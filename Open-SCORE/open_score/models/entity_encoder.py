"""Entity encoders and ALMA controller bridge for the six HAD methods.

Attention and imagined branches are inherited from ALMA commit
81bd5c475972b982ce470b1ed78154ec39d236da. The GNN is adapted from
SPECTra ffababf6187216c9d16b2109ee8ef6fe5fdf1172, SMACv2 gnn_rnn_agent.py:
one graph-convolution layer, separate self/neighbour maps and mean pooling.
Only the graph encoder is borrowed; GNN-QMIX uses the shared ALMA learner.
"""
from copy import copy, deepcopy

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from controllers.entity_controller import EntityMAC
from modules.agents.agent import Agent as ALMAAgent
from modules.layers.attention import EntityAttentionLayer


def relative_entity_views(inputs, args, extras=()):
    """Construct temporary observer-centred views; replay stays global.

    Each observer is moved to row zero. Other geometry is relative to it;
    its own absolute position and velocity are retained, so own speed is not
    lost by subtracting it from itself. Native FF inputs have no HAD geometry.
    Extra per-entity tensors use the same observer order and visibility mask.
    """
    entities = inputs["entities"]
    bs, ts, ne, _ = entities.shape
    na = args.n_agents
    observer_entities = inputs.get("observer_entities")
    if observer_entities is None:
        own = entities[:, :, :na]
        views = entities.unsqueeze(2).expand(bs, ts, na, ne, entities.shape[-1]).clone()
    else:
        if observer_entities.shape[:4] != (bs, ts, na, ne):
            raise ValueError("observer_entities must have shape [batch,time,agents,entities,features]")
        views = observer_entities.clone()
        inds = th.arange(na, device=entities.device)
        own = views[:, :, inds, inds].clone()
    ed = views.shape[-1]
    if observer_entities is None and getattr(args, "feature_layout", "had") == "had":
        views[..., :4] -= own.unsqueeze(3)[..., :4]
        inds = th.arange(na, device=entities.device)
        views[:, :, inds, inds] = own
    order = th.arange(ne, device=entities.device).repeat(na, 1)
    inds = th.arange(na, device=entities.device)
    order[:, 0] = inds
    order[inds, inds] = 0
    order = order.reshape(1, 1, na, ne)
    views = views.gather(3, order.unsqueeze(-1).expand(bs, ts, na, ne, ed))
    hidden = inputs["obs_mask"][:, :, :na].bool()
    hidden = hidden | inputs["entity_mask"].bool().unsqueeze(2)
    hidden = hidden.gather(3, order.expand(bs, ts, na, ne))
    dead = inputs["entity_mask"][:, :, :na].bool()
    views = views.masked_fill(hidden.unsqueeze(-1), 0)
    # REFIL's cross-group branch may mask the observer as a KEY, but its
    # query still uses its own physical features (as in official attention).
    # Zeroing that query would silently change the imagined architecture.
    views[:, :, :, 0] = own.masked_fill(dead.unsqueeze(-1), 0)
    packed = (views.reshape(-1, ne, ed), hidden.reshape(-1, ne), dead.reshape(-1), (bs, ts, na))
    if not extras:
        return packed
    extra_out = []
    for extra in extras:
        width = extra.shape[-1]
        gathered = (extra if extra.ndim == 5 else
                    extra.unsqueeze(2).expand(bs, ts, na, ne, width))
        gathered = gathered.gather(3, order.unsqueeze(-1).expand(bs, ts, na, ne, width))
        gathered = gathered.masked_fill(hidden.unsqueeze(-1), 0)
        extra_out.append(gathered.reshape(-1, ne, width))
    return packed + (extra_out,)


class _EntityEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def init_hidden(self):
        return next(self.parameters()).new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs):
        extras = ()
        if getattr(self, "pre_attn_task", False) and inputs.get("task_embeds") is not None:
            extras = (inputs["task_embeds"],)
        unpacked = relative_entity_views(inputs, self.args, extras=extras)
        views, hidden, dead, shape = unpacked[:4]
        task_x = unpacked[4][0] if len(unpacked) > 4 else None
        chunk_size = int(getattr(self.args, "encoder_chunk_size", 1024))
        if chunk_size < 1:
            raise ValueError("encoder_chunk_size must be positive")
        active = None
        if getattr(self.args, "encoder_skip_dead", True):
            # These encoders never mix the observer/batch axis and mask each
            # inactive observer's output to zero. Compact only that axis;
            # a LIVE REFIL observer with an empty key set must still run.
            active = (~dead).nonzero(as_tuple=False).flatten()
            views, hidden, dead = views[active], hidden[active], dead[active]
            if task_x is not None:
                task_x = task_x[active]
        if views.shape[0] == 0:
            result = views.new_zeros(*shape, self.args.rnn_hidden_dim)
            if th.is_grad_enabled():
                # Preserve zero (rather than missing) parameter/input grads
                # when a recurrent replay timestep consists only of padding.
                zero = views.sum() * 0
                for parameter in self.parameters():
                    zero = zero + parameter.reshape(-1)[0] * 0
                result = result + zero
            return result
        outputs = []
        # Relative features add an observer axis. Recompute these inexpensive
        # attention/GCN activations in backward instead of retaining all of it
        # for a 32-episode full-length minibatch (REFIL has three branches).
        for start in range(0, views.shape[0], chunk_size):
            v = views[start:start + chunk_size]
            m, d = hidden[start:start + chunk_size], dead[start:start + chunk_size]
            extra = None if task_x is None else task_x[start:start + chunk_size]
            if extra is None:
                if self.training and th.is_grad_enabled():
                    out = checkpoint(self._encode, v, m, d, use_reentrant=False)
                else:
                    out = self._encode(v, m, d)
            elif self.training and th.is_grad_enabled():
                out = checkpoint(self._encode, v, m, d, extra, use_reentrant=False)
            else:
                out = self._encode(v, m, d, extra)
            outputs.append(out)
        encoded = th.cat(outputs, dim=0)
        if active is not None:
            encoded = encoded.new_zeros(shape[0] * shape[1] * shape[2],
                                        self.args.rnn_hidden_dim).index_copy(0, active, encoded)
        return encoded.reshape(*shape, self.args.rnn_hidden_dim)


class FlattenEncoder(_EntityEncoder):
    """B0: ordered zero-padded vector plus explicit visibility mask.

    ``pool_slots`` sizes those ordered slots to the training-pool roster, so
    the baseline carries no connection that never receives an input. A roster
    it cannot represent is refused where the configuration is chosen, in
    HADWrapper.reset and FrozenPolicyAdapter; unfilled replay timesteps reach
    this tensor path with an all-zero mask and must not be treated as one.
    """
    def __init__(self, input_shape, args):
        super().__init__(args)
        from open_score.envs.features import pool_slot_indices
        keep = pool_slot_indices(args)
        self.register_buffer("keep", None if keep is None else
                             th.as_tensor(keep, dtype=th.long), persistent=False)
        n_slots = args.n_entities if keep is None else len(keep)
        self.fc1 = nn.Linear(n_slots * (input_shape + 1), args.rnn_hidden_dim)

    def _encode(self, views, hidden, dead):
        if self.keep is not None:
            views, hidden = views[:, self.keep], hidden[:, self.keep]
        x = th.cat((views, (~hidden).to(views.dtype).unsqueeze(-1)), dim=-1)
        return F.relu(self.fc1(x.flatten(1))).masked_fill(dead[:, None], 0)


class AttentionEncoder(_EntityEncoder):
    """Official REFIL/ALMA attention with temporary relative geometry."""
    def __init__(self, input_shape, args):
        super().__init__(args)
        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        self.attn = EntityAttentionLayer(args.attn_embed_dim, args.attn_embed_dim,
                                         args.attn_embed_dim, args)
        self.fc2 = nn.Linear(args.attn_embed_dim, args.rnn_hidden_dim)
        agent = args.agent if isinstance(getattr(args, "agent", None), dict) else {}
        self.pre_attn_task = bool(agent.get("task_embed_in_attn"))

    def _encode(self, views, hidden, dead, task_x=None):
        # An inactive observer has a dummy zero self token, preventing an
        # all-minus-infinity softmax before its output is zeroed.
        pre = hidden.clone()
        pre[dead, 0] = False
        # Official No Mask adds the shared task embedding to every entity
        # before attention, so the mixer/policy can see who shares a task
        # without hiding the rest of the field.
        x = self.fc1(views)
        if task_x is not None:
            x = x + task_x
        x = F.relu(x)
        x = self.attn(x, pre_mask=pre[:, None], post_mask=dead[:, None])
        x = F.relu(self.fc2(F.relu(x[:, 0])))
        return x.masked_fill(dead[:, None], 0)

    def _encode_bundle(self, views, hidden, dead, task_x=None):
        pre = hidden.clone()
        pre[dead, 0] = False
        tokens = self.fc1(views)
        if task_x is not None:
            tokens = tokens + task_x
        tokens = F.relu(tokens)
        if getattr(self.args, "skip_refil_local", False):
            local = F.relu(self.fc2(tokens[:, 0])).masked_fill(dead[:, None], 0)
            tokens = tokens.masked_fill(hidden.unsqueeze(-1), 0)
            return local, tokens
        x = self.attn(tokens, pre_mask=pre[:, None], post_mask=dead[:, None])
        local = F.relu(self.fc2(F.relu(x[:, 0]))).masked_fill(dead[:, None], 0)
        tokens = tokens.masked_fill(hidden.unsqueeze(-1), 0)
        return local, tokens

    def encode_bundle(self, inputs):
        """Local REFIL encoding plus observer-centred tokens for a global branch."""
        entities = inputs["entities"]
        bs, ts, ne, _ = entities.shape
        if "observer_entities" in inputs:
            roles = inputs["observer_entities"][..., :3]
            # Native actor roles remain self/ally/enemy. Shared cardinality
            # and type gating retain HAD's friend/enemy/target semantics.
            types = th.stack((roles[..., 0] + roles[..., 1], roles[..., 2],
                              th.zeros_like(roles[..., 0])), dim=-1)
        else:
            types = entities[..., 7:10]
        origin = th.arange(ne, device=entities.device, dtype=entities.dtype)
        origin = origin.view(1, 1, ne, 1).expand(bs, ts, ne, 1)
        unpacked = relative_entity_views(inputs, self.args, extras=(types, origin))
        views, hidden, dead, shape = unpacked[:4]
        types_f, origin_f = unpacked[4]
        chunk_size = int(getattr(self.args, "encoder_chunk_size", 1024))
        if chunk_size < 1:
            raise ValueError("encoder_chunk_size must be positive")
        active = None
        if getattr(self.args, "encoder_skip_dead", True):
            active = (~dead).nonzero(as_tuple=False).flatten()
            views, hidden, dead = views[active], hidden[active], dead[active]
            types_f, origin_f = types_f[active], origin_f[active]
        attn_dim = int(self.args.attn_embed_dim)
        hid = int(self.args.rnn_hidden_dim)
        if views.shape[0] == 0:
            local = views.new_zeros(*shape, hid)
            tokens = views.new_zeros(*shape, ne, attn_dim)
            type_out = views.new_zeros(*shape, ne, 3)
            origin_out = views.new_zeros(*shape, ne)
            key_mask = views.new_ones(*shape, ne, dtype=th.bool)
            dead_out = views.new_ones(*shape, dtype=th.bool)
            if th.is_grad_enabled():
                zero = views.sum() * 0
                for parameter in self.parameters():
                    zero = zero + parameter.reshape(-1)[0] * 0
                local = local + zero
            return local, tokens, key_mask, dead_out, type_out, origin_out
        locals_, tokens_, types_, origins_, masks_, deads_ = [], [], [], [], [], []
        for start in range(0, views.shape[0], chunk_size):
            sl = slice(start, start + chunk_size)
            v, m, d = views[sl], hidden[sl], dead[sl]
            if self.training and th.is_grad_enabled():
                local, tokens = checkpoint(self._encode_bundle, v, m, d, use_reentrant=False)
            else:
                local, tokens = self._encode_bundle(v, m, d)
            locals_.append(local)
            tokens_.append(tokens)
            types_.append(types_f[sl])
            origins_.append(origin_f[sl])
            masks_.append(m)
            deads_.append(d)
        local = th.cat(locals_, dim=0)
        tokens = th.cat(tokens_, dim=0)
        types_f = th.cat(types_, dim=0)
        origin_f = th.cat(origins_, dim=0)
        hidden = th.cat(masks_, dim=0)
        dead = th.cat(deads_, dim=0)
        rows = shape[0] * shape[1] * shape[2]
        if active is not None:
            def scatter(src, fill=None):
                out = src.new_zeros((rows,) + src.shape[1:])
                if fill is True:
                    out = out.fill_(True)
                return out.index_copy(0, active, src)
            local = scatter(local, None).reshape(*shape, hid)
            tokens = scatter(tokens, None).reshape(*shape, ne, attn_dim)
            types_f = scatter(types_f, None).reshape(*shape, ne, 3)
            origin_f = scatter(origin_f, None).reshape(*shape, ne, 1)
            hidden = scatter(hidden, True).reshape(*shape, ne)
            dead = scatter(dead, True).reshape(*shape)
        else:
            local = local.reshape(*shape, hid)
            tokens = tokens.reshape(*shape, ne, attn_dim)
            types_f = types_f.reshape(*shape, ne, 3)
            origin_f = origin_f.reshape(*shape, ne, 1)
            hidden = hidden.reshape(*shape, ne)
            dead = dead.reshape(*shape)
        return local, tokens, hidden.bool(), dead.bool(), types_f, origin_f.squeeze(-1)


class GraphEncoder(_EntityEncoder):
    """One-layer masked GCN; FullObs or kNN-10, with no fixed-node buffers."""
    def __init__(self, input_shape, args):
        super().__init__(args)
        width = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, width)
        self.lin_layer_neighbor = nn.Linear(width, width)
        self.lin_layer_self = nn.Linear(width, width)
        self.edge_mode = getattr(args, "gnn_edges", "full")
        self.k = int(getattr(args, "gnn_k", 10))
        if self.edge_mode not in ("full", "knn"):
            raise ValueError("gnn_edges must be full or knn")

    def _encode(self, views, hidden, dead):
        valid = ~hidden
        ne = views.shape[1]
        adj = valid[:, :, None] & valid[:, None, :]
        adj &= ~th.eye(ne, dtype=th.bool, device=views.device)[None]
        if self.edge_mode == "knn":
            if getattr(self.args, "feature_layout", "had") != "had":
                raise ValueError("kNN geometry is defined only for the HAD schema")
            # Restore observer-relative position for the own token, which
            # otherwise keeps absolute geometry for the policy input.
            points = views[..., :2].clone()
            points[:, 0] = 0
            distances = th.cdist(points, points).masked_fill(~adj, float("inf"))
            neighbours = distances.topk(min(self.k, max(ne - 1, 1)), largest=False).indices
            chosen = th.zeros_like(adj).scatter_(2, neighbours, True)
            adj &= chosen
        x = F.relu(self.fc1(views)).masked_fill(hidden[..., None], 0)
        # Preserve the donor's separate self/neighbour nonlinearities. The
        # full graph denominator n is now actual degree + the self node.
        neighbour = F.relu(th.bmm(adj.to(x.dtype), self.lin_layer_neighbor(x)))
        own = F.relu(self.lin_layer_self(x))
        out = (own + neighbour) / (adj.sum(-1, keepdim=True) + 1).to(x.dtype)
        out = out.masked_fill(hidden[..., None], 0)
        pooled = out.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return pooled.masked_fill(dead[:, None], 0)


class TemporalHead(nn.Module):
    """GRU for DCG/GNN, retaining the persistent padded agent slots."""
    def __init__(self, args, output_features=None):
        super().__init__()
        self.args = args
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.enemy_head = getattr(args, "action_head", None) == "common6_enemy"
        if self.enemy_head:
            self.fc_base = nn.Linear(args.rnn_hidden_dim, 6)
            self.fc_attack = nn.Sequential(
                nn.Linear(args.rnn_hidden_dim + args.attn_embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        else:
            self.fc_out = nn.Linear(args.rnn_hidden_dim, output_features or args.n_actions)

    def action_values(self, hidden, enemy_tokens=None):
        if not self.enemy_head:
            return self.fc_out(hidden)
        if enemy_tokens is None:
            raise ValueError("common6_enemy requires masked initial enemy encodings")
        expanded = hidden.unsqueeze(-2).expand(*enemy_tokens.shape[:-1], hidden.shape[-1])
        attack = self.fc_attack(th.cat((expanded, enemy_tokens), dim=-1)).squeeze(-1)
        return th.cat((self.fc_base(hidden), attack), dim=-1)

    def forward(self, x, inputs, enemy_tokens=None):
        bs, ts, na, hd = x.shape
        h = inputs.get("hidden_state")
        if h is None:
            h = x.new_zeros(bs, na, hd)
        if h.shape[0] != bs:
            h = h.repeat(bs // h.shape[0], 1, 1)
        h = h.reshape(bs * na, hd)
        outputs = []
        for t in range(ts):
            alive = ~inputs["entity_mask"][:, t, :na].bool()
            h = self.rnn(x[:, t].reshape(bs * na, hd), h)
            h = h.masked_fill(~alive.reshape(-1, 1), 0)
            outputs.append(h.reshape(bs, na, hd))
            # Replay contains complete episodes. The final observation must
            # retain this history for time-limit bootstrapping. Only
            # init_hidden (new episode) and actual death clear memory.
        hs = th.stack(outputs, dim=1)
        return self.action_values(hs, enemy_tokens), h.reshape(bs, na, hd)

    def step(self, x, hidden, alive, enemy_tokens=None):
        bs, na, hd = x.shape
        if hidden.shape[0] != bs:
            hidden = hidden.repeat(bs // hidden.shape[0], 1, 1)
        flat = self.rnn(x.reshape(bs * na, hd), hidden.reshape(bs * na, hd))
        flat = flat.masked_fill(~alive.reshape(-1, 1), 0)
        hidden = flat.reshape(bs, na, hd)
        return self.action_values(hidden, enemy_tokens), hidden


def _phi2_from_counts(*counts):
    parts = []
    for count in counts:
        shifted = (count - 1).clamp_min(0).to(dtype=th.float32)
        parts.append(th.stack((th.square(shifted / (shifted + 2)),
                               (count == 0).to(dtype=th.float32)), dim=-1))
    return th.cat(parts, dim=-1)


class _MHA(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("embed dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** 0.5

    def forward(self, query, key, value, mask=None):
        batch, n_q, dim = query.shape
        n_k = key.shape[1]
        heads = self.n_heads
        query = self.query(query).view(batch, n_q, heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch, n_k, heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch, n_k, heads, self.head_dim).transpose(1, 2)
        logits = th.matmul(query, key.transpose(-2, -1)) / self.scale
        if mask is not None:
            hide = mask[:, None, None, :]
            logits = logits.masked_fill(hide, float("-inf"))
            empty = hide.all(dim=-1, keepdim=True)
            logits = logits.masked_fill(empty, 0)
        weights = th.softmax(logits, dim=-1)
        if mask is not None:
            weights = weights.masked_fill(hide, 0)
            weights = weights.masked_fill(empty, 0)
        if getattr(self, "capture_attention", False):
            self.last_weights = weights.detach()
        out = th.matmul(weights, value).transpose(1, 2).contiguous().view(batch, n_q, dim)
        return self.out(out)


class CardinalityMHA(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("embed dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** 0.5

    def forward(self, query, key, value, mask, types):
        batch, n_q, dim = query.shape
        n_k = key.shape[1]
        heads = self.n_heads
        mask = mask.reshape(batch, n_k)
        types = types.reshape(batch, n_k, 3)
        query = self.query(query).view(batch, n_q, heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch, n_k, heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch, n_k, heads, self.head_dim).transpose(1, 2)
        logits = th.matmul(query, key.transpose(-2, -1)) / self.scale
        mixed = logits.new_zeros(batch, heads, n_q, self.head_dim)
        type_logits = []
        contexts = []
        present = []
        for kind in range(3):
            hide = mask | ~(types[..., kind] > 0)
            hide_b = hide.reshape(batch, 1, 1, n_k)
            empty = hide.all(dim=-1)
            empty_b = empty.reshape(batch, 1, 1, 1)
            kind_logits = logits.masked_fill(hide_b, float("-inf")).masked_fill(empty_b, 0)
            weights = th.softmax(kind_logits, dim=-1).masked_fill(hide_b | empty_b, 0)
            contexts.append(th.matmul(weights, value))
            visible = (~hide).to(logits.dtype)
            denom = visible.sum(dim=-1).clamp_min(1).reshape(batch, 1, 1)
            pooled = (logits.masked_fill(hide_b, 0) * visible.reshape(batch, 1, 1, n_k)).sum(-1)
            pooled = (pooled / denom).masked_fill(empty.reshape(batch, 1, 1), float("-inf"))
            type_logits.append(pooled)
            present.append(~empty)
        stacked = th.stack(type_logits, dim=-1)
        missing = ~th.isfinite(stacked).any(dim=-1, keepdim=True)
        stacked = stacked.masked_fill(missing, 0)
        gates = th.softmax(stacked, dim=-1).masked_fill(missing, 0)
        gates = gates.masked_fill(~th.stack(present, dim=-1).reshape(batch, 1, 1, 3), 0)
        for context, gate in zip(contexts, gates.unbind(-1)):
            mixed = mixed + gate.unsqueeze(-1) * context
        mixed = mixed.masked_fill(mask.all(dim=-1).reshape(batch, 1, 1, 1), 0)
        return self.out(mixed.transpose(1, 2).contiguous().reshape(batch, n_q, dim))


def _global_width(args):
    src = int(args.attn_embed_dim)
    dim = int(getattr(args, "global_embed_dim", 64))
    heads = int(getattr(args, "global_n_heads", 4))
    ffn_mult = int(getattr(args, "global_ffn_mult", 2))
    if dim % heads != 0:
        raise ValueError("global_embed_dim must be divisible by global_n_heads")
    return src, dim, heads, ffn_mult


def intent_classes(args):
    """HAD: the 9 planar actions. SMAC: 6 basic actions + one merged "attack" class."""
    return 7 if getattr(args, "action_head", None) == "common6_enemy" else int(args.n_actions)


class GlobalBranch(nn.Module):
    def __init__(self, args, kind):
        super().__init__()
        self.kind = kind
        self.args = args
        src, dim, heads, ffn_mult = _global_width(args)
        hidden = int(args.rnn_hidden_dim)
        self.n_slots = int(getattr(args, "global_slots", 4))
        self.token_proj = nn.Identity() if src == dim else nn.Linear(src, dim)
        self.count_mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.query_proj = nn.Linear(dim + hidden, dim)
        self.fuse = nn.Linear(dim, hidden)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)
        self.count_to_token = nn.Linear(32, dim)
        if kind == "card":
            self.card_attn = CardinalityMHA(dim, heads)
            self.card_mix = nn.Linear(dim + 32, dim)
        else:
            self.self_attn = _MHA(dim, heads)
            self.read_attn = _MHA(dim, heads)
            self.read_score = nn.Linear(dim, 1)
            self.norm_attn = nn.LayerNorm(dim)
            self.norm_ffn = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.ReLU(),
                                     nn.Linear(dim * ffn_mult, dim))
            if kind == "slot":
                self.init_slots = nn.Parameter(th.randn(1, self.n_slots, dim) * 0.02)
                self.cross_attn = _MHA(dim, heads)
                self.norm_cross = nn.LayerNorm(dim)
            if kind == "feedback":
                self.pref_proj = nn.Linear(int(args.n_actions), dim)
                self.pref_logits = nn.Linear(dim, int(args.n_actions))
        self.rer_update = getattr(args, "rer_update", "tied")
        if self.rer_update not in ("tied", "untied4", "kv0"):
            raise ValueError("rer_update must be tied, untied4 or kv0")
        if self.rer_update != "tied" and kind != "cycle":
            raise ValueError("RER update interventions require global_branch=cycle")
        if self.rer_update == "untied4":
            # The existing block is round one. Deepcopy adds three independent
            # blocks without drawing randomness or changing any shared module.
            names = ("self_attn", "norm_attn", "norm_ffn", "ffn", "count_to_token")
            self.untied_blocks = nn.ModuleList([
                nn.ModuleDict({name: deepcopy(getattr(self, name)) for name in names})
                for _ in range(3)])
        self.intent = bool(getattr(args, "rer_intent", False))
        if self.intent:
            if kind != "cycle" or self.rer_update == "untied4":
                raise ValueError("rer_intent requires a shared-parameter cycle branch")
            n_actions = intent_classes(args)
            # Drawn from a forked RNG so every other module (and the mixer
            # built afterwards) initializes exactly as the same-seed ARR.
            with th.random.fork_rng(devices=[]):
                self.intent_head = nn.Linear(dim, n_actions)
                self.intent_embed = nn.Linear(n_actions, dim, bias=False)
            nn.init.zeros_(self.intent_embed.weight)
        self.intent_override = None
        self.intent_oracle_actions = None
        self.last_intent = None
        self.readout_override = None

    def project(self, tokens):
        return self.token_proj(tokens)

    def allowed_count(self, key_mask, types):
        batch, n_k = key_mask.shape[0], key_mask.shape[-1]
        key_mask = key_mask.reshape(batch, n_k)
        types = types.reshape(batch, n_k, 3)
        visible = ~key_mask
        red = (visible & (types[..., 0] > 0)).sum(-1)
        blue = (visible & (types[..., 1] > 0)).sum(-1)
        tgt = (visible & (types[..., 2] > 0)).sum(-1)
        return self.count_mlp(_phi2_from_counts(red, blue, tgt))

    def expand_depth(self, depth, n_env, n_time, n_agents, device):
        total = n_env * n_time * n_agents
        if isinstance(depth, th.Tensor):
            depth_t = depth.to(device=device, dtype=th.long).reshape(-1)
            if depth_t.numel() == 1:
                return depth_t.expand(total).contiguous()
            if n_env % depth_t.shape[0] == 0 and depth_t.shape[0] != n_env:
                depth_t = depth_t.repeat(n_env // depth_t.shape[0])
            if depth_t.shape[0] == n_env:
                return depth_t[:, None, None].expand(n_env, n_time, n_agents).reshape(-1)
            if depth_t.shape[0] == n_env * n_agents:
                return depth_t.reshape(n_env, 1, n_agents).expand(n_env, n_time, n_agents).reshape(-1)
            if depth_t.shape[0] == total:
                return depth_t
            raise ValueError("global depth batch does not match observers")
        return th.full((total,), int(1 if depth is None else depth), dtype=th.long, device=device)

    def _agent_query(self, own, hidden):
        if getattr(self.args, "global_query_no_memory", False):
            hidden = th.zeros_like(hidden)
        elif getattr(self.args, "global_query_detach_memory", False):
            # Backprop through time via the readout attention explodes on
            # large-norm deep rounds; the GRU's own recurrence keeps its gradient.
            hidden = hidden.detach()
        return self.query_proj(th.cat((own, hidden), dim=-1)).unsqueeze(1)

    def _jk(self, states, query, mem_mask, depth):
        self.captured_read_attn = None
        mode = self.readout_override or getattr(self.args, "rer_readout", "learned")
        if mode not in ("learned", "read1", "read2", "read3", "read4", "uniform"):
            raise ValueError(f"Unknown RER readout {mode!r}")
        if mode != "learned" and self.training:
            raise ValueError("RER readout overrides are evaluation-only")
        capture_indices = getattr(self, "_capture_read_indices", None)
        if getattr(self.args, "read_last_round", False) and mode == "learned" and capture_indices is None:
            stacked = th.stack(states, dim=0)
            index = (depth.clamp(min=1) - 1).long().clamp(max=stacked.shape[0] - 1)
            chosen = stacked[index, th.arange(index.shape[0], device=index.device)]
            read = self.read_attn(query, chosen, chosen, mem_mask).squeeze(1)
            if getattr(self, "capture_attention", False):
                alpha = th.zeros(index.shape[0], stacked.shape[0], device=index.device)
                alpha[th.arange(index.shape[0], device=index.device), index] = 1
                self.last_alpha = alpha.detach()
                if hasattr(self.read_attn, "last_weights"):
                    self.captured_read_attn = self.read_attn.last_weights.mean(1).squeeze(1).detach()
            return read
        scores, values, read_weights = [], [], []
        for memory in states:
            read = self.read_attn(query, memory, memory, mem_mask).squeeze(1)
            values.append(read)
            scores.append(self.read_score(read))
            if getattr(self, "capture_attention", False) and hasattr(self.read_attn, "last_weights"):
                read_weights.append(self.read_attn.last_weights.mean(1).squeeze(1).detach())
        scores = th.cat(scores, dim=-1)
        index = th.arange(scores.shape[-1], device=scores.device)
        valid = index.unsqueeze(0) < depth.unsqueeze(-1)
        scores = scores.masked_fill(~valid, float("-inf"))
        empty = ~valid.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(empty, 0)
        alpha = th.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(~valid, 0)
        if mode != "learned":
            if len(states) != 4 or not bool((depth == 4).all()):
                raise ValueError("RER readout interventions require all four rounds")
            alpha = th.zeros_like(alpha)
            if mode == "uniform":
                alpha.fill_(0.25)
            else:
                alpha[:, int(mode[-1]) - 1] = 1
        elif getattr(self.args, "read_last_round", False):
            alpha = th.zeros_like(alpha).scatter_(1, (depth.clamp_min(1) - 1)[:, None], 1)
        if getattr(self, "capture_attention", False):
            self.last_alpha = alpha.detach()
        stacked = th.stack(values, dim=-1)
        if capture_indices is not None:
            self.captured_reads = stacked.transpose(1, 2).index_select(0, capture_indices).detach().cpu()
            self.captured_alpha = alpha.index_select(0, capture_indices).detach().cpu()
            if read_weights:
                self.captured_read_attn = th.stack(read_weights, dim=1).index_select(0, capture_indices).detach().cpu()
        elif read_weights:
            self.captured_read_attn = th.stack(read_weights, dim=1).detach()
        return (stacked * alpha.unsqueeze(1)).sum(-1)

    def _maybe_checkpoint(self, fn, *tensors):
        if self.training and th.is_grad_enabled():
            return checkpoint(fn, *tensors, use_reentrant=False)
        return fn(*tensors)

    def _map_chunks(self, fn, *aligned, extra=()):
        n = aligned[0].shape[0]
        chunk = int(getattr(self.args, "encoder_chunk_size", 1024))
        if chunk < 1:
            raise ValueError("encoder_chunk_size must be positive")
        if n <= chunk:
            return self._maybe_checkpoint(fn, *aligned, *extra)
        parts = []
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            parts.append(self._maybe_checkpoint(fn, *(tensor[sl] for tensor in aligned), *extra))
        return th.cat(parts, 0)

    def _cycle_round(self, tokens, mask, count, initial=None, round_id=0, kv_extra=None):
        block = self.untied_blocks[round_id - 1] if self.rer_update == "untied4" and round_id else None
        def layer(name):
            return getattr(self, name) if block is None else block[name]
        injected = layer("norm_attn")(tokens)
        if not getattr(self.args, "skip_count_inject", False):
            injected = injected + layer("count_to_token")(count).unsqueeze(1)
        memory = initial if self.rer_update == "kv0" and initial is not None else tokens
        if kv_extra is not None:
            memory = memory + kv_extra
        if getattr(self.args, "global_kv_prenorm", False):
            memory = layer("norm_attn")(memory).masked_fill(mask.unsqueeze(-1), 0)
        attn = layer("self_attn")
        tokens = tokens + attn(injected, memory, memory, mask)
        if getattr(self, "capture_attention", False) and hasattr(attn, "last_weights"):
            self.last_self_attn.append(attn.last_weights.mean(1).detach())
        tokens = tokens + layer("ffn")(layer("norm_ffn")(tokens))
        return tokens.masked_fill(mask.unsqueeze(-1), 0)

    def _slot_round(self, slots, tokens, mask, count):
        query = self.norm_cross(slots)
        if not getattr(self.args, "skip_count_inject", False):
            query = query + self.count_to_token(count).unsqueeze(1)
        slots = slots + self.cross_attn(query, tokens, tokens, mask)
        slots = slots + self.self_attn(self.norm_attn(slots), slots, slots)
        return slots + self.ffn(self.norm_ffn(slots))

    def _intent_probs(self, logits, teammate, origin, bt):
        probs = th.softmax(logits, dim=-1)
        mode = self.intent_override
        if mode is None:
            return probs
        if self.training:
            raise ValueError("Intent overrides are evaluation-only")
        if mode == "uniform":
            return th.full_like(probs, 1.0 / probs.shape[-1])
        if mode == "shuffle":
            # Permute predicted distributions among each observer's visible teammates.
            shuffled = probs.clone()
            for row in range(probs.shape[0]):
                slots = teammate[row].nonzero(as_tuple=False).flatten()
                if slots.numel() > 1:
                    order = slots[th.randperm(slots.numel(), device=slots.device)]
                    shuffled[row, slots] = probs[row, order]
            return shuffled
        if mode == "oracle":
            actions = self.intent_oracle_actions
            if actions is None or bt is None:
                raise ValueError("Oracle intent requires this step's teammate actions")
            n_agents = actions.shape[-1]
            chosen = actions.index_select(0, bt).gather(1, origin.clamp(0, n_agents - 1).long())
            return F.one_hot(chosen.clamp(max=probs.shape[-1] - 1), probs.shape[-1]).to(probs.dtype)
        raise ValueError(f"Unknown intent override {mode!r}")

    def _own_prefs(self, memory):
        return th.softmax(self.pref_logits(memory[:, 0]), dim=-1)

    def _inject_chunk(self, tokens, origin, key_mask, bt, embed):
        n_agents = embed.shape[1]
        gathered = embed.index_select(0, bt).gather(
            1, origin.clamp(0, n_agents - 1).long().unsqueeze(-1).expand(-1, -1, embed.size(-1)))
        red = (origin < n_agents).unsqueeze(-1).to(tokens.dtype)
        live = (~key_mask).unsqueeze(-1).to(tokens.dtype)
        return tokens + gathered * red * live

    def build_memories(self, tokens, key_mask, types, depth_t, origin=None, pack=None):
        count = self.allowed_count(key_mask, types)
        n_rounds = max(int(depth_t.max().item()), 1)
        if self.rer_update == "untied4" and n_rounds > 4:
            raise ValueError("untied4 supports at most four rounds")
        states = []
        if self.kind == "slot":
            memory = self.init_slots.expand(tokens.shape[0], -1, -1)
            for _ in range(n_rounds):
                memory = self._maybe_checkpoint(self._slot_round, memory, tokens, key_mask, count)
                states.append(memory)
            return states, None
        # Keep query construction separate from memory masking: REFIL can
        # retain an own query while excluding that entity from the key set.
        memory = tokens.masked_fill(key_mask.unsqueeze(-1), 0)
        initial = memory
        if getattr(self.args, "global_read_h0", False):
            return [memory], key_mask
        embed = None
        bt = None if pack is None else pack["idx"] // int(pack["na"])
        teammate, logits, kv_extra = None, [], None
        if self.intent:
            # Visible, live Red teammates in this observer's view; slot 0 is the observer.
            n_agents = int(self.args.n_agents)
            teammate = (origin < n_agents) & ~key_mask
            teammate[:, 0] = False
        for round_id in range(n_rounds):
            if self.kind == "feedback" and embed is not None:
                memory = self._map_chunks(
                    self._inject_chunk, memory, origin, key_mask, bt, extra=(embed,))
            if self.kind == "feedback":
                memory = self._map_chunks(self._cycle_round, memory, key_mask, count)
            elif kv_extra is not None:
                def cycle(current, mask, counts, h0, extra, iteration=round_id):
                    return self._cycle_round(current, mask, counts, h0, iteration, extra)
                memory = self._maybe_checkpoint(cycle, memory, key_mask, count, initial, kv_extra)
            else:
                def cycle(current, mask, counts, h0, iteration=round_id):
                    return self._cycle_round(current, mask, counts, h0, iteration)
                memory = self._maybe_checkpoint(cycle, memory, key_mask, count, initial)
            states.append(memory)
            if self.intent:
                logits.append(self.intent_head(memory))
                if round_id + 1 < n_rounds:
                    probs = self._intent_probs(logits[-1], teammate, origin, bt)
                    kv_extra = self.intent_embed(probs) * teammate.unsqueeze(-1).to(memory.dtype)
            if self.intent and round_id + 1 == n_rounds:
                self.last_intent = dict(logits=logits, teammate=teammate, origin=origin)
            if self.kind == "feedback" and round_id + 1 < n_rounds:
                prefs = self._own_prefs(memory)
                dense = prefs.new_zeros(pack["n_obs"], prefs.shape[-1])
                dense = dense.index_copy(0, pack["idx"], prefs)
                embed = self.pref_proj(dense.reshape(pack["bs"] * pack["ts"], pack["na"], -1))
        return states, key_mask

    def read_context(self, states, tokens, key_mask, hidden, depth_t, local, mem_mask=None):
        n = tokens.shape[0]
        chunk = int(getattr(self.args, "encoder_chunk_size", 1024))
        if chunk < 1:
            raise ValueError("encoder_chunk_size must be positive")
        if n <= chunk:
            return self._read_context_body(states, tokens, key_mask, hidden, depth_t, local, mem_mask)
        parts = []
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            mask = None if mem_mask is None else mem_mask[sl]
            parts.append(self._read_context_body(
                [state[sl] for state in states], tokens[sl], key_mask[sl],
                hidden[sl], depth_t[sl], local[sl], mask))
        return th.cat(parts, 0)

    def _read_context_body(self, states, tokens, key_mask, hidden, depth_t, local, mem_mask):
        query = self._agent_query(tokens[:, 0], hidden.reshape(tokens.shape[0], -1))
        if self.kind == "slot":
            mask = None
        else:
            mask = key_mask if mem_mask is None else mem_mask
        return local + self.fuse(self._jk(states, query, mask, depth_t))

    def _read_stacked(self, stacked, tokens, key_mask, hidden, depth_t, local, mem_mask):
        states = [stacked[i] for i in range(stacked.shape[0])]
        mask = None if self.kind == "slot" else mem_mask
        return self._read_context_body(states, tokens, key_mask, hidden, depth_t, local, mask)

    def card_fuse(self, tokens, key_mask, types, hidden, local):
        query = self._agent_query(tokens[:, 0], hidden.reshape(tokens.shape[0], -1))
        count = self.allowed_count(key_mask, types)
        read = self.card_attn(query, tokens, tokens, key_mask, types).squeeze(1)
        context = self.card_mix(th.cat((read, count), dim=-1))
        return local + self.fuse(context)


class EntityAgent(ALMAAgent):
    def __init__(self, input_shape, args):
        # Preserve ALMA's imagined-input construction, subtask conditioning
        # and combined forward.
        input_shape = int(getattr(args, "actor_entity_shape", None) or
                          getattr(args, "observer_entity_shape", None) or input_shape)
        if getattr(args, "action_head", None) == "common6_enemy":
            if args.entity_last_action or getattr(args, "obs_agent_id", False):
                raise ValueError("SMAC entity actors exclude last action and agent IDs")
            if int(args.attn_embed_dim) != 128 or int(args.rnn_hidden_dim) != 64:
                raise ValueError("common6_enemy requires initial encoding 128 and GRU 64")
        super().__init__(input_shape, args, recurrent=False, entity_scheme=True,
                         subtask_cond=args.agent.get("subtask_cond"))
        kind = getattr(args, "encoder", "attention")
        self._base = {"flatten": FlattenEncoder, "attention": AttentionEncoder,
                      "gnn": GraphEncoder}[kind](input_shape, args)
        if args.agent.get("recurrent", False):
            self._head = TemporalHead(args)
        # The completed full-obs probe arms add a task label only after
        # attention, and their checkpoints contain task_cond. Official No Mask
        # puts the label on every entity first; do not keep both paths.
        if (args.agent.get("subtask_cond") == "full_obs"
                and not args.agent.get("task_embed_in_attn")):
            self.task_cond = nn.Linear(args.attn_embed_dim, args.rnn_hidden_dim)
        self.count_cond = getattr(args, "count_cond", None)
        # v4 REFIL-C checkpoints have no count_ln flag and expect LayerNorm.
        self.use_count_ln = bool(getattr(args, "count_ln", True))
        self.global_branch = getattr(args, "global_branch", None)
        if self.global_branch in (None, False, "off", "none", ""):
            self.global_branch = None
        elif self.global_branch not in ("card", "cycle", "slot", "feedback"):
            raise ValueError(f"Unknown global_branch {self.global_branch!r}")
        if self.global_branch is not None:
            if kind != "attention":
                raise ValueError("global_branch requires encoder=attention")
            if not args.agent.get("recurrent", False):
                raise ValueError("global_branch requires a recurrent head")
            self.global_net = GlobalBranch(args, self.global_branch)
            self.global_net.capture_attention = False
            self.global_net.last_self_attn = []
        elif self.count_cond == "phi2":
            hidden = args.rnn_hidden_dim
            self.count_mlp = nn.Linear(6, 32)
            if self.use_count_ln:
                self.count_ln = nn.LayerNorm(hidden)
            self.count_alpha = nn.Linear(32, hidden)
            self.count_beta = nn.Linear(32, hidden)
            nn.init.zeros_(self.count_alpha.weight)
            nn.init.zeros_(self.count_alpha.bias)
            nn.init.zeros_(self.count_beta.weight)
            nn.init.zeros_(self.count_beta.bias)
        elif self.count_cond not in (None, False, "off", "none"):
            raise ValueError(f"Unknown count_cond {self.count_cond!r}")
        hidden = int(args.rnn_hidden_dim)
        extra = int(getattr(args, "matched_hidden", 0) or 0)
        if extra > 0 and self.global_branch is None:
            self.matched_mlp = nn.Sequential(nn.Linear(hidden, extra), nn.ReLU(), nn.Linear(extra, hidden))
            nn.init.zeros_(self.matched_mlp[-1].weight)
            nn.init.zeros_(self.matched_mlp[-1].bias)
        self.capture_attention = False
        self.last_capture = []
        self._capture_limits = None

    def enable_capture(self, max_decisions=1, max_observers=4):
        """Capture a bounded set of real evaluation decisions, detached on CPU."""
        if max_decisions < 1 or max_observers < 1:
            raise ValueError("capture bounds must be positive")
        self._capture_limits = (int(max_decisions), int(max_observers))
        self.last_capture = []

    def disable_capture(self):
        self._capture_limits = None
        if self.global_branch is not None:
            self.global_net._capture_read_indices = None

    def _capture_rows(self, alive):
        if (self._capture_limits is None or self.training or
                not getattr(self, "_capture_real_decision", False) or
                len(self.last_capture) >= self._capture_limits[0]):
            return None
        return alive.reshape(-1).nonzero(as_tuple=False).flatten()[:self._capture_limits[1]]

    def _count_embedding(self, entity_mask):
        n_red = int(self.args.n_agents)
        n_entities = int(entity_mask.shape[-1])
        n_tasks = int(getattr(self.args, "n_tasks", 0))
        if n_tasks <= 0 or n_red + n_tasks > n_entities:
            raise ValueError("count_cond=phi2 needs red, blue and target slots")
        n_blue = n_entities - n_red - n_tasks
        live = ~entity_mask
        counts = (live[..., :n_red].sum(-1),
                  live[..., n_red:n_red + n_blue].sum(-1),
                  live[..., n_red + n_blue:n_red + n_blue + n_tasks].sum(-1))

        def rho(count):
            shifted = (count - 1).clamp_min(0).to(dtype=th.float32)
            return th.stack((th.square(shifted / (shifted + 2)), (count == 0).to(dtype=th.float32)), dim=-1)

        return th.cat([rho(count) for count in counts], dim=-1)

    def _apply_count_cond(self, encoded, entity_mask):
        if self.count_cond != "phi2":
            return encoded
        embed = F.relu(self.count_mlp(self._count_embedding(entity_mask)))
        scale = 1 + self.count_alpha(embed).unsqueeze(2)
        shift = self.count_beta(embed).unsqueeze(2)
        hidden = self.count_ln(encoded) if self.use_count_ln else encoded
        conditioned = scale * hidden + shift
        return conditioned.masked_fill(entity_mask[:, :, :self.args.n_agents].unsqueeze(-1), 0)

    def _compute_network(self, inputs):
        # Modern torch requires bool masks; official ALMA stores uint8 replay.
        inputs = dict(inputs)
        inputs["entity_mask"] = inputs["entity_mask"].bool()
        inputs["obs_mask"] = inputs["obs_mask"].bool()
        if self.global_branch is not None:
            return self._compute_global(inputs)
        enemy_tokens = None
        if getattr(self._head, "enemy_head", False):
            encoded, initial_tokens, _, _, _, _ = self._base.encode_bundle(inputs)
            enemy_tokens = initial_tokens[..., self.args.n_agents:, :]
        else:
            encoded = self._base(inputs)
        encoded = self._apply_count_cond(encoded, inputs["entity_mask"])
        if hasattr(self, "matched_mlp"):
            encoded = encoded + self.matched_mlp(encoded)
            encoded = encoded.masked_fill(inputs["entity_mask"][:, :, :self.args.n_agents].unsqueeze(-1), 0)
        if hasattr(self, "task_cond") and "task_embeds" in inputs:
            encoded = encoded + self.task_cond(inputs["task_embeds"][:, :, :self.args.n_agents])
        if self.use_copa:
            encoded = encoded + inputs["coach_z"]
        if enemy_tokens is None:
            q, hidden = self._head(encoded, inputs)
        else:
            q, hidden = self._head(encoded, inputs, enemy_tokens=enemy_tokens)
        if self._capture_limits is not None and hasattr(self._base, "encode_bundle"):
            self._capture_host(inputs, q)
        agent_mask = inputs["entity_mask"][:, :, :self.args.n_agents]
        return q.masked_fill(agent_mask.unsqueeze(3), 0), hidden

    def _capture_host(self, inputs, q):
        """REFIL / no-cycle capture: observer-centred entity tokens as H0 only."""
        _local, tokens, key_mask, dead, types, origin = self._base.encode_bundle(inputs)
        bs, ts, na, n_ent = tokens.shape[:4]
        dim = tokens.shape[-1]
        n_act = q.shape[-1]
        for t in range(ts):
            alive = ~dead[:, t]
            rows = self._capture_rows(alive)
            if rows is None:
                continue

            def picked(tensor):
                return tensor.index_select(0, rows).detach().cpu()

            tok_t = tokens[:, t].reshape(bs * na, n_ent, dim)
            km_t = key_mask[:, t].reshape(bs * na, n_ent)
            observer_ids = (rows % na).detach().cpu()
            source = origin[:, t].reshape(bs * na, n_ent)
            self.last_capture.append({
                "time_id": int(getattr(self, "_capture_time", 0)) + t,
                "batch_ids": (rows // na).detach().cpu(),
                "observer_ids": observer_ids,
                "H": picked(tok_t).masked_fill(picked(km_t).unsqueeze(-1), 0)[:, None],
                "u": None,
                "alpha": None,
                "read_attn": None,
                "h_prev": None,
                "q": picked(q[:, t].reshape(bs * na, n_act)),
                "key_mask": picked(km_t),
                "types": picked(types[:, t].reshape(bs * na, n_ent, 3)),
                "origin": picked(source),
                "enemy_ids": th.arange(n_ent - na, dtype=th.long),
            })

    def _compute_global(self, inputs):
        local, tokens, key_mask, dead, types, origin = self._base.encode_bundle(inputs)
        enemy_tokens = tokens[..., self.args.n_agents:, :] if getattr(self._head, "enemy_head", False) else None
        if hasattr(self, "task_cond") and "task_embeds" in inputs:
            local = local + self.task_cond(inputs["task_embeds"][:, :, :self.args.n_agents])
        if self.use_copa:
            local = local + inputs["coach_z"]
        tokens = self.global_net.project(tokens)
        bs, ts, na, hidden_dim = local.shape
        n_ent = tokens.shape[-2]
        gdim = tokens.shape[-1]
        n_act = int(self.args.n_actions) if enemy_tokens is None else 6 + enemy_tokens.shape[-2]
        h = inputs.get("hidden_state")
        if h is None:
            h = local.new_zeros(bs, na, hidden_dim)
        if h.shape[0] != bs:
            h = h.repeat(bs // h.shape[0], 1, 1)
        depth = getattr(self, "cycle_depth", 1)
        if depth is None:
            depth = 1
        kind = self.global_branch
        outputs = []
        self.global_net.last_intent = None
        if kind in ("cycle", "slot", "feedback"):
            if getattr(self.global_net, "capture_attention", False):
                self.global_net.last_self_attn = []
                if hasattr(self.global_net, "self_attn"):
                    self.global_net.self_attn.capture_attention = True
                if hasattr(self.global_net, "read_attn"):
                    self.global_net.read_attn.capture_attention = True
            n_obs = bs * ts * na
            live = ~dead.reshape(-1)
            idx = live.nonzero(as_tuple=False).flatten()
            depth_flat = self.global_net.expand_depth(depth, bs, ts, na, tokens.device)
            states_full = None
            mem_mask_full = None
            if idx.numel():
                tok = tokens.reshape(n_obs, n_ent, gdim).index_select(0, idx)
                km = key_mask.reshape(n_obs, n_ent).index_select(0, idx)
                ty = types.reshape(n_obs, n_ent, 3).index_select(0, idx)
                dpt = depth_flat.index_select(0, idx)
                extra = {}
                if kind == "feedback" or self.global_net.intent:
                    extra["origin"] = origin.reshape(n_obs, n_ent).index_select(0, idx)
                    extra["pack"] = dict(idx=idx, n_obs=n_obs, bs=bs, ts=ts, na=na)
                states_live, mem_mask_live = self.global_net.build_memories(tok, km, ty, dpt, **extra)
                if self.global_net.intent:
                    self.global_net.last_intent.update(idx=idx, depth=dpt, shape=(bs, ts, na))
                n_mem = states_live[0].shape[1]
                states_full = []
                for memory in states_live:
                    packed = memory.new_zeros(n_obs, n_mem, gdim).index_copy(0, idx, memory)
                    states_full.append(packed.reshape(bs, ts, na, n_mem, gdim))
                if mem_mask_live is not None:
                    mem_mask_full = key_mask.new_ones(n_obs, n_ent).index_copy(0, idx, mem_mask_live)
                    mem_mask_full = mem_mask_full.reshape(bs, ts, na, n_ent)
            for t in range(ts):
                alive = ~dead[:, t]
                if states_full is None:
                    h = h.masked_fill(~alive.unsqueeze(-1), 0)
                    outputs.append(local.new_zeros(bs, na, n_act))
                    continue
                stacked = th.stack(
                    [state[:, t].reshape(bs * na, -1, gdim) for state in states_full], 0)
                tok_t = tokens[:, t].reshape(bs * na, n_ent, gdim)
                km_t = key_mask[:, t].reshape(bs * na, n_ent)
                mask_t = km_t if mem_mask_full is None else mem_mask_full[:, t].reshape(bs * na, n_ent)
                capture_rows = self._capture_rows(alive)
                self.global_net._capture_read_indices = capture_rows
                h_prev = h
                fused = self.global_net._maybe_checkpoint(
                    self.global_net._read_stacked, stacked, tok_t, km_t,
                    h.reshape(bs * na, hidden_dim),
                    depth_flat.reshape(bs, ts, na)[:, t].reshape(-1),
                    local[:, t].reshape(bs * na, hidden_dim), mask_t)
                q_t, h = self._head.step(fused.reshape(bs, na, hidden_dim), h, alive,
                                         None if enemy_tokens is None else enemy_tokens[:, t])
                outputs.append(q_t)
                if capture_rows is not None:
                    def picked(tensor):
                        return tensor.index_select(0, capture_rows).detach().cpu()
                    h0 = picked(tok_t).masked_fill(picked(km_t).unsqueeze(-1), 0)
                    observer_ids = (capture_rows % na).detach().cpu()
                    source_ids = th.arange(n_ent).expand(len(capture_rows), -1).clone()
                    row_ids = th.arange(len(capture_rows))
                    source_ids[:, 0] = observer_ids
                    source_ids[row_ids, observer_ids] = 0
                    self.last_capture.append({
                        "time_id": int(getattr(self, "_capture_time", 0)) + t,
                        "batch_ids": (capture_rows // na).detach().cpu(),
                        "observer_ids": observer_ids,
                        "H": th.cat((h0[:, None], picked(stacked.transpose(0, 1))), dim=1),
                        "u": self.global_net.captured_reads,
                        "alpha": self.global_net.captured_alpha,
                        "read_attn": getattr(self.global_net, "captured_read_attn", None),
                        "h_prev": picked(h_prev.reshape(bs * na, hidden_dim)),
                        "q": picked(q_t.reshape(bs * na, n_act)),
                        "key_mask": picked(km_t),
                        "types": picked(types[:, t].reshape(bs * na, n_ent, 3)),
                        "origin": source_ids,
                        "enemy_ids": th.arange(n_ent - na, dtype=th.long),
                    })
                self.global_net._capture_read_indices = None
        else:
            for t in range(ts):
                alive = ~dead[:, t]
                fused = self.global_net.card_fuse(
                    tokens[:, t].reshape(bs * na, n_ent, gdim),
                    key_mask[:, t].reshape(bs * na, n_ent),
                    types[:, t].reshape(bs * na, n_ent, 3),
                    h, local[:, t].reshape(bs * na, hidden_dim))
                q_t, h = self._head.step(fused.reshape(bs, na, hidden_dim), h, alive,
                                         None if enemy_tokens is None else enemy_tokens[:, t])
                outputs.append(q_t)
        q = th.stack(outputs, dim=1)
        agent_mask = inputs["entity_mask"][:, :, :na]
        return q.masked_fill(agent_mask.unsqueeze(3), 0), h

    def make_imagined_inputs(self, inputs):
        imagined, groups = super().make_imagined_inputs(inputs)
        if "observer_entities" in inputs:
            imagined["observer_entities"] = inputs["observer_entities"].repeat(2, 1, 1, 1, 1)
        # Keep the random partition constant but exclude deaths at each t.
        inactive = inputs["entity_mask"].bool()
        blocked = inactive.unsqueeze(-1) | inactive.unsqueeze(-2)
        groups = tuple(g.bool() | blocked for g in groups)
        imagined["imagine_mask"] = th.cat(groups, dim=0).to(th.uint8)
        return imagined, groups


def intent_targets(agent, actions, step_mask, n_main):
    """Map every live observer's teammate slot back to that teammate's executed action.

    Returns (record, observer rows, labels [rows, n_ent], valid [rows, n_ent], depth [rows]).
    """
    record = getattr(agent.global_net, "last_intent", None)
    if record is None or "idx" not in record:
        return None
    idx = record["idx"]
    _, ts, na = record["shape"]
    b, t = idx // (ts * na), (idx // na) % ts
    rows = ((b < n_main) & (t < ts - 1)).nonzero(as_tuple=False).flatten()
    if rows.numel() == 0:
        return None
    b, t = b.index_select(0, rows), t.index_select(0, rows)
    origin = record["origin"].index_select(0, rows).long().clamp(0, na - 1)
    classes = record["logits"][0].shape[-1]
    labels = actions[b, t].gather(1, origin).clamp(max=classes - 1)
    valid = record["teammate"].index_select(0, rows) & (step_mask[b, t] > 0).unsqueeze(-1)
    return record, rows, labels, valid, record["depth"].index_select(0, rows)


def intent_auxiliary(agent, actions, step_mask, n_main):
    """Cross-entropy of each round's teammate-action prediction against the
    action that teammate executed at the same step (main batch only).

    actions: [bs, T-1, n_agents] long; step_mask: [bs, T-1] (1 = learnable step).
    Returns (loss averaged over rounds and visible teammates, top-1 accuracy, count).
    """
    targets = intent_targets(agent, actions, step_mask, n_main)
    if targets is None:
        return None
    record, rows, labels, valid, depth = targets
    total, correct, count = 0.0, 0.0, 0
    for round_id, logits in enumerate(record["logits"]):
        keep = valid & (depth > round_id).unsqueeze(-1)
        n = int(keep.sum())
        if n == 0:
            continue
        chosen = logits.index_select(0, rows)[keep]
        target = labels[keep]
        total = total + F.cross_entropy(chosen, target, reduction="sum")
        correct += float((chosen.detach().argmax(-1) == target).sum())
        count += n
    if count == 0:
        return None
    return total / count, correct / count, count


class PolicyValueMixin:
    def state_dict(self):
        state = {"agent": self.agent.state_dict()}
        if getattr(self, "learned_alloc", False):
            state["alloc_policy"] = self.alloc_policy.state_dict()
            state["alloc_critic"] = self.alloc_critic.state_dict()
        return state

    def load_state_dict(self, state):
        self.agent.load_state_dict(state["agent"])
        if getattr(self, "learned_alloc", False):
            self.alloc_policy.load_state_dict(state["alloc_policy"])
            self.alloc_critic.load_state_dict(state["alloc_critic"])

    def evaluation_values(self, batch, t, actions, active, mixer):
        """Use this decision's cached Qs; never advance recurrent state twice."""
        active = list(active)
        q = self._decision_q[active]
        actions = th.as_tensor(actions, device=q.device, dtype=th.long)
        if actions.ndim == 3:
            actions = actions.squeeze(-1)
        chosen = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        alive = ~batch["entity_mask"][active, t, :self.n_agents].bool()
        chosen = chosen.masked_fill(~alive, 0)
        individual = chosen.sum(-1) / alive.sum(-1).clamp_min(1)
        if mixer is None:
            total = individual
        else:
            inputs, _ = self._build_inputs(batch, slice(t, t + 1))
            keys = ("entities", "entity_mask")
            if getattr(self.args, "multi_task", False):
                keys += ("entity2task_mask",)
            mixins = {key: inputs[key][active] for key in keys}
            with th.no_grad():
                # A subtask-conditioned mixer returns one value per subtask;
                # the reported team value is their sum, as in the TD target.
                total = mixer(chosen[:, None], mixins)
                total = mixer.denormalize(total).sum(-1).reshape(-1)
        return {"q_tot": total.detach().cpu().tolist(), "q_i": individual.detach().cpu().tolist()}

    def load_models(self, path, pi_only=False):
        self.agent.load_state_dict(th.load(f"{path}agent.th", map_location="cpu"))


class SharedEntityMAC(PolicyValueMixin, EntityMAC):
    def _build_agents(self, input_shapes):
        self.agent = EntityAgent(input_shapes[0], self.args)
        if self.learned_alloc:
            # This override replaces EntityMAC._build_agents to install the
            # shared encoders, so ALMA's upper layer is built here instead.
            from modules.agents import ALLOC_CRITIC_REGISTRY, ALLOC_POLICY_REGISTRY
            hier_shape = input_shapes[1]
            self.alloc_critic = ALLOC_CRITIC_REGISTRY[self.args.hier_agent["alloc_critic"]](hier_shape, self.args)
            self.alloc_policy = ALLOC_POLICY_REGISTRY[self.args.hier_agent["alloc_policy"]](hier_shape, self.args)

    def forward(self, ep_batch, t=None, **kwargs):
        self.agent._capture_real_decision = isinstance(t, int) and kwargs.get("test_mode", False)
        self.agent._capture_time = t if isinstance(t, int) else 0
        result = super().forward(ep_batch, t, **kwargs)
        if isinstance(t, int):
            self._decision_q = result[0].detach()
        return result

    def _build_inputs(self, batch, t, target=False, imagine_inps=None):
        inputs, imagined = super()._build_inputs(batch, t, target=target, imagine_inps=imagine_inps)
        inputs["entities"] = inputs["entities"].masked_fill(inputs["entity_mask"].bool().unsqueeze(-1), 0)
        if imagined is not None:
            imagined["entities"] = inputs["entities"].repeat(2, 1, 1, 1)
        return inputs, imagined


def build_mac(scheme, groups, args):
    if getattr(args, "method", None) == "transfqmix":
        from open_score.algos.transfqmix import TransfQMixMAC
        return TransfQMixMAC(scheme, groups, args)
    if args.mac == "dcg_mac":
        from open_score.algos.dcg_patch.dcg_controller import DeepCoordinationGraphMAC
        return DeepCoordinationGraphMAC(scheme, groups, args)
    if args.mac == "spectra_mac":
        from open_score.algos.spectra_patch.spectra_controller import SPECTraMAC
        return SPECTraMAC(scheme, groups, args)
    return SharedEntityMAC(scheme, groups, args)


def build_encoder(name, input_shape, args):
    return {"flatten": FlattenEncoder, "attention": AttentionEncoder,
            "gnn": GraphEncoder}[name](input_shape, args)


def run_model_checks(only=None):
    """The approved stage-two V2/V3 numerical checks; no optimizer steps.

    Kept in the implementation module so the experiment has one verification
    entry rather than a separate test project. The caller records every row.
    ``only`` selects exact check IDs so missing checks can be appended without
    repeating already recorded validation.
    """
    import copy as copying
    import itertools
    from open_score.algos import load_config, make_scheme, METHODS
    from open_score.eval.protocol import POOL_ONLY_METHODS
    from open_score.envs.features import (MAX_AGENTS, MAX_BLUE, MAX_ENTITIES, MAX_TARGETS,
                                          ENTITY_DIM, task_masks)
    from components.episode_buffer import EpisodeBatch
    from .mixers import build_mixer

    rows = []
    selected = None if only is None else set(only)
    rng_state = th.get_rng_state()
    th.manual_seed(413)
    info = {"n_agents": MAX_AGENTS, "n_entities": MAX_ENTITIES, "entity_shape": ENTITY_DIM,
            "n_actions": 9, "state_shape": MAX_ENTITIES * ENTITY_DIM + 3, "episode_limit": 100,
            "feature_layout": "had", "n_tasks": MAX_TARGETS}

    def check(name, function):
        if selected is not None and name not in selected:
            return
        try:
            detail = function()
            rows.append(dict(id=name, status="passed", detail=str(detail or "verified")))
        except Exception as error:
            rows.append(dict(id=name, status="failed", detail=f"{type(error).__name__}: {error}"))

    def not_applicable(name, detail, condition=True):
        if selected is None or name in selected:
            rows.append(dict(id=name, status="not_applicable" if condition else "failed",
                             detail=detail if condition else "configuration contradicts this N/A declaration"))

    def sub_info(n_tasks=MAX_TARGETS):
        # One subtask per target slot, so a smaller subtask set is a smaller
        # entity table too, exactly as the environment pads it.
        entities = MAX_AGENTS + MAX_BLUE + n_tasks
        return {**info, "n_tasks": n_tasks, "n_entities": entities,
                "state_shape": entities * ENTITY_DIM + 3}

    def synthetic(counts=((4, 1), (40, 6)), multi_task=False, n_tasks=MAX_TARGETS):
        local = sub_info(n_tasks)
        n_entities = local["n_entities"]
        scheme, groups, preprocess = make_scheme(local, multi_task=multi_task)
        batch = EpisodeBatch(scheme, groups, 2, 3, preprocess=preprocess, device="cpu")
        ent = th.zeros(2, 3, n_entities, ENTITY_DIM)
        inactive = th.ones(2, 3, n_entities, dtype=th.uint8)
        for b, (n, k) in enumerate(counts):
            for start, count, kind in ((0, n, 0), (MAX_AGENTS, n, 1), (MAX_AGENTS + MAX_BLUE, k, 2)):
                inactive[b, :, start:start + count] = 0
                ent[b, :, start:start + count, :4] = th.rand(3, count, 4) * 1.6 - 0.8
                ent[b, :, start:start + count, 7 + kind] = 1
                if kind < 2:
                    ent[b, :, start:start + count, 4:6] = 1
                else:
                    ent[b, :, start:start + count, 2:4] = 0
        obs = inactive.bool().unsqueeze(-1) | inactive.bool().unsqueeze(-2)
        avail = th.zeros(2, 3, MAX_AGENTS, 9, dtype=th.int32)
        avail[..., 0] = 1
        avail = th.where((~inactive[..., :MAX_AGENTS].bool()).unsqueeze(-1), th.ones_like(avail), avail)
        acts = th.arange(MAX_AGENTS).reshape(1, 1, MAX_AGENTS, 1).expand(2, 3, -1, -1) % 9
        acts = acts.masked_fill(inactive[..., :MAX_AGENTS].bool().unsqueeze(-1), 0)
        batch.update({"entities": ent, "entity_mask": inactive, "obs_mask": obs.to(th.uint8),
                      "agent_mask": inactive[..., :MAX_AGENTS], "initial_agent_mask": inactive[..., :MAX_AGENTS],
                      "avail_actions": avail, "actions": acts,
                      "state": th.zeros(2, 3, local["state_shape"]), "reward": th.zeros(2, 3, 1),
                      "terminated": th.zeros(2, 3, 1, dtype=th.uint8),
                      "reset": th.zeros(2, 3, 1, dtype=th.uint8)})
        if multi_task:
            # The same subtask decomposition the environment exposes, plus a
            # decision at the first state as in the runner's clock.
            e2t = th.ones(2, 3, n_entities, n_tasks, dtype=th.uint8)
            active = th.ones(2, 3, n_tasks, dtype=th.uint8)
            for b, (_, k) in enumerate(counts):
                subtasks = task_masks(ent[b, 0].numpy(), inactive[b, 0].numpy(), min(k, n_tasks),
                                      MAX_AGENTS, MAX_BLUE, n_tasks)
                e2t[b] = th.as_tensor(subtasks["entity2task_mask"])
                active[b] = th.as_tensor(subtasks["task_mask"])
            decision = th.zeros(2, 3, 1, dtype=th.uint8)
            decision[:, 0] = 1
            batch.update({"entity2task_mask": e2t, "task_mask": active, "hier_decision": decision,
                          "task_rewards": th.zeros(2, 3, n_tasks),
                          "tasks_terminated": th.zeros(2, 3, n_tasks, dtype=th.uint8)})
        return batch, scheme, groups

    batch, scheme, groups = synthetic()
    # An ordered pool-sized baseline has no parameters for 40v40 K6, so its
    # own checks use the largest roster its slots actually cover.
    pool_batch = synthetic(((4, 1), (10, 3)))[0]
    # The hierarchical arm needs the subtask keys in its replay.
    task_batch, task_scheme, _ = synthetic(multi_task=True)

    def make(name, overrides=None, info_overrides=None):
        args = load_config(name, {"use_cuda": False, "t_max": 20000, **(overrides or {})})
        for key, value in {**info, **(info_overrides or {})}.items():
            setattr(args, key, value)
        model_scheme = copying.deepcopy(scheme)
        model_scheme["entities"]["vshape"] = args.entity_shape
        model_scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
        mac = build_mac(model_scheme, groups, args)
        mac.eval()
        mixer = build_mixer(args)
        if mixer is not None:
            mixer.eval()
        return args, mac, mixer

    def forward(mac, data):
        mac.init_hidden(data.batch_size)
        with th.no_grad():
            result = mac.forward(data, 0)[0]
        assert result.shape == (2, MAX_AGENTS, 9) and th.isfinite(result).all()
        return result

    def permuted(data, permutation):
        other = copying.deepcopy(data)
        keys = ("entities", "entity_mask", "entity2task_mask")
        for key in (key for key in keys if key in data.scheme):
            other.data.transition_data[key] = data[key][:, :, permutation].clone()
        other.data.transition_data["obs_mask"] = data["obs_mask"][:, :, permutation][:, :, :, permutation].clone()
        agent_order = permutation[:MAX_AGENTS]
        for key in ("agent_mask", "initial_agent_mask", "avail_actions", "actions", "actions_onehot"):
            other.data.transition_data[key] = data[key][:, :, agent_order].clone()
        return other

    try:
        for name in METHODS:
            try:
                args, mac, mixer = make(name)
            except Exception as error:
                rows.append(dict(id=f"V2.{name}.construct", status="failed", detail=str(error)))
                continue

            data = pool_batch if name in POOL_ONLY_METHODS else (task_batch if args.multi_task else batch)
            largest = "10v10 K3" if name in POOL_ONLY_METHODS else "40v40 K6"

            def sizes():
                before = sum(p.numel() for p in mac.parameters())
                forward(mac, data)
                assert before == sum(p.numel() for p in mac.parameters())
                return f"same {before} parameters for 4v4 K1 and {largest}"
            check(f"V2.{name}.variable_count", sizes)
            if name == "b0_qmix":
                not_applicable("V2.b0_qmix.permutation",
                               "ordered flatten/QMIX baseline intentionally lacks permutation invariance")
                not_applicable("V2.b0_qmix.mixer_alignment",
                               "ordinary QMIX assigns ordered agent slots; permutation alignment is not a B0 requirement")
                check("V3.b0_qmix.no_identity_history", lambda: _assert_condition(
                    not args.obs_agent_id and not args.obs_last_action and not args.entity_last_action))
            else:
                def entity_permutation():
                    perm = th.cat((th.arange(MAX_AGENTS), th.arange(MAX_ENTITIES - 1, MAX_AGENTS - 1, -1)))
                    th.testing.assert_close(forward(mac, data), forward(mac, permuted(data, perm)), atol=1e-5, rtol=0)
                check(f"V2.{name}.entity_permutation", entity_permutation)

                def agent_permutation():
                    perm = th.arange(MAX_ENTITIES)
                    perm[0], perm[1] = 1, 0
                    expected = forward(mac, data)[:, perm[:MAX_AGENTS]]
                    th.testing.assert_close(expected, forward(mac, permuted(data, perm)), atol=1e-5, rtol=0)
                check(f"V2.{name}.agent_equivariance", agent_permutation)

            def death():
                dead_batch = copying.deepcopy(data)
                dead_batch.data.transition_data["entity_mask"][:, :, 0] = 1
                dead_batch.data.transition_data["entities"][:, :, 0] = 0
                dead_batch.data.transition_data["obs_mask"][:, :, 0] = 1
                dead_batch.data.transition_data["obs_mask"][:, :, :, 0] = 1
                dead_batch.data.transition_data["avail_actions"][:, :, 0] = 0
                dead_batch.data.transition_data["avail_actions"][:, :, 0, 0] = 1
                mac.init_hidden(2)
                with th.no_grad():
                    actions = mac.select_actions(dead_batch, 0, 0, test_mode=True)
                assert (actions[:, 0] == 0).all()
                history = mac.hidden_states.detach().clone()
                values = mac.evaluation_values(dead_batch, 0, actions, [0, 1], mixer)
                assert th.isfinite(th.tensor(values["q_tot"])).all()
                th.testing.assert_close(mac.hidden_states, history, atol=0, rtol=0)
            check(f"V3.{name}.death_noop", death)

            if not args.agent["recurrent"]:
                not_applicable(f"V3.{name}.recurrent_state",
                               "GRU disabled by this configuration; no temporal state to reset")
            else:
                def episode_hidden_reset():
                    assert args.agent["recurrent"]
                    with th.no_grad():
                        mac.init_hidden(2)
                        assert (mac.hidden_states == 0).all()
                        fresh_q = mac.forward(data, 0)[0].clone()
                        fresh_hidden = mac.hidden_states.clone()
                        mac.forward(data, 1)
                        mac.forward(data, 2)
                        assert (mac.hidden_states != 0).any()
                        assert not th.allclose(mac.hidden_states, fresh_hidden, atol=1e-7, rtol=0)
                        # Exercise the same boundary used by runner.run(): a
                        # new episode starts by clearing persistent slots.
                        mac.init_hidden(2)
                        assert (mac.hidden_states == 0).all()
                        reset_q = mac.forward(data, 0)[0]
                        th.testing.assert_close(reset_q, fresh_q, atol=1e-6, rtol=0)
                        th.testing.assert_close(mac.hidden_states, fresh_hidden, atol=1e-6, rtol=0)
                    return "after three recurrent steps, init_hidden clears all slots and reproduces fresh-episode Q and hidden state"
                check(f"V3.{name}.episode_hidden_reset", episode_hidden_reset)

            if name != "spectra":
                def encoder_execution_equivalence():
                    reference = copying.deepcopy(mac)
                    optimized = copying.deepcopy(mac)
                    reference.args.encoder_chunk_size = 256
                    reference.args.encoder_skip_dead = False
                    optimized.args.encoder_chunk_size = 1024
                    optimized.args.encoder_skip_dead = True
                    reference.train()
                    optimized.train()
                    imagined = None
                    if name == "refil":
                        imagined = reference.agent.make_imagined_inputs(data)[0]

                    def outputs(model):
                        model.init_hidden(2)
                        if name == "dcg":
                            return th.stack([model.forward(data, t, actions=data["actions"][:, t],
                                                           policy_mode=False, compute_grads=True)
                                             for t in range(3)], dim=1)
                        return model.forward(data, None, imagine_inps=copying.deepcopy(imagined))[0]

                    ref_q, opt_q = outputs(reference), outputs(optimized)
                    th.testing.assert_close(ref_q, opt_q, atol=1e-6, rtol=1e-5)
                    weights = th.linspace(-1, 1, ref_q.numel()).reshape_as(ref_q)
                    (ref_q * weights).mean().backward()
                    (opt_q * weights).mean().backward()
                    max_grad = 0.0
                    for ref_parameter, opt_parameter in zip(reference.parameters(), optimized.parameters()):
                        assert (ref_parameter.grad is None) == (opt_parameter.grad is None)
                        if ref_parameter.grad is not None:
                            th.testing.assert_close(ref_parameter.grad, opt_parameter.grad, atol=1e-6, rtol=1e-4)
                            max_grad = max(max_grad, float((ref_parameter.grad - opt_parameter.grad).abs().max()))
                    # Cover the zero-live-query branch and its parameter
                    # gradient semantics directly, without a learner update.
                    ref_encoder = reference.agent.encoder if name == "dcg" else reference.agent._base
                    opt_encoder = optimized.agent.encoder if name == "dcg" else optimized.agent._base
                    empty, _ = reference._build_inputs(data, slice(0, 1))
                    empty["entity_mask"] = th.ones_like(empty["entity_mask"])
                    empty["obs_mask"] = th.ones_like(empty["obs_mask"])
                    for encoder in (ref_encoder, opt_encoder):
                        encoder.zero_grad(set_to_none=True)
                        result = encoder(empty)
                        assert (result == 0).all()
                        result.sum().backward()
                        assert all(p.grad is not None and (p.grad == 0).all() for p in encoder.parameters())
                    return (f"chunk256/skip=False versus chunk1024/skip=True: Q max abs diff "
                            f"{float((ref_q.detach() - opt_q.detach()).abs().max()):.3g}; parameter gradient max abs diff "
                            f"{max_grad:.3g}; empty-query outputs and zero gradients preserved")
                check(f"V3.{name}.encoder_execution_equivalence", encoder_execution_equivalence)

            if mixer is not None:
                def monotone():
                    inputs, _ = mac._build_inputs(data, slice(0, 1))
                    qi = th.randn(2, 1, MAX_AGENTS, requires_grad=True)
                    total = mixer(qi, inputs)
                    gradient = th.autograd.grad(total.sum(), qi)[0]
                    assert th.isfinite(total).all() and (gradient >= -1e-7).all()
                check(f"V2.{name}.monotonicity", monotone)
                if name != "b0_qmix":
                    def mixer_alignment():
                        perm = th.arange(MAX_ENTITIES)
                        perm[0], perm[1] = 1, 0
                        inputs, _ = mac._build_inputs(data, slice(0, 1))
                        other_inputs, _ = mac._build_inputs(permuted(data, perm), slice(0, 1))
                        # Mixed signs exercise the nonlinear mixer; a large
                        # positive sum can round tiny row differences away.
                        qi = th.zeros(2, 1, MAX_AGENTS)
                        qi[:, :, 0], qi[:, :, 1] = -40., 60.
                        original = mixer(qi, inputs)
                        aligned = mixer(qi[:, :, perm[:MAX_AGENTS]], other_inputs)
                        th.testing.assert_close(original, aligned, atol=1e-5, rtol=1e-6)
                        misaligned = mixer(qi[:, :, perm[:MAX_AGENTS]], inputs)
                        assert not th.allclose(original, misaligned, atol=1e-8, rtol=0), (original, misaligned)
                    check(f"V2.{name}.mixer_alignment", mixer_alignment)

            if name in ("b2_qmix_atten", "refil", "alma"):
                def last_actions():
                    inputs, _ = mac._build_inputs(data, slice(0, 3))
                    last = inputs["entities"][..., 10:]
                    assert (last[:, 0] == 0).all() and (last[:, :, MAX_AGENTS:] == 0).all()
                    expected = data["actions_onehot"][:, :-1].masked_fill(
                        data["entity_mask"][:, 1:, :MAX_AGENTS].bool().unsqueeze(-1), 0)
                    th.testing.assert_close(last[:, 1:, :MAX_AGENTS], expected)
                    assert args.entity_last_action
                check(f"V3.{name}.last_action", last_actions)
                def attention_mask():
                    masked = copying.deepcopy(data)
                    masked.data.transition_data["obs_mask"][:, :, :, MAX_AGENTS] = 1
                    changed = copying.deepcopy(masked)
                    changed.data.transition_data["entities"][:, :, MAX_AGENTS] += 777
                    th.testing.assert_close(forward(mac, masked), forward(mac, changed), atol=1e-5, rtol=0)
                check(f"V3.{name}.attention_mask", attention_mask)
            if name == "refil":
                def imagination():
                    imagined, masks = mac.agent.make_imagined_inputs(batch)
                    within, cross = masks
                    assert imagined["entities"].shape[0] == 4
                    th.testing.assert_close(within[:, 0], within[:, 1])
                    th.testing.assert_close(cross[:, 0], cross[:, 1])
                    visible = (~batch["entity_mask"].bool()).unsqueeze(-1) & (~batch["entity_mask"].bool()).unsqueeze(-2)
                    assert ((~within & ~cross) & visible).sum() == 0
                    assert th.equal((~within | ~cross) & visible, visible)
                    assert args.lmbda == 0.5 and args.training_iters == 8
                check("V3.refil.episode_partition_and_lambda", imagination)

            if name == "alma":
                def allocation_gates_observation():
                    # The hierarchy must have a causal path into behaviour:
                    # a different allocation has to change the low-level Q,
                    # and an agent must not see outside its own subtask.
                    def assigned(source, task, rows=slice(0, MAX_AGENTS)):
                        other = copying.deepcopy(source)
                        block = other.data.transition_data["entity2task_mask"]
                        block[1, :, rows] = 1
                        block[1, :, rows, task] = 0
                        return other
                    first, second = assigned(data, 0), assigned(data, 1)
                    changed = float((forward(mac, first)[1] - forward(mac, second)[1]).abs().max())
                    assert changed > 1e-6, "the allocation does not reach the low-level policy"
                    # A Blue entity that belongs only to subtask 1 must not
                    # affect agents that the allocation put on subtask 0.
                    hidden = assigned(first, 1, rows=MAX_AGENTS)
                    perturbed = copying.deepcopy(hidden)
                    perturbed.data.transition_data["entities"][1, :, MAX_AGENTS] += 777
                    th.testing.assert_close(forward(mac, hidden)[1], forward(mac, perturbed)[1],
                                            atol=1e-5, rtol=0)
                    return (f"subtask 0 versus subtask 1 moves Q by up to {changed:.3g}; "
                            "an entity outside the agent's subtask is invisible")
                check("V3.alma.allocation_gates_observation", allocation_gates_observation)

                def allocation_validity():
                    fresh = copying.deepcopy(data)
                    fresh.data.transition_data["entity_mask"][:, :, 0] = 1
                    mac.init_hidden(2)
                    with th.no_grad():
                        mac.select_actions(fresh, 0, 0, test_mode=True)
                    alloc = mac.task_allocations
                    live = ~fresh["entity_mask"][:, 0, :MAX_AGENTS].bool()
                    counts = alloc.sum(-1)
                    assert (counts[live] == 1).all() and (counts[~live] == 0).all()
                    inactive = fresh["task_mask"][:, 0].bool().unsqueeze(1).expand_as(alloc)
                    assert (alloc[inactive] == 0).all()
                    stored = 1 - fresh["entity2task_mask"][:, 0, :MAX_AGENTS].float()
                    th.testing.assert_close(stored, alloc.cpu(), atol=0, rtol=0)
                    return ("every live agent holds exactly one active subtask, dead agents hold "
                            "none, and the decision is written back into the replay")
                check("V3.alma.allocation_validity", allocation_validity)

        def subtask_width_extrapolation():
            # Training pads to 3 target slots while evaluation has 6. ALMA's
            # spare subtask embeddings keep the one-hot width fixed, which is
            # what lets best.pt be evaluated on the K=4/6 extrapolation line.
            trained_args, trained_mac, _ = make("alma", info_overrides=sub_info(3))
            assert trained_args.n_extra_tasks == 3
            forward(trained_mac, synthetic(((4, 1), (40, 3)), multi_task=True, n_tasks=3)[0])
            width = int(trained_args.n_tasks) + int(trained_args.n_extra_tasks)
            # The same rebuild load_policy performs for the evaluation pad.
            _, eval_mac, _ = make("alma", {"n_extra_tasks": width - MAX_TARGETS},
                                  info_overrides=sub_info(MAX_TARGETS))
            eval_mac.load_state_dict(trained_mac.state_dict())
            assert th.isfinite(forward(eval_mac, task_batch)).all()
            return f"{width} subtask embeddings trained at K<=3 load and run unchanged at K=6"
        check("V3.alma.subtask_width_extrapolation", subtask_width_extrapolation)

        def dcg_numerics():
            args, dcg, _ = make("dcg")
            dcg.n_agents = 3
            dcg.n_actions = 3
            dcg.args.cg_edges = "star"
            dcg._set_edges([(0, 1), (0, 2)])
            dcg.node_active = th.ones(1, 3, dtype=th.bool)
            fi, fij = th.randn(1, 3, 3), th.randn(1, 2, 3, 3)
            action = dcg.greedy(fi, fij, th.ones_like(fi))
            candidates = [th.tensor(a).reshape(1, 3, 1) for a in itertools.product(range(3), repeat=3)]
            optimum = max(float(dcg.q_values(fi, fij, a)) for a in candidates)
            assert abs(float(dcg.q_values(fi, fij, action)) - optimum) < 1e-6
            # Tied leaves must be decoded conditional on the chosen root.
            dcg.greedy(th.zeros_like(fi), th.zeros_like(fij), th.ones_like(fi))
            dcg.args.cg_edges = "full"
            dcg._set_edges([(0, 1), (0, 2), (1, 2)])
            dcg.node_active = th.tensor([[True, True, False]])
            legal = th.ones(1, 3, 3)
            legal[:, 2, 1:] = 0
            result = dcg.greedy(fi, th.randn(1, 3, 3, 3), legal)
            assert (result[:, 2] == 0).all()
            for n in (2, 3):
                dcg.node_active = th.tensor([[i < n for i in range(3)]])
                value = dcg.q_values(th.ones(1, 3, 3), th.ones(1, 3, 3, 3), th.zeros(1, 3, 1, dtype=th.long))
                th.testing.assert_close(value, th.tensor([2.0]))
        check("V3.dcg.star_exact_padding_normalization", dcg_numerics)

        def payoff_symmetry():
            args, dcg, _ = make("dcg")
            hidden = th.randn(2, MAX_AGENTS, 64)
            pair = dcg.payoffs(hidden)
            old_from, old_to = dcg.edges_from, dcg.edges_to
            dcg.edges_from, dcg.edges_to = old_to, old_from
            reverse = dcg.payoffs(hidden).transpose(-1, -2)
            th.testing.assert_close(pair, reverse, atol=1e-6, rtol=0)
        check("V3.dcg.payoff_symmetry", payoff_symmetry)

        def initial_graph_sequence():
            _, dcg, _ = make("dcg")
            assert dcg.args.cg_edges == "full"
            parameter_ids = tuple(id(p) for p in dcg.parameters())
            counts = []
            for n in (4, 6, 8, 10):
                data, _, _ = synthetic(((n, 2), (n, 2)))
                # Agent zero dies after initialization; its initial graph
                # node and edges remain while its recurrent slot is cleared.
                data.data.transition_data["entity_mask"][:, 1:, 0] = 1
                data.data.transition_data["entities"][:, 1:, 0] = 0
                data.data.transition_data["obs_mask"][:, 1:, 0] = 1
                data.data.transition_data["obs_mask"][:, 1:, :, 0] = 1
                data.data.transition_data["avail_actions"][:, 1:, 0] = 0
                data.data.transition_data["avail_actions"][:, 1:, 0, 0] = 1
                dcg.init_hidden(2)
                assert (dcg.hidden_states == 0).all()
                with th.no_grad():
                    policy = dcg.forward(data, 0)[0]
                    nodes, edges = dcg._masks(dcg.last_utilities)
                    expected_nodes = (th.arange(MAX_AGENTS) < n).expand(2, -1)
                    th.testing.assert_close(nodes, expected_nodes)
                    assert (edges.sum(-1) == n * (n - 1) // 2).all()
                    assert th.isfinite(policy).all() and (dcg.hidden_states[:, n:] == 0).all()
                    # Padding must affect neither the node nor edge mean.
                    fi = th.ones_like(dcg.last_utilities).masked_fill(~nodes[..., None], 1000)
                    fij = th.ones_like(dcg.last_payoffs).masked_fill(~edges[..., None, None], 1000)
                    total = dcg.q_values(fi, fij, th.zeros(2, MAX_AGENTS, 1, dtype=th.long))
                    th.testing.assert_close(total, th.full((2,), 2.0))
                    dead_policy = dcg.forward(data, 1)[0]
                    after_nodes, after_edges = dcg._masks(dcg.last_utilities)
                    th.testing.assert_close(after_nodes, nodes)
                    th.testing.assert_close(after_edges, edges)
                    assert (dcg.hidden_states[:, 0] == 0).all()
                    assert (dead_policy[:, 0].argmax(-1) == 0).all()
                    assert th.isfinite(dcg.last_utilities).all() and th.isfinite(dcg.last_payoffs).all()
                assert tuple(id(p) for p in dcg.parameters()) == parameter_ids
                counts.append(f"N={n}: {n} nodes/{n * (n - 1) // 2} edges")
            return "; ".join(counts) + "; same parameters, reset between episodes, death retains initial graph, padding excluded from means"
        check("V3.dcg.initial_graph_sequence", initial_graph_sequence)

        def knn_fewer_neighbours():
            args, mac, _ = make("gnn_qmix", {"gnn_edges": "knn", "gnn_k": 10})
            q = forward(mac, batch)
            assert th.isfinite(q).all()
            # The first scenario has nine entities, hence fewer than k+1.
            altered = copying.deepcopy(batch)
            altered.data.transition_data["entities"][0, :, 10] = 999
            th.testing.assert_close(q[0], forward(mac, altered)[0], atol=1e-5, rtol=0)
        check("V3.gnn.knn_actual_neighbours", knn_fewer_neighbours)

        def saqa():
            from open_score.algos.spectra_patch.spectra_attention import CrossAttentionBlock
            block = CrossAttentionBlock(64, 4)
            query, entities = th.randn(2, 1, 64), th.randn(2, 7, 64)
            valid = th.ones(2, 1, 7, dtype=th.bool)
            perm = th.arange(6, -1, -1)
            th.testing.assert_close(block(query, entities, valid), block(query, entities[:, perm], valid[:, :, perm]), atol=1e-5, rtol=0)
        check("V3.spectra.SAQA", saqa)

        def lambda_boundaries():
            from types import SimpleNamespace
            from open_score.algos.spectra_patch.rl_utils import build_td_lambda_targets
            rewards = th.tensor([[[1.], [0.]], [[1.], [2.]]])
            mask = th.tensor([[[1.], [0.]], [[1.], [1.]]])
            done = th.tensor([[[0.], [0.]], [[0.], [1.]]])
            boot = th.ones(2, 3, 1) * 10
            targets = build_td_lambda_targets(SimpleNamespace(n_reward=1), rewards, done, mask, boot, .99, .6)
            th.testing.assert_close(targets[0, 0], th.tensor([10.9]))
            th.testing.assert_close(targets[1, 1], th.tensor([2.]))
        check("V3.spectra.TD_lambda_truncation", lambda_boundaries)
    finally:
        th.set_rng_state(rng_state)
    return rows


def _assert_condition(value):
    assert value

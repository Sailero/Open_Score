"""HAD entity attention and ALMA controller bridge for PURE methods.

Attention and imagined branches are inherited from ALMA commit
81bd5c475972b982ce470b1ed78154ec39d236da (MIT; see the vendored LICENSE).
"""
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from controllers.entity_controller import EntityMAC
from modules.agents.agent import Agent as ALMAAgent
from modules.layers.attention import EntityAttentionLayer

from ..leaf import GlobalBranch, intent_auxiliary, intent_targets


def relative_entity_views(inputs, args, extras=()):
    """Construct temporary observer-centred views; replay stays global.

    Each observer is moved to row zero. Other geometry is relative to it;
    its own absolute position and velocity are retained, so own speed is not
    lost by subtracting it from itself.
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




class TemporalHead(nn.Module):
    """GRU head retaining the persistent HAD agent slots."""
    def __init__(self, args, output_features=None):
        super().__init__()
        self.args = args
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_out = nn.Linear(args.rnn_hidden_dim, output_features or args.n_actions)

    def action_values(self, hidden):
        return self.fc_out(hidden)

    def forward(self, x, inputs):
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
        return self.action_values(hs), h.reshape(bs, na, hd)

    def step(self, x, hidden, alive):
        bs, na, hd = x.shape
        if hidden.shape[0] != bs:
            hidden = hidden.repeat(bs // hidden.shape[0], 1, 1)
        flat = self.rnn(x.reshape(bs * na, hd), hidden.reshape(bs * na, hd))
        flat = flat.masked_fill(~alive.reshape(-1, 1), 0)
        hidden = flat.reshape(bs, na, hd)
        return self.action_values(hidden), hidden


class EntityAgent(ALMAAgent):
    def __init__(self, input_shape, args):
        # Preserve ALMA's imagined-input construction, subtask conditioning
        # and combined forward.
        input_shape = int(getattr(args, "actor_entity_shape", None) or
                          getattr(args, "observer_entity_shape", None) or input_shape)
        super().__init__(input_shape, args, recurrent=False, entity_scheme=True,
                         subtask_cond=args.agent.get("subtask_cond"))
        kind = getattr(args, "encoder", "attention")
        self._base = build_encoder(kind, input_shape, args)
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
        self.loop_execution = None
        self.loop_depth_override = None
        self.loop_generator = None
        self.last_loop_stats = {}

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
        encoded = self._base(inputs)
        encoded = self._apply_count_cond(encoded, inputs["entity_mask"])
        if hasattr(self, "matched_mlp"):
            encoded = encoded + self.matched_mlp(encoded)
            encoded = encoded.masked_fill(inputs["entity_mask"][:, :, :self.args.n_agents].unsqueeze(-1), 0)
        if hasattr(self, "task_cond") and "task_embeds" in inputs:
            encoded = encoded + self.task_cond(inputs["task_embeds"][:, :, :self.args.n_agents])
        q, hidden = self._head(encoded, inputs)
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

    def _loop_depths(self, inputs, shape, device, evaluation):
        bs, ts, na = shape
        execution = self.loop_execution if evaluation else None
        execution = {} if execution is None else execution
        mode = execution.get("mode", "fixed")
        if mode not in ("fixed", "adaptive", "random"):
            raise ValueError("loop execution mode must be fixed, adaptive or random")
        override = self.loop_depth_override
        if override is not None:
            if ts != 1 or not isinstance(override, th.Tensor) or override.shape != (bs, na):
                raise ValueError("loop_depth_override must be a tensor [batch,agents] for one decision")
            if override.dtype != th.int64:
                raise ValueError("loop_depth_override must be int64")
            depth = override.to(device).unsqueeze(1)
            mode = "fixed"
            self.loop_depth_override = None
        elif mode == "random":
            probabilities = th.as_tensor(execution.get("probabilities", ()), dtype=th.float64, device="cpu")
            if (probabilities.shape != (4,) or not bool(th.isfinite(probabilities).all())
                    or bool((probabilities < 0).any()) or float(probabilities.sum()) <= 0):
                raise ValueError("random loop probabilities need four finite nonnegative values with positive sum")
            if self.loop_generator is None:
                self.loop_generator = th.Generator(device="cpu").manual_seed(int(getattr(self.args, "seed", 0)))
            depth = (th.multinomial(probabilities, bs * ts * na, replacement=True,
                                    generator=self.loop_generator) + 1).to(device).reshape(bs, ts, na)
        elif evaluation:
            depth = th.full((bs, ts, na), int(execution.get("depth", getattr(self.args, "global_eval_depth", 4))),
                            device=device, dtype=th.int64)
        elif "loop_depth" in inputs:
            saved = inputs["loop_depth"]
            if saved.shape != (bs, ts, na, 1) or saved.dtype != th.int64:
                raise ValueError("loop_depth must be int64 [batch,time,agents,1]")
            depth = saved[..., 0].to(device)
        else:
            # Retain direct tensor/offline callers; learner replay must supply
            # its recorded per-decision schedule through loop_depth.
            depth = self.global_net.expand_depth(getattr(self, "cycle_depth", 4), bs, ts, na, device)
            depth = depth.reshape(bs, ts, na)
        if bool(((depth < 1) | (depth > 4)).any()):
            raise ValueError("main1009 loop depth must be between one and four")
        if mode == "adaptive":
            threshold = float(execution.get("threshold", -1))
            if not 0 <= threshold < float("inf"):
                raise ValueError("adaptive loop threshold must be finite and nonnegative")
        return mode, depth, execution

    def _loop_shared_cost(self, stats, inputs, live, entities):
        if stats is None:
            return
        rows = int(live.sum().item())
        total = int(live.numel())
        encoded_rows = rows if getattr(self.args, "encoder_skip_dead", True) else total
        dim = int(self.args.attn_embed_dim)
        hidden = int(self.args.rnn_hidden_dim)
        feature = self._base.fc1.in_features
        self.global_net._loop_add(stats, decisions=rows,
                                 macs=encoded_rows * entities * feature * dim)
        if not getattr(self.args, "skip_refil_local", False):
            # REFIL in_trans computes Q/K/V for all padded entities, although
            # encode_bundle asks the attention layer for only the own query.
            self.global_net._loop_add(stats,
                attention_entries=encoded_rows * entities * self._base.attn.n_heads,
                macs=encoded_rows * (3 * entities * dim * dim + 2 * entities * dim + dim * dim))
        self.global_net._loop_add(stats, macs=encoded_rows * dim * hidden)
        global_dim = self.global_net.fuse.in_features
        if isinstance(self.global_net.token_proj, nn.Linear):
            # Projection is applied to the full encoded bundle, including
            # zero padded observer rows, not just the compact workspaces.
            self.global_net._loop_add(stats, macs=total * entities * dim * global_dim + rows * dim * global_dim)
        # Independent own token + no-memory query projection.
        self.global_net._loop_add(stats, macs=rows * (feature * dim + (global_dim + hidden) * global_dim))
        if hasattr(self, "task_cond") and "task_embeds" in inputs:
            self.global_net._loop_add(stats, macs=total * self.task_cond.in_features * self.task_cond.out_features)

    def _loop_temporal_cost(self, stats, rows):
        if stats is None:
            return
        hidden = int(self.args.rnn_hidden_dim)
        macs = rows * 6 * hidden * hidden
        macs += rows * hidden * self._head.fc_out.out_features
        self.global_net._loop_add(stats, temporal_previews=rows, macs=macs)

    def _loop_spatial_fixed(self, initial, mask, b0, depths, stats):
        if self.training and th.is_grad_enabled():
            # Checkpoint the complete per-observer loop, not only its layers:
            # otherwise the compacted copies of every previous workspace stay
            # alive until backward across the full B*T*A replay batch.
            chunk = int(getattr(self.args, "encoder_chunk_size", 1024))
            if chunk < 1:
                raise ValueError("encoder_chunk_size must be positive")
            parts = []
            for start in range(0, initial.shape[0], chunk):
                sl = slice(start, start + chunk)
                parts.append(checkpoint(self._loop_spatial_fixed_chunk,
                    initial[sl], mask[sl], b0[sl], depths[sl], None,
                    use_reentrant=False))
            return th.cat(parts, 0)
        return self._loop_spatial_fixed_chunk(initial, mask, b0, depths, stats)

    def _loop_spatial_fixed_chunk(self, initial, mask, b0, depths, stats):
        rows, entities, dim = initial.shape
        result = initial.new_zeros(rows, dim)
        ids = th.arange(rows, device=initial.device)
        work = self.global_net.loop_initialize(initial, mask, b0, stats)
        for round_id in range(1, 5):
            read = self.global_net.loop_round(work, stats)
            done = depths.index_select(0, ids) == round_id
            picked = done.nonzero(as_tuple=False).flatten()
            result = result.index_copy(0, ids.index_select(0, picked), read.index_select(0, picked))
            if stats is not None:
                self.global_net._loop_add(stats, **{f"depth_{round_id}": picked.numel()})
            keep = (~done).nonzero(as_tuple=False).flatten()
            if keep.numel() == 0:
                break
            work = self.global_net.loop_compact(work, keep)
            read = read.index_select(0, keep)
            self.global_net.loop_continue(work, read, stats)
            ids = ids.index_select(0, keep)
        return result

    @staticmethod
    def _loop_stable(q, previous_q, hidden, previous_hidden, legal, threshold):
        count = legal.sum(-1, keepdim=True)
        def normalized(values):
            values = values.masked_fill(~legal, 0)
            centered = (values - values.sum(-1, keepdim=True) / count.clamp_min(1)).masked_fill(~legal, 0)
            norm = centered.norm(dim=-1)
            return centered / norm.clamp_min(1e-6).unsqueeze(-1), norm
        current, current_norm = normalized(q)
        previous, previous_norm = normalized(previous_q)
        masked = q.masked_fill(~legal, float("-inf"))
        old_masked = previous_q.masked_fill(~legal, float("-inf"))
        top = masked.topk(min(2, q.shape[-1]), dim=-1).values
        old_top = old_masked.topk(min(2, q.shape[-1]), dim=-1).values
        if top.shape[-1] < 2:
            return legal.new_zeros(q.shape[0])
        not_tied = ((top[:, 0] - top[:, 1] > 1e-6)
                    & (old_top[:, 0] - old_top[:, 1] > 1e-6))
        q_change = (current - previous).norm(dim=-1)
        h_change = (hidden - previous_hidden).norm(dim=-1) / previous_hidden.norm(dim=-1).clamp_min(1e-6)
        return ((count[:, 0] > 1) & (current_norm > 1e-6) & (previous_norm > 1e-6)
                & not_tied & (masked.argmax(-1) == old_masked.argmax(-1))
                & (th.maximum(q_change, h_change) <= threshold))

    def _loop_execute_online(self, initial, mask, b0, local, alive, h_prev, legal,
                             depth, mode, execution, stats):
        bs, na, hidden_dim = local.shape
        n_act = int(self.args.n_actions)
        ids = alive.reshape(-1).nonzero(as_tuple=False).flatten()
        chosen_q = local.new_zeros(bs * na, n_act)
        chosen_h = local.new_zeros(bs * na, hidden_dim)
        chosen_read = initial.new_zeros(bs * na, initial.shape[-1])
        chosen_depth = th.zeros(bs * na, device=initial.device, dtype=th.int64)
        if ids.numel() == 0:
            return (chosen_q.reshape(bs, na, n_act), chosen_h.reshape(bs, na, hidden_dim),
                    chosen_read, chosen_depth)
        work = self.global_net.loop_initialize(initial.index_select(0, ids), mask.index_select(0, ids),
                                               b0.index_select(0, ids), stats)
        flat_local, flat_h = local.reshape(-1, hidden_dim), h_prev.reshape(-1, hidden_dim)
        legal = legal.reshape(bs * na, n_act).bool()
        depths = depth.reshape(-1)
        previous_q = previous_hidden = None
        for round_id in range(1, 5):
            read = self.global_net.loop_round(work, stats)
            fused = flat_local.index_select(0, ids) + self.global_net.fuse(read)
            n_active = ids.numel()
            q, candidate_h = self._head.step(fused.unsqueeze(1), flat_h.index_select(0, ids).unsqueeze(1),
                                            alive.new_ones(n_active, 1))
            q, candidate_h = q[:, 0], candidate_h[:, 0]
            self._loop_temporal_cost(stats, n_active)
            self.global_net._loop_add(stats, macs=n_active * self.global_net.fuse.in_features * hidden_dim)
            if mode == "adaptive":
                done = th.zeros(n_active, dtype=th.bool, device=q.device)
                if round_id >= 2:
                    done = self._loop_stable(q, previous_q, candidate_h, previous_hidden,
                                             legal.index_select(0, ids), float(execution["threshold"]))
                if round_id == 4:
                    done = th.ones_like(done)
            else:
                done = depths.index_select(0, ids) == round_id
            picked = done.nonzero(as_tuple=False).flatten()
            destination = ids.index_select(0, picked)
            chosen_q = chosen_q.index_copy(0, destination, q.index_select(0, picked))
            chosen_h = chosen_h.index_copy(0, destination, candidate_h.index_select(0, picked))
            chosen_read = chosen_read.index_copy(0, destination, read.index_select(0, picked))
            chosen_depth = chosen_depth.index_fill(0, destination, round_id)
            self.global_net._loop_add(stats, **{f"depth_{round_id}": picked.numel()})
            keep = (~done).nonzero(as_tuple=False).flatten()
            if keep.numel() == 0:
                break
            previous_q, previous_hidden = q.index_select(0, keep), candidate_h.index_select(0, keep)
            work = self.global_net.loop_compact(work, keep)
            self.global_net.loop_continue(work, read.index_select(0, keep), stats)
            ids = ids.index_select(0, keep)
        return chosen_q.reshape(bs, na, n_act), chosen_h.reshape(bs, na, hidden_dim), chosen_read, chosen_depth

    def _capture_loop_exit(self, tokens, mask, types, origin, alive, h_prev, q, read, depth, t):
        rows = self._capture_rows(alive)
        if rows is None:
            return
        def picked(tensor):
            return tensor.index_select(0, rows).detach().cpu()
        na = alive.shape[-1]
        self.last_capture.append({
            "capture_kind": "selected_loop_exit", "loop_core": self.global_net.loop_core,
            "time_id": int(getattr(self, "_capture_time", 0)) + t,
            "batch_ids": (rows // na).detach().cpu(), "observer_ids": (rows % na).detach().cpu(),
            "H": picked(tokens).masked_fill(picked(mask).unsqueeze(-1), 0).unsqueeze(1),
            "u": None, "alpha": None, "read_attn": None,
            "selected_u": picked(read), "selected_depth": picked(depth),
            "h_prev": picked(h_prev.reshape(-1, h_prev.shape[-1])),
            "q": picked(q.reshape(-1, q.shape[-1])), "key_mask": picked(mask),
            "types": picked(types), "origin": picked(origin),
            "enemy_ids": th.arange(tokens.shape[1] - na, dtype=th.long),
        })

    def _compute_loop(self, inputs):
        if "loop_filled" in inputs:
            filled = inputs["loop_filled"]
            if filled.shape != (*inputs["entities"].shape[:2], 1) or filled.dtype != th.bool:
                raise ValueError("loop_filled must be bool [batch,time,1]")
            inputs = dict(inputs)
            inputs["entity_mask"] = inputs["entity_mask"] | ~filled
            inputs["obs_mask"] = inputs["obs_mask"] | ~filled.unsqueeze(-1)
        local, initial_tokens, key_mask, dead, types, origin = self._base.encode_bundle(inputs)
        if hasattr(self, "task_cond") and "task_embeds" in inputs:
            local = local + self.task_cond(inputs["task_embeds"][:, :, :self.args.n_agents])
        tokens = self.global_net.project(initial_tokens)
        bs, ts, na, hidden_dim = local.shape
        n_obs, entities, dim = bs * ts * na, tokens.shape[-2], tokens.shape[-1]
        # Runtime MAC attributes survive ALMA's concatenation of imagined
        # tensor inputs. Target agents use eval() during learning, so module
        # training mode alone must never replace recorded replay depths.
        evaluation = bool(getattr(self, "loop_test_mode", inputs.get("loop_test_mode", not self.training)))
        mode, depths, execution = self._loop_depths(inputs, (bs, ts, na), tokens.device, evaluation)
        stats = ({name: 0 for name in ("decisions", "depth_1", "depth_2", "depth_3", "depth_4",
                  "attention_entries", "read_calls", "temporal_previews", "entity_rounds", "slot_rounds",
                  "query_updates", "temporal_commits", "macs")} if evaluation else None)
        live = ~dead.reshape(-1)
        self._loop_shared_cost(stats, inputs, live, entities)
        self.global_net._loop_add(stats, temporal_commits=int(live.sum().item()) if stats is not None else 0)
        # REFIL may hide self as a KEY in an imagined cross-group branch.
        # Its private query still uses the original own physical features.
        if "observer_entities" in inputs:
            own_ids = th.arange(na, device=tokens.device)
            own = inputs["observer_entities"][:, :, own_ids, own_ids]
        else:
            own = inputs["entities"][:, :, :na]
        active = live.nonzero(as_tuple=False).flatten()
        b0 = tokens.new_zeros(n_obs, dim)
        if active.numel():
            own_encoded = F.relu(self._base.fc1(own.reshape(n_obs, -1).index_select(0, active)))
            own_encoded = self.global_net.project(own_encoded)
            no_history = own_encoded.new_zeros(active.numel(), hidden_dim)
            query = self.global_net.query_proj(th.cat((own_encoded, no_history), -1))
            b0 = b0.index_copy(0, active, query)
        h = inputs.get("hidden_state")
        if h is None:
            h = local.new_zeros(bs, na, hidden_dim)
        if h.shape[0] != bs:
            if bs % h.shape[0]:
                raise ValueError("loop hidden-state batch cannot expand to imagined inputs")
            h = h.repeat(bs // h.shape[0], 1, 1)
        self.global_net.last_intent = None
        if mode in ("fixed", "random"):
            read = tokens.new_zeros(n_obs, dim)
            if active.numel():
                selected = self._loop_spatial_fixed(tokens.reshape(n_obs, entities, dim).index_select(0, active),
                    key_mask.reshape(n_obs, entities).index_select(0, active), b0.index_select(0, active),
                    depths.reshape(-1).index_select(0, active), stats)
                read = read.index_copy(0, active, selected)
            residual = local.new_zeros(n_obs, hidden_dim)
            if active.numel():
                residual = residual.index_copy(0, active, self.global_net.fuse(read.index_select(0, active)))
            fused = local + residual.reshape(bs, ts, na, hidden_dim)
            h_prev = h
            n_act = int(self.args.n_actions)
            outputs = []
            for t in range(ts):
                ids = (~dead[:, t]).reshape(-1).nonzero(as_tuple=False).flatten()
                q_t = local.new_zeros(bs * na, n_act)
                next_h = local.new_zeros(bs * na, hidden_dim)
                if ids.numel():
                    x_t = fused[:, t].reshape(bs * na, hidden_dim).index_select(0, ids)
                    previous = h.reshape(bs * na, hidden_dim).index_select(0, ids)
                    q_live, h_live = self._head.step(x_t.unsqueeze(1), previous.unsqueeze(1),
                        dead.new_ones(ids.numel(), 1))
                    q_t = q_t.index_copy(0, ids, q_live[:, 0])
                    next_h = next_h.index_copy(0, ids, h_live[:, 0])
                    self._loop_temporal_cost(stats, ids.numel())
                else:
                    # Keep the zero-output replay path differentiable without
                    # evaluating padded agent GRUs or fabricating decisions.
                    q_t = q_t + fused[:, t].sum() * 0
                h = next_h.reshape(bs, na, hidden_dim)
                outputs.append(q_t.reshape(bs, na, n_act))
            q = th.stack(outputs, 1)
            selected_depths = depths.masked_fill(dead, 0)
            self.global_net._loop_add(stats, macs=active.numel() * dim * hidden_dim)
            # Real policy decisions have T=1. Never recompute the temporal
            # head merely to capture recurrent histories in a replay batch.
            if ts == 1:
                self._capture_loop_exit(tokens[:, 0].reshape(bs * na, entities, dim),
                    key_mask[:, 0].reshape(bs * na, entities), types[:, 0].reshape(bs * na, entities, 3),
                    origin[:, 0].reshape(bs * na, entities), ~dead[:, 0], h_prev, q[:, 0],
                    read.reshape(bs * na, dim), depths[:, 0].reshape(-1), 0)
        else:
            if not evaluation:
                raise ValueError("adaptive/random deployment is evaluation-only")
            if "avail_actions" not in inputs:
                raise ValueError("adaptive/random loop execution requires avail_actions")
            outputs = []
            exit_depths = []
            for t in range(ts):
                h_prev = h
                q_t, h, selected_read, selected_depth = self._loop_execute_online(tokens[:, t].reshape(bs * na, entities, dim),
                    key_mask[:, t].reshape(bs * na, entities), b0.reshape(bs, ts, na, dim)[:, t].reshape(bs * na, dim),
                    local[:, t], ~dead[:, t], h, inputs["avail_actions"][:, t], depths[:, t], mode, execution, stats)
                outputs.append(q_t)
                exit_depths.append(selected_depth.reshape(bs, na))
                self._capture_loop_exit(tokens[:, t].reshape(bs * na, entities, dim),
                    key_mask[:, t].reshape(bs * na, entities), types[:, t].reshape(bs * na, entities, 3),
                    origin[:, t].reshape(bs * na, entities), ~dead[:, t], h_prev, q_t,
                    selected_read, selected_depth, t)
            q = th.stack(outputs, 1)
            selected_depths = th.stack(exit_depths, 1)
        if stats is not None:
            # Per-decision record for counterfactual probes, not an additive
            # episode cost. Adapters must exclude this list when summing costs.
            stats["selected_depths"] = selected_depths.detach().cpu().tolist()
        self.last_loop_stats = {} if stats is None else stats
        return q.masked_fill(inputs["entity_mask"][:, :, :na].unsqueeze(-1), 0), h

    def _compute_global(self, inputs):
        if getattr(self.global_net, "loop_core", None) is not None:
            return self._compute_loop(inputs)
        local, tokens, key_mask, dead, types, origin = self._base.encode_bundle(inputs)
        if hasattr(self, "task_cond") and "task_embeds" in inputs:
            local = local + self.task_cond(inputs["task_embeds"][:, :, :self.args.n_agents])
        tokens = self.global_net.project(tokens)
        bs, ts, na, hidden_dim = local.shape
        n_ent = tokens.shape[-2]
        gdim = tokens.shape[-1]
        n_act = int(self.args.n_actions)
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
                q_t, h = self._head.step(fused.reshape(bs, na, hidden_dim), h, alive)
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
                q_t, h = self._head.step(fused.reshape(bs, na, hidden_dim), h, alive)
                outputs.append(q_t)
        q = th.stack(outputs, dim=1)
        agent_mask = inputs["entity_mask"][:, :, :na]
        return q.masked_fill(agent_mask.unsqueeze(3), 0), h

    def make_imagined_inputs(self, inputs):
        imagined, groups = super().make_imagined_inputs(inputs)
        if "observer_entities" in inputs:
            imagined["observer_entities"] = inputs["observer_entities"].repeat(2, 1, 1, 1, 1)
        if self.global_branch and getattr(self.global_net, "loop_core", None) is not None:
            for name in ("loop_depth", "loop_filled"):
                if name in inputs:
                    value = inputs[name]
                    imagined[name] = value.repeat(2, *([1] * (value.ndim - 1)))
        # Keep the random partition constant but exclude deaths at each t.
        inactive = inputs["entity_mask"].bool()
        blocked = inactive.unsqueeze(-1) | inactive.unsqueeze(-2)
        groups = tuple(g.bool() | blocked for g in groups)
        imagined["imagine_mask"] = th.cat(groups, dim=0).to(th.uint8)
        return imagined, groups


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
        from open_score.algos.dcg.dcg_controller import DeepCoordinationGraphMAC
        return DeepCoordinationGraphMAC(scheme, groups, args)
    if args.mac == "spectra_mac":
        from open_score.algos.spectra.spectra_controller import SPECTraMAC
        return SPECTraMAC(scheme, groups, args)
    return SharedEntityMAC(scheme, groups, args)


def build_encoder(name, input_shape, args):
    if name != "attention":
        raise ValueError(f"Unsupported PURE encoder: {name}")
    return AttentionEncoder(input_shape, args)

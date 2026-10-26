"""LEAF global reasoning branches and teammate-intent helpers.

Module boundaries preserve the existing parameter names and computations.
"""
from copy import deepcopy

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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

    def project_kv(self, memory):
        """Cache immutable K/V for the fixed-context LEAF cores."""
        batch, length, _ = memory.shape
        key = self.key(memory).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.value(memory).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        return key, value

    def read_cached(self, query, key, value, mask=None):
        batch, length, dim = query.shape
        projected = self.query(query).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        logits = th.matmul(projected, key.transpose(-2, -1)) / self.scale
        if mask is not None:
            hide = mask[:, None, None, :]
            empty = hide.all(-1, keepdim=True)
            logits = logits.masked_fill(hide, float("-inf")).masked_fill(empty, 0)
        weights = th.softmax(logits, -1)
        if mask is not None:
            weights = weights.masked_fill(hide | empty, 0)
        if getattr(self, "capture_attention", False):
            self.last_weights = weights.detach()
        result = th.matmul(weights, value).transpose(1, 2).contiguous().view(batch, length, dim)
        return self.out(result)


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
    """HAD planar action classes."""
    return int(args.n_actions)


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
        self.loop_core = getattr(args, "leaf_loop_core", None)
        if self.loop_core not in (None, "query_feedback", "bidirectional", "requery", "entity_gru", "slot_gru"):
            raise ValueError(f"Unknown leaf_loop_core {self.loop_core!r}")
        if self.loop_core is not None and kind != "cycle":
            raise ValueError("leaf_loop_core requires global_branch=cycle")
        self.stable_cycle = bool(getattr(args, "rer_stable_cycle", False) or self.loop_core)
        self.input_fusion = bool(getattr(args, "rer_input_fusion", False) and not self.loop_core)
        if (self.stable_cycle or self.input_fusion) and kind != "cycle":
            raise ValueError("Stable/input-conditioned updates require global_branch=cycle")
        if self.stable_cycle:
            self.norm_attn_residual = nn.LayerNorm(dim)
            self.norm_ffn_residual = nn.LayerNorm(dim)
        if self.input_fusion:
            self.fixed_input_gate = getattr(args, "rer_input_fixed_gate", None)
            if self.fixed_input_gate is not None:
                self.fixed_input_gate = float(self.fixed_input_gate)
                if not 0 <= self.fixed_input_gate <= 1:
                    raise ValueError("rer_input_fixed_gate must be between zero and one")
            # Preserve the initialization of the host, reader and mixer.
            with th.random.fork_rng(devices=[]):
                self.input_norm = nn.LayerNorm(dim)
                self.input_gate = nn.Linear(2 * dim, dim)
            nn.init.zeros_(self.input_gate.weight)
            # Begin with the 25% input anchor used in the offline diagnostic.
            nn.init.constant_(self.input_gate.bias, -1.0986122886681098)
        if getattr(args, "rer_readout", "learned") == "static":
            rounds = max(int(getattr(args, "global_eval_depth", 4)),
                         *[int(r) for r in getattr(args, "global_depths", [1, 2, 3, 4])])
            self.round_logits = nn.Parameter(th.zeros(rounds))
        if self.rer_update not in ("tied", "untied", "untied4", "kv0"):
            raise ValueError("rer_update must be tied, untied, untied4 or kv0")
        if self.rer_update != "tied" and kind != "cycle":
            raise ValueError("RER update interventions require global_branch=cycle")
        if self.rer_update in ("untied", "untied4"):
            # The existing block is round one. Deepcopy adds the remaining independent
            # blocks without drawing randomness or changing any shared module.
            names = ("self_attn", "norm_attn", "norm_ffn", "ffn", "count_to_token")
            self.untied_depth = (4 if self.rer_update == "untied4" else
                                int(getattr(args, "rer_untied_depth", 2)))
            if self.untied_depth < 1:
                raise ValueError("rer_untied_depth must be positive")
            if self.stable_cycle:
                names += ("norm_attn_residual", "norm_ffn_residual")
            if self.input_fusion:
                names += ("input_norm", "input_gate")
            self.untied_blocks = nn.ModuleList([
                nn.ModuleDict({name: deepcopy(getattr(self, name)) for name in names})
                for _ in range(self.untied_depth - 1)])
        self.intent = bool(getattr(args, "rer_intent", False))
        if self.intent:
            if kind != "cycle" or self.rer_update in ("untied", "untied4"):
                raise ValueError("rer_intent requires a shared-parameter cycle branch")
            n_actions = intent_classes(args)
            # Drawn from a forked RNG so every other module (and the mixer
            # built afterwards) initializes exactly as the same-seed ARR.
            with th.random.fork_rng(devices=[]):
                self.intent_head = nn.Linear(dim, n_actions)
                self.intent_embed = nn.Linear(n_actions, dim, bias=False)
            nn.init.zeros_(self.intent_embed.weight)
        if self.loop_core is not None:
            # New parameters exist only for new methods. A fork also keeps the
            # original host/mixer RNG stream independent of the chosen core.
            with th.random.fork_rng(devices=[]):
                self.loop_query_norm = nn.LayerNorm(dim)
                if self.loop_core in ("query_feedback", "bidirectional", "requery"):
                    self.loop_feedback = nn.Linear(dim, dim)
                    self.loop_gate = nn.Linear(2 * dim, dim)
                    nn.init.zeros_(self.loop_gate.weight)
                    nn.init.zeros_(self.loop_gate.bias)
                elif self.loop_core == "entity_gru":
                    self.loop_message_norm = nn.LayerNorm(dim)
                    self.loop_entity_gru = nn.GRUCell(2 * dim, dim)
                elif self.loop_core == "slot_gru":
                    if self.n_slots != 4:
                        raise ValueError("main1009 slot_gru uses exactly four slots")
                    self.loop_seeds = nn.Parameter(th.randn(1, 4, dim) * 0.02)
                    self.loop_slot_own = nn.Linear(dim, dim)
                    self.loop_slot_norm = nn.LayerNorm(dim)
                    self.loop_slot_query = nn.Linear(dim, dim, bias=False)
                    self.loop_slot_key = nn.Linear(dim, dim, bias=False)
                    self.loop_slot_value = nn.Linear(dim, dim, bias=False)
                    self.loop_slot_gru = nn.GRUCell(dim, dim)
                    self.loop_slot_ffn = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.ReLU(),
                                                       nn.Linear(dim * ffn_mult, dim))
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
        if mode not in ("learned", "static", "read1", "read2", "read3", "read4", "uniform"):
            raise ValueError(f"Unknown RER readout {mode!r}")
        if mode not in ("learned", "static") and self.training:
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
            score_input = (F.normalize(read, dim=-1) * read.shape[-1] ** 0.5
                           if getattr(self.args, "rer_score_normalize", False) else read)
            scores.append(self.read_score(score_input))
            if getattr(self, "capture_attention", False) and hasattr(self.read_attn, "last_weights"):
                read_weights.append(self.read_attn.last_weights.mean(1).squeeze(1).detach())
        scores = th.cat(scores, dim=-1)
        if mode == "static":
            if not hasattr(self, "round_logits") or len(states) > self.round_logits.numel():
                raise ValueError("Static fusion supports only its trained round positions")
            scores = self.round_logits[:len(states)].unsqueeze(0).expand_as(scores)
        index = th.arange(scores.shape[-1], device=scores.device)
        valid = index.unsqueeze(0) < depth.unsqueeze(-1)
        scores = scores.masked_fill(~valid, float("-inf"))
        empty = ~valid.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(empty, 0)
        alpha = th.softmax(scores, dim=-1)
        alpha = alpha.masked_fill(~valid, 0)
        if mode not in ("learned", "static"):
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
        block = self.untied_blocks[round_id - 1] if self.rer_update in ("untied", "untied4") and round_id else None
        def layer(name):
            return getattr(self, name) if block is None else block[name]
        if self.input_fusion:
            if initial is None:
                raise ValueError("Input-conditioned cycles require the initial entity set")
            gate = self.fixed_input_gate
            if gate is None:
                gate = th.sigmoid(layer("input_gate")(th.cat(
                    (layer("input_norm")(initial), layer("input_norm")(tokens)), dim=-1)))
            tokens = (1 - gate) * tokens + gate * initial
        injected = layer("norm_attn")(tokens)
        if not getattr(self.args, "skip_count_inject", False):
            injected = injected + layer("count_to_token")(count).unsqueeze(1)
        memory = initial if self.rer_update == "kv0" and initial is not None else tokens
        if kv_extra is not None:
            memory = memory + kv_extra
        if self.stable_cycle or getattr(self.args, "global_kv_prenorm", False):
            memory = layer("norm_attn")(memory).masked_fill(mask.unsqueeze(-1), 0)
        attn = layer("self_attn")
        tokens = tokens + attn(injected, memory, memory, mask)
        if self.stable_cycle:
            tokens = layer("norm_attn_residual")(tokens)
        if getattr(self, "capture_attention", False) and hasattr(attn, "last_weights"):
            self.last_self_attn.append(attn.last_weights.mean(1).detach())
        tokens = tokens + layer("ffn")(layer("norm_ffn")(tokens))
        if self.stable_cycle:
            tokens = layer("norm_ffn_residual")(tokens)
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
        if self.rer_update in ("untied", "untied4") and n_rounds > self.untied_depth:
            raise ValueError(f"Untied updates support at most {self.untied_depth} rounds")
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

    @staticmethod
    def _loop_add(stats, **costs):
        if stats is not None:
            for name, value in costs.items():
                stats[name] += int(value)

    def _loop_attn_cost(self, stats, rows, queries, keys, *, cached=False):
        dim = int(self.args.global_embed_dim) if hasattr(self.args, "global_embed_dim") else 64
        # Counts actual dense matrix MACs, including masked/padded positions;
        # elementwise, normalization and softmax work is deliberately excluded.
        macs = rows * ((2 * queries + (0 if cached else 2 * keys)) * dim * dim
                       + 2 * queries * keys * dim)
        self._loop_add(stats, attention_entries=rows * queries * keys * self.self_attn.n_heads,
                       macs=macs)

    def _loop_context(self, memory, mask, query=None):
        normalized = self.norm_attn(memory).masked_fill(mask.unsqueeze(-1), 0)
        keys, key_mask = normalized, mask
        if query is not None:
            # A private workspace token, not an entity or another observer.
            keys = th.cat((keys, self.loop_query_norm(query).unsqueeze(1)), 1)
            key_mask = th.cat((mask, mask.new_zeros(mask.shape[0], 1)), 1)
        updated = self.norm_attn_residual(memory + self.self_attn(normalized, keys, keys, key_mask))
        updated = self.norm_ffn_residual(updated + self.ffn(self.norm_ffn(updated)))
        return updated.masked_fill(mask.unsqueeze(-1), 0)

    def _loop_context_cost(self, stats, rows, entities, *, query=False):
        self._loop_attn_cost(stats, rows, entities, entities + int(query))
        dim = self.fuse.in_features
        self._loop_add(stats, entity_rounds=rows,
                       macs=rows * entities * (dim * self.ffn[0].out_features
                                               + self.ffn[0].out_features * dim))

    def loop_initialize(self, initial, mask, b0, stats=None):
        """One independent workspace per observer and physical timestep."""
        initial = initial.masked_fill(mask.unsqueeze(-1), 0)
        work = {"initial": initial, "mask": mask, "b0": b0,
                "query": b0, "memory": initial}
        rows, entities, dim = initial.shape
        if self.loop_core in ("requery", "slot_gru"):
            context = self._map_chunks(self._loop_context, initial, mask)
            self._loop_context_cost(stats, rows, entities)
            work["memory"] = context
            if self.loop_core == "requery":
                work["key"], work["value"] = self.read_attn.project_kv(context)
            else:
                normalized = self.norm_attn(context)
                work["key"] = self.loop_slot_key(normalized)
                work["value"] = self.loop_slot_value(normalized)
                work["slots"] = self.loop_seeds.expand(rows, -1, -1) + self.loop_slot_own(b0).unsqueeze(1)
                self._loop_add(stats, macs=rows * dim * dim)
            self._loop_add(stats, macs=rows * entities * 2 * dim * dim)
        return work

    def _loop_entity_step(self, previous, initial, mask):
        normalized = self.norm_attn(previous).masked_fill(mask.unsqueeze(-1), 0)
        messages = self.self_attn(normalized, normalized, normalized, mask)
        inputs = th.cat((self.norm_attn(initial), self.loop_message_norm(messages)), -1)
        rows, entities, dim = previous.shape
        updated = self.loop_entity_gru(inputs.reshape(-1, 2 * dim), previous.reshape(-1, dim))
        updated = self.norm_attn_residual(updated.reshape(rows, entities, dim))
        updated = self.norm_ffn_residual(updated + self.ffn(self.norm_ffn(updated)))
        return updated.masked_fill(mask.unsqueeze(-1), 0)

    def _loop_slot_step(self, slots, key, value, mask):
        query = self.loop_slot_query(self.loop_slot_norm(slots))
        logits = th.bmm(query, key.transpose(1, 2)) / query.shape[-1] ** 0.5
        # Slots compete for each entity, then each slot takes a weighted mean
        # over entities. Padding contributes neither numerator nor denominator.
        weights = th.softmax(logits, dim=1).masked_fill(mask.unsqueeze(1), 0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        updates = th.bmm(weights, value)
        rows, n_slots, dim = slots.shape
        updated = self.loop_slot_gru(updates.reshape(-1, dim), slots.reshape(-1, dim))
        updated = updated.reshape(rows, n_slots, dim)
        updated = updated + self.loop_slot_ffn(self.loop_slot_norm(updated))
        return updated.masked_fill(mask.all(-1)[:, None, None], 0)

    def loop_round(self, work, stats=None):
        """Perform one spatial iteration and read, without temporal mutation."""
        memory, mask = work["memory"], work["mask"]
        rows, entities, dim = memory.shape
        core = self.loop_core
        if core in ("query_feedback", "bidirectional"):
            anchored = 0.75 * memory + 0.25 * work["initial"]
            if core == "bidirectional":
                memory = self._map_chunks(self._loop_context, anchored, mask, work["query"])
            else:
                memory = self._map_chunks(self._loop_context, anchored, mask)
            self._loop_context_cost(stats, rows, entities, query=core == "bidirectional")
            work["memory"] = memory
        elif core == "entity_gru":
            memory = self._map_chunks(self._loop_entity_step, memory, work["initial"], mask)
            work["memory"] = memory
            self._loop_context_cost(stats, rows, entities)
            self._loop_add(stats, macs=rows * entities * 9 * dim * dim)
        elif core == "slot_gru":
            slots = self._map_chunks(self._loop_slot_step, work["slots"], work["key"], work["value"], mask)
            work["slots"] = slots
            self._loop_add(stats, slot_rounds=rows,
                           attention_entries=rows * self.n_slots * entities,
                           macs=rows * (self.n_slots * dim * dim + 2 * self.n_slots * entities * dim
                                       + 6 * self.n_slots * dim * dim
                                       + 2 * self.n_slots * dim * self.loop_slot_ffn[0].out_features))
            memory = slots
            mask = mask.all(-1).unsqueeze(-1).expand(-1, self.n_slots)
        query = self.loop_query_norm(work["query"]).unsqueeze(1)
        if core == "requery":
            read = self._map_chunks(self.read_attn.read_cached, query, work["key"], work["value"], mask)
        else:
            read = self._map_chunks(self.read_attn, query, memory, memory, mask)
        self._loop_attn_cost(stats, rows, 1, memory.shape[1], cached=core == "requery")
        self._loop_add(stats, read_calls=rows)
        return read.squeeze(1)

    def _loop_update_query(self, b0, previous, read):
        candidate = b0 + self.loop_feedback(self.loop_query_norm(read))
        gate = th.sigmoid(self.loop_gate(th.cat((self.loop_query_norm(previous),
                                                self.loop_query_norm(candidate)), -1)))
        return (1 - gate) * previous + gate * candidate

    def loop_continue(self, work, read, stats=None):
        if self.loop_core in ("query_feedback", "bidirectional", "requery"):
            work["query"] = self._map_chunks(self._loop_update_query, work["b0"], work["query"], read)
            dim = self.fuse.in_features
            self._loop_add(stats, query_updates=read.shape[0], macs=read.shape[0] * 3 * dim * dim)

    @staticmethod
    def loop_compact(work, keep):
        return {name: tensor.index_select(0, keep) for name, tensor in work.items()}


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

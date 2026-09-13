"""Entity encoders and ALMA controller bridge for the six HAD methods.

Attention and imagined branches are inherited from ALMA commit
81bd5c475972b982ce470b1ed78154ec39d236da. The GNN is adapted from
SPECTra ffababf6187216c9d16b2109ee8ef6fe5fdf1172, SMACv2 gnn_rnn_agent.py:
one graph-convolution layer, separate self/neighbour maps and mean pooling.
Only the graph encoder is borrowed; GNN-QMIX uses the shared ALMA learner.
"""
from copy import copy

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from controllers.entity_controller import EntityMAC
from modules.agents.agent import Agent as ALMAAgent
from modules.layers.attention import EntityAttentionLayer


def relative_entity_views(inputs, args):
    """Construct temporary observer-centred views; replay stays global.

    Each observer is moved to row zero. Other geometry is relative to it;
    its own absolute position and velocity are retained, so own speed is not
    lost by subtracting it from itself. Native FF inputs have no HAD geometry.
    """
    entities = inputs["entities"]
    bs, ts, ne, ed = entities.shape
    na = args.n_agents
    own = entities[:, :, :na]
    views = entities.unsqueeze(2).expand(bs, ts, na, ne, ed).clone()
    if getattr(args, "feature_layout", "had") == "had":
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
    return views.reshape(-1, ne, ed), hidden.reshape(-1, ne), dead.reshape(-1), (bs, ts, na)


class _EntityEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def init_hidden(self):
        return next(self.parameters()).new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs):
        views, hidden, dead, shape = relative_entity_views(inputs, self.args)
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
            if self.training and th.is_grad_enabled():
                out = checkpoint(self._encode, v, m, d, use_reentrant=False)
            else:
                out = self._encode(v, m, d)
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

    def _encode(self, views, hidden, dead):
        # An inactive observer has a dummy zero self token, preventing an
        # all-minus-infinity softmax before its output is zeroed.
        pre = hidden.clone()
        pre[dead, 0] = False
        x = F.relu(self.fc1(views))
        x = self.attn(x, pre_mask=pre[:, None], post_mask=dead[:, None])
        x = F.relu(self.fc2(F.relu(x[:, 0])))
        return x.masked_fill(dead[:, None], 0)


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
        self.fc_out = nn.Linear(args.rnn_hidden_dim, output_features or args.n_actions)

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
        return self.fc_out(hs), h.reshape(bs, na, hd)


class EntityAgent(ALMAAgent):
    def __init__(self, input_shape, args):
        # Preserve ALMA's imagined-input construction, subtask conditioning
        # and combined forward.
        super().__init__(input_shape, args, recurrent=False, entity_scheme=True,
                         subtask_cond=args.agent.get("subtask_cond"))
        kind = getattr(args, "encoder", "attention")
        self._base = {"flatten": FlattenEncoder, "attention": AttentionEncoder,
                      "gnn": GraphEncoder}[kind](input_shape, args)
        if args.agent.get("recurrent", False):
            self._head = TemporalHead(args)
        if args.agent.get("subtask_cond") == "full_obs":
            self.task_cond = nn.Linear(args.attn_embed_dim, args.rnn_hidden_dim)

    def _compute_network(self, inputs):
        # Modern torch requires bool masks; official ALMA stores uint8 replay.
        inputs = dict(inputs)
        inputs["entity_mask"] = inputs["entity_mask"].bool()
        inputs["obs_mask"] = inputs["obs_mask"].bool()
        encoded = self._base(inputs)
        if hasattr(self, "task_cond") and "task_embeds" in inputs:
            encoded = encoded + self.task_cond(inputs["task_embeds"][:, :, :self.args.n_agents])
        if self.use_copa:
            encoded = encoded + inputs["coach_z"]
        q, hidden = self._head(encoded, inputs)
        agent_mask = inputs["entity_mask"][:, :, :self.args.n_agents]
        return q.masked_fill(agent_mask.unsqueeze(3), 0), hidden

    def make_imagined_inputs(self, inputs):
        imagined, groups = super().make_imagined_inputs(inputs)
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

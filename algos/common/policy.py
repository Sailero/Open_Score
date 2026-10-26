"""Frozen checkpoint policy on the shared HAD entity interface."""
import copy
import numpy as np

from ...envs.env_wrapper import (
    MAX_AGENTS, N_ACTIONS, available_actions, blue_nearest_from_table,
    entities_from_state, masks_from_entity_mask, resolve_pad, task_masks,
)


class FrozenPolicyAdapter:
    """Greedy checkpoint policy using the same absolute entities as training."""
    def __init__(self, mac, args, scheme, groups, preprocess, mixer=None):
        self.mac, self.args, self.scheme = mac, args, scheme
        self.groups, self.preprocess, self.mixer = groups, preprocess, mixer
        self.mac.agent.eval()
        if self.mixer is not None:
            self.mixer.eval()
        self.device = "cpu"
        self.reset()

    def set_device(self, device):
        import torch
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("Frozen policies support CPU or CUDA devices")
        self.device = str(target)
        self.args.device, self.args.use_cuda = self.device, target.type == "cuda"
        for name in ("agent", "alloc_policy", "alloc_critic", "utility_fun", "payoff_fun"):
            module = getattr(self.mac, name, None)
            if module is not None:
                module.to(target)
        for name in ("edges_from", "edges_to"):
            tensor = getattr(self.mac, name, None)
            if tensor is not None:
                setattr(self.mac, name, tensor.to(target))
        if self.mixer is not None:
            self.mixer.to(target)
        self.reset()

    def set_eval_depth(self, depth):
        depth = int(depth)
        if getattr(self.args, "leaf_loop_core", None):
            self.set_loop_execution("fixed", depth=depth)
        self.args.global_eval_depth = depth
        if hasattr(self.mac, "set_global_depth"):
            self.mac.set_global_depth(depth)

    def set_loop_execution(self, mode, depth=4, threshold=None, probabilities=None):
        if not getattr(self.args, "leaf_loop_core", None):
            raise ValueError("Loop execution is available only for the five loop candidates")
        if mode not in ("fixed", "adaptive", "random"):
            raise ValueError("Loop execution mode must be fixed, adaptive or random")
        depth = int(depth)
        if not 1 <= depth <= 4:
            raise ValueError("Loop execution depth must be in 1..4")
        if threshold is not None and not np.isfinite(float(threshold)):
            raise ValueError("Loop threshold must be finite")
        execution = dict(mode=mode, depth=depth)
        if threshold is not None:
            execution["threshold"] = float(threshold)
        if probabilities is not None:
            values = np.asarray(probabilities, dtype=np.float64)
            if values.shape != (4,) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
                raise ValueError("Loop probabilities require four finite nonnegative entries with positive sum")
            execution["probabilities"] = (values / values.sum()).tolist()
        self.mac.agent.loop_execution = execution

    def set_loop_seed(self, seed):
        import torch as th
        if not getattr(self.args, "leaf_loop_core", None):
            raise ValueError("Loop seeds are available only for loop candidates")
        self.mac.agent.loop_generator = th.Generator(device="cpu").manual_seed(int(seed))

    def set_decision_depths(self, depths):
        import torch as th
        if not getattr(self.args, "leaf_loop_core", None):
            raise ValueError("Decision depths are available only for loop candidates")
        values = th.as_tensor(depths, device=self.device)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape != (1, int(self.args.n_agents)):
            raise ValueError("Decision depths must have shape [A] or [1,A] for the padded agent roster")
        if values.dtype == th.bool or values.is_complex() or (values.is_floating_point() and not bool((values == values.round()).all())):
            raise ValueError("Decision depths must be integers")
        if not bool(((values >= 1) & (values <= 4)).all()):
            raise ValueError("Decision depths must be in 1..4")
        self.mac.agent.loop_depth_override = values.long().clone()

    def get_loop_stats(self):
        return copy.deepcopy(self._loop_stats)

    def _record_loop_stats(self):
        stats = getattr(self.mac.agent, "last_loop_stats", None)
        if not stats:
            return
        # Per-decision selections are an observation, not an additive counter.
        # The runner reads them directly from agent.last_loop_stats.
        stats = {key: value for key, value in stats.items() if key != "selected_depths"}
        def merge(previous, value):
            if isinstance(value, dict):
                previous = {} if previous is None else previous
                return {**previous, **{key: merge(previous.get(key), item) for key, item in value.items()}}
            if isinstance(value, (list, tuple)):
                previous = [0] * len(value) if previous is None else previous
                if len(previous) != len(value):
                    raise ValueError("Loop statistic vector length changed within an episode")
                return [merge(old, item) for old, item in zip(previous, value)]
            if isinstance(value, (int, float, np.number)):
                if isinstance(value, np.number):
                    value = value.item()
                return (0 if previous is None else previous) + value
            return copy.deepcopy(value)
        self._loop_stats = merge(self._loop_stats, stats)

    def reset(self):
        from components.episode_buffer import EpisodeBatch
        self.batch = EpisodeBatch(self.scheme, self.groups, 1, int(self.args.episode_limit) + 1,
                                  preprocess=self.preprocess, device=getattr(self, "device", "cpu"))
        self.mac.init_hidden(batch_size=1)
        self.last_step, self.last_result = -1, None
        self.q_tot, self.q_i = [], []
        self._loop_stats = {}
        if getattr(self.args, "leaf_loop_core", None):
            self.mac.agent.loop_depth_override = None
        self.intent_stats = dict(count={}, correct={}, nll={}, label_hist=[0] * 9)
        self.roster = None
        self._alloc_hold = 0
        self._last_nearest = None
        self._last_alive = None

    def act(self, state, side, action_ids):
        import torch as th
        from dataclasses import replace
        if side.lower() != "red":
            raise ValueError("cross-scale checkpoints control only the Red team")
        if state.step == self.last_step:
            return dict(self.last_result)
        if state.step != self.last_step + 1:
            raise ValueError("policy needs consecutive states and reset() at every new episode")
        if self.roster is None:
            self.roster = {key: tuple(e.id for e in getattr(state, key)) for key in ("red", "blue", "targets")}
            budget = getattr(self.args, "pool_slots", None)
            counts = tuple(len(getattr(state, key)) for key in ("red", "blue", "targets"))
            if budget is not None and any(c > limit for c, limit in zip(counts, budget)):
                raise ValueError(f"configuration {counts} exceeds this checkpoint's ordered slot "
                                 f"budget {tuple(budget)}; it has no parameters for the extra entities")
        ordered = {}
        for key, ids in self.roster.items():
            rows = {e.id: e for e in getattr(state, key)}
            if set(rows) != set(ids):
                raise ValueError("policy physical roster changed within an episode")
            ordered[key] = tuple(rows[identity] for identity in ids)
        state = replace(state, **ordered)
        entities, absent = entities_from_state(state)
        masks = masks_from_entity_mask(absent)
        avail = available_actions(absent)
        initial = np.ones(MAX_AGENTS, dtype=np.uint8)
        initial[:len(state.red)] = 0
        fractions = np.asarray([sum(e.alive for e in state.red) / len(state.red),
                                sum(e.alive for e in state.blue) / len(state.blue),
                                1 - state.step / state.max_steps], dtype=np.float32)
        step = int(state.step)
        data = {"entities": entities[None], "obs_mask": masks["obs_mask"][None],
                "entity_mask": absent[None], "agent_mask": (1 - absent[:MAX_AGENTS])[None],
                "initial_agent_mask": initial[None], "state": np.concatenate((entities.reshape(-1), fractions))[None],
                "avail_actions": avail[None], "reset": [[0]]}
        if getattr(self.args, "multi_task", False):
            # Same subtask decomposition and decision clock as the training
            # runner, so a hierarchical checkpoint acts as it was trained.
            env_args = dict(getattr(self.args, "env_args", None) or {})
            n_red, n_blue, n_targets = resolve_pad("eval")
            subtask_set = env_args.get("subtask_set", "targets")
            subtasks = task_masks(entities, absent, len(state.targets), n_red, n_blue, n_targets,
                                  subtask_set=subtask_set)
            data.update(entity2task_mask=subtasks["entity2task_mask"][None],
                        task_mask=subtasks["task_mask"][None],
                        hier_decision=[[int(self._hier_decision(state, absent, step, env_args))]])
        self.batch.update(data, ts=step)
        with th.no_grad():
            branch = getattr(self.mac.agent, "global_net", None)
            oracle = branch is not None and getattr(branch, "intent_override", None) == "oracle"
            if oracle:
                # Diagnostic upper bound: inject the teammates' actual actions of this step.
                hidden = self.mac.hidden_states.clone()
                branch.intent_override = None
                first = self.mac.select_actions(self.batch, t_ep=step, t_env=0, test_mode=True)
                self.mac.hidden_states = hidden
                branch.intent_oracle_actions = first.reshape(1, -1).long()
                branch.intent_override = "oracle"
            actions = self.mac.select_actions(self.batch, t_ep=step, t_env=0, test_mode=True)
            self._record_loop_stats()
            if oracle:
                branch.intent_oracle_actions = None
            if branch is not None and getattr(branch, "intent", False):
                self._record_intent(branch, actions)
            self.batch.update({"actions": actions.unsqueeze(-1)}, ts=step, mark_filled=False)
            if hasattr(self.mac, "evaluation_values"):
                values = self.mac.evaluation_values(self.batch, step, actions, [0], self.mixer)
                for key, storage in (("q_tot", self.q_tot), ("q_i", self.q_i)):
                    value = values[key][0]
                    if value is not None:
                        storage.append(float(value))
        ids = tuple(map(int, action_ids))
        if len(ids) != N_ACTIONS:
            raise ValueError("checkpoint evaluation requires the 9 native planar action IDs")
        decoded = actions[0].detach().cpu().numpy()
        self.last_result = {entity.id: ids[int(decoded[i])] if entity.alive else ids[0]
                            for i, entity in enumerate(state.red)}
        self.last_step = step
        return dict(self.last_result)

    def _hier_decision(self, state, absent, step, env_args):
        length = int(self.args.hier_agent["action_length"])
        if env_args.get("allocation_clock", "interval") != "event":
            return step % length == 0
        n_red, n_blue, n_targets = resolve_pad("eval")
        nearest = blue_nearest_from_table(entities_from_state(state)[0], absent, n_red, n_blue, n_targets)
        alive = (sum(entity.alive for entity in state.red), sum(entity.alive for entity in state.blue))
        if step == 0:
            self._alloc_hold = 0
            self._last_nearest = nearest
            self._last_alive = alive
            return True
        self._alloc_hold += 1
        death = alive != self._last_alive
        geometry = self._last_nearest is not None and not np.array_equal(nearest, self._last_nearest)
        decide = death or (geometry and self._alloc_hold >= 2) or self._alloc_hold >= int(env_args.get("max_alloc_hold", 10))
        if decide:
            self._alloc_hold = 0
            self._last_nearest = nearest
            self._last_alive = alive
        return decide

    def _record_intent(self, branch, actions):
        """Per-round top-1 hits and NLL of teammate-action predictions against this step's actions."""
        import torch as th
        record = branch.last_intent
        if record is None or "idx" not in record:
            return
        na = actions.shape[-1]
        labels = actions.reshape(-1, na).long()
        bt = record["idx"] // na
        origin = record["origin"].long().clamp(0, na - 1)
        target = labels.index_select(0, bt).gather(1, origin)
        valid = record["teammate"]
        stats = self.intent_stats
        stats["label_hist"] = (np.asarray(stats["label_hist"])
                               + np.bincount(target[valid].cpu().numpy(), minlength=9)[:9]).tolist()
        for round_id, logits in enumerate(record["logits"]):
            keep = valid & (record["depth"] > round_id).unsqueeze(-1)
            n = int(keep.sum())
            if n == 0:
                continue
            chosen, truth = logits[keep], target[keep]
            key = str(round_id + 1)
            stats["count"][key] = stats["count"].get(key, 0) + n
            stats["correct"][key] = stats["correct"].get(key, 0) + int((chosen.argmax(-1) == truth).sum())
            stats["nll"][key] = stats["nll"].get(key, 0.0) + float(
                th.nn.functional.cross_entropy(chosen, truth, reduction="sum"))

    def episode_q_statistics(self):
        values = {"q_tot_mean": float(np.mean(self.q_tot)) if self.q_tot else None,
                  "q_tot_std": float(np.std(self.q_tot)) if self.q_tot else None,
                  "q_i_mean": float(np.mean(self.q_i)) if self.q_i else None}
        if self.intent_stats["count"]:
            values["intent_stats"] = dict(self.intent_stats)
        return values

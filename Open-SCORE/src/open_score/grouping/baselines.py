"""Known-opponent controls and an explicitly labelled ALMA-style adaptation.

The DLOM pairing below is a scoring surrogate only. It never installs pairings
or changes the shared physical world. The ALMA-style control learns an action
Q function and a proposal distribution; it is not a reproduction of ALMA.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .domain import DecisionState, Group, Grouping


@dataclass
class BaselineDecision:
    action: Grouping
    trace: dict
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    released_ids: tuple[int, ...]


def _decision(action, *, value=0.0, released=(), trace=None):
    return BaselineDecision(action, trace or {}, torch.tensor(0.0),
                            torch.tensor(0.0), torch.as_tensor(value), tuple(released))


def _targets(state):
    return tuple(t for t in state.targets if t.health > 0)


def _build(ids, target_assignments, group_size=4, reserve=()):
    by_target = {}
    for identity, target in zip(ids, target_assignments):
        by_target.setdefault(int(target), []).append(int(identity))
    groups = tuple(Group(target, tuple(members[start:start + group_size]))
                   for target, members in by_target.items()
                   for start in range(0, len(members), group_size))
    return Grouping(groups, tuple(map(int, reserve)))


def balanced_initial(state: DecisionState) -> Grouping:
    """Balanced task counts with spatial assignment, then groups of at most four."""
    red, targets = state.alive("red"), _targets(state)
    if not red:
        return Grouping((), ())
    if not targets:
        return Grouping((), tuple(x.id for x in red))
    # Equal-capacity target slots and a global distance match avoid ID-based
    # assignment priorities. Ties are harmless for this fixed rule baseline.
    slots = [targets[i % len(targets)] for i in range(len(red))]
    cost = np.asarray([[np.linalg.norm(np.asarray(x.position) - t.position)
                        for t in slots] for x in red])
    rows, columns = linear_sum_assignment(cost)
    assignments = {int(i): slots[int(j)].id for i, j in zip(rows, columns)}
    action = _build([x.id for x in red], [assignments[i] for i in range(len(red))])
    return action.validate(state.ids("red"), [x.id for x in targets])


def candidates(state: DecisionState, limit: int = 16, rng=None) -> list[Grouping]:
    """A bounded legal pool including the old grouping and diverse partitions."""
    if limit < 1:
        raise ValueError("candidate limit must be positive")
    rng = np.random.default_rng(0) if rng is None else rng
    ids = tuple(state.ids("red"))
    target_ids = tuple(t.id for t in _targets(state))
    pool, seen = [], set()

    def add(action):
        action.validate(ids, target_ids)
        signature = (tuple((g.target, tuple(sorted(g.members))) for g in action.groups),
                     tuple(sorted(action.reserve)))
        if signature not in seen and len(pool) < limit:
            seen.add(signature)
            pool.append(action)

    if state.previous is not None:
        add(state.previous.prune(ids))
    add(balanced_initial(state))
    if not ids or not target_ids:
        return pool
    # Keep same-task split variants before adding concentration/reserves.
    balanced = balanced_initial(state)
    mapping = {i: g.target for g in balanced.groups for i in g.members}
    for size in (1, 2, 3):
        add(_build(ids, [mapping[i] for i in ids], group_size=size))
    for target in target_ids:
        add(_build(ids, [target] * len(ids)))
    add(Grouping((), ids))
    for attempt in range(max(16, limit * 4)):
        if len(pool) >= limit:
            break
        order = tuple(map(int, rng.permutation(ids)))
        reserve_count = int(rng.integers(0, min(3, len(ids)) + 1)) if attempt % 3 == 0 else 0
        active = order[reserve_count:]
        assignments = rng.choice(target_ids, size=len(active))
        add(_build(active, assignments, int(rng.integers(1, 5)), order[:reserve_count]))
    return pool


class StaticPolicy:
    """B0: one balanced initialization, then only remove dead members."""
    config = {"method": "static", "initialization": "balanced_spatial"}

    def act(self, state, **kwargs):
        initial = state.step == 0 or state.previous is None
        action = (balanced_initial(state) if initial
                  else state.previous.prune(state.ids("red")))
        return _decision(action, released=state.ids("red") if initial else ())


class FrozenDLOM:
    """Minimal frozen checkpoint reader; no Stage3 or old training imports."""
    def __init__(self, checkpoint, device="cpu"):
        from open_score.stage2.outcome_model import DynamicHADOutcomeNet
        from open_score.stage2.canonical import HADCanonicalizer

        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        with torch.random.fork_rng(devices=[]):
            self.model = DynamicHADOutcomeNet(payload["horizon_bins"],
                                             payload["entity_hidden_dim"], payload["hidden_dim"])
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.temperature = float(payload["temperature"])
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("invalid DLOM temperature")
        self.horizon_steps = int(payload["steps_per_bin"]) * int(payload["horizon_bins"])
        self.calibration = payload.get("style_calibration", {})
        self.canonicalizer = HADCanonicalizer()

    def encode(self, state, target, red_ids, blue_ids):
        from HAD_Env.config import AeroPoint, initial_health, vDomain

        low = np.asarray([x[0] for x in AeroPoint], dtype=np.float32)
        span = np.asarray([x[1] - x[0] for x in AeroPoint], dtype=np.float32)
        red = {x.id: x for x in state.red}
        blue = {x.id: x for x in state.blue}
        rows = []
        entities = [(red[i], 0) for i in red_ids] + [(blue[i], 1) for i in blue_ids] + [(target, 2)]
        for entity, side in entities:
            types = [float(side == i) for i in range(3)]
            rows.append(list((np.asarray(entity.position) - low) / span)
                        + list(np.asarray(entity.velocity) / float(vDomain[1]))
                        + [entity.health / (initial_health if side == 2 else 1.0), float(entity.health > 0)]
                        + types + [max(0.0, 1.0 - state.step / state.max_steps)])
        return self.canonicalizer.to_entity_set(np.asarray(rows, dtype=np.float32))

    def predict(self, state, requests, style):
        if not requests:
            return []
        if state.max_steps != self.horizon_steps:
            raise ValueError("DLOM baseline requires the checkpoint's 50-step task horizon")
        targets = {x.id: x for x in state.targets}
        encoded = []
        for target, red, blue in requests:
            if not (1 <= len(red) <= 4 and 1 <= len(blue) <= 4):
                raise ValueError("DLOM supports local rosters in 1..4 versus 1..4")
            encoded.append(self.encode(state, targets[target], red, blue))

        def pad(field):
            arrays = [getattr(x, field) for x in encoded]
            values = np.zeros((len(arrays), max(map(len, arrays)), 9), np.float32)
            mask = np.zeros(values.shape[:2], bool)
            for i, array in enumerate(arrays):
                values[i, :len(array)] = array
                mask[i, :len(array)] = True
            return torch.as_tensor(values, device=self.device), torch.as_tensor(mask, device=self.device)

        red, red_mask = pad("red_entities")
        blue, blue_mask = pad("blue_entities")
        with torch.inference_mode():
            probabilities = self.model.probabilities(
                torch.as_tensor(np.stack([x.target for x in encoded]), device=self.device),
                red, blue, torch.as_tensor(np.stack([x.context for x in encoded]), device=self.device),
                red_mask, blue_mask, temperature=self.temperature)
            wins = probabilities[:, :self.model.horizon_bins].sum(-1).cpu().numpy()
        offsets = self.calibration.get("offsets", {})
        global_offset = float(self.calibration.get("global_offsets", {}).get(style, 0.0))
        result = []
        for win, (_, r, b) in zip(wins, requests):
            win = float(np.clip(win, 1e-5, 1 - 1e-5))
            offset = float(offsets.get(f"{style}|{len(r)}|{len(b)}", global_offset))
            adjusted = 1.0 / (1.0 + np.exp(-np.clip(np.log(win / (1 - win)) + offset, -60, 60)))
            result.append(float(adjusted))
        return result


def _proxy_pairs(state, red_action, blue_action):
    """Centroid matching for the local proxy, never an execution assignment."""
    red_entities = {x.id: x for x in state.red}
    blue_entities = {x.id: x for x in state.blue}
    pairs = []
    for target in _targets(state):
        reds = [g for g in red_action.groups if g.target == target.id]
        blues = [g for g in blue_action.groups if g.target == target.id]
        matched_blue = set()
        if reds and blues:
            rp = [np.mean([red_entities[i].position for i in g.members], axis=0) for g in reds]
            bp = [np.mean([blue_entities[i].position for i in g.members], axis=0) for g in blues]
            costs = np.linalg.norm(np.asarray(rp)[:, None, :] - np.asarray(bp)[None, :, :], axis=-1)
            ri, bi = linear_sum_assignment(costs)
            for i, j in zip(ri, bi):
                pairs.append((target.id, reds[int(i)].members, blues[int(j)].members))
                matched_blue.add(int(j))
        pairs.extend((target.id, (), group.members) for j, group in enumerate(blues) if j not in matched_blue)
    return pairs


class DLOMSearchPolicy:
    """B1: maximize expected local proxy utility under the known Blue rule."""
    def __init__(self, checkpoint=None, *, limit=16, device="cpu", seed=0, predictor=None):
        if checkpoint is None:
            checkpoint = Path(__file__).resolve().parents[3] / "assets/frozen/dlom.pt"
        self.predictor = predictor if predictor is not None else FrozenDLOM(checkpoint, device)
        self.limit = int(limit)
        self.rng = np.random.default_rng(seed)
        self.config = {"method": "dlom", "limit": self.limit, "seed": seed,
                       "objective": "known_rule_expected_local_log_survival",
                       "pairing_role": "scoring_only", "checkpoint": str(checkpoint)}

    def rank(self, state, candidate_actions: Sequence[Grouping]) -> list[float]:
        from .opponents import distribution

        blues, probabilities = distribution(state, state.opponent)
        probabilities = np.asarray(probabilities, dtype=float)
        if len(blues) != len(probabilities) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
            raise ValueError("known opponent distribution must be normalized")
        profiles = [[_proxy_pairs(state, red, blue) for blue in blues] for red in candidate_actions]
        requests = sorted({pair for row in profiles for profile in row for pair in profile if pair[1] and pair[2]})
        style = "split_rush" if "split" in str(state.opponent) else "rush"
        lookup = dict(zip(requests, self.predictor.predict(state, requests, style)))
        blue_entities = {x.id: x for x in state.blue}
        targets = {x.id: x for x in state.targets}
        result = []
        for row in profiles:
            utilities = []
            for profile in row:
                total = 0.0
                for target, red_ids, blue_ids in profile:
                    if red_ids:
                        survival = lookup[(target, red_ids, blue_ids)]
                    else:
                        # Conservative reachability proxy; not a calibrated
                        # global probability and not a change to physics.
                        reachable = any(np.linalg.norm(np.asarray(blue_entities[i].position) - targets[target].position)
                                        <= 500 + 300 * (state.max_steps - state.step) for i in blue_ids)
                        survival = 0.0 if reachable else 1.0
                    total += float(np.log(np.clip(survival, 1e-6, 1)))
                utilities.append(total / max(1, len(targets)))
            result.append(float(np.dot(probabilities, utilities)))
        return result

    def act(self, state, **kwargs):
        pool = candidates(state, self.limit, self.rng)
        values = self.rank(state, pool)
        index = int(np.argmax(values))
        return _decision(pool[index], value=values[index], released=state.ids("red"),
                         trace={"candidate_count": len(pool), "proxy_values": values})


class ActionValueNetwork(nn.Module):
    """Set attention over physics, previous groups and candidate action groups."""
    def __init__(self, hidden_dim=128, heads=4, layers=2):
        super().__init__()
        # Same observable recurrent memory and previous lower action as the
        # PPO controls; these can affect the next physical transition.
        self.entity = nn.Sequential(nn.Linear(10 + 64 + 27, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.group = nn.Linear(3 * hidden_dim + 3, hidden_dim)
        self.context = nn.Linear(4, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, 2 * hidden_dim,
                                            dropout=0.0, batch_first=True, activation="gelu")
        self.attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def _tokens(self, state, action):
        device = next(self.parameters()).device
        entities = [(x, side) for side, name in enumerate(("red", "blue", "targets"))
                    for x in getattr(state, name) if x.health > 0]
        rows = []
        for x, side in entities:
            memory = list(state.memory.get(x.id, ())) if side == 0 else []
            if len(memory) > 64:
                raise ValueError("frozen controller memory exceeds 64 features")
            memory += [0.0] * (64 - len(memory))
            last = state.last_actions.get(x.id, -1) if side == 0 else -1
            rows.append(list(np.asarray(x.position) / 10000.0) + list(np.asarray(x.velocity) / 300.0)
                        + [x.health] + [float(side == i) for i in range(3)]
                        + memory + [float(last == i) for i in range(27)])
        embedded = self.entity(torch.tensor(rows, dtype=torch.float32, device=device))
        red = {x.id: embedded[i] for i, (x, side) in enumerate(entities) if side == 0}
        target = {x.id: embedded[i] for i, (x, side) in enumerate(entities) if side == 2}
        tokens = list(embedded.unbind(0))
        zero = embedded.new_zeros(embedded.shape[-1])
        for old, grouping in ((1.0, state.previous), (0.0, action)):
            if grouping is None:
                continue
            grouping = grouping.prune(state.ids("red"))
            for group in grouping.groups:
                members = torch.stack([red[i] for i in group.members if i in red])
                extras = embedded.new_tensor([len(members) / 4.0, old, 0.0])
                tokens.append(self.group(torch.cat((members.sum(0), members.mean(0), target[group.target], extras))))
            reserves = [red[i] for i in grouping.reserve if i in red]
            if reserves:
                members = torch.stack(reserves)
                extras = embedded.new_tensor([len(members) / 4.0, old, 1.0])
                tokens.append(self.group(torch.cat((members.sum(0), members.mean(0), zero, extras))))
        context = embedded.new_tensor([state.step / state.max_steps, 1 - state.step / state.max_steps,
                                       len(state.ids("red")) / 16.0, len(state.ids("blue")) / 16.0])
        return torch.stack([self.context(context)] + tokens)

    def forward(self, states, actions):
        tokens = [self._tokens(state, action) for state, action in zip(states, actions)]
        padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        lengths = torch.tensor([len(x) for x in tokens], device=padded.device)
        mask = torch.arange(padded.shape[1], device=padded.device)[None, :] >= lengths[:, None]
        encoded = self.attention(padded, src_key_padding_mask=mask)
        return self.head(encoded[:, 0]).squeeze(-1)


class AlmaStylePolicy(nn.Module):
    """B5: proposals, action-Q selection, replay TD, and winner-trajectory NLL.

    Candidate maximization changes the behavior distribution. This control is
    trained with Q learning and proposal imitation, never with a PPO ratio.
    """
    def __init__(self, *, hidden_dim=128, heads=4, layers=2, num_candidates=8,
                 batch_size=16, update_every=16, learning_rate=3e-4,
                 replay_capacity=2048, target_tau=0.02, proposal_weight=0.01,
                 gamma=1.0, seed=0):
        super().__init__()
        from .policy import GroupingPolicy

        if num_candidates < 1 or batch_size < 1 or update_every < 1 or replay_capacity < batch_size:
            raise ValueError("invalid ALMA-style candidate/replay configuration")
        if not 0 < target_tau <= 1 or not 0 <= gamma <= 1 or proposal_weight <= 0:
            raise ValueError("invalid ALMA-style discount or update weights")
        self.config = dict(method="alma_style", hidden_dim=hidden_dim, heads=heads, layers=layers,
                           num_candidates=num_candidates, batch_size=batch_size, update_every=update_every,
                           learning_rate=learning_rate, replay_capacity=replay_capacity,
                           target_tau=target_tau, proposal_weight=proposal_weight, gamma=gamma, seed=seed)
        self.proposal = GroupingPolicy(mode="full", hidden_dim=hidden_dim, heads=heads, layers=layers)
        self.q = ActionValueNetwork(hidden_dim, heads, layers)
        self.target = copy.deepcopy(self.q).requires_grad_(False).eval()
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=learning_rate)
        self.proposal_optimizer = torch.optim.Adam(self.proposal.parameters(), lr=learning_rate)
        self.replay = deque(maxlen=replay_capacity)
        self.rng = np.random.default_rng(seed)
        self.pending = None
        self.observed = 0
        self.last_update = 0
        self.updates = 0

    @torch.no_grad()
    def _proposals(self, state, deterministic=False):
        # Greedy evaluation remains an optional distinct protocol. The default
        # samples all proposals and chooses their action-Q maximizer.
        # All action-generation randomness follows the episode's Torch RNG.
        # Replay index sampling has a separate, checkpointed NumPy stream.
        return [self.proposal.act(state, deterministic=deterministic)
                for _ in range(self.config["num_candidates"])]

    @torch.no_grad()
    def act(self, state, deterministic=False, **kwargs):
        choices = self._proposals(state, deterministic)
        values = self.q([state] * len(choices), [x.action for x in choices])
        winner = choices[int(values.argmax())]
        self.pending = (state.to_dict(), winner.action.to_dict(), copy.deepcopy(winner.trace))
        return _decision(winner.action, value=values.max().detach(), released=winner.released_ids,
                         trace={"algorithm": "alma_style", "proposal_trace": winner.trace,
                                "decode_steps": sum(len(x.trace.get('selection', [])) + len(x.trace.get('repairs', [])) for x in choices),
                                "candidate_count": len(choices), "candidate_q": values.cpu().tolist()})

    def observe(self, state, action, reward, delta, done, next_state):
        if delta < 1:
            raise ValueError("macro transitions must advance physical time")
        current, selected = state.to_dict(), action.to_dict()
        trace = None
        if self.pending is not None and self.pending[:2] == (current, selected):
            trace = self.pending[2]
        if not done and next_state is None:
            raise ValueError("nonterminal transition needs its next state")
        self.replay.append(dict(state=current, action=selected, reward=float(reward), delta=int(delta),
                                done=bool(done), next_state=None if next_state is None else next_state.to_dict(), trace=trace))
        self.pending = None
        self.observed += 1

    def update(self):
        config = self.config
        if len(self.replay) < config["batch_size"] or self.observed - self.last_update < config["update_every"]:
            return {}
        indices = self.rng.choice(len(self.replay), config["batch_size"], replace=False)
        batch = [self.replay[int(i)] for i in indices]
        states = [DecisionState.from_dict(x["state"]) for x in batch]
        actions = [Grouping.from_dict(x["action"]) for x in batch]
        predictions = self.q(states, actions)
        targets = []
        self.target.eval()
        with torch.no_grad():
            for sample in batch:
                value = sample["reward"]
                if not sample["done"]:
                    state = DecisionState.from_dict(sample["next_state"])
                    choices = self._proposals(state)
                    candidate_actions = [x.action for x in choices]
                    online_values = self.q([state] * len(choices), candidate_actions)
                    winner = candidate_actions[int(online_values.argmax())]
                    target_value = self.target([state], [winner])[0]
                    value += config["gamma"] ** sample["delta"] * float(target_value)
                targets.append(value)
        q_loss = F.smooth_l1_loss(predictions, predictions.new_tensor(targets))
        if not torch.isfinite(q_loss):
            raise FloatingPointError("nonfinite action-Q loss")
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.q_optimizer.step()
        # Re-propose and re-rank at replay states using the current networks.
        # Historical winners could only reflect an earlier, inaccurate Q.
        winning_traces = []
        with torch.no_grad():
            for state in states:
                choices = self._proposals(state)
                values = self.q([state] * len(choices), [x.action for x in choices])
                winning_traces.append(choices[int(values.argmax())].trace)
        likelihoods = [self.proposal.evaluate_action(state, trace)[0]
                       for state, trace in zip(states, winning_traces)]
        proposal_loss = predictions.new_zeros(())
        if likelihoods:
            proposal_loss = -torch.stack(likelihoods).mean()
            if not torch.isfinite(proposal_loss):
                raise FloatingPointError("nonfinite winner trajectory likelihood")
            self.proposal_optimizer.zero_grad(set_to_none=True)
            (config["proposal_weight"] * proposal_loss).backward()
            nn.utils.clip_grad_norm_(self.proposal.parameters(), 1.0)
            self.proposal_optimizer.step()
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.q.parameters()):
                target.lerp_(online, config["target_tau"])
        self.last_update = self.observed
        self.updates += 1
        return {"q_loss": float(q_loss.detach()), "proposal_nll": float(proposal_loss.detach()),
                "replay_size": len(self.replay), "updates": self.updates,
                "winner_traces": len(likelihoods), "observed_transitions": self.observed}

    def training_state_dict(self):
        return {"schema_version": "alma-style-v2", "config": self.config,
                "model": self.state_dict(), "q_optimizer": self.q_optimizer.state_dict(),
                "proposal_optimizer": self.proposal_optimizer.state_dict(), "replay": list(self.replay),
                "rng": copy.deepcopy(self.rng.bit_generator.state), "observed": self.observed,
                "last_update": self.last_update, "updates": self.updates,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def load_training_state_dict(self, state):
        if state.get("schema_version") != "alma-style-v2" or state["config"] != self.config:
            raise ValueError("ALMA-style resume configuration differs")
        self.load_state_dict(state["model"], strict=True)
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.proposal_optimizer.load_state_dict(state["proposal_optimizer"])
        self.replay.clear()
        self.replay.extend(state["replay"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.observed, self.last_update, self.updates = (int(state[k]) for k in ("observed", "last_update", "updates"))
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        self.pending = None


__all__ = ["balanced_initial", "candidates", "StaticPolicy", "FrozenDLOM", "DLOMSearchPolicy",
           "ActionValueNetwork", "AlmaStylePolicy", "BaselineDecision"]

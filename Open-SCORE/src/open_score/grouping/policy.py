"""One macro-action policy over identity-preserving coalition repairs.

The trace is the augmented action used by PPO: all learned selection and
repair probabilities are summed.  Exogenous release counts and processing
orders are recorded but have no learnable probability factor.  ``entropy``
is the sum of conditional categorical entropies along the recorded prefix,
not an exact entropy calculation over every possible construction trace.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .domain import DecisionState, Group, Grouping


@dataclass
class PolicyDecision:
    action: Grouping
    trace: dict
    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    released_ids: tuple[int, ...]


@dataclass
class _Encoding:
    context: Tensor
    red: dict[int, Tensor]
    targets: dict[int, Tensor]
    previous: Grouping


class GroupingPolicy(nn.Module):
    """Shared full/selective/random/rule decoder with a global state critic.

    ``release_count`` is a controlled-diagnostic override.  It masks STOP in
    selective mode until exactly that many identities have been selected.
    At step zero every mode always deploys all members through full repair.
    Deterministic decoding still uses an exogenous random processing order;
    pass a seeded NumPy Generator as ``rng`` for reproducible comparisons.
    """

    MODES = frozenset({"selective", "full", "random", "rule"})
    MEMORY_DIM = 64
    LOWER_ACTION_DIM = 27
    FEATURE_DIM = 3 + 3 + 3 + 1 + 1 + MEMORY_DIM + LOWER_ACTION_DIM

    def __init__(self, mode: str = "selective", hidden_dim: int = 128,
                 heads: int = 4, layers: int = 2,
                 release_distribution: Mapping | None = None):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads or layers < 1:
            raise ValueError("positive dimensions and hidden_dim divisible by heads required")
        self.mode, self.hidden_dim = mode, int(hidden_dim)
        self.heads, self.layers = int(heads), int(layers)
        self.release_distribution: dict[int, float] = {}
        self.set_release_distribution(release_distribution or {})
        self.entity_input = nn.Sequential(nn.Linear(self.FEATURE_DIM, hidden_dim), nn.Tanh())
        self.context_input = nn.Linear(4, hidden_dim)
        self.old_group = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.Tanh())
        self.reserve_relation = nn.Parameter(torch.zeros(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, dim_feedforward=2 * hidden_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.selection_query = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.Tanh())
        self.selection_key = nn.Linear(hidden_dim, hidden_dim)
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.repair_query = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.Tanh())
        self.group_key = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.Tanh())
        self.new_key = nn.Linear(hidden_dim, hidden_dim)
        self.reserve_key = nn.Linear(hidden_dim, hidden_dim)
        self.repair_bias = nn.Parameter(torch.zeros(3))

    @property
    def config(self) -> dict:
        return {"mode": self.mode, "hidden_dim": self.hidden_dim, "heads": self.heads,
                "layers": self.layers, "release_distribution": dict(self.release_distribution)}

    def set_release_distribution(self, mapping: Mapping) -> None:
        """Freeze B3's count frequencies; counts are clipped to live size later.

        Empty mapping uses a uniform count prior on 0..R during development.
        Formal matched-count evaluation should set training-set frequencies.
        The configuration, rather than the tensor state_dict, persists this.
        """
        values: dict[int, float] = {}
        for raw_count, raw_weight in mapping.items():
            count, weight = int(raw_count), float(raw_weight)
            if str(count) != str(raw_count) and float(raw_count) != count:
                raise ValueError("release distribution keys must be integer counts")
            if count < 0 or weight < 0 or not math.isfinite(weight):
                raise ValueError("release distribution needs nonnegative counts and weights")
            values[count] = values.get(count, 0.0) + weight
        total = sum(values.values())
        if values and (total <= 0 or not math.isfinite(total)):
            raise ValueError("release distribution needs positive total weight")
        self.release_distribution = {key: value / total for key, value in sorted(values.items())}

    def _tensor(self, values) -> Tensor:
        parameter = next(self.parameters())
        return torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device)

    def _encode(self, state: DecisionState) -> _Encoding:
        if state.max_steps < 1:
            raise ValueError("state max_steps must be positive")
        red, blue, targets = (state.alive(side) for side in ("red", "blue", "targets"))
        previous = state.previous.prune(entity.id for entity in red)
        rows = []
        for kind, entities in enumerate((red, blue, targets)):
            for entity in entities:
                memory = state.memory.get(entity.id, (0.0,) * self.MEMORY_DIM) if kind == 0 else (0.0,) * self.MEMORY_DIM
                if len(memory) != self.MEMORY_DIM:
                    raise ValueError("lower-controller memory must have 64 entries per identity")
                history = [0.0] * self.LOWER_ACTION_DIM
                action = int(state.last_actions.get(entity.id, -1)) if kind == 0 else -1
                if not -1 <= action < self.LOWER_ACTION_DIM:
                    raise ValueError("last lower action must be -1 or 0..26")
                if action >= 0:
                    history[action] = 1.0
                rows.append([float(kind == i) for i in range(3)]
                            + [float(v) / 2500.0 for v in entity.position]
                            + [float(v) / 500.0 for v in entity.velocity]
                            + [float(entity.health) / (1.2 if kind == 2 else 1.0),
                               max(0.0, 1.0 - state.step / state.max_steps)]
                            + list(memory) + history)
        base = self.entity_input(self._tensor(rows).reshape(-1, self.FEATURE_DIM))
        red_index = {entity.id: index for index, entity in enumerate(red)}
        target_index = {entity.id: len(red) + len(blue) + index for index, entity in enumerate(targets)}
        zero = base.new_zeros(self.hidden_dim)
        relations, group_tokens = {}, []
        for group in previous.groups:
            members = [base[red_index[i]] for i in group.members if i in red_index]
            if not members:
                continue
            target = base[target_index[group.target]] if group.target in target_index else zero
            token = self.old_group(torch.cat((torch.stack(members).mean(0), target, self._tensor([len(members) / 4.0]))))
            group_tokens.append(token)
            relations.update({i: token for i in group.members})
        tokens = [base[index] + relations.get(entity.id, self.reserve_relation)
                  for index, entity in enumerate(red)]
        tokens.extend(base[index] for index in range(len(red), len(rows)))
        tokens.extend(group_tokens)
        context = self.context_input(self._tensor([
            state.step / state.max_steps, math.log1p(len(red)),
            math.log1p(len(blue)), math.log1p(len(targets)),
        ]))
        # No positional encoding and no identity embeddings: row order is not a feature.
        encoded = self.encoder(torch.stack([context] + tokens).unsqueeze(0))[0]
        return _Encoding(encoded[0], {i: encoded[1 + index] for i, index in red_index.items()},
                         {i: encoded[1 + index] for i, index in target_index.items()}, previous)

    def value(self, state: DecisionState) -> Tensor:
        return self.value_head(self._encode(state).context).squeeze(-1)

    def _selection(self, encoding: _Encoding, remaining: list[int], selected: list[int], stop: bool) -> Categorical:
        pooled = (torch.stack([encoding.red[i] for i in selected]).mean(0)
                  if selected else encoding.context.new_zeros(self.hidden_dim))
        query = self.selection_query(torch.cat((encoding.context, pooled,
                                                self._tensor([len(selected) / max(1, len(encoding.red))]))))
        logits = [(self.selection_key(encoding.red[i]) * query).sum() / math.sqrt(self.hidden_dim)
                  for i in remaining]
        if stop:
            logits.append(self.stop_head(query).squeeze(-1))
        if not logits:
            raise ValueError("selection has no legal choices")
        return Categorical(logits=torch.stack(logits))

    def _repair(self, encoding: _Encoding, agent: int, groups: list[Group], reserve: list[int]):
        pool = (torch.stack([encoding.red[i] for i in reserve]).mean(0)
                if reserve else encoding.context.new_zeros(self.hidden_dim))
        query = self.repair_query(torch.cat((encoding.context, encoding.red[agent], pool)))
        choices, logits = [], []
        for group in groups:
            if len(group.members) >= 4:
                continue
            members = torch.stack([encoding.red[i] for i in group.members]).mean(0)
            key = self.group_key(torch.cat((members, encoding.targets[group.target], self._tensor([len(group.members) / 4.0]))))
            choices.append({"kind": "join", "target": group.target, "members": list(group.members)})
            logits.append((key * query).sum() / math.sqrt(self.hidden_dim) + self.repair_bias[0])
        for target, token in encoding.targets.items():
            choices.append({"kind": "new", "target": target})
            logits.append((self.new_key(token) * query).sum() / math.sqrt(self.hidden_dim) + self.repair_bias[1])
        choices.append({"kind": "reserve"})
        logits.append((self.reserve_key(pool + self.reserve_relation) * query).sum() / math.sqrt(self.hidden_dim) + self.repair_bias[2])
        return Categorical(logits=torch.stack(logits)), choices

    @staticmethod
    def _install(agent: int, choice: dict, groups: list[Group], reserve: list[int]) -> None:
        if choice["kind"] == "reserve":
            reserve.append(agent)
        elif choice["kind"] == "new":
            groups.append(Group(choice["target"], (agent,)))
        else:
            index = next(i for i, group in enumerate(groups)
                         if group.target == choice["target"] and list(group.members) == choice["members"])
            group = groups[index]
            groups[index] = Group(group.target, group.members + (agent,))

    @staticmethod
    def _working(previous: Grouping, released: list[int]):
        released_set = set(released)
        groups = [Group(group.target, tuple(i for i in group.members if i not in released_set))
                  for group in previous.groups if any(i not in released_set for i in group.members)]
        return groups, [i for i in previous.reserve if i not in released_set]

    @staticmethod
    def _order(ids: list[int], rng) -> list[int]:
        if rng is not None:
            if hasattr(rng, "permutation"):
                return [int(i) for i in rng.permutation(ids)]
            result = list(ids)
            rng.shuffle(result)
            return result
        return [ids[i] for i in torch.randperm(len(ids)).tolist()]

    def _random_count(self, size: int, rng) -> int:
        if not self.release_distribution:
            weights = [1.0] * (size + 1)
            counts = list(range(size + 1))
        else:
            counts, weights = zip(*self.release_distribution.items())
        if rng is not None and hasattr(rng, "choice") and hasattr(rng, "permutation"):
            total = sum(weights)
            value = rng.choice(counts, p=[v / total for v in weights])
        else:
            index = int(torch.multinomial(torch.tensor(weights, dtype=torch.float64), 1).item())
            value = counts[index]
        return min(size, int(value))

    @staticmethod
    def _rule_order(state: DecisionState, previous: Grouping) -> tuple[list[int], int]:
        """Under-capacity groups are a public proxy for groups needing repair.

        This does not claim a group was damaged solely because it has <4
        members.  Prefer its members, then nearby reserves, then other peers.
        Without an override, draw just enough reserves to fill its vacancies.
        """
        positions = {entity.id: entity.position for entity in state.alive("red")}
        affected = [group for group in previous.groups if len(group.members) < 4]
        first = [i for group in sorted(affected, key=lambda g: len(g.members)) for i in group.members]
        anchors = [positions[i] for i in first]
        def distance(i):
            return min((sum((a - b) ** 2 for a, b in zip(positions[i], point)) for point in anchors), default=0.0)
        remaining = sorted((i for i in positions if i not in set(first)),
                           key=lambda i: (i not in previous.reserve, distance(i), positions[i], i))
        vacancies = sum(4 - len(group.members) for group in affected)
        count = len(first) + min(len(previous.reserve), vacancies)
        return first + remaining, count

    @staticmethod
    def _draw(distribution: Categorical, deterministic: bool) -> int:
        return int(distribution.logits.argmax().item() if deterministic else distribution.sample().item())

    def act(self, state: DecisionState, deterministic: bool = False, rng=None,
            release_count: int | None = None) -> PolicyDecision:
        encoding = self._encode(state)
        ids = list(encoding.red)
        if not encoding.targets or state.step >= state.max_steps:
            raise ValueError("a terminal state cannot request a grouping action")
        encoding.previous.validate(ids, encoding.targets)
        if release_count is not None and (int(release_count) != release_count or not 0 <= release_count <= len(ids)):
            raise ValueError("release_count must be an integer in 0..live size")
        forced = None if release_count is None else int(release_count)
        initial = state.step == 0
        trace = {"version": 1, "mode": self.mode, "initial": initial,
                 "forced_count": forced, "selection": [], "repairs": []}
        log_prob = encoding.context.sum() * 0.0
        entropy = log_prob
        if not initial and self.mode == "selective":
            trace["release_source"] = "policy"
            released, remaining = [], list(ids)
            limit = len(ids) if forced is None else forced
            while remaining and len(released) < limit:
                distribution = self._selection(encoding, remaining, released, stop=forced is None)
                index = self._draw(distribution, deterministic)
                log_prob = log_prob + distribution.log_prob(torch.tensor(index, device=encoding.context.device))
                entropy = entropy + distribution.entropy()
                if index == len(remaining):
                    trace["selection"].append(None)
                    break
                agent = remaining.pop(index)
                released.append(agent)
                trace["selection"].append(agent)
        else:
            trace["release_source"] = "initial" if initial else "exogenous"
            if initial or self.mode == "full":
                released = self._order(ids, rng)[:len(ids) if initial or forced is None else forced]
            elif self.mode == "random":
                count = self._random_count(len(ids), rng) if forced is None else forced
                released = self._order(ids, rng)[:count]
            else:
                ordered, count = self._rule_order(state, encoding.previous)
                released = ordered[:count if forced is None else forced]
        trace["released_ids"] = list(released)
        groups, reserve = self._working(encoding.previous, released)
        for agent in released:
            distribution, choices = self._repair(encoding, agent, groups, reserve)
            index = self._draw(distribution, deterministic)
            log_prob = log_prob + distribution.log_prob(torch.tensor(index, device=encoding.context.device))
            entropy = entropy + distribution.entropy()
            choice = choices[index]
            trace["repairs"].append({"agent": agent, **choice})
            self._install(agent, choice, groups, reserve)
        action = Grouping(tuple(groups), tuple(reserve)).validate(ids, encoding.targets)
        trace["action"] = action.to_dict()
        return PolicyDecision(action, trace, log_prob, entropy,
                              self.value_head(encoding.context).squeeze(-1), tuple(released))

    def evaluate_action(self, state: DecisionState, trace: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Teacher-force a recorded macro-action with its exact prefix masks."""
        encoding = self._encode(state)
        ids = list(encoding.red)
        if not encoding.targets or state.step >= state.max_steps:
            raise ValueError("a terminal state has no action probability")
        encoding.previous.validate(ids, encoding.targets)
        if trace.get("version") != 1 or trace.get("mode") != self.mode or trace.get("initial") != (state.step == 0):
            raise ValueError("trace version, mode or initial-state flag does not match")
        released = list(map(int, trace["released_ids"]))
        if len(released) != len(set(released)) or not set(released).issubset(ids):
            raise ValueError("trace releases a duplicate or nonliving identity")
        forced = trace.get("forced_count")
        if forced is not None and (int(forced) != forced or not 0 <= forced <= len(ids)):
            raise ValueError("invalid trace forced_count")
        expected_source = "initial" if state.step == 0 else "policy" if self.mode == "selective" else "exogenous"
        if trace.get("release_source") != expected_source:
            raise ValueError("trace release source differs from policy")
        log_prob = encoding.context.sum() * 0.0
        entropy = log_prob
        if expected_source == "policy":
            selected, remaining = [], list(ids)
            selection = trace["selection"]
            stopped = False
            for offset, choice in enumerate(selection):
                if not remaining or stopped or (forced is not None and len(selected) >= forced):
                    raise ValueError("selection continues after its stopping boundary")
                distribution = self._selection(encoding, remaining, selected, stop=forced is None)
                if choice is None:
                    if forced is not None or offset != len(selection) - 1:
                        raise ValueError("STOP is illegal at this prefix")
                    index, stopped = len(remaining), True
                else:
                    if int(choice) not in remaining:
                        raise ValueError("selection repeats or invents an identity")
                    index = remaining.index(int(choice))
                    selected.append(remaining.pop(index))
                log_prob = log_prob + distribution.log_prob(torch.tensor(index, device=encoding.context.device))
                entropy = entropy + distribution.entropy()
            if selected != released or (forced is not None and len(selected) != forced):
                raise ValueError("selection and released identities disagree")
            if forced is None and remaining and not stopped:
                raise ValueError("selection trace is missing its STOP action")
        else:
            if trace["selection"]:
                raise ValueError("exogenous release has no policy selection factors")
            expected_count = len(ids) if state.step == 0 or (self.mode == "full" and forced is None) else forced
            if expected_count is not None and len(released) != expected_count:
                raise ValueError("trace contradicts its forced release count")
        groups, reserve = self._working(encoding.previous, released)
        if len(trace["repairs"]) != len(released):
            raise ValueError("every released identity must have one repair")
        for agent, repair in zip(released, trace["repairs"]):
            if repair.get("agent") != agent:
                raise ValueError("repair order differs from release order")
            distribution, choices = self._repair(encoding, agent, groups, reserve)
            choice = {key: value for key, value in repair.items() if key != "agent"}
            if choice not in choices:
                raise ValueError("repair violates the recorded prefix's legal actions")
            index = choices.index(choice)
            log_prob = log_prob + distribution.log_prob(torch.tensor(index, device=encoding.context.device))
            entropy = entropy + distribution.entropy()
            self._install(agent, choice, groups, reserve)
        action = Grouping(tuple(groups), tuple(reserve)).validate(ids, encoding.targets)
        if action.to_dict() != trace.get("action"):
            raise ValueError("trace action differs from its reconstructed grouping")
        return log_prob, entropy, self.value_head(encoding.context).squeeze(-1)


__all__ = ["GroupingPolicy", "PolicyDecision"]

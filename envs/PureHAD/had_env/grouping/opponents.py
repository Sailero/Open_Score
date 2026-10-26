"""Public, immutable complete opponent policies; lower execution is rush."""
from __future__ import annotations

import numpy as np

from .domain import DecisionState, Group, Grouping


OPPONENTS = ("reactive", "concentrated", "balanced")


def _allocate(state: DecisionState, counts: tuple[int, ...]) -> Grouping:
    available = {entity.id: entity for entity in state.alive("blue")}
    groups = []
    for target, count in zip(state.targets, counts):
        chosen = sorted(available, key=lambda i: (
            float(np.linalg.norm(np.asarray(available[i].position) - target.position)), i))[:count]
        while chosen:
            first = chosen.pop(0)
            nearest = sorted(chosen, key=lambda i: (
                float(np.linalg.norm(np.asarray(available[i].position) - available[first].position)), i))[:3]
            members = (first, *nearest)
            groups.append(Group(target.id, members))
            chosen = [i for i in chosen if i not in nearest]
            for i in members:
                available.pop(i)
    result = Grouping(tuple(groups))
    return result.validate(state.ids("blue"), (target.id for target in state.targets))


def distribution(state: DecisionState, opponent: str) -> tuple[list[Grouping], np.ndarray]:
    """Return the known action law from the decision-before-commit snapshot.

    Reactive uses physical Red coverage, not the Red action being selected.
    Its target probabilities are softmax(-2 * coverage - 0.25 * travel),
    where coverage sums exp(-distance / 850) times surviving Red health and
    travel is mean Blue distance / 5000. Each sampled target gets all Blue.
    This is a fixed stochastic Markov rule, independently sampled each event.
    """
    if opponent not in OPPONENTS:
        raise ValueError(f"opponent must be one of {OPPONENTS}")
    n, count = len(state.ids("blue")), len(state.targets)
    if count == 0:
        raise ValueError("At least one fixed target is required")
    if n == 0:
        return [Grouping(())], np.ones(1)
    if opponent == "balanced":
        quotas = tuple(n // count + int(i < n % count) for i in range(count))
        return [_allocate(state, quotas)], np.ones(1)
    travel = np.asarray([np.mean([np.linalg.norm(np.asarray(blue.position) - target.position)
                                for blue in state.alive("blue")]) / 5000.0
                         for target in state.targets])
    actions = [_allocate(state, tuple(n if j == i else 0 for j in range(count)))
               for i in range(count)]
    if opponent == "concentrated":
        index = int(np.argmin(travel))
        return [actions[index]], np.ones(1)
    coverage = np.asarray([sum(max(0.0, red.health) * np.exp(
        -np.linalg.norm(np.asarray(red.position) - target.position) / 850.0)
        for red in state.alive("red")) for target in state.targets])
    logits = -2.0 * coverage - 0.25 * travel
    probabilities = np.exp(logits - logits.max())
    return actions, probabilities / probabilities.sum()


def sample(state: DecisionState, opponent: str, rng: np.random.Generator) -> Grouping:
    actions, probabilities = distribution(state, opponent)
    return actions[int(rng.choice(len(actions), p=probabilities))]

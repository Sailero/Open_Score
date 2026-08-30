"""Deterministic synthetic benchmark used only to smoke-test the S2 pipeline."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .canonical import HADCanonicalizer
from .data import Stage2Record, record_from_rollout


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0))))


def _synthetic_had_state(
    rng: np.random.Generator, defender_count: int, attacker_count: int
) -> Tuple[np.ndarray, float]:
    """Create a valid normalized HAD-like state and a latent root difficulty."""

    target = np.zeros(11, dtype=np.float32)
    target[:3] = rng.uniform((0.20, 0.25, 0.10), (0.45, 0.75, 0.40))
    target[6:8] = (1.0, 1.0)
    target[10] = 1.0
    entities = []
    for side, count, type_index, offset in (
        ("defender", defender_count, 8, 0.10),
        ("attacker", attacker_count, 9, 0.28),
    ):
        del side
        for _ in range(count):
            entity = np.zeros(11, dtype=np.float32)
            direction = rng.normal(size=3)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            entity[:3] = np.clip(target[:3] + offset * direction, 0.0, 1.0)
            entity[3:6] = rng.normal(0.0, 0.12, size=3)
            entity[6:8] = (1.0, 1.0)
            entity[type_index] = 1.0
            entities.append(entity)
    difficulty = float(rng.normal() + 0.8 * (target[1] - 0.5))
    state = np.stack(entities + [target])
    return state, difficulty


def generate_synthetic_records(
    *,
    n_roots: int = 96,
    scales: Sequence[Tuple[int, int]] = ((1, 1), (2, 1), (2, 2), (3, 2), (4, 3)),
    defender_policies: Sequence[str] = ("guard", "intercept", "adaptive"),
    attacker_policies: Sequence[str] = ("rush", "split_rush", "feint"),
    replicates_per_candidate: int = 2,
    horizon_steps: int = 80,
    command_steps: int = 20,
    seed: int = 7,
) -> List[Stage2Record]:
    """Generate a learnable, non-transitive pipeline test dataset.

    These rows are not environment evidence.  Their only purpose is to catch
    data leakage, model, calibration, serialization, and metric regressions.
    """

    if n_roots < 3 or replicates_per_candidate < 1:
        raise ValueError("synthetic smoke needs at least three roots and one replicate")
    rng = np.random.default_rng(seed)
    canonicalizer = HADCanonicalizer()
    matchup = np.asarray(
        [
            [0.60, -0.40, 0.10],
            [-0.20, 0.65, -0.35],
            [-0.45, -0.10, 0.70],
        ],
        dtype=np.float32,
    )
    records: List[Stage2Record] = []
    for root_index in range(n_roots):
        defender_count, attacker_count = scales[root_index % len(scales)]
        raw_state, difficulty = _synthetic_had_state(rng, defender_count, attacker_count)
        state = canonicalizer(raw_state)
        root_id = f"synthetic:root-{root_index:05d}"
        for defender_index, defender_policy in enumerate(defender_policies):
            for attacker_index, attacker_policy in enumerate(attacker_policies):
                policy_effect = float(
                    matchup[defender_index % matchup.shape[0], attacker_index % matchup.shape[1]]
                )
                defender_margin = (
                    0.75 * (defender_count - attacker_count)
                    + 1.15 * policy_effect
                    - 0.65 * difficulty
                )
                breach_probability = _sigmoid(-defender_margin)
                timeout_probability = 0.05 + 0.12 * np.exp(-abs(defender_margin))
                for replicate in range(replicates_per_candidate):
                    draw = float(rng.random())
                    if draw < timeout_probability:
                        outcome = "timeout"
                        terminal_steps = horizon_steps
                    elif draw < timeout_probability + (1.0 - timeout_probability) * breach_probability:
                        outcome = "breach"
                        fraction = np.clip(rng.beta(1.5, 2.5) * (1.1 - 0.35 * breach_probability), 0.03, 0.98)
                        terminal_steps = max(1, min(horizon_steps - 1, int(round(fraction * horizon_steps))))
                    else:
                        outcome = "defender_win"
                        fraction = np.clip(rng.beta(2.2, 1.8) * (0.75 + 0.15 * breach_probability), 0.03, 0.98)
                        terminal_steps = max(1, min(horizon_steps - 1, int(round(fraction * horizon_steps))))
                    candidate_id = (
                        f"{defender_policy}@synthetic-policy-v1"
                        f"__vs__{attacker_policy}@synthetic-policy-v1"
                    )
                    rollout_id = f"{root_id}:{candidate_id}:rep-{replicate}"
                    records.append(
                        record_from_rollout(
                            canonical_state=state,
                            outcome=outcome,
                            terminal_steps=terminal_steps,
                            environment_id="SYNTHETIC_PIPELINE_ONLY",
                            scenario_id=f"had_like_{defender_count}v{attacker_count}",
                            lineage_group_id=f"synthetic:lineage-{root_index:05d}",
                            root_id=root_id,
                            rollout_id=rollout_id,
                            seed=seed + root_index * 100 + replicate,
                            defender_count=defender_count,
                            attacker_count=attacker_count,
                            defender_policy_id=defender_policy,
                            attacker_policy_id=attacker_policy,
                            defender_policy_version="synthetic-policy-v1",
                            attacker_policy_version="synthetic-policy-v1",
                            capability_version="synthetic-capability-v1",
                            horizon_steps=horizon_steps,
                            command_steps=command_steps,
                            behavior_context=(
                                float(defender_index) / max(1, len(defender_policies) - 1),
                                float(attacker_index) / max(1, len(attacker_policies) - 1),
                            ),
                            candidate_id=candidate_id,
                        )
                    )
    return records

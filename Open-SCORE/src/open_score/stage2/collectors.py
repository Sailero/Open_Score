"""Rollout-to-record adapters for Stage 2."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .canonical import HADCanonicalizer
from .data import Stage2Record, record_from_rollout


def _had_controller(style: str):
    # Import lazily so generic Stage-2 dataset readers do not import HAD/pygame.
    from open_score.stage1 import RandomController, RuleBasedController

    if style == "random":
        return RandomController("random")
    return RuleBasedController(style)


def record_from_had_episode(
    episode,
    *,
    lineage_group_id: str,
    root_id: str,
    rollout_id: str,
    seed: int,
    scenario_id: str,
    horizon_steps: int,
    command_steps: int,
    defender_policy_version: str,
    attacker_policy_version: str,
    capability_version: str,
) -> Stage2Record:
    """Convert any HAD ``CompetitiveEpisode`` (rule or learned) to S2 data."""

    defender_count, attacker_count = map(int, episode.red.scale)
    initial = episode.red.observations[0]
    final_entities = episode.red.observations[-1]["state_entities"]
    target_alive = bool(final_entities[final_entities[:, 10] > 0.5, 7].item() > 0.5)
    attackers_alive = bool(np.any(final_entities[final_entities[:, 9] > 0.5, 7] > 0.5))
    if not target_alive:
        outcome = "breach"
    elif not attackers_alive:
        outcome = "defender_win"
    elif episode.length >= horizon_steps:
        outcome = "timeout"
    else:
        outcome = "defender_win"
    candidate_id = (
        f"{episode.red_policy_name}@{defender_policy_version}"
        f"__vs__{episode.blue_policy_name}@{attacker_policy_version}"
    )
    return record_from_rollout(
        canonical_state=HADCanonicalizer().from_observation(initial),
        outcome=outcome,
        terminal_steps=episode.length,
        environment_id="HAD",
        scenario_id=scenario_id,
        lineage_group_id=lineage_group_id,
        root_id=root_id,
        rollout_id=rollout_id,
        seed=int(seed),
        defender_count=defender_count,
        attacker_count=attacker_count,
        defender_policy_id=episode.red_policy_name,
        attacker_policy_id=episode.blue_policy_name,
        defender_policy_version=defender_policy_version,
        attacker_policy_version=attacker_policy_version,
        capability_version=capability_version,
        horizon_steps=horizon_steps,
        command_steps=command_steps,
        candidate_id=candidate_id,
    )


def collect_had_records(
    *,
    scales: Sequence[Tuple[int, int]],
    seeds: Optional[Iterable[int]] = None,
    seeds_by_scale: Optional[Mapping[str, Sequence[int]]] = None,
    defender_policies: Sequence[str] = ("guard", "intercept"),
    attacker_policies: Sequence[str] = ("rush", "split_rush"),
    max_steps: int = 80,
    command_steps: int = 20,
    capability_version: str = "had-stage1-rules-v1",
    policy_version: str = "rule-v1",
) -> List[Stage2Record]:
    """Collect counterfactual initial-root rollouts from the HAD S1 runner.

    All policy pairs executed from the same ``(scale, seed)`` reset share one
    ``root_id``.  This is the critical property that lets the downstream split
    keep counterfactual siblings out of different data partitions.
    """

    from open_score.stage1 import CompetitiveEpisodeRunner, HADStage1Factory

    if not scales or not defender_policies or not attacker_policies:
        raise ValueError("HAD collection needs scales and policies on both sides")
    if (seeds is None) == (seeds_by_scale is None):
        raise ValueError("provide exactly one of seeds or seeds_by_scale")
    if not 1 <= command_steps <= max_steps:
        raise ValueError("command_steps must lie inside max_steps")
    runner = CompetitiveEpisodeRunner(HADStage1Factory(max_steps=max_steps))
    records: List[Stage2Record] = []
    common_seeds = None if seeds is None else tuple(int(seed) for seed in seeds)
    for scale in scales:
        defender_count, attacker_count = map(int, scale)
        scale_name = f"{defender_count}v{attacker_count}"
        scale_seeds = (
            common_seeds
            if common_seeds is not None
            else tuple(int(seed) for seed in seeds_by_scale[scale_name])
        )
        for seed in scale_seeds:
            lineage_group_id = f"had:master-seed-{int(seed)}"
            root_id = f"{lineage_group_id}:{scale_name}"
            for defender_style in defender_policies:
                for attacker_style in attacker_policies:
                    defender = _had_controller(defender_style)
                    attacker = _had_controller(attacker_style)
                    episode = runner.run(
                        (defender_count, attacker_count), defender, attacker, int(seed)
                    )
                    candidate_id = (
                        f"{defender.name}@{policy_version}"
                        f"__vs__{attacker.name}@{policy_version}"
                    )
                    rollout_id = f"{root_id}:{candidate_id}"
                    records.append(
                        record_from_had_episode(
                            episode,
                            lineage_group_id=lineage_group_id,
                            root_id=root_id,
                            rollout_id=rollout_id,
                            seed=int(seed),
                            scenario_id=f"one_target_{defender_count}v{attacker_count}",
                            defender_policy_version=policy_version,
                            attacker_policy_version=policy_version,
                            capability_version=capability_version,
                            horizon_steps=max_steps,
                            command_steps=command_steps,
                        )
                    )
    return records

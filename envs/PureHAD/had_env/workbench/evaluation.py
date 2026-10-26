"""Explicit ID/OOD scenario splits and paired, multi-training-seed evaluation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .protocols import ScenarioSpec
from .session import SimulationSession


def scenario_signature(scenario):
    """Structural test conditions, excluding opening randomness and split labels."""
    return (scenario.protocol_id, scenario.red_count, scenario.blue_count, scenario.opponent,
            scenario.max_steps, scenario.command_interval, scenario.target_positions,
            scenario.task_mode, scenario.target_health, scenario.horizon_policy, scenario.spatial_dim, scenario.plane_altitude)


@dataclass(frozen=True)
class EvaluationPlan:
    train: tuple[ScenarioSpec, ...]
    validation: tuple[ScenarioSpec, ...]
    test: tuple[ScenarioSpec, ...]

    def __post_init__(self):
        observed_ids, observed_openings = set(), {}
        for split in ("train", "validation", "test"):
            values = tuple(getattr(self, split))
            object.__setattr__(self, split, values)
            for scenario in values:
                if scenario.split != split:
                    raise ValueError("Scenario assigned to a different split")
                if scenario.scenario_id in observed_ids:
                    raise ValueError("Scenario IDs must be unique across splits")
                observed_ids.add(scenario.scenario_id)
                opening = (scenario_signature(scenario), scenario.opening_seed, scenario.opponent_seed)
                if opening in observed_openings:
                    raise ValueError("An exact opening cannot be reused across evaluation families")
                observed_openings[opening] = split
        if not self.train or not self.test:
            raise ValueError("Declare training conditions and at least one held-out test scenario")
        train_signatures = {scenario_signature(s) for s in self.train}
        for scenario in (*self.validation, *self.test):
            is_seen = scenario_signature(scenario) in train_signatures
            if (scenario.distribution == "id") != is_seen:
                raise ValueError("ID must match a training condition; OOD must hold out a structural condition")

    def to_dict(self):
        return {split: [s.to_dict() for s in getattr(self, split)] for split in ("train", "validation", "test")}

    @classmethod
    def from_dict(cls, value):
        return cls(**{split: tuple(ScenarioSpec.from_dict(s) for s in value.get(split, []))
                      for split in ("train", "validation", "test")})


def summarize_evaluation(rows, *, reference="rule"):
    """Keep ID/OOD and training seeds separate; uncertainty is across seed means."""
    groups, seen = defaultdict(list), set()
    for row in rows:
        if row.get("task_mode") == "damage" or row.get("success_native") is None:
            raise ValueError("The survival win-rate evaluator cannot summarize damage-task returns")
        key = (row["method_id"], int(row["training_seed"]), row["scenario_id"])
        if key in seen:
            raise ValueError("Duplicate evaluation unit")
        seen.add(key)
        groups[(row["method_id"], row["split"], row["distribution"])].append(row)
    output = []
    for (method, split, distribution), selected in sorted(groups.items()):
        per_seed = defaultdict(list)
        for row in selected:
            per_seed[int(row["training_seed"])].append(row)
        expected = {r["scenario_id"] for r in selected}
        complete = all({r["scenario_id"] for r in entries} == expected for entries in per_seed.values())
        rates = [float(np.mean([r["success_native"] for r in entries])) for entries in per_seed.values()]
        interval = None
        if len(rates) >= 2 and complete:
            from scipy.stats import t
            radius = float(t.ppf(.975, len(rates)-1) * np.std(rates, ddof=1) / np.sqrt(len(rates)))
            interval = [max(0., float(np.mean(rates))-radius), min(1., float(np.mean(rates))+radius)]
        output.append(dict(method_id=method, split=split, distribution=distribution,
                           episodes=len(selected), wins=sum(bool(r["success_native"]) for r in selected),
                           seed_rates={str(seed): float(np.mean([r["success_native"] for r in entries]))
                                       for seed, entries in sorted(per_seed.items())},
                           training_seeds=len(rates), balanced_scenario_coverage=complete,
                           mean_seed_success_rate=float(np.mean(rates)), across_seed_t95=interval,
                           uncertainty_note="Across designated training-seed means; one seed has no across-seed interval."))
    lookup = {(r["training_seed"], r["scenario_id"]): r for r in rows if r["method_id"] == reference}
    paired = []
    for (method, split, distribution), selected in sorted(groups.items()):
        if method == reference:
            continue
        differences = defaultdict(list)
        for row in selected:
            baseline = lookup.get((row["training_seed"], row["scenario_id"]))
            if baseline is not None:
                if any(row.get(k) != baseline.get(k) for k in ("opening_seed", "opponent_seed", "split", "distribution")):
                    raise ValueError("Paired scenario identity describes different openings")
                differences[int(row["training_seed"])].append(int(row["success_native"])-int(baseline["success_native"]))
        paired.append(dict(method_id=method, reference=reference, split=split, distribution=distribution,
                           pairs=sum(map(len, differences.values())),
                           seed_differences={str(seed): float(np.mean(values)) for seed, values in differences.items()}))
    return dict(schema="had-workbench-evaluation-v1", groups=output, paired=paired, raw_episodes=len(rows))


def evaluate_policies(policies, scenarios, *, training_seeds=(0,), record_dir=None):
    """Evaluate supplied checkpoints; never train, select checkpoints, or change rewards.

    policies maps method names to a built-in policy name, or to {training_seed:
    policy_object}. The latter makes checkpoint ownership explicit for multiple
    training seeds. A single external object is accepted for one seed only.
    """
    scenarios, seeds = tuple(scenarios), tuple(map(int, training_seeds))
    if any(s.task_mode == "damage" for s in scenarios):
        raise ValueError("Damage tasks use cumulative damage/returns, not the registered survival win-rate evaluator")
    if not scenarios or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide scenarios and unique training seeds")
    if len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("Duplicate scenario IDs")
    if any(s.split == "train" for s in scenarios):
        raise ValueError("Evaluate on validation/test/diagnostic scenarios, not training openings")
    if not policies:
        raise ValueError("Provide at least one policy")
    for policy in policies.values():
        if isinstance(policy, dict) and not all(seed in policy for seed in seeds):
            raise ValueError("Missing policy checkpoint for a training seed")
        if len(seeds) > 1 and not isinstance(policy, (str, dict)):
            raise ValueError("Multiple training seeds require explicit per-seed policies")
    destinations = {}
    if record_dir is not None:
        for method in policies:
            for seed in seeds:
                for scenario in scenarios:
                    key = hashlib.sha256(json.dumps([method, seed, scenario.scenario_id]).encode()).hexdigest()[:24]
                    destination = Path(record_dir) / f"{key}.json.gz"
                    if destination.exists():
                        raise FileExistsError("Evaluation recording already exists; use a new output directory")
                    destinations[method, seed, scenario.scenario_id] = destination
    rows = []
    for method, selected in policies.items():
        for seed in seeds:
            policy = selected[seed] if isinstance(selected, dict) else selected
            for scenario in scenarios:
                with SimulationSession(scenario, policy, policy_seed=seed, record=record_dir is not None) as session:
                    episode = session.run()
                    elapsed = [d["decision_seconds"] for d in episode.decisions]
                    row = dict(method_id=method, training_seed=seed, scenario_id=scenario.scenario_id,
                               split=scenario.split, distribution=scenario.distribution,
                               opening_seed=scenario.opening_seed, opponent_seed=scenario.opponent_seed,
                               red_count=scenario.red_count, blue_count=scenario.blue_count,
                               success_native=episode.metadata["success_native"], outcome_red=episode.metadata["outcome_red"],
                               physical_steps=episode.max_step, command_events=len(episode.decisions),
                               termination_reason=episode.metadata["termination_reason"],
                               decision_p95_seconds=float(np.quantile(elapsed, .95)) if elapsed else None)
                    if record_dir is not None:
                        row["recording"] = str(session.save(destinations[method, seed, scenario.scenario_id]))
                    rows.append(row)
    return {**summarize_evaluation(rows), "episodes": rows}

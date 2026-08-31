"""Leakage-safe data contracts for Stage-2 supervised rollout modelling.

The contract deliberately separates a query (information available at command
time) from its realised rollout label.  Environment adapters may therefore
share the JSONL/CSV writer without importing a particular Stage-1 runner.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


VALID_OUTCOMES = ("defender_win", "breach", "timeout")
UNKNOWN_TOKEN = "<UNK>"

# Directly interpretable non-time targets learned by the Stage-2 multi-task
# evaluator.  Survivor/casualty pairs are deliberately both retained: they
# are algebraically redundant when the initial roster is known, but each is a
# common operational question and exposing both avoids forcing downstream
# users to reconstruct the quantity.
PHYSICAL_TARGET_FIELDS = (
    "payoff_red",
    "target_final_health_fraction",
    "target_min_health_fraction",
    "defender_survivors",
    "attacker_survivors",
    "defender_casualties",
    "attacker_casualties",
    "minimum_threat_distance",
    "cumulative_target_damage",
    "cumulative_defender_damage",
    "cumulative_attacker_damage",
    "red_action_cost",
    "blue_action_cost",
)

PHYSICAL_REDUNDANT_RELATIONS = {
    "defender_survivors": "defender_count - defender_casualties",
    "defender_casualties": "defender_count - defender_survivors",
    "attacker_survivors": "attacker_count - attacker_casualties",
    "attacker_casualties": "attacker_count - attacker_survivors",
}


def _finite_tuple(values: Sequence[float], name: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


@dataclass(frozen=True)
class Stage2Query:
    """Information legitimately available before a local rollout.

    ``root_id`` identifies one scale-specific physical state.  The stronger
    ``lineage_group_id`` groups roots that share a master seed, target layout,
    or another parent generator across scales.  Data partitions use the group
    ID; using only ``root_id`` can leak near-duplicate sibling roots.
    ``rollout_id`` is unique for an executed trajectory and must *not* be used
    as the train/calibration/test split key.
    """

    environment_id: str
    scenario_id: str
    lineage_group_id: str
    root_id: str
    rollout_id: str
    seed: int
    defender_count: int
    attacker_count: int
    defender_policy_id: str
    attacker_policy_id: str
    defender_policy_version: str
    attacker_policy_version: str
    capability_version: str
    horizon_steps: int
    command_steps: int
    canonical_state: Tuple[float, ...]
    behavior_context: Tuple[float, ...] = ()
    candidate_id: str = ""
    threat_id: str = ""
    continuation_id: str = "continuation-000"
    stage1_checkpoint_sha256: str = ""
    root_seed: Optional[int] = None
    continuation_seed: Optional[int] = None
    stochasticity_profile: str = "legacy-unspecified"

    def __post_init__(self) -> None:
        for name in (
            "environment_id",
            "scenario_id",
            "lineage_group_id",
            "root_id",
            "rollout_id",
            "defender_policy_id",
            "attacker_policy_id",
            "defender_policy_version",
            "attacker_policy_version",
            "capability_version",
            "continuation_id",
            "stochasticity_profile",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.defender_count < 1 or self.attacker_count < 1:
            raise ValueError("both local teams must contain at least one agent")
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if not 1 <= self.command_steps <= self.horizon_steps:
            raise ValueError("command_steps must lie inside the rollout horizon")
        object.__setattr__(
            self, "canonical_state", _finite_tuple(self.canonical_state, "canonical_state")
        )
        context = tuple(float(value) for value in self.behavior_context)
        if context and not np.all(np.isfinite(context)):
            raise ValueError("behavior_context must be finite")
        object.__setattr__(self, "behavior_context", context)
        if self.root_seed is None:
            object.__setattr__(self, "root_seed", int(self.seed))
        if self.continuation_seed is None:
            object.__setattr__(self, "continuation_seed", int(self.seed))
        if int(self.seed) != int(self.continuation_seed):
            raise ValueError("legacy seed must equal continuation_seed")
        red_candidate = self.defender_policy_key
        blue_threat = self.attacker_policy_key
        if not self.candidate_id:
            object.__setattr__(self, "candidate_id", red_candidate)
        if not self.threat_id:
            object.__setattr__(self, "threat_id", blue_threat)
        if self.candidate_id != red_candidate:
            raise ValueError(
                "candidate_id must identify only the controllable Red/defender policy"
            )
        if self.threat_id != blue_threat:
            raise ValueError("threat_id must identify only the Blue/attacker policy")
        if self.stage1_checkpoint_sha256 and (
            len(self.stage1_checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.stage1_checkpoint_sha256.lower())
        ):
            raise ValueError("stage1_checkpoint_sha256 must be an SHA-256 hex digest")

    @property
    def defender_policy_key(self) -> str:
        return f"{self.defender_policy_id}@{self.defender_policy_version}"

    @property
    def attacker_policy_key(self) -> str:
        return f"{self.attacker_policy_id}@{self.attacker_policy_version}"

    @property
    def policy_pair(self) -> Tuple[str, str]:
        return (self.defender_policy_key, self.attacker_policy_key)


@dataclass(frozen=True)
class Stage2Record:
    """One policy-conditioned rollout label attached to a command query."""

    query: Stage2Query
    outcome: str
    terminal_steps: int
    sample_weight: float = 1.0
    payoff_red: Optional[float] = None
    target_final_health_fraction: Optional[float] = None
    target_min_health_fraction: Optional[float] = None
    defender_survivors: Optional[int] = None
    attacker_survivors: Optional[int] = None
    defender_casualties: Optional[int] = None
    attacker_casualties: Optional[int] = None
    breach_steps: Optional[int] = None
    attackers_neutralized_steps: Optional[int] = None
    minimum_threat_distance: Optional[float] = None
    cumulative_target_damage: Optional[float] = None
    cumulative_defender_damage: Optional[float] = None
    cumulative_attacker_damage: Optional[float] = None
    red_action_cost: Optional[float] = None
    blue_action_cost: Optional[float] = None

    def __post_init__(self) -> None:
        if self.outcome not in VALID_OUTCOMES:
            raise ValueError(f"outcome must be one of {VALID_OUTCOMES}")
        if not 1 <= self.terminal_steps <= self.query.horizon_steps:
            raise ValueError("terminal_steps must lie inside the rollout horizon")
        if not np.isfinite(self.sample_weight) or self.sample_weight <= 0.0:
            raise ValueError("sample_weight must be finite and positive")
        finite_optional = (
            "payoff_red",
            "target_final_health_fraction",
            "target_min_health_fraction",
            "minimum_threat_distance",
            "cumulative_target_damage",
            "cumulative_defender_damage",
            "cumulative_attacker_damage",
            "red_action_cost",
            "blue_action_cost",
        )
        for name in finite_optional:
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        for name in (
            "target_final_health_fraction",
            "target_min_health_fraction",
            "minimum_threat_distance",
            "cumulative_target_damage",
            "cumulative_defender_damage",
            "cumulative_attacker_damage",
            "red_action_cost",
            "blue_action_cost",
        ):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, roster in (
            ("defender_survivors", self.query.defender_count),
            ("attacker_survivors", self.query.attacker_count),
            ("defender_casualties", self.query.defender_count),
            ("attacker_casualties", self.query.attacker_count),
        ):
            value = getattr(self, name)
            if value is not None and not 0 <= int(value) <= roster:
                raise ValueError(f"{name} lies outside the registered roster")
        if self.breach_steps is not None and self.outcome != "breach":
            raise ValueError("breach_steps is only valid for a breach event")
        if self.attackers_neutralized_steps is not None and self.outcome != "defender_win":
            raise ValueError(
                "attackers_neutralized_steps is only valid for a defender-win event"
            )
        # ``timeout`` is an observation mechanism, not a terminal event.  It is
        # represented as right-censoring at ``terminal_steps`` and may happen
        # before the nominal horizon when a rollout budget is interrupted.

    @property
    def short_window_breach(self) -> bool:
        return self.outcome == "breach" and self.terminal_steps <= self.query.command_steps

    @property
    def event_observed(self) -> bool:
        return self.outcome != "timeout"

    @property
    def censor_steps(self) -> Optional[int]:
        return None if self.event_observed else self.terminal_steps

    def terminal_class(self, horizon_bins: int, steps_per_bin: int) -> int:
        """Encode the coherent ``2*K+1`` outcome-time label."""

        if horizon_bins < 1 or steps_per_bin < 1:
            raise ValueError("horizon_bins and steps_per_bin must be positive")
        if self.outcome == "timeout":
            return 2 * horizon_bins
        time_bin = min((self.terminal_steps - 1) // steps_per_bin, horizon_bins - 1)
        return int(time_bin + (horizon_bins if self.outcome == "breach" else 0))

    def to_dict(self) -> Dict[str, object]:
        value: Dict[str, object] = asdict(self.query)
        value.update(
            {
                "outcome": self.outcome,
                "terminal_steps": self.terminal_steps,
                "short_window_breach": self.short_window_breach,
                "event_observed": self.event_observed,
                "censor_steps": self.censor_steps,
                "sample_weight": self.sample_weight,
                "schema_version": 4,
            }
        )
        for name in (
            "payoff_red",
            "target_final_health_fraction",
            "target_min_health_fraction",
            "defender_survivors",
            "attacker_survivors",
            "defender_casualties",
            "attacker_casualties",
            "breach_steps",
            "attackers_neutralized_steps",
            "minimum_threat_distance",
            "cumulative_target_damage",
            "cumulative_defender_damage",
            "cumulative_attacker_damage",
            "red_action_cost",
            "blue_action_cost",
        ):
            value[name] = getattr(self, name)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Stage2Record":
        data = dict(value)
        schema_version = int(data.pop("schema_version", 1))
        declared_short = data.pop("short_window_breach", None)
        declared_event = data.pop("event_observed", None)
        declared_censor = data.pop("censor_steps", None)
        # Version 1/2 encoded a Red-vs-Blue pair as the candidate.  Migrate it
        # on read so old pilot evidence remains inspectable under the v3
        # decision contract, where only Red is controllable.
        if schema_version < 3:
            defender_key = (
                f"{data['defender_policy_id']}@{data['defender_policy_version']}"
            )
            attacker_key = (
                f"{data['attacker_policy_id']}@{data['attacker_policy_version']}"
            )
            data["candidate_id"] = defender_key
            data.setdefault("threat_id", attacker_key)
            data.setdefault("continuation_id", "legacy-continuation-000")
        query_fields = {
            name: data.pop(name)
            for name in Stage2Query.__dataclass_fields__
            if name in data
        }
        query_fields["canonical_state"] = tuple(query_fields["canonical_state"])
        query_fields["behavior_context"] = tuple(query_fields.get("behavior_context", ()))
        record = cls(query=Stage2Query(**query_fields), **data)
        if declared_short is not None and bool(declared_short) != record.short_window_breach:
            raise ValueError("short_window_breach disagrees with outcome/time fields")
        if declared_event is not None and bool(declared_event) != record.event_observed:
            raise ValueError("event_observed disagrees with outcome")
        if declared_censor is not None and declared_censor != record.censor_steps:
            raise ValueError("censor_steps disagrees with outcome/time fields")
        return record


def record_from_rollout(
    *,
    canonical_state: Sequence[float],
    outcome: str,
    terminal_steps: int,
    environment_id: str,
    scenario_id: str,
    lineage_group_id: str,
    root_id: str,
    rollout_id: str,
    seed: int,
    defender_count: int,
    attacker_count: int,
    defender_policy_id: str,
    attacker_policy_id: str,
    defender_policy_version: str,
    attacker_policy_version: str,
    capability_version: str,
    horizon_steps: int,
    command_steps: int,
    behavior_context: Sequence[float] = (),
    candidate_id: str = "",
    threat_id: str = "",
    continuation_id: str = "continuation-000",
    stage1_checkpoint_sha256: str = "",
    root_seed: Optional[int] = None,
    continuation_seed: Optional[int] = None,
    stochasticity_profile: str = "legacy-unspecified",
    sample_weight: float = 1.0,
    payoff_red: Optional[float] = None,
    target_final_health_fraction: Optional[float] = None,
    target_min_health_fraction: Optional[float] = None,
    defender_survivors: Optional[int] = None,
    attacker_survivors: Optional[int] = None,
    defender_casualties: Optional[int] = None,
    attacker_casualties: Optional[int] = None,
    breach_steps: Optional[int] = None,
    attackers_neutralized_steps: Optional[int] = None,
    minimum_threat_distance: Optional[float] = None,
    cumulative_target_damage: Optional[float] = None,
    cumulative_defender_damage: Optional[float] = None,
    cumulative_attacker_damage: Optional[float] = None,
    red_action_cost: Optional[float] = None,
    blue_action_cost: Optional[float] = None,
) -> Stage2Record:
    """Generic hook callable by HAD, SMAClite-AD, or another rollout runner."""

    query = Stage2Query(
        environment_id=environment_id,
        scenario_id=scenario_id,
        lineage_group_id=lineage_group_id,
        root_id=root_id,
        rollout_id=rollout_id,
        seed=seed,
        defender_count=defender_count,
        attacker_count=attacker_count,
        defender_policy_id=defender_policy_id,
        attacker_policy_id=attacker_policy_id,
        defender_policy_version=defender_policy_version,
        attacker_policy_version=attacker_policy_version,
        capability_version=capability_version,
        horizon_steps=horizon_steps,
        command_steps=command_steps,
        canonical_state=tuple(canonical_state),
        behavior_context=tuple(behavior_context),
        candidate_id=candidate_id,
        threat_id=threat_id,
        continuation_id=continuation_id,
        stage1_checkpoint_sha256=stage1_checkpoint_sha256,
        root_seed=root_seed,
        continuation_seed=continuation_seed,
        stochasticity_profile=stochasticity_profile,
    )
    return Stage2Record(
        query,
        outcome,
        terminal_steps,
        sample_weight,
        payoff_red,
        target_final_health_fraction,
        target_min_health_fraction,
        defender_survivors,
        attacker_survivors,
        defender_casualties,
        attacker_casualties,
        breach_steps,
        attackers_neutralized_steps,
        minimum_threat_distance,
        cumulative_target_damage,
        cumulative_defender_damage,
        cumulative_attacker_damage,
        red_action_cost,
        blue_action_cost,
    )


def write_records_jsonl(records: Iterable[Stage2Record], path: Union[str, Path]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def read_records_jsonl(
    path: Union[str, Path],
    *,
    required_schema_version: Optional[int] = None,
) -> List[Stage2Record]:
    """Read Stage-2 JSONL and optionally reject every non-exact schema row.

    Legacy schemas remain readable for historical pilot inspection.  Formal
    training passes ``required_schema_version=4`` so that backward-compatible
    migration cannot silently turn an old dataset into current evidence.
    """

    records: List[Stage2Record] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid Stage-2 JSONL record at line {line_number}") from error
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"invalid Stage-2 JSONL record at line {line_number}"
                    )
                if required_schema_version is not None:
                    try:
                        schema_version = int(payload.get("schema_version", 1))
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"invalid Stage-2 schema_version at line {line_number}"
                        ) from error
                    if schema_version != int(required_schema_version):
                        raise ValueError(
                            "formal Stage-2 JSONL requires schema_version="
                            f"{required_schema_version}; line {line_number} has "
                            f"schema_version={schema_version}"
                        )
                try:
                    records.append(Stage2Record.from_dict(payload))
                except (TypeError, ValueError, KeyError) as error:
                    raise ValueError(
                        f"invalid Stage-2 JSONL record at line {line_number}"
                    ) from error
    if not records:
        raise ValueError("Stage-2 dataset is empty")
    return records


def write_records_csv(records: Iterable[Stage2Record], path: Union[str, Path]) -> None:
    """Write a human-auditable CSV; vectors are JSON arrays inside their cells."""

    rows = [record.to_dict() for record in records]
    if not rows:
        raise ValueError("cannot write an empty Stage-2 dataset")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vector_fields = ("canonical_state", "behavior_context")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for field in vector_fields:
                encoded[field] = json.dumps(encoded[field], separators=(",", ":"))
            writer.writerow(encoded)


def validate_rollout_lineage(records: Sequence[Stage2Record], atol: float = 1e-6) -> None:
    """Reject duplicate rollouts and inconsistent root-to-parent lineage."""

    if not records:
        raise ValueError("cannot audit an empty Stage-2 dataset")
    rollout_ids = [record.query.rollout_id for record in records]
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("rollout_id must be globally unique")
    reference: Dict[str, Stage2Query] = {}
    root_to_group: Dict[str, str] = {}
    static_fields = (
        "environment_id",
        "scenario_id",
        "defender_count",
        "attacker_count",
        "horizon_steps",
        "command_steps",
    )
    for record in records:
        query = record.query
        previous_group = root_to_group.setdefault(query.root_id, query.lineage_group_id)
        if previous_group != query.lineage_group_id:
            raise ValueError("one root_id maps to multiple lineage_group_id values")
        if query.root_id not in reference:
            reference[query.root_id] = query
            continue
        first = reference[query.root_id]
        if any(getattr(first, name) != getattr(query, name) for name in static_fields):
            raise ValueError("one root_id contains inconsistent environment/scenario/scale metadata")
        if (
            first.stochasticity_profile != "legacy-unspecified"
            and query.stochasticity_profile != "legacy-unspecified"
            and int(first.root_seed) != int(query.root_seed)
        ):
            raise ValueError("one root_id contains inconsistent root_seed values")
        if len(first.canonical_state) != len(query.canonical_state) or not np.allclose(
            first.canonical_state, query.canonical_state, rtol=0.0, atol=atol
        ):
            raise ValueError("one root_id contains inconsistent canonical physical states")


def validate_counterfactual_design(
    records: Sequence[Stage2Record],
    *,
    min_continuations_per_cell: int = 2,
    require_common_random_numbers: bool = True,
) -> Dict[str, object]:
    """Audit Red-candidate comparisons under fixed Blue threats.

    A cell is ``root x Red candidate x Blue threat``.  Each cell must contain
    multiple independent continuation IDs.  With common random numbers (CRN),
    all Red candidates evaluated for the same root/threat must use the same
    continuation IDs and the same continuation seed for each ID.  This reduces
    comparison variance without pretending the resulting rows are independent.
    """

    validate_rollout_lineage(records)
    if min_continuations_per_cell < 1:
        raise ValueError("min_continuations_per_cell must be positive")
    cells: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    root_threat_candidates: Dict[Tuple[str, str], Dict[str, Dict[str, int]]] = {}
    for record in records:
        query = record.query
        key = (query.root_id, query.candidate_id, query.threat_id)
        continuations = cells.setdefault(key, {})
        if query.continuation_id in continuations:
            raise ValueError("duplicate continuation_id inside a root/candidate/threat cell")
        continuations[query.continuation_id] = int(query.continuation_seed)
        root_threat_candidates.setdefault(
            (query.root_id, query.threat_id), {}
        )[query.candidate_id] = continuations
    undersampled = [key for key, values in cells.items() if len(values) < min_continuations_per_cell]
    if undersampled:
        raise ValueError(
            f"{len(undersampled)} candidate cells have fewer than "
            f"{min_continuations_per_cell} continuations"
        )
    reused_seeds = [
        key for key, values in cells.items() if len(set(values.values())) != len(values)
    ]
    if reused_seeds:
        raise ValueError(
            f"{len(reused_seeds)} candidate cells reuse a continuation seed under "
            "different continuation_id values"
        )
    if require_common_random_numbers:
        for (root_id, threat_id), candidates in root_threat_candidates.items():
            references = list(candidates.items())
            first_candidate, first = references[0]
            for candidate, continuations in references[1:]:
                if continuations != first:
                    raise ValueError(
                        "common-random-number mismatch for "
                        f"root={root_id}, threat={threat_id}, "
                        f"candidates={first_candidate}/{candidate}"
                    )
    root_candidates: Dict[str, set] = {}
    root_threats: Dict[str, set] = {}
    for root_id, candidate_id, threat_id in cells:
        root_candidates.setdefault(root_id, set()).add(candidate_id)
        root_threats.setdefault(root_id, set()).add(threat_id)
    candidate_sets = {tuple(sorted(values)) for values in root_candidates.values()}
    threat_sets = {tuple(sorted(values)) for values in root_threats.values()}
    if len(candidate_sets) != 1:
        raise ValueError("every root must expose the same Red candidate set")
    if len(threat_sets) != 1:
        raise ValueError("every root must expose the same Blue threat set")
    continuation_counts = [len(values) for values in cells.values()]
    return {
        "status": "passed",
        "controllable_side": "Red",
        "blue_role": "fixed_or_sampled_threat",
        "candidate_cells": len(cells),
        "root_threat_blocks": len(root_threat_candidates),
        "minimum_continuations_per_cell": int(min(continuation_counts)),
        "mean_continuations_per_cell": float(np.mean(continuation_counts)),
        "common_random_numbers": bool(require_common_random_numbers),
        "red_candidates": list(next(iter(candidate_sets))),
        "blue_threats": list(next(iter(threat_sets))),
        "complete_candidate_threat_grid": True,
    }


def validate_formal_dataset_contract(
    records: Sequence[Stage2Record],
    contract: Mapping[str, object],
) -> Dict[str, object]:
    """Enforce the preregistered semantic contract of a frozen formal dataset.

    Dataset SHA-256 protects bytes.  This validator independently protects the
    scientific meaning of those bytes: registered lineages/scales, the full
    Red-candidate/Blue-threat grid, exact CRN replication, complete physical
    supervision, and one locked learned-policy checkpoint lineage.
    """

    if not records:
        raise ValueError("formal Stage-2 dataset is empty")
    values = dict(contract)

    def required_int(name: str) -> int:
        if name not in values:
            raise ValueError(f"formal_data_contract is missing {name}")
        result = int(values[name])
        if result < 1:
            raise ValueError(f"formal_data_contract.{name} must be positive")
        return result

    required_schema_version = required_int("required_schema_version")
    if required_schema_version != 4:
        raise ValueError("formal Stage-2 evidence requires schema_version=4")
    required_lineages = required_int("required_lineage_groups")
    required_red_candidates = required_int("required_red_candidate_count")
    required_blue_threats = required_int("required_blue_threat_count")
    required_continuations = required_int("required_continuations_per_cell")
    required_checkpoint_count = required_int(
        "required_nonempty_checkpoint_sha256_count"
    )
    required_checkpoint_candidates = required_int(
        "required_checkpoint_candidate_count"
    )

    validate_rollout_lineage(records)
    lineage_groups = {record.query.lineage_group_id for record in records}
    if len(lineage_groups) != required_lineages:
        raise ValueError(
            "formal Stage-2 lineage count differs from the preregistered contract"
        )

    scales = {
        (record.query.defender_count, record.query.attacker_count)
        for record in records
    }
    required_scales = {
        tuple(map(int, scale)) for scale in values.get("required_scales", ())
    }
    if not required_scales:
        raise ValueError("formal_data_contract.required_scales must be non-empty")
    if scales != required_scales:
        raise ValueError("formal Stage-2 scales differ from the preregistered contract")

    red_candidates = {record.query.candidate_id for record in records}
    blue_threats = {record.query.threat_id for record in records}
    if len(red_candidates) != required_red_candidates:
        raise ValueError("formal Stage-2 Red-candidate count differs from contract")
    if len(blue_threats) != required_blue_threats:
        raise ValueError("formal Stage-2 Blue-threat count differs from contract")

    cells: Dict[Tuple[str, str, str], set] = {}
    for record in records:
        query = record.query
        cells.setdefault(
            (query.root_id, query.candidate_id, query.threat_id), set()
        ).add(query.continuation_id)
    wrong_continuations = {
        key: len(continuations)
        for key, continuations in cells.items()
        if len(continuations) != required_continuations
    }
    if wrong_continuations:
        raise ValueError(
            "formal Stage-2 candidate cells do not have the exact registered "
            "continuation count"
        )
    design_audit = validate_counterfactual_design(
        records,
        min_continuations_per_cell=required_continuations,
        require_common_random_numbers=True,
    )

    if "required_physical_target_fields" not in values:
        raise ValueError(
            "formal_data_contract is missing required_physical_target_fields"
        )
    required_physical_fields = tuple(
        str(field) for field in values["required_physical_target_fields"]
    )
    if (
        len(required_physical_fields) != len(PHYSICAL_TARGET_FIELDS)
        or len(set(required_physical_fields)) != len(required_physical_fields)
        or set(required_physical_fields) != set(PHYSICAL_TARGET_FIELDS)
    ):
        raise ValueError(
            "formal_data_contract.required_physical_target_fields must list "
            "the 13 canonical Stage-2 physical labels exactly once"
        )
    require_physical = values.get("require_complete_physical_targets") is True
    if not require_physical:
        raise ValueError(
            "formal_data_contract.require_complete_physical_targets must be true"
        )
    missing_physical: Dict[str, int] = {}
    if require_physical:
        missing_physical = {
            field: sum(getattr(record, field) is None for record in records)
            for field in required_physical_fields
        }
        missing_physical = {
            field: count for field, count in missing_physical.items() if count
        }
        if missing_physical:
            raise ValueError(
                "formal Stage-2 dataset has missing 13-head physical labels: "
                + ", ".join(
                    f"{field}={count}" for field, count in missing_physical.items()
                )
            )

    if values.get("require_strict_red_superiority") is not True:
        raise ValueError(
            "formal_data_contract.require_strict_red_superiority must be true"
        )
    if any(
        record.query.defender_count <= record.query.attacker_count
        for record in records
    ):
        raise ValueError("formal Stage-2 dataset violates strict Red>Blue")

    candidate_hashes: Dict[str, set] = {}
    for record in records:
        candidate_hashes.setdefault(record.query.candidate_id, set()).add(
            record.query.stage1_checkpoint_sha256.lower()
        )
    mixed_checkpoint_candidates = [
        candidate
        for candidate, hashes in candidate_hashes.items()
        if "" in hashes and any(hashes)
    ]
    if mixed_checkpoint_candidates:
        raise ValueError(
            "one formal Red candidate mixes checkpoint-tagged and untagged rows"
        )
    checkpoint_hashes = {
        digest
        for hashes in candidate_hashes.values()
        for digest in hashes
        if digest
    }
    checkpoint_candidates = sorted(
        candidate for candidate, hashes in candidate_hashes.items() if any(hashes)
    )
    if len(checkpoint_candidates) != required_checkpoint_candidates:
        raise ValueError(
            "formal Stage-2 dataset checkpoint-tagged Red-candidate count "
            "differs from contract"
        )
    if len(checkpoint_hashes) != required_checkpoint_count:
        raise ValueError(
            "formal Stage-2 dataset does not contain exactly the registered "
            "number of learned checkpoint SHA-256 values"
        )
    expected_checkpoint = str(
        values.get("expected_stage1_checkpoint_sha256", "")
    ).lower()
    if len(expected_checkpoint) != 64 or any(
        character not in "0123456789abcdef" for character in expected_checkpoint
    ):
        raise ValueError(
            "formal_data_contract.expected_stage1_checkpoint_sha256 must "
            "be a 64-character hexadecimal digest"
        )
    if checkpoint_hashes != {expected_checkpoint}:
        raise ValueError(
            "formal Stage-2 dataset checkpoint SHA-256 differs from policy lock"
        )

    snapshot_zero_roots = {
        record.query.root_id
        for record in records
        if ":snapshot-000:" in record.query.root_id
    }
    required_snapshot_zero = values.get("required_snapshot_zero_roots")
    if required_snapshot_zero is not None and len(snapshot_zero_roots) != int(
        required_snapshot_zero
    ):
        raise ValueError(
            "formal Stage-2 snapshot-000 root count differs from contract"
        )
    if bool(values.get("require_snapshot_zero_for_every_lineage_scale", False)):
        snapshot_zero_pairs = {
            (
                record.query.lineage_group_id,
                (record.query.defender_count, record.query.attacker_count),
            )
            for record in records
            if ":snapshot-000:" in record.query.root_id
        }
        expected_pairs = {
            (lineage, scale)
            for lineage in lineage_groups
            for scale in required_scales
        }
        if snapshot_zero_pairs != expected_pairs:
            raise ValueError(
                "formal Stage-2 dataset lacks snapshot-000 for a lineage/scale pair"
            )

    roots = {record.query.root_id for record in records}
    return {
        "status": "passed",
        "required_schema_version": required_schema_version,
        "records": len(records),
        "roots": len(roots),
        "lineage_groups": len(lineage_groups),
        "scales": [list(scale) for scale in sorted(scales)],
        "red_candidates": sorted(red_candidates),
        "blue_threats": sorted(blue_threats),
        "candidate_cells": len(cells),
        "continuations_per_cell": required_continuations,
        "physical_target_fields": list(required_physical_fields),
        "physical_labels_complete": not missing_physical,
        "strict_red_superiority": True,
        "stage1_checkpoint_sha256": sorted(checkpoint_hashes),
        "checkpoint_red_candidates": checkpoint_candidates,
        "snapshot_zero_roots": len(snapshot_zero_roots),
        "counterfactual_design_audit": design_audit,
    }


@dataclass(frozen=True)
class Stage2Split:
    train: Tuple[Stage2Record, ...]
    validation: Tuple[Stage2Record, ...]
    temperature_calibration: Tuple[Stage2Record, ...]
    risk_calibration: Tuple[Stage2Record, ...]
    test: Tuple[Stage2Record, ...]

    @property
    def partitions(self) -> Tuple[Tuple[str, Tuple[Stage2Record, ...]], ...]:
        return (
            ("train", self.train),
            ("validation", self.validation),
            ("temperature_calibration", self.temperature_calibration),
            ("risk_calibration", self.risk_calibration),
            ("test", self.test),
        )

    def assert_no_lineage_leakage(self) -> None:
        validate_rollout_lineage(tuple(record for _, rows in self.partitions for record in rows))
        roots = {
            name: {record.query.root_id for record in records}
            for name, records in self.partitions
        }
        groups = {
            name: {record.query.lineage_group_id for record in records}
            for name, records in self.partitions
        }
        names = list(roots)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if roots[left] & roots[right]:
                    raise ValueError(f"root leakage between {left} and {right}")
                if groups[left] & groups[right]:
                    raise ValueError(f"lineage-group leakage between {left} and {right}")

    def assert_no_root_leakage(self) -> None:
        """Backward-compatible alias that now enforces the stronger group audit."""

        self.assert_no_lineage_leakage()

    def manifest(self) -> Dict[str, object]:
        self.assert_no_lineage_leakage()
        result: Dict[str, object] = {
            "split_key": "lineage_group_id",
            "lineage_consistency_check": "passed",
            "root_leakage_check": "passed",
            "lineage_group_leakage_check": "passed",
            "leakage_check": "passed",
        }
        for name, records in self.partitions:
            root_ids = sorted({record.query.root_id for record in records})
            lineage_groups = sorted(
                {record.query.lineage_group_id for record in records}
            )
            result[name] = {
                "records": len(records),
                "roots": len(root_ids),
                "lineage_groups": len(lineage_groups),
                "rollouts": len({record.query.rollout_id for record in records}),
                "root_ids": root_ids,
                "lineage_group_ids": lineage_groups,
            }
        return result


def split_records_by_lineage_group(
    records: Sequence[Stage2Record],
    train_fraction: float = 0.50,
    validation_fraction: float = 0.125,
    temperature_calibration_fraction: float = 0.125,
    risk_calibration_fraction: float = 0.125,
    seed: int = 0,
    stratify_by_scenario: bool = False,
) -> Stage2Split:
    """Five-way split entire parent lineages, including cross-scale siblings.

    Validation selects model checkpoints/hyperparameters.  Temperature and
    risk calibration remain independent from both selection and final test.
    """

    fractions = (
        train_fraction,
        validation_fraction,
        temperature_calibration_fraction,
        risk_calibration_fraction,
    )
    if any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("train, validation and both calibration fractions must be positive")
    if sum(fractions) >= 1.0:
        raise ValueError("test fraction must be positive")
    by_group: Dict[str, List[Stage2Record]] = {}
    validate_rollout_lineage(records)
    for record in records:
        by_group.setdefault(record.query.lineage_group_id, []).append(record)
    if len(by_group) < 5:
        raise ValueError("at least five independent lineage groups are required")
    rng = np.random.default_rng(seed)

    def partition(group_values: Sequence[str]) -> Tuple[set, set, set, set, set]:
        groups = np.asarray(sorted(group_values), dtype=object)
        rng.shuffle(groups)
        count = len(groups)
        if count < 5:
            raise ValueError("every split stratum needs at least five lineage groups")
        n_train = max(1, min(count - 4, int(round(count * train_fraction))))
        n_validation = max(
            1,
            min(count - n_train - 3, int(round(count * validation_fraction))),
        )
        n_temperature = max(
            1,
            min(
                count - n_train - n_validation - 2,
                int(round(count * temperature_calibration_fraction)),
            ),
        )
        n_risk = max(
            1,
            min(
                count - n_train - n_validation - n_temperature - 1,
                int(round(count * risk_calibration_fraction)),
            ),
        )
        return (
            set(groups[:n_train]),
            set(groups[n_train : n_train + n_validation]),
            set(groups[n_train + n_validation : n_train + n_validation + n_temperature]),
            set(
                groups[
                    n_train + n_validation + n_temperature :
                    n_train + n_validation + n_temperature + n_risk
                ]
            ),
            set(groups[n_train + n_validation + n_temperature + n_risk :]),
        )

    if stratify_by_scenario:
        strata: Dict[str, List[str]] = {}
        for group_id, group_records in by_group.items():
            # A parent may deliberately contain one sibling root per scale.
            signature = "|".join(
                sorted({record.query.scenario_id for record in group_records})
            )
            strata.setdefault(signature, []).append(group_id)
        group_sets = (set(), set(), set(), set(), set())
        for signature in sorted(strata):
            for destination, selected in zip(group_sets, partition(strata[signature])):
                destination.update(selected)
    else:
        group_sets = partition(list(by_group))

    def select(selected: set) -> Tuple[Stage2Record, ...]:
        return tuple(
            record for record in records if record.query.lineage_group_id in selected
        )

    split = Stage2Split(*(select(selected) for selected in group_sets))
    split.assert_no_lineage_leakage()
    return split


def split_records_by_root(*args, **kwargs) -> Stage2Split:
    """Compatibility alias; splitting is now always by ``lineage_group_id``."""

    return split_records_by_lineage_group(*args, **kwargs)


@dataclass(frozen=True)
class Stage2FeatureEncoder:
    """Deterministic, serialisable policy/version one-hot feature encoder."""

    state_dim: int
    context_dim: int
    defender_policies: Tuple[str, ...]
    attacker_policies: Tuple[str, ...]
    capability_versions: Tuple[str, ...]
    include_policy_context: bool = True

    @staticmethod
    def _policy_key(policy_id: str, policy_version: str) -> str:
        return f"{policy_id}@{policy_version}"

    @classmethod
    def fit(
        cls,
        records: Sequence[Stage2Record],
        include_policy_context: bool = True,
    ) -> "Stage2FeatureEncoder":
        if not records:
            raise ValueError("cannot fit features on an empty dataset")
        state_dims = {len(record.query.canonical_state) for record in records}
        context_dims = {len(record.query.behavior_context) for record in records}
        if len(state_dims) != 1 or len(context_dims) != 1:
            raise ValueError("all canonical states and contexts must have fixed dimensions")

        def vocabulary(values: Iterable[str]) -> Tuple[str, ...]:
            return (UNKNOWN_TOKEN,) + tuple(sorted(set(values) - {UNKNOWN_TOKEN}))

        return cls(
            state_dim=state_dims.pop(),
            context_dim=context_dims.pop(),
            defender_policies=vocabulary(
                cls._policy_key(
                    record.query.defender_policy_id, record.query.defender_policy_version
                )
                for record in records
            ),
            attacker_policies=vocabulary(
                cls._policy_key(
                    record.query.attacker_policy_id, record.query.attacker_policy_version
                )
                for record in records
            ),
            capability_versions=vocabulary(
                record.query.capability_version for record in records
            ),
            include_policy_context=include_policy_context,
        )

    @property
    def output_dim(self) -> int:
        policy_dim = 0
        if self.include_policy_context:
            policy_dim = len(self.defender_policies) + len(self.attacker_policies)
        return self.state_dim + self.context_dim + policy_dim + len(self.capability_versions) + 6

    @staticmethod
    def _one_hot(value: str, vocabulary: Tuple[str, ...]) -> np.ndarray:
        result = np.zeros(len(vocabulary), dtype=np.float32)
        try:
            index = vocabulary.index(value)
        except ValueError:
            index = 0
        result[index] = 1.0
        return result

    def transform_queries(self, queries: Sequence[Stage2Query]) -> np.ndarray:
        rows: List[np.ndarray] = []
        for query in queries:
            if len(query.canonical_state) != self.state_dim:
                raise ValueError("canonical state dimension differs from fitted encoder")
            if len(query.behavior_context) != self.context_dim:
                raise ValueError("behavior context dimension differs from fitted encoder")
            horizon = float(query.horizon_steps)
            numeric = np.asarray(
                [
                    query.defender_count / 4.0,
                    query.attacker_count / 4.0,
                    np.log1p(query.defender_count),
                    np.log1p(query.attacker_count),
                    query.command_steps / horizon,
                    np.log1p(horizon) / 10.0,
                ],
                dtype=np.float32,
            )
            parts = [np.asarray(query.canonical_state, dtype=np.float32), numeric]
            if self.include_policy_context:
                parts.extend(
                    [
                        self._one_hot(
                            self._policy_key(
                                query.defender_policy_id, query.defender_policy_version
                            ),
                            self.defender_policies,
                        ),
                        self._one_hot(
                            self._policy_key(
                                query.attacker_policy_id, query.attacker_policy_version
                            ),
                            self.attacker_policies,
                        ),
                    ]
                )
            parts.append(self._one_hot(query.capability_version, self.capability_versions))
            if self.context_dim:
                parts.append(np.asarray(query.behavior_context, dtype=np.float32))
            rows.append(np.concatenate(parts))
        if not rows:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.stack(rows).astype(np.float32, copy=False)

    def transform(self, records: Sequence[Stage2Record]) -> np.ndarray:
        return self.transform_queries([record.query for record in records])

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Stage2FeatureEncoder":
        data = dict(value)
        for name in ("defender_policies", "attacker_policies", "capability_versions"):
            data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureNormalizer":
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("features must be a non-empty 2-D array")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(tuple(float(value) for value in mean), tuple(float(value) for value in scale))

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(mean):
            raise ValueError("feature dimension differs from fitted normalizer")
        return ((values - mean) / scale).astype(np.float32, copy=False)

    def with_passthrough(self, indices: Sequence[int]) -> "FeatureNormalizer":
        """Return a copy that leaves structural mask columns unchanged."""

        mean = np.asarray(self.mean, dtype=np.float32).copy()
        scale = np.asarray(self.scale, dtype=np.float32).copy()
        for index in indices:
            if not 0 <= int(index) < len(mean):
                raise ValueError("normalizer passthrough index is outside feature range")
            mean[int(index)] = 0.0
            scale[int(index)] = 1.0
        return FeatureNormalizer(tuple(map(float, mean)), tuple(map(float, scale)))

    def to_dict(self) -> Dict[str, object]:
        return {"mean": list(self.mean), "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FeatureNormalizer":
        return cls(tuple(value["mean"]), tuple(value["scale"]))

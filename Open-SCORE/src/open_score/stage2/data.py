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
        if not self.candidate_id:
            object.__setattr__(
                self,
                "candidate_id",
                (
                    f"{self.defender_policy_id}@{self.defender_policy_version}"
                    f"__vs__{self.attacker_policy_id}@{self.attacker_policy_version}"
                ),
            )

    @property
    def policy_pair(self) -> Tuple[str, str]:
        return (
            f"{self.defender_policy_id}@{self.defender_policy_version}",
            f"{self.attacker_policy_id}@{self.attacker_policy_version}",
        )


@dataclass(frozen=True)
class Stage2Record:
    """One policy-conditioned rollout label attached to a command query."""

    query: Stage2Query
    outcome: str
    terminal_steps: int
    sample_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.outcome not in VALID_OUTCOMES:
            raise ValueError(f"outcome must be one of {VALID_OUTCOMES}")
        if not 1 <= self.terminal_steps <= self.query.horizon_steps:
            raise ValueError("terminal_steps must lie inside the rollout horizon")
        if not np.isfinite(self.sample_weight) or self.sample_weight <= 0.0:
            raise ValueError("sample_weight must be finite and positive")
        if self.outcome == "timeout" and self.terminal_steps != self.query.horizon_steps:
            raise ValueError("timeout records must end exactly at horizon_steps")

    @property
    def short_window_breach(self) -> bool:
        return self.outcome == "breach" and self.terminal_steps <= self.query.command_steps

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
                "sample_weight": self.sample_weight,
                "schema_version": 2,
            }
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Stage2Record":
        data = dict(value)
        data.pop("schema_version", None)
        declared_short = data.pop("short_window_breach", None)
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
    sample_weight: float = 1.0,
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
    )
    return Stage2Record(query, outcome, terminal_steps, sample_weight)


def write_records_jsonl(records: Iterable[Stage2Record], path: Union[str, Path]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def read_records_jsonl(path: Union[str, Path]) -> List[Stage2Record]:
    records: List[Stage2Record] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(Stage2Record.from_dict(json.loads(line)))
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid Stage-2 JSONL record at line {line_number}") from error
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
        if len(first.canonical_state) != len(query.canonical_state) or not np.allclose(
            first.canonical_state, query.canonical_state, rtol=0.0, atol=atol
        ):
            raise ValueError("one root_id contains inconsistent canonical physical states")


@dataclass(frozen=True)
class Stage2Split:
    train: Tuple[Stage2Record, ...]
    temperature_calibration: Tuple[Stage2Record, ...]
    risk_calibration: Tuple[Stage2Record, ...]
    test: Tuple[Stage2Record, ...]

    @property
    def partitions(self) -> Tuple[Tuple[str, Tuple[Stage2Record, ...]], ...]:
        return (
            ("train", self.train),
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
    temperature_calibration_fraction: float = 1.0 / 6.0,
    risk_calibration_fraction: float = 1.0 / 6.0,
    seed: int = 0,
    stratify_by_scenario: bool = False,
) -> Stage2Split:
    """Four-way split entire parent lineages, including cross-scale siblings."""

    fractions = (
        train_fraction,
        temperature_calibration_fraction,
        risk_calibration_fraction,
    )
    if any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("train and both calibration fractions must be positive")
    if sum(fractions) >= 1.0:
        raise ValueError("test fraction must be positive")
    by_group: Dict[str, List[Stage2Record]] = {}
    validate_rollout_lineage(records)
    for record in records:
        by_group.setdefault(record.query.lineage_group_id, []).append(record)
    if len(by_group) < 4:
        raise ValueError("at least four independent lineage groups are required")
    rng = np.random.default_rng(seed)

    def partition(group_values: Sequence[str]) -> Tuple[set, set, set, set]:
        groups = np.asarray(sorted(group_values), dtype=object)
        rng.shuffle(groups)
        count = len(groups)
        if count < 4:
            raise ValueError("every split stratum needs at least four lineage groups")
        n_train = max(1, min(count - 3, int(round(count * train_fraction))))
        n_temperature = max(
            1,
            min(
                count - n_train - 2,
                int(round(count * temperature_calibration_fraction)),
            ),
        )
        n_risk = max(
            1,
            min(
                count - n_train - n_temperature - 1,
                int(round(count * risk_calibration_fraction)),
            ),
        )
        return (
            set(groups[:n_train]),
            set(groups[n_train : n_train + n_temperature]),
            set(groups[n_train + n_temperature : n_train + n_temperature + n_risk]),
            set(groups[n_train + n_temperature + n_risk :]),
        )

    if stratify_by_scenario:
        strata: Dict[str, List[str]] = {}
        for group_id, group_records in by_group.items():
            # A parent may deliberately contain one sibling root per scale.
            signature = "|".join(
                sorted({record.query.scenario_id for record in group_records})
            )
            strata.setdefault(signature, []).append(group_id)
        group_sets = (set(), set(), set(), set())
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

    def to_dict(self) -> Dict[str, object]:
        return {"mean": list(self.mean), "scale": list(self.scale)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FeatureNormalizer":
        return cls(tuple(value["mean"]), tuple(value["scale"]))

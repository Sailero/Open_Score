"""End-to-end fitting, prediction, and checkpointing for Stage 2."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from .calibration import CalibratedRiskBound, TemperatureScaler
from .data import (
    FeatureNormalizer,
    PHYSICAL_TARGET_FIELDS,
    Stage2FeatureEncoder,
    Stage2Query,
    Stage2Record,
)
from .outcome_model import BootstrapOutcomeEnsemble, competing_risk_nll


@dataclass(frozen=True)
class Stage2TrainingConfig:
    horizon_bins: int = 8
    steps_per_bin: int = 10
    hidden_dim: int = 64
    model_kind: str = "mlp"
    ensemble_members: int = 5
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    calibration_alpha: float = 0.10
    require_conformal_guarantee: bool = False
    include_policy_context: bool = True
    policy_unknown_augmentation_probability: float = 0.15
    early_stopping_patience: int = 12
    minimum_epochs: int = 1
    physical_loss_weight: float = 0.50
    physical_interval_lower_quantile: float = 0.10
    physical_interval_upper_quantile: float = 0.90
    seed: int = 7

    def __post_init__(self) -> None:
        if self.horizon_bins < 1 or self.steps_per_bin < 1:
            raise ValueError("time discretisation must be positive")
        if self.hidden_dim < 1 or self.ensemble_members < 2:
            raise ValueError("hidden_dim must be positive and ensemble needs at least two members")
        if self.model_kind not in {"mlp", "had_deepset"}:
            raise ValueError("model_kind must be mlp or had_deepset")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("invalid Stage-2 optimisation configuration")
        if self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("weight decay/gradient clip are invalid")
        if not 0.0 < self.calibration_alpha < 1.0:
            raise ValueError("calibration_alpha must lie inside (0,1)")
        if not 0.0 <= self.policy_unknown_augmentation_probability <= 1.0:
            raise ValueError("policy_unknown_augmentation_probability must lie inside [0,1]")
        if self.early_stopping_patience < 1 or not 1 <= self.minimum_epochs <= self.epochs:
            raise ValueError("early-stopping settings are invalid")
        if self.physical_loss_weight < 0.0:
            raise ValueError("physical_loss_weight must be non-negative")
        if not (
            0.0 <= self.physical_interval_lower_quantile
            < self.physical_interval_upper_quantile <= 1.0
        ):
            raise ValueError("physical ensemble interval quantiles are invalid")

    @property
    def horizon_steps(self) -> int:
        return self.horizon_bins * self.steps_per_bin

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Stage2TrainingConfig":
        known = {name: value[name] for name in cls.__dataclass_fields__ if name in value}
        return cls(**known)


@dataclass(frozen=True)
class PhysicalTargetScaler:
    """Train-only z-score contract for the non-time Stage-2 supervision heads."""

    fields: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    available_count: tuple[int, ...]

    @classmethod
    def fit(cls, records: Sequence[Stage2Record]) -> "PhysicalTargetScaler":
        means, scales, counts = [], [], []
        for field in PHYSICAL_TARGET_FIELDS:
            values = np.asarray(
                [
                    float(getattr(record, field))
                    for record in records
                    if getattr(record, field) is not None
                ],
                dtype=np.float64,
            )
            counts.append(int(len(values)))
            means.append(float(values.mean()) if len(values) else 0.0)
            standard_deviation = float(values.std()) if len(values) else 1.0
            scales.append(standard_deviation if standard_deviation >= 1e-6 else 1.0)
        return cls(tuple(PHYSICAL_TARGET_FIELDS), tuple(means), tuple(scales), tuple(counts))

    @property
    def available_fields(self) -> tuple[str, ...]:
        return tuple(
            field for field, count in zip(self.fields, self.available_count) if count > 0
        )

    def transform(
        self, records: Sequence[Stage2Record]
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.zeros((len(records), len(self.fields)), dtype=np.float32)
        mask = np.zeros_like(targets, dtype=bool)
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        for row, record in enumerate(records):
            for column, field in enumerate(self.fields):
                value = getattr(record, field)
                if value is not None:
                    targets[row, column] = (float(value) - mean[column]) / scale[column]
                    mask[row, column] = True
        return targets, mask

    def inverse(self, normalised: np.ndarray) -> np.ndarray:
        values = np.asarray(normalised, dtype=np.float32)
        if values.shape[-1] != len(self.fields):
            raise ValueError("physical target dimension differs from fitted scaler")
        return values * np.asarray(self.scale, np.float32) + np.asarray(
            self.mean, np.float32
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "fields": list(self.fields),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "available_count": list(self.available_count),
            "normalization_fit_split": "train_only",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PhysicalTargetScaler":
        return cls(
            tuple(str(field) for field in value["fields"]),
            tuple(float(item) for item in value["mean"]),
            tuple(float(item) for item in value["scale"]),
            tuple(int(item) for item in value["available_count"]),
        )


@dataclass(frozen=True)
class Stage2Prediction:
    probabilities: np.ndarray
    member_probabilities: np.ndarray
    breach_probability: np.ndarray
    breach_upper_marginal: np.ndarray
    breach_upper_selection_safe: np.ndarray
    defender_success_probability: np.ndarray
    short_window_breach_probability: np.ndarray
    expected_remaining_steps: np.ndarray
    survival_through_horizon_probability: np.ndarray
    physical_point: Dict[str, np.ndarray]
    physical_member: Dict[str, np.ndarray]
    physical_interval_lower: Dict[str, np.ndarray]
    physical_interval_upper: Dict[str, np.ndarray]
    physical_std: Dict[str, np.ndarray]


class Stage2System:
    """A fitted feature encoder, ensemble, and calibration state."""

    FORMAT_VERSION = 3

    def __init__(
        self,
        model: BootstrapOutcomeEnsemble,
        feature_encoder: Stage2FeatureEncoder,
        normalizer: FeatureNormalizer,
        temperature: TemperatureScaler,
        risk_bound: CalibratedRiskBound,
        config: Stage2TrainingConfig,
        physical_target_scaler: Optional[PhysicalTargetScaler] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ):
        self.model = model
        self.feature_encoder = feature_encoder
        self.normalizer = normalizer
        self.temperature = temperature
        self.risk_bound = risk_bound
        self.config = config
        self.physical_target_scaler = physical_target_scaler
        self.metadata = dict(metadata or {})

    def _features(self, queries: Sequence[Stage2Query]) -> np.ndarray:
        return self.normalizer.transform(self.feature_encoder.transform_queries(queries))

    @torch.no_grad()
    def predict(
        self,
        values: Sequence[Union[Stage2Record, Stage2Query]],
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 4096,
    ) -> Stage2Prediction:
        if not values:
            raise ValueError("cannot predict an empty Stage-2 query list")
        queries = [value.query if isinstance(value, Stage2Record) else value for value in values]
        features = self._features(queries)
        target_device = torch.device(device)
        self.model.to(target_device).eval()
        member_chunks = []
        physical_chunks = []
        for start in range(0, len(features), batch_size):
            tensor = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=target_device
            )
            member_chunks.append(torch.softmax(self.model(tensor), dim=-1).cpu().numpy())
            if self.physical_target_scaler is not None:
                physical_chunks.append(self.model.physical_forward(tensor).cpu().numpy())
        raw_members = np.concatenate(member_chunks, axis=1)
        raw_mean = raw_members.mean(axis=0)
        probabilities = self.temperature.apply(raw_mean)
        calibrated_members = np.stack(
            [self.temperature.apply(member) for member in raw_members], axis=0
        )
        k = self.config.horizon_bins
        breach_bins = probabilities[:, k : 2 * k]
        breach = breach_bins.sum(axis=1)
        defender_success = probabilities[:, :k].sum(axis=1) + probabilities[:, -1]
        early = np.zeros(len(queries), dtype=np.float32)
        for index, query in enumerate(queries):
            command_bins = min(
                k, max(1, int(np.ceil(query.command_steps / self.config.steps_per_bin)))
            )
            early[index] = float(breach_bins[index, :command_bins].sum())
        midpoints = (
            np.arange(k, dtype=np.float32) + 0.5
        ) * self.config.steps_per_bin
        expected_steps = (
            (probabilities[:, :k] + breach_bins) @ midpoints
            + probabilities[:, -1] * self.config.horizon_steps
        )
        physical_point: Dict[str, np.ndarray] = {}
        physical_member: Dict[str, np.ndarray] = {}
        physical_lower: Dict[str, np.ndarray] = {}
        physical_upper: Dict[str, np.ndarray] = {}
        physical_std: Dict[str, np.ndarray] = {}
        if self.physical_target_scaler is not None and physical_chunks:
            normalised_members = np.concatenate(physical_chunks, axis=1)
            members = self.physical_target_scaler.inverse(normalised_members)
            for column, field in enumerate(self.physical_target_scaler.fields):
                if field not in self.physical_target_scaler.available_fields:
                    continue
                values = members[:, :, column]
                if field in {
                    "target_final_health_fraction",
                    "target_min_health_fraction",
                }:
                    values = np.clip(values, 0.0, 1.0)
                elif field in {
                    "defender_survivors",
                    "defender_casualties",
                }:
                    upper = np.asarray([query.defender_count for query in queries], np.float32)
                    values = np.maximum(0.0, np.minimum(values, upper[None, :]))
                elif field in {
                    "attacker_survivors",
                    "attacker_casualties",
                }:
                    upper = np.asarray([query.attacker_count for query in queries], np.float32)
                    values = np.maximum(0.0, np.minimum(values, upper[None, :]))
                elif field != "payoff_red":
                    values = np.maximum(values, 0.0)
                physical_member[field] = values.astype(np.float32)
                physical_point[field] = values.mean(axis=0).astype(np.float32)
                physical_lower[field] = np.quantile(
                    values, self.config.physical_interval_lower_quantile, axis=0
                ).astype(np.float32)
                physical_upper[field] = np.quantile(
                    values, self.config.physical_interval_upper_quantile, axis=0
                ).astype(np.float32)
                physical_std[field] = values.std(axis=0).astype(np.float32)
        return Stage2Prediction(
            probabilities=probabilities,
            member_probabilities=calibrated_members,
            breach_probability=breach.astype(np.float32),
            breach_upper_marginal=self.risk_bound.upper(breach, selection_safe=False),
            breach_upper_selection_safe=self.risk_bound.upper(breach, selection_safe=True),
            defender_success_probability=defender_success.astype(np.float32),
            short_window_breach_probability=early,
            expected_remaining_steps=expected_steps.astype(np.float32),
            survival_through_horizon_probability=probabilities[:, -1].astype(np.float32),
            physical_point=physical_point,
            physical_member=physical_member,
            physical_interval_lower=physical_lower,
            physical_interval_upper=physical_upper,
            physical_std=physical_std,
        )

    def save(self, path: Union[str, Path]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "model_state": self.model.state_dict(),
            "config": asdict(self.config),
            "feature_encoder": self.feature_encoder.to_dict(),
            "normalizer": self.normalizer.to_dict(),
            "temperature": self.temperature.to_dict(),
            "risk_bound": self.risk_bound.to_dict(),
            "physical_target_scaler": (
                self.physical_target_scaler.to_dict()
                if self.physical_target_scaler is not None
                else None
            ),
            "metadata": self.metadata,
        }
        torch.save(payload, output)

    @classmethod
    def load(
        cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu"
    ) -> "Stage2System":
        try:
            payload = torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # PyTorch 1.13 compatibility.
            payload = torch.load(path, map_location=device)
        format_version = int(payload.get("format_version", -1))
        if format_version not in {1, 2, cls.FORMAT_VERSION}:
            raise ValueError("unsupported Stage-2 checkpoint format")
        config = Stage2TrainingConfig.from_dict(payload["config"])
        encoder = Stage2FeatureEncoder.from_dict(payload["feature_encoder"])
        scaler_payload = payload.get("physical_target_scaler")
        physical_target_scaler = (
            PhysicalTargetScaler.from_dict(scaler_payload) if scaler_payload else None
        )
        model = BootstrapOutcomeEnsemble(
            config.ensemble_members,
            encoder.output_dim,
            config.horizon_bins,
            config.hidden_dim,
            model_kind=config.model_kind,
            state_dim=encoder.state_dim,
            physical_target_count=(
                len(physical_target_scaler.fields)
                if physical_target_scaler is not None
                else 0
            ),
        ).to(device)
        model.load_state_dict(payload["model_state"])
        return cls(
            model=model,
            feature_encoder=encoder,
            normalizer=FeatureNormalizer.from_dict(payload["normalizer"]),
            temperature=TemperatureScaler.from_dict(payload["temperature"]),
            risk_bound=CalibratedRiskBound.from_dict(payload["risk_bound"]),
            config=config,
            physical_target_scaler=physical_target_scaler,
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class Stage2FitResult:
    system: Stage2System
    training_summary: Dict[str, object]


def _validate_horizon(records: Sequence[Stage2Record], config: Stage2TrainingConfig) -> None:
    horizons = {record.query.horizon_steps for record in records}
    if horizons != {config.horizon_steps}:
        raise ValueError(
            f"all query horizons must equal bins*steps_per_bin={config.horizon_steps}; got {horizons}"
        )


def _labels(records: Sequence[Stage2Record], config: Stage2TrainingConfig) -> np.ndarray:
    return np.asarray(
        [record.terminal_class(config.horizon_bins, config.steps_per_bin) for record in records],
        dtype=np.int64,
    )


def _censoring_arrays(
    records: Sequence[Stage2Record], config: Stage2TrainingConfig
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray([record.event_observed for record in records], dtype=bool)
    # Number of fully observed time bins at censoring.  Likelihood mass before
    # this index is ruled out; later event mass plus the survival tail remains.
    censor_bins = np.asarray(
        [
            min(config.horizon_bins, int(record.terminal_steps // config.steps_per_bin))
            for record in records
        ],
        dtype=np.int64,
    )
    return observed, censor_bins


def _numpy_censored_nll(
    probabilities: np.ndarray,
    records: Sequence[Stage2Record],
    config: Stage2TrainingConfig,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    labels = _labels(records, config)
    observed, censor_bins = _censoring_arrays(records, config)
    likelihood = values[np.arange(len(values)), labels].copy()
    k = config.horizon_bins
    for index in np.flatnonzero(~observed):
        start = int(censor_bins[index])
        likelihood[index] = (
            values[index, start:k].sum()
            + values[index, k + start : 2 * k].sum()
            + values[index, -1]
        )
    return float(-np.log(np.clip(likelihood, 1e-12, 1.0)).mean())


def _masked_physical_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Per-row MSE over available normalised physical labels."""

    squared = (predictions - targets).square() * mask.to(predictions.dtype)
    per_row = squared.sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(predictions.dtype)
    row_available = mask.any(dim=1)
    per_row = torch.where(row_available, per_row, torch.zeros_like(per_row))
    if reduction == "none":
        return per_row
    if reduction == "mean":
        return per_row[row_available].mean() if row_available.any() else per_row.sum() * 0.0
    raise ValueError("physical MSE reduction must be none or mean")


def fit_stage2_system(
    train_records: Sequence[Stage2Record],
    validation_records: Sequence[Stage2Record],
    temperature_calibration_records: Sequence[Stage2Record],
    risk_calibration_records: Sequence[Stage2Record],
    config: Stage2TrainingConfig = Stage2TrainingConfig(),
    device: Union[str, torch.device] = "cpu",
) -> Stage2FitResult:
    """Fit/validate an ensemble, then use two independent calibration splits."""

    partitions = {
        "train": tuple(train_records),
        "validation": tuple(validation_records),
        "temperature_calibration": tuple(temperature_calibration_records),
        "risk_calibration": tuple(risk_calibration_records),
    }
    if any(not records for records in partitions.values()):
        raise ValueError(
            "train, validation, temperature-calibration and risk-calibration must be non-empty"
        )
    roots = {
        name: {record.query.root_id for record in records}
        for name, records in partitions.items()
    }
    lineage_groups = {
        name: {record.query.lineage_group_id for record in records}
        for name, records in partitions.items()
    }
    names = list(partitions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if roots[left] & roots[right]:
                raise ValueError(f"root leakage between {left} and {right}")
            if lineage_groups[left] & lineage_groups[right]:
                raise ValueError(f"lineage-group leakage between {left} and {right}")
    all_records = tuple(record for records in partitions.values() for record in records)
    _validate_horizon(all_records, config)
    encoder = Stage2FeatureEncoder.fit(
        train_records, include_policy_context=config.include_policy_context
    )
    raw_train = encoder.transform(train_records)
    augmentation_rng = np.random.default_rng(config.seed + 32452843)
    augmentation_summary: Dict[str, object] = {
        "enabled": bool(
            config.include_policy_context
            and config.policy_unknown_augmentation_probability > 0.0
        ),
        "probability_per_side": config.policy_unknown_augmentation_probability,
        "seed": config.seed + 32452843,
        "defender_unknown_rows": 0,
        "attacker_unknown_rows": 0,
        "capability_unknown_rows": 0,
    }
    if augmentation_summary["enabled"]:
        augmented = raw_train.copy()
        defender_start = encoder.state_dim + 6
        attacker_start = defender_start + len(encoder.defender_policies)
        defender_mask = augmentation_rng.random(len(augmented)) < config.policy_unknown_augmentation_probability
        attacker_mask = augmentation_rng.random(len(augmented)) < config.policy_unknown_augmentation_probability
        capability_mask = augmentation_rng.random(len(augmented)) < config.policy_unknown_augmentation_probability
        augmented[defender_mask, defender_start : defender_start + len(encoder.defender_policies)] = 0.0
        augmented[defender_mask, defender_start] = 1.0
        augmented[attacker_mask, attacker_start : attacker_start + len(encoder.attacker_policies)] = 0.0
        augmented[attacker_mask, attacker_start] = 1.0
        capability_start = attacker_start + len(encoder.attacker_policies)
        augmented[
            capability_mask,
            capability_start : capability_start + len(encoder.capability_versions),
        ] = 0.0
        augmented[capability_mask, capability_start] = 1.0
        augmentation_summary["defender_unknown_rows"] = int(defender_mask.sum())
        augmentation_summary["attacker_unknown_rows"] = int(attacker_mask.sum())
        augmentation_summary["capability_unknown_rows"] = int(capability_mask.sum())
        raw_train = np.concatenate([raw_train, augmented], axis=0)
        train_record_indices = np.tile(np.arange(len(train_records)), 2)
    else:
        train_record_indices = np.arange(len(train_records))
    normalizer = FeatureNormalizer.fit(raw_train)
    if config.model_kind == "had_deepset":
        if encoder.state_dim != 85:
            raise ValueError("had_deepset requires HADCanonicalizer state_dim=85")
        # Canonical state is already dimensionless/normalised. Leaving its
        # entire prefix untouched makes the shared entity phi genuinely
        # invariant to slot permutation; slot-wise z-scoring would break that.
        normalizer = normalizer.with_passthrough(range(encoder.state_dim))
    train_features = normalizer.transform(raw_train)
    validation_features = normalizer.transform(encoder.transform(validation_records))
    temperature_features = normalizer.transform(
        encoder.transform(temperature_calibration_records)
    )
    risk_features = normalizer.transform(encoder.transform(risk_calibration_records))
    physical_scaler = PhysicalTargetScaler.fit(train_records)
    train_physical, train_physical_mask = physical_scaler.transform(train_records)
    validation_physical, validation_physical_mask = physical_scaler.transform(
        validation_records
    )
    train_labels = _labels(train_records, config)
    validation_labels = _labels(validation_records, config)
    temperature_labels = _labels(temperature_calibration_records, config)
    train_observed, train_censor_bins = _censoring_arrays(train_records, config)
    validation_observed, validation_censor_bins = _censoring_arrays(
        validation_records, config
    )
    temperature_observed, temperature_censor_bins = _censoring_arrays(
        temperature_calibration_records, config
    )
    base_train_weights = np.asarray(
        [record.sample_weight for record in train_records], dtype=np.float32
    )
    train_labels = train_labels[train_record_indices]
    train_observed = train_observed[train_record_indices]
    train_censor_bins = train_censor_bins[train_record_indices]
    train_weights = base_train_weights[train_record_indices]
    train_physical = train_physical[train_record_indices]
    train_physical_mask = train_physical_mask[train_record_indices]
    if len(train_record_indices) > len(train_records):
        train_weights *= 0.5

    group_to_indices: Dict[str, list] = {}
    for feature_index, record_index in enumerate(train_record_indices):
        group_id = train_records[int(record_index)].query.lineage_group_id
        group_to_indices.setdefault(group_id, []).append(feature_index)
    cluster_ids = np.asarray(sorted(group_to_indices), dtype=object)

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch.manual_seed(config.seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = BootstrapOutcomeEnsemble(
        config.ensemble_members,
        encoder.output_dim,
        config.horizon_bins,
        config.hidden_dim,
        model_kind=config.model_kind,
        state_dim=encoder.state_dim,
        physical_target_count=len(physical_scaler.fields),
    ).to(target_device)
    x = torch.as_tensor(train_features, dtype=torch.float32, device=target_device)
    y = torch.as_tensor(train_labels, dtype=torch.long, device=target_device)
    weights = torch.as_tensor(train_weights, dtype=torch.float32, device=target_device)
    observed = torch.as_tensor(train_observed, dtype=torch.bool, device=target_device)
    censor_bins = torch.as_tensor(train_censor_bins, dtype=torch.long, device=target_device)
    validation_x = torch.as_tensor(
        validation_features, dtype=torch.float32, device=target_device
    )
    validation_y = torch.as_tensor(
        validation_labels, dtype=torch.long, device=target_device
    )
    validation_event_observed = torch.as_tensor(
        validation_observed, dtype=torch.bool, device=target_device
    )
    validation_censor_bin = torch.as_tensor(
        validation_censor_bins, dtype=torch.long, device=target_device
    )
    physical_y = torch.as_tensor(
        train_physical, dtype=torch.float32, device=target_device
    )
    physical_mask = torch.as_tensor(
        train_physical_mask, dtype=torch.bool, device=target_device
    )
    validation_physical_y = torch.as_tensor(
        validation_physical, dtype=torch.float32, device=target_device
    )
    validation_physical_mask_tensor = torch.as_tensor(
        validation_physical_mask, dtype=torch.bool, device=target_device
    )
    histories = []
    for member_index, member in enumerate(model.members):
        physical_head = model.physical_heads[member_index]
        member_seed = config.seed + 104729 * (member_index + 1)
        rng = np.random.default_rng(member_seed)
        sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        bootstrap = np.concatenate(
            [np.asarray(group_to_indices[str(group_id)], dtype=np.int64) for group_id in sampled_clusters]
        )
        optimizer = torch.optim.AdamW(
            list(member.parameters()) + list(physical_head.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        epoch_losses = []
        epoch_event_losses = []
        epoch_physical_losses = []
        validation_losses = []
        validation_event_losses = []
        validation_physical_losses = []
        best_validation = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(member.state_dict())
        best_physical_state = copy.deepcopy(physical_head.state_dict())
        stale_epochs = 0
        for epoch in range(config.epochs):
            order = bootstrap[rng.permutation(len(bootstrap))]
            total_loss = 0.0
            total_event_loss = 0.0
            total_physical_loss = 0.0
            total_weight = 0.0
            total_physical_weight = 0.0
            member.train()
            physical_head.train()
            for start in range(0, len(order), config.batch_size):
                indices = torch.as_tensor(
                    order[start : start + config.batch_size],
                    dtype=torch.long,
                    device=target_device,
                )
                logits = member(x[indices])
                sample_loss = competing_risk_nll(
                    logits,
                    y[indices],
                    observed[indices],
                    censor_bins[indices],
                    config.horizon_bins,
                    reduction="none",
                )
                sample_physical_loss = _masked_physical_mse(
                    physical_head(x[indices]),
                    physical_y[indices],
                    physical_mask[indices],
                    reduction="none",
                )
                batch_weight = weights[indices]
                event_loss = (sample_loss * batch_weight).sum() / batch_weight.sum().clamp_min(1e-8)
                physical_available = physical_mask[indices].any(dim=1).to(
                    batch_weight.dtype
                )
                physical_loss = (
                    (sample_physical_loss * batch_weight).sum()
                    / (batch_weight * physical_available).sum().clamp_min(1e-8)
                )
                loss = event_loss + config.physical_loss_weight * physical_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(member.parameters()) + list(physical_head.parameters()),
                    config.gradient_clip,
                )
                optimizer.step()
                total_loss += float((sample_loss.detach() * batch_weight).sum().cpu())
                total_event_loss += float((sample_loss.detach() * batch_weight).sum().cpu())
                total_physical_loss += float(
                    (sample_physical_loss.detach() * batch_weight).sum().cpu()
                )
                total_weight += float(batch_weight.sum().cpu())
                total_physical_weight += float(
                    (batch_weight * physical_available).sum().cpu()
                )
            epoch_event = total_event_loss / max(total_weight, 1e-8)
            epoch_physical = total_physical_loss / max(total_physical_weight, 1e-8)
            epoch_losses.append(epoch_event + config.physical_loss_weight * epoch_physical)
            epoch_event_losses.append(epoch_event)
            epoch_physical_losses.append(epoch_physical)
            member.eval()
            physical_head.eval()
            with torch.no_grad():
                validation_event_loss = float(
                    competing_risk_nll(
                        member(validation_x),
                        validation_y,
                        validation_event_observed,
                        validation_censor_bin,
                        config.horizon_bins,
                    ).cpu()
                )
                validation_physical_loss = float(
                    _masked_physical_mse(
                        physical_head(validation_x),
                        validation_physical_y,
                        validation_physical_mask_tensor,
                    ).cpu()
                )
                validation_loss = (
                    validation_event_loss
                    + config.physical_loss_weight * validation_physical_loss
                )
            validation_losses.append(validation_loss)
            validation_event_losses.append(validation_event_loss)
            validation_physical_losses.append(validation_physical_loss)
            if validation_loss < best_validation - 1e-7:
                best_validation = validation_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(member.state_dict())
                best_physical_state = copy.deepcopy(physical_head.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                epoch + 1 >= config.minimum_epochs
                and stale_epochs >= config.early_stopping_patience
            ):
                break
        member.load_state_dict(best_state)
        physical_head.load_state_dict(best_physical_state)
        histories.append(
            {
                "member": member_index,
                "initial_loss": float(epoch_losses[0]),
                "final_loss": float(epoch_losses[-1]),
                "best_loss": float(min(epoch_losses)),
                "best_validation_multitask_loss": float(best_validation),
                "best_validation_censored_nll": float(
                    validation_event_losses[best_epoch - 1]
                ),
                "best_validation_normalized_physical_mse": float(
                    validation_physical_losses[best_epoch - 1]
                ),
                "final_train_censored_nll": float(epoch_event_losses[-1]),
                "final_train_normalized_physical_mse": float(epoch_physical_losses[-1]),
                "best_epoch": int(best_epoch),
                "epochs_ran": len(epoch_losses),
                "early_stopped": len(epoch_losses) < config.epochs,
                "bootstrap_unit": "lineage_group_id",
                "sampled_clusters": len(sampled_clusters),
                "unique_sampled_clusters": len(set(sampled_clusters.tolist())),
            }
        )

    model.eval()
    with torch.no_grad():
        temperature_tensor = torch.as_tensor(
            temperature_features, dtype=torch.float32, device=target_device
        )
        raw_temperature_members = torch.softmax(model(temperature_tensor), dim=-1).cpu().numpy()
    raw_temperature_probability = raw_temperature_members.mean(axis=0)
    temperature = TemperatureScaler.fit_right_censored(
        raw_temperature_probability,
        temperature_labels,
        temperature_observed,
        temperature_censor_bins,
        config.horizon_bins,
    )
    calibrated_temperature_probability = temperature.apply(raw_temperature_probability)

    with torch.no_grad():
        risk_tensor = torch.as_tensor(risk_features, dtype=torch.float32, device=target_device)
        raw_risk_members = torch.softmax(model(risk_tensor), dim=-1).cpu().numpy()
    calibrated_risk_probability = temperature.apply(raw_risk_members.mean(axis=0))
    k = config.horizon_bins
    calibration_breach = calibrated_risk_probability[:, k : 2 * k].sum(axis=1)
    risk_bound = CalibratedRiskBound.fit(
        calibration_breach,
        [record.outcome == "breach" for record in risk_calibration_records],
        [record.query.root_id for record in risk_calibration_records],
        [record.query.candidate_id for record in risk_calibration_records],
        [record.query.lineage_group_id for record in risk_calibration_records],
        alpha=config.calibration_alpha,
        require_finite_sample_guarantee=config.require_conformal_guarantee,
    )
    raw_nll = _numpy_censored_nll(
        raw_temperature_probability, temperature_calibration_records, config
    )
    calibrated_nll = _numpy_censored_nll(
        calibrated_temperature_probability, temperature_calibration_records, config
    )
    summary: Dict[str, object] = {
        "train_records": len(train_records),
        "train_roots": len(roots["train"]),
        "train_lineage_groups": len(lineage_groups["train"]),
        "validation_records": len(validation_records),
        "validation_roots": len(roots["validation"]),
        "validation_lineage_groups": len(lineage_groups["validation"]),
        "temperature_calibration_records": len(temperature_calibration_records),
        "temperature_calibration_roots": len(roots["temperature_calibration"]),
        "temperature_calibration_lineage_groups": len(
            lineage_groups["temperature_calibration"]
        ),
        "risk_calibration_records": len(risk_calibration_records),
        "risk_calibration_roots": len(roots["risk_calibration"]),
        "risk_calibration_lineage_groups": len(lineage_groups["risk_calibration"]),
        "root_and_lineage_group_leakage_check": "passed",
        "temperature_and_risk_calibration_are_disjoint": True,
        "validation_is_disjoint_from_training_and_calibration": True,
        "selection_metric": "validation_competing_risk_nll_plus_normalized_physical_mse",
        "model_kind": config.model_kind,
        "right_censoring_semantics": "timeout_is_not_an_observed_terminal_event",
        "bootstrap_unit": "lineage_group_id",
        "policy_unknown_augmentation": augmentation_summary,
        "physical_multitask": {
            "enabled": True,
            "loss_weight": config.physical_loss_weight,
            "targets": list(physical_scaler.fields),
            "available_train_count": dict(
                zip(physical_scaler.fields, physical_scaler.available_count)
            ),
            "normalization": "train_only_z_score; constant targets use scale=1",
            "derived_redundant_targets": {
                "defender_survivors": "defender_count - defender_casualties",
                "defender_casualties": "defender_count - defender_survivors",
                "attacker_survivors": "attacker_count - attacker_casualties",
                "attacker_casualties": "attacker_count - attacker_survivors",
            },
        },
        "member_losses": histories,
        "temperature": temperature.temperature,
        "temperature_calibration_joint_nll_before": raw_nll,
        "temperature_calibration_joint_nll_after": calibrated_nll,
        "risk_bound": risk_bound.to_dict(),
        "conformal_finite_sample_gate_required": config.require_conformal_guarantee,
    }
    system = Stage2System(
        model=model,
        feature_encoder=encoder,
        normalizer=normalizer,
        temperature=temperature,
        risk_bound=risk_bound,
        config=config,
        physical_target_scaler=physical_scaler,
        metadata={"training_summary": summary},
    )
    return Stage2FitResult(system, summary)

"""End-to-end fitting, prediction, and checkpointing for Stage 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn import functional as F

from .calibration import CalibratedRiskBound, TemperatureScaler
from .data import (
    FeatureNormalizer,
    Stage2FeatureEncoder,
    Stage2Query,
    Stage2Record,
)
from .outcome_model import BootstrapOutcomeEnsemble


@dataclass(frozen=True)
class Stage2TrainingConfig:
    horizon_bins: int = 8
    steps_per_bin: int = 10
    hidden_dim: int = 64
    ensemble_members: int = 5
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 5.0
    calibration_alpha: float = 0.10
    include_policy_context: bool = True
    policy_unknown_augmentation_probability: float = 0.15
    seed: int = 7

    def __post_init__(self) -> None:
        if self.horizon_bins < 1 or self.steps_per_bin < 1:
            raise ValueError("time discretisation must be positive")
        if self.hidden_dim < 1 or self.ensemble_members < 2:
            raise ValueError("hidden_dim must be positive and ensemble needs at least two members")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("invalid Stage-2 optimisation configuration")
        if self.weight_decay < 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("weight decay/gradient clip are invalid")
        if not 0.0 < self.calibration_alpha < 1.0:
            raise ValueError("calibration_alpha must lie inside (0,1)")
        if not 0.0 <= self.policy_unknown_augmentation_probability <= 1.0:
            raise ValueError("policy_unknown_augmentation_probability must lie inside [0,1]")

    @property
    def horizon_steps(self) -> int:
        return self.horizon_bins * self.steps_per_bin

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Stage2TrainingConfig":
        known = {name: value[name] for name in cls.__dataclass_fields__ if name in value}
        return cls(**known)


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


class Stage2System:
    """A fitted feature encoder, ensemble, and calibration state."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        model: BootstrapOutcomeEnsemble,
        feature_encoder: Stage2FeatureEncoder,
        normalizer: FeatureNormalizer,
        temperature: TemperatureScaler,
        risk_bound: CalibratedRiskBound,
        config: Stage2TrainingConfig,
        metadata: Optional[Mapping[str, object]] = None,
    ):
        self.model = model
        self.feature_encoder = feature_encoder
        self.normalizer = normalizer
        self.temperature = temperature
        self.risk_bound = risk_bound
        self.config = config
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
        for start in range(0, len(features), batch_size):
            tensor = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=target_device
            )
            member_chunks.append(torch.softmax(self.model(tensor), dim=-1).cpu().numpy())
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
        return Stage2Prediction(
            probabilities=probabilities,
            member_probabilities=calibrated_members,
            breach_probability=breach.astype(np.float32),
            breach_upper_marginal=self.risk_bound.upper(breach, selection_safe=False),
            breach_upper_selection_safe=self.risk_bound.upper(breach, selection_safe=True),
            defender_success_probability=defender_success.astype(np.float32),
            short_window_breach_probability=early,
            expected_remaining_steps=expected_steps.astype(np.float32),
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
        if int(payload.get("format_version", -1)) != cls.FORMAT_VERSION:
            raise ValueError("unsupported Stage-2 checkpoint format")
        config = Stage2TrainingConfig.from_dict(payload["config"])
        encoder = Stage2FeatureEncoder.from_dict(payload["feature_encoder"])
        model = BootstrapOutcomeEnsemble(
            config.ensemble_members,
            encoder.output_dim,
            config.horizon_bins,
            config.hidden_dim,
        ).to(device)
        model.load_state_dict(payload["model_state"])
        return cls(
            model=model,
            feature_encoder=encoder,
            normalizer=FeatureNormalizer.from_dict(payload["normalizer"]),
            temperature=TemperatureScaler.from_dict(payload["temperature"]),
            risk_bound=CalibratedRiskBound.from_dict(payload["risk_bound"]),
            config=config,
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


def fit_stage2_system(
    train_records: Sequence[Stage2Record],
    temperature_calibration_records: Sequence[Stage2Record],
    risk_calibration_records: Sequence[Stage2Record],
    config: Stage2TrainingConfig = Stage2TrainingConfig(),
    device: Union[str, torch.device] = "cpu",
) -> Stage2FitResult:
    """Fit clustered bootstrap members, temperature, then independent risk bound."""

    partitions = {
        "train": tuple(train_records),
        "temperature_calibration": tuple(temperature_calibration_records),
        "risk_calibration": tuple(risk_calibration_records),
    }
    if any(not records for records in partitions.values()):
        raise ValueError("train, temperature-calibration and risk-calibration must be non-empty")
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
    train_features = normalizer.transform(raw_train)
    temperature_features = normalizer.transform(
        encoder.transform(temperature_calibration_records)
    )
    risk_features = normalizer.transform(encoder.transform(risk_calibration_records))
    train_labels = _labels(train_records, config)
    temperature_labels = _labels(temperature_calibration_records, config)
    base_train_weights = np.asarray(
        [record.sample_weight for record in train_records], dtype=np.float32
    )
    train_labels = train_labels[train_record_indices]
    train_weights = base_train_weights[train_record_indices]
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
    ).to(target_device)
    x = torch.as_tensor(train_features, dtype=torch.float32, device=target_device)
    y = torch.as_tensor(train_labels, dtype=torch.long, device=target_device)
    weights = torch.as_tensor(train_weights, dtype=torch.float32, device=target_device)
    histories = []
    for member_index, member in enumerate(model.members):
        member_seed = config.seed + 104729 * (member_index + 1)
        rng = np.random.default_rng(member_seed)
        sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        bootstrap = np.concatenate(
            [np.asarray(group_to_indices[str(group_id)], dtype=np.int64) for group_id in sampled_clusters]
        )
        optimizer = torch.optim.AdamW(
            member.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        epoch_losses = []
        for _ in range(config.epochs):
            order = bootstrap[rng.permutation(len(bootstrap))]
            total_loss = 0.0
            total_weight = 0.0
            member.train()
            for start in range(0, len(order), config.batch_size):
                indices = torch.as_tensor(
                    order[start : start + config.batch_size],
                    dtype=torch.long,
                    device=target_device,
                )
                logits = member(x[indices])
                sample_loss = F.cross_entropy(logits, y[indices], reduction="none")
                batch_weight = weights[indices]
                loss = (sample_loss * batch_weight).sum() / batch_weight.sum().clamp_min(1e-8)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(member.parameters(), config.gradient_clip)
                optimizer.step()
                total_loss += float((sample_loss.detach() * batch_weight).sum().cpu())
                total_weight += float(batch_weight.sum().cpu())
            epoch_losses.append(total_loss / max(total_weight, 1e-8))
        histories.append(
            {
                "member": member_index,
                "initial_loss": float(epoch_losses[0]),
                "final_loss": float(epoch_losses[-1]),
                "best_loss": float(min(epoch_losses)),
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
    temperature = TemperatureScaler.fit(raw_temperature_probability, temperature_labels)
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
    )
    raw_nll = float(
        -np.log(
            np.clip(
                raw_temperature_probability[
                    np.arange(len(temperature_labels)), temperature_labels
                ],
                1e-12,
                1.0,
            )
        ).mean()
    )
    calibrated_nll = float(
        -np.log(
            np.clip(
                calibrated_temperature_probability[
                    np.arange(len(temperature_labels)), temperature_labels
                ],
                1e-12,
                1.0,
            )
        ).mean()
    )
    summary: Dict[str, object] = {
        "train_records": len(train_records),
        "train_roots": len(roots["train"]),
        "train_lineage_groups": len(lineage_groups["train"]),
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
        "bootstrap_unit": "lineage_group_id",
        "policy_unknown_augmentation": augmentation_summary,
        "member_losses": histories,
        "temperature": temperature.temperature,
        "temperature_calibration_joint_nll_before": raw_nll,
        "temperature_calibration_joint_nll_after": calibrated_nll,
        "risk_bound": risk_bound.to_dict(),
    }
    system = Stage2System(
        model=model,
        feature_encoder=encoder,
        normalizer=normalizer,
        temperature=temperature,
        risk_bound=risk_bound,
        config=config,
        metadata={"training_summary": summary},
    )
    return Stage2FitResult(system, summary)

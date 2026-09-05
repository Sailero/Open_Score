"""Frozen Stage-2 inference helpers for Stage-3 local Blotto payoffs.

New aligned checkpoints carry their independently fitted calibration layer.
The legacy train-split calibration entry point remains readable only so old
Round-01 evidence can still be audited; the aligned Stage-3 protocol never fits
or changes a payoff model at evaluation time.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from open_score.stage2 import (
    DynamicHADOutcomeNet,
    HADVariableSetState,
    project_roster_probability_surface,
)


Progress = Optional[Callable[[str], None]]
SUPPORTED_LOCAL_UTILITIES = frozenset(
    {"centered_red_win_probability", "joint_survival_log_probability"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logit(probability: float, eps: float = 1e-5) -> float:
    value = min(1.0 - eps, max(eps, float(probability)))
    return math.log(value) - math.log1p(-value)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def survival_probability_to_utility(
    probability: np.ndarray,
    *,
    utility_mode: str = "centered_red_win_probability",
    risk_epsilon: float = 0.01,
) -> np.ndarray:
    """Map local survival probabilities to an additive Blotto utility.

    ``joint_survival_log_probability`` implements the series-system mission
    objective used by HAD: every active objective must survive.  Under the
    local conditional-independence surrogate,

        log P(all survive) = sum_m log P(target m survives).

    Dividing by ``-log(risk_epsilon)`` only normalizes the numerical scale and
    leaves every best response and equilibrium unchanged.
    """

    values = np.asarray(probability, dtype=np.float64)
    if utility_mode not in SUPPORTED_LOCAL_UTILITIES:
        raise ValueError(f"unsupported local utility: {utility_mode}")
    if not np.isfinite(risk_epsilon) or not 0.0 < float(risk_epsilon) < 0.5:
        raise ValueError("risk_epsilon must lie strictly between zero and 0.5")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("survival probabilities must be finite and lie in [0, 1]")
    if utility_mode == "centered_red_win_probability":
        return 2.0 * values - 1.0
    clipped = np.clip(values, float(risk_epsilon), 1.0)
    return np.log(clipped) / -math.log(float(risk_epsilon))


def utility_to_survival_probability(
    utility: np.ndarray,
    *,
    utility_mode: str = "centered_red_win_probability",
    risk_epsilon: float = 0.01,
) -> np.ndarray:
    """Invert a supported local utility for diagnostics and reporting."""

    values = np.asarray(utility, dtype=np.float64)
    if utility_mode not in SUPPORTED_LOCAL_UTILITIES:
        raise ValueError(f"unsupported local utility: {utility_mode}")
    if not np.isfinite(risk_epsilon) or not 0.0 < float(risk_epsilon) < 0.5:
        raise ValueError("risk_epsilon must lie strictly between zero and 0.5")
    if utility_mode == "centered_red_win_probability":
        return np.clip((values + 1.0) / 2.0, 0.0, 1.0)
    return np.clip(
        np.exp(values * -math.log(float(risk_epsilon))),
        float(risk_epsilon),
        1.0,
    )


@dataclass(frozen=True)
class StyleCalibration:
    """Roster-conditional Blue-style log-odds offsets."""

    offsets: Mapping[tuple[str, int, int], float]
    global_offsets: Mapping[str, float]
    source_sha256: str
    train_episodes: int

    def offset(self, style: str, red_count: int, blue_count: int) -> float:
        return float(
            self.offsets.get(
                (str(style), int(red_count), int(blue_count)),
                self.global_offsets.get(str(style), 0.0),
            )
        )

    def adjust(
        self, probability: float, style: str, red_count: int, blue_count: int
    ) -> float:
        return _sigmoid(
            _logit(probability) + self.offset(style, red_count, blue_count)
        )


def fit_train_style_calibration(
    dataset_path: Path,
    *,
    smoothing: float = 1.0,
    progress: Progress = None,
) -> StyleCalibration:
    """Fit offsets from unique step-zero training episodes only.

    Using step zero prevents long episodes from receiving more calibration
    weight merely because they produced more snapshots.
    """

    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    by_cell: dict[tuple[str, int, int], list[int]] = defaultdict(lambda: [0, 0])
    by_roster: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    by_style: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    rows = 0
    episodes = 0
    if progress:
        progress(f"[S2校准] 扫描训练集：{path}")
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            rows += 1
            row = json.loads(line)
            if row.get("split") != "train" or int(row.get("step", -1)) != 0:
                continue
            style = str(row["opponent"])
            red = int(row["red_count"])
            blue = int(row["blue_count"])
            win = int(bool(row["red_win"]))
            for bucket in (by_cell[(style, red, blue)], by_roster[(red, blue)]):
                bucket[0] += win
                bucket[1] += 1
            by_style[style][0] += win
            by_style[style][1] += 1
            overall[0] += win
            overall[1] += 1
            episodes += 1
            if progress and episodes % 2000 == 0:
                progress(
                    f"[S2校准] 已读取 {rows:,} 行，纳入 {episodes:,} 个训练 episode"
                )
    if episodes == 0:
        raise ValueError("dataset contains no step-zero training episodes")

    def smoothed(bucket: Sequence[int]) -> float:
        return (bucket[0] + smoothing) / (bucket[1] + 2.0 * smoothing)

    offsets: dict[tuple[str, int, int], float] = {}
    for key, bucket in by_cell.items():
        roster = by_roster[(key[1], key[2])]
        offsets[key] = _logit(smoothed(bucket)) - _logit(smoothed(roster))
    overall_probability = smoothed(overall)
    global_offsets = {
        style: _logit(smoothed(bucket)) - _logit(overall_probability)
        for style, bucket in by_style.items()
    }
    result = StyleCalibration(
        offsets=offsets,
        global_offsets=global_offsets,
        source_sha256=sha256_file(path),
        train_episodes=episodes,
    )
    if progress:
        details = ", ".join(
            f"{name}={value:+.3f}" for name, value in sorted(global_offsets.items())
        )
        progress(f"[S2校准] 完成：{episodes:,} 局；全局 log-odds 偏移 {details}")
    return result


class FrozenStage2Payoff:
    """Load the official dynamic outcome-time model for batched local payoffs."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: torch.device | str = "cpu",
        expected_sha256: str = "",
        style_calibration: Optional[StyleCalibration] = None,
    ) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
            raise ValueError("Stage-2 checkpoint SHA-256 differs from protocol")
        self.path = path
        self.sha256 = actual_sha
        self.device = torch.device(device)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        required = {
            "model",
            "horizon_bins",
            "steps_per_bin",
            "entity_hidden_dim",
            "hidden_dim",
            "temperature",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Stage-2 checkpoint lacks fields: {sorted(missing)}")
        self.model = DynamicHADOutcomeNet(
            int(payload["horizon_bins"]),
            int(payload["entity_hidden_dim"]),
            int(payload["hidden_dim"]),
        ).to(self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.temperature = float(payload["temperature"])
        self.steps_per_bin = int(payload["steps_per_bin"])
        embedded_calibration = payload.get("style_calibration")
        if style_calibration is not None:
            self.style_calibration = style_calibration
        elif isinstance(embedded_calibration, Mapping):
            raw_offsets = embedded_calibration.get("offsets", {})
            offsets: dict[tuple[str, int, int], float] = {}
            for key, value in raw_offsets.items():
                style, red, blue = str(key).split("|", maxsplit=2)
                offsets[(style, int(red), int(blue))] = float(value)
            self.style_calibration = StyleCalibration(
                offsets=offsets,
                global_offsets={
                    str(key): float(value)
                    for key, value in embedded_calibration.get(
                        "global_offsets", {}
                    ).items()
                },
                source_sha256=str(
                    embedded_calibration.get("source_dataset_sha256", "")
                ),
                train_episodes=int(
                    round(
                        float(
                            embedded_calibration.get(
                                "effective_episode_mass", 0.0
                            )
                        )
                    )
                ),
            )
        else:
            self.style_calibration = None
        self.calibration_source_split = (
            str(embedded_calibration.get("source_split"))
            if isinstance(embedded_calibration, Mapping)
            else None
        )
        self.execution_semantics = payload.get("execution_semantics", {})
        self.supported_roster = payload.get("supported_roster", {})
        self.shape_calibration = payload.get("shape_calibration", {})

    @staticmethod
    def _pad(
        states: Sequence[HADVariableSetState], field: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        arrays = [np.asarray(getattr(state, field), dtype=np.float32) for state in states]
        maximum = max(len(array) for array in arrays)
        feature_dim = arrays[0].shape[1]
        padded = np.zeros((len(arrays), maximum, feature_dim), dtype=np.float32)
        mask = np.zeros((len(arrays), maximum), dtype=bool)
        for index, array in enumerate(arrays):
            if array.ndim != 2 or array.shape[1] != feature_dim:
                raise ValueError(f"inconsistent {field} entity table")
            padded[index, : len(array)] = array
            mask[index, : len(array)] = True
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def predict_red_win(
        self,
        states: Sequence[HADVariableSetState],
        *,
        styles: Optional[Sequence[str]] = None,
        rosters: Optional[Sequence[tuple[int, int]]] = None,
        batch_size: int = 512,
    ) -> np.ndarray:
        if not states:
            return np.empty(0, dtype=np.float32)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if styles is not None and len(styles) != len(states):
            raise ValueError("styles must align with states")
        if rosters is not None and len(rosters) != len(states):
            raise ValueError("rosters must align with states")
        target = torch.from_numpy(
            np.stack([np.asarray(state.target, np.float32) for state in states])
        )
        context = torch.from_numpy(
            np.stack([np.asarray(state.context, np.float32) for state in states])
        )
        red, red_mask = self._pad(states, "red_entities")
        blue, blue_mask = self._pad(states, "blue_entities")
        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(states), batch_size):
                stop = min(start + batch_size, len(states))
                probabilities = self.model.probabilities(
                    target[start:stop].to(self.device),
                    red[start:stop].to(self.device),
                    blue[start:stop].to(self.device),
                    context[start:stop].to(self.device),
                    red_mask[start:stop].to(self.device),
                    blue_mask[start:stop].to(self.device),
                    temperature=self.temperature,
                )
                predictions.append(
                    probabilities[..., : self.model.horizon_bins]
                    .sum(-1)
                    .cpu()
                    .numpy()
                )
        result = np.concatenate(predictions).astype(np.float64)
        if self.style_calibration is not None and styles is not None:
            if rosters is None:
                rosters = [
                    (len(state.red_entities), len(state.blue_entities))
                    for state in states
                ]
            result = np.asarray(
                [
                    self.style_calibration.adjust(value, style, *roster)
                    for value, style, roster in zip(result, styles, rosters)
                ],
                dtype=np.float64,
            )
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def centered_utility(self, *args, **kwargs) -> np.ndarray:
        return 2.0 * self.predict_red_win(*args, **kwargs) - 1.0

    def predict_roster_surface(
        self,
        states: Sequence[HADVariableSetState],
        *,
        target_count: int,
        red_cap: int,
        blue_cap: int,
        styles: Sequence[str],
        rosters: Sequence[tuple[int, int]],
        batch_size: int = 512,
    ) -> np.ndarray:
        """Return the frozen Stage-2 shape-calibrated non-empty roster grid."""

        expected = int(target_count) * int(red_cap) * int(blue_cap)
        if len(states) != expected:
            raise ValueError("states do not form a complete non-empty roster surface")
        raw = self.predict_red_win(
            states, styles=styles, rosters=rosters, batch_size=batch_size
        ).reshape(int(target_count), int(red_cap), int(blue_cap))
        calibration = self.shape_calibration
        if not calibration:
            raise ValueError("Stage-2 checkpoint lacks frozen roster-shape calibration")
        if calibration.get("method") != "alternating_logit_concave_isotonic_v1":
            raise ValueError("unsupported Stage-2 roster-shape calibration method")
        return project_roster_probability_surface(
            raw,
            max_iterations=int(calibration.get("max_iterations", 50)),
            tolerance=float(calibration.get("tolerance", 1e-7)),
        ).astype(np.float32)


def build_local_payoff_tensor(
    target_count: int,
    red_cap: int,
    blue_cap: int,
    state_factory: Callable[[int, int, int], HADVariableSetState],
    predictor: FrozenStage2Payoff,
    *,
    blue_style: str,
    batch_size: int = 512,
    direct_red_cap: Optional[int] = None,
    direct_blue_cap: Optional[int] = None,
    tail_decay: Optional[float] = None,
    utility_mode: str = "centered_red_win_probability",
    risk_epsilon: float = 0.01,
    enforce_monotonicity: bool = False,
) -> np.ndarray:
    """Return ``target x (red+1) x (blue+1)`` local Red utilities.

    Empty-side cells are exact task boundaries.  Every non-empty count pair is
    queried directly from one frozen, calibrated Stage-2 checkpoint.  Stage 3
    deliberately performs no monotone projection and no population tail
    extrapolation: roster generalization and diminishing returns belong to the
    Stage-2 training/evaluation contract.

    ``direct_*_cap`` are retained as explicit checkpoint-support declarations,
    not action masks.  They must cover the complete live budget supplied to
    this function.  ``tail_decay`` is a rejected legacy argument.
    """

    if target_count < 1 or red_cap < 1 or blue_cap < 1:
        raise ValueError("target_count and local caps must be positive")
    direct_red = red_cap if direct_red_cap is None else int(direct_red_cap)
    direct_blue = blue_cap if direct_blue_cap is None else int(direct_blue_cap)
    if direct_red < red_cap or direct_blue < blue_cap:
        raise ValueError(
            "the frozen Stage-2 support must cover every legal non-empty count"
        )
    if tail_decay is not None:
        raise ValueError(
            "Stage-3 population-tail extrapolation was removed; train Stage 2 instead"
        )
    if enforce_monotonicity:
        raise ValueError(
            "Stage-3 payoff projection was removed; shape constraints belong to Stage 2"
        )
    if utility_mode not in SUPPORTED_LOCAL_UTILITIES:
        raise ValueError(f"unsupported local utility: {utility_mode}")

    survival = np.empty((target_count, red_cap + 1, blue_cap + 1), np.float64)
    states: list[HADVariableSetState] = []
    indices: list[tuple[int, int, int]] = []
    for target in range(target_count):
        for red in range(red_cap + 1):
            for blue in range(blue_cap + 1):
                if blue == 0:
                    survival[target, red, blue] = 1.0
                elif red == 0:
                    survival[target, red, blue] = 0.0
                else:
                    states.append(state_factory(target, red, blue))
                    indices.append((target, red, blue))
    styles = [blue_style] * len(states)
    rosters = [(red, blue) for _, red, blue in indices]
    if hasattr(predictor, "predict_roster_surface"):
        values = predictor.predict_roster_surface(
            states,
            target_count=target_count,
            red_cap=red_cap,
            blue_cap=blue_cap,
            styles=styles,
            rosters=rosters,
            batch_size=batch_size,
        ).reshape(-1)
        for index, value in zip(indices, values):
            survival[index] = np.clip(float(value), 0.0, 1.0)
    else:
        # Duck-typed predictors are retained for isolated solver/unit tests.
        # Formal aligned runs require FrozenStage2Payoff and are checked by the
        # evaluator before reaching this function.
        values = predictor.centered_utility(
            states,
            styles=styles,
            rosters=rosters,
            batch_size=batch_size,
        )
        for index, value in zip(indices, values):
            survival[index] = np.clip((float(value) + 1.0) / 2.0, 0.0, 1.0)

    if not np.all(np.isfinite(survival)):
        raise RuntimeError("direct Stage-2 payoff query left non-finite cells")
    utility = survival_probability_to_utility(
        survival,
        utility_mode=utility_mode,
        risk_epsilon=risk_epsilon,
    )
    if utility_mode == "joint_survival_log_probability":
        # ``r=0, b>0`` is the registered structural breach boundary, not an
        # uncertain learned prediction.  A plain epsilon clip would give that
        # certain mission failure the same finite score as an ordinary low-
        # probability local outcome.  With several objectives this can make
        # sacrificing one target look better than protecting all of them.
        #
        # Every learned/clipped local log-survival utility lies in [-1, 0].
        # Giving a structural breach the score -(M+1) therefore makes it worse
        # than *any* all-positive M-target profile while keeping the payoff
        # finite for LP/DP solvers.  Zero-defender allocations remain legal;
        # they are discouraged by mission value rather than removed by a mask.
        utility[:, 0, 1:] = -(float(target_count) + 1.0)
    return utility


__all__ = [
    "FrozenStage2Payoff",
    "SUPPORTED_LOCAL_UTILITIES",
    "StyleCalibration",
    "build_local_payoff_tensor",
    "fit_train_style_calibration",
    "sha256_file",
    "survival_probability_to_utility",
    "utility_to_survival_probability",
]

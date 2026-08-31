"""Population-size curricula for one shared Stage-1 policy.

The stage frontier follows the simple small-to-large schedule used by
population curricula.  Within the unlocked frontier, sampling follows a
lightweight learning-progress score inspired by SPMARL/PLR, with an explicit
uniform-coverage floor so that easy scales cannot be forgotten.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Scale = Tuple[int, int]


def supported_scales(max_agents: int = 4, minimum_red_advantage: int = 1) -> List[Scale]:
    """Return registered ``(Red, Blue)`` HAD team sizes.

    HAD attack agents self-destruct after firing.  The Stage-1 protocol
    therefore preregisters only scenarios in which Red starts with a strict
    numerical advantage (``minimum_red_advantage=1``).  A caller may pass zero
    only for a deliberately separate ablation; :class:`HADStage1Adapter`
    itself still rejects non-superior Red rosters.
    """

    if max_agents < 1:
        raise ValueError("max_agents must be positive")
    if minimum_red_advantage < 0:
        raise ValueError("minimum_red_advantage must be non-negative")
    if minimum_red_advantage >= max_agents:
        raise ValueError("minimum_red_advantage leaves no valid HAD scale")
    return [
        (red, blue)
        for red in range(1, max_agents + 1)
        for blue in range(1, max_agents + 1)
        if red - blue >= minimum_red_advantage
    ]


@dataclass(frozen=True)
class CurriculumSnapshot:
    stage: int
    unlocked_scales: Tuple[Scale, ...]
    probabilities: Mapping[Scale, float]
    fast_td: Mapping[Scale, float]
    slow_td: Mapping[Scale, float]
    visits: Mapping[Scale, int]
    sample_visits: Mapping[Scale, int]


class LearningProgressCurriculum:
    """Stage-unlocked, learning-progress sampling over population sizes.

    A scale's score combines |fast TD EMA - slow TD EMA| and a count bonus.
    The score is not called an exact implementation of SPMARL; it is the small,
    auditable specialization needed for the finite 1--4 HAD scale grid.
    """

    def __init__(
        self,
        max_agents: int = 4,
        minimum_red_advantage: int = 1,
        episodes_per_stage: int = 2_000,
        uniform_coverage: float = 0.20,
        temperature: float = 0.25,
        fast_rate: float = 0.25,
        slow_rate: float = 0.025,
        count_bonus: float = 0.10,
    ):
        if episodes_per_stage < 1:
            raise ValueError("episodes_per_stage must be positive")
        if not 0.0 <= uniform_coverage <= 1.0:
            raise ValueError("uniform_coverage must be in [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.max_agents = max_agents
        self.minimum_red_advantage = minimum_red_advantage
        self.episodes_per_stage = episodes_per_stage
        self.uniform_coverage = uniform_coverage
        self.temperature = temperature
        self.fast_rate = fast_rate
        self.slow_rate = slow_rate
        self.count_bonus = count_bonus
        # With strict Red superiority there is no valid scale at stage 1.
        self.stage = 1 + minimum_red_advantage
        self.total_episodes = 0
        self.fast_td: Dict[Scale, float] = {}
        self.slow_td: Dict[Scale, float] = {}
        self.visits: Dict[Scale, int] = {}
        self.sample_visits: Dict[Scale, int] = {}

    @property
    def registered_scales(self) -> Tuple[Scale, ...]:
        return tuple(supported_scales(self.max_agents, self.minimum_red_advantage))

    @property
    def unlocked_scales(self) -> Tuple[Scale, ...]:
        # Stage r unlocks all supported scales with r Red agents.  Under the
        # default HAD constraint this yields 2v1 -> {3v1,3v2} -> ... while one
        # policy remains shared throughout.
        return tuple(scale for scale in self.registered_scales if scale[0] <= self.stage)

    def maybe_advance(self, total_episodes: Optional[int] = None) -> bool:
        if total_episodes is not None:
            self.total_episodes = int(total_episodes)
        first_stage = 1 + self.minimum_red_advantage
        target_stage = min(
            self.max_agents,
            first_stage + self.total_episodes // self.episodes_per_stage,
        )
        changed = target_stage > self.stage
        self.stage = max(self.stage, target_stage)
        return changed

    def update(self, scale: Scale, absolute_td_error: float) -> None:
        if scale not in self.registered_scales:
            raise ValueError(f"scale {scale} is outside the registered training domain")
        value = max(0.0, float(absolute_td_error))
        if scale not in self.fast_td:
            self.fast_td[scale] = value
            self.slow_td[scale] = value
        else:
            self.fast_td[scale] += self.fast_rate * (value - self.fast_td[scale])
            self.slow_td[scale] += self.slow_rate * (value - self.slow_td[scale])
        self.visits[scale] = self.visits.get(scale, 0) + 1

    def probabilities(self) -> Dict[Scale, float]:
        scales = self.unlocked_scales
        scores = []
        for scale in scales:
            progress = abs(self.fast_td.get(scale, 0.0) - self.slow_td.get(scale, 0.0))
            novelty = self.count_bonus / np.sqrt(
                1.0 + self.sample_visits.get(scale, 0)
            )
            scores.append(progress + novelty)
        logits = np.asarray(scores, dtype=np.float64) / self.temperature
        logits -= logits.max()
        prioritized = np.exp(logits)
        prioritized /= prioritized.sum()
        uniform = np.full(len(scales), 1.0 / len(scales), dtype=np.float64)
        mixed = (1.0 - self.uniform_coverage) * prioritized + self.uniform_coverage * uniform
        return {scale: float(probability) for scale, probability in zip(scales, mixed)}

    def sample(self, rng: Optional[np.random.Generator] = None) -> Scale:
        generator = rng if rng is not None else np.random.default_rng()
        probabilities = self.probabilities()
        scales = tuple(probabilities)
        # A probabilistic coverage floor does not guarantee that a short run
        # ever observes every newly unlocked roster.  Force exactly one first
        # visit before reverting to learning-progress sampling.
        unseen = [scale for scale in scales if self.sample_visits.get(scale, 0) == 0]
        if unseen:
            selected = unseen[0]
        else:
            index = int(
                generator.choice(
                    len(scales), p=np.asarray(list(probabilities.values()))
                )
            )
            selected = scales[index]
        self.record_sample(selected)
        return selected

    def record_sample(self, scale: Scale) -> None:
        """Record an externally selected scale (for fixed-scale baselines)."""

        if scale not in self.registered_scales:
            raise ValueError(f"scale {scale} is outside the registered training domain")
        self.sample_visits[scale] = self.sample_visits.get(scale, 0) + 1

    def record_episode(self) -> None:
        self.total_episodes += 1
        self.maybe_advance()

    def snapshot(self) -> CurriculumSnapshot:
        return CurriculumSnapshot(
            stage=self.stage,
            unlocked_scales=self.unlocked_scales,
            probabilities=self.probabilities(),
            fast_td=dict(self.fast_td),
            slow_td=dict(self.slow_td),
            visits=dict(self.visits),
            sample_visits=dict(self.sample_visits),
        )

    def state_dict(self) -> Dict[str, object]:
        def encode(mapping: Mapping[Scale, object]) -> Dict[str, object]:
            return {f"{red}v{blue}": value for (red, blue), value in mapping.items()}

        return {
            "stage": self.stage,
            "total_episodes": self.total_episodes,
            "fast_td": encode(self.fast_td),
            "slow_td": encode(self.slow_td),
            "visits": encode(self.visits),
            "sample_visits": encode(self.sample_visits),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        def decode(mapping: Mapping[str, object], value_type) -> Dict[Scale, object]:
            result = {}
            for key, value in mapping.items():
                red, blue = key.split("v")
                result[(int(red), int(blue))] = value_type(value)
            return result

        self.stage = int(state["stage"])
        self.total_episodes = int(state["total_episodes"])
        self.fast_td = decode(state.get("fast_td", {}), float)
        self.slow_td = decode(state.get("slow_td", {}), float)
        self.visits = decode(state.get("visits", {}), int)
        self.sample_visits = decode(state.get("sample_visits", {}), int)


class StepLearningProgressCurriculum:
    """Frozen 1M-step HAD curriculum from the round-01 protocol.

    The first three phases are deterministic small-to-all scale curricula.  In
    the fourth phase, probabilities are refreshed every ``update_steps`` from
    two non-overlapping windows of per-episode absolute TD errors.  The final
    phase returns to uniform sampling to consolidate every registered scale.
    """

    DEFAULT_BOUNDARIES = (100_000, 250_000, 400_000, 850_000)

    def __init__(
        self,
        scales: Optional[Sequence[Scale]] = None,
        boundaries: Sequence[int] = DEFAULT_BOUNDARIES,
        update_steps: int = 10_000,
        td_window: int = 50,
        uniform_floor: float = 0.30,
        max_scale_probability: float = 0.40,
    ) -> None:
        self.scales = tuple(scales or supported_scales(4, 1))
        if not self.scales:
            raise ValueError("curriculum needs at least one registered scale")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("curriculum scales must be unique")
        self.boundaries = tuple(int(value) for value in boundaries)
        if len(self.boundaries) != 4 or any(
            left >= right
            for left, right in zip(self.boundaries, self.boundaries[1:])
        ):
            raise ValueError("curriculum requires four increasing boundaries")
        if update_steps < 1 or td_window < 1:
            raise ValueError("update_steps and td_window must be positive")
        if not 0.0 <= uniform_floor <= 1.0:
            raise ValueError("uniform_floor must be in [0, 1]")
        if not 0.0 < max_scale_probability <= 1.0:
            raise ValueError("max_scale_probability must be in (0, 1]")
        if max_scale_probability * len(self.scales) < 1.0:
            raise ValueError("probability cap is infeasible for registered scales")
        self.update_steps = int(update_steps)
        self.td_window = int(td_window)
        self.uniform_floor = float(uniform_floor)
        self.max_scale_probability = float(max_scale_probability)
        self.td_history: Dict[Scale, Deque[float]] = {
            scale: deque(maxlen=2 * self.td_window) for scale in self.scales
        }
        self.sample_visits: Dict[Scale, int] = {scale: 0 for scale in self.scales}
        self._probability_bucket = -1
        self._cached_probabilities: Dict[Scale, float] = {}

    def phase(self, environment_steps: int) -> int:
        steps = int(environment_steps)
        for phase, boundary in enumerate(self.boundaries):
            if steps < boundary:
                return phase
        return 4

    def active_scales(self, environment_steps: int) -> Tuple[Scale, ...]:
        phase = self.phase(environment_steps)
        if phase == 0:
            preferred = ((2, 1),)
        elif phase == 1:
            preferred = ((2, 1), (3, 1), (3, 2))
        else:
            preferred = self.scales
        active = tuple(scale for scale in preferred if scale in self.scales)
        return active or (self.scales[0],)

    @staticmethod
    def _uniform(scales: Sequence[Scale]) -> Dict[Scale, float]:
        probability = 1.0 / len(scales)
        return {scale: probability for scale in scales}

    def _cap_and_redistribute(
        self, probabilities: Mapping[Scale, float]
    ) -> Dict[Scale, float]:
        result = {scale: float(value) for scale, value in probabilities.items()}
        # Equal redistribution is repeated because one redistribution can make
        # another scale cross the cap.
        for _ in range(len(result) + 1):
            over = {
                scale: value - self.max_scale_probability
                for scale, value in result.items()
                if value > self.max_scale_probability
            }
            if not over:
                break
            excess = sum(over.values())
            for scale in over:
                result[scale] = self.max_scale_probability
            receivers = [
                scale
                for scale, value in result.items()
                if value < self.max_scale_probability - 1e-12
            ]
            if not receivers:
                raise RuntimeError("could not redistribute curriculum probability")
            share = excess / len(receivers)
            for scale in receivers:
                result[scale] += share
        total = sum(result.values())
        return {scale: value / total for scale, value in result.items()}

    def _adaptive_probabilities(self) -> Dict[Scale, float]:
        if any(
            len(self.td_history[scale]) < 2 * self.td_window
            for scale in self.scales
        ):
            return self._uniform(self.scales)
        scores = []
        for scale in self.scales:
            values = np.asarray(self.td_history[scale], dtype=np.float64)
            old = float(values[: self.td_window].mean())
            new = float(values[self.td_window :].mean())
            score = abs(np.log(old + 1e-6) - np.log(new + 1e-6))
            scores.append(score)
        score_array = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(score_array).all() or score_array.sum() <= 1e-12:
            prioritized = np.full(len(self.scales), 1.0 / len(self.scales))
        else:
            prioritized = score_array / score_array.sum()
        uniform = np.full(len(self.scales), 1.0 / len(self.scales))
        mixed = (
            (1.0 - self.uniform_floor) * prioritized
            + self.uniform_floor * uniform
        )
        return self._cap_and_redistribute(
            {scale: value for scale, value in zip(self.scales, mixed)}
        )

    def probabilities(self, environment_steps: int) -> Dict[Scale, float]:
        phase = self.phase(environment_steps)
        active = self.active_scales(environment_steps)
        if phase != 3:
            return self._uniform(active)
        bucket = int(environment_steps) // self.update_steps
        if bucket != self._probability_bucket or not self._cached_probabilities:
            self._cached_probabilities = self._adaptive_probabilities()
            self._probability_bucket = bucket
        return dict(self._cached_probabilities)

    def sample(
        self, environment_steps: int, rng: Optional[np.random.Generator] = None
    ) -> Scale:
        generator = rng if rng is not None else np.random.default_rng()
        probabilities = self.probabilities(environment_steps)
        scales = tuple(probabilities)
        index = int(
            generator.choice(
                len(scales), p=np.asarray(list(probabilities.values()))
            )
        )
        selected = scales[index]
        self.sample_visits[selected] += 1
        return selected

    def record_td_samples(
        self, samples_by_scale: Mapping[Scale, Sequence[float]]
    ) -> None:
        for scale, values in samples_by_scale.items():
            if scale not in self.td_history:
                raise ValueError(f"scale {scale} is not registered")
            for value in values:
                finite = float(value)
                if np.isfinite(finite):
                    self.td_history[scale].append(max(0.0, finite))

    def state_dict(self) -> Dict[str, object]:
        encode = lambda scale: f"{scale[0]}v{scale[1]}"
        return {
            "scales": [list(scale) for scale in self.scales],
            "boundaries": list(self.boundaries),
            "update_steps": self.update_steps,
            "td_window": self.td_window,
            "uniform_floor": self.uniform_floor,
            "max_scale_probability": self.max_scale_probability,
            "td_history": {
                encode(scale): list(values)
                for scale, values in self.td_history.items()
            },
            "sample_visits": {
                encode(scale): value
                for scale, value in self.sample_visits.items()
            },
            "probability_bucket": self._probability_bucket,
            "cached_probabilities": {
                encode(scale): value
                for scale, value in self._cached_probabilities.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if tuple(tuple(value) for value in state["scales"]) != self.scales:
            raise ValueError("saved curriculum scales do not match")
        if tuple(int(value) for value in state["boundaries"]) != self.boundaries:
            raise ValueError("saved curriculum boundaries do not match")

        def decode(label: str) -> Scale:
            red, blue = label.split("v")
            return int(red), int(blue)

        self.td_history = {
            scale: deque(maxlen=2 * self.td_window) for scale in self.scales
        }
        for label, values in state.get("td_history", {}).items():
            self.td_history[decode(label)].extend(float(value) for value in values)
        self.sample_visits = {scale: 0 for scale in self.scales}
        for label, value in state.get("sample_visits", {}).items():
            self.sample_visits[decode(label)] = int(value)
        self._probability_bucket = int(state.get("probability_bucket", -1))
        self._cached_probabilities = {
            decode(label): float(value)
            for label, value in state.get("cached_probabilities", {}).items()
        }

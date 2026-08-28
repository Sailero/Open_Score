"""Population-size curricula for one shared Stage-1 policy.

The stage frontier follows the simple small-to-large schedule used by
population curricula.  Within the unlocked frontier, sampling follows a
lightweight learning-progress score inspired by SPMARL/PLR, with an explicit
uniform-coverage floor so that easy scales cannot be forgotten.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

Scale = Tuple[int, int]


def supported_scales(max_agents: int = 4, defender_not_outnumbered: bool = True) -> List[Scale]:
    """Return registered (defender, attacker) team sizes in deterministic order."""

    if max_agents < 1:
        raise ValueError("max_agents must be positive")
    return [
        (defenders, attackers)
        for defenders in range(1, max_agents + 1)
        for attackers in range(1, max_agents + 1)
        if not defender_not_outnumbered or defenders >= attackers
    ]


def all_evaluation_scales(max_agents: int = 4) -> List[Scale]:
    """Return the full square grid, including out-of-support stress tests."""

    return [
        (defenders, attackers)
        for defenders in range(1, max_agents + 1)
        for attackers in range(1, max_agents + 1)
    ]


@dataclass(frozen=True)
class CurriculumSnapshot:
    stage: int
    unlocked_scales: Tuple[Scale, ...]
    probabilities: Mapping[Scale, float]
    fast_td: Mapping[Scale, float]
    slow_td: Mapping[Scale, float]
    visits: Mapping[Scale, int]


class LearningProgressCurriculum:
    """Stage-unlocked, learning-progress sampling over population sizes.

    A scale's score combines |fast TD EMA - slow TD EMA| and a count bonus.
    The score is not called an exact implementation of SPMARL; it is the small,
    auditable specialization needed for the finite 1--4 HAD scale grid.
    """

    def __init__(
        self,
        max_agents: int = 4,
        defender_not_outnumbered: bool = True,
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
        self.defender_not_outnumbered = defender_not_outnumbered
        self.episodes_per_stage = episodes_per_stage
        self.uniform_coverage = uniform_coverage
        self.temperature = temperature
        self.fast_rate = fast_rate
        self.slow_rate = slow_rate
        self.count_bonus = count_bonus
        self.stage = 1
        self.total_episodes = 0
        self.fast_td: Dict[Scale, float] = {}
        self.slow_td: Dict[Scale, float] = {}
        self.visits: Dict[Scale, int] = {}

    @property
    def registered_scales(self) -> Tuple[Scale, ...]:
        return tuple(supported_scales(self.max_agents, self.defender_not_outnumbered))

    @property
    def unlocked_scales(self) -> Tuple[Scale, ...]:
        # Stage r unlocks all supported scales with r defenders.  This yields
        # 1v1 -> {2v1,2v2} -> ... while sharing one policy throughout.
        return tuple(scale for scale in self.registered_scales if scale[0] <= self.stage)

    def maybe_advance(self, total_episodes: Optional[int] = None) -> bool:
        if total_episodes is not None:
            self.total_episodes = int(total_episodes)
        target_stage = min(self.max_agents, 1 + self.total_episodes // self.episodes_per_stage)
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
            novelty = self.count_bonus / np.sqrt(1.0 + self.visits.get(scale, 0))
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
        index = int(generator.choice(len(scales), p=np.asarray(list(probabilities.values()))))
        return scales[index]

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

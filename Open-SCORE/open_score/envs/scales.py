"""Frozen cross-scale configurations and an independent episode sampler."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Scale:
    N_R: int
    N_B: int
    K: int

    def __post_init__(self):
        if not (1 <= self.N_R <= 40 and 1 <= self.N_B <= 40 and 1 <= self.K <= 6):
            raise ValueError("HAD scale must fit 40 red, 40 blue and 6 target slots")

    def as_dict(self):
        return {"N_R": int(self.N_R), "N_B": int(self.N_B), "K": int(self.K)}

    @property
    def name(self):
        return f"{self.N_R}v{self.N_B}_K{self.K}"


def as_scale(value):
    if isinstance(value, Scale):
        return value
    if isinstance(value, dict):
        return Scale(int(value.get("N_R", value.get("red"))),
                     int(value.get("N_B", value.get("blue"))),
                     int(value.get("K", value.get("targets"))))
    return Scale(*map(int, value))


TRAIN_POOL = tuple(Scale(n, n, k) for n in (4, 6, 8, 10) for k in (1, 2, 3))
VALIDATION_POOL = (Scale(4, 4, 2), Scale(6, 6, 2), Scale(8, 8, 2), Scale(10, 10, 3))
SCALE_POOLS = {
    "mixed_le10": TRAIN_POOL,
    "single_8v8": (Scale(8, 8, 2),),
    "validation": VALIDATION_POOL,
    # Equal-scale ladder only: both sides intercept by self-destruct, so a
    # red deficit is not a representation-transfer axis. Unseen N at 1:1 is.
    "extrapolation_agents": tuple(Scale(n, n, 2) for n in (10, 15, 20, 25, 30, 40)),
    "extrapolation_ratio": tuple(Scale(n, 2 * n, 2) for n in (2, 4, 8, 15, 20)),
    "extrapolation_targets": tuple(Scale(n, n, k) for n in (10, 15, 20, 30) for k in (4, 6)),
}
TEST_POOL = tuple(c for name, pool in SCALE_POOLS.items() if name.startswith("extrapolation") for c in pool)


class ScaleSampler:
    def __init__(self, seed=0, train_dist="mixed_le10"):
        self.rng = np.random.default_rng(int(seed))
        self.train_dist = train_dist
        self.pool = SCALE_POOLS[train_dist]

    def sample(self):
        if self.train_dist == "mixed_le10":
            n = int(self.rng.choice((4, 6, 8, 10)))
            return Scale(n, n, int(self.rng.choice((1, 2, 3))))
        return self.pool[int(self.rng.integers(len(self.pool)))]

    def state_dict(self):
        return self.rng.bit_generator.state

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state

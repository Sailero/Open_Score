"""Native HAD asset defence; no scenario-specific combat rules are added."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from had_env.core.make_env import HADEnv


@dataclass(frozen=True)
class Scenario:
    """Counts include all roles; unallocated team members are attackers.

    The original random reset and uniform showcase reset remain available.
    ``evaluate=True`` uses the original separated blue starting region.
    """

    red_count: int = 4
    blue_count: int = 4
    target_count: int = 2
    red_scouts: int = 0
    red_disturbers: int = 0
    blue_scouts: int = 0
    blue_disturbers: int = 0
    initialization: str = "random"
    evaluate: bool = False
    target_region: object = None
    task_mode: str = "survival"
    target_health: float | None = None
    spatial_dim: int = 2
    target_initialization: str = "random"
    target_positions: object = None

    def __post_init__(self):
        for name in ("red_count", "blue_count", "target_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("red_scouts", "red_disturbers", "blue_scouts", "blue_disturbers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.red_scouts + self.red_disturbers > self.red_count:
            raise ValueError("red role counts must not exceed red_count")
        if self.blue_scouts + self.blue_disturbers >= self.blue_count:
            raise ValueError("Blue must contain at least one attacker under native HAD termination rules")
        if self.initialization not in ("random", "uniform"):
            raise ValueError("initialization must be 'random' or 'uniform'")
        if self.target_initialization not in ("random", "fixed"):
            raise ValueError("target_initialization must be random or fixed")
        if self.task_mode not in ("survival", "damage"):
            raise ValueError("task_mode must be 'survival' or 'damage'")
        if type(self.spatial_dim) is not int or self.spatial_dim not in (2, 3):
            raise ValueError("spatial_dim must be 2 or 3")
        if self.target_health is not None:
            if isinstance(self.target_health, (bool, np.bool_)):
                raise ValueError("target_health must be a positive finite number")
            health = float(self.target_health)
            if not np.isfinite(health) or health <= 0:
                raise ValueError("target_health must be a positive finite number")
        if self.target_region is not None:
            bounds = np.asarray(self.target_region, dtype=float)
            if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
                raise ValueError("target_region must be finite [x/y/z][low/high] bounds")

    def make_world(self, seed=None):
        return HADEnv(
            self.red_count - self.red_scouts - self.red_disturbers,
            self.blue_count - self.blue_scouts - self.blue_disturbers,
            self.target_count,
            red_scout_n=self.red_scouts, red_disturb_n=self.red_disturbers,
            blue_scout_n=self.blue_scouts, blue_disturb_n=self.blue_disturbers,
            task_type="Normal Showcase" if self.initialization == "uniform" else "Training",
            target_region=self.target_region, seed=seed,
            task_mode=self.task_mode, target_health=self.target_health,
            spatial_dim=self.spatial_dim,
            target_initialization=self.target_initialization, target_positions=self.target_positions,
        )

    def reset_world(self, world, *, seed=None, options=None):
        return world.reset(seed=seed, evaluate=bool((options or {}).get("evaluate", self.evaluate)))

    def observation(self, world):
        """Return unchanged normalized relative rows, including dead entities."""
        return world.get_observation(is_relative_observation=True)

    def reward(self, world):
        """Return the original team RealReward and per-agent LatentReward."""
        return world.get_reward()

    def done(self, world):
        return int(world.is_terminal())

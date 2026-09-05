"""Explicit lower-observation adaptation for the overnight protocol.

The original v2 executor exposes every live Blue entity. The frozen controller
was trained on local Red-superior rosters with at most three Blue entities. This
module offers a declared perception change, not a change to physical opponents,
damage, terminal conditions, task horizon, or frozen controller weights.
"""
from __future__ import annotations

import numpy as np

from open_score.grouping.environment import KnownOpponentEnv
from open_score.grouping.frozen import FrozenExecutor


EXECUTOR_SCOPES = ("full", "nearest3", "nearest_support", "count_clip3")


class _FocusedObservationView:
    """Read-through adapter view: filter only the local actor observation."""

    def __init__(self, adapter, scope: str):
        self._adapter = adapter
        self._scope = scope

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def local_observation(self, side, target_id, red_ids, blue_ids, local_step=0):
        if self._scope in {"full", "count_clip3"} or side != "Red" or not red_ids:
            selected = tuple(blue_ids)
        else:
            reds = self._adapter.agent_states("Red")
            blues = self._adapter.agent_states("Blue")
            center = np.mean([reds[i]["position"] for i in red_ids], axis=0)
            cap = (3 if self._scope == "nearest3"
                   else min(3, max(1, len(red_ids) - 1)))
            live_blue = [i for i in blue_ids if blues[i]["alive"]]
            selected = tuple(sorted(live_blue, key=lambda i: (
                float(np.linalg.norm(blues[i]["position"] - center)), i))[:cap])
        observation = self._adapter.local_observation(
            side, target_id, red_ids, selected, local_step=local_step)
        if self._scope == "count_clip3" and side == "Red":
            # Position 8 is Stage1's log1p(live enemy count). Saturation leaves
            # every entity token visible; the original adapter remains exact.
            blue = self._adapter.agent_states("Blue")
            count = sum(blue[i]["alive"] for i in selected)
            observation["self_obs"][:, 8] = np.log1p(min(3, count))
        return observation


class FocusedExecutor(FrozenExecutor):
    """The same frozen LCL and ID-owned recurrent state, focused observation.

    ``nearest3`` uses the three live Blue entities closest to each Red group's
    centroid. ``nearest_support`` caps that number at group size minus one
    (minimum one for casualty singletons). Neither option makes the *physical*
    battle Red-superior, and both are experimental transfer adaptations.
    ``count_clip3`` retains every Blue entity token and saturates only the
    explicit enemy-count feature at the largest training count, three.
    """

    def __init__(self, stage1_path=None, device="cpu", *, scope="nearest3"):
        if scope not in EXECUTOR_SCOPES:
            raise ValueError(f"executor scope must be one of {EXECUTOR_SCOPES}")
        self.scope = scope
        super().__init__(stage1_path, device)

    def act(self, adapter, grouping):
        view = adapter if self.scope == "full" else _FocusedObservationView(adapter, self.scope)
        return super().act(view, grouping)


def make_env(red, blue=None, opponent="reactive", executor_scope="nearest3", **kwargs):
    """Build the original physical environment with an explicit executor scope."""
    if executor_scope not in EXECUTOR_SCOPES:
        raise ValueError(f"executor scope must be one of {EXECUTOR_SCOPES}")
    env = KnownOpponentEnv(red=red, blue=red if blue is None else blue,
                           opponent=opponent, **kwargs)
    env.executor = FocusedExecutor(kwargs.get("stage1_path"), kwargs.get("device", "cpu"),
                                  scope=executor_scope)
    return env


__all__ = ["EXECUTOR_SCOPES", "FocusedExecutor", "make_env"]

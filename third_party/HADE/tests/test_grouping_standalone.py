"""Standalone regressions recorded before removing the research packages."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from had_env.actions import ACCELERATION_PRIMITIVES
from had_env.grouping.domain import Group, Grouping
from had_env.grouping.environment import KnownOpponentEnv
from had_env.grouping.policies import RulePolicy
from had_env.grouping.rules import make_env


_REFERENCE = json.loads((Path(__file__).parent / "data" / "grouping_extraction_validation.json").read_text())


def _quantize(value):
    # Micrometre-scale rounding avoids asserting platform-level float noise.
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (tuple, list)):
        return [_quantize(item) for item in value]
    if isinstance(value, dict):
        return {key: _quantize(item) for key, item in value.items()}
    return value


@pytest.mark.parametrize("policy_name,roster,seed,expected", [
    (case["policy_name"], case["roster"], case["seed"], case["sha256"])
    for case in _REFERENCE["golden_cases"]
])
def test_complete_episode_matches_pre_extraction_simulation(policy_name, roster, seed, expected):
    """Reference includes each command, state, reward, native event and outcome.

    References were generated from the completed HAD workbench v1.0.0 code,
    before package extraction. This test needs no previous Git history.
    """
    env = make_env(*roster, seed=seed)
    policy = RulePolicy(policy_name, seed + 1)
    try:
        state = env.reset(seed)
        rows = [state.to_dict()]
        while not env.done:
            action = policy.act(state)
            state, reward, done, info = env.step(action)
            rows.append(dict(action=action.to_dict(), state=state.to_dict(),
                             reward=reward, done=done, info=info))
        encoded = json.dumps(_quantize(rows), sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == expected
    finally:
        env.close()


def test_default_grouping_executor_requires_no_checkpoint_and_has_no_size_cap():
    env = KnownOpponentEnv(red=8, blue=8, max_steps=1)
    try:
        state = env.reset(seed=31)
        grouping = Grouping((Group(state.ids("targets")[0], state.ids("red")),))
        _, _, done, info = env.step(grouping)
        assert type(env.executor).__name__ == "RuleExecutor"
        assert env.group_max_size is None
        assert done and info["delta"] == 1
    finally:
        env.close()


def test_grouping_snapshot_restores_exact_independent_future():
    parent, child = make_env(8, seed=8), make_env(8, seed=8)
    policy = RulePolicy("rule")
    try:
        state = parent.reset(seed=8)
        state, _, _, _ = parent.step(policy.act(state))
        child.reset(seed=8)
        child.restore(parent.snapshot())
        before = parent.state().to_dict()
        child_result = child.step(policy.act(child.state()))
        assert parent.state().to_dict() == before
        parent_result = parent.step(policy.act(parent.state()))
        assert parent_result[0].to_dict() == child_result[0].to_dict()
        assert parent_result[1:] == child_result[1:]
    finally:
        parent.close()
        child.close()


def test_primitive_zero_and_26_directions_have_stable_native_order():
    assert ACCELERATION_PRIMITIVES.shape == (27, 3)
    assert ACCELERATION_PRIMITIVES.dtype == np.float32
    np.testing.assert_array_equal(ACCELERATION_PRIMITIVES[0], np.zeros(3))
    np.testing.assert_allclose(np.linalg.norm(ACCELERATION_PRIMITIVES[1:], axis=1), 1.0)
    np.testing.assert_allclose(ACCELERATION_PRIMITIVES[1], -np.ones(3) / np.sqrt(3))
    np.testing.assert_allclose(ACCELERATION_PRIMITIVES[-1], np.ones(3) / np.sqrt(3))

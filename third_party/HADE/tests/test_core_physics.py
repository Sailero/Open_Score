"""Regression checks for the isolated HAD workbench physics protocol."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from had_env.core.make_env import HADEnv
from had_env.core.function.Function import attack_intensity_ratio, distance
from had_env.core.config import AvoidanceDistance
from had_env.core.version import CORE_VERSION


def test_headless_import_steps_and_close_never_import_pygame():
    # A fresh interpreter is necessary: rendering tests in this process may
    # legitimately have already imported pygame during test collection.
    project = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project)
    code = """
import importlib.abc
import sys
class NoRenderer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pygame' or fullname.startswith('pygame.') or fullname.startswith('had_env.core.render'):
            raise AssertionError('headless physics imported renderer: ' + fullname)
sys.meta_path.insert(0, NoRenderer())
from had_env.core.config import AeroPoint
from had_env.core import HADEnv
import numpy as np
env = HADEnv(2, 1, 1)
env.reset(seed=4, evaluate=True)
env.step_physics(np.zeros((3, 3)))
assert len(env.step(np.zeros((3, 3)))) == 6
env.close()
env.close()
assert env.render() is False
assert not any(name == 'pygame' or name.startswith('pygame.') for name in sys.modules)
print('headless-ok')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=project,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "headless-ok"


def physical_state(env):
    return np.asarray([entity.position + list(entity.velocity) + [float(entity.Health)]
                       for entity in env.world])


def set_pose(agent, position, velocity=(20., 0., 0.)):
    agent.position = list(position)
    agent.velocity = list(velocity)


def test_attack_boundary_scalar_vector_agree_and_remain_finite():
    values = [499.999, 500., 500.001]
    expected = [attack_intensity_ratio(value) for value in values]
    assert expected == [1, 0, 0]
    np.testing.assert_array_equal(attack_intensity_ratio(values), expected)
    np.testing.assert_array_equal(attack_intensity_ratio(np.asarray(values)), expected)


def test_exact_boundary_in_a_physical_step_is_finite():
    env = HADEnv(1, 1, 1)
    env.reset(seed=4)
    set_pose(env.red_agents[0], [-1500., 0., 100.])
    set_pose(env.blue_agents[0], [500., 0., 100.])
    set_pose(env.targets[0], [0., 0., 100.], [0., 0., 0.])
    env.step_physics(np.zeros((2, 3)))
    assert np.isfinite(physical_state(env)).all()
    assert env.targets[0].Health == 1.2


def test_heterogeneous_step_preserves_each_roles_existing_reward():
    env = HADEnv(1, 1, 1, red_scout_n=1, red_disturb_n=1,
                 blue_scout_n=1, blue_disturb_n=1)
    env.reset(seed=7, evaluate=True)
    result = env.step(np.zeros((6, 3)))
    rewards = result[3]
    assert len(result) == 6
    for side, agents in (("Red", env.red_agents), ("Blue", env.blue_agents)):
        assert np.asarray(rewards["LatentReward"][side]).shape == (3, 4)
        for agent, reward in zip(agents, rewards["LatentReward"][side]):
            if agent.Health > 0 and agent.Type != "Attack":
                assert reward[:3] == [agent.get_reward(env.world), 0., 0.]


@pytest.mark.parametrize("task_type", ["Training", "Normal Showcase"])
def test_reset_clears_actions_events_and_repairs_first_step_collision(task_type):
    env = HADEnv(2, 1, 1, task_type=task_type)
    env.record_events = True
    env.reset(seed=4)
    for agent in env.agents:
        agent.Health = 0
        agent.IsFire = True
        agent.acceleration = [1., 2., 3.]
    env.targets[0].Health = 0
    env.update_alive_agents()
    env.physics_step_count = 17
    env.last_physics_events = [{"kind": "old"}]
    env.reset(seed=4)
    assert env.alive_agents == env.agents
    assert env.alive_targets == env.targets
    assert env.physics_step_count == 0 and env.last_physics_events == []
    assert all(agent.acceleration == [0., 0., 0.] and not agent.IsFire for agent in env.agents)
    for agent in env.red_agents:
        set_pose(agent, [-500., 0., 100.])
    set_pose(env.blue_agents[0], [2000., 0., 100.])
    env.step_physics(np.zeros((3, 3)))
    assert [agent.Health for agent in env.agents] == [0, 0, 1]
    collisions = [event for event in env.last_physics_events if event["kind"] == "collision"]
    assert len(collisions) == 2
    assert {event["target_id"] for event in collisions} == {0, 1}
    assert all(event["step"] == 1 and event["damage"] == 1 for event in collisions)


@pytest.mark.parametrize("invalid", [
    [], [[0, 0, 0]], np.zeros((3, 3)), np.zeros((2, 2)),
    [[0, 0], [0, 0, 0]], [[float("nan"), 0, 0], [0, 0, 0]],
    [[float("inf"), 0, 0], [0, 0, 0]], [["1", "0", "0"], ["0", "0", "0"]],
    [[1j, 0, 0], [0, 0, 0]],
])
def test_invalid_action_rejected_before_any_physical_mutation(invalid):
    env = HADEnv(1, 1, 1)
    env.reset(seed=4)
    env.record_events = True
    before = physical_state(env).copy()
    with pytest.raises(ValueError):
        env.step_physics(invalid)
    np.testing.assert_array_equal(before, physical_state(env))
    assert env.physics_step_count == 0
    assert env.last_physics_events == []
    assert all(agent.acceleration == [0., 0., 0.] for agent in env.agents)


def legacy_step_order(env, actions):
    """Frozen pre-workbench orchestration, independent of World.step."""
    for i in range(len(env.alive_agents)):
        for j in range(len(env.alive_agents)):
            if i > j and distance(env.alive_agents[i].position, env.alive_agents[j].position) <= AvoidanceDistance:
                env.alive_agents[i].Health = 0
                env.alive_agents[j].Health = 0
    for agent, action in zip(env.agents, actions):
        agent.set_flying_action((np.array(action) * agent.aMax).tolist())
        agent.set_function_action(agent.choose_function_ruled_action(env.world))
    old_world = copy.deepcopy(env.world)
    for agent in env.world:
        agent.update_status(old_world)
    env.update_alive_agents()


@pytest.mark.parametrize("seed", [4, 31, 8123])
def test_optional_recording_preserves_legacy_order_and_trajectory(seed):
    env = HADEnv(4, 4, 2)
    env.reset(seed=seed, evaluate=True)
    reference, silent = copy.deepcopy(env), copy.deepcopy(env)
    env.record_events = True
    rng = np.random.default_rng(seed)
    for _ in range(30):
        actions = rng.uniform(-1, 1, (8, 3)).astype(np.float32)
        legacy_step_order(reference, actions)
        env.step_physics(actions)
        silent.step_physics(actions)
        np.testing.assert_array_equal(physical_state(env), physical_state(reference))
        np.testing.assert_array_equal(physical_state(env), physical_state(silent))
        assert env.is_terminal() == reference.is_terminal()
        assert [a.IsFire for a in env.agents] == [a.IsFire for a in reference.agents]
        assert silent.last_physics_events == []
        json.dumps(env.last_physics_events, allow_nan=False)


def test_damage_events_explain_snapshot_distance_and_firing_self_destruction():
    env = HADEnv(1, 1, 1)
    env.reset(seed=4)
    env.record_events = True
    set_pose(env.red_agents[0], [0., 0., 100.])
    set_pose(env.blue_agents[0], [100., 0., 100.])
    set_pose(env.targets[0], [-2000., 0., 100.], [0., 0., 0.])
    env.step_physics(np.zeros((2, 3)))
    events = env.last_physics_events
    hit = next(event for event in events if event["kind"] == "attack_damage")
    # Blue moves to 120 before taking damage from the Red snapshot at zero.
    assert (hit["source_id"], hit["target_id"], hit["distance"]) == (0, 1, 120.)
    assert hit["source_position"] == [0., 0., 100.]
    assert hit["target_position"] == [120., 0., 100.]
    assert (hit["health_before"], hit["health_after"], hit["damage"]) == (1., 0., 10.)
    assert hit["damage_scope"] == "source_unclipped" and hit["simultaneous"]
    assert any(e["kind"] == "fire" and e["source_id"] == 0 for e in events)
    assert any(e["kind"] == "self_destruct" and e["target_id"] == 0 for e in events)
    assert env.core_version == CORE_VERSION
    assert env.get_info()["core_version"] == CORE_VERSION

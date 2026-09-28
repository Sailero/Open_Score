"""Headless rendering lifecycle and physics isolation checks."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest
pygame = pytest.importorskip("pygame")

from had_env.core.make_env import HADEnv
from had_env.core.config import ScreenLength, ScreenWidth, ScreenHeight


@pytest.fixture
def env():
    instance = HADEnv(2, 1, 1)
    instance.reset(seed=4, evaluate=True)
    yield instance
    instance.close()
    pygame.quit()


def test_rgb_is_cached_stable_and_requires_no_window(env, monkeypatch):
    loads = []
    load = pygame.image.load
    def counted(path):
        loads.append(path)
        return load(path)
    monkeypatch.setattr(pygame.image, "load", counted)
    first = env.render_rgb_array()
    player = env._rgb_player
    second = env.render_rgb_array()
    assert env._rgb_player is player
    assert len(loads) == 2
    assert first.shape == (int(ScreenWidth + ScreenHeight), int(ScreenLength), 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)
    assert pygame.display.get_surface() is None


def test_window_reuses_resources_and_terminal_frame_never_busy_waits(env, monkeypatch):
    calls = []
    set_mode = pygame.display.set_mode
    def counted(*args, **kwargs):
        calls.append(args)
        return set_mode(*args, **kwargs)
    monkeypatch.setattr(pygame.display, "set_mode", counted)
    def forbidden_timer():
        raise AssertionError("Rendering must not busy-wait on result timers")
    monkeypatch.setattr(pygame.time, "get_ticks", forbidden_timer)
    env.blue_agents[0].Health = 0
    assert env.render()
    player = env._display_player
    assert env.render()
    assert env._display_player is player and len(calls) == 1
    assert env.physics_step_count == 0


@pytest.mark.parametrize("event", [pygame.event.Event(pygame.QUIT),
                                  pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
def test_window_quit_returns_control_without_reopening_or_ending_python(env, event):
    assert env.render()
    pygame.event.post(event)
    assert env.render() is False
    assert pygame.display.get_surface() is None
    assert env.render() is False
    env.close()
    env.close()
    # Offscreen export remains available even after interactive close.
    assert env.render_rgb_array().shape[2] == 3
    env.reset(seed=4)
    assert env.render()


def test_render_does_not_mutate_physics_or_random_generators(env):
    before = [(list(a.position), list(a.velocity), a.Health) for a in env.world]
    rng_before = repr(env.np_random.bit_generator.state)
    for _ in range(3):
        env.render_rgb_array()
        env.render()
    after = [(list(a.position), list(a.velocity), a.Health) for a in env.world]
    assert before == after
    assert rng_before == repr(env.np_random.bit_generator.state)
    assert env.physics_step_count == 0

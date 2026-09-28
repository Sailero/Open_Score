"""The standalone research interface remains useful without an ML runtime."""
import copy
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from had_env.workbench.agent_session import FlightScenarioSpec, FlightSession
from had_env.workbench.rng import random_state


def _fake_torch(monkeypatch):
    class State:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

    state = dict(cpu=17, cuda=29, initialized=False)

    def initialize():
        state["initialized"] = True

    cuda = SimpleNamespace(
        is_initialized=lambda: state["initialized"], _lazy_init=initialize,
        get_rng_state_all=lambda: [State(state["cuda"])],
        set_rng_state_all=lambda values: state.update(cuda=values[0].value),
        manual_seed_all=lambda seed: state.update(cuda=seed))
    torch = SimpleNamespace(
        cuda=cuda, get_rng_state=lambda: State(state["cpu"]),
        set_rng_state=lambda value: state.update(cpu=value.value),
        default_generator=SimpleNamespace(manual_seed=lambda seed: state.update(cpu=seed)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch, state


def test_cuda_initialized_after_adapter_is_rejected_before_policy_or_rng_mutation(monkeypatch):
    from had_env.workbench.session import PolicyAdapter
    torch, state = _fake_torch(monkeypatch)
    calls = []
    adapter = PolicyAdapter(lambda observation: calls.append(observation), seed=91)
    torch.cuda._lazy_init()
    before = state.copy()
    python_before, numpy_before = random.getstate(), pickle.dumps(np.random.get_state())
    with pytest.raises(RuntimeError, match="CUDA was initialized after this PolicyAdapter"):
        adapter.act(SimpleNamespace(state=lambda: {}), planning_seed=0)
    assert not calls and state == before
    assert random.getstate() == python_before
    assert pickle.dumps(np.random.get_state()) == numpy_before


@pytest.mark.parametrize("phase", ["reset", "act"])
def test_policy_cannot_lazily_initialize_cuda(monkeypatch, phase):
    from had_env.workbench.session import PolicyAdapter
    torch, state = _fake_torch(monkeypatch)
    original_initializer = torch.cuda._lazy_init

    class Policy:
        def reset(self):
            if phase == "reset":
                torch.cuda._lazy_init()

        def act(self, observation):
            torch.cuda._lazy_init()

    before = state.copy()
    with pytest.raises(RuntimeError, match="Initialize CUDA before constructing PolicyAdapter"):
        adapter = PolicyAdapter(Policy(), seed=91)
        adapter.act(SimpleNamespace(state=lambda: {}), planning_seed=0)
    assert state == before
    assert torch.cuda._lazy_init is original_initializer


def test_numpy_policies_have_private_memory_and_reproducible_branches():
    class Policy:
        def __init__(self):
            self.calls = 0

        def act(self, observation):
            self.calls += 1
            return [(random.randrange(27) + int(np.random.randint(27))) % 27
                    for _ in observation["agent_ids"]]

    policy = Policy()
    ambient = random_state()
    session = FlightSession(FlightScenarioSpec(red_attackers=2, blue_attackers=2, max_steps=4),
                            red_policy=policy, blue_policy=copy.deepcopy(policy))
    child = session.branch()
    try:
        assert session.run().frames == child.run().frames
        assert policy.calls == 0
        current = random_state()
        assert ambient["python"] == current["python"]
        assert pickle.dumps(ambient["numpy"]) == pickle.dumps(current["numpy"])
    finally:
        session.close()
        child.close()


def test_headless_imports_and_lazy_torch_contract_in_fresh_process():
    root = Path(__file__).resolve().parents[1]
    code = '''
import sys, random, pickle
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'PySide6', 'pygame'):
            raise AssertionError('Headless HAD unexpectedly requested ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import had_env.workbench
from had_env.workbench.session import SimulationSession, PolicyAdapter
from had_env.workbench.agent_session import FlightSession, FlightScenarioSpec
from had_env.workbench.rng import random_state
with SimulationSession(record=False) as session:
    session.step()
    class LazyTorch:
        def act(self, observation):
            random.random()
            import torch
    adapter = PolicyAdapter(LazyTorch(), seed=91)
    ambient = pickle.dumps(random_state())
    try:
        adapter.act(session.env, planning_seed=8)
        raise AssertionError('Lazy Torch import should fail with an explicit contract')
    except RuntimeError as error:
        assert 'Import PyTorch before constructing' in str(error)
    assert pickle.dumps(random_state()) == ambient
flight = FlightSession(FlightScenarioSpec(red_attackers=1, blue_attackers=1, max_steps=2))
try:
    assert flight.run().max_step == 2
finally:
    flight.close()
assert not any(name in sys.modules for name in ('torch', 'pygame', 'PySide6'))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root)),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

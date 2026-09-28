import pickle
import random

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from had_env.workbench.rng import random_state, restore_random_state, seed_everything
from had_env.workbench.session import SimulationSession


def test_cpu_session_never_requests_or_seeds_a_cuda_context(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU rule simulation must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    with SimulationSession() as session:
        session.step()
        with session.branch(218) as branch:
            branch.step()


def test_initialized_cuda_and_cpu_streams_are_restored_without_consumption(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    device_state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    calls = []
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [v.clone() for v in device_state])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: calls.append([v.clone() for v in state]))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    ambient = random_state()
    try:
        expected_numpy = pickle.dumps(np.random.get_state())
        expected_python = random.getstate()
        expected_torch = torch.get_rng_state().clone()
        seed_everything(992)
        np.random.rand(3)
        torch.rand(4)
        random.random()
        restore_random_state(ambient)
        assert pickle.dumps(np.random.get_state()) == expected_numpy
        assert random.getstate() == expected_python
        assert torch.equal(torch.get_rng_state(), expected_torch)
        assert len(calls) == 1 and torch.equal(calls[0][0], device_state[0])
    finally:
        restore_random_state(ambient)


def test_restoration_never_implicitly_initializes_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    state = random_state()
    state["cuda"] = [torch.tensor([1], dtype=torch.uint8)]
    with pytest.raises(ValueError, match="already initialized"):
        restore_random_state(state)

"""Small real-environment training checks for the overnight worker contract."""
import copy
import json
import sys

import numpy as np
import pytest
import torch

from open_score.grouping.frozen import DEFAULT_STAGE1_PATH, load_stage1, file_sha256
from open_score.grouping.storage import random_state
from open_score.overnight.config import configuration
from open_score.overnight.environment import make_env
from open_score.overnight.training import Trainer


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_config(route):
    config = configuration(route, seed=862001, smoke=True, device="cpu")
    config.update(model=dict(hidden_dim=32, heads=4, layers=1),
                  train_scales=[4], validation_scales=[4], eval_scales=[4],
                  environments=1, rollout_events=8, batch_size=4, epochs=2,
                  replay_capacity=32, replay_warmup=4, teacher_capacity=16,
                  teacher_candidates=2, teacher_horizon=5, validation_episodes=1,
                  checkpoint_seconds=1000, validation_interval=1000)
    return config


def assert_random_state_equal(first, second):
    assert first["python"] == second["python"]
    assert first["numpy"][0] == second["numpy"][0]
    np.testing.assert_array_equal(first["numpy"][1], second["numpy"][1])
    assert first["numpy"][2:] == second["numpy"][2:]
    torch.testing.assert_close(first["torch"].cpu(), second["torch"].cpu(), rtol=0, atol=0)
    assert len(first["cuda"]) == len(second["cuda"])
    for x, y in zip(first["cuda"], second["cuda"]):
        torch.testing.assert_close(x.cpu(), y.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("route", ["ppo_structured", "ppo_teacher", "candidate_q"])
def test_real_route_updates_checkpoint_and_resume_adds_work(route, tmp_path):
    config = tiny_config(route)
    frozen = load_stage1()
    frozen_before = {name: value.detach().clone() for name, value in frozen.state_dict().items()}
    asset_before = file_sha256(DEFAULT_STAGE1_PATH)
    trainer = Trainer(config, tmp_path, seconds=120, steps=96, checkpoint_every=35)
    initial = {name: value.detach().clone() for name, value in trainer.network.state_dict().items()}
    first = trainer.run()
    assert 96 <= first["physical_steps"] < 110
    assert first["updates"] > 0 and first["optimizer_steps"] > 0
    assert any(not torch.equal(value, initial[name]) for name, value in trainer.network.state_dict().items())
    checkpoint = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["counters"]["physical_steps"] == first["physical_steps"]
    assert checkpoint["counters"]["updates"] == first["updates"]
    first_episodes = (tmp_path / "episodes.jsonl").read_bytes()
    replay_count, teacher_count = len(trainer.replay), len(trainer.teacher)
    resumed = Trainer(config, tmp_path, seconds=120, steps=192, resume=True, checkpoint_every=35)
    assert resumed.counts["physical_steps"] == first["physical_steps"]
    assert resumed.counts["next_episode"] == first["next_episode"]
    assert len(resumed.replay) == replay_count and len(resumed.teacher) == teacher_count
    assert_random_state_equal(random_state(), checkpoint["random_state"])
    for name, value in resumed.network.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)
    second = resumed.run()
    assert 192 <= second["physical_steps"] < 210
    assert second["updates"] > first["updates"]
    assert second["training_seconds"] >= first["training_seconds"]
    assert (tmp_path / "episodes.jsonl").read_bytes().startswith(first_episodes)
    assert len(resumed.replay) <= config["replay_capacity"]
    assert len(resumed.teacher) <= config["teacher_capacity"]
    if route == "candidate_q":
        assert resumed.replay
    if route == "ppo_teacher":
        assert second["teacher_simulation_steps"] > 0
    assert file_sha256(DEFAULT_STAGE1_PATH) == asset_before
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    for name, value in frozen.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)


def test_teacher_branches_use_independent_rng_then_restore_real_transition(tmp_path, monkeypatch):
    from open_score.overnight import training as module
    config = tiny_config("ppo_teacher")
    records, branch_seeds = [], []

    def observed_make_env(*args, **kwargs):
        env = make_env(*args, **kwargs)
        old_step, old_rng = env.step, env.set_rng

        def step(action):
            result = old_step(action)
            records.append((action, result))
            return result

        def set_rng(seed):
            branch_seeds.append(seed)
            return old_rng(seed)

        env.step, env.set_rng = step, set_rng
        return env

    monkeypatch.setattr(module, "make_env", observed_make_env)
    # One deployed macro transition, plus counterfactual candidate branches.
    trainer = Trainer(config, tmp_path, seconds=60, steps=1)
    trainer.teacher_stage()
    assert trainer.counts["upper_events"] == 1
    assert trainer.counts["teacher_simulation_steps"] > 0
    assert len(records) >= config["teacher_candidates"] + 1
    assert branch_seeds.count(config["seed"] + 9_000_000) == config["teacher_candidates"]
    assert config["seed"] + 100_000 in branch_seeds
    deployed_action, deployed_result = records[-1]
    reference = make_env(4, opponent=config["opponent"], executor_scope=config["executor_scope"])
    reference.reset(seed=config["seed"] + 100_000)
    # Branch trials must not consume/replace the actual future opponent RNG.
    assert reference.step(deployed_action) == deployed_result


def test_validation_restores_sampling_rng_and_keeps_replay(tmp_path):
    trainer = Trainer(tiny_config("candidate_q"), tmp_path, seconds=120, steps=32)
    trainer.run()
    expected = random_state()
    counters = copy.deepcopy(trainer.counts)
    replay = list(trainer.replay)
    trainer.validate()
    assert_random_state_equal(random_state(), expected)
    for key in ("physical_steps", "upper_events", "episodes", "wins", "updates"):
        assert trainer.counts[key] == counters[key]
    assert list(trainer.replay) == replay


def test_resume_rejects_configuration_or_frozen_asset_changes(tmp_path, monkeypatch):
    from open_score.overnight import training as module
    config = tiny_config("ppo_structured")
    trainer = Trainer(config, tmp_path, seconds=120, steps=1)
    trainer.save()
    changed = copy.deepcopy(config)
    changed["learning_rate"] *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        Trainer(changed, tmp_path, seconds=120, steps=1, resume=True)
    evidence = copy.deepcopy(trainer.evidence)
    evidence["assets"]["lcl"]["sha256"] = "different"
    monkeypatch.setattr(module, "provenance", lambda: evidence)
    with pytest.raises(ValueError, match="source or frozen assets changed"):
        Trainer(config, tmp_path, seconds=120, steps=1, resume=True)


def test_cli_rejects_changed_budget_before_reusing_completed_training(tmp_path, monkeypatch):
    from open_score.grouping.storage import fingerprint
    from open_score.overnight import cli
    config = configuration("ppo_structured", seed=20260906, smoke=True, device="cpu")
    request = dict(train_seconds=25200.0, eval_seconds=2700.0, steps=96,
                   eval_episodes=100, checkpoint_every=25000, config_hash=fingerprint(config))
    request_path = tmp_path / "worker_request.json"
    request_path.write_text(json.dumps(request), encoding="utf8")
    result_path = tmp_path / "training_result.json"
    result_path.write_text('{"physical_steps": 96}', encoding="utf8")
    before = result_path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["worker", "worker", "--route", "ppo_structured",
        "--output", str(tmp_path), "--steps", "192", "--device", "cpu", "--smoke", "--resume"])
    # This entry-point setting is once-per-process; test worker dispatch in the
    # existing pytest process without attempting to reset Torch's global pool.
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *_: None)
    with pytest.raises(ValueError, match="budgets or configuration changed"):
        cli.main()
    assert result_path.read_bytes() == before
    assert not (tmp_path / "latest.pt").exists()


def test_manifest_records_actual_executor_scope(tmp_path):
    config = tiny_config("ppo_structured")
    config["executor_scope"] = "full"
    Trainer(config, tmp_path, seconds=120, steps=1)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf8"))
    assert "full adapter" in manifest["protocol"]
    assert "count_clip3" not in manifest["protocol"]


@pytest.mark.parametrize("route,update_name", [
    ("ppo_structured", "ppo_update"),
    ("candidate_q", "double_q_update"),
    ("ppo_teacher", "imitation_update"),
])
def test_failed_update_keeps_initial_valid_checkpoint(route, update_name, tmp_path, monkeypatch):
    from open_score.overnight import training as module
    config = tiny_config(route)
    if route == "ppo_teacher":
        config["batch_size"] = 1
        # Give the two actual simulation branches distinct diagnostic labels,
        # ensuring the first teacher update reaches the injected failure.
        scores = iter(range(100))
        monkeypatch.setattr(module, "potential", lambda *_args, **_kwargs: float(next(scores)))
    trainer = Trainer(config, tmp_path, seconds=120, steps=96, checkpoint_every=10000)
    latest = tmp_path / "latest.pt"
    before = file_sha256(latest)
    initial = torch.load(latest, map_location="cpu", weights_only=False)
    assert initial["counters"]["physical_steps"] == 0

    def fail(network, *_args, **_kwargs):
        # Simulate an optimizer/library failure after mutating live weights.
        with torch.no_grad():
            next(network.parameters()).fill_(float("nan"))
        raise FloatingPointError("injected partly failed update")

    monkeypatch.setattr(module, update_name, fail)
    with pytest.raises(FloatingPointError, match="injected partly failed update"):
        trainer.run()
    assert file_sha256(latest) == before
    recovered = torch.load(latest, map_location="cpu", weights_only=False)
    assert all(torch.isfinite(value).all() for value in recovered["model"].values())
    for name, value in recovered["model"].items():
        torch.testing.assert_close(value, initial["model"][name], rtol=0, atol=0)


def test_resume_archives_unsaved_and_partial_log_suffixes(tmp_path):
    config = tiny_config("ppo_structured")
    trainer = Trainer(config, tmp_path, seconds=120, steps=96)
    trainer.counts.update(physical_steps=40, updates=2, optimizer_steps=7, episodes=1)
    trainer.save()
    records = {
        "episodes.jsonl": ({"episode": 1, "physical_steps": 35},
                           {"episode": 2, "physical_steps": 40}),
        "training.jsonl": ({"updates": 2, "optimizer_steps": 7, "physical_steps": 40},
                           {"updates": 3, "optimizer_steps": 8, "physical_steps": 40}),
        "validation_history.jsonl": ({"steps": 40, "summary": {"complete": True}},
                                     {"steps": 45, "summary": {"complete": True}}),
    }
    expected = {}
    for name, (committed, future) in records.items():
        good = json.dumps(committed).encode() + b"\n"
        suffix = json.dumps(future).encode() + b'\n{"incomplete":'
        (tmp_path / name).write_bytes(good + suffix)
        expected[name] = (good, suffix)
    resumed = Trainer(config, tmp_path, seconds=120, steps=96, resume=True)
    assert resumed.counts["physical_steps"] == 40
    archives = list((tmp_path / "resume_discarded").iterdir())
    assert len(archives) == 3
    for name, (good, suffix) in expected.items():
        assert (tmp_path / name).read_bytes() == good
        matches = list((tmp_path / "resume_discarded").glob(name + ".*.bin"))
        assert len(matches) == 1 and matches[0].read_bytes() == suffix
    # Repeated resume is idempotent; it cannot append duplicate archives/rows.
    Trainer(config, tmp_path, seconds=120, steps=96, resume=True)
    assert set((tmp_path / "resume_discarded").iterdir()) == set(archives)


def test_resume_rejects_malformed_interior_log_without_erasing_it(tmp_path):
    config = tiny_config("ppo_structured")
    trainer = Trainer(config, tmp_path, seconds=120, steps=96)
    trainer.counts.update(physical_steps=40, updates=2, episodes=1)
    trainer.save()
    path = tmp_path / "training.jsonl"
    broken = b'{"updates": 1, "physical_steps": 20}\n{"broken":\n{"updates": 2, "physical_steps": 40}\n'
    path.write_bytes(broken)
    checkpoint_hash = file_sha256(tmp_path / "latest.pt")
    with pytest.raises(ValueError, match="Malformed interior log row"):
        Trainer(config, tmp_path, seconds=120, steps=96, resume=True)
    assert path.read_bytes() == broken
    assert file_sha256(tmp_path / "latest.pt") == checkpoint_hash
    assert not (tmp_path / "resume_discarded").exists()


def test_update_log_separates_cumulative_and_current_optimizer_steps(tmp_path, monkeypatch):
    trainer = Trainer(tiny_config("ppo_structured"), tmp_path, seconds=120, steps=96)
    trainer.counts.update(updates=3, optimizer_steps=11)
    trainer.last_metrics = {"optimizer_steps": 2, "loss": .25}
    monkeypatch.setattr(trainer, "elapsed", lambda: 12.5)
    trainer.record_update()
    row = json.loads((tmp_path / "training.jsonl").read_text(encoding="utf8"))
    assert row["updates"] == 3
    assert row["optimizer_steps"] == 11
    assert row["optimizer_steps_this_update"] == 2
    assert row["training_seconds"] == trainer.counts["training_seconds"] == 12.5
    assert row["elapsed_seconds"] >= row["training_seconds"]

"""Small real-world PPO/resume checks and macro-time evaluation contracts."""
import copy
import json

import numpy as np
import pytest
import torch

from open_score.grouping import training
from open_score.grouping.domain import Group, Grouping
from open_score.grouping.environment import KnownOpponentEnv
from open_score.grouping.evaluation import evaluate, grouping_changes, rollout_episode, summarize
from open_score.grouping.frozen import load_stage1
from open_score.grouping.storage import read_jsonl, seed_everything, sha256


@pytest.fixture(autouse=True)
def small_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def training_config():
    return {"method": "selective", "seed": 20260905, "opponent": "reactive",
            "train_scales": [4], "max_steps": 50, "command_interval": 5,
            "device": "cpu", "executor_device": "cpu", "torch_threads": 1,
            "model": {}, "ppo": {"rollout_events": 16, "epochs": 1, "batch_size": 8,
                                  "gamma": 1.0, "gae_lambda": 0.95}}


def test_smdp_gae_uses_physical_duration_and_event_lambda():
    advantage, target = training.smdp_gae(
        [0.2, 1.0], [0.3, 0.4], [0.4, 123.0], [3, 2], [False, True], gamma=0.9, lam=0.95)
    np.testing.assert_allclose(advantage, [0.60713, 0.6], rtol=0, atol=1e-12)
    np.testing.assert_allclose(target, [0.90713, 1.0], rtol=0, atol=1e-12)
    # A nonterminal rollout boundary keeps one value bootstrap, then ends GAE.
    advantage, target = training.smdp_gae([0.2], [0.3], [0.4], [3], [False], gamma=0.9, lam=0.95)
    assert advantage[0] == pytest.approx(0.2 + 0.9**3 * 0.4 - 0.3)
    assert target[0] == pytest.approx(0.2 + 0.9**3 * 0.4)
    # An episode terminal must also prevent advantage from the next episode leaking back.
    advantage, _ = training.smdp_gae([1.0, 50.0], [0.0, 0.0], [99.0, 99.0], [2, 1], [True, True])
    assert advantage[0] == 1.0


def test_native_fifty_step_horizon_is_terminal_without_bootstrap():
    env = KnownOpponentEnv(red=4, blue=4, max_steps=50, command_interval=5)
    try:
        env.reset(seed=17)
        # Controlled pre-horizon snapshot: physical entities are intact and far
        # from targets, so the next real physical step tests native expiry.
        env.adapter.step_count = 49
        state = env.state()
        next_state, reward, done, info = env.step(Grouping((), state.ids("red")))
        assert next_state.step == 50 and info["delta"] == 1
        assert done and info["terminated"] and not info["truncated"]
        assert info["event_reason"] == "horizon"
        _, target = training.smdp_gae([reward], [0.2], [999.0], [1], [done], gamma=1.0)
        assert target[0] == pytest.approx(reward)
    finally:
        env.close()


def test_real_training_reload_pair_evaluation_and_fresh_episode_resume(tmp_path, monkeypatch):
    config = training_config()
    output = tmp_path / "train"
    # Capture actual validated asset/source provenance once. Other agents may
    # edit files concurrently during this integration test; explicit hash
    # rejection is tested below without treating those edits as random noise.
    evidence = training.provenance()
    monkeypatch.setattr(training, "provenance", lambda: copy.deepcopy(evidence))
    reset_seeds = []
    class TrackedEnvironment(KnownOpponentEnv):
        def reset(self, seed=None):
            reset_seeds.append(seed)
            return super().reset(seed)
    monkeypatch.setattr(training, "KnownOpponentEnv", TrackedEnvironment)
    lower = load_stage1()
    frozen = {key: value.clone() for key, value in lower.state_dict().items()}
    result = training.train(config, output, steps=64, wall_seconds=60)
    assert result["physical_steps"] >= 64 and result["updates"] >= 1
    assert result["parameter_change_l2"] > 0 and result["change_from_initial_l2"] > 0
    assert result["episodes"] >= 1
    assert all(torch.equal(frozen[key], value) for key, value in lower.state_dict().items())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in lower.parameters())
    model, payload = training.load_policy(output / "latest.pt")
    assert payload["counters"]["physical_steps"] == result["physical_steps"]
    assert not model.training and model.config["hidden_dim"] == 128
    initialized = torch.load(output / "initialized.pt", map_location="cpu", weights_only=False)
    assert any(not torch.equal(value, initialized["model"][key]) for key, value in model.state_dict().items())
    assert all(torch.equal(value, payload["model"][key]) for key, value in model.state_dict().items())

    summary = evaluate(config, {"trained": output / "latest.pt"}, tmp_path / "evaluation",
                       scales=(4,), episodes=1, wall_seconds=60, matched_static=True)
    assert summary["complete_pairs"] == 1 and summary["incomplete_pairs"] == 0
    assert {row["method"] for row in summary["table"]} == {"trained", "static_matched"}
    assert all(row["episodes"] == 1 and row["constraint_violations"] == 0 for row in summary["table"])
    trained_trace = json.loads((tmp_path / "evaluation/trace_trained_4.json").read_text(encoding="utf-8"))
    static_trace = json.loads((tmp_path / "evaluation/trace_static_matched_4.json").read_text(encoding="utf-8"))
    assert trained_trace[0]["action"] == static_trace[0]["action"]
    assert all(not row["released"] for row in static_trace[1:])

    invocation_resets = len(reset_seeds)
    resumed = training.train(config, output, steps=10, wall_seconds=60, resume=True)
    assert resumed["physical_steps"] == result["physical_steps"] + resumed["invocation_steps"]
    assert resumed["invocation_steps"] >= 10 and resumed["updates"] > result["updates"]
    assert reset_seeds[invocation_resets] == config["seed"] + 100000 + result["next_episode"]
    assert resumed["next_episode"] > result["next_episode"]
    archived = torch.load(output / 'snapshots' / f"step_{result['physical_steps']}.pt", map_location='cpu', weights_only=False)
    assert archived['counters']['physical_steps'] == result['physical_steps']
    rows = read_jsonl(output / "training.jsonl")
    assert [row["physical_steps"] for row in rows] == sorted(row["physical_steps"] for row in rows)

    checkpoint_hash = sha256(output / "latest.pt")
    with pytest.raises(ValueError, match="configuration differs"):
        training.train(dict(config, command_interval=3), output, steps=1, wall_seconds=60, resume=True)
    monkeypatch.setattr(training, "provenance", lambda: {**copy.deepcopy(evidence), "source_hash": "modified-source"})
    with pytest.raises(ValueError, match="source code differs"):
        training.train(config, output, steps=1, wall_seconds=60, resume=True)
    replaced = copy.deepcopy(evidence)
    replaced['assets']['lcl']['sha256'] = 'different-frozen-executor'
    monkeypatch.setattr(training, 'provenance', lambda: copy.deepcopy(replaced))
    with pytest.raises(ValueError, match='frozen assets differ'):
        training.train(config, output, steps=1, wall_seconds=60, resume=True)
    assert sha256(output / "latest.pt") == checkpoint_hash


def test_change_metrics_ignore_casualty_deletion_and_group_order():
    previous = Grouping((Group(0, (0, 1)), Group(1, (2, 3))), (4,))
    action = Grouping((Group(1, (3, 2)), Group(0, (4, 0))))
    changes = grouping_changes(previous, action, (0, 2, 3, 4))
    assert changes["task_change"] == pytest.approx(1 / 4)
    assert changes["team_change"] == pytest.approx(1 / 6)
    assert grouping_changes(previous, previous.prune((0, 2, 3, 4)), (0, 2, 3, 4)) == {"task_change": 0.0, "team_change": 0.0}
    assert grouping_changes(Grouping((), (0, 1)), Grouping((), (1, 0)), (0, 1))["team_change"] == 0.0
    assert grouping_changes(Grouping((), (0,)), Grouping((), (0,)), (0,))["team_change"] == 0.0
    assert grouping_changes(Grouping(()), Grouping(()), ()) == {"task_change": 0.0, "team_change": 0.0}


def test_alma_training_cadence_resume_and_episode_seed_reproduction(tmp_path, monkeypatch):
    config = training_config()
    config.update(method="alma", model={"num_candidates": 2, "batch_size": 2, "update_every": 2})
    # ALMA's cadence is its own replay setting, independent of PPO rollouts.
    config["ppo"]["rollout_events"] = 64
    evidence = training.provenance()
    monkeypatch.setattr(training, "provenance", lambda: copy.deepcopy(evidence))
    output = tmp_path / "alma"
    result = training.train(config, output, steps=20, wall_seconds=60)
    model, payload = training.load_policy(output / "latest.pt")
    state = payload["training_state"]
    assert state["observed"] == result["upper_events"]
    assert result["updates"] == state["updates"] == state["observed"] // 2
    assert len(state["replay"]) == state["observed"]
    assert state["q_optimizer"]["state"] and state["proposal_optimizer"]["state"]
    assert result["decode_steps"] > 0
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.target.parameters())
    env = KnownOpponentEnv(red=4, blue=4, max_steps=50, command_interval=5)
    try:
        seed = 20270913
        seed_everything(seed + 7000000)
        first_state = env.reset(seed=seed)
        expected = model.act(first_state).action
        _, actual, _ = rollout_episode(env, model, seed=seed)
        assert actual == expected  # Same rule used to obtain static_matched's first action.
    finally:
        env.close()
    resumed = training.train(config, output, steps=10, wall_seconds=60, resume=True)
    _, latest = training.load_policy(output / "latest.pt")
    restored = latest["training_state"]
    assert resumed["updates"] == restored["updates"] == restored["observed"] // 2
    assert restored["observed"] > state["observed"] and len(restored["replay"]) > len(state["replay"])
    assert resumed["physical_steps"] == result["physical_steps"] + resumed["invocation_steps"]


def test_summary_uses_only_complete_paired_episodes_and_actual_decision_times():
    def row(seed, method, success, times=(0.01, 0.02)):
        return {"scale": 4, "seed": seed, "method": method, "success": success,
                "decision_seconds": list(times), "release_ratio": 0.25,
                "task_change": 0.1, "team_change": 0.2, "red_survivors": 2,
                "physical_steps": 50, "constraint_violations": 0}
    rows = [row(1, "selective", True), row(1, "static", False),
            row(2, "selective", False), row(2, "static", True),
            row(3, "selective", True, times=(99.0,))]
    summary = summarize(rows, ["selective", "static"])
    assert summary["complete_pairs"] == 2 and summary["incomplete_pairs"] == 1
    assert len(summary["table"]) == 2
    for result in summary["table"]:
        assert result["episodes"] == 2 and result["successes"] == 1
        assert result["success_rate"] == 0.5
        assert result["wilson_low"] < 0.5 < result["wilson_high"]
        assert result["decision_median_ms"] == pytest.approx(15.0)
        assert result["decision_p95_ms"] == pytest.approx(20.0)
        assert result["constraint_violations"] == 0

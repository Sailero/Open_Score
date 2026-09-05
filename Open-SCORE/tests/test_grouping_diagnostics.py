"""Small real-simulator checks of diagnostic controls and truthful completion."""
import json

import numpy as np
import torch

from open_score.grouping.baselines import balanced_initial
from open_score.grouping.diagnostics import (
    continue_from_snapshot, run_diagnostics, same_assignment_partitions,
)
from open_score.grouping.domain import Group, Grouping
from open_score.grouping.environment import KnownOpponentEnv


def test_partition_probe_changes_only_same_target_team_relations():
    old = Grouping((Group(0, (1, 2)), Group(0, (3,)), Group(1, (4, 5))), (6,))
    packed, singles = same_assignment_partitions(old)
    assert packed != singles
    assert packed.assignment() == singles.assignment() == old.assignment()
    assert packed.reserve == singles.reserve == (6,)
    packed.validate(range(1, 7), [0, 1])
    singles.validate(range(1, 7), [0, 1])


def test_physical_diagnostic_restore_is_exact_and_does_not_double_advance_lcl():
    torch.set_num_threads(1)
    env = KnownOpponentEnv(red=4, blue=4, seed=605, max_steps=10)
    state = env.reset()
    action = balanced_initial(state)
    snapshot = env.snapshot()
    first = continue_from_snapshot(env, snapshot, action, 900, record_trajectory=True)
    second = continue_from_snapshot(env, snapshot, action, 900, record_trajectory=True)
    assert first == second
    assert first["return"] in (0.0, 1.0)
    assert len(first["trajectory"]) == first["physical_steps"]
    assert first["trajectory"][0]["step"] == 1
    assert len(first["first_actions"]) == 4
    restored = env.restore(snapshot)
    assert restored == state and all(not any(row) for row in restored.memory.values())


def test_real_diagnostics_complete_with_independent_screening_and_same_repair(tmp_path):
    config = {"seed": 407, "opponent": "reactive", "max_steps": 50,
              "command_interval": 5, "torch_threads": 1,
              "model": {"hidden_dim": 16, "heads": 4, "layers": 1},
              "diagnostics": {"candidate_limit": 8}}
    result = run_diagnostics(config, tmp_path, states=1, rollouts=1, wall_seconds=60)
    assert result["status"] == "complete"
    assert result["completed_states"] == result["completed_e2_states"] == 1
    assert result["completed_e3_modes"] == 4
    assert result["e3_policy_status"] == "untrained_initialization"
    assert result["completed_physical_rollouts"] >= 20
    e1 = result["e1"][0]
    assert e1["same_assignment"] and e1["comparable"] and len(e1["pairs"]) == 1
    e2 = result["e2"][0]
    assert set(e2["screen_seeds"]).isdisjoint(e2["final_seeds"])
    assert e2["candidate_count"] == len(e2["candidates"]) == 8
    assert all(value in (0., 1.) for row in e2["candidates"]
               for key in ("screen_returns", "final_returns") for value in row[key])
    reference, selected = e2["screen_reference_index"], e2["proxy_selected_index"]
    np.testing.assert_allclose(e2["selected_gap"], e2["final_returns"][reference] - e2["final_returns"][selected])
    if not any(e2['final_returns']):
        assert not e2['any_success'] and not e2['action_values_vary']
        assert e2['high_return_min_task_change'] is None
        assert e2['high_return_min_team_change'] is None
    e3 = result["e3"][0]
    assert {row["mode"] for row in e3["matched_count"]} == {"full", "random", "rule", "selective"}
    assert all(len(row["released_ids"]) == e3["release_count"] for row in e3["matched_count"])
    assert {row["mode"] for row in e3["normal_release"]} == {"full", "selective"}
    assert (tmp_path / "diagnostics.md").is_file() and (tmp_path / "states/state_000.pt").is_file()
    assert json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_tiny_diagnostic_budget_reports_zero_completion_instead_of_fabricated_results(tmp_path):
    result = run_diagnostics({"seed": 52, "max_steps": 50, "torch_threads": 1,
                              "model": {"hidden_dim": 16, "heads": 4, "layers": 1}},
                             tmp_path, states=3, rollouts=2, wall_seconds=1e-9)
    assert result["status"] == "partial"
    assert result["completed_states"] == result["completed_physical_rollouts"] == 0
    assert result["e1"] == result["e2"] == result["e3"] == []

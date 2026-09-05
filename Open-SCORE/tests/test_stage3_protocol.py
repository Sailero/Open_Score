from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from open_score.stage3.artifacts import LiveProgress
from scripts.evaluate_stage3_identity import _validate_config, run_exact


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "configs/stage3_aligned.yaml"


def registered_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_registered_protocol_is_identity_level_hard_four():
    config = registered_config()
    _validate_config(config)
    game = config["game"]
    assert config["schema_version"] == "open-score-stage3-identity-cs-bbg-v5"
    assert game["action_representation"] == "labelled_agent_target_channel_partition"
    assert game["identity_level"] is True
    assert game["group_size_support"] == [1, 2, 3, 4]
    assert game["hard_cap_at_four"] is True
    assert game["repeated_target_groups"] is True
    assert game["red_reserve_enabled"] is True
    assert game["blue_reserve_enabled"] is False
    assert game["red_reserve_instant_utility"] == 0.0
    assert game["payoff_normalization"] == "analytic_absolute_bound_M_times_live_blue"
    assert game["public_matching_mechanism"] == "target_channel"
    assert game["channel_symmetry_breaking"] == "nonincreasing_group_size_per_target"
    assert config["solver"]["method"] == "double_oracle_with_set_partitioning_milp_best_responses"
    assert config["physical_evaluation"]["red_commanders"] == [
        "balanced_identity",
        "identity_blotto",
        "revealed_blue_br",
    ]
    assert [row["targets"] for row in config["physical_evaluation"]["scenarios"]] == [2, 4, 5]


@pytest.mark.parametrize(
    "path,value",
    [
        (("game", "simultaneous_upper_actions"), False),
        (("game", "identity_level"), False),
        (("game", "fixed_targets_per_episode"), False),
        (("game", "repeated_target_groups"), False),
        (("game", "action_representation"), "target_counts"),
        (("game", "group_size_support"), [1, 2, 3, 4, 5]),
        (("game", "hard_cap_at_four"), False),
        (("game", "red_reserve_enabled"), False),
        (("game", "blue_reserve_enabled"), True),
        (("game", "red_reserve_instant_utility"), 1.0),
        (("game", "payoff_normalization"), "none"),
        (("game", "public_matching_mechanism"), "posthoc_chunking"),
        (("game", "channel_symmetry_breaking"), "arbitrary_empty_channel_shift"),
    ],
)
def test_protocol_rejects_semantic_drift(path, value):
    config = deepcopy(registered_config())
    parent = config
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        _validate_config(config)


def test_exact_track_compares_double_oracle_with_complete_identity_matrix(tmp_path):
    config = registered_config()
    rows, summary = run_exact(
        config, tmp_path, LiveProgress(tmp_path / "progress.log"), smoke=True
    )
    assert len(rows) == 2
    assert summary["max_value_error"] <= config["acceptance"]["small_game_value_error_max"]
    assert summary["max_exploitability"] <= config["acceptance"]["small_game_exploitability_max"]
    assert summary["all_full_game_certified"]
    assert (tmp_path / "raw/exact_identity_games.csv").is_file()

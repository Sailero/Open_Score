import sys

from open_score.envs import HADStage1Adapter
from scripts.show_stage1_had_policy import parse_args


def test_visualizer_defaults_to_continuous_30_fps(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["show_stage1_had_policy.py"])
    args = parse_args()
    assert args.fps == 30.0
    assert args.episodes == 0


def test_stage1_spawn_sides_match_asset_defence_protocol():
    adapter = HADStage1Adapter(
        6,
        3,
        max_steps=50,
        shaping_scale=0.5,
        allow_unregistered_roster=True,
    )
    adapter.reset(seed=48_260_831)

    assert all(target.position[0] < 0.0 for target in adapter.env.targets)
    assert all(agent.position[0] < 0.0 for agent in adapter.env.red_agents)
    assert all(agent.position[0] >= 0.0 for agent in adapter.env.blue_agents)

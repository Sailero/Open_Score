from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from open_score.envs import HADStage1Adapter
from open_score.stage2 import (
    DynamicHADOutcomeNet,
    HADCanonicalizer,
    HADVariableSetState,
    project_roster_probability_surface,
)
from open_score.stage3.payoff import FrozenStage2Payoff, build_local_payoff_tensor
from scripts import stage2_pipeline_common as stage2_common
from scripts import run_stage2_aligned


PROJECT = Path(__file__).resolve().parents[1]


class RecordingPredictor:
    def __init__(self):
        self.rosters = []

    def centered_utility(self, states, **kwargs):
        self.rosters.extend(kwargs["rosters"])
        return np.zeros(len(states), dtype=np.float32)


def _formal_config():
    return yaml.safe_load(
        (PROJECT / "configs/stage2_aligned.yaml").read_text(encoding="utf-8")
    )


def test_group_value_canonicalizer_supports_the_four_by_four_boundary():
    adapter = HADStage1Adapter(
        4,
        4,
        max_steps=2,
        shaping_scale=0.5,
        allow_unregistered_roster=True,
    )
    observation = adapter.reset(seed=48_260_831)["Red"]
    encoded = HADCanonicalizer().entity_set_from_observation(observation)

    assert encoded.target.shape == (8,)
    assert encoded.red_entities.shape == (4, 9)
    assert encoded.blue_entities.shape == (4, 9)
    assert encoded.context.shape == (5,)
    assert encoded.context[0] == pytest.approx(1.0)
    assert encoded.context[1] == pytest.approx(1.0)
    assert np.all(encoded.red_entities[:, 8] == 1.0)
    assert np.all(encoded.blue_entities[:, 8] == 1.0)


def test_aligned_outcome_model_accepts_variable_permutation_invariant_rosters():
    torch.manual_seed(406)
    model = DynamicHADOutcomeNet(
        horizon_bins=10, entity_hidden_dim=16, hidden_dim=24
    )
    target = torch.randn(3, 8)
    context = torch.randn(3, 5)
    red = torch.randn(3, 4, 9)
    blue = torch.randn(3, 3, 9)
    red[..., 8] = 1.0
    blue[..., 8] = 1.0

    logits = model(target, red, blue, context)
    assert logits.shape == (3, 20)
    summary = model.summarize_probabilities(
        torch.softmax(logits, dim=-1), steps_per_bin=5
    )
    assert torch.allclose(
        summary["red_win_probability"] + summary["blue_win_probability"],
        torch.ones(3),
        atol=1e-6,
    )
    assert torch.all(summary["expected_remaining_steps"] > 0.0)
    assert torch.all(summary["expected_remaining_steps"] <= 50.0)

    permuted = model(
        target,
        red[:, torch.tensor([3, 1, 0, 2])],
        blue[:, torch.tensor([2, 0, 1])],
        context,
    )
    assert torch.allclose(logits, permuted, atol=1e-6, rtol=1e-6)


def test_atomic_json_checkpoint_retries_a_transient_windows_lock(
    tmp_path, monkeypatch
):
    target = tmp_path / "state.json"
    original = Path.replace
    attempts = {"count": 0}

    def transient_lock(path, destination):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError(5, "temporary sharing violation")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", transient_lock)
    stage2_common._write_json(target, {"completed": 10})

    assert attempts["count"] == 3
    assert target.read_text(encoding="utf-8").strip().endswith("}")


def test_training_only_mode_reuses_a_verified_dataset_without_copying(tmp_path):
    config = _formal_config()
    source = tmp_path / "source" / "data"
    source.mkdir(parents=True)
    dataset = source / "dynamic_outcome_time.jsonl"
    dataset.write_text('{"fixed":"evidence"}\n', encoding="utf-8")
    manifest = {
        "schema_version": "stage2-identity-local-value-data-v4",
        "status": "completed",
        "episodes": len(stage2_common._schedule(config)),
        "rows": 1,
        "dataset_sha256": stage2_common._sha256(dataset),
        "schedule_sha256": stage2_common._json_sha(stage2_common._schedule(config)),
        "execution_semantics": config["execution_semantics"],
        "supported_roster": config["supported_roster"],
    }
    stage2_common._write_json(source / "dataset_manifest.json", manifest)
    output = tmp_path / "retrain"

    reused, reused_manifest = run_stage2_aligned.reuse_completed_dataset(
        source.parent, config, output, stage2_common.LiveLog(output)
    )

    assert reused == dataset
    assert reused_manifest["reused_dataset"] == str(dataset)
    reference = json.loads(
        (output / "data/dataset_reference.json").read_text(encoding="utf-8")
    )
    assert reference["dataset_sha256"] == manifest["dataset_sha256"]
    assert not (output / "data/dynamic_outcome_time.jsonl").exists()


def test_training_only_cli_overrides_only_the_training_configuration():
    args = run_stage2_aligned.parse_args(
        [
            "--training-seed",
            "20260905",
            "--max-epochs",
            "140",
            "--patience",
            "24",
            "--learning-rate",
            "0.0004",
        ]
    )
    config = run_stage2_aligned.load_config(args)

    assert config["seed"] == 20260904
    assert config["training"]["seed"] == 20260905
    assert config["training"]["max_epochs"] == 140
    assert config["training"]["patience"] == 24
    assert config["training"]["learning_rate"] == pytest.approx(0.0004)


def state(red: int, blue: int) -> HADVariableSetState:
    red_entities = np.zeros((red, 9), dtype=np.float32)
    blue_entities = np.zeros((blue, 9), dtype=np.float32)
    red_entities[:, 7:] = 1.0
    blue_entities[:, 7:] = 1.0
    return HADVariableSetState(
        target=np.zeros(8, dtype=np.float32),
        red_entities=red_entities,
        blue_entities=blue_entities,
        context=np.asarray(
            [red / 4, blue / 4, red / 4, blue / 4, 1.0], dtype=np.float32
        ),
    )


def test_group_value_schedule_has_disjoint_selection_calibration_and_test_splits():
    config = _formal_config()
    compact = deepcopy(config)
    compact["collection"]["core_scales"] = [[4, 4]]
    compact["collection"]["sparse_scales"] = []
    compact["collection"]["heldout_scales"] = []
    compact["collection"]["core_episodes_per_cell"] = 20
    compact["collection"]["sparse_episodes_per_cell"] = 20
    compact["collection"]["heldout_episodes_per_cell"] = 20
    schedule = stage2_common._schedule(compact)
    observed = {item["split"] for item in schedule if item["group"] != "heldout"}
    assert observed == {"train", "validation", "calibration", "test"}
    assert not any(item["group"] == "heldout" for item in schedule)


def test_group_value_payoff_queries_the_four_by_four_support_and_exact_boundaries():
    predictor = RecordingPredictor()
    payoff = build_local_payoff_tensor(
        1,
        4,
        4,
        lambda _target, red, blue: state(red, blue),
        predictor,
        blue_style="rush",
        direct_red_cap=4,
        direct_blue_cap=4,
    )
    assert payoff.shape == (1, 5, 5)
    assert len(predictor.rosters) == 16
    assert (1, 1) in predictor.rosters
    assert (4, 4) in predictor.rosters
    np.testing.assert_allclose(payoff[:, :, 0], 1.0)
    np.testing.assert_allclose(payoff[:, 0, 1:], -1.0)


def test_checkpoint_embeds_independent_style_calibration(tmp_path: Path):
    model = DynamicHADOutcomeNet(2, 8, 16)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = tmp_path / "stage2.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "horizon_bins": 2,
            "steps_per_bin": 5,
            "entity_hidden_dim": 8,
            "hidden_dim": 16,
            "temperature": 1.0,
            "style_calibration": {
                "source_split": "calibration",
                "source_dataset_sha256": "abc",
                "global_offsets": {"rush": float(np.log(3.0))},
                "offsets": {},
                "effective_episode_mass": 10.0,
            },
            "execution_semantics": {
                "local_group_semantics": "one_target_one_group_pair"
            },
            "supported_roster": {"max_red": 4, "max_blue": 4},
            "shape_calibration": {
                "method": "alternating_logit_concave_isotonic_v1",
                "max_iterations": 50,
                "tolerance": 1e-7,
            },
        },
        checkpoint,
    )
    predictor = FrozenStage2Payoff(checkpoint)
    probability = predictor.predict_red_win(
        [state(1, 1)], styles=["rush"], rosters=[(1, 1)]
    )
    np.testing.assert_allclose(probability[0], 0.75, atol=1e-6)


def test_stage2_shape_calibration_enforces_monotone_diminishing_surface():
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.05, 0.95, size=(2, 4, 4))
    calibrated = project_roster_probability_surface(raw)
    assert np.all(np.diff(calibrated, axis=1) >= -1e-6)
    assert np.all(np.diff(calibrated, axis=2) <= 1e-6)
    logits = np.log(calibrated) - np.log1p(-calibrated)
    assert np.all(np.diff(logits, n=2, axis=1) <= 1e-5)
    assert np.all(np.diff(logits, n=2, axis=2) >= -1e-5)


def test_formal_sampling_is_complete_identity_local_4_by_4_grid():
    config = _formal_config()
    run_stage2_aligned.validate_group_value_protocol(config)
    collection = config["collection"]
    primary = {tuple(map(int, value)) for value in collection["core_scales"]}
    anchors = {tuple(map(int, value)) for value in collection["sparse_scales"]}
    heldout = {tuple(map(int, value)) for value in collection["heldout_scales"]}

    assert primary == {
        (red, blue) for red in range(1, 5) for blue in range(1, 5)
    }
    assert anchors == set()
    assert heldout == set()
    assert primary.isdisjoint(anchors)
    assert primary.isdisjoint(heldout)
    assert anchors.isdisjoint(heldout)

    schedule = stage2_common._schedule(config)
    by_group = {
        name: sum(item["group"] == name for item in schedule)
        for name in ("core", "sparse", "heldout")
    }
    assert by_group == {"core": 19_200, "sparse": 0, "heldout": 0}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["collection"]["core_scales"].pop(),
        lambda value: value["collection"]["sparse_scales"].append([9, 8]),
        lambda value: value["collection"]["heldout_scales"].append([4, 4]),
        lambda value: value["supported_roster"].__setitem__("max_red", 9),
        lambda value: value["execution_semantics"].__setitem__(
            "local_group_semantics", "target_total_then_split"
        ),
    ],
)
def test_group_value_protocol_rejects_domain_drift(mutation):
    config = _formal_config()
    mutation(config)
    with pytest.raises(ValueError):
        run_stage2_aligned.validate_group_value_protocol(config)


def _synthetic_group_state(red: int, blue: int, *, alive: bool = True):
    red_entities = np.zeros((red, 9), dtype=np.float32)
    blue_entities = np.zeros((blue, 9), dtype=np.float32)
    red_entities[:, 7] = float(alive)
    blue_entities[:, 7] = float(alive)
    red_entities[:, 8] = 1.0
    blue_entities[:, 8] = 1.0
    return red_entities, blue_entities


def test_v4_dataset_schema_distinguishes_initial_and_current_alive_counts(tmp_path):
    red_entities, blue_entities = _synthetic_group_state(3, 2)
    row = {
        "episode_id": "core-4v4-rush-0001",
        "split": "train",
        "scale_group": "core",
        "scale": "3v2",
        "origin_scale": "4v4",
        "initial_red_count": 4,
        "initial_blue_count": 4,
        "current_red_count": 3,
        "current_blue_count": 2,
        "red_count": 3,
        "blue_count": 2,
        "opponent": "rush",
        "episode_seed": 1,
        "step": 1,
        "terminal_step": 5,
        "remaining_steps": 4,
        "red_win": 1,
        "execution_semantics": "single_group_replanned_continuation",
        "red_entities": red_entities.tolist(),
        "blue_entities": blue_entities.tolist(),
    }
    dataset = tmp_path / "current_alive.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = stage2_common._read_rows(dataset)
    assert loaded[0]["origin_scale"] == "4v4"
    assert loaded[0]["scale"] == "3v2"
    assert len(loaded[0]["red_entities"]) == loaded[0]["current_red_count"]
    assert len(loaded[0]["blue_entities"]) == loaded[0]["current_blue_count"]

    row["red_entities"][0][7] = 0.0
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only live agents"):
        stage2_common._read_rows(dataset)


def test_episode_state_filters_dead_agents_to_the_current_deployable_roster():
    class FakeAdapter:
        red_ids = (10, 11, 12)
        blue_ids = (20, 21)
        step_count = 7

        def agent_states(self, side):
            ids = self.red_ids if side == "Red" else self.blue_ids
            return {
                agent_id: {"alive": agent_id not in {10, 20}}
                for agent_id in ids
            }

        def local_state_entities(self, target_id, red_ids, blue_ids, local_step):
            assert target_id == 0
            assert tuple(red_ids) == (11, 12)
            assert tuple(blue_ids) == (21,)
            assert local_step == self.step_count
            entities = np.zeros((len(red_ids) + len(blue_ids) + 1, 12), np.float32)
            entities[:, 7] = 1.0
            entities[: len(red_ids), 8] = 1.0
            entities[len(red_ids) : len(red_ids) + len(blue_ids), 9] = 1.0
            entities[-1, 10] = 1.0
            entities[:, 11] = 0.5
            return entities

    encoded = run_stage2_aligned._episode_state(FakeAdapter())
    assert encoded.red_entities.shape == (2, 9)
    assert encoded.blue_entities.shape == (1, 9)
    assert np.all(encoded.red_entities[:, 7:] == 1.0)
    assert np.all(encoded.blue_entities[:, 7:] == 1.0)
    np.testing.assert_allclose(encoded.context[:4], [0.5, 0.25, 0.5, 0.25])


def test_stage2_episode_executes_one_actual_group_without_hidden_rechunking(monkeypatch):
    calls = []

    class FakeAdapter:
        step_count = 0

        def __init__(self, red, blue, targets, **kwargs):
            assert targets == 1
            self.red_ids = tuple(range(red))
            self.blue_ids = tuple(range(100, 100 + blue))

        def reset(self, **kwargs):
            return None

        def agent_states(self, side):
            ids = self.red_ids if side == "Red" else self.blue_ids
            return {agent_id: {"alive": True} for agent_id in ids}

        def local_state_entities(self, target_id, red_ids, blue_ids, local_step):
            entities = np.zeros((len(red_ids) + len(blue_ids) + 1, 12), np.float32)
            entities[:, 7] = 1.0
            entities[: len(red_ids), 8] = 1.0
            entities[len(red_ids) : len(red_ids) + len(blue_ids), 9] = 1.0
            entities[-1, 10] = 1.0
            entities[:, 11] = 1.0
            return entities

        def step(self, actions, **kwargs):
            self.step_count += 1
            return {}, {}, True, {"outcome_red": 1.0}

    class RecordingExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def act(self, adapter, **kwargs):
            calls.append(kwargs["roster_override"])
            return {}

    monkeypatch.setattr(run_stage2_aligned, "HADStage3Adapter", FakeAdapter)
    monkeypatch.setattr(
        run_stage2_aligned, "FrozenStage1GroupExecutor", RecordingExecutor
    )
    config = _formal_config()
    snapshots, red_win, terminal_step, micro_rosters = run_stage2_aligned._run_episode(
        {"scale": [4, 4], "opponent": "rush", "seed": 7},
        object(),
        torch.device("cpu"),
        config,
    )

    assert red_win == 1
    assert terminal_step == 1
    assert micro_rosters == [(4, 4)]
    assert len(snapshots) == 1
    assert calls == [{(0, 0): (tuple(range(4)), tuple(range(100, 104)))}]

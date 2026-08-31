"""Focused tests for Stage-1 checkpoint-selection and transfer protocols."""

import csv
import hashlib
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from open_score.formal_contracts import FormalContractMismatch
from open_score.stage1.replay import EpisodeReplayBuffer


ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _formal_args(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", [str(module.__file__), *argv])
    args = module.parse_args()
    module.validate_args(args)
    return args


HAD_FORMAL_ARGV = [
    "--formal-evidence",
    "--algorithms", "qmix", "vdn", "mappo",
    "--seeds", "20260830", "20260831", "20260832", "20260833", "20260834",
    "--episodes", "2000",
    "--batch-episodes", "12",
    "--replay-episodes", "1024",
    "--updates-per-episode", "1",
    "--target-update-interval", "200",
    "--ppo-epochs", "4",
    "--learning-rate", "0.0005",
    "--shaping-scale", "0.5",
    "--max-steps", "50",
    "--eval-every", "100",
    "--eval-episodes-per-scale", "4",
    "--heldout-episodes-per-scale", "20",
    "--opponent-styles", "rush", "split_rush",
    "--episodes-per-stage", "100",
    "--epsilon-anneal-steps", "50000",
    "--agent-hidden-dim", "128",
    "--critic-hidden-dim", "128",
    "--encoder-kind", "saqa",
    "--attention-heads", "4",
    "--bc-demo-episodes-per-scale-opponent", "16",
    "--bc-algorithms", "qmix",
    "--bc-epochs", "8",
    "--bc-batch-episodes", "12",
    "--bc-learning-rate", "0.001",
]


STOCK_FORMAL_ARGV = [
    "--formal-evidence",
    "--algorithms", "qmix", "vdn", "mappo",
    "--seeds", "20260830", "20260831", "20260832", "20260833", "20260834",
    "--minimum-source-seeds", "5",
    "--encoder-kind", "saqa",
    "--epsilon-start", "0.9",
    "--use-cpp-rvo2",
]


AD_FORMAL_ARGV = [
    "--formal-evidence",
    "--algorithms", "qmix", "vdn", "mappo",
    "--seeds", "20260830", "20260831", "20260832", "20260833", "20260834",
    "--initializations", "scratch", "stock_transfer",
    "--stock-checkpoint-template",
    "outputs/smaclite_stock_entity_formal/{algorithm}_seed{seed}.pt",
    "--ratios", "2:1,3:2,5:3",
    "--train-side", "Red",
    "--opponent", "intercept",
    "--warmup-opponent", "idle",
    "--warmup-fraction", "0.25",
    "--episodes", "2000",
    "--episode-limit", "150",
    "--batch-episodes", "12",
    "--replay-episodes", "1024",
    "--updates-per-episode", "1",
    "--target-update-interval", "200",
    "--validation-episodes-per-ratio", "20",
    "--heldout-episodes-per-ratio", "40",
    "--eval-every", "100",
    "--learning-rate", "0.0005",
    "--agent-hidden-dim", "128",
    "--critic-hidden-dim", "128",
    "--encoder-kind", "saqa",
    "--attention-heads", "4",
    "--ppo-epochs", "4",
    "--epsilon-start", "0.9",
    "--epsilon-finish", "0.05",
    "--epsilon-anneal-steps", "100000",
    "--gamma", "0.99",
    "--reward-mode", "strict_potential",
    "--shaping-scale", "0.5",
    "--approach-weight", "1.0",
    "--spawn-jitter", "2.0",
    "--max-red-agents", "6",
    "--max-blue-agents", "5",
    "--use-cpp-rvo2",
]


@pytest.mark.parametrize(
    ("script_name", "argv", "config_attribute", "field_flag", "bad_value"),
    [
        (
            "train_stage1_baselines",
            HAD_FORMAL_ARGV,
            "HAD_FORMAL_CONFIG",
            "--episodes",
            "1999",
        ),
        (
            "train_smaclite_stock_entity_baselines",
            STOCK_FORMAL_ARGV,
            "STOCK_SOURCE_FORMAL_CONFIG",
            "--replay-episodes",
            "511",
        ),
        (
            "train_smaclite_ad_baselines",
            AD_FORMAL_ARGV,
            "STOCK_TO_AD_FORMAL_CONFIG",
            "--heldout-episodes-per-ratio",
            "39",
        ),
    ],
)
def test_formal_clis_exactly_match_hashed_registered_contract_and_reject_mismatch(
    monkeypatch, script_name, argv, config_attribute, field_flag, bad_value
):
    module = _script(script_name)
    args = _formal_args(monkeypatch, module, argv)
    contract = args._formal_contract
    raw = getattr(module, config_attribute).read_bytes()
    assert contract["validation_status"] == "exact_match"
    assert contract["config_sha256"] == hashlib.sha256(raw).hexdigest()
    assert len(contract["canonical_config_sha256"]) == 64

    mismatched = list(argv)
    if field_flag in mismatched:
        mismatched[mismatched.index(field_flag) + 1] = bad_value
    else:
        mismatched.extend((field_flag, bad_value))
    with pytest.raises(FormalContractMismatch, match=field_flag[2:].replace("-", "_")):
        _formal_args(monkeypatch, module, mismatched)


def test_had_selection_prioritizes_worst_scale_opponent_cell():
    module = _script("train_stage1_baselines")
    robust = {
        "controlled_mean_payoff": 0.30,
        "components": {
            "rush": {"per_scale_controlled_payoff": {"2v1": 0.10, "3v2": 0.5}},
            "split_rush": {
                "per_scale_controlled_payoff": {"2v1": 0.20, "3v2": 0.4}
            },
        },
    }
    brittle = {
        "controlled_mean_payoff": 0.60,
        "components": {
            "rush": {"per_scale_controlled_payoff": {"2v1": -0.20, "3v2": 0.9}},
            "split_rush": {
                "per_scale_controlled_payoff": {"2v1": 0.80, "3v2": 0.9}
            },
        },
    }
    assert module.robust_validation_key(robust) > module.robust_validation_key(
        brittle
    )


def test_ad_selection_prioritizes_worst_ratio_and_replay_is_cleared():
    module = _script("train_smaclite_ad_baselines")
    robust = SimpleNamespace(
        win_rate=0.60,
        mean_return=0.2,
        per_ratio={
            "2:1": {"win_rate": 0.5, "mean_return": 0.1},
            "5:3": {"win_rate": 0.7, "mean_return": 0.3},
        },
    )
    brittle = SimpleNamespace(
        win_rate=0.80,
        mean_return=0.6,
        per_ratio={
            "2:1": {"win_rate": 0.2, "mean_return": -0.1},
            "5:3": {"win_rate": 1.0, "mean_return": 1.0},
        },
    )
    assert module.robust_validation_key(robust) > module.robust_validation_key(
        brittle
    )
    replay = EpisodeReplayBuffer(capacity=9, seed=1)
    fresh, audit = module.reset_replay_at_opponent_transition(
        replay, capacity=9, seed=2, episode=501
    )
    assert fresh is not replay
    assert len(fresh) == 0
    assert audit["performed"]
    assert audit["transition_episode"] == 501


def test_mappo_pending_is_flushed_before_opponent_transition(monkeypatch):
    module = _script("train_smaclite_ad_baselines")
    pending = [object() for _ in range(8)]

    class Learner:
        def __init__(self):
            self.batches = []

        def train_batch(self, batch):
            self.batches.append(batch)
            return SimpleNamespace(learning_signal_by_scale={})

    learner = Learner()
    monkeypatch.setattr(module, "collate_episodes", lambda episodes, device: tuple(episodes))
    metrics, audit = module.flush_mappo_pending_at_opponent_transition(
        pending, learner, torch.device("cpu"), episode=501
    )
    assert metrics is not None
    assert len(learner.batches) == 1 and len(learner.batches[0]) == 8
    assert pending == []
    assert audit["flushed_idle_episode_count"] == 8
    assert audit["discarded_idle_episode_count"] == 0
    assert audit["mixed_opponent_batch_prevented"]


def test_stock_source_gate_rejects_zero_wins_and_aggregates_seeds():
    module = _script("train_smaclite_stock_entity_baselines")
    args = SimpleNamespace(
        heldout_episodes=10,
        minimum_source_win_rate=0.20,
        minimum_source_seeds=2,
        algorithms=["qmix"],
    )
    zero = SimpleNamespace(
        episodes=10,
        win_rate=0.0,
        mean_return=-1.0,
        paired_wins=(False,) * 10,
    )
    assert not module.source_performance_gate(zero, args)["passed"]
    runs = [
        {
            "algorithm": "qmix",
            "seed": seed,
            "heldout_selected": {"win_rate": rate},
            "source_performance_gate": {"passed": True},
        }
        for seed, rate in ((1, 0.3), (2, 0.5))
    ]
    gate = module.aggregate_source_performance_gates(runs, args)
    assert gate["passed"]
    assert gate["algorithms"]["qmix"]["heldout_win_rate"]["seed_count"] == 2


def test_ad_audit_source_integrity_matches_registered_manifest_without_rollout():
    module = _script("audit_smaclite_ad_protocol")
    manifest = json.loads(module.STOCK_MANIFEST.read_text(encoding="utf-8"))
    evidence, gates = module.collect_source_integrity(
        manifest, use_cpp_rvo2=True
    )

    assert all(gates.values())
    maps = evidence["runtime_stock_maps"]
    pin = manifest["source_pins"]["smaclite"]
    assert maps["file_count"] == 13
    assert maps["combined_hash_algorithm"] == (
        "sorted_filename_nul_raw_bytes_lf_sha256"
    )
    assert maps["combined_sha256"] == pin["stock_scenario_combined_sha256"]
    assert evidence["rvo2_backend"]["binary_sha256"] == manifest["source_pins"][
        "smaclite_python_rvo2"
    ]["binary_sha256"]
    assert module.PYTEST_CONTRACT["executed_by_this_audit"] is False


def test_ad_audit_aborts_before_behavior_when_source_integrity_fails(monkeypatch):
    module = _script("audit_smaclite_ad_protocol")
    monkeypatch.setattr(
        module,
        "collect_source_integrity",
        lambda manifest, use_cpp_rvo2: (
            {"reason": "synthetic source mismatch"},
            {"smaclite_checkout_commit_exact": False},
        ),
    )
    monkeypatch.setattr(
        module,
        "SMACliteADFactory",
        lambda **kwargs: pytest.fail("behavior environment must not be constructed"),
    )
    args = SimpleNamespace(
        ratios=((2, 1), (3, 2), (5, 3)),
        episodes_per_ratio=60,
        mapping_episodes_per_ratio=5,
        layout_seeds=50,
        episode_limit=150,
        max_red_agents=6,
        max_blue_agents=5,
        spawn_jitter=2.0,
        reward_mode="strict_potential",
        gamma=0.99,
        shaping_scale=0.5,
        approach_weight=1.0,
        seed=20260831,
        use_cpp_rvo2=True,
        output=Path("unused.json"),
    )
    payload = module.run_audit(args)
    assert payload["passed"] is False
    assert payload["aborted_before_behavior_audit"] is True
    assert payload["pytest_contract"]["executed_by_this_audit"] is False


def test_had_fixed_2v1_engineering_gate_is_machine_checked(tmp_path):
    module = _script("train_stage1_baselines")
    args = SimpleNamespace(
        formal_evidence=False,
        fixed_scale=[2, 1],
        seeds=[20260830],
    )
    results = []
    for algorithm in ("qmix", "vdn", "mappo"):
        final_path = tmp_path / f"{algorithm}_final.pt"
        final_path.write_bytes(f"{algorithm}-checkpoint".encode("ascii"))
        results.append(
            {
                "algorithm": algorithm,
                "seed": 20260830,
                "updates_this_run": 3,
                "selected_parameters_changed": True,
                "selected_parameters_finite": True,
                "final_parameters_finite": True,
                "latest_learning_metrics": {
                    "loss": 0.5,
                    "grad_norm": 1.25,
                    "learner_step": 3,
                    "learning_signal_by_scale": {"2v1": 0.2},
                },
                "checkpoint": str(final_path),
                "checkpoint_sha256": module.checkpoint_sha256(final_path),
                "checkpoint_audit": {
                    "best": {
                        "disk_load_performed": True,
                        "strict_model_load": True,
                        "parameters_finite_after_reload": True,
                    },
                    "final": {
                        "saved": True,
                        "sha256_recorded": True,
                        "disk_reload_performed": False,
                    },
                },
            }
        )

    passing = module.engineering_precheck_gate(results, args)
    assert passing["passed"] is True
    assert passing["algorithm_seed_coverage_exact"] is True
    assert passing["final_checkpoint_reload_claim"] is False

    results[0]["latest_learning_metrics"]["loss"] = float("nan")
    failing = module.engineering_precheck_gate(results, args)
    assert failing["passed"] is False
    assert failing["runs"][0]["checks"]["critical_learner_metrics_finite"] is False


def test_ad_task_success_is_not_invalidated_by_transfer_ceiling():
    module = _script("train_smaclite_ad_baselines")
    results = []
    for seed in range(5):
        common_evaluation = {
            "layout_hashes": [f"layout-{seed}-{index}" for index in range(3)],
            "paired_returns": [1.0, 1.0, 1.0],
            "win_rate": 1.0,
            "mean_return": 1.0,
            "per_ratio": {
                "2:1": {"win_rate": 1.0},
                "3:2": {"win_rate": 1.0},
                "5:3": {"win_rate": 1.0},
            },
        }
        for initialization in ("scratch", "stock_transfer"):
            results.append(
                {
                    "algorithm": "qmix",
                    "seed": seed,
                    "initialization": initialization,
                    "random_initial_model_sha256": f"random-{seed}",
                    "validation_best_parameters_changed": True,
                    "heldout": {"validation_best": common_evaluation},
                    "transfer_manifest": (
                        {
                            "changed_copied_tensor_count": 1,
                            "copied_source_trained_tensor_count": 1,
                            "source_training_learner_updates": 1,
                            "source_selected_checkpoint_learner_updates": 1,
                            "source_total_training_learner_updates": 1,
                        }
                        if initialization == "stock_transfer"
                        else None
                    ),
                }
            )
    aggregate = module.initialization_ablation(results)["aggregates"]["qmix"]
    assert aggregate["task_success_gate"]["passed"]
    assert aggregate["transfer_integrity_gate"]["passed"]
    assert not aggregate["transfer_benefit_gate"]["passed"]
    assert (
        aggregate["transfer_benefit_gate"]["status"]
        == "ceiling_limited_noninferior"
    )


def test_stage1_plotter_uses_seed_bootstrap_and_exposes_seed_traces(
    tmp_path, monkeypatch
):
    module = _script("plot_stage1_results")
    curve_path = tmp_path / "had_curves.csv"
    curve_rows = [
        {
            "algorithm": "qmix",
            "seed": seed,
            "episode": episode,
            "eval_controlled_mean_payoff": value,
        }
        for seed, values in ((1, (-0.4, 0.2)), (2, (0.0, 0.8)))
        for episode, value in zip((0, 100), values)
    ]
    with curve_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    had_summary_path = tmp_path / "had_summary.json"
    had_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "stage1-had-reproduction-v5",
                "results": [
                    {
                        "algorithm": "qmix",
                        "seed": 1,
                        "heldout_best_evaluation": {
                            "controlled_mean_payoff": 0.2
                        },
                    },
                    {
                        "algorithm": "qmix",
                        "seed": 2,
                        "heldout_best_evaluation": {
                            "controlled_mean_payoff": 0.8
                        },
                    },
                ],
                "rule_strategy": {
                    "heldout_evaluation_by_training_seed": {
                        "1": {"controlled_mean_payoff": -0.2},
                        "2": {"controlled_mean_payoff": 0.4},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    ad_summary_path = tmp_path / "ad_summary.json"
    ad_summary_path.write_text(
        json.dumps(
            {
                "schema_version": "smaclite-ad-baseline-v3",
                "results": [
                    {
                        "algorithm": "qmix",
                        "initialization": initialization,
                        "seed": seed,
                        "heldout": {
                            "validation_best": {"mean_return": value}
                        },
                    }
                    for initialization, values in (
                        ("scratch", (-0.1, 0.2)),
                        ("stock_transfer", (0.5, 0.9)),
                    )
                    for seed, value in zip((1, 2), values)
                ],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__file__),
            "--had-summary",
            str(had_summary_path),
            "--had-curves",
            str(curve_path),
            "--ad-summary",
            str(ad_summary_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "500",
            "--bootstrap-seed",
            "1234",
        ],
    )
    module.main()

    manifest = json.loads(
        (output_dir / "figure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "stage1-figure-manifest-v2"
    assert manifest["uncertainty"] == {
        "method": "nonparametric_percentile_bootstrap_of_seed_mean",
        "confidence_level": 0.95,
        "bootstrap_samples": 500,
        "base_seed": 1234,
        "deterministic_stream_derivation": (
            "SHA-256(base_seed|figure|cell), first 64 bits"
        ),
    }
    statistics = manifest["figure_statistics"]
    curves = statistics["had_validation_learning_curves"]
    assert curves["individual_seed_line_count"] == 2
    assert len(curves["bootstrap_cells"]) == 2
    assert {cell["seed_count"] for cell in curves["bootstrap_cells"]} == {2}
    heldout = statistics["had_heldout_payoff"]
    assert heldout["individual_seed_point_count"] == 4
    assert {cell["label"] for cell in heldout["bootstrap_cells"]} == {
        "qmix",
        "rule",
    }
    ad = statistics["smaclite_ad_initialization_ablation"]
    assert ad["individual_seed_point_count"] == 4
    assert ad["paired_seed_line_count"] == 2
    assert ad["paired_bootstrap_stream_by_algorithm"] is True
    assert len(ad["bootstrap_cells"]) == 2
    assert len(manifest["outputs"]) == 6
    assert all(Path(item["path"]).is_file() for item in manifest["outputs"])

    stream_seed = module.derived_bootstrap_seed(1234, "test", "cell")
    first = module.bootstrap_mean_interval(
        [-1.0, 0.0, 1.0], samples=500, seed=stream_seed
    )
    second = module.bootstrap_mean_interval(
        [-1.0, 0.0, 1.0], samples=500, seed=stream_seed
    )
    assert first == second
    assert first["seed_count"] == 3


def test_stage1_policy_lock_binds_relative_checkpoint_once_and_checks_clean_gate(
    tmp_path,
):
    module = _script("lock_stage1_policy")
    project = tmp_path / "repo" / "Open-SCORE"
    checkpoint = (
        project
        / "outputs"
        / "stage1_had_formal"
        / "checkpoints"
        / "best policy #1.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"validation-selected-qmix")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    summary = project / "outputs" / "stage1_had_formal" / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": "stage1-had-reproduction-v5",
                "formal_convergence_claim": True,
                "primary_strategy_gate": {"passed": True},
                "multi_seed_heldout": {
                    "algorithms": {
                        "qmix": {
                            "deployment_candidate": {
                                "selection_split": "validation_only",
                                "heldout_was_not_used_for_selection": True,
                                "seed": 20260831,
                                "checkpoint": checkpoint.relative_to(project).as_posix(),
                                "checkpoint_sha256": digest,
                                "validation_score": [0.2, 0.4],
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    configs = project / "configs"
    configs.mkdir()
    collect = configs / "stage2_had_formal_collect.yaml"
    collect.write_text(
        """data:
  defender_candidates:
    - {kind: rule, style: guard}
    - kind: checkpoint
      path: REPLACE_WITH_STAGE1_QMIX_VALIDATION_BEST_CHECKPOINT_PATH.pt
      expected_sha256: REPLACE_WITH_64_HEX_STAGE1_CHECKPOINT_SHA256
""",
        encoding="utf-8",
    )
    train = configs / "stage2_had_formal.yaml"
    train.write_text(
        """formal_data_contract:
  expected_stage1_checkpoint_sha256: REPLACE_WITH_64_HEX_STAGE1_CHECKPOINT_SHA256
""",
        encoding="utf-8",
    )
    lock_path = project / "outputs" / "stage1_had_formal" / "stage1_policy_lock.json"

    def dirty_reader(**_):
        return {"commit": "a" * 40, "dirty_before_binding": True}

    with pytest.raises(RuntimeError, match="clean committed Git worktree"):
        module.bind_policy(
            Path("outputs/stage1_had_formal/summary.json"),
            Path("configs/stage2_had_formal_collect.yaml"),
            Path("configs/stage2_had_formal.yaml"),
            Path("outputs/stage1_had_formal/stage1_policy_lock.json"),
            protocol_tag="s1s2-protocol-v1",
            apply=True,
            project_root=project,
            git_state_reader=dirty_reader,
        )
    assert module.CHECKPOINT_PATH_PLACEHOLDER in collect.read_text(encoding="utf-8")
    assert not lock_path.exists()

    clean_calls = []

    def clean_reader(*, require_clean, project_root):
        clean_calls.append((require_clean, project_root))
        assert module.CHECKPOINT_PATH_PLACEHOLDER in collect.read_text(
            encoding="utf-8"
        )
        assert not lock_path.exists()
        return {
            "commit": "b" * 40,
            "repository_root": str(project.parent),
            "dirty_before_binding": False,
        }

    result = module.bind_policy(
        Path("outputs/stage1_had_formal/summary.json"),
        Path("configs/stage2_had_formal_collect.yaml"),
        Path("configs/stage2_had_formal.yaml"),
        Path("outputs/stage1_had_formal/stage1_policy_lock.json"),
        protocol_tag="s1s2-protocol-v1",
        apply=True,
        project_root=project,
        git_state_reader=clean_reader,
    )
    assert clean_calls == [(True, project.resolve())]
    expected_relative = checkpoint.relative_to(project).as_posix()
    collect_yaml = yaml.safe_load(collect.read_text(encoding="utf-8"))
    checkpoint_spec = collect_yaml["data"]["defender_candidates"][1]
    assert checkpoint_spec["path"] == expected_relative
    assert checkpoint_spec["expected_sha256"] == digest
    train_yaml = yaml.safe_load(train.read_text(encoding="utf-8"))
    assert (
        train_yaml["formal_data_contract"]["expected_stage1_checkpoint_sha256"]
        == digest
    )
    persisted = json.loads(lock_path.read_text(encoding="utf-8"))
    assert persisted == result
    assert persisted["checkpoint"] == expected_relative
    assert persisted["checkpoint_sha256"] == digest
    assert persisted["collect_config_sha256"] == hashlib.sha256(
        collect.read_bytes()
    ).hexdigest()
    assert persisted["train_config_sha256"] == hashlib.sha256(
        train.read_bytes()
    ).hexdigest()
    assert list(project.rglob("*.pt")) == [checkpoint]

    dry_run = module.bind_policy(
        Path("outputs/stage1_had_formal/summary.json"),
        Path("configs/stage2_had_formal_collect.yaml"),
        Path("configs/stage2_had_formal.yaml"),
        Path("outputs/stage1_had_formal/new_lock.json"),
        protocol_tag="s1s2-protocol-v1",
        apply=False,
        project_root=project,
    )
    assert not dry_run["binding_applied"]


def test_policy_lock_git_gate_audits_outer_repository_and_rejects_dirty_state(
    tmp_path,
):
    module = _script("lock_stage1_policy")
    repository = (tmp_path / "outer").resolve()
    project = repository / "Open-SCORE"
    project.mkdir(parents=True)
    calls = []

    def runner(command, *, cwd, **_):
        calls.append((tuple(command[1:]), Path(cwd).resolve()))
        if command[-1] == "--show-toplevel":
            return SimpleNamespace(stdout=str(repository) + "\n")
        if command[-1] == "HEAD":
            return SimpleNamespace(stdout="c" * 40 + "\n")
        return SimpleNamespace(
            stdout=" M Open-SCORE/configs/stage2_had_formal.yaml\n"
        )

    with pytest.raises(RuntimeError, match="clean committed Git worktree"):
        module._git_state(
            True,
            project_root=project,
            git_executable="git-test",
            runner=runner,
        )
    assert calls[0] == (("rev-parse", "--show-toplevel"), project)
    assert calls[1][1] == repository
    assert calls[2] == (
        ("status", "--porcelain=v1", "--untracked-files=all"),
        repository,
    )

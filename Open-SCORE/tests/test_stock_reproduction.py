from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT
    / "configs"
    / "stock_reproduction"
    / "smaclite_aamas2023_epymarl_v3.json"
)
COLLECTOR_PATH = PROJECT_ROOT / "scripts" / "collect_stock_results.py"
PARALLEL_RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_stock_reproduction_parallel.py"


def _load_collector():
    spec = importlib.util.spec_from_file_location("collect_stock_results", COLLECTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_parallel_runner():
    spec = importlib.util.spec_from_file_location(
        "run_stock_reproduction_parallel", PARALLEL_RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_preregisters_paper_budget_labels_and_hyperparameters():
    manifest = _load_manifest()
    profile = manifest["profiles"][manifest["default_profile"]]

    assert manifest["protocol_revision"] == 3
    assert manifest["paper_protocol"]["training_steps"] == 4_000_000
    assert manifest["paper_protocol"]["number_of_seeds"] == 5
    assert profile["seeds"] == [1, 2, 3, 4, 5]
    assert profile["test_interval"] == 100_000
    assert profile["test_episodes"] == 100
    assert profile["use_cpp_rvo2"] is True
    assert profile["use_cuda"] is False
    assert profile["cpu_threads_per_run"] == 1
    assert profile["require_single_core_affinity"] is True
    assert profile["paper_numeric_gate_eligible"] is False
    assert manifest["profiles"]["windows_v3_paper_cpu_numpy_fallback"]["use_cpp_rvo2"] is False
    assert manifest["profiles"]["windows_v3_paper_cpu_numpy_fallback"]["paper_numeric_gate_eligible"] is False
    assert manifest["profiles"]["windows_v3_gpu_cpp_engineering"]["use_cuda"] is True
    assert manifest["profiles"]["windows_v3_gpu_cpp_engineering"]["paper_numeric_gate_eligible"] is False
    assert manifest["source_pins"]["smaclite_python_rvo2"] == {
        "repository": "https://github.com/uoe-agents/SMAClite-Python-RVO2",
        "commit": "a693b272e387bcf02a8f41ef295d7c3e1b19abb0",
        "python_abi": "cp310-win_amd64",
        "binary_filename": "rvo2.cp310-win_amd64.pyd",
        "binary_sha256": "634b060ef5cbf0dd0561d19c2317ff45bc4e71b9a6ab54b9fd4677a415da034c",
        "windows_build_patch": "upstream/patches/smaclite-python-rvo2-windows.patch",
        "windows_build_patch_sha256": "f0f6eeddaefb2eda3cd4a9af8d187b8d466cbab4a568450b258b7ed35f3b1ef8",
        "build_note": "The patch changes build-system arguments and library paths only; the tracked upstream checkout is restored clean after producing the binary.",
    }
    assert profile["algorithm_overrides"]["qmix"] == {
        "hidden_dim": 128,
        "lr": 0.003,
        "standardise_rewards": False,
        "use_rnn": True,
        "evaluation_epsilon": 0.0,
        "epsilon_anneal_time": 50_000,
        "target_update_interval_or_tau": 200,
    }
    assert profile["algorithm_overrides"]["vdn"]["target_update_interval_or_tau"] == 0.01
    assert profile["algorithm_overrides"]["mappo"]["q_nstep"] == 5
    assert not manifest["profiles"]["algorithm_original_faithful"]["runnable"]
    assert [
        (entry["algorithm"], entry["map"], entry["paper_reference"]["value"])
        for entry in manifest["suites"]["primary"]
    ] == [
        ("qmix", "2s_vs_1sc", 16.22),
        ("vdn", "3s5z", 20.0),
        ("mappo", "MMM2", 24.63),
    ]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _synthetic_record(tmp_path: Path, *, compliance: str = "registered", terminal_step: int = 4_000_000):
    collector = _load_collector()
    manifest = _load_manifest()
    profile_name = manifest["default_profile"]
    profile = manifest["profiles"][profile_name]
    sacred_root = tmp_path / "sacred"
    sacred_run = sacred_root / "qmix" / "2s_vs_1sc" / "1"
    record_path = tmp_path / "records" / "run_record.json"
    checkpoint_root = tmp_path / "checkpoints" / "qmix_seed1_2s_vs_1sc_test"
    checkpoint_dir = checkpoint_root / str(terminal_step)
    for name in ("agent.th", "mixer.th", "opt.th"):
        path = checkpoint_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))
    checkpoint = collector.checkpoint_fingerprint(checkpoint_root)
    assert checkpoint is not None
    manifest_sha = collector.sha256_file(MANIFEST_PATH)
    record = {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "profile": profile_name,
        "protocol_compliance": compliance,
        "algorithm_label": "EPyMARL-QMIX",
        "algorithm": "qmix",
        "map": "2s_vs_1sc",
        "seed": 1,
        "status": "completed",
        "finished_at": "2026-08-31T01:00:00+00:00",
        "manifest_path": str(MANIFEST_PATH),
        "manifest_provenance": {
            "path": str(MANIFEST_PATH),
            "repository_relative_path": "Open-SCORE/configs/stock_reproduction/smaclite_aamas2023_epymarl_v3.json",
            "sha256": manifest_sha,
            "outer_commit": "1" * 40,
        },
        "launcher_sha256": "2" * 64,
        "sacred_run_path": str(sacred_run),
        "command": ["python.exe", "main.py", "--capture=sys"],
        "cpu_threads_per_run": profile["cpu_threads_per_run"],
        "require_single_core_affinity": profile["require_single_core_affinity"],
        "cpu_affinity_mask": 1,
        "cpu_affinity_logical_index": 0,
        "thread_environment": {
            variable: str(profile["cpu_threads_per_run"])
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "runtime_cpu_probe": {
            "path": str(PROJECT_ROOT / "scripts" / "stock_runtime_probe.py"),
            "sha256": manifest["source_pins"]["runtime_cpu_probe"]["sha256"],
            "scope": manifest["source_pins"]["runtime_cpu_probe"]["scope"],
            "result": {
                "process_affinity_mask": 1,
                "torch_num_threads": profile["cpu_threads_per_run"],
                "smaclite_imported_from_expected_checkout": True,
                "stock_scenario_file_count": manifest["source_pins"]["smaclite"]["stock_scenario_file_count"],
                "stock_scenario_combined_sha256": manifest["source_pins"]["smaclite"]["stock_scenario_combined_sha256"],
                "dependency_versions": {
                    name: "test"
                    for name in ("gymnasium", "numpy", "sacred", "smaclite", "torch", "pyyaml")
                },
                "thread_environment": {
                    variable: str(profile["cpu_threads_per_run"])
                    for variable in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
            },
        },
        "source_pins": manifest["source_pins"],
        "open_score_git": {
            "repository_root": str(PROJECT_ROOT.parent),
            "commit": "1" * 40,
            "branch": "test",
            "dirty": False,
            "status_porcelain": [],
        },
        "source_checkout_tracked_clean": True,
        "post_run_source_audit": {
            "passed": True,
            "outer_commit": "1" * 40,
            "outer_dirty": False,
            "upstream": {
                "epymarl": {"commit": manifest["source_pins"]["epymarl"]["commit"], "tracked_clean": True},
                "smaclite": {"commit": manifest["source_pins"]["smaclite"]["commit"], "tracked_clean": True},
                "rvo2": {"commit": manifest["source_pins"]["smaclite_python_rvo2"]["commit"], "tracked_clean": True},
            },
        },
        "produced_checkpoint_root": str(checkpoint_root),
        "produced_checkpoint": checkpoint,
        "runtime_rvo2": {
            "backend": "cpp",
            "source_commit": manifest["source_pins"]["smaclite_python_rvo2"]["commit"],
            "source_checkout_tracked_clean": True,
            "binary_filename": manifest["source_pins"]["smaclite_python_rvo2"]["binary_filename"],
            "binary_sha256": manifest["source_pins"]["smaclite_python_rvo2"]["binary_sha256"],
        },
        "windows_compatibility_overlay": {
            "sha256": manifest["source_pins"]["windows_compatibility_overlay"]["sha256"],
            "sacred_capture_mode": "sys",
        },
    }
    config = collector.expected_config(manifest, profile_name, record)
    config["env_args"] = {
        "map_name": "2s_vs_1sc",
        "time_limit": profile["time_limit"],
        "use_cpp_rvo2": profile["use_cpp_rvo2"],
    }
    metrics = {
        "test_return_mean": {"steps": [0, terminal_step], "values": [0.0, 16.0]},
        "test_return_std": {"steps": [0, terminal_step], "values": [0.0, 0.5]},
        "test_battle_won_mean": {"steps": [0, terminal_step], "values": [0.0, 0.8]},
    }
    run = {
        "status": "COMPLETED",
        "experiment": {
            "repositories": [
                {
                    "commit": manifest["source_pins"]["epymarl"]["commit"],
                    "dirty": False,
                }
            ]
        },
    }
    _write_json(record_path, record)
    _write_json(sacred_run / "config.json", config)
    _write_json(sacred_run / "metrics.json", metrics)
    _write_json(sacred_run / "run.json", run)
    return collector, manifest, profile_name, sacred_root, record_path


def test_collector_accepts_only_config_exact_terminal_completed_run(tmp_path):
    collector, manifest, profile_name, sacred_root, record_path = _synthetic_record(tmp_path)
    run, reasons = collector.validate_record(
        record_path,
        manifest=manifest,
        profile_name=profile_name,
        sacred_root=sacred_root,
    )
    assert reasons == []
    assert run is not None
    assert run["terminal_step"] == 4_000_000
    assert run["terminal_test_return_mean"] == 16.0
    assert run["terminal_test_battle_won_mean"] == 0.8


def test_collector_rejects_override_and_missing_terminal_evaluation(tmp_path):
    collector, manifest, profile_name, sacred_root, record_path = _synthetic_record(
        tmp_path, compliance="override_non_reproduction", terminal_step=100_000
    )
    run, reasons = collector.validate_record(
        record_path,
        manifest=manifest,
        profile_name=profile_name,
        sacred_root=sacred_root,
    )
    assert run is None
    assert "record_not_registered" in reasons
    assert "missing_terminal_evaluation" in reasons


def test_collector_rejects_rvo2_or_windows_capture_provenance_drift(tmp_path):
    collector, manifest, profile_name, sacred_root, record_path = _synthetic_record(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["runtime_rvo2"]["binary_sha256"] = "0" * 64
    record["windows_compatibility_overlay"]["sacred_capture_mode"] = "fd"
    record["runtime_cpu_probe"]["result"]["process_affinity_mask"] = 3
    record["runtime_cpu_probe"]["result"]["smaclite_imported_from_expected_checkout"] = False
    record["open_score_git"]["dirty"] = True
    record["post_run_source_audit"]["passed"] = False
    record["manifest_provenance"]["sha256"] = "0" * 64
    record["command"] = [item for item in record["command"] if not item.startswith("--capture=")]
    _write_json(record_path, record)

    run, reasons = collector.validate_record(
        record_path,
        manifest=manifest,
        profile_name=profile_name,
        sacred_root=sacred_root,
    )
    assert run is None
    assert "rvo2_binary_sha256_mismatch" in reasons
    assert "sacred_capture_mode_mismatch" in reasons
    assert "command_missing_registered_capture_mode" in reasons
    assert "runtime_cpu_affinity_mismatch" in reasons
    assert "runtime_smaclite_import_path_mismatch" in reasons
    assert "open_score_git_not_clean" in reasons
    assert "post_run_source_audit_failed" in reasons
    assert "manifest_sha256_mismatch" in reasons


def test_aggregation_keeps_single_model_paper_value_context_only():
    collector = _load_collector()
    manifest = _load_manifest()
    profile_name = manifest["default_profile"]
    runs = [
        {
            "algorithm": "qmix",
            "map": "2s_vs_1sc",
            "seed": seed,
            "terminal_test_return_mean": value,
            "terminal_test_battle_won_mean": 0.8,
        }
        for seed, value in zip([1, 2, 3, 4, 5], [15.0, 16.0, 16.5, 17.0, 16.5])
    ]
    groups = collector.aggregate_groups(
        runs,
        manifest=manifest,
        profile_name=profile_name,
        bootstrap_samples=500,
        bootstrap_seed=7,
    )
    assert len(groups) == 1
    assert groups[0]["seed_complete"] is True
    assert groups[0]["n_seeds"] == 5
    assert groups[0]["comparison_status"] == "context_only_non_gateable"
    assert groups[0]["paper_reference"]["value"] == 16.22
    assert groups[0]["paper_reference"]["comparable_to_registered_five_seed_mean"] is False
    assert groups[0]["delta_from_paper_reference"] is None


def test_aggregation_bootstraps_terminal_win_rate_by_training_seed():
    collector = _load_collector()
    manifest = _load_manifest()
    profile_name = manifest["default_profile"]
    bootstrap_samples = 2_000
    bootstrap_seed = 17
    wins = [0.0, 0.25, 0.5, 0.75, 1.0]
    runs = [
        {
            "algorithm": "qmix",
            "map": "2s_vs_1sc",
            "seed": seed,
            "terminal_test_return_mean": 10.0 + seed,
            "terminal_test_battle_won_mean": win_rate,
        }
        for seed, win_rate in zip([1, 2, 3, 4, 5], wins)
    ]

    groups = collector.aggregate_groups(
        runs,
        manifest=manifest,
        profile_name=profile_name,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    assert len(groups) == 1
    group = groups[0]
    expected_seed = bootstrap_seed + sum(map(ord, "qmix2s_vs_1sc"))
    expected_interval = collector.bootstrap_mean_ci(
        wins,
        samples=bootstrap_samples,
        seed=expected_seed,
    )
    assert group["terminal_test_battle_won_mean"] == 0.5
    assert group["terminal_test_battle_won_bootstrap_95_ci"] == list(
        expected_interval
    )
    assert expected_interval[0] < 0.5 < expected_interval[1]


def test_launcher_never_claims_algorithm_original_and_saves_models():
    text = (PROJECT_ROOT / "scripts" / "run_stock_reproduction.ps1").read_text(encoding="utf-8")
    assert "EPyMARL-$($AlgName.ToUpperInvariant())" in text
    assert '"save_model=True"' in text
    assert "override_non_reproduction" in text
    assert "Assert-CleanPinnedCheckout" in text
    assert "warm_resume" in text
    assert "$EPyMARLMain" in text
    assert "$env:OPEN_SCORE_EPYMARL_WINDOWS_SAFE = \"1\"" in text
    assert "binary_sha256" in text
    assert '"--capture=$SacredCaptureMode"' in text
    assert "Registered stock reproduction requires a clean committed Open-SCORE worktree" in text
    assert "tracked manifest identical to its HEAD blob" in text
    assert "open_score_git = $OpenScoreProvenance" in text
    assert "Get-CheckpointEvidence" in text
    assert "Expected exactly one new checkpoint root" in text


def test_collector_rejects_missing_checkpoint_and_terminal_metric_misalignment(tmp_path):
    collector, manifest, profile_name, sacred_root, record_path = _synthetic_record(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop("produced_checkpoint_root")
    record.pop("produced_checkpoint")
    _write_json(record_path, record)
    metrics_path = Path(record["sacred_run_path"]) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["test_return_std"]["steps"][-1] -= 1
    _write_json(metrics_path, metrics)
    run, reasons = collector.validate_record(
        record_path,
        manifest=manifest,
        profile_name=profile_name,
        sacred_root=sacred_root,
    )
    assert run is None
    assert "missing_checkpoint" in reasons
    assert "missing_terminal_return_std" in reasons


def test_collector_rejects_checkpoint_hash_drift(tmp_path):
    collector, manifest, profile_name, sacred_root, record_path = _synthetic_record(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    checkpoint_file = Path(record["produced_checkpoint"]["latest_directory"]) / "agent.th"
    checkpoint_file.write_bytes(b"changed")
    run, reasons = collector.validate_record(
        record_path,
        manifest=manifest,
        profile_name=profile_name,
        sacred_root=sacred_root,
    )
    assert run is None
    assert "checkpoint_file_sha256_drift" in reasons


def test_collector_rejects_low_or_incomplete_checkpoint(tmp_path):
    collector, manifest, profile_name, sacred_root, record_path = _synthetic_record(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    original = Path(record["produced_checkpoint"]["latest_directory"])
    low = original.with_name("3000000")
    original.rename(low)
    (low / "mixer.th").unlink()
    checkpoint = collector.checkpoint_fingerprint(low.parent)
    assert checkpoint is not None
    record["produced_checkpoint"] = checkpoint
    _write_json(record_path, record)
    run, reasons = collector.validate_record(
        record_path,
        manifest=manifest,
        profile_name=profile_name,
        sacred_root=sacred_root,
    )
    assert run is None
    assert "checkpoint_below_terminal_threshold" in reasons
    assert "checkpoint_required_files_missing:mixer.th" in reasons


def test_committed_manifest_blob_hash_verification(tmp_path):
    git = r"D:\Software\Git\cmd\git.exe"
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run([git, "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run([git, "config", "core.autocrlf", "false"], cwd=repository, check=True)
    subprocess.run([git, "config", "user.email", "test@example.invalid"], cwd=repository, check=True)
    subprocess.run([git, "config", "user.name", "Test"], cwd=repository, check=True)
    manifest_path = repository / "manifest.json"
    manifest_path.write_text('{"version":1}\n', encoding="utf-8")
    subprocess.run([git, "add", "manifest.json"], cwd=repository, check=True)
    subprocess.run([git, "commit", "-m", "manifest"], cwd=repository, check=True, capture_output=True)
    commit = subprocess.run(
        [git, "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    collector = _load_collector()
    assert collector.committed_blob_sha256(repository, commit, "manifest.json") == collector.sha256_file(manifest_path)
    manifest_path.write_text('{"version":2}\n', encoding="utf-8")
    assert collector.committed_blob_sha256(repository, commit, "manifest.json") != collector.sha256_file(manifest_path)


def test_parallel_scheduler_selects_only_registered_primary_jobs():
    scheduler = _load_parallel_runner()
    manifest = _load_manifest()
    jobs = scheduler.select_jobs(
        manifest,
        profile_name=manifest["default_profile"],
        algorithm="qmix",
        map_name="2s_vs_1sc",
        seeds=[1, 3],
    )
    assert [(job.algorithm, job.map_name, job.seed) for job in jobs] == [
        ("qmix", "2s_vs_1sc", 1),
        ("qmix", "2s_vs_1sc", 3),
    ]
    scheduler_text = PARALLEL_RUNNER_PATH.read_text(encoding="utf-8")
    assert "--cpu-indices is required" in scheduler_text
    assert "share a physical core" in scheduler_text

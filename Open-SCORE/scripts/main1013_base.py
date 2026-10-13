#!/usr/bin/env python3
"""Six resumable HAD baseline runs with LEAF's information and sampling protocol.

plan/progress/report do not import HAD or construct a simulator. The only new
experiment report is OUTPUT/实验报告.md. Author-specific Adam/TD(lambda)/network
settings remain intact; task inputs and sampling controls are shared with LEAF.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
import gc
from datetime import datetime, timezone, timedelta
import importlib.util
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
from queue import Empty
import signal
import statistics
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

from leaf1009 import FINAL_CONFIGS, GROUPS, atomic_document, logical_devices, process_lock

PROFILE = "main1013_base"
REVISION = "main1013_base_last_action_v1"
METHODS, SEEDS = ("spectra", "transfqmix"), (0, 1, 2)
STEPS, QUOTA = 1_000_000, 300
VALIDATION_CONFIGS = ((4, 4, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3))
SCENE_START = {"eval": 9000, "confirm": 110000}
ARMS = {"eval": "standard", "confirm": "confirmation"}
EXPECTED_ENV = {"had_core_version": "had-workbench-2.1.0",
                "physics_protocol": "rebuild-calibrated-v3-r7-target-initialization"}
NAMES = {"spectra": "SPECTra", "transfqmix": "TransfQMix", "regir_nomem": "LEAF"}
CHINA = timezone(timedelta(hours=8))
TOTAL = len(FINAL_CONFIGS) * QUOTA
_MODULES = {}
AUTHOR_LINKS = {
    "SPECTra GRF": "https://raw.githubusercontent.com/funny-rl/SPECTra/ffababf6187216c9d16b2109ee8ef6fe5fdf1172/SPECTra_GRF/config/algs/spectra_qmix_plus.yaml",
    "SPECTra SMACv2": "https://raw.githubusercontent.com/funny-rl/SPECTra/ffababf6187216c9d16b2109ee8ef6fe5fdf1172/SPECTra_SMACv2/config/algs/ss_qmix.yaml",
    "TransfQMix": "https://raw.githubusercontent.com/mttga/pymarl_transformers/2ef0a0726f1f186097b4b560509ceb5a801fdafe/src/config/algs/transf_qmix.yaml",
}


def file_module(name, relative):
    """Reuse file utilities without importing open_score's HAD-dependent init."""
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, PROJECT / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def directory(options, method, seed):
    return options.output / "had" / method / "train" / f"seed_{seed}"


def records(options, stream):
    logging = file_module("main1013_logging", "open_score/utils/logging.py")
    return logging.iter_records(options.output / "had", stream, run="train")


def reference(options):
    """Canonical archive is read only; no weights or datasets are duplicated."""
    root = options.reference_output
    candidates = sorted((root / "had/regir_nomem/models").glob("*/seed_0/config.json"))
    preferred = [p for p in candidates if p.parent.parent.name == "main0928"]
    candidates = preferred or candidates
    if len(candidates) != 1:
        raise ValueError(f"Expected one final LEAF configuration at {root}/had/regir_nomem/models; found {len(candidates)}")
    cfg_path = candidates[0]
    cfg = read_json(cfg_path)
    if cfg.get("method") != "regir_nomem" or not cfg.get("entity_last_action"):
        raise ValueError("Reference must be the final 19-dimensional regir_nomem/LEAF")
    meta = read_json(root / "experiment.json", {})
    source = meta.get("sources", {}).get("main1009", {}).get("experiment.json", {})
    if not source and meta.get("profile") == "leaf1009":
        source = meta
    if not source:
        # The canonical archive keeps the current table in main and historical
        # protocol snapshots in its sibling backup, linked by archive_metadata.
        archive = read_json(root.parent / "backup/experiment.json", {})
        source = archive.get("sources", {}).get("main1009", {}).get("experiment.json", {})
    if any(source.get(k) != v for k, v in EXPECTED_ENV.items()):
        raise ValueError("LEAF main1009 reference lacks the agreed environment/physics identity")
    if ({tuple(c) for c in source.get("final_configs", [])} != set(FINAL_CONFIGS)
            or source.get("episodes_per_config") != QUOTA
            or source.get("confirmation_scene_start") != SCENE_START["confirm"]):
        raise ValueError("Reference formal/confirmation protocol differs from All24 x 300")
    shared_keys = ("batch_size_run", "buffer_size", "batch_size", "gamma", "grad_norm_clip",
                   "target_update_interval", "epsilon_start", "epsilon_finish", "epsilon_anneal_time",
                   "shaping_coef", "shaping_range", "reward_mode", "max_traj_len",
                   "obs_agent_id", "obs_last_action", "entity_last_action", "buffer_cpu_only", "buffer_opt_mem")
    shared = {k: cfg[k] for k in shared_keys}
    expected = dict(batch_size_run=8, buffer_size=5000, batch_size=32, gamma=.99,
                    grad_norm_clip=10, target_update_interval=200, epsilon_start=1.,
                    epsilon_finish=.05, epsilon_anneal_time=100000, shaping_coef=1.,
                    shaping_range=4000., reward_mode="damage", max_traj_len=-1,
                    obs_agent_id=False, obs_last_action=False, entity_last_action=True,
                    buffer_cpu_only=True, buffer_opt_mem=True)
    if shared != expected or int(cfg.get("t_max", 0)) != STEPS:
        raise ValueError("Saved LEAF shared controls differ from the approved fixed protocol")
    env_args = copy.deepcopy(cfg.get("env_args", {}))
    env_args.update(train_dist="mixed_le10", max_steps=100, blue_upper="reactive", blue_lower="rush",
                    target_initialization="random", spatial_dim=2)
    # Values saved explicitly by LEAF may not be silently replaced by defaults.
    for key, value in (cfg.get("env_args") or {}).items():
        if key in env_args and env_args[key] != value:
            raise ValueError(f"LEAF environment setting differs at {key}")
    shared.update(training_iters=8, env_args=env_args, t_max=STEPS)
    return cfg_path, shared


def method_specs():
    """Read declared configurations, without importing model or environment code."""
    import yaml
    spectra = yaml.safe_load((PROJECT / "configs/spectra.yaml").read_text(encoding="utf-8-sig"))
    tree = ast.parse((PROJECT / "open_score/algos/__init__.py").read_text(encoding="utf-8"))
    transf = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAIN_OVERRIDES" for t in node.targets):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value == "transfqmix":
                    transf = ast.literal_eval(value)
    if transf is None or not spectra.get("entity_last_action") or not transf.get("entity_last_action"):
        raise ValueError("Both HAD baseline defaults must enable the real last-action input path")
    return {"spectra": spectra, "transfqmix": transf}


def manifest(options):
    cfg_path, shared = reference(options)
    return dict(profile=PROFILE, implementation_revision=REVISION, methods=list(METHODS), seeds=list(SEEDS),
                shared_controls=shared, method_specifications=method_specs(),
                training_steps=STEPS, validation_points=50, validation_configs=VALIDATION_CONFIGS,
                validation_episodes_per_config=25, validation_scenes=[9500, 9599],
                final_configs=FINAL_CONFIGS, episodes_per_config=QUOTA, formal_scenes=[9000, 9299],
                confirmation_scenes=[110000, 110299], official_checkpoint="final",
                reference=dict(method="regir_nomem", config=str(cfg_path.relative_to(options.reference_output)).replace("\\", "/"),
                               evaluation_source="main1009", summary="had/loop_summary.csv",
                               training_environment_provenance="historical training version was not fully recorded"),
                author_sources=AUTHOR_LINKS, **EXPECTED_ENV)


def canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def initialize(options):
    if options.output == options.reference_output or options.reference_output in options.output.parents:
        raise ValueError("New results must not overwrite or be nested inside the LEAF archive")
    expected = canonical(manifest(options))
    with process_lock(options.output / ".experiment.lock", blocking=True):
        old = read_json(options.output / "experiment.json")
        if old is not None and old != expected:
            raise ValueError("Frozen experiment/configuration changed; existing progress has been preserved")
        (options.output / "had").mkdir(parents=True, exist_ok=True)
        if old is None:
            if any((options.output / "had").glob("*/train/seed_*/config.json")):
                raise ValueError("Existing model runs without this experiment's frozen metadata; refusing overwrite")
            atomic_document(options.output / "experiment.json", expected)
    return expected


def check_environment():
    try:
        from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
    except ImportError as error:
        raise RuntimeError("Install the retained had-env version exposing had_env.core.version before running; plan/progress/report remain available") from error
    actual = dict(had_core_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL)
    if actual != EXPECTED_ENV:
        raise ValueError(f"Environment mismatch: expected {EXPECTED_ENV}, found {actual}; no training started")


def configuration(options, method, seed):
    _, shared = reference(options)
    return dict(shared, method=method, name=method, seed=seed, profile=PROFILE,
                implementation_revision=REVISION, output=str(options.output), env="had", run="train",
                external_report=True, skip_final_eval=True, use_cuda=True, device="cuda",
                progress_interval=10, resume_interval=600)


def qualified_final(options, method, seed):
    """Qualification reads retained weights once per worker, never a file hash."""
    import torch
    path = directory(options, method, seed) / "final.pt"
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg, progress = saved["config"], saved["progress"]
    expected = configuration(options, method, seed)
    for key, value in expected.items():
        if key in ("output", "progress_interval", "resume_interval"):
            continue
        if cfg.get(key) != value:
            raise ValueError(f"Final checkpoint differs at {key}: {path}")
    if (progress.get("status") != "completed" or int(progress.get("t_env", 0)) != STEPS
            or int(progress.get("next_eval", 0)) != 51 or not saved.get("artifact_id")):
        raise ValueError(f"Final checkpoint has not completed the fixed training/validation budget: {path}")
    del saved["networks"]
    return saved


def train_one(options, method, seed, stop):
    from open_score.algos import train, resume_files
    path = directory(options, method, seed)
    if qualified_final(options, method, seed) is not None:
        return True
    recoverable = any(p.exists() for p in resume_files(path))
    if recoverable and not options.resume:
        raise FileExistsError(f"Use --resume to continue {path}")
    if not recoverable and any((path / f).exists() for f in ("config.json", "best.pt")):
        raise ValueError(f"Run has no recoverable snapshot; existing artifacts preserved: {path}")
    cfg = configuration(options, method, seed)
    cfg.update(resume=recoverable, _stop_event=stop)
    result = Path(train(method, cfg))
    return result.name == "final.pt" and qualified_final(options, method, seed) is not None


def config_key(config):
    return tuple(int(config[k]) for k in ("N_R", "N_B", "K")) if isinstance(config, dict) else tuple(config)


def stage_key(stage):
    # The existing trainer uses eval_completed for inline ID validation.
    # Keep formal-evaluation counters separate in the same resource record.
    return "formal" if stage == "eval" else stage


def evaluation_rows(options, method, seed, stage, artifact_id):
    rows = {}
    for row in records(options, "episodes"):
        if row.get("method") != method or row.get("seed") != seed or row.get("phase") != "final_eval" or row.get("arm") != ARMS[stage]:
            continue
        if row.get("protocol") != REVISION or row.get("checkpoint_id") != artifact_id or row.get("version") != PROFILE:
            raise ValueError("Evaluation contains another checkpoint/protocol; existing rows preserved")
        key = (config_key(row["config"]), int(row["episode_seed"]))
        if key[0] not in FINAL_CONFIGS or not SCENE_START[stage] <= key[1] < SCENE_START[stage] + QUOTA:
            raise ValueError("Evaluation record is outside the frozen scene bank")
        if key in rows or not math.isfinite(float(row["D"])):
            raise ValueError("Duplicate or non-finite evaluation record")
        rows[key] = row
    return rows


def resource_update(path, **values):
    state = read_json(path, {})
    state.update(values, updated_at=time.time(), pid=os.getpid())
    atomic_document(path, state)
    return state


def evaluate_one(options, method, seed, stage, stop):
    from open_score.algos import load_policy
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import ExperimentLogger
    saved = qualified_final(options, method, seed)
    if saved is None:
        raise ValueError(f"Completed final.pt is required for {method}/{seed}")
    artifact = saved["artifact_id"]
    existing = evaluation_rows(options, method, seed, stage, artifact)
    if existing and not options.resume:
        raise FileExistsError("Existing evaluation rows; use --resume to skip committed scenes")
    path = directory(options, method, seed)
    resource = path / "resource.json"
    completed, sum_d = len(existing), sum(float(r["D"]) for r in existing.values())
    measured_seconds = sum(float(r.get("episode_wall_seconds") or 0) for r in existing.values())
    counter = stage_key(stage)
    previous = read_json(resource, {})
    if len(existing) == TOTAL:
        resource_update(resource, train_complete=True, **{f"{counter}_completed": TOTAL, f"{counter}_D": sum_d / TOTAL,
                                                        f"{counter}_seconds": measured_seconds})
        return True
    policy = load_policy(method, path / "final.pt")
    policy.set_device("cuda")
    # Configuration says 19 only after construction really produced 19-wide projections.
    width = policy.mac.agent.own_embedding.in_features if method == "spectra" else policy.mac.agent.feat_embedding.in_features
    if width != 19:
        raise ValueError(f"Actual {method} actor input width is {width}, expected 19")
    mixer_width = (policy.mixer.hyper_w1.agent_embedding.in_features if method == "spectra"
                   else policy.mixer.feat_embedding.in_features)
    if mixer_width != 19:
        raise ValueError(f"Actual {method} mixer input width is {mixer_width}, expected 19")
    name = f"{PROFILE}_{method}_{seed}_{stage}_{os.getpid()}"
    register_end_to_end_policy("red", name, name, lambda _: policy)
    logger = ExperimentLogger(options.output, method, seed, "train", env="had")
    tag = f"final@{STEPS}" + ("/confirmation" if stage == "confirm" else "")
    stage_started, heartbeat = time.monotonic(), 0.
    prefix = {k: previous[k] for k in ("formal_completed", "formal_D", "formal_seconds", "confirm_completed", "confirm_D", "confirm_seconds") if k in previous}
    def publish(status="running"):
        values = dict(prefix, env="had", method=method, seed=seed, train_complete=True, status=status,
                      current_phase=stage, stage_started_at=time.time() - (time.monotonic() - stage_started),
                      physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]) if os.environ["CUDA_VISIBLE_DEVICES"].isdigit() else os.environ["CUDA_VISIBLE_DEVICES"],
                      checkpoint_id=artifact, t_env=STEPS, budget_steps=STEPS,
                      **{f"{counter}_completed": completed, f"{counter}_D": sum_d / completed if completed else None,
                         f"{counter}_seconds": measured_seconds, f"{counter}_total": TOTAL})
        resource_update(resource, **values)
        logger.progress(**values)
    publish()
    for config in FINAL_CONFIGS:
        for scene in range(SCENE_START[stage], SCENE_START[stage] + QUOTA):
            if stop.is_set() or (options.output / "stop.request").exists():
                publish("stopped")
                return False
            if (config, scene) in existing:
                continue
            started = time.perf_counter()
            result = run_episode(red=config[0], blue=config[1], targets=config[2], seed=scene,
                                 red_strategy={"architecture": "end_to_end", "policy": name},
                                 blue_strategy={"architecture": "hierarchical", "layers": {"grouping": "reactive", "control": "rush"}},
                                 max_steps=100, record=False, task_mode="damage", spatial_dim=2,
                                 target_initialization="random", diagnostics=True, record_events=False, retain_trajectory=False)
            duration = time.perf_counter() - started
            if result.get("physics_protocol") != EXPECTED_ENV["physics_protocol"]:
                raise ValueError("Episode physics identity changed; result was not committed")
            row = dict(result["episode_summary"], phase="final_eval", config=dict(zip(("N_R", "N_B", "K"), config)),
                       episode_seed=scene, eval_point=50, t_env=STEPS, checkpoint=tag, checkpoint_id=artifact,
                       arm=ARMS[stage], protocol=REVISION, episode_wall_seconds=duration)
            logger.episodes([row])
            completed += 1
            sum_d += float(row["D"])
            measured_seconds += duration
            if time.monotonic() - heartbeat >= 10:
                publish()
                print(f"{method}/{seed} {stage} {completed}/{TOTAL} D~{sum_d/completed:.4f}", flush=True)
                heartbeat = time.monotonic()
    publish("completed")
    del policy
    return completed == TOTAL


def worker(options, method, seed, gpu, stop, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # The train API temporarily installs its own handlers and restores these
    # afterward, so evaluation also finishes its current episode on SIGTERM.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    path = directory(options, method, seed)
    path.mkdir(parents=True, exist_ok=True)
    try:
        with process_lock(path / ".model.lock") as claimed:
            if not claimed:
                raise RuntimeError(f"Model task is already running: {method}/{seed}")
            with (path / "console.log").open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
                check_environment()
                resource_update(path / "resource.json", method=method, seed=seed, env="had", status="starting",
                                current_phase="train" if options.stage in ("run", "train") else options.stage,
                                physical_gpu=gpu, stage_started_at=time.time())
                if options.stage in ("run", "train") and not train_one(options, method, seed, stop):
                    results.put((method, seed, "stopped", None))
                    return
                if qualified_final(options, method, seed) is None:
                    raise ValueError(f"Completed final.pt is required for {method}/{seed}")
                resource_update(path / "resource.json", train_complete=True)
                # Runner callbacks retain a training-cycle closure. Release it
                # before loading the much larger-roster evaluation policy.
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                stages = ("eval", "confirm") if options.stage == "run" else (options.stage,) if options.stage in SCENE_START else ()
                for stage in stages:
                    if stop.is_set() or not evaluate_one(options, method, seed, stage, stop):
                        results.put((method, seed, "stopped", None))
                        return
                resource_update(path / "resource.json", status="completed", current_phase="done")
                results.put((method, seed, "completed", None))
    except BaseException:
        error = traceback.format_exc()
        resource_update(path / "resource.json", status="failed", error=error)
        results.put((method, seed, "failed", error))


def run(options):
    check_environment()  # No model/output initialization when the environment is wrong.
    initialize(options)
    devices = logical_devices(options.devices)
    slots = tuple(gpu for _ in range(options.jobs_per_gpu) for gpu in devices)[:options.max_jobs]
    ctx = get_context("spawn")
    stop, results = ctx.Event(), ctx.Queue()
    waiting = [(method, seed) for seed in SEEDS for method in METHODS]
    active, finished = {}, {}
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda *_: stop.set())
    state_path = options.output / "scheduler.main1013_base.json"
    with process_lock(options.output / ".scheduler.lock") as claimed:
        if not claimed:
            raise RuntimeError("This output already has a running scheduler")
        if (options.output / "stop.request").exists():
            if not options.resume:
                raise ValueError("Stopped output requires --resume")
            (options.output / "stop.request").unlink()
        last_publish, last_render, screen_height = 0., 0., 0
        histories = defaultdict(lambda: deque(maxlen=120))
        try:
            refresh_report(options)
            while waiting or active:
                if (options.output / "stop.request").exists():
                    stop.set()
                if not stop.is_set():
                    for slot, gpu in enumerate(slots):
                        if slot not in active and waiting:
                            method, seed = waiting.pop(0)
                            child = ctx.Process(target=worker, args=(options, method, seed, gpu, stop, results))
                            child.start()
                            active[slot] = dict(child=child, method=method, seed=seed, gpu=gpu, started_at=time.time())
                try:
                    method, seed, status, error = results.get(timeout=.5)
                    finished[(method, seed)] = dict(status=status, error=error)
                    if error:
                        print(error, file=sys.stderr, flush=True)
                except Empty:
                    pass
                for slot, item in list(active.items()):
                    if not item["child"].is_alive():
                        item["child"].join()
                        key = (item["method"], item["seed"])
                        # Queue feeder delivery can lag process exit; drain before deciding.
                        while True:
                            try:
                                method, seed, status, error = results.get_nowait()
                                finished[(method, seed)] = dict(status=status, error=error)
                            except Empty:
                                break
                        if key not in finished:
                            finished[key] = dict(status="failed", error=f"Worker exited without completion, exit={item['child'].exitcode}")
                        del active[slot]
                        refresh_report(options)
                now = time.monotonic()
                if now - last_publish >= 3:
                    payload = dict(profile=PROFILE, stage=options.stage, pid=os.getpid(), updated_at=time.time(),
                                   status="stopping" if stop.is_set() else "running", slots=len(slots),
                                   active=[{k: v for k, v in item.items() if k != "child"} | {"pid": item["child"].pid} for item in active.values()],
                                   pending=[dict(method=m, seed=s) for m, s in waiting],
                                   finished=[dict(method=m, seed=s, **value) for (m, s), value in finished.items()])
                    atomic_document(state_path, payload)
                    last_publish = now
                if sys.stdout.isatty() and now - last_render >= 3:
                    screen_height = render_screen(progress_lines(options, histories), screen_height)
                    last_render = now
                elif not sys.stdout.isatty() and now - last_render >= 30:
                    render_screen(progress_lines(options, histories), 0)
                    last_render = now
                if stop.is_set() and not active:
                    break
            status = ("failed" if any(r["status"] == "failed" for r in finished.values()) else
                      "stopped" if stop.is_set() or any(r["status"] == "stopped" for r in finished.values()) else
                      "completed" if len(finished) == 6 and all(r["status"] == "completed" for r in finished.values()) else "failed")
            atomic_document(state_path, dict(profile=PROFILE, stage=options.stage, status=status, updated_at=time.time(),
                                             pid=os.getpid(), slots=len(slots), active=[], pending=[dict(method=m, seed=s) for m, s in waiting],
                                             finished=[dict(method=m, seed=s, **value) for (m, s), value in finished.items()]))
            refresh_report(options)
            print(f"{PROFILE}: {status}; report={options.output / '实验报告.md'}", flush=True)
            return 130 if status == "stopped" else 1 if status == "failed" else 0
        finally:
            if active:
                stop.set()
                for item in active.values():
                    item["child"].join()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def short_time(seconds):
    if seconds is None:
        return "待估"
    value = max(0, int(seconds))
    days, rest = divmod(value, 86400)
    hours, minutes = rest // 3600, rest % 3600 // 60
    return f"{days}d{hours}h" if days else f"{hours}h{minutes}m" if hours else f"{minutes}m"


def eta_range(history, completed, total, measured_seconds=None):
    if completed >= total:
        return (0., 0.)
    rates = []
    if len(history) > 1:
        now, value = history[-1]
        for window in (60, 300):
            samples = [(t, n) for t, n in history if now - t <= window]
            if samples and now - samples[0][0] >= 10 and value > samples[0][1]:
                rates.append((value - samples[0][1]) / (now - samples[0][0]))
    if measured_seconds and completed:
        rates.append(completed / measured_seconds)
    if not rates:
        return None
    left = max(0, total - completed)
    return left / max(rates), left / min(rates)


def table(rows):
    import unicodedata
    def width(value):
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(value))
    sizes = [max(width(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for n, row in enumerate(rows):
        lines.append("  ".join(str(cell) + " " * (sizes[i] - width(cell)) for i, cell in enumerate(row)).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in sizes))
    return lines


def progress_lines(options, histories):
    scheduler = read_json(options.output / "scheduler.main1013_base.json", {})
    fresh = time.time() - scheduler.get("updated_at", 0) < 30
    active = {(r["method"], r["seed"]): r for r in scheduler.get("active", [])} if fresh else {}
    resources = {(m, s): read_json(directory(options, m, s) / "resource.json", {}) for m in METHODS for s in SEEDS}
    now = time.monotonic()
    completed = {"train": 0, "eval": 0, "confirm": 0}
    units = {"train": 0, "eval": 0, "confirm": 0}
    running = defaultdict(int)
    rows = [["模型/种子", "当前", "设备", "训练", "正式", "确认", "当前用时", "剩余≈", "验证D", "正式D", "确认D"]]
    overall_eta = []
    for key, r in resources.items():
        m, s = key
        counts = {"train": min(STEPS, int(r.get("t_env", 0))), "eval": int(r.get("formal_completed", 0)), "confirm": int(r.get("confirm_completed", 0))}
        for stage in completed:
            total = STEPS if stage == "train" else TOTAL
            units[stage] += counts[stage]
            completed[stage] += bool((r.get("train_complete") or r.get("status") == "completed" and counts[stage] >= total) if stage == "train" else counts[stage] == total)
            histories[(key, stage)].append((now, counts[stage]))
        phase = r.get("current_phase", "train")
        status = r.get("status", "pending")
        is_live = key in active and status not in ("completed", "failed", "stopped")
        if is_live:
            running["train" if phase in ("collecting", "updating", "validating", "initializing", "idle", "train") else phase] += 1
        label = ("验证" if phase == "validating" else "训练" if phase in ("collecting", "updating", "initializing", "idle", "train") else "正式" if phase == "eval" else "确认" if phase == "confirm" else "完成") if is_live else {"failed": "失败", "stopped": "停止", "completed": "完成"}.get(status, "等待" if not r else "离线")
        remaining = []
        for stage in ("train", "eval", "confirm"):
            total = STEPS if stage == "train" else TOTAL
            seconds = (r.get("training_seconds", 0) + r.get("validation_seconds", 0)) if stage == "train" else r.get(f"{stage_key(stage)}_seconds")
            remaining.append(eta_range(histories[(key, stage)], counts[stage], total, seconds))
        estimate = tuple(sum(x[i] for x in remaining) for i in (0, 1)) if all(x is not None for x in remaining) else None
        if estimate is not None:
            overall_eta.append(estimate)
        current_stage = phase if phase in SCENE_START else "train"
        current_eta = remaining[("train", "eval", "confirm").index(current_stage)]
        eta = "待估" if current_eta is None else f"{short_time(current_eta[0])}~{short_time(current_eta[1])}"
        device = str(active.get(key, {}).get("gpu", r.get("physical_gpu", "--")))
        def d(stage):
            value = r.get(f"{stage_key(stage)}_D")
            return "--" if value is None else ("~" if counts[stage] < TOTAL else "") + f"{value:.3f}"
        duration = (time.time() - r.get("stage_started_at", active[key]["started_at"])) if key in active else r.get("elapsed_seconds", 0)
        rows.append([f"{NAMES[m]}/{s}", label, "--" if device == "--" else f"GPU{device}",
                     f"{100*counts['train']/STEPS:.1f}%", f"{100*counts['eval']/TOTAL:.1f}%", f"{100*counts['confirm']/TOTAL:.1f}%",
                     short_time(duration), eta, "--" if r.get("latest_validation_D") is None else f"{r['latest_validation_D']:.3f}", d("eval"), d("confirm")])
    overview = [["阶段", "完成/模型", "运行进程", "剩余模型", "进度"]]
    for stage, title in (("train", "训练"), ("eval", "正式评估"), ("confirm", "确认评估")):
        total = 6 * (STEPS if stage == "train" else TOTAL)
        overview.append([title, f"{completed[stage]}/6", str(running[stage]), str(6-completed[stage]), f"{100*units[stage]/total:.1f}%"])
    eta = "待估" if len(overall_eta) != 6 else f"{short_time(max(x[0] for x in overall_eta))}~{short_time(max(x[1] for x in overall_eta))}"
    status = scheduler.get("status", "未启动")
    if not fresh and status in ("running", "stopping"):
        status = "状态待刷新"
    stamp = datetime.now(CHINA).strftime("%m-%d %H:%M:%S")
    return [f"{PROFILE}  {stamp}  {status}  全部剩余≈{eta}", "", *table(overview), "", *table(rows), "D↓  ~=当前均值  --=待采样；行内剩余为当前阶段；关闭此面板不停止训练"]


def render_screen(lines, previous):
    # Same terminal cursor rewrite used by the existing training status board.
    if previous and sys.stdout.isatty():
        sys.stdout.write(f"\x1b[{previous}F\x1b[J")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    return len(lines) if sys.stdout.isatty() else 0


def progress(options):
    histories, height = defaultdict(lambda: deque(maxlen=120)), 0
    try:
        while True:
            height = render_screen(progress_lines(options, histories), height)
            if options.watch_seconds == 0:
                return 0
            time.sleep(options.watch_seconds)
    except KeyboardInterrupt:
        return 0


def leaf_summary(options):
    result = {}
    path = options.reference_output / "had/loop_summary.csv"
    if not path.exists():
        return result
    with path.open(encoding="utf-8", newline="") as stream:
        for r in csv.DictReader(stream):
            if (r["method"] == "regir_nomem" and r["split"] in SCENE_START and r["arm"] == "original"
                    and r.get("archive_source") == "main1009" and r.get("protocol") == "leaf1009_loop_candidates_v2"
                    and int(r["episodes"]) == QUOTA and int(r["seed"]) in SEEDS):
                key = (r["split"], int(r["seed"]), tuple(json.loads(r["config"])))
                value = float(r["D"])
                if key in result or not math.isfinite(value):
                    raise ValueError("LEAF reference has duplicate/non-finite summary cells")
                result[key] = value
    return result


def refresh_report(options):
    """One report; plots contain only observed values, never pending zeros."""
    initialize(options)
    with process_lock(options.output / ".report.lock", blocking=True):
        cells = defaultdict(dict)
        validation, training = defaultdict(list), defaultdict(list)
        for row in records(options, "episodes"):
            if row.get("version") != PROFILE or row.get("method") not in METHODS or row.get("seed") not in SEEDS:
                continue
            key = (row["method"], row["seed"])
            if row.get("phase") == "train_eval":
                validation[key].append(row)
            elif row.get("phase") == "final_eval" and row.get("protocol") == REVISION:
                stage = next((s for s, a in ARMS.items() if a == row.get("arm")), None)
                if stage is not None:
                    cell = (row["method"], stage, row["seed"], config_key(row["config"]))
                    scene = int(row["episode_seed"])
                    if scene in cells[cell]:
                        raise ValueError("Duplicate committed evaluation scene in report data")
                    cells[cell][scene] = float(row["D"])
        for row in records(options, "learning"):
            if row.get("version") == PROFILE and row.get("method") in METHODS and row.get("seed") in SEEDS:
                if row.get("train_D") is not None:
                    training[row["method"], row["seed"]].append((row["t_env"], row["train_D"]))
        ref = leaf_summary(options)
        scores = {}
        for (method, stage, seed, config), episodes in cells.items():
            if set(episodes) == set(range(SCENE_START[stage], SCENE_START[stage] + QUOTA)) and all(math.isfinite(v) for v in episodes.values()):
                scores[method, stage, seed, config] = statistics.mean(episodes.values())
        scores.update({("regir_nomem", stage, seed, config): value for (stage, seed, config), value in ref.items()})
        group_specs = dict(GROUPS, All24=FINAL_CONFIGS)
        means = {}
        for method in (*METHODS, "regir_nomem"):
            for stage in SCENE_START:
                for group, configs in group_specs.items():
                    values = [statistics.mean(scores[method, stage, seed, config] for config in configs)
                              for seed in SEEDS if all((method, stage, seed, config) in scores for config in configs)]
                    if len(values) == 3:
                        means[method, stage, group] = (statistics.mean(values), statistics.stdev(values))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figures = options.output / "had/figures"
        figures.mkdir(parents=True, exist_ok=True)
        charts = []
        if means:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
            groups = list(group_specs)
            for ax, stage in zip(axes, SCENE_START):
                available = [m for m in ("regir_nomem", *METHODS) if any((m, stage, g) in means for g in groups)]
                w = .8 / max(1, len(available))
                for j, method in enumerate(available):
                    x, y, e = [], [], []
                    for i, group in enumerate(groups):
                        if (method, stage, group) in means:
                            avg, sd = means[method, stage, group]
                            x.append(i - .4 + w * (j + .5)); y.append(avg); e.append(sd)
                    ax.bar(x, y, width=w, yerr=e, capsize=3, label=NAMES[method])
                ax.set_xticks(range(len(groups)), groups, rotation=20)
                ax.set_title("Formal (9000-9299)" if stage == "eval" else "Confirmation (110000-110299)")
                ax.set_ylabel("Target damage D (lower is better)")
                if available:
                    ax.legend()
                ax.grid(axis="y", alpha=.2)
            fig.suptitle("Observed results only; LEAF = retained main1009 reference; error bars = 3-seed SD")
            fig.tight_layout()
            fig.savefig(figures / "comparison.png", dpi=160)
            plt.close(fig)
            charts.append("![正式与确认D比较](had/figures/comparison.png)")
        if training or validation:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            for (method, seed), values in training.items():
                values.sort()
                axes[0].plot([v[0] for v in values], [v[1] for v in values], label=f"{NAMES[method]}/{seed}", alpha=.7)
            for (method, seed), values in validation.items():
                by_point = defaultdict(dict)
                for row in values:
                    by_point[row["eval_point"]][config_key(row["config"]), row["episode_seed"]] = row
                points = [(max(r["t_env"] for r in group.values()), statistics.mean(float(r["D"]) for r in group.values()))
                          for group in by_point.values() if len(group) == 100]
                points.sort()
                axes[1].plot([v[0] for v in points], [v[1] for v in points], label=f"{NAMES[method]}/{seed}")
            for ax, title in zip(axes, ("Training batches", "Complete ID validation points")):
                ax.set_title(title); ax.set_xlabel("Physical environment steps"); ax.set_ylabel("D"); ax.grid(alpha=.2)
                if ax.lines:
                    ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(figures / "learning.png", dpi=160); plt.close(fig)
            charts.append("![训练与验证曲线](had/figures/learning.png)")
        resources = {(m, s): read_json(directory(options, m, s) / "resource.json", {}) for m in METHODS for s in SEEDS}
        finished = sum(bool(r.get("train_complete")) for r in resources.values())
        text = ["# main_1013_base：HAD基线输入对齐重训", "", f"更新：{datetime.now(CHINA).strftime('%Y-%m-%d %H:%M:%S')}（北京时间）。",
                "", f"当前训练完成 {finished}/6。未运行、停止、失败和未满配额的结果均不记为正式完成。",
                "", "## 输入差异与公平性设置", "",
                "原LEAF/REFIL输入为10维实体基础特征加9维红方上一动作；旧SPECTra/TransfQMix仅10维。新版本在actor、online/target mixer及冻结评估统一补齐19维，首步/蓝方/目标动作段为零。时序GRU/token保留，旧10维权重不续训。",
                "", "| 项目 | 冻结设置 |", "|---|---|",
                "| 方法与种子 | SPECTra、TransfQMix；0/1/2，共6模型 |",
                "| 训练 | 每模型1M物理步；N∈{4,6,8,10}，K∈{1,2,3}；8环境；每批8次更新 |",
                "| 共同控制 | replay5000，batch32，γ0.99，clip10，target每200局，ε1→0.05/100k |",
                "| 奖励/对手 | damage；势函数塑形β1、范围4000；reactive/rush；100步；二维随机目标 |",
                "| 方法自身设置 | Adam、lr0.001、Double-Q、TD(λ0.6)、半MSE；原GRU/SAQA或Transformer/token结构 |",
                "| 验证 | 50点；每点4配置×25局；9500–9599；best仅保留，不替代final |",
                "| 正式/确认 | All24各300局；9000–9299 / 110000–110299；每模型每阶段7200局 |",
                f"| 环境 | {EXPECTED_ENV['had_core_version']} / {EXPECTED_ENV['physics_protocol']} |",
                "", "采用共同HAD任务/观测/采样控制，保留作者关键结构及优化参数。这不是完整原论文复现或充分调参声明。SPECTra原GRF探索退火500k，本次共享100k；官方SMACv2也有100k配置。SPECTra原GRF每采集8局更新一次，此处8次；TransfQMix原单进程每局一次，此处按8局批次执行8次。所有差异在运行前固定，不按测试成绩调整。",
                "", "原作者配置：" + "、".join(f"[{k}]({v})" for k, v in AUTHOR_LINKS.items()) + "。",
                "", "## 数据覆盖与实际结果", "", "| 模型/种子 | 训练状态 | 正式局数/7200 | 确认局数/7200 |", "|---|---|---|---|"]
        for m in METHODS:
            for seed in SEEDS:
                r = resources[m, seed]
                counts = [sum(len(v) for (method, split, s, _), v in cells.items() if method == m and split == stage and s == seed) for stage in SCENE_START]
                text.append(f"| {NAMES[m]}/{seed} | {'完成' if r.get('train_complete') else r.get('status', '未启动')} | {counts[0]} | {counts[1]} |")
        text += ["", "仅三训练种子、各配置均满300局时汇总均值±种子SD；缺失单元显示待完成。D是累计目标损伤，不是失败率或胜率。", "",
                 "| 方法 | 阶段 | ID | K-OOD | N-OOD | Joint-OOD | All24 |", "|---|---|---|---|---|---|---|"]
        for m in ("regir_nomem", *METHODS):
            for stage in SCENE_START:
                vals = [f"{means[m,stage,g][0]:.4f}±{means[m,stage,g][1]:.4f}" if (m,stage,g) in means else "待完成" for g in group_specs]
                text.append(f"| {NAMES[m]}{'（归档参照）' if m == 'regir_nomem' else ''} | {'正式' if stage == 'eval' else '确认'} | " + " | ".join(vals) + " |")
        text += ["", *charts, "", "图中只包含真实已完成数据；新基线未运行时，比较图仅展示LEAF归档参照。训练曲线在真实训练数据产生后更新。",
                 "", "## LEAF参照与结论边界", "",
                 "最终LEAF为本地regir_nomem。优先引用main1009的original正式/确认单元（每配置、每种子300局），引用现有数据和模型，不复制目录。旧训练未完整保存环境版本，不能反向证明所有历史训练物理细节一致。归档仅有配置级汇总时，不补造逐局显著性检验。",
                 "", "输入不一致是确定的公平性缺口。补齐后性能如何以本次真实结果为准；训练未完成时不形成优劣结论，不保证改善或声称排除了所有超参数敏感性。",
                 "", "## SSH运行与进度", "", "在远程Open-SCORE目录使用：", "", "```bash",
                 "mkdir -p outputs/main_1013_base", "nohup python -u scripts/main1013_base.py run --devices 0,1 --jobs-per-gpu 3 --max-jobs 6 --resume > outputs/main_1013_base/run.log 2>&1 &",
                 "python scripts/main1013_base.py progress --watch-seconds 3", "python scripts/main1013_base.py stop", "```", "",
                 "每卡最多3个模型任务，总计最多6个，训练/正式/确认共用槽位。--resume恢复完整批次并跳过已提交场景。停止在批次/对局边界保存；OOM不静默改batch、预算或模型。progress只读，Ctrl+C仅关闭面板。没有实测速率时ETA显示待估。",
                 "", "远程需保留只读参照文件：outputs/main/had/regir_nomem/models/main0928/seed_0/config.json、outputs/main/experiment.json、outputs/main/had/loop_summary.csv，以及归档协议元数据outputs/backup/experiment.json。不需要复制LEAF模型权重。--reference-output可指定同结构的参照根目录。",
                 "", "单独阶段：plan、train、eval、confirm、report；所有阶段默认同一输出。未新增GRU重置、短跑、额外种子或超参数搜索。"]
        target = options.output / "实验报告.md"
        pending = target.with_suffix(".md.pending")
        pending.write_text("\n".join(text) + "\n", encoding="utf-8")
        os.replace(pending, target)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "run", "train", "eval", "confirm", "progress", "report", "stop"))
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/main_1013_base")
    parser.add_argument("--reference-output", type=Path, default=PROJECT / "outputs/main")
    parser.add_argument("--devices", default="0,1", help="Logical CUDA indices within inherited CUDA_VISIBLE_DEVICES")
    parser.add_argument("--jobs-per-gpu", type=int, default=3)
    parser.add_argument("--max-jobs", type=int, default=6)
    parser.add_argument("--watch-seconds", type=float, default=3)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    options.output, options.reference_output = options.output.resolve(), options.reference_output.resolve()
    if not 1 <= options.jobs_per_gpu <= 3 or not 1 <= options.max_jobs <= 6:
        parser.error("At most 3 tasks/GPU and 6 total tasks are permitted")
    if not math.isfinite(options.watch_seconds) or options.watch_seconds < 0:
        parser.error("--watch-seconds must be finite and nonnegative")
    try:
        if options.stage == "plan":
            value = manifest(options)
            value.update(output=str(options.output), devices=options.devices, jobs_per_gpu=options.jobs_per_gpu,
                         max_jobs=options.max_jobs, total_models=6, formal_episodes=6*TOTAL, confirmation_episodes=6*TOTAL)
            print(json.dumps(canonical(value), ensure_ascii=False, indent=2))
            return 0
        if options.stage == "progress":
            return progress(options)
        if options.stage == "report":
            refresh_report(options)
            print(options.output / "实验报告.md")
            return 0
        if options.stage == "stop":
            metadata = read_json(options.output / "experiment.json", {})
            if metadata.get("profile") != PROFILE:
                raise ValueError("No main1013_base experiment at the selected output")
            (options.output / "stop.request").write_text("Finish the current batch/episode and preserve progress.\n", encoding="utf-8")
            print("Stop requested; current batches/episodes will finish and remain resumable.")
            return 0
        return run(options)
    except (ValueError, RuntimeError, FileExistsError, ImportError, OSError) as error:
        print(f"{PROFILE}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

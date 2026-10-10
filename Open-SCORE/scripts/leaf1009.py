#!/usr/bin/env python3
"""Cloud runner for five LEAF loop candidates, retained references and legacy tiers.

plan is read-only and imports no simulator. Candidate training uses 1M steps,
seeds 0/1/2. ID calibration and family selection freeze before OOD evaluation.
Execution defaults to all visible logical CUDA devices, four jobs per GPU.
Report updates only the candidate section of the existing sole experiment report.
"""
from __future__ import annotations

import argparse
import copy
import csv
import errno
import math
import random
import statistics
import time
import ast
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import importlib.util
from multiprocessing import get_context
import os
from pathlib import Path
from queue import Empty
import signal
import socket
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
PROFILE = "leaf1009"
LEGACY_REVISION = "leaf1009_input_conditioned_v1"
REVISION = "leaf1009_loop_candidates_v2"
MECHANISM_REVISION = "leaf1009_native_feedback_v1"
NOMEM_DEPTH_REVISION = "leaf1009_nomem_depth_v1"
NOMEM_DEPTH_SPLIT = "nomem_depth_eval"
NOMEM_DEPTHS = (1, 2, 3, 4, 5, 6)
# Complete retained main0928 seed-0 configuration: cloud defaults need no old file.
FROZEN_BASE_CONFIG = {'runner': 'parallel',
 'mac': 'entity_mac',
 'env': 'had',
 'env_args': {},
 'batch_size_run': 8,
 'test_nepisode': 20,
 'test_interval': 2000,
 'test_greedy': True,
 'log_interval': 2000,
 'runner_log_interval': 2000,
 'learner_log_interval': 1000,
 't_max': 1000000,
 'use_cuda': True,
 'buffer_cpu_only': True,
 'buffer_opt_mem': True,
 'multi_task': False,
 'use_tensorboard': False,
 'save_model': False,
 'save_model_interval': 5000,
 'checkpoint_run_name': '',
 'pi_checkpoint_run_name': '',
 'checkpoint_unique_id': '',
 'evaluate': False,
 'load_step': 0,
 'save_replay': False,
 'video_path': None,
 'fps': 2,
 'tb_dirname': 'tb_logs',
 'eval_all_models': False,
 'eval_all_scen': False,
 'eval_sep': False,
 'eval_n_task_range': '',
 'eval_path': None,
 'gamma': 0.99,
 'batch_size': 32,
 'max_traj_len': -1,
 'buffer_size': 5000,
 'lr': 0.0005,
 'optim_alpha': 0.99,
 'optim_eps': 1e-05,
 'grad_norm_clip': 10,
 'weight_decay': 0,
 'alloc_q_weight_decay': 0,
 'target_update_interval': 200,
 'alloc_target_update_interval': 50,
 'popart': False,
 'n_extra_units': 0,
 'n_extra_tasks': 0,
 'vi_lambda': 5e-05,
 'agent': {'recurrent': True, 'entity_scheme': True, 'imagine': True, 'subtask_cond': None},
 'hier_agent': {'task_allocation': None,
                'copa': False,
                'mask_copa': True,
                'copa_vi_loss': True,
                'action_length': 5,
                'alloc_critic': 'standard',
                'alloc_policy': 'autoreg',
                'decay_old': 150000,
                'n_proposals': 32,
                'alloc_eps': '1.0-0.0-0.05',
                'prop_alloc_eps': '1.0-0.05-0.2',
                'entropy_loss': 0.01,
                'alloc_opt': 'rmsprop',
                'max_bs': 400,
                'pi_ag_attn': False,
                'subtask_mask': True,
                'sel_task_upd': True,
                'pi_pointer_net': True,
                'pi_autoreg': True},
 'mask_subtask_actions': False,
 'rnn_hidden_dim': 64,
 'attn_embed_dim': 128,
 'attn_n_heads': 4,
 'alloc_embed_dim': 128,
 'alloc_n_heads': 4,
 'obs_agent_id': False,
 'obs_last_action': False,
 'softmax_mixing_weights': True,
 'mixer_subtask_cond': None,
 'training_iters': 8,
 'action_selector': 'epsilon_greedy',
 'epsilon_start': 1.0,
 'epsilon_finish': 0.05,
 'epsilon_anneal_time': 100000,
 'agent_output_type': 'q',
 'double_q': True,
 'encoder_chunk_size': 1024,
 'encoder_skip_dead': True,
 'mixing_embed_dim': 32,
 'mixer_chunk_size': 256,
 'mixer_skip_unfilled': True,
 'hypernet_embed': 128,
 'name': 'regir_nomem',
 'learner': 'q_learner',
 'mixer': 'flex_qmix',
 'encoder': 'attention',
 'entity_last_action': True,
 'lmbda': 0.5,
 'imagine_group': 'original',
 'count_cond': 'phi2',
 'skip_count_inject': True,
 'global_branch': 'cycle',
 'global_slots': 4,
 'global_depths': [1, 2, 3, 4],
 'global_eval_depth': 4,
 'global_embed_dim': 64,
 'global_ffn_mult': 2,
 'global_n_heads': 4,
 'global_query_no_memory': True,
 'method': 'regir_nomem',
 'seed': 0,
 'entity_scheme': True,
 'feature_layout': 'had',
 'report_interval': 300,
 'progress_interval': 10,
 'resume_interval': 600,
 'resume': True,
 'run': 'train',
 'shaping_coef': 1.0,
 'shaping_range': 4000.0,
 'implementation_revision': 'main0928_looped_nocount_v1',
 'profile': 'main0928',
 'output': '/inspire/hdd/project/urbanlowaltitude/fengkairui-25026/shenqili/TP/Open_Score/Open-SCORE/outputs/main0928',
 'skip_final_eval': True,
 'reward_mode': 'damage',
 'friendly_penalty': 1.0,
 'concurrency': 12,
 'device': 'cuda',
 'count_ln': True,
 'global_feedback_mode': 'gru',
 'skip_refil_local': False,
 'read_last_round': False,
 'rer_update': 'tied',
 'rer_readout': 'learned',
 'matched_hidden': 0,
 'entity_pad': 'train',
 'n_agents': 10,
 'n_entities': 23,
 'n_actions': 9,
 'entity_shape': 10,
 'state_shape': 233,
 'obs_shape': 230,
 'episode_limit': 100,
 'gt_mask_avail': False,
 'n_tasks': 3}
FROZEN_BASE_SOURCE = "main0928/had/regir_nomem/train/seed_0/config.json"
LOOP_METHODS = ("regir_loop_nomem", "regir_bidirectional_nomem", "regir_requery_nomem",
                "regir_entitygru_nomem", "regir_slotgru_nomem")
THRESHOLDS = (.01, .03, .1, .3, 1.)
CF_CONFIGS = ((4,4,2), (10,10,3), (30,30,2), (50,50,2), (30,30,12))
MECHANISM_CONFIGS = ((4,4,2), (10,10,3), (10,10,12), (30,30,2), (50,50,2), (30,30,12))
FEEDBACK_ARMS = {
    "query_feedback": ("normal", "query_update_frozen"),
    "bidirectional": ("normal", "query_update_frozen", "reverse_kv_clamp"),
    "requery": ("normal", "query_update_frozen"),
    "entity_gru": ("normal", "entity_gru_hidden_reset"),
    "slot_gru": ("normal", "slot_gru_hidden_reset", "slot_competition_removed"),
}
LOOP_STATS = ("decisions", "depth_1", "depth_2", "depth_3", "depth_4",
              "attention_entries", "read_calls", "temporal_previews",
              "entity_rounds", "slot_rounds", "query_updates", "macs")
TIERS = {
    "P0": ("regir_nomem_r1", "regir_nomem_fixed4", "regir_nomem_last", "regir_refine_nomem",
           "regir_refine_r1_nomem", "regir_refine_noinput_nomem"),
    "P1": ("regir_nomem_r0", "regir_nomem_static", "regir_nomem_scorenorm",
           "regir_refine_mix_nomem", "regir_refine_fixedinput_nomem"),
    "P2": ("regir_refine_fixed4_nomem", "regir_refine_untied4_nomem", "regir_set2_nomem"),
    "references": ("regir_nomem", "refil", "transfqmix"),
}
GROUPS = {
    "ID": ((4, 4, 2), (6, 6, 2), (8, 8, 2), (10, 10, 3)),
    "K-OOD": tuple((10, 10, k) for k in (4, 6, 9, 12)),
    "N-OOD": tuple((n, n, 2) for n in (15, 20, 25, 30, 40, 50)),
    "Joint-OOD": ((15, 15, 4), (15, 15, 6), (20, 20, 4), (20, 20, 6),
                  (30, 30, 4), (30, 30, 6), (30, 30, 9), (30, 30, 12)),
}
FINAL_CONFIGS = tuple(sorted({c for values in GROUPS.values() for c in values}
                             | {(5, 5, 2), (10, 10, 2)}))
DEPTH_CONFIGS = ((10, 10, 2), (30, 30, 2), (50, 50, 2), (30, 30, 12))


def read_registry():
    """Read the four literal configuration declarations without importing HAD."""
    path = PROJECT / "open_score/algos/__init__.py"
    namespace = {}
    wanted = {"_CYCLE", "_NOMEM", "_REFINE", "REFINEMENT_OVERRIDES", "LOOP_CANDIDATE_OVERRIDES"}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                expression = compile(ast.Expression(node.value), str(path), "eval")
                namespace[target.id] = eval(expression, {"__builtins__": {"dict": dict}}, namespace)
    return {**namespace["REFINEMENT_OVERRIDES"], **namespace.get("LOOP_CANDIDATE_OVERRIDES", {})}


def analysis_module(relative):
    """Reuse a pure file/record utility without importing the environment package."""
    path = PROJECT / "open_score" / relative
    spec = importlib.util.spec_from_file_location("leaf1009_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment_root(root):
    root = Path(root)
    return root / "had" if (root / "had").is_dir() else root


def run_dir(root, method, seed):
    return environment_root(root) / method / "train" / f"seed_{seed}"


def legacy_methods(options):
    selected = tuple(filter(None, (options.methods or "").split(",")))
    if not selected:
        selected = (tuple(m for tier in ("P0", "P1", "P2") for m in TIERS[tier])
                    if options.tier == "all" else TIERS[options.tier])
    allowed = {m for values in TIERS.values() for m in values}
    if len(set(selected)) != len(selected) or set(selected) - allowed:
        raise ValueError("Select distinct registered methods from --tier or --methods")
    return selected


def evaluate_one(options, method, seed, stop):
    import torch
    from open_score.algos import load_policy
    from open_score.eval.anchors import BLUE_STRATEGY
    from open_score.eval.protocol import config_dict, config_key, training_finished
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import ExperimentLogger, read_records, unique_episodes
    source = options.source_output if method in TIERS["references"] else options.output
    checkpoint = run_dir(source, method, seed) / "final.pt"
    if not training_finished(checkpoint.parent):
        raise ValueError(f"Missing completed final checkpoint: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["config"]["method"] != method or int(saved["config"]["seed"]) != seed:
        raise ValueError("Checkpoint method/seed identity differs")
    t_env = int(saved["progress"]["t_env"])
    if int(saved["config"]["t_max"]) != 1_000_000 and options.stage in ("eval", "confirm"):
        raise ValueError("Formal comparison requires the original 1M-step training budget")
    policy = load_policy(method, checkpoint)
    policy.set_device("cuda")
    name = f"{PROFILE}_{method}_{seed}_{options.stage}"
    register_end_to_end_policy("red", name, name, lambda _: policy)
    output = options.output / "had"
    logger = ExperimentLogger(options.output, method, seed, "train", env="had")
    # Use an existing logger phase; the arm and checkpoint tag isolate confirmation.
    phase = "depth_eval" if options.stage == "depth" else "final_eval"
    arm = "confirmation" if options.stage == "confirm" else "standard"
    quota = 300 if options.stage == "eval" else options.episodes
    start = 110000 if options.stage == "confirm" else 9000
    depths = options.depths if options.stage == "depth" else (None,)
    configs = DEPTH_CONFIGS if options.stage == "depth" else FINAL_CONFIGS
    existing = unique_episodes(read_records(output, "episodes", run="train", method=method, seed=seed))
    done = {(r.get("phase"), config_key(r["config"]), int(r["episode_seed"]), r.get("checkpoint"), r.get("arm"))
            for r in existing}
    for depth in depths:
        cfg = saved["config"]
        limit = (4 if cfg.get("rer_update") == "untied4" else
                 int(cfg.get("rer_untied_depth", 2)) if cfg.get("rer_update") == "untied" else
                 max(cfg["global_depths"]) if cfg.get("rer_readout") == "static" else None)
        if depth is not None and (not cfg.get("global_branch") or cfg.get("global_read_h0")
                                  or limit is not None and depth > limit):
            print(f"skip unsupported depth {depth}: {method}", flush=True)
            continue
        tag = f"final@{t_env}" + (f"/R{depth}" if depth is not None else "/confirmation" if arm == "confirmation" else "")
        if depth is not None:
            policy.set_eval_depth(depth)
        for config in configs:
            for scene in range(start, start + quota):
                if stop.is_set():
                    return "stopped"
                if (phase, config, scene, tag, arm) in done:
                    continue
                result = run_episode(red=config[0], blue=config[1], targets=config[2],
                    seed=scene, red_strategy={"architecture": "end_to_end", "policy": name},
                    blue_strategy=BLUE_STRATEGY, max_steps=100, record=False,
                    task_mode="damage", spatial_dim=2, target_initialization="random",
                    diagnostics=True, record_events=False, retain_trajectory=False)
                row = dict(result["episode_summary"], phase=phase, config=config_dict(config),
                           episode_seed=scene, eval_point=50, t_env=t_env, checkpoint=tag,
                           checkpoint_id=saved.get("artifact_id"), cycle_depth=depth,
                           arm=arm, protocol=REVISION)
                logger.episodes([row])
            print(f"{method} s{seed} {phase} R={depth} {config}: {quota} scenes", flush=True)
    return "completed"


# The original tiers remain callable; the candidate protocol below owns new arms.
def base_configuration(options):
    if options.base_config is None:
        return copy.deepcopy(FROZEN_BASE_CONFIG), dict(kind="frozen_literal", source=FROZEN_BASE_SOURCE)
    path = options.base_config.resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("method") != "regir_nomem":
        raise ValueError("--base-config must describe retained main0928 regir_nomem")
    return cfg, dict(kind="explicit_file", source=str(path), configuration=copy.deepcopy(cfg))


def methods(options):
    explicit = tuple(filter(None, (options.methods or "").split(",")))
    if options.suite == "legacy":
        return legacy_methods(options)
    selected = explicit
    if not selected:
        if options.suite == "references":
            selected = TIERS["references"]
        elif options.suite == "selected":
            selected = (frozen_document(options, "selection.json")["selected_method"],)
        else:
            selected = LOOP_METHODS
    allowed = set(LOOP_METHODS + TIERS["references"])
    if len(set(selected)) != len(selected) or set(selected) - allowed:
        raise ValueError("--methods must contain distinct methods from the candidate/reference registry")
    return selected


def atomic_document(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(pending, path)


@contextmanager
def process_lock(path, *, blocking=False):
    """Hold a persistent lock file; never lock an inode replaced by atomic_json."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    acquired=False
    with path.open("a+b") as stream:
        if os.name=="nt":
            import msvcrt
            if path.stat().st_size==0:
                stream.write(b"\0");stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK,1)
                acquired=True
            except OSError as error:
                if blocking or error.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):raise
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                acquired=True
            except OSError as error:
                if blocking or error.errno not in (errno.EACCES,errno.EAGAIN):raise
        try:
            yield acquired
        finally:
            if acquired:
                stream.seek(0)
                if os.name=="nt":msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def initialize(options):
    # Metadata transactions are short; job claims below always remain nonblocking.
    with process_lock(options.output/".experiment.lock",blocking=True):
        return initialize_metadata(options)


def initialize_metadata(options):
    from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
    base, provenance = base_configuration(options)
    registry = read_registry()
    absent = set(methods(options)) - set(registry) - set(TIERS["references"])
    if absent:
        raise ValueError(f"Missing literal method specifications: {sorted(absent)}")
    root = options.output
    path = root / "experiment.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else dict(
        profile=PROFILE, implementation_revision=LEGACY_REVISION if options.suite=="legacy" else REVISION, base_config=base, methods={},
        seeds=[0,1,2], training_steps=1_000_000, final_configs=FINAL_CONFIGS,
        episodes_per_config=300, validation_points=50, validation_episodes_per_config=25,
        official_checkpoint="final", confirmation_scene_start=110000,
        had_core_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL)
    if data.get("profile") != PROFILE:
        raise ValueError("Existing output has a different experiment profile")
    for key, value in (("had_core_version", CORE_VERSION), ("physics_protocol", PHYSICS_PROTOCOL)):
        if data.get(key) not in (None, value):
            raise ValueError(f"Existing experiment differs at {key}; existing data preserved")
        data.setdefault(key, value)
    specs = data.setdefault("methods", {})
    for method in methods(options):
        if method not in registry:
            continue
        spec = registry[method]
        if method in specs and specs[method] != spec:
            raise ValueError(f"Existing method specification changed: {method}")
        specs.setdefault(method, spec)
    protocol = dict(revision=REVISION, base_config=base, base_provenance=provenance,
        methods=list(LOOP_METHODS), seeds=[0,1,2], training_steps=1_000_000,
        calibration=dict(configs=GROUPS["ID"], scenes=[42000,42039], thresholds=THRESHOLDS),
        selection=dict(configs=GROUPS["ID"], scenes=[40000,40099]),
        formal=dict(configs=FINAL_CONFIGS, scenes=[9000,9299]),
        budget_only_scenes=[43000,43009], confirmation_scenes=[110000,110299],
        counterfactual=dict(configs=CF_CONFIGS, scenes=[44000,44009], steps=[0,5,10],
                           agent_rule="first two live observers by stable roster order",
                           maximum_intervention_tails=9000, fixed4_source="factual_reuse"))
    # Append a protocol beside legacy metadata; never rewrite an old method/base cfg.
    protocols = data.setdefault("protocols", {})
    canonical = json.loads(json.dumps(protocol))
    if REVISION in protocols and protocols[REVISION] != canonical:
        raise ValueError("Frozen candidate protocol/base provenance changed")
    protocols.setdefault(REVISION, canonical)
    if options.stage in ("mechanism", "run"):
        mechanism = dict(revision=MECHANISM_REVISION, configs=MECHANISM_CONFIGS,
            scenes=[44000,44009], seeds=[0,1,2], checkpoint="final",
            fixed_depths=[1,2,3,4], feedback_arms=FEEDBACK_ARMS,
            comparison="paired complete episodes; frozen-model interventions",
            selection_use=False, formal_ood=False, maximum_episodes=4860)
        mechanism = json.loads(json.dumps(mechanism))
        if MECHANISM_REVISION in protocols and protocols[MECHANISM_REVISION] != mechanism:
            raise ValueError("Native mechanism protocol changed; existing records preserved")
        protocols.setdefault(MECHANISM_REVISION, mechanism)
    (root / "had").mkdir(parents=True, exist_ok=True)
    atomic_document(path, data)
    return base


def train_one(options, method, seed, stop):
    from open_score.algos import train
    from open_score.eval.protocol import training_finished
    registry = read_registry()
    if method not in registry:
        raise ValueError("Reference checkpoints are reused, not retrained")
    directory = run_dir(options.output, method, seed)
    cfg, _ = base_configuration(options)
    cfg.update(registry[method])
    cfg.update(method=method, name=method, seed=seed, t_max=options.steps,
        profile=PROFILE, output=str(options.output), implementation_revision=REVISION if method in LOOP_METHODS else LEGACY_REVISION,
        external_report=True, use_cuda=True, device="cuda", run="train",
        skip_final_eval=True, resume=options.resume)
    if (directory / "final.pt").is_file():
        if not training_finished(directory):
            raise ValueError(f"Existing final checkpoint is incomplete: {directory}")
        old = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
        for key in ("method", "seed", "t_max", *registry[method]):
            if old.get(key) != cfg.get(key):
                raise ValueError(f"Retained checkpoint differs at {key}: {directory}")
        return "retained"
    cfg["_stop_event"] = stop
    checkpoint = train(method, cfg)
    return "completed" if Path(checkpoint).name == "final.pt" else "stopped"


def config_tuple(value):
    return tuple(int(value[k]) for k in ("N_R", "N_B", "K")) if isinstance(value, dict) else tuple(value)


def record_key(row):
    return (row["method"], int(row["seed"]), row["split"], row["arm"], config_tuple(row["config"]),
            int(row["episode_seed"]), row["checkpoint"], row.get("snapshot_step"), row.get("agent_id"))


def loop_records(options, method=None, seed=None, config=None, split=None):
    path = options.output / "loop_records.jsonl"
    if not path.is_file():
        return []
    bank = {}
    needles=[]
    for key,value in (("method",method),("seed",seed),("split",split)):
        if value is not None:
            suffix="," if key=="seed" else ""
            needles.append((json.dumps(key)+": "+json.dumps(value)+suffix).encode("utf-8"))
    # Writers only append. Read a complete-line prefix without holding their
    # append lock across a full scan, so GPU workers can persist concurrently.
    reader=(analysis_module("utils/logging.py")._locked_file(path)
            if os.name=="nt" else path.open("rb"))
    with reader as stream:
        limit=os.fstat(stream.fileno()).st_size
        for line in stream:
            if stream.tell()>limit or not line.endswith(b"\n"):break
            if not line.strip(): continue
            if any(needle not in line for needle in needles):continue
            row = json.loads(line)
            if method is not None and row["method"] != method: continue
            if seed is not None and int(row["seed"]) != seed: continue
            if config is not None and config_tuple(row["config"]) != tuple(config): continue
            if split is not None and row["split"] != split: continue
            bank[record_key(row)] = row
    return list(bank.values())


def append_loop_record(options, row):
    logging = analysis_module("utils/logging.py")
    path = options.output / "loop_records.jsonl"
    with logging._locked_file(path, create=True) as stream:
        stream.seek(0, os.SEEK_END)
        stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def nomem_depth_context(options, *, stored=False):
    """Inspect retained checkpoints and metadata without importing HAD."""
    path = options.output / "experiment.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("profile") != PROFILE:
        raise ValueError("Depth supplement requires the existing main1009 experiment")
    environment = {key: data.get(key) for key in ("had_core_version", "physics_protocol")}
    if not all(environment.values()):
        raise ValueError("Existing main1009 metadata lacks the HAD version/protocol")
    formal = data.get("protocols", {}).get(REVISION, {}).get("formal", {})
    if (tuple(sorted(map(config_tuple, formal.get("configs", [])))) != FINAL_CONFIGS
            or formal.get("scenes") != [9000, 9299]):
        raise ValueError("Existing formal configuration/scene pool differs")
    retained = data.get("protocols", {}).get(NOMEM_DEPTH_REVISION)
    if stored:
        if retained is None:
            raise ValueError("Run depth-eval before generating its report")
        context = dict(environment=environment, checkpoints=retained["checkpoints"],
                       checkpoint_files={})
        if retained != nomem_depth_protocol(context):
            raise ValueError("Stored NoMem depth protocol differs")
        return context
    import torch
    identities, files = {}, {}
    for seed in (0, 1, 2):
        checkpoint = run_dir(options.source_output, "regir_nomem", seed) / "final.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg, progress = saved["config"], saved["progress"]
        if (cfg.get("method") != "regir_nomem" or cfg.get("env") != "had"
                or cfg.get("profile") != "main0928" or int(cfg.get("seed", -1)) != seed
                or int(cfg.get("t_max", 0)) != 1_000_000
                or progress.get("status") not in ("complete", "completed")
                or int(progress.get("t_env", 0)) < 1_000_000):
            raise ValueError(f"Missing completed original NoMem checkpoint: {checkpoint}")
        if (cfg.get("global_branch") != "cycle" or not cfg.get("global_query_no_memory")
                or cfg.get("rer_update", "tied") != "tied"
                or cfg.get("rer_readout", "learned") != "learned"
                or cfg.get("read_last_round", False) or cfg.get("global_read_h0", False)
                or cfg.get("leaf_loop_core")
                or tuple(cfg.get("global_depths", ())) != (1, 2, 3, 4)
                or int(cfg.get("global_eval_depth", 0)) != 4):
            raise ValueError(f"Checkpoint is not the retained tied/learned NoMem: {checkpoint}")
        identities[str(seed)] = dict(method="regir_nomem", seed=seed,
            checkpoint=f"final@{int(progress['t_env'])}",
            artifact_id=saved.get("artifact_id"), configuration=cfg)
        files[str(seed)] = str(checkpoint)
    context = dict(environment=environment, checkpoints=identities, checkpoint_files=files)
    expected = nomem_depth_protocol(context)
    if retained is not None and retained != expected:
        raise ValueError("Retained NoMem depth protocol/checkpoint identity changed")
    return context


def nomem_depth_protocol(context):
    return json.loads(json.dumps(dict(revision=NOMEM_DEPTH_REVISION,
        method="regir_nomem", seeds=[0, 1, 2], depths=NOMEM_DEPTHS,
        configs=FINAL_CONFIGS, scenes=[9000, 9299], episodes_per_cell=300,
        checkpoint="final", checkpoints=context["checkpoints"],
        training_depth_max=4, environment=context["environment"],
        evaluation=dict(task_mode="damage", spatial_dim=2, max_steps=100,
            target_initialization="random", greedy=True,
            blue_strategy={"architecture": "hierarchical",
                           "layers": {"grouping": "reactive", "control": "rush"}}),
        reuse=dict(split="eval", arm="original", execution_depth=4, protocol=REVISION),
        selection_use=False)))


def nomem_depth_rows(options, context, records=None):
    """Return a logical matrix; R4 source rows are mapped only in memory."""
    records = loop_records(options, method="regir_nomem") if records is None else records
    bank, reused = {}, {}
    for row in records:
        if row.get("official_aggregate") is False or row.get("method") != "regir_nomem":
            continue
        seed = int(row["seed"])
        cfg, scene = config_tuple(row["config"]), int(row["episode_seed"])
        if seed not in (0, 1, 2) or cfg not in FINAL_CONFIGS or not 9000 <= scene <= 9299:
            continue
        identity = context["checkpoints"][str(seed)]
        if row.get("checkpoint") != identity["checkpoint"]:
            continue
        if "D" not in row or not math.isfinite(float(row["D"])):
            continue
        if any(row.get(key, value) != value for key, value in context["environment"].items()):
            continue
        artifact = row.get("checkpoint_artifact_id")
        if artifact is not None and artifact != identity["artifact_id"]:
            continue
        if row.get("split") == NOMEM_DEPTH_SPLIT:
            arm = row.get("arm")
            if (row.get("protocol") != NOMEM_DEPTH_REVISION
                    or arm not in tuple(f"fixed{d}" for d in NOMEM_DEPTHS)
                    or row.get("execution_depth") != int(arm[5:])
                    or any(row.get(k) != v for k, v in context["environment"].items())
                    or artifact != identity["artifact_id"]):
                continue
            bank[arm, cfg, seed, scene] = row
        elif (row.get("split") == "eval" and row.get("arm") == "original"
                and row.get("protocol") == REVISION
                and row.get("execution_depth", 4) == 4):
            mapped = dict(row, split=NOMEM_DEPTH_SPLIT, arm="fixed4",
                protocol=NOMEM_DEPTH_REVISION, execution_depth=4, training_depth_max=4,
                depth_extrapolation=False, loop_stats={},
                checkpoint_artifact_id=identity["artifact_id"],
                source_split="eval", source_arm="original", source_protocol=REVISION,
                source_environment_metadata="experiment.json", reused=True,
                **context["environment"])
            reused["fixed4", cfg, seed, scene] = mapped
    # A directly evaluated R4 has priority over its reusable source.
    return list({**reused, **bank}.values())


def nomem_depth_coverage(rows):
    cells = defaultdict(set)
    for row in rows:
        cells[row["arm"], config_tuple(row["config"]), int(row["seed"])].add(int(row["episode_seed"]))
    scenes = set(range(9000, 9300))
    complete_cells = sum(cells[f"fixed{depth}", cfg, seed] == scenes
        for depth in NOMEM_DEPTHS for cfg in FINAL_CONFIGS for seed in (0, 1, 2))
    complete_shards = sum(all(cells[f"fixed{d}", cfg, seed] == scenes for d in NOMEM_DEPTHS)
        for cfg in FINAL_CONFIGS for seed in (0, 1, 2))
    expected = len(NOMEM_DEPTHS) * len(FINAL_CONFIGS) * 3 * 300
    return dict(kind="nomem_depth_matrix", complete=complete_shards,
        total=len(FINAL_CONFIGS) * 3, complete_cells=complete_cells,
        expected_cells=len(NOMEM_DEPTHS) * len(FINAL_CONFIGS) * 3,
        logical_episodes=len(rows), expected_episodes=expected,
        reused_R4_episodes=sum(bool(r.get("reused")) for r in rows),
        new_completed_episodes=sum(not r.get("reused", False) for r in rows),
        remaining_episodes=expected-len(rows), global_complete=complete_cells == 432)


def initialize_nomem_depth(options, context):
    from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
    from open_score.eval.anchors import BLUE_STRATEGY
    actual = dict(had_core_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL)
    if actual != context["environment"]:
        raise ValueError("Current HAD differs from the formal main1009 environment; data preserved")
    if BLUE_STRATEGY != nomem_depth_protocol(context)["evaluation"]["blue_strategy"]:
        raise ValueError("Current opponent differs from the formal main1009 protocol")
    with process_lock(options.output / ".experiment.lock", blocking=True):
        path = options.output / "experiment.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        protocol = nomem_depth_protocol(context)
        protocols = data.setdefault("protocols", {})
        if NOMEM_DEPTH_REVISION in protocols and protocols[NOMEM_DEPTH_REVISION] != protocol:
            raise ValueError("Frozen NoMem depth supplement changed")
        protocols.setdefault(NOMEM_DEPTH_REVISION, protocol)
        atomic_document(path, data)


def evaluate_nomem_depth_shard(options, job, stop):
    from open_score.eval.anchors import BLUE_STRATEGY
    from open_score.eval.protocol import config_dict
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import ExperimentLogger
    context = options.nomem_depth_context
    method, seed, config = job["method"], job["seed"], job["config"]
    policy, t_env, identity, params = load_frozen_policy(options, method, seed)
    expected = context["checkpoints"][str(seed)]
    if (identity["checkpoint"] != expected["checkpoint"]
            or identity["configuration"] != expected["configuration"]
            or identity.get("artifact_id") != expected["artifact_id"]):
        raise ValueError("NoMem checkpoint differs from the supplement preflight")
    name = f"{PROFILE}_nomem_depth_{seed}_{os.getpid()}"
    register_end_to_end_policy("red", name, name, lambda _: policy)
    logger = ExperimentLogger(options.output, method, seed, "train", env="had")
    records = loop_records(options, method=method, seed=seed, config=config)
    done = {record_key(r) for r in nomem_depth_rows(options, context, records)}
    for scene in range(9000, 9300):
        for arm in job["arms"]:
            if stop.is_set():
                return "stopped"
            depth = int(arm[5:])
            row = dict(method=method, seed=seed, split=NOMEM_DEPTH_SPLIT, arm=arm,
                config=config_dict(config), episode_seed=scene, checkpoint=identity["checkpoint"],
                checkpoint_artifact_id=identity["artifact_id"], actor_params=params,
                protocol=NOMEM_DEPTH_REVISION, execution_depth=depth, training_depth_max=4,
                depth_extrapolation=depth > 4, loop_stats={}, **context["environment"])
            if record_key(row) in done:
                continue
            policy.set_eval_depth(depth)
            policy.reset()
            started = time.perf_counter()
            result = run_episode(red=config[0], blue=config[1], targets=config[2], seed=scene,
                red_strategy={"architecture": "end_to_end", "policy": name},
                blue_strategy=BLUE_STRATEGY, max_steps=100, record=False, task_mode="damage",
                spatial_dim=2, target_initialization="random", diagnostics=True,
                record_events=False, retain_trajectory=False)
            if result.get("physics_protocol") != context["environment"]["physics_protocol"]:
                raise ValueError("Episode physics protocol differs from the formal pool")
            row["episode_wall_seconds"] = time.perf_counter() - started
            summary = result["episode_summary"]
            row.update({key: summary[key] for key in ("D", "return", "ep_len")})
            if not all(math.isfinite(float(row[key])) for key in ("D", "return", "ep_len")):
                raise ValueError("Non-finite NoMem evaluation result")
            logger.episodes([dict({**summary, **row}, phase="depth_eval",
                cycle_depth=depth, arm=f"{NOMEM_DEPTH_SPLIT}:{arm}",
                checkpoint=f"{identity['checkpoint']}/{NOMEM_DEPTH_SPLIT}/{arm}",
                artifact_id=identity["checkpoint"], eval_point=50, t_env=t_env)])
            append_loop_record(options, row)
            done.add(record_key(row))
        if (scene-8999) % 10 == 0:
            print(f"NoMem seed={seed} {config} depth-eval {scene-8999}/300", flush=True)
    return "completed"


def frozen_document(options, name):
    path = options.output / name
    if not path.is_file():
        raise ValueError(f"Run the preceding freeze stage first: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("frozen") or value.get("revision") != REVISION:
        raise ValueError(f"Not a frozen {REVISION} document: {name}")
    return value


def load_frozen_policy(options, method, seed):
    import torch
    from open_score.algos import load_policy
    from open_score.eval.protocol import training_finished
    source = options.source_output if method in TIERS["references"] else options.output
    checkpoint = run_dir(source, method, seed) / "final.pt"
    if not training_finished(checkpoint.parent):
        raise ValueError(f"Missing completed final checkpoint: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, t_env = saved["config"], int(saved["progress"]["t_env"])
    if cfg["method"] != method or int(cfg["seed"]) != seed or int(cfg["t_max"]) != 1_000_000:
        raise ValueError(f"Checkpoint identity/budget mismatch: {checkpoint}")
    if method in LOOP_METHODS and cfg.get("implementation_revision") != REVISION:
        raise ValueError("Candidate final checkpoint must come from this runner protocol")
    policy = load_policy(method, checkpoint)
    policy.set_device("cuda")
    if method in LOOP_METHODS:
        for api in ("set_loop_execution", "set_decision_depths", "get_loop_stats", "set_loop_seed"):
            if not callable(getattr(policy, api, None)):
                raise ValueError(f"Frozen policy lacks required true-loop API {api}")
    identity = dict(method=method, seed=seed, checkpoint=f"final@{t_env}", configuration=cfg)
    if method == "regir_nomem":
        identity["artifact_id"] = saved.get("artifact_id")
    params = sum(p.numel() for p in policy.mac.agent.parameters())
    return policy, t_env, identity, params


def decision_seed(seed, scene):
    return (int(seed) * 1_000_003 + int(scene) * 97 + 1009) % (2**63 - 1)


class FeedbackIntervention:
    def __init__(self, agent, arm):
        from collections import Counter
        import torch as th
        self.agent = agent
        self.net = agent.global_net
        self.arm = arm
        self.core = self.net.loop_core
        if arm not in FEEDBACK_ARMS.get(self.core, ()):
            raise ValueError(f"Unsupported feedback intervention {self.core}/{arm}")
        if getattr(agent, "_native_feedback_intervention", None) is not None:
            raise RuntimeError("Clear the existing intervention before installing another")
        if agent.training:
            raise RuntimeError("Feedback interventions are for eval-mode frozen policies")
        self.stats = Counter()
        self.last_forward = {}
        self._forward_start = Counter()
        self._saved = []
        self._handles = []
        self._context = None
        self._offset = 0
        self._gru_hidden = None
        self._active = True
        if arm == "normal":
            return
        agent._native_feedback_intervention = self
        self._handles.append(agent.register_forward_pre_hook(self._before_forward))
        self._handles.append(agent.register_forward_hook(self._after_forward))
        if arm == "query_update_frozen":
            original = self.net._loop_update_query

            def frozen(net, b0, previous, read):
                original(b0, previous, read)
                self.stats["query_updates_computed"] += int(read.shape[0])
                self.stats["query_updates_overwritten"] += int(read.shape[0])
                self.stats["query_update_calls"] += 1
                return previous

            self._patch(self.net, "_loop_update_query", frozen)
        elif arm == "reverse_kv_clamp":
            original_context = self.net._loop_context
            original_round = self.net.loop_round

            def context(net, memory, mask, query=None):
                if self._context is not None and query is not None:
                    size = int(memory.shape[0])
                    query = self._context[self._offset:self._offset + size]
                    if query.shape[0] != size:
                        raise RuntimeError("Reverse-feedback chunk alignment mismatch")
                    self._offset += size
                    self.stats["reverse_queries_clamped"] += size
                    self.stats["reverse_context_calls"] += 1
                return original_context(memory, mask, query)

            def reverse_round(net, work, stats=None):
                self._context, self._offset = work["b0"], 0
                try:
                    result = original_round(work, stats)
                    if self._offset != work["b0"].shape[0]:
                        raise RuntimeError("Reverse-feedback workspace was not fully consumed")
                    return result
                finally:
                    self._context = None
                    self._offset = 0

            self._patch(self.net, "_loop_context", context)
            self._patch(self.net, "loop_round", reverse_round)
        elif arm == "entity_gru_hidden_reset":
            original = self.net._loop_entity_step

            def entity_step(net, previous, initial, mask):
                self._gru_hidden = initial.reshape(-1, initial.shape[-1])
                try:
                    result = original(previous, initial, mask)
                    self.stats["entity_hidden_resets"] += int(initial.shape[0])
                    self.stats["entity_step_calls"] += 1
                    return result
                finally:
                    self._gru_hidden = None

            self._patch(self.net, "_loop_entity_step", entity_step)
            self._handles.append(self.net.loop_entity_gru.register_forward_pre_hook(self._reset_gru_hidden))
        elif arm == "slot_gru_hidden_reset":
            original_initialize = self.net.loop_initialize
            original_round = self.net.loop_round
            original_step = self.net._loop_slot_step

            def initialize(net, initial, mask, b0, stats=None):
                work = original_initialize(initial, mask, b0, stats)
                # loop_compact index-selects every tensor in work.  Initial
                # slots consequently keep the exact surviving-observer order.
                work["native_feedback_initial_slots"] = work["slots"]
                self.stats["slot_workspaces_initialized"] += int(initial.shape[0])
                return work

            def slot_round(net, work, stats=None):
                self._context, self._offset = work["native_feedback_initial_slots"], 0
                try:
                    result = original_round(work, stats)
                    if self._offset != self._context.shape[0]:
                        raise RuntimeError("Initial-slot workspace was not fully consumed")
                    return result
                finally:
                    self._context = None
                    self._offset = 0

            def slot_step(net, slots, key, value, mask):
                size = int(slots.shape[0])
                initial = self._context[self._offset:self._offset + size]
                self._offset += size
                if initial.shape != slots.shape:
                    raise RuntimeError("Initial-slot chunk alignment mismatch")
                self._gru_hidden = initial.reshape(-1, initial.shape[-1])
                try:
                    result = original_step(slots, key, value, mask)
                    self.stats["slot_hidden_resets"] += size
                    self.stats["slot_step_calls"] += 1
                    return result
                finally:
                    self._gru_hidden = None

            self._patch(self.net, "loop_initialize", initialize)
            self._patch(self.net, "loop_round", slot_round)
            self._patch(self.net, "_loop_slot_step", slot_step)
            self._handles.append(self.net.loop_slot_gru.register_forward_pre_hook(self._reset_gru_hidden))
        elif arm == "slot_competition_removed":

            def independent_slot_step(net, slots, key, value, mask):
                query = net.loop_slot_query(net.loop_slot_norm(slots))
                logits = th.bmm(query, key.transpose(1, 2)) / query.shape[-1] ** 0.5
                # Normalize independently over entities.  All-masked rows
                # use zero logits before softmax and remain zero after mask.
                logits = logits.masked_fill(mask.unsqueeze(1), float("-inf"))
                logits = logits.masked_fill(mask.all(-1)[:, None, None], 0)
                weights = th.softmax(logits, dim=-1).masked_fill(mask.unsqueeze(1), 0)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
                updates = th.bmm(weights, value)
                rows, n_slots, dim = slots.shape
                updated = net.loop_slot_gru(updates.reshape(-1, dim), slots.reshape(-1, dim))
                updated = updated.reshape(rows, n_slots, dim)
                updated = updated + net.loop_slot_ffn(net.loop_slot_norm(updated))
                self.stats["independent_slot_routes"] += int(rows)
                self.stats["slot_step_calls"] += 1
                return updated.masked_fill(mask.all(-1)[:, None, None], 0)

            self._patch(self.net, "_loop_slot_step", independent_slot_step)

    def _patch(self, obj, name, function):
        from types import MethodType
        # Remove the instance override on clear if the original was a class
        # method.  Do not leave a bound-method instance shadow behind.
        had_instance_value = name in obj.__dict__
        self._saved.append((obj, name, had_instance_value, obj.__dict__.get(name)))
        setattr(obj, name, MethodType(function, obj))

    def _reset_gru_hidden(self, module, args):
        if self._gru_hidden is None:
            raise RuntimeError("GRU reset invoked without a matching workspace")
        if args[1].shape != self._gru_hidden.shape:
            raise RuntimeError("GRU hidden-reset shape mismatch")
        return args[0], self._gru_hidden

    def _before_forward(self, module, args):
        if module.training:
            raise RuntimeError("Do not use native feedback interventions for training")
        self._forward_start = self.stats.copy()

    def _after_forward(self, module, args, output):
        self.stats["forwards"] += 1
        self.last_forward = dict(self.stats - self._forward_start)

    def reset_stats(self):
        self.stats.clear()
        self.last_forward = {}

    def summary(self):
        return {"arm": self.arm, "core": self.core, **dict(self.stats)}

    def clear(self):
        if not self._active:
            return
        for handle in self._handles:
            handle.remove()
        for obj, name, had_instance_value, value in reversed(self._saved):
            if had_instance_value:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        if getattr(self.agent, "_native_feedback_intervention", None) is self:
            delattr(self.agent, "_native_feedback_intervention")
        self._context = self._gru_hidden = None
        self._offset = 0
        self._active = False


def configure_execution(policy, arm, threshold=None, probabilities=None):
    if arm.startswith("fixed"):
        policy.set_loop_execution("fixed", depth=int(arm[-1]))
    elif arm.startswith("tau_") or arm == "adaptive":
        policy.set_loop_execution("adaptive", depth=4, threshold=threshold)
    elif arm == "random":
        policy.set_loop_execution("random", depth=4, probabilities=probabilities)
    elif arm != "original":
        raise ValueError(f"Unknown execution arm: {arm}")


def stage_jobs(options, selected=None, split=None, arms=None, configs=None, start=None, quota=None):
    selected = methods(options) if selected is None else tuple(selected)
    stage = options.stage
    if stage == "train":
        return [dict(method=m, seed=s, config=None, split="train", arms=[]) for m in selected for s in options.seeds]
    if configs is None:
        configs = (GROUPS["ID"] if stage in ("calibrate", "select") else
                   MECHANISM_CONFIGS if stage == "mechanism" else
                   CF_CONFIGS if stage == "counterfactual" else FINAL_CONFIGS)
    if start is None:
        start = {"calibrate":42000, "select":40000, "confirm":110000, "counterfactual":44000, "mechanism":44000}.get(stage,9000)
    if quota is None:
        quota = {"calibrate":40, "select":100, "counterfactual":10, "mechanism":10}.get(stage,300)
    split = stage if split is None else split
    jobs = []
    registry = read_registry() if stage == "mechanism" else None
    for method in selected:
        current_arms = arms
        if current_arms is None:
            current_arms = (["fixed1","fixed2","fixed3","fixed4"] +
                list(FEEDBACK_ARMS[registry[method]["leaf_loop_core"]][1:]) if stage == "mechanism" else
                ["original"] if method in TIERS["references"] else
                ["fixed1","fixed2","fixed3","fixed4"] + [f"tau_{t:g}" for t in THRESHOLDS] if stage == "calibrate" else
                ["fixed1","fixed2","fixed3","fixed4","adaptive","random"] if stage == "select" else
                ["adaptive","random"] if stage == "adaptive" else
                ["fixed4","adaptive","random"] if stage == "confirm" else
                ["fixed1","fixed2","fixed3","fixed4"])
        for seed in options.seeds:
            for config in configs:
                jobs.append(dict(method=method, seed=seed, config=tuple(config), split=split,
                                 arms=list(current_arms), start=start, quota=quota))
    return jobs


def budget_mixture(depth_costs, target):
    costs = [float(depth_costs[d]) for d in (1,2,3,4)]
    if any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError("Budget matching requires positive finite measured MAC counts")
    # Choose only adjacent depths; measured costs need not be monotonic.
    candidates = []
    for index in range(3):
        lo, hi = costs[index:index+2]
        if min(lo,hi) <= target <= max(lo,hi) and hi != lo:
            p = (target-lo)/(hi-lo)
            probs = [0.,0.,0.,0.]; probs[index]=1-p; probs[index+1]=p
            candidates.append((abs((1-p)*lo+p*hi-target), index, probs))
    if candidates:
        probs = min(candidates)[2]
    else:
        nearest = min(range(4), key=lambda i: (abs(costs[i]-target), i))
        probs = [float(i == nearest) for i in range(4)]
    predicted = sum(p*c for p,c in zip(probs,costs))
    return dict(probabilities=probs, target_macs=float(target), predicted_macs=predicted,
                predicted_relative_gap=abs(predicted-target)/max(target,1e-12), depth_macs=costs,
                matchable=abs(predicted-target)/max(target,1e-12)<=.05,
                status="matched_prediction" if abs(predicted-target)/max(target,1e-12)<=.05 else "unmatchable")


def execution_parameters(options, job, arm):
    method = job["method"]
    if arm.startswith("tau_"):
        return float(arm[4:]), None
    if arm not in ("adaptive", "random"):
        return None, None
    calibration = frozen_document(options,"calibration.json")["methods"][method]
    tau = calibration["threshold"]
    if arm == "adaptive":
        return tau, None
    if job["split"] in ("adaptive", "confirm"):
        budgets = frozen_document(options,"budgets.json")
        entry = budgets["methods"][method][",".join(map(str,job["config"]))]
        return None, entry["probabilities"]
    return None, calibration["random_budget"]["probabilities"]


def evaluate_shard(options, job, stop):
    from open_score.eval.anchors import BLUE_STRATEGY
    from open_score.eval.protocol import config_dict
    from open_score.rules import register_end_to_end_policy, run_episode
    from open_score.utils.logging import ExperimentLogger
    method, seed, config = job["method"], job["seed"], job["config"]
    policy, t_env, identity, params = load_frozen_policy(options, method, seed)
    name = f"{PROFILE}_{method}_{seed}_{os.getpid()}"
    register_end_to_end_policy("red", name, name, lambda _: policy)
    logger = ExperimentLogger(options.output, method, seed, "train", env="had")
    done = {record_key(row) for row in loop_records(options, method, seed, config, job["split"])}
    calibration = None
    if method in LOOP_METHODS and (any(a in ("adaptive","random") for a in job["arms"])
                                  or job["split"] in ("eval", "confirm")):
        calibration = frozen_document(options,"calibration.json")["methods"][method]
        if calibration["checkpoints"][str(seed)] != identity:
            raise ValueError("Calibrated checkpoint changed; frozen threshold may not transfer")
    selection = (frozen_document(options,"selection.json")
                 if job["split"] in ("eval", "adaptive", "confirm") and method in LOOP_METHODS else None)
    for scene in range(job["start"], job["start"]+job["quota"]):
        for arm in job["arms"]:
            if stop.is_set():
                return "stopped"
            row = dict(method=method,seed=seed,split=job["split"],arm=arm,config=config_dict(config),
                episode_seed=scene,checkpoint=identity["checkpoint"],actor_params=params,protocol=REVISION)
            if selection is not None:
                qualification = selection["methods"][method]
                row.update(selection_status=selection["status"],selected_family=selection["selected_method"],
                           loop_eligible=qualification["loop_eligible"],
                           adaptive_eligible=qualification["adaptive_eligible"])
            if record_key(row) in done:
                continue
            intervention = None
            if job["split"] == "mechanism" and not arm.startswith("fixed"):
                configure_execution(policy,"fixed4")
                intervention = FeedbackIntervention(policy.mac.agent,arm)
            elif arm == "adaptive" and calibration["mode"] == "fixed":
                configure_execution(policy,"fixed4")
            else:
                tau, probabilities = execution_parameters(options,job,arm)
                configure_execution(policy,arm,tau,probabilities)
            if method in LOOP_METHODS:
                policy.set_loop_seed(decision_seed(seed,scene))
            episode_started = time.perf_counter()
            try:
                result = run_episode(red=config[0],blue=config[1],targets=config[2],seed=scene,
                    red_strategy={"architecture":"end_to_end","policy":name},blue_strategy=BLUE_STRATEGY,
                    max_steps=100,record=False,task_mode="damage",spatial_dim=2,
                    target_initialization="random",diagnostics=True,record_events=False,retain_trajectory=False)
                if intervention is not None:
                    row["feedback_intervention_stats"] = intervention.summary()
            finally:
                if intervention is not None:intervention.clear()
            row["episode_wall_seconds"] = time.perf_counter() - episode_started
            stats = policy.get_loop_stats() if method in LOOP_METHODS else {}
            row["loop_stats"] = {key:float(stats.get(key,0)) for key in LOOP_STATS}
            if job["split"] == "mechanism":
                row.update(protocol=MECHANISM_REVISION,analysis_role="frozen_model_diagnostic",
                           formal_ood=False,method_selection_allowed=False)
                if intervention is not None:
                    counts=row["feedback_intervention_stats"]
                    key=("query_updates_overwritten" if arm=="query_update_frozen" else
                         "reverse_queries_clamped" if arm=="reverse_kv_clamp" else
                         "entity_hidden_resets" if arm=="entity_gru_hidden_reset" else
                         "slot_hidden_resets" if arm=="slot_gru_hidden_reset" else "independent_slot_routes")
                    repeats=3 if arm=="query_update_frozen" else 4
                    expected=repeats*int(stats["decisions"])
                    if counts.get(key,0)!=expected or stats["read_calls"]!=4*stats["decisions"]:
                        raise RuntimeError(f"Feedback intervention coverage mismatch: {arm}/{counts}/{stats}")
                    row["feedback_intervention_validated"]=True
            if job["split"] == "calibrate" and scene == job["start"]:
                row["checkpoint_identity"] = identity
            if job["split"] == "budget_only":
                # OOD pilot has compute counters only: never persist/inspect its task reward.
                row["budget_only"] = True
            else:
                summary = result["episode_summary"]
                row.update({key:summary[key] for key in ("D", "return", "ep_len")})
                logger.episodes([dict({**result["episode_summary"], **row},phase="final_eval",arm=f"{job['split']}:{arm}",
                    cycle_depth=int(arm[-1]) if arm.startswith("fixed") else None,
                    checkpoint=identity["checkpoint"]+f"/{job['split']}/{arm}",
                    artifact_id=identity["checkpoint"],eval_point=50,t_env=t_env)])
            append_loop_record(options,row)
            done.add(record_key(row))
        if (scene-job["start"]+1) % 10 == 0:
            print(f"{method} seed={seed} {config} {job['split']} {scene-job['start']+1}/{job['quota']}",flush=True)
    return "completed"


def logical_devices(spec):
    inherited = tuple(filter(None,os.environ.get("CUDA_VISIBLE_DEVICES","").split(",")))
    if spec == "all":
        import torch
        count = torch.cuda.device_count()
        if count <= 0:
            raise ValueError("--devices all found no visible CUDA GPUs")
        indices = tuple(range(count))
    else:
        try:
            indices = tuple(int(v) for v in spec.split(","))
        except ValueError as error:
            raise ValueError("--devices expects all or logical CUDA indices") from error
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("Choose distinct nonnegative logical devices")
    if inherited and max(indices) >= len(inherited):
        raise ValueError("GPU index exceeds inherited visible devices")
    return tuple(inherited[i] if inherited else str(i) for i in indices)


def shard_worker(options, job, gpu, stop, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if options.stage == "run":
        options=copy.copy(options)
        options.stage=job["split"]
    directory = options.output / "had" / job["method"] / "train" / f"seed_{job['seed']}"
    directory.mkdir(parents=True,exist_ok=True)
    label = "train" if job["config"] is None else "_".join(map(str,job["config"]))
    split=job["split"]
    # Same path on every server; no hostname/GPU in the shared task identity.
    claimpath=directory/f".leaf1009_{split}_{label}.lock"
    try:
        with process_lock(claimpath) as claimed:
            if not claimed:
                results.put((job,"claimed_elsewhere",None))
                return
            logpath=directory/f"{split}_{label}.log"
            with logpath.open("a",encoding="utf-8",buffering=1) as stream,redirect_stdout(stream),redirect_stderr(stream):
                try:
                    status = (train_one(options,job["method"],job["seed"],stop) if options.stage == "train" else
                              evaluate_nomem_depth_shard(options,job,stop) if options.stage == "depth-eval" else
                              counterfactual_shard(options,job,stop) if options.stage == "counterfactual" else
                              evaluate_one(options,job["method"],job["seed"],stop) if options.suite == "legacy" else
                              evaluate_shard(options,job,stop))
                    results.put((job,status,None))
                except BaseException:
                    error = traceback.format_exc();print(error,flush=True)
                    results.put((job,"failed",error))
    except BaseException:
        results.put((job,"failed",traceback.format_exc()))


def gpu_slots(devices,jobs_per_gpu):
    return tuple((gpu,slot) for slot in range(jobs_per_gpu) for gpu in devices)


def global_coverage(options,jobs):
    if options.stage == "depth-eval":
        return nomem_depth_coverage(nomem_depth_rows(options, options.nomem_depth_context))
    if options.stage=="train":
        from open_score.eval.protocol import training_finished
        matrix=([(job["method"],job["seed"]) for job in jobs] if options.suite=="legacy"
                else [(m,s) for m in LOOP_METHODS for s in (0,1,2)])
        def finished(method,seed):
            directory=run_dir(options.output,method,seed)
            try:
                cfg=json.loads((directory/"config.json").read_text(encoding="utf-8-sig"))
            except (OSError,ValueError):return False
            expected_budget=options.steps if options.suite=="legacy" else 1_000_000
            if cfg.get("method")!=method or int(cfg.get("seed",-1))!=seed or int(cfg.get("t_max",0))!=expected_budget:return False
            if options.suite!="legacy" and cfg.get("implementation_revision")!=REVISION:return False
            return training_finished(directory)
        complete=sum(finished(m,s) for m,s in matrix)
        return dict(kind="global_training",complete=complete,total=len(matrix),global_complete=complete==len(matrix))
    if options.suite=="legacy":
        return dict(kind="legacy_record_coverage",global_complete=None)
    records=loop_records(options)
    cells=defaultdict(list)
    for row in records:
        cells[row["method"],int(row["seed"]),config_tuple(row["config"]),row["split"],row["arm"]].append(row)
    complete=0
    for job in jobs:
        scenes=set(range(job["start"],job["start"]+job["quota"]))
        if job["split"]=="counterfactual":
            actual={int(r["episode_seed"]) for r in records if r["method"]==job["method"] and int(r["seed"])==job["seed"]
                and config_tuple(r["config"])==tuple(job["config"]) and r["split"]=="counterfactual" and r.get("scene_complete")}
            complete+=actual==scenes
        else:
            values=[cells[job["method"],job["seed"],tuple(job["config"]),job["split"],arm] for arm in job["arms"]]
            complete+=all({int(r["episode_seed"]) for r in v}==scenes and len(v)==len(scenes)
                          and len({r["checkpoint"] for r in v})==1 for v in values)
    return dict(kind="global_shard_records",complete=complete,total=len(jobs),global_complete=complete==len(jobs))


def save_runner_state(options,completed,pending,interrupted,coverage):
    # Lock a separate stable inode: runner_state.json itself is atomically replaced.
    with process_lock(options.output/".runner_state.lock",blocking=True):
        path=options.output/"runner_state.json"
        state=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        host=state.setdefault("hosts",{}).setdefault(socket.gethostname(),{})
        host[options.stage]=dict(pid=os.getpid(),task_results=completed,pending=pending,
                                local_completed=sum(r["status"] in ("completed","retained") for r in completed),
                                claimed_elsewhere=sum(r["status"]=="claimed_elsewhere" for r in completed),
                                interrupted=interrupted,coverage=coverage)
        atomic_document(path,state)


def dispatch_shards(options, jobs, release=None):
    devices = logical_devices(options.devices)
    slots=gpu_slots(devices,options.jobs_per_gpu)
    ctx = get_context("spawn")
    stop, results = ctx.Event(),ctx.Queue()
    handlers = {}
    for signum in (signal.SIGINT,signal.SIGTERM):
        handlers[signum] = signal.getsignal(signum)
        signal.signal(signum,lambda *_:stop.set())
    waiting, active, completed, failures = list(jobs),{},[],[]
    deadline=time.monotonic()+options.max_minutes*60 if options.max_minutes is not None else None
    deadline_reached=False
    unreleased=release is not None
    refresh=0.
    try:
        while waiting or active or unreleased:
            if release is not None and unreleased and not stop.is_set() and time.monotonic()>=refresh:
                additions,unreleased=release()
                jobs.extend(additions)
                waiting.extend(additions)
                waiting.sort(key=lambda job:{"calibrate":0,"select":1,"budget_only":2}.get(job["split"],3))
                refresh=time.monotonic()+30
            if deadline is not None and time.monotonic()>=deadline and not deadline_reached:
                deadline_reached=True;stop.set()
                print("Time limit reached; finishing current episodes and preserving resumable records.",flush=True)
            if not stop.is_set():
                for slot_key in slots:
                    while slot_key not in active and waiting:
                        gpu,_=slot_key
                        job = waiting.pop(0)
                        if release is not None:
                            directory=run_dir(options.output,job["method"],job["seed"])
                            label="_".join(map(str,job["config"]))
                            with process_lock(directory/f".leaf1009_{job['split']}_{label}.lock") as available:
                                if not available:
                                    completed.append(dict(job=job,status="claimed_elsewhere",error=None))
                                    continue
                        process = ctx.Process(target=shard_worker,args=(options,job,gpu,stop,results))
                        process.start(); active[slot_key]=(process,job)
            try:
                job,status,error = results.get(timeout=1)
                completed.append(dict(job=job,status=status,error=error))
                print(json.dumps(dict(job=job,status=status,error=error),ensure_ascii=False),flush=True)
                if status == "failed":
                    failures.append(job)
                    if release is not None:stop.set()
            except Empty:
                pass
            for slot_key,(process,job) in list(active.items()):
                if not process.is_alive():
                    process.join()
                    if process.exitcode and job not in failures:
                        failures.append(job)
                        if release is not None:stop.set()
                    del active[slot_key]
            if stop.is_set() and not active:
                break
        while True:
            try:
                job,status,error = results.get_nowait()
                completed.append(dict(job=job,status=status,error=error))
                if status == "failed": failures.append(job)
            except Empty:
                break
    finally:
        for signum,handler in handlers.items(): signal.signal(signum,handler)
    coverage=global_coverage(options,jobs)
    save_runner_state(options,completed,waiting,stop.is_set(),coverage)
    print(json.dumps(dict(status="local_queue_drained",host=socket.gethostname(),
        jobs_per_gpu=options.jobs_per_gpu,claimed_elsewhere=sum(r["status"]=="claimed_elsewhere" for r in completed),
        coverage=coverage),ensure_ascii=False),flush=True)
    if not coverage.get("global_complete"):
        print("Local tasks ended; global data may still be running on another server. Rerun this stage after shared coverage is complete.",flush=True)
    return 1 if failures else 130 if stop.is_set() else 1 if options.stage in ("run","depth-eval") and not coverage.get("global_complete") else 0


def delta(damage):
    return max(.01,.02*float(damage))


def run_evaluation(options):
    """Release independent banks immediately; publish ID choices before their users."""
    from itertools import zip_longest
    banks={}
    for stage in ("calibrate","select"):
        phase=copy.copy(options);phase.stage=stage
        banks[stage]=stage_jobs(phase,LOOP_METHODS)
    for stage,selected in (("mechanism",LOOP_METHODS),("counterfactual",LOOP_METHODS),
                           ("eval",LOOP_METHODS),("eval",TIERS["references"]),
                           ("adaptive",LOOP_METHODS)):
        phase=copy.copy(options);phase.stage=stage
        name="references" if stage=="eval" and selected==TIERS["references"] else stage
        banks[name]=stage_jobs(phase,selected)
    phase=copy.copy(options);phase.stage="adaptive"
    banks["budget_only"]=stage_jobs(phase,LOOP_METHODS,split="budget_only",
        arms=["fixed1","fixed2","fixed3","fixed4","adaptive"],
        configs=tuple(c for c in FINAL_CONFIGS if c not in GROUPS["ID"]),start=43000,quota=10)
    released=set()
    existing=defaultdict(lambda:dict(scenes=set(),checkpoints=set(),count=0))
    cf_scenes=defaultdict(set)
    for row in loop_records(options):
        prefix=(row["split"],row["method"],int(row["seed"]),config_tuple(row["config"]))
        if row["split"]=="counterfactual":
            if row.get("scene_complete"):cf_scenes[prefix].add(int(row["episode_seed"]))
            continue
        cell=existing[(*prefix,row["arm"])]
        cell["scenes"].add(int(row["episode_seed"]))
        cell["checkpoints"].add(row["checkpoint"]);cell["count"]+=1

    def retained(job):
        prefix=(job["split"],job["method"],job["seed"],tuple(job["config"]))
        scenes=set(range(job["start"],job["start"]+job["quota"]))
        if job["split"]=="counterfactual":
            return cf_scenes[prefix]==scenes
        values=[existing[(*prefix,arm)] for arm in job["arms"]]
        return all(v["scenes"]==scenes and v["count"]==len(scenes)
                   and len(v["checkpoints"])==1 for v in values)

    def publish(name,stage,freeze,jobs,*args):
        if (options.output/name).is_file():
            document=frozen_document(options,name)
            if name!="budgets.json" or all(m in document["methods"] for m in LOOP_METHODS):return True
        if not global_coverage(options,jobs)["global_complete"]:return False
        phase=copy.copy(options);phase.stage=stage
        return freeze(phase,*args)

    def release():
        calibrated=publish("calibration.json","calibrate",freeze_calibration,banks["calibrate"])
        selected=calibrated and publish("selection.json","select",freeze_selection,banks["select"])
        budgeted=calibrated and publish("budgets.json","adaptive",freeze_budgets,banks["budget_only"],LOOP_METHODS)
        ready=["calibrate","mechanism","references"]
        if calibrated:ready.extend(("select","budget_only"))
        if selected:ready.extend(("eval","counterfactual"))
        if selected and budgeted:
            ready.extend(("adaptive","confirm"))
            if "confirm" not in banks:
                phase=copy.copy(options);phase.stage="confirm"
                winner=frozen_document(options,"selection.json")["selected_method"]
                banks["confirm"]=stage_jobs(phase,(winner,*TIERS["references"]))
        new=[name for name in ready if name not in released]
        released.update(new)
        if new:print("Dependency release: "+", ".join(new),flush=True)
        # Keep protocol prerequisites first; fill remaining slots with independent work.
        priority=[j for name in new if name in ("calibrate","select","budget_only") for j in banks[name]]
        independent=[banks[name] for name in new if name not in ("calibrate","select","budget_only")]
        additions=priority+[j for row in zip_longest(*independent) for j in row if j is not None]
        return additions,len(released)<9

    # Retained shards still enter the coverage matrix, but need no worker/GPU.
    def pending_release():
        additions,pending=release()
        skipped=[j for j in additions if retained(j)]
        all_jobs.extend(additions)
        if skipped:print(f"Retained complete shards without GPU startup: {len(skipped)}",flush=True)
        return [j for j in additions if not retained(j)],pending

    all_jobs=[]
    result=dispatch_shards(options,[],release=pending_release)
    coverage=global_coverage(options,all_jobs)
    print(json.dumps(dict(status="dependency_run_coverage",coverage=coverage),ensure_ascii=False),flush=True)
    if not result and not coverage["global_complete"]:result=1
    # Other servers may still own shards. Publication is serialized and never
    # turns local queue completion into a claim of global completion.
    with process_lock(options.output/".report.lock",blocking=True):report(options)
    return result


class IncompleteQuota(ValueError):
    pass


def freeze_when_complete(name):
    """One short publication owner; missing shared coverage leaves a rerunnable stage."""
    def decorate(function):
        def guarded(options,*args):
            with process_lock(options.output/f".{name}.lock") as claimed:
                if not claimed:
                    print(f"Freeze {name} is owned by another server; no local publication.",flush=True)
                    return False
                try:
                    function(options,*args)
                except IncompleteQuota as error:
                    print(f"Global quota incomplete: {error}. {name} not frozen; rerun this stage after other servers finish.",flush=True)
                    return False
                return True
        return guarded
    return decorate


def equal_summary(rows, method, split, arm, configs, quota, seeds=(0,1,2), require_reward=True):
    chosen = [r for r in rows if r["method"]==method and r["split"]==split and r["arm"]==arm]
    cells = defaultdict(list)
    for row in chosen: cells[int(row["seed"]),config_tuple(row["config"])].append(row)
    if any(len(cells[s,c])!=quota for s in seeds for c in configs):
        raise IncompleteQuota(f"{method} {split} {arm}")
    scene_start={"calibrate":42000,"select":40000,"budget_only":43000}.get(split)
    if scene_start is not None and any({int(r["episode_seed"]) for r in cells[s,c]} != set(range(scene_start,scene_start+quota))
                                       for s in seeds for c in configs):
        raise IncompleteQuota(f"Unexpected scene coverage for {method} {split} {arm}")
    per_seed_checkpoints = {}
    for seed in seeds:
        checkpoints = {r["checkpoint"] for r in chosen if int(r["seed"]) == seed}
        if len(checkpoints) != 1:
            raise ValueError(f"Mixed checkpoints for {method} seed={seed} {split} {arm}")
        per_seed_checkpoints[str(seed)] = next(iter(checkpoints))
    per_seed_D, per_seed_cost = [],[]
    for seed in seeds:
        per_seed_cost.append(statistics.mean(statistics.mean(r["loop_stats"]["macs"] for r in cells[seed,c]) for c in configs))
        if require_reward:
            per_seed_D.append(statistics.mean(statistics.mean(float(r["D"]) for r in cells[seed,c]) for c in configs))
    value = dict(macs=statistics.mean(per_seed_cost),per_seed_macs=per_seed_cost,
                 actor_params=chosen[0]["actor_params"],per_seed_checkpoints=per_seed_checkpoints)
    if require_reward: value.update(D=statistics.mean(per_seed_D),per_seed_D=per_seed_D)
    return value


@freeze_when_complete("calibration")
def freeze_calibration(options):
    path = options.output / "calibration.json"
    if path.is_file():
        frozen_document(options,path.name)
        print("Retained frozen calibration; no threshold retuning")
        return
    rows = loop_records(options)
    entries = {}
    for method in methods(options):
        fixed = {d:equal_summary(rows,method,"calibrate",f"fixed{d}",GROUPS["ID"],40) for d in range(1,5)}
        trials = {f"{tau:g}":equal_summary(rows,method,"calibrate",f"tau_{tau:g}",GROUPS["ID"],40) for tau in THRESHOLDS}
        if any(v["per_seed_checkpoints"] != fixed[4]["per_seed_checkpoints"] for v in (*fixed.values(), *trials.values())):
            raise ValueError(f"Calibration arms have different per-seed checkpoints: {method}")
        valid = [(r["macs"],float(tau),r) for tau,r in trials.items() if r["D"]<=fixed[4]["D"]+delta(fixed[4]["D"])]
        if valid:
            cost,tau,chosen=min(valid,key=lambda item:(item[0],item[1])); mode="adaptive"
        else:
            tau=None;chosen=fixed[4];mode="fixed"
        identities = {}
        for seed in (0,1,2):
            identities[str(seed)] = next(r["checkpoint_identity"] for r in rows if r["method"]==method and r["seed"]==seed and r["split"]=="calibrate" and r.get("checkpoint_identity"))
        entries[method] = dict(mode=mode,threshold=tau,chosen=chosen,fixed=fixed,trials=trials,
            random_budget=budget_mixture({d:v["macs"] for d,v in fixed.items()},chosen["macs"]),checkpoints=identities)
    atomic_document(path,dict(frozen=True,revision=REVISION,split="ID_calibration",scene_range=[42000,42039],
                              tolerance="max(0.01,0.02*D)",methods=entries))


@freeze_when_complete("selection")
def freeze_selection(options):
    path = options.output / "selection.json"
    if path.is_file():
        frozen_document(options,path.name);print("Retained frozen family selection");return
    rows = loop_records(options); entries={}
    for method in LOOP_METHODS:
        values = {arm:equal_summary(rows,method,"select",arm,GROUPS["ID"],100) for arm in
                  ("fixed1","fixed2","fixed3","fixed4","adaptive","random")}
        if any(v["per_seed_checkpoints"] != values["fixed4"]["per_seed_checkpoints"] for v in values.values()):
            raise ValueError(f"Selection arms have different per-seed checkpoints: {method}")
        r1,r4,adaptive,random_arm = (values[a] for a in ("fixed1","fixed4","adaptive","random"))
        gains = [a-b for a,b in zip(r1["per_seed_D"],r4["per_seed_D"])]
        loop_ok = (r1["D"]-r4["D"]>delta(r1["D"]) and sum(v>0 for v in gains)>=2 and
                   all(b<=a+delta(a) for a,b in zip(r1["per_seed_D"],r4["per_seed_D"])))
        harm_ok = adaptive["D"]<=r4["D"]+delta(r4["D"])
        saving = 1-adaptive["macs"]/max(r4["macs"],1e-12)
        gap = abs(random_arm["macs"]-adaptive["macs"])/max(adaptive["macs"],1e-12)
        random_gain = random_arm["D"]-adaptive["D"]
        adaptive_ok = harm_ok and (saving>=.10 or (gap<=.05 and random_gain>delta(random_arm["D"])))
        entries[method] = dict(arms=values,loop_eligible=loop_ok,adaptive_eligible=adaptive_ok,
            eligible=loop_ok and adaptive_ok,loop_gains_per_seed=gains,compute_saving=saving,
            random_budget_relative_gap=gap,matched_random_gain=random_gain)
    candidates = [m for m in LOOP_METHODS if entries[m]["eligible"]]
    status = "qualified" if candidates else "loop_unverified"
    if not candidates: candidates=list(LOOP_METHODS)
    best = min(entries[m]["arms"]["fixed4"]["D"] for m in candidates)
    near = [m for m in candidates if entries[m]["arms"]["fixed4"]["D"]<=best+delta(best)]
    order = {m:i for i,m in enumerate((LOOP_METHODS[0],LOOP_METHODS[3],LOOP_METHODS[2],LOOP_METHODS[1],LOOP_METHODS[4]))}
    selected = min(near,key=lambda m:(entries[m]["arms"]["fixed4"]["macs"],
                                      entries[m]["arms"]["fixed4"]["actor_params"],order[m]))
    atomic_document(path,dict(frozen=True,revision=REVISION,selected_method=selected,status=status,
        scene_range=[40000,40099],one_family_across_seeds=True,methods=entries,
        tie_rule="ID D4 within delta(best); lower fixed4 MACs, actor parameters, M1/M4/M3/M2/M5"))
    print(json.dumps(dict(selected_method=selected,status=status),ensure_ascii=False))


@freeze_when_complete("budgets")
def freeze_budgets(options, selected):
    path = options.output / "budgets.json"
    previous = frozen_document(options,path.name) if path.is_file() else dict(frozen=True,revision=REVISION,methods={})
    rows = loop_records(options)
    calibration = frozen_document(options,"calibration.json")
    for method in selected:
        if method in TIERS["references"] or method in previous["methods"]: continue
        entries={}
        for config in FINAL_CONFIGS:
            if config in GROUPS["ID"]:
                costs = {d:equal_summary(rows,method,"calibrate",f"fixed{d}",(config,),40,require_reward=False)["macs"] for d in range(1,5)}
                chosen=calibration["methods"][method]
                arm=f"tau_{chosen['threshold']:g}" if chosen["mode"]=="adaptive" else "fixed4"
                target=equal_summary(rows,method,"calibrate",arm,(config,),40,require_reward=False)["macs"]
            else:
                costs = {d:equal_summary(rows,method,"budget_only",f"fixed{d}",(config,),10,require_reward=False)["macs"] for d in range(1,5)}
                target = equal_summary(rows,method,"budget_only","adaptive",(config,),10,require_reward=False)["macs"]
            entries[",".join(map(str,config))] = budget_mixture(costs,target)
        previous["methods"][method]=entries
    previous["scene_range"]=[43000,43009]
    previous["reward_observed_or_tuned"]=False
    previous["thresholds_source"]="calibration.json: ID only"
    atomic_document(path,previous)


def ensure_budget_jobs(options, selected):
    existing = frozen_document(options,"budgets.json")["methods"] if (options.output/"budgets.json").is_file() else {}
    missing = [m for m in selected if m in LOOP_METHODS and m not in existing]
    if not missing: return 0
    jobs = stage_jobs(options,missing,split="budget_only",arms=["fixed1","fixed2","fixed3","fixed4","adaptive"],
                      configs=tuple(c for c in FINAL_CONFIGS if c not in GROUPS["ID"]),start=43000,quota=10)
    result = dispatch_shards(options,jobs)
    if not result and not freeze_budgets(options,missing):return None
    return result



def rng_snapshot():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def rng_restore(snapshot):
    import numpy as np
    import torch
    random.setstate(snapshot["python"]);np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"]:torch.cuda.set_rng_state_all(snapshot["cuda"])


def policy_snapshot(policy):
    """Copy runtime state, never weights; includes adapter batch, caches and loop RNG."""
    import torch
    excluded={"mac","args","scheme","groups","preprocess","mixer","device"}
    state=dict(policy={k:copy.deepcopy(v) for k,v in vars(policy).items() if k not in excluded},
               mac={},modules=[],generators=[])
    for key,value in vars(policy.mac).items():
        if isinstance(value,torch.Generator):
            state["generators"].append((policy.mac,key,value.get_state().clone()))
        elif key not in ("args","agent") and not isinstance(value,torch.nn.Module):
            state["mac"][key]=copy.deepcopy(value)
    # Plain transient attributes only: torch Module parameters/buffers are immutable here.
    for module in policy.mac.agent.modules():
        values={}
        for key,value in vars(module).items():
            if isinstance(value,torch.Generator):
                state["generators"].append((module,key,value.get_state().clone()))
            elif (key.startswith("last_") or key.startswith("loop_") or key.startswith("_loop_")
                  or key in ("eval_depth","decision_depths","intent_override","intent_oracle_actions")):
                values[key]=copy.deepcopy(value)
        state["modules"].append((module,values))
    return state


def policy_restore(policy,state):
    for key,value in state["policy"].items():setattr(policy,key,copy.deepcopy(value))
    for key,value in state["mac"].items():setattr(policy.mac,key,copy.deepcopy(value))
    for module,values in state["modules"]:
        for key,value in values.items():setattr(module,key,copy.deepcopy(value))
    for owner,key,value in state["generators"]:getattr(owner,key).set_state(value)


def complete_snapshot(wrapper,policy):
    if not callable(getattr(wrapper.adapter,"snapshot",None)) or not callable(getattr(wrapper.adapter,"restore",None)):
        raise RuntimeError("counterfactual invalid: native HAD snapshot/restore unavailable; no tails run")
    extra_keys=("red_grouping","blue_grouping","blue_upper","blue_lower","episode_seed")
    return dict(wrapper=wrapper.snapshot(),physical_state=physical_signature(wrapper.get_policy_state()),
                wrapper_extra={k:copy.deepcopy(getattr(wrapper,k)) for k in extra_keys},
                policy=policy_snapshot(policy),rng=rng_snapshot(),stats=copy.deepcopy(policy.get_loop_stats()))


def physical_signature(state):
    """Native observation fields plus previous actions, without model-derived complexity."""
    def entities(rows):
        return tuple((r.id,tuple(r.position),tuple(r.velocity),float(r.health),bool(r.alive),
                      float(r.initial_health),float(r.cumulative_damage)) for r in rows)
    return (int(state.step),int(state.max_steps),str(state.opponent),
            entities(state.red),entities(state.blue),entities(state.targets),
            tuple(sorted(state.last_actions.items())))


def complete_restore(wrapper,policy,snapshot):
    wrapper.restore(snapshot["wrapper"])
    for key,value in snapshot["wrapper_extra"].items():setattr(wrapper,key,copy.deepcopy(value))
    policy_restore(policy,snapshot["policy"])
    rng_restore(snapshot["rng"])
    if physical_signature(wrapper.get_policy_state()) != snapshot["physical_state"]:
        raise RuntimeError("counterfactual invalid: native restored physical state differs from factual snapshot")


def wrapper_act(wrapper,policy,depths=None):
    import numpy as np
    from open_score.envs.had_wrapper import PLANAR_NATIVE_IDS,NATIVE_TO_PLANAR
    state=wrapper.get_policy_state()
    if depths is not None:policy.set_decision_depths(depths)
    actions=policy.act(state,"red",PLANAR_NATIVE_IDS)
    planar=np.asarray([NATIVE_TO_PLANAR[int(actions[entity.id])] for entity in state.red],dtype=np.int64)
    _,done,_=wrapper.step(planar)
    return done


def cached_decision(policy,step):
    """Read the completed decision; no extra forward, random draw or env step."""
    return dict(q=policy.mac._decision_q[0].detach().cpu().clone(),
                hidden=policy.mac.hidden_states[0].detach().cpu().clone(),
                legal=policy.batch["avail_actions"][0,step].bool().detach().cpu().clone())


def decision_effect(current,factual,index,entering):
    """Separate immediate action agreement from the submitted state difference."""
    legal=factual["legal"][index]
    q,q4=current["q"][index],factual["q"][index]
    h,h4=current["hidden"][index],factual["hidden"][index]
    def direction(value):
        value=value.masked_fill(~legal,0)
        centered=(value-value.sum()/legal.sum().clamp_min(1)).masked_fill(~legal,0)
        return centered/centered.norm().clamp_min(1e-6)
    greedy=int(q.masked_fill(~legal,float("-inf")).argmax())
    greedy4=int(q4.masked_fill(~legal,float("-inf")).argmax())
    previous=entering.detach().cpu()
    return dict(decision_greedy_planar=greedy,factual_greedy_planar=greedy4,
                decision_action_matches_r4=greedy==greedy4,
                decision_q=q.tolist(),factual_decision_q=q4.tolist(),
                decision_q_direction_change=float((direction(q)-direction(q4)).norm()),
                decision_hidden_relative_change_vs_r4=float((h-h4).norm()/h4.norm().clamp_min(1e-6)),
                decision_hidden_change_from_entering=float((h-previous).norm()),
                decision_effect_scope="current_action_and_committed_temporal_state")


def complexity_proxies(state,index):
    import numpy as np
    living_blue=[b for b in state.blue if b.alive]
    living_red=[r for r in state.red if r.alive]
    observer=state.red[index]
    if living_blue:
        nearest=lambda red:min(living_blue,key=lambda blue:(float(np.linalg.norm(np.asarray(red.position[:2])-np.asarray(blue.position[:2]))),blue.id)).id
        target=nearest(observer)
        competitors=sum(nearest(red)==target for red in living_red)-1
    else:
        target=None;competitors=0
    from open_score.envs.features import entities_from_state,masks_from_entity_mask
    _,absent=entities_from_state(state)
    observed=int((masks_from_entity_mask(absent)["obs_mask"][index]==0).sum())
    physical=len(living_red)+len(living_blue)+sum(t.alive for t in state.targets)
    xy_distance=lambda a,b:float(np.linalg.norm(np.asarray(a.position[:2])-np.asarray(b.position[:2])))
    blue=next((b for b in living_blue if b.id==target),None)
    live_targets=[t for t in state.targets if t.alive]
    # Retain the legacy field, but name its actual meaning explicitly. The
    # actor's entity mask retains all target slots, including destroyed targets.
    return dict(visible_count=physical,physical_alive_count=physical,observed_entity_count=observed,
                alive_red_count=len(living_red),alive_blue_count=len(living_blue),
                alive_target_count=len(live_targets),observed_target_count=len(state.targets),
                same_nearest_blue_competitors=competitors,nearest_blue_id=target,
                observer_nearest_blue_distance=xy_distance(observer,blue) if blue is not None else None,
                observer_nearest_alive_target_distance=min(xy_distance(observer,t) for t in live_targets) if live_targets else None,
                nearest_blue_target_distance=min(xy_distance(blue,t) for t in state.targets) if blue is not None and state.targets else None,
                distance_geometry="planar_xy")


def adaptive_snapshot_depths(options,method,policy,wrapper,snapshot):
    calibration=frozen_document(options,"calibration.json")["methods"][method]
    complete_restore(wrapper,policy,snapshot)
    if calibration["mode"]=="fixed":return [4 if r.alive else 0 for r in wrapper.get_policy_state().red]
    configure_execution(policy,"adaptive",calibration["threshold"])
    from open_score.envs.had_wrapper import PLANAR_NATIVE_IDS
    policy.act(wrapper.get_policy_state(),"red",PLANAR_NATIVE_IDS)
    stats=getattr(policy.mac.agent,"last_loop_stats",{}) or {}
    depths=stats.get("depths",stats.get("selected_depths"))
    if hasattr(depths,"detach"):depths=depths.detach().cpu().tolist()
    while depths and isinstance(depths[0],list):depths=depths[0]
    complete_restore(wrapper,policy,snapshot)
    return depths


def counterfactual_shard(options,job,stop):
    import numpy as np
    from open_score.envs.had_wrapper import HADWrapper
    from open_score.eval.protocol import config_dict
    method,seed,config=job["method"],job["seed"],job["config"]
    policy,t_env,identity,params=load_frozen_policy(options,method,seed)
    calibration=frozen_document(options,"calibration.json")["methods"][method]
    if calibration["checkpoints"][str(seed)]!=identity:raise ValueError("Counterfactual checkpoint differs from calibration")
    done_keys={record_key(r) for r in loop_records(options,method,seed,config,"counterfactual")}
    shard_restore_verified=False
    for scene in range(job["start"],job["start"]+job["quota"]):
        if stop.is_set():return "stopped"
        # A complete scene is recoverable. A partial scene is deterministically replayed to snapshots.
        factual_base=dict(method=method,seed=seed,split="counterfactual",arm="factual4",config=config_dict(config),
            episode_seed=scene,checkpoint=identity["checkpoint"],actor_params=params,protocol=REVISION)
        old_scene=[r for r in loop_records(options,method,seed,config,"counterfactual") if r["episode_seed"]==scene]
        if any(r.get("scene_complete") for r in old_scene):continue
        wrapper=HADWrapper(scale=config,max_steps=100,blue_upper="reactive",blue_lower="rush",
            diagnostics=True,retain_trajectory=False,fold_wipeout_tail=False,shaping_coef=0.,pad="eval",
            target_initialization="random",reward_mode="damage")
        try:
            wrapper.reset(seed=scene,evaluate=True)
            configure_execution(policy,"fixed4");policy.reset();policy.set_loop_seed(decision_seed(seed,scene))
            if not callable(getattr(wrapper.adapter,"snapshot",None)) or not callable(getattr(wrapper.adapter,"restore",None)):
                raise RuntimeError("counterfactual invalid: native snapshot unavailable; no substitute rollout quota")
            snapshots=[];ended=False
            while not ended:
                step=int(wrapper.adapter.step_count)
                if step in (0,5,10):
                    state=wrapper.get_policy_state()
                    living=[i for i,r in enumerate(state.red) if r.alive][:2]
                    if living:snapshots.append((step,living,complete_snapshot(wrapper,policy)))
                ended=wrapper_act(wrapper,policy)
                if snapshots and snapshots[-1][0] == step:
                    snapshots[-1][2]["factual_actions"] = copy.deepcopy(policy.last_result)
                    snapshots[-1][2]["factual_decision"] = cached_decision(policy,step)
            factual_D=float(wrapper.adapter.env.target_damage)
            factual_stats=policy.get_loop_stats()
            factual=dict(factual_base,D=factual_D,return_value=-factual_D,loop_stats=factual_stats,
                         scene_complete=not snapshots,snapshot_count=len(snapshots),source="factual_rollout")
            if record_key(factual) not in done_keys:append_loop_record(options,factual)
            # Validate restoration using an already authorized factual decision,
            # then restore again. No additional physical step or R4 tail is run.
            from open_score.envs.had_wrapper import PLANAR_NATIVE_IDS
            if snapshots and not shard_restore_verified:
                step,indices,snapshot=snapshots[0]
                complete_restore(wrapper,policy,snapshot)
                configure_execution(policy,"fixed4")
                replay_actions=policy.act(wrapper.get_policy_state(),"red",PLANAR_NATIVE_IDS)
                if replay_actions != snapshot["factual_actions"]:
                    raise RuntimeError(f"counterfactual invalid: restored fixed4 actions differ at step {step}")
                shard_restore_verified=True
                complete_restore(wrapper,policy,snapshot)
            for snap_index,(step,indices,snapshot) in enumerate(snapshots):
                adaptive_depths=adaptive_snapshot_depths(options,method,policy,wrapper,snapshot)
                for index in indices:
                    complete_restore(wrapper,policy,snapshot)
                    state=wrapper.get_policy_state();proxies=complexity_proxies(state,index)
                    agent_id=int(state.red[index].id)
                    common=dict(factual_base,snapshot_step=step,agent_id=agent_id,observer_index=index,
                        factual_D=factual_D,depth_adaptive=adaptive_depths[index] if adaptive_depths is not None and index<len(adaptive_depths) else None,
                        restore_check="physical_and_fixed4_action_matched" if shard_restore_verified else None,
                        **proxies)
                    for depth in (1,2,4):
                        row=dict(common,arm=f"forced{depth}",forced_depth=depth)
                        if record_key(row) in done_keys:continue
                        if stop.is_set():return "stopped"
                        if depth==4:
                            row.update(D=factual_D,return_value=-factual_D,source="factual_reuse",intervention_tail=False,
                                       benefit_vs_r4=0.,loop_stats=factual_stats)
                            row.update(decision_effect(snapshot["factual_decision"],snapshot["factual_decision"],index,
                                                       snapshot["policy"]["mac"]["hidden_states"][0,index]))
                            row["decision_greedy_native"]=snapshot["factual_actions"][agent_id]
                        else:
                            complete_restore(wrapper,policy,snapshot)
                            configure_execution(policy,"fixed4")
                            n_agents=int(policy.args.n_agents)
                            override=np.full((1,n_agents),4,dtype=np.int64);override[0,index]=depth
                            ended=wrapper_act(wrapper,policy,override)
                            row.update(decision_effect(cached_decision(policy,step),snapshot["factual_decision"],index,
                                                       snapshot["policy"]["mac"]["hidden_states"][0,index]))
                            row["decision_greedy_native"]=policy.last_result[agent_id]
                            while not ended:
                                if stop.is_set():return "stopped"
                                ended=wrapper_act(wrapper,policy)
                            damage=float(wrapper.adapter.env.target_damage);stats=policy.get_loop_stats()
                            row.update(D=damage,return_value=-damage,source="restored_native_tail",intervention_tail=True,
                                benefit_vs_r4=damage-factual_D,loop_stats=stats,
                                tail_loop_stats={k:float(stats.get(k,0))-float(snapshot["stats"].get(k,0)) for k in LOOP_STATS})
                        row["scene_complete"]=(snap_index==len(snapshots)-1 and index==indices[-1] and depth==4)
                        append_loop_record(options,row);done_keys.add(record_key(row))
        finally:wrapper.close()
    return "completed"


def grouped_loop_summary(rows):
    bank=defaultdict(list)
    for row in rows:
        if "D" not in row or row["split"]=="counterfactual":continue
        bank[row["method"],row["split"],row["arm"],row["checkpoint"],config_tuple(row["config"]),int(row["seed"])].append(row)
    summary=[]
    for (method,split,arm,checkpoint,config,seed),values in sorted(bank.items()):
        summary.append(dict(method=method,split=split,arm=arm,checkpoint=checkpoint,config=list(config),seed=seed,
            episodes=len(values),selection_status=values[0].get("selection_status"),
            D=statistics.mean(float(r["D"]) for r in values),
            reward=-statistics.mean(float(r["D"]) for r in values),
            macs=statistics.mean(float(r["loop_stats"]["macs"]) for r in values)
                if all(r.get("loop_stats",{}).get("macs") is not None for r in values) else None,
            episode_wall_seconds=statistics.mean(float(r["episode_wall_seconds"]) for r in values)
                if all("episode_wall_seconds" in r for r in values) else None,
            mean_depth=statistics.mean(float(r["execution_depth"]) if "execution_depth" in r else
                sum(d*r["loop_stats"].get(f"depth_{d}",0) for d in range(1,5))/max(r["loop_stats"].get("decisions",0),1) for r in values)))
    return summary


def native_mechanism_summary(rows):
    """Paired complete-episode diagnostics; OOD scenes never select a family."""
    import numpy as np
    native = [r for r in rows if r.get("split") == "mechanism"
              and r.get("protocol") == MECHANISM_REVISION and "D" in r]
    registry = read_registry()
    scenes = set(range(44000, 44010))
    seeds = (0, 1, 2)
    bank = {}
    invalid = 0
    for row in native:
        cfg, seed, scene = config_tuple(row["config"]), int(row["seed"]), int(row["episode_seed"])
        if (cfg not in MECHANISM_CONFIGS or seed not in seeds or scene not in scenes
                or not math.isfinite(float(row["D"]))
                or (not row["arm"].startswith("fixed") and not row.get("feedback_intervention_validated"))):
            invalid += 1
            continue
        bank[row["method"], row["arm"], cfg, seed, scene] = row

    def complete_cell(method, arm, cfg, seed):
        values = [bank.get((method, arm, cfg, seed, scene)) for scene in sorted(scenes)]
        if any(r is None for r in values):
            return None
        if len({r["checkpoint"] for r in values}) != 1:
            return None
        return values

    def intervals(seed_values, cfg_scene_values):
        mean = float(np.mean(seed_values))
        sd = float(np.std(seed_values, ddof=1))
        half = 4.30265273 * sd / math.sqrt(3)
        rng = np.random.default_rng(1009)
        replicates = np.zeros(2000)
        # One sampled physical scene carries all three frozen training weights.
        # Configurations remain equally weighted, rather than resampling observers.
        for values in cfg_scene_values:
            means = np.mean(np.asarray(values, dtype=float), axis=0)
            draws = rng.integers(0, len(means), size=(2000, len(means)))
            replicates += means[draws].mean(axis=1) / len(cfg_scene_values)
        return dict(mean=mean, seed_sd=sd, seed_t95=[mean-half, mean+half],
                    conditional_scene_bootstrap95=np.quantile(replicates, [.025, .975]).tolist(),
                    seed_values={str(s):float(v) for s,v in zip(seeds, seed_values)})

    def assess(value):
        if not value or not value["complete"]:
            return "证据不足"
        delta = value["practical_margin"]
        intervals95 = (value["seed_t95"], value["conditional_scene_bootstrap95"])
        seed_values = list(value["seed_values"].values())
        if (value["mean"] >= delta and min(i[0] for i in intervals95) > 0
                and sum(v > 0 for v in seed_values) >= 2 and min(seed_values) >= -delta):
            return "支持"
        if max(i[1] for i in intervals95) <= 0:
            return "未支持"
        if max(abs(x) for i in intervals95 for x in i) <= delta:
            return "未支持"
        return "证据不足"

    def contrast(method, arm, reference="fixed4", configs=MECHANISM_CONFIGS):
        valid, cfg_scene, ref_values = [], [], []
        for cfg in configs:
            left = [complete_cell(method, arm, cfg, seed) for seed in seeds]
            right = [complete_cell(method, reference, cfg, seed) for seed in seeds]
            if any(v is None for v in left+right):
                continue
            if any(a[0]["checkpoint"] != b[0]["checkpoint"] for a,b in zip(left,right)):
                continue
            valid.append(cfg)
            cfg_scene.append([[float(a["D"])-float(b["D"]) for a,b in zip(l,r)] for l,r in zip(left,right)])
            ref_values.append([[float(r["D"]) for r in cell] for cell in right])
        if not valid:
            return dict(arm=arm, reference=reference, complete=False, configs=[], paired_episodes=0, status="证据不足")
        per_seed = np.mean(np.asarray(cfg_scene), axis=(0,2)).tolist()
        value = dict(arm=arm, reference=reference, configs=[list(c) for c in valid],
                     complete=len(valid)==len(configs), paired_episodes=30*len(valid),
                     reference_D=float(np.mean(ref_values)))
        value.update(intervals(per_seed, cfg_scene))
        value["practical_margin"] = max(.01, .02*value["reference_D"])
        value["status"] = assess(value)
        value["sign"] = "positive means the reference has lower damage"
        return value

    def arm_summary(method, arm, configs=MECHANISM_CONFIGS):
        cfg_values, valid = [], []
        for cfg in configs:
            cells = [complete_cell(method,arm,cfg,seed) for seed in seeds]
            if any(c is None for c in cells):
                continue
            valid.append(cfg)
            cfg_values.append([dict(D=statistics.mean(float(r["D"]) for r in cell),
                macs_per_decision=sum(float(r["loop_stats"].get("macs",0)) for r in cell)
                    / max(sum(float(r["loop_stats"].get("decisions",0)) for r in cell),1),
                decisions=sum(float(r["loop_stats"].get("decisions",0)) for r in cell),
                episode_wall_seconds=statistics.mean(float(r["episode_wall_seconds"]) for r in cell)) for cell in cells])
        result = dict(arm=arm, complete=len(valid)==len(configs), configs=[list(c) for c in valid], episodes=30*len(valid))
        if cfg_values:
            for key in ("D","macs_per_decision","decisions","episode_wall_seconds"):
                per_seed = [statistics.mean(cfg[seed][key] for cfg in cfg_values) for seed in seeds]
                result[key] = statistics.mean(per_seed)
                result[key+"_seed_values"] = {str(seed):float(v) for seed,v in zip(seeds,per_seed)}
                result[key+"_seed_sd"] = statistics.stdev(per_seed)
        return result

    methods_summary = []
    for method in LOOP_METHODS:
        cuts = FEEDBACK_ARMS[registry[method]["leaf_loop_core"]][1:]
        arms = tuple(f"fixed{d}" for d in range(1,5)) + cuts
        expected = len(arms)*180
        count = sum(key[0] == method for key in bank)
        gains = {f"R{d}_to_R4":contrast(method,f"fixed{d}") for d in (1,2,3)}
        gains["R1_to_R2"] = contrast(method,"fixed1","fixed2")
        cut_results = {arm:contrast(method,arm) for arm in cuts}
        arm_results = {arm:arm_summary(method,arm) for arm in arms}
        per_config = [dict(config=list(cfg), arms={arm:arm_summary(method,arm,(cfg,)) for arm in arms},
            depth_gains={f"R{d}_to_R4":contrast(method,f"fixed{d}",configs=(cfg,)) for d in (1,2,3)},
            feedback_losses={arm:contrast(method,arm,configs=(cfg,)) for arm in cuts}) for cfg in MECHANISM_CONFIGS]
        group_results = {}
        for name,configs in GROUPS.items():
            represented = tuple(c for c in MECHANISM_CONFIGS if c in configs)
            if represented:
                group_results[name] = dict(configs=[list(c) for c in represented],
                    fixed4=arm_summary(method,"fixed4",represented),
                    depth_gains={f"R{d}_to_R4":contrast(method,f"fixed{d}",configs=represented) for d in (1,2,3)},
                    feedback_losses={arm:contrast(method,arm,configs=represented) for arm in cuts})
        ni = gains["R2_to_R4"]
        ni_status = "证据不足"
        savings = None
        if ni.get("complete") and arm_results["fixed2"].get("complete") and arm_results["fixed4"].get("complete"):
            savings = 1-arm_results["fixed2"]["macs_per_decision"]/max(arm_results["fixed4"]["macs_per_decision"],1)
            upper = max(ni["seed_t95"][1],ni["conditional_scene_bootstrap95"][1])
            lower = min(ni["seed_t95"][0],ni["conditional_scene_bootstrap95"][0])
            if (upper <= ni["practical_margin"] and savings > 0
                    and sum(v <= ni["practical_margin"] for v in ni["seed_values"].values()) >= 2
                    and max(ni["seed_values"].values()) <= 2*ni["practical_margin"]):
                ni_status = "支持"
            elif lower > ni["practical_margin"]:
                ni_status = "未支持"
        methods_summary.append(dict(method=method, records=count, expected_records=expected, complete=count==expected,
            arms=arm_results, depth_gains=gains, feedback_losses=cut_results, per_config=per_config, groups=group_results,
            story_tests=dict(reward_value_R1_to_R4=gains["R1_to_R4"]["status"],
                extra_depth_R2_to_R4=gains["R2_to_R4"]["status"],
                feedback_paths={arm:value["status"] for arm,value in cut_results.items()},
                R2_noninferior_and_lower_matrix_MAC=ni_status),
            fixed2_mac_saving_per_decision=savings))
    return dict(revision=MECHANISM_REVISION, records=len(bank), expected_records=4860,
        complete=len(bank)==4860 and all(m["complete"] for m in methods_summary), invalid_records=invalid,
        scene_range=[44000,44009], configs=[list(c) for c in MECHANISM_CONFIGS], training_seeds=list(seeds),
        checkpoint_policy="same frozen final@1M within each paired training seed",
        uncertainty=dict(seed_t95="three training seed means; Student t df=2, 4.30265273",
            bootstrap="2000 physical config/scene cluster resamples, stratified by equally weighted config; conditional on three frozen weights",
            practical_margin="max(0.01, 0.02 * paired reference mean damage)",
            support_rule="complete quota, mean >= margin, both 95% intervals lower > 0, >=2 positive seeds and no seed loss > margin"),
        formal_ood=False, method_selection_allowed=False,
        compute_definition="config-equal matrix MAC/decision from complete live-policy episodes; excludes scalar operations, not wall-clock speed",
        methods=methods_summary)


def plot_native_mechanisms(analysis, save):
    """Render actual native reward diagnostics, including incomplete coverage."""
    import matplotlib.pyplot as plt
    import numpy as np
    if not analysis["records"]:
        return []
    captions = []
    methods_summary = analysis["methods"]
    labels = {m:f"M{i+1}" for i,m in enumerate(LOOP_METHODS)}
    coverage = f"{analysis['records']}/{analysis['expected_records']} episodes; 10 scenes/config; preliminary OOD"
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    for item in methods_summary:
        xs,ys,sd,cost = [],[],[],[]
        for d in range(1,5):
            arm = item["arms"][f"fixed{d}"]
            if "D" not in arm:
                continue
            xs.append(d);ys.append(arm["D"]);sd.append(arm["D_seed_sd"])
            cost.append(arm["macs_per_decision"]/1e6)
        if xs:
            axes[0].errorbar(xs,ys,yerr=sd,marker="o",capsize=3,label=labels[item["method"]])
            axes[1].plot(xs,cost,marker="o",label=labels[item["method"]])
    axes[0].set(xlabel="Fixed depth",ylabel="Damage D (lower is better)",title="Config-equal reward, mean +/- seed SD")
    axes[1].set(xlabel="Fixed depth",ylabel="Measured matrix MAC / decision (million)",title="Live-policy computation")
    for ax in axes:
        ax.set_xticks([1,2,3,4]);ax.legend();ax.grid(alpha=.2)
    fig.suptitle(coverage,fontsize=10)
    save(fig,"loop_native_depth.png","Native depth reward and measured matrix MAC (complete cells only)")
    fig,ax = plt.subplots(figsize=(11,5))
    y, cut_labels = 0, []
    for item in methods_summary:
        for arm,value in item["feedback_losses"].items():
            if "mean" not in value:
                continue
            low,high = value["conditional_scene_bootstrap95"]
            ax.errorbar(value["mean"],y,xerr=[[max(0,value["mean"]-low)],[max(0,high-value["mean"])]],fmt="o",capsize=4,color="C0")
            for seed,number in value["seed_values"].items():
                ax.scatter(number,y+(.09*(int(seed)-1)),marker="|",color=f"C{int(seed)+1}",s=70)
            cut_labels.append(f"{labels[item['method']]} / {arm}")
            captions.append(f"{labels[item['method']]} {arm}: {value['status']}")
            y += 1
    ax.set(yticks=list(range(y)),yticklabels=cut_labels,
           ylim=(-.5,max(y-.5,.5)),xlabel="Damage(cut) - damage(normal R4); positive favors feedback",title=coverage)
    ax.tick_params(axis="y",labelsize=8)
    ax.axvline(0,color="grey",lw=1);ax.grid(axis="x",alpha=.2)
    save(fig,"loop_native_feedback_reward.png","Native feedback cuts: conditional scene bootstrap95, with separate seed estimates")
    fig,axes = plt.subplots(1,4,figsize=(15,4.5))
    for ax,name in zip(axes,GROUPS):
        names,means,sds = [],[],[]
        for item in methods_summary:
            group = item["groups"].get(name,{})
            value = group.get("fixed4",{})
            if "D" not in value:
                continue
            names.append(labels[item["method"]]);means.append(value["D"]);sds.append(value["D_seed_sd"])
        positions = np.arange(len(names))
        ax.bar(positions,means,yerr=sds,capsize=3,color="C0",alpha=.75)
        ax.set(xticks=positions,xticklabels=names,title=name,ylabel="Damage D (lower is better)")
        ax.grid(axis="y",alpha=.2)
    fig.suptitle(coverage+"; mean +/- training seed SD; no family selection",fontsize=10)
    save(fig,"loop_native_ood_pilot.png","Six-config frozen-weight OOD pilot, not the formal 24-config evaluation")
    return captions


def nomem_depth_report(options):
    """Update only the depth supplement in the single formal docs report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    context = nomem_depth_context(options, stored=True)
    rows = nomem_depth_rows(options, context)
    coverage = nomem_depth_coverage(rows)
    summary = grouped_loop_summary(rows)
    cell_bank = {(r["arm"], tuple(r["config"]), int(r["seed"])): r
                 for r in summary if r["episodes"] == 300}
    groups = {"All24": FINAL_CONFIGS, **GROUPS,
        "Extra": tuple(c for c in FINAL_CONFIGS if not any(c in v for v in GROUPS.values()))}

    def seed_stats(values):
        mean, sd = statistics.mean(values), statistics.stdev(values)
        half = 4.30265273 * sd / math.sqrt(3)
        return dict(mean=mean, seed_sd=sd, seed_values=list(values),
                    seed_t95=[mean-half, mean+half])

    def pool_stats(bank, arm, configs):
        keys = [(arm, cfg, seed) for seed in (0, 1, 2) for cfg in configs]
        if any(key not in bank for key in keys):
            return None
        return seed_stats([statistics.mean(bank[arm, cfg, seed]["D"] for cfg in configs)
                           for seed in (0, 1, 2)])

    def gains(depth_stats):
        result = {}
        for shallow, deep in ((1, 2), (2, 4), (4, 5), (4, 6)):
            a, b = depth_stats[str(shallow)], depth_stats[str(deep)]
            result[f"R{shallow}_to_R{deep}"] = (seed_stats([
                x-y for x, y in zip(a["seed_values"], b["seed_values"])]) if a and b else None)
        return result

    group_results = {}
    for name, configs in groups.items():
        depths = {str(d): pool_stats(cell_bank, f"fixed{d}", configs) for d in NOMEM_DEPTHS}
        group_results[name] = dict(configs=list(configs), depths=depths, reward_gains=gains(depths))
    configuration_results = []
    for cfg in FINAL_CONFIGS:
        depths = {str(d): pool_stats(cell_bank, f"fixed{d}", (cfg,)) for d in NOMEM_DEPTHS}
        configuration_results.append(dict(config=list(cfg), depths=depths, reward_gains=gains(depths)))

    # Compare candidates on the same physical pool, with their frozen final tags.
    candidate_rows = [r for r in loop_records(options, split="eval")
        if r["method"] in LOOP_METHODS and r.get("official_aggregate") is not False
        and r.get("protocol") == REVISION and 9000 <= int(r["episode_seed"]) <= 9299
        and config_tuple(r["config"]) in FINAL_CONFIGS and int(r["seed"]) in (0, 1, 2)
        and r["arm"] in ("fixed1", "fixed2", "fixed3", "fixed4")
        and math.isfinite(float(r["D"]))
        and all(r.get(k, v) == v for k, v in context["environment"].items())]
    calibration = json.loads((options.output / "calibration.json").read_text(encoding="utf-8-sig"))
    candidate_rows = [r for r in candidate_rows if r["checkpoint"] ==
        calibration["methods"][r["method"]]["checkpoints"][str(r["seed"])]["checkpoint"]]
    candidate_summary = grouped_loop_summary(candidate_rows)
    comparisons = {"NoMem": dict(depths=group_results["All24"]["depths"],
                                 reward_gains=group_results["All24"]["reward_gains"])}
    candidate_banks = {}
    for index, method in enumerate(LOOP_METHODS, 1):
        bank = {(r["arm"], tuple(r["config"]), int(r["seed"])): r
            for r in candidate_summary if r["method"] == method and r["episodes"] == 300}
        candidate_banks[method] = bank
        depths = {str(d): pool_stats(bank, f"fixed{d}", FINAL_CONFIGS) for d in (1, 2, 3, 4)}
        comparisons[f"M{index}"] = dict(method=method, depths=depths,
            reward_gains={"R1_to_R2": gains({**depths, "5": None, "6": None})["R1_to_R2"],
                          "R2_to_R4": gains({**depths, "5": None, "6": None})["R2_to_R4"]})
    m4_gains = {}
    for d in (1, 2, 3, 4):
        a, b = comparisons["NoMem"]["depths"][str(d)], comparisons["M4"]["depths"][str(d)]
        m4_gains[str(d)] = seed_stats([x-y for x, y in zip(a["seed_values"], b["seed_values"])]) if a and b else None
    analysis = dict(revision=NOMEM_DEPTH_REVISION, coverage=coverage,
        environment=context["environment"], checkpoints=context["checkpoints"],
        R4_source=dict(split="eval", arm="original", protocol=REVISION,
            metadata="experiment.json; retained checkpoint global_eval_depth=4",
            legacy_rows_have_artifact_id=False),
        groups=group_results, configurations=configuration_results, results=summary,
        comparisons=comparisons, M4_reward_gain_vs_NoMem_at_same_depth=m4_gains,
        interpretation=dict(training_depth_max=4, extrapolation_depths=[5, 6],
            learned_fusion_candidate_set_changes=True,
            effect="whole-episode fixed-depth policy including temporal history",
            selection_use=False))
    reportpath = PROJECT.parent / "docs/LEAF_main1009_完整分析报告.md"
    preserved = reportpath.read_text(encoding="utf-8")
    begin, end = "<!-- leaf1009_nomem_depth_v1:start -->", "<!-- leaf1009_nomem_depth_v1:end -->"
    if begin not in preserved and "\n## 4." not in preserved:
        raise ValueError("Formal docs report lacks the section-4 insertion point")
    figures = options.output / "had/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (name, result) in zip(axes.flat, group_results.items()):
        points = [(int(d), v) for d, v in result["depths"].items() if v is not None]
        if points:
            ax.errorbar([d for d, v in points], [v["mean"] for d, v in points],
                yerr=[v["seed_sd"] for d, v in points], marker="o", capsize=3)
        ax.axvspan(4.5, 6.5, alpha=.08, color="orange")
        ax.set(title=name, xlabel="Fixed execution rounds", ylabel="D (lower is better)",
               xticks=NOMEM_DEPTHS, xlim=(.7, 6.3))
        ax.grid(alpha=.2)
    fig.suptitle("Original NoMem: mean and SD across three training seeds")
    fig.tight_layout()
    fig.savefig(figures / "loop_nomem_depth.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, item in comparisons.items():
        points = [(int(d), v) for d, v in item["depths"].items() if v is not None]
        if points:
            ax.errorbar([d for d, v in points], [v["mean"] for d, v in points],
                yerr=[v["seed_sd"] for d, v in points], marker="o", capsize=2, label=name)
    ax.axvspan(4.5, 6.5, alpha=.08, color="orange")
    ax.set(title="All24: frozen checkpoints and matched scene pool",
           xlabel="Fixed execution rounds", ylabel="D (lower is better)",
           xticks=NOMEM_DEPTHS, xlim=(.7, 6.3))
    ax.grid(alpha=.2)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "loop_nomem_comparison.png", dpi=160)
    plt.close(fig)

    def value_text(value):
        return f"{value['mean']:.4f}±{value['seed_sd']:.4f}" if value else "未完整"

    def gain_text(value):
        return (f"{value['mean']:+.4f}; [{value['seed_t95'][0]:+.4f}, {value['seed_t95'][1]:+.4f}]"
                if value else "未完整")

    lines = [begin, "### 3.10 原 NoMem R1–R6：冻结权重的完整深度补评", "",
        f"逻辑矩阵 {coverage['logical_episodes']:,}/{coverage['expected_episodes']:,} 场；"
        f"完整单元 {coverage['complete_cells']}/{coverage['expected_cells']}；"
        f"复用 R4 {coverage['reused_R4_episodes']:,} 场，新增已完成 {coverage['new_completed_episodes']:,} 场。"
        f"全部完成：{'是' if coverage['global_complete'] else '否，缺口不能视作完整结果'}。", "",
        "原 main0928 三个 final 权重，24 配置、种子 0/1/2、场景 9000–9299，每单元 300 场。"
        "D 越低越好，±为三个训练种子的样本标准差；配置等权。"
        "收益为 D浅−D深，t95 用三个配对种子均值、df=2，不将对局当作独立训练种子。", "",
        "R4 来源为既有正式 eval/original；原行无环境版本和 artifact UUID，"
        "复用依据是原 runner 的权重来源、checkpoint 标签、保留的 final 配置与所属 experiment.json。"
        "映射仅参与汇总，不复制或改写旧逐场记录。新行独立保存 HAD 版本、checkpoint artifact 与执行深度。", "",
        "R5/R6 超过训练最大四轮，同时改变 learned fusion 的候选集合与归一化。"
        "这里测量整场固定深度策略，包括时序历史变化，不能直接解释为单个状态的加深收益。"
        "本补评不重新选择方法或阈值，也不覆盖前文与最后的 M4 方法章节。", "",
        "| 分组 | R1 | R2 | R3 | R4 | R5 | R6 |", "|---|---|---|---|---|---|---|"]
    for name, item in group_results.items():
        lines.append("| " + name + " | " + " | ".join(value_text(item["depths"][str(d)]) for d in NOMEM_DEPTHS) + " |")
    lines += ["", "| 分组 | R1→R2收益; t95 | R2→R4收益; t95 | R4→R5收益; t95 | R4→R6收益; t95 |",
              "|---|---|---|---|---|"]
    for name, item in group_results.items():
        lines.append("| " + name + " | " + " | ".join(gain_text(v) for v in item["reward_gains"].values()) + " |")
    lines += ["", "| 配置 (N_R,N_B,K) | R1 | R2 | R3 | R4 | R5 | R6 |",
              "|---|---|---|---|---|---|---|"]
    for item in configuration_results:
        lines.append("| " + str(tuple(item["config"])) + " | " +
                     " | ".join(value_text(item["depths"][str(d)]) for d in NOMEM_DEPTHS) + " |")
    lines += ["", "| 方法（All24） | R1 | R2 | R3 | R4 | R2→R4奖励收益; t95 |",
              "|---|---|---|---|---|---|"]
    for name, item in comparisons.items():
        lines.append("| " + name + " | " + " | ".join(value_text(item["depths"][str(d)]) for d in (1, 2, 3, 4)) +
                     " | " + gain_text(item["reward_gains"]["R2_to_R4"]) + " |")
    lines += ["", "M4 相对原 NoMem 的同轮数奖励收益（正值代表 M4 更好）："
              + "；".join(f"R{d}: {gain_text(v)}" for d, v in m4_gains.items()) + "。", ""]
    for key, value in group_results["All24"]["reward_gains"].items():
        status = ("证据不足（单元未完整）" if value is None else
                  "支持平均奖励改善" if value["seed_t95"][0] > 0 else
                  "支持平均奖励退化" if value["seed_t95"][1] < 0 else
                  "证据不足（种子 t95 跨零）")
        lines.append(f"- 原 NoMem {key.replace('_to_', '→')}：{status}，{gain_text(value)}。")
    for name in ("loop_nomem_depth.png", "loop_nomem_comparison.png"):
        relative = os.path.relpath(figures / name, reportpath.parent).replace(os.sep, "/")
        lines += ["", f"![NoMem深度补评：真实结果与种子波动]({relative})"]
    lines += ["", "配置×深度×种子的明细与来源覆盖见原 main1009 的 summary.json / loop_summary.csv；"
              "原有正式比较、选型与确认统计保留。", end, ""]
    section = "\n".join(lines)
    if begin in preserved:
        prefix, rest = preserved.split(begin, 1)
        if end not in rest:
            raise ValueError("Incomplete NoMem depth report markers")
        preserved = prefix + section.rstrip("\n") + rest.split(end, 1)[1]
    else:
        prefix, rest = preserved.split("\n## 4.", 1)
        preserved = prefix.rstrip() + "\n\n" + section + "\n## 4." + rest
    with process_lock(options.output / ".nomem_depth_report.lock", blocking=True):
        path = options.output / "summary.json"
        combined = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        combined["nomem_depth_analysis"] = analysis
        atomic_document(path, combined)
        csvpath = options.output / "loop_summary.csv"
        if csvpath.is_file():
            with csvpath.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames
                retained = [r for r in reader if r["split"] != NOMEM_DEPTH_SPLIT]
        else:
            retained = combined.get("loop_candidates", {}).get("results", [])
            fieldnames = list((retained or summary)[0]) if retained or summary else None
        if fieldnames:
            pending = csvpath.with_suffix(".csv.pending")
            with pending.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([*retained, *summary])
            os.replace(pending, csvpath)
        pending = reportpath.with_suffix(".md.pending")
        pending.write_text(preserved, encoding="utf-8")
        os.replace(pending, reportpath)
    print(json.dumps(dict(report=str(reportpath), coverage=coverage), ensure_ascii=False, indent=2))


def report(options):
    """Replace only this protocol's section in the sole existing report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging=analysis_module("utils/logging.py")
    rows=loop_records(options)
    metadata=options.output/"experiment.json"
    if metadata.is_file() and NOMEM_DEPTH_REVISION in json.loads(metadata.read_text(encoding="utf-8-sig")).get("protocols",{}):
        context=nomem_depth_context(options,stored=True)
        mapped=nomem_depth_rows(options,context,[r for r in rows if r["method"]=="regir_nomem"])
        rows=[r for r in rows if r["split"]!=NOMEM_DEPTH_SPLIT]+mapped
    summary=grouped_loop_summary(rows)
    path=options.output/"summary.json"
    combined=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    comparisons=[]
    compare_bank={(r["method"],r["split"],r["arm"],r["checkpoint"],tuple(r["config"]),r["seed"]):r for r in summary if r["episodes"]==300}
    for row in summary:
        if row["episodes"]!=300 or row["arm"]!="adaptive" or row["split"] not in ("adaptive","confirm"):continue
        key=(row["method"],row["split"],"random",row["checkpoint"],tuple(row["config"]),row["seed"])
        random_row=compare_bank.get(key)
        fixed_split="eval" if row["split"]=="adaptive" else "confirm"
        fixed_row=compare_bank.get((row["method"],fixed_split,"fixed4",row["checkpoint"],tuple(row["config"]),row["seed"]))
        value=dict(method=row["method"],split=row["split"],checkpoint=row["checkpoint"],config=row["config"],seed=row["seed"])
        if random_row:
            gap=abs(random_row["macs"]-row["macs"])/max(row["macs"],1e-12)
            value.update(actual_random_budget_relative_gap=gap,random_budget_matchable=gap<=.05,
                         adaptive_reward_gain_vs_random=row["reward"]-random_row["reward"])
        if fixed_row:
            value.update(adaptive_reward_gain_vs_fixed4=row["reward"]-fixed_row["reward"],
                         compute_saving_vs_fixed4=1-row["macs"]/max(fixed_row["macs"],1e-12))
        comparisons.append(value)
    combined["loop_candidates"]=dict(revision=REVISION,results=summary,actual_budget_comparisons=comparisons,
        counterfactual_records=sum(r["split"]=="counterfactual" for r in rows),
        intervention_tails=sum(bool(r.get("intervention_tail")) for r in rows))
    native = native_mechanism_summary(rows)
    combined["native_mechanism_analysis"] = native
    if native["complete"]:
        combined.setdefault("native_mechanism_execution",{}).update(status="completed",
            completed_episodes=native["records"],remaining_episodes=0)
    atomic_document(path,combined)
    csvpath=options.output/"loop_summary.csv"
    if summary:
        with csvpath.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    reportpath=options.output/"实验报告.md"
    marker="<!-- leaf1009_loop_candidates_v2 -->"
    preserved=reportpath.read_text(encoding="utf-8").split(marker)[0].rstrip() if reportpath.is_file() else "# main1009：LEAF 循环候选实验"
    text=[marker,"", "## 五个循环候选：正式协议与实际记录", "",
        "训练预算为每方法 1M 物理步，种子 0/1/2。正式 checkpoint 为 final；校准、选择、正式评估和确认使用独立场景。",
        "以下仅列实际数据；冻结阈值只使用 ID 校准池。随机深度独立于当前状态，按实际 MAC 计数匹配，无法匹配时明确标识。", "",
        "episode_wall_seconds 记录正常对局的墙钟耗时，包含环境与诊断开销，不代表纯网络推理测速；没有额外性能探测或对局配额。", "",
        "| 方法 | 保留 final checkpoint 文件 | 记录数 |", "|---|---:|---:|"]
    for method in LOOP_METHODS:
        completed=sum((options.output/"had"/method/"train"/f"seed_{s}"/"final.pt").is_file() for s in (0,1,2))
        text.append(f"| {method} | {completed}/3（文件存在，完成资格以训练进度为准） | {sum(r['method']==method for r in rows)} |")
    for name,label in (("calibration.json","ID 校准"),("selection.json","ID 方法选择"),("budgets.json","计算预算匹配")):
        docpath=options.output/name
        if docpath.is_file():
            doc=json.loads(docpath.read_text(encoding="utf-8"));text.extend(["",f"{label}：[{name}]({name})，frozen={doc.get('frozen')}。"])
            if name=="selection.json":text.append(f"选择 `{doc['selected_method']}`，资格状态 `{doc['status']}`；一个方法家族跨三个种子。")
    if comparisons:
        unmatched=sum(not r.get("random_budget_matchable",True) for r in comparisons)
        text.extend(["",f"正式配对预算比较 {len(comparisons)} 个配置×种子单元，其中 actual MAC 差超过 5% 的 {unmatched} 个标为 unmatchable，不宣称匹配预算优势。实际差与奖励差见 summary.json 的 actual_budget_comparisons。"])
    if summary:
        text.extend(["","结果按 method、split、arm、checkpoint、config、seed 汇总；缺额数据保留为部分结果。",
            "完整配置级数据见 [loop_summary.csv](loop_summary.csv)，逐场景计算与回报见 [loop_records.jsonl](loop_records.jsonl)。", "",
            "| Split | 方法 | Arm | Checkpoint | 配置 | 已完成种子 | 配置等权奖励均值 | MAC 均值 |", "|---|---|---|---|---|---:|---:|"])
        bins=defaultdict(list)
        for r in summary:bins[r["split"],r["method"],r["arm"],tuple(r["config"])].append(r)
        for (split,method,arm,config),values in sorted(bins.items()):
            quota={"calibrate":40,"select":100,"eval":300,"adaptive":300,"confirm":300}.get(split)
            complete=[r for r in values if r["episodes"]==quota]
            if not complete:continue
            if len({r['seed'] for r in complete}) != len(complete):
                continue  # Two final checkpoints of one seed must not be pooled.
            provenance = "; ".join(f"s{r['seed']}:{r['checkpoint']}" for r in sorted(complete,key=lambda r:r['seed']))
            text.append(f"| {split} | {method} | {arm} | {provenance} | {config} | {[r['seed'] for r in complete]} | {statistics.mean(r['reward'] for r in complete):.4f} | {statistics.mean(r['macs'] for r in complete):.0f} |")
    figures=options.output/"had"/"figures";images=[]
    def save(fig,name,label):
        figures.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(figures/name,dpi=160);plt.close(fig)
        images.append((label,f"had/figures/{name}"))
    if native["records"]:
        text.extend(["", "### 原生奖励机制与初步外推", "",
            f"已完成 {native['records']}/{native['expected_records']} 个完整 episode；全部完成={native['complete']}。"
            "仅使用三个冻结 final 权重，每配置十个配对物理场景。六配置外推用于机制判断，不能代替正式 24 配置、每配置 300 场的外推比较或 ID 方法选择。",
            "以下差值均以 D 为单位：深度收益为 D浅−D四，反馈损失为 D切断−D正常四轮。配置等权、训练种子分别公开；t95 使用三个种子均值（df=2），另提供按配置分层、同场景联合三个权重的 2000 次聚类 bootstrap95。正差代表四轮或正常反馈更好。",
            "支持要求完整配额、均值至少达到 δ=max(0.01,0.02×参考D)、两个区间下界均大于零、至少两个种子为正且第三个不显著反向。宽区间保留为证据不足。",
            "| 方法 | 原生记录 | R1→R4 | R2→R4 | 反馈奖励证据 | R2非劣且节省MAC |",
            "|---|---:|---|---|---|---|"])
        for i,item in enumerate(native["methods"],1):
            status = item["story_tests"]
            paths = "; ".join(f"{arm}: {result}" for arm,result in status["feedback_paths"].items())
            text.append(f"| M{i} | {item['records']}/{item['expected_records']} | {status['reward_value_R1_to_R4']} | {status['extra_depth_R2_to_R4']} | {paths} | {status['R2_noninferior_and_lower_matrix_MAC']} |")
        text.extend(["", "| 方法 | 配对比较 | 配置覆盖 | 差值均值±seed SD | seed 0/1/2 | seed t95 | 场景聚类95 |",
                     "|---|---|---:|---:|---|---|---|"])
        for i,item in enumerate(native["methods"],1):
            values = {**item["depth_gains"], **item["feedback_losses"]}
            for name,value in values.items():
                if "mean" not in value:
                    continue
                seed_values = "/".join(f"{value['seed_values'][str(s)]:+.4f}" for s in (0,1,2))
                ci = lambda bounds: f"[{bounds[0]:+.4f}, {bounds[1]:+.4f}]"
                text.append(f"| M{i} | {name} | {len(value['configs'])}/6 | {value['mean']:+.4f}±{value['seed_sd']:.4f} | {seed_values} | {ci(value['seed_t95'])} | {ci(value['conditional_scene_bootstrap95'])} |")
        text.extend(["", "M2 的 reverse_kv_clamp 单独检验反向 query→实体通路；不能用两个切断差值相减拆分贡献。M4/M5 hidden reset 仅去掉直接状态承接，保留迭代注意力/路由。M5 competition removed 检验当前竞争归一化，不能自动证明对象分组。",
            "实际 matrix MAC/decision 来自完整在线策略的矩阵调用；尾段长度和存活状态可能随策略改变，MAC 不代表纯推理速度。全局状态反馈敏感性不能替代奖励贡献。自适应按需计算仍需正式 adaptive 对预算匹配 random 与单决策反事实。",
            "完整配置、种子和路径统计保存在 [summary.json](summary.json) 的 native_mechanism_analysis。"])
        plot_native_mechanisms(native,save)
    def complete_family(values,configs):
        cells={(r["seed"],tuple(r["config"])) for r in values}
        expected={(seed,config) for seed in (0,1,2) for config in configs}
        if cells!=expected or len(values)!=len(expected):return False
        return all(len({r["checkpoint"] for r in values if r["seed"]==seed})==1 for seed in (0,1,2))
    recorded=logging.unique_episodes(logging.read_records(options.output/"had","episodes",run="train"))
    training=[r for r in recorded if r.get("phase")=="train_eval" and r["method"] in LOOP_METHODS]
    if training:
        points=defaultdict(list)
        for r in training:points[r["method"],int(r["seed"]),int(r["eval_point"]),r.get("arm"),r.get("checkpoint")].append(r)
        fig,ax=plt.subplots(figsize=(9,5))
        for method in LOOP_METHODS:
            line=defaultdict(list)
            for (m,seed,point,arm,checkpoint),values in points.items():
                if m==method and len(values)==100:line[point].append((statistics.mean(float(r["t_env"]) for r in values),-statistics.mean(float(r["D"]) for r in values)))
            if line:ax.plot([statistics.mean(v[0] for v in line[p]) for p in sorted(line)],
                            [statistics.mean(v[1] for v in line[p]) for p in sorted(line)],label=method)
        ax.set(xlabel="Physical training steps",ylabel="ID validation reward (-D)");ax.legend(fontsize=7)
        save(fig,"loop_training.png","Training/validation curves (recorded complete points)")
    depth=[r for r in summary if r["split"]=="eval" and r["arm"].startswith("fixed") and r["episodes"]==300]
    if depth:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,(group,configs) in zip(axes.flat,GROUPS.items()):
            for method in LOOP_METHODS:
                xs=[];ys=[]
                for d in range(1,5):
                    v=[r for r in depth if r["method"]==method and r["arm"]==f"fixed{d}" and tuple(r["config"]) in configs]
                    if complete_family(v,configs):xs.append(d);ys.append(statistics.mean(r["reward"] for r in v))
                if xs:ax.plot(xs,ys,marker="o",label=method)
            ax.set(title=group,xlabel="Fixed execution depth",ylabel="Reward (-D)")
        axes.flat[0].legend(fontsize=6);save(fig,"loop_depth.png","Frozen fixed-depth curves")
    pareto=[r for r in summary if r["split"] in ("eval","adaptive","confirm") and r["episodes"]==300 and r["macs"]>0]
    if pareto:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,(group,configs) in zip(axes.flat,GROUPS.items()):
            b=defaultdict(list)
            for r in pareto:
                if tuple(r["config"]) in configs:b[r["method"],r["split"],r["arm"]].append(r)
            for (method,split,arm),v in b.items():
                if not complete_family(v,configs):continue
                ax.scatter(statistics.mean(r["macs"] for r in v),statistics.mean(r["reward"] for r in v),label=f"{method}/{split}/{arm}")
            ax.set(title=group,xlabel="Measured MACs / episode",ylabel="Reward (-D)")
        axes.flat[0].legend(fontsize=5);save(fig,"loop_pareto.png","Compute/reward Pareto points")
    cf=[r for r in rows if r["split"]=="counterfactual" and r.get("intervention_tail")]
    if cf:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,proxy in zip(axes,("physical_alive_count","same_nearest_blue_competitors")):
            b=defaultdict(list)
            for r in cf:b[r["method"],config_tuple(r["config"]),r["forced_depth"],r[proxy]].append(r["benefit_vs_r4"])
            for method,config,depth in sorted({(m,c,d) for m,c,d,x in b}):
                xs=sorted(x for m,c,d,x in b if (m,c,d)==(method,config,depth))
                ax.plot(xs,[statistics.mean(b[method,config,depth,x]) for x in xs],marker=".",label=f"{method}/{config}/R{depth}")
            ax.set(xlabel=proxy,ylabel="Damage(forced depth) - damage(R4)");ax.axhline(0,color="grey",lw=.7)
        axes[0].legend(fontsize=4,ncol=2);save(fig,"loop_complexity_counterfactual.png","Config-stratified physical complexity / counterfactual benefit")
        text.extend(["",f"反事实实际干预尾随 {len(cf)} 条；R4 记录复用同场景 factual final D。原生物理状态、对手 RNG、策略 batch/hidden/cache 与全局 RNG 从同一快照恢复。",
            "图中收益是单个 observer 当前深度改变（含动作与提交 hidden）、未来全为四轮的总回报差；它不等于整条自适应策略的收益。复杂度由物理存活数量与共同最近敌机竞争定义，按配置分层。"])
    for label,path in images:text.extend(["",f"![{label}]({path})"])
    text.extend(["","停机的价值由 adaptive 对 fixed4 的真实回报与计算节省、以及对实际预算匹配 random 的优势判断。",
        "counterfactual 单决策收益只提供归因证据；decision stability 和动作一致性不直接证明回报收益。无合格方法时保留 performance best 并标记 loop_unverified。",""])
    options.output.mkdir(parents=True,exist_ok=True)
    reportpath.write_text(preserved+"\n\n"+"\n".join(text),encoding="utf-8")
    print(reportpath)


def progress(options):
    """View shared records; optional host status publication never starts evaluation."""
    import heapq
    import subprocess
    import shutil
    import unicodedata
    from collections import deque, Counter
    from datetime import datetime, timezone, timedelta

    order=("calibrate","select","budget_only","mechanism","eval","adaptive","counterfactual","confirm")
    root=options.output
    bank=defaultdict(lambda:defaultdict(set))
    costs=defaultdict(lambda:[0.,0,0.])
    history=deque(maxlen=max(11,math.ceil(300/max(options.watch_seconds,1))+1))
    offset=0
    bad=0
    key=lambda j:(j["split"],j["method"],int(j["seed"]),tuple(j["config"]))
    fmt=lambda seconds: (f"{seconds/60:.0f}m" if seconds<3600 else f"{seconds/3600:.1f}h")

    def publish_status(active,mains,slots,denied):
        host=socket.gethostname()
        snapshot=dict(host=host,updated_at=time.time(),interval=options.watch_seconds,
            monitor_pid=os.getpid(),masters=[pid for pid,_ in mains],slots=slots,
            denied=denied,tasks=[dict(pid=pid,task=task) for pid,task in active.items()])
        # A separate snapshot per host avoids touching the evaluation runner's
        # large shared state. Unique pending paths also separate two monitors
        # on the same host (e.g. the collector and interactive viewer).
        path=root/f".leaf1009_status_{host}.json"
        pending=path.with_suffix(f".{os.getpid()}.pending")
        pending.write_text(json.dumps(snapshot,ensure_ascii=False),encoding="utf-8")
        os.replace(pending,path)

    def cluster_status(local_active,local_mains,local_slots):
        publish_status(local_active,local_mains,local_slots,denied)
        active={};lines=[];slots=0;masters=0;fresh=0
        for path in sorted(root.glob(".leaf1009_status_*.json")):
            snap=json.loads(path.read_text(encoding="utf-8"))
            host=snap["host"]
            age=max(0,time.time()-snap["updated_at"])
            stale=age>max(60,3*snap.get("interval",10))
            tasks=snap.get("tasks",[]);pids=snap.get("masters",[])
            label="STALE" if stale else "RUN" if tasks else "WAIT" if pids else "STOP"
            lines.append(f"{label:5} tasks={len(tasks):3} main={len(pids)} age={age:.0f}s | {host}")
            if stale:continue
            fresh+=1;masters+=len(pids)
            if pids or tasks:slots+=snap.get("slots") or 0
            for item in tasks:
                task=item["task"]
                active[host,item["pid"]]=(task[0],task[1],int(task[2]),tuple(task[3]))
        return active,slots or None,masters,fresh,lines

    def queues():
        phase=copy.copy(options)
        groups={}
        for stage in ("calibrate","select","mechanism","eval","adaptive","counterfactual"):
            phase.stage=stage
            selected=LOOP_METHODS+TIERS["references"] if stage=="eval" else LOOP_METHODS
            groups[stage]=stage_jobs(phase,selected)
        phase.stage="adaptive"
        groups["budget_only"]=stage_jobs(phase,LOOP_METHODS,split="budget_only",
            arms=["fixed1","fixed2","fixed3","fixed4","adaptive"],
            configs=tuple(c for c in FINAL_CONFIGS if c not in GROUPS["ID"]),start=43000,quota=10)
        selection=root/"selection.json"
        chosen=json.loads(selection.read_text(encoding="utf-8"))["selected_method"] if selection.is_file() else "selected_pending"
        phase.stage="confirm"
        groups["confirm"]=stage_jobs(phase,(chosen,*TIERS["references"]))
        from itertools import zip_longest
        banks=[groups["mechanism"],groups["counterfactual"],
            [j for j in groups["eval"] if j["method"] in LOOP_METHODS],
            [j for j in groups["eval"] if j["method"] in TIERS["references"]],groups["adaptive"]]
        mixed=[j for row in zip_longest(*banks) for j in row if j is not None]
        return groups,[groups["calibrate"],groups["select"],groups["budget_only"],mixed,groups["confirm"]]

    def processes(lookup):
        active={}; mains=[]; denied=0
        if not Path("/proc").is_dir():return active,mains,denied
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():continue
            try:
                args=[a.decode(errors="replace") for a in (proc/"cmdline").read_bytes().split(b"\0") if a]
                if any(a.endswith("leaf1009.py") for a in args) and "run" in args:
                    output=args[args.index("--output")+1] if "--output" in args else str(PROJECT/"outputs/main1009")
                    cwd=Path(os.readlink(proc/"cwd"))
                    if (cwd/output).resolve()==root:
                        mains.append((int(proc.name),args))
                if not any("spawn_main" in a for a in args):continue
                for fd in (proc/"fd").iterdir():
                    try:
                        parts=Path(os.readlink(fd)).relative_to(root).parts
                        if len(parts)!=5 or parts[0]!="had" or parts[2]!="train":continue
                        split,nr,nb,k=Path(parts[4]).stem.rsplit("_",3)
                        task=(split,parts[1],int(parts[3].split("_")[-1]),(int(nr),int(nb),int(k)))
                        if task in lookup:active[int(proc.name)]=task
                    except (OSError,ValueError):pass
            except PermissionError:denied+=1
            except OSError:pass
        return active,mains,denied

    def capacity(mains,active):
        args=mains[0][1] if len(mains)==1 else []
        spec=args[args.index("--devices")+1] if "--devices" in args else options.devices
        per=int(args[args.index("--jobs-per-gpu")+1]) if "--jobs-per-gpu" in args else options.jobs_per_gpu
        if spec!="all":return len(spec.split(","))*per
        visible=os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
        if mains:
            try:
                env=(Path("/proc")/str(mains[0][0])/"environ").read_bytes().split(b"\0")
                visible=next((v.split(b"=",1)[1].decode() for v in env if v.startswith(b"CUDA_VISIBLE_DEVICES=")),visible)
            except OSError:pass
        if visible and visible!="-1":return len(visible.split(","))*per
        try:
            result=subprocess.run(["nvidia-smi","--query-gpu=index","--format=csv,noheader"],
                capture_output=True,text=True,timeout=3)
            n=len(result.stdout.strip().splitlines()) if result.returncode==0 else 0
            return n*per if n else None
        except (OSError,subprocess.TimeoutExpired):return None

    def sample(method,cfg,arm):
        entries=[(c,a,v) for (m,c,a),v in costs.items() if m==method and v[1]]
        exact=[v for c,a,v in entries if c==cfg and a==arm]
        if exact:
            return sum(v[0] for v in exact)/sum(v[1] for v in exact),"measured",None
        same=[v for c,a,v in entries if c==cfg and (a=="fixed4" if arm not in ("fixed1","fixed2","fixed3") else True)]
        if same:
            mean=sum(v[0] for v in same)/sum(v[1] for v in same)
            return mean,"same-config proxy",(.5*mean,2*mean)
        if entries:
            c,a,v=min(entries,key=lambda x:(abs(math.log(sum(cfg)/sum(x[0]))),x[1]!=arm))
            mean=v[0]/v[1]; ratio=sum(cfg)/sum(c)
            # Planning envelope, not a confidence bound: entity-linear to observer*entity^2.
            lo=.5*ratio*mean
            hi=2*max(ratio,(cfg[0]/c[0])*ratio**2)*mean*max(1,100/max(v[2]/v[1],1))
            return math.sqrt(lo*hi),"unseen-config proxy",(lo,hi)
        other=[v for (m,c,a),v in costs.items() if c==cfg and v[1]]
        if other:
            lo=.5*min(v[0]/v[1] for v in other); hi=2*max(v[0]/v[1] for v in other)
            return math.sqrt(lo*hi),"unmeasured-model proxy",(lo,hi)
        any_cost=[(c,v) for (m,c,a),v in costs.items() if v[1]]
        if any_cost:
            c,v=min(any_cost,key=lambda x:abs(math.log(sum(cfg)/sum(x[0]))))
            mean=v[0]/v[1];ratio=sum(cfg)/sum(c)
            lo=.25*ratio*mean;hi=4*max(ratio,(cfg[0]/c[0])*ratio**2)*mean*max(1,100/max(v[2]/v[1],1))
            return math.sqrt(lo*hi),"unmeasured-model/config proxy",(lo,hi)
        return None,"no timing samples",None

    fixed_screen=sys.stdout.isatty() and options.watch_seconds>0 and not options.status_only
    if fixed_screen:
        sys.stdout.write("\033[?1049h\033[?25l");sys.stdout.flush()
    def render(lines):
        size=shutil.get_terminal_size(fallback=(100,24))
        width=max(1,size.columns-1)
        def clip(line):
            result=[];used=0
            for char in line:
                cells=0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ("W","F") else 1
                if used+cells>width:break
                result.append(char);used+=cells
            return "".join(result)
        visible=lines[:max(1,size.lines-1)] if fixed_screen else lines
        frame="\n".join(clip(line) for line in visible)
        sys.stdout.write(("\033[H" if fixed_screen else "")+frame+("\033[J" if fixed_screen else "\n"))
        sys.stdout.flush()
    try:
        while True:
            groups,phase_queues=queues()
            lookup={key(j):j for jobs in groups.values() for j in jobs}
            if options.status_only:
                active,mains,denied=processes(lookup)
                publish_status(active,mains,capacity(mains,active),denied)
                if options.watch_seconds==0:return 0
                time.sleep(options.watch_seconds)
                continue
            path=root/"loop_records.jsonl"
            if path.is_file():
                with path.open("rb") as stream:
                    stream.seek(offset)
                    while True:
                        line=stream.readline()
                        if not line or not line.endswith(b"\n"):break
                        offset=stream.tell()
                        try:row=json.loads(line)
                        except (ValueError,UnicodeDecodeError):bad+=1;continue
                        if row.get("protocol") not in (REVISION,MECHANISM_REVISION):continue
                        task=(row["split"],row["method"],int(row["seed"]),config_tuple(row["config"]))
                        arm="completed_scene" if row["split"]=="counterfactual" else row["arm"]
                        if row["split"]=="counterfactual" and not row.get("scene_complete"):continue
                        scene=int(row["episode_seed"])
                        if scene in bank[task][arm]:continue
                        bank[task][arm].add(scene)
                        seconds=row.get("episode_wall_seconds")
                        if seconds is not None and math.isfinite(seconds) and seconds>0:
                            v=costs[task[1],task[3],row["arm"]]
                            v[0]+=seconds;v[1]+=1;v[2]+=row.get("ep_len",100)
            done={}; totals={}; remaining={}
            for task,j in lookup.items():
                arms=["completed_scene"] if task[0]=="counterfactual" else j["arms"]
                scenes=set(range(j["start"],j["start"]+j["quota"]))
                remaining[task]={a:j["quota"]-len(bank[task][a]&scenes) for a in arms}
                totals[task]=len(arms)*j["quota"]
                done[task]=totals[task]-sum(remaining[task].values())
            now=time.monotonic();history.append((now,dict(done)))
            def rate(tasks):
                tasks=list(tasks);current=sum(done[t] for t in tasks)
                samples=[(t,sum(v.get(k,0) for k in tasks)) for t,v in history]
                base=next(((t,n) for t,n in samples if n>0),samples[0])
                dt=now-base[0];gain=current-base[1]
                return gain/dt if dt>=60 and gain>0 else None
            def recent_eta(tasks):
                tasks=list(tasks);left=sum(totals[t]-done[t] for t in tasks)
                if not left:return "done"
                speed=rate(tasks)
                return fmt(left/speed) if speed else "sampling"
            active,mains,denied=processes(lookup)
            slots=capacity(mains,active)
            host_lines=[];cluster_masters=len(mains);fresh_hosts=1
            if options.cluster:
                active,slots,cluster_masters,fresh_hosts,host_lines=cluster_status(active,mains,slots)
            bounds={}; reasons=Counter();unpriced=0
            for task,j in lookup.items():
                left=remaining[task];low=high=0.
                for arm,n in left.items():
                    if not n:continue
                    if task[1]=="selected_pending":
                        values=[sample(m,task[3],arm) for m in LOOP_METHODS]
                        available=[v for v in values if v[0] is not None]
                        if available:
                            estimate=math.sqrt(min(v[2][0] if v[2] else v[0] for v in available)*max(v[2][1] if v[2] else v[0] for v in available))
                            spread=(min(v[2][0] if v[2] else v[0] for v in available),max(v[2][1] if v[2] else v[0] for v in available))
                            reason="selection pending"
                        else:estimate,reason,spread=None,"no timing samples",None
                    elif task[0]=="counterfactual":
                        speed=rate([task])
                        if speed:
                            estimate,reason,spread=1/speed,"observed scene rate",None
                        else:
                            estimate,reason,spread=sample(task[1],task[3],"fixed4")
                            if estimate is not None:
                                spread=(spread[0] if spread else estimate,13*(spread[1] if spread else estimate))
                                reason="counterfactual 1-13 rollout proxy"
                    else:estimate,reason,spread=sample(task[1],task[3],arm)
                    reasons[reason]+=n
                    if estimate is None:unpriced+=n;continue
                    lo,hi=spread if spread else (estimate,estimate)
                    low+=n*lo;high+=n*hi
                bounds[task]=(low,high)
            def makespan(jobs,index):
                heap=[0.]*slots
                tasks=[key(j) for j in jobs if totals[key(j)]>done[key(j)]]
                tasks.sort(key=lambda t:t not in active.values())
                for task in tasks:
                    start=heapq.heappop(heap);heapq.heappush(heap,start+bounds[task][index])
                return max(heap)
            def dependency_makespan(index):
                pending={}; remaining=Counter()
                def bank(job):
                    return "references" if job["split"]=="eval" and job["method"] in TIERS["references"] else job["split"]
                for jobs in groups.values():
                    for job in jobs:
                        task=key(job)
                        if totals[task]>done[task]:
                            name=bank(job);pending[task]=name;remaining[name]+=1
                dependencies={"select":("calibrate",),"budget_only":("calibrate",),
                    "eval":("calibrate","select"),"counterfactual":("calibrate","select"),
                    "adaptive":("calibrate","select","budget_only"),
                    "confirm":("calibrate","select","budget_only")}
                running=[];now=0.
                while pending or running:
                    eligible=[t for t,name in pending.items()
                        if all(remaining[d]==0 for d in dependencies.get(name,()))]
                    eligible.sort(key=lambda t:(t not in active.values(),
                        {"calibrate":0,"select":1,"budget_only":2}.get(pending[t],3)))
                    for task in eligible[:slots-len(running)]:
                        name=pending.pop(task)
                        heapq.heappush(running,(now+bounds[task][index],task,name))
                    if not running:raise ValueError("Evaluation dependency graph cannot advance")
                    now,task,name=heapq.heappop(running);remaining[name]-=1
                return now
            lower=upper=None
            if slots and not unpriced:
                lower=dependency_makespan(0)
                upper=dependency_makespan(1)
            stamp=datetime.now(timezone(timedelta(hours=8)))
            running_stages=Counter(t[0] for t in active.values())
            finished=sum(done[t]==totals[t] for t in lookup)
            fraction=sum(done[t]/totals[t] for t in lookup)/max(len(lookup),1)
            filled=int(fraction*24)
            lines=["LEAF main1009  |  "+stamp.strftime("%m-%d %H:%M:%S UTC+08"),
                f"{'集群' if options.cluster else '本机'}正在运行: {len(active)} 个任务  |  并发上限: {slots or '未知'}  |  每 {options.watch_seconds:g} 秒刷新",
                f"整体任务进度: [{'#'*filled+'.'*(24-filled)}] {100*fraction:.1f}% (含部分完成)",
                f"完整任务: {finished}/{len(lookup)}  |  未完整完成: {len(lookup)-finished}",
                "当前阶段: "+(" / ".join(f"{s} ({n}任务)" for s,n in running_stages.items()) or "未识别到运行任务")]
            if options.cluster:
                lines.append(f"状态上报: {fresh_hosts} 台在线 | RUN=有任务 WAIT=主控等待 STOP=无主控 STALE=上报过期")
                lines.extend(host_lines)
            priced=sum(reasons.values())
            direct=reasons["measured"]+reasons["observed scene rate"]
            if lower is not None:
                middle=math.sqrt(lower*upper) if upper else 0
                finish=stamp+timedelta(seconds=middle)
                lines.extend([f"整轮剩余(估计): 约 {fmt(middle)}  |  推算范围: {fmt(lower)} - {fmt(upper)}",
                    "预计结束: "+finish.strftime("%m-%d %H:%M 北京时间")])
            else:lines.extend(["整轮剩余: 等待运行时间/设备采样","预计结束: 暂不可估算"])
            lines.extend([f"耗时实测覆盖: {100*direct/max(priced,1):.1f}%  |  "+("粗估，仍含未测规模/模型" if direct<.8*priced else "仍受负载和对局长度影响"),
                "", "Stage           Done/Quota           %   Tasks    Run   Stage ETA"])
            for stage in order:
                tasks=[key(j) for j in groups[stage]]
                n=sum(done[t] for t in tasks);total=sum(totals[t] for t in tasks)
                complete=sum(done[t]==totals[t] for t in tasks)
                estimate=recent_eta(tasks) if running_stages[stage] else ("done" if n==total else "pending")
                lines.append(f"{stage:15} {str(n)+'/'+str(total):17} {100*n/total:5.1f} {str(complete)+'/'+str(len(tasks)):8} {running_stages[stage]:3}   {estimate}")
            alerts=[]
            if not cluster_masters:alerts.append("无主控: ETA仅为预算推算")
            if not options.cluster and len(mains)>1:alerts.append("多个主控: ETA可能不准")
            if denied:alerts.append("进程访问受限")
            if bad:alerts.append(f"异常记录{bad}行")
            lines.extend(["", "任务比例不等于时间比例；反事实按完整场景计。",
                " | ".join(alerts) if alerts else "ETA为推算，不是保证的上下界。",
                "Ctrl+C 只退出查看器，后台评估继续运行。"])
            render(lines)
            if options.watch_seconds==0:return 0
            time.sleep(options.watch_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        if fixed_screen:
            sys.stdout.write("\033[?25h\033[?1049l");sys.stdout.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("plan","progress","train","run","mechanism","calibrate","select","eval","adaptive","counterfactual","confirm","depth","depth-eval","report"))
    parser.add_argument("--output",type=Path,default=PROJECT/"outputs/main1009")
    parser.add_argument("--base-config",type=Path,default=None,help="Optional retained regir_nomem config override; default is a complete frozen literal")
    parser.add_argument("--source-output",type=Path,default=PROJECT/"outputs/main0928")
    parser.add_argument("--baseline-data",type=Path,default=PROJECT.parent/"Papers/v2/data/paper1007.json")
    parser.add_argument("--suite",choices=("loop_candidates","references","selected"),default=None)
    parser.add_argument("--tier",choices=(*TIERS,"all"),default=None,help="Legacy refinement-tier compatibility")
    parser.add_argument("--methods")
    parser.add_argument("--seeds",default="0,1,2")
    parser.add_argument("--steps",type=int,default=1_000_000)
    parser.add_argument("--devices",default="all",help="all visible logical GPUs (default), or a comma-separated logical subset")
    parser.add_argument("--jobs-per-gpu",type=int,default=4,help="Local concurrent jobs per visible GPU (default: 4)")
    parser.add_argument("--max-minutes",type=float,default=None,help="Stop at episode boundaries after this wall-clock budget; rerun to resume")
    parser.add_argument("--watch-seconds",type=float,default=30,help="progress refresh interval; 0 prints one snapshot")
    parser.add_argument("--cluster",action="store_true",help="progress: publish local status and show all reporting hosts")
    parser.add_argument("--status-only",action="store_true",help="progress: publish local process status only, without reading episode records")
    parser.add_argument("--episodes",type=int,help="Legacy confirm/depth quota only; candidate protocol quotas are frozen")
    parser.add_argument("--depths",default=None,help="depth-eval uses 1..6; legacy depth defaults to 1,2,4,6,8")
    parser.add_argument("--dry-run",action="store_true",help="depth-eval: inspect checkpoints and coverage without games or writes")
    parser.add_argument("--section",choices=("nomem-depth",),help="report: update only the NoMem depth supplement in the formal docs report")
    parser.add_argument("--resume",action="store_true")
    options=parser.parse_args();options.output=options.output.resolve();options.source_output=options.source_output.resolve()
    options.suite="legacy" if options.tier is not None and options.suite is None else options.suite or "loop_candidates"
    options.tier=options.tier or "P0"
    if options.output.name!="main1009":parser.error("New outputs must stay in a directory named main1009")
    try:
        options.seeds=tuple(int(v) for v in options.seeds.split(","))
        options.depths=tuple(int(v) for v in (options.depths or ("1,2,3,4,5,6" if options.stage=="depth-eval" else "1,2,4,6,8")).split(","))
        if options.jobs_per_gpu<=0:raise ValueError("--jobs-per-gpu must be positive")
        if options.max_minutes is not None and (not math.isfinite(options.max_minutes) or options.max_minutes<=0):
            raise ValueError("--max-minutes must be a positive finite value")
        if not math.isfinite(options.watch_seconds) or options.watch_seconds<0:raise ValueError("--watch-seconds must be finite and nonnegative")
        if options.dry_run and options.stage!="depth-eval":raise ValueError("--dry-run is only for depth-eval")
        if options.section and options.stage!="report":raise ValueError("--section is only for report")
        if options.stage=="report" and options.section=="nomem-depth":
            nomem_depth_report(options);return 0
        if options.stage=="progress":return progress(options)
        if not options.seeds or len(set(options.seeds))!=len(options.seeds) or set(options.seeds)-{0,1,2}:raise ValueError("Use distinct seeds from 0,1,2")
        if options.stage=="depth":options.suite="legacy"
        if options.stage=="depth-eval":
            if options.methods not in (None,"regir_nomem"):raise ValueError("depth-eval only evaluates original regir_nomem")
            if options.seeds!=(0,1,2) or options.depths!=NOMEM_DEPTHS:
                raise ValueError("depth-eval requires ordered seeds 0,1,2 and depths 1,2,3,4,5,6")
            if options.episodes is not None or options.steps!=1_000_000 or options.max_minutes is not None:
                raise ValueError("depth-eval uses the full frozen 300-scene, 1M-checkpoint protocol without a time limit")
            options.suite="references";options.methods="regir_nomem"
            context=nomem_depth_context(options)
            coverage=nomem_depth_coverage(nomem_depth_rows(options,context))
            print(json.dumps(dict(stage="depth-eval",revision=NOMEM_DEPTH_REVISION,
                method="regir_nomem",seeds=options.seeds,depths=NOMEM_DEPTHS,configs=FINAL_CONFIGS,
                scenes=[9000,9299],environment=context["environment"],
                checkpoints={s:dict(checkpoint=v["checkpoint"],artifact_id=v["artifact_id"],
                    path=context["checkpoint_files"][s]) for s,v in context["checkpoints"].items()},
                coverage=coverage,dry_run=options.dry_run),ensure_ascii=False,indent=2),flush=True)
            if options.dry_run:return 0
            initialize_nomem_depth(options,context)
            options.nomem_depth_context=context
            jobs=stage_jobs(options,("regir_nomem",),split=NOMEM_DEPTH_SPLIT,
                arms=[f"fixed{d}" for d in NOMEM_DEPTHS],configs=FINAL_CONFIGS,start=9000,quota=300)
            return dispatch_shards(options,jobs)
        selected=methods(options)
        if options.stage=="plan":
            registry=read_registry();base,provenance=base_configuration(options)
            payload=dict(suite=options.suite,revision=REVISION,methods=selected,seeds=options.seeds,steps=options.steps,
                devices=options.devices,jobs_per_gpu=options.jobs_per_gpu,base_provenance=provenance,
                configuration={m:registry.get(m,{"source":str(options.source_output)}) for m in selected})
            if options.suite!="legacy":
                absent=set(selected)-set(registry)-set(TIERS["references"])
                if absent:raise ValueError(f"Missing literal candidate registry: {sorted(absent)}")
                payload.update(train_jobs=sum(m in LOOP_METHODS for m in selected)*len(options.seeds),eval_shards=len(selected)*len(options.seeds)*len(FINAL_CONFIGS),
                    calibration_scenes=[42000,42039],selection_scenes=[40000,40099],formal_scenes=[9000,9299],
                    budget_only_scenes=[43000,43009],confirmation_scenes=[110000,110299],counterfactual_scenes=[44000,44009],
                    calibration_episode_arms=9 if any(m in LOOP_METHODS for m in selected) else 0,
                    selection_episode_arms=6 if any(m in LOOP_METHODS for m in selected) else 0,
                    formal_episode_arms={m:1 if m in TIERS["references"] else 4 for m in selected},
                    adaptive_episode_arms=2 if any(m in LOOP_METHODS for m in selected) else 0,
                    counterfactual_max_intervention_tails=9000 if selected==LOOP_METHODS else 1800*sum(m in LOOP_METHODS for m in selected))
                payload.update(native_mechanism_configs=MECHANISM_CONFIGS,native_mechanism_scenes=[44000,44009],
                    native_mechanism_episode_arms={m:4+len(FEEDBACK_ARMS[registry[m]["leaf_loop_core"]])-1
                        for m in selected if m in LOOP_METHODS},native_mechanism_maximum_episodes=4860)
            print(json.dumps(payload,ensure_ascii=False,indent=2));return 0
        if options.stage=="report":report(options);return 0
        if not options.devices:raise ValueError("--devices must be all or a logical GPU list")
        if options.suite=="legacy":
            if options.stage not in ("train","eval","confirm","depth"):raise ValueError("New calibration/adaptive stages require the loop candidate suite")
            if options.stage=="train" and any(m in TIERS["references"] for m in selected):raise ValueError("Reference checkpoints are reused read-only")
            if options.stage in ("confirm","depth") and (options.episodes is None or options.episodes<=0):raise ValueError("Legacy confirm/depth requires explicit positive --episodes")
            initialize(options)
            jobs=([dict(method=m,seed=s,config=None,split=options.stage,arms=[]) for m in selected for s in options.seeds])
            return dispatch_shards(options,jobs)
        if options.steps!=1_000_000:raise ValueError("Candidate comparison uses exactly 1M physical training steps")
        if options.episodes is not None:raise ValueError("Candidate stage quotas are fixed; --episodes is only for legacy tiers")
        if options.stage in ("run","mechanism","calibrate","select","eval","adaptive","counterfactual","confirm") and options.seeds!=(0,1,2):raise ValueError("Mechanism, freeze and formal paired stages require ordered seeds 0,1,2")
        if options.stage=="train" and any(m in TIERS["references"] for m in selected):raise ValueError("Reference checkpoints are reused read-only")
        if options.stage in ("mechanism","calibrate","select","adaptive","counterfactual") and any(m not in LOOP_METHODS for m in selected):raise ValueError("This stage only applies to new loop candidates")
        if options.stage=="mechanism" and selected!=LOOP_METHODS:raise ValueError("Mechanism comparison requires all five candidates")
        if options.stage=="run" and selected!=LOOP_METHODS:raise ValueError("Full evaluation requires the five-candidate suite")
        if options.stage=="run" and options.max_minutes is not None:raise ValueError("Full evaluation runs on the other machine without a local time limit")
        if options.stage in ("calibrate","select") and selected!=LOOP_METHODS:raise ValueError("Frozen calibration/family selection requires all five candidates")
        if options.stage in ("eval","adaptive","counterfactual","confirm") and any(m in LOOP_METHODS for m in selected):frozen_document(options,"selection.json")
        if options.stage=="confirm":
            selection=frozen_document(options,"selection.json");selected=(selection["selected_method"],*TIERS["references"])
        initialize(options)
        if options.stage=="run":return run_evaluation(options)
        if options.stage=="calibrate" and (options.output/"calibration.json").is_file():
            freeze_calibration(options);return 0
        if options.stage=="select" and (options.output/"selection.json").is_file():
            freeze_selection(options);return 0
        if options.stage in ("select","adaptive","counterfactual","confirm"):frozen_document(options,"calibration.json")
        if options.stage in ("adaptive","confirm"):
            result=ensure_budget_jobs(options,selected)
            if result is None:return 0
            if result:return result
        jobs=stage_jobs(options,selected)
        result=dispatch_shards(options,jobs)
        if result:return result
        if options.stage=="calibrate":freeze_calibration(options)
        elif options.stage=="select":freeze_selection(options)
        return 0
    except (ValueError,KeyError,FileNotFoundError) as error:
        parser.error(str(error))


if __name__=="__main__":
    raise SystemExit(main())

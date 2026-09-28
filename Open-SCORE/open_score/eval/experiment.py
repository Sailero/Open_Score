"""Frozen main0928 protocol: Looped-without-count campaign, flat matrix, no gates.

This module supplies data to the train/eval schedulers and the pipeline loop.
It never starts workers or changes another experiment directory implicitly.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid

_CHECKPOINT_CACHE = {}
# Integration smoke test only (never the formal output): tiny budgets and episode counts.
SMOKE = os.environ.get("OPEN_SCORE_SMOKE") == "1"


def episodes(n):
    return n if not SMOKE else max(1, min(n, 2))

PROFILE, SOURCE_PROFILE = "main0928", "main0921"
IMPLEMENTATION_REVISION = f"{PROFILE}_looped_nocount_v1"
SEEDS = (0, 1, 2)
ALL_SEEDS = SEEDS
# Kept as empty aliases so leftover 0923 helpers fail loudly instead of inventing a gate.
CANDIDATES, LADDER, BRANCH, BRANCH_MECHANISM, GATE = {}, (), {}, {}, {}
FOUNDATION = "regir_r0_sg"
TRIAL_SEEDS, TRIAL_ORDER, INTENT_METHODS, IMPORTED_NO_SG = (), (), (), ()
IMPORTED_BASELINES = ("refil", "refil_matched", "b2_qmix_atten", "dcg", "spectra", "alma", "transfqmix")
# Flat P0→P2 training tables. Looped (`regir_sg`) is the main method and has no count.
TRAIN = {
    "had": (
        ("regir_sg", "P0"), ("regir_norefil_sg", "P0"), ("regir_r0_sg", "P0"),
        ("regir_r1_sg", "P0"), ("regir_nomem", "P0"),
        ("regir_last_sg", "P1"), ("regir_fixed4_sg", "P1"), ("regir_kv0_sg", "P1"),
        ("regir_untied4_sg", "P1"), ("regir_count_sg", "P1"),
    ),
    "smacv2": (
        ("regir_sg", "P0"), ("refil", "P0"), ("spectra", "P0"), ("transfqmix", "P0"),
        ("regir_r1_sg", "P0"), ("b2_qmix_atten", "P1"), ("regir_kv0_sg", "P1"),
        ("regir_untied4_sg", "P1"),
    ),
}
COMMON = TRAIN
MECHANISM = {
    "had": dict(
        depth=("regir_sg", "regir_kv0_sg", "regir_fixed4_sg", "regir_untied4_sg"),
        readout=("regir_sg",),
        probes=("regir_sg", "regir_r0_sg", "regir_r1_sg", "regir_kv0_sg", "regir_untied4_sg", "refil"),
        dup="trained_and_imported",
    ),
    "smacv2": dict(depth=("regir_sg",)),
}
COMMON_PROBES = MECHANISM["had"]["probes"]
PRIORITY_ORDER = ("P0", "P1", "P2", "M")
FAILURE = dict(final_validation_D=1.0, midpoint_step=500_000, midpoint_validation_D=3.0)

SMAC_VALIDATION = tuple((n, n, 0) for n in (4, 6, 8, 10))
SMAC_FINAL = tuple((n, n, 0) for n in (5, 10, 12, 15, 20)) + ((10, 11, 0), (10, 12, 0), (10, 15, 0))
SMAC_DEPTH_CONFIGS = ((10, 10, 0), (15, 15, 0), (20, 20, 0))
DEPTH_CONFIGS = ((10, 10, 2), (30, 30, 2), (50, 50, 2))
GATE_DEPTH_CONFIGS = ((10, 10, 2), (50, 50, 2))
DUP_CONFIGS = ((10, 10, 2), (30, 30, 2), (50, 50, 2))
READOUT_CONFIGS = ((50, 50, 2), (30, 30, 12))
PROBE_CONFIGS = ((10, 10, 2), (10, 10, 3), (30, 30, 2), (50, 50, 2), (30, 30, 12))
READOUTS = ("learned", "read1", "read2", "read3", "read4", "uniform")
INTENT_MODES = ("uniform", "shuffle", "oracle")
EVAL_PHASES = ("final", "dup", "depth", "readout")
DIAGNOSTIC_KINDS = ("coverage", "dynamics", "deep_rounds", "global_probe")
HAD_EVAL_KINDS = EVAL_PHASES + DIAGNOSTIC_KINDS + ("timing",)
COST_METHODS = ("refil", "refil_matched", "regir_r0_sg", "regir_r1_sg", "regir_kv0_sg",
                "regir_sg", "regir_untied4_sg", "regir_count_sg", "transfqmix")


def is_profile(output):
    root = Path(output)
    if root.name == PROFILE:
        return True
    path = root / "experiment.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("profile") == PROFILE
    return False


def _dedupe(values):
    return tuple(dict.fromkeys(values))


def methods(env="had"):
    if env == "had":
        return _dedupe(IMPORTED_BASELINES + tuple(m for m, _ in TRAIN["had"]))
    if env == "smacv2":
        return _dedupe(tuple(m for m, _ in TRAIN["smacv2"]))
    raise ValueError(f"Unsupported {PROFILE} environment: {env}")


SMAC_METHODS = methods("smacv2")


def method_seeds(env, method):
    """Every main0928 method uses three seeds."""
    methods(env)
    return SEEDS


def budget(env="had"):
    methods(env)
    if SMOKE:
        return 6_000 if env == "had" else 4_000
    return 1_000_000 if env == "had" else 5_000_000


def validation_point_count(env="had"):
    methods(env)
    # 40k-step spacing: HAD 1M → 50 points; SMAC 5M → 125 points.
    return 125 if env == "smacv2" else 50


def test_depth(method):
    """Depth used by the formal final evaluation of a relation variant (None: not a cycle)."""
    from open_score.algos import MAIN_OVERRIDES
    cfg = MAIN_OVERRIDES.get(method, {})
    if cfg.get("global_branch") != "cycle":
        return None
    if cfg.get("global_read_h0"):
        return 1
    return int(cfg.get("global_eval_depth", 4))


def max_depth(method):
    from open_score.algos import MAIN_OVERRIDES
    return 4 if MAIN_OVERRIDES.get(method, {}).get("rer_update") == "untied4" else 6


def run_directory(output, method, seed, env="had", run="train"):
    root = Path(output)
    if env != "had":
        root /= env
    return root / method / run / f"seed_{int(seed)}"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def initialize(output):
    from .protocol import FINAL_CONFIGS, VALIDATION_CONFIGS
    root = Path(output)
    if SMOKE and root.name == PROFILE:
        raise ValueError("OPEN_SCORE_SMOKE must never run in the formal output directory")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "experiment.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("profile") != PROFILE:
            raise ValueError(f"Existing experiment identity does not match {PROFILE}")
        return saved
    if any((root / f"{stream}.csv").exists() for stream in ("episodes", "learning", "progress")):
        raise ValueError(f"Cannot relabel an existing experiment as {PROFILE}; select an independent output directory")
    matrix = {env: {m: list(method_seeds(env, m)) for m in methods(env)} for env in ("had", "smacv2")}
    data = dict(profile=PROFILE, schema_version=4, source_profile=SOURCE_PROFILE,
                implementation_revision=IMPLEMENTATION_REVISION,
                seeds=list(SEEDS), run="train",
                official_checkpoint="final", episodes_per_config=300,
                main_method="regir_sg", skip_count_inject_default=True,
                train={env: [list(item) for item in rows] for env, rows in TRAIN.items()},
                mechanism={env: {k: (list(v) if isinstance(v, tuple) else v) for k, v in table.items()}
                           for env, table in MECHANISM.items()},
                common_probes=list(COMMON_PROBES),
                method_seeds=matrix,
                imported=dict(source=SOURCE_PROFILE, baselines=list(IMPORTED_BASELINES)),
                failure_criteria=FAILURE,
                environments={
                    "had": dict(methods=list(methods()), budget=budget(),
                                train_N=[4, 6, 8, 10], train_K=[1, 2, 3],
                                validation_configs=VALIDATION_CONFIGS, validation_points=50,
                                validation_episodes_per_config=25, final_configs=FINAL_CONFIGS,
                                metric="D", direction="min"),
                    "smacv2": dict(methods=list(SMAC_METHODS), budget=budget("smacv2"),
                                   validation_configs=SMAC_VALIDATION,
                                   validation_points=validation_point_count("smacv2"),
                                   validation_episodes_per_config=32, final_configs=SMAC_FINAL,
                                   metric="battle_won", direction="max",
                                   upstream_commit="577ab5a2cff2391f8df582da5731ea9cd6adf3c6",
                                   game_version="4.10.0", state_last_action=False)},
                mechanisms=dict(depth_configs=DEPTH_CONFIGS,
                                dup_configs=DUP_CONFIGS, readout_configs=READOUT_CONFIGS, readouts=READOUTS,
                                smac_depth_configs=SMAC_DEPTH_CONFIGS, smac_depths=[1, 2, 6],
                                probe_configs=PROBE_CONFIGS,
                                dup_pursuit=dict(distance=1500.0, angle_degrees=30.0),
                                read1_equivalence_verified=False,
                                timing_methods=COST_METHODS, timing_warmup=50, timing_repeats=200),
                resources=dict(gpu_ids="auto",
                               had_gpu_per_card=3, had_gpu_per_card_max=4,
                               smac_gpu_per_card=1, smac_gpu_per_card_max=2,
                               continuous_fill=True,
                               gpu_adapt={"had": dict(min=1, default=3, max=4,
                                                      reserve_gib=8.0, peak_multiplier=1.25,
                                                      scale_down_free_gib=8.0, warmup_seconds=180,
                                                      fallback_peak_gib=16.0, scale_up_step=1,
                                                      adapt_pause_seconds=90),
                                          "smacv2": dict(min=1, default=1, max=2,
                                                         reserve_gib=8.0, peak_multiplier=1.25,
                                                         scale_down_free_gib=8.0, warmup_seconds=180,
                                                         fallback_peak_gib=16.0, scale_up_step=1,
                                                         adapt_pause_seconds=90)},
                               cluster=dict(heartbeat_seconds=15, claim_ttl_seconds=120),
                               gpu_reserve_gib=8,
                               had_gpu_admission=dict(reserve_gib=8.0, peak_multiplier=1.25),
                               cuda_oom_auto_retries=3, cuda_oom_backoff_seconds=[60, 120, 240],
                               had_eval_max=32, smac_eval_max=16, diagnostics_max=12,
                               had_workers=8, smac_workers=4),
                sources=dict(transfqmix_commit="2ef0a0726f1f186097b4b560509ceb5a801fdafe",
                             spectra_smac_commit="ffababf6187216c9d16b2109ee8ef6fe5fdf1172"),
                imports=[])
    atomic_json(path, data)
    return data


def config_dict(config):
    if isinstance(config, dict):
        return {key: int(config[key]) for key in ("N_R", "N_B", "K")}
    return dict(zip(("N_R", "N_B", "K"), map(int, config)))


def configs(env="had", validation=False):
    if env == "smacv2":
        return SMAC_VALIDATION if validation else SMAC_FINAL
    from .protocol import FINAL_CONFIGS, VALIDATION_CONFIGS
    return VALIDATION_CONFIGS if validation else FINAL_CONFIGS


def validation_thresholds(env):
    count = validation_point_count(env)
    return [math.ceil(budget(env) * i / count) for i in range(1, count + 1)]


def validation_jobs(env, point, t_env):
    count = episodes(25 if env == "had" else 32)
    base = 9500 if env == "had" else 100_000
    return [dict(env=env, config=config_dict(c), episode_seed=base + ci * count + ei,
                 phase="train_eval", eval_point=int(point), t_env=int(t_env),
                 retain_trajectory=False, arm="validation")
            for ci, c in enumerate(configs(env, True)) for ei in range(count)]


def validation_score(env, rows):
    expected = {(tuple(j["config"].values()), j["episode_seed"]) for j in validation_jobs(env, 1, 0)}
    selected = {(tuple(config_dict(r["config"]).values()), int(r["episode_seed"])): r for r in rows}
    if set(selected) != expected:
        return None
    metric = "D" if env == "had" else "battle_won"
    values = [float(r[metric]) for r in selected.values()]
    return sum(values) / len(values) if all(math.isfinite(v) for v in values) else None


def checkpoint_info(path, *, method=None, seed=None, env=None):
    """Read the actual retained last iterate; never fall back to best."""
    from open_score.algos import setup_runtime, canonical_method
    setup_runtime()
    import torch
    path = Path(path)
    if path.name != "final.pt" or not path.is_file():
        raise ValueError(f"Official final.pt missing: {path}")
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, method, seed, env)
    if cache_key in _CHECKPOINT_CACHE:
        return dict(_CHECKPOINT_CACHE[cache_key])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg, progress = saved["config"], saved["progress"]
    actual_env = cfg.get("env", "had")
    actual_method = canonical_method(cfg["method"])
    actual_seed = int(cfg["seed"])
    if actual_method not in methods(actual_env) or actual_seed not in method_seeds(actual_env, actual_method):
        raise ValueError("Checkpoint is outside the frozen method/seed matrix")
    if method is not None and canonical_method(method) != actual_method:
        raise ValueError("Checkpoint method mismatch")
    if seed is not None and int(seed) != actual_seed:
        raise ValueError("Checkpoint training seed mismatch")
    if env is not None and env != actual_env:
        raise ValueError("Checkpoint environment mismatch")
    actual = int(progress["t_env"])
    if int(cfg["t_max"]) != budget(actual_env) or actual < budget(actual_env):
        raise ValueError(f"Incomplete/incompatible final budget: {actual}/{cfg['t_max']}")
    if progress.get("status") not in ("completed", "complete"):
        raise ValueError("Final checkpoint does not contain a completed training state")
    artifact = saved.get("artifact_id")
    if not artifact:
        raise ValueError("Final checkpoint has no immutable artifact identity; import it first")
    if cfg.get("run") != "train":
        raise ValueError("Formal run must be train")
    result = dict(env=actual_env, method=actual_method, seed=actual_seed,
                  t_env=actual, checkpoint_id=artifact, checkpoint=f"final@{actual}",
                  path=str(path), config=cfg, source_profile=cfg.get("profile"))
    _CHECKPOINT_CACHE[cache_key] = result
    return dict(result)


def _specifications(info, kind):
    """(config, cycle_depth, readout, arm, episodes, extra) rows of one evaluation kind."""
    env, method = info["env"], info["method"]
    final, aux = episodes(300), episodes(100)
    if kind == "final":
        return [(c, None, "learned", "final", final, {}) for c in configs(env)]
    if kind == "gate_depth":
        return [(c, 1, "learned", "gate:depth:R1", aux, {}) for c in GATE_DEPTH_CONFIGS]
    if kind == "dup":
        return [(c, None, "learned", "dup", aux, {}) for c in DUP_CONFIGS]
    if kind == "depth" and env == "had":
        return [(c, d, "learned", f"depth:R{d}", final, {})
                for d in range(1, max_depth(method) + 1) for c in DEPTH_CONFIGS]
    if kind == "depth" and env == "smacv2":
        return [(c, d, "learned", f"depth:R{d}", aux, {}) for d in (1, 2, 6) for c in SMAC_DEPTH_CONFIGS]
    if kind == "readout":
        return [(c, 4, r, f"readout:{r}", final, {}) for r in READOUTS for c in READOUT_CONFIGS]
    if kind == "r1deploy":
        return [(c, 1, "learned", "deploy:R1", final, {}) for c in configs(env)]
    if kind == "intent_intervention":
        return [(c, 4, "learned", f"intent:{m}", aux, {"intent_override": m})
                for m in INTENT_MODES for c in DUP_CONFIGS]
    raise ValueError(f"Invalid evaluation kind: {kind}/{env}")


def evaluation_jobs(info, kind="final"):
    env = info["env"]
    base = 9000 if env == "had" else 110_000
    jobs = []
    for config, depth, readout, arm, episodes, extra in _specifications(info, kind):
        for offset in range(episodes):
            jobs.append(dict(env=env, phase=f"{kind}_eval", eval_point=validation_point_count(env),
                             config=config_dict(config), episode_seed=base + offset,
                             t_env=info["t_env"], checkpoint=info["checkpoint"],
                             checkpoint_id=info["checkpoint_id"], arm=arm,
                             cycle_depth=depth, readout=readout,
                             retain_trajectory=kind == "final" and offset < 2, **extra))
    return jobs


def result_identity(row):
    return (row.get("env", "had"), row.get("checkpoint_id"), row.get("arm"),
            tuple(config_dict(row["config"]).values()), int(row["episode_seed"]))


def equivalent_arms(info, arm, *, read1_equivalent=False):
    """Arms whose episodes are identical to `arm` (same policy, depth, scene seed)."""
    depth = test_depth(info["method"])
    groups = []
    if depth is not None:
        groups.append({"final", f"depth:R{depth}", f"gate:depth:R{depth}", "readout:learned"}
                      | ({"deploy:R1"} if depth == 1 else set()))
        r1 = {"depth:R1", "gate:depth:R1", "deploy:R1"}
        if read1_equivalent and depth == 4:
            r1.add("readout:read1")
        groups.append(r1)
    elif info["env"] == "smacv2":
        groups.append({"final"})
    found = {arm}
    for group in groups:
        if arm in group:
            found |= group
    return found


class EpisodeIndex:
    """Incremental (env, checkpoint_id, arm) -> {(N_R, N_B, K, seed)} over episodes.csv."""

    def __init__(self, output):
        self.path = Path(output) / "episodes.csv"
        self.offset, self.header, self.cells = 0, None, {}

    def refresh(self):
        import csv
        if not self.path.exists():
            return self.cells
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset, self.header, self.cells = 0, None, {}
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            raw = stream.read(size - self.offset)
        end = raw.rfind(b"\n")
        if end < 0:
            return self.cells
        text = raw[:end + 1].decode("utf-8")
        self.offset += end + 1
        lines = text.splitlines()
        if self.header is None:
            self.header = next(csv.reader([lines[0]]))
            lines = lines[1:]
        position = {k: self.header.index(k) for k in ("env", "checkpoint_id", "arm", "config", "episode_seed", "phase")}
        for cells in csv.reader(lines):
            if len(cells) != len(self.header):
                continue
            phase = json.loads(cells[position["phase"]])
            if phase in ("train_eval", "anchor"):
                continue
            checkpoint = json.loads(cells[position["checkpoint_id"]])
            if not checkpoint:
                continue
            config = json.loads(cells[position["config"]])
            key = (json.loads(cells[position["env"]]) or "had", checkpoint, json.loads(cells[position["arm"]]))
            self.cells.setdefault(key, set()).add(
                (int(config["N_R"]), int(config["N_B"]), int(config["K"]), int(json.loads(cells[position["episode_seed"]]))))
        return self.cells

    def done(self, info, arm, *, read1_equivalent=False):
        found = set()
        for alias in equivalent_arms(info, arm, read1_equivalent=read1_equivalent):
            found |= self.cells.get((info["env"], info["checkpoint_id"], alias), set())
        return found


def remaining_evaluations(info, kind, rows=None, *, read1_equivalent=False, index=None):
    """Jobs of `kind` whose scene has no identical recorded episode (rows or an EpisodeIndex)."""
    if index is None:
        index = EpisodeIndex.__new__(EpisodeIndex)
        index.cells = {}
        for row in rows or ():
            if row.get("checkpoint_id") != info["checkpoint_id"]:
                continue
            identity = result_identity(row)
            index.cells.setdefault(identity[:3], set()).add(identity[3] + (identity[4],))
    remaining, cache = [], {}
    for job in evaluation_jobs(info, kind):
        arm = job["arm"]
        if arm not in cache:
            cache[arm] = index.done(info, arm, read1_equivalent=read1_equivalent)
        if tuple(job["config"].values()) + (job["episode_seed"],) not in cache[arm]:
            remaining.append(job)
    return remaining


def new_artifact_id():
    return uuid.uuid4().hex


# ----------------------------------------------------------------------------
# Task matrix: segments, priorities and evaluation kinds
# ----------------------------------------------------------------------------

def read_branch(output):
    return None


def train_matrix(env):
    """{(method, seed): {"segments": {segment: priority}, "order": int}} for trained (non-imported) runs."""
    rows = {}
    order = 0
    for method, priority in TRAIN[env]:
        for seed in method_seeds(env, method):
            rows[(method, seed)] = dict(segments={"train": priority}, order=order)
            order += 1
    return rows


def eval_matrix(env, branch=None):
    """{(method, seed, kind): segments} for evaluation tasks derived from finals."""
    rows = {}

    def add(method, seed, kind, segment, priority):
        rows.setdefault((method, seed, kind), {}).setdefault(segment, priority)

    trained = train_matrix(env)
    for (method, seed), entry in trained.items():
        add(method, seed, "final", "train", entry["segments"]["train"])
    if env == "had":
        for method in IMPORTED_BASELINES:
            for seed in SEEDS:
                add(method, seed, "final", "imported", "P0")
                add(method, seed, "dup", "imported", "P1")
        for (method, seed), entry in trained.items():
            add(method, seed, "dup", "train", "P1")
        table = MECHANISM["had"]
        for method in table["depth"]:
            for seed in SEEDS:
                add(method, seed, "depth", "mechanism", "M")
        for method in table["readout"]:
            for seed in SEEDS:
                add(method, seed, "readout", "mechanism", "M")
        for method in table["probes"]:
            for seed in method_seeds(env, method):
                for kind in DIAGNOSTIC_KINDS:
                    add(method, seed, kind, "mechanism", "M")
    else:
        for method in MECHANISM["smacv2"]["depth"]:
            for seed in SEEDS:
                add(method, seed, "depth", "mechanism", "M")
    return rows


def _segment_rank(segments):
    """Lower rank runs first: P0 training, then P1, then mechanism."""
    ranks = []
    for segment, priority in segments.items():
        if segment == "imported":
            ranks.append(0)
        elif priority in PRIORITY_ORDER:
            ranks.append(PRIORITY_ORDER.index(priority))
        else:
            ranks.append(len(PRIORITY_ORDER))
    return min(ranks) if ranks else len(PRIORITY_ORDER)


def cost_depths(method):
    if method == "regir_kv0_sg":
        return (4, 1)
    if str(method).startswith("regir"):
        return (test_depth(method) or 4,)
    return (None,)


def _timing_task(output):
    from open_score.utils.logging import iter_records
    expected, measurable = 0, []
    for method in COST_METHODS:
        for seed in SEEDS:
            path = run_directory(output, method, seed) / "final.pt"
            info = None
            if path.exists():
                try:
                    info = checkpoint_info(path, method=method, seed=seed, env="had")
                except (ValueError, OSError, KeyError):
                    info = None
            for n in (10, 50):
                for depth in cost_depths(method):
                    expected += 1
                    if info:
                        measurable.append((info["checkpoint_id"], n, depth))
    done = {(row.get("checkpoint_id"), int((row.get("config") or {}).get("N_R", 0)),
             row.get("cycle_depth"))
            for row in iter_records(output, "timing", run="train", env="had")
            if str(row.get("arm", "")).startswith("cost")}
    completed = sum(1 for cell in measurable if cell in done)
    if not measurable:
        status, detail = "waiting", "waiting for cost finals"
    elif completed >= expected:
        status, detail = "complete", "cost arms"
    elif completed >= len(measurable):
        status, detail = "waiting", f"{completed}/{expected} waiting remaining finals"
    else:
        status, detail = "pending", f"{completed}/{expected}"
    return dict(id="eval.timing.had.cost.s0", kind="timing", env="had", method="refil", seed=0,
                segments={"common": "P0"}, rank=_segment_rank({"common": "P0"}),
                status=status, completed=completed, total=expected, detail=detail)


def scan(output, env=None, only=None, probe_collector_seed=0, index=None):
    """Task inventory from the frozen matrix and the actual artifact coverage."""
    root = Path(output)
    manifest = initialize(root)
    read1 = bool(manifest["mechanisms"].get("read1_equivalence_verified"))
    index = index or EpisodeIndex(root)
    index.refresh()
    branch = read_branch(root)
    checkpoints, tasks = [], []
    for domain in ((env,) if env else ("had", "smacv2")):
        trained = train_matrix(domain)
        infos = {}
        seeds_by_method = {}
        for method in methods(domain):
            for seed in method_seeds(domain, method):
                seeds_by_method.setdefault(method, []).append(seed)
        for method, seeds in seeds_by_method.items():
            for seed in seeds:
                directory = run_directory(root, method, seed, domain)
                info, reason = None, "final checkpoint not yet available"
                if (directory / "final.pt").exists():
                    try:
                        info = checkpoint_info(directory / "final.pt", method=method, seed=seed, env=domain)
                    except (ValueError, KeyError, OSError) as error:
                        reason = str(error)
                infos[(method, seed)] = info
                if info:
                    checkpoints.append({k: v for k, v in info.items() if k != "config"})
                imported = domain == "had" and method in IMPORTED_BASELINES
                entry = trained.get((method, seed))
                if imported:
                    continue
                segments = entry["segments"]
                tasks.append(dict(id=f"train.{domain}.{method}.s{seed}", kind="train", env=domain,
                                  method=method, seed=seed, segments=segments,
                                  rank=_segment_rank(segments), order=entry["order"] if entry else -1,
                                  status="complete" if info else "pending",
                                  completed=info["t_env"] if info else 0, total=budget(domain),
                                  detail="qualified final" if info else reason))
        for (method, seed, kind), segments in eval_matrix(domain, branch).items():
            if only is not None and kind not in only:
                continue
            info = infos.get((method, seed))
            identity = f"eval.{kind}.{domain}.{method}.s{seed}"
            base = dict(id=identity, kind=kind, env=domain, method=method, seed=seed,
                        segments=segments, rank=_segment_rank(segments))
            if kind in DIAGNOSTIC_KINDS:
                if not info:
                    failed = (run_directory(root, method, seed, domain) / "task_failure.json").exists()
                    tasks.append(dict(base, status="skipped" if failed else "waiting",
                                      completed=0, total=1,
                                      detail="train failed" if failed else "waiting for final"))
                    continue
                from .probes0923 import probe_complete
                done = probe_complete(root, kind, method, seed, info["checkpoint_id"])
                tasks.append(dict(base, status="complete" if done else "pending",
                                  completed=int(done), total=1, detail="frozen final",
                                  checkpoint=info["path"], checkpoint_id=info["checkpoint_id"]))
                continue
            if not info:
                probe = dict(env=domain, method=method, t_env=0, checkpoint="final@0", checkpoint_id="-")
                tasks.append(dict(base, status="waiting", completed=0,
                                  total=len(evaluation_jobs(probe, kind)), detail="waiting for final"))
                continue
            jobs = evaluation_jobs(info, kind)
            pending = remaining_evaluations(info, kind, read1_equivalent=read1, index=index)
            final_pending = kind != "final" and bool(remaining_evaluations(
                info, "final", read1_equivalent=read1, index=index)) and test_depth(method) is not None
            status = "complete" if not pending else ("waiting" if final_pending else "pending")
            tasks.append(dict(base, status=status, completed=len(jobs) - len(pending), total=len(jobs),
                              detail="reuse final arms first" if status == "waiting" else "frozen final",
                              checkpoint=info["path"], checkpoint_id=info["checkpoint_id"]))
        if domain == "had" and (only is None or "timing" in only):
            tasks.append(_timing_task(root))
    value = dict(profile=PROFILE, branch=branch, checkpoints=checkpoints, tasks=tasks,
                 scanned_at=time.time())
    if env is None:
        atomic_json(root / "inventory.json", value)
    return value


# ----------------------------------------------------------------------------
# Import packed main0921 baselines (or copy from a live source tree)
# ----------------------------------------------------------------------------

def imported_dir(output=None):
    if output is None:
        return Path(__file__).resolve().parents[2] / "outputs" / PROFILE / "imported"
    return Path(output) / "imported"


def unpack_imported(output):
    """Expand git-packed `imported/` weights and CSV slices into the live output."""
    import gzip
    import shutil
    from open_score.utils.logging import append_records
    root = Path(output)
    pack = imported_dir(root)
    if not pack.exists():
        pack = Path(__file__).resolve().parents[2] / "outputs" / PROFILE / "imported"
    manifest = initialize(root)
    if manifest.get("imports"):
        return manifest["imports"]
    pack_manifest = read_json(pack / "manifest.json", {}) or {}
    infos, imported = {}, []
    for method in IMPORTED_BASELINES:
        for seed in SEEDS:
            destination = run_directory(root, method, seed)
            destination.mkdir(parents=True, exist_ok=True)
            origin = pack / "weights" / method / f"seed_{int(seed)}" / "final.pt"
            if not origin.exists():
                raise FileNotFoundError(f"Packed baseline missing: {origin}")
            shutil.copy2(origin, destination / "final.pt")
            for extra in ("config.json", "best.pt"):
                extra_src = origin.parent / extra
                if extra_src.exists():
                    shutil.copy2(extra_src, destination / extra)
            info = checkpoint_info(destination / "final.pt", method=method, seed=seed, env="had")
            infos[(method, seed)] = info
            imported.append(dict(method=method, seed=seed, t_env=info["t_env"],
                                 checkpoint_id=info["checkpoint_id"], source_version=SOURCE_PROFILE,
                                 packed=True))
    counts = {}
    for stream in ("episodes", "learning", "progress"):
        existing = root / f"{stream}.csv"
        if existing.exists() and existing.stat().st_size:
            counts[stream] = "already-present"
            continue
        parts = sorted((pack / "slices").glob(f"{stream}*.csv.gz")) if (pack / "slices").exists() else []
        if not parts:
            continue
        rows, count = [], 0
        for part in parts:
            with gzip.open(part, "rt", encoding="utf-8") as handle:
                import csv
                reader = csv.DictReader(handle)
                for raw in reader:
                    row = {key: json.loads(value) if value else None for key, value in raw.items()}
                    row["version"] = PROFILE
                    row["env"] = row.get("env") or "had"
                    row["source_version"] = SOURCE_PROFILE
                    rows.append(row)
                    if len(rows) >= 2000:
                        append_records(root, stream, rows)
                        count += len(rows)
                        rows.clear()
        if rows:
            append_records(root, stream, rows)
            count += len(rows)
        counts[stream] = count
    manifest = initialize(root)
    manifest["imports"] = imported
    manifest["import_record_counts"] = counts
    manifest["import_pack"] = dict(pack_manifest, unpacked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    atomic_json(root / "experiment.json", manifest)
    inventory = scan(root, env="had")
    missing = [t["id"] for t in inventory["tasks"] if t["kind"] == "final"
               and t["method"] in IMPORTED_BASELINES and t["status"] != "complete"]
    if missing:
        raise ValueError(f"Imported baseline finals are incomplete after unpack: {missing}")
    scan(root)
    return imported


def import_main0921(output, source=None):
    """Install the 7 HAD baselines: unpack the git pack, else copy from a live main0921 tree."""
    from .cluster import nfs_lock
    root = Path(output).resolve()
    with nfs_lock(root / "cluster" / "import.lock", timeout=1800):
        return _import_main0921_locked(root, source)


def _import_main0921_locked(root, source=None):
    manifest = initialize(root)
    if manifest.get("imports"):
        return manifest["imports"]
    pack = imported_dir(root)
    if not (pack / "manifest.json").exists():
        pack = Path(__file__).resolve().parents[2] / "outputs" / PROFILE / "imported"
    if (pack / "manifest.json").exists() and (pack / "weights").exists():
        return unpack_imported(root)
    if source is None:
        source = root.parent / SOURCE_PROFILE
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(
            f"No packed imported/ under {pack} and no live {SOURCE_PROFILE} at {source}")
    return _copy_live_baselines(root, source)


def _copy_live_baselines(root, source):
    import shutil
    from open_score.utils.logging import iter_records, append_records
    if root.resolve() == source.resolve():
        raise ValueError("Import destination must be independent of source")
    if any((root / f"{name}.csv").exists() for name in ("episodes", "learning", "progress", "trajectories")):
        raise ValueError("Incomplete import exists; remove the partial CSV files of the new output before restarting")
    infos, imported = {}, []
    for method in IMPORTED_BASELINES:
        for seed in SEEDS:
            origin, destination = run_directory(source, method, seed), run_directory(root, method, seed)
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("final.pt", "best.pt", "config.json"):
                if (origin / filename).exists():
                    shutil.copy2(origin / filename, destination / filename)
            info = checkpoint_info(destination / "final.pt", method=method, seed=seed, env="had")
            infos[(method, seed)] = info
            imported.append(dict(method=method, seed=seed, t_env=info["t_env"],
                                 checkpoint_id=info["checkpoint_id"], source_version=SOURCE_PROFILE,
                                 checkpoint=str((origin / "final.pt").relative_to(source))))
    from .protocol import FINAL_CONFIGS
    final_configs = set(FINAL_CONFIGS)
    counts = {}
    for stream in ("episodes",):
        rows, count = [], 0
        for row in iter_records(source, stream, run="train", env="had"):
            key = (row.get("method"), row.get("seed"))
            anchor = stream == "episodes" and row.get("phase") == "anchor"
            baseline = key in infos
            if not (baseline or anchor):
                continue
            item = dict(row, version=PROFILE, env="had", source_version=SOURCE_PROFILE)
            phase = row.get("phase")
            config = tuple(config_dict(item["config"]).values())
            if anchor:
                if config not in final_configs or not 9000 <= int(item["episode_seed"]) < 9300:
                    continue
                item.update(arm="anchor", checkpoint_id=None)
            elif phase == "final_eval" and item.get("arm") == "final":
                if item.get("checkpoint_id") != infos[key]["checkpoint_id"]:
                    continue
            else:
                continue
            rows.append(item)
            if len(rows) >= 2000:
                append_records(root, stream, rows); count += len(rows); rows.clear()
        if rows:
            append_records(root, stream, rows); count += len(rows)
        counts[stream] = count
    manifest = initialize(root)
    manifest["imports"] = imported
    manifest["import_record_counts"] = counts
    manifest["reference_speed"] = _reference_speeds(source)
    atomic_json(root / "experiment.json", manifest)
    inventory = scan(root, env="had")
    missing = [t["id"] for t in inventory["tasks"] if t["kind"] == "final"
               and t["method"] in IMPORTED_BASELINES and t["status"] != "complete"]
    if missing:
        raise ValueError(f"Imported baseline finals are incomplete after import: {missing}")
    scan(root)
    return imported


def pack_imported(source, dest=None, *, max_part_bytes=9_000_000):
    """Build the git-sized pack: 7×3 final.pt + gzipped final-eval/anchor slices."""
    import csv
    import gzip
    import shutil
    from open_score.utils.logging import iter_records, schema_for
    source = Path(source)
    dest = Path(dest) if dest is not None else imported_dir()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "weights").mkdir(exist_ok=True)
    (dest / "slices").mkdir(exist_ok=True)
    for stale in (dest / "slices").glob("*.csv.gz"):
        stale.unlink()
    packed = []
    infos = {}
    for method in IMPORTED_BASELINES:
        for seed in SEEDS:
            origin = run_directory(source, method, seed) / "final.pt"
            info = checkpoint_info(origin, method=method, seed=seed, env="had")
            infos[(method, seed)] = info
            target = dest / "weights" / method / f"seed_{int(seed)}"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target / "final.pt")
            config = run_directory(source, method, seed) / "config.json"
            if config.exists():
                shutil.copy2(config, target / "config.json")
            packed.append(dict(method=method, seed=seed, bytes=(target / "final.pt").stat().st_size,
                               checkpoint_id=info["checkpoint_id"]))
    from .protocol import FINAL_CONFIGS
    final_configs = set(FINAL_CONFIGS)
    wanted_ids = {info["checkpoint_id"] for info in infos.values()}

    def keep_episode(row):
        if row.get("phase") == "anchor":
            config = tuple(config_dict(row["config"]).values())
            return config in final_configs and 9000 <= int(row["episode_seed"]) < 9300
        return (row.get("phase") == "final_eval" and row.get("arm") == "final"
                and row.get("checkpoint_id") in wanted_ids)

    columns = schema_for(dest.parent if dest.parent.name == PROFILE else dest.parent, "episodes")
    part, handle, writer, nbytes, index = None, None, None, 0, 0

    def close_part():
        nonlocal handle
        if handle is not None:
            handle.close()
            handle = None

    def new_part():
        nonlocal part, handle, writer, nbytes, index
        close_part()
        part = dest / "slices" / f"episodes.{index:02d}.csv.gz"
        handle = gzip.open(part, "wt", encoding="utf-8", newline="")
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        nbytes, index = 0, index + 1

    new_part()
    n_rows = 0
    for row in iter_records(source, "episodes", run="train", env="had"):
        if not keep_episode(row):
            continue
        item = dict(row, version=PROFILE, env="had", source_version=SOURCE_PROFILE)
        if item.get("phase") == "anchor":
            item.update(arm="anchor", checkpoint_id=None)
        encoded = [json.dumps(item.get(key), ensure_ascii=False, separators=(",", ":"),
                              allow_nan=False) for key in columns]
        line = ",".join(encoded) + "\n"
        if nbytes + len(line.encode("utf-8")) > max_part_bytes and nbytes:
            new_part()
        writer.writerow(encoded)
        nbytes += len(line.encode("utf-8"))
        n_rows += 1
    close_part()
    payload = dict(profile=PROFILE, source_profile=SOURCE_PROFILE, baselines=list(IMPORTED_BASELINES),
                   seeds=list(SEEDS), weights=packed, episode_rows=n_rows,
                   packed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    atomic_json(dest / "manifest.json", payload)
    return payload


def _reference_speeds(source):
    """Mean main0921 training speed (steps/s over training seconds) by method."""
    speeds = {}
    for method in ("regir_kv0", "regir_r1", "regir", "regir_untied4", "regir_fixed4", "refil"):
        values = []
        for seed in SEEDS:
            row = read_json(run_directory(source, method, seed) / "resource.json", {})
            value = row.get("training_steps_per_second")
            if value:
                values.append(float(value))
        if values:
            speeds[method] = sum(values) / len(values)
    return speeds


def reference_speed(manifest, method):
    """main0921 speed of the same architecture; relation variants fall back to ARR (~20 steps/s)."""
    speeds = manifest.get("reference_speed", {})
    for key in (method, method.replace("_sg", ""), method.replace("_intent", "").replace("_sg", "")):
        if key in speeds:
            return speeds[key]
    return speeds.get("regir_kv0") if method.startswith("regir") else None


# ----------------------------------------------------------------------------
# Integration audit
# ----------------------------------------------------------------------------

ACCEPTANCE_KEY = "main0928_methods"
SMAC_ACCEPTANCE_KEY = "smac_main0928"


def new_had_methods():
    return tuple(m for m in methods("had") if m not in IMPORTED_BASELINES)


def validate_integration(output, env="had"):
    """Audit the frozen protocol and recorded numerical/native acceptance in place."""
    from open_score.algos import load_config
    from open_score.utils.logging import schema_for
    import csv
    root = Path(output)
    manifest = initialize(root)
    assert tuple(manifest["seeds"]) == SEEDS
    assert len(configs("had")) == 24 and len(configs("smacv2")) == 8
    assert len(validation_jobs("had", 1, 0)) == 100
    assert len(validation_jobs("smacv2", 1, 0)) == 128
    assert validation_thresholds("had")[-1] == 1_000_000
    assert validation_thresholds("smacv2")[-1] == 5_000_000
    assert validation_point_count("had") == 50
    assert validation_point_count("smacv2") == 125
    assert validation_thresholds("smacv2")[49] == 2_000_000
    assert manifest.get("main_method") == "regir_sg"
    from open_score.algos import MAIN_OVERRIDES
    for method, _ in TRAIN[env]:
        if not str(method).startswith("regir"):
            continue
        skip = bool(MAIN_OVERRIDES[method].get("skip_count_inject", True))
        if method == "regir_count_sg":
            assert skip is False, f"{method} is the sole +count variant"
        else:
            assert skip is True, f"{method} must skip count inject"
    for method in methods(env):
        args = load_config(method, dict(profile=PROFILE, output=str(root), env=env,
            t_max=budget(env), run="train", seed=0, use_cuda=False,
            batch_size_run=8 if env == "had" else 4))
        assert args.device == "cpu" and args.run == "train"
        if env == "smacv2":
            assert not any((args.entity_last_action, args.obs_last_action, args.obs_agent_id,
                            args.env_args["obs_last_action"], args.env_args["state_last_action"]))
        if method.startswith("regir") and method not in IMPORTED_BASELINES:
            detached = bool(getattr(args, "global_query_detach_memory", False))
            nomem = bool(getattr(args, "global_query_no_memory", False))
            assert detached or nomem, f"{method} must detach the readout memory or zero it"
            if method == "regir_count_sg":
                assert not bool(getattr(args, "skip_count_inject", True))
            else:
                assert bool(getattr(args, "skip_count_inject", False)), f"{method} must skip count"
    for stream in ("episodes", "learning", "progress", "timing", "trajectories"):
        path = root / f"{stream}.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                assert next(csv.reader(handle)) == list(schema_for(root, stream)), f"{stream} header differs"
    inventory = scan(root)
    imported_train = [t for t in inventory["tasks"]
                      if t["kind"] == "train" and "imported" in t.get("segments", {})]
    assert not imported_train, "imported baselines must not appear as train tasks"
    print(f"{PROFILE} {env}: protocol passed; {len(inventory['checkpoints'])} qualified finals")
    return inventory


def accepted_had_methods(output):
    return set(methods("had"))


def require_had_method_acceptance(output, acceptance=None, methods_required=None):
    return accepted_had_methods(output)


def require_smac_method_acceptance(output, methods_required=None):
    return set(methods("smacv2"))

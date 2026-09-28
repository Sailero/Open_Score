"""main0923 formal report: identity-checked IQM, stratified bootstrap, G7 sections.

Reads retained CSVs and inventory only. Never writes inventory, never loads a
policy into the environment, and never interpolates an incomplete cell.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import csv
import json
import math
import os
import statistics

import numpy as np

from . import experiment as X
from .report0921 import HAD_GROUPS, T95, _cfg, _escape, _finite, _label, _num, _table
from ..utils.logging import iter_records

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260924
FORMAL_GROUPS = ("ID", "N-OOD", "K-OOD", "联合 OOD")
OOD_GROUPS = ("N-OOD", "K-OOD", "联合 OOD")
SMAC_KEY = ((10, 10, 0), (15, 15, 0), (20, 20, 0), (10, 15, 0))
SMAC_LEARNING_CFG = (8, 8, 0)
T_CRIT = {2: 12.7062047361747, 3: T95, 4: 3.182446305284263, 5: 2.7764451051977987}
# Three-seed IQM equals the 0921 mean; these are the published 3-decimal cells.
REF_0921 = {
    "refil": {"ID": 0.715, "N-OOD": 2.209, "K-OOD": 1.796, "联合 OOD": 3.308},
    "refil_matched": {"ID": 0.914, "N-OOD": 2.269, "K-OOD": 2.493, "联合 OOD": 4.170},
    "b2_qmix_atten": {"ID": 2.318, "N-OOD": 5.139, "K-OOD": 5.009, "联合 OOD": 7.415},
    "dcg": {"ID": 2.028, "N-OOD": 5.021, "K-OOD": 5.839, "联合 OOD": 8.092},
    "spectra": {"ID": 2.662, "N-OOD": 6.455, "K-OOD": 6.933, "联合 OOD": 11.184},
    "alma": {"ID": 1.326, "N-OOD": 3.605, "K-OOD": 3.022, "联合 OOD": 5.375},
    "transfqmix": {"ID": 2.157, "N-OOD": 6.226, "K-OOD": 6.128, "联合 OOD": 10.597},
    "rule_nv1": {"ID": 1.462, "N-OOD": 1.545, "K-OOD": 3.212, "联合 OOD": 3.060},
}
PROBE_FILES = {
    "coverage": "coverage.csv",
    "dynamics": "dynamics.csv",
    "deep_rounds": "deep_rounds.csv",
    "global_probe": "global_results.csv",
    "readout_attention": "readout_attention.csv",
    "intent_accuracy": "intent.csv",
}
PAPER_NAMES = {
    "regir_r1_sg": "1-round", "regir_kv0_sg": "ARR", "regir_kv0_intent_sg": "IAR",
    "regir_r0_sg": "R0", "regir_sg": "Looped", "regir_untied4_sg": "Untied4",
    "regir_r1_nomem": "1-round-NoMem", "regir_r1_norefil_sg": "1-round-NoREFIL",
    "regir_r1_nocount_sg": "1-round-NoCount", "regir_kv0_nomem": "ARR-NoMem",
    "regir_kv0_norefil_sg": "ARR-NoREFIL", "regir_kv0_fixed4_sg": "ARR-Fixed4",
    "regir_fixed4_sg": "Tied-Fixed4", "regir_kv0_nocount_sg": "ARR-NoCount",
    "regir_kv0_intent_noaux_sg": "IAR-NoAux", "regir_kv0_intent_nomem": "IAR-NoMem",
    "regir_intent_sg": "IAR-Looped", "regir_kv0_intent_norefil_sg": "IAR-NoREFIL",
    "refil": "REFIL", "refil_matched": "REFIL-matched", "b2_qmix_atten": "QMIX-Atten",
    "dcg": "DCG", "spectra": "SPECTra", "alma": "ALMA", "transfqmix": "TransfQMix",
    "rule_nv1": "Rule", "random": "Random",
    "regir": "Full (no-SG)", "regir_r1": "Single (no-SG)", "regir_kv0": "KV0 (no-SG)",
    "regir_fixed4": "Fixed4 (no-SG)", "regir_untied4": "Untied4 (no-SG)",
    "regir_last": "Last (no-SG)", "regir_nocount": "No-count (no-SG)",
    "regir_norefil": "No-REFIL (no-SG)",
}
STYLES = {
    "regir_r1_sg": ("#D55E00", "s", "--"),
    "regir_kv0_sg": ("#004C78", "o", "-"),
    "regir_kv0_intent_sg": ("#7B61A8", "D", "-"),
    "regir_r0_sg": ("#56B4E9", "P", ":"),
    "regir_sg": ("#8C564B", "v", (0, (5, 2))),
    "regir_untied4_sg": ("#CC79A7", "^", (0, (3, 1, 1, 1))),
    "refil": ("#009E73", "^", "-."),
    "refil_matched": ("#009E73", "P", (0, (3, 1, 1, 1))),
    "b2_qmix_atten": ("#E69F00", "v", (0, (5, 2))),
    "dcg": ("#882255", "X", "--"),
    "spectra": ("#44AA99", ">", "-."),
    "alma": ("#CC79A7", "D", ":"),
    "transfqmix": ("#332288", "P", (0, (1, 1))),
    "rule_nv1": ("#666666", "x", (0, (2, 2))),
}


def paper_name(method, branch=None):
    if method == "regir_r1_sg" and branch == "C":
        return "RCR"
    return PAPER_NAMES.get(method, method)


def iqm(values):
    """Agarwal et al. 2021 IQM; same trim as branch_gate.iqm."""
    values = np.sort(np.asarray(list(values), dtype=float), axis=-1)
    if values.size == 0:
        return None
    n = values.shape[-1]
    k = int(math.floor(0.25 * n))
    return float(values[..., k:n - k].mean(-1))


def _iqm_axis(values):
    values = np.sort(np.asarray(values, dtype=float), axis=-1)
    n = values.shape[-1]
    k = int(math.floor(0.25 * n))
    return values[..., k:n - k].mean(-1)


def bootstrap_iqm(matrix, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """95% percentile interval of IQM; seeds and configs resampled independently."""
    matrix = np.asarray(matrix, dtype=float)
    n_seeds, n_cfgs = matrix.shape
    rng = np.random.default_rng(seed)
    seed_idx = rng.integers(0, n_seeds, size=(reps, n_seeds))
    cfg_idx = rng.integers(0, n_cfgs, size=(reps, n_cfgs))
    sampled = matrix[seed_idx]
    taken = np.take_along_axis(sampled, np.broadcast_to(cfg_idx[:, None, :], sampled.shape), 2)
    scores = _iqm_axis(taken.mean(axis=2))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def _quota(env, arm):
    if env == "smacv2" and isinstance(arm, str) and arm.startswith("depth:R"):
        return 100
    if arm in ("final", "deploy:R1") or (isinstance(arm, str) and (
            arm.startswith("depth:R") or arm.startswith("readout:"))):
        return 300
    if arm == "dup" or (isinstance(arm, str) and (arm.startswith("gate:") or arm.startswith("intent:"))):
        return 100
    return 300


def _json(path, default):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, ValueError):
        return default


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(text, encoding="utf-8")
    pending.replace(path)
    return path


def _t_interval(values):
    values = [float(v) for v in values if _finite(v)]
    n = len(values)
    if n < 2 or n not in T_CRIT:
        return None
    mean = statistics.mean(values)
    return mean, T_CRIT[n] * statistics.stdev(values) / math.sqrt(n), tuple(values)


def _metric(row, env):
    value = row.get("D" if env == "had" else "battle_won")
    if not _finite(value):
        return None
    value = float(value)
    if env == "smacv2" and value not in (0.0, 1.0):
        return None
    return value


def _episode_return(row):
    """Official env episode return (HAD: -D; SMACv2: shaped SC2 reward)."""
    value = row.get("return")
    if value is None:
        value = row.get("episode_return")
    return float(value) if _finite(value) else None


def _smac_scenario(cfg):
    """SMACv2 README name: protoss_{n_units}_vs_{n_enemies} on 10gen_protoss."""
    allies, enemies, _ = cfg
    return f"protoss_{int(allies)}_vs_{int(enemies)}"


def _extra(row, key):
    value = row.get(key)
    return float(value) if _finite(value) else None


def _collision_rate(row, cfg):
    """Fraction of red agents that died by friendly collision (the paper's 相撞率)."""
    deaths = row.get("red_deaths_by_cause")
    if not isinstance(deaths, dict) or not cfg or int(cfg[0]) <= 0:
        return None
    value = deaths.get("friendly_collision")
    return float(value) / float(cfg[0]) if _finite(value) else None


def _pct(value):
    return f"{100 * float(value):.1f}%" if _finite(value) else "—"


def _fmt_iqm(result, *, percent=False):
    if not result:
        return "—"
    if percent:
        return f"{100 * result['iqm']:.1f}% [{100 * result['lo']:.1f}%, {100 * result['hi']:.1f}%]"
    return f"{_num(result['iqm'])} [{_num(result['lo'])}, {_num(result['hi'])}]"


class Results:
    """Identity-checked episode store. Incomplete cells return None, never a partial mean."""

    def __init__(self, output, inventory, metadata):
        self.output = Path(output)
        self.inventory = inventory or {}
        self.metadata = metadata or {}
        self.branch = X.read_branch(self.output)
        self.read1 = bool(self.metadata.get("mechanisms", {}).get("read1_equivalence_verified"))
        self.checkpoints = {}
        self._failed_qualify = set()
        for info in self.inventory.get("checkpoints", []):
            env, method, seed = info.get("env", "had"), info.get("method"), int(info.get("seed", -1))
            if (env in ("had", "smacv2") and method in X.methods(env)
                    and seed in X.method_seeds(env, method) and info.get("checkpoint_id")
                    and int(info.get("t_env", 0)) >= X.budget(env)):
                self.checkpoints[(env, method, seed)] = info
        self.episodes = defaultdict(dict)
        self.anchors = defaultdict(dict)
        self.legacy = defaultdict(dict)
        self.validation = defaultdict(dict)
        self.validation_return = defaultdict(dict)
        self.validation_steps = {}
        self.intent = {}
        self.conflicts = set()
        self.excluded = Counter()
        self.sources = Counter()
        self.field_missing = Counter()
        self.progress = {}
        self.spikes, self.grad_peaks = {}, {}
        self.intent_loss, self.intent_acc = {}, {}
        self.costs = []
        self.probes = {kind: [] for kind in PROBE_FILES}
        self.probe_complete = {}
        self.measurement_gpu = self.metadata.get("resources", {}).get("measurement_gpu")
        self._expected_validation = {
            env: {(tuple(job["config"].values()), job["episode_seed"])
                  for job in X.validation_jobs(env, 1, 0)}
            for env in ("had", "smacv2")}
        self._read_episodes()
        self._read_learning()
        self._read_progress()
        self._read_timing()
        self._read_probes()
        self.decision = _json(self.output / "decision" / "decision.json", {})
        self.branch_record = _json(self.output / "decision" / "branch.json", {})

    def _info(self, env, method, seed):
        key = (env, method, seed)
        if key in self.checkpoints or key in self._failed_qualify:
            return self.checkpoints.get(key)
        if method not in X.methods(env) or seed not in X.method_seeds(env, method):
            self._failed_qualify.add(key)
            return None
        path = X.run_directory(self.output, method, seed, env) / "final.pt"
        if not path.exists():
            self._failed_qualify.add(key)
            return None
        try:
            info = X.checkpoint_info(path, method=method, seed=seed, env=env)
        except (ValueError, KeyError, OSError):
            self._failed_qualify.add(key)
            return None
        self.checkpoints[key] = info
        return info

    def _aliases(self, env, method, seed, arm):
        info = self.checkpoints.get((env, method, seed))
        if info:
            return X.equivalent_arms(info, arm, read1_equivalent=self.read1)
        depth = X.test_depth(method)
        found = {arm}
        if depth is None:
            return found
        group = {"final", f"depth:R{depth}", f"gate:depth:R{depth}", "readout:learned"}
        if depth == 1:
            group.add("deploy:R1")
        if arm in group:
            found |= group
        r1 = {"depth:R1", "gate:depth:R1", "deploy:R1"}
        if self.read1 and depth == 4:
            r1.add("readout:read1")
        if arm in r1:
            found |= r1
        return found

    def _put(self, mapping, key, episode, value, *, compare=None):
        old = mapping[key].get(episode)
        if old is not None:
            left = old[0] if isinstance(old, tuple) else old
            right = value[0] if isinstance(value, tuple) else value
            if compare is None:
                compare = left != right
            if compare:
                self.conflicts.add(key)
        mapping[key][episode] = value

    def _accepted(self, env, method, arm, cfg):
        if arm == "final" and cfg in X.configs(env):
            return True
        if env == "had":
            if isinstance(arm, str) and arm.startswith("depth:R"):
                try:
                    depth = int(arm.split("R", 1)[1])
                except (IndexError, ValueError):
                    return False
                return 1 <= depth <= X.max_depth(method) and cfg in X.DEPTH_CONFIGS
            if isinstance(arm, str) and arm.startswith("readout:"):
                return arm.split(":", 1)[1] in X.READOUTS and cfg in X.READOUT_CONFIGS
            if arm == "deploy:R1":
                return cfg in X.configs("had")
            if arm == "gate:depth:R1":
                return cfg in X.GATE_DEPTH_CONFIGS
            if arm == "dup":
                return cfg in X.DUP_CONFIGS
            if isinstance(arm, str) and arm.startswith("intent:"):
                return arm.split(":", 1)[1] in X.INTENT_MODES and cfg in X.DUP_CONFIGS
        if env == "smacv2" and isinstance(arm, str) and arm.startswith("depth:R"):
            try:
                depth = int(arm.split("R", 1)[1])
            except (IndexError, ValueError):
                return False
            return depth in (1, 2, 6) and cfg in X.SMAC_DEPTH_CONFIGS
        return False

    def _read_episodes(self):
        for row in iter_records(self.output, "episodes", run="train"):
            env, method = row.get("env", "had"), row.get("method")
            if env not in ("had", "smacv2") or not row.get("config"):
                self.excluded["非协议环境/缺配置"] += 1
                continue
            cfg, episode = _cfg(row["config"]), int(row["episode_seed"])
            phase, arm = row.get("phase"), row.get("arm")
            seed = int(row.get("seed") or 0)
            metric = _metric(row, env)
            if metric is None:
                self.excluded["指标非有限值/缺失"] += 1
                continue
            record = (metric, _collision_rate(row, cfg), _extra(row, "dup_pursuit_frac"))
            if env == "had":
                if record[1] is None:
                    self.field_missing["collision_rate"] += 1
                if record[2] is None:
                    self.field_missing["dup_pursuit_frac"] += 1
            if phase == "anchor" and env == "had" and method in ("random", "rule_nv1"):
                if cfg in X.configs("had") and 9000 <= episode < 9300:
                    self._put(self.anchors, (method, cfg), episode, record)
                else:
                    self.excluded["anchor配置/随机流不符"] += 1
                continue
            if row.get("protocol") == "no_sg":
                if (method in X.IMPORTED_NO_SG and seed in X.SEEDS and arm == "final"
                        and cfg in X.configs("had") and 9000 <= episode < 9300):
                    self._put(self.legacy, (method, seed, cfg), episode, record)
                    self.sources[row.get("source_version") or "no_sg"] += 1
                else:
                    self.excluded["no_sg行不符"] += 1
                continue
            if method not in X.methods(env) or seed not in X.method_seeds(env, method):
                self.excluded["非登记方法/种子"] += 1
                continue
            if phase == "train_eval":
                point = int(row.get("eval_point") or 0)
                if 1 <= point <= X.validation_point_count(env) and (cfg, episode) in self._expected_validation[env]:
                    key = (env, method, seed, point)
                    self._put(self.validation, key, (cfg, episode), metric)
                    self.validation_steps[key] = max(self.validation_steps.get(key, 0), int(row.get("t_env") or 0))
                    reward = _episode_return(row)
                    if reward is None:
                        self.field_missing["return"] += 1
                    else:
                        self._put(self.validation_return, key, (cfg, episode), reward)
                else:
                    self.excluded["验证配置/随机流/点位不符"] += 1
                continue
            if not self._accepted(env, method, arm, cfg):
                self.excluded["阶段/实验臂/配置不符"] += 1
                continue
            info = self._info(env, method, seed)
            if not info or row.get("checkpoint_id") != info["checkpoint_id"]:
                self.excluded["无合格final身份/身份不符"] += 1
                continue
            if int(row.get("t_env") or 0) != int(info["t_env"]):
                self.excluded["final步数不符"] += 1
                continue
            base = 9000 if env == "had" else 110000
            if episode not in range(base, base + 300):
                self.excluded["终评随机流不符"] += 1
                continue
            self._put(self.episodes, (env, method, seed, arm, cfg), episode, record)
            self.sources[row.get("source_version") or row.get("version") or "未标识"] += 1
            stats = row.get("intent_stats")
            if stats:
                self.intent[(method, seed, cfg, episode)] = stats

    def _read_learning(self):
        for row in iter_records(self.output, "learning", run="train"):
            key = (row.get("env", "had"), row.get("method"), int(row.get("seed") or 0))
            if "grad_spikes" in row:
                self.spikes[key] = max(self.spikes.get(key, 0), int(row["grad_spikes"]))
            if "grad_norm_max" in row:
                self.grad_peaks[key] = max(self.grad_peaks.get(key, 0.0), float(row["grad_norm_max"]))
            if _finite(row.get("intent_loss")):
                self.intent_loss[key] = float(row["intent_loss"])
            if _finite(row.get("intent_acc")):
                self.intent_acc[key] = float(row["intent_acc"])

    def _read_progress(self):
        for row in iter_records(self.output, "progress", run="train"):
            cfg = _cfg(row["config"]) if row.get("phase") == "timing" and row.get("config") else None
            self.progress[(row.get("env", "had"), row.get("method"), row.get("seed"),
                           row.get("phase"), row.get("arm"), cfg)] = row

    def _read_timing(self):
        costs = {}
        for row in iter_records(self.output, "timing", run="train"):
            env, method, seed = row.get("env", "had"), row.get("method"), int(row.get("seed") or 0)
            info = self.checkpoints.get((env, method, seed)) or self._info(env, method, seed)
            arm = row.get("arm") or "cost"
            if not (info and row.get("checkpoint_id") == info["checkpoint_id"]
                    and env == "had" and method in X.COST_METHODS
                    and (arm == "cost" or (isinstance(arm, str) and arm.startswith("cost")))
                    and row.get("device") == "cuda:0"
                    and type(self.measurement_gpu) is int and self.measurement_gpu in (0, 1)
                    and row.get("physical_gpu") == self.measurement_gpu
                    and row.get("n_steps") == 200 and row.get("config")
                    and _cfg(row["config"]) in ((10, 10, 2), (50, 50, 2))
                    and all(_finite(row.get(k)) for k in ("ms_per_step", "p25_ms", "p75_ms", "p95_ms",
                                                         "actor_params", "training_params"))):
                continue
            costs[row["checkpoint_id"], _cfg(row["config"]), arm] = row
        self.costs = list(costs.values())

    def _read_probes(self):
        for kind, name in PROBE_FILES.items():
            path = self.output / "probe" / name
            if not path.exists():
                continue
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if not row.get("method") or row.get("seed") is None:
                        continue
                    item = dict(row)
                    item["seed"] = int(row["seed"])
                    if "round" in row and row["round"] not in ("", None):
                        item["round"] = int(row["round"])
                    for key in ("auc", "acc", "r2", "nll", "alpha", "cosine", "spread",
                                "readout_norm", "shift_from_10v10", "norm", "ratio",
                                "share_pursued", "share_unmarked", "share_teammate",
                                "share_target"):
                        if key in row:
                            item[key] = float(row[key]) if _finite(row[key]) else None
                    self.probes[kind].append(item)
            for method in {r["method"] for r in self.probes[kind]}:
                for seed in X.method_seeds("had", method):
                    info = self.checkpoints.get(("had", method, seed))
                    from .probes0923 import probe_complete
                    self.probe_complete[(kind, method, seed)] = bool(
                        info and probe_complete(self.output, kind, method, seed, info["checkpoint_id"]))

    def values(self, env, method, seed, cfg, arm="final"):
        key = (env, method, seed, arm, cfg)
        if key in self.conflicts:
            return {}
        quota = _quota(env, arm)
        base = 9000 if env == "had" else 110000
        allowed = range(base, base + quota)
        merged = {}
        for alias in self._aliases(env, method, seed, arm):
            alias_key = (env, method, seed, alias, cfg)
            if alias_key in self.conflicts:
                self.conflicts.add(key)
                return {}
            for episode, record in self.episodes.get(alias_key, {}).items():
                if episode not in allowed:
                    continue
                if episode in merged and merged[episode][0] != record[0]:
                    self.conflicts.add(key)
                    return {}
                merged[episode] = record
        return merged if len(merged) == quota else {}

    def seed_mean(self, env, method, seed, configs, arm="final", field=0):
        means = []
        quota = _quota(env, arm)
        for cfg in configs:
            rows = self.values(env, method, seed, cfg, arm)
            if len(rows) != quota:
                return None
            numbers = [rec[field] for rec in rows.values()]
            if field and any(v is None for v in numbers):
                return None
            means.append(statistics.mean(float(v) for v in numbers))
        return statistics.mean(means) if means else None

    def seed_O(self, method, seed):
        groups = [self.seed_mean("had", method, seed, HAD_GROUPS[name]) for name in OOD_GROUPS]
        if any(v is None for v in groups):
            return None
        return statistics.mean(groups)

    def matrix(self, env, method, configs, arm="final", field=0):
        seeds = X.method_seeds(env, method)
        rows = []
        for seed in seeds:
            line = [self.seed_mean(env, method, seed, [cfg], arm, field) if field == 0
                    else self.seed_mean(env, method, seed, [cfg], arm, field)
                    for cfg in configs]
            if any(v is None for v in line):
                return None, seeds
            rows.append(line)
        return np.asarray(rows, dtype=float), seeds

    def iqm_summary(self, env, method, configs, arm="final", field=0):
        matrix, seeds = self.matrix(env, method, configs, arm, field)
        if matrix is None:
            return None
        per_seed = [float(row.mean()) for row in matrix]
        expected = len(X.method_seeds(env, method))
        if len(per_seed) != expected:
            return None
        lo, hi = bootstrap_iqm(matrix)
        tstats = _t_interval(per_seed)
        return dict(iqm=float(iqm(per_seed)), lo=lo, hi=hi, per_seed=tuple(per_seed),
                    seeds=tuple(seeds), mean=float(statistics.mean(per_seed)),
                    t_half=None if tstats is None else tstats[1])

    def anchor_mean(self, method, configs, field=0):
        means = []
        for cfg in configs:
            rows = self.anchors.get((method, cfg), {})
            if len(rows) != 300 or (method, cfg) in self.conflicts:
                return None
            numbers = [rec[field] if isinstance(rec, tuple) else rec for rec in rows.values()]
            if field and any(v is None for v in numbers):
                return None
            means.append(statistics.mean(float(v) for v in numbers))
        return statistics.mean(means) if means else None

    def cell(self, env, method, configs, arm="final", *, points=False, field=0):
        percent = field in (1, 2)
        if method in ("random", "rule_nv1"):
            value = self.anchor_mean(method, configs, field)
            if value is None:
                return "—"
            shown = _pct(value) if percent else _num(value)
            return f"{shown}（共同参照）"
        result = self.iqm_summary(env, method, configs, arm, field)
        text = _fmt_iqm(result, percent=percent)
        if points:
            if result and result.get("t_half") is not None:
                if percent:
                    text += f"; mean {_pct(result['mean'])} ± {_pct(result['t_half'])}"
                else:
                    text += f"; mean {_num(result['mean'])} ± {_num(result['t_half'])}"
            seeds = X.method_seeds(env, method)
            per = [self.seed_mean(env, method, seed, configs, arm, field) for seed in seeds]
            text += " [" + "; ".join(_pct(v) if percent else _num(v) for v in per) + "]"
            if result is None:
                counts = [sum(len(self.episodes.get((env, method, seed, arm, cfg), {}))
                              for cfg in configs) for seed in seeds]
                text += (f" n({ '/'.join(map(str, seeds)) })="
                         f"{'/'.join(map(str, counts))}; quota={_quota(env, arm) * len(configs)}")
        return text

    def ood_wins(self, method, reference="refil"):
        shared = [s for s in X.method_seeds("had", method) if s in X.method_seeds("had", reference)]
        wins = compared = 0
        for seed in shared:
            ours, theirs = self.seed_O(method, seed), self.seed_O(reference, seed)
            if ours is None or theirs is None:
                continue
            compared += 1
            if ours < theirs:
                wins += 1
        return wins, compared

    def val_mean(self, env, method, seed, point):
        key = (env, method, seed, point)
        rows = self.validation.get(key, {})
        quota = len(self._expected_validation[env])
        if len(rows) != quota or key in self.conflicts:
            return None
        return statistics.mean(rows.values())

    def failed(self, method, seed, env="had"):
        if env != "had":
            return None
        final, mid = self.val_mean(env, method, seed, 50), self.val_mean(env, method, seed, 25)
        if final is None and mid is None:
            return None
        return bool((final is not None and final > X.FAILURE["final_validation_D"])
                    or (mid is not None and mid > X.FAILURE["midpoint_validation_D"]))

    def params(self, method, seed=0):
        for row in self.costs:
            if row.get("method") == method and int(row.get("seed") or 0) == seed:
                return int(row["actor_params"]), int(row["training_params"])
        return None

    def integrity(self):
        """Imported 3-seed IQM (equals mean) must match the published main0921 cells."""
        rows = []
        ok = True
        for method, groups in REF_0921.items():
            for group, expected in groups.items():
                if method == "rule_nv1":
                    value = self.anchor_mean(method, HAD_GROUPS[group])
                else:
                    summary = self.iqm_summary("had", method, HAD_GROUPS[group])
                    value = None if summary is None else summary["iqm"]
                match = value is not None and f"{value:.3f}" == f"{expected:.3f}"
                if value is not None and not match:
                    ok = False
                rows.append(dict(method=method, group=group, expected=expected,
                                 observed=None if value is None else float(value), match=match,
                                 pending=value is None))
        return ok, rows


def _ladder_upto(branch):
    names = [("R0", X.FOUNDATION)]
    for name in X.LADDER:
        names.append((name, X.CANDIDATES[name]))
        if branch and name == branch:
            break
    if not branch:
        return [("R0", X.FOUNDATION)] + [(n, X.CANDIDATES[n]) for n in X.LADDER]
    return names


def _table1_methods(branch):
    methods = []
    if branch in X.CANDIDATES:
        methods.append(X.CANDIDATES[branch])
    else:
        methods.extend([X.FOUNDATION, X.CANDIDATES["C"], X.CANDIDATES["A"], X.CANDIDATES["B"]])
    for method in ("refil", "b2_qmix_atten", "dcg", "spectra", "alma", "transfqmix",
                   "refil_matched", "rule_nv1"):
        if method not in methods:
            methods.append(method)
    return methods


def _curve_methods(branch):
    main = X.CANDIDATES[branch] if branch in X.CANDIDATES else None
    methods = []
    if main:
        methods.append(main)
    for method in ("refil", "b2_qmix_atten", "dcg", "spectra", "alma", "transfqmix", "rule_nv1"):
        methods.append(method)
    return methods


def _ablations(branch):
    if branch not in X.BRANCH:
        return []
    return [method for method, _ in X.BRANCH[branch]["had"]]


def _fmt_wins(pair):
    wins, n = pair
    return "—" if n == 0 else f"{wins}/{n}"


def _save_figure(fig, directory, stem, plt):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        target = directory / f"{stem}.{extension}"
        pending = directory / f".{stem}.pending.{extension}"
        fig.savefig(pending, dpi=200, bbox_inches="tight", facecolor="white")
        pending.replace(target)
    plt.close(fig)
    return stem


def _empty(ax, message="pending"):
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color="#707070")


def _style_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    # Latin first: using Noto CJK as the sole family spaces English letters.
    families = ["DejaVu Sans"]
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        families.append(font_manager.FontProperties(fname=str(font)).get_name())
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": families,
                         "axes.unicode_minus": False, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    return plt


def render_figures(results, output, *, paper=False):
    plt = _style_pyplot()
    from matplotlib.patches import FancyBboxPatch
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    branch = results.branch
    prefix = "paper_" if paper else ""
    _fig_architecture(plt, FancyBboxPatch, directory, prefix, branch)
    _fig_scale(results, plt, directory, prefix, paper=paper)
    _fig_mechanism(results, plt, directory, prefix, paper=paper)
    if not paper:
        _fig_learning(results, plt, directory)
        _fig_learning_smac_appendix(results, plt, directory)


def _fig_architecture(plt, FancyBboxPatch, directory, prefix, branch):
    fig, ax = plt.subplots(figsize=(12.2, 4.2))
    ax.set(xlim=(0, 12.4), ylim=(0, 4.2))
    ax.axis("off")
    boxes = [
        (.15, 2.45, 1.7, .85, "#e8e8e8", "Entities + masks"),
        (2.15, 2.45, 1.55, .85, "#e8e8e8", "REFIL host\nH0 (grey)"),
        (4.05, 2.45, 2.55, .85, "#d6eaf8", "Relation module\nC: 1-round · A: ARR · B: IAR"),
        (6.95, 2.45, 2.15, .85, "#d6eaf8", "Memory-cond.\nreadout"),
        (9.45, 2.45, 2.2, .85, "#e8e8e8", "Zero-init residual\n+ GRU / Q"),
        (2.15, .7, 2.3, .85, "#f4f4f4", "1st order (R0)\nread H0 only"),
        (5.05, .7, 2.55, .85, "#d6eaf8", "2nd order\npairwise attention"),
        (8.15, .7, 3.4, .85, "#fdebd0", "K/V: Looped drifts\nARR/IAR stay at H0"),
    ]
    for x, y, w, h, color, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.05", linewidth=.9,
                                    edgecolor="#366480", facecolor=color))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)
    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#4b5861", lw=1.15))
    for a, b in [((1.9, 2.87), (2.1, 2.87)), ((3.75, 2.87), (4.0, 2.87)),
                 ((6.65, 2.87), (6.9, 2.87)), ((9.15, 2.87), (9.4, 2.87)),
                 ((2.9, 2.4), (3.2, 1.6)), ((5.3, 2.4), (6.1, 1.6))]:
        arrow(a, b)
    chosen = f"selected branch: {branch}" if branch else "branch not selected"
    ax.text(6.2, 3.85, f"Plugin relation module on REFIL · {chosen} · schematic, not a result",
            ha="center", fontsize=8)
    _save_figure(fig, directory, f"{prefix}fig1_architecture", plt)


def _series_means(results, env, method, configs, arm="final"):
    if method == "rule_nv1":
        return [results.anchor_mean(method, [cfg]) for cfg in configs]
    return [None if (s := results.iqm_summary(env, method, [cfg], arm)) is None else s["iqm"]
            for cfg in configs]


def _fig_scale(results, plt, directory, prefix, *, paper):
    branch = results.branch
    methods = _curve_methods(branch)
    panels = (
        ("HAD team size (K=2)", "had", tuple((n, n, 2) for n in (5, 10, 15, 20, 25, 30, 40, 50)),
         lambda c: c[0], "N", True),
        ("HAD targets at N=30", "had", tuple((30, 30, k) for k in (2, 4, 6, 9, 12)),
         lambda c: c[2], "K", True),
        ("SMACv2 team size", "smacv2", tuple((n, n, 0) for n in (5, 10, 12, 15, 20)),
         lambda c: c[0], "N", False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)
    for ax, (title, env, configs, xs, xlabel, log_y) in zip(axes, panels):
        shown = False
        for method in methods:
            if env == "smacv2" and method == "rule_nv1":
                continue
            if env == "smacv2" and method not in X.methods("smacv2"):
                continue
            y = _series_means(results, env, method, configs)
            if all(v is None for v in y):
                continue
            style = STYLES.get(method, ("#333333", "o", "-"))
            x = [xs(c) for c in configs]
            plot = [float("nan") if v is None else v for v in y]
            ax.plot(x, plot, color=style[0], marker=style[1], linestyle=style[2],
                    linewidth=1.5, markersize=4, label=paper_name(method, branch))
            shown = True
        ax.set(title=title, xlabel=xlabel,
               ylabel="D ↓" if env == "had" else "Win rate ↑")
        if log_y and env == "had":
            ax.set_yscale("log")
        ax.grid(alpha=.25)
        if shown:
            ax.legend(fontsize=6.5, ncol=2)
        else:
            _empty(ax)
    _save_figure(fig, directory, f"{prefix}fig2_scale", plt)


def _fig_mechanism(results, plt, directory, prefix, *, paper):
    branch = results.branch
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), constrained_layout=True)
    ax = axes[0, 0]
    space = [("R0", X.FOUNDATION), ("C", X.CANDIDATES["C"]), ("A", X.CANDIDATES["A"]),
             ("B", X.CANDIDATES["B"]), ("Looped", "regir_sg"), ("Untied4", "regir_untied4_sg")]
    shown = False
    for name, method in space:
        summary = results.iqm_summary("had", method, sum((HAD_GROUPS[g] for g in OOD_GROUPS), ()))
        params = results.params(method)
        if summary is None or params is None:
            continue
        ax.scatter(params[0] / 1e6, summary["iqm"], s=40, label=name)
        ax.annotate(name, (params[0] / 1e6, summary["iqm"]), textcoords="offset points",
                    xytext=(4, 3), fontsize=7)
        shown = True
    ax.set(title="a. Design space (O-IQM vs actor params)", xlabel="Actor params (M)", ylabel="O IQM ↓")
    if shown:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    else:
        _empty(ax, "pending M4 / trial finals")
    ax = axes[0, 1]
    _panel_branch_mechanism(results, ax, branch)
    ax = axes[1, 0]
    depths = range(1, 7)
    depth_methods = {
        "A": ("regir_kv0_sg", "regir_sg", "regir_kv0_fixed4_sg", "regir_untied4_sg"),
        "B": ("regir_kv0_intent_sg", "regir_kv0_sg"),
        "C": ("regir_r1_sg",),
    }.get(branch, ("regir_kv0_sg", "regir_r1_sg", "regir_kv0_intent_sg"))
    shown = False
    cfg = (50, 50, 2)
    for method in depth_methods:
        y = []
        for depth in depths:
            if X.max_depth(method) < depth:
                y.append(None)
                continue
            y.append(None if (s := results.iqm_summary("had", method, [cfg], f"depth:R{depth}")) is None
                     else s["iqm"])
        if all(v is None for v in y):
            continue
        style = STYLES.get(method, ("#333", "o", "-"))
        ax.plot(list(depths), [float("nan") if v is None else v for v in y],
                color=style[0], marker=style[1], label=paper_name(method, branch))
        shown = True
    ax.set(title="c. Execution depth at 50v50 K2", xlabel="R", ylabel="D ↓", xticks=list(depths))
    if shown:
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    else:
        _empty(ax)
    ax = axes[1, 1]
    ns = (10, 30, 50)
    shown = False
    methods = _curve_methods(branch)
    if branch:
        methods = [X.CANDIDATES[branch], "refil", X.CANDIDATES.get("A", "regir_kv0_sg")]
        methods = list(dict.fromkeys(m for m in methods if m))
    for method in methods:
        coll = [results.iqm_summary("had", method, [(n, n, 2)], field=1)
                for n in ns]
        dups = [results.iqm_summary("had", method, [(n, n, 2)], field=2) for n in ns]
        cy = [None if s is None else s["iqm"] for s in coll]
        dy = [None if s is None else s["iqm"] for s in dups]
        style = STYLES.get(method, ("#333", "o", "-"))
        if any(v is not None for v in cy):
            ax.plot(ns, [float("nan") if v is None else v for v in cy],
                    color=style[0], marker=style[1], linestyle="-",
                    label=f"{paper_name(method, branch)} collide %")
            shown = True
        if any(v is not None for v in dy):
            ax.plot(ns, [float("nan") if v is None else v for v in dy],
                    color=style[0], marker=style[1], linestyle="--",
                    label=f"{paper_name(method, branch)} dup")
            shown = True
    ax.set(title="d. Friendly-collision and duplicate-pursuit rates vs N",
           xlabel="N (K=2)", ylabel="rate (0–1)")
    if shown:
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(alpha=.25)
    else:
        _empty(ax)
    _save_figure(fig, directory, f"{prefix}fig3_mechanism", plt)


def _panel_branch_mechanism(results, ax, branch):
    ax.set_title("b. Branch mechanism", loc="left")
    if branch == "A":
        rows = [r for r in results.probes["dynamics"]
                if r.get("method") == "regir_kv0_sg" and r.get("config") == "50v50_K2"
                and results.probe_complete.get(("dynamics", r["method"], r["seed"]))]
        if not rows:
            return _empty(ax, "pending dynamics")
        by_round = defaultdict(list)
        for row in rows:
            if _finite(row.get("shift_from_10v10")):
                by_round[row["round"]].append(float(row["shift_from_10v10"]))
        if not by_round:
            return _empty(ax, "pending K/V shift")
        xs = sorted(by_round)
        ax.plot(xs, [statistics.mean(by_round[r]) for r in xs], marker="o")
        ax.set(xlabel="round", ylabel="K/V shift from 10v10")
        ax.grid(alpha=.25)
        return
    if branch == "B":
        rows = [r for r in results.probes["intent_accuracy"]
                if r.get("control") == "trained" and results.probe_complete.get(
                    ("intent_accuracy", r["method"], r["seed"]))]
        if not rows:
            return _empty(ax, "pending intent probe")
        by_cfg = defaultdict(list)
        for row in rows:
            if _finite(row.get("acc")):
                by_cfg[row.get("config")].append(float(row["acc"]))
        labels = sorted(by_cfg)
        ax.bar(range(len(labels)), [statistics.mean(by_cfg[k]) for k in labels])
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=7)
        ax.set(ylabel="intent top-1")
        return
    rows = [r for r in results.probes["coverage"]
            if r.get("control") == "trained" and r.get("split") == "ood_confirm"
            and results.probe_complete.get(("coverage", r["method"], r["seed"]))]
    if not rows:
        return _empty(ax, "pending coverage")
    by = defaultdict(list)
    for row in rows:
        if _finite(row.get("auc")):
            by[(row["method"], row["round"])].append(float(row["auc"]))
    for method in ("regir_r0_sg", "regir_r1_sg", "regir_kv0_sg"):
        xs = sorted({r for m, r in by if m == method})
        if not xs:
            continue
        ax.plot(xs, [statistics.mean(by[(method, r)]) for r in xs], marker="o",
                label=paper_name(method, branch))
    ax.set(xlabel="round", ylabel="coverage AUC")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    else:
        _empty(ax, "pending coverage")


def _draw_learning(ax, results, env, methods, *, store, title, ylabel, cfg=None,
                   ylim=None):
    expected = results._expected_validation[env]
    quota = len(expected) if cfg is None else sum(1 for item, _ in expected if item == cfg)
    shown = False
    for method in methods:
        seeds = X.method_seeds(env, method)
        style = STYLES.get(method, ("#333333", "o", "-"))
        xs, ys, lo, hi = [], [], [], []
        for point in range(1, X.validation_point_count(env) + 1):
            values, steps = [], []
            for seed in seeds:
                key = (env, method, seed, point)
                rows = store.get(key, {})
                if cfg is not None:
                    rows = {item: value for item, value in rows.items() if item[0] == cfg}
                if len(rows) != quota or key in results.conflicts:
                    continue
                values.append(statistics.mean(rows.values()))
                steps.append((results.validation_steps.get(key, 0)
                              or X.validation_thresholds(env)[point - 1]) / 1e6)
            if not values:
                xs.append(X.validation_thresholds(env)[point - 1] / 1e6)
                ys.append(float("nan"))
                lo.append(float("nan"))
                hi.append(float("nan"))
                continue
            mean = statistics.mean(values)
            spread = statistics.stdev(values) if len(values) >= 2 else 0.0
            xs.append(statistics.mean(steps))
            ys.append(mean)
            lo.append(mean - spread)
            hi.append(mean + spread)
        if not any(math.isfinite(v) for v in ys):
            continue
        ax.plot(xs, ys, color=style[0], linewidth=1.45,
                label=paper_name(method, results.branch))
        ax.fill_between(xs, lo, hi, color=style[0], alpha=.18)
        shown = True
    ax.set(title=title, xlabel="steps (M)", ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=.25)
    if shown:
        ax.legend(fontsize=6.5, ncol=2)
    else:
        _empty(ax)


def _smac_learning_methods():
    return list(X.methods("smacv2"))


def _fig_learning(results, plt, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8), constrained_layout=True)
    _draw_learning(
        axes[0], results, "had",
        [X.FOUNDATION, *X.CANDIDATES.values(), *X.IMPORTED_BASELINES],
        store=results.validation, title="HAD", ylabel="D ↓")
    _draw_learning(
        axes[1], results, "smacv2", _smac_learning_methods(),
        store=results.validation, cfg=SMAC_LEARNING_CFG,
        title=_smac_scenario(SMAC_LEARNING_CFG), ylabel="win rate ↑", ylim=(0, 1))
    _save_figure(fig, directory, "fig_learning", plt)


def _fig_learning_smac_appendix(results, plt, directory):
    extra = [cfg for cfg in X.SMAC_VALIDATION if cfg != SMAC_LEARNING_CFG]
    fig, axes = plt.subplots(1, max(len(extra), 1), figsize=(11.2, 3.4), constrained_layout=True)
    if len(extra) == 1:
        axes = [axes]
    for ax, cfg in zip(axes, extra):
        _draw_learning(
            ax, results, "smacv2", _smac_learning_methods(),
            store=results.validation, cfg=cfg,
            title=_smac_scenario(cfg), ylabel="win rate ↑", ylim=(0, 1))
    for ax in axes[len(extra):]:
        ax.axis("off")
    _save_figure(fig, directory, "fig_learning_smac_appendix", plt)


def _figure_lines(stem, title, caption):
    return [f"![{title}](figures/{stem}.png)", "", caption,
            f"[PDF](figures/{stem}.pdf) · [PNG](figures/{stem}.png)", ""]


def _cost_table(results):
    rows = []
    for method in X.COST_METHODS:
        records = [r for r in results.costs if r.get("method") == method]
        actor = sorted({int(r["actor_params"]) for r in records})
        total = sorted({int(r["training_params"]) for r in records})
        latencies = []
        for n in (10, 50):
            cells = []
            for seed in X.SEEDS:
                selected = [r for r in records if int(r.get("seed") or 0) == seed
                            and _cfg(r["config"]) == (n, n, 2)]
                devices = {(r.get("physical_gpu"), r.get("device")) for r in selected}
                if len(devices) != 1 or not selected:
                    cells.append("—")
                    continue
                last = selected[-1]
                cells.append(f"{_num(last['ms_per_step'])} [{_num(last.get('p25_ms'))}, "
                             f"{_num(last.get('p75_ms'))}]; p95={_num(last.get('p95_ms'))}")
            latencies.append("<br>".join(f"s{s}: {v}" for s, v in zip(X.SEEDS, cells)))
        rows.append([paper_name(method, results.branch),
                     "/".join(map(str, actor)) or "—", "/".join(map(str, total)) or "—",
                     *latencies])
    return _table(["方法", "actor 参数", "训练总参数", "N=10 ms/决策", "N=50 ms/决策"], rows)


def _probe_pointer(results, kind):
    path = results.output / "probe" / PROBE_FILES[kind]
    complete = sum(1 for key, ok in results.probe_complete.items() if key[0] == kind and ok)
    total = sum(1 for key in results.probe_complete if key[0] == kind)
    status = "已有" if path.exists() else "尚未写出"
    return f"`probe/{PROBE_FILES[kind]}`：{status}；完成标记 {complete}/{total}"


def _snapshot(results):
    had = {}
    for method in X.methods("had") + ("rule_nv1",):
        had[method] = {}
        for group in FORMAL_GROUPS:
            if method == "rule_nv1":
                value = results.anchor_mean(method, HAD_GROUPS[group])
                had[method][group] = None if value is None else dict(iqm=value, kind="anchor")
            else:
                summary = results.iqm_summary("had", method, HAD_GROUPS[group])
                had[method][group] = None if summary is None else {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in summary.items()}
    ok, checks = results.integrity()
    return dict(
        profile=X.PROFILE, created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        branch=results.branch, identity="300 episodes, checkpoint_id, t_env, scene 9000–9299 / 110000–110299",
        iqm="Agarwal 2021; 5-seed trim 1+1; 3-seed = mean",
        bootstrap=dict(reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED),
        had_groups=had,
        excluded=dict(results.excluded),
        field_missing=dict(results.field_missing),
        sources=dict(results.sources),
        conflicts=[str(k) for k in sorted(results.conflicts, key=str)],
        checkpoints={f"{e}/{m}/s{s}": info.get("checkpoint_id")
                     for (e, m, s), info in results.checkpoints.items()},
        integrity_ok=ok, integrity=checks,
        probe_complete={f"{k[0]}/{k[1]}/s{k[2]}": v for k, v in results.probe_complete.items()},
        costs=len(results.costs),
        coverage={
            f"{env}/{method}/s{seed}":
            f"{sum(1 for cfg in X.configs(env) if results.values(env, method, seed, cfg))}/"
            f"{len(X.configs(env))}"
            for env in ("had", "smacv2")
            for method in X.methods(env)
            for seed in X.method_seeds(env, method)
        },
    )


def render_markdown(results, *, paper=False):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    branch = results.branch
    ok, checks = results.integrity()
    mismatches = [c for c in checks if c["observed"] is not None and not c["match"]]
    pending_import = [c for c in checks if c["pending"]]
    lines = []
    if paper:
        lines += [f"# main0923 论文版（{now}）", "",
                  "3 图 2 表。主统计量为组内等权后的跨种子 IQM 与分层 bootstrap 95% 区间。"
                  "缺完整种子的方法不连成曲线；格内显示 —，不插值。", ""]
    else:
        lines += [f"# main0923：试训、闸门与机制报告（{now}）", "",
                  f"主方法：{branch or '未定'}。唯一正式报告；正文与附录共用同一数据源。"
                  "正式格必须匹配 inventory / 磁盘 final 的 checkpoint_id 与 t_env，"
                  "且每种子完成规定随机流的全部回合（HAD 9000–9299 共 300 局；"
                  "SMACv2 110000–110299；闸门/重复追击/意图干预为前 100 局）。", "",
                  "统计顺序：每种子每配置求回合均值 → 组内配置等权 → 跨种子 IQM"
                  "（5 种子去掉两端各 1 个，与闸门 `branch_gate.iqm` 相同；3 种子即均值）。"
                  "区间为分层 bootstrap 10,000 次（种子与组内配置有放回重采样，种子 20260924）的 2.5/97.5 分位。"
                  "附录另给均值 ± 95% t 与逐种子值。Rule 是同一场景库的单组共同参照，没有虚构训练种子。", ""]
    if mismatches:
        lines += ["**身份核对失败：导入基线与 main0921 发表格不一致，下表数字不得写入论文。**", ""]
        lines += _table(["方法", "组", "0921", "本次"],
                        [[c["method"], c["group"], c["expected"], _num(c["observed"])] for c in mismatches])
    elif pending_import:
        lines += ["导入基线中仍有未完成的正式格，已完成项与 main0921 一致；未完成项显示 —。", ""]
    else:
        lines += ["导入基线四组 IQM（3 种子即均值）与 main0921 表 1 三位小数一致。", ""]
    if results.conflicts:
        lines += [f"**冲突格 {len(results.conflicts)} 个，已排除，不进入任何均值。**", ""]

    if not paper:
        lines += ["## 1. 试训与闸门", ""]
        decision = results.decision
        if decision:
            lines += [f"判定文件时间：{decision.get('computed_at', '—')}；"
                      f"方式：{decision.get('mode', '—')}；选择：{decision.get('choice') or '未定'}。", ""]
            lines += _table(["级别", "O-IQM", "失败", "种子数"],
                            [[n, _num((decision.get("stats") or {}).get(n, {}).get("O")),
                              (decision.get("stats") or {}).get(n, {}).get("failures", "—"),
                              (decision.get("stats") or {}).get(n, {}).get("n", "—")]
                             for n in ("R0", "C", "A", "B") if n in (decision.get("stats") or {})])
            lines += ["理由：", ""] + [f"- {r}" for r in decision.get("reasons") or ["尚未生成"]] + [""]
        else:
            lines += ["闸门尚未写出 `decision/decision.json`。试训未齐时不推断选择。", ""]
        if results.branch_record:
            lines += [f"已放行分支：`{results.branch_record.get('branch')}`，"
                      f"mode={results.branch_record.get('mode')}。", ""]

    lines += ["## 图1：方法结构" if not paper else "### 图 1", ""]
    stem = "paper_fig1_architecture" if paper else "fig1_architecture"
    lines += _figure_lines(stem, "图1 方法结构",
                           "REFIL 宿主为灰色，关系模块为彩色。图示不是机制实验的证据，不含数值结果。")

    groups = FORMAL_GROUPS
    table_methods = _table1_methods(branch)
    lines += ["## 表1：主比较" if not paper else "### 表 1", "",
              "HAD：IQM [bootstrap 2.5%, 97.5%]。相对 REFIL 的逐种子胜负只比较双方都完整的重叠种子（通常 0–2），"
              "比较量为三个 OOD 组的等权平均 $O$。SMAC 为胜率 IQM。参数来自合格 M4 行。", ""]
    rows = []
    for method in table_methods:
        if method == "rule_nv1":
            line = [paper_name(method, branch),
                    *[f"{_num(results.anchor_mean(method, HAD_GROUPS[g]))}（共同参照）" for g in groups],
                    "—", "—", "—"]
        else:
            wins = results.ood_wins(method)
            params = results.params(method)
            smac = [results.cell("smacv2", method, [cfg]) if method in X.methods("smacv2") else "—"
                    for cfg in SMAC_KEY]
            line = [paper_name(method, branch),
                    *[results.cell("had", method, HAD_GROUPS[g]) for g in groups],
                    "—" if method == "refil" else _fmt_wins(wins),
                    "/".join(map(str, params)) if params else "—",
                    " / ".join(smac)]
        rows.append(line)
    lines += _table(["方法", *groups, "相对 REFIL $O$", "actor/训练参数",
                     "SMAC 10/15/20/10v15"], rows)

    lines += ["## 图2：规模曲线" if not paper else "### 图 2", ""]
    stem = "paper_fig2_scale" if paper else "fig2_scale"
    lines += _figure_lines(stem, "图2 规模曲线",
                           "HAD 用对数纵轴，只画完整方法的 IQM 均值；缺格断开，不插值。"
                           "SMACv2 胜率在 0–1，用线性轴。配色全文统一。")

    lines += ["## 表2：关系阶梯与消融" if not paper else "### 表 2", "",
              "阶梯始终为 R0 → C → A → B，到已选定的主方法为止（未选定时列全阶梯，缺格为 —）。"
              "50v50 K2 的相撞率是红方碰撞死亡 / $N_R$（与规划文档一致），"
              "重复追击是 `dup_pursuit_frac`；缺字段不填零。", ""]
    ladder = _ladder_upto(branch)
    extra = _ablations(branch)
    rows = []
    for name, method in list(ladder) + [(paper_name(m, branch), m) for m in extra]:
        pretty = paper_name(method, branch)
        label = pretty if name == pretty or name not in ("R0", "C", "A", "B") else f"{name} / {pretty}"
        rows.append([
            label,
            *[results.cell("had", method, HAD_GROUPS[g]) for g in groups],
            results.cell("had", method, [(50, 50, 2)], field=1),
            results.cell("had", method, [(50, 50, 2)], field=2),
        ])
    lines += _table(["方法", *groups, "50v50 相撞率", "50v50 重复追击"], rows)
    lines += ["导入基线在同一 50v50 K2 终评上的协同指标（缺 `dup_pursuit_frac` 不填零）：", ""]
    lines += _table(["方法", "50v50 相撞率", "50v50 重复追击"], [
        [paper_name(m, branch),
         results.cell("had", m, [(50, 50, 2)], field=1),
         results.cell("had", m, [(50, 50, 2)], field=2)]
        for m in (*X.IMPORTED_BASELINES, "rule_nv1")])

    lines += ["## 图3：机制" if not paper else "### 图 3", ""]
    stem = "paper_fig3_mechanism" if paper else "fig3_mechanism"
    lines += _figure_lines(stem, "图3 机制",
                           "(a) 设计空间点图需合格 M4 参数与 O-IQM；(b) 随分支切换："
                           "A 为 K/V 漂移，B 为意图准确率，C/未选为覆盖探针；"
                           "(c) 执行深度；(d) 相撞率与重复追击率随 N。"
                           "任一面板缺完整数据时标为待完成，不画占位曲线。")

    if paper:
        return "\n".join(lines).rstrip() + "\n"

    lines += ["## 2. 学习曲线", ""] + _figure_lines(
        "fig_learning", "训练内验证",
        "训练中每隔一段步数会对四个验证场景各跑前向评估；正文只看 "
        f"`{_smac_scenario(SMAC_LEARNING_CFG)}`。"
        "SMAC 纵轴是训练中验证胜率 `battle_won`，与正式指标相同。"
        "其余验证场景的同口径曲线见附录 B。"
        "每个验证点：该场景 32 局齐了的种子算均值 ±1 标准差；1 个种子时带宽为 0。")

    lines += ["## 3. 成本 M4", "",
              "仅保留匹配当前 final 身份、登记物理 GPU、float32、200 次同步计时的行。", ""]
    lines += _cost_table(results)

    lines += ["## 4. 探针文件", ""]
    lines += [f"- {_probe_pointer(results, kind)}" for kind in PROBE_FILES]
    lines += [""]

    lines += ["## 附录 A. HAD 全部配置 × 方法", "",
              "每格：IQM [bootstrap]；mean ± 95% t（与 main0921 同口径）；方括号为逐种子组均值。"
              "不完整格保留已有种子值与有效局数，这些值不进入正式 IQM。", ""]
    for group, cfgs in HAD_GROUPS.items():
        lines += [f"### A.{list(HAD_GROUPS).index(group) + 1} {group}", ""]
        methods = list(X.methods("had")) + ["rule_nv1"]
        lines += _table(["方法", *map(_label, cfgs)], [
            [paper_name(method, branch),
             *(results.cell("had", method, [cfg], points=True) for cfg in cfgs)]
            for method in methods])

    lines += ["### A.x 旧协议 no-SG（附录对照，不进正文）", "",
              "导入行标注 `protocol=no_sg`，无本目录权重；仅当 300 局场景齐全时给均值。", ""]
    no_sg_rows = []
    for method in X.IMPORTED_NO_SG:
        line = [paper_name(method, branch)]
        for group in FORMAL_GROUPS:
            seeds = []
            for seed in X.SEEDS:
                means = []
                complete = True
                for cfg in HAD_GROUPS[group]:
                    rows = results.legacy.get((method, seed, cfg), {})
                    if len(rows) != 300:
                        complete = False
                        break
                    means.append(statistics.mean(rec[0] for rec in rows.values()))
                seeds.append(statistics.mean(means) if complete else None)
            if any(v is None for v in seeds):
                line.append("— [" + "; ".join(_num(v) for v in seeds) + "]")
            else:
                line.append(f"{_num(iqm(seeds))} [" + "; ".join(_num(v) for v in seeds) + "]")
        no_sg_rows.append(line)
    lines += _table(["方法", *FORMAL_GROUPS], no_sg_rows)

    lines += [f"## 附录 B. SMACv2（{len(X.SMAC_FINAL)} 配置）", ""]
    extra = "、".join(_smac_scenario(cfg) for cfg in X.SMAC_VALIDATION if cfg != SMAC_LEARNING_CFG)
    lines += _figure_lines(
        "fig_learning_smac_appendix", "训练内验证（其余场景）",
        f"与正文同一套训练中验证胜率 `battle_won`；此处为 {extra}。"
        f"正文主看 `{_smac_scenario(SMAC_LEARNING_CFG)}`。正式外推评估仍用胜率，见下表。")
    lines += _table(["方法", *map(_label, X.SMAC_FINAL)], [
        [paper_name(method, branch),
         *(results.cell("smacv2", method, [cfg], points=True) for cfg in X.SMAC_FINAL)]
        for method in X.methods("smacv2")])

    lines += ["## 附录 C. 执行深度、读出、意图干预、重复追击", ""]
    depth_methods = list(dict.fromkeys(
        [X.CANDIDATES["A"], X.CANDIDATES["B"], "regir_sg", "regir_kv0_fixed4_sg", "regir_untied4_sg"]))
    for method in depth_methods:
        lines += [f"### {paper_name(method, branch)} 执行深度", ""]
        lines += _table(["R", *map(_label, X.DEPTH_CONFIGS)], [
            [f"R{d}", *(results.cell("had", method, [cfg], f"depth:R{d}", points=True)
                        for cfg in X.DEPTH_CONFIGS)]
            for d in range(1, X.max_depth(method) + 1)])
    lines += _table(["读出", *map(_label, X.READOUT_CONFIGS)], [
        [r, *(results.cell("had", X.CANDIDATES.get(branch, X.CANDIDATES["A"]), [cfg],
                           f"readout:{r}", points=True) for cfg in X.READOUT_CONFIGS)]
        for r in X.READOUTS])
    lines += _table(["干预", *map(_label, X.DUP_CONFIGS)], [
        [m, *(results.cell("had", X.CANDIDATES["B"], [cfg], f"intent:{m}", points=True)
              for cfg in X.DUP_CONFIGS)]
        for m in X.INTENT_MODES])
    lines += ["导入基线 `arm=dup`（每格 100 局）重复追击：", ""]
    lines += _table(["方法", *map(_label, X.DUP_CONFIGS)], [
        [paper_name(m, branch),
         *(results.cell("had", m, [cfg], "dup", points=True, field=2) for cfg in X.DUP_CONFIGS)]
        for m in X.IMPORTED_BASELINES])

    lines += ["## 附录 D. 失败、尖峰、排除与来源", "",
              f"失败判据：最终验证 D>{X.FAILURE['final_validation_D']} 或 50 万步验证 D>{X.FAILURE['midpoint_validation_D']}。", ""]
    fail_rows = []
    for env in ("had", "smacv2"):
        for method in X.methods(env):
            for seed in X.method_seeds(env, method):
                fail_rows.append([
                    env, paper_name(method, branch), seed,
                    {True: "失败", False: "否", None: "验证未齐"}[results.failed(method, seed, env)],
                    results.spikes.get((env, method, seed), "—"),
                    _num(results.grad_peaks.get((env, method, seed))),
                    _num(results.intent_loss.get((env, method, seed))),
                    results.checkpoints.get((env, method, seed), {}).get("checkpoint_id", "—"),
                    results.checkpoints.get((env, method, seed), {}).get("t_env", "—"),
                ])
    lines += _table(["环境", "方法", "seed", "训练失败", "尖峰", "最大梯度", "intent_loss",
                     "artifact", "t_env"], fail_rows)
    lines += _table(["排除原因", "记录数"], sorted(results.excluded.items()) or [["本次读取未发现排除", 0]])
    lines += _table(["缺字段（有限值才计数；导入行缺 dup 属预期）", "记录数"],
                    sorted(results.field_missing.items()) or [["无", 0]])
    lines += [f"冲突格：**{len(results.conflicts)}**。", ""]
    lines += _table(["来源", "行数"], sorted(results.sources.items()) or [["—", 0]])
    tasks = results.inventory.get("tasks", [])
    lines += ["任务状态来自当前 inventory；`completed` 文字本身不能使结果进入统计。", ""]
    lines += _table(["任务", "环境", "方法", "seed", "状态", "完成/总量"], [
        [t.get("id", "—"), t.get("env", "—"), t.get("method", "—"), t.get("seed", "—"),
         t.get("status", "—"), f"{t.get('completed', '—')}/{t.get('total', '—')}"]
        for t in tasks[:400]] or [["inventory 尚未生成", "—", "—", "—", "待完成", "—"]])
    if len(tasks) > 400:
        lines += [f"（inventory 共 {len(tasks)} 条，上表截断至 400。）", ""]
    lines += ["原始逐局：[episodes.csv](episodes.csv)；训练：[learning.csv](learning.csv)；"
              "进度：[progress.csv](progress.csv)；时延：[timing.csv](timing.csv)；"
              "机器可读汇总：[aggregates.json](aggregates.json)；"
              "冻结协议：[experiment.json](experiment.json)。不另造第二套汇总 CSV。", ""]
    return "\n".join(lines).rstrip() + "\n"


def _load(output, run="train"):
    if run != "train":
        raise ValueError("main0923 正式报告只读 train run")
    output = Path(output)
    metadata = _json(output / "experiment.json", {})
    if metadata.get("profile") not in (None, X.PROFILE) and metadata.get("profile") != X.PROFILE:
        raise ValueError(f"report0923 refuses profile={metadata.get('profile')}")
    inventory = _json(output / "inventory.json", {})
    return Results(output, inventory, metadata)


def render_report(output, *, run="train"):
    output = Path(output)
    results = _load(output, run=run)
    X.atomic_json(output / "aggregates.json", _snapshot(results))
    try:
        render_figures(results, output, paper=False)
    except Exception as error:
        print(f"formal figures skipped: {type(error).__name__}: {error}", flush=True)
    return _atomic_text(output / "实验报告.md", render_markdown(results, paper=False))


def render_paper(output, *, run="train"):
    output = Path(output)
    results = _load(output, run=run)
    X.atomic_json(output / "aggregates.json", _snapshot(results))
    try:
        render_figures(results, output, paper=True)
    except Exception as error:
        print(f"paper figures skipped: {type(error).__name__}: {error}", flush=True)
    return _atomic_text(output / "实验报告_论文版.md", render_markdown(results, paper=True))

"""Read-only, bounded-I/O terminal status for the main0921 experiment.

No checkpoint loads, CSV scans, scheduler calls, or persistent dashboard state.
Scheduler snapshots describe live work; inventory remains the authority for
qualified completion. Console tails only fill missing live progress fields.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata


_HAD = ("regir", "refil", "b2_qmix_atten", "dcg", "spectra", "alma", "regir_norefil",
        "regir_nocount", "regir_r1", "regir_last", "refil_matched", "transfqmix",
        "regir_fixed4", "regir_untied4", "regir_kv0")
_SMAC = ("regir", "regir_r1", "refil", "b2_qmix_atten", "spectra", "transfqmix")
_NAMES = dict(train="训练", final="正式终评", depth="M1深度", readout="M2读出", probe="M3探针", timing="M4成本")
_DONE = {"complete", "completed"}
_ACTIVE = {"running", "training", "evaluating", "starting", "waiting", "stopping", "pending"}
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAIN = re.compile(r"\[(?P<method>[^]]+)\]\s+(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)\s+"
                    r"(?P<done>[\d,]+)/(?P<total>[\d,]+)\s+(?P<rate>[\d.]+)/s(?P<rest>[^\n]*)")
_EVAL = re.compile(r"(?P<id>eval\.(?:final|depth|readout|probe|timing)\.[\w.]+)\s+"
                   r"(?P<phase>\w+)\s+(?P<done>[\d,]+)/(?P<total>[\d,]+)(?P<rest>[^\n]*)")


def _number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _json(path, issues=None):
    path = Path(path)
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("小状态文件超过4MiB，拒绝读取")
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        if issues is not None:
            issues.append(f"{path.name}: {error}")
        return {}


def _tail(path, limit=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - limit))
            return _ANSI.sub("", stream.read(limit).decode("utf-8", errors="replace"))
    except OSError:
        return ""


def _stamp(value):
    numeric = _number(value)
    if numeric is not None:
        return numeric
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _age(path, now):
    try:
        return max(0, now - Path(path).stat().st_mtime)
    except OSError:
        return None


def _alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        # kill(pid, 0) alone considers a zombie alive.
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().rpartition(")")[2].split()[0] == "Z":
            return False
        return True
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError, IndexError):
        return False


def _duration(seconds):
    value = _number(seconds)
    if value is None:
        return "暂不可估算"
    seconds = max(0, int(value))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}天{hours}小时{minutes}分"
    if hours:
        return f"{hours}小时{minutes}分"
    return f"{minutes}分{rest % 60}秒"


def _range(values):
    if not values or len(values) != 2 or any(_number(x) is None for x in values):
        return "暂不可估算"
    lo, hi = sorted(max(0., float(x)) for x in values)
    return f"{_duration(lo)} ～ {_duration(hi)}"


def _directory(root, task):
    prefix = root if task.get("env", "had") == "had" else root / task["env"]
    return prefix / (task.get("method") or "eval") / "train" / f"seed_{int(task.get('seed') or 0)}"


def _identity(row, kind, env):
    if row.get("id"):
        return str(row["id"])
    method, seed = row.get("method"), row.get("seed")
    if not method or seed is None:
        return "eval.timing.had.all" if row.get("kind") == "timing" else None
    return (f"train.{env}.{method}.s{seed}" if kind == "train" else
            f"eval.{row.get('kind', 'final')}.{env}.{method}.s{seed}")


def _tasks(metadata, inventory):
    tasks = {row["id"]: dict(row) for row in inventory.get("tasks", []) if row.get("id")}
    checkpoints = {(r.get("env", "had"), r.get("method"), r.get("seed")): r
                   for r in inventory.get("checkpoints", [])}
    for env, fallback in (("had", _HAD), ("smacv2", _SMAC)):
        config = metadata.get("environments", {}).get(env, {})
        for method in config.get("methods", fallback):
            for seed in (0, 1, 2):
                base = dict(env=env, method=method, seed=seed)
                identity = f"train.{env}.{method}.s{seed}"
                final = checkpoints.get((env, method, seed))
                tasks.setdefault(identity, dict(base, id=identity, kind="train",
                    status="complete" if final else "pending", completed=final.get("t_env", 0) if final else 0,
                    total=config.get("budget", 1000000 if env == "had" else 4000000),
                    detail="已登记合格final" if final else "尚未登记训练完成"))
                for kind in (("final", "depth", "readout") if env == "had" and method == "regir" else ("final",)):
                    identity = f"eval.{kind}.{env}.{method}.s{seed}"
                    total = (7200 if env == "had" else 2400) if kind == "final" else 5400
                    tasks.setdefault(identity, dict(base, id=identity, kind=kind, status="waiting", completed=0,
                                                    total=total, detail="等待本版本任务清单/依赖登记"))
    for seed in (0, 1, 2):
        identity = f"eval.probe.had.regir.s{seed}"
        tasks.setdefault(identity, dict(id=identity, kind="probe", env="had", method="regir", seed=seed,
                                       status="waiting", completed=0, total=1, detail="公共状态库及冻结Full"))
    tasks.setdefault("eval.timing.had.all", dict(id="eval.timing.had.all", kind="timing", env="had", method=None,
                     seed=None, status="waiting", completed=0, total=48, detail="全部成本权重、公共状态库及空闲测量卡"))
    return tasks


def _train_tail(text):
    rows = []
    for match in _TRAIN.finditer(text):
        row = match.groupdict()
        elapsed = int(row["h"]) * 3600 + int(row["m"]) * 60 + int(row["s"])
        if rows and elapsed < rows[-1]["session_elapsed_seconds"]:
            rows.clear()  # A resumed process starts a new elapsed-time clock.
        extra = re.search(r"\bev=(\d+)/(\d+)", row["rest"])
        rows.append(dict(t_env=int(row["done"].replace(",", "")), budget_steps=int(row["total"].replace(",", "")),
                         session_elapsed_seconds=elapsed, steps_per_second=float(row["rate"]),
                         current_phase="validation" if extra else "training",
                         eval_completed=int(extra[1]) if extra else None, eval_total=int(extra[2]) if extra else None))
    if not rows:
        return {}
    result = rows[-1]
    # Recent wall-clock throughput, not the cumulative average printed by the
    # trainer. Stay within the current uninterrupted training/validation phase.
    first = result
    for previous in reversed(rows[:-1]):
        if previous["current_phase"] != result["current_phase"] or previous["t_env"] > result["t_env"]:
            break
        first = previous
        if result["session_elapsed_seconds"] - first["session_elapsed_seconds"] >= 60:
            break
    seconds = result["session_elapsed_seconds"] - first["session_elapsed_seconds"]
    if seconds > 0 and result["current_phase"] == "training":
        result["recent_steps_per_second"] = (result["t_env"] - first["t_env"]) / seconds
    rates = [r["steps_per_second"] for r in rows[-8:] if r["steps_per_second"] > 0]
    if rates:
        result["rate_range"] = (min(rates), max(rates))
    return result


def _eval_tail(text):
    result = {}
    for match in _EVAL.finditer(text):
        row = match.groupdict()
        extra = re.search(r"\beta (\d+):(\d+):(\d+)", row["rest"])
        item = dict(completed=int(row["done"].replace(",", "")), total=int(row["total"].replace(",", "")), phase=row["phase"])
        if extra:
            item["remaining_seconds"] = int(extra[1]) * 3600 + int(extra[2]) * 60 + int(extra[3])
        result[row["id"]] = item
    return result


def _overlay(task, live):
    for key, value in live.items():
        if value is not None and key not in ("id", "kind", "completed", "total", "status"):
            task[key] = value
    if task["kind"] == "train":
        task["completed"] = min(task["total"], max(task.get("completed", 0), int(live.get("t_env") or 0)))
    elif live.get("total") == task.get("total"):
        task["completed"] = min(task["total"], max(task.get("completed", 0), int(live.get("completed") or 0)))
        if task["kind"] in ("final", "depth", "readout") and all(task.get(k) is not None for k in
                ("independent_completed", "independent_total", "reused_completed")):
            task["independent_completed"] = max(task["independent_completed"],
                min(task["independent_total"], max(0,task["completed"]-task["reused_completed"])))
    else:
        task["worker_completed"], task["worker_total"] = live.get("completed"), live.get("total")
    task["live"] = True


def _snapshot(root, now):
    issues = []
    metadata = _json(root / "experiment.json", issues)
    if not metadata.get("runtime_estimate"):
        acceptance = _json(root / "implementation_acceptance.json", issues)
        estimate = acceptance.get("matrix_smoke_runtime", {}).get("runtime_estimate")
        if estimate:
            metadata["runtime_estimate"] = estimate
    inventory = _json(root / "inventory.json", issues)
    tasks = _tasks(metadata, inventory)
    registered = {row["id"] for row in inventory.get("tasks", []) if row.get("id")}
    try:
        inventory_time = (root / "inventory.json").stat().st_mtime
    except OSError:
        inventory_time = math.inf
    schedulers = []
    for kind in ("train", "eval"):
        for env in ("had", "smacv2"):
            path = root / f"scheduler.{kind}.{env}.json"
            scheduler = _json(path, issues)
            if not scheduler:
                continue
            scheduler = dict(scheduler, kind=kind, env=env, alive=_alive(scheduler.get("pid")), age=_age(path, now))
            schedulers.append(scheduler)
            try:
                snapshot_time = path.stat().st_mtime
            except OSError:
                snapshot_time = 0.
            published = _stamp(scheduler.get("updated_at"))
            newer = snapshot_time > inventory_time and (published is None or published > inventory_time)
            eligible = scheduler["alive"] or scheduler.get("status") in ("completed", "stopped")
            if newer and eligible:
                for identity in scheduler.get("completed_ids", []):
                    if (not isinstance(identity, str) or identity not in registered
                            or tasks[identity].get("env") != env
                            or (tasks[identity].get("kind") == "train") != (kind == "train")):
                        issues.append(f"{path.name}: 忽略未登记或不属于本队列的完成ID {identity}")
                        continue
                    task = tasks[identity]
                    if task.get("status") in _DONE:
                        continue
                    task.update(status="complete", completed=task["total"], completion_source=path.name,
                                completion_source_updated_at=scheduler.get("updated_at"))
                    if task.get("independent_total") is not None:
                        task["independent_completed"] = task["independent_total"]
            if not scheduler["alive"] and scheduler.get("status") in _ACTIVE:
                issues.append(f"{path.name}: 登记为{scheduler.get('status')}，父PID已退出；不当作运行中")
            for failure in scheduler.get("failures", []):
                issues.append(f"{kind}/{env}: " + (str(failure.get("id", "")) + " " + str(failure.get("error", failure.get("reason", failure)))
                              if isinstance(failure, dict) else str(failure)))
            for row in scheduler.get("waiting", []):
                identity = _identity(row, kind, env)
                if identity in tasks and tasks[identity].get("status") not in _DONE:
                    tasks[identity]["queued"] = scheduler["alive"] and row.get("status", "queued") in ("pending", "queued", "resource_wait")
                    for key in ("independent_completed", "independent_total", "reused_completed"):
                        if row.get(key) is not None:
                            tasks[identity][key] = row[key]
                    if row.get("reason") or row.get("detail"):
                        tasks[identity]["detail"] = row.get("reason") or row["detail"]
            for row in scheduler.get("live", []):
                identity = _identity(row, kind, env)
                if identity not in tasks:
                    issues.append(f"调度快照含未登记任务: {identity}")
                elif tasks[identity].get("status") in _DONE:
                    continue
                elif _alive(row.get("pid")):
                    _overlay(tasks[identity], row)
                else:
                    issues.append(f"{identity}: 工作PID已退出，快照尚未更新")
    for task in tasks.values():
        if task["kind"] != "train":
            continue
        directory = _directory(root, task)
        resource = _json(directory / "resource.json", issues)
        for key in ("validation_completed_episodes", "validation_total_episodes",
                    "validation_measured_episodes", "validation_measured_seconds",
                    "training_seconds", "training_steps_per_second"):
            if resource.get(key) is not None:
                task[key] = resource[key]
        if resource and task.get("status") not in _DONE:
            task["completed"] = min(task["total"], max(task.get("completed", 0), int(resource.get("t_env") or 0)))
            if _alive(resource.get("pid")) and resource.get("status") not in _DONE | {"stopped", "failed"}:
                _overlay(task, resource)
            elif resource.get("status") in ("failed", "stopped", "interrupted"):
                task["last_status"] = resource["status"]
            elif resource.get("status") in _ACTIVE and not _alive(resource.get("pid")):
                issues.append(f"{task['id']}: resource登记{resource['status']}但PID已退出；保留可恢复进度")
        if task.get("live"):
            observed = _train_tail(_tail(directory / "console.log"))
            if task.get("current_phase"):
                observed.pop("current_phase", None)
            # The resource snapshot can advance beyond the last printed line.
            observed["t_env"] = max(task.get("completed", 0), observed.get("t_env", 0))
            _overlay(task, observed)
            task["log_age"] = _age(directory / "console.log", now)
    # Old schedulers have no snapshot yet. Only a live, output-matched parent
    # plus a recent console tail can imply evaluation activity.
    eval_lease = _json(root / ".cpu-eval.lock")
    parent_alive = _alive(eval_lease.get("pid")) and Path(eval_lease.get("output", "")).resolve() == root.resolve()
    for env in ("had", "smacv2"):
        path = root / f"eval.{env}.console.log"
        age = _age(path, now)
        observed = _eval_tail(_tail(path))
        for identity, row in observed.items():
            if identity not in tasks or tasks[identity].get("status") in _DONE:
                continue
            if tasks[identity].get("live"):
                _overlay(tasks[identity], row)
            elif parent_alive and age is not None and age < 120 and not any(s["kind"] == "eval" and s["env"] == env for s in schedulers):
                _overlay(tasks[identity], row)
                tasks[identity]["pid_note"] = "日志推测；父进程存活，worker PID未登记"
    bank = _json(root / "probe/collection_complete.json")
    if "trajectories" not in bank:
        # At most 500 public scene names, no gzip payloads or feature files.
        directory = root / "probe/trajectories"
        bank["trajectory_files"] = min(500, sum(1 for _ in directory.glob("scene_*.json.gz"))) if directory.exists() else 0
    return dict(metadata=metadata, inventory=inventory, tasks=list(tasks.values()), schedulers=schedulers,
                issues=list(dict.fromkeys(issues)), inventory_age=_age(root / "inventory.json", now),
                bank=bank)


def _gpu_snapshot():
    """One low-frequency query, restricted to the two authorized devices."""
    try:
        result = subprocess.run(["nvidia-smi", "--id=0,1", "--query-gpu=index,name,utilization.gpu,memory.free,memory.total",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
        rows = []
        for index, name, use, free, total in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if int(index) in (0, 1):
                rows.append(f"GPU{index} {name}: 利用率{use}% / 空闲{float(free)/1024:.1f}/{float(total)/1024:.1f}GiB")
        return "；".join(rows) or "GPU0/1未返回信息"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return f"GPU0/1信息暂不可用：{error}"


def _independent(task, by_id, equivalent):
    total = task.get("independent_total")
    completed = task.get("independent_completed")
    if total is not None and completed is not None:
        return int(completed), int(total)
    if task["kind"] == "final":
        return int(task.get("completed", 0)), int(task["total"])
    if task["kind"] not in ("depth", "readout"):
        return None, None
    total = 4500 if task["kind"] == "depth" or not equivalent else 3900
    reused = task.get("reused_completed")
    if reused is not None:
        return max(0, int(task.get("completed", 0)) - int(reused)), total
    final = by_id.get(f"eval.final.had.regir.s{task['seed']}", {})
    depth = by_id.get(f"eval.depth.had.regir.s{task['seed']}", {})
    if task.get("status") in _DONE:
        return total, total
    if final.get("status") in _DONE and (task["kind"] == "depth" or not equivalent or depth.get("status") in _DONE):
        return max(0, int(task.get("completed", 0)) - (int(task["total"]) - total)), total
    return None, total


def _task_eta(task):
    explicit = task.get("remaining_seconds_range") or task.get("eta_seconds_range")
    if explicit:
        return _range(explicit), "调度实测区间"
    remaining = max(0, task["total"] - task.get("completed", 0))
    if task["kind"] == "train":
        measured = _number(task.get("training_steps_per_second"))
        if measured is not None and measured > 0:
            return f"约{_duration(remaining / measured)}", "累计采集+更新实测；不含验证，验证另列"
        rates = task.get("rate_range")
        if rates and min(rates) > 0:
            return _range((remaining / max(rates), remaining / min(rates))), "近期训练日志吞吐（含已发生验证耗时）"
    value = _number(task.get("remaining_seconds"))
    if value is not None:
        return f"约{_duration(value)}", "调度器按实际已执行回合估计"
    return "暂不可估算", "尚无该任务实际速度"


def _compact(tasks):
    grouped = defaultdict(list)
    for row in tasks:
        grouped[row.get("env", "had"), row["kind"], row.get("method") or "all"].append(row)
    result = []
    for (env, kind, method), rows in sorted(grouped.items()):
        seeds = ",".join(f"s{r['seed']}" for r in rows if r.get("seed") is not None) or "全部"
        reasons = sorted({str(r.get("last_status", "")) + (": " if r.get("last_status") else "") + str(r.get("detail", ""))
                          for r in rows if r.get("detail") or r.get("last_status")})
        result.append(f"  {env}/{_NAMES.get(kind, kind)} {method} [{seeds}]" + (" — " + "; ".join(reasons) if reasons else ""))
    return result or ["  无"]


def _estimate_ranges(estimate, tasks=(), metadata=None):
    """Use registered evidence and remaining work for both terminal views."""
    if not isinstance(estimate, dict):
        return {}
    scaled = {}
    for env in ("had", "smacv2"):
        section = estimate.get(env, {})
        if not isinstance(section, dict):
            continue
        registered = (metadata or {}).get("environments", {}).get(env, {}).get("methods")
        measured = section.get("methods")
        if registered and measured and set(registered) != set(measured):
            # An old four-method estimate cannot be scaled by total steps
            # to masquerade as measured throughput for new architectures.
            continue
        reference = section.get("reference_remaining", {})
        selected = [t for t in tasks if t.get("env") == env]
        training = [t for t in selected if t["kind"] == "train"]
        validation = [t for t in training if env == "smacv2" or t["method"] in
                      (metadata or {}).get("environments", {}).get("had", {}).get("new_methods", _HAD[-4:])]
        remaining = dict(training_steps=sum(max(0, t["total"]-t.get("completed",0)) for t in training),
                         final_episodes=sum(max(0,t["total"]-t.get("completed",0)) for t in selected if t["kind"]=="final"))
        if validation and all(t.get("validation_completed_episodes") is not None for t in validation):
            remaining["validation_episodes"] = sum(max(0, t.get("validation_total_episodes",5000 if env=="had" else 5120)-t["validation_completed_episodes"]) for t in validation)
        for field, reference_key in (("training_seconds","training_steps"), ("validation_seconds","validation_episodes"),
                                     ("evaluation_seconds","final_episodes")):
            values = section.get(field)
            if isinstance(values, (list, tuple)) and len(values)==2 and all(_number(x) is not None for x in values):
                baseline = _number(reference.get(reference_key))
                factor = remaining[reference_key]/baseline if baseline and reference_key in remaining else 1.
                scaled[env,field] = [float(x)*factor for x in values]
        rates = section.get("training_rates", {})
        pending = [t for t in training if t["completed"] < t["total"]]
        if training and rates and all(t["method"] in rates for t in pending):
            cards = max(1, int(estimate.get("assumed_full_gpus", 2)))
            scaled[env, "training_seconds"] = [
                sum((t["total"] - t["completed"]) / rates[t["method"]][i] for t in pending) / cards
                for i in (1, 0)]
        mechanism = section.get("mechanism_seconds")
        if isinstance(mechanism, (list, tuple)) and len(mechanism)==2:
            # Probe fitting and M4 are fixed work, not proportional to episode count.
            pending = any(t["kind"] in ("depth","readout","probe","timing") and t.get("status") not in _DONE for t in selected)
            scaled[env,"mechanism_seconds"] = mechanism if pending else [0,0]
    components = ("training_seconds","validation_seconds","evaluation_seconds","mechanism_seconds")
    for env in ("had","smacv2"):
        if all((env,key) in scaled and all(_number(x) is not None for x in scaled[env,key]) for key in components):
            scaled[env,"stage_seconds"] = [sum(scaled[env,key][i] for key in components) for i in (0,1)]
    if all((env,"stage_seconds") in scaled for env in ("had","smacv2")):
        scaled["overall_seconds"] = [sum(scaled[env,"stage_seconds"][i] for env in ("had","smacv2")) for i in (0,1)]
    return scaled


def _estimate_lines(estimate, tasks=(), metadata=None):
    """Render explicitly registered ranges; do not invent a global multiplier."""
    if not isinstance(estimate, dict) or not estimate:
        return ["  整体：暂不可估算（runtime_estimate尚未登记；未测方法/SMAC不能由HAD外推）"]
    lines = []
    updated = estimate.get("measured_at", estimate.get("updated_at"))
    if updated:
        lines.append(f"  登记估计更新时间：{updated}")
    labels = dict(had="HAD", smacv2="SMAC", training_seconds="训练", validation_seconds="验证",
                  evaluation_seconds="终评", mechanism_seconds="机制", stage_seconds="阶段串行上界", overall_seconds="整体")
    scaled = _estimate_ranges(estimate, tasks, metadata)
    def visit(value, path):
        if not isinstance(value, dict):
            return
        span = value.get("remaining_seconds_range", value.get("seconds_range"))
        if span is None and value.get("lower_seconds") is not None and value.get("upper_seconds") is not None:
            span = (value["lower_seconds"], value["upper_seconds"])
        if span is not None:
            lines.append(f"  {'/'.join(path) or '整体'}：{_range(span)}；依据：{value.get('basis', '下方登记实测与条件')}")
        for key, item in value.items():
            if key == "superseded_four_method_estimate":
                continue
            if isinstance(item, dict):
                visit(item, path + [str(key)])
            elif key.endswith("_seconds") and (item is None or isinstance(item, (list, tuple))):
                identity = (path[0],key) if len(path)==1 else key
                item = scaled.get(identity,item)
                label = "/".join(labels.get(p,p) for p in path+[key])
                lines.append(f"  {label}：{_range(item)}；依据：下方登记实测与条件")
    visit(estimate, [])
    if estimate.get("basis"):
        basis = estimate["basis"]
        lines += ["  登记依据："+str(item) for item in (basis if isinstance(basis,list) else [basis])]
    for env in ("had","smacv2"):
        section = estimate.get(env, {})
        for key in ("evaluation_basis","mechanism_basis"):
            if isinstance(section,dict) and section.get(key):
                lines.append(f"  {labels[env]} {'终评' if key=='evaluation_basis' else '机制'}依据：{section[key]}")
    if estimate.get("had",{}).get("mechanism_basis"):
        lines.append("  机制区间含尚未测得的探针拟合/M4额外耗时假设；该部分不是实测结论。")
    if estimate.get("assumptions"):
        lines.append(f"  条件：{estimate['assumptions']}")
    if not any("依据：" in line for line in lines):
        lines.append("  整体：暂不可估算（登记中尚无可解释的剩余时间区间）")
    lines.append("  训练/验证/终评按已知剩余量相对reference_remaining缩放；缺累计量时保留登记区间。机制含固定探针拟合/计时开销，不按局数机械缩小。")
    lines.append("  整体为各阶段相加的条件区间，CPU可与训练重叠；不是置信区间。未测环节不自动补点估计。")
    return lines


def _render_details(output, *, gpu_status=None, now=None, data=None):
    """Return one plain-text snapshot; useful for --once and captured terminals."""
    root = Path(output)
    now = time.time() if now is None else float(now)
    data = _snapshot(root, now) if data is None else data
    tasks, metadata = data["tasks"], data["metadata"]
    by_id = {t["id"]: t for t in tasks}
    completed = [t for t in tasks if t.get("status") in _DONE]
    live = [t for t in tasks if t.get("live") and t.get("status") not in _DONE]
    queued = [t for t in tasks if t.get("status") not in _DONE and not t.get("live") and
              (t.get("queued") or (not t.get("last_status") and t.get("status") in ("pending", "queued")))]
    waiting = [t for t in tasks if t not in completed and t not in live and t not in queued]
    lines = [f"main0921 实验总览  {datetime.fromtimestamp(now).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
             f"输出：{root.resolve()}",
             f"任务 总数{len(tasks)} | 已完成{len(completed)} | 当前{len(live)} | 剩余{len(tasks)-len(completed)} "
             f"(排队{len(queued)} / 等待依赖或恢复{len(waiting)})",
             "每卡训练上限2路，两卡合计4路；正式评估+机制共享4路。验证属于训练子过程，不重复算任务。"]
    if data["inventory_age"] is None:
        lines.append("任务清单尚未生成：矩阵占位只表示计划，不表示已完成。")
    else:
        lines.append(f"inventory快照距今{_duration(data['inventory_age'])}；活动进度由存活PID/最新小快照补充。")
    promoted = sum(bool(task.get("completion_source")) for task in tasks)
    if promoted:
        lines.append(f"其中{promoted}项新完成来自较新的合格调度快照；面板更新不改变正式报告资格。")
    if gpu_status:
        lines.append(gpu_status)
    lines += ["", "训练进度（按当前实验矩阵）"]
    for env in ("had", "smacv2"):
        selected = [t for t in tasks if t["kind"] == "train" and t["env"] == env]
        done = sum(t.get("status") in _DONE for t in selected)
        progress = sum(min(int(t.get("completed", 0)), int(t["total"])) for t in selected)
        total = sum(int(t["total"]) for t in selected)
        lines.append(f"  {env}: 完成{done}/{len(selected)}项，当前{sum(bool(t.get('live')) for t in selected)}项；"
                     f"物理/联合步 {progress:,}/{total:,}，剩余{max(0,total-progress):,}")
    lines += ["", "正式评估与机制（任务/回合/分析格分开；复用臂不重复计对局）"]
    equivalent = bool(metadata.get("mechanisms", {}).get("read1_equivalence_verified"))
    independent_done, independent_total, unknown = 0, 0, False
    for env, kind in (("had", "final"), ("smacv2", "final"), ("had", "depth"), ("had", "readout"), ("had", "probe"), ("had", "timing")):
        selected = [t for t in tasks if t["kind"] == kind and t["env"] == env]
        count = sum(t.get("status") in _DONE for t in selected)
        coverage = sum(int(t.get("completed", 0)) for t in selected)
        total = sum(int(t["total"]) for t in selected)
        base = f"  {env}/{_NAMES[kind]}: 任务{count}/{len(selected)}；"
        if kind in ("final", "depth", "readout"):
            pairs = [_independent(t, by_id, equivalent) for t in selected]
            actual = None if any(d is None for d, _ in pairs) else sum(d for d, _ in pairs)
            budget = sum(n for _, n in pairs)
            independent_total += budget
            if actual is None:
                unknown = True
            else:
                independent_done += actual
            base += f"覆盖{coverage:,}/{total:,}局；独立对局{'待登记' if actual is None else format(actual, ',')}/{budget:,}"
            if actual is not None:
                base += f"，剩余{budget-actual:,}"
        else:
            base += f"{coverage}/{total} {'checkpoint分析' if kind=='probe' else '成本测量格'}"
        lines.append(base)
    lines.append(f"  独立正式对局合计：{'部分待登记（不将复用量猜入总数）' if unknown else format(independent_done, ',')} / {independent_total:,}")
    bank = data["bank"].get("trajectories")
    bank_label = str(bank) if bank is not None else f"已保存{data['bank'].get('trajectory_files',0)}个场景文件，完成标记待生成"
    lines.append(f"  M3公共轨迹库：{bank_label}/500场，只采集一次；3个checkpoint分析不算1500场。")
    lines += ["", "训练内验证（与正式终评分开）"]
    for env in ("had", "smacv2"):
        selected = [t for t in tasks if t["kind"] == "train" and t["env"] == env and
                    (env == "smacv2" or t["method"] in metadata.get("environments", {}).get("had", {}).get("new_methods", _HAD[-4:]))]
        expected = len(selected) * (5000 if env == "had" else 5120)
        registered = [t for t in selected if t.get("validation_completed_episodes") is not None]
        actual = sum(int(t["validation_completed_episodes"]) for t in registered)
        label = f"{actual:,}" if len(registered) == len(selected) else f"已登记{actual:,}（其余累计未知）"
        lines.append(f"  {env}: 本版本新增计划{expected:,}局；{label}；条件ETA见下方分阶段估计。")
    lines += ["", "当前任务（逐一列出；PID已核实，未登记者明确标注）"]
    if not live:
        lines.append("  无已核实的活动任务")
    for task in sorted(live, key=lambda row: row["id"]):
        eta, basis = _task_eta(task)
        device = f"GPU{task['physical_gpu']}" if task.get("physical_gpu") is not None else "CPU/设备未登记"
        phase = task.get("current_phase", task.get("phase", _NAMES.get(task["kind"], task["kind"])))
        lines.append(f"  {task['id']} | {phase} | {device} | PID={task.get('pid', '未登记')} | "
                     f"{int(task.get('completed',0)):,}/{int(task['total']):,} | 剩余ETA {eta}")
        lines.append(f"    依据：{basis}" + (f"；{task['pid_note']}" if task.get("pid_note") else ""))
        validation_done = _number(task.get("validation_completed_episodes"))
        measured_episodes = _number(task.get("validation_measured_episodes"))
        measured_seconds = _number(task.get("validation_measured_seconds"))
        validation_rate = (measured_episodes / measured_seconds
                           if measured_episodes and measured_episodes > 0 and measured_seconds and measured_seconds > 0 else None)
        if task.get("eval_total"):
            pending = max(0, task["eval_total"] - (task.get("eval_completed") or 0))
            batch_eta = f"约{_duration(pending / validation_rate)}" if validation_rate else "暂不可估算（配对验证速率未测）"
            lines.append(f"    当前训练内验证 {task.get('eval_completed',0)}/{task['eval_total']}局；该批验证ETA {batch_eta}")
        if task["kind"] == "train" and task.get("validation_total_episodes"):
            if validation_rate and validation_done is not None:
                pending = max(0, task["validation_total_episodes"] - validation_done)
                lines.append(f"    全程剩余验证约{_duration(pending / validation_rate)}；依据：配对实测{int(measured_episodes)}局/{_duration(measured_seconds)}，累计覆盖仅用于计算剩余局数")
            else:
                lines.append("    全程剩余验证ETA暂不可估算（缺配对实测局数/耗时或累计覆盖；不使用旧完成数/耗时混算）")
        if task.get("worker_total"):
            lines.append(f"    当前子阶段 {task.get('worker_completed',0)}/{task['worker_total']}（不加到独立对局总量）")
    lines += ["", "排队（全部任务，按方法合并seed）"] + _compact(queued)
    lines += ["", "等待依赖/恢复（全部任务）"] + _compact(waiting)
    lines += ["", "整体与分阶段ETA"] + _estimate_lines(metadata.get("runtime_estimate"), tasks, metadata)
    lines += ["", "调度与异常"]
    for scheduler in data["schedulers"]:
        lines.append(f"  {scheduler['kind']}/{scheduler['env']}: {scheduler.get('status','未知')} / "
                     f"父PID{'存活' if scheduler['alive'] else '已退出'} / 并发上限{scheduler.get('max_concurrent','未登记')} / "
                     f"快照距今{_duration(scheduler['age'])}")
    if not data["schedulers"]:
        lines.append("  尚无调度快照；使用resource+PID和有限console尾，排队顺序待登记。")
    lines += ["  " + issue.replace("\n", " ") for issue in data["issues"]] or ["  未发现小状态文件/PID异常；不代表所有实验已完成。"]
    lines += ["", "只读面板：不扫描结果CSV、不加载模型。Ctrl+C仅关闭面板，不停止实验。"]
    return "\n".join(lines)


def _width(text):
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in str(text))


def _fit(text, width):
    text = str(text)
    if _width(text) <= width:
        return text
    result, used = [], 0
    for char in text:
        size = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if used + size > max(0, width - 1):
            break
        result.append(char)
        used += size
    return "".join(result) + ("…" if width else "")


def _cell(text, width):
    text = _fit(text, width)
    return text + " " * max(0, width - _width(text))


def _short_duration(seconds):
    value = _number(seconds)
    if value is None:
        return "待估"
    value = max(0., value)
    if value >= 86400:
        return f"{value / 86400:.1f}天"
    if value >= 3600:
        return f"{value / 3600:.1f}h"
    if value >= 60:
        return f"{value / 60:.0f}m"
    return f"{value:.0f}s"


def _short_range(values):
    if not isinstance(values, (list, tuple)) or len(values) != 2 or any(_number(x) is None for x in values):
        return "待估"
    low, high = sorted(max(0., float(value)) for value in values)
    unit, scale = ("天", 86400.) if high >= 86400 else ("h", 3600.) if high >= 3600 else ("m", 60.)
    return f"{low / scale:.1f}–{high / scale:.1f}{unit}"


def _short_task_eta(task):
    if task["kind"] == "train":
        rate = _recent_training_rate(task)
        remaining = max(0, task.get("total", 0) - task.get("completed", 0))
        return _short_duration(remaining / rate) if rate and rate > 0 else "—"
    span = task.get("remaining_seconds_range") or task.get("eta_seconds_range")
    if span:
        return _short_range(span)
    return _short_duration(task.get("remaining_seconds"))


def _recent_training_rate(task):
    if (task.get("kind") != "train"
            or task.get("current_phase") in ("validation", "validating")
            or task.get("log_age", math.inf) > 60):
        return None
    return _number(task.get("recent_steps_per_second"))


def _cpu_snapshot():
    """Host CPU use between refreshes; one short sample on first display."""
    try:
        import psutil
        warmed = getattr(_cpu_snapshot, "warmed", False)
        percent = psutil.cpu_percent(interval=None if warmed else .1)
        _cpu_snapshot.warmed = True
        threads = psutil.cpu_count() or os.cpu_count() or 1
        return f"CPU 整机利用率 {percent:.1f}%（约{percent * threads / 100:.1f}/{threads}逻辑核）"
    except (ImportError, OSError):
        return "CPU 整机利用率暂不可读"


def _gpu_lines(gpu_status, live, width, per_gpu=2):
    rows = []
    for device in (0, 1):
        count = sum(task["kind"] == "train" and str(task.get("physical_gpu")) == str(device) for task in live)
        match = re.search(rf"GPU{device} [^:]+: 利用率([^%]+)% / 空闲([\d.]+)/([\d.]+)GiB", gpu_status or "")
        usage = (f"利用率{match[1]}%  显存{float(match[3]) - float(match[2]):.1f}/{float(match[3]):.1f}GiB"
                 if match else "利用率/显存待查询")
        rows.append(f"GPU{device} {usage}  训练{count}/{per_gpu}")
    combined = "  |  ".join(rows)
    return [combined] if _width(combined) <= width else rows


def render_status(output, *, gpu_status=None, cpu_status=None, now=None, details=False, width=100,
                  height=None, page=0):
    """Default to a concise view for live terminals, pipes, and --once alike."""
    root = Path(output)
    now = time.time() if now is None else float(now)
    data = _snapshot(root, now)
    if details:
        return _render_details(root, gpu_status=gpu_status, now=now, data=data)
    width = max(1, int(width))
    tasks, metadata = data["tasks"], data["metadata"]
    completed = [task for task in tasks if task.get("status") in _DONE]
    live = sorted((task for task in tasks if task.get("live") and task.get("status") not in _DONE),
                  key=lambda task: (task["kind"] != "train", str(task.get("physical_gpu", "")), task["id"]))
    stopping = bool(live) and any((root / name).exists() for name in ("stop.request", "eval.stop.request"))
    if stopping:
        state = "正在停止"
    elif live:
        state = "运行中"
    elif len(completed) == len(tasks):
        state = "已完成"
    elif any(scheduler["alive"] for scheduler in data["schedulers"]):
        state = "等待资源/依赖"
    else:
        state = "已暂停"
    lines = [f"main0921  {state}  {datetime.fromtimestamp(now).strftime('%m-%d %H:%M:%S')}",
             f"总任务 {len(tasks)}  完成 {len(completed)}  当前 {len(live)}  剩余 {len(tasks) - len(completed)}"]
    for env, name in (("had", "HAD"), ("smacv2", "SMAC")):
        groups = []
        for kinds, label in ((("train",), "训练"), (("final",), "终评"), (("depth", "readout", "probe", "timing"), "机制")):
            selected = [task for task in tasks if task.get("env") == env and task["kind"] in kinds]
            if selected:
                done = sum(task.get("status") in _DONE for task in selected)
                groups.append(f"{label} {done}/{len(selected)}")
        lines.append(f"{name:<4}  " + "  ".join(groups))
    smac_active = any(t.get("env") == "smacv2" and t["kind"] == "train" for t in live)
    cap_key = "smac_gpu_per_card_max" if smac_active else "gpu_per_card_max"
    lines += _gpu_lines(gpu_status, live, width, int(metadata.get("resources", {}).get(cap_key, 2)))
    cpu_tasks = sum(task["kind"] != "train" and task.get("physical_gpu") is None for task in live)
    lines.append(f"{cpu_status or _cpu_snapshot()}  评估{cpu_tasks}/4")
    lines.append("─" * min(width, 76))
    narrow = width < 60
    device_width, progress_width, speed_width, eta_width = (3, 5, 7, 6) if narrow else (5, 7, 8, 11)
    task_width = max(5, min(33, width - device_width - progress_width - speed_width - eta_width - 4))
    def row(device, task, progress, speed, eta):
        return " ".join((_cell(device, device_width), _cell(task, task_width),
                         _cell(progress, progress_width), _cell(speed, speed_width), _cell(eta, eta_width))).rstrip()
    lines.append(row("卡" if narrow else "设备", "当前任务", "进度", "步/秒", "剩余ETA"))
    names = {"regir": "Full", "regir_r1": "Single", "refil": "REFIL", "b2_qmix_atten": "QMIX-Atten",
             "transfqmix": "TransfQMix", "regir_fixed4": "Fixed4", "regir_untied4": "Untied4", "regir_kv0": "KV0",
             "regir_norefil": "norefil", "regir_nocount": "nocount", "regir_last": "last", "refil_matched": "matched"}
    kind_names = dict(train="训", final="评", depth="深度", readout="读出", probe="探针", timing="计时")
    task_start = len(lines)
    for task in live:
        device = f"{'G' if narrow else 'GPU'}{task['physical_gpu']}" if task.get("physical_gpu") is not None else "CPU"
        env = "H" if task.get("env") == "had" else "S"
        method = names.get(task.get("method"), task.get("method") or "all")
        phase = "验" if task.get("current_phase") in ("validation", "validating") else kind_names.get(task["kind"], task["kind"])
        label = f"{env}/{phase} {method}" + (f" s{task['seed']}" if task.get("seed") is not None else "")
        fraction = min(100., 100. * task.get("completed", 0) / max(1, task.get("total", 0)))
        rate = _recent_training_rate(task)
        speed = ("验证" if phase == "验" else
                 f"{rate:.1f}" if rate is not None else "—")
        lines.append(row(device, label, f"{fraction:.1f}%", speed, _short_task_eta(task)))
    if not live:
        lines.append("  无活动任务；已有进度保留" if state == "已暂停" else "  无活动任务")
    task_end = len(lines)
    estimates = _estimate_ranges(metadata.get("runtime_estimate"), tasks, metadata)
    prefix = "恢复后预计" if state == "已暂停" else "预计剩余"
    review = metadata.get("runtime_estimate", {}).get("deadline_review", {})
    if metadata.get("runtime_estimate", {}).get("status") == "six_method_throughput_pending":
        had = [estimates.get(("had", key)) for key in ("training_seconds", "validation_seconds")]
        had_total = [sum(part[i] for part in had) for i in (0, 1)] if all(had) else None
        lines.append(f"训练剩余：HAD {_short_range(had_total)} | SMAC 待补测（18次训练）")
        lines.append("7天目标待复核：旧12次SMAC估算已停用")
    elif review.get("decision") == "authorized_restart_full_two_gpu":
        training = {}
        for env in ("had", "smacv2"):
            parts = [estimates.get((env, key)) for key in ("training_seconds", "validation_seconds")]
            if all(parts):
                training[env] = [sum(part[i] for part in parts) for i in (0, 1)]
        transition = review.get("had_transition_tail_seconds", [0, 0])
        had_pending = any(task.get("env") == "had" and task.get("status") not in _DONE for task in tasks)
        total = ([sum(training[env][i] for env in training) + (transition[i] if had_pending else 0)
                  for i in (0, 1)] if len(training) == 2 else None)
        lines.append(f"{prefix}训练 {_short_range(total)} | 目标{review.get('deadline_days', 7)}天（两卡全速假设）")
        lines.append(f"含验证：HAD {_short_range(training.get('had'))} | SMAC {_short_range(training.get('smacv2'))}")
        deadline_at = _stamp(review.get("deadline_at"))
        remaining_deadline = max(0, deadline_at - now) if deadline_at else review.get('deadline_days', 7) * 86400
        if total and total[1] > remaining_deadline:
            lines.append("工期区间跨过目标：尚不能保证7天内训练完")
    else:
        lines.append(f"{prefix} {_short_range(estimates.get('overall_seconds'))}  （共享资源等待另计）")
        lines.append(f"HAD {_short_range(estimates.get(('had', 'stage_seconds')))}  |  SMAC {_short_range(estimates.get(('smacv2', 'stage_seconds')))}")
    if state == "已暂停" and review.get("decision") == "paused_deadline_not_supported":
        lines.append(f"{review.get('deadline_days', 5)}天目标：当前估计不满足，保持暂停")
    if data["issues"]:
        lines.append(f"提示：{len(data['issues'])}项状态异常，使用 --details 查看")
    lines.append("训练步/秒与ETA：近60秒，不含验证；Ctrl+C 仅关闭面板")
    if height is not None and len(lines) > max(1, int(height)):
        height = max(1, int(height))
        slots = height - (len(lines) - (task_end - task_start)) - 1
        if slots >= 1:
            count = task_end - task_start
            pages = math.ceil(count / slots)
            current = int(page) % pages
            first = current * slots
            shown = lines[task_start + first:min(task_end, task_start + first + slots)]
            lines = (lines[:task_start] + shown
                     + [f"任务 {first + 1}–{min(count, first + slots)}/{count} · {current + 1}/{pages}页（自动轮换）"]
                     + lines[task_end:])
        else:
            lines = lines[:height - 1] + ["窗口过矮，请增大高度或用 --once 查看"]
    return "\n".join(_fit(line, width) for line in lines)


def watch_status(output, interval=5, once=False, details=False):
    """TTY refreshes in place; redirected output prints one compact snapshot."""
    interval = max(1., float(interval))
    tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    overwrite = tty and not once
    gpu_status, last_gpu = None, -math.inf
    frame = 0
    try:
        if overwrite:
            # Use the alternate screen so refreshes never accumulate in the
            # user's scrollback; restore their original screen on exit.
            sys.stdout.write("\x1b[?1049h\x1b[?25l")
            sys.stdout.flush()
        while True:
            tick = time.monotonic()
            if tick - last_gpu >= interval:
                gpu_status, last_gpu = _gpu_snapshot(), tick
            size = shutil.get_terminal_size((100, 24))
            available = max(1, size.lines - 1)
            panel = render_status(output, gpu_status=gpu_status, cpu_status=_cpu_snapshot(),
                                  details=details, width=max(1, size.columns - 1),
                                  height=available if overwrite else None, page=frame // 2)
            if overwrite:
                lines = panel.splitlines()
                if len(lines) > available:
                    lines = lines[:max(0, available - 1)] + ["… 请增大终端高度以显示完整面板"]
                sys.stdout.write("\x1b[H" + "\r\n".join("\x1b[2K" + line for line in lines) + "\x1b[J")
            else:
                sys.stdout.write(panel + "\n")
            sys.stdout.flush()
            frame += 1
            if once or not tty:
                return
            time.sleep(max(.1, interval - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if overwrite:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()

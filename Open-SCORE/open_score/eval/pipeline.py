"""main0923 pipeline (section 5.2): trial -> branch gate -> common + branch experiments.

`run` is the long-lived main loop started by run_all.sh. Every 60 s it scans the
artifact inventory, advances the stage, runs the gate when its inputs are
complete, and writes queue.{env}.json. Four scheduler processes (HAD/SMAC
training, HAD/SMAC evaluation) read those queues; the loop restarts a crashed
scheduler at most three times per hour. All other subcommands only read or
edit small state files and return immediately.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import experiment as X

STAGES = ("prepare", "trial", "gate", "branch", "done")
LOOP_SECONDS = 60
REPO = Path(__file__).resolve().parents[3]
PY = "/home/dell/anaconda3/envs/saileron/bin/python"
SMAC_PY = "/data3/dell/Saileron/envs/saileron-smac/bin/python"


def _now():
    return time.time()


def _stamp(value=None):
    return datetime.fromtimestamp(value or _now()).astimezone().isoformat(timespec="seconds")


def state_path(out):
    return Path(out) / "pipeline.json"


def load_state(out):
    state = X.read_json(state_path(out), None)
    if not isinstance(state, dict):
        state = dict(stage="prepare", branch=None, decided_by=None, decided_at=None, cut=[], cut_now=[],
                     intent_ready=False, abandoned=[], parallel_since=None, smac_paused=False,
                     restarts={}, errors=[], created_at=_stamp())
    return state


def save_state(out, state):
    state["updated_at"] = _stamp()
    X.atomic_json(state_path(out), state)


def control_path(out):
    return Path(out) / "control.json"


def load_control(out):
    """User edits (cut, abandoned, events). Only CLI commands write it; the loop only reads it."""
    value = X.read_json(control_path(out), None)
    return value if isinstance(value, dict) else dict(cut=[], cut_now=[], abandoned=[], events=[])


def save_control(out, control):
    control["updated_at"] = _stamp()
    X.atomic_json(control_path(out), control)


def merge_control(out, state):
    control = load_control(out)
    for key in ("cut", "cut_now", "abandoned"):
        state[key] = list(control.get(key, []))
    state["user_events"] = control.get("events", [])[-20:]


def decision_dir(out):
    return Path(out) / "decision"


def branch_record(out):
    return X.read_json(decision_dir(out) / "branch.json")


# ----------------------------------------------------------------------------
# Task classification
# ----------------------------------------------------------------------------

def _failure_count(out, env, method, seed):
    value = X.read_json(X.run_directory(out, method, seed, env) / "task_failure.json", {}) or {}
    return int(value.get("count", 0))


def _started(out, env, method, seed):
    from open_score.algos import resume_files
    return any(p.exists() for p in resume_files(X.run_directory(out, method, seed, env)))


def _exhausted(out, task):
    return task["kind"] == "train" and _failure_count(out, task["env"], task["method"], task["seed"]) > 2


def _accepted(out):
    had = X.accepted_had_methods(out)
    smac = set()
    for method in X.SMAC_METHODS:
        try:
            X.require_smac_method_acceptance(out, methods_required=(method,))
            smac.add(method)
        except (ValueError, OSError, KeyError):
            pass
    return had, smac


def classify(out, inventory, state):
    """Derive stage facts from the inventory."""
    tasks = inventory["tasks"]
    intent = X.CANDIDATES["B"]
    trial_train = [t for t in tasks if t["env"] == "had" and t["kind"] == "train" and "trial" in t["segments"]]
    trial_eval = [t for t in tasks if t["env"] == "had" and t["kind"] != "train" and "trial" in t["segments"]]
    counted = [t for t in trial_train if state["intent_ready"] or t["method"] != intent]
    ended = [t for t in counted if t["status"] == "complete" or _exhausted(out, t)]
    b_train = [t for t in trial_train if t["method"] == intent]
    b_eval = [t for t in trial_eval if t["method"] == intent]
    rest = lambda rows: [t for t in rows if t["method"] != intent]
    return dict(
        trial_training_done=len(ended) == len(counted),
        trial_training_done_without_b=all(t["status"] == "complete" or _exhausted(out, t) for t in rest(trial_train)),
        core_complete=(all(t["status"] == "complete" for t in rest(trial_train))
                       and all(t["status"] == "complete" for t in rest(trial_eval))),
        b_complete=(all(t["status"] == "complete" for t in b_train) and all(t["status"] == "complete" for t in b_eval)),
        b_failed=any(_exhausted(out, t) for t in b_train),
        core_failed=[t["id"] for t in rest(trial_train) if _exhausted(out, t)])


def _segment_open(segment, facts, state, env):
    if segment == "trial":
        return True
    if segment == "common":
        return env == "smacv2" or facts["trial_training_done"]
    if segment.startswith("branch:"):
        return state.get("branch") == segment.split(":", 1)[1]
    return False


def _prefetch(task, state):
    branches = [s for s in task["segments"] if s.startswith("branch:")]
    return state["stage"] == "gate" and state.get("branch") is None and len(branches) >= 2


def _cut(task, state):
    segments = {s: p for s, p in task["segments"].items() if s not in ("trial", "imported")}
    if not segments or not state.get("cut"):
        return False
    return all(p in state["cut"] for p in segments.values())


def plan(out, inventory, state, facts):
    """queue.{env}.json contents for both environments."""
    had_ok, smac_ok = _accepted(out)
    intent = X.CANDIDATES["B"]
    queues = {}
    for env in ("had", "smacv2"):
        accepted = had_ok if env == "had" else smac_ok
        train, evals = [], []
        for task in inventory["tasks"]:
            if task["env"] != env or "imported" in task["segments"]:
                continue
            if task["method"] not in accepted:
                continue
            if task["method"] == intent and env == "had" and not state["intent_ready"]:
                continue
            open_ = any(_segment_open(s, facts, state, env) for s in task["segments"]) or _prefetch(task, state)
            if not open_ or task["status"] == "complete":
                continue
            if _cut(task, state):
                # A cut keeps an already-started training unless it was cut with --now.
                started = task["kind"] == "train" and _started(out, env, task["method"], task["seed"])
                stop_now = any(p in state.get("cut_now", []) for p in task["segments"].values())
                if not started or stop_now:
                    continue
            if task["kind"] == "train":
                if _exhausted(out, task):
                    continue
                train.append(task)
            else:
                evals.append(task["id"])
        train.sort(key=lambda t: (t["rank"], t["order"]))
        per_gpu = 3 if env == "had" else (2 if facts["trial_training_done"] else 1)
        pause = []
        if env == "smacv2" and state.get("smac_paused") and not facts["trial_training_done"]:
            pause = [t["id"] for t in train]
            train = []
        queues[env] = dict(train=[dict(id=t["id"], method=t["method"], seed=t["seed"]) for t in train],
                           eval=evals, per_gpu=per_gpu, pause=pause, closed=state["stage"] == "done")
    return queues


def write_queues(out, queues):
    changed = False
    for env, value in queues.items():
        path = Path(out) / f"queue.{env}.json"
        previous = X.read_json(path, {}) or {}
        if {k: v for k, v in previous.items() if k != "updated_at"} != value:
            X.atomic_json(path, dict(value, updated_at=_stamp()))
            changed = True
    return changed


# ----------------------------------------------------------------------------
# Speed guard and gate
# ----------------------------------------------------------------------------

def _scheduler(out, kind, env):
    return X.read_json(Path(out) / f"scheduler.{kind}.{env}.json", {}) or {}


def speed_guard(out, state, facts, manifest):
    if facts["trial_training_done"] or state.get("smac_paused"):
        return
    trial = {m for m, _ in X.TRIAL_ORDER}
    had = [r for r in _scheduler(out, "train", "had").get("live", []) if r.get("method") in trial]
    smac = _scheduler(out, "train", "smacv2").get("live", [])
    if not had or not smac:
        return
    if state.get("parallel_since") is None:
        state["parallel_since"] = _now()
        return
    guard = manifest["resources"]["speed_guard"]
    if _now() - state["parallel_since"] < guard["after_hours"] * 3600:
        return
    ratios = []
    for row in had:
        reference = X.reference_speed(manifest, row["method"])
        speed = row.get("training_steps_per_second")
        if reference and speed:
            ratios.append(float(speed) / reference)
    if ratios and sum(ratios) / len(ratios) < guard["ratio"]:
        state["smac_paused"] = True
        state.setdefault("events", []).append(dict(at=_stamp(), event="smac_paused",
            detail=f"HAD trial speed {sum(ratios) / len(ratios):.2f}x main0921 (< {guard['ratio']})"))


def maybe_gate(out, state, facts):
    """Run the gate when its inputs are complete (or IAR's waiting period has ended)."""
    from . import branch_gate
    if state["stage"] != "trial" or not facts["core_complete"]:
        return
    deadline = datetime.fromisoformat(X.GATE["intent_wait_until"]).timestamp()
    include_b = state["intent_ready"] and facts["b_complete"]
    if not include_b and _now() < deadline and not facts["b_failed"]:
        return
    state["gate_without_b"] = not include_b
    try:
        result = branch_gate.decide(out, include_b=include_b)
    except Exception as error:
        import traceback
        state["gate_error"] = dict(at=_stamp(), error=f"{type(error).__name__}: {error}")
        print(f"{_stamp()} gate failed (retried next loop): {error}\n{traceback.format_exc()}", flush=True)
        return
    state.pop("gate_error", None)
    state["gate"] = dict(mode=result["mode"], choice=result["choice"], reasons=result["reasons"],
                         computed_at=result["computed_at"], include_b=include_b)
    record = branch_record(out)
    if record:
        state.update(stage="branch", branch=record["branch"], decided_by=record["mode"],
                     decided_at=record.get("decided_at"))
    else:
        state["stage"] = "gate"
        print(f"{_stamp()} 等待选择主方法：bash run_all.sh select A|B|C", flush=True)


# ----------------------------------------------------------------------------
# Scheduler children
# ----------------------------------------------------------------------------

def child_commands(out):
    out = str(out)
    base_env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                    NUMEXPR_NUM_THREADS="1", PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1",
                    CUDA_VISIBLE_DEVICES="0,1")
    smac_env = dict(base_env, PYTHONNOUSERSITE="1", SC2PATH="/data3/dell/Saileron/envs/StarCraftII")
    train, evaluate = str(REPO / "Open-SCORE/scripts/train.py"), str(REPO / "Open-SCORE/scripts/eval.py")
    return {
        "train.had": ([PY, "-u", train, "--profile", X.PROFILE, "--stage", "train", "--env", "had",
                       "--devices", "0,1", "--per-gpu", "3", "--queue", f"{out}/queue.had.json",
                       "--output", out], base_env, f"{out}/train.had.console.log"),
        "train.smacv2": ([SMAC_PY, "-u", train, "--profile", X.PROFILE, "--stage", "train", "--env", "smacv2",
                          "--devices", "0,1", "--per-gpu", "2", "--queue", f"{out}/queue.smacv2.json",
                          "--output", out], smac_env, f"{out}/train.smacv2.console.log"),
        "eval.had": ([PY, "-u", evaluate, "--profile", X.PROFILE, "--stage", "eval", "--env", "had",
                      "--device", "auto", "--devices", "0,1", "--gpu-workers-per-device", "2",
                      "--final-shards", "96", "--max-concurrent", "32", "--queue", f"{out}/queue.had.json",
                      "--output", out], base_env, f"{out}/eval.had.console.log"),
        "eval.smacv2": ([SMAC_PY, "-u", evaluate, "--profile", X.PROFILE, "--stage", "eval", "--env", "smacv2",
                         "--only", "final,depth", "--device", "auto", "--devices", "0,1",
                         "--gpu-workers-per-device", "2", "--final-shards", "32", "--max-concurrent", "16",
                         "--queue", f"{out}/queue.smacv2.json", "--output", out], smac_env,
                        f"{out}/eval.smacv2.console.log"),
    }


class Children:
    def __init__(self, out, state):
        self.out, self.state, self.procs = Path(out), state, {}

    def ensure(self, stopping):
        commands = child_commands(self.out)
        for name, (argv, env, log) in commands.items():
            proc = self.procs.get(name)
            if proc is not None and proc.poll() is None:
                continue
            if proc is not None:
                code = proc.returncode
                self.procs.pop(name)
                closed = (X.read_json(self.out / f"queue.{name.split('.')[1]}.json", {}) or {}).get("closed")
                if stopping or (closed and code == 0):
                    continue
                history = [t for t in self.state["restarts"].get(name, []) if _now() - t < 3600]
                history.append(_now())
                self.state["restarts"][name] = history
                print(f"{_stamp()} scheduler {name} exited with {code}; restart {len(history)}/3 this hour", flush=True)
                if len(history) > 3:
                    self.state["errors"].append(dict(at=_stamp(), error=f"{name} restarted more than 3 times in one hour"))
                    raise RuntimeError(f"{name} restarted more than 3 times in one hour; see {log}")
            if stopping:
                continue
            closed = (X.read_json(self.out / f"queue.{name.split('.')[1]}.json", {}) or {}).get("closed")
            if closed and self.state["stage"] == "done":
                continue
            with open(log, "ab") as handle:
                self.procs[name] = subprocess.Popen(argv, env=env, cwd=str(REPO), stdin=subprocess.DEVNULL,
                                                    stdout=handle, stderr=subprocess.STDOUT)
            print(f"{_stamp()} started scheduler {name} pid={self.procs[name].pid}", flush=True)

    def running(self):
        return {name: p.pid for name, p in self.procs.items() if p.poll() is None}

    def wait_all(self):
        last = 0
        while any(p.poll() is None for p in self.procs.values()):
            if _now() - last > 60:
                print(f"{_stamp()} waiting for schedulers to save and exit: {sorted(self.running())}", flush=True)
                last = _now()
            time.sleep(1)


def request_stop(out):
    from open_score.utils.resources import write_stop
    write_stop(Path(out) / "stop.request", "Stop after the current complete sampling/learning batch.")
    write_stop(Path(out) / "eval.stop.request", "Stop after the current complete evaluation episode.")


def run(out):
    out = Path(out).resolve()
    manifest = X.initialize(out)
    state = load_state(out)
    if state["stage"] == "prepare":
        state["stage"] = "trial"
    state["intent_ready"] = X.CANDIDATES["B"] in X.accepted_had_methods(out)
    for path in (out / "stop.request", out / "eval.stop.request"):
        path.unlink(missing_ok=True)
    save_state(out, state)
    stopping = {"flag": False}

    def on_signal(signum, frame):
        stopping["flag"] = True
        print(f"{_stamp()} stop requested: schedulers save resume points and exit", flush=True)
        request_stop(out)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    children = Children(out, state)
    index = X.EpisodeIndex(out)
    print(f"{_stamp()} pipeline {X.PROFILE}: stage={state['stage']} branch={state.get('branch')} "
          f"intent_ready={state['intent_ready']}", flush=True)
    consecutive = 0
    try:
        while not stopping["flag"]:
            try:
                finished = iteration(out, state, index, children, stopping, manifest)
                consecutive = 0
            except RuntimeError:
                raise
            except Exception as error:
                import traceback
                consecutive += 1
                state["errors"] = (state.get("errors", []) + [
                    dict(at=_stamp(), error=f"{type(error).__name__}: {error}")])[-20:]
                print(f"{_stamp()} pipeline iteration failed ({consecutive}/10): {error}\n"
                      f"{traceback.format_exc()}", flush=True)
                if consecutive >= 10:
                    raise
                finished = False
            if finished:
                break
            for _ in range(LOOP_SECONDS):
                if stopping["flag"]:
                    break
                time.sleep(1)
    except BaseException as error:
        state["errors"].append(dict(at=_stamp(), error=f"{type(error).__name__}: {error}"))
        print(f"{_stamp()} pipeline error: {type(error).__name__}: {error}", flush=True)
        request_stop(out)
        raise
    finally:
        children.ensure(True)
        children.wait_all()
        save_state(out, state)
    return 0


def iteration(out, state, index, children, stopping, manifest):
    """One pass of the main loop; returns True when the experiment is finished."""
    merge_control(out, state)
    state["intent_ready"] = X.CANDIDATES["B"] in X.accepted_had_methods(out)
    record = branch_record(out)
    if record and record.get("branch") != state.get("branch"):
        state.update(branch=record["branch"], decided_by=record["mode"], decided_at=record.get("decided_at"))
        if state["stage"] == "gate":
            state["stage"] = "branch"
    inventory = X.scan(out, index=index)
    facts = classify(out, inventory, state)
    maybe_gate(out, state, facts)
    speed_guard(out, state, facts, manifest)
    queues = plan(out, inventory, state, facts)
    if state["stage"] == "branch" and not any(q["train"] or q["eval"] for q in queues.values()):
        if all(not _scheduler(out, k, e).get("live") for k in ("train", "eval") for e in ("had", "smacv2")):
            state["stage"] = "done"
            queues = plan(out, inventory, state, facts)
            print(f"{_stamp()} all runnable tasks complete; queues closed", flush=True)
    state["facts"] = facts
    write_queues(out, queues)
    save_state(out, state)
    children.ensure(stopping["flag"])
    return state["stage"] == "done" and not children.running()


# ----------------------------------------------------------------------------
# User commands
# ----------------------------------------------------------------------------

def select(out, branch, force=False):
    out = Path(out)
    if branch not in X.LADDER:
        raise SystemExit("select A|B|C")
    state = load_state(out)
    current = branch_record(out)
    if current and current.get("branch") != branch and not force:
        raise SystemExit(f"主方法已由 {current['mode']} 选定为 {current['branch']}；改选请加 --force")
    if current and current.get("branch") == branch:
        print(f"主方法已是 {branch}（{current['mode']}）")
        return
    if current and force:
        inventory = X.read_json(out / "inventory.json", {}) or {}
        old = f"branch:{current['branch']}"
        abandoned = [t["id"] for t in inventory.get("tasks", []) if old in t.get("segments", {})
                     and f"branch:{branch}" not in t["segments"] and t["status"] != "complete"]
        control = load_control(out)
        control.setdefault("abandoned", []).extend(abandoned)
        control.setdefault("events", []).append(dict(at=_stamp(), event="reselect", old=current["branch"],
                                                     new=branch, abandoned=abandoned))
        save_control(out, control)
    preselected = state["stage"] in ("prepare", "trial")
    X.atomic_json(decision_dir(out) / "branch.json", dict(branch=branch, mode="user", decided_at=_stamp(),
                                                          preselected=preselected, gate_version=X.GATE["version"]))
    (decision_dir(out) / "DECISION_REQUIRED").unlink(missing_ok=True)
    print(f"主方法：{branch}（user{'，试训结束前的预先选择，闸门仍会生成判定报告' if preselected else ''}）")


def cut(out, priority, now=False):
    if priority not in ("P0", "P1", "P2"):
        raise SystemExit("cut P0|P1|P2")
    control = load_control(out)
    if priority not in control.setdefault("cut", []):
        control["cut"].append(priority)
    if now and priority not in control.setdefault("cut_now", []):
        control["cut_now"].append(priority)
    control.setdefault("events", []).append(dict(at=_stamp(), event="cut", priority=priority, now=now))
    save_control(out, control)
    print(f"已取消 {priority} 中尚未开始的任务" + ("，并停止正在运行的" if now else ""))


def retry(out, method, seed, env="had"):
    path = X.run_directory(out, method, int(seed), env) / "task_failure.json"
    if not path.exists():
        print(f"{env}/{method}/s{seed} 没有失败记录")
        return
    value = X.read_json(path, {})
    archive = path.with_name(f"task_failure.{int(_now())}.json")
    shutil.move(str(path), str(archive))
    print(f"已清除 {env}/{method}/s{seed} 的失败标记（原记录 {value.get('count')} 次，存于 {archive.name}）；主循环会重新排队")


def decide(out, preview=False):
    from . import branch_gate
    state = load_state(out)
    include_b = state.get("intent_ready", False) and not state.get("gate_without_b", False)
    result = branch_gate.decide(out, include_b=include_b, preview=preview)
    print(f"判定：{result['mode']} {result['choice'] or ''}\n" + "\n".join(result["reasons"]))
    print(f"报告：{decision_dir(out) / ('分支判定_预览.md' if preview else '分支判定.md')}")


# ----------------------------------------------------------------------------
# Status panel
# ----------------------------------------------------------------------------

def _hms(seconds):
    seconds = int(max(0, seconds or 0))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def render_status(out):
    out = Path(out)
    state = load_state(out)
    inventory = X.read_json(out / "inventory.json", {}) or {}
    tasks = inventory.get("tasks", [])
    lines = []
    if (decision_dir(out) / "DECISION_REQUIRED").exists() and not branch_record(out):
        lines.append("\x1b[1;31m等待选择主方法：bash run_all.sh select A|B|C（见 decision/分支判定.md）\x1b[0m")
    record = branch_record(out)
    lines.append(f"{X.PROFILE}  阶段={state.get('stage')}  主方法={record['branch'] + '（' + record['mode'] + '）' if record else '未定'}"
                 f"  IAR就绪={state.get('intent_ready')}  更新于 {state.get('updated_at', '-')}")
    if state.get("gate_error"):
        lines.append(f"\x1b[31m闸门计算出错 {state['gate_error']['at']}: {state['gate_error']['error']}\x1b[0m")
    if state.get("gate"):
        lines.append(f"闸门：{state['gate']['mode']} {state['gate'].get('choice') or ''}  " + "；".join(state['gate']['reasons']))
    if state.get("cut"):
        lines.append(f"已取消优先级：{state['cut']}" + (f"（立即停止：{state['cut_now']}）" if state.get("cut_now") else ""))
    if state.get("smac_paused"):
        lines.append("速度守护：SMAC 队列已暂停，直到试训训练结束")
    for error in state.get("errors", [])[-3:]:
        lines.append(f"\x1b[31m错误 {error['at']}: {error['error']}\x1b[0m")

    def bucket(task):
        segs = task.get("segments", {})
        if "imported" in segs:
            return "导入"
        if "trial" in segs:
            return "试训"
        parts = []
        for s, p in segs.items():
            parts.append(("公共" if s == "common" else s.replace("branch:", "分支")) + ("-机制" if p == "M" else f"-{p}"))
        return "/".join(sorted(parts))

    groups = {}
    for task in tasks:
        key = (task["env"], bucket(task), "训练" if task["kind"] == "train" else "评估")
        row = groups.setdefault(key, dict(done=0, total=0, failed=0))
        row["total"] += 1
        row["done"] += task["status"] == "complete"
        if task["kind"] == "train" and _failure_count(out, task["env"], task["method"], task["seed"]) > 2:
            row["failed"] += 1
    lines.append("")
    lines.append(f"{'环境':<7}{'段':<28}{'类型':<5}{'完成':>9}{'失败':>6}")
    for (env, seg, kind), row in sorted(groups.items()):
        lines.append(f"{env:<8}{seg:<28}{kind:<5}{row['done']:>5}/{row['total']:<4}{row['failed'] or '':>5}")
    for env in ("had", "smacv2"):
        sched = _scheduler(out, "train", env)
        live = sched.get("live", [])
        lines.append("")
        lines.append(f"[{env} 训练] 调度器 {sched.get('status', '未运行')}  每卡≤{sched.get('per_gpu', '-')}  "
                     f"运行 {len(live)}  排队 {len(sched.get('waiting', []))}")
        for row in sorted(live, key=lambda r: r["id"]):
            total = row.get("total") or X.budget(env)
            t_env = int(row.get("t_env") or 0)
            sps = row.get("training_steps_per_second")
            spikes = int(row.get("grad_spikes") or 0)
            flag = "\x1b[33m" if spikes > 5 else ""
            extra = ""
            if row.get("intent_loss") is not None:
                extra = f"  intent={row['intent_loss']:.3f}/acc={row.get('intent_acc') or 0:.2f}"
            val = row.get("latest_validation_D")
            lines.append(f"  {flag}{row['method']:<28} s{row['seed']}  GPU{row.get('physical_gpu')}  "
                         f"{t_env:>9,}/{total:,}  {(sps or 0):5.1f}步/秒  尖峰={spikes}"
                         f"  验证D={'-' if val is None else f'{val:.3f}'}{extra}\x1b[0m")
        sched = _scheduler(out, "eval", env)
        lines.append(f"[{env} 评估] 调度器 {sched.get('status', '未运行')}  进程 {len(sched.get('live', []))}"
                     f"  已完成任务 {sched.get('completed', 0)}/{sched.get('total', 0)}")
    abandoned = state.get("abandoned", [])
    if abandoned:
        lines.append(f"\n已废弃（改选分支）：{len(abandoned)} 个任务")
    return "\n".join(lines)


def status(out, once=False, interval=10):
    try:
        while True:
            text = render_status(out)
            if once:
                print(text)
                return
            sys.stdout.write("\x1b[H\x1b[2J" + text + "\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "select", "cut", "retry", "decide"))
    parser.add_argument("args", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--now", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--env", default="had", choices=("had", "smacv2"))
    options = parser.parse_args()
    out = options.output.resolve()
    if options.command == "run":
        sys.exit(run(out))
    if options.command == "status":
        status(out, once=options.once)
    elif options.command == "select":
        select(out, (options.args or [None])[0], force=options.force)
    elif options.command == "cut":
        cut(out, (options.args or [None])[0], now=options.now)
    elif options.command == "retry":
        if len(options.args) != 2:
            raise SystemExit("retry METHOD SEED [--env smacv2]")
        retry(out, options.args[0], options.args[1], env=options.env)
    elif options.command == "decide":
        decide(out, preview=options.preview)


if __name__ == "__main__":
    main()

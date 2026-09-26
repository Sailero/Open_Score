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
import csv
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import unicodedata

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
    had = X.accepted_had_methods(out) | set(X.IMPORTED_BASELINES)
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
            if not open_ or task["status"] in ("complete", "skipped"):
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
# Status panel (compact first screen; details after; refresh overwrites)
# ----------------------------------------------------------------------------

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_STAGE = dict(prepare="准备", trial="试训中", gate="等待选分支", branch="分支实验", done="已完成")
_METHOD = {
    "regir_kv0_intent_sg": "候选B 意图推断（IAR）",
    "regir_kv0_intent_noaux_sg": "IAR 去掉意图监督",
    "regir_kv0_intent_nomem": "IAR 去掉记忆查询",
    "regir_intent_sg": "IAR 未锚定",
    "regir_kv0_intent_norefil_sg": "IAR 去掉REFIL部件",
    "regir_kv0_sg": "候选A 锚定关系（ARR）",
    "regir_kv0_nomem": "ARR 去掉记忆查询",
    "regir_kv0_norefil_sg": "ARR 去掉REFIL部件",
    "regir_kv0_fixed4_sg": "ARR 固定4轮",
    "regir_kv0_nocount_sg": "ARR 去掉数量编码",
    "regir_r1_sg": "候选C 单轮关系（RCR）",
    "regir_r1_nomem": "RCR 去掉记忆查询",
    "regir_r1_norefil_sg": "RCR 去掉REFIL部件",
    "regir_r1_nocount_sg": "RCR 去掉数量编码",
    "regir_r0_sg": "地基对照 只读原始实体（R0）",
    "regir_sg": "未锚定循环（Looped）",
    "regir_untied4_sg": "逐层独立（Untied4）",
    "regir_fixed4_sg": "未锚定固定4轮",
    "refil": "基线 REFIL",
    "refil_matched": "基线 REFIL-matched",
    "b2_qmix_atten": "基线 QMIX-Atten",
    "dcg": "基线 DCG",
    "spectra": "基线 SPECTra",
    "alma": "基线 ALMA",
    "transfqmix": "基线 TransfQMix",
}
_KIND = dict(train="训练", final="终评", gate_depth="闸门辅助评估", dup="重复追击评估",
             coverage="覆盖探针", dynamics="动态/漂移", deep_rounds="延长轮次",
             global_probe="全局探针", readout_attention="读出注意力",
             intent_accuracy="意图准确率", timing="M4时延",
             depth="执行深度评估", readout="读出干预", r1deploy="单轮部署评估",
             intent_intervention="意图干预")
_PHASE = dict(collecting="正在采集对局", updating="正在更新网络",
              validation="正在做训练内验证", validating="正在做训练内验证",
              training="正在训练", evaluating="正在评估", starting="正在启动")


def _plain(text):
    return _ANSI.sub("", str(text))


def _width(text):
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in _plain(text))


def _fit(text, width):
    text = str(text)
    if _width(text) <= width:
        return text
    result, used = [], 0
    for char in _plain(text):
        size = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if used + size > max(0, width - 1):
            break
        result.append(char)
        used += size
    return "".join(result) + ("…" if width else "")


def _cell(text, width):
    text = _fit(text, width)
    return text + " " * max(0, width - _width(text))


def _short_eta(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    seconds = max(0, float(seconds))
    if seconds >= 86400:
        return f"{seconds / 86400:.1f}天"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"


def _frac(done, total):
    return f"{done}/{total}" if total else "—"


def _count(tasks, pred):
    selected = [t for t in tasks if pred(t)]
    done = sum(t.get("status") == "complete" for t in selected)
    return done, len(selected)


def _family(task):
    segs = task.get("segments") or {}
    if "imported" in segs:
        return "import"
    if "trial" in segs:
        return "trial"
    if "common" in segs:
        return "common"
    branches = [s.split(":", 1)[1] for s in segs if s.startswith("branch:")]
    if branches:
        return "branch:" + (branches[0] if len(branches) == 1 else "+".join(sorted(branches)))
    return "other"


def _method_name(method):
    return _METHOD.get(method, method or "?")


def _short_gpu_name(name):
    name = name.replace("NVIDIA ", "").replace("Tesla ", "")
    for suffix in (" Workstation Edition", " Founders Edition", " Graphics"):
        name = name.replace(suffix, "")
    name = re.sub(r"\s+Blackwell.*", "", name)
    return re.sub(r"\s+", " ", name).strip() or "GPU"


def _join_bits(prefix, bits, width):
    if not bits:
        return []
    indent = " " * _width(prefix)
    lines, current = [], prefix
    for index, bit in enumerate(bits):
        piece = bit if index == 0 else "  ·  " + bit
        if current == prefix or _width(current) + _width(piece) <= width:
            current += piece
        else:
            lines.append(current)
            current = indent + bit
    lines.append(current)
    return lines


def _gpu_rows():
    query = ("index,name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0,1", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as error:
        return [dict(index=i, error=str(error)) for i in (0, 1)]
    rows = []
    for cells in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(cells) < 8:
            continue
        index, name, use, used, total, temp, draw, limit = cells[:8]
        try:
            rows.append(dict(
                index=int(index),
                name=_short_gpu_name(name),
                util=float(use), used=float(used) / 1024, total=float(total) / 1024,
                temp=None if temp.upper() == "N/A" else float(temp),
                draw=None if draw.upper() == "N/A" else float(draw),
                limit=None if limit.upper() == "N/A" else float(limit)))
        except ValueError:
            rows.append(dict(index=int(index) if index.isdigit() else len(rows), error="读数失败"))
    return rows or [dict(index=i, error="未返回信息") for i in (0, 1)]


def _cpu_row():
    info = dict(percent=None, threads=os.cpu_count() or 1, load=None, mem_used=None, mem_total=None)
    try:
        info["load"] = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass
    try:
        import psutil
        warmed = getattr(_cpu_row, "warmed", False)
        info["percent"] = psutil.cpu_percent(interval=None if warmed else 0.1)
        _cpu_row.warmed = True
        info["threads"] = psutil.cpu_count() or info["threads"]
        memory = psutil.virtual_memory()
        info["mem_used"] = memory.used / (1024 ** 3)
        info["mem_total"] = memory.total / (1024 ** 3)
    except (ImportError, OSError):
        try:
            raw = Path("/proc/meminfo").read_text()
            values = {line.split(":")[0]: float(line.split()[1]) for line in raw.splitlines() if ":" in line}
            total = values.get("MemTotal")
            available = values.get("MemAvailable", values.get("MemFree", 0))
            if total:
                info["mem_total"] = total / (1024 ** 2)
                info["mem_used"] = (total - available) / (1024 ** 2)
        except (OSError, ValueError, KeyError):
            pass
    return info


def _live_jobs(out):
    jobs = []
    for env in ("had", "smacv2"):
        for kind in ("train", "eval"):
            sched = _scheduler(out, kind, env)
            for row in sched.get("live") or []:
                job = dict(row)
                job.setdefault("env", env)
                job.setdefault("kind", kind if kind == "train" else job.get("kind") or "final")
                jobs.append(job)
    jobs.sort(key=lambda r: (r.get("kind") != "train", str(r.get("physical_gpu", "")), r.get("id", "")))
    return jobs


def _detail_bits(tasks, env, family, label):
    train = _count(tasks, lambda t: t["env"] == env and t["kind"] == "train" and _family(t) == family)
    evals = _count(tasks, lambda t: t["env"] == env and t["kind"] != "train" and _family(t) == family)
    if train[1] == 0 and evals[1] == 0:
        return None
    return f"{label} 训{_frac(*train)} 评{_frac(*evals)}"


def _job_bucket(job, tasks):
    kind = job.get("kind") or "train"
    matched = None
    for task in tasks:
        if (task.get("env") == job.get("env") and task.get("method") == job.get("method")
                and task.get("seed") == job.get("seed")):
            if kind == "train" and task.get("kind") == "train":
                matched = task
                break
            if task.get("kind") == kind:
                matched = task
                break
    family = _family(matched) if matched else None
    trial = {m for m, _ in X.TRIAL_ORDER}
    if family == "trial" or (job.get("env") == "had" and job.get("method") in trial):
        return "trial_train" if kind == "train" else "trial_eval"
    if job.get("env") == "smacv2" and (family in (None, "common") or (family or "").startswith("branch:")):
        if family == "common" or job.get("method") in {m for m, _ in X.COMMON.get("smacv2", ())}:
            return "smac_common"
        return "branch"
    if job.get("env") == "had" and family == "common":
        return "had_common"
    if family and family.startswith("branch:"):
        return "branch"
    if kind not in ("train", "final", "gate_depth"):
        return "mech"
    return "other"


def _job_block(job):
    device = f"GPU{job['physical_gpu']}" if job.get("physical_gpu") is not None else "CPU"
    env = "拦截任务" if job.get("env") == "had" else "星际争霸"
    kind = _KIND.get(job.get("kind") or "train", job.get("kind") or "训练")
    name = _method_name(job.get("method"))
    seed = "" if job.get("seed") is None else f" 种子{job['seed']}"
    phase = _PHASE.get(job.get("current_phase") or job.get("phase"), "")
    if job.get("kind") == "train" or job.get("t_env") is not None:
        done = int(job.get("t_env") or job.get("completed") or 0)
        total = int(job.get("total") or job.get("budget_steps") or 0) or 1
        rate = job.get("training_steps_per_second")
        speed = f"{rate:.1f}步/秒" if rate else "速度未知"
        eta = _short_eta((total - done) / rate) if rate and rate > 0 else "—"
    else:
        done = int(job.get("completed") or 0)
        total = int(job.get("total") or 0) or 1
        speed, eta = "评估中", _short_eta(job.get("remaining_seconds"))
    pct = min(100.0, 100.0 * done / max(1, total))
    title = f"    ▶ {device}  {name}{seed} · {env}{kind}"
    bits = [f"进度{pct:.1f}%（{done/10000:.1f}万/{total/10000:.0f}万步）" if total >= 1000
            else f"进度{pct:.1f}%（{done}/{total}）", speed]
    if phase:
        bits.append(phase)
    if job.get("latest_validation_D") is not None:
        bits.append(f"最近验证损伤{job['latest_validation_D']:.3f}")
    bits.append(f"约剩{eta}")
    spikes = job.get("grad_spikes")
    if spikes and int(spikes) > 0:
        bits.append(f"梯度尖峰{int(spikes)}次")
    detail = f"        {'  '.join(bits)}"
    extra = []
    if job.get("intent_loss") is not None:
        acc = job.get("intent_acc") or 0
        extra.append(f"        附带：预测可见队友下一步动作  损失{job['intent_loss']:.2f}  "
                     f"准确率{acc:.0%}（随机约11%，现在还几乎没学会）")
    lines = [title, detail, *extra]
    if spikes is not None and int(spikes) > 5:
        return [f"\x1b[33m{line}\x1b[0m" for line in lines]
    return lines


def _queue_line(out, env, pred, width):
    waiting = [row for row in (_scheduler(out, "train", env).get("waiting") or []) if pred(row)]
    if not waiting:
        return []
    grouped = {}
    for row in waiting:
        grouped.setdefault(row.get("method"), []).append(row.get("seed"))
    names = []
    for method, seeds in grouped.items():
        seeds = sorted(s for s in seeds if s is not None)
        if not seeds:
            names.append(_method_name(method))
        elif len(seeds) == 1:
            names.append(f"{_method_name(method)} 种子{seeds[0]}")
        elif seeds == list(range(seeds[0], seeds[-1] + 1)):
            names.append(f"{_method_name(method)} 种子{seeds[0]}–{seeds[-1]}")
        else:
            names.append(f"{_method_name(method)} 种子{','.join(str(s) for s in seeds)}")
    return _join_bits(f"    排队还有{len(waiting)}个：", names, width)


_DIM, _YEL, _RED, _RST = "\x1b[2m", "\x1b[33m", "\x1b[1;31m", "\x1b[0m"
_FREEZE_AT = datetime.fromisoformat("2026-09-29T22:00:00+08:00")
_CUT_AT = datetime.fromisoformat("2026-09-28T18:00:00+08:00")
_UNWRITTEN = (
    "覆盖探针M-V", "动态/漂移M-B/J", "延长轮次M-F", "全局探针M-I",
    "读出注意力M-K", "意图面板M-T1", "M4时延", "论文出图",
    "Read-1登记", "TransfQMix作者仓核实",
)


def _parse_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).astimezone()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _clock(when):
    if when is None:
        return "—"
    return when.astimezone().strftime("%m-%d %H:%M")


def _slots(out, env):
    sched = _scheduler(out, "train", env)
    per = sched.get("per_gpu") or (3 if env == "had" else 1)
    devices = sched.get("devices") or [0, 1]
    return max(1, int(per) * max(1, len(devices)))


def _live_index(jobs):
    return {(j.get("env"), j.get("method"), j.get("seed")): j
            for j in jobs if j.get("kind") == "train"}


def _method_rate(method, live_rate, reference):
    if live_rate:
        return live_rate
    if method in (reference or {}):
        return reference[method]
    base = method[:-3] if str(method).endswith("_sg") else method
    if base in (reference or {}):
        return reference[base]
    if method and "untied4" in method:
        return (reference or {}).get("regir_untied4")
    if method and "fixed4" in method:
        return (reference or {}).get("regir_fixed4")
    return (reference or {}).get("regir_kv0")


def _train_eta(tasks, jobs, pred, slots, reference, default_rate=18.0):
    live = _live_index(jobs)
    remaining, rates, pending = 0.0, [], 0
    for task in tasks:
        if task.get("kind") != "train" or task.get("status") == "complete" or not pred(task):
            continue
        pending += 1
        job = live.get((task.get("env"), task.get("method"), task.get("seed")))
        total = float(task.get("total") or X.budget(task.get("env", "had")))
        done = float((job or {}).get("t_env") or task.get("completed") or 0)
        remaining += max(0.0, total - done)
        rate = _method_rate(task.get("method"), (job or {}).get("training_steps_per_second"), reference)
        if rate:
            rates.append(float(rate))
    if pending == 0 or remaining <= 0:
        return 0.0, 0
    avg = (sum(rates) / len(rates)) if rates else default_rate
    if not avg or avg <= 0:
        return None, pending
    return remaining / (avg * max(1, slots)), pending


def _eval_frac(tasks, pred):
    return _count(tasks, lambda t: t.get("kind") != "train" and pred(t))


def _your_move(out, state, tasks, record, now):
    if (decision_dir(out) / "DECISION_REQUIRED").exists() and not record:
        return True, "select A|B|C  ← decision/分支判定.md"
    if state.get("gate_error"):
        return True, "gate error · decide 重试"
    failed = [t for t in tasks if t.get("kind") == "train"
              and _failure_count(out, t["env"], t["method"], t["seed"]) > 2]
    if failed:
        row = failed[0]
        extra = " --env smacv2" if row.get("env") == "smacv2" else ""
        return True, f"retry {row['method']} {row['seed']}{extra}  ({len(failed)} failed)"
    if now >= _FREEZE_AT:
        return True, "freeze 已过 · 核对数字/出图"
    if now >= _CUT_AT and state.get("stage") in ("branch", "gate"):
        return True, "考虑 cut P2"
    if state.get("stage") == "done":
        return True, "pipeline done · plot/G10 收尾"
    if record:
        return False, f"idle · branch {record.get('branch')} 自动跑 · 9/28 cut?"
    return False, "idle · gate ~9/26"


def _stage_lines(out, state, tasks, jobs, record, now, width):
    manifest = X.read_json(out / "experiment.json", {}) or {}
    reference = manifest.get("reference_speed") or {}
    facts = state.get("facts") or {}
    branch = (record or {}).get("branch")
    trial_done, trial_total = _count(tasks, lambda t: t["kind"] == "train" and _family(t) == "trial")
    trial_eval = _eval_frac(tasks, lambda t: _family(t) == "trial")
    smac_common = _count(tasks, lambda t: t["env"] == "smacv2" and t["kind"] == "train" and _family(t) == "common")
    had_common = _count(tasks, lambda t: t["env"] == "had" and t["kind"] == "train" and _family(t) == "common")
    had_common_eval = _eval_frac(tasks, lambda t: t["env"] == "had" and _family(t) == "common")
    smac_common_eval = _eval_frac(tasks, lambda t: t["env"] == "smacv2" and _family(t) == "common")
    trial_eta, _ = _train_eta(
        tasks, jobs, lambda t: t["env"] == "had" and _family(t) == "trial",
        _slots(out, "had"), reference)
    smac_eta, _ = _train_eta(
        tasks, jobs, lambda t: t["env"] == "smacv2" and _family(t) == "common",
        _slots(out, "smacv2"), reference, default_rate=18.0)
    had_common_eta, _ = _train_eta(
        tasks, jobs, lambda t: t["env"] == "had" and _family(t) == "common",
        _slots(out, "had"), reference)
    live_trial = sum(1 for j in jobs if j.get("env") == "had" and j.get("kind") == "train"
                     and j.get("method") in {m for m, _ in X.TRIAL_ORDER})
    live_smac = sum(1 for j in jobs if j.get("env") == "smacv2" and j.get("kind") == "train")
    prepare_done = bool(manifest.get("imports")) and state.get("stage") != "prepare"
    gate_waiting = (decision_dir(out) / "DECISION_REQUIRED").exists() and not record
    core_complete = bool(facts.get("core_complete"))
    trial_train_done = bool(facts.get("trial_training_done")) or (trial_total and trial_done == trial_total)
    deadline = _parse_time(X.GATE.get("intent_wait_until"))

    def eta_text(seconds, prefix="约剩"):
        if seconds is None:
            return "时长待估"
        if seconds <= 0:
            return "可结束"
        finish = now + timedelta(seconds=seconds)
        return f"{prefix}{_short_eta(seconds)}（{_clock(finish)}）"

    grouped = {}
    for job in jobs:
        grouped.setdefault(_job_bucket(job, tasks), []).append(job)
    trial_methods = {m for m, _ in X.TRIAL_ORDER}
    common_smac = {m for m, _ in X.COMMON.get("smacv2", ())}
    rows = []

    def add(number, mark, title, status, note, purpose=None, bucket=None, queue=None, paint=None):
        line = f"{number} {mark} {_cell(title, 10)} {_cell(status, 10)} {note}"
        if paint:
            line = f"{paint}{line}{_RST}"
        rows.append(line)
        if purpose:
            rows.append(f"    {purpose}")
        for job in grouped.get(bucket) or []:
            rows.extend(_job_block(job))
        if queue:
            rows.extend(queue)

    add("1", "✓" if prepare_done else "▶", "准备",
        "已完成" if prepare_done else "进行中",
        "导入旧结果、验收新方法、启动流水线")
    if trial_train_done:
        add("2", "✓", "试训训练", "已完成",
            f"{trial_done}/{trial_total}  拦截任务上的三个候选+地基",
            "用来比较谁更好，之后据此选主方法。")
    else:
        add("2", "▶", "试训训练", "进行中",
            f"{trial_done}/{trial_total}完成  {live_trial}个正在跑  {eta_text(trial_eta)}",
            "拦截任务上训练：候选B意图推断、候选A锚定关系、候选C单轮关系、地基R0。",
            bucket="trial_train",
            queue=_queue_line(out, "had", lambda row: row.get("method") in trial_methods, width))
    if trial_eval[1] and trial_eval[0] == trial_eval[1]:
        add("3", "✓", "试训终评", "已完成", f"{trial_eval[0]}/{trial_eval[1]}",
            bucket="trial_eval")
    elif trial_eval[0] or grouped.get("trial_eval"):
        add("3", "▶", "试训终评", "进行中",
            f"{trial_eval[0]}/{trial_eval[1]}  每个模型训完立刻评，不用你",
            bucket="trial_eval")
    else:
        add("3", "·", "试训终评", "未开始",
            "每个试训模型训完就自动评24个场景，大约再加数小时")
    if smac_common[1] and smac_common[0] == smac_common[1]:
        add("4", "✓", "公共SMAC", "已完成",
            f"训{_frac(*smac_common)}  评{_frac(*smac_common_eval)}")
    else:
        smac_note = eta_text(smac_eta)
        if trial_eta and smac_eta and smac_eta > trial_eta:
            smac_note += "；试训结束后每卡会升到2路"
        add("4", "▶" if live_smac or smac_common[0] else "·", "公共SMAC",
            "进行中" if live_smac or smac_common[0] else "排队中",
            f"{_frac(*smac_common)}完成  {live_smac}个正在跑  {smac_note}",
            "与第2段同时开始。星际争霸上的基线和单轮关系，和选哪个主方法无关。",
            bucket="smac_common",
            queue=_queue_line(out, "smacv2", lambda row: row.get("method") in common_smac, width))
    if record:
        add("5", "✓", "分支闸门", "已选定",
            f"主方法 {record.get('branch')}（{record.get('mode')}）")
    elif gate_waiting:
        add("5", "!", "分支闸门", "要你选",
            "读 decision/分支判定.md，然后 bash run_all.sh select A|B|C",
            paint=_RED)
    elif core_complete:
        add("5", "▶", "分支闸门", "计算中", "终评已齐，正在出判定报告")
    else:
        until = f"候选B最晚等到{_clock(deadline)}。" if deadline else ""
        add("5", "!", "分支闸门", "还没到",
            f"自动选或要你选。{until}你每多等1小时，后面的特有实验就晚1小时。",
            paint=_YEL)
    if trial_train_done and had_common[1]:
        if had_common[0] == had_common[1]:
            add("6", "✓", "公共HAD", "已完成", f"训{_frac(*had_common)}  评{_frac(*had_common_eval)}")
        else:
            add("6", "▶", "公共HAD", "进行中",
                f"训{_frac(*had_common)}  {eta_text(had_common_eta)}",
                "拦截任务上的未锚定循环和逐层独立对照。",
                bucket="had_common")
    else:
        add("6", "·", "公共HAD", "未开始",
            "等第2段18个训练全部结束后自动开，不用你")
    if branch:
        own = f"branch:{branch}"
        special_tr = _count(tasks, lambda t: t["kind"] == "train" and own in (t.get("segments") or {}))
        special_ev = _eval_frac(tasks, lambda t: own in (t.get("segments") or {}))
        special_eta, _ = _train_eta(
            tasks, jobs, lambda t: t["kind"] == "train" and own in (t.get("segments") or {}),
            _slots(out, "had"), reference)
        if special_tr[1] and special_tr[0] == special_tr[1]:
            add("7", "✓", f"特有{branch}", "已完成", f"训{_frac(*special_tr)}  评{_frac(*special_ev)}")
        else:
            add("7", "▶", f"特有{branch}", "进行中",
                f"训{_frac(*special_tr)}  评{_frac(*special_ev)}  {eta_text(special_eta)}",
                bucket="branch",
                queue=_queue_line(out, "had",
                                 lambda row: own in ((next((t.get("segments") or {} for t in tasks
                                                           if t.get("kind")=="train" and t.get("method")==row.get("method")
                                                           and t.get("seed")==row.get("seed")), {}))),
                                 width))
    else:
        add("7", "·", "分支特有", "未开始",
            "第5段选定 A/B/C 后，自动开该分支的消融训练")
    mech = [t for t in tasks if t.get("kind") not in ("train", "final", "gate_depth")
            and (not branch or f"branch:{branch}" in (t.get("segments") or {}) or _family(t) == "common")]
    mech_done, mech_total = _count(mech, lambda t: True)
    if mech_total and mech_done == mech_total:
        add("8", "✓", "机制对局", "已完成", f"{mech_done}/{mech_total}")
    elif mech_done or grouped.get("mech"):
        add("8", "▶", "机制对局", "进行中", f"{mech_done}/{mech_total}",
            bucket="mech")
    else:
        add("8", "·", "机制对局", "未开始",
            "选分支后自动跑：改执行轮数、只读某一轮、意图替换等；探针挂 HAD eval 队列")
    add("9", "·", "探针/M4", "已接线",
        "覆盖/动态/深度/全局/读出/意图探针 + M4（试训训练结束后有空闲 GPU 再测）。出图仍手跑 plot.py")
    freeze_left = (_FREEZE_AT - now).total_seconds()
    if now >= _FREEZE_AT:
        add("10", "!", "冻结收尾", "已到期", "要你核对数字、写下判定、出图", paint=_RED)
    else:
        add("10", "!", "冻结收尾", "9/29 22:00",
            f"还有{_short_eta(freeze_left)}。到点要你核对数字、写下判定、出图。",
            paint=_YEL)
    return [_fit(line, width) for line in rows]


_ALIAS = {
    "regir_kv0_intent_sg": "IAR", "regir_kv0_intent_noaux_sg": "IAR-noaux",
    "regir_kv0_intent_nomem": "IAR-nomem", "regir_intent_sg": "IAR-loop",
    "regir_kv0_intent_norefil_sg": "IAR-norefil", "regir_kv0_sg": "ARR",
    "regir_kv0_nomem": "ARR-nomem", "regir_kv0_norefil_sg": "ARR-norefil",
    "regir_kv0_fixed4_sg": "ARR-F4", "regir_kv0_nocount_sg": "ARR-nocnt",
    "regir_r1_sg": "RCR", "regir_r1_nomem": "RCR-nomem",
    "regir_r1_norefil_sg": "RCR-norefil", "regir_r1_nocount_sg": "RCR-nocnt",
    "regir_r0_sg": "R0", "regir_sg": "Looped", "regir_untied4_sg": "Untied4",
    "regir_fixed4_sg": "Tied-F4", "refil": "REFIL", "refil_matched": "matched",
    "b2_qmix_atten": "QMIX-A", "dcg": "DCG", "spectra": "SPECTra",
    "alma": "ALMA", "transfqmix": "TFQMix",
}
_PHASE_SHORT = dict(collecting="collect", updating="update", validation="val",
                    validating="val", training="train", evaluating="eval", starting="start")


def _alias(method):
    return _ALIAS.get(method, method or "?")


def _is_train_job(job):
    return (job.get("kind") or "train") == "train"


def _job_eta(job):
    phase = job.get("current_phase") or job.get("phase")
    if _is_train_job(job):
        done = int(job.get("t_env") or job.get("completed") or 0)
        total = int(job.get("total") or job.get("budget_steps") or 0) or 1
        rate = job.get("training_steps_per_second")
        if phase in ("validation", "validating"):
            measured = job.get("validation_measured_episodes") or 0
            seconds = job.get("validation_measured_seconds") or 0
            speed = f"{measured / seconds:.1f}ep/s" if seconds else "val"
        else:
            speed = f"{rate:.1f}/s" if rate else "—"
        return (min(100.0, 100.0 * done / total), speed,
                _short_eta((total - done) / rate) if rate and rate > 0 else "—")
    done = int(job.get("completed") or 0)
    total = int(job.get("total") or 0) or 1
    workers = job.get("_shard_n")
    speed = f"{workers}w" if workers else "—"
    return min(100.0, 100.0 * done / total), speed, _short_eta(job.get("remaining_seconds"))


_WIN = re.compile(r"\bwin=([\d.-]+)")


def _console_win(out, job):
    if job.get("method") is None or job.get("seed") is None:
        return None
    path = X.run_directory(out, job["method"], job["seed"], job.get("env") or "smacv2") / "console.log"
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 4096))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        match = _WIN.search(line)
        if match and match.group(1) != "-":
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _job_score(out, job):
    if not _is_train_job(job):
        done = int(job.get("completed") or 0)
        total = int(job.get("total") or 0)
        bits = [f"{done}/{total}ep"] if total else []
        cpu_n = job.get("_shard_cpu")
        if cpu_n:
            bits.append(f"{cpu_n}cpu")
        return " ".join(bits) or "—"
    env = job.get("env")
    points = job.get("validation_completed_episodes") or 0
    quota = 100 if env == "had" else 128
    n_pts = int(points) // quota if points else 0
    if env == "had":
        value = job.get("latest_validation_D")
        text = f"{value:.2f}" if value is not None else "—"
    else:
        value = job.get("latest_validation_win_rate")
        if value is None:
            value = _console_win(out, job)
        text = f"{value:.0%}" if value is not None else "—"
    if n_pts:
        text += f"#{n_pts}"
    if env == "had" and job.get("intent_acc") is not None and "intent" in str(job.get("method") or ""):
        text += f"/{job['intent_acc']:.0%}"
    if job.get("current_phase") in ("validation", "validating"):
        done = job.get("eval_completed") or 0
        total = job.get("eval_total") or 0
        if total:
            text = f"val {done}/{total}" if text == "—" else f"{text} {done}/{total}"
    return text


def _bucket(task):
    family = _family(task)
    return "branch" if family.startswith("branch:") else family


def _job_kind(job):
    return "train" if _is_train_job(job) else (job.get("kind") or "final")


def _display_jobs(jobs):
    """One row per trainer; eval shards of the same parent collapse to one row."""
    trains = [job for job in jobs if _is_train_job(job)]
    groups = {}
    for job in jobs:
        if _is_train_job(job):
            continue
        key = (job.get("env"), job.get("method"), job.get("seed"),
               job.get("kind") or "final", job.get("parent_id") or job.get("id"))
        groups.setdefault(key, []).append(job)
    collapsed = []
    for rows in groups.values():
        if len(rows) == 1 and not rows[0].get("parent_id"):
            collapsed.append(rows[0])
            continue
        gpus = sorted({str(row.get("physical_gpu")) for row in rows
                       if row.get("physical_gpu") is not None})
        cpu_n = sum(1 for row in rows if row.get("physical_gpu") is None)
        remain = [float(row["remaining_seconds"]) for row in rows
                  if row.get("remaining_seconds") is not None]
        summary = dict(rows[0])
        summary.update(
            completed=sum(int(row.get("completed") or 0) for row in rows),
            total=sum(int(row.get("total") or 0) for row in rows),
            remaining_seconds=max(remain) if remain else None,
            _shard_n=len(rows), _shard_cpu=cpu_n, _shard_gpus=gpus,
            physical_gpu=gpus[0] if len(gpus) == 1 and cpu_n == 0 else None)
        summary.pop("t_env", None)
        collapsed.append(summary)
    return trains + collapsed


def _job_device(job):
    shards = job.get("_shard_n")
    gpus = job.get("_shard_gpus") or []
    cpu_n = job.get("_shard_cpu") or 0
    if shards:
        bits = []
        if gpus:
            bits.append("GPU" + (gpus[0] if len(gpus) == 1 else f"×{len(gpus)}"))
        if cpu_n:
            bits.append(f"CPU×{cpu_n}" if cpu_n > 1 else "CPU")
        return "+".join(bits) or f"×{shards}"
    if job.get("physical_gpu") is not None:
        return f"GPU{job['physical_gpu']}"
    return "CPU"


def _task_match(task, env, method, seed, kind):
    return (task.get("env") == env and task.get("method") == method
            and int(task.get("seed", -2)) == int(seed) and task.get("kind") == kind)


def _eval_overview(out, tasks, jobs):
    """Parent-level evals actually running, plus ready items still in the queue."""
    groups = {}
    for job in jobs:
        if _is_train_job(job):
            continue
        key = (job.get("env"), job.get("method"), int(job.get("seed", -1)), job.get("kind") or "final")
        groups.setdefault(key, []).append(job)
    running = []
    for (env, method, seed, kind), rows in groups.items():
        parent = next((task for task in tasks if _task_match(task, env, method, seed, kind)), None)
        gpu_n = sum(row.get("physical_gpu") is not None for row in rows)
        cpu_n = len(rows) - gpu_n
        remain = [float(row["remaining_seconds"]) for row in rows
                  if row.get("remaining_seconds") is not None]
        done = int((parent or {}).get("completed") or 0)
        if not done:
            done = sum(int(row.get("completed") or 0) for row in rows)
        total = int((parent or {}).get("total") or 0)
        if not total:
            total = sum(int(row.get("total") or 0) for row in rows)
        running.append(dict(env=env, method=method, seed=seed, kind=kind,
                            workers=len(rows), gpu=gpu_n, cpu=cpu_n,
                            completed=done, total=total or 1,
                            eta=max(remain) if remain else None))
    live = set(groups)
    by_id = {task.get("id"): task for task in tasks}
    queued = []
    for env in ("had", "smacv2"):
        for eid in (X.read_json(Path(out) / f"queue.{env}.json", {}) or {}).get("eval") or []:
            task = by_id.get(eid)
            if not task or task.get("status") in ("complete", "skipped") or not task.get("checkpoint"):
                continue
            key = (task.get("env"), task.get("method"), int(task.get("seed", -1)), task.get("kind"))
            if key not in live:
                queued.append(task)
    return running, queued


def _eval_banner(out, tasks, jobs):
    running, queued = _eval_overview(out, tasks, jobs)
    workers = sum(item["workers"] for item in running)
    cpu_n = sum(item["cpu"] for item in running)
    gpu_n = sum(item["gpu"] for item in running)
    if not running:
        return [f"eval  0 running  ·  {workers} workers"]
    where = []
    if gpu_n:
        where.append(f"{gpu_n} GPU")
    if cpu_n:
        where.append(f"{cpu_n} CPU")
    lines = [f"eval  {len(running)} task running  ·  {workers} workers"
             + (f" ({'+'.join(where)})" if where else "")]
    for item in running:
        pct = 100.0 * item["completed"] / item["total"]
        lines.append(
            f"      {_alias(item['method'])} s{item['seed']} {item['kind']}  "
            f"{pct:.0f}%  {item['completed']}/{item['total']}ep  "
            f"{item['workers']}w  {_short_eta(item['eta'])}")
    if queued:
        nxt = queued[0]
        extra = f" +{len(queued) - 1}" if len(queued) > 1 else ""
        lines.append(f"      next {_alias(nxt.get('method'))} s{nxt.get('seed')} {nxt.get('kind')}{extra}")
    return lines


def _task_key(task):
    return (task.get("env"), task.get("method"), task.get("seed"), task.get("kind"))


def _job_key(job):
    return (job.get("env"), job.get("method"), job.get("seed"), _job_kind(job))


def _row_id(task):
    bucket = _bucket(task)
    kind = task.get("kind") or "train"
    env = "HAD" if task.get("env") == "had" else "SMAC"
    if bucket == "import":
        return "import"
    if bucket == "trial":
        if kind == "train":
            return "trial-tr"
        if kind == "gate_depth":
            return "trial-gate"
        return "trial-ev"
    if bucket == "common":
        if kind == "train":
            return f"cmn-{env}-tr"
        if kind == "dup":
            return "cmn-HAD-dup"
        if kind in ("coverage", "dynamics", "deep_rounds", "global_probe",
                    "readout_attention", "intent_accuracy"):
            return "cmn-HAD-probe" if bucket == "common" else "branch-probe"
        if kind == "timing":
            return "cmn-HAD-m4"
        return f"cmn-{env}-ev"
    if bucket == "branch":
        if kind == "train":
            return "branch-tr"
        if kind == "final":
            return "branch-ev"
        if kind in ("coverage", "dynamics", "deep_rounds", "global_probe",
                    "readout_attention", "intent_accuracy"):
            return "branch-probe"
        return "branch-mech"
    return "other"


_ROW_ORDER = (
    "import", "trial-tr", "trial-ev", "trial-gate",
    "cmn-HAD-tr", "cmn-HAD-ev", "cmn-HAD-dup", "cmn-HAD-probe", "cmn-HAD-m4",
    "cmn-SMAC-tr", "cmn-SMAC-ev",
    "branch-tr", "branch-ev", "branch-mech", "branch-probe", "other",
)


def _job_row_id(job, by_key):
    key = _job_key(job)
    if key in by_key:
        return _row_id(by_key[key])
    env = "HAD" if job.get("env") == "had" else "SMAC"
    kind = _job_kind(job)
    if job.get("env") == "had" and job.get("method") in {m for m, _ in X.TRIAL_ORDER}:
        return "trial-tr" if kind == "train" else ("trial-gate" if kind == "gate_depth" else "trial-ev")
    if kind == "train":
        return f"cmn-{env}-tr"
    return f"cmn-{env}-ev"


def _progress_table(tasks, jobs):
    by_key = {_task_key(task): task for task in tasks}
    live = {_job_key(job) for job in jobs}
    stats = {name: dict(done=0, run=0, pend=0, tot=0) for name in _ROW_ORDER}
    for task in tasks:
        row = stats.setdefault(_row_id(task), dict(done=0, run=0, pend=0, tot=0))
        row["tot"] += 1
        if task.get("status") == "complete":
            row["done"] += 1
        elif _task_key(task) in live:
            row["run"] += 1
        else:
            row["pend"] += 1
    unmatched = sum(1 for job in jobs if _job_key(job) not in by_key)
    if unmatched:
        stats.setdefault("other", dict(done=0, run=0, pend=0, tot=0))
        stats["other"]["run"] += unmatched
        stats["other"]["tot"] += unmatched
    names = [name for name in _ROW_ORDER if stats.get(name, {}).get("tot")]
    names += [name for name in stats if name not in _ROW_ORDER and stats[name]["tot"]]
    totals = dict(done=0, run=0, pend=0, tot=0)
    for name in names:
        for key in totals:
            totals[key] += stats[name][key]
    w_cat, w_n = 13, 5

    def cells(cat, row):
        return " ".join((_cell(cat, w_cat),
                         _cell(str(row["done"]), w_n),
                         _cell(str(row["run"]), w_n),
                         _cell(str(row["pend"]), w_n),
                         _cell(str(row["tot"]), w_n))).rstrip()

    lines = [cells("cat", dict(done="done", run="run", pend="pend", tot="tot"))]
    for name in names:
        line = cells(name, stats[name])
        if stats[name]["run"]:
            lines.append(line)
        elif stats[name]["done"] == stats[name]["tot"]:
            lines.append(f"{_DIM}{line}{_RST}")
        else:
            lines.append(line)
    lines.append(cells("ALL", totals))
    return lines, by_key


def render_status(out, width=100, height=None):
    out = Path(out)
    state = load_state(out)
    inventory = X.read_json(out / "inventory.json", {}) or {}
    tasks = inventory.get("tasks", [])
    record = branch_record(out)
    branch = (record or {}).get("branch") or state.get("branch")
    mode = (record or {}).get("mode")
    jobs = _live_jobs(out)
    now = datetime.now().astimezone()
    stopping = any((out / name).exists() for name in ("stop.request", "eval.stop.request"))
    if stopping:
        headline = "stopping"
    elif state.get("stage") == "gate" and (decision_dir(out) / "DECISION_REQUIRED").exists() and not record:
        headline = "wait-select"
    else:
        headline = state.get("stage") or "?"
    need_you, move = _your_move(out, state, tasks, record, now)
    done_n = sum(t.get("status") == "complete" for t in tasks)
    n_train = sum(1 for job in jobs if _is_train_job(job))
    n_eval_proc = sum(1 for job in jobs if not _is_train_job(job))
    n_eval_task = len({(job.get("env"), job.get("method"), job.get("seed"), job.get("kind"))
                       for job in jobs if not _is_train_job(job)})
    live_txt = f"{n_train} train"
    if n_eval_task:
        live_txt += f" + {n_eval_task} eval/{n_eval_proc}w"
    lines = [f"{X.PROFILE}  {headline}  branch={branch + '/' + mode if branch else '?'}  "
             f"{now.strftime('%m-%d %H:%M:%S')}  live {live_txt}  ·  {done_n}/{len(tasks)} done"]
    lines.append(f"{_RED if need_you else _DIM}你现在  {move}{_RST}")
    if state.get("gate_error"):
        lines.append(f"{_RED}gate  {state['gate_error'].get('error')}{_RST}")
    if state.get("smac_paused"):
        lines.append(f"{_YEL}SMAC paused (speed guard){_RST}")
    if state.get("cut"):
        lines.append("cut " + "/".join(state["cut"]))
    for error in (state.get("errors") or [])[-1:]:
        lines.append(f"{_RED}{error.get('error')}{_RST}")

    trial_tr = _count(tasks, lambda t: t["kind"] == "train" and _family(t) == "trial")
    trial_ev = _count(tasks, lambda t: t["kind"] != "train" and _family(t) == "trial")
    facts = state.get("facts") or {}
    if facts.get("trial_training_done") or (trial_tr[1] and trial_tr[0] == trial_tr[1]):
        nxt = "trial-eval → gate" if trial_ev[0] < trial_ev[1] else ("gate" if not record else f"branch {branch}")
    else:
        nxt = "trial-train"
    manifest = X.read_json(out / "experiment.json", {}) or {}
    trial_eta, _ = _train_eta(
        tasks, jobs, lambda t: t["env"] == "had" and _family(t) == "trial",
        _slots(out, "had"), manifest.get("reference_speed") or {})
    lines.append(f"now   {nxt}  ·  trial ETA {_short_eta(trial_eta) if trial_eta else '—'}")
    table, by_key = _progress_table(tasks, jobs)
    lines.extend(table)

    train_jobs = [j for j in jobs if _is_train_job(j)]
    shown = train_jobs
    for gpu in _gpu_rows():
        index = gpu.get("index", "?")
        ours = sum(str(j.get("physical_gpu")) == str(index) for j in train_jobs)
        had_n = sum(str(j.get("physical_gpu")) == str(index) and j.get("env") == "had" for j in train_jobs)
        smac_n = ours - had_n
        cap = f"HAD{had_n}+SMAC{smac_n}" if smac_n and had_n else f"{ours}"
        if gpu.get("error"):
            lines.append(f"GPU{index}  n/a  ours {cap}")
            continue
        bits = [f"GPU{index} {gpu['name']}",
                f"util {gpu['util']:.0f}%",
                f"mem {gpu['used']:.1f}/{gpu['total']:.1f}GiB"]
        if gpu.get("temp") is not None:
            bits.append(f"{gpu['temp']:.0f}°C")
        if gpu.get("draw") is not None:
            bits.append(f"{gpu['draw']:.0f}W" + (f"/{gpu['limit']:.0f}W" if gpu.get("limit") is not None else ""))
        bits.append(f"ours {cap}")
        lines.append("  ".join(bits))
    cpu = _cpu_row()
    ev_n = n_eval_proc
    cpu_bits = ["CPU"]
    if cpu.get("percent") is not None:
        busy = cpu["percent"] * cpu["threads"] / 100
        cpu_bits.append(f"{cpu['percent']:.1f}% ({busy:.1f}/{cpu['threads']})")
    if cpu.get("load") is not None:
        cpu_bits.append(f"load {cpu['load']:.1f}")
    if cpu.get("mem_used") is not None and cpu.get("mem_total") is not None:
        cpu_bits.append(f"ram {cpu['mem_used']:.0f}/{cpu['mem_total']:.0f}GiB")
    cpu_bits.append(f"eval {n_eval_task} task/{ev_n}w" if n_eval_task else "eval 0")
    lines.append("  ".join(cpu_bits))
    lines.extend(_eval_banner(out, tasks, jobs))
    lines.append("─" * min(width, 76))

    narrow = width < 90
    w_dev, w_cat, w_task, w_pct, w_spd, w_ph, w_d, w_eta = (
        (4, 10, 13, 6, 8, 6, 10, 5) if narrow else (5, 12, 16, 6, 8, 7, 12, 6))
    def row(*cells, widths=(w_dev, w_cat, w_task, w_pct, w_spd, w_ph, w_d, w_eta)):
        return " ".join(_cell(c, w) for c, w in zip(cells, widths)).rstrip()
    lines.append(row("GPU", "cat", "task", "pct", "speed", "phase", "D/acc", "ETA"))
    task_start = len(lines)
    for job in shown:
        env = "HAD" if job.get("env") == "had" else "SMAC"
        kind = "tr" if _is_train_job(job) else (job.get("kind") or "ev")[:4]
        label = f"{env} {_alias(job.get('method'))} s{job.get('seed')} {kind}".strip()
        device = _job_device(job)
        pct, speed, eta = _job_eta(job)
        phase = _PHASE_SHORT.get(job.get("current_phase") or job.get("phase"), job.get("current_phase") or "—")
        line = row(device, _job_row_id(job, by_key), label, f"{pct:.1f}%", speed, phase,
                   _job_score(out, job), eta)
        spikes = job.get("grad_spikes")
        if spikes and int(spikes) > 5:
            line = f"{_YEL}{line}{_RST}"
        lines.append(line)
    if not jobs:
        lines.append("  (idle)")
    task_end = len(lines)

    lines.append(f"next  {nxt}  ·  freeze 9/29 22:00  ·  probe+M4 on HAD eval queue")
    lines.append("cat tot=清单任务（终评一条种子算1）  ·  工人数见 eval 行  ·  D=HAD损伤↓")

    if height is not None and len(lines) > max(1, int(height)):
        height = max(1, int(height))
        extra = len(lines) - (task_end - task_start)
        slots = height - extra
        if slots >= 1:
            count = task_end - task_start
            shown = lines[task_start:task_start + slots]
            lines = lines[:task_start] + shown + lines[task_end:]
            if count > slots:
                lines.insert(task_start + len(shown), f"  … {count - slots} more")
        else:
            lines = lines[: height - 1] + ["窗口过矮"]
    return "\n".join(_fit(line, width) for line in lines)


def status(out, once=False, interval=5):
    """TTY refreshes in place on the alternate screen; --once prints one snapshot."""
    interval = max(1.0, float(interval))
    tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    overwrite = tty and not once
    try:
        if overwrite:
            sys.stdout.write("\x1b[?1049h\x1b[?25l")
            sys.stdout.flush()
        while True:
            tick = time.monotonic()
            size = shutil.get_terminal_size((100, 24))
            available = max(1, size.lines - 1)
            text = render_status(out, width=max(1, size.columns - 1),
                                 height=available if overwrite else None)
            if overwrite:
                rows = text.splitlines()
                if len(rows) > available:
                    rows = rows[: max(0, available - 1)] + ["… 窗口过矮，增大高度看完整面板"]
                sys.stdout.write("\x1b[H" + "\r\n".join("\x1b[2K" + row for row in rows) + "\x1b[J")
            else:
                sys.stdout.write(text + "\n")
            sys.stdout.flush()
            if once or not tty:
                return
            time.sleep(max(0.1, interval - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if overwrite:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()


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

#!/usr/bin/env bash
# main0923 single entry point (section 5.1). Installed as $OUT/run_all.sh.
set -Eeuo pipefail
OUT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT="$OUT/run_all.sh"
REPO=${REGIR_REPO:-/data3/dell/Saileron/projects/ReGIR}
cd -- "$REPO"
PY=/home/dell/anaconda3/envs/saileron/bin/python
SMAC_PY=/data3/dell/Saileron/envs/saileron-smac/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SC2PATH=/data3/dell/Saileron/envs/StarCraftII
export PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/data3/dell/Saileron/tmp
export XDG_CACHE_HOME=/data3/dell/Saileron/.cache
PIPE=(env PYTHONPATH="$REPO/Open-SCORE" "$PY" -m open_score.eval.pipeline --output "$OUT")

usage() {
  cat >&2 <<'EOF'
Usage: bash run_all.sh COMMAND
  start                    first launch: import main0921, acceptance, validate, then the pipeline (background)
  resume                   restart the pipeline after stop or any interruption (training resumes from resume.pt)
  stop                     every trainer saves resume.pt after its batch; evaluators finish the current episode
  status [--once]          live panel (Ctrl+C closes the view only)
  decide [--preview]       (re)compute decision/分支判定.md; --preview uses finished seeds and never triggers a branch
  select A|B|C [--force]   choose the main method (--force to override an existing choice)
  cut P0|P1|P2 [--now]     cancel not-yet-started tasks of a priority; --now also stops running ones
  retry METHOD SEED [--env smacv2]   clear a task's failure mark so it is queued again
  prepare                  only the start-time preparation (idempotent)
EOF
  exit 2
}
command=${1:-}
[[ -n "$command" ]] || usage
shift || true

request_stop() {
  "$PY" - "$OUT" <<'STOP' 9>&-
import sys
from pathlib import Path
for name, text in (("stop.request", "Stop after the current complete sampling/learning batch."),
                   ("eval.stop.request", "Stop after the current complete evaluation episode.")):
    Path(sys.argv[1], name).write_text(text + "\n", encoding="utf-8")
STOP
}

# A stale/reused PID must never signal an unrelated process.
wrapper_pid_matches() {
  local candidate=$1 candidate_cwd argument resolved
  local -a arguments=()
  [[ "$candidate" =~ ^[0-9]+$ && -r "/proc/$candidate/cmdline" ]] || return 1
  candidate_cwd=$(readlink -- "/proc/$candidate/cwd") || return 1
  mapfile -d '' -t arguments < "/proc/$candidate/cmdline" || return 1
  for argument in "${arguments[@]}"; do
    [[ "$argument" == */run_all.sh || "$argument" == run_all.sh ]] || continue
    if [[ "$argument" == /* ]]; then
      resolved=$(realpath -e -- "$argument" 2>/dev/null) || continue
    else
      resolved=$(realpath -e -- "$candidate_cwd/$argument" 2>/dev/null) || continue
    fi
    [[ "$resolved" == "$SCRIPT" ]] && return 0
  done
  return 1
}

stop_existing() {
  local owner= attempt
  exec 9> "$OUT/run_all.lock"
  if ! flock -n 9; then
    for attempt in {1..20}; do
      owner=$(cat "$OUT/run_all.pid" 2>/dev/null || true)
      if wrapper_pid_matches "$owner"; then break; fi
      sleep 0.1
    done
    if ! wrapper_pid_matches "$owner"; then
      echo 'Wrapper lock is held, but its owner could not be verified; no process was signalled.' >&2
      return 1
    fi
    echo "Stopping main0923 wrapper PID $owner; waiting for recoverable checkpoints and child exit..."
    kill -TERM -- "$owner"
    flock 9
  fi
  request_stop
  # Worker processes may outlive a killed parent. Wait until this output's
  # OS-held job/eval/scheduler locks are free and no verified parent remains.
  "$PY" - "$OUT" 9>&- <<'WAIT_OLD_QUEUES'
import fcntl, json, os, sys, time
from pathlib import Path
output = Path(sys.argv[1]).resolve()
cache = Path("/data3/dell/Saileron/.cache")
leases = [cache / f"main0923.gpus.{env}.lock" for env in ("had", "smacv2")]
last_notice, quiet_since = 0.0, None

def parent_matches(pid):
    try:
        process = Path("/proc") / str(int(pid))
        if process.joinpath("stat").read_text().split(")", 1)[1].split()[0] == "Z":
            return False
        arguments = [os.fsdecode(v) for v in process.joinpath("cmdline").read_bytes().split(b"\0") if v]
        value = Path(arguments[arguments.index("--output") + 1])
        if not value.is_absolute():
            value = process.joinpath("cwd").resolve() / value
        return value.resolve() == output
    except (OSError, ValueError, IndexError, TypeError):
        return False

while True:
    held, handles = [], []
    paths = list(output.rglob(".job.*.lock")) + list(output.rglob(".eval.*.lock"))
    paths += [p for p in output.glob(".cpu-eval*.lock")]
    try:
        for path in paths:
            try:
                handle = path.open("a+")
                handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(str(path.relative_to(output)))
        for path in [*output.glob("scheduler.*.json"), *leases]:
            try:
                entry = json.loads(path.read_text())
                if path in leases and Path(entry.get("output", "")).resolve() != output:
                    continue
                if parent_matches(entry.get("pid")):
                    held.append("parent PID " + str(entry["pid"]))
            except (OSError, ValueError, TypeError):
                continue
    finally:
        for handle in handles:
            handle.close()
    if not held:
        quiet_since = quiet_since or time.monotonic()
        if time.monotonic() - quiet_since >= 2:
            break
    else:
        quiet_since = None
        if time.monotonic() - last_notice >= 30:
            print(f"Waiting for {len(set(held))} main0923 queue/worker locks to exit safely...", flush=True)
            last_notice = time.monotonic()
    time.sleep(0.2)
WAIT_OLD_QUEUES
}

prepare() {
  if pgrep -f "profile main0921" >/dev/null; then
    echo 'main0921 processes are still running; stop them first (pgrep -fa "profile main0921").' >&2
    exit 1
  fi
  if ! "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/experiment.json')).get('imports') else 1)" 2>/dev/null; then
    echo '[prepare] importing main0921 baselines, legacy rows, probe bank and SMAC scenes'
    "$PY" Open-SCORE/scripts/eval.py --profile main0923 --stage migrate --output "$OUT"
  fi
  if ! "$PY" - "$OUT" <<'CHECK'
import json, sys
sys.path.insert(0, "/data3/dell/Saileron/projects/ReGIR/Open-SCORE")
from open_score.eval import experiment as X
required = set(X.new_had_methods())
sys.exit(0 if required <= X.accepted_had_methods(sys.argv[1]) else 1)
CHECK
  then
    echo '[prepare] HAD acceptance (G4): CPU smoke, readout detach, IAR checks'
    (cd Open-SCORE && "$PY" -m open_score.eval.acceptance0923 --output "$OUT" --env had --processes 24)
  fi
  if ! (cd Open-SCORE && PYTHONNOUSERSITE=1 "$SMAC_PY" -c "
import sys
from open_score.eval import experiment as X
X.require_smac_method_acceptance('$OUT')" 2>/dev/null); then
    echo '[prepare] SMAC acceptance (G3): real SC2 CPU smoke and 20v20 strict load'
    (cd Open-SCORE && CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 "$SMAC_PY" -m open_score.eval.acceptance0923 \
      --output "$OUT" --env smacv2 --processes 12)
  fi
  echo '[prepare] validate'
  "$PY" Open-SCORE/scripts/train.py --profile main0923 --stage validate --env had --output "$OUT"
  PYTHONNOUSERSITE=1 "$SMAC_PY" Open-SCORE/scripts/train.py --profile main0923 --stage validate --env smacv2 --output "$OUT"
}

launch() {
  exec 8> "$OUT/run_all.control.lock"
  if ! flock -n 8; then
    echo 'Another main0923 start/resume/stop command is active; wait for it to finish.' >&2
    exit 75
  fi
  stop_existing
  exec 9>&-
  nohup setsid bash "$SCRIPT" _run </dev/null >> "$OUT/run_all.console.log" 2>&1 8>&- &
  launched=$!
  for attempt in {1..50}; do
    owner=$(cat "$OUT/run_all.pid" 2>/dev/null || true)
    if [[ "$owner" == "$launched" ]] && wrapper_pid_matches "$owner"; then
      echo "main0923 pipeline running: PID $owner"
      echo "Log: $OUT/run_all.console.log    Panel: bash $SCRIPT status"
      exit 0
    fi
    if ! kill -0 "$launched" 2>/dev/null; then
      echo "Pipeline startup failed; see $OUT/run_all.console.log" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "Pipeline PID $launched did not publish startup state; inspect $OUT/run_all.console.log" >&2
  exit 1
}

case "$command" in
  start)
    prepare
    launch ;;
  resume)
    launch ;;
  prepare)
    prepare ;;
  stop)
    exec 8> "$OUT/run_all.control.lock"
    flock -n 8 || { echo 'Another main0923 control command is active.' >&2; exit 75; }
    stop_existing
    echo 'main0923 stopped; recoverable checkpoints retained. Continue with: bash run_all.sh resume' ;;
  status)
    "${PIPE[@]}" status "$@" ;;
  decide)
    "${PIPE[@]}" decide "$@" ;;
  select)
    "${PIPE[@]}" select "$@" ;;
  cut)
    "${PIPE[@]}" cut "$@" ;;
  retry)
    "${PIPE[@]}" retry "$@" ;;
  _run)
    exec 9> "$OUT/run_all.lock"
    if ! flock -n 9; then
      echo 'main0923 wrapper already running; use resume to replace it safely.' >&2
      exit 75
    fi
    printf '%s\n' "$$" > "$OUT/run_all.pid"
    child=
    cleanup() {
      if [[ "$(cat "$OUT/run_all.pid" 2>/dev/null || true)" == "$$" ]]; then rm -f -- "$OUT/run_all.pid"; fi
    }
    forward() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
    trap forward INT TERM
    trap cleanup EXIT
    echo "$(date '+%F %T') main0923 pipeline starting"
    "${PIPE[@]}" run 9>&- &
    child=$!
    result=0
    while kill -0 "$child" 2>/dev/null; do
      wait "$child" || result=$?
    done
    echo "$(date '+%F %T') main0923 pipeline exited ($result)"
    exit "$result" ;;
  *)
    usage ;;
esac

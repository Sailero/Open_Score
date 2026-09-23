#!/usr/bin/env bash
# Authorized staged experiment queue; no seed expansion or budget extension.
set -Eeuo pipefail
OUT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$OUT/../../.." && pwd)
SCRIPT="$OUT/run_all.sh"
cd -- "$REPO"
PY=/home/dell/anaconda3/envs/saileron/bin/python
SMAC_PY=/data3/dell/Saileron/envs/saileron-smac/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SC2PATH=/data3/dell/Saileron/envs/StarCraftII
export PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/data3/dell/Saileron/tmp
export XDG_CACHE_HOME=/data3/dell/Saileron/.cache

if (( $# > 1 )); then
  echo "Usage: bash $SCRIPT [resume|stop]" >&2
  exit 2
fi
command=${1:-resume}
case "$command" in resume|stop|_run) ;; *)
  echo "Usage: bash $SCRIPT [resume|stop]" >&2
  exit 2
esac

request_stop() {
  "$PY" Open-SCORE/scripts/train.py --stage stop --output "$OUT" 9>&- || return
  "$PY" Open-SCORE/scripts/eval.py --stage stop --output "$OUT" 9>&-
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
    echo "Stopping main0921 wrapper PID $owner; waiting for recoverable checkpoints and child exit..."
    kill -TERM -- "$owner"
    flock 9
  fi
  # Also stop a standalone queue for this output, if one exists. Its own queue
  # lock prevents the replacement queue from clearing the stop before it exits.
  request_stop
  # A killed parent may leave worker processes alive. Keep stop files present
  # until this output's OS-held worker/CPU locks and verified parents are gone.
  "$PY" - "$OUT" "$REPO" 9>&- <<'WAIT_OLD_QUEUES'
import fcntl
import json
import os
from pathlib import Path
import sys
import time

output, repo = (Path(value).resolve() for value in sys.argv[1:])
gpu_lease = repo.parent.parent / ".cache/main0921.gpus.lock"
last_notice = 0.0
quiet_since = None

def parent_matches(pid):
    try:
        process = Path("/proc") / str(int(pid))
        if process.joinpath("stat").read_text().split(")", 1)[1].split()[0] == "Z":
            return False
        arguments = process.joinpath("cmdline").read_bytes().split(b"\0")
        arguments = [os.fsdecode(value) for value in arguments if value]
        index = arguments.index("--output") + 1
        value = Path(arguments[index])
        if not value.is_absolute():
            value = process.joinpath("cwd").resolve() / value
        return value.resolve() == output
    except (OSError, ValueError, IndexError, TypeError):
        return False

while True:
    held, handles = [], []
    paths = list(output.rglob(".job.*.lock")) + list(output.rglob(".eval.*.lock"))
    if (output / ".cpu-eval.lock").exists():
        paths.append(output / ".cpu-eval.lock")
    try:
        for path in paths:
            try:
                handle = path.open("a+")
                handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(str(path.relative_to(output)))
        for path in [*output.glob("scheduler.*.json"), gpu_lease]:
            try:
                entry = json.loads(path.read_text())
                if path == gpu_lease and Path(entry.get("output", "")).resolve() != output:
                    continue
                if parent_matches(entry.get("pid")):
                    held.append("parent PID " + str(entry["pid"]))
            except (OSError, ValueError, TypeError):
                continue
    finally:
        for handle in handles:
            handle.close()
    if not held:
        if quiet_since is None:
            quiet_since = time.monotonic()
        if time.monotonic() - quiet_since >= 2:
            break
    else:
        quiet_since = None
        if time.monotonic() - last_notice >= 30:
            print(f"Waiting for {len(set(held))} previous main0921 queue/worker locks to exit safely...", flush=True)
            last_notice = time.monotonic()
    time.sleep(0.2)
WAIT_OLD_QUEUES
}

if [[ "$command" != _run ]]; then
  # Concurrent resume/stop commands cannot stop each other's new wrapper.
  exec 8> "$OUT/run_all.control.lock"
  if ! flock -n 8; then
    echo 'Another main0921 resume/stop command is active; wait for it to finish.' >&2
    exit 75
  fi
  stop_existing
  if [[ "$command" == stop ]]; then
    echo 'main0921 staged queue stopped; recoverable checkpoints retained.'
    exit 0
  fi
  exec 9>&-
  nohup setsid bash "$SCRIPT" _run </dev/null >> "$OUT/run_all.console.log" 2>&1 8>&- &
  launched=$!
  # Only check startup, never monitor training here. The worker owns its PID file.
  for attempt in {1..50}; do
    owner=$(cat "$OUT/run_all.pid" 2>/dev/null || true)
    if [[ "$owner" == "$launched" ]] && wrapper_pid_matches "$owner"; then
      echo "main0921 resumed: PID $owner; GPUs 0,1 (HAD <=3/card; SMAC <=2/card)."
      echo "Log: $OUT/run_all.console.log"
      exit 0
    fi
    if ! kill -0 "$launched" 2>/dev/null; then
      echo "Wrapper startup failed; see $OUT/run_all.console.log" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "Wrapper PID $launched did not publish startup state; inspect $OUT/run_all.console.log" >&2
  exit 1
fi

exec 9> "$OUT/run_all.lock"
if ! flock -n 9; then
  echo 'main0921 wrapper already running; use the resume command to stop and replace it safely.' >&2
  exit 75
fi
printf '%s\n' "$$" > "$OUT/run_all.pid"
trainer= evaluator= measurer= reporter=
stopping=0 completed=0
stop_children() {
  if (( stopping )); then return 0; fi
  stopping=1
  request_stop || true
  for child in "$trainer" "$evaluator" "$measurer" "$reporter"; do
    if [[ -n "$child" ]]; then wait "$child" || true; fi
  done
}
cleanup() {
  local result=$?
  trap - EXIT
  if (( ! completed )); then stop_children || true; fi
  if [[ "$(cat "$OUT/run_all.pid" 2>/dev/null || true)" == "$$" ]]; then
    rm -f -- "$OUT/run_all.pid"
  fi
  exit "$result"
}
trap 'stop_children; exit 130' INT TERM
trap cleanup EXIT
wait_pair() {
  local result=0
  wait -n "$trainer" "$evaluator" || result=$?
  if (( result )); then
    stop_children
    return "$result"
  fi
  wait "$trainer" || result=$?
  wait "$evaluator" || result=$?
  if (( result )); then stop_children; return "$result"; fi
  trainer= evaluator=
}
echo 'Starting HAD: GPU0/GPU1 training and CPU+GPU final/mechanism queues'
CUDA_VISIBLE_DEVICES=0,1 "$PY" -u Open-SCORE/scripts/train.py \
  --profile main0921 --stage train --group main --env had --steps 1000000 \
  --batch-size-run 8 --per-gpu 3 --devices 0,1 --run train --resume --output "$OUT" \
  >> "$OUT/train.had.console.log" 2>&1 9>&- &
trainer=$!
CUDA_VISIBLE_DEVICES=0,1 "$PY" -u Open-SCORE/scripts/eval.py \
  --profile main0921 --stage eval --env had --only final,depth,readout,probe \
  --device auto --devices 0,1 --gpu-workers-per-device 2 --final-shards 96 \
  --max-concurrent 80 --resume --output "$OUT" \
  >> "$OUT/eval.had.console.log" 2>&1 9>&- &
evaluator=$!
echo "HAD trainer=$trainer evaluator=$evaluator"
wait_pair
echo 'HAD queues complete; measuring M4 on one registered GPU'
CUDA_VISIBLE_DEVICES=0,1 "$PY" -u Open-SCORE/scripts/eval.py \
  --profile main0921 --stage eval --env had --only timing --device cuda \
  --max-concurrent 1 --resume --output "$OUT" >> "$OUT/timing.console.log" 2>&1 9>&- &
measurer=$!
wait "$measurer"
measurer=
echo 'Starting SMACv2: up to two trainers and two admitted evaluators per GPU, with CPU evaluation fallback'
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1 "$SMAC_PY" -u Open-SCORE/scripts/train.py \
  --profile main0921 --stage train --group main --env smacv2 --steps 4000000 \
  --batch-size-run 4 --per-gpu 2 --devices 0,1 --run train --resume --output "$OUT" \
  >> "$OUT/train.smacv2.console.log" 2>&1 9>&- &
trainer=$!
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1 "$SMAC_PY" -u Open-SCORE/scripts/eval.py \
  --profile main0921 --stage eval --env smacv2 --only final --device auto \
  --devices 0,1 --gpu-workers-per-device 2 --final-shards 32 --max-concurrent 16 \
  --resume --output "$OUT" >> "$OUT/eval.smacv2.console.log" 2>&1 9>&- &
evaluator=$!
echo "SMACv2 trainer=$trainer evaluator=$evaluator"
wait_pair
CUDA_VISIBLE_DEVICES='' "$PY" Open-SCORE/scripts/plot.py --output "$OUT" --run train 9>&- &
reporter=$!
wait "$reporter"
reporter=
completed=1
echo 'All registered main0921 training, evaluation and mechanism tasks complete'

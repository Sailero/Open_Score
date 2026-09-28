#!/usr/bin/env bash
# main0928 single entry point. Installed as $OUT/run_all.sh.
# Multi-host: every machine sources envs/local.env and runs the same command
# against the shared outputs/main0928 directory.
set -Eeuo pipefail
OUT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT="$OUT/run_all.sh"
HOST=$(hostname -s 2>/dev/null || hostname)
if [[ -n "${REGIR_ROOT:-}" ]]; then
  REPO=$REGIR_ROOT
elif [[ -f "${OUT}/../../../scripts/setup_farm.sh" ]]; then
  REPO=$(cd -- "$OUT/../../.." && pwd)
else
  REPO=${REGIR_REPO:-}
fi
[[ -n "$REPO" && -d "$REPO/Open-SCORE" ]] || { echo "Set REGIR_ROOT to the cloned ReGIR root." >&2; exit 1; }
cd -- "$REPO"
if [[ -f "$REPO/envs/local.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/envs/local.env"
fi
PY=${PY:-python}
SMAC_PY=${SMAC_PY:-$PY}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1 REGIR_ROOT="$REPO"
PIPE=(env PYTHONPATH="$REPO/Open-SCORE" "$PY" -m open_score.eval.pipeline --output "$OUT")
LOCK_DIR="$OUT/cluster/locks"
mkdir -p "$LOCK_DIR"

usage() {
  cat >&2 <<'EOF'
Usage: bash run_all.sh COMMAND
  start                    unpack imported baselines, validate, then the pipeline (background)
  resume                   restart the pipeline after stop (training resumes from resume.pt)
  stop                     every trainer saves resume.pt after its batch; this host releases claims
  status [--once]          live in-place panel (any machine; Ctrl+C closes the view only)
  retry METHOD SEED [--env smacv2]   clear a task's failure mark so it is queued again
  prepare                  only the start-time preparation (idempotent)
EOF
  exit 2
}
command=${1:-}
[[ -n "$command" ]] || usage
shift || true

request_stop() {
  "$PY" - "$OUT" <<'STOP'
import sys
from pathlib import Path
for name, text in (("stop.request", "Stop after the current complete sampling/learning batch."),
                   ("eval.stop.request", "Stop after the current complete evaluation episode.")):
    Path(sys.argv[1], name).write_text(text + "\n", encoding="utf-8")
STOP
}

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
  exec 9> "$LOCK_DIR/${HOST}.run_all.lock"
  if ! flock -n 9; then
    for attempt in {1..20}; do
      owner=$(cat "$LOCK_DIR/${HOST}.run_all.pid" 2>/dev/null || true)
      if wrapper_pid_matches "$owner"; then break; fi
      sleep 0.1
    done
    if ! wrapper_pid_matches "$owner"; then
      echo 'Wrapper lock is held, but its owner could not be verified; no process was signalled.' >&2
      return 1
    fi
    echo "Stopping main0928 wrapper PID $owner on $HOST; waiting for checkpoints and child exit..."
    kill -TERM -- "$owner"
    flock 9
  fi
  request_stop
}

prepare() {
  echo '[prepare] unpack imported baselines (or copy from a live main0921 tree)'
  PYTHONPATH="$REPO/Open-SCORE" "$PY" Open-SCORE/scripts/eval.py --profile main0928 --stage migrate --output "$OUT"
  echo '[prepare] validate HAD protocol'
  PYTHONPATH="$REPO/Open-SCORE" "$PY" Open-SCORE/scripts/train.py --profile main0928 --stage validate --env had --output "$OUT" || true
  if [[ -n "${SC2PATH:-}" && -d "${SC2PATH}/Versions" ]]; then
    echo '[prepare] validate SMAC protocol'
    PYTHONNOUSERSITE=1 PYTHONPATH="$REPO/Open-SCORE" "$SMAC_PY" Open-SCORE/scripts/train.py \
      --profile main0928 --stage validate --env smacv2 --output "$OUT" || true
  else
    echo '[prepare] no SC2; SMAC tasks stay pending'
  fi
}

launch() {
  exec 8> "$LOCK_DIR/${HOST}.control.lock"
  if ! flock -n 8; then
    echo "Another main0928 start/resume/stop command is active on $HOST; wait for it to finish." >&2
    exit 75
  fi
  stop_existing
  exec 9>&-
  nohup setsid bash "$SCRIPT" _run </dev/null >> "$OUT/run_all.${HOST}.console.log" 2>&1 8>&- &
  launched=$!
  for attempt in {1..50}; do
    owner=$(cat "$LOCK_DIR/${HOST}.run_all.pid" 2>/dev/null || true)
    if [[ "$owner" == "$launched" ]] && wrapper_pid_matches "$owner"; then
      echo "main0928 pipeline running on $HOST: PID $owner"
      echo "Log: $OUT/run_all.${HOST}.console.log    Panel: bash $SCRIPT status"
      exit 0
    fi
    if ! kill -0 "$launched" 2>/dev/null; then
      echo "Pipeline startup failed; see $OUT/run_all.${HOST}.console.log" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "Pipeline PID $launched did not publish startup state; inspect $OUT/run_all.${HOST}.console.log" >&2
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
    exec 8> "$LOCK_DIR/${HOST}.control.lock"
    flock -n 8 || { echo "Another main0928 control command is active on $HOST." >&2; exit 75; }
    stop_existing
    echo "main0928 stopped on $HOST; claims released after schedulers exit. Continue with: bash run_all.sh resume" ;;
  status)
    "${PIPE[@]}" status "$@" ;;
  retry)
    "${PIPE[@]}" retry "$@" ;;
  _run)
    exec 9> "$LOCK_DIR/${HOST}.run_all.lock"
    if ! flock -n 9; then
      echo 'main0928 wrapper already running on this host; use resume to replace it safely.' >&2
      exit 75
    fi
    printf '%s\n' "$$" > "$LOCK_DIR/${HOST}.run_all.pid"
    child=
    cleanup() {
      if [[ "$(cat "$LOCK_DIR/${HOST}.run_all.pid" 2>/dev/null || true)" == "$$" ]]; then
        rm -f -- "$LOCK_DIR/${HOST}.run_all.pid"
      fi
    }
    forward() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi; }
    trap forward INT TERM
    trap cleanup EXIT
    echo "$(date '+%F %T') main0928 pipeline starting on $HOST"
    "${PIPE[@]}" run 9>&- &
    child=$!
    result=0
    while kill -0 "$child" 2>/dev/null; do
      wait "$child" || result=$?
    done
    echo "$(date '+%F %T') main0928 pipeline exited ($result)"
    exit "$result" ;;
  *)
    usage ;;
esac

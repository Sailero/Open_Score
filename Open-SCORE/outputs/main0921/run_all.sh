#!/usr/bin/env bash
# Authorized staged experiment queue; no seed expansion or budget extension.
set -Eeuo pipefail
OUT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$OUT/../../.." && pwd)
cd -- "$REPO"
PY=/home/dell/anaconda3/envs/saileron/bin/python
SMAC_PY=/data3/dell/Saileron/envs/saileron-smac/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SC2PATH=/data3/dell/Saileron/envs/StarCraftII
export PYTHONPATH="" PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/data3/dell/Saileron/tmp
export XDG_CACHE_HOME=/data3/dell/Saileron/.cache
trainer= evaluator=
stopping=0
stop_children() {
  if (( stopping )); then return; fi
  stopping=1
  "$PY" Open-SCORE/scripts/train.py --stage stop --output "$OUT"
  "$PY" Open-SCORE/scripts/eval.py --stage stop --output "$OUT"
  if [[ -n "$trainer" ]]; then wait "$trainer" || true; fi
  if [[ -n "$evaluator" ]]; then wait "$evaluator" || true; fi
}
trap 'stop_children; exit 130' INT TERM
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
echo 'Starting HAD: GPU0/GPU1 training and CPU final/mechanism queues'
CUDA_VISIBLE_DEVICES=0,1 "$PY" -u Open-SCORE/scripts/train.py \
  --profile main0921 --stage train --group main --env had --steps 1000000 \
  --batch-size-run 8 --max-concurrent 2 --devices 0,1 --run train --resume --output "$OUT" \
  >> "$OUT/train.had.console.log" 2>&1 &
trainer=$!
CUDA_VISIBLE_DEVICES='' "$PY" -u Open-SCORE/scripts/eval.py \
  --profile main0921 --stage eval --env had --only final,depth,readout,probe \
  --device cpu --max-concurrent 16 --resume --output "$OUT" \
  >> "$OUT/eval.had.console.log" 2>&1 &
evaluator=$!
echo "HAD trainer=$trainer evaluator=$evaluator"
wait_pair
echo 'HAD queues complete; measuring M4 on one registered GPU'
CUDA_VISIBLE_DEVICES=0,1 "$PY" -u Open-SCORE/scripts/eval.py \
  --profile main0921 --stage eval --env had --only timing --device cuda \
  --max-concurrent 1 --resume --output "$OUT" >> "$OUT/timing.console.log" 2>&1
echo 'Starting SMACv2: one automatically placed GPU trainer and up to four CPU evaluators'
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1 "$SMAC_PY" -u Open-SCORE/scripts/train.py \
  --profile main0921 --stage train --group main --env smacv2 --steps 4000000 \
  --batch-size-run 4 --max-concurrent 1 --devices 0,1 --run train --resume --output "$OUT" \
  >> "$OUT/train.smacv2.console.log" 2>&1 &
trainer=$!
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "$SMAC_PY" -u Open-SCORE/scripts/eval.py \
  --profile main0921 --stage eval --env smacv2 --only final --device cpu \
  --max-concurrent 4 --resume --output "$OUT" >> "$OUT/eval.smacv2.console.log" 2>&1 &
evaluator=$!
echo "SMACv2 trainer=$trainer evaluator=$evaluator"
wait_pair
CUDA_VISIBLE_DEVICES='' "$PY" Open-SCORE/scripts/plot.py --output "$OUT" --run train
echo 'All registered main0921 training, evaluation and mechanism tasks complete'

#!/usr/bin/env bash
# Bootstrap HAD/SMAC prefixes, vendored packages, and SC2 4.10.
# Reuses an existing conda env when possible (default name: sarc).
# Matches that env's torch wheel to the machine CUDA (13.2 -> cu132, Python 3.10 OK).
# Each host writes envs/local.$HOST.env so a shared clone can keep per-machine Pythons.
# Run from the cloned repository root.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$ROOT"
# Cache/tmp stay inside the clone unless the caller sets SAILERON_ROOT.
SAILERON_ROOT=${SAILERON_ROOT:-$ROOT}
REUSE_ENV=${REUSE_ENV:-sarc}
FRESH=${FRESH:-0}
SKIP_SC2=${SKIP_SC2:-0}
KEEP_TORCH=${KEEP_TORCH:-0}
FORCE_TORCH=${FORCE_TORCH:-0}
HOST=$(hostname -s 2>/dev/null || hostname)
MAMBA_ROOT="$ROOT/envs/micromamba"
HAD_PREFIX="$ROOT/envs/saileron.${HOST}"
SMAC_PREFIX="$ROOT/envs/saileron-smac.${HOST}"
SC2_DIR=${SC2PATH:-"$ROOT/envs/StarCraftII"}
LOCAL_ENV="$ROOT/envs/local.env"
SC2_ZIP_URL=${SC2_ZIP_URL:-http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip}
MAPS_URL=${MAPS_URL:-https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip}

log() { printf '[setup_farm] %s\n' "$*"; }
log "ROOT=$ROOT REUSE_ENV=$REUSE_ENV FRESH=$FRESH SAILERON_ROOT=$SAILERON_ROOT KEEP_TORCH=$KEEP_TORCH"

install_micromamba() {
  if [[ -x "$MAMBA_ROOT/bin/micromamba" ]]; then
    return
  fi
  log "installing micromamba into $MAMBA_ROOT"
  mkdir -p "$MAMBA_ROOT"
  python3 - "$MAMBA_ROOT" <<'PY'
import os, sys, urllib.request, tarfile, io
root = sys.argv[1]
url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
data = urllib.request.urlopen(url, timeout=120).read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as archive:
    member = archive.getmember("bin/micromamba")
    archive.extract(member, path=root)
os.chmod(os.path.join(root, "bin/micromamba"), 0o755)
PY
}

mamba() {
  "$MAMBA_ROOT/bin/micromamba" "$@"
}

find_named_python() {
  local name=$1 py base line prefix
  local -a candidates=()
  if [[ -n "${CONDA_PREFIX:-}" && ( "${CONDA_DEFAULT_ENV:-}" == "$name" || "$(basename -- "$CONDA_PREFIX")" == "$name" ) ]]; then
    candidates+=("$CONDA_PREFIX/bin/python")
  fi
  if command -v conda >/dev/null 2>&1; then
    base=$(conda info --base 2>/dev/null || true)
    [[ -n "$base" ]] && candidates+=("$base/envs/$name/bin/python")
    while IFS= read -r line; do
      prefix=${line##* }
      [[ -n "$prefix" && "$prefix" != "#" ]] && candidates+=("$prefix/bin/python")
    done < <(conda env list 2>/dev/null | awk -v n="$name" '$1==n {print $NF}')
  fi
  if [[ -n "${CONDA_EXE:-}" ]]; then
    base=$(cd -- "$(dirname -- "$CONDA_EXE")/.." && pwd)
    candidates+=("$base/envs/$name/bin/python")
  fi
  if [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
    candidates+=("$MAMBA_ROOT_PREFIX/envs/$name/bin/python")
  fi
  for base in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/miniforge3" \
              "$HOME/.conda" "/opt/conda" "/opt/miniconda3" \
              "/data3/dell/anaconda3" "/data3/dell/miniconda3" "/data3/dell/mambaforge"; do
    candidates+=("$base/envs/$name/bin/python")
  done
  for py in "${candidates[@]}"; do
    if [[ -x "$py" ]]; then
      printf '%s\n' "$py"
      return 0
    fi
  done
  return 1
}

prefix_of_python() {
  local py=$1
  cd -- "$(dirname -- "$py")/.." && pwd
}

require_python() {
  "$1" - <<'PY'
import sys
v = sys.version_info
print(f"python {v.major}.{v.minor}.{v.micro}")
if v < (3, 10):
    raise SystemExit("need Python >= 3.10; CUDA 13.2 PyTorch wheels start at cp310")
if (v.major, v.minor) == (3, 10):
    print("Python 3.10 is supported (official torch+cu132 cp310 wheels)")
PY
}

has_torch() {
  "$1" - <<'PY' >/dev/null 2>&1
import torch
assert torch.__version__
PY
}

protobuf_is_smac() {
  "$1" - <<'PY' >/dev/null 2>&1
import google.protobuf as p
print(p.__version__)
raise SystemExit(0 if str(getattr(p, "__version__", "")).startswith("3.20") else 1)
PY
}

driver_cuda() {
  local cuda=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)
  fi
  if [[ -z "$cuda" ]] && command -v nvcc >/dev/null 2>&1; then
    cuda=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n1)
  fi
  printf '%s\n' "$cuda"
}

cuda_tag() {
  if [[ -n "${TORCH_CUDA:-}" ]]; then
    echo "$TORCH_CUDA"
    return
  fi
  local cuda
  cuda=$(driver_cuda)
  case "$cuda" in
    13.2*|13.3*|13.4*|13.5*|13.6*) echo "cu132" ;;
    13.0*|13.1*) echo "cu130" ;;
    13.*) echo "cu132" ;;
    12.8*|12.9*) echo "cu128" ;;
    12.6*|12.7*) echo "cu126" ;;
    12.4*|12.5*) echo "cu124" ;;
    12.1*|12.2*|12.3*) echo "cu121" ;;
    12.*) echo "cu128" ;;
    11.*) echo "cu118" ;;
    "") echo "cpu" ;;
    *) echo "cu132" ;;
  esac
}

torch_fallbacks() {
  case "$1" in
    cu132) echo cu132 cu130 cu128 ;;
    cu130) echo cu130 cu128 cu126 ;;
    cu128) echo cu128 cu126 cu124 ;;
    cu126) echo cu126 cu124 ;;
    cu124) echo cu124 cu121 ;;
    cu121) echo cu121 cu118 ;;
    cu118) echo cu118 ;;
    cpu) echo cpu ;;
    *) echo "$1" cu132 cu130 cu128 ;;
  esac
}

torch_matches_tag() {
  local py=$1 tag=$2
  "$py" - "$tag" <<'PY' >/dev/null 2>&1
import sys
tag = sys.argv[1]
try:
    import torch
except Exception:
    raise SystemExit(1)
cuda = str(getattr(torch.version, "cuda", None) or "")
want = {
    "cpu": "",
    "cu118": "11.8",
    "cu121": "12.1",
    "cu124": "12.4",
    "cu126": "12.6",
    "cu128": "12.8",
    "cu130": "13.0",
    "cu132": "13.2",
}.get(tag)
if want is None:
    raise SystemExit(0 if tag in str(torch.__version__) else 1)
if tag == "cpu":
    raise SystemExit(0 if not cuda else 1)
raise SystemExit(0 if cuda.startswith(want) else 1)
PY
}

pip_install_torch() {
  local py=$1 tag=$2
  if [[ "$tag" == "cpu" ]]; then
    "$py" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
  else
    "$py" -m pip install --upgrade torch --index-url "https://download.pytorch.org/whl/$tag"
  fi
}

install_torch() {
  local py=$1 tag candidate
  tag=$(cuda_tag)
  if [[ "$FORCE_TORCH" != "1" && "$KEEP_TORCH" == "1" ]] && has_torch "$py"; then
    log "KEEP_TORCH=1: leave $($py -c 'import torch; print(torch.__version__, torch.version.cuda)')"
    return
  fi
  if [[ "$FORCE_TORCH" != "1" ]] && torch_matches_tag "$py" "$tag"; then
    log "reuse torch $($py -c 'import torch; print(torch.__version__, "cuda="+str(torch.version.cuda))') for $tag"
    return
  fi
  log "driver CUDA $(driver_cuda); installing torch for $tag into $py"
  for candidate in $(torch_fallbacks "$tag"); do
    log "pip torch from https://download.pytorch.org/whl/$candidate"
    if pip_install_torch "$py" "$candidate"; then
      if torch_matches_tag "$py" "$candidate" || has_torch "$py"; then
        log "torch now $($py -c 'import torch; print(torch.__version__, "cuda="+str(torch.version.cuda), "gpu="+str(torch.cuda.is_available()))')"
        return
      fi
    fi
    log "wheel index $candidate failed; trying fallback"
  done
  log "FAILED to install a CUDA-matching torch ($tag) into $py"
  return 1
}

create_fresh_prefix() {
  local prefix=$1
  if [[ -x "$prefix/bin/python" ]]; then
    log "reuse $prefix"
    return
  fi
  install_micromamba
  log "creating $prefix (python 3.10)"
  mamba create -y -p "$prefix" python=3.10 pip
}

resolve_had_python() {
  local py
  if [[ -n "${HAD_PY:-}" && -x "${HAD_PY}" ]]; then
    HAD_PYTHON=$HAD_PY
    HAD_PREFIX=$(prefix_of_python "$HAD_PYTHON")
    log "HAD python from HAD_PY=$HAD_PYTHON"
    return
  fi
  if [[ "$FRESH" != "1" ]]; then
    if py=$(find_named_python "$REUSE_ENV"); then
      HAD_PYTHON=$py
      HAD_PREFIX=$(prefix_of_python "$HAD_PYTHON")
      log "reusing conda env '$REUSE_ENV' at $HAD_PREFIX"
      return
    fi
    if [[ "$REUSE_ENV" != "saileron" ]] && py=$(find_named_python saileron); then
      HAD_PYTHON=$py
      HAD_PREFIX=$(prefix_of_python "$HAD_PYTHON")
      log "reusing conda env 'saileron' at $HAD_PREFIX"
      return
    fi
  fi
  create_fresh_prefix "$HAD_PREFIX"
  HAD_PYTHON="$HAD_PREFIX/bin/python"
}

resolve_smac_python() {
  local py
  if [[ -n "${SMAC_PY:-}" && -x "${SMAC_PY}" ]]; then
    SMAC_PYTHON=$SMAC_PY
    SMAC_PREFIX=$(prefix_of_python "$SMAC_PYTHON")
    log "SMAC python from SMAC_PY=$SMAC_PYTHON"
    return
  fi
  if protobuf_is_smac "$HAD_PYTHON"; then
    SMAC_PYTHON=$HAD_PYTHON
    SMAC_PREFIX=$HAD_PREFIX
    log "HAD env already has protobuf 3.20.x; SMAC uses the same interpreter"
    return
  fi
  if [[ -x "$SMAC_PREFIX/bin/python" ]]; then
    SMAC_PYTHON="$SMAC_PREFIX/bin/python"
    log "reuse $SMAC_PREFIX"
    return
  fi
  log "creating SMAC venv with system site packages from HAD (keeps sarc torch; pins protobuf)"
  mkdir -p "$(dirname -- "$SMAC_PREFIX")"
  if "$HAD_PYTHON" -m venv --system-site-packages "$SMAC_PREFIX"; then
    SMAC_PYTHON="$SMAC_PREFIX/bin/python"
    return
  fi
  log "venv failed; falling back to a fresh micromamba prefix for SMAC"
  create_fresh_prefix "$SMAC_PREFIX"
  SMAC_PYTHON="$SMAC_PREFIX/bin/python"
}

install_had() {
  log "HAD: Open-SCORE + HADE into $HAD_PYTHON"
  if [[ "$FRESH" == "1" ]]; then
    "$HAD_PYTHON" -m pip install -U pip setuptools wheel
  fi
  install_torch "$HAD_PYTHON"
  "$HAD_PYTHON" -m pip install -e "$ROOT/third_party/HADE"
  "$HAD_PYTHON" -m pip install -e "$ROOT/Open-SCORE[training]"
  install_torch "$HAD_PYTHON"
}

install_smac() {
  log "SMAC: protobuf 3.20.3 + SMACv2 into $SMAC_PYTHON"
  if [[ "$FRESH" == "1" || "$SMAC_PYTHON" != "$HAD_PYTHON" ]]; then
    "$SMAC_PYTHON" -m pip install -U pip setuptools wheel || true
  fi
  install_torch "$SMAC_PYTHON"
  "$SMAC_PYTHON" -m pip install 'protobuf==3.20.3'
  "$SMAC_PYTHON" -m pip install -e "$ROOT/third_party/HADE"
  "$SMAC_PYTHON" -m pip install -e "$ROOT/Open-SCORE[training]"
  "$SMAC_PYTHON" -m pip install -e "$ROOT/third_party/SMACv2"
  "$SMAC_PYTHON" -m pip install pysc2 || log "pysc2 pip failed; SMAC may stay pending until it is installed"
  "$SMAC_PYTHON" -m pip install 'protobuf==3.20.3'
  install_torch "$SMAC_PYTHON"
}

install_sc2() {
  if [[ -n "${SC2PATH:-}" && -d "$SC2PATH/Versions" ]]; then
    log "SC2 already at $SC2PATH"
    SC2_DIR=$SC2PATH
    return
  fi
  if [[ -d "$SC2_DIR/Versions" ]]; then
    log "SC2 already at $SC2_DIR"
    return
  fi
  if [[ "$SKIP_SC2" == "1" ]]; then
    log "SKIP_SC2=1: SMAC stays pending until StarCraft II is installed"
    return
  fi
  log "downloading StarCraft II 4.10 (Blizzard EULA zip)"
  mkdir -p "$ROOT/envs"
  local zip="$ROOT/envs/SC2.4.10.zip"
  if [[ ! -f "$zip" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      log "curl missing; skip SC2 download"
      return
    fi
    curl -L --fail --retry 3 -o "$zip" "$SC2_ZIP_URL" || {
      log "SC2 download failed; HAD still works. Retry later or set SC2PATH."
      return
    }
  fi
  log "unzipping SC2 (EULA password)"
  local tmp="$ROOT/envs/.sc2_unpack"
  rm -rf "$tmp"
  mkdir -p "$tmp"
  if command -v unzip >/dev/null 2>&1; then
    unzip -P iagreetotheeula -q "$zip" -d "$tmp" || {
      log "unzip failed; install unzip or unpack SC2 yourself"
      return
    }
  else
    log "unzip not installed; skip SC2 unpack"
    return
  fi
  mkdir -p "$(dirname -- "$SC2_DIR")"
  if [[ -d "$tmp/StarCraftII" ]]; then
    mv "$tmp/StarCraftII" "$SC2_DIR"
  else
    mkdir -p "$SC2_DIR"
    mv "$tmp"/* "$SC2_DIR" 2>/dev/null || true
  fi
  rm -rf "$tmp"
  mkdir -p "$SC2_DIR/Maps"
  local maps_dir="$ROOT/third_party/SMACv2/smacv2/env/starcraft2/maps/SMAC_Maps"
  if [[ ! -d "$SC2_DIR/Maps/SMAC_Maps" ]]; then
    if [[ -d "$maps_dir" ]]; then
      cp -a "$maps_dir" "$SC2_DIR/Maps/SMAC_Maps"
    else
      local maps="$ROOT/envs/SMAC_Maps.zip"
      curl -L --fail --retry 3 -o "$maps" "$MAPS_URL" && unzip -q "$maps" -d "$SC2_DIR/Maps" || \
        log "SMAC_Maps download failed; copy maps into $SC2_DIR/Maps/SMAC_Maps"
    fi
  fi
}

write_env() {
  local host
  host=$(hostname -s 2>/dev/null || hostname)
  mkdir -p "$ROOT/envs" "$SAILERON_ROOT/tmp" "$SAILERON_ROOT/.cache"
  local payload
  payload=$(cat <<EOF
# Generated by scripts/setup_farm.sh on $host — sourced by run_all.sh
export REGIR_ROOT="$ROOT"
export SAILERON_ROOT="$SAILERON_ROOT"
export PY="$HAD_PYTHON"
export SMAC_PY="$SMAC_PYTHON"
export SC2PATH="$SC2_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/Open-SCORE"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${SAILERON_ROOT}/tmp"
export XDG_CACHE_HOME="${SAILERON_ROOT}/.cache"
EOF
)
  printf '%s\n' "$payload" > "$ROOT/envs/local.${host}.env"
  printf '%s\n' "$payload" > "$LOCAL_ENV"
  log "wrote $ROOT/envs/local.${host}.env (and $LOCAL_ENV)"
  log "  PY=$HAD_PYTHON"
  log "  SMAC_PY=$SMAC_PYTHON"
}

smoke() {
  log "HAD smoke"
  PYTHONPATH="$ROOT/Open-SCORE" "$HAD_PYTHON" - <<'PY'
import had_env
import torch
from open_score.envs.entity_env import HADEntityEnv
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device", torch.cuda.get_device_name(0))
env = HADEntityEnv(seed=0, config=(4, 4, 1))
obs = env.reset()
assert obs is not None
env.close()
print("HADEntityEnv ok", had_env.__file__)
PY
  if [[ -d "$SC2_DIR/Versions" ]]; then
    log "SMAC smoke"
    PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT/Open-SCORE" SC2PATH="$SC2_DIR" "$SMAC_PYTHON" - <<'PY' || log "SMAC adapter smoke failed (SC2 maps?); HAD is still usable"
from open_score.envs.smacv2_env import MixedScaleSMACAdapter
print("MixedScaleSMACAdapter", MixedScaleSMACAdapter)
PY
  else
    log "no SC2: skip SMAC smoke"
  fi
}

resolve_had_python
require_python "$HAD_PYTHON"
if [[ "$SKIP_SC2" == "1" ]]; then
  SMAC_PYTHON=$HAD_PYTHON
  SMAC_PREFIX=$HAD_PREFIX
  log "SKIP_SC2=1: skip SMAC python packages"
else
  resolve_smac_python
fi
install_had
if [[ "$SKIP_SC2" != "1" ]]; then
  install_smac
fi
install_sc2
write_env
smoke
log "done. Next:"
log "  source $ROOT/envs/local.\$(hostname -s).env"
log "  bash Open-SCORE/outputs/main0928/run_all.sh start"
log "  (operator guide: Open-SCORE/outputs/main0928/远程操作说明.md)"

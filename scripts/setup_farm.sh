#!/usr/bin/env bash
# Bootstrap HAD/SMAC prefixes, vendored packages, and SC2 4.10.
# Reuses an existing conda env when possible (default name: sarc).
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
HOST=$(hostname -s 2>/dev/null || hostname)
MAMBA_ROOT="$ROOT/envs/micromamba"
HAD_PREFIX="$ROOT/envs/saileron.${HOST}"
SMAC_PREFIX="$ROOT/envs/saileron-smac.${HOST}"
SC2_DIR=${SC2PATH:-"$ROOT/envs/StarCraftII"}
LOCAL_ENV="$ROOT/envs/local.env"
SC2_ZIP_URL=${SC2_ZIP_URL:-http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip}
MAPS_URL=${MAPS_URL:-https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip}

log() { printf '[setup_farm] %s\n' "$*"; }
log "ROOT=$ROOT REUSE_ENV=$REUSE_ENV FRESH=$FRESH SAILERON_ROOT=$SAILERON_ROOT"

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

cuda_tag() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi
  local cuda
  cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n1)
  case "$cuda" in
    12.4*|12.5*|12.6*|12.8*|13.*) echo "cu124" ;;
    12.1*|12.2*|12.3*) echo "cu121" ;;
    11.*) echo "cu118" ;;
    *) echo "cu124" ;;
  esac
}

install_torch() {
  local py=$1 tag
  if has_torch "$py"; then
    log "reuse torch $($py -c 'import torch; print(torch.__version__)')"
    return
  fi
  tag=$(cuda_tag)
  log "installing torch ($tag) into $py"
  if [[ "$tag" == "cpu" ]]; then
    "$py" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
  else
    "$py" -m pip install --upgrade torch --index-url "https://download.pytorch.org/whl/$tag"
  fi
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
from open_score.envs.entity_env import HADEntityEnv
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

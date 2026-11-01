#!/usr/bin/env bash
# FP32 fast / fast+batch timing on the RTX 5090 after the FP32 gates were removed (user, 2026-10-23).
# Protocol: IMPLEMENTATION_2026-10-23_POLAR_SHELL.md S4 (GPU step). Sequential, so runs do not share the GPU.
# Each out-dir must be new (profile_mlmm.py refuses existing dirs); rerunning skips finished dirs.
set -euo pipefail
cd /home/ruigengji/E3D
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
OUT=experiments/2026-10-23_polar_shell/gpu_5090_nogate
mkdir -p "$OUT"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv > "$OUT/nvidia_smi_before.csv"

run() {  # name, extra args...
  local name=$1; shift
  if [[ -f "$OUT/$name.log" ]] && grep -q -- '^-> ' "$OUT/$name.log"; then
    echo "skip $name (done)"; return; fi
  [[ -d "$OUT/$name" ]] && mv "$OUT/$name" "$OUT/${name}_incomplete_$(date +%H%M%S)"
  echo "=== $name $(date +%T)"
  "$PY" -B -u profile_mlmm.py --threads 8 --out-dir "$OUT/$name" "$@" 2>&1 | grep --line-buffered -v Warning > "$OUT/$name.log"
  echo "=== $name done $(date +%T)"
}

QS_POLAR=0,1.5,2,2.5,3,3.5,4,4.5,5
QS_D19=0,1.5,2,2.5,3,3.5,4
run polar_fast       --polar-shell --qs $QS_POLAR
run polar_fast_batch --polar-shell --qs $QS_POLAR --batch-fragments
run d19_fast         --qs $QS_D19
run d19_fast_batch   --qs $QS_D19 --batch-fragments
echo ALL DONE $(date +%T)

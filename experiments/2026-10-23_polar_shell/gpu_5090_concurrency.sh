#!/usr/bin/env bash
# N = 1, 2, 4, 8 concurrent POLAR-shell copies on the RTX 5090 (fulda), 3 CPU threads each
# (8 x 3 = 24 cores). q = 0, 2, 3.5 only: q >= 4.5 goes NaN within 5 steps from the leg start frame (nan_trace_q*.log).
# Production defaults: FP32, fast forces, fragment batching. Rerun skips finished N.
set -euo pipefail
cd /home/ruigengji/E3D
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
OUT=experiments/2026-10-23_polar_shell/gpu_5090_concurrency
mkdir -p "$OUT"
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
for N in 1 2 4 8; do
  D=$OUT/n$N
  if [[ -d $D ]] && [[ $(ls $D/rank*.json 2>/dev/null | wc -l) -eq $N ]]; then echo "skip n$N"; continue; fi
  rm -rf "$D"; mkdir -p "$D"
  echo "=== n$N $(date +%T)"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 2 > "$D/nvidia_smi.csv" &
  SMI=$!
  pids=()
  for ((r = 0; r < N; r++)); do
    "$PY" -B -u experiments/2026-10-23_polar_shell/concurrency.py --out-dir "$D" --rank $r --nproc $N --qs 0,2,3.5 \
      > "$D/rank$r.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  kill $SMI
  echo "=== n$N done rc=$rc $(date +%T)"
done
echo ALL DONE $(date +%T)

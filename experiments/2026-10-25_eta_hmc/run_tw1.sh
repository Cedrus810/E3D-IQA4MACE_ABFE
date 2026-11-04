#!/usr/bin/env bash
# T1 window ladder, first round (TORSION_WINDOW_PLAN_DRAFT_2026-11-04.md): groups A, B (windows) and CA, CB (same-cost
# unbiased control), 200 sweeps each, two processes per GPU. Two or more visible GPUs: all four at once; one GPU: two
# waves (A + B, then CA + CB). Resumable: the same command continues every group from its last commit.
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${OMP_NUM_THREADS:-8} JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
LOGS=logs/tw1_${PBS_JOBID:-manual}
mkdir -p "$LOGS"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "GPUs: ${GPUS[*]}; logs $LOGS"
fail=0
run_wave() {
  local pids=() names=() i=0
  for g in "$@"; do
    local dev=${GPUS[$(( i / 2 % ${#GPUS[@]} ))]}
    CUDA_VISIBLE_DEVICES=$dev $P -B -u torsion_window.py run --group "$g" --sweeps 200 --tag _tw1_v1 > "$LOGS/$g.log" 2>&1 &
    pids+=($!); names+=("$g"); echo "group $g on GPU $dev pid $!"; i=$((i + 1))
  done
  for k in "${!pids[@]}"; do
    if wait "${pids[$k]}"; then echo "group ${names[$k]} ok"; else echo "group ${names[$k]} FAILED ($LOGS/${names[$k]}.log)"; fail=$((fail + 1)); fi
  done
}
if (( ${#GPUS[@]} >= 2 )); then run_wave A CA B CB; else run_wave A B; run_wave CA CB; fi
if (( fail == 0 )); then echo "ALL DONE $(date '+%F %T')"; else echo "INCOMPLETE $(date '+%F %T'): $fail failures"; exit 1; fi

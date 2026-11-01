#!/usr/bin/env bash
# Checks of S13.64 (user 2026-10-23): seeds 12, 13 at r_in 0.25; r_in 0.28 at seed 11. Phases A, B, D
# (C, the MC histogram, cannot reach N = 0). Three jobs side by side on the 5090. Resumable.
set -uo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-23_jump_swap
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
job() {  # r_in seed tag
  for P in A B; do $PY -B jump_swap.py $P --r-in $1 --seed $2 --tag $3 --ps 2000; done
  $PY -B jump_swap.py D --r-in $1 --seed $2 --tag $3 --ps 500 --equil-ps 50
  $PY -B jump_swap.py analyze --r-in $1 --tag $3
}
job 0.25 12 _s12 > check_s12.log 2>&1 &
job 0.25 13 _s13 > check_s13.log 2>&1 &
job 0.28 11 _s11 > check_r028.log 2>&1 &
wait
$PY -B jump_swap.py analyze --r-in 0.25 > check_s11_rebootstrap.log 2>&1   # seed 11 again, with the bootstrap
echo ALL DONE $(date '+%F %T')

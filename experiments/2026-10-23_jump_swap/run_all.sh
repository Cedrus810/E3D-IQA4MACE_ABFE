#!/usr/bin/env bash
# One-step jump + water term on the acetamide leg, MM only, r_in 0.25 nm. Resumable (finished phases skipped).
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-23_jump_swap
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
for P in A B; do $PY -B jump_swap.py $P --r-in 0.25 --ps 2000; done
$PY -B jump_swap.py C --r-in 0.25 --ps 1000
$PY -B jump_swap.py D --r-in 0.25 --ps 500 --equil-ps 50
$PY -B jump_swap.py analyze --r-in 0.25
echo ALL DONE $(date '+%F %T')

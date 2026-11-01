#!/usr/bin/env bash
# Group-by-group dummy MM sampling, one leg, one rep, on fulda (5090). Resumable (finished states skipped).
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-23_groupwise_dummy
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
$PY -B groupwise_mm.py run "$1" "$2" 100 500
$PY -B groupwise_mm.py analyze "$1"
echo ALL DONE $(date '+%F %T')

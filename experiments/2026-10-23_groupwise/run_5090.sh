#!/usr/bin/env bash
# Groupwise MM path, acetamide (DESIGN_2026-10-23_groupwise_dummy.md S4): reps 1-3 in order CH3 -> amide,
# rep 1 in order amide -> CH3; all four on the 5090 (user, 2026-10-23). Resumable per state.
set -uo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-23_groupwise
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
$PY -B -u groupwise_mm.py run 1 --order 0,1 > rep1_order01.log 2>&1 &
$PY -B -u groupwise_mm.py run 2 --order 0,1 > rep2_order01.log 2>&1 &
$PY -B -u groupwise_mm.py run 3 --order 0,1 > rep3_order01.log 2>&1 &
$PY -B -u groupwise_mm.py run 1 --order 1,0 > rep1_order10.log 2>&1 &
wait
$PY -B groupwise_mm.py analyze --order 0,1 > analyze_order01.log 2>&1
$PY -B groupwise_mm.py analyze --order 1,0 > analyze_order10.log 2>&1
echo ALL DONE $(date '+%F %T')

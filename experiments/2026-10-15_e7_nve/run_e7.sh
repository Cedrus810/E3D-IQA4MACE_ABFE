#!/bin/bash
# E7 (v3 S6.3.5): NVE continuity of the ML/MM + FIRES + wall potential, all
# states in series on one GPU, then a summary. Rerunning skips states whose
# npz exists.
set -euo pipefail
cd "$(dirname "$0")"
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
START=/home/ruigengji/E3D/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz:0
run() {   # name q n_equil n_nve dt
  [ -f "e7_$1.npz" ] && return 0
  echo "--- $1 (q $2, dt $5 fs) $(date '+%T')"
  if $P -u e7_nve.py "$2" "e7_$1.npz" "$3" "$4" "$5" 7 "$START" > "e7_$1.log" 2>&1; then
    grep -E "Langevin|NVE" "e7_$1.log"
  else
    echo "FAILED: $1"; grep -E "Exception|Error" "e7_$1.log" | tail -2
  fi
}
run mm     mm  500 1000 2.0
run q0.0   0.0 500 1000 2.0
run q1.0   1.0 500 1000 2.0
run q1.5   1.5 500 1000 2.0
run q2.0   2.0 500 1000 2.0
run q2.5   2.5 500 1000 2.0
run q3.5   3.5 500 1000 2.0
run q1.5_dt1 1.5 1000 2000 1.0
echo "=== e7 done $(date '+%F %T')"

#!/bin/bash
# MM frames at lambda_elec 0.5 and 0 (LJ full): 20 test frames (seed 23, every
# 50 ps, as thermal_test_solvent) and 200 training frames (seed 11, every 10 ps,
# as thermal200_solvent). Rerunning skips states whose output exists.
set -euo pipefail
cd "$(dirname "$0")"
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
for L in 0.5 0.0; do
  [ -f "lam${L}_test_solvent.npz" ] || $P -u lambda_frames.py "$L" 23 "lam${L}_test_solvent.npz" 50 20 CUDA
  [ -f "lam${L}_train200_solvent.npz" ] || $P -u lambda_frames.py "$L" 11 "lam${L}_train200_solvent.npz" 10 200 CUDA
done
echo "=== md done"

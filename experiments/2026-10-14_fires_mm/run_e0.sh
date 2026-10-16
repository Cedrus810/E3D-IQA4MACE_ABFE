#!/bin/bash
# E0 (v3 S6.3.5): FIRES vs free, pure MM, 3 ns each, in parallel on one GPU,
# then the comparison. Rerunning skips a run whose npz exists.
set -euo pipefail
cd "$(dirname "$0")"
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -f e0_fires.npz ] || $P -u fires_mm.py fires 7 e0_fires.npz 3 CUDA cuda > e0_fires.log 2>&1 &
[ -f e0_free.npz ]  || $P -u fires_mm.py free 7 e0_free.npz 3 CUDA cpu > e0_free.log 2>&1 &
wait
$P analyze_e0.py e0_free.npz e0_fires.npz 5 > e0_analysis.log 2>&1
echo "=== e0 done $(date '+%F %T')"

#!/bin/bash
# Acetamide MM hydration free energy: 3 independent reps in parallel on one GPU, then MBAR.
# Rerunning skips finished states (runs/rep*_state*.npz).
set -euo pipefail
cd "$(dirname "$0")"
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "=== start $(date '+%F %T') on $(hostname)"
for r in 1 2 3; do
  $P -u abfe_mm.py run $r 100 1000 > "mm_rep$r.log" 2>&1 &
done
wait
grep -h "done\|Error" mm_rep*.log || true
$P abfe_mm.py analyze 2>&1 | grep -E "^rep|^mean" | tee mm_result.log
echo "=== done $(date '+%F %T')"

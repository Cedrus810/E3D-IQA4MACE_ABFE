#!/usr/bin/env bash
# CypD-8 G1 per-atom MM path derivatives b_(1,i) on the saved bridge frames (atom_deriv.py), reps 1-3 in parallel on
# one GPU, all 500 frames per window. Resumable: the same command reuses finished window files. Ends with ALL DONE
# only if every eval and report finished.
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-26_mm_bridge
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONDONTWRITEBYTECODE=1
SYS=CyclophilinD_ligand-8
REPS="1 2 3"
mkdir -p logs
pids=()
for r in $REPS; do
  $P -B -u atom_deriv.py eval $SYS $r --group 1 --check 5 > logs/atom_deriv_G1_rep$r.log 2>&1 &
  pids+=($!)
done
fail=0
i=0
for r in $REPS; do
  if wait ${pids[$i]}; then echo "rep $r eval ok"; else echo "rep $r eval FAILED (logs/atom_deriv_G1_rep$r.log)"; fail=$((fail + 1)); fi
  i=$((i + 1))
done
for r in $REPS; do
  $P -B atom_deriv.py report $SYS $r --group 1 > logs/atom_deriv_G1_rep${r}_report.log 2>&1 \
    || { echo "rep $r report FAILED"; fail=$((fail + 1)); }
done
if (( fail == 0 )); then echo "ALL DONE $(date '+%F %T')"; else echo "INCOMPLETE $(date '+%F %T'): $fail failures"; exit 1; fi

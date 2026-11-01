#!/usr/bin/env bash
# CypD-8 per-atom MM path derivatives b_(k,i), groups 1-4, on the saved chain frames (atom_deriv.py; run after
# run_chain.sh), reps 1-3 in parallel on one GPU. Resumable: finished window files (G1 from run_atom_deriv_G1.sh) are
# reused. Ends with ALL DONE only if every eval and report finished.
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-26_mm_bridge
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONDONTWRITEBYTECODE=1
SYS=CyclophilinD_ligand-8
REPS="1 2 3"
GROUPS_K="1 2 3 4"
mkdir -p logs
pids=()
for r in $REPS; do
  ( for k in $GROUPS_K; do $P -B -u atom_deriv.py eval $SYS $r --group $k --check 5 || exit 1; done ) \
    > logs/atom_deriv_chain_rep$r.log 2>&1 &
  pids+=($!)
done
fail=0
i=0
for r in $REPS; do
  if wait ${pids[$i]}; then echo "rep $r eval ok"; else echo "rep $r eval FAILED (logs/atom_deriv_chain_rep$r.log)"; fail=$((fail + 1)); fi
  i=$((i + 1))
done
for r in $REPS; do
  for k in $GROUPS_K; do
    $P -B atom_deriv.py report $SYS $r --group $k > logs/atom_deriv_G${k}_rep${r}_report.log 2>&1 \
      || { echo "rep $r G$k report FAILED"; fail=$((fail + 1)); }
  done
done
if (( fail == 0 )); then echo "ALL DONE $(date '+%F %T')"; else echo "INCOMPLETE $(date '+%F %T'): $fail failures"; exit 1; fi

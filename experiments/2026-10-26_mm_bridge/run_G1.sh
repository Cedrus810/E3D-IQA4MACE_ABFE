#!/usr/bin/env bash
# CypD-8 G1 bridge (S0 -> S1), reps 1-3 in parallel on one GPU, 25 windows x (100 + 500 ps). Resumable: the same
# command reuses finished windows (bridge_mm.py run). Ends with ALL DONE only if every rep and analysis finished.
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-26_mm_bridge
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONDONTWRITEBYTECODE=1
SYS=CyclophilinD_ligand-8
ELEC=1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0
VDW=1,0.9,0.8,0.7,0.6,0.5,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0
REPS="1 2 3"
mkdir -p logs
pids=()
for r in $REPS; do
  $P -B -u bridge_mm.py run $SYS $r --groups 1 --elec $ELEC --vdw $VDW --equil-ps 100 --prod-ps 500 --frame-ps 1 \
    > logs/G1_rep$r.log 2>&1 &
  pids+=($!)
done
fail=0
i=0
for r in $REPS; do
  if wait ${pids[$i]}; then echo "rep $r run ok"; else echo "rep $r run FAILED (logs/G1_rep$r.log)"; fail=$((fail + 1)); fi
  i=$((i + 1))
done
for r in $REPS; do
  $P -B bridge_mm.py analyze $SYS $r > logs/G1_rep${r}_analyze.log 2>&1 \
    || { echo "rep $r analyze FAILED"; fail=$((fail + 1)); }
done
if (( fail == 0 )); then echo "ALL DONE $(date '+%F %T')"; else echo "INCOMPLETE $(date '+%F %T'): $fail failures"; exit 1; fi

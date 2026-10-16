#!/bin/bash
# v3 S8 route step 1: paired 2 fs / 1 fs NVE + float64 force check at one q.
# Usage: ./run_step1.sh Q   (q2.0 and q2.5 run as two chains on one GPU).
# Rerunning skips a finished part (its output exists).
set -euo pipefail
cd "$(dirname "$0")"
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
Q=$1
echo "--- q $Q paired_dt $(date '+%T')"
[ -f "q${Q}_frames.npz" ] || $P -u paired_dt.py "$Q" "q${Q}" 500 2.0 7 > "paired_q${Q}.log" 2>&1
grep -E "^q |Langevin|dt |cap|cover|done" "paired_q${Q}.log" || true
echo "--- q $Q fd_check $(date '+%T')"
[ -f "fd_q${Q}.done" ] || { $P -u fd_check.py "q${Q}_frames.npz" 8 1e-3,5e-4 > "fd_q${Q}.log" 2>&1 && touch "fd_q${Q}.done"; }
grep -E "^q |^\[|total|ML\+|wall" "fd_q${Q}.log" || true
echo "=== q $Q done $(date '+%F %T')"

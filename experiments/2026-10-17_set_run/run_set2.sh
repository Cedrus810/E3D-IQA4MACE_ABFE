#!/bin/bash
# ML/MM horizontal test on the revised path (v3 S6.3.4, S13.50: SW before an all-MM W3).
# Usage: ./run_set2.sh JOB [JOB ...]   JOB = full:MOL  (whole path q 0-5, 28 states, from the leg frame)
#                                           tail:MOL  (SW + W3, q 3-5, 14 states, from MOL's first-pass q = 3 state)
# MOL = an IBS system name (300 K) or "acetamide" (298.15 K). One job after another on one GPU;
# rerunning resumes each job from its checkpoint.
set -eo pipefail
cd /home/ruigengji/E3D
set +u; source /home/ruigengji/miniforge3/etc/profile.d/conda.sh && conda activate openmm_dev
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu
D=experiments/2026-10-17_set_run
for J in "$@"; do
  KIND=${J%%:*}; M=${J#*:}
  if [ "$M" = acetamide ]; then LEG=experiments/2026-10-15_acetamide_leg/em_acetamide.npz; T=298.15
  else LEG=experiments/2026-10-17_ibs_legs/$M.npz; T=300; fi
  if [ "$KIND" = full ]; then QS=$(head -1 $D/qs_full.txt); EXTRA=""; OUT=$D/v2_full_$M
  else QS=$(head -1 $D/qs_tail.txt); EXTRA="--init-from $D/$M --init-q 3.0"; OUT=$D/v2_tail_$M; fi
  echo "=== $J start $(date '+%F %T')"
  python -u run_abfe.py --mode mlmm --leg-npz $LEG --T $T --qs "$QS" $EXTRA \
    --ckpt runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt \
    --backbone /home/ruigengji/MLP/mace/MACE-POLAR-1-M.model --n-inner 20 \
    --equil 2 --iters 10 --steps-per-iter 250 --dt 2.0 --save-every 1 \
    --out-dir $OUT > $OUT.log 2>&1 || echo "FAILED: $J"
  grep -E "dG_|closure|overlap" $OUT.log || true
done
echo "=== done $(date '+%F %T')"

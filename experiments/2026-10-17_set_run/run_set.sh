#!/bin/bash
# ML/MM horizontal test (v3 S13.50): the full path on several molecules with the operating
# head, compared with each molecule's MM value. One molecule after another on one GPU.
# Usage: ./run_set.sh MOL [MOL ...]    MOL = an IBS system name (300 K) or "acetamide" (298.15 K)
# Rerunning resumes each molecule from its checkpoint.
set -eo pipefail
cd /home/ruigengji/E3D
set +u; source /home/ruigengji/miniforge3/etc/profile.d/conda.sh && conda activate openmm_dev
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu
D=experiments/2026-10-17_set_run
QS=$(head -1 $D/qs24.txt)
for M in "$@"; do
  if [ "$M" = acetamide ]; then LEG=experiments/2026-10-15_acetamide_leg/em_acetamide.npz; T=298.15
  else LEG=experiments/2026-10-17_ibs_legs/$M.npz; T=300; fi
  echo "=== $M start $(date '+%F %T')"
  python -u run_abfe.py --mode mlmm --leg-npz $LEG --T $T --qs "$QS" \
    --ckpt runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt \
    --backbone /home/ruigengji/MLP/mace/MACE-POLAR-1-M.model --n-inner 20 \
    --equil 2 --iters 10 --steps-per-iter 250 --dt 2.0 --save-every 1 \
    --out-dir $D/$M > $D/$M.log 2>&1 || echo "FAILED: $M"
  grep -E "dG_|closure|overlap" $D/$M.log || true
done
echo "=== done $(date '+%F %T')"

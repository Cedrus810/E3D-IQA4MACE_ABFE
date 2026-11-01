#!/bin/bash
# D19 acetamide full ladder (PLAN.md S2-S3), one process on 180's 2080 Ti (user, 2026-10-19).
# Resumable: rerun the same script; --init-from only acts on the first start.
set -euo pipefail
cd /home/ruigengji/E3D
set +u; source /home/ruigengji/miniforge3/etc/profile.d/conda.sh && conda activate openmm_dev; set -u
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu
D=experiments/2026-10-19_d19_acetamide
QS=0,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.94,1.97,1.99,2,2.25,2.5,2.75,3,3.2,3.4,3.55,3.7,3.8,3.86,3.91,3.95,3.975,3.99,4
echo "=== $(date '+%F %T') host $(hostname)"
python -u run_abfe.py --mode mlmm --shell-mm \
  --leg-npz experiments/2026-10-15_acetamide_leg/em_acetamide.npz \
  --ckpt runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt \
  --backbone /home/ruigengji/MLP/mace/MACE-POLAR-1-M.model --n-inner 20 \
  --qs "$QS" --equil 5 --iters 50 --steps-per-iter 500 --dt 2.0 --seed 0 \
  --init-from experiments/2026-10-16_mm_path_control/fires_seed1 \
  --out-dir $D/full
echo "=== done $(date '+%F %T')"

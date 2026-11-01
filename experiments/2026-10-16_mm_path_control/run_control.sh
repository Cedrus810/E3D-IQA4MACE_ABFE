#!/bin/bash
# All-MM path control (v3 S7.2) on the acetamide leg: must reproduce +34.04 +- 0.14 kJ/mol (S13.44).
# Usage: ./run_control.sh nofires SEED [ITERS]   (default 1000 x 1 ps per state after 20 ps)
#        ./run_control.sh fires SEED [ITERS]     (FIRES on MM inner waters, default 200)
# Rerunning resumes from the checkpoint; a larger ITERS extends a finished run.
set -euo pipefail
cd /home/ruigengji/E3D
set +u; source /home/ruigengji/miniforge3/etc/profile.d/conda.sh && conda activate openmm_dev; set -u   # conda activate.d scripts reference unset vars
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 JAX_PLATFORMS=cpu
KIND=$1; SEED=$2; ITERS=${3:-}
QS=$(head -1 experiments/2026-10-16_mm_path_control/qs.txt)
OUT=experiments/2026-10-16_mm_path_control/${KIND}_seed${SEED}
if [ "$KIND" = nofires ]; then EXTRA="--no-fires --iters ${ITERS:-1000}"; else EXTRA="--iters ${ITERS:-200}"; fi
python -u run_abfe.py --mode mlmm --mm-only $EXTRA --seed "$SEED" \
  --leg-npz experiments/2026-10-15_acetamide_leg/em_acetamide.npz --qs "$QS" \
  --equil 20 --steps-per-iter 500 --dt 2.0 --save-every 20 --eval-precision mixed --no-per-atom \
  --out-dir "$OUT"

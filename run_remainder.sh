#!/bin/bash
# S13.23 risk check: spread of the cut-step work r = truth - sumD over
# configurations, for both 20-frame heads (<=24 and <=120 waters).
#
#   nohup ./run_remainder.sh > runs/remainder.log 2>&1 & disown
#   tail -f runs/remainder.log
#
# Test frames: a fresh MM trajectory (seed 23), so neither head has seen any
# cluster cut from them. Summary per size at the end of each scan: r mean,
# r sd and sd/kT -- a few kT means one BAR step can bridge the cut.
set -eu
cd "$(dirname "$0")"

P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
N='warning|visitor|cuequiv|detach|^  return|^  print|^  model|_Jd|Consider'
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p runs/polar-M/logs
echo "=== $(date '+%F %T')  host $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-all} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ -f frames_test_solvent.npz ]; then
  echo "--- test frames: present, skipped"
else
  echo "--- test frames $(date '+%T')"
  $P -u -c "import em_system; em_system.frames(seed=23, out='frames_test_solvent.npz', heldout_out='md2000s23_solvent.npz')"
fi

for W in "" _w120; do
  CK=runs/polar-M/ckpt/joint_organic_n50000_c10000_frames_solvent${W}_h256_s60000_D.pt
  [ -f "$CK" ] || { echo "missing $CK"; exit 1; }
  echo "--- scan head frames_solvent${W} $(date '+%T')"
  $P -u test_scale.py MACE-POLAR-1-M "$CK" 5,10,20,50,100 frames_test_solvent.npz 2>&1 \
    | grep -viE --line-buffered "$N" \
    | tee "runs/polar-M/logs/remainder_frames${W}.log"
done
echo "=== done $(date '+%F %T') ==="

#!/bin/bash
# S13.19 end to end: MD frames -> POLAR-1-M with multi-frame clusters -> size scans.
#
#   nohup ./run_multiframe.sh > runs/multiframe.log 2>&1 & disown
#   CLUSTER_MAX=120 nohup ./run_multiframe.sh > runs/multiframe_w120.log 2>&1 & disown
#   tail -f runs/multiframe*.log
#
# Resumable: frames are skipped if present, run_polar.sh / test_joint.py resume
# from their own caches, so rerunning after a crash picks up where it stopped.
# Pick the card with CUDA_VISIBLE_DEVICES=... in front of nohup.
set -eu
cd "$(dirname "$0")"

P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
N='warning|visitor|cuequiv|detach|^  return|^  print|^  model|_Jd|Consider'
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p runs/polar-M/logs
echo "=== $(date '+%F %T')  host $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-all} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# 1. frames (GPU OpenMM; ~17 h on CPU)
if [ -f frames_solvent.npz ] && [ -f md2000_solvent.npz ]; then
  echo "--- frames: present, skipped"
else
  echo "--- frames $(date '+%T')"
  $P -u em_system.py frames
fi

# 2. train (arm D, 60k steps, batch 16, 10,000 clusters from the 20 frames)
W=${CLUSTER_MAX:-24}
WT=$([ "$W" = 24 ] || echo "_w$W")
echo "--- train $(date '+%T')  clusters up to $W waters"
FRAMES=frames_solvent.npz CLUSTER_MAX=$W TAGS=M ./run_polar.sh

# 3. size scans: two held-out frames, then the training frame for regression
CK=runs/polar-M/ckpt/joint_organic_n50000_c10000_frames_solvent${WT}_h256_s60000_D.pt
[ -f "$CK" ] || { echo "missing $CK -- training did not finish"; exit 1; }
for f in md50_solvent.npz md2000_solvent.npz em_solvent.npz; do
  echo "--- scan $f $(date '+%T')"
  $P -u test_scale.py MACE-POLAR-1-M "$CK" 5,10,20,50,100 "$f" 2>&1 \
    | grep -vE --line-buffered "$N" \
    | tee "runs/polar-M/logs/scale_frames${WT}_${f%.npz}.log"
done
echo "=== done $(date '+%F %T') ==="

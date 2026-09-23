#!/bin/bash
# Overnight MACE-POLAR-1 runs, one directory per backbone.
#
# The separate directories are not tidiness. test_joint.py's E/F targets ARE the
# backbone's own energy, so a target cache written under one backbone and reused
# under another trains the head against the wrong labels and nothing errors. One
# E3D_OUT per backbone makes that impossible.
#
#   ./run_polar.sh              M then L, arm D, defaults below
#   TAGS=L ./run_polar.sh       just the large one
#   STEPS=32000 ./run_polar.sh  shorter
#   ARMS=ABCD ./run_polar.sh    the full ladder, not just the five-term arm
#
# Needs graph_electrostatics v0.4.0 -- NOT v0.4.4, which mace-torch 0.3.16
# cannot call (see test_polar.py). Check with:
#   python -c "import graph_longrange as g; print(g.__version__)"
set -u
cd "$(dirname "$0")"

P=/home/ruigengji/miniforge3/envs/openmm_dev_ubio/bin/python
N='warning|visitor|cuequiv|detach|^  return|^  print|_Jd|rho\(r\)'
MDIR=/home/ruigengji/MLP/mace

# The per-edge pair tensor product allocates in large blocks and the run is long;
# L died 726 MiB short on an 11 GiB card with 589 MiB reserved-but-unallocated.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TAGS=${TAGS:-"M L"}       # POLAR-1-S is scalars only -- the head's pair branch
                          # loses all angular information on it. Don't.
CFG=${CFG:-organic}
NTR=${NTR:-50000}
STEPS=${STEPS:-60000}     # S13.12: arm D was still descending at 32,000
HIDDEN=${HIDDEN:-256}
ARMS=${ARMS:-D}
# Measured at hidden=256, arm D, on an 11 GiB RTX 2080 Ti:
#   M, batch 16   10.7 GiB   0.30 s/step
#   L, batch 16   OOM at ~11.3 GiB -- fits a 16 GiB card, not this one
# Lower this first if the card runs out of memory; it is the only knob that
# does not change the model.
BATCH=${BATCH:-16}
# Backbone-labelled many-body clusters (decomp/clusters.py). 0 reproduces the
# dimer-only recipe, whose heads carry a per-edge bias that grows with system
# size (test_scale.py). 10,000 against 50,000 dimers is ~17% of batches; the
# labels take ~30 min once and are cached in $OUT/data/clusters_N.npy.
CLUSTERS=${CLUSTERS:-10000}

for TAG in $TAGS; do
  MODEL=$MDIR/MACE-POLAR-1-$TAG.model
  OUT=runs/polar-$TAG
  [ -f "$MODEL" ] || { echo "missing $MODEL"; exit 1; }
  mkdir -p "$OUT/logs"
  echo "=== MACE-POLAR-1-$TAG  cfg=$CFG n=$NTR clusters=$CLUSTERS steps=$STEPS h=$HIDDEN arms=$ARMS -> $OUT ==="
  MACE_MODEL=$MODEL E3D_OUT=$OUT E3D_BATCH=$BATCH E3D_CLUSTERS=$CLUSTERS \
    $P -u test_joint.py "$CFG" "$NTR" "$STEPS" "$HIDDEN" "$ARMS" 2>&1 \
    | grep -viE --line-buffered "$N" \
    | tee -a "$OUT/logs/joint.log"
done

echo "=== done ==="
echo "results:  runs/polar-*/logs/joint_*.json"
echo "heads:    runs/polar-*/ckpt/"

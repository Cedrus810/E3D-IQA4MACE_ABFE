#!/bin/bash
# step_eval.py on the 5090 (fulda): model loaded once per system, all frames of S0..S{K-1}, then the step table.
# Resumable: a finished state's steps_N50_S{k}.npz is skipped on rerun.
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-23_groupwise_dummy
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
for s in PWWP1_ligand-8 CyclophilinD_ligand-8; do
  $P step_eval.py eval "$s" --batch 16 && $P step_eval.py steps "$s"
done

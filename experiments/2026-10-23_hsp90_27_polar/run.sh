#!/usr/bin/env bash
# HSP90-27 solvent leg, POLAR-shell path (user 2026-10-23: path decided, ion wall, ladder EM), RTX 5090.
# Two ladders side by side on one GPU (v3 S13.62: <= 4 processes):
#   segAB  q 0-3  (W1, W2, cut; same Hamiltonian as the old path's A, now with the ion wall), fresh
#          from the leg start frame, EM per replica.
#   segC   q 3-4  (POLAR-shell W3), every replica from the old longA run's equilibrated q = 3 frame
#          (inner waters picked on that frame), EM per replica.
# The shape leg (q 4-5) is not run: in the ladder smoke its EM stalled and q = 5 went NaN (smoke.log).
# 300 K (IBS leg), 2 + 20 iterations x 500 steps x 2 fs = 20 ps per state, as the acetamide D19 full20.
# Resumable: rerun this script; each ladder resumes from its state.npz (init-from/EM only on a fresh start).
set -euo pipefail
cd /home/ruigengji/E3D
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
D=experiments/2026-10-23_hsp90_27_polar
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
F='warn|^ *(return|model|system|warnings)|^\*|^$'
COMMON=(--mode mlmm --polar-shell --leg-npz experiments/2026-10-17_ibs_legs/HSP90_ligand-27.npz --T 300
        --ckpt runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt
        --backbone /home/ruigengji/MLP/mace/MACE-POLAR-1-M.model --n-inner 20
        --equil 2 --iters 20 --steps-per-iter 500 --dt 2.0 --seed 0 --save-every 1)
QAB=0,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.94,1.97,1.99,2,2.25,2.5,2.75,3
QC=3,3.2,3.4,3.55,3.7,3.8,3.86,3.91,3.95,3.975,3.99,4
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > $D/nvidia_smi_start.txt
"$PY" -B -u run_abfe.py "${COMMON[@]}" --qs $QAB --out-dir $D/segAB 2>&1 | grep --line-buffered -viE "$F" >> $D/segAB.log &
P1=$!
"$PY" -B -u run_abfe.py "${COMMON[@]}" --qs $QC --init-from experiments/2026-10-18_checks/longA_HSP90_ligand-27 \
  --init-q 3 --out-dir $D/segC 2>&1 | grep --line-buffered -viE "$F" >> $D/segC.log &
P2=$!
rc=0; wait $P1 || rc=1; wait $P2 || rc=1
echo "ALL DONE rc=$rc $(date '+%F %T')"

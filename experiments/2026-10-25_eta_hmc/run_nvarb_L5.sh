#!/usr/bin/env bash
# nvar b frozen, CypD-8 S0, eta 1.0 x starts rep1-4, L=5 x 2 fs (10 fs), 500 cycles each (same model-call
# budget as the L=20 _nvarb_ms chains). One process, chains in sequence (181, 2080 Ti). Resumable: rerun.
set -euo pipefail
cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
for s in 1 2 3 4; do
  $P -u -B eta_hmc.py chain CyclophilinD_ligand-8 --eta 1.0 --start $s --cycles 500 --L 5 \
    --tag _nvarb_L5 --head $HEAD --log-every 50 > logs/nvarb_L5_eta1.0_start${s}.log 2>&1
done
$P -u -B eta_hmc.py analyze CyclophilinD_ligand-8 --tag _nvarb_L5 > logs/nvarb_L5_analyze.log 2>&1
echo "ALL DONE $(date '+%F %T')"

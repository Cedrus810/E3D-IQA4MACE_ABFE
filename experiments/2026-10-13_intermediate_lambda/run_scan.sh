#!/bin/bash
# Remainder scans of the operating head (thermal200 N+es, v3 S13.35) on the
# lambda_elec test sets. Waits for each set; rerun skips finished logs.
set -euo pipefail
cd /home/ruigengji/E3D
X=experiments/2026-10-13_intermediate_lambda
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
CK=runs/polar-M/ckpt/joint_organic_n50000_c10000_thermal200_solvent_es_h256_s60000_N.pt
for L in 0.5 0.0; do
  f=$X/lam${L}_test_solvent.npz
  until [ -f "$f" ]; do sleep 10; done
  [ -s "$X/scale_N_es_thermal200_lam${L}.log" ] && grep -q "sd/kT" "$X/scale_N_es_thermal200_lam${L}.log" && continue
  E3D_ES=1 $P -u test_scale.py MACE-POLAR-1-M "$CK" 5,10,20,50,100 "$f" > "$X/scale_N_es_thermal200_lam${L}.log" 2>&1
done
echo "=== scans done"

#!/bin/zsh
#PBS -q default
#PBS -l select=1:ncpus=16:ngpus=1:mem=60gb:host=yayoi08
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N e3d_checks
# Separating checks for the +27 kJ/mol E3D - MM offset (v3 S13.51), on yayoi08 gpu0 (RTX 3090),
# one after another (one ladder process per GPU, S13.48):
#   1. dG_ref for acetamide's box (no salt, 29.9 A): gives acetamide its full E3D value, the one
#      molecule with an all-MM path too
#   2. dG_ref for HSP90-27's box: is one dG_ref good enough for six IBS boxes?
#   3. longer A segment (W1 + W2 + cut, 15 states) for HSP90-27, the largest offset (+46.7):
#      continues from v2_full_HSP90_ligand-27's states, 40 more iterations (5 -> 25 ps per state)
# Time (basis: this 3090, 284 s per dG_ref iteration = 0.19 s per replica step, S13.51; the full ML
# region assumed ~0.2 s per step here, not measured): 1 + 2 ~ 2 x 1.7 h; 3 ~ 15 x 250 x 0.2 s
# ~ 12.5 min per iteration x 40 ~ 8 h. Resumable: resubmitting skips finished parts.
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev

cd /home/ruigengji/E3D
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 JAX_PLATFORMS=cpu
D=experiments/2026-10-18_checks
S=experiments/2026-10-17_set_run
CK="--ckpt runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt --backbone /home/ruigengji/MLP/mace/MACE-POLAR-1-M.model --n-inner 20 --dt 2.0 --save-every 1"
F="warning|visitor|detach|as_tensor|^\*|jax"
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

export CUDA_VISIBLE_DEVICES=0                      # gpu0 (user, 2026-10-18)
python -u run_abfe.py --mode mlmm --ref --leg-npz experiments/2026-10-15_acetamide_leg/em_acetamide.npz --T 298.15 \
  --qs 3,3.2,3.4,3.6,3.8,4 --init-from $S/v2_tail_acetamide --init-q 5 \
  --equil 2 --iters 20 --steps-per-iter 250 $=CK \
  --out-dir $D/ref_acetamide 2>&1 | grep -viE "$F" > $D/ref_acetamide.log
python -u run_abfe.py --mode mlmm --ref --leg-npz experiments/2026-10-17_ibs_legs/HSP90_ligand-27.npz --T 300 \
  --qs 3,3.2,3.4,3.6,3.8,4 --init-from $S/v2_full_HSP90_ligand-27 --init-q 5 \
  --equil 2 --iters 20 --steps-per-iter 250 $=CK \
  --out-dir $D/ref_HSP90_ligand-27 2>&1 | grep -viE "$F" > $D/ref_HSP90_ligand-27.log
python -u run_abfe.py --mode mlmm --leg-npz experiments/2026-10-17_ibs_legs/HSP90_ligand-27.npz --T 300 \
  --qs 0,1,1.2,1.4,1.6,1.8,1.9,1.94,1.97,1.99,2,2.25,2.5,2.75,3 \
  --init-from $S/v2_full_HSP90_ligand-27 --equil 0 --iters 40 --steps-per-iter 250 $=CK \
  --out-dir $D/longA_HSP90_ligand-27 2>&1 | grep -viE "$F" > $D/longA_HSP90_ligand-27.log
echo "=== done $(date '+%F %T')"

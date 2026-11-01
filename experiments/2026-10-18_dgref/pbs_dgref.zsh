#!/bin/zsh
#PBS -q default
#PBS -l select=1:ncpus=16:ngpus=1:mem=60gb:host=yayoi08
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -N e3d_dgref
# D18's neat-solvent reference leg dG_ref (v3 S6.3.4, S13.50) on the PWWP1-8 box: the n = 20 inner
# waters switched ML -> MM (q 3 -> 4, 6 states) with the ligand decoupled, the wall off and FIRES
# around one ghost atom. Starts from the refilled q = 5 state of the PWWP1-8 SW + W3 tail. The six
# IBS boxes are alike (37.9-40.9 A, 150 mM NaCl), so this dG_ref serves them for a first comparison;
# acetamide (no salt, 29.9 A) needs its own.
# Time: smoke on a 2080 Ti 0.27 s per replica step -> 6 x 250 x 0.27 s ~ 7 min per iteration,
# 22 iterations ~ 2.5 h; a 3090 should be faster. Resumable: resubmitting continues from the checkpoint.
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev

cd /home/ruigengji/E3D
export CUDA_VISIBLE_DEVICES=0                      # gpu0 (user, 2026-10-18)
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 JAX_PLATFORMS=cpu
D=experiments/2026-10-18_dgref
echo "=== $(date '+%F %T')  host $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
python -u run_abfe.py --mode mlmm --ref --leg-npz experiments/2026-10-17_ibs_legs/PWWP1_ligand-8.npz --T 300 \
  --qs 3,3.2,3.4,3.6,3.8,4 \
  --init-from experiments/2026-10-17_set_run/v2_tail_PWWP1_ligand-8 --init-q 5 \
  --ckpt runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt \
  --backbone /home/ruigengji/MLP/mace/MACE-POLAR-1-M.model --n-inner 20 \
  --equil 2 --iters 20 --steps-per-iter 250 --dt 2.0 --save-every 1 \
  --out-dir $D/ref_PWWP1_ligand-8 2>&1 | grep -viE "warning|visitor|detach|as_tensor|^\*|jax" | tee $D/ref_PWWP1_ligand-8.log
echo "=== done $(date '+%F %T')"

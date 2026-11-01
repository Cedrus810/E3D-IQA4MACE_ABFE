#!/bin/zsh
#PBS -q default
#PBS -l select=1:ncpus=32:ngpus=2:mem=100gb:host=yayoi01
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N e3d_d19_acet
# D19 acetamide full ladder (PLAN.md S2), 20 ps per state: 2 equil + 20 production iterations x
# 500 steps x 2 fs. Split at q = 2 into two ladders, one per GPU (one ladder process per GPU,
# v3 S13.48): segA = W1 + W2 (q 0-2, 15 states) on gpu0, segB = cut + W3 (q 2-4, 16 states) on gpu1.
# Total = MBAR(segA) + MBAR(segB), joined at q = 2 as in v3 S13.51.
# Both start from full/ (1 D19 equil iteration on 180, 2026-10-19), nearest q per state.
# Resumable: resubmit the same script; --init-from only acts on the first start.
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev

cd /home/ruigengji/E3D
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 JAX_PLATFORMS=cpu
D=experiments/2026-10-19_d19_acetamide
QA=0,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.94,1.97,1.99,2
QB=2,2.25,2.5,2.75,3,3.2,3.4,3.55,3.7,3.8,3.86,3.91,3.95,3.975,3.99,4
CK=(--mode mlmm --shell-mm
    --leg-npz experiments/2026-10-15_acetamide_leg/em_acetamide.npz
    --ckpt runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt
    --backbone /home/ruigengji/MLP/mace/MACE-POLAR-1-M.model --n-inner 20
    --equil 2 --iters 20 --steps-per-iter 500 --dt 2.0 --seed 0 --init-from $D/full)
F="warning|visitor|detach|as_tensor|^\*|jax"
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
python -c "import torch; n = torch.cuda.device_count(); print('cuda devices', n); assert n >= 2" || exit 1

CUDA_VISIBLE_DEVICES=0 python -u run_abfe.py $CK --qs $QA --out-dir $D/segA 2>&1 | grep -viE "$F" > $D/segA.log &
CUDA_VISIBLE_DEVICES=1 python -u run_abfe.py $CK --qs $QB --out-dir $D/segB 2>&1 | grep -viE "$F" > $D/segB.log &
wait
echo "=== done $(date '+%F %T')"

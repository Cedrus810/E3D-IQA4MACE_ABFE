#!/bin/zsh
#PBS -q default
#PBS -l select=1:ncpus=16:ngpus=1:mem=60gb:host=yayoi30
#PBS -l walltime=8:00:00
#PBS -j oe
#PBS -N e3d_paired
# Step 3 of the D19 review (v3 S13.57 end): 4 starts x {old path, D19} at q = 1, 10 ps each,
# one process on the one GPU. Resumable: resubmit; finished trajectories are skipped.
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev

cd /home/ruigengji/E3D
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 JAX_PLATFORMS=cpu
D=experiments/2026-10-21_paired_sampling
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
# stop at once if the node has no usable GPU (yayoi26 failed that way on 2026-10-20)
python -c "import torch; assert torch.cuda.device_count() >= 1, 'no CUDA GPU'; print(torch.cuda.get_device_name())" || exit 1
python -u $D/paired_sampling.py 2>&1 | grep --line-buffered -viE "warning|visitor|detach|as_tensor|torch.load|^\*|jax" > $D/paired.log
echo "=== done $(date '+%F %T')"

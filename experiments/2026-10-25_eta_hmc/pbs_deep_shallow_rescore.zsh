#!/bin/zsh
# 8 次保存构型重算；无 MD。尚未提交。
#PBS -q default
#PBS -l select=1:ncpus=8:ngpus=1:mem=20gb:host=yayoi29
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -N e3d_deep_rescore

set -e
if [[ -n ${PBS_O_WORKDIR:-} ]]; then
  cd "$PBS_O_WORKDIR"
fi
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev
cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
RUN_LOGS=$(mktemp -d "logs/deep_rescore_${PBS_JOBID:-manual}_XXXXXXXX")
print "Start $(date '+%F %T') on $(hostname); outputs: $RUN_LOGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$P" -u -B deep_shallow_panel.py rescore \
  --panel logs/deep_shallow_panel_20261031.npz --out "$RUN_LOGS/result.json" \
  > "$RUN_LOGS/rescore.log" 2>&1
print "All eight source caches reproduced; outputs: $RUN_LOGS"

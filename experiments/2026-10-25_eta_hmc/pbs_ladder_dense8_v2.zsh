#!/bin/zsh
# 加密布点的有界对照；方案见 ETA_DENSIFICATION_2026-10-30.md。
# 每卡同时一个进程。此脚本只准备，尚未提交。
#PBS -q default
#PBS -l select=1:ncpus=16:ngpus=2:mem=80gb:host=yayoi10
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -N e3d_dense8_v2

# 不使用 set -u：现有 mamba 激活脚本含未定义变量。
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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
GPUS=(${(s:,:)${CUDA_VISIBLE_DEVICES:-0,1}})
if (( ${#GPUS[@]} < 2 )); then
  print -u2 'Expected two allocated CUDA devices'
  exit 1
fi
RUN_LOGS=$(mktemp -d "logs/dense8_v2_${PBS_JOBID:-manual}_XXXXXXXX")
print "Start $(date '+%F %T') on $(hostname); logs: $RUN_LOGS; GPUs: ${GPUS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

typeset -a worker_pids
for slot in 1 2; do
  (
    export CUDA_VISIBLE_DEVICES=${GPUS[$slot]}
    for r in $slot $((slot + 2)); do
      "$P" -u -B eta_exchange.py ladder CyclophilinD_ligand-8 \
        --etas 0,0.125,0.25,0.5,0.625,0.75,0.875,1 --start "$r" --init last \
        --sweeps 500 --tag _ex_dense8_last_v2 --head "$HEAD" \
        --checkpoint-every 50 --coords-every 10 \
        > "$RUN_LOGS/dense8_start$r.log" 2>&1 || exit $?
      "$P" -u -B eta_exchange.py ladder CyclophilinD_ligand-8 \
        --etas 0,0.25,0.5,1 --start "$r" --init last \
        --sweeps 1001 --tag _ex_coarse4_last_v2 --head "$HEAD" \
        --checkpoint-every 50 --coords-every 10 \
        > "$RUN_LOGS/coarse4_start$r.log" 2>&1 || exit $?
    done
  ) &
  worker_pids+=($!)
done

run_status=0
for child_pid in ${worker_pids[@]}; do
  if wait "$child_pid"; then
    print "Worker $child_pid completed"
  else
    child_status=$?
    print -u2 "Worker $child_pid failed: exit $child_status; logs: $RUN_LOGS"
    run_status=1
  fi
done
if (( run_status )); then
  exit "$run_status"
fi
print "All eight ladders completed successfully $(date '+%F %T'); logs: $RUN_LOGS"

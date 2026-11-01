#!/bin/zsh
# 单次续长预算：只将已完成的四个 _dense8_v1 从 500 续到 1000 sweep。
# 方案：ETA_CONTINUATION_2026-10-31.md。尚未提交。
#PBS -q default
#PBS -l select=1:ncpus=32:ngpus=1:mem=60gb:host=yayoi32
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -N e3d_dense8_x1000

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
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
RUN_LOGS=$(mktemp -d "logs/dense8_x1000_${PBS_JOBID:-manual}_XXXXXXXX")
print "Start $(date '+%F %T') on $(hostname); logs: $RUN_LOGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

typeset -a worker_pids
for pair in "1 3" "2 4"; do
  (
    for r in ${=pair}; do
      "$P" -u -B eta_exchange.py ladder CyclophilinD_ligand-8 \
        --etas 0,0.125,0.25,0.5,0.625,0.75,0.875,1 --start "$r" --init last \
        --sweeps 1000 --tag _dense8_x1000_v1 --extend-from _dense8_v1 --head "$HEAD" \
        --checkpoint-every 50 --coords-every 10 \
        > "$RUN_LOGS/start$r.log" 2>&1 || exit $?
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
print "All four continuations completed successfully $(date '+%F %T'); logs: $RUN_LOGS"

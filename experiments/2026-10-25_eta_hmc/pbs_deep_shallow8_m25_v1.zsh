#!/bin/zsh
# η=0 m=25 对照（ETA0_MM_UPDATES_DECISION_2026-11-01.md）：同一八点梯子，深／浅初态各四个随机重复，η=0 每 sweep 25 次 MM-HMC 子步。
#PBS -q default
#PBS -l select=1:ncpus=32:ngpus=2:mem=100gb:host=yayoi08
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -N e3d_deepshallow8_m25

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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1    # 4 workers x 8 = ncpus 32
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
PANEL=/home/ruigengji/E3D/experiments/2026-10-25_eta_hmc/logs/deep_shallow_panel_20261031.npz
RUN_LOGS=$(mktemp -d "logs/deepshallow8_m25_${PBS_JOBID:-manual}_XXXXXXXX")
print "Start $(date '+%F %T') on $(hostname); logs: $RUN_LOGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

typeset -a worker_pids gpus
# two GPUs, two workers each: worker = one rep (its deep then its shallow ladder), rep r on GPU (r-1)%2
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  gpus=(${(s:,:)CUDA_VISIBLE_DEVICES})
else
  gpus=(0 1)
fi
if (( ${#gpus} < 2 )); then
  print -u2 "need 2 GPUs, got: ${gpus[*]}"
  exit 1
fi
for r in 1 2 3 4; do
  dev=${gpus[$(( (r - 1) % 2 + 1 ))]}
  (
    export CUDA_VISIBLE_DEVICES=$dev
    for branch in deep shallow; do
      if [[ "$branch" == deep ]]; then
        frame_index=0
        stream_base=$((500000 + 1000*r))
      else
        frame_index=$((r + 3))
        stream_base=$((600000 + 1000*r))
      fi
      "$P" -u -B eta_exchange.py ladder CyclophilinD_ligand-8 \
        --etas 0,0.125,0.25,0.5,0.625,0.75,0.875,1 --start "$r" \
        --init panel --init-panel "$PANEL" --init-frame "$frame_index" --seed-base "$stream_base" \
        --sweeps 500 --tag "_start${branch}8_m25_v1" --eta0-mm-updates 25 --head "$HEAD" \
        --checkpoint-every 50 --coords-every 10 \
        > "$RUN_LOGS/${branch}_start$r.log" 2>&1 || exit $?
    done
  ) &
  worker_pids+=($!)
  print "worker rep $r: pid $! on GPU $dev"
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
print "All eight m=25 initial-state ladders completed successfully $(date '+%F %T'); logs: $RUN_LOGS"

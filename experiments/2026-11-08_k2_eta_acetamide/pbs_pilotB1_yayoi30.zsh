#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）
#PBS -l select=1:ncpus=32:ngpus=1:mem=60gb:host=yayoi30
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=24:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N k2B1_30
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# module load CUDA                                         # ← 只有 pyscf-env 需要（用户印象，未核实），其他环境别加

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc     # ← 项目目录（k2_exchange.py 写 runs/acetamide/）
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8                 # ← 2 进程 × 8 = 16 = 5950X 物理核
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← 运行命令：试点 B1（--eta0-mm-updates 1），prep 3 4 各一个 11 标签梯子，同时跑；重提同一脚本即续跑
HC=/home/ruigengji/E3D/experiments/2026-11-07_head_calibration
LG=/home/ruigengji/E3D/experiments/2026-11-08_k2_eta_acetamide/logs/pilotB
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
BB=/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model
for p in 3 4; do
  ( $P -u k2_exchange.py acetamide --etas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 --start-file $HC/mm/prep${p}_S0.npz \
      --start-label prep$p --head $HEAD --backbone $BB --init last --sweeps 500 --checkpoint-every 50 --coords-every 10 \
      --eta0-mm-updates 1 --tag _pilotB1 > $LG/B1_prep$p.log 2>&1 || echo "FAILED B1 prep$p" ) &
done
wait
echo "=== done $(date '+%F %T') ==="

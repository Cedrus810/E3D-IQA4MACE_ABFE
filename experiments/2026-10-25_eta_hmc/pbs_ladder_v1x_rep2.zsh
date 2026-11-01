#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）。yayoi10（GroupA，EPYC 7452，2×RTX 3090，空 16/32 核；本作业 1 卡）
#PBS -l select=1:ncpus=8:ngpus=1:mem=30gb:host=yayoi10
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=02:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N e3d_ladder_v1x_rep2
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# module load CUDA                                         # ← 只有 pyscf-env 需要（用户印象，未核实），其他环境别加

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc     # ← 项目目录
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8                 # ← 1 进程 × 8 线程 = ncpus 8
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← 运行命令：rep2 交换梯子 _ex_v1 续到 1000 sweep（新 tag _ex_v1x1000，源文件不动）。上次被续接复查拦下，
#   2026-10-30 修正 hmc（缓存坐标不再吸收读回舍入）与复查容差后重交。检查点来自 yayoi42（4080 SUPER）；在 3090 上加载成功过，在 2080 Ti 上失败。
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
$P -u -B eta_exchange.py ladder CyclophilinD_ligand-8 --etas 0,0.25,0.5,1 --start 2 --sweeps 1000 \
  --tag _ex_v1x1000 --extend-from _ex_v1 --head $HEAD > logs/ladder_ex_v1x1000_start2.log 2>&1
wait
echo "=== done $(date '+%F %T') ==="

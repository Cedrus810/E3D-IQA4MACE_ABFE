#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）
#PBS -l select=1:ncpus=32:ngpus=2:mem=100gb:host=yayoi10
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=03:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N e3d_nvarb_L5
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# module load CUDA                                         # ← 只有 pyscf-env 需要（用户印象，未核实），其他环境别加

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc     # ← 项目目录
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8                 # ← 4 进程 × 8 线程 = ncpus 32
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← 运行命令：nvar b 冻结，CypD-8 S0，eta 1.0 × 起点 rep1–4，L=5×2 fs（10 fs），各 500 循环（与 _nvarb_ms 的 L=20 链模型调用数相同）；两卡各 2 进程。续跑：重新 qsub。
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
i=0
for s in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$(( i % 2 )) $P -u -B eta_hmc.py chain CyclophilinD_ligand-8 --eta 1.0 --start $s --cycles 500 --L 5 \
    --tag _nvarb_L5 --head $HEAD --log-every 50 > logs/nvarb_L5_eta1.0_start${s}.log 2>&1 &
  i=$(( i + 1 ))
done
wait
$P -u -B eta_hmc.py analyze CyclophilinD_ligand-8 --tag _nvarb_L5 > logs/nvarb_L5_analyze.log 2>&1
echo "=== done $(date '+%F %T') ==="

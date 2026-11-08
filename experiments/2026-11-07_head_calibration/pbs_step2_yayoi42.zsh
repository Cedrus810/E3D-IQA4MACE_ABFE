#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）
#PBS -l select=1:ncpus=20:ngpus=1:mem=60gb:host=yayoi42
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=4:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N hcal2_42
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# module load CUDA                                         # ← 只有 pyscf-env 需要（用户印象，未核实），其他环境别加

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/E3D/experiments/2026-11-07_head_calibration   # ← 项目目录
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8                 # ← 2 条链 × 8 线程 ≤ ncpus 20
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← 运行命令：prep 9 10 各一条链，链内先 MM 再评价，链之间并行；重提同一脚本即续跑
for p in 9 10; do
  ( $P -u step2_mm.py $p 500 2000 2 CUDA mm > mm/prep${p}_mm.log 2>&1 && \
    $P -u step2_eval.py mm/prep${p}_S0.npz 5 cpu 8 > mm/prep${p}_eval.log 2>&1 ) &
done
wait
echo "=== done $(date '+%F %T') ==="

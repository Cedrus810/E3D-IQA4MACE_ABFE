#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）
#PBS -l select=1:ncpus=16:ngpus=1:mem=60gb:host=yayoi05
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=24:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N e3d_collect
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# module load CUDA                                         # ← 只有 pyscf-env 需要（用户印象，未核实），其他环境别加

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/E3D/experiments/2026-10-23_groupwise_dummy   # ← 项目目录
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3                 # ← 等于 ncpus（4 个 MM 进程 × 3 + 1 个 δU 进程 × 4 = 16）
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← 运行命令：
# (a) 新的独立重复 rep 2..21，两个体系，整条 S0..S4 链，每态 100 ps 平衡 + 500 ps 生产（groupwise_mm.py，rep1 已跑通）。
#     4 个进程共用 1 块 GPU；同一 rep 的已完成状态在重交时跳过。
# (b) 同时在 rep1 全部端点帧上算 δU 与 a_j（local_E.py，181 冒烟通过的 batch 4）。
L=/home/ruigengji/E3D/experiments/2026-10-24_local_E/logs
mkdir -p $L
for w in 0 1 2 3; do
  ( for r in {2..21}; do
      (( (r - 2) % 4 == w )) || continue
      for s in PWWP1_ligand-8 CyclophilinD_ligand-8; do
        $P -B -u groupwise_mm.py run $s $r 100 500 >> $L/pbs_mm_w$w.log 2>&1
      done
    done ) &
done
( cd /home/ruigengji/E3D/experiments/2026-10-24_local_E && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 &&
  for s in PWWP1_ligand-8 CyclophilinD_ligand-8; do
    $P -B -u local_E.py eval $s --batch 4 && $P -B -u local_E.py analyze $s
  done >> $L/pbs_dU_rep1.log 2>&1 ) &
wait
for s in PWWP1_ligand-8 CyclophilinD_ligand-8; do
  $P -B -u groupwise_mm.py analyze $s >> $L/pbs_mm_analyze.log 2>&1
done
echo "=== done $(date '+%F %T') ==="

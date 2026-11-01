#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）。yayoi42（GroupE，i7-12700K 20 线程，1×RTX 4080 SUPER 16GB，128GB）
#PBS -l select=1:ncpus=16:ngpus=1:mem=60gb:host=yayoi42
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=08:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N e3d_ladder_v1
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# module load CUDA                                         # ← 只有 pyscf-env 需要（用户印象，未核实），其他环境别加

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc     # ← 项目目录
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8                 # ← 2 进程 × 8 线程 = ncpus 16
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← 运行命令：nvar b 冻结，CypD-8 S0，eta {0,0.25,0.5,1}，L=20×2 fs，500 sweep；同卡 2 进程（16 GB 显存，每梯子进程实测约 4.3 GB），
#   进程 1 跑 rep1、rep3，进程 2 跑 rep2、rep4；每个 rep 先交换梯子 _ex_v1、再无交换对照 _ctl_v1（同标签、同核、同起始帧、同调用数）。续跑：重新 qsub 同一脚本。
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
for pair in "1 3" "2 4"; do
  (
  for r in ${=pair}; do
    $P -u -B eta_exchange.py ladder CyclophilinD_ligand-8 --etas 0,0.25,0.5,1 --start $r --sweeps 500 \
      --tag _ex_v1 --head $HEAD > logs/ladder_ex_v1_start$r.log 2>&1
    $P -u -B eta_exchange.py ladder CyclophilinD_ligand-8 --etas 0,0.25,0.5,1 --start $r --sweeps 500 \
      --tag _ctl_v1 --head $HEAD --no-exchange > logs/ladder_ctl_v1_start$r.log 2>&1
  done
  ) &
done
wait
echo "=== done $(date '+%F %T') ==="

#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）。yayoi32（GroupE，Ryzen 9950X，1×RTX 4080 SUPER 16GB，64GB）
#PBS -l select=1:ncpus=32:ngpus=1:mem=60gb:host=yayoi32
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=08:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N e3d_dense8_v1
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# module load CUDA                                         # ← 只有 pyscf-env 需要（用户印象，未核实），其他环境别加

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/E3D/experiments/2026-10-25_eta_hmc     # ← 项目目录
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16               # ← 2 进程 × 16 线程 = ncpus 32
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← 运行命令（用户 2026-10-30 指定的有界加密对照）：nvar b，CypD-8 S0，L=20×2 fs；两协议的全部 walker 从该 rep 的 MM 末帧新开（--init last）。
#   加密梯子 _dense8_v1：eta {0,0.125,0.25,0.5,0.625,0.75,0.875,1}，500 sweep，8×501 = 4008 次模型调用；
#   对照 _ex4last_v1：原 4 点交换梯子 eta {0,0.25,0.5,1}，1001 sweep，4×1002 = 4008 次。
#   4 进程各负责一个 rep（先加密、后对照），每卡 2 进程；GPU 取 PBS 分配的 CUDA_VISIBLE_DEVICES。续跑：重新 qsub。
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
# 全部 8 个梯子在本作业（yayoi32，1 卡 16 GB）：同卡 2 进程，进程 1 跑 rep1、rep3，进程 2 跑 rep2、rep4；每个 rep 先加密 _dense8_v1 再对照 _ex4last_v1。
for pair in "1 3" "2 4"; do
  (
  for r in ${=pair}; do
    $P -u -B eta_exchange.py ladder CyclophilinD_ligand-8 --etas 0,0.125,0.25,0.5,0.625,0.75,0.875,1 --start $r \
      --init last --sweeps 500 --tag _dense8_v1 --head $HEAD > logs/ladder_dense8_v1_start$r.log 2>&1
    $P -u -B eta_exchange.py ladder CyclophilinD_ligand-8 --etas 0,0.25,0.5,1 --start $r \
      --init last --sweeps 1001 --tag _ex4last_v1 --head $HEAD > logs/ladder_ex4last_v1_start$r.log 2>&1
  done
  ) &
done
wait
echo "=== done $(date '+%F %T') ==="

#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）。yayoi29（GroupD，Ryzen 5950X，1×RTX 3090，64GB）
#PBS -l select=1:ncpus=32:ngpus=1:mem=50gb:host=yayoi29
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=03:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N e3d_dense8_v1x
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

# ← 运行命令（用户 2026-10-31 定）：4 条加密梯子 _dense8_v1（500 sweep）续到 1000 sweep，新 tag _dense8_v1x1000，源文件不动；
#   布点 eta {0,0.125,0.25,0.5,0.625,0.75,0.875,1}、nvar b、L=20×2 fs 不变；粗梯子对照不重跑。
#   检查点写于 yayoi32（RTX 4080 SUPER）：同型号 GPU 加载检查点，否则自动从缓存坐标续接（记录在 meta continuations）。
#   同卡 2 进程，进程 1 跑 rep1、rep3，进程 2 跑 rep2、rep4。续跑：重新 qsub。
HEAD=/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt
for pair in "1 3" "2 4"; do
  (
  for r in ${=pair}; do
    $P -u -B eta_exchange.py ladder CyclophilinD_ligand-8 --etas 0,0.125,0.25,0.5,0.625,0.75,0.875,1 --start $r \
      --init last --sweeps 1000 --tag _dense8_v1x1000 --extend-from _dense8_v1 --head $HEAD \
      > logs/ladder_dense8_v1x1000_start$r.log 2>&1
  done
  ) &
done
wait
echo "=== done $(date '+%F %T') ==="

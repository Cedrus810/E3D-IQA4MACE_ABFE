#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）
#PBS -l select=1:ncpus=32:ngpus=1:mem=60gb:host=yayoi18
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=72:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N e3d_nvar
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
. /home/apps/Modules/init/profile.sh
export MODULEPATH=/home/ruigengji/modulefiles:$MODULEPATH
# module load CUDA                                         # ← 只有 pyscf-env 需要（用户印象，未核实），其他环境别加

export MAMBA_EXE=/home/ruigengji/miniforge3/bin/mamba
export MAMBA_ROOT_PREFIX=/home/ruigengji/miniforge3
source /home/ruigengji/miniforge3/etc/profile.d/mamba.sh
mamba activate openmm_dev                                  # ← 环境名

cd /home/ruigengji/E3D                                     # ← 项目目录
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32               # ← 等于 ncpus
P=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python    # ← 和环境名一起改
echo "=== $(date '+%F %T')  host $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ← 运行命令：RESULT.md「方差归一化」两臂串行，a = 60k 步（只改归一化），b = 104k 步（再补回被稀释的步数）；
#   训练目标缓存与 v2 共用（runs/polar-M/data/targets_..._resp_v2.pt），各臂训完接 test_scale 扫描
N='warning|visitor|cuequiv|detach|^  return|^  print|^  model|_Jd|Consider'
H=experiments/2026-10-25_head_retrain
X=experiments/2026-10-13_intermediate_lambda
for ST in 60000 104000; do
  CK=runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s${ST}_nvar_N.pt
  echo "=== arm nvar s$ST $(date '+%F %T') ==="
  E3D_ES=1 E3D_RESP=$H/data/resp_v2.npz E3D_RESP_NORM=var STEPS=$ST CLUSTERS=10000 FRAMES=$X/lampool400_solvent.npz CLUSTER_MAX=24 TAGS=M ARMS=N ./run_polar.sh
  [ -f "$CK" ] || { echo "missing $CK -- training did not finish"; continue; }
  for f in experiments/2026-10-12_thermal_frames/thermal_test_solvent.npz $X/lam0.5_test_solvent.npz $X/lam0.0_test_solvent.npz frames_test_solvent.npz; do
    b=${f:t:r}
    echo "--- scan s$ST $b $(date '+%T')"
    E3D_ES=1 $P -u test_scale.py MACE-POLAR-1-M "$CK" 5,10,20,50,100 "$f" 2>&1 | grep -viE --line-buffered "$N" | tee runs/polar-M/logs/scale_N_es_resp_v2_s${ST}_nvar_${b}.log
  done
done
echo "=== done $(date '+%F %T') ==="

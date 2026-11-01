#!/bin/zsh
# PBS 模板。只改标了 ← 的几处，其余不动。来源：/home/ruigengji/E3D/learn_thermal.zsh 头部（跑通过）。
# 不加 set -u（mamba activate 遇到未定义变量会退出）；不夹带冒烟/测试步骤。
# 提交：qsub job.zsh
# 节点备注：yayoi07 物理 5×RTX A4000 16GB，PBS 只登记 ngpus=4、ncpus=32、mem≈126GB（2026-10-23 pbsnodes + nvidia-smi 核实）
# #PBS 行末尾不能带注释（qsub 报 directive error，2026-10-23 实测），要改的项说明写在上一行：
# ← -q：walltime > 72h 改成 week
#PBS -q default
# ← select：host 填空节点；ncpus/ngpus/mem 按节点改（纯 CPU 节点去掉 ngpus）
#PBS -l select=1:ncpus=32:ngpus=1:mem=60gb:host=yayoi28
# ← walltime：default 最多 72h；week 最多 144h
#PBS -l walltime=72:00:00
#PBS -j oe
# ← -N：作业名
#PBS -N e3d_resp
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

# ← 运行命令：PLAN.md §1 数据（rep5-21 × 30 帧 × 4 时间点，按 rep 续跑）→ pack → arm N 重训（learn_lampool 配方 + E3D_RESP）→ test_scale 扫描
H=experiments/2026-10-25_head_retrain
N='warning|visitor|cuequiv|detach|^  return|^  print|^  model|_Jd|Consider'
STEM=joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s60000
CK=runs/polar-M/ckpt/${STEM}_N.pt
X=experiments/2026-10-13_intermediate_lambda
{ for S in CyclophilinD_ligand-8 PWWP1_ligand-8; do $P -u $H/make_data.py gen $S --reps 5-21 --frames 30 --batch 1 || exit 1; done } 2>&1 | grep -viE --line-buffered "$N" | tee -a $H/logs/data_v2.log
$P -u $H/make_data.py pack v2 CyclophilinD_ligand-8:5-21 PWWP1_ligand-8:5-21 2>&1 | tee -a $H/logs/data_v2.log
[ -f $H/data/resp_v2.npz ] || { echo "missing $H/data/resp_v2.npz"; exit 1; }
echo "=== train $(date '+%F %T') ==="
E3D_ES=1 E3D_RESP=$H/data/resp_v2.npz CLUSTERS=10000 FRAMES=$X/lampool400_solvent.npz CLUSTER_MAX=24 TAGS=M ARMS=N ./run_polar.sh
[ -f "$CK" ] || { echo "missing $CK -- training did not finish"; exit 1; }
for f in experiments/2026-10-12_thermal_frames/thermal_test_solvent.npz $X/lam0.5_test_solvent.npz $X/lam0.0_test_solvent.npz frames_test_solvent.npz; do
  b=${f:t:r}
  echo "--- scan $b $(date '+%T')"
  E3D_ES=1 $P -u test_scale.py MACE-POLAR-1-M "$CK" 5,10,20,50,100 "$f" 2>&1 | grep -viE --line-buffered "$N" | tee runs/polar-M/logs/scale_N_es_resp_v2_${b}.log
done
echo "=== done $(date '+%F %T') ==="

# D19 acetamide 完整 ladder：计划表（2026-10-19，待批准，未启动）

**2026-10-19 记录**：按用户要求在 180（2080 Ti）上整条 30 态 ladder 启动（`run_full.sh`，`full.log`）。
第 1 次平衡迭代用时 2,877 s，即每个副本每步 0.19 s（冒烟估的是 0.28 s）；55 次迭代合计约 44 h。
用户判断 44 h 没有意义，跑完 1 次迭代后按 PID 杀掉（128061 / 128126）。`full/` 里的 checkpoint 只有 1 次平衡迭代。

**2026-10-20**：改在 yayoi01（两块 3090）上，以 q=2 为界分成 segA / segB 两条，每态 20 ps（`pbs_yayoi01.zsh`），
07:58–23:11 跑完。结果：MBAR +67.05 ± 2.54 kJ/mol，相对 MM 高 +33.0。详见 v3 §13.56。

依据：v3 §6.3.4a（D19 方案 3）、§13.53（代码、精确检查、冒烟）、§13.48（E8 全 MM 路径对照）。

## 1. 这一步推进什么

- D19 路径的终点和 E8 完全相同（气相配体 + 纯 TIP3P），所以 E3D(D19) 与 MM 参考值 +34.04 ± 0.14 kJ/mol（§13.44）之差，只剩耦合态那一处：
  - L–I 耦合：20 个水上的 POLAR 对 GAFF；
  - 配体内能：ML 对 GAFF。
- 同时拿到：
  - 各段与 E8 逐段对比（E8 的 W2 为 +57.4，W3 为 −24.5，§13.48）；
  - cut 占总量的比例；
  - 逐原子求和是否闭合；
  - 旧路径 A 段 +75.4（§13.51）在新路径下变成多少。
- 这是对比，不是门：不设通过阈值。

## 2. Ladder（30 个态，q ∈ [0, 4]）

| 段 | q | 来源 |
|---|---|---|
| W1 | 0, 1 | E8 `qs.txt` |
| W2 | 1, 1.1, …, 1.9, 1.94, 1.97, 1.99, 2 | E8 `qs.txt`（λ→0 端已加密，§13.48） |
| cut | 2, 2.25, 2.5, 2.75, 3 | 与 IBS 首轮的 `qs_full.txt` 相同；acetamide 首轮在这套 ladder 上，全局最小 overlap 为 0.116，dG_cut 为 +0.09 |
| W3 | 3, 3.2, 3.4, 3.55, 3.7, 3.8, 3.86, 3.91, 3.95, 3.975, 3.99, 4 | E8 `qs.txt`（s_w = 4 − q，映射与 D19 相同） |

    QS=0,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.94,1.97,1.99,2,2.25,2.5,2.75,3,3.2,3.4,3.55,3.7,3.8,3.86,3.91,3.95,3.975,3.99,4

## 3. 起点与采样

- **起点**：`--init-from experiments/2026-10-16_mm_path_control/fires_seed1`。
  - 这是 E8 带 FIRES 的那条 run：220 次迭代，每态 200 ps，起始帧同为 acetamide 腿的坐标，所以内层水集合相同。
  - 每个态取 E8 里 q 最近的构型，cut 的中间态取 q = 2 或 3 的构型。
  - W3 的空腔已经在 MM 下回填平衡过，绕开了 §13.48 里"从耦合帧起跑，W3 回填要 0.5–1 ns"的问题。
- **采样**：5 次平衡迭代 + 50 次生产迭代，每次 500 步 × 2 fs。每个态 50 ps、50 个样本。
- **选 50 ps 的依据**：
  - §13.51 里 A 段每态 5 ps，MBAR 误差为 ±2.8–4.5；按 1/√t 缩放到 50 ps，约 ±1；
  - MBAR 误差约低估一半（§13.48 分块的散布是 MBAR 误差的 2 倍），所以实际约 ±2 kJ/mol；
  - 这个精度足以读出 E3D − MM（旧路径上约 +20 量级），但读不出 0.5 kJ/mol 量级的差别。
- 一个 seed（0）。种子重复不在本表内。

## 4. 耗时依据

- **2080 Ti 实测**：D19 冒烟（§13.53，181）每个副本每步 0.28 s，含 u_kn 计算。
  - 冒烟每次迭代只有 50 步，计算量的占比比生产时（500 步）高，所以 0.28 s 偏保守。
  - W3 的态只算配体的 ML，应该更便宜，但没有单独测过，一律按 0.28 计。
- **5090 推算**：acetamide 没在 5090 上测过。Atenolol n = 20 时 2080 Ti 与 5090 之比为 3.5（§13.45），据此推出约 0.08 s。【推算，未实测】

## 5. 两个方案

### 方案 A：5090 上跑一条 ladder（PBS，由你提交）

- 每次迭代 30 × 500 × 0.08 = 1,200 s；55 次共约 18 h【推算】。
- 生产 GPU 就是 5090（项目 CLAUDE.md）。

### 方案 B：三台 2080 Ti 并行，按段拆成三条 ladder

| ladder | q | 态数 | 机器 | 每次迭代 | 55 次 |
|---|---|---|---|---|---|
| A（W1 + W2） | 0 … 2 | 15 | 181 | 15 × 500 × 0.28 = 2,100 s | 32 h |
| B（cut） | 2 … 3 | 5 | 183 | 700 s | 10.7 h |
| C（W3） | 3 … 4 | 12 | 180 | 1,680 s（上限） | 25.7 h（上限） |

- 总量 dG = 三条 ladder 各自 dG_MBAR 之和。在 q = 2、3 两个边界，相邻两条 ladder 各有一份同一态的样本。§13.51 也是这样分段合并的。
- 墙钟时间约 32 h，由 A 决定。
- **与项目规定不同**：项目 CLAUDE.md 规定 2080 Ti 只用于冒烟，在 2080 Ti 上测的耗时也不能当生产耗时。方案 B 是在 2080 Ti 上跑接近生产规模的作业。原因：5090 要排 PBS。之前 IBS 首轮也在工作站上跑过。由你决定。

## 6. 机器当前负载（2026-10-19 19:02 查看）

| 机器 | GPU | CPU |
|---|---|---|
| 181 | 2080 Ti 0%，1 MiB | load 0.9；有别的窗口在跑 pytest（PID 2577370，CPU） |
| 180 | 2080 Ti 1%，368 MiB | load 0.00 |
| 183 | 2080 Ti 0%，1 MiB | load 0.00 |
| 5090（P4） | 看不到，由你查 | |

一块 GPU 只放一个 ladder 进程（§13.48：多个进程共用一块 GPU 会报 "No compatible CUDA device"）。

## 7. 命令

每条 ladder 都是同一条命令，只换 `--qs` 和 `--out-dir`。线程数：181、183 设 `OMP_NUM_THREADS=8`，180（8 核）设 4。

    cd /home/ruigengji/E3D
    set +u; source /home/ruigengji/miniforge3/etc/profile.d/conda.sh && conda activate openmm_dev; set -u
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 JAX_PLATFORMS=cpu
    python -u run_abfe.py --mode mlmm --shell-mm \
      --leg-npz experiments/2026-10-15_acetamide_leg/em_acetamide.npz \
      --ckpt runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt \
      --backbone /home/ruigengji/MLP/mace/MACE-POLAR-1-M.model --n-inner 20 \
      --qs "$QS" --equil 5 --iters 50 --steps-per-iter 500 --dt 2.0 --seed 0 \
      --init-from experiments/2026-10-16_mm_path_control/fires_seed1 \
      --out-dir experiments/2026-10-19_d19_acetamide/<name>

- 方案 A：`<name>` = `full`，`QS` 用第 2 节的整串。
- 方案 B 的三条：
  - `<name>` = `segA`，`QS=0,1,1.1,…,1.99,2`；
  - `<name>` = `segB`，`QS=2,2.25,2.5,2.75,3`；
  - `<name>` = `segC`，`QS=3,3.2,…,3.99,4`。
- 启动按全局规则第 3 章的写法：`setsid nohup … &`，PID 记进本目录的 `pids.txt`；远程命令用 `bash -c` 包起来。
- 能续跑：每次迭代都写 checkpoint，重跑同一条命令就从 checkpoint 接着跑；`--init-from` 只在第一次启动时生效。
- 长任务，按规则放在下班时间跑，时间由你定。

## 8. 跑完读什么

- dG_W1 / W2 / cut / W3、TI 与 MBAR、overlap_min、swap acceptance；
- 前后两半的 MBAR 值，看有没有漂移；
- 逐原子求和是否闭合；
- 和 E8 逐段比较，和 +34.04 比较；
- 结果写进 v3 §13 和 STATUS。

## 9. 不在本表内（之后另报计划）

- **有限 k 下 FIRES 的残余项**（§6.3.4a）：测法还没设计。
- **6 条 IBS 溶剂腿走 D19**：需要先写一段代码，把旧 run 的 q 映射到新路径（旧 W3 的 q 4…5 对应新的 3…4），才能复用已经回填好的构型。
- **种子重复**。

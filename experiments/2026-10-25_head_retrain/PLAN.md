# head 重训计划：加入配体内部形变响应监督（2026-10-25，计划稿，未实施）

本文件只写计划，不含代码改动；实施前等用户确认。

## 依据（实施前读完）

- `DESIGN_2026-10-23_groupwise_dummy.md` §7.1（local_E v1 的 E 构造）、§10（η 修正采样）。
- `MODEL_DEFINITION_from_v3.md` §2.2–2.3：当前 head 的配方（arm N，D6 显式 Coulomb，60k 步，≤24 水）。
- `experiments/2026-10-25_eta_hmc/RESULT.md` 最后四节：内部形变诊断、通道拆分、S_K 测试、按 T 重算。
- `test_joint.py`（训练脚本，`E3D_ES=1` 时标签减去 `rec["cross"]`，见 188–189 行）、`learn_lampool.zsh`（当前 head 的 PBS 配方）。

## 用户原话（2026-10-25）

- “可以重新训练”
- “训练标签必须与当前显式通道一致。令 X 为显式 Coulomb 的 cross，则：T(x)=C_bb(x)−X(x)，Σ_c D_c(x)→T(x)。
  并约束形变响应：Δ(Σ_c D_c)→ΔT。标签应是 C_bb−cross，不是 C_bb−d_es。后者会漏掉两者之间的密度响应项。
  四通道的大幅抵消也不能仅凭这组结果判为错误；应保留通道监督，修正合计响应，而非手动缩放通道。”
- “重训后的 head 需要新版本：虽然目标构造不换，E 的数值仍会改变，旧 η 链不能直接当作新目标样本。”
- “先验收耦合响应，再改提议势；暂不继续调 L、加密 η 或扩展复合物腿。”

## 现状（已核实）

- 当前 head：`runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt`。
- 它的簇数据来自 `experiments/2026-10-13_intermediate_lambda/lampool400_solvent.npz`：400 帧，配体 41 原子（单一配体），
  簇 ≤24 水，其中 25% 为配体–溶剂划分（`decomp/clusters.solvent_clusters` 的 `ligand_frac=0.25`）。CypD-8、PWWP1-8 未出现在训练中。
- local_E v1 在评价时用 50 水的 Ω，大于训练簇上限 24 水。
- CypD-8 S0 上，学习通道之和对配体内部形变的响应：相对 T 的相关 0.669、斜率 0.490，差的 sd 6.71 kJ/mol；
  合计水平：学习通道之和 −260.33 vs T −210.45 kJ/mol（`RESULT.md`）。
- 上次同配方训练耗时（`e3d_lampool.o13460`）：簇标签 642 s / 1 万簇，目标预计算 1215 s，60k 步 12563 s（约 3.5 h）。

## 计划

### 1. 新增训练数据（只用全耦合 S0 帧）

- 来源：`experiments/2026-10-23_groupwise_dummy/runs/{CyclophilinD_ligand-8,PWWP1_ligand-8}/rep{5..21}_S0.npz`。
  rep1–4 的 S0 全部留作验证（之前的诊断起点都在 rep1–4）。
- 簇：按 local_E v1 的规则截取（配体＋最近 50 个整水，离子不进），与评价时的簇完全相同。
- 形变响应对：从每个选中的帧出发，用与诊断相同的 MM 短轨迹（20 步 × 2 fs，新速度），把配体刚性拟合回起点，
  环境和 Ω 固定在起点；取 t = 0 与若干 t>0 的构型组成对。
- 标签（float64，冻结 backbone）：每个构型的 T = C_bb − cross（`coupling_labels` 与 `coulomb_cross` 的同一次密度）；
  对的标签为 ΔT。
- 规模（待冒烟测耗时后定）：初定每体系每 rep 30 帧 × 每帧 1 条轨迹 × 4 个时间点（含 t=0）＝ 2×17×30×4 ≈ 4080 个构型。

### 2. 损失

保留 arm N 的全部项（L_E、L_F、二聚体 L_int、原簇 L_int、SAPT 四通道），权重不变。新增：

- 新簇合计：Σ_c D_c → T，进入现有簇 L_int 项（同一归一化方式：除以新簇标签的均方）。
- 形变响应：L_resp = mean_pairs[(Δ Σ_c D_c − ΔT)²] / mean(ΔT²)，权重 w_resp 初定 1（与其他项同样按均方归一化）。
- 不对单通道做任何缩放；四通道仍由 SAPT 项监督（新簇没有 SAPT 标签，只进合计与响应）。

### 3. 训练方式与版本

- 从头训练（同一随机种子与步数 60k），不在旧 head 上微调，便于与旧版逐项对比。
- 新版本名：`joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N_resp<tag>.pt`，在所有输出与 E 构造标记中写明
  “local_E v1 + head v2”。旧 head 与旧 η 链原样保留，不作为新目标的样本。

### 4. 验收时报告的量（只报实测，不设放行线）

在 rep1–4 S0（未参与训练）上，与旧 head 并列：
1. 内部形变响应：学习通道之和 vs T 的相关、斜率、差的 sd（与 `diag_head_channels.py` 同一口径；
   并用已保存的 `runs/diag_head_channels_data.npz` 构型复算）。
2. 合计水平：(Σ_c D_c + cross) − C_bb 的均值与 sd。
3. 原有指标：DES370K test 的 E_int、四通道 MAE、E/atom、F（`test_joint.evaluate`），以及 `learn_lampool.zsh` 里的 `test_scale` 扫描。
4. δU = C^ML − C^MM 在 rep1 S0 的分布与 40 fs 快变 sd（`local_E.py`、`diag_fast_dU.py` 口径）。

### 5. 实施步骤（确认后）

1. 写数据生成脚本（簇＋响应对＋float64 标签），输出到 `experiments/2026-10-25_head_retrain/data/`。
2. `test_joint.py` 加可选环境变量（新簇文件、响应对、w_resp）；默认值下行为与现在完全相同。
3. 在 181 冒烟：生成少量数据、训练几百步，测单构型标签耗时与每步耗时，再给 PBS 计划表（耗时按实测外推）。
4. 正式训练与验收按计划表执行。

## 实施状态（2026-10-25，用户：“你先冒烟一下”）

- 步骤 1：`make_data.py`（`gen` 按 rep 输出 `data/<体系>/rep<r>.npz`，元数据一致则跳过；`pack` 合并成 `data/resp_<名>.npz`）。
  标签 C_bb 用 `coupling_labels`、X 用 `coulomb_cross`（整簇一次前向），都是 float64，单位 eV；时间点取第 0/5/10/20 步（0/10/20/40 fs）。
- 步骤 2：`test_joint.py` 新增 `E3D_RESP`（pack 文件）与 `E3D_W_RESP`（默认 1）。原文件备份为 `test_joint.orig.py`。
  - 响应簇排在训练集末尾。每个响应对（同一轨迹的 t=0 与一个 t>0）单独成一个 batch，Δ 在同一次前向里算。
  - 新簇的合计项用自己的均方归一化。
  - 文件名带上 pack 文件名的 stem（例如 `..._es_resp_smoke_h256_...`），w_resp≠1 时再加 `_wr<w>`。
  - 不设 `E3D_RESP` 时，batch 划分与各项损失和原来相同：这一点只按代码核对过，没有另跑一次对照。
- 步骤 3：在 181（2080 Ti）上冒烟通过，没有做完整训练。日志：`logs/smoke_make_data.log`、`logs/smoke_train.log`。
  - 数据：两个体系 rep5 各 2 帧 × 4 个时间点，共 16 个构型，簇为 185 原子（CypD-8）和 181 原子（PWWP1-8）。
    - 第一次 CypD-8 标签用 batch 4，在 2080 Ti 上 OOM（同卡还有 OpenMM context）。改成 `--batch 1` 后通过。
    - 每个构型：C_bb 0.66–0.73 s，X 0.22–0.23 s；MM 加截簇可以忽略。
  - 训练：200 个二聚体、20 个旧簇加 16 个响应构型，arm N，300 步，共 138 s（0.46 s/步）。
    - 300 步里有 143 步抽到响应对。
    - 响应项（归一化后）从 64.6 降到 3.8。只跑 300 步，不说明收敛。
- 冒烟里看到的计划外事项：
  - 每条轨迹的 t=0 构型出现在 3 个 batch 里，所以在新簇合计项里的权重是其他时间点的 3 倍。
  - 按计划的规模（4080 构型），会有 3060 个响应对 batch。二聚体 batch 是 50000/16 = 3125 个，旧簇 batch 数没有统计，
    所以响应对 batch 在训练里的占比远高于旧簇在原配方中的占比（这一点放进下方未确认）。

## 未确认（Claude 推断）

- S1–S3（部分 dummy）的构型是否也要进训练：head 在这些端点上剪掉 dummy–环境边，但对应的 backbone 标签在部分 dummy 下尚无定义
  （`RESULT.md`：按片段成图只对 S_K 适用）。本计划暂只用 S0。
- 是否同时监督环境运动引起的响应（诊断中环境运动部分 sd 10.2 kJ/mol），以及 Ω 换水的不连续。本计划只按用户原话监督内部形变。
- 用 ΔT 差分监督还是用 T 对配体坐标的梯度（力匹配）监督；后者信息更多但需改 backbone 求导路径。
- 训练中是否加入其他配体以保持迁移性；只加两个目标体系可能让 head 偏向它们。
- w_resp=1、每帧 4 个时间点、30 帧/rep 都是初定，没有依据，冒烟后按耗时再定。
- 响应对每对单独成 batch（避免 4 × 185 原子超过现有 BATCH*34 原子上限），带来两点：
  t=0 在新簇合计项中权重 ×3；响应 batch 在训练中占比高。要不要降低抽样频率或改成整条轨迹一个 batch（只能在 5090 上），未定。
- 训练簇上限 24 水而评价用 50 水，新簇用 50 水后是否改变旧数据上的指标，需由第 4 节第 3 项实测。

## 正式作业（2026-10-25，用户在 yayoi10 / yayoi28 中让我选，选 yayoi28：作业只用 1 块 GPU）

- `pbs_retrain.zsh`：数据（rep5–21 × 30 帧 × 4 时间点，`--batch 1`）→ `pack v2` → arm N 重训（learn_lampool 配方＋`E3D_RESP=data/resp_v2.npz`）→ test_scale 四组扫描。
- head v2 文件：`runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s60000_N.pt`。
- 日志：`logs/data_v2.log`、`runs/polar-M/logs/joint.log`、`runs/polar-M/logs/scale_N_es_resp_v2_*.log`。

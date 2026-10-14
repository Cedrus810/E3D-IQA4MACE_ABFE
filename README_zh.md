# E3D-IQA4MACE · ABFE

[English](README.md) | **中文** | [日本語](README_ja.md)

**由物理规范固定的等变势给出原子分辨的结合自由能——一次计算，每个原子都有交代。**

多数等变 MLIP 把能量写成 `E = Σᵢ Eᵢ`，但由于消息传递把所有信息混进每个节点，
某个原子的 `Eᵢ` 并不能唯一地拆成分子内、静电与极化部分。把配体原子的 `Eᵢ`
乘上 λ，你并没有移除它与环境的相互作用——你移除的是它与配体自身内能的某种
任意混合，而且每个原子的混合比例都不同。

本工程把"移除"做成严格的：

```
E = Σᵢ E_intra(i) + Σ_{i<j} D_ij          （分解，钉在可观测量上）
D_LE = Σ_{i∈L} Σ_{a∈E} D_ia                （配体–环境耦合）
```

由于分解**规范固定到物理可观测量**（而不只是总能量的重新参数化），用逐原子
λᵢ 缩放耦合项并沿对角路径积分，就得到解耦自由能的逐原子分解：它严格加和等于
总自由能，且不依赖任何去原子顺序——这就是 Shapley 值，代价只是一次普通 ABFE。

| | 逐原子 dG | QM 精度 | 极化 / 电荷转移 |
|---|---|---|---|
| MM-GBSA 逐残基 | 有（不可靠） | 无 | 无 |
| MM ABFE 按组解耦 | 有 | 无 | 无 |
| 朴素 `Σᵢ Eᵢ` MLIP | 无 | 有 | 有 |
| **本工作** | **有** | **有** | **有** |

完整设计、实测与 go/no-go 标准：**[RESEARCH_PLAN_v3.md](RESEARCH_PLAN_v3.md)**
（2026-10-10 起；[RESEARCH_PLAN_v2.md](RESEARCH_PLAN_v2.md) S13 是此前的冻结实测
记录，v1 保留为 [PROJECT_PLAN_v1.md](PROJECT_PLAN_v1.md)）。当前进展与半成品清单：
**[STATUS.md](STATUS.md)**。

## 状态——进行中

规范固定纲领（WP1–WP7）已完成并实测，第一个完整的五项模型已训练。WP8 的机器
已建好并端到端验证；尚缺的是带真实采样的自由能计算——目前所有结果都刻意对固定
构象取平均，把归属算术与采样收敛分开。验证阶梯：**A** 二聚体 → **B** FreeSolv
溶剂化 → **C** SAMPL 主客体 → **D** 蛋白质–配体 ABFE（当前在 DES370K 二聚体的
A 阶段尺度）。

- **阴性对照可复现**（WP1）：没有 IQA 监督时，节点/边拆分不会自己收敛到
  IQA——合成标签下差 28 倍，冻结 `mace-omol-0` 骨架下差 49 倍；加了监督，
  它在 E/F 精度几乎不变的情况下完成重组。
- **`L_int` 钉住对加和**（WP3）：`Σ D_AB` 与 CCSD(T)/CBS 相互作用能在 228 个
  留出 DES370K 体系上吻合到 **0.27 kcal/mol**。
- **端点严格**（WP6）：λ 必须屏蔽*图*，而不只是边能量——只缩放边能量会让
  `U(R,0) ≠ U_L + U_E` 偏离耦合的 5.9%；图屏蔽通过全部六项一致性检验。
- **顺序无关归属**（WP7）：对角路径与穷举蒙特卡洛 Shapley 吻合到总量的 **0.82%**。
- **SAPT 分量监督**（es/ex/ind/disp 逐通道）：把与逐边 oracle 的差距缩小 85%，
  真实 SAPT 数据上 84–94%。只监督边界*总量*会让各通道比真实分量大 3–7 倍——
  这是对非可辨识性的直接展示。工作点：`sapt+total`。
- **炼金哈密顿量在 OpenMM 下运行**（WP8）：经由 `openmm.PythonForce`，
  不需要任何东西 TorchScript 化，λ 只是一个普通属性。端点恒等式经过桥接后
  仍保持到 1.2e-7 eV。

**第一个完整模型**（arm D，五项 `E + F + L_int + L_IQA + L_SAPT`，冻结
`mace-omol-0`，只训头）：对骨架自身能量 4.3 meV/atom，力 0.062 eV/Å，
相互作用能 0.53 kcal/mol，SAPT 各分量 0.6–1.7 kcal/mol——各通道比无监督基线
好 6–11 倍，能量代价约为 arm C 的 2 倍。这是上界：D 在 32k 步时仍在下降。
检查点在 `ckpt/`，数字在 `logs/joint_*.json`。

**凝聚相**（Atenolol + 最多 100 水，骨架现为 `MACE-POLAR-1-M`）：二聚体训练的
头带有每交叉边 -2.0 meV 的逐边偏差，且随体系增大；用冻结骨架自身标注的簇把它
在样本外削掉 7 倍，从 20 帧 MD 取簇把留出对加和带进 0.35 eV 以内。头的总耦合
在簇覆盖的交叉边数之外失效；更大的簇修好尾部却把误差挪到别处
（`RESEARCH_PLAN_v2.md` §13.15–13.23）。

**锚定总能量**（v3，2026-10-10 定）：总能量现在来自冻结骨架，
`U = E_bb − Σᵢ(1−λᵢ)cᵢ`，头只负责拆分；头缺的部分记为一项 cut 项，于是
`Σᵢ dGᵢ + dG_cut = dG` 严格成立。恒等式在代码中保持到 float64（v3 §13.29）。
cut 项的散布在 n = 20–50 时为 6–10 kT，所以 cut 要走几个窗口（§13.25, §13.37）。
头在跨炼金态池化的未最小化 MD 帧上重训，因为它拟合的正是簇所来自的系综
（§13.33–13.37）。

**尚未完成**：锚定采样所需的 MM 墙与 ML/MM 体系、带真实采样的自由能计算
（WP8/WP9），以及真实 IQA 标签——迄今所有节点规范结果用的都是合成教师，所以
它们说明的是"可以对冻结特征施加*某个*规范"，而不是"*IQA* 规范就在那里"。

## 仓库布局

```
decomp/                 包本体
  head.py               与骨架无关的分解头（逐节点 E_intra，逐边 D_ij）
  mace_adapter.py       把头接到冻结的 MACE 骨架上（WP4）
  lambda_mask.py        逐原子炼金耦合——"edge" / "graph" / "anchored"（v3 S3）
  polar_es.py           来自 MACE-POLAR-1 的显式配体–环境 Coulomb 通道（v3 S2.3）
  losses.py             混合监督：E、F、IQA 规范、L_int、SAPT（S4.3）
  train.py              定义性实验的训练步
  data.py               DES370K 二聚体批次
  fe.py                 沿对角路径的 TI、MBAR（WP8）
  sampling.py           炼金哈密顿量上的 Langevin 采样
  openmm_bridge.py      在 OpenMM 里运行学到的哈密顿量；刻出 ML 区
  hremd.py              lambda 复本交换
  clusters.py           由冻结骨架标注的溶剂簇

test_*.py               一文件一实验（见 RESEARCH_PLAN_v2.md §13、v3 §13）
test_anchored.py        锚定哈密顿量的严格检查
                        端点 / 归属 / openmm 运行按秒计，无需数据
test_scale.py           头耦合 vs 骨架真值，n_solvent = 5..100
run_polar.sh            MACE-POLAR-1 训练（每个骨架一个输出目录）
em_system.py            MM 最小化一条制备好的 ABFE leg
prepare_data.py         一次性 DES370K → train/val/test 切分（按体系，不按行）
run_queue.sh            实验序列的集群队列
RESEARCH_PLAN_v3.md     当前设计 + §13.24 起的实测（真正的文档）
RESEARCH_PLAN_v2.md     早期设计；§13.1–13.23 冻结实测记录
experiments/            按日期命名的实验目录（脚本、日志、结果）
PROJECT_PLAN_v1.md      v1 框架，已被取代
```

## 复现

环境：`e3nn`、`torch`、`mace-torch`、`pymbar`、`openmm`。
（开发环境：miniforge `openmm_dev`，e3nn 0.4.4——为 MACE-POLAR-1 降级——/
torch 2.12.0 / mace-torch 0.3.16；完整清单见 STATUS.md）

```bash
# 1. 从 DESRES 下载 DES370K（https://www.deshawresearch.com/downloads/），
#    并把 DES370K.csv 放到仓库根目录。
# 2. 构建元素/电荷过滤的切分（hcno、organic、full）：
python prepare_data.py

# 3. 跑实验序列（阴性对照、规范、端点、归属、joint arms A–D）：
./run_queue.sh            # 或单独运行各个 test_*.py
```

每个 `test_*.py` 在 `logs/` 下写一条 JSON 记录、在 `ckpt/` 下写一个头检查点
（均已 gitignore）；上文数字出自 `RESEARCH_PLAN_v2.md §13`，那里给出了引用。

## 参考文献

- Shimamura & Nomura, *E3D-IQA* — arXiv:2609.00674（定义性实验：IQA 监督重组
  等变势的节点/边拆分）。
- DES370K: Merchant et al.，37 万个经 counterpoise 校正的 CCSD(T)/CBS 二聚体
  相互作用能的 def2-TZVPD DFT 数据集（D. E. Shaw Research）。
- MACE: Batatia et al.；`mace-omol-0` 骨架来自 OMol25 模型库。

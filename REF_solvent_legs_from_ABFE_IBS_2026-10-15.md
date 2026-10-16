# 参考：从 ABFE-IBS 拿来的 6 条溶剂腿（Stage B 候选 + MM 对照值）

日期：2026-10-15
性质：**参考材料，不是设计稿**。不改 `RESEARCH_PLAN_v3.md` 的任何决定。D9（药物集）仍开放，下面这 6 个是**候选**，用不用、用几个由用户定。
来源：`/home/ruigengji/fragment-opt-abfe-benchmark/openmmIBS/runs_lcr_s1long15/`（fragment-opt 基准上跑完的 ABFE-IBS，3 rep/体系，2026-10-04 至 10-08 跑完）。
ΔG 数字是 ABFE-IBS 输出文件里的原值，E3D 没有复测；协议细节是从 IBS 的 config 和 log 里读出来的。

## 必读（用这批体系开工前全部读完）

| 文件 | 为什么读 |
|---|---|
| 本文件 | 体系名单、MM 对照值、和 Atenolol 腿的差异（§4 不处理会直接出错） |
| `data/solvent_legs_ibs/manifest.tsv` | 每个体系逐 rep 的 ΔG_solv、电荷段/vdW 段拆分、IBS 原 run 目录 |
| `/home/ruigengji/fragment-opt-abfe-benchmark/openmmIBS/README.md` | 这批 run 的协议（"s1long15"：去电荷 0.5+1.5 ns）、中和过的拓扑、HMR 的说明 |
| `/home/ruigengji/fragment-opt-abfe-benchmark/openmmIBS/configs/<system>_lcr.json` | 每个体系实际用的 config |
| 任一 run 的 `rep1/thermodynamic_cycle.md` | IBS 的循环定义、PME 自能、LJ 长程修正的处理方式（和 E3D 比总量时要知道 MM 那边包含了什么） |

---

## 1. 这 6 个体系

全部是中性、只含 H/C/N/O（在 DES370K `hcno` 子集里；Atenolol 也是 HCNO），配体 31–38 个原子，都不比 Atenolol（41）大。

ΔG 单位 kJ/mol，方向是**解耦**（耦合 → 解耦），和 E3D 的 `dG_decouple` 同向。mean ± SD 是 3 个 rep 的样本 SD。
ΔG_elec 是 stage 1 去电荷（adjacent BAR）；ΔG_vdW 是总量减去 ΔG_elec，也就是 IBS stage 2 的 vdW 消失。

| 体系 | SMILES（RDKit 从坐标推出） | 配体原子 | 盒子原子 / 水 | ΔG_solv | ΔG_elec | ΔG_vdW |
|---|---|---|---|---|---|---|
| HSP90_ligand-17 | `CCN(CC)C(=O)c1ccc(O)c(C(C)C)c1` | 38 | 5382 / 1778 | 48.86 ± 0.45 | 56.14 ± 0.71 | −7.29 ± 0.70 |
| HSP90_ligand-19 | `CCN(CC)C(=O)c1cc(C(C)C)ccc1O` | 38 | 5385 / 1779 | 35.94 ± 2.19 | 45.51 ± 0.95 | −9.57 ± 2.30 |
| HSP90_ligand-27 | `CC(C)c1cc(C(=O)N2CCOCC2)ccc1O` | 37 | 5390 / 1781 | 56.78 ± 0.45 | 62.42 ± 0.28 | −5.64 ± 0.25 |
| CyclophilinD_ligand-4 | `CCN1Cc2cc3c(cc2NCCC1=O)OCO3` | 34 | 5348 / 1768 | 65.24 ± 0.46 | 61.39 ± 0.34 | +3.85 ± 0.26 |
| CyclophilinD_ligand-8 | `O=C(NCc1ccccn1)C(=O)NC1CCCC1` | 35 | 6752 / 2235 | 60.63 ± 0.86 | 64.22 ± 0.50 | −3.59 ± 0.57 |
| PWWP1_ligand-8 | `Cn1cnc(-c2ccccc2)c1-c1ccncc1` | 31 | 5711 / 1890 | 47.59 ± 0.27 | 51.71 ± 0.89 | −4.12 ± 0.62 |

为什么挑这 6 个（我的选择，用户可以换）：

- **成对的化学对照**，用于 v3 §7.3「同一官能团在不同药物上的逐原子值应该相近」：
  - HSP90-17 和 HSP90-19 是同分异构体（酚 OH 和 iPr 互换位置）。19 的 OH 在酰胺邻位，ΔG_solv 少 12.9 kJ/mol，其中电荷段少 10.6。邻位分子内氢键是我的猜测，没有核实。这一对的逐原子水合图应该把差别落在 OH 和 C=O 上。
  - HSP90-19 和 HSP90-27 母核相同，二乙基酰胺换成吗啉酰胺（多一个醚 O），ΔG_solv 多 20.8。
- **另外三类化学型**：CypD-4（苯并二噁茂 + 内酰胺）、CypD-8（草酰二胺 + 吡啶）、PWWP1-8（N-甲基咪唑 + 吡啶，不含 O）。
- **MM 对照干净**：除 HSP90-19 以外，rep 间 SD ≤ 0.9 kJ/mol。HSP90-19 留着是为了那组异构体对照，它的 SD 主要来自 vdW 段（2.3）。

**这里没有的**：
- acetamide（Stage A，D8）。这批 run 里没有，还得另外准备。
- 带电配体（MCL-1 全部，以及 CypD-7、PWWP1-12/13/17/BI-*）。按 STATUS item 6，带电药物需要 DES370K `full`；而且 IBS 的带电路线本身还不完整（openmmIBS README 的 "Charged ligands" 一节）。
- 同一批 run 里其余的中性 HSP90 / CypD / PWWP1 也都可以用，数字见 IBS 侧 `runs_lcr_s1long15/`。这批 HSP90 里还有 18/20（tBu 异构体对，37.49 ± 0.06 / 42.00 ± 0.95）和 21/30（吗啉系列）。

---

## 2. 文件在哪、怎么加载

拷进 E3D 的小文件（共 25 MB），由 `data/solvent_legs_ibs/fetch_from_ibs.py` 生成，重跑会覆盖成同样的文件：

```
data/solvent_legs_ibs/
  manifest.tsv                       逐 rep 数字 + IBS 原目录
  <system>/system_solvent.xml        纯 MM System（3 个 rep 逐字节相同，脚本里检查过）
  <system>/rep{1,2,3}/
    system_solvent.xml -> ../system_solvent.xml   软链接，让 rep 目录能直接被 loader 读
    topology_solvent.cif             拓扑 + 建盒时的坐标（预平衡之前的，见 §4.5）
    pre_equil_final_state.npz        NPT 预平衡后的坐标 positions_nm + 盒 box_nm（IBS 从这里起跑）
    box_vectors_solvent.npy, ligand_indices_solvent.json
    solvent_final_results.json       IBS 溶剂腿完整结果（含诊断）
    stage1_decharging.json           去电荷段结果
```

加载（2026-10-15 在 181 上 CPU 冒烟，6 个体系都通过）：

```python
from decomp.openmm_bridge import from_abfe_output
system, top, pos, numbers, frag = from_abfe_output(
    "data/solvent_legs_ibs/HSP90_ligand-17/rep1", ligand_resname="LIG")   # 不是默认的 "MOL"
```

冒烟检查了这几项：`frag == 0` 恰好是 `ligand_indices_solvent.json` 里的原子（都是 0..n−1）；npz 原子数和 System 一致；配体净电荷 |q| < 2e-16 e；用 npz 里的坐标和盒能算出势能（例如 HSP90-17 是 −75056.6 kJ/mol）。

**大文件没拷，只记路径**（IBS 侧，只读）：`<ibs_run_dir>/repN/solvent_leg/`

- `pre_equilibration.dcd`：NPT 预平衡轨迹。
- `decharging/decharging_rep{i}.dcd`：去电荷段第 i 个热力学**状态**的轨迹（ABFE_IBS `free_energy_engine.py:770` 的注释说文件名里的 rep 指状态，不是物理 replica【据代码注释，没在数据上核实】）。每个 1.5 ns，150 帧，帧间隔 10 ps，全体系坐标（按文件大小算的：HSP90-20 是 150 × 64628 + 276 字节）。λ_coul 从 1 到 0 线性取 12 档：状态 0 是 λ=1，5 和 6 是 0.545 / 0.455，11 是 0。
- `decharging/decharging_pme_u_kn.npy`：去电荷段的 u_kn。

---

## 3. MM 对照用的是什么协议

从 config / log 里读出来的（以 `configs/<system>_lcr.json` 和 run 的 `launch_config.json` 为准）：

- 力场 GAFF2（配体）+ TIP3P（`amber14/tip3p.xml`），150 mM NaCl（这 6 个盒里各有 5–6 对 Na⁺/Cl⁻）。配体电荷先经过 `topologies_neutral/` 处理，残差平均摊到每个原子上（每个原子 ≤ 2e-4 e）。
- PME，截断 1.0 nm，开色散修正，ewaldTolerance 5e-4，HBond 约束 + 刚性水。
- **300 K**，2 fs，**HMR**（配体 H 质量 4.032，水的 H 仍是 1.008）。
- 预平衡：NPT 最多 10 ns，可以提前停（HSP90-17 rep1 在 5.5 ns 停）；盒体积取后半段的平均值，然后**固定盒 NVT** 做两段。
- Stage 1 去电荷：λ_coul 12 档，REMD，0.5 ns 平衡（不进 MBAR）+ 每档 1.5 ns，adjacent BAR。`couple-intramol=no`，配体内部 ≥1-5 的库仑冻结为与 λ 无关。
- Stage 2 vdW：26 档均匀 λ_vdw 的软核 IBS，分 5 个窗口，每窗口 250000 步，`stage2_lcr_rounds=true`。LJ 截断外的尾部修正已经并进采样哈密顿量。

---

## 4. 和 E3D 现在用的 Atenolol 腿有哪些不同（要处理）

| # | 这批 IBS 腿 | Atenolol 腿（`ABFE_IBS/Atenolol-rank11/output/`） | 影响 |
|---|---|---|---|
| 1 | 配体残基名 `LIG` | `MOL` | `from_abfe_output` 默认值是 `"MOL"`，不改就直接报 "no residue named"。`carve_ml_region` 等函数的 `ligand_resname` 默认值也要一起检查 |
| 2 | **有 Na⁺/Cl⁻**（5–6 对） | 只有配体 + 4035 个水原子 | FIRES（D11）只给水排序，ML 区里也只放水。离子会落在 MM 远场，但 `mlmm.py` 的 MM 远场逐原子 λ、L–O LJ、WCA 墙这几处有没有假设"非配体的都是水"，**没查过**。正式用之前先跑一遍 E0–E7 里便宜的那几项 |
| 3 | 300 K | E3D 用 298.15 K（`run_abfe.py --T`、`lambda_frames.py`） | E3D 和 MM 对照的总量比较要在同一温度下做。E3D 用 300 K 跑最简单；否则要说明 2 K 的差别 |
| 4 | 配体 H 做了 HMR（4.032 Da） | 标准质量 | 构型系综不受影响；但 E3D 在 NVE、步长上的结论（S13.42，λ=0+ 漂移）是在标准质量下测的，换成 HMR 以后不能直接沿用 |
| 5 | `topology_solvent.cif` 里是**建盒时**的坐标（cif 写于预平衡之前；HSP90-20 的 cif 盒是 3.826 nm，预平衡后是 3.79 nm 左右） | — | `from_abfe_output` 的 docstring 写着 "equilibrated"，对这批文件不成立。要用平衡好的起点，读 `pre_equil_final_state.npz`（坐标和盒一起用） |
| 6 | 盒子 5348–6752 个原子 | 4076 | MM 部分稍微贵一点；ML 区大小只取决于 n_solvent，跟盒子大小无关 |

另外，按 STATUS 里的 "Recurring lessons"：现在的 head 只见过 Atenolol 的 cluster。每个新药都要做三件事：用自己的帧做 cluster（v3 §5.2：λ=1、0.5、0 三个态各 200 帧）；重算 D15 的逐对墙半径；用 remainder scan 和短 NVE 确认 head 在它上面可用。

---

## 5. 怎么用（建议，不是决定）

1. **v3 §7.2 的总量对照**：E3D 的 `dG_decouple` 对上表的 ΔG_solv。两边是不同的哈密顿量，所以这是比较，不是过关门。MM 这边的噪声尺度用 rep 间 SD（0.3–2.2 kJ/mol），不要用单次 run 报的 MBAR ±。
2. **cluster 用的帧**：最直接的做法是照 E3D 自己的配方，用 `lambda_frames.py` 在 298.15/300 K 下重新出帧（v3 §5.2 实测每个 200 帧的态 217–313 s，2080 Ti）。IBS 的去电荷轨迹可以当免费的交叉检查：λ=1 和 λ=0 各 3 rep × 150 帧。但它们是 300 K、带离子的，λ_elec 也没有恰好 0.5 这一档，直接拿来当训练帧之前要先想清楚这些差别。
3. **顺序**：v3 §8 的路线是 Stage A（acetamide）走通以后才到 Stage B。这批体系在那之前能做的只有不花 GPU 的准备工作：加载、§4 的适配、出 cluster 帧。

---

## 6. 已知问题

- **HSP90-27 rep3 被 IBS 标成 `results_untrusted=true`**：stage 2 去相关后的样本数是 19，门槛是 20。IBS 自己说明这个门是"未标定的临时阻断条件"，这个 rep 的数（56.74）和另外两个 rep（57.25、56.35）也一致。manifest 里照常算进了平均。
- 18 个 rep 里有 10 个带 `max_endpoint_uncertainty_kJ_mol > 1.0` 的提示，都是 `affects_trust=False`。IBS 说那个 1.0 kJ/mol 的阈值没有依据。
- IBS 的 `precision_status=UNMEASURED` 是针对单次 run 的说法；这里每个体系有 3 个独立 rep，SD 是我在 manifest 里算的。
- 论文自带的 `free_energy_data/abfe.csv` 也有溶剂腿的列（`lig_coul`、`lig_vdw`）。这两列的单位和符号约定我没弄清楚（HSP90-20：−2.1 / 50.2；CypD-8：−0.8 / −7.6，和 IBS 的数对不上，也不像同一个口径）。弄清楚之前别拿来当对照。

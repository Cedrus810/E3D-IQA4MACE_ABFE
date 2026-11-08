# E3D 模型定义（从旧 RESEARCH_PLAN_v3.md 原样摘出，2026-10-23）

只摘模型本身（v3 §2）。

2026-10-24 连接采样修订：溶剂腿和复合物腿采用相同逐组拆除顺序。
K+1 是模型报告端点数；各组内部增加 MM 去电荷／软核 LJ 桥接窗口，每窗重新平衡、生产。
MM 采样，模型计算能量：模型只在 S0…SK 端点评价，不提供 MD 力、不需要定义组内 λ 中间态；2026-10-25 起在端点 η 窗口里以模型能量参与 MC 接拒（设计 §10）。
执行和自由能定义以 `DESIGN_2026-10-23_groupwise_dummy.md` 为准；旧 S3/cut/λ 驱动公式不覆盖当前协议。

本版局部目标 E_j=V_j−C^MM_{j,Ω}+C^ML_{j,Ω}，V 为实际 MM 采样能量。
**2026-11-07 K2 修订替代上句的总量定义**：当前 S0 采用 E_0=V_0−CMM_(0,Ω)+C_bb,0，
C_bb,0=E_bb(L+Ω)−E_bb(L)−E_bb(Ω)。head 保留学习通道＋显式 Coulomb 的原子读出 cH_i，
全局差 q_0=C_bb,0−Σ_i cH_i 独立积分和报告；原子贡献与全局余项共同闭合总量。
全 dummy 耦合为零；部分 dummy 的 C_bb 尚待定义，近期只交付整腿。该决策尚未切换旧运行代码或后端，
实现身份与合同见 [`K2 决策`](experiments/2026-11-07_head_calibration/K2_DECISION_2026-11-07.md)。
保留同帧共享的内部、环境、未替换远场和约束背景；替换项采用匹配的局部作用范围。
这是模型耦合重构目标，不能自动当作冻结 backbone 整盒能量。Ω、匹配 MM 长程／例外项、
周期边界与蛋白片段构造需先实现并验证，不能用局部模型评分扣掉 MM 全盒耦合。
按用户判断，已 dummy 原子的间接读出影响暂作可忽略近似。

主设计 §7.3：MM 中间窗用相邻 BAR 得到 ΔF^V_k；各端点以自己的 MM 帧计算
 a_j=−RT log <exp[−β(E_j−V_j)]>_{MM,j}，再用 ΔF^E_k=ΔF^V_k+a_k−a_{k−1}。
MM 窗口连接与模型端点修正分别验证；模型权重不能代替缺失的 MM 桥接。
相邻 BAR 不要求旧全状态共同测度或模型组内可加性。§7.2 原比值仅保留为精确恒等式。
§7.4 的 g_k=ΔF^E_solvent,k−ΔF^E_complex,k 与独立循环修正保持不变。

当前实现状态见 `HANDOFF_2026-10-28.md`。
旧 MM 总量／残差代码及旧模型拼接原型不自动转为本版实现。

**模型的功能是拆（分解）**：给定一帧构型，把配体与环境的耦合拆到每个配体原子上（D_iE = Σ_a D_ia）。
它不驱动 MD。MM 负责采样。

## 2. Model

### 2.1 Backbone: MACE-POLAR-1-M, frozen  [decided]

Measured against the two alternatives tried (v2-S13.15, S13.17):

| | at scale (Atenolol, 5-100 waters) | DES370K `E_int` | cost |
|---|---|---|---|
| `mace-omol-0` | coupling sign-flipped at 4 of 5 sizes; non-periodic training | -- | 19456-dim features |
| POLAR-1-L | sumD worse at every size held out | 0.537 kcal/mol | 2.2x M; 341 atoms do not fit 11 GiB |
| **POLAR-1-M** | linear, correctable per-edge bias | **0.428** | baseline |

For a drug set it also has what the others lack: periodic training, a total
charge input and explicit long-range electrostatics, which charged drugs need.

Locked by the user on 2026-10-10. Not measured: POLAR-1-M against MACE-OFF24
on the condensed-phase scan, and POLAR-1-M's ML/MM step time (S6.3).
Reopening the backbone later costs a full retrain of the head.

Environment: `miniforge3/envs/openmm_dev` (user, 2026-10-10; `openmm_dev_ubio`
runs the same code identically). Both are **production** -- see `CLAUDE.md`. Backbones in `/home/ruigengji/MLP/mace/`.

### 2.2 Head

`decomp/head.py` unchanged: `E_intra` per node, four pair channels per edge
(SAPT es / ex / ind / disp), permutation-symmetric, 0e readouts, hidden 256, on
frozen POLAR-1-M node features via `decomp/mace_adapter.py`.

Training recipe [decided, D7, 2026-10-12]: **arm N** -- arm D without the
synthetic-teacher `L_IQA` -- 60k steps, <= 24 waters, with the explicit L/E
Coulomb channel (D6, `E3D_ES=1`). Clusters (S5.2): 10,000, cut from
unminimised MD frames pooled over alchemical states. Current head (S13.37):
`runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt`
(DES370K E 1.44 meV/atom, `E_int` 0.319 kcal/mol; remainder mean within
2 SE of zero at n <= 20 at lambda = 1 and on the MM lambda_elec 0.5 / 0
proxy; -0.115 eV at n = 50, lambda = 1). For n <= 50.
Head history: 20 minimised frames (S13.26) -> + D6 (S13.32) -> 200 thermal
frames (S13.35) -> + lambda_elec 0.5 / 0 frames (S13.37). Each step fixed the
bias on the ensemble it added and nothing else: the head fits the ensemble its
clusters come from.

Under S3 the head's `E_intra` and its total enter no free energy: `U` uses only
`E_bb` and `D_ia`. They stay in training as regularisers through `L_E + L_F`.

### 2.3 Long-range electrostatics: an explicit pair channel  [decided, D6, 2026-10-12]

POLAR-1's electrostatic energy (`mace/modules/extensions.py`, `PolarMACE`;
`graph_longrange.energy.GTOElectrostaticEnergy`) is the Coulomb energy of
Gaussian-smeared charge densities after the field-update iterations -- a
bilinear form in the final densities `rho`:

    E_es = 1/2 sum_{i,j} rho_i K_ij rho_j

So the ligand-environment part `E_es^LE = sum_{i in L} sum_{a in E} rho_i K_ia rho_a`
is exactly pair-resolved for a given set of densities. What is not pairwise is
how `rho` itself responds to the environment.

Two options, both with an exact total under S3:

- **Pair channel (chosen).** Add `E_es^LE` per pair as a fifth, physics-
  defined channel of `D_ia`; the head's learned channels then fit the targets
  minus it. The long-range direct electrostatics becomes attributable per atom;
  only the charge response lands in `dG_cut`.
- **Cut step.** Leave it to the remainder. No new code, but the most
  chemically interesting part for charged drugs becomes unattributable.

Cost of the pair channel: the head must be retrained, because today's targets
already contain `E_es^LE` and adding it explicitly would double-count.

Evidence (S13.27): at n >= 10 the remainder tracks `E_es^LE` with slope ~1 --
the head carries none of its configuration dependence -- and removing it cuts
the spread most at large n (11.4 -> 7.1 kT at n = 50 with the full
electrostatic part). It does not remove the need for cut windows.

**Implementation route.**

- Total: `E_es(rho_L + rho_E) - E_es(rho_L) - E_es(rho_E)` on the full-system
  densities, by calling `PolarMACE.coulomb_energy` with masked `source_feats`
  (done in `experiments/2026-10-12_r_vs_es/r_vs_es.py`). No pair matrix needed.
- Per ligand atom: the cross term is linear in `rho_L`, so by Euler's theorem
  `E_es,i^LE = rho_i . dE_cross/drho_i` (summed over the density components) --
  one autograd call gives the exact per-atom split for the diagonal path.
- In the anchored U: `D_ia` -> learned channels + `E_es^LE` per ligand atom,
  both scaled by `(1 - lambda_i)`; densities from the full system at R.

**Retraining targets** (to fix in code, not here): subtract the explicit
Coulomb from every target that contains it -- dimer `L_int`, cluster
coupling, the SAPT `es` channel, and the head's total (`E_bb` minus the
pairwise Coulomb). With pbc all-False PolarMACE takes the real-space
evaluator (`graph_longrange.energy.GTOElectrostaticEnergy.forward`), so no
k-space term enters a non-periodic carve.

**Implemented 2026-10-12** (S13.28): `decomp/polar_es.py` (`coulomb_cross`,
`es_coupling`); `E3D_ES=1` in `test_joint.py` (targets: E minus the full
`E_es`, F minus its forces, coupling and SAPT es minus the cross term; metrics
reported with it added back) and in `test_scale.py` (scoring). Retrain:
`learn_d6.zsh` (PBS, node P1).

---

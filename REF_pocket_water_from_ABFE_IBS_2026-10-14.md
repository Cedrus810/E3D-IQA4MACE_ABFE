# 参考：ABFE_IBS 的口袋水「猴子补丁」—— 给 E3D W3 / D15 的启发

日期：2026-10-14
性质：**参考材料，不是设计稿**。不改 `RESEARCH_PLAN_v3.md` 的任何决定（D14 已定、D15 仍开放）。
来源：ABFE_IBS 仓库两份设计稿（下文数字全部出自那里，E3D 侧**未复核**）。

## 必读（用到这条线之前全部读完）

| 文件 | 为什么读 |
|---|---|
| `/home/ruigengji/ABFE_IBS/ABFE_IBS/docs/design/PLAN_pocket_water_swap_sampler_2026-09-27.md` | **现行版（v2）**：固定 N 的 ABFE + `kT ln P(N)`，采样器是搬水 MC；含 S1 接受率实测（§8–9）和 brd4 MC vs MD 对照（§10） |
| `/home/ruigengji/ABFE_IBS/ABFE_IBS/docs/design/PLAN_pocket_water_gcmc_patch.md` | **v1（09-18 暂停）**：GCNCMC + B 阶梯路线；`grand` 1.1.0 的实测成本和踩坑都在 §8–9 |
| `/home/ruigengji/ABFE_IBS/ABFE_IBS/PLAN_pocket_water_sampler_redesign.md` | 用户起草的原稿。v3 §3 引用的就是这份，但修订后落地的是上面的 v2 |
| `/home/ruigengji/ABFE_IBS/ABFE_IBS/acs.jctc.6c00178.md` | SwapMC 论文（Ding et al., JCTC 2026），v2 的出发点 |

代码：`/home/ruigengji/ABFE_IBS/ABFE_IBS/tools/pocket_water/`（`relocation.py`、`u0.py`、`run_sampler.py`，约 600 行），
测试 `tests/test_pocket_water_relocation_2026_09_28.py`。v1 的 grand 子类代码**已丢失**。

---

## 1. 热力学环（两版共用）

```
F_X = F_X(N) + kT ln P_X(N)
ΔG°_bind = ΔG_bind(N*) + kT ln P_holo(N*) − kT ln P_apo(N*)
```

- N = 区域 R 内的水氧数。两端必须是**同一个 N\***、同一个 R 定义。
- 成立前提：ABFE 全程 N 真的被约束在 N\*。v1 只是把 N 个水放进初始构型，并没有约束住；v2 用区域墙把这一步做严格（v2 §5）。

### 和 E3D W1–W3 的对应（我的推导，待 E3D 侧核对）

E3D 的 W3 是「把纯排斥墙去掉，让水填满空腔」。如果墙是**硬墙**（区域内有水，能量就是 ∞），去墙这一步的自由能**正好**就是上式 N\*=0 的那一项：

```
ΔF(W3: 墙→无墙) = kT ln P_无墙(N_wall = 0)        R = 墙排除的体积
```

墙刚度有限时，严格式为 `ΔF = kT ln ⟨e^{−βW}⟩_无墙`（单步 FEP，从无墙态往有墙态做）。
因此 W3 有两种算法：MM 软核把墙逐步去掉（溶剂腿），或者在无墙态上测出 N_wall 的分布、读 `P(0)`（复合物腿）。
后者本质上就是 ABFE_IBS 的补丁，可以直接借它的采样器和教训。

ABFE_IBS v2 的结论在这里同样适用：**N\*=0 落在 P_apo(N) 的远尾**（jnk1 apo 口袋均值约 14 个水），直方图数不出来，必须加 N 偏置 g(N)（预热期 Wang–Landau 压平，进生产前冻结；见 v2 §3.3）。

---

## 2. 先测、再决定要不要 GCMC（最值钱的一条）

ABFE_IBS 花了两轮才发现：**它要修的那两个体系，口袋水 MD 本来就采得到。**

- v2 §9：λ=0 时，配体体积这块区域在 JNK1 和 BRD4 里补水都很快，每个水每 ns 交换约 23–30 次，一次完整驻留的中位数约 20 ps。
- v2 §10：brd4 两个配体的端点态，MD+MC 与纯 MD 得到的 N_R 在误差内一致，MC 净插入约为 0。
- BRD4 确实有约 4 个慢水，但它们在 holo 和 apo 两态都满占，3 个 rep 的结果也一致 ⟹ 两边的 `kT ln P` 互相抵消，没有需要修的东西。

⟹ D14 写的是「GCMC 用在交换慢的地方」。动手之前先用最便宜的探针判断「慢」到底存不存在：
在 λ=0、无墙的态上跑普通 MD 1 ns，**逐 ps 追踪每个水的身份**，数出区域内每个水每 ns 的交换次数，以及 900 ps 内一次都没换过的位点。
探针脚本可以照搬 `ABFE_IBS/tools/diagnostics/pocket_water_s1_probe.py`（注意 v2 §9 记录的那个已修 bug：每帧第二个源水要先 `ctx.setPositions(pos)`）。
只有出现 BPTI 型的深埋口袋（文献给的数字是：默认间隔下 10 ns 内插入成功 0 次），才值得上 GCMC。

---

## 3. 如果真要 GCMC：两条路线的实测成本

| 路线 | 实测 | 出处 |
|---|---|---|
| `grand` GCNCMC（v1） | jnk1，77k 原子，2080 Ti：switching 约 1 ps 最优，**约 37 s 换一次被接受的插入**；加长 switching 接受率并不上去 | v1 §9 |
| 搬水 MC，均匀选源 | **死路**：体相水 βu≈−32，随机放进口袋的位点几乎都做不到这么好，接受率约 10⁻¹¹ | v2 §8 |
| 搬水 MC，能量加权选源（SwapMC eq 1） | apo 每个候选的接受率约 2×10⁻⁵；1.8 万个候选一批，至少接受一次的概率 jnk1 约 0.22、brd4 约 0.10（重尾，帧间可差 5 个数量级） | v2 §8–9 |
| 搬水 MC + Python 精确 PME | 生产尺寸**跑不动**（2.8 ms 一次能量 × 1.8 万个候选，Z 还得另算），必须写批量逐水能量 GPU 核 | v2 §8 |

**对 D15「GCMC 插进来的水用 ML 还是 MM」的启示**（推论，未测）：
上表每次接受要评估 10⁴ 量级的候选能量，这个量级用 ML 势逐个算不现实。可行的组合是 **两级 delayed acceptance**（Christen–Fox，v2 §3.4）：
第一级用 MM 做 U_0，跑完整个 MTM；第二级只对被选中的那个候选补一次 `exp(−β[(U_ML−U_0)(y) − (U_ML−U_0)(x)])`。两级合起来的平稳分布严格是 ML/MM 哈密顿量的分布。
要求 U_0 是**整个构型**良定义的函数，不能用「只算附近几个原子」这类随候选变化的近似。
另外要和 D11（ML 水的身份方案，FIRES）一起看：搬水改变的是「哪些水在区域里」，如果区域里的水恰好是 ML 水，就会碰上 ML 水身份必须保持连续的问题。

---

## 4. 方法上的坑（ABFE_IBS 已踩过）

1. **SwapMC 的 first-accept 破坏细致平衡**：按固定顺序试 M 个候选、取第一个通过的，转移核里会带上 `[1−(1−Ā)^M]/Ā`，正反两个方向的这个因子不相等。要用标准 MTM（Liu–Liang–Wong 2000），把正反向的 Rosenbluth 权重都放进接受率（v2 §2）。能量加权选源和 K 候选 MTM 合用时，接受率要单独推导，不能照抄（v2 §3.2 ⚠️）。
2. **SwapMC 用反应场能量直接做接受判据**：这样采到的是 RF 哈密顿量的分布。要么用精确能量，要么用 delayed acceptance 修回来。
3. **N 的计数和墙必须用同一个区域中心函数。** 不同的话，P(N) 里的 N 和 ΔG(N\*) 里的 N 就不是同一个东西（v2 §3.6）。E3D 的墙以配体为中心；如果改用蛋白侧的区域，也要保证 apo/holo 和各 rep 用的是同一个区域。
4. **墙挡得住水，挡不住蛋白**：v2 §5 记录了 HREX 末帧里蛋白侵入鬼影配体体积的情况。E3D 进入复合物腿以后，W2 的 λ=0 端也要看这一点。
5. **收敛门对慢变量失明**（v1 §6）：block 平均、ESS、split-half 量到的都是快自由度。这条线的唯一验收判据是**跨 rep 散布是否缩小**，不是「更接近实验值」。
6. **N 之外还有其他慢变量**：ABFE_IBS 09-24 发现，stage1 的 rep 间差异主要来自口袋 Arg/Gln 侧链的离散态，不是水。

## 5. `grand` 1.1.0 的坑（E3D 的 `openmm_dev` 里装的就是它，生产环境不改，下面都在进程内处理）

出自 v1 §8：

- `calc_mu_ex` 按 pymbar 3 写，环境里是 pymbar 4.2.0。有 4 个调用点要在进程内加兼容 shim（`u_kln` 3D 入参在 pymbar 4 里已取消），**不要改 site-packages**。
- `customiseForces()` **就地**改 System：把 NonbondedForce 里所有 ε 清零，sterics 搬进新建的 CustomNonbondedForce，同时关掉色散校正。⟹ 每个协议档都必须重建 System。
  对 E3D 的**额外风险**（未测）：混合 ML/MM 的 System 里，ML 区域的非键已经被 openmm-ml 去掉或改过，grand 的这一步很可能和它冲突。
- `initialise` 必须在 `minimizeEnergy` **之前**调，否则最小化时鬼影水还是真水，体系会被拧坏。
- `updateGCMCSphere` 是 O(N_water²)，2.3 万个水的体系每个 move 约 5.6 亿次操作，需要在子类里向量化。
- explosion 路径会把 NaN 坐标传给区域更新（`samplers.py:1358`）。单球版碰巧没事，换成自定义区域就会抛错。
- 非球形区域（球并集）的子类只需要覆写 4 处：`initialise`、`updateGCMCSphere`、`insertRandomWater`、`__init__`。插入在区域内均匀、B 里的 V 等于区域体积、成员判定一致，三者都满足时，接受率公式不用改。
- 实测 TIP3P 的 μ_ex = −5.986 ± 0.117 kcal/mol，与 grand 的默认值 −6.09 在误差内一致。

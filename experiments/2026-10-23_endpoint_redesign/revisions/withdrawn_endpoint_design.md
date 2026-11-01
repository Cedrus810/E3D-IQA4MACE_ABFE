# E3D 设计：全拆／全恢复，MM 采样，ML 离线自由能分解

2026-10-23 重新设计。保留本文件名作为唯一设计入口。
用户本轮要求拿新设计稿重新设计；下面的具体统计方法是本轮提出的方案，
不冒称为用户过去的逐字决定。改写前版本保存在
`experiments/2026-10-23_endpoint_redesign/revisions/design_before_endpoint_redesign.md`。

## 1. 本轮选择和交付目标

先做完整配体的两个物理端点：**全部耦合 A、全部 dummy B**。
“全恢复”是在 B 的帧上计算 A 的能量；不在 B 的穿透构型上突然打开硬核后直接积分 MD。
MM 各自采样这两个端点，保存整盒坐标；完成采样后模型加载一次、分批评估。
第一次交付是 acetamide 的总解耦自由能、原子分解及完整闭合记录。
逐基团状态保留为后续归因路径扩展，RDKit 分组先用于汇总原子结果，不要求先跑 K+1 个状态。

继承约束：冻结 MACE-POLAR-1-M 与现有 head；ML 不提供 MD 力；
MM dummy 保留配体分子内项，环境耦合用已有 Beutler 软核实现；
不自动加入基团内 MD λ 窗口、墙、GCMC、搬水或 ML/MM ladder。
下文 t 是离线积分坐标，不对应新增 MD 窗口。

先限定在现有中性 HCNO 溶剂腿。温度、盒子、约束、HMR、配体内部势和长程约定与对照匹配。
acetamide 使用 298.15 K，IBS 药物腿使用 300 K。结合腿与带电体系另定义参考态后扩展。

## 2. 同一算法的两种输入：MM 复现与 ML 局部修正

符号固定为 ΔG = F_B − F_A，单位 kJ/mol；水合方向取相反号。
固定盒子下实际估计 Helmholtz 自由能差 ΔF，沿用项目 ΔG 标签，不额外声称包含 NPT 或标准态修正。

| 输入 | A 的目标能量 | B 的目标能量 | 结果含义 |
|---|---|---|---|
| MM 恒等替换 | U_A^MM | U_B^MM | 同参数、同参考态的 MM 总量复现；检查算法 |
| ML 局部耦合替换 | U_A^MM + C_bb − C_loc^MM | U_B^MM | 明确定义的 ML 局部修正自由能 |

两种输入使用 §4–5 同一套统计分析。ML 总量对照 IBS，不强迫等于 IBS，
不拿 IBS 数字归一化原子份额。已有 MM 路线是基准，不因重设计而重做整套 ABFE。

对每个完整盒子构型 R：

    C_all^MM = U_A^MM(R) − U_B^MM(R)
    C_bb     = E_bb(L+E_c) − E_bb(L) − E_bb(E_c)
    C_T      = C_all^MM + C_bb − C_loc^MM
    U_A^T    = U_B^MM + C_T
    U_B^T    = U_B^MM

E_c 为同帧最近 N 个完整水分子；先固定 N=50，记录每帧水身份。
同帧 N=20/50 对照用来衡量边界敏感性，不悄悄更换目标定义。
C_loc^MM 是同一有限非周期簇的 L–水直接交叉对能：原始 MM 电荷的真空 Coulomb
加原始组合参数的 LJ，使用解包后距离，不加截断、LRC 或 PME。
PME、离子、远处水和长程 LJ 修正通过 C_all^MM − C_loc^MM 保留。
这是局部相互作用替换的定义，不宣称为整盒纯 POLAR 自由能。
能量导出器必须直接、稳定地评估保留项：B 帧上的近程 MM LJ 可能极大，
不能将两个约 1e20 的浮点数相减来恢复远场。上式规定数学目标；实现要按交叉对、
PME 修正和 LRC 拆账，并在正常帧上与原完整势核对。在 r=0 的帧上不能计算硬核再做无穷减无穷。

骨架三个能量在相同片段坐标上评估；中性配体和水簇各给实际净电荷 0。
配体保持完整，评估时不剪 dummy 图边，也不把输出设为 NaN 来模拟解耦。
B 的目标能量直接取 U_B^MM，完全关闭替换耦合；B 帧上评估 C_bb
只用于计算恢复 A 的反向能量，不能把穿透构型冒称为物理耦合构型。
非有限值、极端短接触和训练覆盖外构型全部保留并标记，不能删掉后提高有效样本数。

## 3. 原子耦合与残差一起闭合

每帧在完整图上取现有模型输出：

    d_i = Σ_a D_ia + E_es,i^LE
    r_bb = C_bb − Σ_i d_i
    c_far = C_all^MM − C_loc^MM
    C_T = Σ_i d_i + r_bb + c_far

显式 Coulomb 使用完整体系响应后的密度；直接交叉项进入 d_i，
密度响应及 head 未拟合的部分通过 r_bb 保留。
不假设 head 已精确拟合总耦合，也不把残差均值接近零当作自由能贡献为零。

正式输出同时给每原子贡献、r_bb 和 c_far 的贡献。
若下游需要“仅原子、严格加和”的表，另输出固定约定

    d_i^closed = d_i + (r_bb + c_far)/N_L

N_L 含全部配体原子。均分是明确的记账约定，不能称为学到的原子物理真值；
原始原子表与两项全局贡献必须同时保留。禁止按 Σd_i 的比例缩放总自由能。

## 4. 先改统计权重，再做双端自由能估计

模型在 MM 帧上评估后，不把 ML 能量差直接塞进普通 BAR。
使用自归一化重要性权重构造两端目标期望，再解等先验的非 Boltzmann Bennett 方程。
这是本轮选定的具体分析方案，下面给出方程，不再把统计权重问题挂成待用户判断的事项。

u=βU，β=1/(k_B T)。源势 m_A=u_A^MM、m_B=u_B^MM，目标势 v_A=u_A^T、v_B=u_B^T。
两端先去平衡、按相关时间取样；N_A、N_B 是用于分析的样本数。

    a_n = exp[−(v_A−m_A)(x_n)] / Σ_{x∈A} exp[−(v_A−m_A)(x)]
    b_n = exp[−(v_B−m_B)(y_n)] / Σ_{y∈B} exp[−(v_B−m_B)(y)]
    w_F(x)=v_B(x)−v_A(x),  w_R(y)=v_A(y)−v_B(y)
    σ(z)=1/(1+exp(−z))

求 δ=f_B^T−f_A^T，使

    Σ_A a_n σ(δ−w_F) = Σ_B b_n σ(−δ−w_R).

精确目标期望下，由 p_B/p_A=exp(δ−w_F) 直接得到此式。
MM 恒等替换时 a、b 均匀，退化为等先验双端 Bennett 方程。
用 log-sum-exp、稳定 sigmoid 和括区间求根，不复制高权重帧代替加权。
等先验在样本数不等时仍一致，但不声称最优方差；有限样本重要性估计也不声称无偏。

## 5. 总量和原子贡献使用同一个离线统计测度

由每端自己的源帧计算

    δf_s = −log mean_s exp[−(v_s−m_s)] = f_s^T−f_s^MM
    (f_A^T,f_B^T)=(0,δ),  f_s^MM=f_s^T−δf_s.

拼接两端样本，每帧保存两端源能量和两端目标能量。定义

    log D_n = logsumexp_s [log N_s + f_s^MM − m_s(R_n)]
    v_t = (1−t)v_B + t v_A,  t∈[0,1]
    ℓ_n(t) = −v_t(R_n) − log D_n
    W_n(t) = exp ℓ_n(t) / Σ_m exp ℓ_m(t)
    f_hat(t) = −log Σ_n exp ℓ_n(t)
    ΔG = kT [f_hat(0)−f_hat(1)].

原子和残差使用相同 W(t)：

    ΔG_i   = −∫_0^1 Σ_n W_n(t) d_i(R_n) dt
    ΔG_r   = −∫_0^1 Σ_n W_n(t) r_bb(R_n) dt
    ΔG_far = −∫_0^1 Σ_n W_n(t) c_far(R_n) dt
    Σ_i ΔG_i + ΔG_r + ΔG_far = ΔG.

上式分量以 kJ/mol 给出；原型输入为 β 乘后的 reduced units，最后乘 kT。
32/64 点 Gauss–Legendre 离线积分逐项检查收敛，无新增 MD。
闭合对共同有限样本测度也成立到积分误差，不在结果出来后补差值。
NBB 和共同测度的端点差在有限样本下可不同，二者都记录并检验差异。
原子结果称为指定连续路径的归因，不宣称等于离散 Shapley。

## 6. 一次完整执行和输入协议

1. **MM 采样**：取现有软核系统的全部耦合／全部 dummy 两端，独立平衡，保存整盒坐标、盒子、速度和源状态。
   首轮预算每端 100 ps 平衡 +500 ps 生产、1 ps 存帧；按相关时间和独立重复调整。
   已有 groupwise S0/SK 整盒帧可进入首轮导出，不要求先重跑中间状态。
2. **能量导出**：每帧评估完整 MM 两端源势；模型加载一次，分批计算 C_bb、d_i、r_bb；
   同帧计算 C_loc^MM、c_far。保存系统 XML、坐标来源、参数、分组和 checkpoint 的 SHA256。
3. **统一分析**：先在同帧做 MM 恒等替换，再接真实 ML 输出。输入 NPZ：
   `u_source[2,N]`、`u_target[2,N]`、`components[C,N]`、`n_source[2]`；
   行序 A/B，列序 A 源帧后接 B 源帧，能量全部 reduced units。
   components 为原子、r_bb、c_far，逐帧验证 sum= u_target[A]−u_target[B]。
   另含 `units='reduced'`、`decorrelated=True`、`temperature_K`、`component_names[C]`。
4. **重复与误差**：各重复独立给总量／原子表；完整序列用块 bootstrap 重做权重、NBB 和积分，
   报告分量协方差，不能把原子误差当成独立量平方求和。另报告重复间离散及重训后的归因敏感性。

本轮新增统计原型：`experiments/2026-10-23_endpoint_redesign/estimate.py`。
在该实验目录运行：`/home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B estimate.py energies.npz result.json`。
原型不调用 OpenMM、Torch 或 MD；检查输入、计算权重、NBB、共同测度积分，返回诊断。
它尚不是整盒 MM／ML 能量导出器，也不输出校准后的物理置信区间。
真实 ML 端点和生产采样不因原型通过而视为完成。

## 7. 验收和支持不足时的动作

以下是预先写定的工程筛选门槛，不是理论误差保证：

| 检查 | 首轮条件 | 未满足时动作 |
|---|---|---|
| 系统／温度／参数 | 元数据与源状态一致 | 修正导出，不能混用输入 |
| 逐帧闭合 | reduced units 绝对 1e−7 + 相对 1e−10 | 修正记账，禁止补总量 |
| 权重 | 两端及积分节点 Kish ESS≥50、ESS/N≥0.02、max W≤0.05 | 返回 insufficient_support |
| 端点连接 | 对称 Bennett overlap≥0.03 | 返回 insufficient_support |
| 数值一致性 | NBB／共同测度差≤0.1 kT；32/64 点逐项差、闭合误差≤1e−5 kT | 增加离线积分点或修正实现 |
| 科学验收 | 独立重复、块误差、接触覆盖、N 敏感性、留出归因检查 | 标记未完成，不能仅凭 ESS 放行 |

两个端点减少状态和任务数，不保证有限帧覆盖全部恢复事件。
未通过先在相同两个 MM 端点补独立重复和分块诊断，不改目标能量、不删除极端帧。
若证据显示结构性支持缺口，下一项方法工作确定为**离线可逆构型映射**：
B→A 恢复时映射水坐标，在源密度／权重中计入 Jacobian，做可逆性及正反检验。
这属于后续实现；不能用删水、任意推水或省略 Jacobian 冒充已经修好。
本轮不默认启动映射训练、墙或新 MD λ 窗口，由真实端点数据决定是否需要后续方法工作。

## 8. 已有证据和旧路线

- `jump_swap/result_r0.25.json`：+33.89795 = +0.00158 +56.84502 −22.94866 kJ/mol，
  MM 参考 +34.04；另两种子约 +35.46、+36.13。
  它说明完整拆除可以构成可计算的热力学循环；包含墙自由能修正，不能去掉修正再称裸端点跳跃已验证。
  原文件 jump 的 BAR_err 为 NaN，保留它和 bootstrap 记录，不把接近参考当作统计验收。
- 现有 groupwise 整盒轨迹保留。`step_eval.py` 的平均耦合差是能量诊断，不能改标签当自由能。
- “MM 帧 + ML 能量差直接普通 BAR”由 §4 替换；不再把统计权重问题推回用户。
- 不打开、修改或移入 `Archive/`；旧 ML/MM 代码不作为本路线入口。
- 模型权重和训练来源见 `MODEL_DEFINITION_from_v3.md`。其中旧 S3、cut 等只作历史模型说明，执行以本文为准。

参考：[König et al., JCTC 2014, Multiscale Free Energy Simulations](https://pmc.ncbi.nlm.nih.gov/articles/PMC3985817/)。
共同测度归因、局部替换边界和操作门槛是本轮新设计，尚未在真实分子上完成验收。

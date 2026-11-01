# 2026-10-28：η 原子记录、续接与统计接口补齐

用户要求：“你看哪些是你顺手能做的”。本轮直接完成不改目标、不重训的 η 侧接口。
未运行真实分子 MD、模型能量查询、GPU 或 PBS 作业；未修改生产环境或历史链文件。

## 已实现

1. `eta_hmc.ModelEnergy` 同一次查询保留逐原子 C_ML、C_MM、δu、簇的整盒原子映射和水身份。
   标量目标及显式 Coulomb 边界不变，保存分解不增加模型调用。
2. `hmc` 保存实际链状态：接受则替换原子分解，拒绝则重复旧分解与旧坐标。
   dH / dE 仍是提议诊断；V / dU / 原子项 / 坐标是接拒之后的状态。
3. 新 `eta_records.py` 实现 schema 2 的分块提交与严格检查点恢复。
4. 新 `atomic_report.py` 给出 A_i 的 η TI、标量 BAR、r_η=BAR−Σ_i A_i、各起点结果和可选块重采样。
   原子联合行在各自链内重采样，保留原子间协方差；不只取接受帧，不分摊闭合残差。
5. 支持 `chain --head ... --backbone ...` 显式选择已有模型；默认加载路径不变。
6. 新批处理默认 tag `_atoms_v2`、同卡并发 1；失败／分析／worker 日志按 tag 分开，保留旧日志。
   本轮没有运行该批处理或提交 PBS。

## 新保存合同

| 字段 | 频率与语义 |
|---|---|
| acc / dH / dE | 每循环，接拒及提议诊断 |
| V / dU / C_ML / C_MM / dU_atom | 每循环的实际链状态，包括拒绝后的重复状态 |
| lig_xyz / cycle | 每循环的配体坐标与从 1 开始的循环编号 |
| cluster_system_indices / omega_water_resid | 每循环实际状态的簇原子映射及水身份 |
| xyz / xyz_cycles | 整盒 float64 坐标及其循环编号；默认每 10 循环及末循环保存 |
| last_positions / checkpoint / rng_state | 提交边界的完整位置、OpenMM Context 检查点、NumPy 接拒 RNG |
| box / meta | 单位、映射、dummy、参数、模型路径、System / integrator XML、Context 参数、种子、完成标记等 |

默认每 50 循环写一次 `.progress.npz`，先写临时文件再原子替换；最后提交正式 `.npz`。
未提交片段发生中断后，从最后一次提交重新执行；只有兼容的 Context checkpoint 才称为精确续接。
检查点无法加载时停止，不把只有坐标／速度的续接混称为精确恢复。
完成文件的复用也要通过 schema、数据和运行合同检查，旧格式不能原地升级或被静默覆盖。

模型文件继续要求使用不可变的版本路径；mtime 仅作来源记录，不作为 hash／续跑裁决。
续接时额外查询一次已存实际状态，核对标量及原子目标，然后用已存标量继续接拒。
MM System 的实际 XML、Context 参数与积分器定义按内容检查，防止仅 XML 路径相同就误续跑。
最初在 Reference CPU 上验证检查点重放；随后一次真实模型 CUDA 续接冒烟通过，见末尾章节。
CUDA 提议数组的完整逐位重放及更广泛的恢复覆盖仍未验证。

## 原子统计的边界

- 要求显式给出完整 η 网格及起点；缺任一条时标 incomplete，不生成完整 A_i。
- 不同模型、状态、原子映射或 MM Hamiltonian 不能混合。
- 均值保留所有循环；每个起点另给 A_i、BAR 和残差，避免拼接平均掩盖起点不一致。
- 可指定块长度，或以标量及各原子统计无效率的最大值选择；原子联合行在链内抽块。
- 块重采样输出是经验诊断分位数，不是已校准的物理置信区间。
- 标量 BAR 使用全部平衡后循环求点估计；相关性由可选块重采样处理。
  这与旧 scalar analyze 的中位 g 抽稀算法不同，旧分析器和旧结果保留用于历史回归。
- 本轮没有新分子链，因而没有新分子的 A_i / a_j 可报告。

## 使用接口（仅示例，本轮未执行）

在项目根目录，单个已有模型端点的链：

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B experiments/2026-10-25_eta_hmc/eta_hmc.py chain CyclophilinD_ligand-8 --eta 0.5 --start 1 --cycles 500 --tag _atoms_v2 --checkpoint-every 50 --coords-every 10

可额外传 `--head <已有版本路径>`，不更改默认模型或旧权重。模型 MC 接拒限定已验证的 float64 查询。
`--cycles`、记录间隔及目标合同变化时，应使用另一 tag，不能覆盖现有链冒称同一次续跑。

当实际 11×4 的新链完整时，原子报告示例（shell 展开 files）：

    /home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B experiments/2026-10-25_eta_hmc/atomic_report.py experiments/2026-10-25_eta_hmc/runs/CyclophilinD_ligand-8/chain_S0_eta*_start*_atoms_v2.npz --expect-etas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 --expect-starts 1,2,3,4 --bootstrap 200 --out experiments/2026-10-25_eta_hmc/runs/CyclophilinD_ligand-8/atomic_S0_atoms_v2.json

报告默认不覆盖已有输出；只传正式完成的链，不包含 `.progress.npz`。
这不是启动整批采样的指令；实际运行仍按已确认的单端点范围和预算推进。

## 验证

`logs/test_atomic_records.log`：6 个 unittest 通过，使用五粒子谐振 Reference Context、解析能量与合成高斯记录。

- 原子单位与 dummy 处理；40 次提议只有 41 次能量查询，记录不增加查询。
- 接受和拒绝均出现，逐循环原子能量与实际坐标一致，拒绝步保留旧值；整盒抽样编号正确。
- 在第二个片段中故意消耗 3 次提议和两套 RNG 后模拟中断，从第一次提交点恢复；
  坐标、接拒、标量／原子能量与未中断运行逐项完全一致。
- 完成 progress 已写但正式文件未落盘的情况能够完成提交；完成链复用不再查询能量。
- 旧格式、错误目标及不一致原子数据被拒绝，原文件字节不变。
- 原子 TI 均值和未分摊残差正确；联合块重采样保留已知原子协方差；缺链和错目标被识别。
- 调度假执行器验证：4 条失败→INCOMPLETE／退出 1；全部成功→ALL DONE／退出 0；旧日志未覆盖。

`logs/test_legacy_fixes.log`：旧链只读回归，仍复现 BAR −102.84 / TI −105.84 kJ/mol、23/44 缺失标记、
部分区间 not a_j、空 tag 隔离、错 head 分组以及活跃原子 NaN 抛错。
这些旧分子数值仍不可用，不因回归复现而升级为科学结果。

## 还没顺手解决的部分

- 完整 MM 原子参数延拓／导数评价器：需要对 PME、例外、LRC 的同步能量与导数逐项验证，未实现。
- CUDA 上的实际模型查询／记录／恢复冒烟：已做一次，见下节“CUDA 冒烟”。5090 上未跑。
- 真实 CypD-8 S0 的 η 混合与可用 a_0：未通过；其他端点及完整模型溶剂自由能也未验收。

旧 `groupwise_mm.py` 的端点续跑路径 D1 没有改；新的 η schema 2 只修复本轮明确的 η 侧记录和恢复接口。

## CUDA 冒烟（2026-10-28 14:52–14:55，181 RTX 2080 Ti，Claude 窗口）

命令（`experiments/2026-10-25_eta_hmc/` 下，两次相同）：

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B eta_hmc.py chain CyclophilinD_ligand-8 --eta 0.5 --start 1 --cycles 60 --checkpoint-every 20 --coords-every 10 --log-every 10 --tag _smoke_cuda

第一次在 `committed 20/60` 之后按 PID 杀掉（第二段已跑约 10 循环），第二次同命令续跑到完成。
日志 `eta_hmc/logs/smoke_cuda_run{1,2}.log`，PID `eta_hmc/logs/smoke_cuda_pids.txt`，
链文件 `eta_hmc/runs/CyclophilinD_ligand-8/chain_S0_eta0.500_start1_smoke_cuda.npz`（26 MB）。

接口结果：
- 真实模型（默认 head / backbone，float64）＋CUDA mixed 上，记录、提交、`validate_record` 全部通过；
  max|Σ_i δu_i − δU| = 9.9e−14 kJ/mol；整盒帧 float64，循环 10…60。
- 第 20、40 循环都是拒绝步，段首 `np.array_equal` 坐标比对两次都通过；CUDA 上没有出现事先担心的误拦（一次运行，不是全覆盖证明）。
- 从 Context 检查点＋RNG 续接成功：续跑的模型复查（容差 atol 1e-7 kJ/mol）通过；
  续跑第二段前 10 循环的日志统计（acc 0.00，⟨dH⟩ −1.14）与被杀进程同一段一致。只比了这一行，没有逐位比数组。
- 续跑进程 41 次模型调用（1 次复查＋40 次提议），60 循环累计 36.0 s（2080 Ti，非生产耗时）。

顺带看到的采样现象（不是这次冒烟要验的内容，原因未查）：
接受只出现在第 1–3、5–6 循环；第 6 循环接受了 ΔδU = −25.2 的提议，δU 到 −122.74 kJ/mol，
之后 54 个循环全部拒绝。这些提议的 ΔδU 在 +15.5…+70.0 kJ/mol，dH 在 −5.5…+6.1 kJ/mol，
所以拒绝主要来自 η·ΔδU（η=0.5 时约 3–14 kT），不是 MM 积分误差。链卡在 δU 低尾的一个构型上。

## CUDA 链的后续只读诊断（2026-10-28，本对话）

详见 `diagnostics/SMOKE_CUDA_LOW_TAIL.md` 及对应 JSON，可用 `diagnose_smoke_cuda.py` 复现。
只读已有链和静态拓扑；没有重新查询能量、调用模型或运行 MD。

- 第 5→6 的 −25.2104 kJ/mol 来自模型合计耦合变化 −17.7361、MM LJ 的修正贡献 −7.9657、MM Coulomb 的修正贡献 +0.4914。
- 后 54 个提议的接受概率中位数约 0.01%；由保存种子重建的全部 60 个接拒决定及最终 NumPy RNG 均一致。
- 冒烟使用原始旧 head；不能把它当成 v2／nvar 的采样验证。CML 的学习项与显式 Coulomb 未分别保存。
- Ω 换 3 水、内部几何改变同时出现。全部整盒快照都在停滞之后，且没有被拒提议的坐标／原子项；因此无法进一步作通道或因果归因。
- 固定构型上的 V 记录有 6.70e−5 kJ/mol 峰峰差，远小于提议修正代价，不将它列为主要停滞解释。

这次完成的是已有记录的诊断，不是新的采样验收，也没有修改记录器增加新字段。

## schema 3：提议诊断导出（2026-10-28 15:10，Claude 窗口）

起因：CUDA 冒烟链卡在 δU 低尾，旧记录只有被拒提议的标量 ΔδU / ΔH，分不出是哪个通道（`diagnostics/SMOKE_CUDA_LOW_TAIL.md`）。

改动（不增加模型调用，不改接拒目标）：
- `eta_hmc.couplings_dt(..., split=True)` 在同一次 forward 里另外返回学习通道和显式 Coulomb 的逐原子量；合计的算法与原来相同。
- `ModelEnergy.last_components` 新增 `C_ML_learned`、`C_ML_coul`（kJ/mol，dummy 为 0）。
- `hmc` 每个循环另存这次提议本身：`prop_` + 全部原子分解字段、`prop_V`、`prop_lig_xyz`；整盒提议坐标 `prop_xyz` 与 `xyz` 用同一组循环。这些只用于诊断，不进入链平均。
- `eta_records.SCHEMA` 2→3：校验 learned + coul = C_ML（提议和链状态都校验）、被接受的循环中提议字段与链状态一致、形状一致；续接时一并读入。
  schema 2 文件（只有 `_smoke_cuda` 那一条）不能再被 `read_record` / `atomic_report` 读取，原文件不动。
- 测试：`test_atomic_records.py` 补上述字段的检查（提议 ΔδU = 前一状态 dU + dE、提议原子能量与提议坐标一致、拒绝步的提议坐标不同于链状态），
  6 项通过，`logs/test_atomic_records_schema3.log`。

GPU 短链（181 2080 Ti，PID 2632268，`eta_hmc/logs/propdiag_run1.log`）：

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B eta_hmc.py chain CyclophilinD_ligand-8 --eta 0.5 --start 1 --cycles 60 --checkpoint-every 20 --coords-every 1 --log-every 10 --tag _propdiag

产物 `eta_hmc/runs/CyclophilinD_ligand-8/chain_S0_eta0.500_start1_propdiag.npz`（45 MB），`validate_record` 通过。
接受循环仍为 1、2、3、5、6，之后 54 次全拒，停在 δU −122.745（上次 −122.74；同种子，CUDA 上不逐位相同）。
分析结果见 `diagnostics/SMOKE_CUDA_LOW_TAIL.md`“追加”。

## `eta_hmc.hmc` 拒绝分支修正（2026-10-29，Claude 窗口）

CUDA mixed 下拒绝后 `setPositions(x)` 读回最多差 1.4e−14 nm，交换梯子每 sweep 调用 `hmc(cycles=1)` 时被段首严格相等检查拦下（单链只在 50 循环提交边界检查，此前未触发）。
修正：恢复后以 Context 读回值作为缓存状态；严格检查不变。已有单链记录不受影响（其边界检查均通过，段内差异量级 ~1e−11 kJ/mol）。
`test_atomic_records.py` 6 项、`test_eta_exchange.py` 4 项通过（`logs/test_atomic_records_fix.log`、`logs/test_eta_exchange_fix.log`）。

## `eta_hmc.hmc` 拒绝分支再修正（2026-10-30，Claude 窗口）

10-29 的做法（拒绝后以 Context 读回值作缓存坐标）会让 CUDA float32 级读回舍入随连续拒绝累积（rep2 walker 3 连续 275 次拒绝后水坐标漂移 7.1e−7 nm，能量与缓存差 8.9e−5 kJ/mol）。
现改为缓存坐标始终保留精确值、拒绝时从该值写回，段首检查容差 POS_TOL = 1e−6 nm。详见 `../2026-10-25_eta_hmc/RESULT.md`“rep2 续接失败的修复”。

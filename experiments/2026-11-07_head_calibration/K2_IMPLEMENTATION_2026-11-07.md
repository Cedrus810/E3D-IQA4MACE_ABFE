# K2 实施第一项：新身份、拒绝混链、K2 查询（2026-11-07，Claude）

依据：`K2_DECISION_2026-11-07.md` §5 的第一项，以及主设计 §7.1、§7.5、§10（已全文读完）。用户 2026-11-07 回复“可以”批准实施。
这一项只实现了查询和身份，**没有接入 η 链**：记录格式（q、R 字段）、统计和 CUDA 都是下一项。

## 改动

| 文件 | 内容 |
|---|---|
| `experiments/2026-10-25_eta_hmc/k2_energy.py`（新） | `ModelEnergyK2`：同一个 carved 簇上算 δU_K2 = C_bb − C_MM；head 原子读出 cH_i、d_i = cH_i − cMM_i、q = C_bb − Σ cH_i，满足 Σd + q = δU_K2。S_K 直接取零，不调用模型；部分 dummy 拒绝；配体净电荷不为 0 时拒绝，因为 `coupling_labels` 的总电荷写死为 0；只允许 float64；head 和 backbone 路径必须显式给出。`k2_identity` 记录 E_construction=`local_E_bb_v1`、attribution_definition、backbone／head 路径、实际加载的实空间求值器（从模块读出类名和 offset）、dtype、总电荷、Ω 规则 |
| `experiments/2026-10-25_eta_hmc/test_k2_energy.py`（新） | 9 项 CPU 测试，日志 `experiments/2026-10-25_eta_hmc/logs/test_k2_energy_20261107.log` |
| `k2_query_smoke.py`、`k2_query_smoke.json`、`k2_query_smoke.log`（本目录，新） | 真实模型冒烟 |

没有改 `eta_hmc.py`、`eta_records.py`、`eta_exchange.py` 和旧目标的任何代码。K2 的分量名和旧的不同：K2 不提供 `dU_atom`（它在旧目标里的和就是标量 δU）。这样，旧的验证器和报告工具遇到 K2 数据会直接报错，不会悄悄按旧口径求和。

## CPU 测试（9 项通过）

Σd + q = δU_K2；换 head 时 δU_K2 不变、只有 q 变；dummy 行非零或出现非有限值时报错；S0／S_K 识别，部分 dummy 拒绝；S_K 返回零且不调用模型；带电配体拒绝；缺路径或 float32 拒绝；后端标签区分 stencil／analytic，混用或找不到求值器时报错；`check_contract` 双向拒绝新旧目标混链，换后端拒绝，只改 mtime 不拒绝。

## 真实模型冒烟（181 CPU，8 线程，float64，0 MD）

acetamide 第 2 步 prep1 的第 0、500、995 帧，和第 2 步独立写的逐帧记录对照（单位 kJ/mol）：

| 检查 | 实测 |
|---|---|
| Σd + q − δU_K2 | ≤ 1.8e−14 |
| δU_K2 对第 2 步的 C_bb − C_MM | 0、0、−1.4e−9 |
| q 对第 2 步的 −r | ≤ 1.5e−9 |
| Σd 对第 2 步的旧 head δU | ≤ 1.5e−10 |
| Ω 水身份 | 3 帧都与第 2 步相同 |
| 同一帧重复查询 | δU 差 0，全部分量逐位相同 |
| S_K（acetamide 状态 2） | δU = q = 0，模型调用 0 次 |
| 部分 dummy（状态 1） | 拒绝 |
| 入口冒烟的旧链合同（`local_E v1`）当作 K2 续 | 拒绝（E_construction、attribution_definition、realspace_backend 等字段不同） |
| K2 合同当作旧链续 | 拒绝（E_construction 等字段不同） |

身份里实际读到的后端：`graph_longrange_stencil`，能量 offset 0.02 Å，场特征 offset 0.1 Å。
和第 2 步的 1e−9 量级差异，有的帧有、有的帧为 0，像多线程归约顺序带来的；没有专门查。

## 尚未完成（写于第一项完成时；前三条已在下面第二项完成）

- η 记录格式：加入 q、C_bb、d_atom 字段和 K2 身份，验证器逐帧检查 Σd + q = δU；续跑时复查 K2 分量。
- 原子统计：A_i^H、R_0 和数值误差 ε_eta 分开报告（`atomic_report.py` 目前只认旧口径）。
- CUDA 路径：只跑了 CPU。
- 每次查询 4 次 backbone forward（head 1 次 + C_bb 3 次）；复用 L+Ω 那次可以少 1 次，没做。

# K2 实施第二项：η 链记录、续跑、原子统计、CUDA 冒烟（2026-11-07，Claude，用户批准）

## 改动（全部是新文件；`eta_hmc.py`、`eta_records.py`、`atomic_report.py` 没改）

| 文件（`experiments/2026-10-25_eta_hmc/`） | 内容 |
|---|---|
| `k2_records.py` | K2 记录格式 `k2_1`，事务和续跑规则照搬 `eta_records.run_recorded`：分块提交、原子替换、精确 Context + RNG 续接、每个循环都记（包括拒绝重复）。验证器逐循环检查 C_H = learned + Coulomb、d = C_H − ΣC_MM、q = C_bb − ΣC_H、Σd + q = dU、C_bb − ΣC_MM = dU，以及 dummy 行、提议记录、坐标、Ω 映射。续跑时复查一次末状态，标量和所有 K2 分量（含 C_bb、q、Ω）都要对上。`E_construction` 不是 `local_E_bb_v1` 的直接拒绝 |
| `k2_report.py` | 用每个循环的 [d_1…d_n, q] 联合行做 η 积分：A_i^H、R（单列，不分摊）、标量 TI、相邻 BAR，以及 ε_eta = a^BAR − (ΣA + R)。块 bootstrap 联合重采样原子和 q。复用 `atomic_report.integrate`／`moving_block_sample` |
| `k2_chain.py` | K2 η 链入口：`ModelEnergyK2` + `k2_records`。起点由 `--start-file/--start-frame/--start-label` 显式给出（比如第 2 步的 prep 末帧）；输出文件名前缀是 `k2chain_`，不会和旧链同名 |
| `test_k2_records.py` | 4 项：玩具体系上的记账与拒绝状态、篡改 q／d／C_bb 被拦下、中断后重放结果逐位一致、完成后复用不调模型、两种记录格式互相拒绝、换后端拒绝续接、报告里 A + R 闭合且 R 单列 |

测试：K2 共 13 项全部通过（`logs/test_k2_records_20261107.log`）；旧 `test_atomic_records` 6 项回归也通过。

## CUDA 冒烟（181 2080 Ti，模型 CUDA float64，MM CUDA mixed）

acetamide S0，起点是第 2 步 prep1 的末帧，η = 0／0.5／1，各 20 个循环，tag `_cudasmoke`，日志在 `experiments/2026-10-25_eta_hmc/logs/k2_cudasmoke/`。

- 每次 K2 查询 0.81–1.04 s（2080 Ti，float64；只是冒烟耗时，不是生产耗时）。
- η=1 在第 10 个循环提交后被 timeout 中断，同一命令续跑：复查 1 次，加 10 次提议，共 11 次模型调用，跑到 20 循环完成。η=0 再跑一次时直接复用，模型调用 0 次。
- 中断时留下一个半写的 `k2chain_S0_eta1.000_prep1_cudasmoke.progress.tmp.npz`（旧的 eta_records 也会这样）。续跑不读它；文件还在 `runs/acetamide/`，没删。
- 三条链的记录都通过验证，Σd + q − dU 最大 4.3e−14 kJ/mol。
- 在 CPU 上复算 η=1 的 4 个整盒快照：dU、C_bb 差 ≤ 4.2e−9，q 差 ≤ 3.4e−9，d_i 差 ≤ 5.5e−10 kJ/mol，Ω 完全相同；CPU 和 CUDA 的身份字段一致。
- K2 报告端到端跑通，单起点、各 20 循环，数值没有意义：ΣA + R − a^TI = −7e−15，ε_eta = −0.33 kJ/mol，R 单列为 +14.05。

## 仍未做

- 生产规模的 K2 η 运行（多起点、η 网格、预算），需要单独的计划表。
- 每次查询 4 次 backbone forward，没有优化。
- 5090 上没跑过。

# K2 η 交换试点 B 计划：acetamide S0（2026-11-08，Claude 起草，等用户说“跑”）

依据：试点 A 结果（`RESULT.md`：各起点的 a_0^K2 相差 6.3 kJ/mol；η=0 的 4 个起点 ⟨δU⟩ 在 −11.8 到 −19.5 之间，而 MM 参考值是 −11.56；高 η 接受率 0.01–0.08），以及 `../2026-11-07_head_calibration/K2_IMPLEMENTATION_2026-11-07.md` 第三节（交换移植，已冒烟）。只用已冒烟的 `k2_exchange.py` 和 `k2_report.py`，不改代码。

要回答的问题：模型查询数和试点 A 相同时，η 交换能不能让 4 个起点的 a_0^K2 靠拢，η=0 标签能不能回到 MM 参考值。

## 计划表

| 项 | 内容 |
|---|---|
| 方法 | `k2_exchange.py`：每个起点一个 11 标签的交换梯子。每个 sweep，每个 walker 做一次 MM-HMC 提议（L=20，2 fs），之后相邻标签交换（只读缓存的 δU_K2，不额外调用模型） |
| η 网格 | 和 A 相同：0–1，间隔 0.1。依据：A 中相邻 η 的工作 sd ≤ 0.6 RT；冒烟时交换接受率 0.6–1.0 |
| 起点 | prep1–4 末帧，和 A 相同；每个梯子的 11 个 walker 都从该 prep 末帧出发（`--init last`） |
| 两种配置 | **B1**：`--eta0-mm-updates 1`，查询数和 A 一样，是干净的对照。**B2**：`--eta0-mm-updates 25`：每个 sweep 里，η=0 的 walker 先做 25 次纯 MM 子步，最后才查询一次模型（K2 版已冒烟）。查询数和 B1 相同，只多 MM 步。加 B2 的理由：A 显示 η=0 连 MM 分布都没覆盖，而交换只能把已有构型在标签之间搬运，生成新构型靠 η=0 |
| 规模 | 2 种配置 × 4 个起点 = 8 个梯子，每个 500 sweep。每个梯子 11 × 501 = 5,511 次查询（等于 A 每个起点 11 条 × 501），合计 44,088 次。MM：B1 每梯子 11 × 500 × 20 = 110,000 步；B2 再加 500 × 24 × 20 = 240,000 步 |
| 命令（每个梯子） | `python k2_exchange.py acetamide --etas 0,0.1,…,1 --start-file ../2026-11-07_head_calibration/mm/prepP_S0.npz --start-label prepP --head <nvar-b> --backbone <POLAR-1-M> --init last --sweeps 500 --checkpoint-every 50 --coords-every 10 --eta0-mm-updates {1,25} --tag {_pilotB1,_pilotB2}`，在 `experiments/2026-10-25_eta_hmc/` 下运行 |
| 机器 | PBS GPU 节点（用户给）或 5090。每个梯子一个进程（加载 1 份模型、11 个 MM Context）。同卡进程数按显存定：3090（24 GB）4 个，5080（16 GB）3 个（A 中 5080 同卡 8 个进程 OOM，单进程峰值 4.4 GiB；梯子进程多了 10 个 Context，显存没测过） |
| 预计耗时 | 依据：A 在 3090 上 4 进程同卡，实测每次查询 1.4–2.0 s（中位 1.93）。按此，每个梯子 5,511 × 1.93 s ≈ 3.0 h；B2 的额外 MM 每个 sweep 约 500 步，按 A 中 2080 Ti 的 0.12–0.18 ms/步，约 0.1 s，可忽略。一个 3090 节点跑 4 个梯子约 3 h；8 个梯子在一个节点上约 6 h，两个节点约 3 h。5080／5090 上梯子进程的耗时没测过 |
| 能否续跑 | 能：同一命令重跑，从最后提交的 sweep 续接；换 GPU 型号时不加载检查点，改从缓存坐标续接，并记一条 continuation |
| 输出 | `experiments/2026-10-25_eta_hmc/runs/acetamide/k2ladder_S0_prep{1..4}_pilotB{1,2}.npz`；报告写本目录 `report_pilotB.json` |
| 不做 | 不改 Ω／head／后端，不训练，不自动加长；B 之后是否加长（`--extend-from`）由用户定 |

## 跑完后报告（只报实测值，不设通过线）

和 A 并排，同一查询预算（discard 0.2）：

- **起点间差：** 4 个起点各自的 a_0^K2、R_0、ΣA_i^H，以及它们的极差；A 是 6.3 kJ/mol。
- **η=0 标签对 MM 参考：** η=0 标签的 ⟨δU_K2⟩ 是 V 系综下的量，和第 2 步 2000 个独立 MM 帧的 −11.56 kJ/mol 是同一个物理量，可以直接比。A 的 4 个起点给出 −11.8 到 −19.5。
- **每个标签：** 接受率、δU 的统计低效率 g、有效样本数；A 在高 η 只有 3–20。
- **交换和运输：** 各相邻对的交换接受率；walker 在 η=0 和 η=1 之间的往返次数；η=1 标签上新结构的来源（walker 编号）。
- **数值闭合：** ε_eta = a^BAR − (ΣA + R)。
- **B1 对 B2：** 多做 η=0 的 MM 更新有没有改善上面几项。

## 已知局限

- 4 个 prep 的初始几何相同（种子不同），起点间差可能低估真正的初态依赖。
- 一个梯子内各标签相关，`report_ladders` 的 bootstrap 把各标签独立重采样，只作诊断。
- 500 sweep 够不够没有依据；它只是和 A 等预算，目的是做对照。

## 提交（2026-11-08，用户给了 yayoi28、yayoi30，两台都是 3090）

- **只交 B1。** B2 的 `--eta0-mm-updates 25` 在 K2 上只做过 CPU 玩具测试，没有做过 CUDA 冒烟（181 上的冒烟只跑了 m=1）。按规定，没在 GPU 上跑通的代码不交 PBS，节点空着的时候也不现场冒烟。B2 等之后在 181 上冒烟通过再交。
- B1 的 4 个梯子分到两台：yayoi28 跑 prep1、prep2，yayoi30 跑 prep3、prep4，每台 2 个进程 × 8 线程（16 = 物理核数）。脚本 `pbs_pilotB1_yayoi28.zsh`、`pbs_pilotB1_yayoi30.zsh`，日志 `logs/pilotB/`。
- 耗时：同卡 2 个梯子进程时每次查询多快没测过；按 A 中 3090 同卡 4 进程的 1.93 s 估计上限，每个梯子 ≤ 3.0 h。

## B2 冒烟（2026-11-09，181 2080 Ti，用户同意挤用 181）

- `k2_exchange.py --eta0-mm-updates 25`，3 个标签（η 0／0.5／1），prep3 末帧，12 sweep，每 4 个 sweep 提交一次。45 s 时中断（已提交 8 个 sweep），同一命令续到 12 个，再跑一次直接复用（0 次模型调用）。日志在 `../2026-10-25_eta_hmc/logs/k2_ladder_m25_smoke/`。
- 模型调用 39 次（3 × 13），每次 K2 查询 1.03 s。记录通过验证，Σd + q − dU 最大 6.4e−14；η=0 行记录了 25 个子步（子步接受率 0.73），这些行的 dH 记为 NaN，其他行有限。
- η=0 walker 每个 sweep 的配体 RMS 位移平均 1.12 Å（试点 A 中 m=1 时，η=0 每次查询约 0.2 Å）。只有 12 个 sweep，只说明这条代码路径在 GPU 上跑通了。
- B2 的 PBS 脚本已备好：`pbs_pilotB2_slotA.zsh`（prep1、prep2）、`pbs_pilotB2_slotB.zsh`（prep3、prep4），host 写的是 `yayoiXX`，等用户给节点后填上。
- 用户 2026-11-09 给了 yayoi31（9950X 16 物理核，RTX 4080 SUPER 16 GB）。4 个 B2 梯子都放这一台：`pbs_pilotB2_yayoi31.zsh`，`xargs -P 3` 同时最多 3 个（3 × 4.4 GiB ≈ 13 GiB），每个 5 线程。slotA／slotB 两份脚本改名为 `_UNUSED`，不用。
- 耗时：4080S 上没测过。按 B1 在 3090 上每次查询约 1.0 s（同卡 2 进程），每个梯子约 1.5 h；4 个梯子分两轮，约 3 h（同卡 3 进程时会慢多少不知道）。

## B2 第一次运行（PBS 25406，yayoi31，2026-11-09 13:45–15:00）

- prep1、prep3 完成（500 sweep，每次查询 0.75 s）；prep2、prep4 在启动时 CUDA OOM，没有留下任何进度文件。
- 原因：我按独立链的进程峰值 4.4 GiB 设了同卡 3 个进程，但梯子进程实测 5.2–5.9 GiB（多了 10 个 MM Context），3 个放不下 16 GB。PLAN 里写过“梯子进程显存没测过”，却仍按 4.4 估了，这是我的错。
- 修改：`pbs_pilotB2_yayoi31.zsh` 改为同时 2 个进程 × 8 线程。重提时 prep1、prep3 直接复用，只新跑 prep2、prep4。这两个没有检查点，任何 GPU 型号都能跑；同卡 2 个进程约 12 GB，16 GB 和 24 GB 的卡都放得下。
- 预览（只有 2 个起点，不作结论）：η=0 标签 ⟨δU_K2⟩：prep1 −9.51（B1 −11.35），prep3 −11.08（B1 −14.03），MM 参考值 −11.56；a_0^K2：prep1 −29.41（B1 −33.37），prep3 −32.15（B1 −31.62）。见 `report_pilotB2_partial.json`。
- 2026-11-10：yayoi31 的重提没有执行过。改交 yayoi43（5950X 16 物理核，4080 SUPER 16 GB）：`pbs_pilotB2_yayoi43.zsh`，只跑 prep2、prep4，同卡 2 进程 × 8 线程；日志写 `logs/pilotB/B2rerun_prep*.log`，第一次 OOM 的日志保留不覆盖。

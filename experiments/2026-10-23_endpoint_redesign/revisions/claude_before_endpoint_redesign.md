# E3D — project rules

## 设计稿
- **新窗口先读 `HANDOFF_2026-10-23_groupwise.md`**（2026-10-23 交接，必读清单在它第一节）。
- **`DESIGN_2026-10-23_groupwise_dummy.md`**（2026-10-23 起，用户定）：**唯一的设计稿**。逐基团 dummy + MM 采样 + 模型对帧批量评估；**不用 ML/MM 采样（`--mode mlmm`）**。开工、验收前全文读完。
- `MODEL_DEFINITION_from_v3.md`：模型本身的定义（backbone、head、Coulomb 通道、D_ia），从旧 v3 原样摘出。设计稿里写的"v3 §2 / §3"就看这个文件。

## `Archive/` 是垃圾桶，不要动
- `Archive/` 里是作废的旧设计和旧状态：v1 / v2 / v3 计划、旧 `STATUS.md`、POLAR-shell 实现、性能计划、10-10 交接。全是"ML 驱动采样 / ML/MM"那条**已经放弃**的路线。
- **当它不存在**：不打开、不读、不 grep、不 ls 进去、不引用、不往外拿东西、不往里放东西、不删。
- 设计稿里出现的"v3 §13.x"只是出处标注，不用去 `Archive/` 查。
- 真需要旧文件里的东西，先问用户。

## 参考材料（不是设计稿）
- **`REF_pocket_water_from_ABFE_IBS_2026-10-14.md`**（2026-10-14 用户放入，来自 ABFE_IBS 那边）。
  - **性质**：外部参考，不改 v3 的任何决定。权威顺序排在用户决定和 v3 之后。
  - **数字未复核**：里面的数字 E3D 都没有复核过，引用时一律标【未核实】；要用的话，先在 E3D 体系上复测。
  - **用在哪**：复合物腿的 W3（撤墙、回填）；D15 里和 GCMC 有关的两项（插入的水用 ML 还是 MM、GCMC 跑在哪个环境）；以及 ML 水身份（D11、FIRES）和区域内水的交互。做到这几条线之前，先读完笔记开头的必读清单。
  - **不用在哪**：W1、W2，以及溶剂腿的 W3。溶剂腿撤墙按 v3 §3 已定的做法，用 MM 软核。
- **`REF_solvent_legs_from_ABFE_IBS_2026-10-15.md`**（2026-10-15 用户让放入，来自 fragment-opt 基准上的 ABFE-IBS）。
  - **是什么**：6 条中性 HCNO 溶剂腿，作为 Stage B 的候选，附 3-rep 的 MM ΔG_solv 对照值。文件在 `data/solvent_legs_ibs/`。
  - **性质**：参考，不改 v3。D9 仍开放；ΔG 是 IBS 的原值，E3D 没有复测，引用时标【未核实】。
  - **用之前**：先读该文件 §4。残基名是 `LIG` 不是 `MOL`；盒里有 Na⁺/Cl⁻；温度 300 K；配体 HMR；cif 坐标是预平衡之前的。

## 环境
**E3D 用 `openmm_dev`**（2026-10-10 用户决定；`run_*.sh`、`learn*.zsh` 已改指向它）。两个环境都是生产环境，都不许改。

| 环境 | 生产？ | 说明 |
|---|---|---|
| `/home/ruigengji/miniforge3/envs/openmm_dev_ubio` | **是（生产，2026-10-10 用户确认）** | 不许 pip/conda/mamba 安装、卸载、升级、拷贝。torch 2.12.1、mace-torch 0.3.16、graph_longrange 0.4.0、pymbar 4.0.3；e3nn 0.5.1（2026-10-08 16:18 重装）。2026-10-10 在 181 冒烟：POLAR-1-M + 20-frame head 正常，与 MACECalculator 一致到 0e+00（`experiments/2026-10-10_env_smoke/`）。STATUS 里"需 e3nn 0.4.4"已过时 |
| `/home/ruigengji/miniforge3/envs/openmm_dev` | **是（生产，2026-10-10 用户确认）** | e3nn 0.4.4、torch 2.12.0、mace-torch 0.3.16、graph_longrange 0.4.0、grand 1.1.0（GCMC）、openmmtools、pymbar 4.2.0。同一冒烟通过，结果与 ubio 逐位相同；`test_endpoints` / `test_attribution` / `test_openmm` 在 181 上通过（`experiments/2026-10-10_env_smoke/`）|

Backbone 文件：`/home/ruigengji/MLP/mace/`。Backbone：MACE-POLAR-1-M，已锁定（2026-10-10 用户决定，v3 §2.1 / D5）。

## 算力
- **生产 GPU 是 RTX 5090**（2026-10-11 用户说明）。181/183 的 2080 Ti 只用于冒烟；在 2080 Ti 上测的耗时不能当生产耗时。

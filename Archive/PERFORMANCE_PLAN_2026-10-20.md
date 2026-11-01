# E3D 性能优化：先测分项，再减少重复工作

2026-10-20。范围：D19 `--mode mlmm --shell-mm`，固定 POLAR-1-M、当前 head、n=20、D17、FIRES、时间步和路径。依据 `RESEARCH_PLAN_v3.md` §8、§11 R7、§13.45–13.46、§13.53，以及当前源码。优化后的物理与归因结果须单独验收。

## 1. 当前判断

应把性能作为下一条实现主线。现有运行记录支持程序能够推进；它们没有关闭采样收敛、有限 k FIRES 残余、跨分子转移和含盐腿 TI 等问题。性能优化以相同构型上的能量、力、导数和闭合保持为前提。

- 5090 的 **Atenolol、旧路径**测量是 n=20 时约 91 ms/step（§13.45）；不能拿它当 acetamide D19 的速度。
- acetamide D19 的 `full.log` 第一轮为 2,877 s / (30 × 500) = 192 ms/replica-step，含评估和保存。它与短跑约 0.28 s 的测量工作量不同。
- `segB.log`（cut + W3）记录末轮约 1,650 s / (16 × 500) = 206 ms/replica-step，含迭代开销。这里是混合分段的平均值，不能说明单个 W2 或 W3 的速度。
- 5090 的尺寸缩放支持固定开销值得调查；2080 Ti 与 3090 的跨作业耗时不能单独证明 GPU 不重要。CPU/线程、系统规模、分段比例及环境均需配对控制。

本轮只做源码审计、性能工具和执行方案。当前会话的 `nvidia-smi` 无法连接驱动；未启动 MD、GPU profiling、生产任务或环境修改，未声称实测加速。

## 2. 源码中可以确定的工作量

| 位置 | 现在做的工作 | 优化候选与约束 |
|---|---|---|
| `decomp/lambda_mask.py:anchored_energy` | 常规力调用做 full、L、I 三次 backbone；cap 与 D19 都用片段能量 | 合批三个独立图。保留各图电荷、batch/ptr、POLAR 非周期场和独立能量；先验证 POLAR 的 batch 独立性 |
| `decomp/polar_es.py:_Recorder.forward` | `per_atom=True` 时做两组 `_cross`，每组各调用三次 Coulomb；第二组还求 Euler 分摊梯度 | `AnchoredForce` 每步即使 `need_dudl=False` 仍请求逐原子 Coulomb。均匀 lambda 的传播可用总 cross 及其位置梯度；记录/非均匀 lambda 仍保留准确逐原子分摊 |
| `decomp/lambda_mask.py` 与 `decomp/head.py` | head 计算 node 分支和全部边，anchored 随后只用 L–I 边 | 独立推理接口只算 L–I pair readout；backbone 必须仍看到全图，训练接口和权重保持兼容 |
| `decomp/openmm_bridge.py:_graph` | 每次重建 CPU 邻居表并转成设备张量；片段再切图 | 先量 graph 与拷贝占比；如引入 skin 缓存，必须证明缓存失效规则覆盖位移、周期重包裹和原子身份变化 |
| `decomp/mlmm.py:MLMMForce.evaluate` | D19 q=3、t=0 仍走 full + head + fragments；q>3 才走单配体分支 | q=3 的传播能量可化为 E_L；求 cut 的单侧导数和 all-state 评估仍需要完整部分 |
| `decomp/mlmm_hremd.py:iterate` | 逐副本 `integrator.step(steps)`，每个 PythonForce 各算一次 ML | 跨副本合批要重做同步调度。仅改为线程并发有共享 backbone Coulomb wrapper 竞争风险；先做单配置合批 |

这些是调用结构，不是已测出的耗时占比。尤其不能把「三次 backbone」直接等同于「可加速三倍」。

## 3. 第一阶段：可重复基线

新增 `profile_mlmm.py`，默认 acetamide D19/n=20，分别测 W1、W2、lambda=0+、cut、q=3 及 W3。它只新建一个临时 OpenMM Context，使用新的输出目录；可只读已有 checkpoint 的各 q 构型。默认每态 5 步预热、3 组 × 10 步基线、3 次固定构型分项调用，全部为有界短测。

输出 `profile.json` 包括：

- 不插桩的 OpenMM ms/step、ns/day，及固定构型 force call 的原始重复计时；
- unwrap、graph、head、full/fragment backbone、Coulomb、autograd、FIRES 的嵌套计时与调用次数；
- inclusive 与 self 时间分开：只累加 self；各边界 CUDA 同步的分项时间含插桩开销，不能替代未插桩基线；
- 插桩前后相同构型的能量与力差；可选额外 torch CPU/CUDA trace；
- 主机、GPU、库版本、线程、参数、内水身份与当前源码 SHA256。模型/输入记录路径、大小、mtime，尚非完整权重哈希凭据。

GPU 空闲且可访问时，在 `openmm_dev` 内执行；不要与当前 ladder 共卡：

```sh
PYTHONDONTWRITEBYTECODE=1 /home/ruigengji/miniforge3/envs/openmm_dev/bin/python \
  /home/ruigengji/E3D/profile_mlmm.py \
  --threads 8 --qs 0,1.5,2,2.5,3,3.5,4 --trace \
  --out-dir /home/ruigengji/E3D/experiments/2026-10-20_performance/baseline_01
```

`--snapshot .../state.npz --qs ...` 只取 checkpoint 中明确存在的 q，拒绝路径不符。内水集合仍按原始 `--start stack.npz:index` 或 leg 起点选择，必须与该 run 一致；不能按每个 checkpoint 构型重选内水。默认预热和基线会推进临时轨迹；固定构型分项测量使用所选原构型。该工具不执行 HREMD 或自由能分析。

先看 CPU/CUDA trace 区分 kernel、CPU 调度和设备同步，再按占比决定下阶段顺序。实际 HREMD 的评估、逐原子 TI、交换及 checkpoint 开销仍须在后续短 ladder 配对测量中单独报告；此工具只覆盖传播与力调用。

## 4. 第二阶段：按可验证的收益逐项实现

1. **去掉传播中不需要的归因工作**：均匀 lambda 下，总 Coulomb 与其位置梯度单独进入传播；准确逐原子 Euler 分摊留在采样记录路径。并只计算 head 的跨边 readout、处理 exact endpoint 的传播短路。每项独立 A/B，保留参考实现。
2. **full/L/I 三图合批**：先测 batch=1/3 的单图能量、梯度、电荷响应与独立图一致性，再比较延迟与显存。若 batch 相互影响或显存显著增长而无端到端收益，停止该路线。
3. **跨副本合批**：定义同步推进和约束处理接口后再改 ladder。报告整条 ladder 的总 replica-steps/s、GPU-h/ns/state 和显存，不把 batch 吞吐改善冒充单副本延迟改善。
4. **剩余开销**：只有分项测量显示 graph、FIRES、host/device 往返或 kernel launch 占比足够大，才推进缓存、设备常驻、编译或 CUDA Graph。每项保留正确失效/回退路径。

第一轮目标是选出有可重复端到端收益的等价实现，再决定是否值得投入跨副本调度。暂不承诺倍数。MTS/更大时间步会改变积分误差，需要另行决定和动力学验收；不放进本轮等价实现优化。

## 5. 验收和停止条件

- 固定同一模型、构型、内水身份、n、wall、cap、精度、dt、线程和 GPU；初始正常构型与已有 cap/switch 活跃构型都要覆盖。
- float64 检查端点、能量/力、dU/dq、逐原子值及闭合；float32 配对误差须与当前实现重复调用的舍入噪声比较。阈值先记录再判断，不因候选结果调整。
- 保留 q=1/2/3 的单侧导数、D19 及旧路径、`need_dudl` 记录行为；不能通过只保总和改变逐原子归因。
- 未插桩计时至少三组，原始数据全部保留；收益必须超过重复计时波动。首轮检查不通过则停在候选实现，不替换生产路径。
- 正确性后才做短 ladder，比较分段耗时、完整迭代开销、交换与保存。性能与科学收敛分别报告；本方案不把短测通过写成独立科学验收。
- 使用新输出目录和源码身份；既有生产结果不覆盖。长跑仍按既有资源安排执行。

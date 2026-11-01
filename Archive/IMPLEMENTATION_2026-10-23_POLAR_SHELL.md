# POLAR 壳层候选路径与等价传播优化

2026-10-23，用户要求修正并提速。本次实现方案 2 的显式候选入口，保留 D19 和旧 SW 路径用于对照；它不构成“D19 已被证明物理错误”或“+33 已解释”的结论。没有启动生产采样、独立 QM 计算或修改生产环境。

## 1. 模型路径

`run_abfe.py --mode mlmm --polar-shell`：

| q | 段 | ML 部分 | FIRES / 墙 |
|---|---|---|---|
| 0–1 | W1 | E_POLAR(L∪I) | 配体形状 FIRES，开墙 |
| 1–2 | W2 | 原 anchored Hamiltonian，完整 POLAR 壳层 | 配体形状 FIRES，墙全开 |
| 2–3 | cut | 连续桥接至 E_POLAR(L)+E_POLAR(I) | 同上 |
| 3–4 | W3 | E_POLAR(L)+E_POLAR(I) | 保留 POLAR 壳层，软核撤墙 |
| 4–5 | shape | E_POLAR(L)+E_POLAR(I) | 墙关闭，FIRES 从配体形状桥接到首个重原子这个点 |

I–I 的 MM 非键相互作用保持关闭，`t_sw=0`；不存在 SW 或 dG_ref 段。shape 的具体实现是**能量的线性桥接**：

    U_F(a) = (1-a) U_F,shape + a U_F,point,   a=q-4
    dU/dq = U_F,point - U_F,shape

两套 FIRES 使用同一组固定内水，shape 中不重新选水；力由上述能量求导。端点及每个中间态均是明确的保守势，是否具备足够窗口重叠需另行采样验证。该桥接可能需要较密窗口，不以能量连续替代重叠验收。

**终点是点中心 POLAR 水壳层 + 外层 MM 溶剂中的气相 ML 配体参考，不是 bulk-MM 水。** 输出记录 `endpoint_reference` / `reference_note`，shape 作为独立全局自由能项进入闭合。与 D19/MM 的绝对总量对比仍需匹配参考态或完整转换循环；不能用旧总量直接替换。

CLI 的 `--polar-shell` 与 `--shell-mm` 互斥；也拒绝 `--ref`、`--mm-only`、`--no-fires` 的不相容组合。已有两条路径的坐标含义保留，新增入口必须显式选择。`--shape-windows` 控制 shape 段窗口数。

## 2. 防止不同模型的历史混用

Checkpoint 新增 `polar_shell`、`ref`、`inner`。续跑在恢复任何坐标或历史记录前检查路径、窗口和内水身份；不同 Hamiltonian 直接报错，替代原来的 shell_mm 警告后继续加载。旧 checkpoint 缺少新增标识时按原路径解释。

跨路径初始化只复用坐标和速度，不复用统计历史。涉及 POLAR-shell 候选时必须显式指定 `--init-q`，避免把旧 SW 的 q=4 当成新 shape 的 q=4。所有新计算使用新输出目录。

## 3. 性能修改

- FP64 快速传播在 lambda=1 直接求 backbone 能量：POLAR 壳层仅 full，D19 为 full-I，不计算无权重的 head、Coulomb 分摊或 cap；FP32 的该捷径未通过预设力差检查，因此 FP32 的 lambda=1 保留原路径。记录导数时始终走完整路径。
- 精确 cut 终点直接求独立片段，避免 full/head/Coulomb；记录 cut 的单侧导数时保留完整参考实现。
- FP64 中间态均匀 lambda 的传播直接使用总 Coulomb cross 及其位置梯度，不做逐原子 Euler 分摊；逐原子记录和 all-state 评价仍使用准确分摊。该接口要求标量 lambda，拒绝非均匀输入。**FP32 中间态保留原算法**：真实模型对照的力差超过了预设 0.05 kJ/mol/A，未放宽阈值放行。
- `--batch-fragments` 将独立 L、I 两张图合为一次 backbone 调用，保留各自 graph、charge、spin、cell 和 POLAR 场。**目前只允许 FP64**（CLI 同时设 `--ml-dtype float64`）；真实 FP64 检查通过，FP32 在 D19 上未通过力差验收，接口明确拒绝 FP32 合批。
- `--reference-forces` 禁用快速传播，供配对比较；结合不启用片段合批可回到原力调用计算结构。

最终放行范围：默认 FP32 只保留精确解耦端点捷径；FP32 原参考重复调用也出现超过既定阈值的差异，尚未校准误差基线，不能据此放行中间态优化。详细成功、失败及 CPU 计时全部保留在 RESULT 中。

没有以更大时间步或 MTS 换取吞吐；没有削弱多体项、修改 head 权重或调 cap 参数。GPU 驱动在当前会话不可访问，不能报告生产 GPU 加速倍数。

## 4. 验证入口

CPU 的小系统回归，不加载预训练权重、不积分：

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENMM_CPU_THREADS=2 \
  /home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B -m unittest -v test_mlmm_paths
```

覆盖快速/参考传播、显式 Coulomb、cap 开关、D19、片段合批、能量/力/逐原子导数，以及三条完整路径的直接 Context 能量、参数导数、shape 力和全局闭合、错误路径 checkpoint 拒绝。

真实 POLAR-1-M 的固定构型等价检查与 CPU 延迟比较（原始计时保存，输出文件必须新建；不推进轨迹）：

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 JAX_PLATFORMS=cpu \
  /home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B -u \
  experiments/2026-10-23_polar_shell/check_native.py --threads 2 --repeats 3 \
  --out experiments/2026-10-23_polar_shell/native_cpu_01.json
```

真实 FP64 模型的固定阈值：|dU|≤2e-5 kJ/mol、max|dF|≤1e-5 kJ/mol/A。FP32 检查的力差阈值为 0.05 kJ/mol/A，能量阈值为完整 backbone 绝对能量的 4 倍机器精度；超阈值的标量传播/合批组合保留失败记录并停止放行。CPU 延迟只描述本机该固定构型和精度，不能替代 GPU 计时。验收结果另写 `experiments/2026-10-23_polar_shell/RESULT.md`。

GPU 可访问时，用 `profile_mlmm.py --polar-shell --qs 0,1.5,2,2.5,3,3.5,4,4.5,5`，先对 `--reference-forces` 与默认快速传播做新目录配对短测。合批仅用于显式 FP64 对照，不能从 CPU FP64 推出生产 FP32 收益。之后才做短 ladder / NVE，并单独验证形状腿重叠、有限 k 残余和独立物理参考；单点回归不等于这些科学验收。

## 5. 更新（2026-10-23 10:30，用户决定）

生产模型只有 FP32，FP32 下 0.05 kJ/mol/Å 的力差门判不出等价（GPU 上同一实现重复调用就差 0.14–0.55）。上文第 3 节中所有"FP32 保留原算法 / 拒绝合批"的限制已删除，代数等价以 FP64 对照为准。5090 计时见 `experiments/2026-10-23_polar_shell/RESULT.md` 末节、v3 §13.60–13.61。

`--batch-fragments` 改为默认开（用户 2026-10-23），`--no-batch-fragments` 关闭；`--reference-forces` 同时关闭快速传播和合批，作为完整的参考路径。

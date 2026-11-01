> **已撤回当前路线（2026-10-23）**：此目录是上轮擅自添加的两端点重设计。
> 用户明确要求每拆一组 dummy 后重新运行该状态的 MM MD、补水和采样；本原型不实现该工作流，不能作为设计或执行入口。
> 下文仅保留已发生的数值实验记录，测试通过不表示用户路线已实现。

# 2026-10-23：端点路线重新设计与离线统计原型

本轮交付为可执行的设计和数值原型，不是新的分子自由能结果。

## 设计已落地

- 根目录唯一设计稿已重写为全部耦合／全部 dummy 两端先行，逐基团保留为扩展。
- 保留 MM 采样、ML 采样后一次加载批量评估；不运行 ML 力驱动的 MD。
- 明确定义 MM 恒等替换与 ML 局部替换两种输入，不借 IBS 数字缩放原子贡献。
- 以重要性权重校正系综，再用等先验加权 Bennett 和共同样本测度计算总量及分解。
- 原始原子项、backbone 残差和 MM 远场项共同闭合；另外提供明确均分约定的原子闭合表定义。
- 交接、CLAUDE、模型说明和中英日 README 入口同步；本轮没有修改 Archive 或旧 ML/MM 实现。
- 改写前主设计、交接和 CLAUDE 的副本保存在 `revisions/`。

## 实现

`estimate.py`：纯 NumPy；接收两端源／目标 reduced energies、分量与样本数。
输出源到目标的权重诊断、加权 Bennett 端点差、共同测度端点差、离线积分分量和闭合误差。
支持不足返回 `insufficient_support`；数值失败返回 `numerical_failure`。
CLI 两类失败退出码为 2，成功仅标 `diagnostics_passed`；`physical_validation` 始终为 `not_assessed`。
没有误差校准，`uncertainty` 为 null；不能把闭合或 ESS 门槛当作科学正确性证明。

## 实际验证

使用现有生产环境的 Python，只读使用依赖，未安装或修改环境：

```sh
cd /home/ruigengji/E3D/experiments/2026-10-23_endpoint_redesign
/home/ruigengji/miniforge3/envs/openmm_dev/bin/python -B -m unittest -v test_estimate
```

6 项通过：

1. 三状态模型，两个不同源分布，精确计数积分；对照独立配分函数与直接目标 Boltzmann 分量积分。
2. 故意省略源到目标权重的阴性对照，证实会得到错误答案。
3. 共同能量零点及各源能量常数改变后结果不变。
4. 两端样本数不同仍恢复精确配分函数比。
5. 无重叠时，即使形式上可闭合也拒绝支持验收。
6. 分量不闭合与非有限能量拒绝输入，不静默删帧。

`finite_state_validation.json` 保存独立三状态验证：

| 项目 | 实测，reduced units |
|---|---|
| 精确 Δf | −0.005138382969000423 |
| NBB Δf | −0.005138382969100177 |
| 共同测度 Δf | −0.005138382969027688 |
| 分量闭合误差 | 1.14e−15 |
| 32/64 点逐项积分差 | 5.55e−16 |

CLI 从 `/tmp/e3d_endpoint_finite_state.npz` 读取相同有限状态输入，正常退出，
输出 `cli_validation.json`；该 JSON 是数值示例，不是 acetamide 数据。
两份 Python 源码 AST 检查通过，设计和入口文档无行末空白；`git diff --check` 通过。

## 完成边界

已完成重设计、统计原型和独立数值测试。尚未实现整盒 MM／POLAR 能量导出器，
尚未计算真实分子的这套 ML 端点估计、块 bootstrap、N 敏感性或独立重复。
本轮没有启动模拟、调用 GPU、改动模型权重或重算已有 MM ABFE。
下一步接口与动作已经写进主设计 §6–7，不再让用户选择统计事实来推进实现。

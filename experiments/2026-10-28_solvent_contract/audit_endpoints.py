"""Read-only audit of existing rep1 S0..S4 exports; no model / MD calls.

Writes NEW JSON / Markdown audit artifacts in this directory only.
Run with the existing openmm_dev Python, using -B.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYSTEMS = ("CyclophilinD_ligand-8", "PWWP1_ligand-8")
R_EXPORT = 0.0083144626  # groupwise_mm.KT coefficient, not a changed energy convention


def audit_system(name):
    group_file = ROOT / "experiments/2026-10-23_groupwise_dummy/groups.json"
    definitions = json.loads(group_file.read_text())[name]["groups"]
    ligfile = ROOT / "data/solvent_legs_ibs" / name / "rep1/ligand_indices_solvent.json"
    lig = json.loads(ligfile.read_text())["ligand_indices"]
    groups = [g["system_index"] for g in definitions]
    if len(set(lig)) != len(lig) or sorted(sum(groups, [])) != sorted(lig):
        raise ValueError(f"{name}: groups do not partition ligand")
    local = {i: k for k, i in enumerate(lig)}
    gidx = [[local[i] for i in g] for g in groups]
    rows = []
    for j in range(len(groups) + 1):
        path = ROOT / "experiments/2026-10-24_local_E/runs" / name / f"endpoint_N50_S{j}.npz"
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            ml, mm, du, v = (np.asarray(z[k], float) for k in ("C_ML", "C_MM", "dU", "V"))
            frames, omega = z["frames"], z["omega_water_resid"]
            n, na = ml.shape
            assert na == len(lig) and mm.shape == (n, na, 2) and du.shape == v.shape == (n,)
            assert meta["system"] == name and meta["rep"] == 1 and meta["endpoint"] == j
            assert meta["K"] == len(groups) and meta["groups"] == groups
            assert meta["n_water"] == 50 and omega.shape == (n, 50)
            assert len(np.unique(frames)) == n and np.issubdtype(frames.dtype, np.integer)
            assert np.isfinite(ml).all() and np.isfinite(mm).all() and np.isfinite(du).all() and np.isfinite(v).all()
            dummy = np.zeros(na, bool)
            for g in gidx[:j]:
                dummy[g] = True
            assert np.array_equal(dummy, np.asarray(meta["dummy"], bool))
            assert np.all(ml[:, dummy] == 0) and np.all(mm[:, dummy] == 0)
            assert np.issubdtype(omega.dtype, np.integer)
            assert all(len(np.unique(w)) == 50 for w in omega)
            atomic = ml - mm.sum(axis=2)
            grouped = np.stack([atomic[:, g].sum(axis=1) for g in gidx], axis=1)
            atom_error = float(np.max(np.abs(atomic.sum(axis=1) - du)))
            group_error = float(np.max(np.abs(grouped.sum(axis=1) - du)))
            assert atom_error < 1e-9 and group_error < 1e-9
            source = Path(meta["source"])
            with np.load(source, allow_pickle=False) as src:
                assert int(src["state"]) == j
                assert frames.min() >= 0 and frames.max() < src["u"].shape[1]
                ref_v = src["u"][j, frames] * R_EXPORT * meta["T_K"]
                v_error = float(np.max(np.abs(v - ref_v)))
                assert v_error < 1e-8
                box = np.asarray(src["box"], float)
                assert np.all(np.diag(box) > 0)
            weights = np.exp(-(du - du.min()) / (R_EXPORT * meta["T_K"]))
            weights /= weights.sum()
            row = {"endpoint": j, "frames": n, "atoms": na, "coupled_atoms": int((~dummy).sum()),
                   "file": str(path), "source": str(source), "head": meta["head"], "backbone": meta["backbone"],
                   "temperature_K": meta["T_K"], "array_fields": list(z.files),
                   "max_abs_atomic_sum_minus_dU": atom_error, "max_abs_group_sum_minus_dU": group_error,
                   "max_abs_V_minus_source": v_error,
                   "mean_C_ML": float(ml.sum(axis=1).mean()), "mean_C_MM": float(mm.sum(axis=(1, 2)).mean()),
                   "mean_dU": float(du.mean()), "sd_dU_ddof1": float(du.std(ddof=1)),
                   "group_mean_C_ML": [float(ml[:, g].sum(axis=1).mean()) for g in gidx],
                   "group_mean_C_MM": [float(mm[:, g].sum(axis=(1, 2)).mean()) for g in gidx],
                   "group_mean_delta_u": grouped.mean(axis=0).tolist(),
                   "max_EXP_weight_diagnostic_only": float(weights.max()) if j < len(groups) else None,
                   "Kish_diagnostic_only": float(1 / (weights @ weights)) if j < len(groups) else None,
                   "a_status": "unvalidated_sampling_estimate" if j < len(groups) else "analytic_zero",
                   "separate_learned_channels_saved": False, "separate_explicit_Coulomb_saved": False,
                   "E_status": "derived_as_V_plus_dU_not_independently_exported",
                   "checks": "saved-array bookkeeping and source energy passed; no fresh model evaluation"}
            if dummy.all():
                assert np.all(du == 0)
            rows.append(row)
    assert len({r["head"] for r in rows}) == len({r["backbone"] for r in rows}) == 1
    return {"system": name, "ligand_system_indices": lig, "groups_system_indices": groups,
            "group_charge_sum_e": [g["q"] for g in definitions], "endpoints": rows}


def main():
    data = {"kind": "saved solvent endpoint bookkeeping audit; not scientific acceptance",
            "scope": "two systems, rep1, five endpoints, existing old-head exports only",
            "units": "kJ/mol", "systems": [audit_system(s) for s in SYSTEMS]}
    out = HERE / "endpoint_audit.json"
    out.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    lines = ["# 已有溶剂端点数据核对（2026-10-28）", "",
             "仅核对已有数组与来源能量；未调用模型、未运行 MD。sd 使用 ddof=1。", "",
             "| 体系 | 状态 | 帧数 | 耦合原子 | 平均 δU | sd(δU) | 原子求和最大误差 | a_j 状态 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for s in data["systems"]:
        for r in s["endpoints"]:
            status = "解析零" if r["a_status"] == "analytic_zero" else "未获得可用采样估计"
            lines.append(f"| {s['system']} | S{r['endpoint']} | {r['frames']} | {r['coupled_atoms']} | "
                         f"{r['mean_dU']:.3f} | {r['sd_dU_ddof1']:.3f} | {r['max_abs_atomic_sum_minus_dU']:.2e} | {status} |")
    lines += ["", "每态均检查：组和原子映射、dummy 零值、有限性、50 个唯一水身份、",
              "逐原子及逐组求和，以及 V 与来源 u[j]×RT 的一致性。", "",
              "四个学习通道与显式 Coulomb 已合并成 C_ML，旧端点文件无法拆回；",
              "S0 内部形变诊断另存了 D[20,21,35,4] 与 cross[20,21,35]，",
              "该诊断不能代替两体系各 S_j 的 500 帧生产导出。", "",
              "E=V+δU 仅按定义可重建，没有独立 E 文件可供交叉验证。模型版本仅有路径来源记录。", "",
              "上述检查只验证记账；不证明 head 准确、目标分布混合或 a_j 收敛。"]
    (HERE / "ENDPOINT_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(f"audited {sum(len(s['endpoints']) for s in data['systems'])} endpoints; {out}")


if __name__ == "__main__":
    main()

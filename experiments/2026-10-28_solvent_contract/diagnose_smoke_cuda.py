"""Postprocess the already saved 60-cycle CUDA smoke chain only.

Imports NumPy and the record validator, never a model, OpenMM or a simulator.
Input chain / topology / logs are not modified. New diagnostic files only.
"""

import json
import shlex
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/2026-10-25_eta_hmc"))
from eta_records import read_record  # noqa: E402

SOURCE = ROOT / "experiments/2026-10-25_eta_hmc/runs/CyclophilinD_ligand-8/chain_S0_eta0.500_start1_smoke_cuda.npz"
TOPOLOGY = ROOT / "data/solvent_legs_ibs/CyclophilinD_ligand-8/rep1/topology_solvent.cif"


def topology_atoms(path):
    headers, atoms, started = [], [], False
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            headers.append(line.split()[0])
            started = True
        elif started:
            if line.startswith("#") or line.startswith("loop_"):
                break
            if line.strip():
                values = shlex.split(line)
                assert len(values) == len(headers)
                atoms.append(dict(zip(headers, values)))
    resids, previous, resid = [], None, -1
    for a in atoms:
        key = tuple(a[k] for k in ("_atom_site.label_asym_id", "_atom_site.label_comp_id",
                                   "_atom_site.label_seq_id", "_atom_site.auth_seq_id"))
        if key != previous:
            previous, resid = key, resid + 1
        resids.append(resid)
    return atoms, np.asarray(resids)


def main():
    m, z = read_record(SOURCE)
    atoms, resids = topology_atoms(TOPOLOGY)
    assert len(atoms) == m["n_particles"]
    accepted = z["cycle"][z["acc"]]
    assert accepted.tolist() == [1, 2, 3, 5, 6]
    previous, frozen = 4, 5  # zero-based rows: cycle 5 -> cycle 6
    ml = z["C_ML"][frozen] - z["C_ML"][previous]
    mm = z["C_MM"][frozen] - z["C_MM"][previous]
    delta = z["dU_atom"][frozen] - z["dU_atom"][previous]
    np.testing.assert_allclose(delta, ml - mm.sum(axis=1), atol=1e-12, rtol=0)
    np.testing.assert_allclose(delta.sum(), z["dE"][frozen], atol=1e-10, rtol=0)
    indices = np.asarray(m["ligand_system_indices"])
    local = {int(i): n for n, i in enumerate(indices)}
    gid = {int(i): k + 1 for k, g in enumerate(m["groups"]) for i in g}
    atomic = []
    for i, si in enumerate(indices):
        atomic.append({"atom_local": i, "system_index": int(si), "group": gid[int(si)],
                       "element": atoms[si]["_atom_site.type_symbol"], "name": atoms[si]["_atom_site.label_atom_id"],
                       "change_C_ML": float(ml[i]), "change_C_MM_Coulomb": float(mm[i, 0]),
                       "change_C_MM_LJ": float(mm[i, 1]), "change_delta_u": float(delta[i]),
                       "frozen_delta_u": float(z["dU_atom"][frozen, i])})
    groups = []
    for k, g in enumerate(m["groups"]):
        ix = [local[int(i)] for i in g]
        groups.append({"group": k + 1, "change_C_ML": float(ml[ix].sum()),
                       "change_C_MM_Coulomb": float(mm[ix, 0].sum()), "change_C_MM_LJ": float(mm[ix, 1].sum()),
                       "change_delta_u": float(delta[ix].sum())})
    kt = 0.0083144626 * m["T_K"]
    logp = -(z["dH"] + m["eta"] * z["dE"]) / kt
    pacc = np.exp(np.minimum(logp, 0))
    rng = np.random.default_rng(m["acceptance_seed"])
    reconstructed = np.log(rng.random(len(logp))) < logp
    assert np.array_equal(reconstructed, z["acc"])
    assert rng.bit_generator.state == json.loads(str(z["rng_state"]))
    frozen_fields = {}
    for k in ("dU_atom", "C_ML", "C_MM", "omega_water_resid", "cluster_system_indices", "lig_xyz", "dU"):
        a = z[k][frozen:]
        frozen_fields[k] = bool(np.array_equal(a, np.repeat(a[:1], len(a), axis=0)))
        assert frozen_fields[k]
    assert np.all(z["xyz"] == z["xyz"][0])
    assert np.array_equal(z["xyz"][0, indices], z["lig_xyz"][frozen])
    x, box = z["xyz"][0], z["box"]
    bd = np.diag(box)
    assert np.array_equal(box, np.diag(bd))
    water = np.array([i for i, a in enumerate(atoms) if a["_atom_site.label_comp_id"] == "HOH"])
    oxy = np.array([i for i in water if atoms[i]["_atom_site.type_symbol"] == "O"])
    heavy = np.array([i for i in indices if atoms[i]["_atom_site.type_symbol"] != "H"])
    d = x[heavy, None] - x[None, oxy]
    d -= bd * np.round(d / bd)
    rr = np.linalg.norm(d, axis=2)
    hi, wi = np.unravel_index(rr.argmin(), rr.shape)
    ranked = []
    for r in np.unique(resids[water]):
        ix = np.flatnonzero(resids == r)
        dv = x[ix, None] - x[None, indices]
        dv -= bd * np.round(dv.reshape(-1, 3).mean(axis=0) / bd)
        ranked.append((float(np.linalg.norm(dv, axis=2).min()), int(r)))
    ranked.sort()
    assert {r for _, r in ranked[:50]} == set(z["omega_water_resid"][frozen])
    changed_in = np.setdiff1d(z["omega_water_resid"][frozen], z["omega_water_resid"][previous])
    changed_out = np.setdiff1d(z["omega_water_resid"][previous], z["omega_water_resid"][frozen])
    a, b = z["lig_xyz"][frozen], z["lig_xyz"][previous]
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    aligned = (a - a.mean(0)) @ rotation + b.mean(0)
    step = {"from_cycle": 5, "to_cycle": 6, "change_delta_U": float(delta.sum()),
            "change_C_ML": float(ml.sum()), "change_C_MM_Coulomb": float(mm[:, 0].sum()),
            "change_C_MM_LJ": float(mm[:, 1].sum()), "change_V": float(z["V"][5] - z["V"][4]),
            "change_U_eta": float(z["V"][5] - z["V"][4] + m["eta"] * delta.sum()),
            "dH": float(z["dH"][5]), "acceptance_probability": float(pacc[5]),
            "ligand_RMS_move_nm": float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))),
            "ligand_aligned_RMS_change_nm": float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=1)))),
            "omega_resid_in": changed_in.tolist(), "omega_resid_out": changed_out.tolist()}
    result = {"kind": "read-only stored-chain diagnosis; no energy queries / model / MD",
              "source": str(SOURCE), "topology_labels": str(TOPOLOGY),
              "head": m["head"], "eta": m["eta"], "T_K": m["T_K"], "units": "kJ/mol; distances nm",
              "accepted_cycles": accepted.tolist(), "frozen_delta_U": float(z["dU"][5]),
              "step_5_to_6": step, "group_changes": groups, "atom_changes": atomic,
              "post6": {"cycles": 54, "dH_range": [float(z["dH"][6:].min()), float(z["dH"][6:].max())],
                        "proposal_delta_U_range": [float(z["dE"][6:].min()), float(z["dE"][6:].max())],
                        "acceptance_probability_quantiles_0_25_50_75_100": np.percentile(pacc[6:], [0, 25, 50, 75, 100]).tolist(),
                        "sum_of_recorded_proposal_accept_probabilities": float(pacc[6:].sum())},
              "acceptance_decisions_reconstructed": True, "final_acceptance_RNG_matches": True,
              "post6_recorded_fields_identical": frozen_fields, "all_full_box_snapshots_identical": True,
              "post6_V_peak_to_peak_kJmol": float(np.ptp(z["V"][5:])),
              "full_box_snapshot_cycles": z["xyz_cycles"].tolist(),
              "geometry": {"min_ligand_heavy_water_O_nm": float(rr[hi, wi]),
                           "ligand_system_index": int(heavy[hi]), "water_residue_index": int(resids[oxy[wi]]),
                           "rank50_water": ranked[49], "rank51_water": ranked[50]},
              "limits": ["No rejected-proposal coordinates or per-atom proposal energies were saved.",
                         "No full-box state before cycle 6 is saved in this chain.",
                         "C_ML combines learned channels and explicit Coulomb; they cannot be separated here.",
                         "This smoke used the original head, not resp_v2 or nvar.",
                         "No claim of a physical metastable basin, collapse, or convergence follows from 60 cycles."]}
    out = HERE / "diagnostics"
    out.mkdir(exist_ok=True)
    (out / "smoke_cuda_low_tail.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# CUDA 冒烟低尾与停滞：已有记录分析（2026-10-28）", "",
             "只读 60 循环链及静态拓扑标签；没有调用模型、重算势能或运行 MD。", "",
             "## 可确认的接拒机制", "",
             "接受循环为 1、2、3、5、6；第 6 次以后 54 次全部拒绝。",
             "由保存的种子重建 60 个接拒随机数，所有接拒决定及最终 NumPy RNG 状态完全一致。",
             "这是接拒逻辑／记录一致性证据，不是 CUDA 提议数组逐位重放证明。", "",
             f"第 7–60 次提议的接受概率中位数 {np.median(pacc[6:]):.3g}，最大 {pacc[6:].max():.4f}；",
             f"记录提议的概率之和 {pacc[6:].sum():.4f}，不将它外推为其他链的等待时间。", "",
             "## 第 5→6 循环：低尾怎样形成", "",
             "| 项 | 变化 kJ/mol | 对 ΔδU 的贡献 |", "|---|---:|---:|",
             f"| CML（四学习通道＋显式 Coulomb 合计） | {ml.sum():+.4f} | {ml.sum():+.4f} |",
             f"| CMM 裸 Coulomb | {mm[:,0].sum():+.4f} | {-mm[:,0].sum():+.4f} |",
             f"| CMM 截断 LJ | {mm[:,1].sum():+.4f} | {-mm[:,1].sum():+.4f} |",
             f"| δU | {delta.sum():+.4f} | {delta.sum():+.4f} |", "",
             "模型合计耦合更有利，同时 MM LJ 变得较不有利，二者共同使修正下降。",
             "不能把 CML 的变化直接归因于学习 head：其显式 Coulomb 没有单独保存。", "",
             "| 组 | ΔCML | ΔCMM Coulomb | ΔCMM LJ | ΔδU |", "|---|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| G{g['group']} | {g['change_C_ML']:+.3f} | {g['change_C_MM_Coulomb']:+.3f} | "
                     f"{g['change_C_MM_LJ']:+.3f} | {g['change_delta_u']:+.3f} |")
    lines += ["", "逐原子静电有很大的正负抵消，不能将最大的负 δu 原子直接当作停滞原因。",
              "例：N11（系统索引 3）Δδu=−62.224，C9（14）则为 +46.994 kJ/mol；",
              "全配体 ΔCMM Coulomb 却仅 −0.491 kJ/mol。全原子表见 JSON。", "",
              "## 几何与 Ω 记录", "",
              f"配体第 5→6 次 RMS 位移 {step['ligand_RMS_move_nm']:.5f} nm；刚性拟合后仍为 {step['ligand_aligned_RMS_change_nm']:.5f} nm。",
              f"Ω 换入 {changed_in.tolist()}，换出 {changed_out.tolist()}（拓扑残基索引，不是水原子索引）。",
              "模型／MM 变化、配体变形与换水同时发生；未保存前一整盒状态，不能分离各自因果贡献。", "",
              f"停滞构型中最近的配体重原子—水 O 距离 {rr[hi,wi]:.5f} nm（O10—水残基 {resids[oxy[wi]]}）；",
              "单个距离不能证明或否定目标异常吸引。六个整盒抽样帧（10…60）完全相同，均是第 6 次接受后的构型。", "",
              f"固定坐标上的 V 记录存在 {np.ptp(z['V'][5:]):.2e} kJ/mol 的极小峰峰差；不将 V 误写成逐位常数，",
              "也不据此解释 15–70 kJ/mol 的提议修正代价。", "",
              "## 必须保留的结论边界", "",
              "本次使用原始 head（文件名无 resp_v2 / nvar），不能将停滞推广到尚未在本链使用的其他 head。",
              "已确认的是该状态下 MM 提议与模型修正严重失配；未确定它来自学习通道、显式 Coulomb、",
              "Ω 边界还是环境／内部运动的哪一部分，也未证明该低尾是合理的物理势阱。", "",
              "被拒提议只有标量 ΔδU / ΔH，没有逐原子分解和提议坐标；仅靠当前文件不能进一步定位拒绝通道。",
              "若后续确需因果诊断，应先确定最小的提议诊断导出内容，再另行安排有界计算；本轮不启动它。"]
    (out / "SMOKE_CUDA_LOW_TAIL.md").write_text("\n".join(lines) + "\n")
    print(f"saved read-only diagnostics: {out}")


if __name__ == "__main__":
    main()

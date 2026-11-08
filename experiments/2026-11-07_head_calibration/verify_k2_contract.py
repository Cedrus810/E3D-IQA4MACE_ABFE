"""Read-only data audit and exact finite-state K2 accounting checks. No molecular/model calls."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RT = .0083144626 * 298.15


def free_energy(energy):
    x = -np.asarray(energy) / RT
    return float(-RT * (x.max() + np.log(np.exp(x - x.max()).sum())))


def weights(energy):
    x = -np.asarray(energy) / RT
    p = np.exp(x - x.max())
    return p / p.sum()


def integrate(start, delta, parts):
    eta = np.linspace(0., 1., 4001)
    u = start[None, :] + eta[:, None] * delta[None, :]
    logw = -u / RT
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return np.trapezoid(w @ np.asarray(parts).T, eta, axis=0)


def finite_state_checks():
    background = np.array([0., 1.3, -.7, 2.])
    mm = np.array([[-2., -1., -3., .4], [-.5, 1., -1.5, -.2]])
    bb = np.array([-4.5, -1.2, -1., -3.8])
    head = np.array([[-1.1, -2.1, -.9, -.6], [-2.4, -.1, -.4, -1.]])
    v0 = background + mm.sum(axis=0)
    e0 = background + bb
    delta = bb - mm.sum(axis=0)
    remainder = bb - head.sum(axis=0)
    a_exact = free_energy(e0) - free_energy(v0)
    a_parts = integrate(v0, delta, np.vstack([head - mm, remainder]))
    b = integrate(v0, -mm.sum(axis=0), -mm)
    mm_exact = free_energy(background) - free_energy(v0)
    total = free_energy(background) - free_energy(e0)
    atoms, remainder_leg = b - a_parts[:-1], -a_parts[-1]
    r_mm = mm_exact - b.sum()
    r_eta = a_exact - a_parts.sum()
    closure = total - atoms.sum() - remainder_leg
    assert abs(a_exact - a_parts.sum()) < 1e-6
    assert abs(total - atoms.sum() - remainder_leg) < 1e-6
    assert abs(closure - (r_mm - r_eta)) < 1e-12
    # Change the head's distribution over atoms and configurations. Total K2
    # remains fixed; the named structural remainder compensates in attribution.
    changed = head.copy()
    changed[0] += np.array([.9, -.4, 1.1, .3])
    q2 = bb - changed.sum(axis=0)
    changed_parts = integrate(v0, delta, np.vstack([changed - mm, q2]))
    changed_total = (b - changed_parts[:-1]).sum() - changed_parts[-1]
    assert abs(changed_total - atoms.sum() - remainder_leg) < 1e-12
    assert np.max(np.abs(changed_parts[:-1] - a_parts[:-1])) > .01
    # Negative controls: a plain MM average and a separate target-to-target
    # free-energy difference are different quantities from this eta-path term.
    q_mm_mean = float(weights(v0) @ remainder)
    a_head = free_energy(background + head.sum(axis=0)) - free_energy(v0)
    target_difference = a_exact - a_head
    assert abs(q_mm_mean - a_parts[-1]) > 1e-3
    assert abs(target_difference - a_parts[-1]) > 1e-3
    return {"kind": "exact four-state independent arithmetic; not molecular acceptance",
            "a_K2_exact": a_exact, "A_head_atom_TI": a_parts[:-1].tolist(), "R0_structural_TI": float(a_parts[-1]),
            "f_atom": atoms.tolist(), "f_remainder": float(remainder_leg), "total_exact": total,
            "total_closure": float(closure), "numerical_r_MM": float(r_mm), "numerical_r_eta": float(r_eta),
            "changed_head_total_difference": float(changed_total - atoms.sum() - remainder_leg),
            "negative_controls": {"MM_mean_q": q_mm_mean, "separate_a_K2_minus_a_head": target_difference}}


def data_audit():
    r_all, bb_all, head_all, means, files = [], [], [], [], []
    identity = None
    for prep in range(1, 11):
        path = HERE / f"mm/prep{prep}_S0_r_s5.json"
        d = json.loads(path.read_text())
        rows = sorted(d["rows"], key=lambda row: row["frame"])
        assert d["complete_for_saved_frames"] and len(rows) == 200
        assert [q["frame"] for q in rows] == list(range(0, 1000, 5))
        target = {k: d[k] for k in ("N", "head", "backbone", "dtype", "units")}
        if identity is None:
            identity = target
        assert identity == target and target["N"] == 50 and target["dtype"] == "float64"
        r = np.array([q["r"] for q in rows])
        bb = np.array([q["bb_minus_MM"] for q in rows])
        head = np.array([q["dU"] for q in rows])
        assert np.isfinite(np.r_[r, bb, head]).all()
        assert np.max(np.abs(head - bb - r)) < 1e-8
        assert np.max(np.abs(np.diff([q["time_ps"] for q in rows]) - 10)) < 1e-8
        r_all.extend(r); bb_all.extend(bb); head_all.extend(head)
        means.append(float(r.mean()))
        files.append(str(path))
    r, bb, head = map(np.array, (r_all, bb_all, head_all))
    shift = float(r[:1000].mean())
    result = {"files": files, "identity": identity, "n_frames": len(r), "mean_r": float(r.mean()),
              "sd_r": float(r.std(ddof=1)), "SE_from_seed_run_means": float(np.std(means, ddof=1) / np.sqrt(10)),
              "validation_RMSE_after_constant_shift": float(np.sqrt(np.mean((r[1000:] - shift)**2))),
              "sd_deltaU_head": float(head.std(ddof=1)), "sd_deltaU_backbone": float(bb.std(ddof=1)),
              "mean_deltaU_backbone_MM_ensemble": float(bb.mean()),
              "scope": "MM S0 ensemble; distinct random seeds share the same initial geometry; no eta=1 claim"}
    saved = json.loads((HERE / "step2_summary.json").read_text())
    assert abs(result["mean_r"] - saved["pooled"]["mean_r"]) < 1e-12
    assert abs(result["validation_RMSE_after_constant_shift"] - saved["K1_split"]["validation_rmse_after_shift"]) < 1e-12
    return result


if __name__ == "__main__":
    output = {"data_audit": data_audit(), "finite_state": finite_state_checks(),
              "new_target_queries": 0, "new_backbone_forwards": 0, "new_MD_steps": 0}
    (HERE / "k2_decision_checks.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))

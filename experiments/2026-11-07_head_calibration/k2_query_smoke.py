"""K2 query smoke on real acetamide frames, CPU float64, 0 MD (K2_DECISION §5, first item).

Compares ModelEnergyK2 with the independently written step-2 rows on the same frames:
dU_K2 vs C_bb - C_MM, q vs -r, sum_i d_i vs the legacy head dU, C_bb vs C_bb; repeat-query reproducibility;
S_K zero without a model call; partial dummy refused; legacy acetamide chain contract refused for K2.
Differences are reported, not gated.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "experiments/2026-10-25_eta_hmc")]
import eta_hmc as eh  # noqa: E402
import k2_energy as k2  # noqa: E402
from eta_records import check_contract  # noqa: E402

FRAMES = (0, 500, 995)


def main():
    torch.set_num_threads(8)
    rows = {r["frame"]: r for r in json.loads((HERE / "mm/prep1_S0_r_s5.json").read_text())["rows"]}
    ident = json.loads((HERE / "mm/prep1_S0_r_s5.json").read_text())
    with np.load(HERE / "mm/prep1_S0.npz", allow_pickle=False) as z:
        xyz = z["xyz"].astype(np.float64)
    leg = eh.local_E.Leg("acetamide")
    me = k2.ModelEnergyK2(leg, 0, head=ident["head"], backbone=ident["backbone"], device="cpu")
    out = {"identity": me.identity, "frames": []}
    for f in FRAMES:
        du = me(xyz[f])
        c = me.last_components
        r = rows[f]
        out["frames"].append({
            "frame": f, "dU_K2": du, "q": float(c["q"]), "sum_d": float(c["d_atom"].sum()),
            "closure_sum_d_plus_q_minus_dU": float(c["d_atom"].sum() + c["q"] - du),
            "vs_step2_dU_K2": du - r["bb_minus_MM"], "vs_step2_q": float(c["q"]) + r["r"],
            "vs_step2_head_dU": float(c["d_atom"].sum()) - r["dU"], "vs_step2_C_bb": float(c["C_bb"]) - r["C_bb"],
            "omega_same_as_step2": c["omega_water_resid"].tolist() == r["omega_water_resid"]})
        print(json.dumps(out["frames"][-1]), flush=True)
    first = dict(me.last_components)
    du_again = me(xyz[FRAMES[-1]])
    out["repeat_dU_difference"] = du_again - out["frames"][-1]["dU_K2"]
    out["repeat_components_identical"] = all(np.array_equal(first[k], me.last_components[k]) for k in k2.K2_DETAILS)
    out["S0_model_calls"] = me.calls
    sk = k2.ModelEnergyK2(leg, leg.K, head=ident["head"], backbone=ident["backbone"], device="cpu")
    out["SK"] = {"dU": sk(xyz[0]), "model_calls": sk.calls, "q": float(sk.last_components["q"])}
    try:
        k2.ModelEnergyK2(leg, 1, head=ident["head"], backbone=ident["backbone"], device="cpu")
        out["partial_dummy"] = "NOT refused"
    except NotImplementedError as e:
        out["partial_dummy"] = f"refused: {e}"
    legacy = json.loads(str(np.load(ROOT / "experiments/2026-11-06_acetamide_entry/chain_eta1.0_smoke.npz")["meta"]))
    k2_request = {**{k: legacy[k] for k in legacy if k in ("system", "state", "T_K", "n_water")}, **me.identity}
    out["legacy_E_construction"] = legacy.get("E_construction")
    for name, (saved, req) in {"legacy_as_K2": (legacy, k2_request), "K2_as_legacy": (k2_request, legacy)}.items():
        try:
            check_contract(saved, req)
            out[name] = "NOT refused"
        except ValueError as e:
            out[name] = f"refused: {str(e)[:160]}"
    print(json.dumps({k: v for k, v in out.items() if k != "frames"}, indent=1, default=str))
    (HERE / "k2_query_smoke.json").write_text(json.dumps(out, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()

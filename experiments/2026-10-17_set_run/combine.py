"""E3D ML/MM hydration free energy per molecule against its MM value (v3 S13.51).

Run: python combine.py      -> combine.log (table) and combine.json

dG = [f(q 3) - f(q 0)]_A + [f(q 5) - f(q 3)]_B - dG_ref, all by MBAR on each run's own samples
(u shifted per state; exact, S13.50 fix):
  A (W1 + W2 + cut): the first-pass run <MOL> (q <= 3 there is the current Hamiltonian) or the
    revised-path run v2_full_<MOL>
  B (SW + W3): v2_tail_<MOL> (started from A's q = 3 state) or v2_full_<MOL>
  dG_ref: experiments/2026-10-18_dgref/ref_PWWP1_ligand-8, the neat-solvent reference on the
    PWWP1-8 box, used for all six IBS boxes; acetamide (no salt, smaller box) has none yet
Errors: MBAR's, added in quadrature; they understate the scatter (S13.48).
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.fe import build_mbar, free_energy_differences
HERE = os.path.dirname(os.path.abspath(__file__))
REF = "/home/ruigengji/E3D/experiments/2026-10-18_dgref/ref_PWWP1_ligand-8"
MM = {"HSP90_ligand-17": (48.86, 0.45), "HSP90_ligand-19": (35.94, 2.19), "HSP90_ligand-27": (56.78, 0.45),
      "CyclophilinD_ligand-4": (65.24, 0.46), "CyclophilinD_ligand-8": (60.63, 0.86),
      "PWWP1_ligand-8": (47.59, 0.27), "acetamide": (34.04, 0.14)}


def seg(d, qa, qb, kT):
    """MBAR f(qb) - f(qa) in kJ/mol with its error, from run dir d."""
    z = np.load(f"{d}/state.npz", allow_pickle=True)
    qs = np.asarray(z["qs"], float)
    u = np.array(json.load(open(f"{d}/records.json"))["u"])
    n, K, _ = u.shape
    u_kn = u.transpose(1, 0, 2).reshape(K * n, K).T
    shift = np.median(u_kn, axis=1)
    df, ddf = free_energy_differences(build_mbar(u_kn - shift[:, None], [n] * K))
    a, b = int(np.argmin(abs(qs - qa))), int(np.argmin(abs(qs - qb)))
    assert abs(qs[a] - qa) < 1e-6 and abs(qs[b] - qb) < 1e-6
    return (df[a, b] + shift[b] - shift[a]) * kT, ddf[a, b] * kT, n


rows = {}
kT300 = 0.008314462618 * 300.0
ref, ref_err, _ = seg(REF, 3.0, 4.0, kT300)
print(f"dG_ref (PWWP1-8 box, q 3 -> 4): {ref:+.2f} +- {ref_err:.2f} kJ/mol\n")
print(f"{'molecule':22s} {'A: W1+W2+cut':>16s} {'B: SW+W3':>18s} {'B - ref':>14s} {'E3D dG':>16s} {'MM (IBS)':>14s} {'E3D - MM':>9s}  samples A/B")
for M, (mm, mm_sd) in MM.items():
    kT = 0.008314462618 * (298.15 if M == "acetamide" else 300.0)
    full = f"{HERE}/v2_full_{M}"
    if os.path.exists(f"{full}/records.json") and os.path.exists(f"{full}/result.json"):
        A = seg(full, 0.0, 3.0, kT); B = seg(full, 3.0, 5.0, kT)
    elif os.path.exists(f"{HERE}/v2_tail_{M}/result.json"):
        A = seg(f"{HERE}/{M}", 0.0, 3.0, kT); B = seg(f"{HERE}/v2_tail_{M}", 3.0, 5.0, kT)
    else:
        print(f"{M:22s} not finished"); continue
    if M == "acetamide":
        print(f"{M:22s} {A[0]:+9.2f} +- {A[1]:4.2f} {B[0]:+11.2f} +- {B[1]:4.2f}   (no dG_ref for this box yet)"
              f"{'':16s} {mm:+7.2f} +- {mm_sd:.2f}           {A[2]}/{B[2]}")
        rows[M] = dict(A=A[0], A_err=A[1], B=B[0], B_err=B[1], mm=mm); continue
    g = A[0] + B[0] - ref
    e = float(np.sqrt(A[1] ** 2 + B[1] ** 2 + ref_err ** 2))
    rows[M] = dict(A=A[0], A_err=A[1], B=B[0], B_err=B[1], B_minus_ref=B[0] - ref, dG=g, err=e, mm=mm, mm_sd=mm_sd)
    print(f"{M:22s} {A[0]:+9.2f} +- {A[1]:4.2f} {B[0]:+11.2f} +- {B[1]:4.2f} {B[0] - ref:+8.2f}       "
          f"{g:+8.2f} +- {e:4.2f} {mm:+7.2f} +- {mm_sd:.2f} {g - mm:+9.2f}  {A[2]}/{B[2]}")
d = [r["dG"] - r["mm"] for r in rows.values() if "dG" in r]
if d:
    print(f"\n{len(d)} IBS molecules: mean(E3D - MM) {np.mean(d):+.2f}, MAE {np.mean(np.abs(d)):.2f}, "
          f"sd {np.std(d, ddof=1) if len(d) > 1 else float('nan'):.2f} kJ/mol")
json.dump(dict(ref=ref, ref_err=ref_err, rows=rows), open(f"{HERE}/combine.json", "w"), indent=1)

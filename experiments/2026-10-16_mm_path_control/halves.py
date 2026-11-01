"""Block check of the all-MM control vs the abfe_mm reference: MBAR on the first / second half."""
import glob, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.fe import mbar_free_energy
KT = 0.008314462618 * 298.15
print("path control (HREMD, 1000 samples per state):")
for s in (1, 2, 3):
    rec = json.load(open(f"nofires_seed{s}/records.json"))
    u = np.array(rec["u"]); n, K, _ = u.shape
    out = []
    for name, sl in (("all", slice(0, n)), ("1st half", slice(0, n // 2)), ("2nd half", slice(n // 2, n))):
        x = u[sl]; m = len(x)
        r = mbar_free_energy(x.transpose(1, 0, 2).reshape(K * m, K).T, [m] * K, kT=KT)
        out.append(f"{name} {r['dG']:+.2f} +- {r['dG_err']:.2f}")
    print(f"  seed {s}: " + "; ".join(out))
print("abfe_mm reference (31 states in series, 1000 samples per state, no subsampling here):")
L = "/home/ruigengji/E3D/experiments/2026-10-15_acetamide_leg/runs"
for rep in (1, 2, 3):
    U = [np.load(f"{L}/rep{rep}_state{k:02d}.npz")["u"] for k in range(31)]
    n = U[0].shape[1]; out = []
    for name, sl in (("all", slice(0, n)), ("1st half", slice(0, n // 2)), ("2nd half", slice(n // 2, n))):
        blocks = [u[:, sl] for u in U]
        r = mbar_free_energy(np.concatenate(blocks, axis=1), [b.shape[1] for b in blocks], kT=KT)
        out.append(f"{name} {r['dG']:+.2f} +- {r['dG_err']:.2f}")
    print(f"  rep {rep}: " + "; ".join(out))

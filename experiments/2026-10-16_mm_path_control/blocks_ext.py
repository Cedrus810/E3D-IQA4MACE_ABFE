"""MBAR per 500-sample block (0.5 ns per state) and on the last 1000 samples, for the extended control runs."""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.fe import mbar_free_energy
KT = 0.008314462618 * 298.15
for s in (1, 2, 3):
    u = np.array(json.load(open(f"nofires_seed{s}/records.json"))["u"]); n, K, _ = u.shape
    def g(sl):
        x = u[sl]; m = len(x)
        r = mbar_free_energy(x.transpose(1, 0, 2).reshape(K * m, K).T, [m] * K, kT=KT)
        return f"{r['dG']:+.2f} +- {r['dG_err']:.2f}"
    blocks = [g(slice(i, i + 500)) for i in range(0, n - 499, 500)]
    line = f"seed {s} ({n} samples): blocks of 0.5 ns: " + ", ".join(blocks)
    if n >= 2000:
        line += f"; last 1 ns {g(slice(n - 1000, n))}; all {g(slice(0, n))}"
    print(line, flush=True)

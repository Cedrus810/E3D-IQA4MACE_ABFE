"""Re-run mlmm_hremd.analyze on finished runs (records.json + state.npz), e.g. after the MBAR shift fix.

Run: python reanalyze.py DIR [DIR ...]     -> DIR/result_reanalysed.json, one summary line per run
"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.mlmm_hremd import analyze
for d in sys.argv[1:]:
    z = np.load(f"{d}/state.npz", allow_pickle=True)
    rec = json.load(open(f"{d}/records.json"))
    rec["u"] = [np.array(x) for x in rec["u"]]
    T = 300.0 if "ligand" in d else 298.15
    r = analyze(z["qs"], rec, 0.008314462618 * T, z["n_acc"] / np.maximum(z["n_att"], 1))
    json.dump(r, open(f"{d}/result_reanalysed.json", "w"), indent=1)
    keys = [k for k in ("dG_W1", "dG_W2", "dG_cut", "dG_SW", "dG_W3", "dG_TI", "dG_MBAR", "dG_MBAR_err", "overlap_min") if k in r]
    print(os.path.basename(d.rstrip("/")) + ": " + ", ".join(f"{k[3:] if k.startswith('dG_') else k} {r[k]:+.2f}" for k in keys)
          + ("" if "dG_MBAR" in r else f"  MBAR: {r.get('mbar_error', '')[:80]}"), flush=True)

"""Why batch 4 != batch 1 in step_eval.couplings (smoke 2026-10-23): repeat, identical copies, float64."""
import sys
import numpy as np
import torch
sys.path.insert(0, "..")
import step_eval as se
from model_eval import EV, load_model, topology

name = "PWWP1_ligand-8"
top = topology(name)
lig = top[0]
model = load_model()
xyz = np.load(se.gm.HERE / "runs" / name / "rep1_S0.npz")["xyz"][:2]
c0, c1 = [se.carve(x, top, 50) for x in xyz]
nd = np.zeros(len(lig), bool)
for dt in (torch.float32,):
    model.to(dt)
    torch.set_default_dtype(dt)
    a = se.couplings(model, [c0], nd)
    b = se.couplings(model, [c0], nd)
    cc = se.couplings(model, [c0, c0, c0, c0], nd)
    d = se.couplings(model, [c0, c1], nd)
    f = lambda x: f"{np.abs(x).max() * EV:.2e}"
    print(f"{dt}: alone twice {f(a - b)}; 4 copies vs alone {f(cc - a)}; copies among themselves {f(cc - cc[0])}; "
          f"[c0,c1][0] vs alone {f(d[0] - a[0])}; sum alone {a.sum() * EV:+.4f}", flush=True)

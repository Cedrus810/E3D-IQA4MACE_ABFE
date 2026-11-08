"""Model evaluation of the saved MM frames: per-atom D_iE, per-group D_gE, TI by group (HANDOFF §1, §4).

Run: python model_eval.py eval SYSTEM [N_WATER] [STRIDE] [MAX_FRAMES]   (all states of runs/SYSTEM/rep1_S*.npz)
     python model_eval.py ti SYSTEM [N_WATER]                           (averages + TI from the eval npz)

MM only made the frames; every energy here is the model's (MODEL_DEFINITION_from_v3.md §3):
    U(R, lambda) = E_bb(R) - sum_i (1 - lambda_i) D_iE(R),   dU/dlambda_i = D_iE
D_iE = d U / d lambda_i at lambda = 1 (full graph; D_ia does not depend on lambda) plus the explicit
L/E Coulomb channel per ligand atom (polar_es.coulomb_cross per_atom; head trained with E3D_ES=1),
as sumD + cross in test_scale.py. Cluster per frame: ligand + the N_WATER nearest waters (ions left
out), non-periodic, as the head was trained (n <= 50).
TI by group, decoupling g (on_g 1 -> 0) with every frame of S_g (on_g = 1) and S_{g+1} (on_g = 0):
    dG_g = -int_0^1 <D_gE>_on d on  ~=  -(<D_gE>_{S_g} + <D_gE>_{S_{g+1}}) / 2,   total = sum_g dG_g
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ruigengji/E3D")
import decomp  # noqa: E402,F401
from decomp.lambda_mask import alchemical_energy  # noqa: E402
from decomp.mace_adapter import MACEDecomposition  # noqa: E402
from decomp.openmm_bridge import AlchemicalForce, carve_from_arrays  # noqa: E402
from decomp.polar_es import coulomb_cross  # noqa: E402

import groupwise_mm as gm  # noqa: E402
from openmm import app  # noqa: E402

DEV = "cuda"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = "/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
EV = 96.485332                                   # kJ/mol per eV


def load_model(ckpt=None, backbone=None, device=DEV):
    bb = torch.load(backbone or BB, map_location="cpu", weights_only=False).to(torch.float32).to(device).eval()
    torch.manual_seed(0)
    model = MACEDecomposition(bb, hidden=256, n_pair_channels=4).to(device).eval()
    model.head.load_state_dict(torch.load(ckpt or CKPT, map_location=device))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def topology(name):
    _, lig, groups, _, box = gm.leg(name)
    cif = app.PDBxFile(str(gm.input_dir(name) / "topology_solvent.cif"))
    atoms = list(cif.topology.atoms())
    numbers = np.array([a.element.atomic_number for a in atoms])
    resids = np.array([a.residue.index for a in atoms])
    resnames = np.array(["MOL" if i in set(lig) else a.residue.name for i, a in enumerate(atoms)])
    keep = np.where((resnames == "MOL") | (resnames == "HOH"))[0]          # ions out of the cluster
    return lig, groups, box, numbers, resids, resnames, keep


def d_ie(model, force, pos):
    """Per-ligand-atom D_iE (eV) for one carved cluster."""
    data = force._graph(pos)
    lam = torch.ones(len(force.frag_id), dtype=force.dtype, device=force.device, requires_grad=True)
    with coulomb_cross(model.backbone, force.frag_id, per_atom=True) as rec:
        U, _, _ = alchemical_energy(model, data, lam, force.frag_id, mode="graph")
    g, = torch.autograd.grad(U.sum(), lam)
    L = force.frag_id == 0
    return (g[L] + rec["per_atom"][L]).detach().cpu().numpy().astype(np.float64)


def evaluate(name, n_water=50, stride=1, max_frames=None):
    lig, groups, box, numbers, resids, resnames, keep = topology(name)
    K = len(groups)
    model = load_model()
    out = gm.HERE / "runs" / name / (f"model_N{n_water}" + ("_smoke" if max_frames else "") + ".npz")
    res, force = {}, None
    t0 = time.perf_counter()
    nf = 0
    for k in range(K + 1):
        z = np.load(gm.HERE / "runs" / name / f"rep1_S{k}.npz")
        frames = range(0, len(z["xyz"]), stride)
        if max_frames:
            frames = list(frames)[:max_frames]
        D = []
        for s in frames:
            X = z["xyz"][s].astype(np.float64)[keep] * 10
            pos, num, frag, idx = carve_from_arrays(X, numbers[keep], resids[keep], resnames[keep],
                                                    n_solvent=n_water, box_ang=np.diag(box) * 10)
            assert (keep[idx[:len(lig)]] == np.array(lig)).all()
            if force is None:
                force = AlchemicalForce(model, num, frag, mode="graph")
            D.append(d_ie(model, force, pos))
            nf += 1
        res[f"D_S{k}"] = np.array(D) * EV                                  # [frames, n_lig] kJ/mol
        print(f"{name} S{k}: {len(D)} frames, {time.perf_counter() - t0:.0f} s so far "
              f"({(time.perf_counter() - t0) / nf * 1e3:.0f} ms/frame)", flush=True)
    np.savez(out, n_water=n_water, stride=stride, **res)
    print(f"wrote {out}")


def ti(name, n_water=50):
    from pymbar import timeseries
    _, lig, groups, _, _ = gm.leg(name)
    pos_in_lig = {a: i for i, a in enumerate(lig)}
    K = len(groups)
    z = np.load(gm.HERE / "runs" / name / f"model_N{n_water}.npz")
    gidx = [[pos_in_lig[a] for a in g] for g in groups]

    def mean_se(x):
        g = timeseries.statistical_inefficiency(x) if len(x) > 10 and x.std() > 0 else 1.0
        return x.mean(), x.std(ddof=1) * np.sqrt(g / len(x))

    total, var = 0.0, 0.0
    print(f"{name}, N_water {n_water}  (kJ/mol; D_gE < 0 is attractive coupling)")
    for g in range(K):
        a, sa = mean_se(z[f"D_S{g}"][:, gidx[g]].sum(1))
        b, sb = mean_se(z[f"D_S{g + 1}"][:, gidx[g]].sum(1))
        dG = -(a + b) / 2
        total += dG
        var += (sa ** 2 + sb ** 2) / 4
        print(f"  G{g + 1}: <D_gE> on {a:+9.2f} +- {sa:.2f}   off {b:+11.2f} +- {sb:.2f}   dG {dG:+9.2f}")
    ref = gm.REF_IBS.get(name)
    reference = f"   IBS {ref[0]:+.2f} +- {ref[1]:.2f} [unverified]" if ref else ""
    print(f"  total {total:+.2f} +- {np.sqrt(var):.2f}{reference}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "eval":
        evaluate(a[1], *(int(x) for x in a[2:5]))
    elif a[0] == "ti":
        ti(a[1], *(int(x) for x in a[2:3]))

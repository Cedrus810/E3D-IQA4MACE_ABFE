"""Does the head's ligand-environment coupling survive the move from dimers to
a solvated cluster?

Run: python test_scale.py BACKBONE CKPT [n_solvent,n_solvent,...] [structure.npz]
     python test_scale.py MACE-POLAR-1-M runs/polar-M/ckpt/joint_..._D.pt

The head is trained on DES370K dimers: 20-50 cross edges per structure, and
L_int pins only their SUM. A per-edge bias is invisible there and grows as
n_edges * delta in condensed phase. This carves the prepared Atenolol leg at
increasing solvation and compares the head against the backbone's own coupling,
which needs no head and no alchemy:

    truth = E(all) - E(ligand) - E(environment)

First measured without backbone-labelled clusters in training:

    POLAR-1-M  err/edge -1.5 .. -2.1 meV, linear from 229 to 1524 edges
               (-3.1 eV of a -1.5 eV coupling at 341 atoms)
    omol-0     sign flips: +0.35, +1.31, +1.52, +0.99, then -1.69 eV at 341
               atoms -- the S13.14 carve, where it happens to agree to 0.07 eV

`truth` is computed in float64 (`coupling_labels`, as for the training
clusters): in float32 it is a difference of ~3e4 eV totals, good only to
+-0.05 eV, which is 2 kT and would swamp the spread measured below.

A stacked structure file ([n_frames, n_atoms, 3], e.g. from `em_system.py
frames`) scans every frame and summarises each size over frames. The number
that matters for S13.23 is the spread of the remainder r = truth - sumD: with
the total anchored to the backbone, r(R) IS the work of the lambda=0 cut step,
so its standard deviation over configurations sets that step's BAR overlap
(a few kT is fine, tens of kT is not).
Read-only on /home/ruigengji/MLP/mace; downloads nothing.
"""

import os
import sys

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.clusters import coupling_labels
from decomp.mace_adapter import MACEDecomposition
from decomp.openmm_bridge import AlchemicalForce, carve_from_arrays
from decomp.lambda_mask import alchemical_energy
from decomp.polar_es import es_coupling

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MDIR = "/home/ruigengji/MLP/mace"
KT = 0.025693                                    # eV at 298.15 K
# Head trained with E3D_ES=1 (v3 S2.3): it carries no L/E Coulomb, so add the
# explicit channel to sumD and the backbone's electrostatic coupling to direct,
# which keeps every column comparable with heads trained without it.
ES = os.environ.get("E3D_ES", "0") == "1"


def coupling(force, pos):
    """direct = U(1) - U(0) and sumD = dU/dlambda at lambda=1, summed over the
    ligand. test_peel.energies also peels 41 atoms; nothing here uses that."""
    data, frag = force._graph(pos), force.frag_id
    one = torch.ones(len(frag), device=force.lam.device)
    U = lambda lam: alchemical_energy(force.model, data, lam, frag, mode=force.mode)[0].sum()
    with torch.no_grad():
        direct = float(U(one) - U(torch.zeros_like(one)))
    lam = one.clone().requires_grad_(True)
    g = torch.autograd.grad(U(lam), lam)[0]
    return direct, float(g[frag == 0].sum())


def main(bb_name, ckpt, sizes=(5, 10, 20, 50, 100), hidden=256, channels=4,
         structure="em_solvent.npz"):
    d = np.load(structure, allow_pickle=True)
    box = d["box"] if "box" in d else None
    stack = d["positions"][None] if d["positions"].ndim == 2 else d["positions"]

    bb = torch.load(f"{MDIR}/{bb_name}.model", map_location="cpu",
                    weights_only=False).to(torch.float32).to(DEV).eval()
    torch.manual_seed(0)
    model = MACEDecomposition(bb, hidden=hidden,
                              n_pair_channels=channels).to(DEV).eval()
    model.head.load_state_dict(torch.load(ckpt, map_location=DEV))
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"{bb_name}  <-  {ckpt}  [{structure}, {len(stack)} frame(s)]")
    print(f"{'frame':>6s}{'n_solv':>7s}{'atoms':>7s}{'edges':>8s}{'truth':>10s}{'sumD':>10s}"
          f"{'direct':>10s}{'dir-tru':>10s}{'err/edge':>10s}   (eV; meV/edge)")
    rows = []                                   # (frame, n, edges, truth, sumD, direct)
    for f, positions in enumerate(stack):
        for n in sizes:
            pos, numbers, frag, _ = carve_from_arrays(
                positions, d["numbers"], d["resids"], d["resnames"],
                n_solvent=n, box_ang=box)
            truth = float(coupling_labels(bb, [pos], [numbers], [frag], device=DEV)[0])

            force = AlchemicalForce(model, numbers, frag, mode="graph")
            ei = force._graph(pos)["edge_index"]
            n_cross = int((force.frag_id[ei[0]] != force.frag_id[ei[1]]).sum()) // 2
            direct, sumD = coupling(force, pos)
            if ES:
                cross, d_es = es_coupling(bb, pos, numbers, frag, device=DEV)
                sumD, direct = sumD + cross, direct + d_es
            rows.append((f, n, n_cross, truth, sumD, direct))
            print(f"{f:6d}{n:7d}{len(numbers):7d}{n_cross:8d}{truth:10.4f}{sumD:10.4f}"
                  f"{direct:10.4f}{direct-truth:10.4f}"
                  f"{(sumD-truth)/max(n_cross,1)*1e3:10.3f}", flush=True)

    R = np.array(rows)
    e, err = R[:, 2], R[:, 4] - R[:, 3]
    slope = float((e * err).sum() / (e * e).sum())      # fit through the origin
    print(f"\n  sumD error ~ {slope*1e3:+.3f} meV x n_edges"
          f"   (-> {slope*1524:+.2f} eV at the 1524-edge S13.14 carve)")
    if len(stack) > 1:
        print(f"\n  over {len(stack)} frames; r = truth - sumD is the anchored cut-step work"
              f" (S13.23)\n{'n_solv':>7s}{'edges':>8s}{'truth':>14s}{'r mean':>9s}"
              f"{'r sd':>7s}{'sd/kT':>7s}{'dir-tru':>14s}")
        for n in sizes:
            m = R[:, 1] == n
            t, r, dt = R[m, 3], R[m, 3] - R[m, 4], R[m, 5] - R[m, 3]
            print(f"{n:7d}{R[m, 2].mean():8.0f}{t.mean():8.3f}+-{t.std():4.2f}"
                  f"{r.mean():9.3f}{r.std():7.3f}{r.std()/KT:7.1f}"
                  f"{dt.mean():8.3f}+-{dt.std():4.2f}", flush=True)
    return slope


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sizes = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 \
        else (5, 10, 20, 50, 100)
    main(sys.argv[1], sys.argv[2], sizes,
         structure=sys.argv[4] if len(sys.argv) > 4 else "em_solvent.npz")

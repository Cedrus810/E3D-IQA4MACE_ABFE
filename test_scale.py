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

`truth` is float32 differences of ~3e4 eV totals, good to about +-0.05 eV.
Read-only on /home/ruigengji/MLP/mace; downloads nothing.
"""

import sys

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.data import mace_batch
from decomp.mace_adapter import MACEDecomposition
from decomp.openmm_bridge import AlchemicalForce, carve_from_arrays
from test_peel import energies

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MDIR = "/home/ruigengji/MLP/mace"


def main(bb_name, ckpt, sizes=(5, 10, 20, 50, 100), hidden=256, channels=4,
         structure="em_solvent.npz"):
    d = np.load(structure, allow_pickle=True)
    box = d["box"] if "box" in d else None

    bb = torch.load(f"{MDIR}/{bb_name}.model", map_location="cpu",
                    weights_only=False).to(torch.float32).to(DEV).eval()
    torch.manual_seed(0)
    model = MACEDecomposition(bb, hidden=hidden,
                              n_pair_channels=channels).to(DEV).eval()
    model.head.load_state_dict(torch.load(ckpt, map_location=DEV))
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def E(p, z):
        b = mace_batch([p], [z], bb, charges=[0], device=DEV)
        return float(bb(b, training=False, compute_force=False)["energy"])

    print(f"{bb_name}  <-  {ckpt}")
    print(f"{'n_solv':>7s}{'atoms':>7s}{'edges':>8s}{'truth':>10s}{'sumD':>10s}"
          f"{'direct':>10s}{'dir-tru':>10s}{'err/edge':>10s}   (eV; meV/edge)")
    rows = []
    for n in sizes:
        pos, numbers, frag, _ = carve_from_arrays(
            d["positions"], d["numbers"], d["resids"], d["resnames"],
            n_solvent=n, box_ang=box)
        lig, env = frag == 0, frag == 1
        truth = E(pos, numbers) - E(pos[lig], numbers[lig]) - E(pos[env], numbers[env])

        force = AlchemicalForce(model, numbers, frag, mode="graph")
        ei = force._graph(pos)["edge_index"]
        n_cross = int((force.frag_id[ei[0]] != force.frag_id[ei[1]]).sum()) // 2
        direct, sumD, _, _ = energies(force, pos)
        err = sumD - truth
        rows.append((n_cross, err))
        print(f"{n:7d}{len(numbers):7d}{n_cross:8d}{truth:10.4f}{sumD:10.4f}"
              f"{direct:10.4f}{direct-truth:10.4f}{err/max(n_cross,1)*1e3:10.3f}",
              flush=True)

    e, r = np.array(rows).T
    slope = float((e * r).sum() / (e * e).sum())        # fit through the origin
    print(f"\n  sumD error ~ {slope*1e3:+.3f} meV x n_edges"
          f"   (-> {slope*1524:+.2f} eV at the 1524-edge S13.14 carve)")
    return slope


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sizes = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 \
        else (5, 10, 20, 50, 100)
    main(sys.argv[1], sys.argv[2], sizes,
         structure=sys.argv[4] if len(sys.argv) > 4 else "em_solvent.npz")

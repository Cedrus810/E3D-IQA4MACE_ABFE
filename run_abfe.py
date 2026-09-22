"""Double decoupling with the learned alchemical Hamiltonian (WP8/WP9).

Run: python run_abfe.py --leg solvent --lambdas 12 --iters 200
     python run_abfe.py --leg complex --lambdas 16 --iters 400 --restraint boresch.json

Two legs, each an HREMD ladder over the per-atom coupling:

    dG_bind = dG_solvent - dG_complex + dG_restraint + dG_standard

The solvent leg needs neither restraint nor standard-state term, so it is also
the whole calculation for a hydration free energy (validation ladder stage B,
FreeSolv). Run it first: it exercises the entire pipeline with two fewer moving
parts, and it has experimental references.

What is NOT reimplemented here: restraint construction, standard-state
corrections, GROMACS topology handling and the production protocol, which exist
in /home/ruigengji/ABFE_IBS and should be called rather than rewritten. This
file is the alchemical Hamiltonian wired into a replica ladder; the surrounding
ABFE machinery is borrowed.
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.hremd import HREMD
from decomp.mace_adapter import MACEDecomposition
from decomp.openmm_bridge import AlchemicalForce, make_system


def load_model(ckpt, backbone_path, hidden=256, n_pair_channels=4, device="cuda"):
    bb = torch.load(backbone_path, map_location="cpu",
                    weights_only=False).to(torch.float32).to(device).eval()
    model = MACEDecomposition(bb, hidden=hidden,
                              n_pair_channels=n_pair_channels).to(device)
    if ckpt:
        model.head.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def lambda_schedule(n, kind="linear"):
    """Windows go 1 -> 0, so TI integrates straight to the decoupling free energy.

    'quadratic' clusters windows near lambda=0, where the coupling changes
    fastest and neighbouring windows stop overlapping first.
    """
    t = np.linspace(0.0, 1.0, n)
    return 1.0 - (t if kind == "linear" else t ** 2)


def build_leg(model, numbers, frag_id, positions_ang, lambdas, mode="graph",
              charge=0, multiplicity=1, need_systems=True):
    systems, forces = [], []
    for lam in lambdas:
        f = AlchemicalForce(model, numbers, frag_id, mode=mode,
                            charge=charge, multiplicity=multiplicity)
        f.set_lambda(float(lam))
        forces.append(f)
        systems.append(make_system(numbers, f) if need_systems else None)
    return systems, forces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=["solvent", "complex"], default="solvent")
    ap.add_argument("--structure", required=True,
                    help=".npz with positions (Angstrom), numbers, frag_id")
    ap.add_argument("--ckpt", default="ckpt/joint_organic_n50000_h256_s32000_D.pt")
    ap.add_argument("--backbone",
                    default="/home/ruigengji/MLP/mace/mace-omol-0-extra-large-4M.model")
    ap.add_argument("--lambdas", type=int, default=12)
    ap.add_argument("--schedule", choices=["linear", "quadratic"], default="linear")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--steps-per-iter", type=int, default=500)
    ap.add_argument("--equil", type=int, default=20)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--mode", choices=["graph", "edge"], default="graph",
                    help="'edge' is the v1 control: wrong endpoint by design (S13.7)")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--multiplicity", type=int, default=1)
    ap.add_argument("--backend", choices=["torch", "openmm"], default="torch",
                    help="torch for dimer-scale methodology; openmm for PBC/PME/"
                         "restraints at condensed-phase scale (S13.13)")
    ap.add_argument("--device", default="cpu",
                    help="cpu is faster than cuda below ~100 atoms")
    ap.add_argument("--record-every", type=int, default=1,
                    help="u_kn costs n^2 energy evaluations per recording")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.structure, allow_pickle=True)
    pos, numbers, frag = d["positions"], d["numbers"], d["frag_id"]
    n_lig = int((frag == 0).sum())
    lambdas = lambda_schedule(args.lambdas, args.schedule)

    print(f"leg={args.leg}  {len(numbers)} atoms  ligand {n_lig}  "
          f"{args.lambdas} windows ({args.schedule})  mode={args.mode}")
    print(f"  {args.iters} iterations x {args.steps_per_iter} steps x "
          f"{args.dt} fs = {args.iters*args.steps_per_iter*args.dt/1000:.1f} ps "
          f"per window after {args.equil} equilibration iterations", flush=True)

    model = load_model(args.ckpt, args.backbone, device=args.device)
    systems, forces = build_leg(model, numbers, frag, pos, lambdas,
                                mode=args.mode, charge=args.charge,
                                multiplicity=args.multiplicity,
                                need_systems=args.backend == "openmm")

    t0 = time.perf_counter()
    r = HREMD(systems, forces, pos * 0.1, lambdas, T=args.T, dt_fs=args.dt,
              backend=args.backend, numbers=numbers,
              record_every=args.record_every).run(
        args.iters, args.steps_per_iter, n_equil=args.equil)
    r["wall_seconds"] = time.perf_counter() - t0
    r["leg"], r["mode"], r["n_ligand"] = args.leg, args.mode, n_lig

    print(f"\n  dG_TI   {r['dG_TI']:+.4f} kJ/mol = "
          f"{r['dG_TI']/4.184:+.3f} kcal/mol")
    if "dG_MBAR" in r:
        print(f"  dG_MBAR {r['dG_MBAR']:+.4f} +- {r['dG_MBAR_err']:.4f} kJ/mol = "
              f"{r['dG_MBAR']/4.184:+.3f} kcal/mol")
        ov = r["overlap"]
        print(f"  overlap min {ov['min']:.3f}   "
              + ("all windows connected" if ov["ok"]
                 else f"poor windows {ov['poor_windows']} -- add lambda points there"))
    else:
        print(f"  MBAR failed: {r.get('mbar_error')}")
    print(f"  swap acceptance {np.round(r['acceptance'], 2)}")
    print(f"\n  per-atom dG (ligand, kcal/mol):")
    pa = r["dG_TI_per_atom"][:n_lig] / 4.184
    for i, v in enumerate(pa):
        print(f"    atom {i:3d}  Z={int(numbers[i]):3d}  {v:+8.3f}")
    print(f"    {'sum':>9s}      {pa.sum():+8.3f}   "
          f"(total {r['dG_TI']/4.184:+.3f})")
    print(f"\n  {r['wall_seconds']:.0f}s")

    out = args.out or f"logs/abfe_{args.leg}_{args.mode}_{args.lambdas}w.json"
    pathlib.Path("logs").mkdir(exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in r.items() if k != "overlap"} |
        {"overlap_min": r.get("overlap", {}).get("min")}, indent=2))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()

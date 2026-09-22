"""Peel the ligand atom by atom and check it against a direct reference.

Run: python test_peel.py [--structure X.npz] [--steps 2000]

The reference needs no alchemy: at any configuration, the exact
ligand-environment coupling is just two energy evaluations,

    dU_total = U(lambda = 1) - U(lambda = 0)

because graph masking makes lambda = 0 exactly the isolated fragments (S13.7).
The decomposition claims that same number is the sum of per-atom pieces,

    dU_total  ==  sum_i sum_{a in E} D_ia

and that peeling atoms off one at a time, in any order, accumulates to it.
So MD supplies an ensemble and every frame is checked three ways:

    direct      U(1) - U(0)
    decomposed  sum over ligand atoms of sum_a D_ia
    peeled      remove atom 1, then 2, ... recording each step's energy change

Free energies need the lambda path and sampling convergence on top of this;
energies do not, which is why this comes first. If the three disagree on a
single frame, no amount of sampling will fix it.
"""

import argparse

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.lambda_mask import alchemical_energy
from decomp.mace_adapter import MACEDecomposition
from decomp.openmm_bridge import AlchemicalForce
from decomp.sampling import Langevin, masses


def energies(force, pos_ang, order=None):
    """Direct, decomposed and peeled coupling energies at one configuration."""
    from decomp.lambda_mask import lambda_edge_weight
    data = force._graph(pos_ang)
    frag, dev = force.frag_id, force.lam.device
    n = len(frag)
    lig = (frag == 0).nonzero().squeeze(-1)

    def U(lam_vec):
        return float(alchemical_energy(force.model, data, lam_vec, frag,
                                       mode=force.mode)[0].sum())

    one, zero = torch.ones(n, device=dev), torch.zeros(n, device=dev)
    direct = U(one) - U(zero)

    # per-atom dU/dlambda at full coupling: sum_a D_ia for each ligand atom
    lam = one.clone().requires_grad_(True)
    Uc, _, _ = alchemical_energy(force.model, data, lam, frag, mode=force.mode)
    g = torch.autograd.grad(Uc.sum(), lam)[0].detach()
    decomposed = float(g[lig].sum())

    # peel: switch ligand atoms off one at a time, recording each step
    idx = lig.tolist() if order is None else list(order)
    lam = one.clone()
    prev, steps = U(lam), []
    for i in idx:
        lam[i] = 0.0
        cur = U(lam)
        steps.append(prev - cur)
        prev = cur
    return direct, decomposed, np.array(steps), g[lig].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default="geom_water_dimer.npz")
    ap.add_argument("--from-abfe", default=None,
                    help="an ABFE_IBS output dir; loads the prepared solvent leg")
    ap.add_argument("--leg", default="solvent")
    ap.add_argument("--n-solvent", type=int, default=100,
                    help="nearest whole solvent residues kept in the ML region")
    ap.add_argument("--backbone",
                    default="/home/ruigengji/MLP/mace/MACE-OFF24_medium.model")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--equil", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--minimise", type=int, default=200,
                    help="LBFGS steps before MD; 0 disables")
    ap.add_argument("--restraint-k", type=float, default=2.0,
                    help="eV/A^2 harmonic restraint on the fragment COM "
                         "separation. A gas-phase dimer is unbound at 298 K and "
                         "drifts past the cutoff in under a picosecond, after "
                         "which D_LE is identically zero and every check passes "
                         "on nothing. 0 disables.")
    ap.add_argument("--restraint-r0", type=float, default=3.0,
                    help="Angstrom; target COM separation.")
    args = ap.parse_args()

    if args.from_abfe:
        from decomp.openmm_bridge import carve_ml_region, from_abfe_output
        sysm, top, full_pos, _, _ = from_abfe_output(args.from_abfe, leg=args.leg)
        from openmm import unit as _u
        bv = sysm.getDefaultPeriodicBoxVectors()
        box = np.array([bv[i][i].value_in_unit(_u.angstrom) for i in range(3)])
        pos, numbers, frag, _ = carve_ml_region(top, full_pos,
                                                n_solvent=args.n_solvent,
                                                box_ang=box)
        print(f"  carved from {args.from_abfe}: {len(numbers)} atoms "
              f"({args.n_solvent} nearest solvent residues of "
              f"{top.getNumResidues()-1})")
    else:
        d = np.load(args.structure, allow_pickle=True)
        if "resnames" in d:               # a minimised box from em_system.py
            from decomp.openmm_bridge import carve_from_arrays
            pos, numbers, frag, _ = carve_from_arrays(
                d["positions"], d["numbers"], d["resids"], d["resnames"],
                n_solvent=args.n_solvent,
                box_ang=d["box"] if "box" in d else None)
            print(f"  carved from {args.structure}: {len(numbers)} atoms "
                  f"({args.n_solvent} nearest solvent residues)")
        else:
            pos, numbers, frag = d["positions"], d["numbers"], d["frag_id"]
    n_lig = int((frag == 0).sum())

    bb = torch.load(args.backbone, map_location="cpu",
                    weights_only=False).to(torch.float32).to(args.device).eval()
    torch.manual_seed(0)
    model = MACEDecomposition(bb, hidden=args.hidden,
                              n_pair_channels=args.channels).to(args.device).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=args.device)
        try:
            model.head.load_state_dict(sd)
        except RuntimeError as exc:
            raise SystemExit(
                f"checkpoint does not match the head built here.\n"
                f"  {exc}\n"
                f"  pass --hidden/--channels matching how it was trained, and a "
                f"--backbone whose node_feats width matches.") from None
        print(f"  loaded head from {args.ckpt}")
    for p in model.parameters():
        p.requires_grad_(False)
    force = AlchemicalForce(model, numbers, frag, mode="graph")

    print(f"peel check: {len(numbers)} atoms, ligand {n_lig}, "
          f"{args.steps} MD steps at {args.T} K, sampling every {args.stride}")

    # MD at full coupling supplies the ensemble
    p0 = torch.tensor(pos, dtype=torch.float32, device=args.device)

    lig_mask = torch.as_tensor(frag == 0, device=args.device)
    env_mask = ~lig_mask

    def energy_fn(p):
        data = force._graph(p.detach().cpu().numpy())
        data["positions"] = p
        U = alchemical_energy(model, data, force.lam, force.frag_id,
                              mode="graph")[0].sum()
        if args.restraint_k > 0:
            # The restraint biases the ensemble but not the identities under
            # test: at every configuration the three energies must still agree.
            d = (p[lig_mask].mean(0) - p[env_mask].mean(0)).norm()
            U = U + args.restraint_k * (d - args.restraint_r0) ** 2
        return U

    if args.minimise:
        from decomp.sampling import minimize
        p0 = minimize(energy_fn, p0, n_steps=args.minimise)

    dyn = Langevin(energy_fn, p0, masses(numbers, p0.device), T=args.T,
                   dt=0.5, gamma=1.0, seed=0)
    print(f"  T = {dyn.temperature():.0f} K at start", flush=True)

    rows, fwd, rev, per_atom_all = [], [], [], []
    rng = np.random.default_rng(0)
    if args.steps == 0:
        # No dynamics: the identities are per-configuration, so one physical
        # structure decides them. Sampling only matters for free energies.
        frames = [pos]
    else:
        frames = None
    for step in range(args.equil + args.steps):
        dyn.step()
        if step == 0 or step == args.equil:
            print(f"  T = {dyn.temperature():.0f} K at step {step}", flush=True)
        if step < args.equil or (step - args.equil) % args.stride:
            continue
        x = dyn.pos.detach().cpu().numpy()
        # diagnostics first: all-zero energies mean the fragments drifted apart,
        # not that the identities hold
        sep = float(np.linalg.norm(x[frag == 0].mean(0) - x[frag == 1].mean(0)))
        dmat = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        cross = int(((dmat < force.r_max) &
                     (frag[:, None] != frag[None, :])).sum() // 2)
        direct, decomposed, peeled, per_atom = energies(force, x)
        if cross == 0:
            print(f"  frame {len(rows)+1:3d}  SKIPPED: fragments {sep:.2f} A "
                  f"apart, no cross-boundary edges", flush=True)
            continue
        # the same peel in reverse order: the sum is a state function
        _, _, peeled_rev, _ = energies(force, x, order=list(range(n_lig))[::-1])
        rows.append((direct, decomposed, peeled.sum(), peeled_rev.sum()))
        fwd.append(peeled); rev.append(peeled_rev[::-1])
        per_atom_all.append(per_atom)
        print(f"  frame {len(rows):3d}  COM {sep:5.2f} A  {cross:3d} cross edges"
              f"   direct {direct:+10.6f}  decomposed {decomposed:+10.6f}  "
              f"peeled {peeled.sum():+10.6f}  (rev {peeled_rev.sum():+10.6f}) eV",
              flush=True)

    if frames is not None:
        for x in frames:
            sep = float(np.linalg.norm(x[frag == 0].mean(0) - x[frag == 1].mean(0)))
            dmat = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
            cross = int(((dmat < force.r_max) &
                         (frag[:, None] != frag[None, :])).sum() // 2)
            direct, decomposed, peeled, per_atom = energies(force, x)
            _, _, peeled_rev, _ = energies(force, x,
                                           order=list(range(n_lig))[::-1])
            rows.append((direct, decomposed, peeled.sum(), peeled_rev.sum()))
            fwd.append(peeled); rev.append(peeled_rev[::-1])
            per_atom_all.append(per_atom)
            print(f"  static frame  COM {sep:5.2f} A  {cross:4d} cross edges"
                  f"   direct {direct:+10.6f}  decomposed {decomposed:+10.6f}  "
                  f"peeled {peeled.sum():+10.6f}  (rev {peeled_rev.sum():+10.6f}) eV",
                  flush=True)

    if not rows:
        raise SystemExit("no frame had cross-boundary edges -- raise "
                         "--restraint-k or lower --restraint-r0")
    a = np.array(rows)
    print(f"\n  over {len(a)} frames, mean +- sd (eV):")
    for k, name in enumerate(["direct U(1)-U(0)", "decomposed sum D_ia",
                              "peeled (forward)", "peeled (reverse)"]):
        print(f"    {name:24s} {a[:,k].mean():+10.6f} +- {a[:,k].std():.6f}")
    print(f"\n  |direct - decomposed|  max {np.abs(a[:,0]-a[:,1]).max():.3e} eV")
    print(f"  |direct - peeled|      max {np.abs(a[:,0]-a[:,2]).max():.3e} eV")
    print(f"  |forward - reverse|    max {np.abs(a[:,2]-a[:,3]).max():.3e} eV"
          f"   <- order dependence of the TOTAL (must vanish)")

    f, r, g = np.array(fwd), np.array(rev), np.array(per_atom_all)
    print(f"\n  per-atom coupling energy (eV), mean over frames:")
    print(f"    {'atom':>5s} {'dU/dlambda_i':>14s} {'peel fwd':>12s} "
          f"{'peel rev':>12s}")
    for i in range(n_lig):
        print(f"    {i:5d} {g[:,i].mean():14.6f} {f[:,i].mean():12.6f} "
              f"{r[:,i].mean():12.6f}")
    print(f"\n  per-atom spread between peel orders: "
          f"{np.abs(f.mean(0)-r.mean(0)).max():.3e} eV"
          f"   <- order dependence PER ATOM (real, and why the diagonal path exists)")


if __name__ == "__main__":
    main()

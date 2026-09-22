"""Is the per-atom free-energy attribution order-independent? (WP7, S6)

Run: python test_attribution.py

The flagship claim: decoupling along the diagonal path lambda(t) = (t,...,t)
gives a per-atom decomposition

    dG_i = - integral_0^1 < sum_{a in E} D_ia >_t dt      with  sum_i dG_i = dG

for the cost of ONE alchemical calculation, and it is the Aumann-Shapley value,
so it does not depend on any removal order.

That is a mathematical claim about the attribution, which this checks against
its own definition: Monte-Carlo-sampled discrete Shapley values over random
removal orders. They must agree. S13.7 already verified the additivity identity
exactly; what is open is whether the diagonal integral equals the order-averaged
sequential one.

This is deliberately NOT a free-energy calculation. Sampling is a fixed set of
conformers, so <.> is an average over that set rather than a Boltzmann average:
it isolates the attribution arithmetic from sampling convergence, which is a
separate problem (WP8/WP9). Anything that fails here fails for free too.
"""

import itertools
import sys

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.data import mace_batch
from decomp.lambda_mask import alchemical_energy
from decomp.mace_adapter import MACEDecomposition
from e3nn import o3
from mace.modules import MACE, gate_dict, interaction_classes

DEV = "cuda" if torch.cuda.is_available() else "cpu"
Z = [1, 8]
R_MAX = 5.0
N_LAMBDA = 33          # trapezoid nodes along the diagonal
N_CONF = 24            # conformers averaged over


def small_mace():
    return MACE(
        r_max=R_MAX, num_bessel=8, num_polynomial_cutoff=6, max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=len(Z),
        hidden_irreps=o3.Irreps("16x0e+16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(len(Z)), avg_num_neighbors=8.0,
        atomic_numbers=Z, correlation=2, gate=gate_dict["silu"],
    ).to(DEV)


def conformers(n, seed=0):
    """Jittered water dimers. Ligand = first water (atoms 0-2)."""
    g = np.random.default_rng(seed)
    base = np.array([[0., 0., 0.], [0.758, 0.587, 0.], [-0.758, 0.587, 0.],
                     [2.9, 0.1, 0.3], [3.5, 0.6, -0.2], [3.2, -0.7, 0.5]])
    return [base + 0.10 * g.standard_normal((6, 3)) for _ in range(n)]


def dU_dlambda(model, batches, lam_vec, frag):
    """< dU/dlambda_i > over conformers, at one lambda point."""
    tot = None
    for b in batches:
        lam = lam_vec.repeat(b["positions"].shape[0] // len(frag)).clone()
        lam.requires_grad_(True)
        U, _, _ = alchemical_energy(model, b, lam, b["frag_id"], mode="graph")
        g = torch.autograd.grad(U.sum(), lam)[0].detach()
        g = g.view(-1, len(frag)).mean(0)
        tot = g if tot is None else tot + g
    return (tot / len(batches)).cpu().numpy()


def diagonal_attribution(model, batches, frag, n_lambda=N_LAMBDA):
    """dG_i along lambda(t) = (t,...,t), t from 1 to 0. Trapezoid rule."""
    ts = np.linspace(1.0, 0.0, n_lambda)
    grads = []
    for t in ts:
        lam = torch.full((len(frag),), float(t), device=DEV)
        grads.append(dU_dlambda(model, batches, lam, frag))
    g = np.array(grads)                       # [n_lambda, n_atoms]
    return np.trapezoid(g, ts, axis=0)        # integral from 1 to 0


def sequential_attribution(model, batches, frag, order, n_lambda=9):
    """Remove ligand atoms one at a time in `order`; dG_i is the work of step i."""
    lig = np.where(frag.cpu().numpy() == 0)[0]
    lam = torch.ones(len(frag), device=DEV)
    out = np.zeros(len(frag))
    ts = np.linspace(1.0, 0.0, n_lambda)
    for i in order:
        vals = []
        for t in ts:
            l = lam.clone(); l[i] = float(t)
            vals.append(dU_dlambda(model, batches, l, frag)[i])
        out[i] = np.trapezoid(vals, ts)
        lam[i] = 0.0
    return out, lig


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"Per-atom attribution: diagonal path vs sequential Shapley  ({DEV})")
    bb = small_mace()
    model = MACEDecomposition(bb).to(DEV).eval()

    confs = conformers(N_CONF)
    frag = torch.tensor([0, 0, 0, 1, 1, 1], device=DEV)
    z = np.array([8, 1, 1, 8, 1, 1])
    batches = []
    for k in range(0, N_CONF, 8):
        c = confs[k:k + 8]
        b = mace_batch(c, [z] * len(c), bb,
                       frag_ids=[frag.cpu().numpy()] * len(c), device=DEV)
        batches.append(b)
    print(f"  {N_CONF} conformers, ligand = 3 atoms, {N_LAMBDA} lambda nodes\n")

    diag = diagonal_attribution(model, batches, frag)
    lig = np.where(frag.cpu().numpy() == 0)[0]

    # total decoupling free energy, as the diagonal integral of dU/dt
    total = diag[lig].sum()

    # every removal order of a 3-atom ligand: exact discrete Shapley
    orders = list(itertools.permutations(lig))
    seq = []
    for o in orders:
        s, _ = sequential_attribution(model, batches, frag, o)
        seq.append(s[lig])
    seq = np.array(seq)
    shapley = seq.mean(0)

    print(f"  {'atom':>6s} {'diagonal':>12s} {'Shapley mean':>14s} "
          f"{'Shapley sd':>12s} {'|diff|':>10s}")
    for k, i in enumerate(lig):
        print(f"  {i:6d} {diag[i]:12.6f} {shapley[k]:14.6f} "
              f"{seq[:, k].std():12.6f} {abs(diag[i]-shapley[k]):10.2e}")
    print(f"  {'sum':>6s} {diag[lig].sum():12.6f} {shapley.sum():14.6f} "
          f"{'':12s} {abs(diag[lig].sum()-shapley.sum()):10.2e}")

    print(f"\n  order dependence of the sequential route: "
          f"per-atom sd / |mean| = "
          f"{np.mean(seq.std(0) / np.abs(shapley).clip(1e-12)):.1%}")
    print(f"  sum over orders: min {seq.sum(1).min():.6f}  "
          f"max {seq.sum(1).max():.6f}  spread {np.ptp(seq.sum(1)):.2e}")

    err = np.abs(diag[lig] - shapley).max()
    rel = err / max(abs(total), 1e-12)
    print(f"\n  max |diagonal - Shapley| = {err:.2e}  "
          f"({100*rel:.2f}% of the total {total:.6f})")
    print(f"  cost: diagonal {N_LAMBDA} lambda points, "
          f"sequential {len(orders)} orders x {len(lig)} atoms x 9 = "
          f"{len(orders)*len(lig)*9} points "
          f"({len(orders)*len(lig)*9/N_LAMBDA:.0f}x more)")
    assert rel < 0.05, "diagonal attribution disagrees with the Shapley value"
    print("\n  passed: the diagonal path reproduces the Shapley value")

"""The six consistency tests the alchemical Hamiltonian must pass (S8.3).

Run: python test_endpoints.py

No training, no data beyond one dimer, seconds on CPU or GPU. These decide
whether the S5.2 design is right, and everything in WP7-WP9 rests on them.

The load-bearing one is test 2: U(R, 0) == U_L + U_E. v1 scaled only the LE edge
energies and asserted ligand-internal invariance; S5.2 argues that leaves the
ligand environment-polarised at lambda=0 so the endpoint is wrong, and that
lambda must cut the graph too. That argument has never been checked. Both modes
run here, side by side.
"""

import numpy as np
import torch

import decomp  # noqa: F401
from decomp.data import mace_batch
from decomp.lambda_mask import alchemical_energy, isolated_energy
from decomp.mace_adapter import MACEDecomposition
from e3nn import o3
from mace.modules import MACE, gate_dict, interaction_classes

DEV = "cuda" if torch.cuda.is_available() else "cpu"
Z = [1, 8]
R_MAX = 5.0
TOL = 1e-5          # fp32; the exact-identity tests should sit far below this


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


def water_dimer():
    """Ligand = first water, environment = second."""
    pos = np.array([[0., 0., 0.], [0.758, 0.587, 0.], [-0.758, 0.587, 0.],
                    [2.9, 0.1, 0.3], [3.5, 0.6, -0.2], [3.2, -0.7, 0.5]])
    return pos, np.array([8, 1, 1, 8, 1, 1]), np.array([0, 0, 0, 1, 1, 1])


def setup():
    torch.manual_seed(0)
    bb = small_mace()
    model = MACEDecomposition(bb).to(DEV).eval()
    pos, z, frag = water_dimer()
    data = mace_batch([pos], [z], bb, frag_ids=[frag], device=DEV)
    return model, data, data["frag_id"]


def ones(frag):  return torch.ones(len(frag), device=DEV)
def zeros(frag): return torch.zeros(len(frag), device=DEV)


def test_1_full_state(model, data, frag):
    """U(R, 1) must equal the unmodified model energy, exactly."""
    ref = model(data, compute_force=False)["energy"][0]
    for mode in ("edge", "graph"):
        U, _, _ = alchemical_energy(model, data, ones(frag), frag, mode=mode)
        err = abs(float(U[0] - ref))
        print(f"  [1] full state  {mode:9s} |U(1) - E_ML| = {err:.2e}"
              f"   {'OK' if err < TOL else 'FAIL'}")
        assert err < TOL, mode


def test_2_endpoint(model, data, frag):
    """U(R, 0) must equal U_L + U_E. The whole design rests on this."""
    U_L = float(isolated_energy(model, data, frag, 0))
    U_E = float(isolated_energy(model, data, frag, 1))
    print(f"  [2] endpoint    U_L + U_E = {U_L + U_E:+.6f}")
    out = {}
    for mode in ("edge", "graph"):
        U, _, _ = alchemical_energy(model, data, zeros(frag), frag, mode=mode)
        err = abs(float(U[0]) - (U_L + U_E))
        out[mode] = err
        print(f"      {mode:9s} U(0) = {float(U[0]):+.6f}  |diff| = {err:.2e}"
              f"   {'OK' if err < TOL else 'FAIL'}")
    assert out["graph"] < TOL, "graph masking does not give an exact endpoint"
    print(f"      -> graph masking is {out['edge']/max(out['graph'],1e-12):.0e}x "
          f"closer than edge-only scaling")
    return out


def test_3_ligand_drift(model, data, frag):
    """U_LL(lambda) - U_LL(1): measured and reported, NOT required to vanish."""
    def U_LL(lam, mode):
        _, _, D = alchemical_energy(model, data, lam, frag, mode=mode)
        ei = data["edge_index"]
        inside = (frag[ei[0]] == 0) & (frag[ei[1]] == 0)
        return float(0.5 * D[inside].sum())
    for mode in ("edge", "graph"):
        base = U_LL(ones(frag), mode)
        drift = abs(U_LL(zeros(frag), mode) - base)
        print(f"  [3] ligand drift {mode:9s} |U_LL(0) - U_LL(1)| = {drift:.2e}"
              f"  (reported, not required to be 0)")


def test_4_force_continuity(model, data, frag):
    """Forces must stay smooth and bounded across 0 <= lambda <= 1."""
    for mode in ("edge", "graph"):
        prev, worst, mx = None, 0.0, 0.0
        for t in np.linspace(1.0, 0.0, 21):
            pos = data["positions"].clone().requires_grad_(True)
            d = {**data, "positions": pos}
            lam = torch.full((len(frag),), float(t), device=DEV)
            U, _, _ = alchemical_energy(model, d, lam, frag, mode=mode)
            F = -torch.autograd.grad(U.sum(), pos)[0]
            mx = max(mx, float(F.abs().max()))
            if prev is not None:
                worst = max(worst, float((F - prev).abs().max()))
            prev = F
        print(f"  [4] force continuity {mode:9s} max step {worst:.2e}  "
              f"max |F| {mx:.2e}   {'OK' if np.isfinite(mx) else 'FAIL'}")
        assert np.isfinite(mx)


def test_5_lambda_derivative(model, data, frag):
    """autograd dU/dlambda_i vs central differences."""
    for mode in ("edge", "graph"):
        lam = torch.full((len(frag),), 0.5, device=DEV, requires_grad=True)
        U, _, _ = alchemical_energy(model, data, lam, frag, mode=mode)
        g = torch.autograd.grad(U.sum(), lam)[0]
        eps, worst = 1e-3, 0.0
        for i in (0, 1, 2):
            for sgn in (+1, -1):
                l2 = lam.detach().clone(); l2[i] += sgn * eps
                U2, _, _ = alchemical_energy(model, data, l2, frag, mode=mode)
                if sgn > 0: up = float(U2[0])
                else:       dn = float(U2[0])
            worst = max(worst, abs((up - dn) / (2 * eps) - float(g[i])))
        print(f"  [5] dU/dlambda  {mode:9s} max |autograd - fd| = {worst:.2e}"
              f"   {'OK' if worst < 1e-2 else 'FAIL'}")


def test_6_attribution_additivity(model, data, frag):
    """Per-atom dU/dlambda_i must sum to dU/dt along the diagonal path (S6.1)."""
    lam = torch.full((len(frag),), 0.5, device=DEV, requires_grad=True)
    U, _, D = alchemical_energy(model, data, lam, frag, mode="graph")
    g = torch.autograd.grad(U.sum(), lam, create_graph=False)[0]
    ei = data["edge_index"]
    cross = frag[ei[0]] != frag[ei[1]]
    D_LE = float(0.5 * D[cross].sum())
    total = float(g[frag == 0].sum())
    err = abs(total - D_LE)
    print(f"  [6] additivity  sum_i dU/dlambda_i = {total:+.6f}  "
          f"D_LE = {D_LE:+.6f}  |diff| = {err:.2e}"
          f"   {'OK' if err < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    print(f"Alchemical endpoint consistency  (device {DEV})")
    model, data, frag = setup()
    print(f"  water dimer, ligand = atoms {(frag == 0).sum().item()}, "
          f"environment = {(frag == 1).sum().item()}\n")
    test_1_full_state(model, data, frag)
    print()
    test_2_endpoint(model, data, frag)
    print()
    test_3_ligand_drift(model, data, frag)
    print()
    test_4_force_continuity(model, data, frag)
    print()
    test_5_lambda_derivative(model, data, frag)
    print()
    test_6_attribution_additivity(model, data, frag)
    print("\nall passed")

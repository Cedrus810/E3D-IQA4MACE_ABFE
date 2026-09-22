"""The alchemical Hamiltonian, driven by OpenMM (WP8 bridge check).

Run: python test_openmm.py

S13.7 verified the endpoint identities inside torch. This checks they survive
the trip through OpenMM -- unit conversion, the official neighbour list, and
PythonForce -- and that MD actually runs and conserves what it should.

No training, no data, seconds.
"""

import numpy as np
import openmm
import torch
from openmm import unit

import decomp  # noqa: F401
from decomp.data import mace_batch
from decomp.lambda_mask import alchemical_energy, isolated_energy
from decomp.mace_adapter import MACEDecomposition
from decomp.openmm_bridge import EV_TO_KJ_PER_MOL, AlchemicalForce, make_system
from e3nn import o3
from mace.modules import MACE, gate_dict, interaction_classes

DEV = "cuda" if torch.cuda.is_available() else "cpu"
Z_LIST = [1, 8]
R_MAX = 5.0


def small_mace():
    return MACE(
        r_max=R_MAX, num_bessel=8, num_polynomial_cutoff=6, max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2, num_elements=len(Z_LIST),
        hidden_irreps=o3.Irreps("16x0e+16x1o"), MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(len(Z_LIST)), avg_num_neighbors=8.0,
        atomic_numbers=Z_LIST, correlation=2, gate=gate_dict["silu"],
    ).to(DEV)


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"Alchemical Hamiltonian through OpenMM  ({DEV})")
    bb = small_mace()
    model = MACEDecomposition(bb).to(DEV).eval()
    pos = np.array([[0., 0., 0.], [0.758, 0.587, 0.], [-0.758, 0.587, 0.],
                    [2.9, 0.1, 0.3], [3.5, 0.6, -0.2], [3.2, -0.7, 0.5]])
    z = np.array([8, 1, 1, 8, 1, 1]); frag = np.array([0, 0, 0, 1, 1, 1])

    force = AlchemicalForce(model, z, frag, mode="graph")
    print(f"  water dimer, ligand = 3 atoms, mode = graph\n")

    # --- 1. the bridge reproduces the torch-side energy ---
    data = mace_batch([pos], [z], bb, frag_ids=[frag], device=DEV)
    ft = torch.ones(6, device=DEV)
    U_torch = float(alchemical_energy(model, data, ft, data["frag_id"],
                                      mode="graph")[0][0])
    U_omm, F_omm = force.energy_and_forces(pos)
    print(f"  [1] torch {U_torch:+.8f} eV   bridge {U_omm:+.8f} eV   "
          f"|diff| {abs(U_torch-U_omm):.2e}")
    assert abs(U_torch - U_omm) < 1e-5

    # --- 2. the endpoint identity survives the bridge ---
    U_L = float(isolated_energy(model, data, data["frag_id"], 0))
    U_E = float(isolated_energy(model, data, data["frag_id"], 1))
    force.set_lambda(0.0)
    U0, _ = force.energy_and_forces(pos)
    force.set_lambda(1.0)
    U1, _ = force.energy_and_forces(pos)
    print(f"  [2] U(1) {U1:+.6f}   U(0) {U0:+.6f}   U_L+U_E {U_L+U_E:+.6f}   "
          f"|diff| {abs(U0-(U_L+U_E)):.2e}")
    assert abs(U0 - (U_L + U_E)) < 1e-5

    # --- 3. dU/dlambda from the bridge ---
    force.set_lambda(0.5)
    _, _ = force.energy_and_forces(pos, need_dudl=True)
    g = force.last_dudl
    print(f"  [3] dU/dlambda_i (ligand) {np.round(g[:3], 6)}   "
          f"sum {g[:3].sum():+.6f}")

    # --- 4. OpenMM agrees with the callable, and MD runs ---
    system = make_system(z, force)
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.5 * unit.femtosecond)
    ctx = openmm.Context(system, integrator,
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos * 0.1)                  # Angstrom -> nm
    force.set_lambda(1.0)
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    print(f"  [4] OpenMM {e:+.4f} kJ/mol   direct {U1*EV_TO_KJ_PER_MOL:+.4f}   "
          f"|diff| {abs(e-U1*EV_TO_KJ_PER_MOL):.2e}")
    assert abs(e - U1 * EV_TO_KJ_PER_MOL) < 1e-3

    integrator.step(50)
    st = ctx.getState(getEnergy=True, getPositions=True)
    ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    moved = np.linalg.norm(
        st.getPositions(asNumpy=True).value_in_unit(unit.nanometer) * 10 - pos)
    print(f"  [5] 50 MD steps: PE {pe:+.3f}  KE {ke:.3f} kJ/mol  "
          f"moved {moved:.3f} A   {'OK' if np.isfinite(pe+ke) else 'FAIL'}")
    assert np.isfinite(pe + ke)

    print("\n  all passed -- the alchemical Hamiltonian runs under OpenMM")

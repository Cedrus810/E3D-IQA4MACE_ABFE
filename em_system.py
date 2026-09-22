"""Minimise a prepared ABFE leg with OpenMM and its own MM force field.

Run: python em_system.py /home/ruigengji/ABFE_IBS/Atenolol-rank11/output

The MM force field has hard repulsive cores and OpenMM's LocalEnergyMinimizer is
well tested; minimising the same structure with a learned potential is not the
same thing at all -- a cluster carved from a periodic box sits outside the
training domain at its surface, and an MLIP has no wall to stop atoms collapsing
into each other. So: minimise here, evaluate the decomposition afterwards.
"""

import sys
import time

import numpy as np
import openmm
import openmm.app as app
from openmm import unit


def main(out_dir, leg="solvent", tol=10.0, max_iter=0, out=None):
    suffix = "" if leg == "complex" else f"_{leg}"
    system = openmm.XmlSerializer.deserialize(
        open(f"{out_dir}/system{suffix}.xml").read())
    st = app.PDBxFile(f"{out_dir}/topology{suffix}.cif")
    top, pos = st.topology, st.positions

    integrator = openmm.LangevinMiddleIntegrator(
        298.15 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    ctx = openmm.Context(system, integrator)
    ctx.setPositions(pos)

    e0 = ctx.getState(getEnergy=True).getPotentialEnergy()
    f0 = np.linalg.norm(ctx.getState(getForces=True).getForces(asNumpy=True)
                        .value_in_unit(unit.kilojoule_per_mole/unit.nanometer),
                        axis=1).max()
    print(f"{leg}: {system.getNumParticles()} particles")
    print(f"  before  U {e0.value_in_unit(unit.kilojoule_per_mole):14.2f} kJ/mol"
          f"   max|F| {f0:12.1f} kJ/mol/nm", flush=True)

    t0 = time.perf_counter()
    openmm.LocalEnergyMinimizer.minimize(
        ctx, tolerance=tol * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=max_iter)
    e1 = ctx.getState(getEnergy=True).getPotentialEnergy()
    f1 = np.linalg.norm(ctx.getState(getForces=True).getForces(asNumpy=True)
                        .value_in_unit(unit.kilojoule_per_mole/unit.nanometer),
                        axis=1).max()
    print(f"  after   U {e1.value_in_unit(unit.kilojoule_per_mole):14.2f} kJ/mol"
          f"   max|F| {f1:12.1f} kJ/mol/nm   [{time.perf_counter()-t0:.0f}s]")

    pos_ang = ctx.getState(getPositions=True).getPositions(asNumpy=True)\
        .value_in_unit(unit.angstrom)
    numbers = np.array([a.element.atomic_number for a in top.atoms()])
    resnames = np.array([a.residue.name for a in top.atoms()])
    resids = np.array([a.residue.index for a in top.atoms()])
    bv = system.getDefaultPeriodicBoxVectors()
    box = np.array([bv[i][i].value_in_unit(unit.angstrom) for i in range(3)])

    out = out or f"em_{leg}.npz"
    np.savez(out, positions=pos_ang, numbers=numbers, resnames=resnames,
             resids=resids, box=box,
             source=f"{out_dir}/system{suffix}.xml")
    print(f"  -> {out}")
    return out


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/ruigengji/ABFE_IBS/Atenolol-rank11/output"
    main(d, leg=sys.argv[2] if len(sys.argv) > 2 else "solvent")

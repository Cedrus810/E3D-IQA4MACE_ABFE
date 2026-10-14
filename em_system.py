"""Minimise a prepared ABFE leg with OpenMM and its own MM force field.

Run: python em_system.py ~/ABFE_IBS/Atenolol-rank11/output
     python em_system.py frames      # MD frames for clusters, see frames()

The MM force field has hard repulsive cores and OpenMM's LocalEnergyMinimizer is
well tested; minimising the same structure with a learned potential is not the
same thing at all -- a cluster carved from a periodic box sits outside the
training domain at its surface, and an MLIP has no wall to stop atoms collapsing
into each other. So: minimise here, evaluate the decomposition afterwards.
"""

import os
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


def frames(start="em_solvent.npz", n_frames=20, skip_ps=100, every_ps=50,
           heldout_ps=2000, seed=11, out="frames_solvent.npz",
           heldout_out="md2000_solvent.npz", tol=10.0):
    """Geometries for decomp/clusters.py: MM MD of a minimised leg, each frame
    MM-minimised, exactly the recipe of md50_solvent.npz (seed 7, 50 ps).

    Clusters from one frame leave the held-out pair sum 0.4-0.8 eV off; 20
    frames fix it (S13.20). Frames at skip_ps, skip_ps+every_ps,
    ... go to `out`, stacked [n_frames, n_atoms, 3]; one more at heldout_ps goes
    to `heldout_out` in the single-frame format, never for training. A fresh
    seed keeps md50 held out too. NVT, so one box for all frames.
    OPENMM_DEFAULT_PLATFORM=CPU keeps it off a shared GPU.
    """
    d = np.load(start, allow_pickle=True)
    system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    integ = openmm.LangevinMiddleIntegrator(
        298.15 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ)
    b = d["box"] / 10
    ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * b[i])) for i in range(3)])
    ctx.setPositions(d["positions"] * unit.angstrom)
    ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
    print(f"frames: {ctx.getPlatform().getName()}, seed {seed}", flush=True)
    lig = d["resids"] == d["resids"][0]

    def snap(t_ps):
        # minimise a copy of the state, then put the trajectory back untouched
        st = ctx.getState(getPositions=True, getVelocities=True)
        openmm.LocalEnergyMinimizer.minimize(
            ctx, tolerance=tol * unit.kilojoule_per_mole / unit.nanometer)
        pos = ctx.getState(getPositions=True).getPositions(asNumpy=True)\
            .value_in_unit(unit.angstrom)
        ctx.setState(st)
        disp = np.sqrt(((pos[lig] - d["positions"][lig]) ** 2).sum(1).mean())
        print(f"  {t_ps:6.0f} ps  ligand RMSD vs start {disp:5.2f} A"
              f"  [{time.perf_counter()-t0:.0f}s]", flush=True)
        return pos

    t0 = time.perf_counter()
    integ.step(skip_ps * 500)                         # 2 fs steps
    stack = [snap(skip_ps)]
    for k in range(1, n_frames):
        integ.step(every_ps * 500)
        stack.append(snap(skip_ps + k * every_ps))
    t_last = skip_ps + (n_frames - 1) * every_ps
    integ.step((heldout_ps - t_last) * 500)
    held = snap(heldout_ps)

    meta = dict(numbers=d["numbers"], resnames=d["resnames"], resids=d["resids"],
                box=d["box"])
    np.savez(out, positions=np.stack(stack), source=str(d["source"]),
             note=f"{start} + MM MD seed {seed}, frames {skip_ps}..{t_last} ps "
                  f"every {every_ps}, each minimised", **meta)
    np.savez(heldout_out, positions=held,
             source=f"{start} + {heldout_ps} ps MM MD + minimise, seed {seed}",
             **meta)
    print(f"  -> {out} ({n_frames} frames), {heldout_out}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["frames"]:
        frames()
        sys.exit()
    d = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/ABFE_IBS/Atenolol-rank11/output")
    main(d, leg=sys.argv[2] if len(sys.argv) > 2 else "solvent")

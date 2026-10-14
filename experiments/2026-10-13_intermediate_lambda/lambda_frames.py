"""Unminimised MM frames of the Atenolol solvent leg at an intermediate alchemical state.

Run: python lambda_frames.py LAMBDA_ELEC SEED OUT.npz [every_ps] [n_frames] [platform]

A proxy for the intermediate-lambda ensemble of the anchored Hamiltonian
(v3 S13.35: the head fits the lambda = 1 thermal ensemble; nothing says it holds
where the ligand is partly decoupled). The anchored ML ensemble itself needs
ML/MM + FIRES + the wall, none written yet (S8.1). Here: ligand-water
electrostatics scaled by LAMBDA_ELEC (scaled_system), LJ kept in full (its
repulsion stands in for the wall), ligand intramolecular terms untouched
(decoupling, not annihilation). Checked (check_scaled.log): at 1 it matches
the reference to -0.034 +- 0.014 kJ/mol, and E is linear in LAMBDA_ELEC to 0.08 kJ/mol.

Same MD as thermal_frames.py (em_solvent.npz, Langevin 298.15 K, 1/ps, 2 fs,
frames from 100 ps), and the same per-0.5 ps ligand-water contact minima.
"""

import sys
import time

import numpy as np
import openmm
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D/experiments/2026-10-12_thermal_frames")
from thermal_frames import LIG_EL, SYM, W_EL, contact_minima  # noqa: E402

E3D = "/home/ruigengji/E3D"


def scaled_system(ref, lig_idx, lam):
    """Ligand-environment electrostatics x lam, everything else as in `ref`.

    Ligand charges are scaled in the NonbondedForce, so the ligand-water
    Coulomb (direct and reciprocal) goes as lam. Every ligand-ligand pair gets
    an exception with its original charge product (existing exceptions keep
    theirs), so the ligand's own electrostatics is lam-independent: PME's
    exclusion correction removes the scaled reciprocal pair term and the
    exception adds the unscaled Coulomb back. LJ is untouched. At lam = 1 this
    differs from `ref` only by the ligand's self-images (box 35 A).
    openmmtools' "direct-space" decoupling was tried first and rejected: at
    lam = 1 it is 33 +- 13 kJ/mol off the reference over 20 thermal frames,
    with ligand force errors up to 210 kJ/mol/nm (check_l1.log).
    """
    system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(ref))
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    assert len(nb) == 1, "expected one NonbondedForce"
    nb = nb[0]
    q, sig, eps = {}, {}, {}
    for i in lig_idx:
        q[i], sig[i], eps[i] = nb.getParticleParameters(int(i))
    have = set()
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        have.add((min(i, j), max(i, j)))
    lig_idx = [int(i) for i in lig_idx]
    for a, i in enumerate(lig_idx):
        for j in lig_idx[a + 1:]:
            if (i, j) not in have:
                nb.addException(i, j, q[i] * q[j], 0.5 * (sig[i] + sig[j]),
                                (eps[i] * eps[j]).sqrt())
    for i in lig_idx:
        nb.setParticleParameters(i, q[i] * lam, sig[i], eps[i])
    return system


def main(lam_elec, seed, out, every_ps=50.0, n_frames=20, platform="CUDA",
         t0_ps=100, dense_ps=0.5):
    d = np.load(f"{E3D}/em_solvent.npz", allow_pickle=True)
    ref = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    num, box = d["numbers"], d["box"]
    lig = d["resids"] == d["resids"][0]
    wat = d["resnames"] == "HOH"
    system = scaled_system(ref, np.where(lig)[0], lam_elec)

    integ = openmm.LangevinMiddleIntegrator(
        298.15 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName(platform))
    b = box / 10
    ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * b[i])) for i in range(3)])
    ctx.setPositions(d["positions"] * unit.angstrom)
    ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
    dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3

    print(f"lambda_elec {lam_elec}, seed {seed}, {ctx.getPlatform().getName()}, "
          f"{len(num)} atoms, ligand {lig.sum()}", flush=True)

    def pos():
        return ctx.getState(getPositions=True).getPositions(asNumpy=True)\
            .value_in_unit(unit.angstrom)

    t = time.perf_counter()
    integ.step(t0_ps * 500)
    print(f"  equilibrated {t0_ps} ps [{time.perf_counter()-t:.0f}s]", flush=True)
    per = int(round(dense_ps * 500))
    k_frame = int(round(every_ps / dense_ps))
    n_dense = (n_frames - 1) * k_frame + 1
    frames, contacts = [], []
    for k in range(n_dense):
        if k:
            integ.step(per)
        p = pos()
        contacts.append(contact_minima(p, lig, wat, num, box))
        if k % k_frame == 0:
            frames.append(p)
            ke = ctx.getState(getEnergy=True).getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            if len(frames) % 10 == 1:
                print(f"  {t0_ps + k * dense_ps:7.1f} ps  frame {len(frames):3d}  "
                      f"T {2 * ke / (dof * 0.0083144626):.0f} K  [{time.perf_counter()-t:.0f}s]",
                      flush=True)

    t_last = t0_ps + (n_frames - 1) * every_ps
    np.savez(out, positions=np.stack(frames), numbers=num, resnames=d["resnames"],
             resids=d["resids"], box=box, source=str(d["source"]),
             note=f"em_solvent.npz + MM MD at lambda_elec {lam_elec} (LJ full, openmmtools "
                  f"decouple), seed {seed}, frames {t0_ps}..{t_last} ps every {every_ps}, "
                  f"NOT minimised")
    pairs = [f"{SYM[a]}-{SYM[b]}w" for a in LIG_EL for b in W_EL]
    np.savez(out.replace(".npz", "_contacts.npz"), contacts=np.array(contacts),
             pairs=np.array(pairs), dense_ps=dense_ps, t0_ps=t0_ps)
    print(f"  -> {out} ({len(frames)} frames), {len(contacts)} contact snapshots"
          f"  [{time.perf_counter()-t:.0f}s]", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 3:
        sys.exit(__doc__)
    main(float(a[0]), int(a[1]), a[2],
         every_ps=float(a[3]) if len(a) > 3 else 50.0,
         n_frames=int(a[4]) if len(a) > 4 else 20,
         platform=a[5] if len(a) > 5 else "CUDA")

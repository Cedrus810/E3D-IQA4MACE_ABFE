"""Unminimised MM frames of the Atenolol solvent leg, and their ligand-water contacts.

Run: python thermal_frames.py SEED OUT.npz [platform] [every_ps] [n_frames]

Same MD as em_system.frames() (em_solvent.npz start, Langevin 298.15 K, 1/ps,
2 fs, frames at 100, 150, ..., 1050 ps) but the frames are NOT minimised:
v3 S13.31 found every cluster the head trained on comes from minimised frames,
so thermal contacts were never seen. Seed 23 is the trajectory of
frames_test_solvent.npz, seed 11 that of frames_solvent.npz (same seeds,
not the same trajectories bit for bit: the platform may differ).

Besides the 20 frames, every 0.5 ps from 100 to 1050 ps it records, per
(ligand element, water element), the shortest ligand-water distance
(minimum image) -- the 298 K contact distribution of the fully coupled
system, i.e. the upper bound for the D15 wall radius.
"""

import sys
import time

import numpy as np
import openmm
from openmm import unit

E3D = "/home/ruigengji/E3D"
LIG_EL, W_EL = (6, 7, 8, 1), (8, 1)
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}


def contact_minima(pos, lig, wat, num, box):
    d = pos[lig][:, None, :] - pos[wat][None, :, :]
    d -= box * np.round(d / box)
    r = np.sqrt((d ** 2).sum(-1))                       # [n_lig, n_wat]
    zl, zw = num[lig], num[wat]
    return [r[np.ix_(zl == a, zw == b)].min() for a in LIG_EL for b in W_EL]


def main(seed, out, platform="CPU", t0_ps=100, every_ps=50, n_frames=20, dense_ps=0.5):
    d = np.load(f"{E3D}/em_solvent.npz", allow_pickle=True)
    system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    integ = openmm.LangevinMiddleIntegrator(
        298.15 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName(platform))
    b = d["box"] / 10
    ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * b[i])) for i in range(3)])
    ctx.setPositions(d["positions"] * unit.angstrom)
    ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
    num, box = d["numbers"], d["box"]
    dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    lig = d["resids"] == d["resids"][0]
    wat = d["resnames"] == "HOH"
    print(f"seed {seed}, {ctx.getPlatform().getName()}, {len(num)} atoms, ligand {lig.sum()}",
          flush=True)

    def pos():
        return ctx.getState(getPositions=True).getPositions(asNumpy=True)\
            .value_in_unit(unit.angstrom)

    t = time.perf_counter()
    integ.step(t0_ps * 500)
    print(f"  equilibrated {t0_ps} ps [{time.perf_counter()-t:.0f}s]", flush=True)
    per = int(round(dense_ps * 500))
    k_frame = int(round(every_ps / dense_ps))
    n_dense = (n_frames - 1) * k_frame + 1
    frames, contacts, temps = [], [], []
    for k in range(n_dense):
        if k:
            integ.step(per)
        p = pos()
        contacts.append(contact_minima(p, lig, wat, num, box))
        if k % k_frame == 0:
            frames.append(p)
            st = ctx.getState(getEnergy=True)
            temps.append(2 * st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                         / (dof * 0.0083144626))
            print(f"  {t0_ps + k * dense_ps:7.1f} ps  frame {len(frames):2d}  "
                  f"T~{temps[-1]:.0f} K  [{time.perf_counter()-t:.0f}s]", flush=True)

    t_last = t0_ps + (n_frames - 1) * every_ps
    np.savez(out, positions=np.stack(frames), numbers=num, resnames=d["resnames"],
             resids=d["resids"], box=box, source=str(d["source"]),
             note=f"em_solvent.npz + MM MD seed {seed} ({ctx.getPlatform().getName()}), "
                  f"frames {t0_ps}..{t_last} ps every {every_ps}, NOT minimised")
    pairs = [f"{SYM[a]}-{SYM[b]}w" for a in LIG_EL for b in W_EL]
    np.savez(out.replace(".npz", "_contacts.npz"), contacts=np.array(contacts),
             pairs=np.array(pairs), dense_ps=dense_ps, t0_ps=t0_ps)
    print(f"  -> {out} ({len(frames)} frames), {len(contacts)} contact snapshots"
          f"  [{time.perf_counter()-t:.0f}s]")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    main(int(a[0]), a[1], a[2] if len(a) > 2 else "CPU",
         every_ps=float(a[3]) if len(a) > 3 else 50, n_frames=int(a[4]) if len(a) > 4 else 20)

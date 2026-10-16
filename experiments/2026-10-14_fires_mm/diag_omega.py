"""Is the e0_fires NaN a step-size instability of the FIRES restraint?

Run: python diag_omega.py METRIC [ps] [dt_fs] [seed]
     METRIC = all   (d over every water atom x every ligand atom: D16 as coded)
            | heavy (water O x ligand heavy atoms)

Hypothesis: the softmin closest-atom distance puts the restraint on H atoms
(1 amu), several violating pairs stack on one atom, and the stiffness
k_eff / m exceeds what the step can integrate. For velocity-Verlet-type
integrators the stability limit of a harmonic mode is omega dt < 2.

Every 50 steps, when the order is violated, it builds the exact Hessian of
U_F for every atom that feels a FIRES force (autograd, 3 x 3 diagonal blocks),
takes omega_a = sqrt(lambda_max(H_aa) / m_a) and reports max_a omega_a dt.
(Diagonal blocks give a lower bound on the largest mode; enough to show the
bound is crossed.) Same start (em_solvent.npz), Langevin 298.15 K, 1/ps.
Stops at the first NaN.
"""

import sys
import time

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp.fires import Fires, water_indices  # noqa: E402

E3D = "/home/ruigengji/E3D"
a = sys.argv[1:]
metric = a[0]
ps = float(a[1]) if len(a) > 1 else 100.0
dt = float(a[2]) if len(a) > 2 else 2.0
seed = int(a[3]) if len(a) > 3 else 7
chunk = 50

d = np.load(f"{E3D}/em_solvent.npz", allow_pickle=True)
system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
box = np.asarray(d["box"], dtype=np.float64)
num = d["numbers"]
lig = np.where(d["resids"] == d["resids"][0])[0]
W = water_indices(d["resnames"], d["resids"])
if metric == "all":
    fires = Fires(lig, W, n_inner=20, device="cuda")
elif metric == "heavy":
    fires = Fires(lig[num[lig] > 1], W[:, :1], n_inner=20, device="cuda")
else:
    sys.exit(__doc__)
inner = fires.select_inner(d["positions"], box)
ref = Fires(lig, W, n_inner=20).select_inner(d["positions"], box)
print(f"metric {metric}, dt {dt} fs, seed {seed}; inner set equals the all-atom one: "
      f"{set(inner) == set(ref)} ({len(set(inner) & set(ref))}/20 shared)", flush=True)
mass = torch.tensor([system.getParticleMass(i).value_in_unit(unit.dalton)
                     for i in range(system.getNumParticles())], dtype=torch.float64)
f = openmm.PythonForce(fires)
f.setUsesPeriodicBoundaryConditions(True)
system.addForce(f)
integ = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1.0 / unit.picosecond,
                                        dt * unit.femtosecond)
integ.setRandomNumberSeed(seed)
ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
ctx.setPositions(d["positions"] * unit.angstrom)
ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
KJ_A2_AMU_TO_PS2 = 100.0     # kJ/mol/A^2 / amu = 100 ps^-2


def omega_dt(p):
    """max_a sqrt(lambda_max(H_aa)/m_a) * dt over atoms with a FIRES force."""
    pos = fires._t(p).requires_grad_(True)
    U = fires.energy(pos, fires._t(box))
    if float(U.detach()) == 0.0:
        return 0.0, None
    (g,) = torch.autograd.grad(U, pos, create_graph=True)
    atoms = torch.nonzero(g.detach().abs().sum(1) > 1e-8).flatten()
    best, who = 0.0, None
    for at in atoms.tolist():
        H = torch.stack([torch.autograd.grad(g[at, c], pos, retain_graph=True)[0][at]
                         for c in range(3)])
        lam = torch.linalg.eigvalsh(0.5 * (H + H.T)).max().item()
        if lam > 0:
            w = np.sqrt(lam * KJ_A2_AMU_TO_PS2 / mass[at].item())       # 1/ps
            if w * dt / 1000 > best:
                best, who = w * dt / 1000, at
    return best, who


n_chunks = int(round(ps * 1000 / dt / chunk))
t, rec = time.perf_counter(), []
for c in range(n_chunks):
    try:
        integ.step(chunk)
        p = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        if not np.isfinite(p).all():
            raise ValueError("NaN positions")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED in chunk {c} ({(c + 1) * chunk * dt / 1000:.2f} ps): {e}", flush=True)
        break
    wdt, who = omega_dt(np.asarray(p))
    L = fires.last
    rec.append((wdt, L["n_viol"], L["max_depth"]))
    if wdt > 1.0 or c % 100 == 0:
        print(f"  {(c + 1) * chunk * dt / 1000:7.2f} ps  viol {L['n_viol']:3d}  depth {L['max_depth']:.3f}  "
              f"omega dt {wdt:.2f}" + (f" on Z={num[who]}" if who is not None else ""), flush=True)
else:
    print(f"no failure in {ps} ps", flush=True)
r = np.array(rec)
v = r[:, 1] > 0
print(f"{metric}: {len(r)} snapshots, violated in {100 * v.mean():.1f}%; omega dt when violated: "
      f"median {np.median(r[v, 0]) if v.any() else 0:.2f}, p99 {np.percentile(r[v, 0], 99) if v.any() else 0:.2f}, "
      f"max {r[:, 0].max():.2f}; snapshots with omega dt > 2: {(r[:, 0] > 2).sum()}; "
      f"max depth {r[:, 2].max():.3f} A [{time.perf_counter() - t:.0f}s]", flush=True)

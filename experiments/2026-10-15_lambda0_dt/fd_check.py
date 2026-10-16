"""Forces against the energy gradient on the full ML/MM system (v3 S8 route step 1).

Run: python fd_check.py PREFIX_frames.npz [n_atoms] [h1,h2]

Builds the same system as paired_dt.py, but in float64 throughout: the
model in float64 on the CPU, OpenMM CUDA at double precision. On each saved
frame (largest cap excess, most switch-zone pairs, closest contact), at that
frame's q, it compares the analytic force with -dU/dx by central
differences, for the ligand-water pairs closest relative to the wall radius
(smallest r / r0, i.e. where the wall and the cover switch act; n_atoms
atoms in all), x/y/z each. Two step sizes h are run, so the FD
truncation error shows up as the change between them. Reported for the total
U and for the ML + FIRES group (G_ML) and the wall group (G_WALL) alone.
"""

import sys
import time

import numpy as np
import openmm
import torch
from openmm import unit

sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm  # noqa: E402
from run_abfe import load_model  # noqa: E402

E3D = "/home/ruigengji/E3D"
LEG = f"{E3D}/em_solvent.npz"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"
CKPT = f"{E3D}/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
START = f"{E3D}/experiments/2026-10-12_thermal_frames/thermal200_solvent.npz"
KJ = unit.kilojoule_per_mole
torch.set_num_threads(8)

a = sys.argv[1:]
fr = np.load(a[0])
n_atoms = int(a[1]) if len(a) > 1 else 8
hs = [float(x) for x in a[2].split(",")] if len(a) > 2 else [1e-3, 5e-4]
q = float(fr["q"])

d = np.load(LEG, allow_pickle=True)
box = np.asarray(d["box"], dtype=np.float64)
num = np.asarray(d["numbers"])
pos0 = np.asarray(np.load(START, allow_pickle=True)["positions"][0], dtype=np.float64)
model = load_model(CKPT, BB, device="cpu").to(torch.float64)
system, mlf, info = mlmm.build_mlmm_system(LEG, model, n_inner=20, es=True, device="cpu", positions=pos0)
lig = info["lig"]
watm = np.where(d["resnames"] == "HOH")[0]
integ = openmm.VerletIntegrator(1.0 * unit.femtosecond)
ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"), {"Precision": "double"})
ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)])
mlmm.set_path(ctx, mlf, q, lig)
GROUPS = {"total": None, "ML+FIRES": {mlmm.G_ML}, "wall": {mlmm.G_WALL}}


def energy(P, groups):
    ctx.setPositions(P * unit.angstrom)
    kw = {} if groups is None else {"groups": groups}
    return ctx.getState(getEnergy=True, **kw).getPotentialEnergy().value_in_unit(KJ)


def force(P, groups):
    ctx.setPositions(P * unit.angstrom)
    kw = {} if groups is None else {"groups": groups}
    return ctx.getState(getForces=True, **kw).getForces(asNumpy=True).value_in_unit(KJ / unit.nanometer) / 10


print(f"q {q}; float64 (model CPU, OpenMM CUDA double); h {hs} A", flush=True)
for name in ("max_excess", "max_zone", "min_contact"):
    P = np.asarray(fr[name], dtype=np.float64)
    dv = P[lig][:, None] - P[watm][None]
    dv -= box * np.round(dv / box)
    D = np.sqrt((dv ** 2).sum(-1))
    r0 = np.array([[mlmm.WALL_R0[int(z)][0 if int(w) == 8 else 1] for w in num[watm]] for z in num[lig]])
    order = np.dstack(np.unravel_index(np.argsort(D / r0, axis=None), D.shape))[0]
    atoms = []
    for i, j in order:
        for x in (int(lig[i]), int(watm[j])):
            if x not in atoms and len(atoms) < n_atoms:
                atoms.append(x)
        if len(atoms) >= n_atoms:
            break
    t = time.perf_counter()
    print(f"[{name}] t = {float(fr[name + '_t_ps']):.3f} ps; atoms {atoms}; closest L-w r/r0 {(D / r0).min():.3f}", flush=True)
    for g, grp in GROUPS.items():
        F = force(P, grp)
        line = []
        for h in hs:
            err = []
            for x in atoms:
                for k in range(3):
                    Pp, Pm = P.copy(), P.copy()
                    Pp[x, k] += h; Pm[x, k] -= h
                    fd = -(energy(Pp, grp) - energy(Pm, grp)) / (2 * h)
                    err.append(abs(fd - F[x, k]))
            line.append(f"h {h:g}: max |F - FD| {max(err):.2e}")
        print(f"   {g:9s} max |F| {np.abs(F[atoms]).max():9.2f} kJ/mol/A;  " + ";  ".join(line), flush=True)
    print(f"   [{time.perf_counter() - t:.0f}s]", flush=True)

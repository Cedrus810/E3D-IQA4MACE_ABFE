"""Which force group gives NaN when a water atom sits on a ligand atom at the decoupled end?

Run: python overlap_nan.py     (acetamide leg, mm_only, no FIRES; q 4 and 3.99)
A water O (and, separately, a water H) is moved rigidly so that it lies at
distance d from ligand atom 0 (C), d = 1e-2 ... 1e-5 nm; per-group energy and
max |force| on CUDA mixed (the production precision) and double.
"""
import sys
import numpy as np, openmm
from openmm import unit
sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm
LEG = "/home/ruigengji/E3D/experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
d = np.load(LEG, allow_pickle=True); P0 = np.asarray(d["positions"]) / 10; box = np.asarray(d["box"]) / 10
system, mlf, info = mlmm.build_mlmm_system(LEG, None, n_inner=20, device="cpu", positions=P0 * 10,
                                           mm_only=True, use_fires=False)
lig, W = info["lig"], info["water"]
names = {0: "bonded", 1: "NB", 2: "L-env LJ", 3: "wall"}
for prec in ("mixed", "double"):
    c = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("CUDA"),
                       {"Precision": prec})
    c.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[i] * box[i])) for i in range(3)])
    for q in (4.0, 3.99):
        mlmm.set_path(c, mlf, q, lig)
        for which, a in (("O", 0), ("H", 1)):
            for dist in (1e-2, 1e-3, 1e-4, 1e-5):
                P = P0.copy()
                w = W[50]
                shift = P[lig[0]] + np.array([dist, 0, 0]) - P[w[a]]
                P[w] += shift
                c.setPositions(P)
                out = []
                for g, nm in names.items():
                    st = c.getState(getEnergy=True, getForces=True, groups={g})
                    e = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    f = np.abs(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)).max()
                    out.append(f"{nm} E {e:.4g} |F| {f:.3g}")
                print(f"{prec:6s} q {q:4g} water {which} at {dist:g} nm from ligand C: " + "; ".join(out), flush=True)

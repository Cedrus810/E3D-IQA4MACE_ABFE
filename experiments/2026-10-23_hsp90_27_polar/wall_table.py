"""Print the wall r0 table (A) of the HSP90-27 leg, incl. the ion rows (user, 2026-10-23). No dynamics."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, openmm
from decomp import mlmm
leg = ROOT / "experiments/2026-10-17_ibs_legs/HSP90_ligand-27.npz"
system, _, info = mlmm.build_mlmm_system(str(leg), None, mm_only=True, device="cpu")
wall = [f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce) and f.getForceGroup() == mlmm.G_WALL][0]
nt, _, vals = wall.getTabulatedFunction(0).getFunctionParameters()
tab = np.asarray(vals).reshape(nt, nt, order="F") * 10
names = ["H", "C", "N", "O", "Ow", "Hw"] + [f"ion{k}" for k in range(nt - 6)]
types = {int(wall.getParticleParameters(i)[0]) for i in range(system.getNumParticles())}
d = np.load(leg, allow_pickle=True)
for k in range(6, nt):
    i = next(i for i in range(system.getNumParticles()) if int(wall.getParticleParameters(i)[0]) == k)
    names[k] = str(d["resnames"][i])
print("cutoff", wall.getCutoffDistance(), "groups", wall.getNumInteractionGroups())
for a in range(4):
    print(names[a], " ".join(f"{names[b]} {tab[a, b]:.2f}" for b in range(4, nt)))

"""End checks of the converted IBS legs (OpenMM CUDA double).

  [mm]   build_mlmm_system(mm_only=True, use_fires=False) at q 0 vs the IBS System at the same frame
  [ions] ML/MM build: ions in the L-env LJ group, and that group = all non-ligand, non-inner atoms
  [ml]   one ML + FIRES evaluation (operating head, CPU float64) is finite; FIRES 0 at the start frame
"""
import glob, os, sys
import numpy as np, openmm, torch
from openmm import unit
sys.path.insert(0, "/home/ruigengji/E3D")
from decomp import mlmm
from run_abfe import load_model
HERE = os.path.dirname(os.path.abspath(__file__))
KJ = unit.kilojoule_per_mole
plat = openmm.Platform.getPlatformByName("CUDA")
torch.set_num_threads(8)
model = load_model("/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt",
                   "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model", device="cpu").to(torch.float64)
for leg in sorted(glob.glob(f"{HERE}/*.npz")):
    d = np.load(leg, allow_pickle=True); P = np.asarray(d["positions"]); box = np.asarray(d["box"])
    vec = [openmm.Vec3(*(np.eye(3)[i] * box[i] / 10)) for i in range(3)]
    def E(system, q=None, mlf=None, lig=None):
        c = openmm.Context(system, openmm.VerletIntegrator(0.001), plat, {"Precision": "double"})
        c.setPeriodicBoxVectors(*vec)
        if q is not None:
            mlmm.set_path(c, mlf, q, lig)
        c.setPositions(P * unit.angstrom)
        return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    ref = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    s1, f1, i1 = mlmm.build_mlmm_system(leg, None, device="cpu", positions=P, mm_only=True, use_fires=False)
    e_ref, e_mm = E(ref), E(s1, 0.0, f1, i1["lig"])
    s2, f2, i2 = mlmm.build_mlmm_system(leg, model, device="cpu", positions=P)
    ions = np.where(np.isin(d["resnames"], ["NA", "CL"]))[0]
    env = set(i2["lj_env"].tolist())
    expect = set(range(len(P))) - set(i2["lig"].tolist()) - set(i2["inner_atoms"].tolist())
    U, F = f2.evaluate(P, box)
    print(f"{os.path.basename(leg)[:-4]:22s} [mm] q0 - IBS System {e_mm - e_ref:+.4f} kJ/mol (|E| {abs(e_ref):.0f}); "
          f"[ions] {len(ions)} ions, all in LJ group {set(ions.tolist()) <= env}, group exact {env == expect}; "
          f"[ml] U finite {np.isfinite(U) and np.isfinite(F).all()}, FIRES {f2.last['U_fires']:.4f}", flush=True)

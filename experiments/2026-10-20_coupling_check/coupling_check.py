"""Ligand - inner-shell coupling at lambda = 1: POLAR (truth, capped sum c) vs GAFF2/TIP3P (v3 S13.56).

D19 on acetamide gives W2 +84 against the all-MM path's +58; its integrand at lambda = 1 is 199.6
against 116.5 kJ/mol. This asks whether that is the Hamiltonian itself: on the same configurations,
is POLAR's ligand-20-water coupling much stronger than GAFF2/TIP3P's?

Frames: (a) a fresh GAFF2/TIP3P MD at lambda = 1 (the leg's own System), 20 frames every 10 ps after
20 ps; (b) the lambda = 1 states (q 0, 1) of the D19 checkpoints segA and full, and of E8 fires_seed1.
Per frame: the 20 inner waters by the FIRES ordering (heavy atoms, as build_mlmm_system), then
  truth  = E_bb(L u I) - E_bb(L) - E_bb(I)                      POLAR-1-M, float64, CPU
  sum_c  = the head's coupling after the D17 cover switch and cap (what U uses at lambda < 1)
  E_MM   = GAFF2/TIP3P Coulomb + LJ between L and I, as a non-periodic cluster (truth's definition)
All in kJ/mol; negative = attractive.
"""
import json
import pathlib
import sys

import numpy as np
import openmm
import torch
from openmm import unit

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from decomp.fires import Fires, heavy_selection, water_indices  # noqa: E402
from decomp.lambda_mask import anchored_energy  # noqa: E402
from decomp.mlmm import CAP_M, MLMMForce  # noqa: E402
from run_abfe import load_model  # noqa: E402

EV = 96.48533212
ONE_4PI_EPS0 = 138.935456  # kJ/mol nm / e^2
OUT = pathlib.Path(__file__).resolve().parent
LEG = ROOT / "experiments/2026-10-15_acetamide_leg/em_acetamide.npz"
CKPT = ROOT / "runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt"
BB = "/home/ruigengji/MLP/mace/MACE-POLAR-1-M.model"


def mm_frames(system, pos_ang, box_ang, n=20, every_ps=10, equil_ps=20, seed=11):
    integ = openmm.LangevinMiddleIntegrator(298.15 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CUDA"))
    ctx.setPeriodicBoxVectors(*[openmm.Vec3(*(np.eye(3)[k] * box_ang[k] / 10)) for k in range(3)])
    ctx.setPositions(pos_ang * unit.angstrom)
    ctx.setVelocitiesToTemperature(298.15 * unit.kelvin, seed)
    integ.step(int(equil_ps * 500))
    out = []
    for _ in range(n):
        integ.step(int(every_ps * 500))
        out.append(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    return [np.asarray(p, dtype=np.float64) for p in out]


def checkpoint_frames(path, qs=(0.0, 1.0)):
    z = np.load(path, allow_pickle=True)
    return [(f"{path.parent.name} q{q:g}", np.asarray(z["P"][int(np.argmin(np.abs(z["qs"] - q)))], float))
            for q in qs]


def main():
    torch.set_num_threads(8)
    d = np.load(LEG, allow_pickle=True)
    system = openmm.XmlSerializer.deserialize(open(str(d["source"])).read())
    numbers, box = np.asarray(d["numbers"]), np.asarray(d["box"], float)
    lig = np.where(d["resids"] == d["resids"][0])[0]
    W = water_indices(d["resnames"], d["resids"])
    centre, wsel = heavy_selection(lig, numbers, W)
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    prm = np.array([[p.value_in_unit(u) for p, u in zip(nb.getParticleParameters(i),
                     (unit.elementary_charge, unit.nanometer, unit.kilojoule_per_mole))]
                    for i in range(system.getNumParticles())])

    model = load_model(str(CKPT), BB, device="cpu").double()

    frames = [(f"MM {k}", p) for k, p in enumerate(mm_frames(system, np.asarray(d["positions"], float), box))]
    for ck in ("experiments/2026-10-19_d19_acetamide/segA/state.npz",
               "experiments/2026-10-19_d19_acetamide/full/state.npz",
               "experiments/2026-10-16_mm_path_control/fires_seed1/state.npz"):
        frames += checkpoint_frames(ROOT / ck)

    rows = []
    for name, P in frames:
        fires = Fires(centre, wsel, n_inner=20, device="cpu")
        inner = fires.select_inner(P, box)
        mlf = MLMMForce(model, numbers, lig, W[inner], fires, es=True, device="cpu", cap=CAP_M)
        af = mlf.anch
        X = mlf.ml_positions(P, box)                          # L u I, unwrapped (A)
        with torch.no_grad():
            data = af._graph(X)
            lam = torch.ones_like(af.lam)
            _, c, parts = anchored_energy(af.model, data, lam, af.frag_id, ligand=af.ligand, es=af.es,
                                          cut=0.5, charges=af.charges, cover=af.cover, cap=af.cap)
        truth, sum_c, excess = (float(parts[k]) * EV for k in ("truth", "sum_c", "excess"))
        nl = len(lig)
        ia = mlf.ml_idx                                       # system indices of X's rows
        XL, XI = X[:nl] / 10, X[nl:] / 10                     # nm
        qL, sL, eL = prm[ia[:nl]].T
        qI, sI, eI = prm[ia[nl:]].T
        r = np.linalg.norm(XL[:, None] - XI[None], axis=-1)
        coul = ONE_4PI_EPS0 * (qL[:, None] * qI[None] / r).sum()
        sig, eps = 0.5 * (sL[:, None] + sI[None]), np.sqrt(eL[:, None] * eI[None])
        x6 = (sig / r) ** 6
        lj = (4 * eps * (x6 * x6 - x6)).sum()
        rows.append(dict(frame=name, truth=truth, sum_c=sum_c, cap_excess=excess, mm_coul=coul, mm_lj=lj,
                         mm=coul + lj, min_LI_A=float(10 * r.min())))
        print("%-22s truth %8.1f  sum_c %8.1f  excess %6.1f | MM %8.1f (C %8.1f  LJ %6.1f) | truth-MM %7.1f"
              % (name, truth, sum_c, excess, coul + lj, coul, lj, truth - coul - lj), flush=True)

    out = dict(rows=rows, summary={})
    for label, sel in (("MM ensemble", lambda n: n.startswith("MM ")),
                       ("D19 checkpoints", lambda n: "seg" in n or "full" in n),
                       ("E8 checkpoint", lambda n: "fires_seed1" in n)):
        R = [x for x in rows if sel(x["frame"])]
        s = {k: (float(np.mean([x[k] for x in R])), float(np.std([x[k] for x in R], ddof=1)) if len(R) > 1 else 0.0)
             for k in ("truth", "sum_c", "cap_excess", "mm", "mm_coul", "mm_lj")}
        diff = [x["truth"] - x["mm"] for x in R]
        s["truth_minus_mm"] = (float(np.mean(diff)), float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(R) > 1 else 0.0)
        s["n"] = len(R)
        out["summary"][label] = s
        print(f"\n{label} (n = {len(R)}): mean +- sd, kJ/mol")
        for k, v in s.items():
            if k != "n":
                print("  %-15s %8.1f +- %.1f%s" % (k, v[0], v[1], "  (SE)" if k == "truth_minus_mm" else ""))
    (OUT / "coupling_check.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

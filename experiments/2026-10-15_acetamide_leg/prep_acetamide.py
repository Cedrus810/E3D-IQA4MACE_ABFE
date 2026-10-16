"""Acetamide solvent leg for Stage A (D8, v3 S8.2), built in E3D's own settings.

Run on a GPU node: python prep_acetamide.py [box_nm] [npt_ns]

ABFE_IBS builds a solvent leg only from a complex, and acetamide has none,
so this builds it directly in OpenMM, matching the Atenolol leg
(ABFE_IBS/Atenolol-rank11/output/system_solvent.xml): PME, 1.0 nm cutoff,
Ewald tolerance 5e-4, dispersion correction, no switching, HBonds
constraints + rigid TIP3P, standard masses, no ions. Ligand: GAFF2
(gaff-2.11) with AM1-BCC charges (openmmforcefields + AmberTools sqm),
residue name MOL, first in the topology (mlmm.py takes the first residue as
the ligand). Temperature 298.15 K (E3D), not IBS's 300 K.

Steps: solvate a cubic box -> minimise -> NPT (1 bar, 298.15 K) for npt_ns
-> fix the box at the mean volume of the second half -> rescale the last
frame into it -> minimise -> 100 ps NVT. Writes, in this directory:
  system_solvent.xml      the NVT System (no barostat)
  topology_solvent.cif    topology + the final positions
  em_acetamide.npz        E3D leg file (positions A, numbers, resnames, resids, box A, source)
  prep.log                volumes, energies, ligand charges
"""

import sys
import time

import numpy as np
import openmm
from openff.toolkit import Molecule
from openmm import app, unit
from openmmforcefields.generators import GAFFTemplateGenerator

OUT = "/home/ruigengji/E3D/experiments/2026-10-15_acetamide_leg"
T = 298.15 * unit.kelvin
box_nm = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
npt_ns = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
t0 = time.perf_counter()

mol = Molecule.from_smiles("CC(N)=O")
mol.generate_conformers(n_conformers=1)
mol.assign_partial_charges("am1bcc")
q = np.asarray(mol.partial_charges.m_as("elementary_charge"), dtype=float)
gaff = GAFFTemplateGenerator(molecules=mol, forcefield="gaff-2.11")
ff = app.ForceField("amber14/tip3p.xml")
ff.registerTemplateGenerator(gaff.generator)

top = mol.to_topology().to_openmm()
for r in top.residues():
    r.name = "MOL"
pos = mol.conformers[0].to_openmm()
modeller = app.Modeller(top, pos)
modeller.addSolvent(ff, model="tip3p", boxSize=openmm.Vec3(box_nm, box_nm, box_nm) * unit.nanometer,
                    neutralize=False)
kw = dict(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds,
          rigidWater=True, ewaldErrorTolerance=5e-4)
system = ff.createSystem(modeller.topology, **kw)
for f in system.getForces():
    if isinstance(f, openmm.NonbondedForce):
        f.setUseDispersionCorrection(True)
        f.setUseSwitchingFunction(False)
n_wat = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
log = [f"acetamide CC(N)=O, GAFF2 + AM1-BCC charges {np.round(q, 4).tolist()} (sum {q.sum():+.2e} e)",
       f"box {box_nm} nm cubic, {n_wat} TIP3P waters, {system.getNumParticles()} atoms, "
       f"{system.getNumConstraints()} constraints"]
print(*log, sep="\n", flush=True)

plat = openmm.Platform.getPlatformByName("CUDA")
npt = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
npt.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, T, 25))
integ = openmm.LangevinMiddleIntegrator(T, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
integ.setRandomNumberSeed(7)
sim = app.Simulation(modeller.topology, npt, integ, plat)
sim.context.setPositions(modeller.positions)
sim.minimizeEnergy()
sim.context.setVelocitiesToTemperature(T, 7)
n_steps = int(npt_ns * 1e6 / 2)
vols = []
for k in range(100):
    sim.step(n_steps // 100)
    vols.append(sim.context.getState().getPeriodicBoxVolume().value_in_unit(unit.nanometer ** 3))
L = float(np.mean(vols[50:])) ** (1 / 3)
log.append(f"NPT {npt_ns} ns: volume first / mean 2nd half / sd 2nd half {vols[0]:.3f} / "
           f"{np.mean(vols[50:]):.3f} / {np.std(vols[50:]):.3f} nm^3 -> L = {L:.4f} nm")
print(log[-1], flush=True)

st = sim.context.getState(getPositions=True)
P = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
Lnow = np.diag(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
# rescale molecule centres (not atoms) into the fixed box
res_atoms = [[a.index for a in r.atoms()] for r in modeller.topology.residues()]
for idx in res_atoms:
    c = P[idx].mean(0)
    P[idx] += c * (L / Lnow - 1)
box_vec = [openmm.Vec3(L, 0, 0), openmm.Vec3(0, L, 0), openmm.Vec3(0, 0, L)] * unit.nanometer
system.setDefaultPeriodicBoxVectors(*box_vec)
integ2 = openmm.LangevinMiddleIntegrator(T, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
integ2.setRandomNumberSeed(8)
sim2 = app.Simulation(modeller.topology, system, integ2, plat)
sim2.context.setPeriodicBoxVectors(*box_vec)
sim2.context.setPositions(P * unit.nanometer)
sim2.minimizeEnergy()
sim2.context.setVelocitiesToTemperature(T, 8)
sim2.step(50000)
st = sim2.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=False)
P = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
log.append(f"NVT 100 ps at L = {L:.4f} nm: PE {st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
print(log[-1], flush=True)

xml = f"{OUT}/system_solvent.xml"
open(xml, "w").write(openmm.XmlSerializer.serialize(system))
app.PDBxFile.writeFile(modeller.topology, P * unit.angstrom, open(f"{OUT}/topology_solvent.cif", "w"))
atoms = list(modeller.topology.atoms())
np.savez(f"{OUT}/em_acetamide.npz", positions=np.asarray(P, dtype=np.float64),
         numbers=np.array([a.element.atomic_number for a in atoms]),
         resnames=np.array([a.residue.name for a in atoms]),
         resids=np.array([a.residue.index for a in atoms]),
         box=np.array([L, L, L]) * 10.0, source=np.array(xml))
log.append(f"wrote system_solvent.xml, topology_solvent.cif, em_acetamide.npz [{time.perf_counter() - t0:.0f}s]")
print(log[-1], flush=True)
open(f"{OUT}/prep.log", "w").write("\n".join(log) + "\n")

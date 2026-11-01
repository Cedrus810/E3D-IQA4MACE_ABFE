"""RDKit grouping of a ligand into the groups that become dummy one by one (DESIGN_2026-10-23 §3).

Run: python groups.py            (acetamide + the six IBS solvent legs; writes groups.json, prints a table)

Rule (user, 2026-10-23: "first CH3, then polar, non-polar, ..."):
  1. methyl   CH3 on one heavy atom, each its own group
  2. rings    each RDKit SSSR ring is one group, kept whole; RDKit says which are heterocycles
              (fused atoms go to aromatic rings first, then heterocycles); exocyclic =O / =S / =N and halogens on a
              ring atom go with the ring
  3. polar    off-ring connected components of: N, O, S, P, plus any atom with a non-aromatic
              multiple bond to N / O / S (C=O, C=N, C#N, ...)
  4. chain    the remaining off-ring heavy atoms, by connected piece; halogens stay with their atom
  Hydrogens follow their heavy atom. Order = methyl -> polar -> heterocycle -> carbocycle -> chain,
  within a class by lowest atom index. S0 = all coupled, S_k = first k groups dummy, S_K = all dummy.
  (First version, rings cut by their heteroatoms: groups_v1.py / groups_v1.log.)
The molecule comes from the leg's own topology (cif bonds) and MM charges (system xml); bond orders
from rdDetermineBonds with the net MM charge.
"""

import json
import pathlib

import numpy as np
import openmm
from openmm import app
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

HERE = pathlib.Path(__file__).resolve().parent
E3D = HERE.parents[1]
LEGS = {"acetamide": (E3D / "experiments/2026-10-15_acetamide_leg/topology_solvent.cif",
                      E3D / "experiments/2026-10-15_acetamide_leg/system_solvent.xml", None)}
for name in ["HSP90_ligand-17", "HSP90_ligand-19", "HSP90_ligand-27",
             "CyclophilinD_ligand-4", "CyclophilinD_ligand-8", "PWWP1_ligand-8"]:
    r = E3D / "data/solvent_legs_ibs" / name / "rep1"
    LEGS[name] = (r / "topology_solvent.cif", r / "system_solvent.xml", r / "ligand_indices_solvent.json")


def ligand_mol(cif, xml, idx_json):
    pdbx = app.PDBxFile(str(cif))
    atoms = list(pdbx.topology.atoms())
    lig = json.load(open(idx_json))["ligand_indices"] if idx_json else \
        [a.index for a in atoms if a.residue.index == atoms[0].residue.index]
    pos = {i: k for k, i in enumerate(lig)}
    system = openmm.XmlSerializer.deserialize(open(xml).read())
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    q = np.array([nb.getParticleParameters(i)[0]._value for i in lig])
    m = Chem.RWMol()
    for i in lig:
        m.AddAtom(Chem.Atom(atoms[i].element.atomic_number))
    for b in pdbx.topology.bonds():
        if b[0].index in pos and b[1].index in pos:
            m.AddBond(pos[b[0].index], pos[b[1].index], Chem.BondType.SINGLE)
    xyz = pdbx.getPositions(asNumpy=True)._value[lig] * 10
    conf = Chem.Conformer(len(lig))
    for k, p in enumerate(xyz):
        conf.SetAtomPosition(k, p.tolist())
    m.AddConformer(conf)
    mol = m.GetMol()
    rdDetermineBonds.DetermineBondOrders(mol, charge=int(round(q.sum())))
    return mol, lig, q


def groups(mol):
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    hs = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1] for i in heavy}
    hnb = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() > 1] for i in heavy}
    out, used = [], set()

    def add(kind, core):
        core = sorted(core)
        out.append({"kind": kind, "heavy": core, "atoms": sorted(core + [h for i in core for h in hs[i]])})
        used.update(core)

    for i in heavy:                                                       # 1. methyl
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() == 6 and len(hs[i]) == 3 and len(hnb[i]) <= 1:
            add("methyl", [i])

    def is_polar(i):
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() in (7, 8, 15, 16):
            return True
        return any(b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) and not b.GetIsAromatic()
                   and b.GetOtherAtom(a).GetAtomicNum() in (7, 8, 16) for b in a.GetBonds())

    def components(pool, linked):
        pool, comps = set(pool), []
        while pool:
            stack, c = [min(pool)], set()
            while stack:
                i = stack.pop()
                if i in c:
                    continue
                c.add(i)
                stack += [j for j in hnb[i] if j in pool and j not in c and linked(i, j)]
            pool -= c
            comps.append(sorted(c))
        return sorted(comps)

    halo = {i for i in heavy if mol.GetAtomWithIdx(i).GetAtomicNum() in (9, 17, 35, 53)}
    rings = [sorted(r) for r in Chem.GetSymmSSSR(mol)]                    # 2. rings, RDKit SSSR
    hetero = lambda r: any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r)
    ring_groups = []
    arom = lambda r: all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)
    for r in sorted(rings, key=lambda r: (not arom(r), not hetero(r), r[0])):  # aromatic, then hetero, claim fused atoms
        own = [i for i in r if i not in used]
        for i in list(own):                                               # exocyclic =O / =S / =N, halogens
            for b in mol.GetAtomWithIdx(i).GetBonds():
                j = b.GetOtherAtomIdx(i)
                if j in used or j in own or mol.GetAtomWithIdx(j).IsInRing():
                    continue
                if (b.GetBondType() == Chem.BondType.DOUBLE and len(hnb[j]) == 1) or j in halo:
                    own.append(j)
        if own:
            ring_groups.append(("heterocycle" if hetero(r) else "carbocycle", own))
            used.update(own)

    polar = [i for i in heavy if i not in used and is_polar(i)]           # 3. exocyclic polar
    for c in components(polar, lambda i, j: True):
        add("polar", c)
    for kind, own in ring_groups:
        add(kind, own)

    rest = [i for i in heavy if i not in used and i not in halo]          # 4. chain pieces
    for c in components(rest, lambda i, j: True):
        add("chain", c + [h for i in c for h in hnb[i] if h in halo and h not in used])
    for i in halo - used:                                                 # halogen on a polar atom
        add("halogen", [i])
    order = {"methyl": 0, "polar": 1, "heterocycle": 2, "carbocycle": 3, "chain": 4, "halogen": 4}
    out.sort(key=lambda g: (order[g["kind"]], g["heavy"][0]))
    assert sorted(a for g in out for a in g["atoms"]) == list(range(mol.GetNumAtoms()))
    return out


def main():
    res = {}
    for name, (cif, xml, idx) in LEGS.items():
        mol, lig, q = ligand_mol(cif, xml, idx)
        gs = groups(mol)
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        print(f"\n{name}: {smi}   {mol.GetNumAtoms()} atoms, net q {q.sum():+.3f}, "
              f"{len(gs)} groups -> {len(gs) + 1} states")
        for k, g in enumerate(gs):
            frag = Chem.MolFragmentToSmiles(mol, atomsToUse=g["atoms"], allHsExplicit=False)
            g["smiles"] = frag
            g["q"] = float(q[g["atoms"]].sum())
            g["system_index"] = [int(lig[a]) for a in g["atoms"]]
            print(f"  G{k + 1:<2d} {g['kind']:<8s} {len(g['atoms']):3d} atoms  q {g['q']:+.3f}  {frag}")
        res[name] = {"smiles": smi, "groups": gs}
    json.dump(res, open(HERE / "groups.json", "w"), indent=1)
    print(f"\nwrote {HERE / 'groups.json'}")


if __name__ == "__main__":
    main()

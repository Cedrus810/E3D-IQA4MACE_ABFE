"""One-time DES370K split. Run: python prepare_data.py

Writes train/val/test caches under data/. Splits by `system_id`, NOT by row:
DES370K holds many geometries of the same dimer pair, so a random row split puts
the same pair at a slightly different geometry in both train and test and the
held-out numbers come out flattering and wrong.
"""

import pathlib
import sys
import time

import numpy as np
import pandas as pd
from ase.data import atomic_numbers

from decomp.data import KCAL_PER_MOL_IN_EV

CSV = "DES370K.csv"
OUT = pathlib.Path("data")
TARGET = "cbs_CCSD(T)_all"
# SAPT components, stored alongside the total. They decompose the SAME
# interaction energy into pieces with very different spatial decay --
# electrostatics ~1/r, exchange ~exp(-r), dispersion ~1/r^6 -- so supervising
# all four constrains how D_ij is distributed over the boundary, not just its
# sum. One structure becomes four constraints at no extra quantum chemistry.
# 6.1% of DES370K rows have no SAPT; those come back as NaN and must be masked.
SAPT = ["sapt_es", "sapt_ex", "sapt_ind", "sapt_disp", "sapt_all"]
SPLIT = (0.80, 0.10, 0.10)
SEED = 0

CONFIGS = {
    # name           elements                             max_atoms  neutral_only
    "hcno_small":   (set("H C N O".split()),                     16, True),
    "hcno":         (set("H C N O".split()),                    None, True),
    "organic":      (set("H C N O F P S Cl Br I".split()),      None, True),
    "full":         (None,                                      None, False),
}


def build(name, elements, max_atoms, neutral_only, df):
    d = df
    if neutral_only:
        d = d[(d.charge0 == 0) & (d.charge1 == 0)]
    if max_atoms is not None:
        d = d[d.natoms0 + d.natoms1 <= max_atoms]
    if elements is not None:
        d = d[d.elements.map(lambda e: elements.issuperset(e.split()))]
    if len(d) == 0:
        print(f"  {name}: empty, skipped")
        return

    # group split: every geometry of one dimer pair lands in the same split
    systems = d.system_id.unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(systems)
    n_tr = int(SPLIT[0] * len(systems))
    n_va = int(SPLIT[1] * len(systems))
    where = {}
    for s in systems[:n_tr]:            where[s] = "train"
    for s in systems[n_tr:n_tr + n_va]: where[s] = "val"
    for s in systems[n_tr + n_va:]:     where[s] = "test"
    d = d.assign(split=d.system_id.map(where))

    OUT.mkdir(exist_ok=True)
    print(f"  {name}: {len(d):,} dimers over {len(systems):,} systems")
    for split in ("train", "val", "test"):
        part = d[d.split == split]
        pos, num, frag = [], [], []
        for sym_s, xyz_s, n0, n1 in zip(part.elements, part.xyz,
                                        part.natoms0, part.natoms1):
            sym = sym_s.split()
            xyz = np.fromstring(xyz_s, sep=" ").reshape(-1, 3)
            assert len(sym) == len(xyz) == n0 + n1, "elements/xyz/natoms disagree"
            pos.append(xyz)
            num.append(np.array([atomic_numbers[s] for s in sym]))
            frag.append(np.array([0] * n0 + [1] * n1))
        E = part[TARGET].to_numpy() * KCAL_PER_MOL_IN_EV
        path = OUT / f"des370k_{name}_{split}.npz"
        np.savez_compressed(
            path,
            positions=np.array(pos, dtype=object),
            numbers=np.array(num, dtype=object),
            frags=np.array(frag, dtype=object),
            energies=E,
            charges=(part.charge0 + part.charge1).to_numpy().astype(int),
            system_id=part.system_id.to_numpy(),
            sapt=np.stack([part[c].to_numpy() for c in SAPT], 1) * KCAL_PER_MOL_IN_EV,
            sapt_names=np.array(SAPT))
        sz = "-" if not len(pos) else f"{min(map(len,pos))}-{max(map(len,pos))}"
        n_sapt = int((~part[SAPT[0]].isna()).sum())
        print(f"    {split:5s} {len(part):7,d}  {part.system_id.nunique():5,d} sys  "
              f"sapt {100*n_sapt/max(len(part),1):3.0f}%  atoms {sz:8s} "
              f"|E_int| {np.abs(E).mean() if len(E) else 0:.4f} eV  "
              f"-> {path} ({path.stat().st_size/2**20:.0f} MiB)")


if __name__ == "__main__":
    if not pathlib.Path(CSV).exists():
        sys.exit(f"{CSV} not found")
    t0 = time.perf_counter()
    print(f"reading {CSV} ...", flush=True)
    df = pd.read_csv(CSV, usecols=["elements", "xyz", "natoms0", "natoms1",
                                   "charge0", "charge1", "system_id", TARGET] + SAPT)
    print(f"  {len(df):,} rows, {df.system_id.nunique():,} systems, "
          f"{time.perf_counter()-t0:.0f}s\n")
    for name, (el, ma, no) in CONFIGS.items():
        build(name, el, ma, no, df)
    print(f"\ndone in {time.perf_counter()-t0:.0f}s")

"""Data plumbing: MACE batches and the DES370K dimer set.

DES370K supplies counterpoise-corrected CCSD(T)/CBS interaction energies for
370,959 dimers -- the supervision L_int needs (RESEARCH_PLAN_v2.md S3.4, S13.5).
Fragment membership is free: the first `natoms0` atoms are fragment 0.
"""

import pathlib

import numpy as np
import torch

KCAL_PER_MOL_IN_EV = 0.043364104  # DES370K energies are kcal/mol; MACE works in eV


def mace_batch(positions, numbers, model, charges=None, spins=None,
               frag_ids=None, r_max=None, device="cpu", dtype=torch.float32):
    """Collate a list of structures into one MACE input dict.

    positions: list of [n_i, 3] arrays        numbers: list of [n_i] atomic numbers
    charges:   per-structure total charge     spins: per-structure multiplicity
    frag_ids:  list of [n_i] fragment labels, carried through as "frag_id"

    `total_spin`/`total_charge` are shaped [n_graphs], NOT [n_graphs, 1]: MACE
    gathers them per atom, so a trailing singleton dim broadcasts to
    [n_atoms, n_atoms, emb] and the forward pass dies in the joint embedding.
    """
    zs = model.atomic_numbers.tolist()
    r_max = float(model.r_max) if r_max is None else r_max
    ns = len(positions)

    pos, attrs, ei, batch, frag, ptr, off = [], [], [], [], [], [0], 0
    for i, (p, z) in enumerate(zip(positions, numbers)):
        p = torch.as_tensor(np.asarray(p), dtype=dtype)
        n = p.shape[0]
        a = torch.zeros(n, len(zs), dtype=dtype)
        a[torch.arange(n), [zs.index(int(x)) for x in z]] = 1.0
        # Self-pairs are excluded BY INDEX, never by distance. torch.cdist
        # switches to a matmul-based algorithm above 25 rows, whose cancellation
        # error leaves the diagonal at ~1e-4 instead of 0; a `d > 0` filter then
        # admits self-loops, edge_vec becomes the zero vector, and the spherical
        # harmonics normalise by zero -> NaN. Sharp onset at exactly 26 atoms.
        d = torch.cdist(p, p)
        adj = d < r_max
        adj.fill_diagonal_(False)
        ei.append(adj.nonzero().t() + off)
        pos.append(p); attrs.append(a)
        batch.append(torch.full((n,), i, dtype=torch.long))
        if frag_ids is not None:
            frag.append(torch.as_tensor(np.asarray(frag_ids[i]), dtype=torch.long))
        off += n; ptr.append(off)

    e = torch.cat(ei, 1).contiguous().to(device)
    out = {
        "positions": torch.cat(pos).to(device),
        "node_attrs": torch.cat(attrs).to(device),
        "edge_index": e,
        "shifts": torch.zeros(e.shape[1], 3, dtype=dtype, device=device),
        "unit_shifts": torch.zeros(e.shape[1], 3, dtype=dtype, device=device),
        "cell": torch.zeros(3, 3, dtype=dtype, device=device),
        "batch": torch.cat(batch).to(device),
        "ptr": torch.tensor(ptr, device=device),
        "head": torch.zeros(ns, dtype=torch.long, device=device),
        "total_charge": torch.as_tensor(
            [0] * ns if charges is None else charges, dtype=torch.long, device=device),
        "total_spin": torch.as_tensor(
            [1] * ns if spins is None else spins, dtype=torch.long, device=device),
    }
    if frag_ids is not None:
        out["frag_id"] = torch.cat(frag).to(device)
    return out


def load_des370k(path="DES370K.csv", n=2000, max_atoms=None, neutral_only=False,
                 elements=None, target="cbs_CCSD(T)_all", seed=0, cache=None):
    """Read DES370K into (positions, numbers, frag_ids, E_int_eV, charges).

    The csv is 293 MB and mostly xyz strings, so it is read in chunks with a
    random subsample taken from each and the result cached to a small .npz.
    Re-runs with the same cache path skip the csv entirely.

    elements: keep only structures whose atoms all lie in this set of symbols.
    target:   'cbs_CCSD(T)_all' is the gold standard; 'sapt_all' and the
              individual sapt_* components are available as separate signals.
    """
    import pandas as pd
    from ase.data import atomic_numbers

    if cache is not None and pathlib.Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        return (list(z["positions"]), list(z["numbers"]), list(z["frags"]),
                z["energies"], z["charges"])

    cols = ["elements", "xyz", "natoms0", "natoms1", "charge0", "charge1", target]
    keep_el = None if elements is None else set(elements)
    rng = np.random.default_rng(seed)
    parts, got = [], 0

    for chunk in pd.read_csv(path, usecols=cols, chunksize=20_000):
        if neutral_only:
            chunk = chunk[(chunk.charge0 == 0) & (chunk.charge1 == 0)]
        if max_atoms is not None:
            chunk = chunk[chunk.natoms0 + chunk.natoms1 <= max_atoms]
        if keep_el is not None:
            chunk = chunk[chunk.elements.map(lambda e: keep_el.issuperset(e.split()))]
        if len(chunk) == 0:
            continue
        # subsample each chunk so no single region of the file dominates
        take = min(len(chunk), max(1, n // 8))
        parts.append(chunk.sample(n=take, random_state=int(rng.integers(1 << 30))))
        got += take
        if got >= 4 * n:            # enough to draw a clean sample from
            break

    df = pd.concat(parts)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)

    positions, numbers, frags = [], [], []
    for sym_s, xyz_s, n0, n1 in zip(df.elements, df.xyz, df.natoms0, df.natoms1):
        sym = sym_s.split()
        xyz = np.fromstring(xyz_s, sep=" ").reshape(-1, 3)
        assert len(sym) == len(xyz) == n0 + n1, "elements/xyz/natoms disagree"
        positions.append(xyz)
        numbers.append(np.array([atomic_numbers[s] for s in sym]))
        frags.append(np.array([0] * n0 + [1] * n1))

    energies = df[target].to_numpy() * KCAL_PER_MOL_IN_EV
    charges = (df.charge0 + df.charge1).to_numpy().astype(int)

    if cache is not None:
        np.savez_compressed(
            cache,
            positions=np.array(positions, dtype=object),
            numbers=np.array(numbers, dtype=object),
            frags=np.array(frags, dtype=object),
            energies=energies, charges=charges)
    return positions, numbers, frags, energies, charges

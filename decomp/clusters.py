"""Many-body training structures, labelled by the backbone itself.

DES370K is dimers: two fragments, 20-50 cross edges, one CCSD(T) number per
structure. `L_int` pins the SUM over those edges, so a per-edge bias `delta` is
only ever seen as `n_edges * delta` -- and at dimer scale that is small enough
to be absorbed by the rest of the fit. In a solvated cluster `n_edges` is 1500,
and the same `delta` becomes the whole answer. Measured on the Atenolol carve:

    backbone truth   -1.51 eV      sum D_ia (POLAR-1-M head)   -4.65 eV
    per-edge bias    -2.0 meV, near-constant from 229 to 1524 cross edges

More dimers cannot fix that. More edges per label can, and they are free: the
frozen backbone gives an exact label for ANY geometry and ANY whole-molecule
partition,

    E_couple = E(A u B) - E(A) - E(B),

with no QM. Clusters of 2-25 residues span the edge counts the dimers miss.
The CCSD(T) and SAPT anchors stay on the dimers -- those are real data, and a
backbone label only teaches the head to reproduce its own backbone.

ponytail: one minimised frame is the only geometry source here, so diversity
comes from which residues are kept and how they are split. Add frames from an
MD trajectory if the head starts memorising this box.
"""

import numpy as np
import torch

from .data import mace_batch


def _residue_groups(resids):
    """Atom indices per residue, in a stable order."""
    uniq = np.unique(resids)
    return [np.where(resids == r)[0] for r in uniq]


def _carve(centroids, seed, n_keep, box, exact=None):
    """The `n_keep` residues nearest the seed, each minimum-imaged onto it.

    Whole residues only: a half water has no meaning to the backbone, and the
    label E(A u B) - E(A) - E(B) is nonsense if a fragment is a radical.

    Selection is by residue centroid, vectorised over all residues at once.
    `carve_from_arrays` uses the closest-atom distance instead, which matters
    for an elongated 41-atom ligand but not for a 3-atom water -- so `exact`
    carries precomputed closest-atom distances for the seeds where it does,
    and this only ever has to pick training structures, not a canonical carve.
    ponytail: centroid selection for waters, upgrade if a solvent with large
    residues shows up.
    """
    if box is not None:
        shifts = -np.round((centroids - centroids[seed]) / box) * box
    else:
        shifts = np.zeros_like(centroids)
    d = (np.linalg.norm(centroids + shifts - centroids[seed], axis=-1)
         if exact is None else exact.copy())
    d[seed] = -1.0                       # the seed is always kept, and first
    order = np.argpartition(d, min(n_keep + 1, len(d) - 1))[:n_keep + 1]
    return order[np.argsort(d[order])], shifts


def solvent_clusters(path="em_solvent.npz", n=2000, rng=None, size=(2, 24),
                     ligand_frac=0.25, ligand_resname="MOL"):
    """Clusters of whole residues with an A|B partition.

    `ligand_frac` of them are ligand-vs-solvent, the case WP9 actually needs;
    the rest are solvent-only with a random split, which is where most of the
    cross-edge counts come from and costs nothing to generate.

    Returns (positions, numbers, frag_ids) lists, ready for `mace_batch`.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    z = np.load(path, allow_pickle=True)
    pos_all = np.asarray(z["positions"])
    num_all = np.asarray(z["numbers"])
    resids = np.asarray(z["resids"])
    resnames = np.asarray(z["resnames"])
    box = np.asarray(z["box"]) if "box" in z.files else None

    groups = _residue_groups(resids)
    is_lig = np.array([bool((resnames[g] == ligand_resname).all()) for g in groups])
    lig_seeds = np.where(is_lig)[0]
    wat_seeds = np.where(~is_lig)[0]
    centroids = np.array([pos_all[g].mean(0) for g in groups])

    # Closest-atom distances from each ligand, computed once: the ligand is a
    # fixed 41-atom residue and thousands of clusters reuse the same ranking.
    exact = {}
    for s in lig_seeds:
        sh = (-np.round((centroids - centroids[s]) / box) * box
              if box is not None else np.zeros_like(centroids))
        exact[int(s)] = np.array([
            np.linalg.norm(pos_all[g] + sh[k] - pos_all[groups[s]][:, None],
                           axis=-1).min()
            for k, g in enumerate(groups)])

    out_pos, out_num, out_frag = [], [], []
    for _ in range(n):
        want_lig = lig_seeds.size and rng.random() < ligand_frac
        seed = int(rng.choice(lig_seeds if want_lig else wat_seeds))
        n_keep = int(rng.integers(size[0], size[1] + 1))
        order, shifts = _carve(centroids, seed, n_keep, box,
                               exact=exact.get(seed))

        parts = [pos_all[groups[k]] + shifts[k] for k in order]
        sizes = [len(groups[k]) for k in order]

        if want_lig:
            # the target case: one ligand against everything else
            a = np.zeros(len(order), dtype=bool)
            a[0] = True
        else:
            # a random non-empty proper subset, so cross-edge counts spread
            # over the whole range instead of clustering at "one residue vs
            # the rest", which is the low end of it.
            k = int(rng.integers(1, len(order)))
            a = np.zeros(len(order), dtype=bool)
            a[rng.choice(len(order), k, replace=False)] = True

        frag = np.concatenate([np.full(s, 0 if flag else 1, dtype=np.int64)
                               for s, flag in zip(sizes, a)])
        keep = np.concatenate([groups[k] for k in order])
        out_pos.append(np.concatenate(parts))
        out_num.append(num_all[keep])
        out_frag.append(frag)
    return out_pos, out_num, out_frag


@torch.no_grad()
def coupling_labels(bb, positions, numbers, frags, device="cuda", batch=8):
    """E(A u B) - E(A) - E(B) from the frozen backbone, in eV.

    In float64. The totals are ~1e4 eV and the answer is ~1 eV, so float32
    leaves ~1e-2 eV of cancellation noise -- a few percent of the smaller
    labels, and this runs once.
    """
    was = next(bb.parameters()).dtype
    bb.to(torch.float64)
    try:
        def E(ps, zs):
            out = []
            for k in range(0, len(ps), batch):
                sl = slice(k, k + batch)
                b = mace_batch(ps[sl], zs[sl], bb, charges=[0] * len(ps[sl]),
                               device=device, dtype=torch.float64)
                out.append(bb(b, training=False,
                              compute_force=False)["energy"].detach())
            return torch.cat(out)

        whole = E(positions, numbers)
        a = E([p[f == 0] for p, f in zip(positions, frags)],
              [z[f == 0] for z, f in zip(numbers, frags)])
        b_ = E([p[f == 1] for p, f in zip(positions, frags)],
               [z[f == 1] for z, f in zip(numbers, frags)])
    finally:
        bb.to(was)
    return (whole - a - b_).cpu().numpy()

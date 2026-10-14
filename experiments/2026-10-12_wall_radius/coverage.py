"""D15, part 1: which short ligand-water contacts has the head's training data seen?

Exactly the training set of the operating head (arm N / D6): DES370K `organic`
train, the 50,000 drawn by test_joint.load_split (seed 0), plus the 10,000
clusters solvent_clusters(frames_solvent.npz, size 2-24, seed 1). Per
structure, the shortest cross-fragment distance for each element pair; then
quantiles over structures. CPU only. Run from the repo root.
"""
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np
from decomp.clusters import solvent_clusters

SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
LIG = (1, 6, 7, 8)          # Atenolol's elements
WAT = (8, 1)                # water O, H
OUT = os.path.dirname(os.path.abspath(__file__))


def per_structure_min(pos_l, num_l, frag_l, pairs, directed=False, need=None):
    """directed: x on fragment 0, y on fragment 1 only (ligand -> water).
    need: keep only structures whose fragment 0 contains this element."""
    res = {p: [] for p in pairs}
    for P, Z, F in zip(pos_l, num_l, frag_l):
        P, Z, F = np.asarray(P), np.asarray(Z), np.asarray(F)
        A, B = F == 0, F == 1
        if need is not None and not (Z[A] == need).any():
            continue
        d = np.linalg.norm(P[A][:, None] - P[B][None], axis=-1)
        za, zb = Z[A][:, None], Z[B][None]
        for (x, y) in pairs:
            m = (za == x) & (zb == y)
            if not directed:
                m = m | ((za == y) & (zb == x))
            if m.any():
                res[(x, y)].append(d[m].min())
    return {p: np.array(v) for p, v in res.items()}


def table(name, res):
    rows = {}
    print(f"\n{name}: per-structure shortest cross distance (A)")
    print(f"  {'pair':6s}{'n':>8s}{'min':>7s}{'p0.1':>7s}{'p1':>7s}{'p5':>7s}{'median':>8s}")
    for (x, y), v in res.items():
        if len(v) == 0:
            continue
        q = np.quantile(v, [0.001, 0.01, 0.05, 0.5])
        key = f"{SYM[x]}-{SYM[y]}"
        rows[key] = dict(n=int(len(v)), min=float(v.min()), p01=float(q[0]), p1=float(q[1]),
                         p5=float(q[2]), median=float(q[3]))
        print(f"  {key:6s}{len(v):8d}{v.min():7.2f}{q[0]:7.2f}{q[1]:7.2f}{q[2]:7.2f}{q[3]:8.2f}")
    return rows


pairs = [(x, y) for x in LIG for y in WAT]
z = np.load("data/des370k_organic_train.npz", allow_pickle=True)
# Read each array ONCE: z["positions"] re-reads and unpickles the whole
# 232k-object array on every access -- inside a comprehension that is
# 50,000 full reads, which is why the first run timed out at 900 s.
Pz, Zz, Fz = z["positions"], z["numbers"], z["frags"]
i = np.random.default_rng(0).choice(len(Pz), 50000, replace=False)
des = per_structure_min([Pz[k] for k in i], [Zz[k] for k in i], [Fz[k] for k in i], pairs)
print(f"DES370K done", flush=True)
cp, cn, cf = solvent_clusters("frames_solvent.npz", n=10000, size=(2, 24),
                              rng=np.random.default_rng(1))
# fragment 0 of a cluster is the ligand only if it has carbon (Atenolol);
# the rest are water-water clusters
clu = per_structure_min(cp, cn, cf, pairs, directed=True, need=6)
out = {"des370k_organic_train_50k (any fragment pair; second element = partner atom)":
           table("DES370K organic train, 50k (unordered element pair across the fragment boundary)", {p: v for p, v in des.items() if p != (1, 8)}),
       "clusters_10k (ligand element - water element)":
           table("Training clusters with the ligand (ligand element - water element)", clu)}
json.dump(out, open(f"{OUT}/coverage.json", "w"), indent=1)

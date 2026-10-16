"""E0 analysis: does FIRES distort the solvent around the ligand? (v3 S6.3.5)

Run: python analyze_e0.py FREE.npz FIRES.npz [n_blocks]

Rowley & Roux's criterion (Fig. 2): the restrained and the free run must give
the same solvent structure, with small deviations only where the restraint
acts. Here, on the restraint's own ordering distance d:

  - n(d): mean number of waters per 0.1 A bin of d, 1.5-8 A, free vs FIRES,
    with block-average standard errors (n_blocks contiguous blocks per run);
  - N(d) = number of waters with d < x, at a few x;
  - the 20th-smallest d in the free run against R_inner in the FIRES run:
    the same random variable if FIRES only relabels;
  - FIRES diagnostics: fraction of snapshots with a violation, depth, <U_F>.
"""

import sys

import numpy as np


def blocks(x, nb):
    """mean and SE over nb contiguous blocks of axis 0."""
    b = np.array_split(x, nb)
    m = np.array([bb.mean(0) for bb in b])
    return m.mean(0), m.std(0, ddof=1) / np.sqrt(nb)


def main(free, fires, nb=5):
    A, B = np.load(free), np.load(fires)
    dA, dB = A["d"], B["d"]
    print(f"free: {dA.shape[0]} snapshots, FIRES: {dB.shape[0]} snapshots "
          f"({float(A['dense_ps'])} ps apart), {dA.shape[1]} waters, n_inner {int(B['n_inner'])}, "
          f"{nb} blocks; mean T {A['T'].mean():.1f} / {B['T'].mean():.1f} K")
    edges = np.arange(1.5, 8.01, 0.1)
    hA = np.stack([np.histogram(x, edges)[0] for x in dA]).astype(float)
    hB = np.stack([np.histogram(x, edges)[0] for x in dB]).astype(float)
    mA, sA = blocks(hA, nb)
    mB, sB = blocks(hB, nb)
    z = (mB - mA) / np.sqrt(sA ** 2 + sB ** 2 + 1e-12)
    print("\nn(d), waters per 0.1 A bin: d  free +- SE   FIRES +- SE   z")
    for i in range(len(mA)):
        if mA[i] + mB[i] > 0.02:
            print(f"  {edges[i]:4.1f}-{edges[i + 1]:4.1f}  {mA[i]:6.3f} +- {sA[i]:.3f}   "
                  f"{mB[i]:6.3f} +- {sB[i]:.3f}   {z[i]:+5.1f}")
    print(f"  bins with |z| > 3: {int((np.abs(z) > 3).sum())} of {int((mA + mB > 0.02).sum())} populated")

    print("\nN(d < x): free +- SE   FIRES +- SE")
    for x in (2.5, 3.0, 3.5, 4.0, 5.0, 6.0):
        a, sa = blocks((dA < x).sum(1).astype(float), nb)
        b, sb = blocks((dB < x).sum(1).astype(float), nb)
        print(f"  x = {x:.1f} A   {a:7.2f} +- {sa:.2f}   {b:7.2f} +- {sb:.2f}")

    n = int(B["n_inner"])
    kth = np.sort(dA, axis=1)[:, n - 1]
    ra = B["r_inner"]
    qs = (5, 25, 50, 75, 95)
    print(f"\n{n}th-smallest d (free) vs R_inner (FIRES), A: mean +- SE; quantiles {qs}")
    for name, x in (("free", kth), ("FIRES", ra)):
        m, s = blocks(x, nb)
        print(f"  {name:6s} {m:.3f} +- {s:.3f}   " + " ".join(f"{np.percentile(x, q):.2f}" for q in qs))

    v = B["n_viol"] > 0
    print(f"\nFIRES diagnostics: snapshots with a violation {100 * v.mean():.1f}%; "
          f"violating pairs per snapshot {B['n_viol'].mean():.2f} (max {B['n_viol'].max()}); "
          f"depth when violated: median {np.median(B['max_depth'][v]) if v.any() else 0:.3f}, "
          f"max {B['max_depth'].max():.3f} A; <U_F> {B['U'].mean():.2f} kJ/mol "
          f"(max {B['U'].max():.1f})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 5)

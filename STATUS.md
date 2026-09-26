# Status — 2026-09-26 (rev 6)

Snapshot of where the project stands. Numbers cite `RESEARCH_PLAN_v2.md §13`,
which holds the full experimental record; this file is the short version plus
the things that are half-done and easy to forget.

## Current conclusions

What stands, with the number that supports it:

1. **A node gauge can be imposed on frozen backbone features** at almost
   unchanged E/F: 28× (toy), 49× (`mace-omol-0`) off without supervision
   (§13.1–13.3). Synthetic teacher only -- not yet shown for the *IQA* gauge.
2. **`L_int` pins the pair sum**: `Σ D_AB` to 0.27 kcal/mol on 228 held-out
   DES370K systems (`mace-omol-0`, §13.5); 0.43 kcal/mol on the DES370K test
   split for the current POLAR-1-M head.
3. **The alchemical arithmetic is exact**: graph masking passes all endpoint
   identities, edge-only scaling is off by 5.9% (§13.7); the diagonal path equals
   Shapley to 0.82% (§13.8); peel = direct to 1e-4 on a 341-atom carve.
4. **SAPT supervision resolves individual pairs**: 6–16× better channels, toy
   and real data agree (§13.10, §13.11).
5. **Dimer-only heads carry a per-edge bias** (-2.0 meV/edge, grows with the
   system); **backbone-labelled clusters remove it**: -0.14 meV/edge on the
   training frame, +0.34 held out, 7× smaller (§13.15, §13.16).
6. **Backbone: POLAR-1-M.** `mace-omol-0` sign-flips at scale (§13.15);
   POLAR-1-L is no better and 2.2× the cost (§13.17).

What does not stand, or is not yet known:

7. **`direct` does not extrapolate** past the training cluster size: held out,
   +0.36 eV at n=50 and +2.07 at n=100. Clusters from one frame are memorised,
   on both M and L. Next experiment: §13.19.
8. **§13.14's physics is withdrawn**: the "47% polarisation" (+0.797 eV) was a
   per-edge bias. Its identities (peel = direct, forward = reverse) stand.
9. **No free energy with real sampling, and no real IQA labels**, yet.

## Work packages

| WP | | Evidence |
|---|---|---|
| 1 e3nn reimplementation + negative control | done | 28× (§13.1) |
| 2 portable decomposition head | done | 4 backbone configs, multi-channel, projection (§13.2) |
| 3 `L_int` pins `Σ D_AB` | done | 0.27 kcal/mol, 228 held-out systems (§13.5) |
| 4 frozen pretrained backbone | done | 1.12× of fine-tuning at n=512, 3% of parameters (§13.5) |
| ~~5 train from scratch on OMol25~~ | **cut** | frozen backbone suffices (§13.5) |
| 6 λ masking + endpoint identities | done | v1's design off by 5.9% of the coupling; graph masking exact (§13.7) |
| 7 per-atom attribution | done | 0.82% against exhaustive Shapley (§13.8) |
| — per-atom resolution of `D_ia` | done | SAPT components: 6–16×, toy and real data agree (§13.10, §13.11) |
| — first complete model | done | five terms, all metrics usable (§13.12) |
| 8 sampling + TI/MBAR pipeline | assembled; **a design flaw found and not yet fixed** | §13.14 |
| — size extrapolation of `D_ij` | **fixed on POLAR-1-M** by backbone-labelled clusters; L tested, no gain — M is the backbone | below |
| 9 validation ladder | blocked on scale | ML/MM is the only route (§13.13) |

## What exists in code

```
decomp/
  head.py             E_intra per node, D_ij per edge; backbone-agnostic
  mace_adapter.py     head on a frozen MACE backbone
  lambda_mask.py      "edge" / "graph" / body-order coupling + isolated-fragment reference
  losses.py           E, F, IQA, L_int, SAPT (dataset-level scaling)
  data.py             DES370K batching; MACE input dicts
  train.py            training step
  fe.py               TI, per-atom TI, MBAR, overlap  (pymbar shims from ABFE_IBS)
  sampling.py         torch Langevin (small systems)
  openmm_bridge.py    PythonForce bridge; loads prepared ABFE_IBS legs;
                      carves an ML region (whole residues, minimum image)
  hremd.py            replica exchange over lambda; torch or openmm backend
  clusters.py         ligand + n-water clusters carved from a leg, coupling labelled by the frozen backbone (fp64)

em_system.py          minimise a prepared leg with OpenMM's own MM force field
run_abfe.py           two-leg double decoupling driver
test_peel.py          peel the ligand atom by atom against a direct reference
test_scale.py         head coupling vs backbone truth at n_solvent = 5..100
test_polar.py         MACE-POLAR-1 adapter checks (box independence, batching, forces)
run_polar.sh          POLAR-1 M/L training; CLUSTERS=10000 by default
```

**Minimise with MM, score with ML.** Minimising a carved cluster with the
learned potential drives atoms into each other — outside its training domain an
MLIP has no repulsive wall, and LBFGS happily runs the energy to -5e5 eV. The MM
force field has hard cores and OpenMM's minimiser is well tested: 4,076 atoms,
max|F| 158755 -> 2360 kJ/mol/nm in one second.

Borrowed from `/home/ruigengji/ABFE_IBS`, not rewritten: the pymbar shims
(`fe.py`), and the prepared Atenolol legs (`output/system_solvent.xml` +
`topology_solvent.cif`, 4,076 atoms, already solvated and equilibrated).
Restraints, standard-state corrections and the production protocol should be
called from there too when the complex leg is run.

All eleven `test_*.py` pass. `test_endpoints.py`, `test_attribution.py` and
`test_openmm.py` need no data and run in seconds.

## First real-system result, and what it broke

Atenolol (41 atoms) + its 100 nearest waters, carved from the prepared solvent
leg after minimising the full 4,076-atom box with OpenMM's own MM force field.
One configuration; the identities are per-configuration.

    U(1) - U(0)        -1.6855 eV = -162.6 kJ/mol = -38.9 kcal/mol
    41-step peel       agrees to 1.4e-4 relative
    forward vs reverse agrees to 2.7e-4 eV

Sign and magnitude are right for a neutral polar drug. **Two things broke:**
*(Rev 4: the size scan below shows this agreement is a coincidence of this one
carve size, and that item 1's +0.797 eV is mostly absorbing a per-edge bias,
not measuring polarisation. Item 2's order dependence also shrinks 8x once the
bias is fixed.)*

1. **TI would miss 47% of the answer.** The graph mask is `keep = w > 0`, a step
   function: the graph is complete for every lambda > 0 and cuts only at
   lambda = 0. TI integrates `sum_a D_ia` and never sees the jump, which here is
   +0.797 eV = 18.4 kcal/mol — the polarisation cost of staying coupled. Pair
   terms attract by 57 kcal/mol, polarisation costs 18, net 39. On a randomly
   initialised head this gap was 4.2%; trained, it is 47% of the net coupling.

2. **Sequential per-atom values are not usable.** Removal order changes them by
   up to 0.619 eV = 14.3 kcal/mol — larger than most atoms contribute, enough to
   flip signs. The diagonal path's values span -0.35 to +0.14 eV and are
   order-independent by construction. S13.8 measured 1.3% order dependence on a
   water dimer; on a real drug it is the size of the signal. This turns the
   diagonal path from "cleaner" into "the only usable route".

**The fix, not yet implemented:** a two-stage protocol, TI over the continuous
segment plus one discrete graph-cut step by BAR. `dG_cut` is not a correction to
hide — it is the ligand's polarisation stabilisation by solvent, a physical
observable MM FEP cannot decompose out.

## Size scan: a per-edge bias, and the fix (§13.15, §13.16)

`test_scale.py` carves the same Atenolol frame at n_solvent = 5/10/20/50/100 and
compares the head's coupling with the backbone's own
`E(all) - E(ligand) - E(water)`. Heads trained on DES370K dimers only:

    edges   POLAR-1-M direct-truth   omol direct-truth
      229        -0.48 eV                +0.96 eV
      423        -0.74                   +2.26
      719        -1.63                   +2.69
     1228        -2.94                   +2.77
     1524        -4.25                   +0.07   <- the carve above

- **POLAR-1-M**: linear in edge count, -2.0 meV/edge. `L_int` pins one scalar
  per dimer to the *sum* of a few dozen cross edges; a uniform per-edge offset
  is invisible there and grows as n_edges x delta in solution. This is the pair
  non-identifiability of §13.10/13.11, now measured in the condensed phase.
- **omol**: non-monotonic and sign-flipped (coupling repulsive at 4 of 5 sizes).
  The 4% agreement above is where the curve happens to cross zero; the
  "relaxation" term `direct - sumD` runs 45%–201% of the true coupling.
- Not the long-range term: POLAR's electrostatic + electron energy is 9.9% of
  this coupling, the head's error was 3.7x.
- `truth` is an fp32 difference of three ~3e4 eV numbers: good to ±0.05 eV.

**Fix:** the arm-D target is the backbone's own energy, so any geometry can be
labelled for free. `decomp/clusters.py` cuts 10,000 ligand + 2–24 water clusters
from `em_solvent.npz`; `test_joint.py` (`E3D_CLUSTERS`) adds `L_E + L_F` and a
coupling loss pinning `sum D_LE` to the backbone coupling, normalised separately
from the dimer `L_int`. Result on POLAR-1-M, 60k steps:

    edges   direct-truth   old       Atenolol n=100      old POLAR-M   new
      229     -0.18       -0.48      direct (eV)          -5.758       -1.757
      423     -0.21       -0.74      sum D_ia             -4.654       -1.772
      719     -0.25       -1.63      direct - sumD        -1.104       +0.014
     1228     -0.29       -2.94      peel order, /atom     0.236        0.077
     1524     -0.28       -4.25      peel vs direct        1.2e-4       4.6e-5

Per-edge bias -2.0 -> -0.14 meV; the error no longer grows with size.
n = 50/100 are larger than any training cluster (<= 24 waters), so those rows
are real extrapolation. The cut term is now ~0, so ΣD alone carries the
coupling. Cost on DES370K: `E_int` 0.393 -> 0.428 kcal/mol, E 2.54 -> 2.83
meV/atom, F 30.1 -> 33.3 meV/Å.

**Held-out frame** (`md50_solvent.npz`: em_solvent + 50 ps MM MD + MM
minimise; ligand moved 5 Å, all waters rearranged; the carve has 2684 cross
edges, not 1524):

    edges  truth    new sumD  new direct  dir-tru   |  old sumD  old dir-tru
      307  -0.91    -0.41     -0.95       -0.04     |  -0.89     -0.63
      547  -1.13    -0.72     -1.26       -0.13     |  -1.75     -1.28
      975  -1.64    -1.20     -1.75       -0.11     |  -3.31     -2.36
     2127  -2.47    -1.75     -2.10       +0.36     |  -7.63     -6.05
     2684  -2.84    -2.07     -0.78       +2.07     |  -9.92     -7.84

- The per-edge bias fix holds out of sample: +0.34 vs -2.43 meV/edge, 7x.
- **`direct` does not extrapolate**: good to 0.13 eV up to n=20 (training
  cluster sizes), then +0.36 and +2.07 eV. The cut term `direct - sumD` is
  -0.5 eV at n=5 and +1.3 eV at n=100 -- the "+0.014, ≈0" on the training frame
  was memorisation of that box.
- Next: clusters from many MD frames (the `ponytail:` in `clusters.py`), then
  rescan on a frame not in training.

## POLAR-1-L: no gain over M (§13.17)

Same recipe as M but batch 8 (L at 16 does not fit 11 GiB). DES370K: E 2.83 ->
3.60 meV/atom, F 33.3 -> 40.2 meV/Å, `E_int` 0.428 -> 0.537 kcal/mol. Held out,
sumD is worse at every size (+0.41 vs +0.34 meV/edge) and `direct` fails past 24
waters exactly as on M. Same failure on both backbones -> the bottleneck is the
one-frame clusters, not backbone size. Full table in §13.17; logs in
`runs/polar-L/logs/scale*.log`. The 1524-edge training-frame row is unverified
(L's truth equals M's to four decimals).

## The gap

**No free-energy calculation with real sampling has been done yet.** The
identities above are single-configuration; everything about sampling
convergence, lambda spacing and overlap is still untested. Every result
so far averages over a fixed conformer set, deliberately, to separate the
attribution arithmetic from sampling convergence. The pieces for the real thing
are built and individually verified; assembling them is the next task:

```
λ windows → OpenMM MD → collect dU/dλ → TI + MBAR → per-atom dG_i
                                       ↘ overlap matrix
             control: message masking (S7.1) — right endpoint, no attribution
```

That control is the experiment that justifies the decomposition: both routes
must give the same total ΔG (state function), but the decomposition should
converge faster, and only it yields per-atom numbers.

## Scale, measured

Full-ML condensed phase is out of reach: the 4,076-atom solvent box needs ~55
GiB and 11 days per lambda window (§13.13). ML/MM over ligand-plus-near-water
is ~200 ms/step, about 66 h for a 12-window leg at 100 ps each — that is WP9's
path. Every remaining *methodological* question (lambda spacing, overlap,
softcore, the message-masking control, attribution vs Shapley) lives in 6–50
atom systems where sampling takes minutes, so WP8 is not blocked by this.

Also measured, on a 6-atom system: the neighbour list costs 0.4 ms and graph
construction 0.2 ms, but the MACE forward+backward costs 30 ms and the OpenMM
PythonForce round trip pushes the step to ~150 ms. CPU beats CUDA below ~100
atoms. None of this matters at complex scale, where compute dominates — which is
why `hremd.py` has both backends rather than one tuned for dimers.

## Open items, in priority order

0. **Clusters from many MD frames, on POLAR-1-M** -- the full plan with pass
   criteria is §13.19. First code change: the cluster and target cache names
   must include the frame source, or the run silently reuses single-frame
   labels. Frames come from CPU OpenMM, so this does not need the GPU until
   training.
1. **Two-stage protocol** (above) -- only if §13.19 leaves a cut term
   `direct - sumD` above the truth noise; until decided, any free energy
   from this pipeline may be off by that term. Changes land in
   `lambda_mask.py` (an explicit cut Hamiltonian) and `run_abfe.py` (BAR after
   the TI segment).
2. **Then the Stage A comparison**: decomposition vs message-masking baseline.
   Both must give the same total dG; the decomposition should converge faster
   and is the only one that yields per-atom numbers.
3. **Rerun arm D longer.** Its loss was still descending at 32k steps, so the
   +106% energy cost of adding SAPT is an upper bound, not a measurement. The
   per-batch SAPT normalisation that made its gradients noisy is now fixed, so
   the rerun should also be cleaner. §13.12.
4. **Real IQA labels.** Every node-gauge result uses a synthetic teacher. They
   show *a* gauge can be imposed on frozen features, not that the *IQA* gauge
   lives there. Blocked on AIMAll: the pipeline was never run, and the
   per-structure cost and recovery error are still unmeasured (§8.0b).
5. **Why L_IQA helps `D_ij` on MACE-OFF24 but not on `mace-omol-0`** (§13.5).
   Same code, same design, opposite sign. The backbone we need is the one where
   it does not help.
6. **Measure arm A at 32k steps** to replace the estimate in §13.6. Two hours,
   will not change the conclusion.

## Recurring lesson

Four conclusions in this file were wrong at 8,000 steps and right at 32,000+:
sample size (§13.5), head capacity (§13.6), channel count (§13.11), and
probably the SAPT cost (§13.12). **Confirm a loss plateau before comparing two
configurations.** Each of those cost a full run.

## Environment

`miniforge3/envs/openmm_dev_ubio` — e3nn 0.4.4 (downgraded for MACE-POLAR-1), torch 2.12.1 (CUDA 13.0),
mace-torch 0.3.16, openmm 8.5.2, openmm-torch 1.5.1, openmm-ml 1.7, pymbar 4.0.3.
Backbones in `/home/ruigengji/MLP/mace/`. Current backbone: `MACE-POLAR-1-M`
(needs graph_electrostatics v0.4.0, see `run_polar.sh`). Earlier results used
`mace-omol-0-extra-large-4M` (51.3M parameters, 19456-dim node features, r_max 6.0, 82 elements, and it
needs per-graph `total_spin`/`total_charge` shaped `[n_graphs]`).

Two traps worth not rediscovering: never exclude self-pairs by distance
(`torch.cdist` switches algorithm above 25 rows and leaves the diagonal at
~1e-4, which silently NaNs everything), and cache node features in fp32, not
fp16 (the boundary sum accumulates fp16 rounding to 26% of the signal).

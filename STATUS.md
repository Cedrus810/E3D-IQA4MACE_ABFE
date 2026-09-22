# Status — 2026-09-23 (rev 3)

Snapshot of where the project stands. Numbers cite `RESEARCH_PLAN_v2.md §13`,
which holds the full experimental record; this file is the short version plus
the things that are half-done and easy to forget.

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

em_system.py          minimise a prepared leg with OpenMM's own MM force field
run_abfe.py           two-leg double decoupling driver
test_peel.py          peel the ligand atom by atom against a direct reference
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

1. **Implement the two-stage protocol** (above). Until then any free energy
   from this pipeline is wrong by the polarisation term. Changes land in
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

`miniforge3/envs/openmm_dev_ubio` — e3nn 0.5.1, torch 2.12.1 (CUDA 13.0),
mace-torch 0.3.16, openmm 8.5.2, openmm-torch 1.5.1, openmm-ml 1.7, pymbar 4.0.3.
Backbones in `/home/ruigengji/MLP/mace/`; `mace-omol-0-extra-large-4M` is the one
used (51.3M parameters, 19456-dim node features, r_max 6.0, 82 elements, and it
needs per-graph `total_spin`/`total_charge` shaped `[n_graphs]`).

Two traps worth not rediscovering: never exclude self-pairs by distance
(`torch.cdist` switches algorithm above 25 rows and leaves the diagonal at
~1e-4, which silently NaNs everything), and cache node features in fp32, not
fp16 (the boundary sum accumulates fp16 rounding to 26% of the signal).

# Status — 2026-10-15 (rev 10)

Snapshot of where the project stands. The design document is
`RESEARCH_PLAN_v3.md` (since 2026-10-10; decisions in its S12). Numbers cite
v3 S13.24 onward as **S13.x**, and the frozen v2 record
(`RESEARCH_PLAN_v2.md` S13.1-S13.23) as **v2-S13.x**. This file is the short
version plus the things that are half-done and easy to forget. Rev 8
(2026-09-28) was written against v2; its open items are resolved or replaced
below.

## Design in one paragraph (v3 S3)

The total comes from the frozen backbone, and the head only splits it:
`U = E_bb - sum_i (1 - lambda_i) c_i`, with `c_i` the head's per-ligand-atom
coupling (`sumD`, plus the explicit L/E Coulomb `cross` under D6). At
lambda = 0+ what is left of the coupling is the remainder `r = truth - sumD`,
bridged by a cut segment at lambda = 0. So `dG = sum_i dG_i + dG_cut` holds
exactly, and `dG_cut` is reported as one unattributed term. The endpoint is
held by an MM repulsive wall, followed by a refill (D14). Waters in the ML
region are identified with FIRES (D11).

## Decisions (v3 S12)

Decided: D1 anchored total · D2 drug set first · D3 both envs production ·
D4 v3 replaces v2 S0-S12 · D5 POLAR-1-M locked · D6 explicit L/E Coulomb
channel · D7 arm N (no synthetic `L_IQA`) · D11 FIRES · D14 MM wall + refill,
GCMC where exchange is slow.
D10 far field via MM per-atom lambda (2026-10-14) · D16 ML/MM + FIRES + wall
design (2026-10-14, v3 S6.3) · D17 bound the attributed coupling: cover
switch + backbone cap, m = 0.5 eV (2026-10-15, v3 S3).
D15: radius revised 2026-10-15 to the larger of the training-cluster
minimum contact and DES370K p1 per pair (the first radius, DES370K p1,
failed at lambda = 0+, S13.41). Open in D15: a second radius for Stage A's
sensitivity check, and the GCMC parts.
D8 Stage A = acetamide, solvent leg (2026-10-15).
D18 reference state of the decoupled end = bulk MM water: B -> B' leg (inner waters ML -> MM) minus a neat-solvent dG_ref (2026-10-16; dG_ref to confirm).
Open: D9 drug set · D12 one flagship paper · D13 mark v2
S0-S12 superseded in the file.

## Current conclusions

What stands, with the number that supports it:

1. **The anchored Hamiltonian is exact in code.** Endpoints match `E_bb`
   with zero error; the cut-step work equals `truth - sumD` to 3e-12 eV; and
   per-atom sums close. A two-segment HREMD smoke closes the bookkeeping
   (S13.29).
2. **The cut needs windows, not one BAR step.** The remainder spread is
   3-13 kT (S13.25). On thermal frames with the current head it is 5.8-6.5 kT
   at n = 20 and 7-10 kT at n = 50 (S13.37). At 1-2 kT per window, that is
   ~4-8 cut windows at n = 20.
3. **Dropping the synthetic `L_IQA` helps everywhere it was measured**:
   DES370K E -40%, F -31%, and the remainder mean is better at n >= 50
   (S13.26, D7).
4. **Part of r is the Coulomb term the head cannot see** (slope ~1 at
   n >= 10, S13.27). The explicit channel (D6) narrows the spread and
   improves DES370K `E_int` from 0.391 to 0.319 kcal/mol (S13.32).
5. **POLAR-1-M stays repulsive down to 0.7-0.9 A** (S13.30), so the wall
   radius comes from training coverage, not from a turnover point. The
   DES370K dimers reach deep, with O-H p1 at 1.32 A; the many-body cluster
   labels do not (S13.31). At 298 K, contacts are 0.2-0.5 A shorter than in
   minimised frames (S13.33).
6. **The head fits the ensemble its clusters come from**, measured three
   times:
   - Minimised-frame clusters leave a bias of -0.25 eV at n = 20 on thermal
     frames (S13.33).
   - Thermal clusters remove that bias but overbind minimised frames
     (S13.35).
   - λ = 1 clusters overbind at MM λ_elec 0.5 / 0 by +0.25 eV at n = 20
     (S13.36). Pooling states into the clusters removes that bias: +0.07 /
     +0.02 eV, within 2 SE (S13.37).
7. **Per-atom attribution machinery from v2 still stands.**
   - Graph masking is exact (v2-S13.7).
   - The diagonal path equals Shapley to 0.82% (v2-S13.8).
   - SAPT channels resolve pairs 6-16x better (v2-S13.10, 13.11).
   - Backbone-labelled clusters remove the per-edge bias of dimer-only
     heads (v2-S13.15, 13.16, 13.20).

8. **The ML/MM + FIRES + wall Hamiltonian runs end to end** at every
   path state (E0-E7, S13.38-S13.42). FIRES leaves the solvent unchanged
   outside a 0.2 A boundary band (E0). The NVE energy sd is at the pure-MM
   level, and no state shows a jump (E7). W1 = +0.006 kJ/mol by FEP. Two
   things were needed at lambda = 0+: wall radii from the training-cluster
   coverage and the D17 bound on c. Without them, the head's sum c ran away
   in the crowded shell (S13.41).

What is not known:

9. **The real intermediate-lambda ensemble of the anchored ML Hamiltonian.**
   Everything in item 6 at λ < 1 uses an MM proxy (λ_elec-scaled
   ligand-water electrostatics, LJ in full). Electrostatics and LJ are not
   split in the anchored U.
10. **Run-to-run training noise.** Every head was trained with seed 7
   (hard-coded in `test_joint.py`). So the λ = 1, n = 50 shift of the
   current head (-0.070 ± 0.014 eV paired, S13.37) cannot yet be assigned to
   pooling rather than retraining in general.
11. **No free energy with real sampling has been run**, and the ML/MM step
    takes 91 ms at n = 20 on the RTX 5090 (S13.45).
13. **Whether the full path is right, and what its end state is**
    (S13.47; review 2026-10-16). The Atenolol pilot gives +60 to +69
    kJ/mol, a diagnostic output: MBAR's error is NaN and there are two
    gaps. IBS's MM gives +152 [not verified, old run]. The decoupled end is
    not a defined reference state:
    - the 20 ML waters stay ML;
    - FIRES still orders them around the ghost ligand and pushes on it.

    So G_end holds an undefined solvent-reorganisation term (v3 S7.2).
    Order, one check at a time:
    1. D18, the reference state -- decided: bulk MM water. Add a B -> B' leg
       (inner waters ML -> MM at the decoupled end) and subtract a
       neat-solvent dG_ref (same n, FIRES around a dummy atom). Not in
       code yet.
    2. the all-MM path control with matched ends on acetamide (vs +34.04);
    3. the W2 / W3 window gaps.
12. **How much D17 reshapes c at lambda = 0+.** Forces are the energy
    gradient, and the 2 fs drift is integration error (S13.43). But the cap
    takes > 1 kT in ~half the q 2.0 samples, and 83-99% of samples have
    pairs inside the cover switch (S13.43). That part goes to dG_cut.
    Whether retraining on ML/MM lambda = 0+ frames shrinks it is not
    known (item 4).

## Operating head

`runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt`
(S13.37). Use it for n <= 50.

| | |
|---|---|
| Recipe | arm N + D6 (`E3D_ES=1`), 60k steps, hidden 256, POLAR-1-M frozen |
| Training data | 50,000 DES370K `organic` dimers + 10,000 clusters (≤ 24 waters) |
| Cluster frames | `experiments/2026-10-13_intermediate_lambda/lampool400_solvent.npz`, unminimised: 200 frames at λ = 1, 100 at λ_elec 0.5, 100 at λ_elec 0 |
| DES370K | E 1.44 meV/atom, F 21.4 meV/A, `E_int` 0.319 kcal/mol |

Remainder mean (eV) / sd/kT, 20 frames per test set:

| n | λ = 1 | λ_elec 0.5 | λ_elec 0 |
|---|---|---|---|
| 20 | -0.007 / 6.1 | +0.073 / 6.5 | +0.019 / 6.5 |
| 50 | -0.115 / 10.3 | +0.123 / 7.4 | +0.085 / 7.9 |

Alternative: the c20000 arm of the same run. It is better at n = 50 at
intermediate λ by 0.07 eV, but worse on DES370K `E_int` (0.341 kcal/mol).

Head history:

| Step | Head | Section |
|---|---|---|
| 1 | 20 minimised frames | S13.26 |
| 2 | + D6 | S13.32 |
| 3 | 200 thermal frames | S13.35 |
| 4 | + λ_elec 0.5 / 0 frames | S13.37 |

All of them are in `runs/polar-M/ckpt/`.

## Open items, in priority order

Route (user, 2026-10-15; v3 S8 top): (1) close the lambda = 0+ dynamics
question -- **done 2026-10-15, S13.43**, (2) `--mode mlmm` + RTX 5090
timing -- 5090 timed (S13.45), driver written (S13.46), pilot next, (3) Stage A on acetamide (leg built here, S13.44), (4) a few
neutral drugs (6 IBS solvent legs in `data/solvent_legs_ibs/`). Solvent legs
first. Before those IBS legs: `mlmm.py` drops ligand-ion LJ (its L-O LJ
group is ligand x outer waters only) and walls only waters; fix both first.

0. **D15 wall radius**: decided and revised (2026-10-15, v3 S12): per pair
   the larger of the training-cluster minimum contact and DES370K p1. With
   it, W1 = +0.006 kJ/mol by FEP (S13.42). A second radius for Stage A's
   sensitivity check is still open.
1. **MM wall (W1/W3) and the ML/MM system with FIRES (D11).** Designed in
   v3 S6.3 (D16, decided 2026-10-14). Code: `decomp/fires.py`, `decomp/mlmm.py`;
   checks E0-E6 pass (S13.38-S13.40; E0: boundary band of +0.5 waters at
   R_inner, otherwise unchanged). E7 found lambda = 0+ unstable (S13.41);
   fixed by revised wall radii + the S3 cover switch and backbone cap (D17).
   E7 rerun (S13.42): all 8 states stable, sd and step jumps at the
   pure-MM level; eps_w = 25 keeps lambda = 0+ contacts >= 0.92 r0; W1 =
   +0.006 kJ/mol. Open: lambda = 0+ (q 2.0 / 2.5) drifts +4-6 kJ/mol/ps,
   above the ~3 kJ/mol/ps run-to-run noise. Route step 1 (S13.43): paired
   2 / 1 fs runs and float64 force-vs-FD checks show integration error,
   not an implementation error. The cap and the cover switch act in most
   lambda = 0+ samples (unknown 12). The cap makes the 2080 Ti step 0.30 s
   (was 0.165 s). Today `run_abfe.py --mode anchored` runs the torch backend on the
   ML region alone (v3 S8.1); `--mode mlmm` is not written yet. For the complex-leg W3 and the GCMC parts of
   D15, the ABFE_IBS reference is
   `REF_pocket_water_from_ABFE_IBS_2026-10-14.md`. It is reference only, and
   its numbers have not been re-measured here (see `CLAUDE.md`).
2. **`run_abfe.py --mode mlmm`**: written and checked (S13.46). It
   gives one ML forward per configuration for the whole u_kn row, per-atom
   TI that closes to the total, and checkpoint/resume. RTX 5090 step: 91 /
   119 / 200 ms at n = 20 / 50 / 100 (S13.45). Pilot ladder on the 5090
   (S13.47; 26 states, 10 ps each, 4 h): overlap 0.13-0.33 everywhere
   except two gaps, lambda 0.09 -> 0 (0.02) and s_w 0.2 -> 0 in W3 (0.00).
   The cut is about a third of the total. Next: pack states into the two
   gaps (W3 maybe also a softer softcore), then set the budget. Before
   that or alongside: the path-only control (all-MM path on acetamide vs
   its MM +34.04 kJ/mol, v3 S7.2), which tests the cycle's implementation
   independently of the head. Batching replicas into one ML call is not
   done.
3. **Stage A on acetamide** (D8, 2026-10-15). The leg is built
   (`experiments/2026-10-15_acetamide_leg/`, GAFF2/TIP3P, 2,634 atoms) and
   its MM dG_decouple is +34.04 +- 0.14 kJ/mol (3 reps, S13.44). Next:
   its own clusters, head and coverage radii. It should measure:
   - softcore need;
   - cut-step overlap in the real ensemble;
   - λ spacing;
   - agreement with the message-masking control;
   - attribution against Shapley.
4. **Replace the MM-proxy intermediate frames** in the clusters with frames
   from the anchored ML/MM windows once item 1 exists. Then rescan.
5. **Seed replicates.** This needs an `E3D_SEED` knob in `test_joint.py`
   (seed 7 and batch rng 0 are hard-coded). Without it, no head-vs-head
   difference can be compared against training noise.
6. **Drug set** (D9). Charged drugs need DES370K `full` (3,691 systems, not
   yet run). Each drug needs its own solvent-leg frames at all three states
   (v3 S5.2). Candidates (2026-10-15, user-supplied, not yet chosen): six
   neutral HCNO legs from the fragment-opt ABFE-IBS runs, with 3-rep MM
   dG_solv, in `data/solvent_legs_ibs/`. See
   `REF_solvent_legs_from_ABFE_IBS_2026-10-15.md`, and read its S4 before
   loading them (resname `LIG`, Na/Cl in the box, 300 K, HMR).
7. **IQA as a validation set only** (D7). It is gated on AIMAll cost and
   recovery error, which have never been measured (v2 S8.0b).

## What exists in code

```
decomp/
  head.py             E_intra per node, D_ij per edge (4 SAPT channels); backbone-agnostic
  mace_adapter.py     head on a frozen MACE backbone
  polar_es.py         explicit L/E Coulomb channel from POLAR-1's densities, per-atom Euler split (D6)
  lambda_mask.py      "edge" / "graph" / body-order coupling; `anchored_energy` (v3 S3)
  losses.py           E, F, IQA, L_int, SAPT (dataset-level scaling)
  data.py             DES370K batching; MACE input dicts
  train.py            training step
  fe.py               TI, per-atom TI, MBAR, overlap  (pymbar shims from ABFE_IBS)
  sampling.py         torch Langevin (small systems)
  openmm_bridge.py    PythonForce bridge; loads prepared ABFE_IBS legs; carves an ML region
  hremd.py            replica exchange over lambda; torch or openmm backend (dU/dlambda units fixed, S13.29)
  clusters.py         ligand + n-water clusters from a frame stack, coupling labelled by the frozen backbone (fp64)
  fires.py            FIRES (D11): softmin ordering water O x ligand heavy atoms, pairwise C1 restraint (v3 S6.3.2)
  mlmm_hremd.py       HREMD over the S6.3.4 path (`run_abfe.py --mode mlmm`): all-state energies from one ML
                      forward, per-segment dU/dq, per-atom W1/W2/W3 terms, checkpoint/resume, TI + MBAR
  mlmm.py             build_mlmm_system: ML on L u I + FIRES (one PythonForce), MM far field with per-atom lambda,
                      L-O LJ, WCA wall (revised D15 radii), cover switch + cap on (D17); set_path(q) over W1/W2/cut/W3

em_system.py          MM-minimise a prepared leg; `frames` = minimised MD frames
run_abfe.py           two-leg driver; `--mode anchored` = continuous + cut segment on one path p in [0, 2]
test_anchored.py      anchored Hamiltonian's exact checks 1-6 (S13.29)
test_scale.py         remainder r = truth - sumD(+cross) at n_solvent = 5..100 over a frame stack
test_joint.py         head training (E3D_CLUSTERS, E3D_ES, ...); seed hard-coded 7
run_polar.sh          training driver; knobs CLUSTERS, FRAMES, CLUSTER_MAX, TAGS, ARMS
run_remainder.sh      S13.25 remainder spread
learn_{d7,d6,thermal,lampool}.zsh   PBS jobs behind S13.26 / S13.32 / S13.35 / S13.37

experiments/2026-10-12_thermal_frames/thermal_frames.py     unminimised MM frames + contact minima
experiments/2026-10-13_intermediate_lambda/lambda_frames.py MM frames at lambda_elec (scaled_system)
experiments/2026-10-14_lampool_result/paired.py             paired per-frame head comparison
experiments/2026-10-14_fires_mm/        E4 (check_fires.py), E0 (fires_mm.py, analyze_e0.py)
experiments/2026-10-14_mlmm_checks/     E1, E2, E3, E5, E6 (check_mlmm.py)
experiments/2026-10-15_e7_nve/          E7 (e7_nve.py, run_e7.sh, e7_summary.py); lambda = 0+ diagnosis and D17 checks
experiments/2026-10-15_remainder_scan/  single-water remainder scans; training-cluster contact coverage
```

**Minimise with MM, score with ML.** Minimising a carved cluster with "the
learned potential" once ran the energy to -5e5 eV; which model it was is not
recorded. POLAR-1-M itself stays repulsive down to sub-angstrom contacts
(S13.30). The MM force field has hard cores, and OpenMM's minimiser is well
tested. All MD frames are MM.

**MM λ_elec states.** Use `lambda_frames.scaled_system`: ligand charges
times λ, plus an exception carrying the original q_i q_j for every
ligand-ligand pair. It matches the reference at λ = 1 to 0.03 kJ/mol.
openmmtools' "direct-space" decoupling is 33 ± 13 kJ/mol off already at
λ = 1, and its "exact" PME mode refuses decoupled electrostatics (S13.36).

Borrowed from `/home/ruigengji/ABFE_IBS`, not rewritten:
- the pymbar shims (`fe.py`);
- the prepared Atenolol legs (`output/system_solvent.xml` +
  `topology_solvent.cif`, 4,076 atoms).

Restraints and standard-state corrections should come from there when the
complex leg is run.

## Scale

Full-ML condensed phase is out of reach: about 55 GiB and 11 days per λ
window for the 4,076-atom box (v2-S13.13). ML/MM over the ligand plus near
water is the only route.

- **ML force call** on a 2080 Ti: 121-331 ms (S13.24). That gives 40-110 h
  per 12-window solvent leg at 100 ps, before MM, W1/W3 and cut windows.
- **Full ML/MM step** with the D17 cap, RTX 5090: 91 / 119 / 200 ms at
  n = 20 / 50 / 100 (S13.45); 2080 Ti n = 20: 0.30-0.32 s. At n = 20 on the
  5090, a 100 ps window is ~1.3 GPU-h at 2 fs.
- **Head training** (60k steps): 0.21 s/step on an RTX 4080, 0.25-0.26 on an
  RTX 3090. Labels plus targets for 10,000 clusters take 31 min on the 4080
  and 40 min on the 3090 (S13.35, S13.37).

## Recurring lessons

- **The head fits the ensemble its clusters come from** (item 6). Any new
  ensemble the sampling visits must be checked with a remainder scan before
  the head is trusted there.
- **Confirm a loss plateau before comparing two configurations.** Four v2
  conclusions were wrong at 8,000 steps and right at 32,000+ (v2-S13.5, 13.6,
  13.11, 13.12).
- **c is free; bound it where the head is out of range.** The head's
  coupling overshot in an ensemble it had not seen (the crowded lambda = 0+
  shell) and drove a runaway. Two per-pair patches failed before D17
  (S13.41). Any new ensemble needs a short NVE run, not only a static scan.
- **Compare heads frame by frame.** On the same frames and carves, paired
  differences have 4-7x smaller SE than the difference of two means (S13.37).

## Environment

`miniforge3/envs/openmm_dev` (since 2026-10-10; production, do not modify):
e3nn 0.4.4, torch 2.12.0, mace-torch 0.3.16, graph_longrange 0.4.0,
openmm 8.5.2, openmm-ml 1.6, pymbar 4.2.0, grand 1.1.0 (GCMC), openmmtools.

Results up to rev 8 were produced in `openmm_dev_ubio` (also production;
e3nn 0.5.1, torch 2.12.1, pymbar 4.0.3). Both envs give identical POLAR-1-M
and head outputs (`experiments/2026-10-10_env_smoke/`).

Backbones are in `/home/ruigengji/MLP/mace/`. The backbone is
`MACE-POLAR-1-M`, locked (D5).

Production GPU is an RTX 5090. The 2080 Ti workstations are for smoke
runs only.

Traps worth not rediscovering:
- **Self-pairs.** Never exclude them by distance: `torch.cdist` switches
  algorithm above 25 rows and leaves the diagonal at ~1e-4, which silently
  NaNs everything.
- **Cached node features.** Store them in fp32, not fp16: the boundary sum
  accumulates fp16 rounding to 26% of the signal.
- **Files written on another node** can look stale or empty through this
  NFS mount for minutes. Read them on that node via ssh.
- **Logs on a live remote run.** Check the process on the node before
  calling a run dead. A log on another node can look stalled during
  CPU-only loading (467 s).
- **MD temperature.** Count degrees of freedom as 3N − constraints − 3.
  Leaving out the constraints prints ~199 K for a 298 K run.

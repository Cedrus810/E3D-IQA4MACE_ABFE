# Atom-Resolved Solvation and Binding Free Energies with an Exact Total

**Version 3.0, 2026-10-10.** Supersedes S0-S12 of `RESEARCH_PLAN_v2.md`.
v2 S13 (S13.1-S13.23) stays where it is as the frozen measurement record and is
cited here as **v2-S13.x**. New measurements are appended to S13 of this file,
numbered from S13.24 on.

Items marked **[decided]** were settled by the user; items marked **[open]**
carry a recommendation and wait for a decision. The full list is S12.

---

## 0. What changed from v2, and why

v2 was written on 2026-09-20, before the condensed-phase work. Its S0-S12 assume
the head's own total energy defines the alchemical Hamiltonian, that IQA labels
fix the gauge, and that validation starts on dimers. The measurements since
then (v2-S13.13-S13.23) overturned all three.

| v2 | Measured | v3 |
|---|---|---|
| `U = sum E_intra + D_LL + D_EE + sum lambda_i D_LE`, total from the head (v2 S5.1) | The head's total cannot be consistent between a 341-atom carve and its parts to the ~0.1 meV/atom a 1 kcal/mol coupling needs; its own fit is 3-4 meV/atom. Every cluster recipe moves the error rather than removing it (v2-S13.20, S13.22) | **Total from the backbone, head only splits it** (S3). [decided] |
| `sum_i dG_i = dG_decouple` exactly (v2 S6.1) | Not reachable with a learned total; the graph cut at lambda = 0 carries a jump TI cannot see (v2-S13.14) | `sum_i dG_i + dG_cut = dG_decouple` exactly; `dG_cut` reported as one global term (S4) |
| Graph masking for every lambda (v2 S5.2) | Needed only because the head defined the total | Full graph for lambda > 0, one cut at lambda = 0 (S3) |
| OMol25 E/F supervision (v2 S3.1) | Not used; DES370K plus clusters labelled by the frozen backbone (v2-S13.16) | S5 |
| Two gauge constraints, IQA + L_int (v2 S4.3) | SAPT components are what resolve the pair distribution, 6-16x (v2-S13.10, S13.11); IQA was never run, only a synthetic teacher | L_int + SAPT + cluster coupling; IQA off the critical path (S5.3). [decided] |
| Allegro / NequIP / MACE; condensed phase on MACE-OFF24 (v2 S4.1, S11 R6) | `mace-omol-0` sign-flips at scale; POLAR-1-L is worse and 2.2x the cost (v2-S13.15, S13.17) | MACE-POLAR-1-M, frozen (S2.1). [decided] |
| Long range: "decide before Stage B" (v2 S11 R5) | POLAR carries explicit electrostatics; 9.9% of the Atenolol coupling (v2-S13.15) | Explicit cross-Coulomb as its own pair channel (S2.3). [decided] |
| Stage A = dimers (v2 S8.2) | Work is already at Atenolol + 100 waters | Stage A = small solute with real sampling, Stage B = a drug set (S8). [decided: drug set] |
| Phase 0 = AIMAll (v2 S8.0) | Never run; WP1-7 went ahead without it | Stage 0 = the two checks that gate S3 (S8.0) |
| Three papers (v2 S10) | Paper II rested on `mace-omol-0` | One flagship paper (S10). [open] |

What carries over unchanged: the decomposition head (v2 S4.1), loss
normalisation by target scale (v2 S4.4), the diagonal path as the Aumann-Shapley
attribution (v2 S6.1-6.2, measured v2-S13.8), the message-masking control
(v2 S7.1), and ML/MM as the only reachable condensed-phase route (v2-S13.13).

---

## 1. Core claim

A frozen pretrained potential gives the total energy. A decomposition head on
its features gives pair energies `D_ia` between ligand atom i and environment
atom a, pinned to CCSD(T) interaction energies, SAPT components and the
backbone's own cluster couplings. Using the backbone for the total and the head
only to split it:

    dG_decouple = sum_{i in L} dG_i  +  dG_cut          exactly

- `dG_i` is atom i's share of the pair-resolved coupling, from one diagonal-path
  TI: order-independent (Aumann-Shapley), one ordinary alchemical calculation.
- `dG_cut` is the part of the coupling the pairs do not carry -- polarisation
  and charge response, many-body terms, whatever the head misses -- obtained in
  one discrete step. It is a physical observable of the backbone, reported, not
  hidden.
- The total is the backbone's own ABFE whatever the head's quality, because
  both endpoints are the backbone's energies. Head quality decides only how much
  lands in `dG_cut`.

The figure of merit for the head is therefore the unattributed share
`|dG_cut| / |dG_decouple|`, not its total-energy error.

### 1.1 Objectives and their knobs (user, 2026-10-11)

The project has several objectives, and the measurements so far say they are
largely decoupled, so each is worked on with its own knob rather than by a joint
(Pareto) search:

| Objective | Knob | Evidence that it is separate |
|---|---|---|
| Total dG | backbone (fixed, S2.1) | by construction (S3): independent of the head |
| Remainder mean (unattributed share) | cluster recipe; operating size n | moves with recipe (v2-S13.22, S13.25) |
| Remainder spread (sets the cut windows) | physics terms (S2.3); cut windows (S3) | identical for three heads (S13.25, S13.26); tracks the cross Coulomb term (S13.27) |
| Physical meaning of the per-atom split | SAPT supervision; validation (S7.3) | -- |
| Order independence | diagonal path | arithmetic (v2-S13.8) |
| Cost | n; step size | S13.24 |

Still coupled, through the head's shared capacity: remainder mean across sizes
and DES370K accuracy (v2-S13.22), and possibly SAPT vs energy accuracy
(v2-S13.12, measured on `mace-omol-0` only). Caveat: "decoupled" rests on three
heads, one seed each, within these recipes.

---

## 2. Model

### 2.1 Backbone: MACE-POLAR-1-M, frozen  [decided]

Measured against the two alternatives tried (v2-S13.15, S13.17):

| | at scale (Atenolol, 5-100 waters) | DES370K `E_int` | cost |
|---|---|---|---|
| `mace-omol-0` | coupling sign-flipped at 4 of 5 sizes; non-periodic training | -- | 19456-dim features |
| POLAR-1-L | sumD worse at every size held out | 0.537 kcal/mol | 2.2x M; 341 atoms do not fit 11 GiB |
| **POLAR-1-M** | linear, correctable per-edge bias | **0.428** | baseline |

For a drug set it also has what the others lack: periodic training, a total
charge input and explicit long-range electrostatics, which charged drugs need.

Locked by the user on 2026-10-10. Not measured: POLAR-1-M against MACE-OFF24
on the condensed-phase scan, and POLAR-1-M's ML/MM step time (S6.3).
Reopening the backbone later costs a full retrain of the head.

Environment: `miniforge3/envs/openmm_dev` (user, 2026-10-10; `openmm_dev_ubio`
runs the same code identically). Both are **production** -- see `CLAUDE.md`. Backbones in `/home/ruigengji/MLP/mace/`.

### 2.2 Head

`decomp/head.py` unchanged: `E_intra` per node, four pair channels per edge
(SAPT es / ex / ind / disp), permutation-symmetric, 0e readouts, hidden 256, on
frozen POLAR-1-M node features via `decomp/mace_adapter.py`.

Training recipe [decided, D7, 2026-10-12]: **arm N** -- arm D without the
synthetic-teacher `L_IQA` -- 60k steps, <= 24 waters, with the explicit L/E
Coulomb channel (D6, `E3D_ES=1`). Clusters (S5.2): 10,000, cut from
unminimised MD frames pooled over alchemical states. Current head (S13.37):
`runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt`
(DES370K E 1.44 meV/atom, `E_int` 0.319 kcal/mol; remainder mean within
2 SE of zero at n <= 20 at lambda = 1 and on the MM lambda_elec 0.5 / 0
proxy; -0.115 eV at n = 50, lambda = 1). For n <= 50.
Head history: 20 minimised frames (S13.26) -> + D6 (S13.32) -> 200 thermal
frames (S13.35) -> + lambda_elec 0.5 / 0 frames (S13.37). Each step fixed the
bias on the ensemble it added and nothing else: the head fits the ensemble its
clusters come from.

Under S3 the head's `E_intra` and its total enter no free energy: `U` uses only
`E_bb` and `D_ia`. They stay in training as regularisers through `L_E + L_F`.

### 2.3 Long-range electrostatics: an explicit pair channel  [decided, D6, 2026-10-12]

POLAR-1's electrostatic energy (`mace/modules/extensions.py`, `PolarMACE`;
`graph_longrange.energy.GTOElectrostaticEnergy`) is the Coulomb energy of
Gaussian-smeared charge densities after the field-update iterations -- a
bilinear form in the final densities `rho`:

    E_es = 1/2 sum_{i,j} rho_i K_ij rho_j

So the ligand-environment part `E_es^LE = sum_{i in L} sum_{a in E} rho_i K_ia rho_a`
is exactly pair-resolved for a given set of densities. What is not pairwise is
how `rho` itself responds to the environment.

Two options, both with an exact total under S3:

- **Pair channel (chosen).** Add `E_es^LE` per pair as a fifth, physics-
  defined channel of `D_ia`; the head's learned channels then fit the targets
  minus it. The long-range direct electrostatics becomes attributable per atom;
  only the charge response lands in `dG_cut`.
- **Cut step.** Leave it to the remainder. No new code, but the most
  chemically interesting part for charged drugs becomes unattributable.

Cost of the pair channel: the head must be retrained, because today's targets
already contain `E_es^LE` and adding it explicitly would double-count.

Evidence (S13.27): at n >= 10 the remainder tracks `E_es^LE` with slope ~1 --
the head carries none of its configuration dependence -- and removing it cuts
the spread most at large n (11.4 -> 7.1 kT at n = 50 with the full
electrostatic part). It does not remove the need for cut windows.

**Implementation route.**

- Total: `E_es(rho_L + rho_E) - E_es(rho_L) - E_es(rho_E)` on the full-system
  densities, by calling `PolarMACE.coulomb_energy` with masked `source_feats`
  (done in `experiments/2026-10-12_r_vs_es/r_vs_es.py`). No pair matrix needed.
- Per ligand atom: the cross term is linear in `rho_L`, so by Euler's theorem
  `E_es,i^LE = rho_i . dE_cross/drho_i` (summed over the density components) --
  one autograd call gives the exact per-atom split for the diagonal path.
- In the anchored U: `D_ia` -> learned channels + `E_es^LE` per ligand atom,
  both scaled by `(1 - lambda_i)`; densities from the full system at R.

**Retraining targets** (to fix in code, not here): subtract the explicit
Coulomb from every target that contains it -- dimer `L_int`, cluster
coupling, the SAPT `es` channel, and the head's total (`E_bb` minus the
pairwise Coulomb). With pbc all-False PolarMACE takes the real-space
evaluator (`graph_longrange.energy.GTOElectrostaticEnergy.forward`), so no
k-space term enters a non-periodic carve.

**Implemented 2026-10-12** (S13.28): `decomp/polar_es.py` (`coulomb_cross`,
`es_coupling`); `E3D_ES=1` in `test_joint.py` (targets: E minus the full
`E_es`, F minus its forces, coupling and SAPT es minus the cross term; metrics
reported with it added back) and in `test_scale.py` (scoring). Retrain:
`learn_d6.zsh` (PBS, node P1).

---

## 3. Alchemical Hamiltonian (anchored)

    U(R, lambda) = E_bb(R) - sum_{i in L} (1 - lambda_i) sum_{a in E} D_ia(R)      lambda > 0, full graph
    U(R, 0)      = E_bb(R_L) + E_bb(R_E)                                            fragments alone

- lambda = 1: `U = E_bb` exactly.
- lambda -> 0+: `U = E_bb - sum D_LE = E_bb(L) + E_bb(E) + r`, with the remainder
  `r(R) = truth - sum D_LE`, `truth = E_bb(all) - E_bb(L) - E_bb(E)`.
- lambda = 0: fragments alone, exactly.

`D_ia` does not depend on lambda (the graph is complete for every lambda > 0),
so `U` is linear in each `lambda_i` and

    dU/dlambda_i = sum_{a in E} D_ia          exactly, no message-masking term

Cost: `E_bb` and `D_ia` come from the same backbone pass; the lambda = 0
fragments are what `isolated_energy` already computes. Forces need `dD/dR`
through the backbone features, i.e. one backbone forward + backward per step.

**Two stages.**

    dG_decouple = - integral_0^1 < sum_{i,a} D_ia >_t dt     TI, diagonal path    -> sum_i dG_i
                  + dG_cut                                  lambda = 0+ -> 0     -> BAR

The cut step's energy difference is `U(0) - U(0+) = -r(R)`. What decides
whether one BAR step bridges it is the spread of `r` over configurations, not
its mean (v2-S13.23 risk). If it does not, intermediate windows on the cut
segment: `U_s = (1 - s) U(0+) + s U(0)`, s in [0, 1].

**Required exact checks** (replace v2 S8.3):

1. `U(R, 1) = E_bb(R)` to fp precision.
2. `U(R, 0) = E_bb(R_L) + E_bb(R_E)` to fp precision.
3. `sum_i dU/dlambda_i = sum D_LE` to fp precision; autograd matches finite
   differences.
4. `U(0+) - U(0) = r` reproduces `test_scale.py`'s `truth - sumD` on the same
   carve.
5. Forces finite and smooth on lambda in (0, 1].

**Endpoint: an MM repulsive wall, then refill** [decided, user 2026-10-10;
details open]. Without one, waters move into the ligand as lambda -> 0 and the
backbone and head are evaluated far outside their domain (S11 R1). Three
segments, total still exact:

    W1  wall on     lambda = 1, add an MM softcore repulsive wall between
                    ligand atoms and every water                         MM
    W2  decouple    lambda 1 -> 0 as above, wall held fixed              ML (TI + cut step)
    W3  refill      remove the wall, water fills the cavity              MM; GCMC where
                                                                         exchange is slow

- W2 is where the attribution lives; with the wall on, every configuration it
  samples -- including the lambda = 0 ensemble the cut-step BAR is evaluated
  on -- keeps waters outside the wall.
- W3: in the solvent leg water floods the cavity in ps by plain MD, so ordinary
  MM softcore removal of a purely repulsive wall is enough (`ABFE_IBS` has
  it). GCMC is what the complex leg needs, where a buried pocket refills
  slowly.
- W1 and W3 are MM pair terms, so they are per-atom attributable too (a
  per-atom cavity term).

Open within this decision:

- **Wall radius**, per element pair, from two measurements (method agreed
  2026-10-12): where the MLP itself turns non-physical at short range (a
  radial scan of the backbone coupling: inside its maximum the learned
  "wall" gives way), and the shortest ligand-water contact the training data
  covers (DES370K dimers + clusters). The wall excludes both.
- **The attribution now depends on the wall.** The total does not; the per-atom
  split of W2 vs W1+W3 shifts with the wall radius. State it; measure the
  sensitivity in Stage A with two radii.
- **GCMC-inserted waters: ML or MM**, and how that interacts with the ML-region
  scheme (S6.3).
- **GCMC code**: `grand` 1.1.0 is in `openmm_dev` (production, used as is);
  not in `openmm_dev_ubio`. Both run POLAR-1-M + the head identically
  (smoke, workstation A, 2026-10-10). The pocket-water cycle already designed in
  `ABFE_IBS/ABFE_IBS/PLAN_pocket_water_sampler_redesign.md` (fixed-N ABFE +
  kT ln P(N)) is the candidate for the complex leg's W3.

---

## 4. Attribution

Diagonal path `lambda(t) = (t, ..., t)`:

    dG_i = - integral_0^1 < sum_{a in E} D_ia >_t dt
    sum_{i in L} dG_i + dG_cut = dG_decouple

Per environment atom or residue, sum over `i in L` instead; the `<D_ia>` matrix
is the interaction free-energy map (v2 S6.3). Order independence is exact for
the attributed part (Aumann-Shapley, v2 S6.2; measured to 0.82%, v2-S13.8).
`dG_cut` is one global number; it is not spread over atoms.

Scope (v2 S6.4 still holds): restraint and standard-state terms of a binding
calculation are global and reported separately.

**What the attribution means physically.** `L_int` pins the boundary sum; SAPT
pins four channel sums per dimer. Below that, the split of a channel over
individual atom pairs is set by the head's architecture, not by a label -- the
pair non-identifiability of v2-S13.2/S13.3, reduced but not removed by SAPT
(v2-S13.10: 25% cross-boundary recovery error on a toy teacher). The per-atom
values are well defined and order-independent; how far they are *the* physical
split is a validation question (S7.3), not a given.

---

## 5. Data and supervision

### 5.1 Dimers

DES370K `organic` split by system (v2-S13.5): `L_int` against CCSD(T)/CBS,
`L_SAPT` against es / ex / ind / disp, operating point `sapt+total`
(v2-S13.11). Charged pairs (`full`, 3,691 systems) are not yet run; the drug
set needs them if it includes charged drugs (S11 R5).

### 5.2 Backbone-labelled clusters

`decomp/clusters.py`: ligand + 2-24 water clusters cut from MM MD frames,
coupling labelled by the frozen backbone, `E_couple = E(A u B) - E(A) - E(B)`,
no QM (v2-S13.16, S13.20). Frame diversity is what fixed sumD (v2-S13.20).
Each cluster draws its frame uniformly from the stack.

Frames, current recipe (S13.33-S13.37): **unminimised** (minimised frames
miss the 298 K contacts, 0.2-0.5 A shorter; S13.33), **200 frames** (20
barely widen the contact coverage, since all clusters from one frame share
its nearest waters; S13.34), and **pooled over alchemical states**: 200 at
lambda = 1 + 100 each at MM lambda_elec 0.5 and 0 (`lampool400_solvent.npz`,
`experiments/2026-10-13_intermediate_lambda/`; S13.36, S13.37). The
intermediate states are an MM proxy; once the anchored ML/MM sampling exists
(S8.1), frames from its own intermediate windows should replace them.

**For a drug set** the clusters must come from every drug's own solvent leg,
pooled into one head: today's head has seen Atenolol only. Per drug that is one
MM trajectory per state (lambda = 1, 0.5, 0; 200 frames each on the current
recipe) and one labelling pass. Measured off the 5090: MD 217-313 s per
200-frame state on workstation A (2080 Ti); labels 642 s and targets 1,215 s for
10,000 clusters on an RTX 4080 (S13.37). (20 minimised frames on the 5090:
frames 96 s, labels 5 min, targets 8 min, v2-S13.20.) One pooled retrain of the head.

### 5.3 IQA: off the critical path  [decided, D7, 2026-10-12]

IQA is a real-space partition of the molecular energy into intra-atomic and
interatomic terms. Its role in v2 was to fix the **node gauge** -- how energy
is split between `E_intra` and `D_ij` -- not long-range electrostatics. POLAR's
electrostatics are a separate thing: they get the total right at long range but
fix no split.

Under S3 the node gauge enters no free energy. What IQA could still supply is
the one thing nothing else does: per-atom-pair interatomic energies
`E_ia^inter` across a fragment boundary, a physical reference for the
sub-channel split of S4. That makes it a **validation set**, not a training
term:

- Drop the synthetic-teacher `L_IQA` from the head's training recipe (it pins a
  random gauge and costs capacity; v2-S13.12 measured +87% on the node term).
  Ablation, not assumption: retrain arm D without it and compare remainder and
  DES370K.
- Keep IQA as a validation set of a few hundred dimers, gated on v2 S8.0b
  (AIMAll cost and recovery error, never measured). If 0b fails, the
  attribution is validated chemically and against MM (S7.3) instead.

---

## 6. Deployment in the condensed phase

### 6.1 ML/MM

Full ML is out of reach (v2-S13.13: ~55 GiB, 11 days per window). The ML region
is the ligand plus its nearest waters; the rest is MM.
`decomp/openmm_bridge.py` already loads prepared legs (`from_abfe_output`), carves
whole residues under minimum image and strips the MM ligand-environment
nonbonded terms the ML region replaces.

### 6.2 Far field: also attributable  [open]

The ligand's MM nonbonded interactions with waters outside the ML region are
pairwise by construction. Scale them with the same per-atom `lambda_i` (MM
softcore as in ordinary ABFE) and their `dU/dlambda_i` is per-atom too, so the
diagonal path covers the far field as well. This removes v2-S13.13's
"near-field only" limitation, at the cost of mixing ML and MM physics in one
map; the two parts are reported separately and summed.

### 6.3 Not yet designed

- **Which waters are ML.** The nearest-n set changes as waters diffuse; a
  Hamiltonian that switches waters between ML and MM mid-run is discontinuous.
  **Decided (D11, 2026-10-12): FIRES first** (Rowley & Roux, JCTC 2012 -- not
  re-read here): the n ML waters keep their identity, and a boundary set by
  the farthest inner water keeps inner and outer waters from crossing, so the
  Hamiltonian stays continuous and the ML waters cannot diffuse away.
- **Region size.** Under S3 the total is exact at any n; what grows with n is
  the remainder (v2-S13.23: 9-20% at n = 100) and the cost. Choose n from the
  measured remainder.
- **Step time on POLAR-1-M.** v2-S13.13's ~200 ms/step and 66 h per 12-window
  leg were measured on MACE-OFF24, not POLAR-1-M. Measure it (S8.0).

---

## 7. Controls and validation

### 7.1 Message-masking baseline

Scale messages across the L/E boundary of the backbone itself (v2 S7.1). Same
endpoints as S3, so the same total dG; no attribution, and a path through
out-of-distribution features. The decomposition must match its total and should
converge faster with lower variance. If it does not converge faster, the claim
narrows to attribution only (S9).

### 7.2 Totals

- Solvent-leg dG against the MM ABFE of the same prepared legs
  (`/home/ruigengji/ABFE_IBS`). A comparison, not a pass/fail: the two are
  different Hamiltonians.
- Hydration free energies against experiment, for drugs where it exists.

### 7.3 Attribution

- Exhaustive or Monte-Carlo Shapley on a small system (v2-S13.8, now with
  sampling).
- Chemical sense across the drug set: the same functional group should carry
  similar per-atom values across drugs; polar groups should dominate the
  hydration map.
- IQA `E_ia^inter` on dimers, if S5.3's gate passes.
- Against MM per-group decoupling on the same legs: agreement is not expected,
  disagreement must be explainable.

---

## 8. Staging

### 8.0 Stage 0 -- before writing the anchored code

Both cheap; both decide whether S3 is workable as written.

1. **Remainder spread.** `run_remainder.sh` -- **done 2026-10-10, S13.25: sd 3-13 kT,
   cut windows needed beyond n = 5**:
   `r = truth - sumD` over configurations of a fresh MM trajectory, for both
   20-frame heads, sd/kT per size. Few kT -> one BAR step; many kT -> cut
   windows (S3). Static configurations are a proxy for the lambda = 0+ ensemble,
   not the ensemble itself.
2. **POLAR-1-M ML/MM step time** -- ML part measured, S13.24 (121-331 ms);
   full ML/MM step still to measure: a few hundred steps on the Atenolol leg at
   n = 20 / 50 / 100. Sets the cost of everything after.

### 8.1 Code (once S3, S2.1 and S2.3 are decided)

- `lambda_mask.py`: `anchored` mode, checks 1-5 of S3. **Done 2026-10-12**
  (`anchored_energy`; `test_anchored.py`, S13.29).
- `run_abfe.py`: TI over lambda > 0 on the diagonal path, BAR (or cut windows)
  at lambda = 0, per-atom accumulation. **Done 2026-10-12** (`--mode anchored`,
  one path coordinate p in [0, 2] over both segments, `AnchoredForce`; S13.29).
- Not yet: the MM wall (W1/W3, needs D15) and the ML/MM system with the
  ML-water scheme (D11); both need OpenMM, `run_abfe.py` runs the torch
  backend on the ML region alone today.
- S2.3 (decided): explicit `E_es^LE` channel with the per-atom Euler split,
  target subtraction in `test_joint.py`, then one retrain on the arm-N recipe.
  **Done** (S13.28, retrain S13.32); the head has since been retrained twice on
  new frames (S13.35, S13.37).

### 8.2 Stage A -- method validation with real sampling

A small solute in ML/MM water, sampling in minutes to hours. Answers what no
static scan can: softcore need (S11 R1), cut-step overlap, ML-water identity
(S6.3), lambda spacing, convergence against the message-masking control (S7.1),
attribution against Shapley (S7.3). System: [open] -- recommend a drug fragment
already in DES370K's chemistry (e.g. acetamide or phenol), so the head is in
domain.

The drug set is the first *real* system [decided]; Stage A is the protocol test
in front of it, so that 66-h legs do not run on an unvalidated protocol.

### 8.3 Stage B -- a drug set, solvent legs

Per drug: prepared solvent leg, clusters (S5.2), anchored decoupling, per-atom
hydration map. One pooled head for the set. Totals against MM and experiment
(S7.2), attribution against chemistry (S7.3).

Selection [open]: how many drugs; neutral only or with charged ones (needs
DES370K `full`, S5.1); hydration data available; whether they share one target
so Stage C can follow. Only Atenolol is prepared today
(`ABFE_IBS/Atenolol-rank11/output/`).

Cost: ML force call on POLAR-1-M 121 / 197 / 331 ms at n = 20 / 50 / 100
(S13.24, lower bound on the step), i.e. 40-110 h per 12-window solvent leg at
100 ps on a 2080 Ti; ten drugs at n = 50 is ~660 GPU-h before W1/W3 and cut
windows. Production is an RTX 5090, not yet timed; replace these with
the 5090 measurement.

### 8.4 Stage C -- complex legs, binding

Same protocol on the complex leg; restraints and standard-state corrections from
`ABFE_IBS`. Per-residue maps of binding free energy.

---

## 9. Go / no-go

Gates name what is measured and what decides it. Numerical thresholds are set
from the noise measured in the step before, not here.

| Gate | Go if | Kill / pivot if |
|---|---|---|
| Stage 0.1 | remainder spread small enough for BAR, or for a few cut windows | spread needs many windows -> the cut segment is itself an alchemical leg; cost rises, claim stands |
| Stage A, exactness | checks 1-5 of S3 pass on the real system | -- (arithmetic; fix the code) |
| Stage A, endpoint | sampling near lambda = 0+ stays in the head's and backbone's domain | waters collapse onto the ligand -> softcore design needed before Stage B |
| Stage A, control | anchored total = message-masking total within error | they disagree -> a bug, since both are state functions of the same endpoints |
| Stage A, efficiency | decomposition converges faster than message masking | no faster -> claim narrows to attribution only |
| Stage B | `dG_cut` a minority of `dG_decouple` across the set | `dG_cut` dominates -> the attribution covers too little to be the headline |

---

## 10. Deliverable  [open]

One flagship paper: **atom-resolved hydration (then binding) free energies for a
set of drugs, with an exact total and an explicitly reported unattributed
term**, against message masking, MM ABFE and experiment.

The gauging results of v2-S13.1-S13.12 (frozen-backbone gauging, `L_int`, SAPT
resolving the pair distribution, pair non-identifiability) become its methods
section or a short companion paper, rather than v2's Papers I and II.

---

## 11. Risks

**R1 - Endpoint with an ML potential.** Near lambda = 0+ the remaining
ligand-water interaction is only `r`; waters can move into the ligand, where
the backbone and head are far outside their training domain and an MLIP has no
repulsive wall (minimising with the learned potential ran to -5e5 eV, STATUS).
MM softcore has no ML analogue yet. **Addressed by the wall + refill scheme
(S3, W1-W3)**; what remains is the wall-radius choice and the attribution's
dependence on it. Stage A measures both.

**R2 - Cut-step overlap.** Mean remainder 0.3-0.5 eV (7-12 kcal/mol)
(v2-S13.23). Stage 0.1.

**R3 - ML/MM region.** S6.3; the waters switching between ML and MM.

**R4 - Transfer across drugs.** The head has seen Atenolol clusters only.
Pooled clusters (S5.2) are the remedy; whether one head holds across a set is
measured in Stage B.

**R5 - Charged drugs.** Charged DES370K pairs not run; POLAR's total-charge
input and long range help; the cluster coupling of a charged ligand is larger
and longer-ranged, which grows the remainder unless S2.3's pair channel carries
it.

**R6 - Physical meaning of the per-atom split.** S4. Validation, S7.3.

**R7 - Cost.** Unknown until S8.0.2.

---

## 12. Decisions

| # | Decision | Status | By / date |
|---|---|---|---|
| D1 | Total from the backbone, head splits it (S3; v2-S13.23) | **decided** | user, 2026-10-10 ("按照新的写设计") |
| D2 | First real system is a drug set (S8.3) | **decided** | user, 2026-10-10 |
| D3 | `openmm_dev_ubio` and `openmm_dev` are production environments | **decided** | user, 2026-10-10 |
| D4 | v3 replaces v2 S0-S12 as the design document | **decided** | user, 2026-10-10 |
| D5 | Lock POLAR-1-M (S2.1) | **decided** | user, 2026-10-10 |
| D6 | Long-range electrostatics as an explicit pair channel (S2.3) | **decided** | user, 2026-10-12 |
| D7 | IQA: validation only, drop synthetic `L_IQA` from training (S5.3); arm N recipe | **decided** | user, 2026-10-12 |
| D8 | Stage A system (S8.2) | open -- recommend a small drug fragment | |
| D9 | Drug set composition (S8.3) | open | |
| D10 | Far field attributed via MM per-atom lambda (S6.2) | open | |
| D11 | ML-water identity scheme (S6.3): FIRES first | **decided** | user, 2026-10-12 |
| D12 | One flagship paper (S10) | open | |
| D13 | Mark v2 S0-S12 as superseded in the file itself | open | |
| D14 | Endpoint: MM repulsive wall held through the ML decoupling, then refill; GCMC where water exchange is slow (S3 W1-W3) | **decided** | user, 2026-10-10 |
| D15 | Wall radius; GCMC waters ML or MM; GCMC environment (S3) | open -- method agreed (user, 2026-10-12): radius from the MLP's non-physical short-range region and the training data's contact coverage; GCMC parts open | |

---

## 13. Measurements

v2-S13.1-S13.23: `RESEARCH_PLAN_v2.md` S13, unchanged. New measurements below,
from S13.24.

Machines: workstations A and B (RTX 2080 Ti, smoke and timing checks only);
PBS nodes P1 and P2 (RTX 3090) and P3 (RTX 4080); production GPU RTX 5090.

### 13.24 Stage 0 smoke on workstation A (2026-10-10), env `openmm_dev`

`experiments/2026-10-10_stage0_smoke/` (`A_frames.log`, `BC.log`). RTX 2080 Ti,
4 CPU threads, POLAR-1-M + the 20-frame <= 24-water head.

**ML force call** (`time_ml_step.py`: backbone forward + backward + head, one
Atenolol carve from `md50_solvent.npz`, mean of 20 after 3 warm-ups). Excludes
the MM part and the PythonForce round trip, so a lower bound on the ML/MM step:

    n_solv  atoms  ms/call   sd   peak GiB
        20    101    121.3  5.0     1.10
        50    191    196.8  5.3     2.13
       100    341    330.7  2.6     3.94

At 1 fs and 100 ps per window, 12 windows: 40 h (n=20), 66 h (n=50), 110 h
(n=100) per leg, before MM, round trip, W1/W3 and any cut windows.

**These are 2080 Ti numbers. Production runs on an RTX 5090, which has not
been timed**; no speed-up factor is assumed here. Rerun `time_ml_step.py`
on the 5090 before planning Stage A/B wall time. (The remainder statistics of
S13.25 do not depend on the card.)

**MM frames**: 30 ps of the 4,076-atom leg plus three minimisations in ~4 s
on CUDA, so `run_remainder.sh`'s 2 ns trajectory is minutes.

**Remainder path** (`test_scale.py` on 2 smoke frames at 10/20 ps, seed 23):
runs end to end in `openmm_dev`; `direct - truth` +2.23 / +2.24 eV at n=100,
the same failure as v2-S13.20 (+2.18). r sd/kT 2.2 / 8.8 / 10.0 at n=5/20/100
-- **from two configurations, so not a measurement of the spread**, only that
the path works. The 20-frame run (`run_remainder.sh`) is the measurement.

### 13.25 Stage 0.1: the remainder spread is 3-13 kT -- one BAR step does not bridge the cut

`run_remainder.sh` on workstation A (2080 Ti, env `openmm_dev`, 2026-10-10 19:34-19:40,
6 min end to end). Fresh MM trajectory, seed 23, 20 frames 100-1,050 ps, each
MM-minimised; neither head has seen a cluster cut from it. `runs/remainder.log`,
`runs/polar-M/logs/remainder_frames{,_w120}.log`. kT = 0.0257 eV (298 K).

    r = truth - sumD over 20 frames (eV)
                         <= 24 waters                 <= 120 waters
    n   edges  truth     r mean  r sd  sd/kT  share   r mean  r sd  sd/kT  share
    5    260   -1.03     +0.008  0.084   3.3   1%     -0.250  0.089   3.4   24%
    10   531   -1.63     -0.040  0.113   4.4   2%     -0.477  0.172   6.7   29%
    20   980   -2.10     -0.037  0.189   7.4   2%     -0.380  0.211   8.2   18%
    50  1986   -3.11     -0.314  0.324  12.6  10%     -0.020  0.329  12.8    1%
    100 2549   -3.52     -0.930  0.250   9.7  26%     +0.012  0.266  10.4    0.3%

    share = |r mean| / |truth|, of the coupling energy (not of a free energy)

**The spread does not depend on the head.** The two recipes move the mean
(the <= 24 head is near zero up to n = 20, the <= 120 head at n >= 50, as in
v2-S13.22) but give the same sd at every size, 0.08 -> 0.33 eV. Whatever
carries the fluctuation, cluster recipe does not change it.

**One BAR step is not enough beyond the smallest carve.** For a Gaussian
`dU` with sd sigma, the forward and reverse distributions are separated by
sigma^2/kT, i.e. by sigma/kT standard deviations: 3.3 at n = 5 (marginal),
7-13 at n >= 20 (no overlap). So the cut segment needs windows
`U_s = (1-s) U(0+) + s U(0)` (S3). Splitting it into M linear steps divides
the per-step sd by M; at the common rule of thumb of 1-2 kT per step that is
~4-8 windows at n = 20 and ~6-13 at n = 50-100. Each cut window evaluates
both Hamiltonians.

**Caveat.** Minimised MM frames, not the lambda = 0+ ensemble of the anchored
Hamiltonian; thermal configurations would more likely widen the spread than
narrow it. Stage A measures it in the real ensemble.

**For S8:** with the <= 24 head, n = 20 gives the smallest unattributed mean
(2%) at a spread of 7.4 kT; this is the operating point to carry into Stage A
unless the ML-region scheme (S6.3) needs a larger n. `direct - truth`
reproduces v2-S13.20 (+2.02 eV at n = 100, <= 24 head).

### 13.26 D7 ablation: dropping the synthetic L_IQA helps everywhere it is measured

Arm N = arm D without `L_IQA` (`test_joint.py`, weights (1, 1, 0, 1, 1)), same
seed, data, clusters (20 frames, <= 24 waters), 60k steps, batch 16, POLAR-1-M.
PBS `learn_d7.zsh` on node P1 (RTX 3090), finished 2026-10-12 01:26; training
15,171 s (0.25 s/step). Loss flat at 0.020-0.024 over the last 11k steps (D:
~0.027). `runs/polar-M/logs/{joint.log, joint_..._s60000.json, scale_N_*.log}`.

    DES370K test (1,000 dimers, 178 systems)      D        N
      E      meV/atom                             2.87     1.72    -40%
      F      meV/A                                32.3     22.3    -31%
      E_int  kcal/mol                             0.429    0.391
      SAPT es / ex / ind / disp  meV              27.0 / 59.8 / 15.3 / 8.9   26.1 / 59.5 / 14.4 / 8.5
      E_intra vs synthetic teacher  meV           2.8      110      (unsupervised, expected)

    remainder r = truth - sumD, frames_test_solvent.npz (20 frames)
    n    r mean D -> N      r sd D -> N      sd/kT D -> N
    5    +0.008 -> +0.002   0.084 -> 0.077   3.3 -> 3.0
    10   -0.040 -> -0.033   0.113 -> 0.102   4.4 -> 4.0
    20   -0.037 -> -0.026   0.189 -> 0.184   7.4 -> 7.1
    50   -0.314 -> -0.145   0.324 -> 0.286  12.6 -> 11.1
    100  -0.930 -> -0.569   0.250 -> 0.213   9.7 -> 8.3
    sumD fit, meV/edge: +0.261 -> +0.153

    held-out single frames, D (v2-S13.20) -> N, eV
              n:      5      10      20      50     100
    md50   sumD-tru  +0.04/+0.07  -0.17/-0.09  -0.35/-0.28  -0.06/-0.25  +0.56/+0.41
           dir-tru   -0.20/-0.09  -0.31/-0.12  -0.25/-0.12  +0.54/+0.09  +2.18/+1.18
    md2000 sumD-tru  -0.19/-0.11  -0.14/-0.04  -0.14/-0.04  -0.15/-0.17  +0.29/+0.25
           dir-tru   -0.20/-0.09  -0.52/-0.30  -0.21/-0.09  +0.36/+0.25  +1.86/+1.13

**What is significant.** DES370K E and F improve by a third or more. The
remainder mean improves at n = 50 (by 2.4 standard errors, SE = sd/sqrt(20))
and n = 100 (6 SE); at n <= 20 both are within noise of zero. **The spread
does not change**: the standard error of an sd from 20 samples is ~16%, and
the 4-15% reductions are inside it. One seed per arm.

**Reading.** The synthetic teacher pinned a random node gauge and cost
capacity; without it the head fits the backbone and the interaction labels
better, and nothing the anchored Hamiltonian uses gets worse. The cut-step
spread (S13.25) is unaffected, so cut windows are still needed.

### 13.27 The remainder is partly the Coulomb term the head cannot see -- not mostly

`experiments/2026-10-12_r_vs_es/r_vs_es.py` (workstation A, 2080 Ti, `openmm_dev`, fp64,
a few minutes). Same 100 carves as S13.25/S13.26 (20 frames x 5 sizes); r per
carve read from the D and N scan logs; the recomputed `truth` matches the logs
to 4 decimals on every carve. Per carve, from the backbone alone:

- `cross` = `E_es(rho_L + rho_E) - E_es(rho_L) - E_es(rho_E)` with the full-system
  GTO densities: the ligand-environment Coulomb at fixed charges, exactly
  pair-resolvable (S2.3). Computed by wrapping `PolarMACE.coulomb_energy`.
- `d_es` = electrostatic part of the coupling (`cross` + charge response);
  `d_el` = the same for POLAR's local electron energy.

    per size over 20 frames: c = corr(r, x), b = slope, sd of r - (a + b x), kT
    head N   sd(r)    x = cross               x = d_es + d_el
      n=5     3.09    c -0.02  b -0.03  3.17   c -0.26  b -0.24  3.06
       10     4.08    c +0.52  b +1.00  3.57   c +0.33  b +0.29  3.95
       20     7.33    c +0.53  b +1.19  6.40   c +0.50  b +0.60  6.54
       50    11.41    c +0.68  b +1.20  8.65   c +0.80  b +0.99  7.09
      100     8.49    c +0.52  b +0.72  7.43   c +0.62  b +0.64  6.84
    (head D: same pattern, c 0.38-0.61 for cross at n >= 10; full table in the log)

    magnitude, mean +- sd over frames (eV):  cross -0.03+-0.04 (n=5) ... -0.42+-0.16 (n=100)

**What it shows.** At n >= 10, r's fluctuation tracks the cross Coulomb term
with slope ~1 (1.0-1.2 at n = 10-50): the head carries none of its
configuration dependence, as expected from node features that never see the
long-range block (`test_polar.py`). Correlations 0.5-0.8 are 3+ standard errors
from zero (SE ~ (1-c^2)/sqrt(19) ~ 0.15).

**What it does not.** Removing it leaves 6-9 kT at n >= 20. The clear reduction
is at n = 50 (11.4 -> 7.1 kT with `d_es + d_el`, beyond the ~16% SE of an sd);
at n = 20, the operating size, 7.3 -> 6.4 kT is within noise. At n = 5 r is
not electrostatic at all. Fits are in-sample, 2 parameters on 20 points, so
the reductions are if anything optimistic.

**For D6.** The explicit pair channel would move a real, physically defined
piece out of `dG_cut` into the per-atom attribution, and its benefit grows
with n. It does not remove the need for cut windows; the rest of r (short-range
polarisation, many-body, head error) is a different term. A symbolic-regression
search on that rest (the MOT-SR idea) needs far more than 20 carves per size --
backbone labels are free, so that is a data-generation step, not a blocker.

### 13.28 D6 code: identity checks and smoke (2026-10-12)

`experiments/2026-10-12_d6_smoke/`. Checks in fp64 on a 101-atom carve
(`frames_test_solvent.npz` frame 0, n = 20) unless marked:

    wrapper leaves the backbone energy unchanged        |dE| 1.5e-11 eV
    cross vs S13.27's analysis, same carve               |d| 1.3e-11 eV
    Euler split: sum over ligand rows = sum over env rows = cross   to 1e-8
    d cross/dR, autograd vs central FD (h 1e-4 A, 9 components)    4.5e-6 eV/A (|g| 0.10)
    es_coupling d_es vs S13.27                           to 1e-6
    E3D_ES targets, fp32, 4 DES370K dimers: E0 - E_ES = E_es   8.6e-4 eV (4 eps|E_bb| = 1.3e-2)
                                            F0 - F_ES = F_es   3.8e-4 eV/A (run-to-run floor 3.5e-4)

The fp32 force yardstick is measured, not chosen: MACE's own forces for
POLAR-1-M differ by 2.5e-4 eV/A between two identical fp32 calls on the GPU
and by 7e-13 in fp64, where its forces and `-dE/dR` by autograd also agree to
6e-13 (`check_forces_dtype.py`).

Smoke on workstation B (2080 Ti; workstation A's GPU was busy with another job): `test_joint.py`
with `E3D_ES=1`, 2,000 dimers + 200 clusters, 200 steps, runs end to end
(explicit L/E Coulomb |mean| 0.019 eV, dimers 0.014 eV); `test_scale.py` with
`E3D_ES=1` on that head runs; `truth` matches S13.24 to 4 decimals. The
200-step numbers themselves mean nothing.

### 13.29 The anchored Hamiltonian: exact checks and a two-segment HREMD smoke (2026-10-12)

`test_anchored.py` on workstation B (2080 Ti), float64, Atenolol + 10 waters (71 atoms,
`frames_test_solvent.npz` frame 0), head N; `experiments/2026-10-12_anchored_smoke/`.

    [1] U(p=0) = E_bb(R)                                 |d| 0
    [2] U(p=2) = E_bb(L) + E_bb(E), fragments built apart |d| 0
    [3] dU/dp autograd vs finite difference (h 1e-4)       2.7e-8 (continuous), 2.3e-8 (cut)
        per-atom sum = total; zero off the ligand         exact
    [4] U(0+) - U(0) = truth - sumD (coupling_labels)     2.7e-12 eV
        U on the cut is linear in s                       exact
    [5] forces vs central FD, 3 components, p = 0.5, 1.5   6.0e-6, 2.8e-6 eV/A (|F| ~4)
        finite at 9 points p = 0..2; max |F| 3.8 -> 4.6 eV/A, smooth
    [6] es=True (200-step D6 head): sum c(es) - sum c(no es) = cross   5.9e-12 eV
        forces vs FD at p = 0.5                           7.4e-6 eV/A

The FD residuals are the O(h^2) truncation of a central difference at h = 1e-4
A, not an error of the forces; the identities [1]-[4] are float64-exact.

**Two-segment HREMD smoke** (`run_abfe.py --mode anchored`, torch backend,
GPU): Atenolol + 5 waters, 4 continuous + 2 cut windows, 3 x 10 steps. Runs
end to end; prints `dG = sum_i dG_i + dG_cut` (11.96 + 1.43 = 13.39
kcal/mol on this 30-step run -- the numbers mean nothing, only that the
bookkeeping closes). TI and MBAR now agree in units (56.0 vs 59.0 +- 4.1
kJ/mol).

**Bug fixed on the way.** `hremd.HREMD` recorded dU/dlambda in eV, so TI came
out in eV while MBAR (from u_kn = U/kT, kT in kJ/mol) came out in kJ/mol, and
`run_abfe.py` printed both as kJ/mol. No free energy had been run with
sampling, so no result is affected. Also: per-atom TI is integrated over the
continuous segment only (a trapezoid across p = 1 would add a spurious
triangle), and `TorchReplica` with `--device cuda` failed on a CPU/GPU
mismatch in the old code as well; both fixed.

### 13.30 D15: the backbone does not turn over above ~0.7-0.9 A -- the turnover criterion does not bind

`experiments/2026-10-12_wall_radius/scan.py` on workstation B (2080 Ti; it shared the
GPU with another user's job for its 648 s -- launched without checking,
a mistake). One water moved rigidly toward one Atenolol atom along its own
approach direction, probe (water O or H) from 3.5 to 0.7 A, `frames_test_solvent.npz`
frame 0, float64; 26 scans over C/N/O/H x probe O/H.

    backbone coupling truth(r): position of its maximum (inside it the learned "wall" gives way)
    ligand-probe   n   probe distance at the peak     d_min (any L-w pair) at the peak    peak height
    C, N, O - O    9   0.70-0.95 A                    0.09-0.73 A (water H hits first)    +36 .. +68 eV
    all others    17   0.70 A (the grid edge) except one H-H at 0.80   = probe distance   +5 .. +46 eV

POLAR-1-M is repulsive all the way in to sub-angstrom contacts: by tens of eV
before it turns over, and in 16 of 26 scans not at all inside the grid. The
head's sumD peaks at the same place in most scans, a little earlier in a few
(N1 <- water O at 1.40 A probe distance, d_min 0.47 A).

**So "where the MLP turns non-physical" is not where the wall goes.** The
turnover sits far inside any thermally reachable contact; the backbone is
not wall-less in the sense STATUS records (that collapse was a minimisation
with "the learned potential"; which model is not recorded). Repulsive is not the same as
right, though: whether the repulsion is the right *size* at 1.5-2.5 A needs a
reference, and the head's reliability there is what the training data covers.
The wall radius therefore comes from the coverage measurement (part 1; its
first run on workstation A timed out at 900 s before printing anything, to be rerun
batched).

### 13.31 D15: what the head's training data covers at short range

`experiments/2026-10-12_wall_radius/coverage.py` on workstation A, CPU, 12 s (the first
run timed out: it re-read the 232k-object npz array on every access). The
operating head's exact training set: the 50,000 DES370K `organic` dimers of
`test_joint.load_split` (seed 0) and the 10,000 `solvent_clusters` (seed 1),
of which 2,493 contain the ligand. Per structure, the shortest cross distance
per element pair; quantiles over structures (A):

    DES370K dimers (CCSD(T) labels; unordered pair)      clusters with the ligand (backbone labels; ligand - water)
    pair    n      min   p0.1  p1    p5    median        pair    min   p0.1  p1    p5    median
    O-H   38108   0.92  1.10  1.32  1.70  3.22           O-Hw   1.71  1.71  1.71  1.73  1.79
    N-H   13785   0.97  1.13  1.39  1.97  3.98           N-Hw   1.83  1.83  1.83  1.86  1.93
    C-H   45504   1.33  1.55  1.83  2.16  3.60           C-Hw   2.06  2.06  2.06  2.09  2.47
    H-H   49845   0.72  0.87  1.18  1.62  3.04           H-Hw   1.93  1.93  1.93  1.97  2.12
    O-O   18955   1.88  1.98  2.14  2.49  4.65           O-Ow   2.67  2.67  2.67  2.67  2.72
    N-O    9817   1.95  2.05  2.21  2.56  4.66           N-Ow   2.78  2.78  2.78  2.79  2.88
    C-O   35084   1.93  2.15  2.38  2.71  4.13           C-Ow   2.97  2.97  2.97  3.02  3.27
                                                         H-Ow   1.74  1.75  1.75  1.80  1.93

**Two very different coverages.** The dimers reach deep into the repulsive
wall (DES370K includes compressed geometries): an O-H contact at 1.32 A or
less in 1% of them. The clusters, the only many-body labels, sit at one
distance per pair: every quantile is the same to 0.1 A, because all ligand
clusters are cut from 20 MM-*minimised* frames and contain the same nearest
waters. Their shortest ligand O - water H is 1.71 A.

**For the wall (D15).** Below the dimer minimum (O-H 0.92, C-O 1.93 A, ...)
there is no data at all; between that and the cluster minimum there are
CCSD(T) pair labels but no many-body ones. A radius per pair between the
DES370K minimum and p1 excludes what the head has essentially never seen;
where exactly is a choice (open), and it must also stay below the contacts
the fully coupled system visits at 298 K, so that W1 costs nothing -- not yet
measured (the frames are minimised).

**A larger consequence.** Thermal contacts at 298 K will regularly be shorter
than the minimised ones the clusters cover (an H-bond fluctuates below 1.71
A), so in real sampling the many-body part of the head runs outside its
training range at exactly the contacts that matter, and the remainder spread
of S13.25 was measured on minimised frames too. Clusters (and the remainder
check) should come from unminimised MD frames as well. The D6 retrain now
running uses the minimised-frame clusters.


### 13.32 D6 retrain: the explicit Coulomb channel narrows the cut spread, but biases large carves

Arm N with `E3D_ES=1` (`learn_d6.zsh`, PBS 13210 on node P1, RTX 3090,
2026-10-12 10:06-14:54; 60k steps, same seed, data and minimised-frame
clusters as S13.26). The head is trained on `E - cross` and `sapt_es - cross`;
`cross` (S13.27) is added back exactly at evaluation and in `test_scale`.
`runs/polar-M/ckpt/joint_organic_n50000_c10000_frames_solvent_es_h256_s60000_N.pt`,
`runs/polar-M/logs/{joint_..._es_h256_s60000.json, scale_N_es_*.log}`.
Explicit cross term in training: |mean| 0.025 eV (dimers 0.014).

    DES370K test (1,000 dimers, 178 systems)      N        N+es
      E      meV/atom                             1.72     1.53
      F      meV/A                                22.3     21.3
      E_int  kcal/mol                             0.391    0.319   -18%
      SAPT es channel  meV                        26.1     22.1
      SAPT ex / ind / disp  meV                   59.5 / 14.4 / 8.5   59.7 / 14.5 / 8.5

    remainder r = truth - sumD(+cross), frames_test_solvent.npz (20 minimised frames)
    n    r mean N -> N+es    r sd N -> N+es    sd/kT N -> N+es   dir-tru N -> N+es
    5    +0.002 -> +0.013    0.077 -> 0.066     3.0 -> 2.6       -0.11 -> -0.13
    10   -0.033 -> -0.024    0.102 -> 0.090     4.0 -> 3.5       -0.16 -> -0.17
    20   -0.026 -> -0.040    0.184 -> 0.140     7.1 -> 5.5       -0.11 -> -0.12
    50   -0.145 -> -0.279    0.286 -> 0.195    11.1 -> 7.6       +0.25 -> +0.13
    100  -0.569 -> -0.845    0.213 -> 0.203     8.3 -> 7.9       +1.03 -> +0.71
    sumD fit, meV/edge: +0.153 -> +0.236

    held-out single MD frames, sumD - truth, N -> N+es, eV
              n:      5            10           20           50           100
    md50          +0.07/+0.07  -0.09/-0.07  -0.28/-0.17  -0.25/+0.20  +0.41/+0.88
    md2000        -0.11/-0.10  -0.04/-0.00  -0.04/+0.05  -0.17/+0.08  +0.25/+0.78

**What is significant** (SE of a mean = sd/sqrt(20) ~ 0.03-0.06 eV; SE of an
sd ~ 16%; one seed per arm).
- The spread falls at every size, by 13-31%. At n = 50 the 31% drop is about
  twice the SE. At n = 20, the operating size, the 23% drop is borderline. At
  n = 5, 10 and 100 the change is within noise. Same carves for both heads, so
  the comparison is paired; no paired test was run.
- The mean gets worse at large n: n = 50 by -0.13 eV (~3 SE), n = 100 by
  -0.28 eV (~6 SE). The two held-out MD frames show the same at n = 100
  (+0.4 -> +0.9 and +0.25 -> +0.8 eV). At n <= 20 the means are within noise of N.
- DES370K E_int improves 18% and the es channel 15%; ex / ind / disp are
  unchanged. Seed-to-seed scatter on DES370K has not been measured.

**Reading.** As S13.27 predicted, part of r's fluctuation was the cross Coulomb
term, and making it explicit removes some of that fluctuation. The cut windows
are still needed: 5.5 kT at n = 20. The new bias at n >= 50 is outside the
training range (clusters <= 24 waters, minimised). The head was trained to
absorb `E - cross`. On larger carves the exact `cross` keeps growing in
magnitude (-0.42 eV mean at n = 100, S13.27), and the head's remaining part
does not track it. The mechanism has not been checked. Under the anchored
Hamiltonian a mean bias only moves work between dG_continuous and dG_cut,
and the total stays exact. What it costs is attribution: the per-atom split
is cleanest at n <= 20, where N+es is no worse than N.

**Operating head.** N+es for D6 work at n <= 20. Large carves need clusters
from more and unminimised frames, and larger ones (S13.31) -- one retrain
covers both.

### 13.33 Thermal (unminimised) frames: contacts 0.2-0.5 A shorter, remainder biased for both heads

`experiments/2026-10-12_thermal_frames/` (workstation A, 2080 Ti, `openmm_dev`, 2026-10-12).
`thermal_frames.py`: the MD of `em_system.frames()` (em_solvent.npz,
Langevin 298.15 K, 1/ps, 2 fs, frames 100..1050 ps every 50) on CUDA,
**without** minimising. Seed 23 -> `thermal_test_solvent.npz` (frames_test's
seed), seed 11 -> `thermal_solvent.npz` (frames_solvent's seed). Same seeds do not
mean the same trajectories (platform differs). MD took about 110 s per 1.05 ns.
Besides the frames, the shortest ligand-water contact per element pair is
recorded every 0.5 ps (1,901 snapshots per seed). `contacts.py` -> `contacts.log/json`.

    shortest ligand-water contact per snapshot (A)
    pair   | thermal 298 K (3,802 snapshots)      | minimised (20) | clusters | DES370K
           |  min  p0.1   p1    p5  median        |  min  median   |   min    |  min   p1
    C-Ow   | 2.71  2.73  2.81  2.89  3.11         | 3.02  3.14     |  2.97    | 1.93  2.38
    C-Hw   | 1.77  1.87  1.95  2.05  2.39         | 2.09  2.33     |  2.06    | 1.33  1.83
    N-Ow   | 2.54  2.59  2.65  2.72  2.89         | 2.78  2.91     |  2.78    | 1.95  2.21
    N-Hw   | 1.61  1.65  1.72  1.81  2.17         | 1.85  1.96     |  1.83    | 0.97  1.39
    O-Ow   | 2.40  2.46  2.52  2.56  2.70         | 2.68  2.72     |  2.67    | 1.88  2.14
    O-Hw   | 1.50  1.54  1.59  1.64  1.81         | 1.72  1.78     |  1.71    | 0.92  1.32
    H-Ow   | 1.54  1.58  1.66  1.74  1.95         | 1.76  1.91     |  1.74    | 0.92  1.32
    H-Hw   | 1.38  1.47  1.61  1.73  1.98         | 1.85  2.13     |  1.93    | 0.72  1.18
    (snapshots 0.5 ps apart are correlated: n overstates the independent count)

**Contacts.** At 298 K the closest contact runs 0.2-0.5 A shorter than in the
minimised frames. In at least 5% of snapshots the contact is below the
cluster minimum, for every pair except H-Ow, where p5 equals it. **For D15**, the DES370K p1 sits below the thermal p0.1 for
every pair, so a wall between them excludes what the head never saw and
leaves the coupled system alone. The gap is wide for polar pairs (O-Hw
1.32 vs 1.54) and narrow for C-Hw (1.83 vs 1.87).

    remainder r = truth - sumD(+cross), 20 frames, eV; minimised = S13.26/S13.32
    head  frames     n:   5             10            20            50            100
                     r mean / sd/kT
    N     minimised    +0.002/3.0    -0.033/4.0    -0.026/7.1    -0.145/11.1   -0.569/8.3
    N     thermal      +0.002/4.0    -0.092/4.7    -0.251/7.3    -0.702/12.2   -1.265/11.5
    N+es  minimised    +0.013/2.6    -0.024/3.5    -0.040/5.5    -0.279/7.6    -0.845/7.9
    N+es  thermal      +0.006/3.8    -0.076/3.9    -0.238/7.0    -0.791/11.6   -1.431/11.3
    truth (coupling), thermal vs minimised: -1.72 vs -2.10 eV at n = 20
    sumD fit, meV/edge, thermal: N +0.425, N+es +0.476 (minimised +0.153 / +0.236)
    logs: scale_N_thermal.log, scale_N_es_thermal.log

**Remainder.** On thermal frames both heads underbind: at n = 20 the mean r is
-0.24 to -0.25 eV, about 6 standard errors (SE = sd/sqrt(20) ~ 0.04), against
about -0.03 on minimised frames; at n = 50 it is -0.7 to -0.8. The spread
grows a little: n = 100 goes from 8 to 11.5 kT, and the rest are within
the ~16% SE of an sd. **N+es's narrower spread (S13.32) does not survive on
thermal frames** (7.0 vs 7.3 kT at n = 20, inside noise). Both heads are off
the same way, so the domain (minimised clusters) matters more here than the
Coulomb channel. One explanation, not checked: the many-body
cluster labels never contain the short thermal contacts (table above).

**Consequence.** Sampling runs on thermal configurations, so S13.25/S13.32
measured on minimised frames understate the cut-step work and the attribution
loss. At n = 20, 0.25 of 1.72 eV (15%) of the coupling falls into dG_cut and
cannot be assigned per atom. The total stays exact (anchored). The next head
should train on clusters cut from unminimised frames (`thermal_solvent.npz`,
seed 11, ready), with remainder checks on `thermal_test_solvent.npz`.

### 13.34 Plan: thermal-cluster retrain (`learn_thermal.zsh`, run: S13.35)

What changes against S13.26 / S13.32 is the frame source of the clusters,
nothing else: 200 unminimised frames (`thermal200_solvent.npz`, seed 11,
every 10 ps, 100-2090 ps, MD 217 s on workstation A's GPU). The same 10,000 clusters,
<= 24 waters (v2 S13.22: <= 120 worsens n <= 20 and DES370K). 20 thermal
frames are not enough: every cluster from a frame shares that frame's nearest
waters, so coverage barely moves. Per-cluster shortest ligand-water contact,
min (A), from `experiments/2026-10-12_thermal_retrain/cluster_contacts.log`:

    pair    minimised 20   thermal 20   thermal 200   | 298 K p0.1 / p1 (S13.33)
    O-Hw    1.71           1.61         1.54          | 1.54 / 1.59
    N-Hw    1.83           1.86         1.69          | 1.65 / 1.72
    H-Ow    1.75           1.72         1.66          | 1.58 / 1.66
    H-Hw    1.93           1.66         1.46          | 1.47 / 1.61
    C-Hw    2.06           1.96         1.77          | 1.87 / 1.95
    O-Ow    2.67           2.53         2.46          | 2.46 / 2.52

Two arms, N and N+es, in one PBS job (~10 h on node P1, from the D6 arm's
4 h 48 min). They are scanned on `thermal_test_solvent.npz` (seed 23,
unminimised; the measure) and on the three minimised sets (continuity).
Smoke on workstation A (200 clusters from thermal frames, 200 steps, ES on): labels,
targets, training and checkpoint all pass, and the cache names carry the
frame stem (`experiments/2026-10-12_thermal_retrain/train_smoke.log`).
It answers: does the thermal-frame bias (-0.25 eV at n = 20) go, and does the
explicit Coulomb channel help on thermal frames.

### 13.35 Thermal-cluster retrain: the bias on thermal frames is gone

S13.34 as planned. PBS 13315 on node P2 (RTX 3090; the user moved it from P1),
2026-10-12 15:53 to 10-13 01:33. Cluster labels took 823 s once. Targets took
1,257 s (N) and 1,547 s (N+es). Each arm trained 60k steps in about 15,290 s
(0.255 s/step), and the loss ended at 0.022 (N) and 0.021 (N+es). Output is
`e3d_thermal.o13315`,
`runs/polar-M/ckpt/joint_organic_n50000_c10000_thermal200_solvent{,_es}_h256_s60000_N.pt`
and `runs/polar-M/logs/scale_N_thermal{,_es}_*.log`.

    remainder r = truth - sumD(+cross) on thermal_test_solvent.npz (20 unminimised frames)
    r mean (eV) / sd/kT           n:   5            10           20           50           100
    N     minimised clusters  S13.33   +0.002/4.0   -0.092/4.7   -0.251/7.3   -0.702/12.2  -1.265/11.5
    N     thermal200 clusters          +0.040/4.0   -0.007/4.7   -0.000/6.5   +0.040/11.6  -0.059/10.9
    N+es  minimised clusters  S13.33   +0.006/3.8   -0.076/3.9   -0.238/7.0   -0.791/11.6  -1.431/11.3
    N+es  thermal200 clusters          +0.041/3.7   +0.006/3.5   +0.013/5.8   -0.044/10.6  -0.212/9.8
    sumD fit, meV/edge: N +0.425 -> +0.006, N+es +0.476 -> +0.053
    SE of a mean = sd/sqrt(20): 0.02 (n=5) ... 0.07 (n=50) eV; SE of an sd ~16%

    DES370K test                 N(min)  N(thermal)  N+es(min)  N+es(thermal)
      E   meV/atom               1.72    1.70        1.53       1.51
      F   meV/A                  22.3    22.1        21.3       21.3
      E_int kcal/mol             0.391   0.394       0.319      0.319

    the same thermal heads on MINIMISED frames, r mean (eV)   n: 20      50      100
      N     frames_test_solvent                                +0.245  +0.713  +0.825
      N+es  frames_test_solvent                                +0.207  +0.520  +0.476

**The bias goes.** On thermal frames, the mean of r is now within about 2 SE of
zero at every size for N, and up to n = 50 for N+es. At n = 20 it went from
-0.25 to 0.00 eV. N+es keeps -0.21 eV at n = 100 (about 4 SE), which is S13.32's
large-carve pattern again. The spread barely changes: at n = 20, 7.3 -> 6.5 kT
(N) and 7.0 -> 5.8 kT (N+es), both within or at the edge of the SE. So the
cut windows stay. DES370K is unchanged. The S13.33 explanation (the many-body
labels lacked thermal contacts) is supported, not proven: frames changed in
both count and minimisation.

**The bias moves to minimised frames.** The thermal heads now overbind
minimised geometry (+0.25 eV at n = 20, +0.7 at n = 50 for N). The md50 /
md2000 single frames, also minimised, show the same. The head fits the
ensemble it was trained on. That is the right ensemble for lambda = 1. It
does not say the head holds at intermediate lambda, where the solvation shell
of a partly decoupled ligand is neither of these. That check is still to do:
frames from intermediate-lambda sampling, scanned the same way.

**N vs N+es, both on thermal clusters.** The spread is 7-26% lower with es at
every size; only n = 10 (4.7 -> 3.5 kT) is beyond one SE. DES370K E_int is 19%
better, the n = 100 mean is worse, and n <= 50 is equal. One seed per arm.

**Operating head.** `..._thermal200_solvent_es_h256_s60000_N.pt` (N+es, D6
kept), for n <= 50. Below n = 100 it is no worse than N on any measure here.
At n = 100 N is better (mean -0.06 vs -0.21 eV).
Superseded by the pooled-state head of S13.37.

### 13.36 Intermediate lambda (MM proxy): the operating head overbinds once the shell loses its H-bonds

`experiments/2026-10-13_intermediate_lambda/` (workstation A, 2080 Ti, `openmm_dev`,
2026-10-13). The real intermediate-lambda ensemble of the anchored ML
Hamiltonian needs ML/MM + FIRES + the wall, and none of them is written. So
this uses MM frames at partly decoupled states as a proxy.
`lambda_frames.py` runs the MD of S13.33 with the ligand-water electrostatics
scaled by lambda_elec. LJ stays in full: its repulsion stands in for the
wall. Ligand intramolecular terms are unchanged.

- **Implementation (`scaled_system`).** The ligand charges are multiplied by
  lambda_elec, and every ligand-ligand pair gets an exception carrying its
  original charge product. Checks (`check_scaled.log`, 20 thermal frames,
  double precision): at lambda_elec = 1 it matches the reference to
  -0.034 +- 0.014 kJ/mol, with max |dF| 0.25 kJ/mol/nm; it is linear in
  lambda_elec to 0.08 kJ/mol.
- **Tried first and rejected:** openmmtools `AbsoluteAlchemicalFactory`.
  Its default "exact" PME refuses decoupled electrostatics. "direct-space"
  is already off at lambda = 1: 33 +- 13 kJ/mol over the same frames, and
  ligand force errors up to 210 kJ/mol/nm (`check_l1.log`).

Test sets: seed 23, 20 frames from 100 to 1050 ps; MD took 111-147 s each.
Training pools: seed 11, 200 frames every 10 ps.

    operating head (thermal200 N+es, S13.35); r = truth - sumD(+cross), 20 frames, eV
    lambda_elec      n:   5            10           20           50           100
                     r mean / sd/kT
    1   (S13.35)          +0.041/3.7   +0.006/3.5   +0.013/5.8   -0.044/10.6  -0.212/9.8
    0.5                   +0.081/2.0   +0.152/4.0   +0.249/6.9   +0.416/7.2   +0.322/6.5
    0                     +0.098/2.6   +0.178/4.3   +0.237/6.0   +0.444/7.7   +0.383/7.8
    truth at n = 20:  -1.72 (1)  -0.87 (0.5)  -0.61 (0) eV

    shortest ligand-water contact, p1 / median (A), lambda_elec 1 | 0.5 | 0
    O-Hw  1.58/1.81 | 1.74/2.19 | 2.03/2.60     N-Hw  1.72/2.15 | 1.98/2.78 | 2.17/2.81
    H-Ow  1.65/1.94 | 1.89/2.33 | 2.01/2.42     H-Hw  1.63/1.98 | 1.56/1.97 | 1.42/1.90
    below the thermal200 cluster minimum: <= 1.4% of snapshots for any pair at
    any state (H-Hw at 0: 1.7%)   (contacts_lambda.log)

**Bias.** Below lambda = 1 the head overbinds. At n = 20 the mean of r is
+0.24 to +0.25 eV, about 6 SE (SE = sd/sqrt(20) ~ 0.04). At lambda = 1 it is
+0.01. At n = 50 it is +0.42 to +0.44. Against a coupling that shrinks to
-0.61 eV at lambda_elec = 0, that is 40% left unattributed on this proxy.
The spread does not grow: 6-7 kT at n = 20.

**Not a short-contact problem.** With the H-bonds gone, the shell moves out,
and the contacts are longer than at lambda = 1 and inside the cluster
coverage. The one exception is ligand H against water H, which comes closer
without the electrostatic repulsion. The configurations are new in kind,
not in range: a non-H-bonded shell around a polar solute. The lambda = 1
clusters do not sample that. Same lesson as S13.33/S13.35: the head fits the
ensemble its clusters come from.

**Proxy caveat.** MM lambda_elec states are not the anchored ML ensemble at
intermediate lambda. In the anchored U the coupling fades through -(1 - lambda)
sumD, and electrostatics and LJ are not split. What the proxy does show is
that one plausible class of decoupled-shell configurations is outside the
head's domain.

**Next (S13.34-style retrain, `learn_lampool.zsh`, run: S13.37):** pool
clusters over states. `lampool400_solvent.npz` holds 200 frames at
lambda_elec 1 (thermal200), plus 100 each at 0.5 and 0 (every 2nd of the
200), so clusters fall 2:1:1. Two N+es arms: 20,000 clusters (lambda = 1
keeps ~10,000) and 10,000 (the dimer/cluster balance kept). Cutting 20,000
clusters from the 400 frames takes 15 s on CPU (`pool.log`).

### 13.37 Pooled-state retrain: the intermediate-lambda bias goes, at a small cost at lambda = 1, n = 50

`learn_lampool.zsh` as planned in S13.36. PBS 13460 on node P3 (RTX 4080; the user
moved it from P2, an RTX 3090), 2026-10-13 06:28 to 15:31 (9.05 h;
estimate was ~10 h, from the 3090).

    arm        cluster labels  targets   60k steps            final loss
    c20000     1,291 s         1,713 s   14,435 s (0.241 s/step)  0.022
    c10000       642 s         1,215 s   12,563 s (0.209 s/step)  0.022

Output: `e3d_lampool.o13460`,
`runs/polar-M/ckpt/joint_organic_n50000_c{20000,10000}_lampool400_solvent_es_h256_s60000_N.pt`,
`runs/polar-M/logs/scale_N_es_lampool_c{20000,10000}_*.log`. Paired analysis:
`experiments/2026-10-14_lampool_result/paired.py` (`paired.log`, `paired.json`).

    r = truth - sumD(+cross), 20 frames per set, r mean (eV) / sd/kT
    set           head            n:  5            10           20           50           100
    lambda 1      S13.35 operating    +0.041/3.7   +0.006/3.5   +0.013/5.8   -0.044/10.6  -0.212/9.8
     (thermal)    lampool c20000      +0.050/3.5   +0.020/3.5   -0.012/5.8   -0.145/9.8   -0.304/8.9
                  lampool c10000      +0.048/3.5   +0.012/3.7   -0.007/6.1   -0.115/10.3  -0.303/9.3
    lambda_e 0.5  S13.35 operating    +0.081/2.0   +0.152/4.0   +0.249/6.9   +0.416/7.2   +0.322/6.5
                  lampool c20000      +0.025/2.2   +0.049/4.0   +0.056/5.8   +0.054/7.1   -0.050/6.7
                  lampool c10000      +0.029/2.2   +0.054/4.3   +0.073/6.5   +0.123/7.4   +0.006/7.0
    lambda_e 0    S13.35 operating    +0.098/2.6   +0.178/4.3   +0.237/6.0   +0.444/7.7   +0.383/7.8
                  lampool c20000      +0.029/2.4   +0.055/4.0   +0.011/6.3   +0.017/7.4   -0.062/6.4
                  lampool c10000      +0.027/2.5   +0.050/4.4   +0.019/6.5   +0.085/7.9   -0.002/7.3
    minimised     S13.35 operating    +0.036       +0.033       +0.207       +0.520       +0.476
     (frames_test) lampool c20000     +0.060/2.6   +0.080/3.0   +0.180/4.9   +0.383/6.5   +0.410/7.0
                  lampool c10000      +0.067/2.8   +0.083/3.3   +0.211/5.4   +0.483/7.4   +0.465/7.1
    SE of a mean = sd/sqrt(20): ~0.035 eV at n = 20, ~0.045-0.06 at n = 50

    paired per-frame differences (same 20 frames and carves), mean +- paired SE, eV
    set           n:   20                        50                        100
    c10000 - S13.35 operating
      lambda 1         -0.020+-0.010 (-1.9)      -0.070+-0.014 (-5.2)      -0.091+-0.014 (-6.5)
      lambda_e 0.5     -0.175+-0.010 (-17)       -0.293+-0.015 (-20)       -0.315+-0.014 (-23)
      lambda_e 0       -0.218+-0.008 (-29)       -0.359+-0.010 (-36)       -0.385+-0.011 (-35)
      minimised        +0.004+-0.013 (+0.3)      -0.037+-0.015 (-2.5)      -0.010+-0.016 (-0.6)
    c20000 - c10000
      lambda 1         -0.004+-0.005 (-0.9)      -0.030+-0.008 (-4.0)      -0.001+-0.008 (-0.2)
      lambda_e 0.5     -0.017+-0.006 (-3.1)      -0.069+-0.006 (-12)       -0.056+-0.008 (-6.8)
      lambda_e 0       -0.009+-0.005 (-1.8)      -0.068+-0.008 (-8.9)      -0.060+-0.010 (-6.0)
      minimised        -0.031+-0.007 (-4.6)      -0.100+-0.010 (-10)       -0.056+-0.010 (-5.4)

    DES370K test                 S13.35 N+es  c20000  c10000
      E   meV/atom               1.51         1.53    1.44
      F   meV/A                  21.3         22.4    21.4
      E_int kcal/mol             0.319        0.341   0.319

**The intermediate-lambda bias goes.** At n = 20 the mean of r drops from
+0.25 / +0.24 eV (lambda_elec 0.5 / 0) to +0.07 / +0.02 (c10000) and
+0.06 / +0.01 (c20000), all within 2 SE of zero. The paired shifts are 17-36
paired SE. At lambda_elec 0 the unattributed share of the n = 20 coupling
(-0.61 eV) goes from 39% to 3%. At n = 50, c10000 keeps +0.12 / +0.09 eV
(2.9 / 1.9 SE); c20000 is at +0.05 / +0.02. The spread does not change
(sd/kT within one SE everywhere), so the cut windows stay as in S13.25.
Pooling the states into the clusters is enough; the S13.36 reading (the head
fits the ensemble its clusters come from) holds a third time.

**Cost at lambda = 1.** n <= 20 is unchanged (paired shifts within 2.2 SE).
At n = 50 both heads move toward underbinding: c10000 by -0.070 +- 0.014,
c20000 by -0.101 +- 0.015 eV, so the n = 50 mean is -0.115 (1.9 SE from
zero) and -0.145 (2.6 SE). At n = 100 both lose a further 0.09 eV
(-0.30 eV). The paired SE is frame noise only. All heads used the same
training seed (`test_joint.py` hard-codes 7), so run-to-run variance is not in
it, and whether pooling causes this shift or any retrain would move n = 50 by
this much is not known without seed replicates.

**c20000 vs c10000.** More clusters help most at n = 50: -0.07 eV at both
intermediate states (9-12 paired SE) and -0.10 on minimised frames, with
-0.03 at lambda = 1. DES370K E_int is 7% worse (0.341 vs 0.319 kcal/mol),
the S13.22 trade-off again in small. One seed each, so the size of that
difference against training noise is not known. Minimised frames still
overbind with both (+0.18 to +0.21 eV at n = 20), as in S13.35.

**Operating head:**
`runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt`
(lampool c10000, N+es), for n <= 50. It replaces the S13.35 head. Reasons: it
keeps DES370K, and it has the S13.35 dimer/cluster balance, so the only
change from that head is the pooled states. c20000 is the alternative where
n = 50 at intermediate lambda matters more than DES370K. Both remain
MM-proxy results (S13.36 caveat): the anchored ML ensemble at intermediate
lambda is still unsampled.

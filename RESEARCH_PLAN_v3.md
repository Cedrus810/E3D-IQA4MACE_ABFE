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

**c is free, so bound it** [decided, D17, user 2026-10-15; S13.41]. Both
endpoints are the backbone's own energies, so the total is exact for any
c. That freedom is used to keep U well-behaved where the head is out of its
training range. Without it, at lambda -> 0+ the crowded shell made sum c
overshoot the backbone's coupling, and U = E_bb - sum c ran away (S13.41).

*Scope of "exact"* (review, 2026-10-16). At fixed endpoints, with a correct
implementation and enough sampling, changing the head changes only the path
and the attribution, not the endpoint free-energy difference. It does not
guarantee three things:
- that the endpoints are the reference states we want (S7.2, D18);
- that the total is unchanged when the ML region size n changes, since n
  changes the hybrid Hamiltonian;
- that the endpoint is physical just because the per-atom terms close.

- **Cover switch.** Each ligand-environment D_ia is multiplied by
  `cover_switch(r_ia)`. The switch is 1 above the shortest contact in the
  head's training clusters (`COVER_R`) and smoothstep to 0 over 0.2 A below
  it. The explicit Coulomb is not switched.
- **Backbone cap.** Define truth = E_bb(R) - E_bb(R_L) - E_bb(R_E) and
  S = sum c. Then
  `S_eff = smin_tau(S, softplus_tau(truth + m))`, and the excess S - S_eff is
  taken evenly off every c_i. Here m = 0.5 eV (about 3 sd of r at n = 20,
  S13.37) and tau = 0.025 eV.
  - It guarantees U(0+) >= min(E_bb(R), E_bb(R_L) + E_bb(R_E) - m), so U
    stays above the two physical endpoints minus m at every lambda.
  - Cost: two extra fragment backbone calls per step; the 2080 Ti ML/MM
    step at n = 20 goes from 0.165 to 0.30-0.32 s (S13.42).
  - The excess depends on R only, so dU/dlambda_i = c_i still holds.
  - The cost is the two fragment energies at every step.
- **Effect on attribution.** What the switch and the cap remove stays in r,
  and so in dG_cut. In ordinary configurations neither acts: the start frame
  changes by 0.00 kJ/mol.

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
  (smoke, workstation A, 2026-10-10). The pocket-water cycle of `ABFE_IBS`
  (fixed-N ABFE + kT ln P(N)) is the candidate for the complex leg's W3.
  Correction (2026-10-14): the file cited here before,
  `ABFE_IBS/ABFE_IBS/PLAN_pocket_water_sampler_redesign.md`, is the user's
  draft; what ABFE_IBS built is
  `ABFE_IBS/ABFE_IBS/docs/design/PLAN_pocket_water_swap_sampler_2026-09-27.md`
  (relocation MC), after the paused GCNCMC route of
  `docs/design/PLAN_pocket_water_gcmc_patch.md`. Reference note:
  `REF_pocket_water_from_ABFE_IBS_2026-10-14.md`. It is a reference, not
  design; its numbers are ABFE_IBS's and are not re-measured here.
- **W3 as a P(0) estimate** (from the reference note; derivation checked
  here, nothing measured). For a wall `W >= 0`,
  `F(no wall) - F(wall) = kT ln < exp(-beta W) >_(no wall)`. For a hard
  wall this is `kT ln P_(no wall)(N_wall = 0)`, where `N_wall` counts the
  waters inside the wall. In the solvent leg, `P(0)` for a drug-sized
  cavity is far in the tail, so stepwise MM softcore removal stays the
  method there. In the complex leg it is the same far-tail problem as
  ABFE_IBS's N* = 0 and needs an N bias. The counting region must be the
  wall's own region.
- **Is exchange slow? Measure before building GCMC** [open; recommendation].
  D14 says GCMC "where water exchange is slow". ABFE_IBS found MD already
  refills the ligand volume in JNK1 and BRD4 at lambda = 0
  (`REF_...` S2, [not verified here]). So the first complex-leg step is a
  cheap probe: 1 ns of plain MD at lambda = 0 with no wall, tracking water
  identities. GCMC follows only if it finds sites that do not exchange.

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

### 6.2 Far field: also attributable  [decided, D10, 2026-10-14]

The ligand's MM nonbonded interactions with waters outside the ML region are
pairwise by construction. Scale them with the same per-atom `lambda_i` (MM
softcore as in ordinary ABFE) and their `dU/dlambda_i` is per-atom too, so the
diagonal path covers the far field as well. This removes v2-S13.13's
"near-field only" limitation, at the cost of mixing ML and MM physics in one
map; the two parts are reported separately and summed.

### 6.3 ML/MM with FIRES and the wall  [decided, D16, 2026-10-14]

D10, D11, D14 and D15's first radius are decided; this section turns them into
one Hamiltonian. Items marked *(choice)* are recommendations still to approve.

**Source.** Rowley & Roux, JCTC 8, 3526 (2012), doi 10.1021/ct300091w,
read in full 2026-10-14 (`ct300091w.pdf`, project root, supplied by the user;
not for the repository). What it says (eq numbers are the paper's):

- **Exact factorisation (eq 2-3).** The configurational integral with N
  indistinguishable solvent molecules is rewritten as an integral over the n
  molecules nearest the solute, times one over the N - n others restricted to
  outside the outermost of the n, with weights 1/n! 1/(N - n)!. Sampling with
  the n inner molecules strictly kept nearer than the N - n outer ones is
  therefore "strictly exact". Averages are rigorous as long as inner and
  outer molecules never swap, independent of the potential energy surface.
- **Boundary (eq 4):** `R_inner = max(r_1, ..., r_n)`, recomputed every step;
  r is the solute-solvent distance.
- **Restraint (eq 5):** `E_FIRES = 1/2 sum_j k_FIRES (r_j - R_inner)^2`, a
  half-harmonic repulsion on every outer molecule j that is closer than
  R_inner. Through R_inner it also acts on the outermost inner molecule.
- **k_FIRES = 500 kcal/mol/A^2.** This was found effective without numerical
  instability at a 1 fs step (VV2, CHARMM, dual Langevin). All waters were
  rigid (SHAKE).
- **Check (Fig. 2).** In a pure MM test (451 SWM4-NDP waters around K+, 3 ns,
  n = 12), the FIRES RDF equals the unrestrained RDF for r < 4 A. It
  deviates slightly at 4-5 A, where the restraint is active.
- **Limitations stated:**
  - The restraint fires more often as n grows.
  - Dynamics, for example exchange rates, are not physical.
  - A reflective boundary, or exchanging momenta at a "virtual collision",
    would be the fix. SAFIRES (Kirchhoff et al., JCTC 17, 5863, 2021) later
    did exactly that, and records that Bulo et al. saw molecules accumulate
    at a FIRES boundary.

#### 6.3.1 Partition of the Hamiltonian

The atoms are L (the ligand), I (the n inner waters, ML, fixed identity) and
O (the outer waters, MM). The system is the prepared solvent leg: 4,076 atoms,
1,345 TIP3P waters, a 35 A box, PME.

| term | computed by | lambda dependence |
|---|---|---|
| L internal, L-I, I-I | ML on L u I: anchored U of S3 (`E_bb(L u I)` and `c_i`; `E_bb(L) + E_bb(I)` at lambda = 0) | anchored (S3) |
| L-O | MM: Coulomb (PME) + LJ | linear in lambda_i (D10) |
| I-O, O-O, O internal | MM (TIP3P) | none |
| wall, L x (I u O) | MM | switch s_w (W1, W3) |
| FIRES, I vs O relative to L | restraint | none |

**MM terms removed**, because ML supplies them:
- the ligand's bonded terms;
- L-L, L-I and I-I nonbonded terms, removed by exceptions. PME's exclusion
  correction then removes their reciprocal part. This is the same mechanism
  `scaled_system` relies on, checked to 0.03 kJ/mol in S13.36.

`strip_ligand_environment_nonbonded` in `openmm_bridge.py` strips L from the
whole environment. That fits an all-ML environment, not this split.

**Embedding is mechanical.** ML-MM electrostatics use MM charges (TIP3P;
ligand charges scaled by lambda_i). The ML region is not polarised by the MM
waters.

**All waters stay rigid (SETTLE), the inner ones too** *(choice)*. Reasons:
- FIRES relabels waters, so inner and outer waters must have the same degrees
  of freedom.
- It keeps 2 fs steps.
- The head's clusters were cut from TIP3P MM frames (S5.2), so ML is
  evaluated on exactly the geometry it was trained on.

**PBC.** The ML coordinates of L u I are unwrapped around the ligand each
step (minimum image per whole residue, as in `carve_from_arrays`). The ML
region (~8 A at n = 20) is far below half the box.

#### 6.3.2 FIRES

- **Ordering distance** (D16, revised by the user 2026-10-14 after S13.38):
  `d_w = softmin_beta(|r_O(w) - r_i|)` over the water oxygen and the ligand
  heavy atoms i, with `softmin_beta(x) = -(1/beta) ln sum exp(-beta x)`,
  beta = 20 /A.
  - The softmin offset from the true minimum is at most ln(19)/beta =
    0.15 A, and measured at 0.034 A or less.
  - Why not all atoms. The first version ran over all atoms, the
    closest-atom metric of `carve_from_arrays`. It put the restraint on H,
    and stacked pairs pushed omega dt to >= 1.8. The run went NaN at
    57-63 ps at 2 fs (S13.38). On heavy atoms omega dt is <= 0.66 and the
    run is stable.
  - What it costs. The inner set is not exactly the closest-atom carve of
    the remainder scans: 17 of 20 waters are shared on em_solvent.npz.
  - The paper's own ordering was also heavy-atom, ion to water.
  - A centre-of-mass sphere does not fit: Atenolol is 12 x 6 x 10 A.
- **Restraint** *(choice: a C1 variant of eq 5)*:
  `U_F = sum_{w in I} sum_{v in O} (k/2) (d_w - d_v)^2 theta(d_w - d_v)`.
  - Eq 5 with `R_inner = max` is continuous in energy. But its force jumps
    when the outermost inner water changes while an outer water is inside
    R_inner, because the max hands the reaction force to a different water.
  - The pairwise sum has no such jump: it is C1 everywhere.
  - It equals eq 5 whenever only the outermost inner water is beyond a given
    outer water, which is the usual violation.
  - Both are exactly zero on the allowed region, so the exactness argument
    holds for either.
  - It costs n x 1,325 pair terms per step and is computed in the same torch
    force as ML.
- **What the free energy is exact for.** This is the paper's eq 3, applied
  here to the alchemical path.
  - Let H_n be the adaptive Hamiltonian in which the n waters nearest the
    ligand by d are ML. H_n is discontinuous wherever the order changes.
  - With the order strictly kept (k -> infinity), the FIRES ensemble is
    H_n's ensemble times C(1345, n), the same at every lambda. So dG under
    H + U_F tends to dG under H_n.
  - At finite k, dG is still exact for H + U_F, which is the Hamiltonian at
    every lambda. Only its distance from H_n is approximate.
  - That distance is measured, not assumed: <U_F>, and how often and how deep
    the order is violated, per lambda window.
- **At lambda = 0 the ligand is a ghost**, but U_F still orders waters by
  distance to it.
  - In the hard limit that is only a relabelling, with no physical effect.
  - At finite k it is a weak ghost-water coupling. The same measurement
    bounds it.
- **k** *(choice)*: start from the paper's 500 kcal/mol/A^2 (2,092
  kJ/mol/A^2). That was tested at 1 fs; at 2 fs, check stability in the smoke
  run and lower k only if it fails. Report the distribution of violation
  depth.
- **n**: 20, the operating size (S13.25). The head is valid for n <= 50
  (S13.37).
- **Boundary artefact.** Use the paper's own check (Fig. 2), on our box
  and metric: pure MM, the same 4,076-atom leg, FIRES at n = 20 against no
  restraint. Compare the density of waters against d and the ligand-water
  RDFs. This needs no ML, so it is the first smoke.
  - If it accumulates the way Bulo et al. report, SAFIRES's elastic-collision
    integrator is the known fix. It needs a custom integrator, so it is not
    planned now.

#### 6.3.3 The wall (D14, D15)

- **Form**: `U_wall = s_w sum_{i in L} sum_{a in I u O} eps_w g(r_ia / r0_ia)`,
  with `g(x) = x^-12 - 2 x^-6 + 1` for x < 1 and 0 otherwise.
  - This is WCA shifted: g = g' = 0 at r0, so the forces are continuous.
- **r0 per element pair** (D15, revised by the user 2026-10-15 after
  S13.41): the larger of the shortest contact in the operating head's
  training clusters and DES370K p1, per (ligand element, water element):
  - ligand O / N / C / H against water O: 2.46 / 2.62 / 2.69 / 1.66 A;
  - against water H: 1.54 / 1.69 / 1.83 / 1.39 A.

  Why: at lambda = 0+ only r = truth - sum c acts, and r is trusted only
  where the backbone-labelled clusters reach. The first radius (DES370K p1)
  let ligand H be crushed to 1.1-1.25 A, where r is -1,243 kJ/mol (S13.41).
  The cost is that these radii sit at the edge of the 298 K contacts, so W1
  is no longer ~0: it is measured (FEP or windows).
- **eps_w** *(choice)*: set in the smoke run at lambda = 0. It is the
  smallest value at which the closest L-water contact stays above ~0.9 r0;
  at x = 0.8, g = 7.9. E7 (S13.42): eps_w = 25 kJ/mol keeps it at
  >= 0.92 r0 at lambda = 0+; smaller values not tried.
- **W1** (lambda = 1, s_w from 0 to 1). With the revised radii (S13.41)
  the wall touches the edge of the 298 K contacts, so dG_W1 is small but not
  zero. The first estimate is one FEP step, -kT ln <exp(-beta U_wall)> at
  s_w = 0 (E7 state q = 0). Windows are added if that has poor overlap.
  Measured (S13.42): dG_W1 = +0.006 kJ/mol, largest sample 0.12 kT, so one
  step suffices on this start frame.
- **W3** (lambda = 0, s_w from 1 to 0). The cavity floods, so the wall is
  softcored: r is replaced by `(alpha (1 - s_w) r0^6 + r^6)^(1/6)`, a few
  windows.
  - ML is still on in W3 (`E_bb(L) + E_bb(I)`), so W3 windows cost as much
    as W2 windows.
  - Releasing I to MM first would add a separate ML -> MM water-cluster
    term, which is not zero.
  - For the complex leg's W3, see `REF_pocket_water_from_ABFE_IBS_2026-10-14.md`
    (S3).
- **Per-atom attribution**: W1 and W3 use per-atom s_w,i on the diagonal
  path. dU/ds_w,i is the per-atom wall energy, which gives the cavity term
  of S3.

#### 6.3.4 One path, all segments

    W1    lambda = 1,  s_w 0 -> 1                       MM wall, FEP / few windows
    W2    lambda 1 -> 0+, wall on                       TI: ML c_i + MM far-field, per atom
    cut   lambda = 0+ -> 0, s 0 -> 1, wall on           cut windows (S3, S13.25)
    W3    lambda = 0,  s_w 1 -> 0 (softcore)            MM wall, windows

    dG = dG_W1 + sum_i (dG_i^ML + dG_i^MM) + dG_cut + dG_W3

- Per atom: dG_i^ML and dG_i^MM from W2, and the wall terms from W1 and W3.
  Global: dG_cut.
- With the wall on, W2 needs no softcore: the wall keeps every L-water
  distance above ~r0, so linear LJ and Coulomb in lambda_i stay finite. That
  is what D14 buys.
- One HREMD ladder over a single path coordinate, as `AnchoredForce` does
  today for W2 and the cut.

#### 6.3.5 Exact checks (to code; float64 where the platform allows)

- **E1 Partition.** At lambda = 1 with s_w = 0, the MM part equals the full
  MM energy minus the removed terms. The removed terms are computed
  independently from subsystem energies, to the precision of the double
  platform.
- **E2 Far field.** The L-O MM energy is linear in lambda_i. Per-atom
  dU/dlambda_i sums to the total and matches finite differences.
- **E3 Anchored.** S3 checks 1-5, inside the full system.
- **E4 FIRES.** U_F = 0 when ordered. Forces match finite differences, and
  stay continuous across an order swap.
- **E5 Wall.** Zero above r0. Force continuous at r0. Softcore finite at
  r = 0 for s_w < 1.
- **E6 PBC.** Energy unchanged when the whole system is translated by a box
  vector, and when molecules are rewrapped.
- **E0 FIRES on pure MM.** The paper's Fig. 2 check, as above. It runs
  before any ML.
- **E7 Continuity, end to end.** Short NVE runs at lambda = 1, 0.5 and 0 with
  the wall on, and in W3. The drift is compared against the same box under
  pure MM. This is the test of "the potential is continuous".

Status (2026-10-15): E0-E6 pass (S13.38-S13.40). E7 passes after the D15
revision and D17 (S13.42): no state shows a jump, and the sd is at the MM
level. Open: at lambda = 0+ the 2 fs run drifts +4-6 kJ/mol/ps, which
shrinks at 1 fs, so it is probably integration error from a stiff term.

#### 6.3.6 Implementation

- **New `decomp/mlmm.py`**:
  `build_mlmm_system(leg, n, model, radii, k, eps_w) -> (System, MLMMForce)`.
  - **NonbondedForce**:
    - exceptions for L-L (zero, since ML supplies the ligand's internal
      terms), L-I and I-I;
    - ligand charges through particle-parameter offsets driven by per-atom
      global lambda_i, with `addEnergyParameterDerivative` for exact per-atom
      dU/dlambda_i under PME;
    - ligand LJ epsilon set to 0.
  - **Ligand-O LJ**: a `CustomNonbondedForce` with interaction group (L, O),
    linear in lambda, using the system's cutoff and switching.
    - Its per-atom derivative is the unscaled per-atom pair sum (short-range,
      exactly pairwise), computed on stored frames.
    - The long-range correction must be treated consistently; to check in E1.
  - **Wall**: a `CustomNonbondedForce` with interaction group (L, I u O),
    and r0 from a `Discrete2DFunction` over element types.
  - **ML + FIRES**: one `PythonForce` on L u I (unwrapped), reusing
    `AnchoredForce`, plus U_F over all waters.
  - Ligand bonded MM forces are removed. The prepared leg's constraints are
    kept.
- **`run_abfe.py --mode mlmm`**: the ladder of S6.3.4 on `hremd.HREMD`'s
  OpenMM backend.

#### 6.3.7 Smoke and measurements before any production plan

On a 2080 Ti workstation (smoke only, minutes each):
- E1-E7;
- the step time at n = 20;
- <U_F> and order violations;
- <U_wall> at lambda = 1;
- the closest contact at lambda = 0 (sets eps_w);
- the boundary density.

Then a timing run on the RTX 5090 before planning Stage A.

Done so far (2026-10-15, S13.38-S13.42): E0-E7; the step time at n = 20
(2080 Ti: 0.30-0.32 s with the cap); <U_F> and violations (U_F <= 17.4
kJ/mol, <= 10 pairs); <U_wall> at lambda = 1 (W1 = +0.006 kJ/mol); the
closest contact at lambda = 0 (eps_w = 25 holds >= 0.92 r0); the boundary
density (E0). Not done: the RTX 5090 timing.

- **Region size.** Under S3 the total does not depend on the head at a given
  n. It does depend on n, because n defines the hybrid ML/MM Hamiltonian:
  which waters are ML (corrected 2026-10-16; the earlier text said "exact at
  any n"). What also grows with n is the remainder (v2-S13.23: 9-20% at
  n = 100) and the cost. Choose n from the measured remainder, and measure
  the total's n dependence (D18).
- **Step time on POLAR-1-M.** The ML force call is measured (S13.24); the
  full ML/MM step, MM and the PythonForce round trip included, is not.

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
- **The decoupled endpoint is not yet a defined reference state** (review,
  2026-10-16; S13.47). At q = 4 the code has:
  - the 20 inner waters still ML (`E_bb(I)`);
  - inner-outer still MM;
  - FIRES still ordering the waters by distance to the ligand's heavy
    atoms (`mlmm.py`, `heavy_selection(lig)`, independent of lambda).

  So H_end = E_ML(L) + E_ML(I) + U_MM,rest + U_FIRES(L, I, O). Three
  consequences:
  - The solvent's partition is still defined around the ghost ligand.
  - The ghost is not fully decoupled: FIRES forces act on its atoms, the
    ML droplet follows it, and its conformational ensemble is not the gas
    phase.
  - FIRES is exact only when relabelling solvent molecules leaves the
    energy unchanged. That holds in all-MM water, but not in hybrid water,
    where membership of the inner set decides which interactions are ML.

  G_end therefore holds a ligand-shape-dependent solvent reorganisation
  term of the ML droplet around the ghost. It is not defined or measured,
  so the total is not yet "ligand in gas + bulk solvent". The reference
  state to aim for is decision D18 (S12).

  **Decided (D18 (a), user 2026-10-16): bulk MM water.**
  - The path gets one more leg at the decoupled end: B -> B', the n inner
    waters ML -> MM, with FIRES kept.
  - A neat-solvent reference dG_ref (same box and n, FIRES around a
    non-interacting dummy atom) removes the ligand-independent part.

  Then
      dG_hyd = dG(A -> B) + dG(B -> B') - dG_ref,
  and dG(B -> B') - dG_ref is the reorganisation term above. It is reported
  as a global term (the dG_ref leg is a Claude addition, to confirm).
- **Path-only control (planned, 2026-10-16).** The same W1 | W2 | cut | W3
  path with every term MM must reproduce the standard MM decoupling of the
  same leg. Both ends must match the reference calculation: ligand
  internal MM, L-L and L-I MM with L-I scaled by lambda like L-O, and
  FIRES kept at n = 20 on MM inner waters. Then relabelling is a symmetry,
  FIRES is exact, and:
  - acetamide, +34.04 +- 0.14 kJ/mol (S13.44): same Hamiltonian, same
    endpoints;
  - this tests the wall, the refill and the path bookkeeping as a whole,
    which the local exact checks (S13.43, S13.46) do not;
  - if it agrees, any remaining E3D-vs-MM difference is the Hamiltonian
    (ML vs GAFF2, and the hybrid water above) plus window error; if not,
    the cycle's implementation is wrong, independently of the head.

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

**Route from here** (user, 2026-10-15). Solvent legs first; complex legs and
pocket water come later.

1. **Close the dynamics question.** At the suspect lambda = 0+ states
   (q 2.0, and the cut at q 2.5), run paired 2 fs / 1 fs NVE from one saved
   state. Check forces against the energy gradient on the full system. Log
   how often the D17 cap and the cover switch act. This decides whether the
   lambda = 0+ drift is integration error or the implementation (S13.42).
   **Done 2026-10-15 (S13.43)**: integration error, forces correct. The cap
   and the cover switch act in most lambda = 0+ samples.
2a. **Endpoint audit** (user, 2026-10-16, after the review of S13.47),
    before more GPU time on the full path:
    - fix the target reference state -- done: D18 (a), bulk MM water, via the
      B -> B' leg and the neat-solvent dG_ref;
    - run the path-only control with matched ends on acetamide (S7.2);
    - then fill the W2 / W3 window gaps.
2. **Wire the full calculation and measure its cost.** Put ML/MM, FIRES,
   W1, W2, the cut and W3 into the production sampling entry
   (`run_abfe.py --mode mlmm`, S6.3.4). Time the full step on the RTX 5090,
   then set windows and the sampling budget from that.
3. **Stage A on acetamide** (D8, S8.2).
   - First check its own configuration coverage: clusters, head, and
     `COVER_R` / wall radii recomputed from its training clusters.
   - Then check window overlap, independent replicates and convergence by
     block.
   - Key results:
     - the total agrees, within error, with a control path between the same
       endpoints;
     - the per-atom terms plus the cut and wall terms close to the total;
     - the cut share, the cap's firing rate and the sensitivity to the wall
       radius are all reported.
4. **After Stage A, a small set of neutral drugs** (Stage B). Different
   functional groups test whether the attribution transfers.

Solvent legs: acetamide was built here (S13.44; MM dG_decouple +34.04 +-
0.14 kJ/mol). Six neutral drug legs came from ABFE-IBS
(`REF_solvent_legs_from_ABFE_IBS_2026-10-15.md`, `data/solvent_legs_ibs/`).
They hold Na+/Cl-, and `mlmm.py` currently drops ligand-ion LJ and walls
only waters, so it must be fixed before they are used.

### 8.0 Stage 0 -- before writing the anchored code

Both cheap; both decide whether S3 is workable as written.

1. **Remainder spread.** `run_remainder.sh` -- **done 2026-10-10, S13.25: sd 3-13 kT,
   cut windows needed beyond n = 5**:
   `r = truth - sumD` over configurations of a fresh MM trajectory, for both
   20-frame heads, sd/kT per size. Few kT -> one BAR step; many kT -> cut
   windows (S3). Static configurations are a proxy for the lambda = 0+ ensemble,
   not the ensemble itself.
2. **POLAR-1-M ML/MM step time** -- ML part measured, S13.24 (121-331 ms);
   full ML/MM step measured at n = 20 on a 2080 Ti: 0.165 s, 0.30-0.32 s
   with the D17 cap (S13.41, S13.42); **on the RTX 5090: 91 / 119 / 200 ms
   at n = 20 / 50 / 100 (S13.45)**. Originally planned: a few hundred steps on the Atenolol leg at
   n = 20 / 50 / 100. Sets the cost of everything after.

### 8.1 Code (once S3, S2.1 and S2.3 are decided)

- `lambda_mask.py`: `anchored` mode, checks 1-5 of S3. **Done 2026-10-12**
  (`anchored_energy`; `test_anchored.py`, S13.29).
- `run_abfe.py`: TI over lambda > 0 on the diagonal path, BAR (or cut windows)
  at lambda = 0, per-atom accumulation. **Done 2026-10-12** (`--mode anchored`,
  one path coordinate p in [0, 2] over both segments, `AnchoredForce`; S13.29).
- The MM wall (W1/W3) and the ML/MM system with FIRES (D11), designed in
  S6.3 (D16): **code done 2026-10-14/15** (`decomp/fires.py`,
  `decomp/mlmm.py`; D17 in `lambda_mask.anchored_energy`), checks E0-E7
  (S13.38-S13.42). `run_abfe.py --mode mlmm` (the S6.3.4 ladder,
  `decomp/mlmm_hremd.py`): **done 2026-10-16**, exact checks and smoke in
  S13.46. Not yet: batching replicas into one ML call.
- S2.3 (decided): explicit `E_es^LE` channel with the per-atom Euler split,
  target subtraction in `test_joint.py`, then one retrain on the arm-N recipe.
  **Done** (S13.28, retrain S13.32); the head has since been retrained twice on
  new frames (S13.35, S13.37).

### 8.2 Stage A -- method validation with real sampling

A small solute in ML/MM water, sampling in minutes to hours. Answers what no
static scan can: softcore need (S11 R1), cut-step overlap, ML-water identity
(S6.3), lambda spacing, convergence against the message-masking control (S7.1),
attribution against Shapley (S7.3). System: **acetamide** [decided, D8, user
2026-10-15], a drug fragment in DES370K's chemistry, so the head is in domain.
Its solvent leg comes from the user. The checks and key results are in the
route at the top of S8.

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
the 5090 measurement. The full ML/MM step with the D17 cap is 0.30-0.32 s
at n = 20 on a 2080 Ti (S13.42), so the same leg at 100 ps is ~100 h at
1 fs or ~50 h at 2 fs per 12 windows. On the RTX 5090 it is 91 ms
(S13.45): ~15 GPU-h per 12 windows at 2 fs.

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
dependence on it. Stage A measures both. The wall alone was not enough:
the head's sum c overshot in the crowded lambda = 0+ shell and ran away
(S13.41); D17 (cover switch + backbone cap) bounds it, and E7 is stable
(S13.42).

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

**R7 - Cost.** Full ML/MM step on the RTX 5090 with the D17 cap: 91 / 119 /
200 ms at n = 20 / 50 / 100 (S13.45), i.e. ~1.3 GPU-h per 100 ps window at
n = 20. About half of the n = 20 step is fixed overhead, so batching
replicas is the next lever.

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
| D8 | Stage A system (S8.2): acetamide, solvent leg first; leg supplied by the user | **decided** | user, 2026-10-15 |
| D9 | Drug set composition (S8.3) | open | |
| D10 | Far field attributed via MM per-atom lambda (S6.2): the ligand's MM nonbonded terms with waters outside the ML region scaled by the same lambda_i, MM softcore | **decided** | user, 2026-10-14 |
| D11 | ML-water identity scheme (S6.3): FIRES first | **decided** | user, 2026-10-12 |
| D12 | One flagship paper (S10) | open | |
| D13 | Mark v2 S0-S12 as superseded in the file itself | open | |
| D14 | Endpoint: MM repulsive wall held through the ML decoupling, then refill; GCMC where water exchange is slow (S3 W1-W3) | **decided** | user, 2026-10-10 |
| D15 | Wall radius; GCMC waters ML or MM; GCMC environment (S3) | method agreed (user, 2026-10-12): radius from the MLP's non-physical short-range region and the training data's contact coverage. First radius (user, 2026-10-14): DES370K p1 per element pair (S13.31) -- **failed at lambda = 0+** (S13.41: ligand H crushed, r a well). **Revised** (user, 2026-10-15): per (ligand, water) element pair the larger of the operating head's training-cluster minimum contact and DES370K p1: C-Ow 2.69, C-Hw 1.83, N-Ow 2.62, N-Hw 1.69, O-Ow 2.46, O-Hw 1.54, H-Ow 1.66, H-Hw 1.39 A; W1 measured by FEP: +0.006 kJ/mol (S13.42); eps_w = 25 kJ/mol keeps lambda = 0+ contacts >= 0.92 r0 (S13.42); a second radius for Stage A's sensitivity check still open. GCMC parts open | user |
| D16 | ML/MM + FIRES + wall design (S6.3): partition, softmin ordering over water O x ligand heavy atoms (revised from all atoms after S13.38's 2 fs instability), C1 pairwise variant of FIRES eq 5, rigid waters, k from 500 kcal/mol/A^2, n = 20, wall form, path, checks E0-E7 | **decided** | user, 2026-10-14 |
| D17 | S3: bound the attributed coupling -- cover switch on D_ia (training-cluster contact coverage) and backbone cap on sum c (m = 0.5 eV) (S3, S13.41) | **decided** | user, 2026-10-15 |
| D18 | Reference state of the decoupled end (S7.2, S13.47): **bulk MM water**, option (a). B -> B': at the decoupled end, with FIRES kept, the n inner waters are switched ML -> MM. At B' relabelling is a symmetry and FIRES is exact, so B' = gas-phase ligand (ML internal) + bulk MM water. *Added by Claude 2026-10-16, to confirm:* this leg also carries the n waters' own ML-vs-MM free energy (the absolute energy offset of the ML water, plus the bulk chemical-potential difference). That part does not depend on the ligand, so it is removed by a neat-solvent reference leg dG_ref: the same box and n, FIRES around a non-interacting dummy atom, n waters ML -> MM. dG_hyd = dG(A -> B) + dG(B -> B') - dG_ref. The difference dG(B -> B') - dG_ref is the ghost-shape solvent-reorganisation term, measured explicitly and reported as a global term like dG_cut | **decided** (a); the dG_ref leg to confirm | user, 2026-10-16 |

---

## 13. Measurements

v2-S13.1-S13.23: `RESEARCH_PLAN_v2.md` S13, unchanged. New measurements below,
from S13.24.

Machines: workstations A, B and C (RTX 2080 Ti, smoke and timing checks only);
PBS nodes P1 and P2 (RTX 3090), P3 (RTX 4080) and P4 (RTX 5090, the
production GPU).

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

### 13.38 FIRES code: E4 checks pass; the paper's eq 5 has a force jump the pairwise form removes

`decomp/fires.py` (`Fires`: softmin closest-atom ordering, pairwise C1 variant
of eq 5, k = 500 kcal/mol/A^2, beta = 20 /A; S6.3.2, D16).
`experiments/2026-10-14_fires_mm/check_fires.py` on workstation A, CPU,
float64, `em_solvent.npz`, n = 20 (`check_fires.log`):

    [1] ordered frame (inner set selected on it)   U = 0, F = 0; R_inner 3.722 A, nearest outer 3.766 A
    [2] nearest outer water pushed 0.3 A inside    U = 173.6 kJ/mol, 3 violating pairs
        forces vs central FD (h 1e-4 A, 21 comp.)  max |d| 1.9e-5 kJ/mol/A (|F| up to 1,015)
    [3] same water moved radially through R_inner  max neighbour step of U and dU/ds halves when the
                                                   step halves (ratios 1.99, 2.00): C1
    [4] outermost inner water changes during a     pairwise: step halves (2.00) -- continuous;
        violation (radial force on it)             eq 5 (max): 315 kJ/mol/A jump, not shrinking (1.00)
    [5] box translation / rewrapping               |dU| 5e-12 / 0, max |dF| 2e-11 / 0
    [6] softmin vs true closest-atom distance      0.000-0.032 A over the nearest 60 waters (bound 0.24);
                                                   same 20 inner waters as by the true minimum
    [7] CPU float64, 8 threads                     7.6 ms per energy + force call

[4] is the measured reason for the pairwise form (D16): eq 5's force on the
outermost inner water jumps by k x (violation depth) when another inner water
overtakes it.

**E0 smoke** (`fires_mm.py`, workstation C (RTX 2080 Ti), 2 + 10 ps per run, seed 7,
`smoke*.log`). Pure MM with FIRES as an `openmm.PythonForce`:
- torch on CPU: 8.4 ms/step;
- torch on the GPU: 2.2 ms/step;
- MM alone: 0.24 ms/step.

The run is stable at 2 fs with k from the paper, T 301 K. U_F stays under
30 kJ/mol, with 0-4 violating pairs per snapshot. In the free run, the 20
waters nearest at t = 0 have spread to R_inner 13 A by 10 ps: without FIRES
the ML set would leave. The full E0 (3 ns per run) is not run yet.

**E0 first attempt: the FIRES run went NaN; diagnosed as a step-size
instability on H atoms** (2026-10-14, workstation C). In `run_e0.sh`, the
free run finished (3 ns, 0.31 ms/step, T 298.5 K, `e0_free.npz`). The FIRES
run hit a NaN during its 100 ps equilibration.

`diag_nan.py` reproduces it: NaN at 56.8 ps. Before that, the deepest
violation was 0.16 A and the largest FIRES force 521 kJ/mol/A, almost always
on an H atom.

Hypothesis: the closest-atom ordering puts the restraint on H (1 amu), and
several violating pairs stack on one atom. A harmonic mode is stable under a
Verlet-type step only while omega dt < 2. `diag_omega.py` tests this. Every
0.1 ps it takes the exact Hessian of U_F (autograd, 3 x 3 diagonal blocks)
for every atom that feels a FIRES force, and computes
omega = sqrt(lambda_max / m). Diagonal blocks, sampled every 0.1 ps, give a
lower bound.

    ordering metric (k = 500 kcal/mol/A^2, 2 fs, seed 7)   result            omega dt when violated: median / p99 / max   atom
    all atoms (D16 as coded)                               NaN at 63.2 ps    1.24 / 1.82 / 1.82                           H
    water O x ligand heavy atoms                           150 ps, no NaN    0.35 / 0.55 / 0.66                           C, N, O
                                                           (violated in 64% of snapshots; deepest 0.105 A)

The lower bound on the all-atom run sits at 1.2-1.8 throughout and the run
fails; on heavy atoms it is ~3.5x lower, as sqrt(m_C / m_H) predicts, and the
run is stable. The hypothesis is supported. **D16 revised** (user,
2026-10-14): the ordering now runs over water O x ligand heavy atoms
(`heavy_selection`). The heavy-atom inner set shares 17 of
20 waters with the all-atom set on em_solvent.npz.

Also noted, not investigated: the prepared leg's `PeriodicTorsionForce` has
0 torsions; the ligand carries 19 bonds, 22 constraints, 71 angles and 205
nonbonded exceptions. All MM frames (clusters, remainder scans) were sampled
with this torsion-free ligand. In ML/MM, ML supplies the ligand's internal
energy.

Also: files written on workstation C can show a stale size and mtime over
NFS here for minutes (a 11 kB log read as 0 bytes). Read remote outputs on
the node.

**E4 rerun on the heavy-atom ordering** (`check_fires_heavy.log`,
workstation A, CPU float64): all pass.

    [1] U = 0 on the ordered frame; R_inner 4.578 A
    [2] forces vs FD                5.0e-6 kJ/mol/A (|F| up to 2,074)
    [3] C1                          ratios 1.99, 2.00
    [4] pairwise                    continuous (ratio 2.00); eq 5 still jumps by 434 kJ/mol/A (ratio 1.00)
    [5] PBC                         translation 1e-11, rewrap 0
    [6] softmin offset              <= 0.034 A
    [7] time                        2.7 ms per CPU call

E0 was relaunched with the heavy-atom ordering for both runs. The free
run's d must use the same metric, so the all-atom outputs are kept as
`*_allatom*`.


### 13.39 ML/MM system code: E1, E2, E3, E5, E6 pass after one PBC bug in the exceptions

`decomp/mlmm.py` (`build_mlmm_system`, `MLMMForce`, `set_path`,
`mm_far_field_dudl`, `mm_far_field_per_atom_coulomb`; S6.3.6).
`experiments/2026-10-14_mlmm_checks/check_mlmm.py` on workstation A: MM on
the Reference platform (double precision), ML in float64 on the CPU, Ewald
tolerance 1e-7, operating head (S13.37), em_solvent.npz, n = 20
(`check_mlmm.log`).

    E1  MM(ours), lambda = 1, s_w = 0   vs MM(ref) - (A-A pairs removed, summed in numpy at minimum image)
                                        diff 1.2e-5 kJ/mol (removed -1,067 kJ/mol; A = L u I)
    E2  Coulomb vs lambda                quadratic, not linear: deviation 0.015 kJ/mol at lambda = 0.5
        per-atom sum (central diff.)     equals mm_far_field_dudl to 6e-10; that equals a FD of U to 2e-7
    E3  unwrapped L u I                  equal to minimum-image distances (0 over 6,263 pairs); extent 18 x 11 x 15 A
        p = 0 / p = 2                    E_bb(L u I) / E_bb(L) + E_bb(I) to 0 eV
    E5  wall, two particles              zero above r0; force step halves with the step (2.00);
                                         finite at r = 1e-4 A for s_w < 1; at s_w = 1, U(0.8 r0) = eps_w g(0.8)
    E6  total U at q = 1.5 (all terms)   unchanged (0) when the system, one inner water, or the ligand
                                         moves by a box vector

**Bug found by E6 and fixed.** NonbondedForce computes exceptions without PBC
by default. The new L-I and I-I exceptions join different molecules, and the
raw coordinates can place them a box apart. Before the fix, moving one inner
water by a box vector changed U by 0.68 kJ/mol, and moving the ligand changed
it by 3.1. The fix is `setExceptionsUsePeriodicBoundaryConditions(True)`.

**Two findings that changed the code.**
- **The L-O Coulomb is quadratic in lambda, not linear.** The A region's MM
  charges interact with their own periodic images 35 A away (1.45 kJ/mol in
  total, present in the reference MM too), and the ligand's share scales as
  lambda^2. So dU/dlambda uses an exact central difference.
- **Per-atom dU/dlambda_i cannot come from switching on one atom's charge.**
  That charges the cell: the naive per-atom sum came out at -321 kJ/mol
  against a total of -38. Central differences in lambda_i are exact,
  because the energy is quadratic in each lambda_i.

**E1 method note.** Comparing against "ref minus A alone in the same box"
leaves the A-A image term. Estimating that term by box scaling was swamped
by PME error at the default tolerance, and was not 1/L^3 at 35 A. The direct
pair sum closes it to 1e-5.

Not yet run: E7 (NVE continuity with ML on the GPU), the step time of the
full ML/MM system, and eps_w from a lambda = 0 run.

### 13.40 E0: FIRES leaves the solvent unchanged except a 0.2 A band at the boundary

`run_e0.sh` on workstation C (RTX 2080 Ti), 2026-10-14 18:38 to 19:52, pure
MM, heavy-atom ordering (D16), n = 20, k = 500 kcal/mol/A^2, 2 fs. Each run
is 100 ps of equilibration plus 3 ns, seed 7, sampled every 0.5 ps (6,000
snapshots); standard errors come from 5 blocks. FIRES ran at 2.68 ms/step
(67 min); the free run at 0.58 ms/step while sharing the GPU. Mean T is
298.4 / 298.6 K. The log is `e0_analysis.log`, read on the node.

    R_inner (FIRES) vs the 20th-smallest d (free)    3.696 +- 0.001 vs 3.687 +- 0.002 A;
                                                      quantiles 5/25/50/75/95 within 0.01 A
    n(d) per 0.1 A, 55 populated bins                 |z| > 3 in 2: 3.6-3.7 A +0.30 waters (+10%, z 12),
                                                      3.7-3.8 A +0.21 (z 5); all others within noise
    N(d < x), free vs FIRES                           3.0 A: 4.57 / 4.63; 3.5: 14.29 / 14.31; 4.0: 29.14 / 29.78;
                                                      5.0: 54.38 / 54.78; 6.0: 82.09 / 82.39
    FIRES restraint                                   violated in 66% of snapshots, 1.65 pairs (max 13);
                                                      depth median 0.028, max 0.117 A; <U_F> 1.28 kJ/mol (max 26)

**What it shows.**
- The boundary FIRES selects is the free shell's own 20th-water distance.
- The solvent structure is unchanged within noise at every d except a band
  at the boundary (median R_inner 3.69 A).
- In that band waters accumulate: ~0.5 extra waters at 3.6-3.8 A, taken
  from just outside. The excess within 4 A is 0.64; within 6 A it is down
  to 0.3, about 1 SE.

This is the paper's own observation (Fig. 2: deviations only where the
restraint acts) and the accumulation Bulo et al. reported. The run is stable
at 2 fs with the paper's k.

**For the free energy.** dG is exact for H + U_F at any k. The boundary
band measures how far H + U_F is from the ideal adaptive H_n. Whether that
matters for dG depends on whether the band and <U_F> change with lambda.
That is measured per window in E7 and in Stage A. SAFIRES's
elastic-collision integrator is the known fix, if that comparison asks for
it.

### 13.41 E7, first pass: continuous at lambda = 1 and 0.5; at lambda = 0+ the remainder pulls waters through the DES370K-p1 wall

`experiments/2026-10-15_e7_nve/run_e7.sh` on workstation B (RTX 2080 Ti),
2026-10-15. Each state starts from thermal200 frame 0, which also picks the
20 inner waters, with the operating head (S13.37) in float32, eps_w = 25
kJ/mol, alpha = 0.5. Langevin runs 500 steps, then NVE 1,000 steps at 2 fs.

    state            E_tot drift   sd     step sd / max   T       closest L-w contact r/r0   FIRES U_F max
                     kJ/mol/ps     kJ/mol kJ/mol          K       min / p1 / p50
    pure MM          +0.27         2.17   0.57 / 3.4      298.2   1.00 / 1.02 / 1.23         --
    q 1.0 (l = 1)    +0.05         2.46   0.79 / 4.6      302.1   1.18 / 1.18 / 1.26         14.8
    q 1.5 (l = 0.5)  +2.02         2.25   0.86 / 6.4      305.7   1.17 / 1.17 / 1.25         12.3
    q 2.0 (l = 0+)   NaN during Langevin

(The float32 resolution of E_bb at 6.6e4 eV is 0.75 kJ/mol, a floor for
every ML state's per-step jumps. The step time is 164-167 ms in Langevin;
the 345-353 ms in NVE includes one extra energy evaluation per step for the
record.)

At lambda = 1 and 0.5, E_tot behaves as in pure MM: the same sd, and a drift
of 0.05-2 kJ/mol/ps over 2 ps. The 1 fs comparison and q 2.5 / 3.5 were not
reached, because the driver stops on the first error.

**The q = 2.0 NaN, diagnosed** (`diag_q.py`, reproduced at step 150;
`diag_parts.py` on the last good state at step 140, float64):

                         truth      sum c = sumD + cross       r = truth - sum c   (kJ/mol, ML region)
    start (frame 0)      -125.5     -140.9 = -114.8 - 26.1       +15.4
    step 140             +1,902     +3,145 = +3,193 - 47         -1,243

- **Who is crushed.** The four largest c_i belong to ligand H atoms. Each has
  the closest water atom at 1.11-1.25 A, and 3-5 waters within 2.5 A.
- **Why.** The backbone sees strong repulsion there, and the head's pair
  term sees much more. At lambda = 0+, U = E_bb(L) + E_bb(I) + r, so a
  water that moves in lowers U: r is a well, and it deepens without bound.
- **Not the explicit Coulomb.** The cross term barely moves.
- **Why the wall did not stop it.** For ligand H the wall is thin: H-Ow
  1.32 A, H-Hw 1.18 A.

**Single-water remainder scans** (`experiments/2026-10-15_remainder_scan/scan_r.py`,
the S13.30 geometry, operating head with es, float64, 26 scans): r falls
1 kT below its starting value at d_min 1.0-1.66 A. The latest onset is
ligand H against water O at 1.66 A. Further in, r drops by hundreds to
thousands of kJ/mol, with inward pulls of 1e3-1e6 kJ/mol/A.

**Training-label coverage** (`cluster_contacts_lampool.py`: the operating
head's 2,474 ligand clusters on lampool400; shortest contact per cluster):

    pair   min    p1     |  DES370K p1 (the current r0)
    C-Ow   2.69   2.83   |  2.38
    C-Hw   1.77   2.00   |  1.83
    N-Ow   2.62   2.69   |  2.21
    N-Hw   1.69   1.78   |  1.39
    O-Ow   2.46   2.53   |  2.14
    O-Hw   1.54   1.60   |  1.32
    H-Ow   1.66   1.69   |  1.32
    H-Hw   1.39   1.46   |  1.18

**Reading.** At lambda = 0+ the only ML ligand-water term left is r, which
contains the backbone's truth. So r can be trusted only as far as the
backbone-labelled clusters reach. The dimers' CCSD(T) coverage does not help
there. The single-water onset (H-Ow 1.66 A) sits at the cluster minimum for
H-Ow (1.66 A). D15's first radius, DES370K p1, lies inside that range for
every pair except C-Hw. A revised radius is asked for (2026-10-15).

**Revised radii, same failure** (`diag_q.py`, q = 2.0, 2026-10-15, radii as
in the revised D15). NaN at step 140. By step 120 a ligand H was at
0.71 r0 from an inner water O (1.18 A) and T had reached 1,132 K. The MM
(wall) force on that H was 30,000 kJ/mol/A and sum c was +1,244 kJ/mol. The
wall pushes, but at eps_w = 25 kJ/mol the remainder's pull wins. A
much stiffer wall would bring back the 2 fs stability problem of S13.38, on
H again. Two patches in a row have failed, so the next step goes back to the
design (S3), which allows any c: see the question of 2026-10-15.

**The S3 cover switch** (user, 2026-10-15; `lambda_mask.cover_switch`,
`COVER_R` = the training-cluster minima above, delta = 0.2 A, smoothstep).
Each ligand-environment D_ia is multiplied by s(r_ia); the explicit Coulomb
is not switched. Checks (`check_cover.py`, float64):
- The start frame has no pair below r_cov, so sum c is unchanged.
- Forces vs FD on atoms with a pair in the switch zone: 3e-5 eV/A.
- dU/dp vs FD: to 1e-8.
- The endpoints p = 0 and 2 are bit-identical with and without the switch.

**It does not stop the runaway** (`diag_q.py`, q = 2.0, revised wall +
switch). The run reached step 320 before the NaN, against 140-150 before.
The contacts stayed at 0.93-0.98 r0, within or at the edge of coverage, yet
sum c rose 243 -> 524 -> 1,812 kJ/mol. At the last good state
(`diag_parts_cover.log`): truth +696, sum c +1,812 (sumD +1,820, cross -9),
r = -1,116 kJ/mol. A single ligand H carries c_i up to 217 kJ/mol, with 4-6
waters within 2.5 A and the nearest at 1.39-1.42 A.

**Mechanism.** At lambda -> 0+ the ligand barely attracts water, so water
packs against the wall all round it. That crowded many-body environment is
outside the head's training, and its c overshoots the backbone's own
coupling. Since U = E_bb - (1 - lambda) sum c, every overshoot lowers U and
packs the shell tighter. A per-pair distance switch cannot catch this.
Asked 2026-10-15: bound the attributed coupling (S3).

**The S3 backbone cap** (D17, user 2026-10-15; `anchored_energy(cap=m)`,
m = 0.5 eV, tau = 0.025 eV, on top of the cover switch).

Checks (`check_cap.py`, float64):
- The start frame is unchanged: -140.95 kJ/mol.
- On the two crushed states, r goes from -477 and -196 kJ/mol to -48.2
  kJ/mol = -m.
- Forces vs FD with the cap active: 1.0e-4 eV/A (|F| 2.3).
- dU/dp vs FD: to 2e-8.
- The endpoints are bit-identical.

**Dynamics** (`diag_q.py`, q = 2.0, revised wall + switch + cap): stable
for 1,000 steps (2 ps), the end of the test. T stayed at 296-308 K and the
closest contact at >= 0.96 r0. Sum c peaked at 373 kJ/mol and relaxed back
to about 0. The step took 0.36 s with diagnostics, against 0.165 s before
the cap; a clean timing comes from E7. E7 was rerun with all three changes
(earlier outputs kept as `*_v0`).


### 13.42 E7 rerun (revised wall + cover switch + backbone cap): every state stable; lambda = 0+ drifts +4-6 kJ/mol/ps

`run_e7.sh` on workstation B (RTX 2080 Ti), 2026-10-15 10:11-11:59, with
the revised D15 radii, the cover switch and the backbone cap (D17,
m = 0.5 eV), eps_w = 25 kJ/mol, alpha = 0.5. The setup is as in S13.41:
thermal200 frame 0, Langevin for 500 steps, then NVE for 1,000 steps at
2 fs. The q 1.5 dt1 state runs 1,000 + 2,000 steps at 1 fs. Summary:
`e7_summary.py` -> `e7_summary.log`.

    state            dt   E_tot drift  sd      step sd / max  T      closest L-w r/r0   FIRES U_F max,
                     fs   kJ/mol/ps    kJ/mol  kJ/mol         K      min / p1 / p50     violations max
    pure MM          2    -0.36        2.15    0.59 / 3.5     302.0  0.97 / 0.98 / 1.06  --
    q 0.0 (W1 s_w 0) 2    +2.46        2.08    0.83 / 4.9     304.3  0.98 / 0.99 / 1.07  14.5, 6
    q 1.0 (l = 1)    2    +2.75        2.25    0.84 / 4.5     302.3  1.00 / 1.00 / 1.10   6.9, 10
    q 1.5 (l = 0.5)  2    -0.13        2.17    0.78 / 3.6     300.6  0.99 / 1.01 / 1.07   8.8, 9
    q 1.5 (l = 0.5)  1    -0.27        1.97    0.60 / 3.2     302.9  0.99 / 1.00 / 1.10   7.6, 6
    q 2.0 (l = 0+)   2    +5.85        3.17    0.87 / 5.7     302.5  0.92 / 0.93 / 0.97   7.9, 5
    q 2.5 (cut)      2    +3.89        2.17    0.83 / 6.3     301.5  0.93 / 0.93 / 0.97  17.4, 6
    q 3.5 (W3 s_w .5) 2   +0.32        2.05    0.74 / 4.4     302.5  0.80 / 0.81 / 0.90   8.9, 7

(r/r0 uses the revised wall radii in every state, pure MM included. The
contacts at lambda = 1 sit at 1.00 r0 because the revised radii are at the
edge of the 298 K contacts.)

- **No NaN in any state**, including lambda = 0+ (q 2.0) and the cut
  (q 2.5), which failed before the cap (S13.41).
- **The sd and the step jumps are at the MM level.** The sd is 2.0-2.2
  kJ/mol in every state except q 2.0 (3.2, 1.5x MM). The per-step sd is
  0.74-0.87 against 0.59 for MM, and the largest step is 6.3 against 3.5.
  The float32 floor of E_bb (0.75 kJ/mol) is of the size of that excess.
  There is no step of tens or hundreds of kJ/mol, which a jump in U would
  give.
- **The dt test cannot tell here.** At q 1.5, going 2 fs -> 1 fs changes
  the sd by 1.11x and the per-step sd by 1.31x, not the ~4x of an
  integration error. The fluctuation is set by something that does not
  scale with dt. The pure-MM sd is the same size, so it is not ML-specific.
  The cause has not been looked at (candidates: SETTLE / constraint
  tolerance, PME, the float32 floor). Pure MM was not run at 1 fs.
- **Drift.** Re-running a state shows how noisy a 2 ps drift is: q 1.0 went
  +0.05 -> +2.75 and q 1.5 +2.02 -> -0.13 kJ/mol/ps between S13.41 and here
  (different radii, but a stable state), and the two halves of one run
  differ by up to 3.4 (q 1.5: +0.9 / -2.5). So |drift| <= ~3 kJ/mol/ps is
  within run-to-run noise. q 2.0 and q 2.5 (lambda = 0+) lie above it.
  - q 2.0: +5.9 kJ/mol/ps; both halves are positive (+9.9 / +6.1);
    E_tot(2 ps) - E_tot(0) = +11.3 kJ/mol.
  - q 2.5: +3.9 kJ/mol/ps (halves +1.6 / +6.9); end - start = +11.3 kJ/mol.

  Over 2 ps that is 11 kJ/mol in a 4,076-atom box. Under the Langevin
  thermostat of production it is absorbed. The cause is not identified. The
  candidates are the stiff parts that are active only at lambda = 0+: the
  smooth-min of the cap (tau = 0.025 eV), the cover switch (0.2 A
  smoothstep) and the wall, against which the waters press (0.92 r0). A
  1 fs run at q 2.0 would separate integration error (drift / 4) from a
  non-conservative force (drift unchanged). Not run.
- **eps_w.** At lambda = 0+ the closest contact stays at >= 0.92 r0 (p1
  0.93) with eps_w = 25 kJ/mol, so 25 meets the S6.3.3 criterion
  (>= ~0.9 r0). Smaller values were not tried, so it is not shown to be the
  smallest. One 2 ps run from one start frame.
- **W1 by FEP from q 0.0** (101 samples, every 20 fs of the 2 ps NVE):
  dG_W1 = +0.006 kJ/mol. 4.0% of the samples have U_wall(s_w = 1) > 0, and
  the largest is 0.29 kJ/mol (0.12 kT), so one FEP step has full overlap.
  At lambda = 1 with the revised radii, W1 is ~0 after all. These samples
  are correlated, and this is a single start.
- **FIRES** holds in every state: U_F <= 17.4 kJ/mol, at most 10 violating
  pairs at any sample.
- **Step time** (2080 Ti, smoke only; production is the RTX 5090): the
  Langevin step is 303-312 ms in every ML state, against 164-167 ms before
  the cap (S13.41). The cap costs about +140 ms per step for the two extra
  fragment backbone calls. The NVE step (633-656 ms) includes the
  per-step energy record. Pure MM is 0.2 ms.

**q 2.0 at 1 fs** (workstation C, `e7_nve.py 2.0 e7_q2.0_dt1.npz 1000 2000
1.0`, same start and seed, 2026-10-15). It separates integration error
(the excess shrinks with dt) from a non-conservative force (it does not):

    state          dt   drift      halves         end - start   sd     step sd / max   contact min
                   fs   kJ/mol/ps  kJ/mol/ps      kJ/mol        kJ/mol kJ/mol          r/r0
    q 2.0          2    +5.85      +9.9 / +6.1    +11.3         3.17   0.87 / 5.7      0.92
    q 2.0          1    +1.83      +3.0 / +4.0     +6.0         1.92   0.51 / 3.7      0.91
    q 1.5 (ref)    2    -0.13      +0.9 / -2.5     -1.9         2.17   0.78 / 3.6      0.99
    q 1.5 (ref)    1    -0.27      +0.1 / -0.9     +2.4         1.97   0.60 / 3.2      0.99

- At 1 fs the q 2.0 sd (1.92) is back at the MM level. 2 fs -> 1 fs shrinks
  the q 2.0 sd 1.65x, against 1.11x at q 1.5, so the q 2.0 excess scales
  with dt.
- The drift falls 3.2x (5.85 -> 1.83), near the dt^2 ratio of 4. At 1 fs it
  is inside the run-to-run noise (~3 kJ/mol/ps).
- **Reading.** This points to integration error from a stiff term active at
  lambda = 0+, not a non-conservative force. It is not conclusive: one run
  per dt, and the drift difference (4 kJ/mol/ps) is about the size of the
  noise. Which term is stiff (the cap's smooth-min, the cover switch, or the
  wall) is not separated.
- Step time on workstation C's 2080 Ti: 321 ms Langevin, 664 ms NVE (with
  the record), the same as workstation B.

### 13.43 Route step 1: the lambda = 0+ drift is integration error, the forces are right; the cap and switch act all the time at lambda = 0+

`experiments/2026-10-15_lambda0_dt/` (`paired_dt.py`, `fd_check.py`,
`run_step1.sh`), workstation B, 2026-10-15. Production settings as in
S13.42. One Langevin run gives one state: thermal200 frame 0, 500 x 2 fs,
seed 7. From that same state, NVE runs 2 ps at 2 fs and at 1 fs. Records
every 10 fs.

    state         dt   drift kJ/mol/ps (halves)   end-start  sd     T
    q 2.0 (l 0+)  2    +4.13 (+6.70 / -0.50)       +4.27      2.77   298.4
                  1    +0.93 (+0.45 / +0.96)       +2.24      1.72   300.1
    q 2.5 (cut)   2    -2.91 (-8.98 / +3.36)       -3.35      2.81   301.9
                  1    +0.57 (+0.85 / +3.59)       +1.35      2.04   301.3

- At q 2.0 the drift falls 4.4x and the sd 1.6x from 2 fs to 1 fs.
- At q 2.5 the 2 fs drift changes sign between halves, so it is within
  the noise; the sd falls 1.4x.
- With S13.42 (unpaired, q 2.0: 5.85 -> 1.83), both comparisons point the
  same way: the lambda = 0+ excess scales with dt.

**Forces against the energy gradient** (`fd_check.py`). Float64 throughout
(model on the CPU, OpenMM CUDA double). Frames: three per state from the 2 fs
run (largest cap excess, most switch-zone pairs, closest contact). On each,
the 8 atoms of the pairs with the smallest r / r0, x/y/z, central
differences at h = 1e-3 and 5e-4 A.
- ML + FIRES group: max |F - FD| 4.5-5.8e-3 (h 1e-3) -> 1.2-1.4e-3 (h
  5e-4) kJ/mol/A, with |F| up to 202. The ~4x drop is the h^2 truncation.
- Wall group: 1.7-6.5e-3 -> 0.4-1.6e-3, with |F| up to 231.
- Total U: the same, except 5 of 12 (frame, h) cases at 4.88 kJ/mol/A,
  doubling to 9.75 when h halves in one frame. That is a fixed energy step
  of 4.88 x 2h = 0.0098 kJ/mol, outside the ML and wall groups.
  - Its size is that of one water O-H pair crossing the 1.0 nm cutoff of
    the PME direct sum without switching: 0.834 x 0.417 x 138.9 x
    erfc(alpha r_c) ~ 0.0094 kJ/mol at tolerance 5e-4.
  - The prepared MM leg has no switching, so pure MM has the same steps.
    They may be the dt-independent NVE floor of S13.42 (pure MM sd 2.1);
    not tested.
- **Reading**: the forces of every ML/MM-specific term (ML, cap, cover
  switch, FIRES, wall) are the gradient of the energy. The lambda = 0+ drift
  at 2 fs is integration error from stiff terms, not an implementation error.

**The D17 terms at lambda = 0+** (every 10 fs, both runs per state):

    state   cap excess kJ/mol p50 / p90 / max   samples > 1 kT   sum c p50   truth p50   samples with a pair in the switch zone (max pairs)
    q 2.0   2.4-2.7 / 10.0-10.5 / 23-29          48-52%           207-233     161-185     83% (7-9)
    q 2.5   0.0-0.3 / 3.2-7.6 / 21-32            13-27%           315-417     287-372     96-99% (10-13)

- At lambda = 0+, sum c sits about m (48 kJ/mol) above truth.
- About half the samples at q 2.0 lose more than 1 kT to the cap.
- Nearly every sample has ligand-water pairs inside the cover switch,
  because the waters press on the wall and `COVER_R` ~ `WALL_R0`.
- What this means:
  - D17 is not a rarely active guard at lambda = 0+; it reshapes c there.
  - What it removes goes to dG_cut, not to atoms.
  - The head overshoots in the confined lambda = 0+ shell. That is the
    case for clusters from real ML/MM window frames (STATUS item 4).
  - The cap's firing rate is a Stage A report item (S8 route).
- Step time with these records: Langevin 508 ms, NVE 557-614 ms per step,
  two chains sharing one 2080 Ti.

### 13.44 Acetamide solvent leg (Stage A, D8) and its MM hydration free energy

`experiments/2026-10-15_acetamide_leg/`.

**The leg** (`prep_acetamide.py`, workstation C, 2026-10-15).
- Why built here: ABFE_IBS builds a solvent leg only from a complex, so
  it was built directly in OpenMM.
- Settings, matching the Atenolol leg: PME, 1.0 nm cutoff, Ewald
  tolerance 5e-4, dispersion correction, no switching, HBonds + rigid
  TIP3P, standard masses, no ions; 298.15 K.
- Ligand: GAFF2 (gaff-2.11) with AM1-BCC charges (openmmforcefields +
  AmberTools sqm), net charge 6e-17 e, residue MOL, first in the topology.
- Box: 875 waters, 2,634 atoms. NPT 1 ns (1 bar) -> mean volume of the
  second half 26.64 +- 0.21 nm^3 -> fixed cubic L = 29.87 A -> 100 ps NVT.
- Files: `system_solvent.xml`, `topology_solvent.cif`, `em_acetamide.npz`
  (E3D leg format).

**MM dG_decouple** (`abfe_mm.py`, `run_mm.sh`, workstation A, 2026-10-15/16,
3 independent reps in parallel on one 2080 Ti, 4.6 h).
- Decoupling (couple-intramol = no).
  - Electrostatics: ligand charges x lambda_elec by NonbondedForce
    offsets; ligand-ligand pairs kept by exceptions.
  - Sterics: ligand-water LJ by a Beutler softcore (alpha 0.5) with the
    long-range correction.
  - At lambda = 1 it matches the reference System to 0.01 kJ/mol, forces
    to 0.0014 kJ/mol/nm.
- 11 electrostatic + 20 steric states; 100 ps equilibration + 1 ns per
  state, sampled every 1 ps. MBAR after subsampling each state by its
  statistical inefficiency.

    rep    dG_decouple       elec      vdW      samples/state   min adjacent overlap
    1      +33.90 +- 0.24    +41.45    -7.55    436-1000        0.087
    2      +34.05 +- 0.23    +41.45    -7.40    519-1000        0.057
    3      +34.18 +- 0.24    +41.75    -7.57    465-1000        0.090
    mean   +34.04 +- 0.14 (SD over reps)  +41.55 +- 0.17  -7.51 +- 0.09   (kJ/mol)

- **What the number is for.** It is the MM control for Stage A's total
  (S7.2): a comparison, not a gate, since the Hamiltonians differ.
- The rep-to-rep SD (0.14) agrees with the per-rep MBAR error (0.23-0.24).
- Experiment: dG_hyd = -9.71 kcal/mol, i.e. +40.6 kJ/mol in the
  decoupling sign (FreeSolv) [not verified; from memory]. GAFF2 / AM1-BCC
  is 6.6 kJ/mol less favourable.
- **Differences from the IBS solvent legs**
  (`REF_solvent_legs_from_ABFE_IBS_2026-10-15.md`): 298.15 K, not 300 K;
  standard masses, not HMR; no salt; plain softcore MBAR, not the IBS
  26-state stage 2.
- Speed: 177-180 ns/day per rep, with three reps on one GPU and the u_kn
  evaluation included.

### 13.45 Full ML/MM step time (route step 2): RTX 5090 91 / 119 / 200 ms at n = 20 / 50 / 100

`experiments/2026-10-16_5090_timing/time_step.py`, workstation C
(RTX 2080 Ti, smoke only), 2026-10-16. Production defaults of
`build_mlmm_system` (revised D15 radii, cover switch, cap, eps_w 25); head in
float32 on the GPU; LangevinMiddle at 2 fs; 50 timed steps after 20 warm-up
steps, with no energy records. "ML call" is one `MLMMForce.evaluate`
(ML + FIRES) on its own.

    case                        ms/step   ns/day   ML call ms
    pure MM (4,076 atoms)        0.20     850      --
    n 20 (101 ML atoms), q 1.5   322      0.54     335
    n 20, q 2.5 (cut)            319      0.54     323
    n 50 (191 ML atoms), q 1.5   465      0.37     472
    n 50, q 2.5                  471      0.37     477

The step is the ML call, and the MM part does not show. The cut state costs
no more than W2, because the cap already evaluates both fragments.

**RTX 5090** (PBS node P4, the production GPU, 2026-10-16, same
script, 200 timed steps, OMP 8 threads; the user's run):

    case                        ms/step   ns/day   ML call ms   vs 2080 Ti
    pure MM                      0.09     1,889    --           2.2x
    n 20 (101 ML atoms), q 1.5   90.7     1.91     99.8         3.5x
    n 20, q 2.5                  90.9     1.90     97.4         3.5x
    n 50 (191 ML atoms), q 1.5   119.2    1.45     124.6        3.9x
    n 50, q 2.5                  118.9    1.45     126.1        4.0x
    n 100 (341 ML atoms), q 1.5  200.4    0.86     206.6        --
    n 100, q 2.5                 200.7    0.86     207.7        --

- **Scaling with the ML region.** 101 -> 191 -> 341 ML atoms (x1.9, x3.4)
  costs x1.31 and x2.21 in time. A line through n = 20 and 100 gives
  ~45 ms per step fixed plus ~0.46 ms per ML atom; n = 50 sits 13 ms below
  it.
  - So at n = 20 about half the step does not scale with the region: graph
    build, Python, and three backbone calls (full, plus two fragments for
    the cap), each small.
  - At these sizes the GPU is not filled by one replica. Batching several
    HREMD replicas into one ML call is therefore the first speed lever for
    `--mode mlmm`. Not measured.
- **Budget at n = 20**: 1.9 ns/day per replica. 100 ps per window at 2 fs
  is 1.26 GPU-h; a ~20-window solvent leg (W1, ~12 W2, 4-8 cut, a few W3)
  is ~25 GPU-h run one replica at a time. Acetamide has fewer ML atoms
  (9 + 60) and was not timed.

### 13.46 `run_abfe.py --mode mlmm`: the full path on one HREMD ladder; exact checks and smoke

`decomp/mlmm_hremd.py` (new) and `run_abfe.py --mode mlmm`, 2026-10-16
(route step 2).

**How it works.**
- One OpenMM replica per path state q (W1 | W2 | cut | W3, S6.3.4).
  - Each replica is built from the same start frame, so the inner waters
    are the same in all of them; the build checks this.
  - Neighbour swaps exchange positions and velocities.
- One ML forward per configuration gives the energy at every state. E_bb,
  the capped S_eff = sum c and the fragment sum U0 do not depend on q:
  - W1: U_ML = E_bb.
  - W2: E_bb - (q - 1) S_eff.
  - cut: (1 - s)(E_bb - S_eff) + s U0.
  - W3: U0.
- The MM part per configuration:
  - NonbondedForce: three PME evaluations, because it is exactly quadratic
    in the common lambda;
  - L-O LJ: linear in lambda;
  - wall: one evaluation per distinct s_w.

  So a full u_kn row costs one ML forward, and the swaps use the same rows.
- dU/dq per segment, in closed form, from the same parts. Boundary states
  carry both one-sided values.
- Per-atom terms:
  - W2: the ML c_i, the MM Coulomb per atom (exact central differences)
    and the L-O LJ pair sum per atom. The LJ long-range correction is split
    by sum 4 eps sigma^6; the total is OpenMM's.
  - W1 / W3: the wall's dU/ds_w per atom.
- The evaluator context runs at double precision (MM only).
- Checkpoint every iteration; a rerun resumes.
- Analysis: TI per segment, per-atom TI, and MBAR over all states.

**Exact checks** (`experiments/2026-10-16_mlmm_hremd/check_mlmm_hremd.py`).
Float64 throughout (model on the CPU, OpenMM CUDA double), 12 states. Two
frames: thermal200 frame 0, and the q 2.0 frame with the largest cap excess
from S13.43.
- [1] all_states against the full Context energy at each q:
  1.1-1.4e-5 kJ/mol, with |U| ~ 6.4e6.
- [3] Per-atom sums against their totals: <= 2.4e-4 kJ/mol over 11
  (state, segment) pairs on each frame, the cap active on the second.
- [2] dU/dq against a finite difference in q:
  - W1, cut and W3: <= 5.8e-3 kJ/mol.
  - W2: 0.8 kJ/mol with a one-sided h = 1e-4. That miss is the noise of
    the double-precision PME energy (~5e-5 kJ/mol over h).
    `diag_w2_fd.py`: the NB-group slope converges 23.99 -> 24.76 ->
    24.79 for h = 1e-4 / 1e-3 / 1e-2. The total converges to the closed
    form, 253.914 vs 253.918. The ML and LJ slopes are exact at every h.

**Smoke** (workstation C, 2080 Ti, float32). Atenolol, 7 states
(--w1 2 --w2 3 --cut-windows 2 --w3 2), 1 + 2 iterations of 10 steps.
- It runs, checkpoints and resumes (`resumed at iteration 2`). MBAR runs
  with pymbar's JAX on the CPU; on the GPU it ran out of memory next to
  torch.
- Per-atom + cut = total: 0.000 kJ/mol.
  - Before that fix it was -0.2: S_eff was taken as E_bb - U_plus, which
    float32 resolves to 0.75 kJ/mol. Now it is sum_c.
- The free energies of this smoke (20 fs per state) mean nothing.
- One iteration of 7 replicas x 10 steps took 26-31 s.
  - Stepping is ~22 s at 0.32 s per step.
  - The rest is the u_kn rows and the per-atom terms, about 0.6-1.2 s per
    replica.

### 13.47 Pilot ladder on the RTX 5090 (Atenolol, n = 20): two gaps, at lambda -> 0 and at the end of W3

`run_abfe.py --mode mlmm` on PBS node P4 (RTX 5090), 2026-10-16
13:47-17:47, the user's run.
- Ladder: 26 states (--w1 2 --w2 12 --cut-windows 8 --w3 6).
- Sampling: 4 + 20 iterations of 250 steps x 2 fs, so 10 ps per state
  after equilibration and 20 samples per state.
- Start: thermal200 frame 0, operating head.
- Diagnostics: `experiments/2026-10-16_mlmm_pilot/pilot_analysis.py` ->
  `pilot_analysis.log`.

**Cost.** An iteration took 587-595 s, and 26 x 250 x 91 ms = 592 s. The
u_kn rows, derivatives and per-atom terms add almost nothing. The step
time is as in S13.45.

**Overlap.** The MBAR nearest-neighbour overlap is 0.13-0.33 for 23 of the
25 pairs, and swaps are accepted on all of those. Two pairs fail:

    pair                         swaps   NN overlap   reverse work mean / sd (kT)
    q 1.909 - 2.000 (W2, l 0.09 -> 0)    0/15    0.020        +7.5 / 3.4
    q 3.800 - 4.000 (W3, s_w 0.2 -> 0)   0/9     0.000      +153 / 12.6

**<dU/dq> profiles** (kJ/mol per unit q; naive SE from 20 samples):
- W2 falls smoothly from +315 (lambda = 1) to +82 (lambda = 0.09), then
  jumps to -204 +- 21 at lambda = 0.
  - At lambda = 0 the waters press on the wall. The capped coupling sum c
    is then repulsive, about +200 kJ/mol (cf. S13.43), so the integrand
    turns over in the last 0.09 of lambda.
  - That interval carries 41% of the W2 variance, and its trapezoid is not
    reliable.
- cut: +45 down to -57 at s = 1. Smooth, and the overlap is fine
  throughout.
- W3: -17 to -76 at s_w = 0.2, then -833 +- 16 at s_w = 0.
  - That last interval alone gives (-76 - 833) / 2 x 0.2 = -91 of the
    -115 kJ/mol.
  - At s_w = 0 water fills the cavity, and those configurations cost
    +153 kT at s_w = 0.2: the alpha = 0.5 softcore does not bridge the
    last step.

**Totals of this pilot** (not results: two gaps, 10 ps per state).

    dG_W1 +0.16   dG_W2 +160.66   dG_cut +23.40   dG_W3 -115.26   ->  TI +68.96, MBAR +59.96 kJ/mol
    per-atom: W1 +0.16, W2 ML +62.66, W2 MM Coulomb +15.38, W2 MM LJ +82.61, W3 wall -115.26
    per-atom sum + cut = TI total to 0.000

- TI and MBAR differ by 9 kJ/mol, which is the two gaps.
- The cut is 23 of ~65 kJ/mol, about a third of the total, and is not
  attributed. This is the Stage A report item. Part of it is the cap's
  excess (S13.43).
- The per-atom MM Coulomb ranges from -54 to +64 kJ/mol over atoms and sums
  to +15: the charged groups cancel.

**What this sets for the production ladder:**
- W2 needs states packed towards lambda = 0.
- W3 needs states packed towards s_w = 0, or a softer wall softcore, or
  both.
- W1 (overlap 0.33, +0.16 +- 0.16) and the cut (>= 0.15) are adequate as
  spaced.

**Against the MM number** (2026-10-16). The only MM reference for this leg is
ABFE_IBS's `Atenolol-rank11/output/final_binding_results.json`:
`solvent_delta_G_kJ_mol` = +152.05 [not verified]. It is an old run, and its
own diagnostics flag `stage2_lambda_endpoints.ok = False`. The pilot gives
+60 (MBAR) to +69 (TI), about 85 kJ/mol less.

Is the cycle wrong? This run cannot say. The local checks hold:
- every segment's energy and derivative matches direct evaluation (S13.43,
  S13.46);
- the per-atom terms close.

The review of 2026-10-16 tightened two claims made here earlier. The
decoupled end is not a defined reference state (S7.2: hybrid water
partitioned around a ghost ligand that FIRES still couples to the waters).
And "anchored is exact" covers the head, not the endpoint or n (S3 scope
note). So the 85 kJ/mol can come from four places, which this run cannot
separate:
1. **Numerics**: the two gaps above. The last W3 interval alone gives -91
   kJ/mol, and that trapezoid is not reliable. MBAR's error is NaN.
2. **The endpoint**: the undefined solvent-reorganisation term of the ML
   droplet around the ghost (S7.2, D18).
3. **The Hamiltonian**: ML vs GAFF2/TIP3P near the ligand.
4. **The implementation of the path as a whole**: wall, refill and FIRES.

The +60 to +69 is a diagnostic output, not a result. It neither supports
dropping the anchored scheme or the backbone, nor reads as "ML vs GAFF2
physics".

The order agreed with the user (2026-10-16), one check at a time:
1. Fix the target reference state (D18). Done: (a), bulk MM water (S7.2).
2. Run the path-only control with matched ends (S7.2), which decides
   item 4.
3. Pack states into the two gaps, judged by overlap, independent repeats
   and block convergence, which decides item 1.
4. Only then discuss the total and the per-atom terms.

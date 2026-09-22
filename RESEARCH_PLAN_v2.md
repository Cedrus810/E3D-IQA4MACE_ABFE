# Atom-Resolved Binding Free Energy from Physically Gauged Equivariant Potentials

**Version 2.1.** Supersedes `PROJECT_PLAN.md — OMol25-scale E3D-IQA...`.
Status: implementation started, two experiments run (S13). Last revised 2026-09-20.

---

## 0. What changed from v1, and why

v1 framed the project as "scale E3D-IQA to OMol25, then do ABFE." That framing
has two problems: it makes the flagship deliverable (ABFE) depend on the
scaling work, and it competes with conventional ABFE on a metric
(binding affinity accuracy) where a young method cannot win.

v2 reframes around what the decomposition uniquely enables:

> **A rigorous, order-independent, per-atom and per-residue decomposition of the
> binding free energy, at quantum-mechanical accuracy, for the cost of a single
> ABFE calculation.**

Conventional methods cannot deliver this:

| Method | Per-atom dG | QM accuracy | Polarization / charge transfer |
|---|---|---|---|
| MM-GBSA per-residue | yes, but unreliable | no | no |
| MM ABFE (per-group decoupling) | yes | no | no |
| Plain `sum_i E_i` MLIP | **no** (see S2) | yes | yes |
| **This work** | **yes** | yes | yes |

Concrete changes from v1:

1. **New deliverable framing** (S1): atom-resolved dG, not "better ABFE."
2. **Aumann-Shapley diagonal path** (S6) replaces v1's sequential atom removal.
   Same cost as one ordinary ABFE; attribution is provably order-independent.
3. **Architecture-agnostic decomposition head** (S4.1) replaces the Allegro
   fork. Deliverable is an `e3nn` module usable on Allegro / NequIP / MACE.
4. **Second gauge constraint** (S4.3): fragment interaction energies pin
   `sum D_LE` directly. IQA alone does not constrain the quantity alchemy uses.
5. **lambda masks the graph, not just edge energies** (S5.2). v1's requirement
   that ligand-internal energy be exactly lambda-invariant is dropped: it is an
   aesthetic preference, not a thermodynamic requirement, and keeping it makes
   the lambda=0 endpoint wrong.
6. **Body-order generalization of lambda** (S5.3), required for MACE.
7. **Message-masking baseline** (S7.1) as the primary control. It gets endpoints
   right for free; the decomposition must beat it on *path quality* and is the
   only route to *attribution*.
8. **Two single points per IQA structure** (S3.3) resolves the functional
   mismatch between OMol25 labels and AIMAll-compatible IQA.
9. **Phase 0** (S8.0): three cheap experiments that can kill the project in
   under a month. All were buried at step 5-9 of v1's linear plan.
10. **Kill criteria** alongside every go criterion (S9).

### Changes in v2.1, from measurements (S13)

11. **WP5 deleted.** Training from scratch on OMol25 is not needed. The gauge can
    be imposed on a frozen pretrained backbone, and the result *saturates* in
    w_I, so the limit is not supervision strength. Was the single most expensive
    work package.
12. **WP1 rewritten.** The original code is not published and Allegro's E3D-IQA
    variant does not exist publicly, so there is nothing to reproduce from. The
    project is a from-scratch e3nn reimplementation; MACE is the backbone.
13. **L_int promoted from "new component" to the top priority**, on evidence
    rather than argument: D_ij is not identifiable from total energy plus node
    supervision. Training 5x longer improves total energy 11x and E_intra 3.3x
    and D_ij 1.09x, leaving it worse than predicting zero (S13.2, S13.3).
14. **Loss terms are normalised by target scale** (S4.4). Copying the paper's
    w_I = 0.1 across label scales silently starves the IQA term.

---

## 1. Core claim

Given an equivariant potential whose energy is decomposed as

    E = sum_i E_i^intra + sum_{i<j} D_ij

with the decomposition pinned to physical observables (not merely a
reparameterization of the total energy), the ligand-environment coupling

    D_LE = sum_{i in L} sum_{a in E} D_ia

is a per-atom-resolved quantity. Scaling it by per-atom couplings lambda_i and
integrating along the diagonal path yields a decomposition of the decoupling
free energy into atomic contributions that sum exactly to the total and do not
depend on any removal order.

The three ingredients, in dependency order:

    physically gauged decomposition  ->  alchemical Hamiltonian  ->  atom-resolved dG
            (S4)                              (S5)                      (S6)

---

## 2. Why a plain atomic-energy MLIP cannot do this

Most equivariant MLIPs write `E = sum_i E_i`. Because message passing mixes
neighborhood information, `E_i` contains intramolecular, intermolecular,
electrostatic and polarization contributions with no unique split. Multiplying
a ligand atom's `E_i` by lambda therefore does not correspond to removing that
atom's interaction with the environment; it also removes part of the ligand's
internal energy, and does so differently for each atom. Sequential atom removal
compounds this: ligand-internal terms are removed repeatedly or migrate between
steps.

The requirement is **decoupling**, not deletion. That requires the
ligand-environment coupling to exist as a separately addressable term, which
requires the decomposition.

---

## 3. Reference data

### 3.1 Energy / force supervision (large)

OMol25, wB97M-V/def2-TZVPD, ORCA 6. Retain atomic numbers, coordinates, total
charge, spin multiplicity, energy, forces. Staged 10k -> 100k -> 1M -> 4M.

### 3.2 IQA supervision (small, expensive)

QTAIM/IQA intra-atomic energies via AIMAll. Following Shimamura & Nomura
(arXiv:2609.00674), reference each atom to its isolated-atom value:

    dE_intra(i) = E_intra(i) - E_iso_atom(Z_i)

This removes the ~10^5 kJ/mol per-atom offsets and is required for numerical
viability in float32.

**The IQA subset is NOT constrained to be a subset of OMol25.** v1 required
`D_IQA subset D_EF`; there is no reason for this. The IQA subset's only job is
to fix the node gauge, so it should be constructed for coverage of local
chemical environments, free of OMol25's sampling distribution.

### 3.3 Resolving the functional mismatch

AIMAll's DFT-IQA supports global hybrids (B3LYP); wB97M-V is range-separated
with VV10 nonlocal correlation and is very unlikely to be supported. **Verify
this first (S8.0a).** Assuming it is not:

    selected geometry
        |
        +-- wB97M-V/def2-TZVPD SP  -> E, F      (consistent with OMol25)
        |
        +-- B3LYP/def2-TZVPPD SP   -> wfx -> AIMAll -> E_intra

Two single points per structure. The extra SP is negligible against AIMAll's
O(N^2) basin integration. E/F labels stay self-consistent across the whole
dataset; the residual is that the IQA partition sums to E_B3LYP rather than
E_wB97M-V. Most of that difference is a smooth per-element offset absorbed by
the isolated-atom referencing; the remainder is absorbed by the edge branch.

**Consequence:** OMol25's 4M wavefunction subset provides no advantage.
AIMAll is the cost, not the DFT. Drop that dependency from the data plan
and with it the ~200 TB download it implied.

### 3.4 Interaction-energy supervision (new, cheap, large)

Counterpoise-corrected fragment interaction energies at fixed geometry:

    E_int(A,B) = E_AB - E_A - E_B

Sources: DES370K, SPICE dimers, S66x8, Splinter. These cover exactly the regime
the IQA paper reports as its weakest (weak and repulsive interactions) and that
ABFE depends on entirely. No wavefunction analysis required; three single
points per datapoint. Scalable to 10^5 without difficulty.

---

## 4. The decomposition

### 4.1 Architecture-agnostic head

The deliverable is an `e3nn` module, not a fork of any one model. For any
backbone producing equivariant node features `h_i`:

    E = sum_i E_i^intra(h_i) + sum_{i<j} D_ij(h_i, h_j, r_ij)

Requirements:

- **Permutational symmetry by construction.** Build pair features from
  `h_i + h_j` (and the symmetric part of `h_i (x) h_j`), not by averaging
  `D_ij` and `D_ji` post hoc as in v1. Note `r_hat_ij = -r_hat_ji`, so odd-l
  spherical harmonics change sign; restrict the pair channel to even l or use
  explicitly symmetric combinations.
- **Energy-conserving.** Both branches feed the total energy; forces by
  autograd. No side outputs.
- **Readouts are 0e.**

Backbone support:

| Backbone | D_ij route | Role |
|---|---|---|
| Allegro | native (edge-centric) | **control** - isolates "does gauging work" from "is my invented pair decomposition good" |
| NequIP | external head (above) | generalization test |
| MACE | body-order expansion (S4.2) or external head | generalization test |

Implement against the `nequip` module interface (Allegro is a `nequip`
extension, so both come free); keep the head itself a pure `e3nn` module with
no `nequip` dependency so MACE needs only a thin adapter.

### 4.2 MACE and body order

MACE's node energy is a polynomial in the atomic basis A_i, i.e. an explicit
body-ordered expansion to order nu+1:

    E_i = E_i^(1) + sum_j E_ij^(2) + sum_jk E_ijk^(3) + ...

The two-body term is therefore *readable*, not invented. Higher-body terms are
genuinely not pairwise and are handled by the lambda rule in S5.3.

### 4.3 Two gauge constraints

IQA supervision alone does **not** constrain the quantity the alchemy uses.
It pins the one-body/two-body split; it says nothing about how `D_LE` is
partitioned against `D_LL`. Two constraints are needed:

    L = w_E L_E + w_F L_F + m_IQA w_I L_IQA + m_int w_int L_int

| Term | Pins | Cost | Scale |
|---|---|---|---|
| `L_IQA` | node gauge | high (AIMAll) | 10^3 - 10^4 |
| `L_int` | `sum D_LE` | low (3 SPs) | 10^4 - 10^5 |

with

    L_int = | sum_{i in A} sum_{b in B} D_ib  -  E_int(A,B) |^2

Note `L_int` constrains a **physical observable**. Matching IQA pair energies
`E_ij^inter` is a *diagnostic*, not a requirement: ABFE needs `sum D_LE` to be
right, not the internal covalent-bond partition. The IQA paper's large pair
error (2.1 eV MAE) is dominated by covalent allocation, and no covalent bond
crosses an L/E boundary.

`E_ij^inter` is retained as a held-out validation target only, preserving the
defining experiment of the original work.

**This is now measured, not argued.** In a controlled teacher/student test where
the target decomposition is exactly representable, node supervision drives
E_intra to 1.1% relative error but leaves D_ij at 30%, and full fine-tuning does
not improve the pair term at all (S13.2). Total energy plus node supervision is
mathematically insufficient to determine the individual D_ij. Since every
ligand-environment term is a D_ij, L_int is not optional.

### 4.4 Loss balancing

The IQA paper sweeps lambda_intra over {0, 0.01, 0.05, 0.1, 1.0, 10.0} with the
optimum at 0.1, and total-energy MAE degrades slightly (0.890 -> 0.945 eV) when
IQA supervision is added. The branches are in genuine tension. Expect a narrow
weight window; budget for a sweep at every scale, not just once.

**Normalise every term by the mean square of its own target.** The published
w_I = 0.1 was tuned for that paper's label units and does not transfer. With
forces of order 14 and intra energies of order 0.35, an unnormalised w_I = 0.1
puts the IQA term at ~1e-6 of the total loss: training converges cleanly, the
E/F metrics look healthy, and the IQA term contributes no usable gradient at
all. This failure is silent -- it cost one full run to find (S13.1). Normalised,
the weights mean the same thing across datasets and w_I is interpretable.

---

## 5. Alchemical Hamiltonian

### 5.1 Two-body form

    U(R, lambda) = sum_i E_i^intra + D_LL + D_EE + sum_{i in L} lambda_i sum_{a in E} D_ia

lambda_i = 1: fully interacting. lambda_i = 0: invisible to the environment.

A dummy atom is **not** a new element. Do not introduce `Z = dummy` or a learned
dummy embedding. The ligand atom stays chemically real inside the ligand; the
dummy state is defined solely by ligand-environment coupling.

### 5.2 lambda must also mask the graph

v1 required `dE_i^intra/dlambda = 0` and `dD_LL/dlambda = 0`. Algebraically
those hold, but `E_i^intra` and `D_LL` are functions of features computed on a
graph that includes the environment. At lambda = 0 the ligand is still polarized
by the environment, so

    U(R, 0) != U_L + U_E

and v1's test S21.3 fails. Fix: lambda scales the **messages crossing the L/E
boundary** as well as the edge energies. Then at lambda = 0 the graph is
disconnected and the endpoint identity is exact for any architecture.

Cost: `dD_LL/dlambda != 0`. This is acceptable. Ligand-internal invariance is an
aesthetic preference, not a thermodynamic requirement - TI needs only
`dU/dlambda`, which autograd supplies including the extra term. Trading an
unachievable invariance for an exact endpoint is the correct trade.

**Revised tests:** ligand-internal invariance is *measured and reported*, not
required. Endpoint identity is *required*.

Allegro is strictly local (edge features depend only on atoms within one cutoff
of the centre, independent of depth), so contamination is bounded by the cutoff
rather than growing with layers. Quantify it; do not assume it.

### 5.3 Arbitrary body order

For a body-ordered term over atom set S:

    Lambda(S) = prod_{i in S ^ L} lambda_i   if  S ^ L != {} and S ^ E != {}
              = 1                            otherwise

    U(lambda) = sum_S Lambda(S) E_S

Every term touching both L and E decays with the product of its ligand members'
couplings; pure-L and pure-E terms are untouched. At lambda = 0 exactly
`U_L + U_E` remains. Reduces to S5.1 at two-body order.

### 5.4 Softcore

Linear scaling `D_ia -> lambda_i D_ia` is the first implementation only. If the
learned interaction is strongly repulsive it will produce endpoint singularities
and poor overlap. Later: softcore coordinate transforms, bounded energy
transforms, nonlinear lambda schedules. **Keep the decomposition and the
alchemical path conceptually and architecturally separate.**

---

## 6. Atom-resolved free energy (the flagship result)

### 6.1 The diagonal path

Take the diagonal path `lambda(t) = (t, t, ..., t)`, t: 1 -> 0. Then

    dU/dt = sum_{i in L} sum_{a in E} D_ia = D_LE

and per-atom contributions are

    dG_i = - integral_0^1 < sum_{a in E} D_ia >_t dt

with, exactly,

    sum_{i in L} dG_i = dG_decouple

**This costs one ordinary ABFE calculation.** No sequential removal, no
multiplication of lambda windows by ligand size. The per-atom attribution is
obtained by recording per-atom averages in each window instead of only the
total - an analysis change, not a sampling change.

### 6.2 Why the diagonal path is the right one

Sequential removal gives order-dependent `dG_i` (the total is a state function;
the individual terms are not). The diagonal path is the **Aumann-Shapley** value
- the continuous-limit Shapley value - and is the unique attribution satisfying
additivity, symmetry, dummy and consistency. Order dependence is therefore not
mitigated but eliminated.

Sequential decoupling is retained as a *validation tool*: on a small system,
Monte-Carlo-sampled discrete Shapley values over random removal orders must
agree with the diagonal-path attribution. Unlike v1's path-independence test
(which only probes sampling convergence, since both paths share endpoints and
dG is a state function), this test probes the attribution itself.

### 6.3 Free by-products

From the same trajectory set:

- **Per-residue decomposition:** sum over ligand atoms instead,
  `dG_a = - int < sum_{i in L} D_ia >_t dt`.
- **Interaction free-energy map:** the full `< D_ia >` matrix, ligand atom by
  environment residue. A direct, rigorous replacement for MM-GBSA per-residue
  decomposition.
- **Enthalpic attribution at zero cost:** `< sum_a D_ia >` at lambda = 1
  requires no integration at all.

### 6.4 Honest scope of the decomposition

    dG_bind = dG_decouple^solvent - dG_decouple^complex + dG_restraint + dG_standard

Per-atom attribution applies to the **decoupling terms only**. Restraint and
standard-state corrections are global and are not atom-decomposable. Report them
separately; do not smear them across atoms.

---

## 7. Controls and ablations

### 7.1 Message-masking baseline (primary control)

Scaling messages across the L/E boundary by lambda gives an **exact** endpoint
for any architecture, with no decomposition, no IQA and no gauge. It must be run
as the baseline.

What it cannot do:

- **No attribution.** A severed graph has no addressable per-atom quantity.
  This is the decomposition's unique capability and the project's core claim.
- **Poor path.** At intermediate lambda the features are out of distribution -
  half-strength messages correspond to no physical state - so `dU/dlambda` along
  the path is unconstrained extrapolation. Endpoints are right; overlap is not.

**Therefore:** total `dG` from both routes must agree (state function). The
decomposition route must show *faster convergence and lower variance*. That
comparison, plus attribution itself, is the strongest evidence the project can
produce. It is cheap and should exist from day one.

### 7.2 Model ablations

| | Training | Purpose |
|---|---|---|
| A | `L_E + L_F` | unconstrained gauge (negative control) |
| B | `+ L_IQA`, small data only | reproduce arXiv:2609.00674 |
| C | `+ L_IQA + L_int`, small data | does interaction supervision pin `D_LE`? |
| D | `+ L_IQA + L_int`, OMol25 scale | **main model** |
| E | direct pair-IQA supervision | does explicit pair supervision help or hurt? |
| F | message masking, no decomposition | alchemical baseline (S7.1) |

The published negative control is strong and should reproduce: intra-atomic
error 12.210 eV at lambda_intra = 0 vs 0.229 eV at 0.1, a 53x reorganization at
nearly unchanged E/F accuracy. If this does not reproduce, the implementation is
wrong.

---

## 8. Staging

### 8.0 Phase 0 - cheap experiments that can kill the project

Run before anything else. Target: one month.

- **0a. Functional compatibility.** Can AIMAll do IQA on a wB97M-V/ORCA GBW?
  Determines whether S3.3's two-single-point workaround is needed.
- **0b. Cost and noise.** Install AIMAll, run 5 molecules end to end. Measure
  per-structure wall time and **IQA recovery error**
  (`sum E_intra + sum E_inter - E_DFT`). The paper reports neither. Recovery
  error is the label-noise floor and caps achievable node accuracy; wall time
  determines whether a 10^4 IQA subset is real or fantasy.
- **0c. Code and data.** The paper states code and data "will be made publicly
  available" with no URL. Email the corresponding author (Shimamura, Kumamoto
  University) for the 1,287-structure IQA set and training code, and ask 0a/0b
  directly - cheapest available source of both answers.
- **0d. Weak-interaction diagnostic.** On the paper's own systems, stratify
  `D_ij` vs `E_ij^inter` by interaction class and measure MAE **and sign
  accuracy** for weak pairs (|E_ij| < 20 kJ/mol). The paper names these as its
  worst failure mode and they are the entire L/E regime.
- **0e. Endpoint check.** Toy model, dimer separated to 20 A: does
  `U(R, 0) = U_A + U_B` hold? Confirms S5.2's necessity numerically. Needs no
  IQA labels; can run today.

### 8.1 Work packages

    WP1  e3nn reimplementation + negative control        DONE  (S13.1)
    WP2  Decomposition head as a portable e3nn module     DONE  (S13.1)
    WP3  L_int; verify it pins sum D_LE                   DONE  (S13.5, 0.27 kcal/mol)
    WP4  Frozen pretrained backbone + head                <- replaces WP5
    WP5  (deleted -- train from scratch on OMol25)        CUT   (S13.2)
    WP6  lambda masking; endpoint identities              DONE  (S13.7)
    WP7  Attribution: diagonal path vs MC Shapley          DONE  (S13.8)
    WP8  Softcore; overlap and convergence vs message-masking baseline
    WP9  Validation ladder (S8.2)

WP1-WP4 are independent of the ABFE programme and constitute a complete result
on their own (S10, Paper I).

WP1 is not a reproduction of the authors' code: it is not published, and no
public Allegro variant of E3D-IQA exists. The implementation is from scratch in
e3nn, on a MACE backbone, and WP1 is considered met by reproducing the paper's
*defining experiment* rather than its code (S13.1).

### 8.2 Validation ladder

| Stage | System | Tests |
|---|---|---|
| A | molecular dimers (water, methane-water, benzene-water, ammonia-water) | endpoint identity; edge decoupling; attribution vs MC Shapley |
| B | small neutral molecules, explicit solvent | dG_solv vs **FreeSolv**; per-atom attribution chemically sensible |
| C | rigid host-guest | **SAMPL** sets; restraints; full ligand decoupling |
| D | protein-ligand ABFE | neutral ligands, compact pockets, well-defined poses, modest conformational change |

v1 named no benchmark datasets. Without named references there is no
quantitative definition of success.

### 8.3 Consistency tests (before any free energy)

1. `U(R, 1) = E_ML(R)` exactly.
2. `U(R, 0) = U_L + U_E` exactly **(required; needs S5.2)**.
3. Ligand-internal drift `U_LL(lambda) - U_LL(1)` **measured and reported**, not
   required to vanish.
4. Forces smooth and bounded across `0 <= lambda <= 1`.
5. Autograd `dU/dlambda_i` matches finite differences.
6. `sum_i dG_i = dG_decouple` to within integration error.

---

## 9. Go / no-go

| Gate | Go if | **Kill / pivot if** |
|---|---|---|
| Phase 0 | AIMAll pipeline runs; recovery error << target node accuracy; weak-pair signs mostly correct | recovery error >= node MAE - the labels cannot support the claim. Pivot: drop IQA, use `L_int` alone as the gauge |
| WP1 | negative control reproduces | **MET** - 28x on synthetic labels, 49x on a frozen pretrained backbone (S13) |
| WP3 | `sum D_AB` matches `E_int` to a few kJ/mol on held-out dimers | **MET** - 0.27 kcal/mol on 228 held-out systems (S13.5) |
| WP4 | head transfers to NequIP/MACE with comparable decomposition quality | Allegro-only - narrows Paper I but does not kill it |
| WP4 | frozen-backbone gauge holds on REAL IQA labels | frozen cannot express the IQA gauge - then, and only then, train from scratch (the deleted WP5 comes back) |
| WP6 | all six tests in S8.3 pass | **MET** with graph masking; edge-only scaling fails test 2 by 5.9% of the coupling (S13.7) |
| WP7 | diagonal attribution agrees with MC Shapley within uncertainty | **MET** - 0.82% of the total, exhaustive over all removal orders (S13.8) |
| WP8 | overlap and convergence beat the message-masking baseline | no advantage - the decomposition buys attribution only, not efficiency. Still a result; narrow the claim |

---

## 10. Deliverables

Three independently publishable results, in dependency order:

**Paper I - Portable physical gauging of equivariant potentials.**
WP1-WP4. Energy decomposition can be pinned to physical observables (IQA +
interaction energies) independently of backbone, including in node-centric
architectures where pair energies must be constructed rather than read off.
Does not depend on the ABFE programme.

**Paper II - Scaling without retraining.**
WP4. The gauge is imposed on a pretrained OMol25 backbone (`mace-omol-0`) by
training only the decomposition head, at ~3% of the trainable parameters and
none of the pretraining cost. Supported by
the published transfer result: adding 10-19 atom E/F-only data improved 20-23
atom external validation from 1.656 to 0.706 eV (energy) and 0.249 to 0.036
eV/A (forces) while maintaining decomposition quality.

**Paper III - Atom-resolved binding free energy.**
WP6-WP9. The flagship. An order-independent per-atom and per-residue
decomposition of binding free energy at QM accuracy, for the cost of one ABFE
calculation, validated against MC Shapley and benchmarked against the
message-masking baseline and conventional ABFE.

---

## 11. Risks

**R1 - Weak and repulsive interactions.** The source paper's stated worst
failure mode ("sign-allocation errors for positive interatomic terms and
magnitude errors for attractive pairs") is exactly the L/E regime; no covalent
bond crosses the boundary. *Mitigation:* `L_int` targets this directly with
cheap, abundant data. *Test:* 0d, then WP3. This is the single most important
technical risk.

**R2 - Pair gauge non-uniqueness.** Confirmed by the paper's own data:
increasing the cutoff from 5.0 to 5.5 A "did not monotonically improve the
correspondence," and the residual "originates from latent energy allocation
within the modeled graph, rather than from omitted cutoff-excluded IQA
interactions." Node supervision does not uniquely determine each `D_ij`.
**Now measured directly** (S13.2): with a target that is exactly representable
and full fine-tuning allowed, D_ij still lands at 30% relative error while
E_intra reaches 1.1%, and extra capacity does not help. The pair split is
underdetermined by total energy plus node supervision, as a matter of
information, not optimisation.
*Mitigation:* `L_int` constrains the sum that matters; pair-level uniqueness is
not required for the claim. Report the residual honestly.

**R3 - IQA labels are noisy and expensive.** Recovery error is typically
1-10 kJ/mol, potentially comparable to the node accuracy being targeted. The
paper quantifies neither error nor cost. *Mitigation:* measure in 0b; if the
floor is too high, `L_int` can carry the gauge alone.

**R4 - Gauge drift at scale is undetectable.** No IQA labels exist on the 4M
set. *Mitigation:* a label-free size-consistency test - separate any two
fragments to large distance and require `E_total -> E_A + E_B`, all cross-fragment
`D_ij -> 0`, and each `E_i^intra` to converge to its isolated-molecule value.
Free, batchable over unlabelled structures, and it simultaneously probes the
alchemical endpoint. Add to standard metrics.

**R5 - Long range.** IQA inter-atomic energies contain full Coulomb and never
vanish. Not the dominant error at 4-9 atoms (see R2), but it returns for charged
ligands and explicit solvent, where solvation free energy is dominated by
electrostatics beyond any practical cutoff. *Decision required before Stage B*,
not at WP9: adding an explicit electrostatic branch puts that energy outside
`D_ij` and therefore outside lambda's control, which changes the alchemical
Hamiltonian. Initial validation: neutral ligands only.

**R6 - Deployment in condensed-phase MD.** OMol25 is non-periodic finite
clusters with global charge and spin as model inputs; neither concept survives
in a periodic solvated box. A solvated protein-ligand system is 3-5 x 10^4
atoms. *Likely answer:* ML/MM with the ML region covering ligand plus pocket -
but then the L/E boundary crosses into the MM region and `D_LE` must be
redefined there. *Decide before Stage C.* v1 had no deployment plan at all.

**R7 - Scope.** Three papers is a multi-year programme. The work packages are
ordered so WP1-WP4 stand alone; do not serialize the whole chain before the
first publishable unit.

---

## 12. Summary

    OMol25 (E, F)  +  sparse IQA  +  fragment interaction energies
                            |
                            v
              physically gauged decomposition
                E_i^intra  +  D_ij       (portable e3nn module)
                            |
                            v
                  D_LL  +  D_EE  +  D_LE
                            |
                            v
              lambda_i on D_LE, masking edges AND messages
                            |
                            v
            diagonal path: dG_i = -int < sum_a D_ia >_t dt
                            |
                            v
              atom- and residue-resolved binding free energy

Principle unchanged from v1: **fix the energy gauge first, then do alchemy.**
Added in v2: **the gauge must be fixed on the quantity the alchemy actually
uses, and the payoff is attribution, not merely a working endpoint.**

---

## 13. Measurements

Runs are in this repository; `decomp/` is the implementation and the `test_*.py`
files are the experiments. Backbone: MACE (`mace-torch` 0.3.16), e3nn 0.5.1,
fp32 on one RTX 2080 Ti. Both experiments use **synthetic teacher labels**: a
randomly initialised head defines one particular (E_intra, D_ij) gauge and a
student is trained on its E, F and E_intra. This tests the machinery, not the
chemistry -- see S13.3.

### 13.1 The defining experiment reproduces

`test_negative_control.py`, MACE trained from scratch, 16 water dimers:

    w_I      E/atom MAE    F MAE    E_intra MAE
    0.0         0.00114   0.01245        0.38588
    0.1         0.00124   0.01357        0.01370        28x

Matches arXiv:2609.00674 in all three respects: w_I = 0 fits E/F just as well
(slightly better) yet does not recover the split; w_I > 0 reorganises the node
branch by more than an order of magnitude; and E/F degrade slightly rather than
improve (+9% here, +6% in the paper). Adding node and edge channels to the
architecture is not sufficient for an IQA-like decomposition to emerge.

### 13.2 The gauge does not require retraining the backbone -- but pair energies stay loose

`test_frozen_gauge.py`. Backbone: pretrained MACE-OFF24_medium. The teacher head
sits on the same frozen features, so the target gauge is representable by
construction. 32 water dimers, 3000 steps, w_I swept.

                           E/atom MAE     F MAE   E_intra MAE    D MAE   trainable
    predict zero              0.59003   14.44186     0.35219    0.34173          0
    frozen    w_I=0           0.00838    0.76874     0.32282    0.16011     45,696
    frozen    w_I=1           0.00988    0.74139     0.01025    0.10255     45,696
    frozen    w_I=10          0.01094    0.77711     0.00673    0.10352     45,696
    frozen    w_I=100         0.01117    0.78548     0.00663    0.10359     45,696
    finetuned w_I=10          0.00492    0.65378     0.00384    0.10119  1,474,064

**Finding 1 -- the negative control holds on a frozen pretrained backbone.**
w_I = 0 fits E/F best of the frozen runs yet leaves E_intra at 0.3228 against a
predict-zero baseline of 0.3522: 8% better than predicting nothing.

**Finding 2 -- a frozen backbone suffices, and the result saturates.**
w_I = 1 -> 10 -> 100 gives 0.0103 -> 0.0067 -> 0.0066. Raising the weight a
hundredfold changes nothing, so the residual is the expressive limit of the
frozen features, not weak supervision. That limit sits 1.73x from full
fine-tuning while training 3% of the parameters. **This is why WP5 is cut.**

**Finding 3 -- node supervision does not pin the pair energies.**
D_ij reaches 0.1012 against |D| = 0.342, i.e. 30% relative error, versus 1.1%
for E_intra -- and frozen (0.1035) and fine-tuned (0.1012) are indistinguishable,
so capacity is not the constraint. The target is exactly representable and the
optimiser still cannot find it, because total energy plus node supervision does
not determine the individual D_ij. Every ligand-environment term is a D_ij.
**This is why L_int is required, and it is the sharpest result so far.**

### 13.3 The finding transfers to an OMol25-pretrained backbone

`test_omol_transfer.py`. Frozen `mace-omol-0-extra-large-4M`: 51.3M parameters,
19456-dim node features, 82 elements, r_max 6.0, with per-graph `total_spin` and
`total_charge` categorical embeddings. 8 water dimers, 3000 steps, frozen only.

                          E/atom MAE     F MAE   E_intra MAE    D MAE
    predict zero             0.12167    2.44765     0.31057    0.16379
    frozen w_I=0             0.00147    0.27316     0.20283    0.19847
    frozen w_I=10            0.00180    0.28814     0.00644    0.20390

**The gauge is imposable on OMol25 features.** E_intra reaches 0.0064 against a
predict-zero baseline of 0.3106 -- 48x -- and 31.5x better than w_I = 0. S13.2
carries from 640-dim MACE-OFF24 to 19456-dim mace-omol-0. The head trains in
~140 s at 781 MiB on one RTX 2080 Ti, so the pretrained-backbone route is cheap
enough to iterate on.

**D_ij is not identifiable, and this is now the cleanest result in the file.**
Running 5x longer improves everything except the pair energies:

    steps      E/atom      E_intra       D_ij
    600       0.01668      0.02144     0.22280
    3000      0.00147      0.00644     0.20390
    gain          11x         3.3x       1.09x

D_ij ends at 0.2039 against |D| = 0.1638, i.e. **worse than predicting zero**,
and w_I = 0 (0.1985) and w_I = 10 (0.2039) are indistinguishable. The mechanism
is visible in the constraint itself: once E_total and E_intra are both pinned,
`sum D_ij` is pinned too, but the distribution over individual edges is not.
The student finds a different distribution with the same sum -- per-edge error
0.204 against |D| 0.164 means large cancelling redistribution.

This is an identifiability problem, not an optimisation problem. More steps,
more parameters and more supervision on the node branch cannot fix it, because
the information is not in the loss.

*Caveat:* 124% here vs 30% in S13.2 is not a like-for-like comparison -- 8
structures vs 32. Fewer structures means fewer constraints and more freedom to
redistribute, so the gap may be sample size rather than backbone. What holds
across both, at two backbones, two step counts and with and without
fine-tuning: D_ij is recovered poorly, node supervision does not help it, and
neither capacity nor training time helps.

### 13.4 L_int pins sum D_AB to chemical accuracy

`test_lint.py`. Frozen `mace-omol-0-extra-large-4M`, head only. DES370K
`hcno_small`: H/C/N/O, <= 16 atoms, neutral. 73,409 train / 11,042 test dimers,
split **by system_id**, so held-out means unseen molecule pairs rather than
another geometry of a pair already in training. Target `cbs_CCSD(T)_all`.

**Read the system count, not the dimer count.** `hcno_small` holds 93,174
geometries of only **246 distinct molecule pairs**, so the split is roughly
196 / 24 / **26** systems and the test number below rests on 26 pairs (400
geometries sampled from them). That is 6.7% of DES370K's 3,691 systems. The
result is a promising signal on a narrow slice, not a general one.

                             val MAE            test MAE
    w_int=0 (untrained)   22.65 kcal/mol     20.12 kcal/mol
    w_int=1                0.78 kcal/mol      0.80 kcal/mol
    predict zero                               3.81 kcal/mol

**Test MAE 0.80 kcal/mol, below chemical accuracy**, with val 0.78 vs test 0.80
-- essentially no generalisation gap. 4.8x better than predicting zero, 25x
better than the untrained head.

This closes the gap S13.3 opened. D_ij is not identifiable from total energy
plus node supervision, but `sum_{i in A, b in B} D_ib` against a measured
interaction energy pins it directly, and that sum is exactly the quantity the
alchemical Hamiltonian scales. **WP3 passes.**

Cost: 5,000 steps at batch 16, 231 s, 2110 MiB on one RTX 2080 Ti. That is
80,000 samples against 73,409 training structures -- **1.09 epochs**, and the
val curve was still falling. The earlier failure (S13.5 note) used 2,000
structures for 8 epochs and stalled at the predict-zero baseline; it was data
starvation, not a structural limit.

**What this does not pin.** E_int is one scalar per structure constraining a sum
over several hundred cross-fragment edges. The total over a boundary is pinned;
the individual D_ia are not. The flagship per-atom attribution (S6) needs
`sum_a D_ia` correct *per ligand atom*, which this supervision does not reach.
Three cheap routes, none needing new quantum chemistry beyond more single
points: small fragments (so the sum spans few atoms), many fragmentations of one
geometry (each gives a different linear combination of D_ij), and 3-body terms.
DES370K also ships the full SAPT decomposition (`sapt_es`, `sapt_ex`,
`sapt_ind`, `sapt_disp`), which turns one constraint per structure into four
with different spatial character, at no extra compute.

**Scope.** 26 held-out molecule pairs, H/C/N/O, <= 16 atoms, neutral, and
trained on L_int *alone* -- no energy or force anchoring, so this is not yet a
valid potential. Two things must hold before 0.80 kcal/mol means anything
general:

    config        systems   train dimers   status
    hcno_small        246         73,409   0.80 kcal/mol (S13.4)
    hcno            1,174        151,200   not run
    organic         2,274        232,011   not run
    full            3,691        295,617   not run

Chemical breadth (`organic` adds F P S Cl Br I at 9x the system count) and joint
training against E, F, IQA and E_int together -- whether the constraints
cooperate or fight.

### 13.5 The D_ij gap is the backbone, not the sample size -- and WP3 passes

Two results from the same queue.

**L_int scales.** `organic` (1,819 train / 228 test systems, F P S Cl Br I
added, 8.8x the chemical diversity of `hcno_small`):

    w_int=1        test 0.0119 eV = 0.27 kcal/mol
    predict zero        0.1308 eV = 3.02 kcal/mol

0.27 kcal/mol, 11x below the predict-zero baseline and **better** than the 0.80
obtained on `hcno_small`'s 26 systems. More chemical breadth improved the result
rather than breaking it, confirming the `hcno_small` number was data-starved
rather than lucky. **WP3 passes.**

**The sample-size hypothesis was wrong.** S13.4 flagged the 30% (S13.2,
MACE-OFF24, 32 structures) versus 124% (S13.3, mace-omol-0, 8 structures)
comparison as confounded and guessed sample size. Sweeping n on a fixed
backbone settles it:

    n      |D|    frozen w_I=10   rel    finetuned   rel
      8   0.5416       0.18115   33.4%     0.20163  37.2%
     32   0.3417       0.10352   30.3%     0.10119  29.6%
    128   0.3020       0.10936   36.2%     0.09827  32.5%
    512   0.4707       0.13326   28.3%     0.20464  43.5%

Flat at 28-36% across a 64x range in sample size. The gap is a property of the backbone,
not the sample size. This makes the non-identifiability finding **stronger**:
16x the data, 32x the trainable parameters, and 100x the supervision weight all
leave D_ij pinned at roughly a third of its own magnitude.

**One difference between backbones that is not yet understood.** On MACE-OFF24,
L_IQA does improve the pair term -- 56% -> 33% (n=8), 47% -> 30% (n=32),
63% -> 36% (n=128), a consistent 1.5-1.8x. On mace-omol-0 (S13.3) it does not:
w_I=0 gives 0.1985 and w_I=10 gives 0.2039, flat to slightly worse. Node
supervision helps the pair split on one backbone and not the other. Worth
understanding before relying on either.

**WP5 stays cut, and the case strengthens with data.** The frozen-to-fine-tuned
E_intra ratio runs 1.61 / 1.73 / 1.75 / **1.12** at n = 8 / 32 / 128 / 512 --
the slow upward creep reverses and frozen essentially catches up. L_IQA
reorganises E_intra by 121.6x / 48.7x / 54.0x / 62.8x throughout.

At n = 512 the fine-tuned model is worse than the frozen one on every metric --
E/atom 0.03214 vs 0.02944, forces 1.652 vs 1.446, D_ij 0.20464 vs 0.13326 --
with 32x the trainable parameters. **Do not read that as fine-tuning being
harmful.** Both ran 3,000 steps, which is far too few for 1.47M parameters on
512 structures; the fine-tuned arm is simply undertrained, and a fair comparison
would give it a step budget matched to its parameter count. The defensible
claim is the weaker one: a frozen backbone reaches the same place, not that it
reaches a better one.

### 13.6 The four constraints trade against each other, and the trade is real

`test_joint.py`. Frozen `mace-omol-0-extra-large-4M`, head only. `organic`,
50,000 train (1,817 systems) / 1,000 test (178 systems). E and F targets are the
backbone's own output minus per-element references (option D, S13.8); E_int is
DES370K CCSD(T); E_intra is a fixed random teacher. Conditions: A = E+F,
B = +L_int, C = +L_IQA+L_int.

**At 8,000 steps everything looked like a capacity problem. It was not.**

    hidden      A E/atom    B E/atom    C E/atom   L_int cost
        64       0.00146     0.00371     0.00486        +154%
       256       0.00174     0.00337     0.00454         +94%

The apparent improvement from 64 to 256 is an artefact: the cost ratio fell
because the *denominator* got worse (A degraded 0.00146 -> 0.00174 with 4x the
capacity), while B improved only 9%. A harder task fitting worse with more
parameters is a signature of undertraining, and the head turned out to be
nearly free anyway -- benchmarking on the real model gives 712 ms/step at
hidden=128 and 732 ms at hidden=512, a 3% difference, because the 51.3M-parameter
backbone forward dominates. (Raising `hidden` is limited by memory, not compute:
the run sits at 9.2-9.6 GiB of an 11 GiB card. `DecompositionHead` now takes an
`irreps_proj` argument that projects the backbone's 19456 dims down before the
per-edge tensor product, worth ~7% memory and ~12% time.)

**At 32,000 steps both A and C converge, and they converge to different places.**

    condition    steps    E/atom        F     E_intra    E_int
    C             8000   0.00454  0.06887     0.03250  0.64 kcal
    C            32000   0.00208  0.03035     0.01500  0.40 kcal
    A             8000   0.00174  0.03354        --       --
    A            32000  ~0.00062  ~0.0119        --       --   (estimated)

Training loss plateaus for both -- A at 0.0005 (flat over the last 4,000 steps),
C at 0.0147 (0.0171 -> 0.0147 over the last 6,000). Neither is still falling.

*The A row at 32,000 steps is an estimate, not a measurement.* A's training
completed but the old code printed its table only after all three conditions,
and the process was killed during B, so the numbers were lost. The estimate
scales the measured 8,000-step values by sqrt(0.004 / 0.0005) = 2.83 from the
loss ratio; MAE-vs-MSE and unequal scaling of the two loss terms put perhaps a
1.5x uncertainty on it. The code now writes a JSON result and a head checkpoint
after each condition, so this cannot recur.

**The trade, quantified.** C pays roughly 3.4x on total energy relative to A
(2.6x at 8,000 steps), and the gap *widens* with training rather than closing:
A improved 2.8x between 8,000 and 32,000 steps, C only 2.2x. Both converged.
This is a genuine trade-off, not an optimisation failure.

On the other axis the tax shrinks:

    E_int          L_int alone    joint C    tax
    8,000 steps          0.27       0.64    2.4x
    32,000 steps         0.27       0.40    1.5x

So the joint model gets steadily closer to a dedicated interaction-energy model
while paying a fixed premium on total energy.

**Is that premium affordable? Probably.** C reaches 2.08 meV/atom against the
backbone's own energy. The backbone's own error against DFT is itself of order
1-2 meV/atom, so the stack is around 3 meV/atom -- about 1.7 kcal/mol of
absolute energy on a 25-atom dimer. Free energies are differences, where
systematic error largely cancels. The point is that the cost is now measured
rather than unknown, and it is the price of having a decomposition at all.

**The trained head is saved** at
`ckpt/joint_organic_n50000_h256_s32000_C.pt` -- the first head trained under all
four constraints, and the starting point for WP7.

### 13.7 The endpoint: v1's design is wrong, S5.2's fix is exact

`test_endpoints.py` + `decomp/lambda_mask.py`. Water dimer, ligand = first
water. Seconds, no training, no data. These are exact-arithmetic identities:
they hold or they do not.

    [1] full state   U(1) - E_ML          edge 1.19e-07   graph 0.00e+00   OK
    [2] endpoint     U_L + U_E = -0.831327
                     edge   U(0) = -0.820455   diff 1.09e-02   FAIL
                     graph  U(0) = -0.831327   diff 1.19e-07   OK
    [3] ligand drift |U_LL(0) - U_LL(1)|  edge 0.00e+00   graph 2.04e-03
    [4] forces       max step / max |F|   finite and smooth on both
    [5] dU/dlambda   autograd vs fd       5.1e-05 / 4.7e-05   OK
    [6] additivity   sum_i dU/dlambda_i = D_LE = -0.184579, diff 0.00e+00  OK

**v1's Hamiltonian has the wrong endpoint.** Scaling only the LE edge energies
leaves U(R,0) off by 1.09e-02 eV = 0.25 kcal/mol on a water dimer whose entire
LE coupling is 0.1846 eV -- a **5.9% systematic bias on the coupling energy**,
and one that does not cancel between the complex and solvent legs because the
residual polarisation differs between them. Masking the graph as well (S5.2)
makes the identity exact to fp32, 9e4 times closer.

**The trade S5.2 argued for is now measured on both sides.** Edge-only scaling
keeps ligand-internal energy exactly invariant (0.00e+00) and gets the endpoint
wrong; graph masking breaks that invariance (2.04e-03) and gets the endpoint
exactly right. Giving up an aesthetic invariance for a correct endpoint is the
right trade, and S8.3 already demotes test 3 to "measure and report".

**Attribution is exactly additive.** `sum_i dU/dlambda_i = D_LE` to machine
precision, so the S6.1 diagonal-path decomposition satisfies
`sum_i dG_i = dG_decouple` by construction, not approximately. The flagship
result's arithmetic foundation holds. **WP6 passes.**

*Scope:* a randomly initialised small MACE on one water dimer. The identities
are exact arithmetic and the mechanism is architectural, so they should carry,
but repeating on the pretrained `mace-omol-0` and on a larger ligand is cheap
and should be done before relying on it.

### 13.8 The diagonal path gives the Shapley value

`test_attribution.py`. Water dimer, ligand = first water (3 atoms), 24
conformers, 33 lambda nodes, graph masking. Deliberately not a free-energy
calculation: the average is over a fixed conformer set rather than a Boltzmann
ensemble, which isolates the attribution arithmetic from sampling convergence
(WP8/WP9). Anything failing here fails for free too.

     atom     diagonal   Shapley mean   Shapley sd    |diff|
        0     0.320391       0.318764     0.001776  1.63e-03
        1    -0.126608      -0.127019     0.000183  4.12e-04
        2     0.004380       0.004518     0.000140  1.38e-04
      sum     0.198163       0.196263               1.90e-03

**The diagonal integral reproduces the exhaustive discrete Shapley value** over
all 3! removal orders to 1.63e-03, or 0.82% of the total -- and that residual is
trapezoid discretisation over 33 nodes, not a difference between the two
attributions.

**Order dependence is real, not a straw man.** Sequential removal gives per-atom
values with a 1.3% spread across orders, and the totals range 0.194474 to
0.198050. "Atom i contributes X" is genuinely ill-defined under sequential
decoupling; the diagonal path removes the ambiguity rather than mitigating it.
(The spread in the *total* is a state function and should be exactly zero; 3.6e-03
is the same discretisation residual.)

**The cost argument is already visible at three atoms** -- 33 lambda points
against 162 for the exhaustive route, 5x -- and becomes decisive at ligand
scale: 30 heavy atoms have 30! orders. The diagonal path costs one ordinary
alchemical calculation regardless of ligand size, because the per-atom
attribution is an analysis-side change (record per-atom averages in each window
instead of only the total), not a sampling-side one.

Together with the exact additivity of S13.7 (`sum_i dU/dlambda_i = D_LE` to
machine precision), the arithmetic behind the flagship result is complete.
**WP7 passes.** What remains for atom-resolved free energies is sampling and
the per-atom resolution of D_ia itself (S13.4), not the attribution theory.

### 13.9 SAPT component supervision, as tried, makes the total worse

`test_sapt.py`. Frozen `mace-omol-0-extra-large-4M`, head only, `organic`,
50,000 train (1,817 systems) / 2,000 test (216 systems), 8,000 steps,
hidden=256. Three arms at matched budget, scored on held-out total E_int.

    arm      channels   supervised on              test MAE
    total           1   cbs_CCSD(T)_all            0.73 kcal/mol
    sum             4   the sum of the 4 channels  1.29 kcal/mol
    sapt            4   the 4 components           3.92 kcal/mol
    predict zero    --  --                         2.93 kcal/mol

**Component-only supervision is worse than predicting zero.** The per-channel
fits are each about half right -- es 1.945/3.96 = 49%, ex 3.776/8.38 = 45%,
ind 1.703/2.55 = 67%, disp 1.353/3.11 = 43% -- and the total is a small residue
of large cancelling terms: exchange repulsion of +8.4 kcal/mol offsets
electrostatics and dispersion to leave 2.93. Four independently 50%-wrong
components do not add up to a usable total.

**Two problems with the experiment, both mine.**

*The sapt arm never sees the total.* The loss weights each channel by its own
mean square and sums them, which optimises "every component is right", not "the
total is right" -- and the total is cancellation-dominated. Components **plus**
the total is the configuration that should have been run.

*More seriously, this measures the wrong quantity.* The motivation (S13.4) was
that four constraints with different spatial decay would pin how D_ij is
distributed over the boundary, giving the per-atom resolution the flagship
result needs. Total E_int MAE does not report that. It was measured because no
reference per-pair decomposition exists to compare against -- which is the real
difficulty, and the reason the question stayed open.

The 4-channel head also costs 1.8x on the total even when supervised identically
(`sum` 1.29 vs `total` 0.73), which at fixed steps and learning rate is most
likely an optimisation artefact rather than a property of the parameterisation.

**The direct measurement is available and cheap.** The S13.2 teacher/student
setup answers the actual question: give the teacher four pair channels and ask
whether a student under component supervision recovers the teacher's D_ij better
than one under total-only supervision. That measures the pair distribution
itself, needs no new data, and has a target that is representable by
construction.

### 13.10 Measured directly, component supervision does pin the pair distribution

`test_pair_recovery.py`. The S13.9 follow-up, using the teacher/student
construction to measure the quantity S13.9 could not: a fixed random 4-channel
teacher head IS a reference pair decomposition, representable by the student by
construction, so D_ij recovery is directly measurable. 64 dimers, toy MACE,
1,500 steps, 4 channels in every arm so the comparison is supervision rather
than parameterisation.

    arm           D recovery, all edges   cross-boundary
    total                        141.2%           154.5%
    components                    35.3%            25.1%
    both                          35.5%            25.4%
    oracle (edge-by-edge)          1.5%             1.9%

**Component supervision closes 85% of the gap** between total-only supervision
and the edge-by-edge oracle, on the cross-boundary terms that per-atom
attribution actually integrates -- a 6.2x improvement.

**This reconciles with S13.9 rather than contradicting it.** Components pin how
D_ij is distributed; they do not pin the total, because the total is a small
residue of large cancelling terms (exchange +8.4 against attraction, leaving
2.93). Both statements are true at once, and together they prescribe the
configuration: **supervise the components AND the total**. That `both` matches
`components` on D recovery (25.4% vs 25.1%) says adding the total costs nothing
on the distribution, so the combination should get both -- which is what the
`sapt+total` arm of `test_sapt.py` now tests on real DES370K data.

**Three limits.** The teacher's four channels are random, not SAPT, so what is
demonstrated is that *four independent linear functionals of D constrain it far
better than one* -- a statement about information content. It does not show that
SAPT's particular split is better than any other four, only that SAPT is the
four we have measured data for. The `total` baseline here carries no energy,
force or IQA supervision, which is why it sits at 154.5% against the 28-36% of
S13.2/S13.5 where those terms constrain D indirectly; in a realistic joint
setting the baseline is stronger and 85% will shrink. And this is 64 dimers on a
toy backbone.

Still: this is the first positive evidence for the per-atom resolution the
flagship result needs, and it says the route is worth pursuing.

### 13.11 On real SAPT data the trade is strongly favourable

`test_sapt.py`, 40,000 steps (S13.9 ran 8,000). Frozen `mace-omol-0`, head only,
`organic`, 20,000 train / 2,000 test. Node features are cached once -- nothing
here needs dH/dR -- which makes the head-only step ~50x cheaper and put this run
within reach at all. **In fp32: fp16 quantises h to 1.7e-4 relative, harmless
per edge at 2e-4 eV, but the boundary sum runs over hundreds of edges and
accumulates 0.071 kcal/mol, 26% of the 0.27 already achieved.**

    arm           total MAE     per-channel MAE (es / ex / ind / disp)
    total          0.31 kcal     --
    sum            0.31 kcal     7.86  9.93  7.05  6.35
    sapt           1.88 kcal     0.64  1.17  0.37  0.36
    sapt+total     0.47 kcal     0.73  1.59  0.44  0.44
    predict zero   2.93 kcal

**The 1.8x penalty for four channels was undertraining.** At 8,000 steps `sum`
cost 1.76x against `total`; at 40,000 they are identical (1.02x). The
parameterisation is free. That is the third time 8,000 steps has produced a
spurious conclusion in this file -- after capacity (S13.6) and sample size
(S13.5).

**Component supervision buys 6-16x on the distribution for 1.5x on the total.**
Both totals sit inside chemical accuracy (0.31 -> 0.47 kcal/mol), while the
per-channel errors fall 10.8x / 6.2x / 16.1x / 14.5x, from 45-70% of each
component's own magnitude at 8,000 steps to 14-19% here. Helping the total was
never the objective -- S13.4 already pinned it; the objective was constraining
how D_ij is spread over the boundary, which S13.10 showed is what per-atom
attribution needs.

**The toy prediction holds quantitatively on real data.** S13.10, on a synthetic
teacher, put component supervision at 85% of the way to an edge-by-edge oracle.
Taking perfect component recovery as the oracle here gives 91% / 84% / 94% / 93%
across the four channels. A prediction made on a random teacher reproduces on
CCSD(T)/SAPT labels, which is worth more than either result alone.

**Operating point: `sapt+total`.** Components alone (1.88 kcal/mol on the total)
are not usable, because the total is a small residue of cancelling components
and must be supervised directly. Components plus total gets both.

### 13.12 What these do not show

The teacher's gauge is expressible from the frozen features by construction,
because the teacher is itself a head on those features. The experiments show
that the optimiser can find *a* gauge in frozen features; they do not show that
the *IQA* gauge lives there. Only real E_i^IQA,intra labels can answer that, and
that is now the one experiment that could overturn S13.2.

Scale caveats: MACE-OFF24, water dimers, 32 structures, one element pair.
Nothing here has touched OMol25, real IQA labels, or any molecule larger than
six atoms.

### 13.13 Next

    1. DONE (S13.4, S13.5). L_int reaches 0.27 kcal/mol on `organic`'s 228
       held-out systems. WP3 passes.
    2. DONE (S13.6). Not a capacity problem and not undertraining: both A and
       C converge, ~3.4x apart on total energy, and the gap widens slightly
       with training. The premium is affordable (2.08 meV/atom) and now
       measured. Optional: rerun A at 32,000 steps to replace the estimate
       with a measurement -- 2 h, and it will not change the conclusion.
    3. Per-atom resolution of D_ia. First attempt failed (S13.9): SAPT
       component supervision alone makes the total worse, and the experiment
       measured the total rather than the pair distribution it was meant to
       probe. Resolved in S13.10: measured directly, component supervision
       closes 85% of the gap to an edge-by-edge oracle, and S13.11 confirms
       84-94% on real SAPT data. Operating point is `sapt+total`. Next is
       folding it into joint training alongside E, F and IQA.
    4. Charged pairs (`full`, 3,691 systems) after `organic`.
    5. Real IQA labels -> redo S13.2/S13.3. Still the decisive test of whether
       the IQA gauge in particular lives in frozen features.

Two implementation traps, both found the expensive way:

**Never exclude self-pairs by distance.** `torch.cdist` defaults to
`use_mm_for_euclid_dist_if_necessary` and switches to a matmul-based algorithm
**above 25 rows**; its cancellation error leaves the diagonal near 1e-4 rather
than 0, so a `(d < r_max) & (d > 0)` filter admits self-loops. The edge vector
becomes the zero vector, the spherical harmonics normalise by zero, and every
feature is NaN -- with a sharp onset at exactly 26 atoms, independent of dtype,
geometry and neighbour density. Build the mask and `fill_diagonal_(False)`.

Every result in S13 predates the fix but is unaffected: all of them used 6-16
atom systems, below the cdist threshold. The first run of `organic` (up to 34
atoms) hit it, as did an earlier 60-atom synthetic benchmark that was wrongly
dismissed at the time as a bad test geometry. After the fix the edge lists match
`mace.data.AtomicData.from_config` exactly, structure for structure.

**`mace-omol-0` needs per-graph `total_spin` and `total_charge`, shaped
`[n_graphs]`.** A trailing singleton dim broadcasts to
`[n_atoms, n_atoms, 1024]` and the run dies inside the joint embedding.

The lesson both share: validate the graph against the backbone's own data
pipeline before trusting anything built on it.

---

## References

- Shimamura & Nomura, *Diagnosing Latent Energy Decomposition in Machine-Learning
  Interatomic Potentials via Interacting Quantum Atoms*, arXiv:2609.00674v1 (2026).
  B3LYP/def2-TZVPPD (Psi4) + AIMAll; Transition1x; H/C/N/O; 1,287 IQA training
  structures (4-9 atoms) + 22,653 E/F-only (10-19 atoms); Allegro, 3 layers,
  64 scalar / 32 tensor, l_max = 2, 5.0 A cutoff, lambda_intra = 0.1.
- OMol25 - wB97M-V/def2-TZVPD, ORCA 6.
- Interaction-energy sets - DES370K, SPICE dimers, S66x8, Splinter.
- Benchmarks - FreeSolv (solvation), SAMPL (host-guest).

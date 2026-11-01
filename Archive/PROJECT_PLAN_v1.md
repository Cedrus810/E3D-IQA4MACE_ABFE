# OMol25-scale E3D-IQA for Atom-wise Alchemical Free-Energy Calculations

## 1. Project Summary

### 1.1 Goal

The objective of this project is to develop a large-scale equivariant machine-learning interatomic potential in which the total energy is explicitly decomposed into:

\[
E_{\mathrm{ML}}
=
\sum_i E_i^{\mathrm{intra}}
+
\sum_{i<j} D_{ij}^{\mathrm{inter}},
\]

with:

- \(E_i^{\mathrm{intra}}\): an atom-centered intra-atomic energy;
- \(D_{ij}^{\mathrm{inter}}\): a latent pairwise interatomic energy contribution.

The decomposition follows the central logic of **E3D-IQA**, where the node-energy branch is constrained using Interacting Quantum Atoms (IQA) intra-atomic energies while the edge-energy branch is not directly fitted to IQA pair energies. Instead, total energies and forces constrain the full potential, and IQA node supervision fixes the otherwise arbitrary internal energy gauge.

The model will be trained at OMol25 scale using:

1. a large set of OMol25 structures labeled only with total energies and forces;
2. a substantially smaller subset carrying additional IQA intra-atomic energy labels.

The resulting energy decomposition will then be used to define an **atom-wise alchemical dummy transformation** for absolute and relative binding free-energy calculations.

The key alchemical requirement is:

> Turning a ligand atom into a dummy atom must remove only its interaction with the environment while retaining all ligand-internal contributions.

Therefore, ligand internal energy must not be repeatedly removed or counted during sequential atom decoupling.

---

# 2. Scientific Motivation

## 2.1 Problem with conventional atomic-energy MLIPs

Many modern equivariant MLIPs express the total energy as:

\[
E=\sum_i E_i.
\]

Although this is convenient computationally, the individual \(E_i\) generally have no unique physical meaning.

Because message passing mixes information from neighboring atoms,

\[
E_i
=
E_i(
\mathbf R_i,
\mathbf R_{\mathcal N(i)},
\ldots
),
\]

an atomic contribution can contain information originating from:

- intramolecular interactions;
- intermolecular interactions;
- electrostatics;
- many-body polarization;
- environmental effects.

Consequently, directly multiplying the atomic energy of a ligand atom by an alchemical parameter,

\[
E_i\rightarrow\lambda_i E_i,
\]

does **not** rigorously correspond to removing only the interaction of that atom with its environment.

This is especially problematic for atom-by-atom free-energy transformations.

---

## 2.2 The ligand-internal double-counting problem

Consider a ligand \(L\) interacting with an environment \(E\).

Ideally,

\[
U
=
U_{LL}
+
U_{EE}
+
U_{LE},
\]

where:

- \(U_{LL}\): ligand-internal contribution;
- \(U_{EE}\): environment-internal contribution;
- \(U_{LE}\): ligand-environment interaction.

For binding free-energy decoupling, the desired endpoint is:

\[
U_{\mathrm{dummy}}
=
U_{LL}
+
U_{EE}.
\]

The ligand should remain internally intact while becoming invisible to the environment.

If atoms are instead removed sequentially using a decomposition that mixes ligand-internal and ligand-environment energies, terms belonging to \(U_{LL}\) can be removed multiple times or redistributed between successive alchemical steps.

This produces an ill-defined atom-wise decomposition and potentially strong path dependence.

The desired alchemical operation is therefore **decoupling**, not deletion.

---

# 3. Core Hypothesis

The project is based on the following hypothesis:

\[
\boxed{
\text{IQA supervision can fix the latent energy gauge sufficiently well}
}
\]

such that an equivariant MLIP can learn a meaningful decomposition

\[
E
=
\sum_i E_i^{\mathrm{intra}}
+
\sum_{i<j}D_{ij}^{\mathrm{inter}}.
\]

Once the edge contribution \(D_{ij}\) behaves as an interatomic energy representation, ligand-environment interactions can be isolated according to atom identity:

\[
D_{ij}
\rightarrow
\begin{cases}
D_{ij}^{LL},\\
D_{ij}^{EE},\\
D_{ij}^{LE}.
\end{cases}
\]

The alchemical transformation can then operate exclusively on \(LE\) edges.

---

# 4. Relation to E3D-IQA

E3D-IQA introduces an Allegro-type MLIP in which the usual latent edge-energy pathway is retained while an additional node-energy pathway is trained against IQA intra-atomic energies.

The total energy is represented schematically as:

\[
E_{\mathrm{ML}}
=
\sum_i E_i^{\mathrm{node}}
+
\sum_{i<j}D_{ij}^{\mathrm{edge}}.
\]

The important feature is that IQA pair energies are **not required as direct training targets**.

Instead:

\[
E_i^{\mathrm{node}}
\approx
E_i^{\mathrm{IQA,intra}}
\]

is enforced, while total energy and force fitting constrain the remainder.

The resulting latent edge terms can subsequently be compared with IQA interatomic energies.

The published E3D-IQA work shows that energy/force training alone does not recover an IQA-like decomposition, whereas partial IQA intra-atomic supervision substantially reorganizes the latent energy representation. It also demonstrates that additional energy/force-only structures improve transfer beyond the IQA-labeled subset.

This mixed-supervision structure is exactly what makes scaling to OMol25 attractive.

---

# 5. Why OMol25

OMol25 contains more than \(10^8\) DFT calculations spanning organic and inorganic molecules, transition-metal complexes and electrolyte configurations.

The primary labels include:

\[
E_{\mathrm{DFT}},\qquad
\mathbf F_{\mathrm{DFT}},
\]

calculated using:

\[
\omega\mathrm{B97M\!-\!V}/\mathrm{def2\!-\!TZVPD}
\]

with ORCA6.

OMol25 therefore provides the large-scale \(E/F\) supervision necessary to build a transferable potential.

Importantly, a four-million-structure subset additionally exposes raw ORCA outputs, GBW files and density matrices, providing a practical route for selecting structures for additional electronic-structure post-processing rather than recomputing the underlying DFT calculations from scratch.

The project will **not** attempt IQA analysis for the full OMol25 dataset.

Instead:

\[
\boxed{
\text{large }E/F\text{ dataset}
+
\text{small IQA-labeled subset}
}
\]

will be used.

---

# 6. Model Architecture

## 6.1 Backbone

The initial implementation should remain as close as possible to the original E3D-IQA formulation.

Preferred first implementation:

\[
\boxed{
\text{Allegro/e3nn backbone}
}
\]

rather than immediately moving to MACE.

Reason:

- Allegro already possesses an explicit edge-centric representation;
- the original E3D-IQA formulation is already defined on an Allegro-like architecture;
- \(D_{ij}\) naturally exists as a latent edge quantity;
- this avoids simultaneously solving both the IQA problem and the problem of inventing a pair-energy decomposition for MACE.

MACE can be investigated later as an alternative backbone.

---

## 6.2 Equivariant representations

Each edge carries equivariant latent features:

\[
h_{ij}^{(l)}
\]

constructed from:

\[
Z_i,\quad
Z_j,\quad
r_{ij},\quad
Y_{\ell m}(\hat{\mathbf r}_{ij}),
\]

and higher-order neighborhood information.

The equivariant network produces both scalar and higher-order irreducible representations.

Energy readouts must ultimately transform as:

\[
0e.
\]

---

# 7. Energy Decomposition

## 7.1 Node branch

The node branch represents intra-atomic energy:

\[
E_i^{\mathrm{intra}}
=
f_{\mathrm{node}}(h_i^{0e}).
\]

with:

\[
h_i
=
\sum_{j\in\mathcal N(i)}
\Phi_{\mathrm{node}}(h_{ij}).
\]

For IQA-labeled structures:

\[
E_i^{\mathrm{intra}}
\rightarrow
E_i^{\mathrm{IQA,intra}}.
\]

---

## 7.2 Edge branch

Each edge produces a scalar contribution:

\[
D_{ij}
=
f_{\mathrm{edge}}(h_{ij}^{0e}).
\]

For directed graphs, enforce symmetric pair energy:

\[
D_{ij}^{\mathrm{sym}}
=
\frac{1}{2}
\left(
D_{ij}+D_{ji}
\right).
\]

Total edge energy:

\[
E_{\mathrm{edge}}
=
\sum_{i<j}D_{ij}^{\mathrm{sym}}.
\]

---

## 7.3 Total energy

The total potential is:

\[
\boxed{
E_{\mathrm{ML}}
=
\sum_i E_i^{\mathrm{intra}}
+
\sum_{i<j}D_{ij}
}
\]

and forces are obtained from:

\[
\mathbf F_i
=
-\frac{\partial E_{\mathrm{ML}}}
{\partial\mathbf R_i}.
\]

Energy conservation therefore remains exact.

---

# 8. Training Dataset

## 8.1 Dataset A: OMol25 E/F dataset

Large dataset:

\[
\mathcal D_{EF}
=
\{
\mathbf R,Z,Q,M,E,\mathbf F
\}.
\]

Available metadata should retain:

- atomic numbers;
- coordinates;
- total charge;
- spin multiplicity;
- total energy;
- forces.

OMol25 explicitly includes charge and spin metadata and covers a much broader chemical domain than the original E3D-IQA proof-of-concept.

---

## 8.2 Dataset B: IQA subset

Construct:

\[
\mathcal D_{\mathrm{IQA}}
\subset
\mathcal D_{EF}.
\]

Additional labels:

\[
\{
E_i^{\mathrm{IQA,intra}}
\}.
\]

Optional diagnostic labels:

\[
E_{ij}^{\mathrm{IQA,inter}}
\]

may also be stored.

However, these pairwise IQA quantities should initially be used **only for validation**, not as direct training targets.

This preserves the defining experiment of E3D-IQA.

---

# 9. IQA Subset Selection

The IQA subset should not simply be selected randomly.

It should deliberately cover chemically important local environments.

Priority categories:

- H, C, N, O;
- S, P, F, Cl;
- neutral functional groups;
- charged functional groups;
- aromatic systems;
- heterocycles;
- alcohols;
- amines;
- carboxylates;
- amides;
- phosphates;
- sulfonamides;
- halogens;
- conjugated systems;
- hydrogen-bond donors/acceptors;
- ion pairs;
- strongly polarized motifs.

Later stages may add:

- metal coordination;
- unusual charge states;
- radicals;
- transition metals.

Initial development should avoid trying to reproduce all 83 OMol25 elements simultaneously.

---

# 10. IQA Data Generation

OMol25 electronic-structure releases provide a 4M subset with ORCA outputs, GBW files and density matrices.

The proposed IQA pipeline is:

```text
OMol25 index
    |
    v
select chemically representative structures
    |
    v
retrieve ORCA electronic structure
    |
    v
wavefunction / density conversion if required
    |
    v
IQA analysis
    |
    +--> atomic intra energies
    |
    +--> pair interatomic energies
    |
    v
store aligned with OMol25 structure ID
```

The expensive IQA calculation is therefore restricted to a relatively small subset.

---

# 11. Mixed-Supervision Training

For all samples:

\[
\mathcal L_E
=
\left|
E_{\mathrm{ML}}-E_{\mathrm{DFT}}
\right|^2,
\]

and:

\[
\mathcal L_F
=
\frac{1}{3N}
\sum_i
\left|
\mathbf F_i^{\mathrm{ML}}
-
\mathbf F_i^{\mathrm{DFT}}
\right|^2.
\]

For IQA-labeled structures only:

\[
\mathcal L_{\mathrm{IQA}}
=
\frac{1}{N}
\sum_i
\left|
E_i^{\mathrm{intra}}
-
E_i^{\mathrm{IQA,intra}}
\right|^2.
\]

Total loss:

\[
\boxed{
\mathcal L
=
w_E\mathcal L_E
+
w_F\mathcal L_F
+
w_I m_{\mathrm{IQA}}\mathcal L_{\mathrm{IQA}}
}
\]

where:

\[
m_{\mathrm{IQA}}
=
\begin{cases}
1,&\text{IQA label available}\\
0,&\text{otherwise}.
\end{cases}
\]

---

# 12. Training Strategy

## Phase 1 — Reproduce E3D-IQA

Goal:

Reproduce the published small-scale result before touching OMol25.

Requirements:

- reproduce total-energy accuracy;
- reproduce force accuracy;
- reproduce node-IQA correlation;
- reproduce latent edge/IQA-inter correspondence;
- reproduce the no-IQA negative control.

This stage verifies the implementation.

---

## Phase 2 — OMol25-small

Use a manageable OMol25 subset.

Example progression:

```text
10k
 -> 100k
 -> 1M
 -> 4M
```

The purpose is not immediately maximum accuracy.

The goal is to determine whether the IQA gauge survives increasingly heterogeneous \(E/F\) supervision.

---

## Phase 3 — Add IQA supervision

Start with a modest IQA dataset.

Possible initial scale:

```text
~1k structures
```

then:

```text
5k
10k
50k
```

depending on cost.

The critical quantity is not simply the number of IQA structures but the diversity of local environments.

---

# 13. Essential Ablation Study

At least four models should be compared.

### Model A — Energy/force only

\[
\mathcal L
=
\mathcal L_E+\mathcal L_F.
\]

Purpose:

establish the unconstrained latent gauge.

---

### Model B — E3D-IQA small only

Train only on IQA-labeled structures.

Purpose:

determine the decomposition obtainable without large-scale OMol25 regularization.

---

### Model C — OMol25 + IQA mixed training

\[
\boxed{
\text{main model}
}
\]

Purpose:

test whether large \(E/F\)-only supervision improves transfer while retaining IQA decomposition.

---

### Model D — Direct pair-IQA supervision

Optional control:

\[
\mathcal L
\rightarrow
\mathcal L
+
w_{\mathrm{pair}}
\mathcal L(D_{ij},E_{ij}^{IQA}).
\]

This should not initially be the primary model.

It tests whether explicit pair supervision improves or harms the decomposition.

---

# 14. Metrics

## 14.1 Conventional MLIP metrics

Report:

- energy MAE;
- energy RMSE;
- force MAE;
- force RMSE.

---

## 14.2 IQA node metrics

Evaluate:

\[
E_i^{\mathrm{ML,intra}}
\quad\text{vs}\quad
E_i^{\mathrm{IQA,intra}}.
\]

Metrics:

- MAE;
- RMSE;
- Pearson \(r\);
- Spearman \(\rho\);
- element-resolved errors;
- functional-group-resolved errors.

---

## 14.3 IQA edge metrics

Without direct pair training, evaluate:

\[
D_{ij}^{\mathrm{ML}}
\quad\text{vs}\quad
E_{ij}^{\mathrm{IQA,inter}}.
\]

Evaluate separately:

- attractive interactions;
- repulsive interactions;
- weak interactions;
- covalent pairs;
- hydrogen bonds;
- nonbonded close contacts;
- long-range pairs.

This diagnostic is central to determining whether the model has learned a useful physical gauge.

---

# 15. Transition to Alchemical Free Energy

Only after the node/edge decomposition is validated should the ABFE layer be introduced.

For a system containing ligand \(L\) and environment \(E\):

\[
E
=
E_{\mathrm{node}}
+
D_{LL}
+
D_{EE}
+
D_{LE}.
\]

where:

\[
D_{LL}
=
\sum_{i<j,\ i,j\in L}D_{ij},
\]

\[
D_{EE}
=
\sum_{a<b,\ a,b\in E}D_{ab},
\]

and:

\[
D_{LE}
=
\sum_{i\in L}
\sum_{a\in E}
D_{ia}.
\]

---

# 16. Definition of the Dummy Atom

A dummy atom is **not** a new chemical element.

Do not define:

```text
Z = dummy
```

and do not introduce a learned dummy embedding.

The ligand atom remains chemically real inside the ligand.

Its atom type, geometry and ligand-internal interactions remain unchanged.

Instead, dummy state is defined solely by ligand-environment coupling.

For ligand atom \(i\):

\[
\lambda_i=1
\]

means fully interacting.

\[
\lambda_i=0
\]

means dummy with respect to the environment.

---

# 17. Alchemical Hamiltonian

Define:

\[
\boxed{
U(\mathbf R,\boldsymbol\lambda)
=
\sum_iE_i^{\mathrm{intra}}
+
D_{LL}
+
D_{EE}
+
\sum_{i\in L}
\lambda_i
\sum_{a\in E}D_{ia}
}
\]

Thus:

\[
\frac{\partial E_i^{\mathrm{intra}}}
{\partial\lambda_j}
=
0,
\]

and:

\[
\frac{\partial D_{LL}}
{\partial\lambda_i}
=
0.
\]

Only cross-boundary interactions are decoupled.

---

# 18. Atom-wise Decoupling

For ligand atoms:

\[
L=\{1,\ldots,N_L\},
\]

start from:

\[
\boldsymbol\lambda
=
(1,1,\ldots,1).
\]

Sequentially transform:

\[
(1,1,1,1)
\rightarrow
(1,1,1,0)
\rightarrow
(1,1,0,0)
\rightarrow
(1,0,0,0)
\rightarrow
(0,0,0,0).
\]

At the final state:

\[
D_{LE}=0.
\]

But:

\[
E_L^{\mathrm{intra}}
+
D_{LL}
\]

remains unchanged.

The endpoint therefore contains an internally intact but environmentally decoupled ligand.

---

# 19. Thermodynamic Derivative

For atom \(i\):

\[
\frac{\partial U}
{\partial\lambda_i}
=
\sum_{a\in E}D_{ia}.
\]

This gives an immediately interpretable alchemical observable.

Thermodynamic integration becomes:

\[
\Delta G_i
=
\int_0^1
\left\langle
\sum_aD_{ia}
\right\rangle_{\lambda_i}
d\lambda_i.
\]

Alternatively:

- BAR;
- MBAR;
- λ-REMD;
- expanded ensemble;
- IBS-like adaptive sampling

can be applied.

---

# 20. Important Caveat: Linear Edge Scaling

The simplest Hamiltonian,

\[
D_{ia}(\lambda_i)
=
\lambda_iD_{ia},
\]

is the first implementation only.

It may produce poor overlap or endpoint singularities if the learned interaction becomes strongly repulsive.

Therefore later implementations should explore a smooth alchemical transform:

\[
D_{ia}^{\lambda}
=
s(\lambda_i,r_{ia},D_{ia}).
\]

Possible strategies include:

- softcore coordinate transforms;
- bounded energy transforms;
- ACE-style regularized softcore;
- nonlinear λ schedules.

The decomposition architecture and the alchemical path should remain conceptually separate.

---

# 21. Fundamental Consistency Tests

Before performing free-energy calculations, the model must satisfy several tests.

## 21.1 Full-state identity

At:

\[
\lambda_i=1\quad\forall i,
\]

require:

\[
U(\mathbf R,\mathbf 1)
=
E_{\mathrm{ML}}(\mathbf R).
\]

This should hold exactly.

---

## 21.2 Ligand internal invariance

For arbitrary \(\boldsymbol\lambda\):

\[
\boxed{
U_{LL}(\boldsymbol\lambda)
=
U_{LL}(\mathbf1)
}
\]

numerically to machine precision.

---

## 21.3 Final dummy state

At:

\[
\boldsymbol\lambda=\mathbf0,
\]

require:

\[
U(\mathbf R,\mathbf0)
=
U_L
+
U_E.
\]

No ligand-environment edge contribution may survive.

---

## 21.4 Force continuity

Check:

\[
\mathbf F_i(\lambda)
\]

across:

\[
0\le\lambda\le1.
\]

No discontinuities or divergent forces should occur.

---

## 21.5 λ derivative

Compare autograd:

\[
\frac{\partial U}{\partial\lambda_i}
\]

against finite differences.

---

# 22. Path Independence Test

Because free energy is a state function, the final decoupling free energy should not depend on the order of atom removal when sampling is converged.

Example:

```text
A -> B -> C -> D
```

versus:

```text
D -> C -> B -> A
```

must satisfy:

\[
\Delta G_{\mathrm{path1}}
\approx
\Delta G_{\mathrm{path2}}.
\]

This is one of the most important validation experiments.

A substantial discrepancy indicates:

- inadequate sampling;
- invalid decomposition;
- problematic λ path;
- or hidden λ dependence of ligand-internal terms.

---

# 23. Free-Energy Validation Ladder

Do not start with a GPCR system.

Use a staged hierarchy.

## Stage A — Molecular dimers

Examples:

- water dimer;
- methane-water;
- benzene-water;
- ammonia-water.

Purpose:

verify edge decoupling.

---

## Stage B — Solvation free energy

Small neutral molecules in explicit solvent.

Compare:

\[
\Delta G_{\mathrm{solv}}^{\mathrm{alchemical}}
\]

against conventional FEP/experimental benchmarks.

---

## Stage C — Host–guest binding

Use relatively rigid host–guest systems.

Purpose:

test restraint handling and complete ligand decoupling.

---

## Stage D — Protein–ligand ABFE

Only after earlier tests pass.

Initial systems should preferably involve:

- neutral ligands;
- compact binding pockets;
- well-defined poses;
- modest conformational changes.

---

# 24. Binding Free Energy

For a ligand:

\[
\Delta G_{\mathrm{decouple}}^{\mathrm{complex}}
\]

and:

\[
\Delta G_{\mathrm{decouple}}^{\mathrm{solvent}}
\]

are evaluated with the same alchemical Hamiltonian.

Then:

\[
\boxed{
\Delta G_{\mathrm{bind}}
=
\Delta G_{\mathrm{decouple}}^{\mathrm{solvent}}
-
\Delta G_{\mathrm{decouple}}^{\mathrm{complex}}
+
\Delta G_{\mathrm{restraint}}
+
\Delta G_{\mathrm{standard}}
}
\]

with the appropriate sign convention fixed consistently in implementation.

---

# 25. Relation to Existing ABFE

Conventional molecular-mechanics ABFE often separates:

```text
charge decoupling
      |
      v
vdW decoupling
      |
      v
dummy ligand
```

The proposed method instead provides a learned interatomic decomposition:

```text
E3D-IQA
   |
   v
D_ij
   |
   +--> ligand-ligand
   |
   +--> environment-environment
   |
   +--> ligand-environment
               |
               v
          atom-wise λ
```

The defining feature is therefore not merely the use of an ML potential.

The methodological novelty is:

\[
\boxed{
\text{physically constrained latent energy decomposition}
\rightarrow
\text{alchemical Hamiltonian}
}
\]

---

# 26. Connection to RBFE

After ABFE validation, the same framework can be extended to RBFE.

For transformation:

\[
L_A\rightarrow L_B,
\]

shared atoms remain active while disappearing/appearing atoms carry alchemical occupation parameters.

The node/edge decomposition potentially allows a cleaner treatment of:

- disappearing atoms;
- appearing atoms;
- maximum-common-substructure transformations;
- multi-topology mappings.

RBFE should remain a later-stage extension.

ABFE provides the cleaner first validation because the final endpoint is unambiguous.

---

# 27. Major Technical Risks

## Risk 1 — IQA gauge does not scale

A small amount of IQA supervision may fail to control the decomposition when millions of heterogeneous OMol25 structures dominate the optimization.

Mitigation:

- increase IQA sampling ratio;
- stratified IQA mini-batches;
- periodic IQA-only optimization steps;
- higher \(w_I\);
- curriculum training.

---

## Risk 2 — Local cutoff loses important interactions

IQA pair energies are not strictly local.

A finite graph cutoff may omit long-range electrostatic contributions.

Possible solutions:

- increase cutoff;
- explicit electrostatic branch;
- multipolar long-range model;
- separate long-range correction;
- evaluate only within the ML interaction cutoff during initial development.

---

## Risk 3 — Pair gauge remains non-unique

Node IQA supervision fixes a large part of the energy gauge but may not uniquely determine every individual \(D_{ij}\).

Evaluation must therefore examine:

\[
D_{ij}
\leftrightarrow
E_{ij}^{IQA}
\]

carefully.

If necessary, introduce weak pair supervision later.

---

## Risk 4 — Alchemical path produces poor overlap

Even a correct endpoint decomposition does not guarantee good free-energy convergence.

Potential solution:

- softcore edge scaling;
- staged atom removal;
- adaptive λ placement;
- REMD;
- IBS.

---

## Risk 5 — Charged ligands

Charged systems introduce additional complications:

- long-range electrostatics;
- finite-size effects;
- net-charge corrections;
- potentially nonlocal IQA contributions.

Initial ABFE validation should therefore focus on neutral ligands.

Charged systems should be treated as a separate methodological extension.

---

# 28. Computational-Cost Strategy

The project must avoid full-scale retraining before the concept is validated.

Recommended progression:

```text
E3D-IQA reproduction
        |
        v
10k OMol25
        |
        v
100k OMol25
        |
        v
1M OMol25
        |
        v
4M OMol25
        |
        v
larger scale only if justified
```

Likewise for IQA:

```text
100
 -> 1k
 -> 5k
 -> 10k+
```

The project should determine the minimum amount of IQA supervision required to stabilize the decomposition.

---

# 29. Minimal Viable Project

The MVP is **not** an ABFE calculation.

The MVP is:

\[
\boxed{
\text{OMol25-trained E3D-IQA model}
}
\]

that demonstrates simultaneously:

### Criterion 1

Good total energies:

\[
E_{\mathrm{ML}}\approx E_{\mathrm{DFT}}.
\]

### Criterion 2

Good forces:

\[
\mathbf F_{\mathrm{ML}}
\approx
\mathbf F_{\mathrm{DFT}}.
\]

### Criterion 3

Node energies recover IQA intra-atomic energies:

\[
E_i^{\mathrm{intra}}
\approx
E_i^{\mathrm{IQA,intra}}.
\]

### Criterion 4

Edge energies correlate with withheld IQA interatomic terms:

\[
D_{ij}
\sim
E_{ij}^{\mathrm{IQA,inter}}.
\]

Only after these four criteria pass should alchemical development begin.

---

# 30. Second MVP: Dummy Hamiltonian

The second milestone is:

\[
\boxed{
\text{atom-wise LE edge decoupling}
}
\]

without performing full binding calculations.

For a small molecular complex verify:

\[
\lambda_i=1
\Rightarrow
\text{full interaction}
\]

and:

\[
\lambda_i=0
\Rightarrow
D_{iE}=0
\]

while:

\[
D_{iL}
\]

and:

\[
E_i^{\mathrm{intra}}
\]

remain unchanged.

---

# 31. Third MVP: Free-Energy Closure

Perform sequential decoupling in multiple orders.

For example:

\[
1\rightarrow2\rightarrow3\rightarrow4
\]

and:

\[
4\rightarrow3\rightarrow2\rightarrow1.
\]

Require:

\[
\left|
\Delta G_A-\Delta G_B
\right|
\]

to be within statistical uncertainty.

This provides the first strong evidence that the learned decomposition is suitable for thermodynamic integration.

---

# 32. Proposed Repository Structure

```text
e3d_iqa_omol/
|
├── README.md
├── PROJECT_PLAN.md
|
├── configs/
│   ├── e3d_iqa_small.yaml
│   ├── omol_10k.yaml
│   ├── omol_100k.yaml
│   ├── omol_1m.yaml
│   └── alchemical.yaml
|
├── data/
│   ├── omol/
│   ├── iqa/
│   └── splits/
|
├── src/
│   ├── model/
│   │   ├── backbone.py
│   │   ├── node_energy.py
│   │   ├── edge_energy.py
│   │   └── e3d_iqa.py
│   │
│   ├── training/
│   │   ├── losses.py
│   │   ├── mixed_sampler.py
│   │   └── trainer.py
│   │
│   ├── iqa/
│   │   ├── parser.py
│   │   ├── mapping.py
│   │   └── dataset.py
│   │
│   └── alchemy/
│       ├── partition.py
│       ├── lambda_mask.py
│       ├── softcore.py
│       └── calculator.py
|
├── validation/
│   ├── energy_force/
│   ├── iqa_node/
│   ├── iqa_edge/
│   ├── lambda_derivative/
│   └── path_independence/
|
└── scripts/
    ├── build_omol_subset.py
    ├── prepare_iqa.py
    ├── train.py
    ├── evaluate_iqa.py
    ├── test_dummy.py
    └── run_ti.py
```

---

# 33. Suggested Development Order

```text
[1] Reproduce E3D-IQA
          |
          v
[2] Port OMol25 E/F loader
          |
          v
[3] Mixed IQA + E/F batches
          |
          v
[4] Train 10k / 100k OMol pilot
          |
          v
[5] Generate targeted IQA subset
          |
          v
[6] Quantify node/edge gauge
          |
          v
[7] Scale to 1M–4M OMol25
          |
          v
[8] Add LL / LE / EE edge labels
          |
          v
[9] Add atom-wise λ mask
          |
          v
[10] Validate endpoint identities
          |
          v
[11] Add softcore transformation
          |
          v
[12] TI / MBAR on molecular dimers
          |
          v
[13] Solvation FE
          |
          v
[14] Host–guest ABFE
          |
          v
[15] Protein–ligand ABFE
```

---

# 34. Go / No-Go Criteria

## Go from E3D-IQA to OMol scaling

Proceed if:

- original E3D-IQA behavior is reproducible;
- IQA supervision clearly changes the latent decomposition;
- E/F accuracy is not substantially degraded.

---

## Go from OMol training to alchemy

Proceed if:

\[
E_i^{\mathrm{intra}}
\]

generalizes to held-out molecules and:

\[
D_{ij}
\]

shows a stable relationship with held-out IQA interatomic energies.

---

## Go from dummy model to free energy

Proceed if:

- \(\lambda=1\) exactly reproduces the original model;
- \(\lambda=0\) eliminates all selected LE edges;
- LL energies are exactly invariant;
- forces are smooth;
- finite-difference and autograd λ derivatives agree.

---

## Go from toy systems to ABFE

Proceed only if atom-removal-order closure is achieved within sampling uncertainty.

---

# 35. Expected Scientific Contributions

The project potentially contributes three separate methodological results.

## Contribution I — Large-scale E3D-IQA

Demonstrate that sparse IQA supervision can constrain latent energy decomposition in a large equivariant potential trained on OMol25-scale data.

---

## Contribution II — IQA-constrained alchemical decomposition

Convert the learned node/edge representation into a thermodynamic decomposition:

\[
\text{latent physical representation}
\rightarrow
\text{alchemical coordinate}.
\]

---

## Contribution III — Atom-wise dummy free-energy path

Introduce a ligand decoupling scheme in which individual ligand atoms become environmentally invisible while preserving the full ligand-internal Hamiltonian.

This directly avoids repeated counting/removal of ligand internal contributions.

---

# 36. Central Concept

The complete project can be summarized as:

\[
\boxed{
\begin{array}{c}
\text{OMol25}\\
E,\mathbf F\\
+\text{ sparse IQA}\\
\downarrow\\
\textbf{E3D-IQA}\\
\downarrow\\
E_i^{\mathrm{intra}}
+
D_{ij}^{\mathrm{inter}}\\
\downarrow\\
D_{LL}+D_{EE}+D_{LE}\\
\downarrow\\
D_{iE}\rightarrow\lambda_iD_{iE}\\
\downarrow\\
\text{atom becomes environment-dummy}\\
\downarrow\\
\textbf{ABFE / RBFE}
\end{array}
}
\]

The core methodological principle is therefore:

\[
\boxed{
\textbf{First fix the energy gauge, then perform alchemy.}
}
\]

The project should not begin by inventing a dummy atom inside an unconstrained ML potential.

It should first establish a physically meaningful decomposition of the learned Hamiltonian and only then use that decomposition to define the alchemical pathway.
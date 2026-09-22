# E3D-IQA4MACE · ABFE

**Atom-resolved binding free energies from a physically gauged equivariant
potential — one calculation, every atom accounted for.**

Most equivariant MLIPs write `E = Σᵢ Eᵢ`, and because message passing mixes
everything into each node, an atom's `Eᵢ` has no unique split into
intramolecular, electrostatic and polarization parts. Scale a ligand atom's
`Eᵢ` by λ and you have not removed its interaction with the environment —
you have removed an arbitrary mixture of it and the ligand's own internal
energy, differently for each atom.

This project makes the removal exact:

```
E = Σᵢ E_intra(i) + Σ_{i<j} D_ij          (decomposition, pinned to observables)
D_LE = Σ_{i∈L} Σ_{a∈E} D_ia                (ligand–environment coupling)
```

With the decomposition **gauged to physical observables** (not merely a
reparameterization of the total energy), scaling the coupling by per-atom λᵢ
and integrating along the diagonal path yields a per-atom decomposition of the
decoupling free energy that sums exactly to the total and is independent of any
atom-removal order — it is the Shapley value, for the cost of one ordinary ABFE.

| | per-atom dG | QM accuracy | polarization / charge transfer |
|---|---|---|---|
| MM-GBSA per-residue | yes (unreliable) | no | no |
| MM ABFE per-group decoupling | yes | no | no |
| plain `Σᵢ Eᵢ` MLIP | no | yes | yes |
| **this work** | **yes** | **yes** | **yes** |

Full design, measurements and go/no-go criteria: **[RESEARCH_PLAN_v2.md](RESEARCH_PLAN_v2.md)**
(v1 is preserved as [PROJECT_PLAN_v1.md](PROJECT_PLAN_v1.md)).
Where things stand right now, and what is half-done: **[STATUS.md](STATUS.md)**.

## Status — work in progress

The gauging programme (WP1–WP7) is done and measured, and the first complete
five-term model is trained. WP8's machinery is built and verified end to end;
what is missing is a free-energy calculation with real sampling — every result
so far deliberately averages over fixed conformers to isolate the attribution
arithmetic from sampling convergence. Validation ladder: **A** dimers →
**B** FreeSolv solvation → **C** SAMPL host–guest → **D** protein–ligand ABFE
(currently at stage **A** scale on DES370K dimers).

- **Negative control reproduces** (WP1): without IQA supervision the node/edge
  split does not converge to IQA on its own — 28× off on synthetic labels, 49×
  on a frozen `mace-omol-0` backbone; with supervision it reorganises at almost
  unchanged E/F accuracy.
- **`L_int` pins the pair sum** (WP3): `Σ D_AB` matches CCSD(T)/CBS interaction
  energies to **0.27 kcal/mol** on 228 held-out DES370K systems.
- **Endpoints exact** (WP6): λ must mask the *graph*, not just edge energies —
  edge-only scaling leaves `U(R,0) ≠ U_L + U_E` by 5.9% of the coupling; graph
  masking passes all six consistency tests.
- **Order-independent attribution** (WP7): the diagonal path agrees with
  exhaustive Monte-Carlo Shapley to **0.82%** of the total.
- **SAPT component supervision** (per-channel es/ex/ind/disp): closes 85% of
  the gap to an edge-by-edge oracle, 84–94% on real SAPT data. Supervising only
  the boundary *total* leaves the individual channels 3–7× larger than the true
  components — a direct display of pair non-identifiability on real data.
  Operating point: `sapt+total`.
- **The alchemical Hamiltonian runs under OpenMM** (WP8): via
  `openmm.PythonForce`, so nothing needs to be TorchScript-able and λ is an
  ordinary attribute. The endpoint identity survives the bridge to 1.2e-7 eV.

**First complete model** (arm D, five terms `E + F + L_int + L_IQA + L_SAPT`,
frozen `mace-omol-0`, head only): 4.3 meV/atom against the backbone's own
energy, 0.062 eV/Å forces, 0.53 kcal/mol interaction energies, 0.6–1.7 kcal/mol
per SAPT component — 6–11× better channels than the unsupervised baseline, for
about 2× on energy against arm C. That cost is an upper bound: D was still
descending at 32k steps. Checkpoint in `ckpt/`, numbers in `logs/joint_*.json`.

**Not yet done:** a free-energy calculation with real sampling (WP8/WP9), and
real IQA labels — every node-gauge result to date uses a synthetic teacher, so
they show that *a* gauge can be imposed on frozen features, not that the *IQA*
gauge lives there.

## Repository layout

```
decomp/                 the package
  head.py               backbone-agnostic decomposition head (E_intra per node, D_ij per edge)
  mace_adapter.py       attach the head to a frozen MACE backbone (WP4)
  lambda_mask.py        atom-wise alchemical coupling — "edge" vs "graph" modes (S5)
  losses.py             mixed supervision: E, F, IQA gauge, L_int, SAPT (S4.3)
  train.py              training step for the defining experiment
  data.py               DES370K dimer batches
  fe.py                 TI along the diagonal path, MBAR (WP8)
  sampling.py           Langevin sampling on the alchemical Hamiltonian
  openmm_bridge.py      run the learned Hamiltonian inside OpenMM

test_*.py               one experiment per file (see RESEARCH_PLAN_v2.md §13)
                        endpoints / attribution / openmm run in seconds, no data
prepare_data.py         one-time DES370K → train/val/test splits (by system, not by row)
run_queue.sh            cluster queue for the experiment sequence
RESEARCH_PLAN_v2.md     design + measurements (the real documentation)
PROJECT_PLAN_v1.md      v1 framing, superseded
```

## Reproduce

Environment: `e3nn >= 0.5.1`, `torch`, `mace-torch`, `pymbar`, `openmm`.
(dev env: miniforge, e3nn 0.5.1 / torch 2.12.1 / mace-torch 0.3.16)

```bash
# 1. Download DES370K from DESRES (https://www.deshawresearch.com/downloads/)
#    and place DES370K.csv in the repo root.
# 2. Build element/charge-filtered splits (hcno, organic, full):
python prepare_data.py

# 3. Run the experiment sequence (negative control, gauge, endpoints,
#    attribution, joint arms A–D):
./run_queue.sh            # or run individual test_*.py
```

Each `test_*.py` writes a JSON record under `logs/` and a head checkpoint
under `ckpt/` (both gitignored); the numbers quoted above are from
`RESEARCH_PLAN_v2.md §13`, which cites them.

## References

- Shimamura & Nomura, *E3D-IQA* — arXiv:2609.00674 (the defining experiment:
  IQA supervision reorganises the node/edge split of an equivariant potential).
- DES370K: Merchant et al., a def2-TZVPD DFT dataset of 370k counterpoise-corrected
  CCSD(T)/CBS dimer interaction energies (D. E. Shaw Research).
- MACE: Batatia et al.; `mace-omol-0` backbone from the OMol25 model zoo.

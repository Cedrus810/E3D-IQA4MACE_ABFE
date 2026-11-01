# Status — 2026-10-23 (rev 12; open items updated through 2026-10-23, the rest as of rev 10)

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
D19 the decoupled end (user, 2026-10-18; v3 S6.3.4a): **option 3**. ML
carries only L and truth(L, I); the shell water is TIP3P; SW and dG_ref are
dropped. A finite-k FIRES residual that depends on the ghost's shape
remains, to measure. In code (`run_abfe.py --mode mlmm --shell-mm`), exact
checks and an acetamide smoke pass (v3 S13.53); no free energy yet.
Existing totals stay as diagnostics.
**D19 paused for expansion (user, 2026-10-21)**: a candidate to validate; see the D19 review at the
top of the open items (v3 S13.56-13.59).

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
    2. the all-MM path control with matched ends on acetamide (vs +34.04):
       done (S13.48). MBAR +33.54 +- 0.10 over 3 seeds, -0.50 from the
       reference; still drifting up between halves (second halves +33.78),
       so most likely under-equilibrated W3 refill. Seed 1 extended to
       2 ns on the 5090: last ns +34.05 +- 0.24, and the 0.5 ns blocks
       scatter by 0.72, so the -0.5 is sampling. FIRES on MM waters gives
       +34.87 +- 0.54. **The path reproduces the MM decoupling**; it is not
       the source of the 85 kJ/mol;
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

**Current design (user, 2026-10-23): `DESIGN_2026-10-23_groupwise_dummy.md`** -- groups become dummy one by one (states = groups + 1, no lambda windows / walls / GCMC), MM samples and saves frames, the model evaluates all frames in one batch. ML-driven sampling (`--mode mlmm`) is not used. Earlier note: First step, MM only on
acetamide (CH3 | C(=O)NH2, 61 states): smoke on 183 (10 samples/state) +34.43 +- 1.62 vs MM +34.04.
Full run on the 5090 (15:09, `run_5090.sh`) STOPPED by the user at 15:34 after ~2-4 states per job:
the script saves only neighbour energies, not frames, so the model cannot be applied to them afterwards.

**One-step jump + water term on acetamide, MM only (2026-10-23, v3 S13.64).** Total +33.90 against
MM MBAR +34.04 kJ/mol (jump BAR +56.85 + wall-growth -22.95 + 0.002), but the jump's forward / reverse
FEP disagree by 53 kJ/mol. Checks: r_in 0.25 over 3 seeds +35.16 +- 1.14 (1.1 above MM); r_in 0.28
+37.33 (should equal it): within 1-3 kJ/mol of MM, a residual r_in dependence of ~2 kJ/mol unresolved;
the jump is the weak step. The P(N=0) term needs wall growth; the MC histogram never reaches N = 0.

**Direction (user, 2026-10-23, ~12:30): the model is a decomposition tool, not a sampling engine.**
Removing every ligand atom's coupling is the binding / solvation free energy, so the total comes from
the per-atom attribution itself. MM samples (the existing MM ABFE windows, frames saved); the model is
evaluated on those frames in large batches (per-atom D_iE = dU/dlambda_i), integrated over lambda to
per-atom dG_i, summing to the total. ML-driven ML/MM ladders (`--mode mlmm`) are not the way forward.
Open: whether to reweight MM -> ML (the user's call); which IBS runs saved frames.

**HSP90-27 on the POLAR-shell path -- STOPPED 2026-10-23 ~12:30 by the user (direction above).** User: the path is
POLAR-shell (+33 cause fixed by it); the wall now covers ions; the ladder minimises each replica on a
fresh start. `experiments/2026-10-23_hsp90_27_polar/run.sh`: segAB q 0-3 and segC q 3-4 (W3),
20 ps per state, est. ~18:00 / ~14:00 from the first iterations. Shape leg (q 4-5) not run: in the
ladder smoke q = 5 went NaN after EM (a separate EM of the same state is stable; not explained).

**POLAR-shell candidate and propagation optimizations (2026-10-23, user: implement and speed up).**
`--polar-shell` implements option 2 as W1 | W2 | cut | W3 | shape, with POLAR water-water retained
and a fixed-identity energy bridge to one-point FIRES. Its endpoint is a POLAR droplet in MM solvent,
not bulk-MM water; it is an explicit candidate, not evidence that the +33 gap is repaired.
Exact decoupled-endpoint shortcuts are implemented, with a `--reference-forces` A/B path.
Coupled-endpoint and scalar interior propagation plus optional `--batch-fragments` pass FP64
checks only; FP32 force differences failed the preset threshold, so FP32 coupled/interior states
retain the reference algorithm and FP32 batching is rejected. Checkpoint model/identity mismatches
fail before restoring histories.
Protocol and verification: `IMPLEMENTATION_2026-10-23_POLAR_SHELL.md` and
`experiments/2026-10-23_polar_shell/RESULT.md`. Production-GPU timing and scientific acceptance remain open.
CPU validation: 9 regressions, 16 real-POLAR FP64 cases, and three FP32 cut-endpoint cases pass.
FP32 q=3 force-call latency ratios are 2.20x (POLAR shell) and 17.64x (D19); these are not GPU
or full-ladder speedups. The FP32 reference-repeat check also exceeded the preset force threshold;
retain that failure and calibrate the noise before enabling additional FP32 optimizations.
**5090, FP32, paired reference/fast (2026-10-23, v3 S13.60)**: lambda = 1 and interior states unchanged
(~90 ms/step); POLAR-shell q >= 3 92 -> 57 ms/step (1.6x); D19 only q = 3 gains (94 -> 32). GPU FP32
repeat calls on one frame differ by 0.14-0.55 kJ/mol/A in force (ML states), 3-11x the 0.05 threshold:
that threshold cannot decide FP32 equivalence on the GPU.
**FP32 gates removed (user, 2026-10-23, v3 S13.61)**: the model is FP32 only; FP64 checks prove the algebra.
With `--batch-fragments` on the 5090: POLAR-shell lambda=1 106 -> 48, interior 92 -> 65, W3 94 -> 34
ms/step (shape-leg timings void: q >= 4.5 goes NaN within 5 steps from the leg start frame, any force variant;
a test set-up error, no EM before MD; with steepest-descent EM they run (shape ~50 ms/step, 34 ms force call);
the ladder itself has no EM step, so q >= 4.5 needs EM or `--init-from`; RESULT.md "更正", "EM"); D19 lambda=1 107 -> 73, interior 92 -> 66, q=3 94 -> 33. `--batch-fragments` is
now on by default in `run_abfe.py` / `profile_mlmm.py` (user, 2026-10-23); `--no-batch-fragments` turns it off,
`--reference-forces` turns off both shortcuts and batching.
**Concurrency (v3 S13.62)**: N processes on one 5090 saturate at N = 4 (total 1.28-1.44x of N = 1; N = 8 adds
2-5%; 2.7 GB each). Beyond that: replica batching in one backbone call (not implemented).

**D19 review: conclusions as of 2026-10-21 (v3 S13.56-13.59).** D19 is **paused for expansion**
and kept as a candidate to validate (user, 2026-10-21). What stands:
1. **D19 on acetamide, first full path** (S13.56): MBAR +67.05 +- 2.54 kJ/mol against MM +34.04,
   E3D - MM = +33.0. W1 and W3 match the all-MM control E8; the gap is W2 (+25.9) and the cut (+5.4,
   not converged: halves +1.4 / +9.4). 20 ps per state, one seed.
2. **Not POLAR on the same configurations** (S13.57): on GAFF2/TIP3P frames POLAR's ligand-shell
   coupling differs from MM by -4.0 +- 2.4 kJ/mol. This points to an ensemble effect; it does not
   exclude other energy terms or establish the source of the free-energy gap.
3. **D19 samples shells with a large POLAR many-body term** (S13.57-13.59): -71 (production
   identity) against -5 on MM frames and -16 on the old path's frames; the two-body sum stays near
   MM. From four shared starts (same identity, seed, velocities; 10 ps) D19 ends with the stronger
   many-body term every time (mean -24) and the stronger coupling (truth mean -35).
4. **The old model penalises those shells** (S13.58): on D19's frames U_old - U_D19 is ~+41 kJ/mol
   above the MM baseline, offsetting ~60% of their many-body advantage. The two water models drive
   the shell apart within ~0.5 ps; their relative preferences for the two shell ensembles differ
   by ~150-180 kJ/mol (S13.59), not a separate penalty of that size under each model.
5. **Reading** (hypothesis, supported, not proven): replacing the shell's POLAR energy by TIP3P
   changes the relative cost of shell configurations and over-favours some cooperative ones.
   The old model also has more negative coupling energies (truth ~-166 vs -99 on MM frames).
   A common contribution to the +33 free-energy difference remains a hypothesis, not a deduction.
6. **Not settled**: no independent reference says which shell is right; configuration energies,
   not free energies; 10-20 ps, one seed each. Not to be "fixed" by suppressing the many-body term,
   the head or the cap (they do not set the lambda = 1 energy).
**Next**: validate the implemented option-2 candidate and choose an independent physical reference
before judging shell accuracy. Option 1's reference-leg convergence problem remains.

**Next after the D19 acetamide ladder: speed (user, 2026-10-20; v3 S8 top).** The ML/MM step is
0.19-0.33 s per replica on a 2080 Ti and on an RTX 3090 alike, 200-300x slower per step than MM
ABFE; fix the per-step overhead (profile first; batch replicas / backbone calls) before going wide.
Implementation audit and bounded profiling entry: `PERFORMANCE_PLAN_2026-10-20.md`,
`profile_mlmm.py`. Per-step Euler Coulomb attribution and all-edge head readouts are additional
optimization candidates before replica batching. This was the pre-profile audit; equivalent force
optimizations were subsequently implemented on 2026-10-23. The current session cannot access the
NVIDIA driver; production-GPU speedups remain unmeasured.
Baseline profile run 2026-10-20 (v3 S13.55, 2080 Ti): ML states 342-367 ms/step, W3 (ligand
only) 142-152; backward ~47% and three backbone forwards ~45%; 4-14 thousand CUDA kernels per force
call with the GPU busy ~10-30% -- launch-bound. Coulomb split ~4%, head < 1%.

**HSP90-27 longer A run (old path, diagnostic only; 2026-10-19; v3 S13.54).** MBAR +168.47 +- 1.79
at 20 ps per state against +168.06 +- 3.57 at 5 ps: A does not move with sampling. Its TI is broken
(dG_W2 = -2.6e7 kJ/mol) because Na+/Cl- enter the ghost ligand at lambda = 0 (the wall covers waters
only), so the unscaled ligand-ion LJ in dU/dlambda diverges. Energies stay finite; only TI and the
per-atom split are affected. Matters only when the salted IBS legs are run on D19 for attribution.
Acetamide has no ions.

**Correction (2026-10-18; v3 S13.52):** dG_ref differs by +89.6 kJ/mol between two alike IBS boxes,
and its ML-water end drifts within each run. So SW / dG_ref are unconverged, and the +27 offset below is
not established; with HSP90-27's own-box ref its E3D - MM goes from +46.7 to -42.9.

**First cross-molecule result (2026-10-18; v3 S13.51).** E3D vs IBS MM on 5 neutral legs:
E3D - MM = +13 / +47 / +17 / +19 / +37 kJ/mol (mean +26.8, SD 14.5), all positive. The causes are
not separated (5 ps per state; the ML-vs-MM Hamiltonian and the seam; the cut ~+30; one dG_ref for
six boxes). HSP90-19's tail and acetamide's dG_ref are still to come.

**Running at hand-off (2026-10-17 15:25; v3 S13.50).** ML/MM horizontal test on the revised path
(SW before an all-MM W3). Queues: `experiments/2026-10-17_set_run/run_set2_183.log` (workstation B:
full HSP90-27, CypD-8, then tails 17, 19) and `run_set2_181.log` (workstation C: after acetamide's
first pass, full CypD-4, then tails PWWP1-8, acetamide); PIDs in that directory's `pids.txt`.
Each job resumes if rerun. **Next**: the dG_ref leg (`run_abfe.py --mode mlmm --ref --qs 3,...,4
--init-from <v2_tail or v2_full run> --init-q 5`) once a tail has a refilled q = 5 state; until it
is subtracted every total carries the ~4e6 kJ/mol SW constant and must not be compared with MM.
Then: one table of all molecules against their IBS MM values.

**Reframed 2026-10-17 (user; v3 S8 top, S13.49):** the method is general.
It is judged across neutral molecules (6 IBS legs + acetamide + Atenolol)
against their MM values, breadth first. The IBS MM values are the
reference; the earlier 85 kJ/mol gap was E3D's incomplete total, not the
reference. Next: frames and clusters for all molecules -> one pooled head
(some held out) -> ML/MM on all of them, with the D18 legs and the packed
windows.

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
- **0 x inf in OpenMM custom expressions.** A term scaled to zero by a
  global (lam_lj = 0) still evaluates (s/r)^12. In mixed precision that
  overflows when atoms overlap, which they do at a decoupled end, and gives
  NaN. Floor r or use a softcore (S13.48).
- **Several ladder processes on one GPU** fail at the first Context with
  "No compatible CUDA device". This happened on the 2080 Ti and on the
  5090, and is not memory; the cause is not known (S13.48). Run one
  process per GPU.
- **`set -u` in launch scripts** breaks `conda activate`, whose activate.d
  scripts read unset variables. Wrap the activation in `set +u` ... `set -u`.
- **MBAR's error on HREMD samples understates the scatter.** In S13.48
  the 0.5 ns blocks scatter twice as much as the MBAR error. Judge
  convergence by blocks and by independent repeats.
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

"""Same-temperature eta-label exchange on one endpoint (ETA_EXCHANGE_PLAN_2026-10-29.md; DESIGN §10).

R walkers each carry a full configuration, their own MM Context and the cached dU / atomic decomposition.
Every sweep: one existing MM-HMC proposal per walker at its current label (eta_hmc.hmc; ML gives no force),
then exchanges of non-overlapping adjacent label pairs (k, k+1), k = sweep % 2 (mod 2), reading only cached dU:
    log R = (eta_k - eta_{k+1}) [dU(x_a) - dU(x_b)] / RT,   a = walker at eta_k, b = walker at eta_{k+1}.
Labels move; coordinates, Context, dU, atoms and Omega stay with the walker. --no-exchange keeps the labels
fixed (matched control: same labels, kernel and model-call budget).
The ladder is committed as one transaction (all Context checkpoints, local and exchange RNGs, permutation,
caches); a resume replays an uninterrupted run. No hash; no splicing of independent chains.

--eta0-mm-updates m (> 1): the walker holding eta=0 at the start of a sweep instead does m MM-only HMC substeps and
one model query at their end (eta0_mm_kernel; ETA0_MM_UPDATES_DECISION_2026-11-01.md); other labels unchanged.

Run: python eta_exchange.py ladder SYSTEM --etas 0,0.25,0.5,1 --start REP --sweeps N --tag T [--head H]
     ... --sweeps 1000 --tag T2 --extend-from T     (continue a completed T ladder; T is not modified)
"""

import argparse
import json
import pathlib
import re
import time

import numpy as np

SCHEMA = "eta_exchange-1"
DETAILS = ("C_ML", "C_ML_learned", "C_ML_coul", "C_MM", "dU_atom", "cluster_system_indices", "omega_water_resid")
PER_WALKER = ("acc", "dH", "dE", "V", "dU", "prop_V")
PROVENANCE_ONLY = {"head_mtime", "backbone_mtime", "checkpoint_device", "continuations"}
# OpenMM Context checkpoints loaded on a different GPU silently corrupt the following MM integration (2026-10-30:
# 4080 SUPER -> 3090 / 2080 Ti gave dH +48 kJ/mol, NaN or segfaults; same-GPU restarts normal). A checkpoint is
# therefore loaded only on the GPU model that wrote it; otherwise the walker restarts from its exact cached
# coordinates (HMC redraws momenta every cycle, so positions are the full Markov state) with a new integrator seed,
# recorded as a continuation (DESIGN §5: a coordinate-only restart is a new continuation segment, not exact replay).
SEED_STRIDE = 7919
# Restart recheck of the cached energy / atoms at the committed state (kJ/mol). It guards against a different
# target (head, Omega rule, mask: differences of 1-100 kJ/mol); 1e-3 kJ/mol = 4e-4 kT is far below any effect on
# acceptance and above the ~1e-4 kJ/mol cache/Context rounding found in pre-2026-10-30 ladder records.
RESTART_ATOL = 1e-3


# ---------------------------------------------------------------- exchange step (pure)

def exchange_step(perm, e, etas, phase, rng, rt, enabled=True):
    """One round of non-overlapping adjacent label swaps. perm[k] = walker at label k; e[w] = cached dU of walker w.
    Returns new perm and per-pair (attempted, logR, log_u, accepted); rejection leaves perm unchanged."""
    K = len(etas)
    perm = np.array(perm, copy=True)
    att = np.zeros(K - 1, bool)
    logR, logu = np.full(K - 1, np.nan), np.full(K - 1, np.nan)
    acc = np.zeros(K - 1, bool)
    if enabled:
        for k in range(phase, K - 1, 2):
            a, b = perm[k], perm[k + 1]
            att[k] = True
            logR[k] = (etas[k] - etas[k + 1]) * (e[a] - e[b]) / rt
            logu[k] = np.log(rng.random())
            if logu[k] < logR[k]:
                acc[k] = True
                perm[k], perm[k + 1] = b, a
    return perm, att, logR, logu, acc


# ---------------------------------------------------------------- record validation

def validate_ladder(z):
    m = json.loads(str(z["meta"]))
    if m.get("storage_schema") != SCHEMA:
        raise ValueError("not an eta-exchange ladder record")
    etas, n = np.asarray(m["etas"], float), int(m["completed_sweeps"])
    K, na, rt = len(etas), len(m["ligand_system_indices"]), float(m["RT"])
    if n < 1 or n > int(m["sweeps"]) or bool(m["complete"]) != (n == int(m["sweeps"])):
        raise ValueError("invalid completion count / flag")
    m_up, eta0 = int(m.get("eta0_mm_updates", 1)), None
    if m_up > 1:                                       # eta=0 rows of the multi-update kernel (ETA0_SEMANTICS)
        if etas[0] != 0.0 or np.asarray(z["walker_eta_local"]).shape != (n, K):
            raise ValueError("eta=0 MM updates without an eta=0 label")
        eta0 = np.asarray(z["walker_eta_local"]) == 0.0
        sub_dH, sub_acc = np.asarray(z["mm_sub_dH"]), np.asarray(z["mm_sub_acc"])
        if sub_dH.shape != (n, K, m_up) or sub_acc.shape != (n, K, m_up) or sub_acc.dtype != bool:
            raise ValueError("invalid eta=0 substep records")
        if not np.isfinite(sub_dH[eta0]).all() or not np.isnan(sub_dH[~eta0]).all() or sub_acc[~eta0].any():
            raise ValueError("eta=0 substep records on the wrong walkers")
        if not np.array_equal(np.asarray(z["acc"], bool)[eta0], sub_acc[eta0].any(-1)):
            raise ValueError("eta=0 state change disagrees with its substeps")
    for k in PER_WALKER:
        v = np.asarray(z[k])
        if v.shape != (n, K):
            raise ValueError(f"invalid per-walker record {k}")
        if k == "dH" and eta0 is not None:
            if not np.isfinite(v[~eta0]).all() or not np.isnan(v[eta0]).all():
                raise ValueError("invalid per-walker record dH")
        elif not np.isfinite(v).all():
            raise ValueError(f"invalid per-walker record {k}")
    atom, ml = np.asarray(z["dU_atom"]), np.asarray(z["C_ML"])
    for p in ("", "prop_"):
        if np.asarray(z[p + "dU_atom"]).shape != (n, K, na) or np.asarray(z[p + "C_MM"]).shape != (n, K, na, 2):
            raise ValueError(f"invalid {p}atomic dimensions")
        if not np.allclose(z[p + "C_ML_learned"] + z[p + "C_ML_coul"], z[p + "C_ML"], rtol=1e-12, atol=1e-8):
            raise ValueError(f"{p}C_ML is not learned + explicit Coulomb")
        if not np.allclose(z[p + "dU_atom"], z[p + "C_ML"] - z[p + "C_MM"].sum(-1), rtol=1e-12, atol=1e-8):
            raise ValueError(f"{p}atomic difference disagrees with C_ML - C_MM")
    if not np.allclose(atom.sum(-1), z["dU"], rtol=1e-12, atol=1e-8):
        raise ValueError("atomic energies disagree with walker dU")
    acc = np.asarray(z["acc"], bool)
    for k in DETAILS:
        if not np.array_equal(np.asarray(z["prop_" + k])[acc], np.asarray(z[k])[acc]):
            raise ValueError(f"accepted local proposals differ from walker state: {k}")
    # exchanges: replay every decision from the cached walker dU
    perm = np.asarray(z["perm"])
    prev = np.asarray(m["initial_perm"])
    for s in range(n):
        if sorted(perm[s]) != list(range(K)):
            raise ValueError(f"sweep {s + 1}: labels are not a permutation of walkers")
        att = np.asarray(z["ex_attempt"][s])
        want = np.zeros(K - 1, bool)
        if m["exchange"]:
            want[(s % 2)::2] = True
        if not np.array_equal(att, want):
            raise ValueError(f"sweep {s + 1}: exchange schedule differs")
        p = prev.copy()
        for k in np.flatnonzero(att):
            a, b = p[k], p[k + 1]
            lr = (etas[k] - etas[k + 1]) * (z["dU"][s, a] - z["dU"][s, b]) / rt
            if not np.isclose(lr, z["ex_logR"][s, k], rtol=1e-12, atol=1e-12):
                raise ValueError(f"sweep {s + 1}: log R differs from cached dU")
            if bool(z["ex_acc"][s, k]) != bool(z["ex_logu"][s, k] < lr):
                raise ValueError(f"sweep {s + 1}: exchange decision inconsistent")
            if z["ex_acc"][s, k]:
                p[k], p[k + 1] = b, a
        if np.any(z["ex_acc"][s][~att]) or not np.array_equal(p, perm[s]):
            raise ValueError(f"sweep {s + 1}: permutation does not follow the exchange decisions")
        prev = perm[s]
    idx = np.arange(n)[:, None]
    if not np.array_equal(z["label_dU"], z["dU"][idx, perm]) or not np.array_equal(z["label_dU_atom"], atom[idx, perm]):
        raise ValueError("fixed-eta label view is not the walker state under the recorded permutation")
    walker_eta = np.empty((n, K))
    walker_eta[idx, perm] = etas[None, :]
    lab0 = np.empty(K)
    lab0[np.asarray(m["initial_perm"])] = etas
    if not np.array_equal(z["walker_eta_local"], np.vstack([lab0, walker_eta[:-1]])):
        raise ValueError("local moves did not use the label held at the start of the sweep")
    xs = np.asarray(z["xyz_sweeps"])
    if np.asarray(z["xyz"]).shape != (len(xs), K, int(m["n_particles"]), 3) or np.asarray(z["prop_xyz"]).shape != np.asarray(z["xyz"]).shape:
        raise ValueError("invalid full-box snapshot dimensions")
    lig = np.asarray(m["ligand_system_indices"])
    if len(xs) and not np.array_equal(np.asarray(z["xyz"])[:, :, lig], np.asarray(z["lig_xyz"])[xs - 1]):
        raise ValueError("full-box snapshots are not the recorded walker states")
    if np.asarray(z["last_positions"]).shape != (K, int(m["n_particles"]), 3):
        raise ValueError("invalid restart positions")
    for w in range(K):
        if not np.asarray(z[f"checkpoint_w{w}"]).size:
            raise ValueError("missing Context checkpoint")
    return m


def read_ladder(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return validate_ladder(data), data


def atomic_save(path, data):
    path = pathlib.Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    validate_ladder(data)
    np.savez(tmp, **data)
    tmp.replace(path)


# ---------------------------------------------------------------- recorded ladder

def same_contract_value(key, a, b):
    if key == "integrator_definition" and isinstance(a, str) and isinstance(b, str):
        return re.sub(r'randomSeed="-?\d+"', "", a) == re.sub(r'randomSeed="-?\d+"', "", b)
    return a == b


def run_ladder(ctxs, igs, energy, rngs_local, rng_x, x0s, etas, sweeps, out, contract, kernel, rt,
               exchange=True, checkpoint_every=50, coords_every=10, extend_from=None, device=None, eta0_mm_updates=1):
    """Run / continue a committed ladder. Model calls: one per walker at the start and one per local proposal;
    exchanges add none. A crash loses at most one uncommitted chunk, replayed from the last commit.
    extend_from: a completed ladder with the same contract and fewer sweeps; its final committed state is continued
    exactly (Context checkpoints, RNGs, permutation, caches) into the new file `out`; the source is left untouched.
    eta0_mm_updates > 1: `kernel` must be eta0_mm_kernel with the same m; the count and record semantics enter the
    contract (m = 1 leaves the contract and records exactly as before)."""
    K = len(etas)
    if not (len(ctxs) == len(igs) == len(rngs_local) == len(x0s) == K) or K < 2:
        raise ValueError("one Context / integrator / RNG / start per label")
    expected_rt = 0.0083144626 * float(contract["T_K"])
    if not np.isfinite(rt) or rt <= 0 or not np.isclose(rt, expected_rt, rtol=0, atol=1e-12):
        raise ValueError("exchange temperature differs from the recorded target")
    if any(not np.isclose(ig.getGlobalVariableByName("kT"), rt, rtol=0, atol=1e-12) for ig in igs):
        raise ValueError("integrator temperature differs from the exchange target")
    if sweeps < 1 or checkpoint_every < 1 or coords_every < 1 or eta0_mm_updates < 1:
        raise ValueError("sweep / checkpoint / coordinate / update counts must be positive")
    if eta0_mm_updates > 1 and float(etas[0]) != 0.0:
        raise ValueError("eta=0 MM updates need an eta=0 label")
    out = pathlib.Path(out)
    progress = out.with_name(out.stem + ".progress.npz")
    want = {**contract, "storage_schema": SCHEMA, "etas": list(map(float, etas)), "RT": float(rt),
            "exchange": bool(exchange), "sweeps": int(sweeps), "initial_perm": list(range(K)),
            "checkpoint_every": int(checkpoint_every), "coords_every": int(coords_every)}
    if eta0_mm_updates > 1:
        want["eta0_mm_updates"], want["eta0_semantics"] = int(eta0_mm_updates), ETA0_SEMANTICS
    lig = np.asarray(want["ligand_system_indices"], int)
    want["checkpoint_device"], want["continuations"] = device, []
    states, perm, done, arrays, prior_seconds = [None] * K, np.arange(K), 0, {}, 0.0
    saved = out if out.exists() else progress if progress.exists() else None
    extending = saved is None and extend_from is not None
    if extending:
        saved = pathlib.Path(extend_from)
    if saved is not None:
        old, data = read_ladder(saved)
        skip = PROVENANCE_ONLY | ({"sweeps"} if extending else set())
        diff = [k for k, v in want.items() if k not in skip and not same_contract_value(k, old.get(k), v)]
        if diff:
            raise ValueError("ladder contract differs: " + ", ".join(diff) + "; preserve files and use a new tag")
        if extending:
            if not old["complete"] or int(old["completed_sweeps"]) >= sweeps:
                raise ValueError("extend_from must be a completed ladder with fewer sweeps than requested")
            want["extended_from"] = {"file": saved.name, "sweeps": int(old["completed_sweeps"])}
        elif old["complete"]:
            if saved != out:
                atomic_save(out, data)
                progress.unlink(missing_ok=True)
            print(f"{out.name}: complete ladder reused", flush=True)
            return old
        elif "extended_from" in old:
            want["extended_from"] = old["extended_from"]
        by_position = old.get("checkpoint_device") != device
        want["continuations"] = list(old.get("continuations", []))
        if by_position:
            want["continuations"].append({"after_sweep": int(old["completed_sweeps"]), "mode": "positions",
                                          "checkpoint_device": old.get("checkpoint_device"), "device": device})
            print(f"{out.name}: checkpoint from {old.get('checkpoint_device')} not loaded on {device}; "
                  f"continuing from cached coordinates with new integrator seeds", flush=True)
        for w in range(K):
            if by_position:
                ctxs[w].setPositions(data["last_positions"][w])
            else:
                try:
                    ctxs[w].loadCheckpoint(data[f"checkpoint_w{w}"].tobytes())
                except Exception as err:
                    raise RuntimeError("Context checkpoint is incompatible; keep this run and use a new tag") from err
            rngs_local[w].bit_generator.state = json.loads(str(data[f"rng_local_w{w}"]))
            x = data["last_positions"][w]
            if not np.isclose(energy(x), data["dU"][-1, w], rtol=0, atol=RESTART_ATOL):
                raise ValueError(f"walker {w}: current model energy differs from the committed cache")
            det = {k: data[k][-1, w].copy() for k in DETAILS}
            if any(not np.allclose(energy.last_components[k], det[k], rtol=0, atol=RESTART_ATOL) for k in DETAILS):
                raise ValueError(f"walker {w}: current atomic target / Omega differs from the committed cache")
            states[w] = {"x": x, "e": float(data["dU"][-1, w]), "details": det}
        rng_x.bit_generator.state = json.loads(str(data["rng_exchange"]))
        perm, done, prior_seconds = data["perm"][-1].copy(), int(old["completed_sweeps"]), float(old["seconds"])
        arrays = {k: data[k] for k in data if k not in ("meta", "last_positions", "rng_exchange")
                  and not k.startswith(("checkpoint_w", "rng_local_w"))}
        print(f"{out.name}: resumed committed sweep {done}", flush=True)
    calls0 = getattr(energy, "calls", 0)
    started = time.perf_counter()
    while done < sweeps:
        count = min(checkpoint_every, sweeps - done)
        rows = {k: [] for k in ("perm", "ex_attempt", "ex_logR", "ex_logu", "ex_acc", "walker_eta_local")}
        wrows = []
        for s in range(done, done + count):
            inv = np.empty(K, int)
            inv[perm] = np.arange(K)
            eta_w = np.asarray(etas, float)[inv]
            res = []
            for w in range(K):
                o = kernel(ctxs[w], igs[w], x0s[w], float(eta_w[w]), 1, energy, rngs_local[w],
                           record=lambda x: np.asarray(x, np.float64).copy(),
                           capture=lambda: {k: np.asarray(v).copy() for k, v in energy.last_components.items()},
                           initial=states[w])
                states[w] = o["final_state"]
                res.append(o)
            e = np.array([st["e"] for st in states])
            perm, att, lr, lu, ac = exchange_step(perm, e, etas, s % 2, rng_x, rt, exchange)
            for k, v in zip(("perm", "ex_attempt", "ex_logR", "ex_logu", "ex_acc", "walker_eta_local"),
                            (perm, att, lr, lu, ac, eta_w)):
                rows[k].append(v)
            wrows.append(res)
        sw = np.arange(done + 1, done + count + 1)
        take = sw % coords_every == 0
        if done + count == sweeps:
            take[-1] = True
        new = {k: np.array(v) for k, v in rows.items()}
        for k in ("acc", "dH", "dE", "V", "prop_V"):
            new[k] = np.array([[o[k][0] for o in r] for r in wrows])
        new["dU"] = np.array([[o["e"][0] for o in r] for r in wrows])
        for k in DETAILS:
            new[k] = np.array([[o["details"][k][0] for o in r] for r in wrows])
            new["prop_" + k] = np.array([[o["prop_details"][k][0] for o in r] for r in wrows])
        if eta0_mm_updates > 1:
            m_up = eta0_mm_updates
            new["mm_sub_dH"] = np.array([[o.get("mm_sub_dH", np.full(m_up, np.nan)) for o in r] for r in wrows])
            new["mm_sub_acc"] = np.array([[o.get("mm_sub_acc", np.zeros(m_up, bool)) for o in r] for r in wrows])
        rec = np.array([[o["rec"][0] for o in r] for r in wrows])
        prec = np.array([[o["prop_rec"][0] for o in r] for r in wrows])
        new.update({"sweep": sw, "lig_xyz": rec[:, :, lig], "prop_lig_xyz": prec[:, :, lig],
                    "xyz": rec[take], "prop_xyz": prec[take], "xyz_sweeps": sw[take]})
        idx = np.arange(count)[:, None]
        new["label_dU"] = new["dU"][idx, new["perm"]]
        new["label_dU_atom"] = new["dU_atom"][idx, new["perm"]]
        arrays = {k: np.concatenate([arrays[k], v]) if k in arrays else v for k, v in new.items()}
        done += count
        meta = {**want, "completed_sweeps": done, "complete": done == sweeps,
                "seconds": prior_seconds + time.perf_counter() - started,
                "model_calls_this_process": getattr(energy, "calls", 0) - calls0,
                "model_calls_ladder": K * (done + 1),
                "restart_policy": "all Context checkpoints + local / exchange RNG; no coordinate-only fallback"}
        data = {**arrays, "meta": json.dumps(meta), "last_positions": np.array([st["x"] for st in states]),
                "rng_exchange": json.dumps(rng_x.bit_generator.state)}
        for w in range(K):
            data[f"checkpoint_w{w}"] = np.frombuffer(ctxs[w].createCheckpoint(), np.uint8)
            data[f"rng_local_w{w}"] = json.dumps(rngs_local[w].bit_generator.state)
        atomic_save(progress, data)
        print(f"{out.name}: committed {done}/{sweeps} sweeps; exchange acc so far "
              f"{arrays['ex_acc'].sum()}/{arrays['ex_attempt'].sum()}; local acc {arrays['acc'].mean():.2f}", flush=True)
    atomic_save(out, data)
    progress.unlink(missing_ok=True)
    return meta


# ---------------------------------------------------------------- molecular ladder

def load_initial_panel(path, frame_index):
    """Read one exact full-box panel state, retaining its values as the resume contract (no hash)."""
    path = pathlib.Path(path).resolve()
    with np.load(path, allow_pickle=False) as z:
        m = json.loads(str(z["meta"]))
        if m.get("schema") != "e3d-deep-shallow-panel-1":
            raise ValueError("unsupported initialization panel schema")
        if not 0 <= frame_index < len(m["frames"]):
            raise ValueError("initialization frame index out of range")
        x = np.asarray(z["xyz_nm"][frame_index], np.float64)
        if x.shape != (m["contract"]["n_particles"], 3) or not np.isfinite(x).all():
            raise ValueError("invalid panel initial positions")
        expected = float(z["expected_dU"][frame_index])
        details = {k: np.asarray(z[k][frame_index]).copy() for k in DETAILS}
        if not np.isfinite(expected) or any(not np.isfinite(v).all() for v in details.values()):
            raise ValueError("non-finite panel initial energy / components")
    info = {"file": str(path), "frame_index": int(frame_index), "source": m["frames"][frame_index],
            "positions_nm": x.tolist(), "expected_dU": expected,
            "expected_components": {k: v.tolist() for k, v in details.items()}}
    return x.copy(), m["contract"], info


def check_panel_contract(panel_contract, current):
    different = [k for k, v in panel_contract.items() if k not in current or not same_contract_value(k, v, current[k])]
    if different:
        raise ValueError("initialization panel target differs: " + ", ".join(different))


def initial_state(c, x0, energy, info=None):
    """eta_hmc.hmc's setPositions / applyConstraints / energy sequence; with a panel, recheck the selected state."""
    import eta_hmc as eh
    c.setPositions(x0)
    c.applyConstraints(eh.TOL)
    x = eh.positions(c)
    value = energy(x)
    details = {k: np.asarray(energy.last_components[k]).copy() for k in DETAILS}
    if info is not None:
        from deep_shallow_panel import compare_frame
        result = compare_frame(value, details, info["expected_dU"], info["expected_components"])
        if not result["cache_reproduced"]:
            raise ValueError("panel initial state fails current energy / atoms / Omega recheck: " + json.dumps(result))
    return {"x": x, "e": float(value), "details": details}


def panel_checked_kernel(info):
    """Use the ordinary initial query to recheck the selected panel state before any proposal.

    Initialization reproduces eta_hmc.hmc's setPositions / applyConstraints / energy sequence.
    It adds zero queries: the checked state is passed as initial to the unchanged HMC kernel.
    Resumed states are checked by run_ladder instead and never reset to the panel.
    """
    def kernel(c, ig, x0, eta, cycles, energy, rng, **kwargs):
        import eta_hmc as eh
        if kwargs.get("initial") is None:
            kwargs["initial"] = initial_state(c, x0, energy, info)
        return eh.hmc(c, ig, x0, eta, cycles, energy, rng, **kwargs)
    return kernel


ETA0_SEMANTICS = ("eta=0 label (held at the start of the sweep): m MM-HMC substeps (L, dt as above, fresh momenta, "
                  "Metropolis on each substep's own dH, rejection restores the substep's start), then one model query "
                  "at the final state. Record: acc = at least one substep accepted (state changed), dH = NaN (no single "
                  "Metropolis work), dE = final - initial cached dU, V / prop_V / prop_* = final state; substeps in "
                  "mm_sub_dH / mm_sub_acc. Other labels: one unchanged HMC proposal.")


def eta0_mm_kernel(base, m, info=None):
    """The eta = 0 label does m MM-only substeps (eta_hmc.mm_updates) and one final model query
    (ETA0_MM_UPDATES_DECISION_2026-11-01.md); other labels, and m = 1, run `base` unchanged."""
    def kernel(c, ig, x0, eta, cycles, energy, rng, record=None, capture=None, initial=None, **kw):
        import eta_hmc as eh
        if m == 1 or eta != 0.0:
            return base(c, ig, x0, eta, cycles, energy, rng, record=record, capture=capture, initial=initial, **kw)
        if cycles != 1 or record is None or capture is None:
            raise ValueError("the eta=0 multi-update kernel runs one recorded sweep at a time")
        if initial is None:
            initial = initial_state(c, x0, energy, info)
        elif np.abs(eh.positions(c) - initial["x"]).max() > eh.POS_TOL:
            raise ValueError("checkpoint positions differ from the saved accepted state")
        e0 = float(initial["e"])
        x, V, dH, acc = eh.mm_updates(c, ig, np.array(initial["x"], np.float64), m, rng)
        e = energy(x)
        details = capture()
        if not np.isfinite(e):
            raise FloatingPointError("non-finite model energy at the final eta=0 state")
        one = {k: np.asarray(v)[None] for k, v in details.items()}
        return {"acc": np.array([acc.any()]), "dH": np.array([np.nan]), "dE": np.array([e - e0]),
                "V": np.array([V]), "e": np.array([e]), "prop_V": np.array([V]),
                "rec": np.array([record(x)]), "prop_rec": np.array([record(x)]),
                "details": one, "prop_details": {k: v.copy() for k, v in one.items()},
                "final_state": {"x": np.array(x, copy=True), "e": float(e), "details": details},
                "mm_sub_dH": dH, "mm_sub_acc": acc}
    return kernel


def start_frames(name, rep, j, K, init, spacing):
    import local_E
    z = np.load(local_E.GW / "runs" / name / f"rep{rep}_S{j}.npz")
    if init == "last":
        return [z["last_positions"].astype(np.float64)] * K, ["last_positions"] * K
    n = len(z["xyz"])
    idx = [n - 1 - spacing * w for w in range(K)]
    if min(idx) < 0:
        raise ValueError("not enough saved frames for the requested spacing")
    return [z["xyz"][i].astype(np.float64) for i in idx], [f"xyz[{i}]" for i in idx]


def ladder(a):
    import openmm
    import eta_hmc as eh
    import local_E
    etas = [float(x) for x in a.etas.split(",")]
    if etas != sorted(etas) or len(set(etas)) != len(etas) or etas[0] < 0 or etas[-1] > 1:
        raise ValueError("etas must increase within [0, 1]")
    if a.dtype != "float64":
        raise ValueError("model MC acceptance requires the validated float64 energy query")
    if (a.init == "panel") != (a.init_panel is not None and a.init_frame is not None):
        raise ValueError("--init panel requires --init-panel and --init-frame; panel flags require --init panel")
    if a.init != "panel" and (a.init_panel is not None or a.init_frame is not None):
        raise ValueError("panel flags require --init panel")
    K = len(etas)
    leg = local_E.Leg(a.system)
    me = eh.ModelEnergy(leg, a.state, a.dtype, ckpt=a.head, backbone=a.backbone)
    import torch
    device = torch.cuda.get_device_name(torch.cuda.current_device())
    base = 100000 + 1000 * a.start if a.seed_base is None else a.seed_base
    if not 1 <= base < 2**31 - 1000000:
        raise ValueError("seed base must be positive and leave room for continuation streams")
    out = eh.HERE / "runs" / a.system / f"ladder_S{a.state}_start{a.start}{a.tag}.npz"
    src = None
    if a.extend_from is not None:
        src = eh.HERE / "runs" / a.system / f"ladder_S{a.state}_start{a.start}{a.extend_from}.npz"
        if src == out:
            raise ValueError("--extend-from needs a different tag; the source ladder is left untouched")
    # integrator seeds: a new stream for every coordinate continuation (seed must be set before Context creation)
    progress = out.with_name(out.stem + ".progress.npz")
    prior = next((q for q in (out, progress, src) if q is not None and q.exists()), None)
    k = 0
    if prior is not None:
        with np.load(prior, allow_pickle=False) as zz:
            pm = json.loads(str(zz["meta"]))
        k = len(pm.get("continuations", [])) + int(not (prior == out and pm["complete"])
                                                     and pm.get("checkpoint_device") != device)
    ctxs, igs, rngs = [], [], []
    for w in range(K):
        system, groups, box = eh.mm_system(a.system)
        ig = eh.proposal_integrator(a.L, a.dt_fs, leg.temperature_K)
        ig.setRandomNumberSeed(base + 10 * w + 1 + SEED_STRIDE * k)
        c = eh.make_context(system, box, ig)
        eh.gm.set_state(c, a.state, len(groups))
        ctxs.append(c)
        igs.append(ig)
        rngs.append(np.random.default_rng(base + 10 * w))
    rng_x = np.random.default_rng(base + 9)
    panel = None
    if a.init == "panel":
        x, panel_contract, panel = load_initial_panel(a.init_panel, a.init_frame)
        x0s, frames = [x.copy() for _ in range(K)], [f"panel[{a.init_frame}]"] * K
    else:
        x0s, frames = start_frames(a.system, a.start, a.state, K, a.init, a.spacing)
    out.parent.mkdir(parents=True, exist_ok=True)
    ident = eh.model_identity(ckpt=a.head, backbone=a.backbone)
    want = {"system": a.system, "state": a.state, "start_rep": a.start, "start_frames": frames, "L": a.L, "dt_fs": a.dt_fs,
            "dtype": a.dtype, "T_K": leg.temperature_K, "E_construction": "local_E v1", **ident,
            "leg": "solvent", "n_water": 50, "n_particles": system.getNumParticles(),
            "ligand_system_indices": list(map(int, leg.lig)), "groups": [list(map(int, g)) for g in leg.groups],
            "dummy_mask": leg.dummy(a.state).tolist(), "box_nm": np.asarray(box, float).tolist(),
            "system_xml": str(eh.gm.leg(a.system)[0]), "softcore_alpha": eh.gm.ALPHA,
            "system_definition": openmm.XmlSerializer.serialize(system),
            "context_parameters": dict(ctxs[0].getParameters()),
            "integrator_definition": openmm.XmlSerializer.serialize(igs[0]),
            "openmm_version": openmm.__version__, "seed_base": base,
            "energy_units": "kJ/mol", "xyz_units": "nm"}
    if panel is not None:
        check_panel_contract(panel_contract, want)
        want["panel_initialization"] = panel
    kernel = panel_checked_kernel(panel) if panel is not None else eh.hmc
    if a.eta0_mm_updates > 1:
        kernel = eta0_mm_kernel(kernel, a.eta0_mm_updates, panel)
    meta = run_ladder(ctxs, igs, me, rngs, rng_x, x0s, etas, a.sweeps, out, want, kernel, leg.rt,
                      exchange=not a.no_exchange, checkpoint_every=a.checkpoint_every, coords_every=a.coords_every,
                      extend_from=src, device=device, eta0_mm_updates=a.eta0_mm_updates)
    print(f"{out.name}: complete={meta['complete']}; {meta['completed_sweeps']} sweeps; "
          f"ladder model calls {meta['model_calls_ladder']}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ladder"])
    ap.add_argument("system")
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--etas", default="0,0.25,0.5,1")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--init", choices=["spread", "last", "panel"], default="spread",
                    help="spread: saved MM production frames n-1, n-1-spacing, ... of rep START; last: its last frame")
    ap.add_argument("--init-panel", default=None, help="exact full-box saved-state panel for --init panel")
    ap.add_argument("--init-frame", type=int, default=None, help="zero-based index in the initialization panel")
    ap.add_argument("--seed-base", type=int, default=None, help="explicit random-stream base; default remains 100000+1000*start")
    ap.add_argument("--spacing", type=int, default=100)
    ap.add_argument("--sweeps", type=int, default=500)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--dt-fs", type=float, default=2.0)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--tag", default="")
    ap.add_argument("--head", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--no-exchange", action="store_true", help="matched control: labels fixed")
    ap.add_argument("--extend-from", default=None,
                    help="tag of a completed ladder (same contract, fewer sweeps) continued exactly into --tag")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--coords-every", type=int, default=10)
    ap.add_argument("--eta0-mm-updates", type=int, default=1,
                    help="MM-HMC substeps per sweep for the eta=0 label, one model query at their end; 1 = original kernel")
    a = ap.parse_args()
    {"ladder": ladder}[a.cmd](a)

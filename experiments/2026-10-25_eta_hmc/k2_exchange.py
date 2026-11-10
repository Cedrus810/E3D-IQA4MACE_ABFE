"""eta-label replica exchange on the K2 target (port of eta_exchange.py; K2_DECISION_2026-11-07.md, DESIGN §10).

Same algorithm and transaction / restart policy as eta_exchange.run_ladder:
  every sweep each walker makes one MM-HMC proposal at its current label (MM forces only), Metropolis on
  V + eta * dU_K2; then non-overlapping adjacent label swaps from cached dU_K2 only:
      log R = (eta_k - eta_{k+1}) [dU_K2(x_a) - dU_K2(x_b)] / RT.
  Labels move; coordinates, Context, dU_K2, d_i, q, C_bb and Omega stay with the walker.
  Context checkpoints are loaded only on the GPU model that wrote them; otherwise the walker continues from its exact
  cached coordinates with new integrator seeds, recorded as a continuation (eta_exchange.py:31-35).
K2 differences: storage schema k2_exchange-1; per-walker K2 components (k2_records.DETAILS) with the K2 accounting
checked for every walker and sweep; fixed-eta label views label_dU / label_d_atom / label_q for k2_report.
Legacy ladders and their tools are untouched; the two schemas refuse each other.

Run: python k2_exchange.py SYSTEM --etas 0,0.1,...,1 --start-file NPZ --start-label L --head H --backbone B
       [--init last|spread --spacing N] --sweeps N --tag T [--no-exchange] [--eta0-mm-updates m]
Output: runs/SYSTEM/k2ladder_S{state}_{label}{tag}.npz (rerun the same command to resume).
"""

import argparse
import json
import pathlib
import time
import zlib

import numpy as np

from eta_exchange import (ETA0_SEMANTICS, RESTART_ATOL, SEED_STRIDE, exchange_step, same_contract_value)
import k2_records as kr

SCHEMA = "k2_exchange-1"
DETAILS = kr.DETAILS
PER_WALKER = ("acc", "dH", "dE", "V", "dU", "prop_V")
PROVENANCE_ONLY = {"head_mtime", "backbone_mtime", "checkpoint_device", "continuations"}


# ---------------------------------------------------------------- record validation

def _flat(z, p, n, K):
    return {p + k: np.asarray(z[p + k]).reshape((n * K,) + np.asarray(z[p + k]).shape[2:])
            for k in ("C_H", "C_H_learned", "C_H_coul", "C_MM", "d_atom", "C_bb", "q")}


def validate_ladder(z):
    m = json.loads(str(z["meta"]))
    if m.get("storage_schema") != SCHEMA or m.get("E_construction") != kr.TARGET:
        raise ValueError(f"not a {SCHEMA} / {kr.TARGET} ladder record")
    etas, n = np.asarray(m["etas"], float), int(m["completed_sweeps"])
    K, na, rt = len(etas), len(m["ligand_system_indices"]), float(m["RT"])
    if n < 1 or n > int(m["sweeps"]) or bool(m["complete"]) != (n == int(m["sweeps"])):
        raise ValueError("invalid completion count / flag")
    m_up, eta0 = int(m.get("eta0_mm_updates", 1)), None
    if m_up > 1:
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
    closure, target, ch, mm = kr._accounting(_flat(z, "", n, K), "", n * K, na)
    kr._accounting(_flat(z, "prop_", n, K), "prop_", n * K, na)
    du = np.asarray(z["dU"]).reshape(-1)
    if not np.allclose(closure, du, rtol=1e-12, atol=kr.ATOL):
        raise ValueError("sum d_atom + q disagrees with walker dU")
    if not np.allclose(target, du, rtol=1e-12, atol=kr.ATOL):
        raise ValueError("C_bb - C_MM disagrees with walker dU")
    dummy = np.asarray(m["dummy_mask"], bool)
    if dummy.shape != (na,) or np.any(ch[:, dummy] != 0) or np.any(mm[:, dummy] != 0):
        raise ValueError("dummy rows must be zero")
    acc = np.asarray(z["acc"], bool)
    for k in DETAILS:
        if not np.array_equal(np.asarray(z["prop_" + k])[acc], np.asarray(z[k])[acc]):
            raise ValueError(f"accepted local proposals differ from walker state: {k}")
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
    for lab, src in (("label_dU", "dU"), ("label_d_atom", "d_atom"), ("label_q", "q")):
        if not np.array_equal(z[lab], np.asarray(z[src])[idx, perm]):
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


# ---------------------------------------------------------------- kernels with K2 details

def initial_state(c, x0, energy):
    import eta_hmc as eh
    c.setPositions(x0)
    c.applyConstraints(eh.TOL)
    x = eh.positions(c)
    value = energy(x)
    return {"x": x, "e": float(value), "details": {k: np.asarray(energy.last_components[k]).copy() for k in DETAILS}}


def eta0_mm_kernel(base, m):
    """eta_exchange.eta0_mm_kernel with K2 details (same semantics, ETA0_SEMANTICS)."""
    def kernel(c, ig, x0, eta, cycles, energy, rng, record=None, capture=None, initial=None, **kw):
        import eta_hmc as eh
        if m == 1 or eta != 0.0:
            return base(c, ig, x0, eta, cycles, energy, rng, record=record, capture=capture, initial=initial, **kw)
        if cycles != 1 or record is None or capture is None:
            raise ValueError("the eta=0 multi-update kernel runs one recorded sweep at a time")
        if initial is None:
            initial = initial_state(c, x0, energy)
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


# ---------------------------------------------------------------- recorded ladder

def run_ladder(ctxs, igs, energy, rngs_local, rng_x, x0s, etas, sweeps, out, contract, kernel, rt,
               exchange=True, checkpoint_every=50, coords_every=10, extend_from=None, device=None, eta0_mm_updates=1):
    """eta_exchange.run_ladder with the K2 schema (see module docstring)."""
    if contract.get("E_construction") != kr.TARGET:
        raise ValueError(f"K2 ladders require E_construction={kr.TARGET}")
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
                raise ValueError(f"walker {w}: current K2 energy differs from the committed cache")
            det = {k: data[k][-1, w].copy() for k in DETAILS}
            if any(not np.allclose(energy.last_components[k], det[k], rtol=0, atol=RESTART_ATOL) for k in DETAILS):
                raise ValueError(f"walker {w}: current K2 components / Omega differ from the committed cache")
            states[w] = {"x": x, "e": float(data["dU"][-1, w]), "details": det}
        rng_x.bit_generator.state = json.loads(str(data["rng_exchange"]))
        perm, done, prior_seconds = data["perm"][-1].copy(), int(old["completed_sweeps"]), float(old["seconds"])
        arrays = {k: data[k] for k in data if k not in ("meta", "last_positions", "rng_exchange")
                  and not k.startswith(("checkpoint_w", "rng_local_w"))}
        print(f"{out.name}: resumed committed sweep {done}", flush=True)
    calls0 = getattr(energy, "calls", 0)
    started = time.perf_counter()
    meta = data = None
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
                           capture=lambda: {k: np.asarray(energy.last_components[k]).copy() for k in DETAILS},
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
        new["label_d_atom"] = new["d_atom"][idx, new["perm"]]
        new["label_q"] = new["q"][idx, new["perm"]]
        arrays = {k: np.concatenate([arrays[k], v]) if k in arrays else v for k, v in new.items()}
        done += count
        meta = {**want, "completed_sweeps": done, "complete": done == sweeps,
                "seconds": prior_seconds + time.perf_counter() - started,
                "model_calls_this_process": getattr(energy, "calls", 0) - calls0,
                "model_calls_ladder": K * (done + 1),
                "restart_policy": "Context checkpoints on the same GPU model, else cached coordinates (continuation); "
                                  "local / exchange RNG restored"}
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

def start_frames(path, K, init, spacing):
    with np.load(path, allow_pickle=False) as z:
        if init == "last":
            x = z["xyz"][-1] if "xyz" in z.files and len(z["xyz"]) else z["last_positions"]
            return [np.asarray(x, np.float64)] * K, ["xyz[-1]"] * K
        n = len(z["xyz"])
        idx = [n - 1 - spacing * w for w in range(K)]
        if min(idx) < 0:
            raise ValueError("not enough saved frames for the requested spacing")
        return [np.asarray(z["xyz"][i], np.float64) for i in idx], [f"xyz[{i}]" for i in idx]


def ladder(a):
    import openmm
    import torch
    import eta_hmc as eh
    import k2_energy as k2
    etas = [float(x) for x in a.etas.split(",")]
    if etas != sorted(etas) or len(set(etas)) != len(etas) or etas[0] < 0 or etas[-1] > 1:
        raise ValueError("etas must increase within [0, 1]")
    K = len(etas)
    leg = eh.local_E.Leg(a.system)
    me = k2.ModelEnergyK2(leg, a.state, head=a.head, backbone=a.backbone, device=a.device)
    device = torch.cuda.get_device_name(torch.cuda.current_device()) if a.platform == "CUDA" else a.platform
    base = 100000 + 1000 * (zlib.crc32(a.start_label.encode()) % 100000) if a.seed_base is None else a.seed_base
    base = base % (2**31 - 2000000) + 1
    out = eh.HERE / "runs" / a.system / f"k2ladder_S{a.state}_{a.start_label}{a.tag}.npz"
    src = None
    if a.extend_from is not None:
        src = eh.HERE / "runs" / a.system / f"k2ladder_S{a.state}_{a.start_label}{a.extend_from}.npz"
        if src == out:
            raise ValueError("--extend-from needs a different tag; the source ladder is left untouched")
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
        c = eh.make_context(system, box, ig, a.platform)
        eh.gm.set_state(c, a.state, len(groups))
        ctxs.append(c)
        igs.append(ig)
        rngs.append(np.random.default_rng(base + 10 * w))
    rng_x = np.random.default_rng(base + 9)
    x0s, frames = start_frames(a.start_file, K, a.init, a.spacing)
    out.parent.mkdir(parents=True, exist_ok=True)
    want = {"system": a.system, "state": a.state, "start_label": a.start_label, "start_file": str(a.start_file),
            "start_frames": frames, "L": a.L, "dt_fs": a.dt_fs, "T_K": leg.temperature_K, **me.identity,
            "leg": "solvent", "n_particles": system.getNumParticles(),
            "ligand_system_indices": list(map(int, leg.lig)), "groups": [list(map(int, g)) for g in leg.groups],
            "dummy_mask": leg.dummy(a.state).tolist(), "box_nm": np.asarray(box, float).tolist(),
            "system_xml": str(eh.gm.leg(a.system)[0]), "softcore_alpha": eh.gm.ALPHA,
            "system_definition": openmm.XmlSerializer.serialize(system),
            "context_parameters": dict(ctxs[0].getParameters()),
            "integrator_definition": openmm.XmlSerializer.serialize(igs[0]),
            "openmm_version": openmm.__version__, "seed_base": base, "platform": a.platform, "model_device": a.device,
            "energy_units": "kJ/mol", "xyz_units": "nm"}
    kernel = eh.hmc if a.eta0_mm_updates == 1 else eta0_mm_kernel(eh.hmc, a.eta0_mm_updates)
    meta = run_ladder(ctxs, igs, me, rngs, rng_x, x0s, etas, a.sweeps, out, want, kernel, leg.rt,
                      exchange=not a.no_exchange, checkpoint_every=a.checkpoint_every, coords_every=a.coords_every,
                      extend_from=src, device=device, eta0_mm_updates=a.eta0_mm_updates)
    print(f"{out.name}: complete={meta['complete']}; {meta['completed_sweeps']} sweeps; ladder model calls "
          f"{meta['model_calls_ladder']}; {me.seconds / max(me.calls, 1):.2f} s per K2 query", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--etas", required=True)
    ap.add_argument("--start-file", required=True)
    ap.add_argument("--start-label", required=True)
    ap.add_argument("--init", choices=["last", "spread"], default="last")
    ap.add_argument("--spacing", type=int, default=50, help="spread: frames n-1, n-1-spacing, ... of the start file")
    ap.add_argument("--head", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--sweeps", type=int, default=500)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--dt-fs", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-exchange", action="store_true")
    ap.add_argument("--extend-from", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--coords-every", type=int, default=10)
    ap.add_argument("--eta0-mm-updates", type=int, default=1)
    ladder(ap.parse_args())

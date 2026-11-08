"""K2 eta-chain records and checkpoint transactions (K2_DECISION_2026-11-07.md §3-§5).

Same transaction / restart policy as eta_records.run_recorded (exact compatible Context + acceptance RNG,
committed chunks, atomic replace, every cycle incl. rejected repeats), but a separate storage schema whose
validator checks the K2 accounting every cycle:
    dU = C_bb - sum C_MM,   d_atom = C_H - sum_k C_MM,   C_H = learned + explicit Coulomb,   sum d_atom + q = dU.
Kept separate from eta_records so legacy local_E v1 chains and their tools are untouched; legacy records
(schema 3) are refused here and K2 records are refused by eta_records (different storage_schema / keys).
"""

import json
import time
from pathlib import Path

import numpy as np

from eta_records import check_contract

SCHEMA = "k2_1"
TARGET = "local_E_bb_v1"
DETAILS = ("C_H", "C_H_learned", "C_H_coul", "C_MM", "d_atom", "C_bb", "q",
           "cluster_system_indices", "omega_water_resid")
PROP = tuple("prop_" + k for k in DETAILS) + ("prop_V", "prop_lig_xyz", "prop_xyz")
ATOL = 1e-8                 # kJ/mol; same as eta_records' stored-record identities (pure float64 bookkeeping)


def _accounting(z, p, n, na):
    ch, lr, co = (np.asarray(z[p + k]) for k in ("C_H", "C_H_learned", "C_H_coul"))
    mm, d = np.asarray(z[p + "C_MM"]), np.asarray(z[p + "d_atom"])
    cbb, q = np.asarray(z[p + "C_bb"]), np.asarray(z[p + "q"])
    if ch.shape != (n, na) or lr.shape != ch.shape or co.shape != ch.shape or d.shape != ch.shape:
        raise ValueError(f"{p}invalid atomic dimensions")
    if mm.shape != (n, na, 2) or cbb.shape != (n,) or q.shape != (n,):
        raise ValueError(f"{p}invalid C_MM / C_bb / q dimensions")
    if not all(np.isfinite(a).all() for a in (ch, lr, co, mm, d, cbb, q)):
        raise ValueError(f"{p}non-finite K2 component")
    if not np.allclose(lr + co, ch, rtol=1e-12, atol=ATOL):
        raise ValueError(f"{p}C_H is not learned + explicit Coulomb")
    if not np.allclose(d, ch - mm.sum(axis=2), rtol=1e-12, atol=ATOL):
        raise ValueError(f"{p}d_atom disagrees with C_H - C_MM")
    if not np.allclose(q, cbb - ch.sum(axis=1), rtol=1e-12, atol=ATOL):
        raise ValueError(f"{p}q disagrees with C_bb - sum C_H")
    return d.sum(axis=1) + q, cbb - mm.sum(axis=(1, 2)), ch, mm


def validate_record(z):
    m = json.loads(str(z["meta"]))
    if m.get("storage_schema") != SCHEMA or m.get("E_construction") != TARGET:
        raise ValueError(f"not a {SCHEMA} / {TARGET} chain; keep it and use its own tools")
    n = int(m["completed_cycles"])
    if n < 1 or n > int(m["cycles"]) or bool(m["complete"]) != (n == int(m["cycles"])):
        raise ValueError("invalid completion count / flag")
    indices = np.asarray(m["ligand_system_indices"], int)
    na = len(indices)
    for key in ("acc", "dH", "dE", "V", "dU"):
        if np.asarray(z[key]).shape != (n,) or not np.isfinite(z[key]).all():
            raise ValueError(f"invalid scalar record: {key}")
    closure, target, ch, mm = _accounting(z, "", n, na)
    if not np.allclose(closure, z["dU"], rtol=1e-12, atol=ATOL):
        raise ValueError("sum d_atom + q disagrees with the accepted-state scalar dU")
    if not np.allclose(target, z["dU"], rtol=1e-12, atol=ATOL):
        raise ValueError("C_bb - C_MM disagrees with the accepted-state scalar dU")
    _accounting(z, "prop_", n, na)
    acc = np.asarray(z["acc"], bool)
    for k in DETAILS:
        if np.asarray(z["prop_" + k]).shape != np.asarray(z[k]).shape:
            raise ValueError(f"invalid proposal record: {k}")
        if not np.array_equal(np.asarray(z["prop_" + k])[acc], np.asarray(z[k])[acc]):
            raise ValueError(f"accepted proposals differ from the recorded chain state: {k}")
    if np.asarray(z["prop_V"]).shape != (n,) or np.asarray(z["prop_lig_xyz"]).shape != (n, na, 3):
        raise ValueError("invalid proposal V / ligand coordinates")
    if np.asarray(z["prop_xyz"]).shape != np.asarray(z["xyz"]).shape:
        raise ValueError("proposal full-box snapshots must use the chain's coordinate cycles")
    dummy = np.asarray(m["dummy_mask"], bool)
    if dummy.shape != (na,) or np.any(ch[:, dummy] != 0) or np.any(mm[:, dummy] != 0):
        raise ValueError("dummy rows must be zero")
    if not np.array_equal(z["cycle"], np.arange(1, n + 1)):
        raise ValueError("chain record is missing or repeats cycle indices")
    xyz, xc, lig = np.asarray(z["xyz"]), np.asarray(z["xyz_cycles"]), np.asarray(z["lig_xyz"])
    nparticles = int(m["n_particles"])
    if xyz.shape != (len(xc), nparticles, 3) or lig.shape != (n, na, 3):
        raise ValueError("invalid full-box / ligand coordinate dimensions")
    if not np.isfinite(xyz).all() or not np.isfinite(lig).all():
        raise ValueError("non-finite saved coordinates")
    if not np.issubdtype(xc.dtype, np.integer) or (len(xc) and (xc[0] < 1 or xc[-1] > n or np.any(np.diff(xc) <= 0))):
        raise ValueError("invalid coordinate sample cycle indices")
    if len(xc) and not np.array_equal(xyz[:, indices], lig[xc - 1]):
        raise ValueError("full-box snapshots are not the actual recorded chain states")
    last = np.asarray(z["last_positions"])
    if last.shape != (nparticles, 3) or not np.isfinite(last).all() or not np.array_equal(last[indices], lig[-1]):
        raise ValueError("invalid checkpoint coordinates")
    box = np.asarray(z["box"])
    if box.shape != (3, 3) or not np.isfinite(box).all() or not np.array_equal(box, m["box_nm"]):
        raise ValueError("box differs from the run contract")
    nw = int(m["n_water"])
    si, omega = np.asarray(z["cluster_system_indices"]), np.asarray(z["omega_water_resid"])
    if si.shape != (n, na + 3 * nw) or omega.shape != (n, nw):
        raise ValueError("invalid cluster atom / water mapping")
    if not np.issubdtype(si.dtype, np.integer) or not np.issubdtype(omega.dtype, np.integer):
        raise ValueError("cluster mappings must be integers")
    if np.any(si < 0) or np.any(si >= nparticles) or not np.all(si[:, :na] == indices):
        raise ValueError("cluster system indices are inconsistent")
    if any(len(np.unique(s)) != len(s) for s in si) or any(len(np.unique(w)) != nw for w in omega):
        raise ValueError("duplicate cluster atoms / water identities")
    if not np.asarray(z["checkpoint"]).size or np.asarray(z["checkpoint"]).dtype != np.uint8:
        raise ValueError("missing Context checkpoint")
    json.loads(str(z["rng_state"]))
    return m


def read_record(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return validate_record(data), data


def atomic_save(path, data):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    validate_record(data)
    np.savez(tmp, **data)
    tmp.replace(path)


def run_recorded(c, ig, energy, rng, x0, eta, cycles, out, contract, kernel,
                 checkpoint_every=50, coords_every=10, log_every=0):
    """eta_records.run_recorded with the K2 schema. The restart recheck re-queries the last state once and
    compares the scalar and every K2 component (incl. C_bb, q, Omega) with the committed record."""
    if contract.get("E_construction") != TARGET:
        raise ValueError(f"K2 records require E_construction={TARGET}")
    if cycles < 1 or checkpoint_every < 1 or coords_every < 1:
        raise ValueError("cycle / checkpoint / coordinate counts must be positive")
    rt = 0.0083144626 * float(contract["T_K"])
    if not np.isfinite(rt) or rt <= 0 or not np.isclose(ig.getGlobalVariableByName("kT"), rt, rtol=0, atol=1e-12):
        raise ValueError("integrator temperature differs from the recorded target")
    out = Path(out)
    progress = out.with_name(out.stem + ".progress.npz")
    want = {**contract, "storage_schema": SCHEMA, "cycles": int(cycles),
            "checkpoint_every": int(checkpoint_every), "coords_every": int(coords_every)}
    initial, arrays, done, prior_seconds = None, {}, 0, 0.0
    saved_path = out if out.exists() else progress if progress.exists() else None
    if saved_path is not None:
        old, data = read_record(saved_path)
        check_contract(old, want)
        if old["complete"]:
            if saved_path != out:
                atomic_save(out, data)
                progress.unlink(missing_ok=True)
            print(f"{out.name}: complete {SCHEMA} chain reused", flush=True)
            return old
        try:
            c.loadCheckpoint(data["checkpoint"].tobytes())
        except Exception as err:
            raise RuntimeError("Context checkpoint is incompatible; keep this run and start a new continuation tag") from err
        rng.bit_generator.state = json.loads(str(data["rng_state"]))
        xlast = data["last_positions"]
        checked = energy(xlast)
        if not np.isclose(checked, data["dU"][-1], rtol=1e-12, atol=1e-7):
            raise ValueError("current K2 energy differs from checkpoint target")
        details = {k: data[k][-1].copy() for k in DETAILS}
        if any(not np.allclose(energy.last_components[k], details[k], rtol=1e-12, atol=1e-7) for k in DETAILS):
            raise ValueError("current K2 components / Omega differ from checkpoint")
        initial = {"x": xlast, "e": float(data["dU"][-1]), "details": details}
        done, prior_seconds = int(old["completed_cycles"]), float(old["seconds"])
        arrays = {k: data[k] for k in ("acc", "dH", "dE", "V", "dU", "cycle", "lig_xyz", "xyz", "xyz_cycles",
                                       *DETAILS, *PROP)}
        print(f"{out.name}: resumed committed cycle {done} from Context and NumPy RNG", flush=True)
    indices = np.asarray(want["ligand_system_indices"], int)
    started = time.perf_counter()
    meta = data = None
    while done < cycles:
        count = min(checkpoint_every, cycles - done)
        o = kernel(c, ig, x0, eta, count, energy, rng,
                   record=lambda x: np.asarray(x, np.float64).copy(), log_every=log_every,
                   capture=lambda: {k: np.asarray(energy.last_components[k]).copy() for k in DETAILS},
                   initial=initial)
        now = done + count
        cycle = np.arange(done + 1, now + 1)
        take = cycle % coords_every == 0
        if now == cycles:
            take[-1] = True
        new = {k: o[k] for k in ("acc", "dH", "dE", "V")}
        new.update({"dU": o["e"], "cycle": cycle, "lig_xyz": o["rec"][:, indices],
                    "xyz": o["rec"][take], "xyz_cycles": cycle[take], **o["details"],
                    "prop_V": o["prop_V"], "prop_lig_xyz": o["prop_rec"][:, indices], "prop_xyz": o["prop_rec"][take],
                    **{"prop_" + k: v for k, v in o["prop_details"].items()}})
        arrays = {k: np.concatenate([arrays[k], v]) if k in arrays else v for k, v in new.items()}
        initial, done = o["final_state"], now
        meta = {**want, "completed_cycles": done, "complete": done == cycles,
                "seconds": prior_seconds + time.perf_counter() - started,
                "model_calls_this_process": getattr(energy, "calls", 0),
                "restart_policy": "exact compatible Context + acceptance RNG; no coordinate-only fallback"}
        data = {**arrays, "meta": json.dumps(meta), "last_positions": initial["x"],
                "box": np.asarray(want["box_nm"], np.float64),
                "checkpoint": np.frombuffer(c.createCheckpoint(), np.uint8),
                "rng_state": json.dumps(rng.bit_generator.state)}
        atomic_save(progress, data)
        print(f"{out.name}: committed {done}/{cycles} cycles", flush=True)
    atomic_save(out, data)
    progress.unlink(missing_ok=True)
    return meta

"""Compare old, v2 and both existing nvar heads on held-out S0 frames (PLAN.md §4 items 1, 2, 4).

Configurations: make_data.py gen SYSTEM --reps 1-4 (rep1-4 never in training): per S0 frame one MM trajectory of
20 x 2 fs, ligand rigidly fitted back, environment and Omega (50 waters) of x0; labels C_bb, X (float64).
Per configuration and head (float64, S0 = nothing dummy, local_E v1 C^ML):
    C_head = sum_i (sum_c D_ic + cross_i)      (eta_hmc.couplings_dt)
    learned sum = C_head - X  vs  T = C_bb - X  (the two differ from C_head - C_bb only by cross rounding)
    C_MM   = bare Coulomb + cutoff LJ over the same pairs (local_E v1)
Reported, no thresholds:
    1  response: changes t>0 minus t=0 within a trajectory, learned sum vs T: corr, slope, sd of the difference
    2  level: mean, sd of (learned sum - T)
    4  dU = C_head - C_MM: sd over frames at t=0, and sd of its internal-deformation change at 40 fs
       (environment motion and Omega swaps are not in these configurations)
Run: python validate.py SYSTEM [--reps 1-4] [--heads old v2 nvar_a nvar_b] [--tag _allheads]
The JSON and per-configuration NPZ are separate from historical validation outputs; existing outputs are protected.
This panel fixes the environment and Omega within each trajectory: it does not validate full moving-environment
HMC proposals, partial-dummy endpoints, equilibrium mixing, or physical accuracy of the backbone labels.
"""

import argparse
import json
import os

import numpy as np
import torch

from make_data import DATA, HERE, eh, reps_of

HEADS = {"old": "/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_h256_s60000_N.pt",
         "v2": "/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s60000_N.pt",
         "nvar_a": "/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s60000_nvar_N.pt",
         "nvar_b": "/home/ruigengji/E3D/runs/polar-M/ckpt/joint_organic_n50000_c10000_lampool400_solvent_es_resp_v2_h256_s104000_nvar_N.pt"}


def load(ckpt):
    from decomp.mace_adapter import MACEDecomposition
    from model_eval import BB
    bb = torch.load(BB, map_location="cpu", weights_only=False).to(torch.float64).to("cuda").eval()
    torch.manual_seed(0)
    m = MACEDecomposition(bb, hidden=256, n_pair_channels=4).to(torch.float64).to("cuda").eval()
    m.head.load_state_dict({k: v.to(torch.float64) for k, v in torch.load(ckpt, map_location="cuda").items()})
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def stats(a, b):
    """a vs reference b (pooled changes)."""
    return {"corr": float(np.corrcoef(a, b)[0, 1]), "slope": float((a * b).sum() / (b * b).sum()),
            "sd_diff": float((a - b).std(ddof=1)), "sd_ref": float(b.std(ddof=1)), "sd_pred": float(a.std(ddof=1))}


def summarize(L, T, cmm, X, steps):
    """All inputs are aligned [trajectory, saved step] arrays, in kJ/mol."""
    dU = L + X - cmm
    return {"level_mean": float((L - T).mean()), "level_sd": float((L - T).std(ddof=1)),
            "learned_mean": float(L.mean()), "T_mean": float(T.mean()),
            "level_t0_sd": float((L[:, 0] - T[:, 0]).std(ddof=1)),
            "level_t0_fit": stats(L[:, 0], T[:, 0]),
            "response_pooled": stats((L[:, 1:] - L[:, :1]).ravel(), (T[:, 1:] - T[:, :1]).ravel()),
            "response_by_step": {str(s): stats(L[:, k] - L[:, 0], T[:, k] - T[:, 0])
                                 for k, s in enumerate(steps) if k},
            "dU_t0_mean": float(dU[:, 0].mean()), "dU_t0_sd": float(dU[:, 0].std(ddof=1)),
            "dU_internal_sd_by_step": {str(s): float((dU[:, k] - dU[:, 0]).std(ddof=1))
                                       for k, s in enumerate(steps) if k}}


def main(a):
    reps = reps_of(a.reps)
    if len(set(reps)) != len(reps):
        raise ValueError("duplicate repetitions would count the same configurations twice")
    tag = "" if a.reps == "1-4" else f"_reps{a.reps}"
    if not a.tag or any(c in a.tag for c in "/\\"):
        raise ValueError("use a nonempty filename tag without path separators")
    prefix = HERE / "logs" / f"validate_{a.system}{tag}{a.tag}"
    for ext in (".json", ".npz"):
        if prefix.with_suffix(ext).exists():
            raise FileExistsError(f"preserving {prefix.with_suffix(ext)}; choose a new --tag")
    leg = eh.local_E.Leg(a.system)
    paths = [DATA / a.system / f"rep{r}.npz" for r in reps]
    zs = [np.load(p, allow_pickle=False) for p in paths]
    metas = [json.loads(str(z["meta"])) for z in zs]
    steps = json.loads(str(zs[0]["meta"]))["steps"]
    for rep, z, meta in zip(reps, zs, metas):
        if meta["system"] != a.system or meta["rep"] != rep or meta["state"] != 0 or meta["steps"] != steps:
            raise ValueError(f"rep{rep}: inconsistent system/rep/state/steps")
        if not np.array_equal(z["numbers"], zs[0]["numbers"]) or not np.array_equal(z["frag"], zs[0]["frag"]):
            raise ValueError(f"rep{rep}: inconsistent cluster elements or ligand mapping")
        if len(z["positions"]) != len(meta["frames"]) * len(steps):
            raise ValueError(f"rep{rep}: configuration count differs from frame metadata")
    n_t = len(steps)
    P = np.concatenate([z["positions"] for z in zs])                  # [n_traj * n_t, n, 3]
    has_si = all("sysidx" in z.files for z in zs)          # training files (rep5-21) carry no sysidx: no C_MM
    SI = np.concatenate([z["sysidx"] for z in zs]) if has_si else [None] * len(P)
    num, frag = zs[0]["numbers"], zs[0]["frag"]
    EV = eh.EV
    T = np.concatenate([(z["C_bb"] - z["cross"]).ravel() for z in zs]) * EV
    X = np.concatenate([z["cross"].ravel() for z in zs]) * EV
    no_dummy = np.zeros(len(leg.lig), bool)
    cmm = np.array([leg.c_mm(p, s, no_dummy).sum() if has_si else np.nan for p, s in zip(P, SI)])
    out = {"system": a.system, "reps": a.reps, "n_traj": len(P) // n_t, "steps": steps,
           "heads": {h: HEADS[h] for h in a.heads}, "dtype": "float64", "units": "kJ/mol",
           "sources": [{"path": str(p), "meta": m} for p, m in zip(paths, metas)],
           "scope": "S0, fixed environment/Omega per trajectory; fitted ligand deformation; no mixing or physical-accuracy claim"}
    traj_rep = np.concatenate([np.full(len(z["positions"]) // n_t, r) for z, r in zip(zs, reps)])
    raw = {"T": T.reshape(-1, n_t), "X": X.reshape(-1, n_t), "C_MM": cmm.reshape(-1, n_t),
           "steps": np.array(steps), "traj_rep": traj_rep,
           "source_frame": np.concatenate([m["frames"] for m in metas])}
    for name in a.heads:
        ckpt = HEADS[name]
        print(f"{a.system}: {name}, {len(P)} saved configurations, batch {a.batch}", flush=True)
        m = load(ckpt)
        ch = []
        for k in range(0, len(P), a.batch):
            cl = [(p, num, frag) for p in P[k:k + a.batch]]
            ch.append(np.nansum(eh.couplings_dt(m, cl, no_dummy, torch.float64), 1) * EV)
        ch = np.concatenate(ch)
        if not np.isfinite(ch).all():
            raise ValueError(f"{name}: nonfinite model coupling")
        L = ch - X
        del m
        torch.cuda.empty_cache()
        Lr = L.reshape(-1, n_t)
        r = summarize(Lr, raw["T"], raw["C_MM"], raw["X"], steps)
        r["by_rep"] = {str(rep): summarize(Lr[traj_rep == rep], raw["T"][traj_rep == rep],
                                          raw["C_MM"][traj_rep == rep], raw["X"][traj_rep == rep], steps)
                       for rep in reps}
        out[name] = r
        raw[f"{name}_learned"] = Lr
        print(f"  residual sd {r['level_sd']:.3f}; response sd(diff) {r['response_pooled']['sd_diff']:.3f} kJ/mol", flush=True)
    bbmm = (T + X - cmm).reshape(-1, n_t)
    out["C_bb_minus_C_MM_internal_sd_by_step"] = {str(s): float((bbmm[:, k] - bbmm[:, 0]).std(ddof=1))
                                                  for k, s in enumerate(steps) if k}
    kT = eh.KT
    print(f"{a.system} reps {a.reps}: {out['n_traj']} trajectories x steps {steps} (2 fs); kJ/mol; kT {kT:.3f}")
    print(f"  {'':34s}" + "".join(f"{h:>12s}" for h in a.heads))
    rows = [("level: learned - T mean", lambda r: r["level_mean"]), ("level: learned - T sd", lambda r: r["level_sd"]),
            ("level: learned - T sd, t=0 only", lambda r: r["level_t0_sd"]),
            ("response pooled: corr", lambda r: r["response_pooled"]["corr"]),
            ("response pooled: slope", lambda r: r["response_pooled"]["slope"]),
            ("response pooled: sd(diff)", lambda r: r["response_pooled"]["sd_diff"])]
    last = str(steps[-1])
    rows += [(f"response {2 * steps[-1]} fs: corr", lambda r: r["response_by_step"][last]["corr"]),
             (f"response {2 * steps[-1]} fs: slope", lambda r: r["response_by_step"][last]["slope"]),
             (f"response {2 * steps[-1]} fs: sd(diff)", lambda r: r["response_by_step"][last]["sd_diff"]),
             ("dU t=0: sd over frames", lambda r: r["dU_t0_sd"])]
    rows += [(f"dU internal change sd, {2 * int(s)} fs", (lambda s: lambda r: r["dU_internal_sd_by_step"][s])(s))
             for s in map(str, steps[1:])]
    for lab, f in rows:
        print(f"  {lab:34s}" + "".join(f"{f(out[h]):12.3f}" for h in a.heads))
    print(f"  sd of T change at {2 * steps[-1]} fs (reference): {out[a.heads[0]]['response_by_step'][last]['sd_ref']:.3f};"
          f"  C_bb - C_MM internal change sd: "
          + ", ".join(f"{2 * int(s)} fs {v:.2f}" for s, v in out["C_bb_minus_C_MM_internal_sd_by_step"].items()))
    raw["meta"] = np.array(json.dumps({k: out[k] for k in ("system", "reps", "heads", "sources", "dtype", "units", "scope")}))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp = prefix.with_suffix(".tmp.npz")
    np.savez(tmp, **raw)
    os.replace(tmp, prefix.with_suffix(".npz"))
    tmp = prefix.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(out, indent=1) + "\n")
    os.replace(tmp, prefix.with_suffix(".json"))
    for z in zs:
        z.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--reps", default="1-4")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--heads", nargs="+", choices=list(HEADS), default=list(HEADS))
    ap.add_argument("--tag", default="_allheads", help="new output suffix; historical validation files are never overwritten")
    main(ap.parse_args())

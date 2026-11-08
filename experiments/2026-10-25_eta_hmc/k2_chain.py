"""K2 eta chain: MM-HMC proposals on V_j, Metropolis on U = V_j + eta * dU_K2 (DESIGN §10, K2 revision).

The integrator uses MM forces only; the scalar acceptance uses C_bb - C_MM (head-independent); the head
atom readout and q are recorded on the same chain (k2_records, schema k2_1). Endpoints: S0 or S_K only.

python k2_chain.py SYSTEM --eta E --start-file NPZ [--start-frame -1] --start-label LABEL \
       --head CKPT --backbone MODEL [--state 0] [--L 20] [--dt-fs 2] [--cycles N] [--tag T] [--platform CUDA]
Output: runs/SYSTEM/k2chain_S{state}_eta{eta:.3f}_{label}{tag}.npz (rerun the same command to resume).
"""

import argparse
import zlib

import numpy as np
import openmm

import eta_hmc as eh
import k2_energy as k2
from k2_records import run_recorded


def start_coordinates(path, frame):
    with np.load(path, allow_pickle=False) as z:
        if "xyz" in z.files and len(z["xyz"]):
            x = z["xyz"][frame]
        elif frame in (-1, None):
            x = z["last_positions"]
        else:
            raise ValueError(f"{path}: no saved frames")
    return np.asarray(x, np.float64)


def main(a):
    if not 0 <= a.eta <= 1:
        raise ValueError("eta must be in [0,1]")
    leg = eh.local_E.Leg(a.system)
    system, groups, box = eh.mm_system(a.system)
    K = len(groups)
    ig = eh.proposal_integrator(a.L, a.dt_fs, leg.temperature_K)
    seed = (zlib.crc32(a.start_label.encode()) % 100000) * 1000 + int(round(a.eta * 1000))
    ig.setRandomNumberSeed(seed + 1)
    c = eh.make_context(system, box, ig, a.platform)
    eh.gm.set_state(c, a.state, K)
    me = k2.ModelEnergyK2(leg, a.state, head=a.head, backbone=a.backbone, device=a.device)
    rng = np.random.default_rng(seed)
    out = eh.HERE / "runs" / a.system / f"k2chain_S{a.state}_eta{a.eta:.3f}_{a.start_label}{a.tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    x0 = start_coordinates(a.start_file, a.start_frame)
    want = {"system": a.system, "state": a.state, "eta": a.eta, "start_label": a.start_label,
            "start_file": str(a.start_file), "start_frame": a.start_frame, "L": a.L, "dt_fs": a.dt_fs,
            "T_K": leg.temperature_K, **me.identity,
            "leg": "solvent", "n_particles": system.getNumParticles(),
            "ligand_system_indices": list(map(int, leg.lig)), "groups": [list(map(int, g)) for g in leg.groups],
            "dummy_mask": leg.dummy(a.state).tolist(), "box_nm": np.asarray(box, float).tolist(),
            "system_xml": str(eh.gm.leg(a.system)[0]), "softcore_alpha": eh.gm.ALPHA,
            "system_definition": openmm.XmlSerializer.serialize(system),
            "context_parameters": dict(c.getParameters()),
            "integrator_definition": openmm.XmlSerializer.serialize(ig), "openmm_version": openmm.__version__,
            "platform": a.platform, "model_device": a.device,
            "acceptance_seed": seed, "proposal_seed": seed + 1, "energy_units": "kJ/mol", "xyz_units": "nm"}
    meta = run_recorded(c, ig, me, rng, x0, a.eta, a.cycles, out, want, eh.hmc,
                        checkpoint_every=a.checkpoint_every, coords_every=a.coords_every, log_every=a.log_every)
    print(f"{out.name}: complete={meta['complete']}; {meta['completed_cycles']} cycles; "
          f"model calls this process {me.calls}; {me.seconds / max(me.calls, 1):.2f} s per K2 query", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--start-file", required=True)
    ap.add_argument("--start-frame", type=int, default=-1)
    ap.add_argument("--start-label", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--dt-fs", type=float, default=2.0)
    ap.add_argument("--cycles", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--device", default="cuda", help="model device; cpu allowed for checks")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--coords-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=0)
    main(ap.parse_args())

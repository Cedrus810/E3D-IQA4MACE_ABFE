"""Is a backbone coupling C_bb exactly 0 at S_K (all ligand dummy) under edge cut + electrostatic masking? (user 2026-10-25)

C_bb,j = E_bb,j(L + Omega) - E_bb(L) - E_bb(Omega), where E_bb,j is the frozen POLAR-1-M on the ligand + Omega cluster
with the dummy atoms' interactions to Omega switched off. At S_K every ligand atom is dummy. Variants of "switched off":
  A  all ligand-Omega graph edges cut (message passing only)
  B  A + ligand-Omega Coulomb removed from the electrostatic energy (coulomb_energy -> E_es(L) + E_es(Omega))
  C  B + the field each fragment feels comes only from its own fragment's densities (field-update iterations)
Left in C: PolarMACE's total-charge constraint (Fukui redistribution) acts on the whole graph, not per fragment.
  D  A + batch relabelled by fragment (per-fragment total-charge constraint, field and Coulomb); no B/C patches
Frames: CypD-8 rep1, first N frames of S4 (all dummy) and S0 (for scale), Omega = 50 nearest waters. float64.
Run: python test_SK_zero.py [--frames 10]
"""

import argparse
import contextlib

import numpy as np
import torch

import eta_hmc as eh

NAME = "CyclophilinD_ligand-8"


class CoulombNoCross(torch.nn.Module):
    def __init__(self, inner, frag):
        super().__init__()
        self.inner, self.frag = inner, frag

    def forward(self, **kw):
        sf = kw["source_feats"]
        m = (self.frag == 0).to(sf.dtype).view(-1, *([1] * (sf.dim() - 1)))
        return self.inner(**{**kw, "source_feats": sf * m}) + self.inner(**{**kw, "source_feats": sf * (1 - m)})


@contextlib.contextmanager
def masked(bb, frag, coulomb, field):
    inner_c = bb.coulomb_energy
    desc = bb.electric_potential_descriptor
    inner_f = desc.forward_dynamic
    if coulomb:
        bb.coulomb_energy = CoulombNoCross(inner_c, frag)
    if field:
        def fd(cache, source_feats, pbc):
            m = (frag == 0).to(source_feats.dtype).view(-1, *([1] * (source_feats.dim() - 1)))
            fL = inner_f(cache=cache, source_feats=source_feats * m, pbc=pbc)
            fE = inner_f(cache=cache, source_feats=source_feats * (1 - m), pbc=pbc)
            mo = (frag == 0).to(fL.dtype).view(-1, *([1] * (fL.dim() - 1)))
            return mo * fL + (1 - mo) * fE
        desc.forward_dynamic = fd
    try:
        yield
    finally:
        bb.coulomb_energy = inner_c
        desc.forward_dynamic = inner_f


def per_fragment_graphs(b):
    """Same nodes and edges, but batch = fragment: PolarMACE then applies its total-charge (Fukui) constraint,
    its real-space field and its Coulomb energy per fragment. Ligand atoms come first in the carve."""
    fr = b["frag_id"].long()
    nL = int((fr == 0).sum())
    assert bool((fr[:nL] == 0).all())
    out = dict(b)
    out["batch"] = fr
    out["ptr"] = torch.tensor([0, nL, len(fr)], device=fr.device)
    for k in ("volume", "pbc", "fermi_level", "external_field", "head", "total_charge", "total_spin"):
        out[k] = b[k].repeat(2, *([1] * (b[k].dim() - 1)))
    for k in ("cell", "rcell"):
        out[k] = b[k].repeat(2, 1)
    return out


def energy(bb, p, n, f, cut, coulomb=False, field=False, charge=False):
    from decomp.data import mace_batch
    b = mace_batch([p], [n], bb, charges=[0], frag_ids=[f], device="cuda", dtype=torch.float64)
    if charge:
        b = per_fragment_graphs(b)
    if cut:
        inL = b["frag_id"] == 0
        s, d = b["edge_index"]
        keep = inL[s] == inL[d]
        b = {**b, "edge_index": b["edge_index"][:, keep], "shifts": b["shifts"][keep], "unit_shifts": b["unit_shifts"][keep]}
    with torch.no_grad(), masked(bb, b["frag_id"], coulomb, field):
        return float(bb(b, training=False, compute_force=False)["energy"].sum()) * eh.EV


def main(a):
    leg = eh.local_E.Leg(NAME)
    model = __import__("model_eval").load_model().to(torch.float64)
    bb = model.backbone
    for j in (leg.K, 0):
        xyz = np.load(eh.local_E.GW / "runs" / NAME / f"rep1_S{j}.npz")["xyz"][:a.frames].astype(np.float64)
        res = []
        for x in xyz:
            p, n, f, _ = leg.carve(x, 50)
            L, O = f == 0, f == 1
            sep = energy(bb, p[L], n[L], f[L], False) + energy(bb, p[O], n[O], f[O], False)
            full = energy(bb, p, n, f, False)
            A = energy(bb, p, n, f, True)
            B = energy(bb, p, n, f, True, coulomb=True)
            C = energy(bb, p, n, f, True, coulomb=True, field=True)
            D = energy(bb, p, n, f, True, charge=True)
            res.append([full - sep, A - sep, B - sep, C - sep, D - sep])
        r = np.array(res)
        print(f"S{j} ({'all dummy' if j == leg.K else 'all coupled'}), {len(r)} frames, kJ/mol: "
              f"E_variant(L+Omega) - E(L) - E(Omega)")
        for k, lab in enumerate(("no masking (= C_bb)", "A edges cut", "B A + no L-Omega Coulomb",
                                 "C B + own-fragment field", "D A + per-fragment graphs")):
            print(f"  {lab:28s} mean {r[:, k].mean():+10.4f}  sd {r[:, k].std():8.4f}  max|.| {np.abs(r[:, k]).max():9.4f}")
    print("float64; one backbone pass per variant per frame")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    main(ap.parse_args())

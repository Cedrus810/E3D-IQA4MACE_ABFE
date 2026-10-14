"""298 K ligand-water contacts vs the minimised frames and the head's training data.

Run: python contacts.py   (after thermal_frames.py for seeds 23 and 11)

Per (ligand element, water element): the shortest contact in each snapshot.
Thermal = unminimised MD every 0.5 ps, 100-1050 ps, both seeds pooled
(snapshots 0.5 ps apart are correlated, so n overstates the independent
count). Minimised = the 20 frames of frames_test_solvent.npz. Clusters and
DES370K columns are from ../2026-10-12_wall_radius/coverage.json.
"""

import json

import numpy as np

from thermal_frames import LIG_EL, SYM, W_EL, contact_minima

E3D = "/home/ruigengji/E3D"
cov = json.load(open(f"{E3D}/experiments/2026-10-12_wall_radius/coverage.json"))
des, clu = cov[list(cov)[0]], cov[list(cov)[1]]

th = np.concatenate([np.load(f, allow_pickle=True)["contacts"]
                     for f in ("thermal_test_solvent_contacts.npz", "thermal_solvent_contacts.npz")])
z = np.load(f"{E3D}/frames_test_solvent.npz", allow_pickle=True)
lig = z["resids"] == z["resids"][0]
wat = z["resnames"] == "HOH"
mn = np.array([contact_minima(p, lig, wat, z["numbers"], z["box"]) for p in z["positions"]])

print(f"shortest ligand-water contact per snapshot (A); thermal n = {len(th)} snapshots, "
      f"minimised n = {len(mn)} frames")
print(f"{'pair':6s} | {'thermal 298 K':^34s} | {'minimised':^11s} | clusters | DES370K")
print(f"{'':6s} | {'min':>6s}{'p0.1':>7s}{'p1':>7s}{'p5':>7s}{'median':>7s} | "
      f"{'min':>5s}{'med':>6s} | {'min':>8s} | {'min':>5s}{'p1':>6s}")
out = {}
k = 0
for a in LIG_EL:
    for b in W_EL:
        sa, sb = SYM[a], SYM[b]
        dk = f"{sb}-H" if sa == "H" and sb != "H" else f"{sa}-{sb}"
        x = th[:, k]
        q = dict(min=x.min(), p01=np.percentile(x, 0.1), p1=np.percentile(x, 1),
                 p5=np.percentile(x, 5), median=np.median(x))
        out[f"{sa}-{sb}w"] = dict(thermal={kk: float(v) for kk, v in q.items()},
                                  minimised_min=float(mn[:, k].min()),
                                  minimised_median=float(np.median(mn[:, k])),
                                  clusters_min=clu[f"{sa}-{sb}"]["min"],
                                  des_min=des[dk]["min"], des_p1=des[dk]["p1"])
        print(f"{sa}-{sb}w  | {q['min']:6.2f}{q['p01']:7.2f}{q['p1']:7.2f}{q['p5']:7.2f}"
              f"{q['median']:7.2f} | {mn[:, k].min():5.2f}{np.median(mn[:, k]):6.2f} | "
              f"{clu[f'{sa}-{sb}']['min']:8.2f} | {des[dk]['min']:5.2f}{des[dk]['p1']:6.2f}")
        k += 1
json.dump(out, open("contacts.json", "w"), indent=1)

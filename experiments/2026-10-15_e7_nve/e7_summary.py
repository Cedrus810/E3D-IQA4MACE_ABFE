"""E7 summary: NVE energy conservation of the ML/MM + FIRES + wall states against pure MM.

Run: python e7_summary.py > e7_summary.log

Per state (e7_*.npz):
  drift   slope of E_tot against t (kJ/mol/ps)
  sd      sd of E_tot about that line (kJ/mol)
  step    sd and max of |E_tot(k+1) - E_tot(k)| (kJ/mol)
  T       mean temperature
  contact closest ligand-water r / r0 (wall radii), quantiles over samples
  FIRES   max U_F and violating pairs
Velocity Verlet on a smooth potential: the E_tot fluctuation scales as dt^2,
so q1.5 at 1 fs should show ~1/4 of the 2 fs fluctuation; a discontinuity
in U adds steps that do not shrink with dt. The ML energy is float32: at
|E_bb| ~ 6.6e4 eV its resolution is 2^-7 eV = 0.75 kJ/mol, a floor for the
per-step jumps of every ML state.
"""

import glob

import numpy as np

EV = 96.48533212
DOF = 3 * 4076 - 4057 - 3          # atoms, constraints (SETTLE + ligand X-H), COM
print(f"float32 resolution of E_bb at 6.6e4 eV: {np.spacing(np.float32(65595.0)) * EV:.2f} kJ/mol\n")
print(f"{'state':10s} {'dt':>4s} {'n':>5s} {'drift':>9s} {'sd':>7s} {'step sd':>8s} {'step max':>9s} "
      f"{'T':>6s}  contact r/r0 min/p1/p50   FIRES U_F max, viol max")
rows = {}
for f in sorted(glob.glob("e7_*.npz")):
    d = np.load(f, allow_pickle=True)
    name = f[3:-4]
    E, t = d["E_tot"], d["t_ps"]
    a, b = np.polyfit(t, E, 1)
    res = E - (a * t + b)
    st = np.abs(np.diff(E))
    rows[name] = dict(sd=res.std(), step=st.std(), dt=float(d["dt"]))
    con = d["contact"]
    fr = d["fires"]
    T = 2 * d["KE"].mean() / (DOF * 0.0083144626)
    print(f"{name:10s} {float(d['dt']):4.1f} {len(E) - 1:5d} {a:+9.3f} {res.std():7.3f} {st.std():8.3f} "
          f"{st.max():9.3f} {T:6.1f}  "
          f"{con.min():.2f}/{np.percentile(con, 1):.2f}/{np.median(con):.2f}"
          + (f"          {fr[:, 0].max():.2f}, {int(fr[:, 1].max())}" if fr.size else "          --"))
for f in sorted(glob.glob("e7_q0.0.npz")):
    w = np.load(f)["wall1"] / 2.479
    print(f"\nW1 by FEP from q 0.0 (s_w = 0 -> 1, {len(w)} samples): dG_W1 = {-2.479 * np.log(np.mean(np.exp(-w))):+.3f} "
          f"kJ/mol; samples with U_wall(s_w = 1) > 0: {100 * (w > 0).mean():.1f}%, max {w.max() * 2.479:.2f} kJ/mol")
if "q1.5" in rows and "q1.5_dt1" in rows:
    r2, r1 = rows["q1.5"], rows["q1.5_dt1"]
    print(f"\nq1.5, 2 fs / 1 fs: sd ratio {r2['sd'] / r1['sd']:.2f}, step-sd ratio {r2['step'] / r1['step']:.2f} "
          f"(smooth potential: ~4; floor-dominated: ~1)")
if "mm" in rows:
    print("ratios to pure MM (2 fs): " + ", ".join(
        f"{k} sd {v['sd'] / rows['mm']['sd']:.1f}x" for k, v in rows.items() if k != "mm" and v["dt"] == 2.0))

"""CPU checks of the 2026-10-26 fixes (REVIEW_2026-10-26_nontraining.md C1-C3): analyze coverage labels,
exact tag match, target split, NaN on active atoms. Reads the existing v1 chains through symlinks in WORK;
nothing under runs/ is written.

Run: python test_fixes.py WORK_DIR
"""

import contextlib
import io
import json
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eta_hmc as eh  # noqa: E402

SYS = "CyclophilinD_ligand-8"
SRC = HERE / "runs" / SYS
WORK = pathlib.Path(sys.argv[1])


def args(**kw):
    d = dict(system=SYS, state=0, tag="_v1", discard=0.2, expect_etas="", expect_starts="")
    d.update(kw)
    return types.SimpleNamespace(**d)


def run(files, **kw):
    """Symlink files into WORK/runs/SYS, analyze there, return (stdout, json)."""
    d = WORK / "runs" / SYS
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        f.unlink()
    for f in files:
        (d / f.name).symlink_to(f) if f.parent == SRC else f.rename(d / f.name)
    eh.HERE = WORK
    buf = io.StringIO()
    a = args(**kw)
    with contextlib.redirect_stdout(buf):
        eh.analyze(a)
    return buf.getvalue(), json.load(open(d / f"analysis_S0{a.tag}.json"))


v1 = sorted(SRC.glob("chain_S0_eta*_start*_v1.npz"))
grid = ",".join(f"{e / 10:.1f}" for e in range(11))
print(f"{len(v1)} v1 chains")

out, js = run(v1, expect_etas=grid, expect_starts="1,2,3,4")
print("[1] all v1, expected 11 x 4:\n" + "\n".join(l for l in out.splitlines() if "INCOMPLETE" in l or "BAR sum" in l
                                                  or l.startswith("target")))
print(f"    complete={js[0]['complete']} missing={len(js[0]['missing'])} label={js[0]['label']!r}")

out, js = run([f for f in v1 if 0.15 < float(f.name.split("eta")[1][:5]) < 0.85])
print("[2] eta 0.2-0.8 only: " + [l for l in out.splitlines() if "BAR sum" in l][0].strip())
print(f"    complete={js[0]['complete']}")

out, js = run(v1 + [SRC / "chain_S0_eta0.500_start1_smoke.npz"], tag="")
print(f"[3] empty tag, v1 + smoke present: {len(js)} targets, files {[g['files'] for g in js]}")

f0 = v1[0]
z = dict(np.load(f0))
m = json.loads(str(z["meta"]))
m["head"] = "/some/other/head.pt"
z["meta"] = json.dumps(m)
alt = WORK / ("alt_" + f0.name)
np.savez(alt, **z)
alt = alt.rename(WORK / f0.name.replace("start", "start9"))
out, js = run(v1 + [alt])
print(f"[4] one chain with another head: {len(js)} targets; heads {[g['target']['head'] for g in js]}; "
      f"files per target {[len(g['files']) for g in js]}")

# [5] NaN: real carve / C^MM on a saved S1 frame, couplings_dt replaced by a stub
leg = eh.local_E.Leg(SYS)
xyz = np.load(eh.local_E.GW / "runs" / SYS / "rep1_S1.npz")["xyz"][0].astype(np.float64)
me = object.__new__(eh.ModelEnergy)
me.leg, me.dm, me.model, me.dtype, me.calls, me.seconds = leg, leg.dummy(1), None, None, 0, 0.0
n = len(leg.lig)
good = np.full((1, n), 0.01)
good[0, me.dm] = np.nan
eh.couplings_dt = lambda *a_: good.copy()
e_good = me(xyz)
bad = good.copy()
bad[0, np.where(~me.dm)[0][0]] = np.nan
eh.couplings_dt = lambda *a_: bad.copy()
try:
    me(xyz)
    print("[5] NaN on active atom: NOT raised")
except FloatingPointError as err:
    print(f"[5] dummy NaN only: dU {e_good:+.3f} kJ/mol (finite); NaN on active atom -> FloatingPointError: {err}")

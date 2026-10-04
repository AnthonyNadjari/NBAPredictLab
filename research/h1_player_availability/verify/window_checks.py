"""Is the 'beats open but not close' contrast for the realistic proxy an artifact of the training
window? Open WF trains on 2023-24(+2024-25) only; close WF also trains on 2021-22/2022-23.
Here: close WF with the SAME training window as open WF (seasons >= 2023-24), same 2 test seasons,
and on the same games (games that have an opening line)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

H1 = Path(r"C:\Users\nadja\NBAPredictLab\research\h1_player_availability")
sys.path.insert(0, str(H1))
sys.argv = [sys.argv[0]]
import run  # noqa

g = run.load()
both = g[g.mkt_open.notna()].copy()  # 2023-24..2025-26, games with both lines
tests = ["2024-25", "2025-26"]
print("matched window (train seasons >= 2023-24), test 2024-25 + 2025-26, same games, WF")
for spec in ("offset", "lr"):
    for name in run.PRIMARY:
        line = []
        for base in ("mkt_open", "mkt_close"):
            o, c = run.evaluate(both, base, run.PRIMARY[name], tests, "wf", spec)
            dd = o.ll_model - o.ll_mkt
            m, lo, hi = run.boot_mean(dd)
            per = o.assign(dd=dd).groupby("season").dd.mean()
            line.append(f"{base[4:]} {1e3*m:+.2f} [{1e3*lo:+.2f},{1e3*hi:+.2f}] per-season {np.round(1e3*per.values,2)}")
        print(f"{spec:6s} {name:10s} | " + " | ".join(line))

print("\nmatched LOSO on 2023-24..2025-26 (same games) - close vs open")
for spec in ("offset",):
    for name in run.PRIMARY:
        line = []
        for base in ("mkt_open", "mkt_close"):
            o, c = run.evaluate(both, base, run.PRIMARY[name], ["2023-24", "2024-25", "2025-26"], "loso", spec)
            dd = o.ll_model - o.ll_mkt
            m, lo, hi = run.boot_mean(dd)
            per = o.assign(dd=dd).groupby("season").dd.mean()
            line.append(f"{base[4:]} {1e3*m:+.2f} [{1e3*lo:+.2f},{1e3*hi:+.2f}] per-season {np.round(1e3*per.values,2)}")
        print(f"{spec:6s} {name:10s} | " + " | ".join(line))

print("\nin-season offset GLM (z-stats), standardized within season")
for base in ("mkt_close", "mkt_open"):
    d = g[g[base].notna()]
    for s in sorted(d.season.unique()):
        x = d[d.season == s]
        row = []
        for name in ("prev_min", "oracle_min", "prev_gs", "oracle_gs"):
            f = run.PRIMARY[name][0]
            z = (x[f] - x[f].mean()) / x[f].std()
            tr = x.assign(**{f: z})
            _, c = run.fit_offset(tr, tr, base, [f])
            b = c[1]
            zz = ((tr[f] - tr[f].mean()) / tr[f].std()).to_numpy()
            p = 1 / (1 + np.exp(-(run.logit(x[base]) + b * zz)))
            se = 1 / np.sqrt(np.sum(p * (1 - p) * zz ** 2))
            row.append(f"{name} b={b:+.3f} z={b/se:+.2f}")
        print(f"{base} {s}: " + "  ".join(row))

# rolling 1-season training window for the close (sensitivity, post hoc)
print("\nclose, offset, train on previous season only (post hoc sensitivity)")
for name in run.PRIMARY:
    out = []
    for s in run.TEST_SEASONS:
        prev = {"2022-23": "2021-22", "2023-24": "2022-23", "2024-25": "2023-24", "2025-26": "2024-25"}[s]
        tr, te = g[g.season == prev], g[g.season == s]
        p, c = run.fit_offset(tr, te, "mkt_close", run.PRIMARY[name])
        out.append(1e3 * (run.ll(te.y, p) - run.ll(te.y, te.mkt_close)).mean())
    print(f"{name:10s} per-season {np.round(out, 2)}  mean {np.mean(out):+.2f}")

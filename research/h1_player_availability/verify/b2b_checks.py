"""B2B split for the realistic-proxy vs-open effect (claimed in README but not computed in run.py),
plus per-season coefficient stability and a check of the README's player-level claims."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

H1 = Path(r"C:\Users\nadja\NBAPredictLab\research\h1_player_availability")
sys.path.insert(0, str(H1))
sys.argv = [sys.argv[0]]
import run, build  # noqa

g = run.load()
t = build.team_logs().sort_values(["team", "date"])
t["prev_date"] = t.groupby("team").date.shift()
t["rest"] = (t.date - t.prev_date).dt.days
rest = t[["team", "GAME_ID", "rest"]]
for side in ("home", "away"):
    g = g.merge(rest.rename(columns={"team": side, "GAME_ID": "game_id", "rest": f"{side}_rest"}),
                on=["game_id", side], how="left")
g["b2b"] = (g.home_rest == 1) | (g.away_rest == 1)
# does a team with a non-zero prev-proxy absence play a b2b?
g["b2b_with_prev_abs"] = ((g.home_rest == 1) & (g.home_miss_min_prev > 0)) | ((g.away_rest == 1) & (g.away_miss_min_prev > 0))

for scheme, tests in (("wf", run.TEST_SEASONS), ("loso", ["2023-24", "2024-25", "2025-26"])):
    for spec in ("offset", "lr"):
        for name in ("prev_min", "oracle_min"):
            o, c = run.evaluate(g, "mkt_open", run.PRIMARY[name], tests, scheme, spec)
            o = o.merge(g[["game_id", "b2b", "b2b_with_prev_abs"]], on="game_id")
            o["dd"] = o.ll_model - o.ll_mkt
            out = []
            for lab, m in (("b2b", o.b2b), ("not b2b", ~o.b2b), ("b2b & team w/ prev abs", o.b2b_with_prev_abs)):
                mm, lo, hi = run.boot_mean(o.dd[m])
                out.append(f"{lab}: n={m.sum()} {1e3*mm:+.2f} [{1e3*lo:+.2f},{1e3*hi:+.2f}]")
            print(f"open {scheme} {spec} {name}: " + " | ".join(out))

# per-season coefficient of offset prev_min fitted on each season alone (open)
d = g[g.mkt_open.notna()]
for s in sorted(d.season.unique()):
    tr = d[d.season == s]
    _, c = run.fit_offset(tr, tr, "mkt_open", ["d_miss_min_prev"])
    _, c2 = run.fit_offset(tr, tr, "mkt_open", ["d_miss_min_oracle"])
    _, c3 = run.fit_offset(tr, tr, "mkt_close", ["d_miss_min_prev"])
    print(f"in-season offset coef {s}: open prev_min {c[1]:+.3f}  open oracle_min {c2[1]:+.3f}  close prev_min {c3[1]:+.3f}")
d = g[g.mkt_close.notna()]
for s in sorted(d.season.unique()):
    tr = d[d.season == s]
    cs = {n: run.fit_offset(tr, tr, "mkt_close", run.PRIMARY[n])[1][1] for n in run.PRIMARY}
    print(f"in-season close offset coefs {s}: " + " ".join(f"{k} {v:+.3f}" for k, v in cs.items()))

# player-level README claims
a = pd.read_csv(build.CACHE / "rotation_player_games.csv")
print("P(out today | out last game) =", round(a[a.absent_prev].absent_oracle.mean(), 3),
      " P(out today | played last) =", round(a[~a.absent_prev].absent_oracle.mean(), 3))
print("mean rotation size", a.groupby(["team", "GAME_ID"]).size().mean())

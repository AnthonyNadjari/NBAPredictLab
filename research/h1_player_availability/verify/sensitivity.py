"""Rotation-definition sensitivity (in memory only, nothing written to the repo)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

H1 = Path(r"C:\Users\nadja\NBAPredictLab\research\h1_player_availability")
sys.path.insert(0, str(H1))
sys.argv = [sys.argv[0]]
import build, run  # noqa

base = pd.read_csv(build.CACHE / "h1_games.csv", dtype={"game_id": str})
keep = [c for c in base.columns if not c.startswith(("home_miss", "away_miss", "home_top", "away_top", "home_n_rot",
                                                       "away_n_rot", "home_rot", "away_rot", "has_rot"))]
games = base[keep]
p_all, t_all = build.player_logs(), build.team_logs()
build.REPL = build.replacement_rate(p_all)
vals = build.player_values(p_all, t_all, build.REPL)

for rmin, rwin in ((15, 10), (10, 10), (20, 10), (15, 5), (15, 20)):
    build.ROT_MIN, build.ROT_WINDOW = rmin, rwin
    a = build.availability(p_all, t_all, vals)
    tf = build.team_features(a)
    g = games.copy()
    for side in ("home", "away"):
        x = tf.rename(columns={c: f"{side}_{c}" for c in tf.columns if c not in ("team", "GAME_ID")})
        g = g.merge(x, left_on=["game_id", side], right_on=["GAME_ID", "team"], how="left").drop(columns=["GAME_ID", "team"])
    num = [c for c in g.columns if c.startswith(("home_miss", "away_miss", "home_top", "away_top"))]
    g[num] = g[num].fillna(0.0)
    for c in [c for c in g.columns if c.startswith("home_") and c[5:].startswith(("miss_", "top2_", "top1_"))]:
        g["d_" + c[5:]] = g[c] - g["away_" + c[5:]]
    g["y"] = g.home_win.astype(float)
    out = []
    for name in ("prev_min", "prev_gs", "prev_oo", "oracle_min"):
        for lab, b, tests, sch in (("closeWF", "mkt_close", run.TEST_SEASONS, "wf"),
                                    ("openWF", "mkt_open", run.TEST_SEASONS, "wf"),
                                    ("openLOSO", "mkt_open", ["2023-24", "2024-25", "2025-26"], "loso")):
            o, _ = run.evaluate(g, b, run.PRIMARY[name], tests, sch, "offset")
            dd = o.ll_model - o.ll_mkt
            m, lo, hi = run.boot_mean(dd)
            per = o.assign(dd=dd).groupby("season").dd.mean()
            out.append(f"  {name:10s} {lab:8s} {1e3*m:+.2f} [{1e3*lo:+.2f},{1e3*hi:+.2f}] seasons {np.round(1e3*per.values,2)}")
    print(f"ROT_MIN={rmin} ROT_WINDOW={rwin}\n" + "\n".join(out), flush=True)

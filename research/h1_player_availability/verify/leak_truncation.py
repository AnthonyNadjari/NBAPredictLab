"""Leak test: truncate player logs at cutoff D (keep rows < D), keep schedule <= D.
Recompute features. Expect: every feature for games < D identical; *_prev / n_rot /
rot_gs_total identical for games on D (only oracle-type features may change on D)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

H1 = Path(r"C:\Users\nadja\NBAPredictLab\research\h1_player_availability")
sys.path.insert(0, str(H1))
import build  # noqa

ref = pd.read_csv(build.CACHE / "h1_games.csv", dtype={"game_id": str})
p_all, t_all = build.player_logs(), build.team_logs()
build.REPL = build.replacement_rate(p_all)

cutoffs = sys.argv[1:] or ["2023-01-15", "2024-02-10", "2025-03-05", "2026-04-25"]
for D in cutoffs:
    Dt = pd.Timestamp(D)
    p = p_all[p_all.date < Dt].copy()
    t = t_all[t_all.date <= Dt].copy()
    vals = build.player_values(p, t, build.REPL)
    a = build.availability(p, t, vals)
    tf = build.team_features(a)
    tf = tf.merge(t[["team", "GAME_ID", "date"]], on=["team", "GAME_ID"])
    r = []
    for side in ("home", "away"):
        cols = [c for c in ref.columns if c.startswith(side + "_") and c != side + "_win"]
        x = ref[["game_id", "game_date", side] + cols].rename(columns={side: "team", **{c: c[len(side) + 1:] for c in cols}})
        r.append(x)
    r = pd.concat(r)
    m = r.merge(tf, left_on=["game_id", "team"], right_on=["GAME_ID", "team"], suffixes=("_ref", "_new"))
    before = m[pd.to_datetime(m.game_date) < Dt]
    on = m[pd.to_datetime(m.game_date) == Dt]
    feats = [c[:-4] for c in m.columns if c.endswith("_ref") and c[:-4] + "_new" in m.columns]
    prevf = [f for f in feats if f.endswith("_prev") or f in ("n_rot", "rot_gs_total")]
    mb = max(np.nanmax(np.abs(before[f + "_ref"] - before[f + "_new"])) for f in feats)
    mo = max(np.nanmax(np.abs(on[f + "_ref"] - on[f + "_new"])) for f in prevf) if len(on) else np.nan
    mo_or = max(np.nanmax(np.abs(on[f + "_ref"] - on[f + "_new"])) for f in feats if "oracle" in f) if len(on) else np.nan
    print(f"cutoff {D}: games-before compared {len(before)} max|diff| all feats = {mb:.3g}; "
          f"games on D {len(on)} team-rows, max|diff| prev/rot feats = {mo:.3g}; oracle feats diff on D = {mo_or:.3g}",
          flush=True)

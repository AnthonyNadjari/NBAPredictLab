"""CLV of the headline bets measured from the open and from the Kalshi 09:00 UTC price to the close.

    python research/h7_own_model/blend/verify_data/clv.py
"""
import numpy as np
import pandas as pd

import verify as v

u = pd.read_csv(v.DATA / "upsets_dataset.csv", usecols=["game_id", "game_date", "season", "home", "away", "home_win",
                                                         "mkt_open", "mkt_close"])
pr = pd.read_csv(v.PREDS)[["GAME_ID", "p"]].rename(columns={"GAME_ID": "game_id", "p": "own"})
o = pd.concat([pd.read_csv(v.DATA / f"espn_{s}.csv") for s in ("2023-24", "2024-25", "2025-26")], ignore_index=True)
o["home"], o["away"] = o.home.replace(v.MAP), o.away.replace(v.MAP)
o["game_date"] = pd.to_datetime(o.date_utc, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
e = u.merge(pr, on="game_id").merge(o.drop(columns=["season"]), on=["game_date", "home", "away"])
e = e[e.mkt_open.notna() & e.home_ml_open.notna()].sort_values(["game_date", "game_id"]).reset_index(drop=True)
p = np.full(len(e), np.nan)
for s in ("2024-25", "2025-26"):
    past, cur = (e.season < s).to_numpy(), (e.season == s).to_numpy()
    w = min(v.W, key=lambda w: v.ll(e.home_win.to_numpy()[past], v.bl(e.own.to_numpy()[past], e.mkt_open.to_numpy()[past], w)).mean())
    p[cur] = v.bl(e.own.to_numpy()[cur], e.mkt_open.to_numpy()[cur], w)
e["blend"] = p
k = pd.read_csv(v.DATA / "h6_timing" / "kalshi_snapshots.csv")[["event_id", "k_utc09"]]
t = e[e.blend.notna()].merge(k, on="event_id").dropna(subset=["k_utc09"]).reset_index(drop=True)
sh, sa = v.value(t.blend.to_numpy(), v.am_dec(t.home_ml_open), v.am_dec(t.away_ml_open))
sign = np.where(sh == 1, 1.0, np.where(sa == 1, -1.0, np.nan))
for nm, a, b in (("open -> 09UTC", t.mkt_open, t.k_utc09), ("09UTC -> close", t.k_utc09, t.mkt_close),
                 ("open -> close", t.mkt_open, t.mkt_close)):
    mv = sign * (b - a).to_numpy()
    print(f"{nm}: toward our side {np.nanmean(mv > 0.005) * 100:.1f}%, away {np.nanmean(mv < -0.005) * 100:.1f}%, mean {np.nanmean(mv) * 100:+.2f} pts (n={int(np.isfinite(mv).sum())})")

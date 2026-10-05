"""Does the morning blend (own model + opening line) make money at the real opening prices?

Bet 1 unit on a side when blend_prob * decimal_open_odds - 1 > threshold, at the main book's
opening moneyline (vig included). w(own) chosen walk-forward as in run.py. Bootstrap by game day.

    python research/h7_own_model/blend/roi.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from run import H7, DATA, W, ll, blend

ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX"}


def dec(ml):
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml > 0, 1 + ml / 100, 1 + 100 / ml.abs())


def main():
    u = pd.read_csv(DATA / "upsets_dataset.csv", usecols=["game_id", "game_date", "season", "home", "away",
                                                           "home_win", "mkt_open"])
    own = pd.read_csv(H7 / "ensemble" / "preds.csv")[["GAME_ID", "p"]].rename(columns={"GAME_ID": "game_id", "p": "own"})
    odds = pd.concat([pd.read_csv(DATA / f"espn_{s}.csv") for s in ("2023-24", "2024-25", "2025-26")])
    odds["game_date"] = pd.to_datetime(odds.date_utc).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    for c in ("home", "away"):
        odds[c] = odds[c].replace(ESPN_TO_NBA)
    odds = odds.dropna(subset=["home_ml_open", "away_ml_open"])[["game_date", "home", "away", "home_ml_open", "away_ml_open"]]
    ev = u.merge(own, on="game_id").merge(odds, on=["game_date", "home", "away"], how="inner")
    ev = ev[ev.mkt_open.notna()].sort_values("game_date").reset_index(drop=True)
    ev["dh"], ev["da"] = dec(ev.home_ml_open), dec(ev.away_ml_open)
    print(f"{len(ev)} games with opening prices")

    p = np.full(len(ev), np.nan)
    for s in ("2024-25", "2025-26"):
        past, cur = (ev.season < s).to_numpy(), (ev.season == s).to_numpy()
        yp = ev.home_win.to_numpy()[past]
        w = min(W, key=lambda w: ll(yp, blend(ev.own.to_numpy()[past], ev.mkt_open.to_numpy()[past], w)).mean())
        p[cur] = blend(ev.own.to_numpy()[cur], ev.mkt_open.to_numpy()[cur], w)
        print(f"  {s}: w(own) = {w}")
    m = ~np.isnan(p)
    e, p = ev[m].reset_index(drop=True), p[m]
    y = e.home_win.to_numpy()
    days = e.game_date.to_numpy()
    rng = np.random.default_rng(11)

    def report(name, stake_home, stake_away):
        pnl = stake_home * np.where(y == 1, e.dh - 1, -1) + stake_away * np.where(y == 0, e.da - 1, -1)
        n = int((stake_home + stake_away).sum())
        if n == 0:
            print(f"  {name}: no bets"); return
        uniq, inv = np.unique(days, return_inverse=True)
        sp, sn = np.bincount(inv, pnl), np.bincount(inv, stake_home + stake_away)
        boots = []
        for _ in range(2000):
            k = rng.integers(0, len(uniq), len(uniq))
            boots.append(sp[k].sum() / max(sn[k].sum(), 1))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {name}: {n} bets ({n / (len(np.unique(days)) / 7):.1f}/week), ROI {100 * pnl.sum() / n:+.1f}% "
              f"[95% CI {100 * lo:+.1f}%, {100 * hi:+.1f}%], hit {100 * ((stake_home * (y == 1) + stake_away * (y == 0)).sum() / n):.1f}%")

    print(f"\n{len(e)} walk-forward games (2024-25, 2025-26), opening prices of the main book (vig included)")
    fav_home = (e.mkt_open.to_numpy() >= .5).astype(float)
    report("always the opening favourite", fav_home, 1 - fav_home)
    for thr in (0.0, 0.02, 0.04, 0.06):
        evh, eva = p * e.dh - 1, (1 - p) * e.da - 1
        report(f"blend value > {thr:.0%}", ((evh > thr) & (evh >= eva)).astype(float), ((eva > thr) & (eva > evh)).astype(float))
    for thr in (0.0, 0.04):
        po = e.own.to_numpy()
        evh, eva = po * e.dh - 1, (1 - po) * e.da - 1
        report(f"own model alone, value > {thr:.0%}", ((evh > thr) & (evh >= eva)).astype(float), ((eva > thr) & (eva > evh)).astype(float))


if __name__ == "__main__":
    main()

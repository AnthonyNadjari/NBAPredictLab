"""H8 (first look): is the over/under (total points) market beatable with simple team ratings?

Model: total points ~ ridge regression on leak-free team features (ortg/drtg/pace EWMAs, rest,
back-to-back, season month), walk-forward by season. Compared with the ESPN main-book total line
(posted pre-game; ESPN stores the last pre-game value). Bet over/under when the model differs from
the line by >= k points; a standard -110 bet needs 52.4% to break even.

    python research/h8_totals/run.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DATA = Path(__file__).resolve().parents[1] / "data"
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX"}
FEATS = ["ortg_ewm_h", "drtg_ewm_h", "pace_ewm_h", "ortg_ewm_a", "drtg_ewm_a", "pace_ewm_a",
         "rest_h", "rest_a", "b2b_h", "b2b_a", "month"]


def main():
    g = pd.read_csv(DATA / "games_features_odds.csv")
    g["total_pts"] = g.PTS_h + g.PTS_a
    g["month"] = pd.to_datetime(g.date).dt.month.map(lambda m: m if m >= 10 else m + 12)
    odds = pd.concat([pd.read_csv(DATA / f"espn_{s}.csv") for s in ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")])
    odds["date"] = pd.to_datetime(odds.date_utc).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    for c in ("home", "away"):
        odds[c] = odds[c].replace(ESPN_TO_NBA)
    odds = odds.dropna(subset=["total"])[["date", "home", "away", "total"]].drop_duplicates(["date", "home", "away"])
    e = g.merge(odds, on=["date", "home", "away"], how="inner").dropna(subset=FEATS)
    e = e[e.season_type == "Regular Season"]
    print(f"{len(e)} regular-season games with a total line")
    rows = []
    for s in ["2022-23", "2023-24", "2024-25", "2025-26"]:
        tr, te = g.dropna(subset=FEATS)[lambda d: (d.season < s) & (d.season >= "2018-19") & (d.season_type == "Regular Season")], e[e.season == s]
        m = Ridge(alpha=1.0).fit(tr[FEATS], tr.total_pts)
        rows.append(te.assign(pred=m.predict(te[FEATS])))
    t = pd.concat(rows)
    print(f"mean abs error: model {np.abs(t.pred - t.total_pts).mean():.2f} pts, line {np.abs(t.total - t.total_pts).mean():.2f} pts")
    t = t[t.total_pts != t.total]                                        # pushes
    for k in (0, 2, 4, 6, 8):
        d = t[np.abs(t.pred - t.total) >= k]
        over = d.pred > d.total
        hit = np.where(over, d.total_pts > d.total, d.total_pts < d.total)
        n = len(d)
        se = np.sqrt(.25 / n) if n else 0
        print(f"  |model - line| >= {k}: {n:5d} bets, hit {100 * hit.mean():.1f}% (+/- {196 * se:.1f}), "
              f"ROI at -110 {100 * (hit.mean() * 1.909 - 1):+.1f}%  by season "
              + str({s: f'{100 * h.mean():.1f}%' for s, h in pd.Series(hit, index=d.season).groupby(level=0)}))


if __name__ == "__main__":
    main()

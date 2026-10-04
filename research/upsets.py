"""Where does the betting market fail? Error analysis of favourite losses.

1. Is the market calibrated (are upsets simply priced in)?
2. Segments where favourites lose more/less than priced.
3. Does any feature predict the outcome *beyond* the market price?
   (walk-forward: fit on earlier seasons, score on the next)
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine import features, history  # noqa: E402
from src.engine.teams import from_espn  # noqa: E402


def am_to_p(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load():
    o = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "research/data/espn_*.csv"))])
    o["home"], o["away"] = o.home.map(from_espn), o.away.map(from_espn)
    o["game_date"] = (pd.to_datetime(o.date_utc, utc=True).dt.tz_convert("America/New_York")
                      .dt.strftime("%Y-%m-%d"))
    for k in ("", "_open", "_close"):
        h, a = am_to_p(o[f"home_ml{k}"]), am_to_p(o[f"away_ml{k}"])
        o[f"mkt{k}"] = h / (h + a)
        o[f"vig{k}"] = h + a - 1
    # spread-implied probability (NBA: ~ normal margin, sd ~ 12.5)
    from scipy.stats import norm
    o["mkt_spread"] = norm.cdf(-pd.to_numeric(o.spread, errors="coerce") / 12.5)
    f = features.build(history.load())
    d = f.merge(o[["game_date", "home", "away", "mkt", "mkt_open", "mkt_close", "vig", "spread",
                   "mkt_spread", "total", "season_type"]].rename(columns={"season_type": "espn_type"}),
                on=["game_date", "home", "away"], how="inner")
    d = d[d.home_win.notna() & d.mkt_close.notna()].copy()
    d["p"] = d.mkt_close
    d["fav_home"] = d.p >= 0.5
    d["fav_p"] = np.where(d.fav_home, d.p, 1 - d.p)
    d["fav_won"] = np.where(d.fav_home, d.home_win, 1 - d.home_win)
    d["month"] = pd.to_datetime(d.game_date).dt.month
    d["move"] = d.mkt_close - d.mkt_open  # line movement toward home
    return d


def calibration(d):
    print("\n=== 1. Market calibration (closing line) ===")
    print(f"games {len(d)} | favourite won {d.fav_won.mean():.1%} | priced {d.fav_p.mean():.1%}"
          f" -> upsets {1 - d.fav_won.mean():.1%} vs expected {1 - d.fav_p.mean():.1%}")
    d["bucket"] = pd.cut(d.fav_p, [.5, .55, .6, .65, .7, .75, .8, .85, .9, 1])
    t = d.groupby("bucket", observed=True).agg(n=("fav_won", "size"), priced=("fav_p", "mean"),
                                                won=("fav_won", "mean"))
    t["gap"] = t.won - t.priced
    t["z"] = t.gap / np.sqrt(t.priced * (1 - t.priced) / t.n)
    print(t.round(3).to_string())


def segments(d):
    print("\n=== 2. Segments: favourite win rate vs price (z = gap / binomial sd) ===")
    fav = lambda col: np.where(d.fav_home, d[f"home_{col}"], d[f"away_{col}"])
    dog = lambda col: np.where(d.fav_home, d[f"away_{col}"], d[f"home_{col}"])
    segs = {
        "fav at home": d.fav_home, "fav on road": ~d.fav_home,
        "fav on B2B": fav("back_to_back") == 1, "dog on B2B": dog("back_to_back") == 1,
        "fav 3+ games in 5 days": fav("games_last5d") >= 3, "dog 3+ games in 5 days": dog("games_last5d") >= 3,
        "fav rested 2+ days more": (fav("rest_days") - dog("rest_days")) >= 2,
        "fav cold (lost 3+ in row)": fav("streak") <= -3, "dog hot (won 3+ in row)": dog("streak") >= 3,
        "fav hot (won 5+)": fav("streak") >= 5,
        "Elo disagrees with market": (d.elo_win_prob >= .5) != d.fav_home,
        "fav worse L10 net rating": fav("last10_net_rating") < dog("last10_net_rating"),
        "line moved toward dog 3pt+": np.where(d.fav_home, -d.move, d.move) >= .03,
        "line moved toward fav 3pt+": np.where(d.fav_home, d.move, -d.move) >= .03,
        "October-November": d.month.isin([10, 11]), "March-April (tanking)": d.month.isin([3, 4]),
        "playoffs/play-in": d.season_type != "Regular Season",
        "big fav 80%+": d.fav_p >= .8, "coin flip <55%": d.fav_p < .55,
        "high total 235+": pd.to_numeric(d.total, errors="coerce") >= 235,
        "low total <220": pd.to_numeric(d.total, errors="coerce") < 220,
        "early season (<10 games)": np.minimum(d.home_games_played, d.away_games_played) < 10,
    }
    rows = []
    for name, m in segs.items():
        s = d[np.asarray(m, bool)]
        if len(s) < 50:
            continue
        gap = s.fav_won.mean() - s.fav_p.mean()
        sd = np.sqrt((s.fav_p * (1 - s.fav_p)).sum()) / len(s)
        rows.append({"segment": name, "n": len(s), "priced": s.fav_p.mean(), "fav_won": s.fav_won.mean(),
                     "gap_pts": 100 * gap, "z": gap / sd})
    print(pd.DataFrame(rows).sort_values("z").round(3).to_string(index=False))


CANDIDATES = {
    "elo_logit_gap": lambda d: logit(d.elo_win_prob) - logit(d.p),
    "net_diff": lambda d: d.net_diff, "form_diff": lambda d: d.form_diff,
    "l10_net_diff": lambda d: (d.home_last10_net_rating - d.away_last10_net_rating).fillna(0),
    "rest_diff": lambda d: d.rest_diff, "b2b_diff": lambda d: d.b2b_diff, "g5_diff": lambda d: d.g5_diff,
    "streak_diff": lambda d: d.home_streak - d.away_streak,
    "line_move": lambda d: (logit(d.mkt_close) - logit(d.mkt_open)).fillna(0),
    "spread_vs_ml": lambda d: (logit(d.mkt_spread) - logit(d.p)).fillna(0),
    "venue_split": lambda d: (d.home_team_home_win_pct - d.away_team_road_win_pct).fillna(0),
    "fg3_l10_diff": lambda d: (d.home_last10_fg3_pct - d.away_last10_fg3_pct).fillna(0),
}


def residual_tests(d):
    print("\n=== 3. Does a feature beat the market? (walk-forward log-loss, lower = better) ===")
    seasons = sorted(d.season.unique())
    base_ll, rows = [], {}
    for k in list(CANDIDATES) + ["ALL"]:
        rows[k] = []
    for s in seasons[1:]:
        tr, te = d[d.season < s], d[d.season == s]
        y_te = te.home_win.to_numpy()
        base = te.p.to_numpy()
        base_ll.append(_ll(y_te, base))
        for k in list(CANDIDATES) + ["ALL"]:
            cols = list(CANDIDATES) if k == "ALL" else [k]
            Xtr = np.c_[logit(tr.p)] if False else np.column_stack([logit(tr.p)] + [CANDIDATES[c](tr) for c in cols])
            Xte = np.column_stack([logit(te.p)] + [CANDIDATES[c](te) for c in cols])
            m = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, tr.home_win)
            rows[k].append(_ll(y_te, m.predict_proba(Xte)[:, 1]))
    b = np.mean(base_ll)
    out = [{"feature": k, "logloss": np.mean(v), "vs_market": np.mean(v) - b} for k, v in rows.items()]
    print(f"market alone: {b:.5f}")
    print(pd.DataFrame(out).sort_values("vs_market").round(5).to_string(index=False))


def _ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def upset_profile(d):
    print("\n=== 4. Upsets explained: how surprising were they? ===")
    u = d[d.fav_won == 0]
    print(f"{len(u)} upsets. Favourite price when it lost: median {u.fav_p.median():.1%}")
    print(f"  coin flips (fav <55%): {(u.fav_p < .55).mean():.1%} of upsets")
    print(f"  fav 55-70%:            {((u.fav_p >= .55) & (u.fav_p < .7)).mean():.1%}")
    print(f"  fav 70%+:              {(u.fav_p >= .7).mean():.1%}")
    margin = np.where(u.fav_home, u.margin, -u.margin)  # fav margin (negative = lost by)
    print(f"  lost by 1-5 pts: {(margin >= -5).mean():.1%} | by 6-10: {((margin < -5) & (margin >= -10)).mean():.1%}"
          f" | by 11+: {(margin < -10).mean():.1%}")


if __name__ == "__main__":
    d = load()
    d.to_csv(ROOT / "research/data/upsets_dataset.csv", index=False)
    calibration(d)
    upset_profile(d)
    segments(d)
    residual_tests(d)

"""Two questions on the combined model (own model blended with the market):

A. If we published later, knowing exactly who plays (a perfect injury report), how good would it be?
   Own model with the player part fed the real rotation (player_impact "oracle_roster_plus_team":
   players who actually played, minutes still projected from earlier games), blended with the close.
B. Where does the combined model miss? Calibration of the published favourite by group (price band,
   close finishes, rest, back-to-back, home/away favourite, month, playoffs, disagreement with our
   model, team): a group where the favourite systematically wins more or less than announced would
   be a fixable bias.

    python research/h7_own_model/player_impact/run.py --dump <all.csv>
    python research/h7_own_model/late_info/run.py <all.csv>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H7 = Path(__file__).resolve().parents[1]
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
W = np.round(np.arange(0, 1.0001, 0.05), 2)


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(y, a, b, days, n=2000, seed=3):
    d = ll(y, a) - ll(y, b)
    u, inv = np.unique(days, return_inverse=True)
    s, c = np.bincount(inv, d), np.bincount(inv)
    rng = np.random.default_rng(seed)
    bs = [s[k].sum() / c[k].sum() for k in (rng.integers(0, len(u), len(u)) for _ in range(n))]
    return f"{d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]"


def walk_blend(e, own_col, mkt_col="mkt_close"):
    """logit blend, w(own) picked on earlier seasons only; first season uses w=0.15 (production)."""
    out = np.full(len(e), np.nan)
    for s in SEASONS:
        past, cur = (e.season < s).to_numpy(), (e.season == s).to_numpy()
        if past.any():
            yp = e.home_win.to_numpy()[past]
            w = min(W, key=lambda w: ll(yp, sig(w * lg(e[own_col][past]) + (1 - w) * lg(e[mkt_col][past]))).mean())
        else:
            w = 0.15
        out[cur] = sig(w * lg(e[own_col][cur]) + (1 - w) * lg(e[mkt_col][cur]))
    return out


def main(dump: str):
    e = pd.read_csv(dump)
    for fam, f in {"margin_r": H7 / "margin_ratings" / "preds.csv", "elo_plus": H7 / "elo_plus" / "preds.csv"}.items():
        e = e.merge(pd.read_csv(f)[["GAME_ID", "p"]].rename(columns={"p": fam}), on="GAME_ID")
    e = e[e.mkt_close.notna()].sort_values("date").reset_index(drop=True)
    y = e.home_win.to_numpy()
    e["own"] = sig((lg(e.pregame_plus_team) + lg(e.margin_r) + lg(e.elo_plus)) / 3)            # production avg3
    e["own_full_info"] = sig((lg(e.oracle_roster_plus_team) + lg(e.margin_r) + lg(e.elo_plus)) / 3)
    e["combined"] = walk_blend(e, "own")
    e["combined_full_info"] = walk_blend(e, "own_full_info")
    days = e.date.to_numpy()
    acc = lambda p: f"{100 * ((p > .5) == y).mean():.1f}%"

    print(f"A. {len(e)} games 2022-23..2025-26 with a closing price\n")
    print(f"{'predictor':34s} {'log-loss':>9s} {'accuracy':>9s}")
    for name in ["own", "own_full_info", "mkt_close", "combined", "combined_full_info", "oracle_roster_plus_team"]:
        print(f"{name:34s} {ll(y, e[name]).mean():9.4f} {acc(e[name]):>9s}")
    print("\nown_full_info - own        ", boot(y, e.own_full_info, e.own, days))
    print("own_full_info - close      ", boot(y, e.own_full_info, e.mkt_close, days))
    print("combined_full_info - close ", boot(y, e.combined_full_info, e.mkt_close, days))
    print("combined - close           ", boot(y, e.combined, e.mkt_close, days))

    # ---------------------------------------------------------------- B: where the combined favourite misses
    p = e.combined.to_numpy()
    fav_home = p >= .5
    pf = np.where(fav_home, p, 1 - p)                       # announced favourite probability
    won = np.where(fav_home, y == 1, y == 0)
    own_side = np.where(fav_home, e.own, 1 - e.own)
    margin_abs = e.margin.abs()
    fav_rest = np.where(fav_home, e.rest_diff, -e.rest_diff)
    fav_b2b = np.where(fav_home, e.b2b_diff, -e.b2b_diff)
    month = pd.to_datetime(e.date).dt.month
    groups = {
        "price 50-60%": (pf < .6), "price 60-70%": (pf >= .6) & (pf < .7), "price 70-80%": (pf >= .7) & (pf < .8),
        "price 80%+": pf >= .8,
        "favourite at home": fav_home, "favourite on the road": ~fav_home,
        "favourite more rested": fav_rest > 0, "favourite less rested": fav_rest < 0,
        "favourite on a back-to-back only": fav_b2b > 0, "underdog on a back-to-back only": fav_b2b < 0,
        "Oct-Nov": month.isin([10, 11]).to_numpy(), "Dec-Feb": month.isin([12, 1, 2]).to_numpy(),
        "Mar-Apr": month.isin([3, 4]).to_numpy(),
        "playoffs/play-in": e.season_type.ne("Regular Season").to_numpy(),
        "our model likes the favourite less (>5 pts)": (pf - own_side) > .05,
        "our model likes the favourite more (>5 pts)": (own_side - pf) > .05,
    }
    miss = ~won
    print(f"\nB. combined favourite: announced {100 * pf.mean():.1f}%, won {100 * won.mean():.1f}%; "
          f"{miss.sum()} misses, {100 * (miss & (margin_abs.to_numpy() <= 5)).sum() / miss.sum():.0f}% of them by 5 points or fewer\n")
    print(f"{'group':44s} {'games':>6s} {'announced':>9s} {'won':>6s} {'gap':>6s} {'z':>5s}")
    rows = []
    for name, m in groups.items():
        m = np.asarray(m, bool)
        n = m.sum()
        exp, obs = pf[m].mean(), won[m].mean()
        se = np.sqrt((pf[m] * (1 - pf[m])).sum()) / n
        rows.append((name, n, exp, obs, (obs - exp) / se))
        print(f"{name:44s} {n:6d} {100 * exp:8.1f}% {100 * obs:5.1f}% {100 * (obs - exp):+5.1f} {(obs - exp) / se:+5.1f}")
    # teams: favourites that keep under/over-performing their price
    fav_team = np.where(fav_home, e.home, e.away)
    t = pd.DataFrame({"team": fav_team, "pf": pf, "won": won})
    tg = t.groupby("team").agg(n=("won", "size"), exp=("pf", "mean"), obs=("won", "mean"),
                               var=("pf", lambda s: (s * (1 - s)).sum()))
    tg["z"] = (tg.obs - tg.exp) * tg.n / np.sqrt(tg["var"])
    tg = tg.sort_values("z")
    print("\nteams as favourite, most over / under their price (|z| > 2.9 needed for 30 teams x 1 test):")
    for team, r in pd.concat([tg.head(3), tg.tail(3)]).iterrows():
        print(f"  {team}: {int(r.n)} games, announced {100 * r.exp:.1f}%, won {100 * r.obs:.1f}%, z {r.z:+.1f}")
    print(f"\n{len(groups)} groups tested: |z| > 2.7 needed to count as real (Bonferroni, 5%).")


if __name__ == "__main__":
    main(sys.argv[1])

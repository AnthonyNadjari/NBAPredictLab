"""One pick per game day: which selection rule gives the best hit rate (and at what price)?

Uses the combined model (own avg3 blended walk-forward with the close, as in run.py). For each
rule, one game per day: hit rate, average decimal price of the pick (fair close price,
1 / p_close), and flat-stake ROI at the fair close (an upper bound: real books take ~4%).

    python research/h7_own_model/late_info/pick_one.py <all.csv>
"""
import sys

import numpy as np
import pandas as pd

from run import H7, lg, sig, walk_blend


def main(dump):
    e = pd.read_csv(dump)
    for fam, f in {"margin_r": H7 / "margin_ratings" / "preds.csv", "elo_plus": H7 / "elo_plus" / "preds.csv"}.items():
        e = e.merge(pd.read_csv(f)[["GAME_ID", "p"]].rename(columns={"p": fam}), on="GAME_ID")
    e = e[e.mkt_close.notna()].sort_values("date").reset_index(drop=True)
    e["own"] = sig((lg(e.pregame_plus_team) + lg(e.margin_r) + lg(e.elo_plus)) / 3)
    e["comb"] = walk_blend(e, "own")
    fh = e.comb >= .5
    e["pf"] = np.where(fh, e.comb, 1 - e.comb)                       # our favourite's probability
    e["won"] = np.where(fh, e.home_win == 1, e.home_win == 0)
    e["pc"] = np.where(fh, e.mkt_close, 1 - e.mkt_close)              # market's price on the same side
    e["po"] = np.where(fh, e.own, 1 - e.own)                         # our model alone, same side
    e["agree"] = (e.po >= .5) & (e.pc >= .5)
    e["price"] = 1 / e.pc

    rules = {
        "random game of the day": lambda d: d.sample(1, random_state=0),
        "biggest favourite": lambda d: d.nlargest(1, "pf"),
        "biggest favourite, models agree": lambda d: d[d.agree].nlargest(1, "pf"),
        "biggest favourite with price >= 1.25": lambda d: d[d.price >= 1.25].nlargest(1, "pf"),
        "biggest favourite with price >= 1.40": lambda d: d[d.price >= 1.40].nlargest(1, "pf"),
        "our model most above the market (value)": lambda d: d.assign(v=d.po - d.pc).nlargest(1, "v"),
        "agree + biggest gap own-market, price >= 1.25": lambda d: d[d.agree & (d.price >= 1.25)].assign(v=d.po - d.pc).nlargest(1, "v"),
    }
    days = e.groupby("date")
    print(f"{e.date.nunique()} game days, {len(e)} games (2022-23..2025-26)\n")
    print(f"{'rule (1 pick per day)':48s} {'picks':>6s} {'hit':>7s} {'announced':>9s} {'avg price':>9s} {'ROI fair':>8s}")
    for name, rule in rules.items():
        picks = pd.concat([rule(d) for _, d in days if len(rule(d))])
        roi = (np.where(picks.won, picks.price - 1, -1)).mean()
        print(f"{name:48s} {len(picks):6d} {100 * picks.won.mean():6.1f}% {100 * picks.pf.mean():8.1f}% "
              f"{picks.price.mean():9.2f} {100 * roi:+7.1f}%")
    # by season for the leading rule
    picks = pd.concat([d.nlargest(1, "pf") for _, d in days])
    print("\nbiggest favourite, by season:", {s: f"{100 * g.won.mean():.1f}%" for s, g in picks.groupby("season")})


if __name__ == "__main__":
    main(sys.argv[1])

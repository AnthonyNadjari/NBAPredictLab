"""Can the bookmakers' price make our own model better, or our model make the price better?

Blend in logit space: logit p = w * logit(own) + (1 - w) * logit(market), w chosen walk-forward
(on earlier test seasons only, so 2022-23 is used only for fitting), for the market at the open
(the morning line) and at the close. Paired bootstrap by game day on the log-loss difference.

    python research/h7_own_model/blend/run.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

H7 = Path(__file__).resolve().parents[1]
DATA = H7.parents[0] / "data"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
W = np.round(np.arange(0, 1.0001, 0.05), 2)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def blend(own, mkt, w):
    return 1 / (1 + np.exp(-(w * logit(own) + (1 - w) * logit(mkt))))


def boot(y, a, b, days, n=2000, seed=7):
    d = ll(y, a) - ll(y, b)
    uniq, inv = np.unique(days, return_inverse=True)
    sums, counts = np.bincount(inv, d), np.bincount(inv)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        k = rng.integers(0, len(uniq), len(uniq))
        stats.append(sums[k].sum() / counts[k].sum())
    return round(float(d.mean()), 5), [round(float(np.percentile(stats, q)), 5) for q in (2.5, 97.5)]


def main():
    u = pd.read_csv(DATA / "upsets_dataset.csv", usecols=["game_id", "game_date", "season", "home_win",
                                                           "mkt_close", "mkt_open"])
    own = pd.read_csv(H7 / "ensemble" / "preds.csv")[["GAME_ID", "p"]].rename(columns={"GAME_ID": "game_id", "p": "own"})
    ev = u.merge(own, on="game_id")
    print(f"{len(ev)} games")
    for mk in ("mkt_close", "mkt_open"):
        e = ev[ev[mk].notna()].reset_index(drop=True)
        y = e.home_win.to_numpy()
        # in-sample curve (optimistic, for reading only)
        curve = {float(w): round(float(ll(y, blend(e.own.to_numpy(), e[mk].to_numpy(), w)).mean()), 5) for w in W}
        best_in = min(curve, key=curve.get)
        # walk-forward: w picked on earlier seasons only
        preds, chosen = np.full(len(e), np.nan), {}
        for s in SEASONS:
            past, cur = e.season < s, (e.season == s).to_numpy()
            if not past.any() or not cur.any():
                continue
            yp = y[past.to_numpy()]
            w = min(W, key=lambda w: ll(yp, blend(e.own[past].to_numpy(), e[mk][past].to_numpy(), w)).mean())
            chosen[s] = float(w)
            preds[cur] = blend(e.own[cur].to_numpy(), e[mk][cur].to_numpy(), w)
        m = ~np.isnan(preds)
        res_m = boot(y[m], preds[m], e[mk].to_numpy()[m], e.game_date.to_numpy()[m])
        res_o = boot(y[m], preds[m], e.own.to_numpy()[m], e.game_date.to_numpy()[m])
        acc = lambda p: round(float(((p > .5) == y[m]).mean()) * 100, 2)
        print(f"\n== {mk}: {m.sum()} walk-forward games, w(own) chosen per season {chosen}")
        print(f"   in-sample best w(own) = {best_in} (log-loss {curve[best_in]} vs market alone {curve[0.0]}, own alone {curve[1.0]})")
        print(f"   log-loss  blend {ll(y[m], preds[m]).mean():.5f} | market {ll(y[m], e[mk].to_numpy()[m]).mean():.5f} | own {ll(y[m], e.own.to_numpy()[m]).mean():.5f}")
        print(f"   accuracy  blend {acc(preds[m])}% | market {acc(e[mk].to_numpy()[m])}% | own {acc(e.own.to_numpy()[m])}%")
        print(f"   blend - market: {res_m[0]} CI {res_m[1]}   (negative = blend better)")
        print(f"   blend - own:    {res_o[0]} CI {res_o[1]}")


if __name__ == "__main__":
    main()

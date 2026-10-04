"""Where does the opening-line improvement of F_move_ridge come from?

1. Decomposition: a constant shift (the average open->close move of the training seasons, which is a
   pure "the open is biased toward one side" correction) vs the game-specific part of the prediction.
2. Timing check: games where neither team played the day before. For those, every engine feature was
   known long before the opening line was posted, so an improvement there cannot come from the open
   being stale with respect to last night's results.
3. Ridge coefficients (last fold) of the move model.

Run after run.py (reads results/predictions_open.csv). Writes results/open_diagnostics.txt.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h5data  # noqa: E402
from h5models import RidgeMove, ll_vec, logit, make_features, sigmoid  # noqa: E402
from run import Tee, boot_means  # noqa: E402

RES = HERE / "results"


def main():
    sys.stdout = Tee(RES / "open_diagnostics.txt")
    d = h5data.build()
    P = pd.read_csv(RES / "predictions_open.csv")
    has = d.p_open.notna()
    X = make_features(d, "open")
    mv = logit(d.p_close) - logit(d.p_open)
    rows, coefs = [], None
    for s in ["2024-25", "2025-26"]:
        tr = ((d.season < s) & has).to_numpy()
        te = ((d.season == s) & has).to_numpy()
        mean_mv = float(np.nanmean(mv[tr]))
        r = RidgeMove().fit(d[tr], X[tr], mv[tr])
        pred = r.predict(X[te])
        L = logit(d.p_open.to_numpy())[te]
        y = d.home_win.to_numpy(float)[te]
        b2b = (d.home_back_to_back.to_numpy()[te] == 1) | (d.away_back_to_back.to_numpy()[te] == 1)
        out = pd.DataFrame(dict(season=s, y=y, b2b=b2b, open=sigmoid(L), close=d.p_close.to_numpy()[te],
                                const_shift=sigmoid(L + mean_mv), ridge=sigmoid(L + pred),
                                ridge_no_const=sigmoid(L + pred - pred.mean()),
                                pred=pred, move=mv[te]))
        rows.append(out)
        print(f"{s}: train mean move {mean_mv:+.4f} logit | test mean move {mv[te].mean():+.4f} | "
              f"ridge alpha {r.alpha} | pred mean {pred.mean():+.4f} sd {pred.std():.4f} | "
              f"corr(pred, move) {np.corrcoef(pred, mv[te])[0, 1]:+.3f}")
        coefs = pd.Series(r.m.coef_, index=X.columns)
    T = pd.concat(rows, ignore_index=True)
    print("\nlog-loss delta vs the opening line (negative = better), paired bootstrap 95% CI:")
    for col in ("const_shift", "ridge", "ridge_no_const", "close"):
        for name, m in (("all games", np.ones(len(T), bool)), ("no back-to-back", ~T.b2b.to_numpy()),
                        ("a team on a back-to-back", T.b2b.to_numpy())):
            diff = ll_vec(T.y[m], T[col][m]) - ll_vec(T.y[m], T.open[m])
            bs = boot_means(diff.to_numpy() if hasattr(diff, "to_numpy") else diff, 4000, seed=5)
            per = {s: float((ll_vec(T.y[m & (T.season == s)], T[col][m & (T.season == s)]) -
                             ll_vec(T.y[m & (T.season == s)], T.open[m & (T.season == s)])).mean())
                   for s in ("2024-25", "2025-26")}
            print(f"  {col:15s} {name:26s} n={m.sum():5d} delta {diff.mean():+.5f} "
                  f"[{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]  per season "
                  + " ".join(f"{k} {v:+.5f}" for k, v in per.items()))
    print("\nhome-win rate vs opening price by season (is the open biased toward one side?):")
    print(T.groupby("season").agg(n=("y", "size"), home_won=("y", "mean"), open_p=("open", "mean"),
                                  close_p=("close", "mean"), mean_move=("move", "mean")).round(4).to_string())
    print("\nridge move model, last fold: largest standardised coefficients (logit units per SD)")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).round(4).to_string())
    T.to_csv(RES / "open_diagnostics_games.csv", index=False)


if __name__ == "__main__":
    main()

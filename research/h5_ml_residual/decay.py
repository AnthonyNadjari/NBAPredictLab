"""Did the closing line USE to leave residual structure that has since disappeared?

For every pre-game feature, z = corr(feature, home_win - p_close) * sqrt(n), separately for
2018-19..2022-23 (older price sources: Caesars 'last stored' line, DraftKings 'current' line, then the
ESPN close) and 2023-24..2025-26. Because the 150 features are strongly correlated, the share of
|z| > 1.96 is compared with a permutation null: the residuals are shuffled WITHIN each season
(500 times), which keeps the feature correlation structure and breaks only the link to outcomes.
Exploratory: this looks at all seasons, test seasons included, so it is descriptive, not a test of
a trading rule (the walk-forward results in run.py are the test).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h5data  # noqa: E402
from h5models import make_features  # noqa: E402
from run import Tee  # noqa: E402

RES = HERE / "results"
B = 500


def zscores(Xs, r):
    rs = (r - r.mean()) / r.std()
    return Xs.T @ rs / np.sqrt(len(r))


def main():
    sys.stdout = Tee(RES / "decay.txt")
    d = h5data.build()
    X = make_features(d, "close").drop(columns=["mkt_logit", "spread_missing", "line_move"])
    X = X.astype(float).fillna(X.median())
    r_all = (d.home_win - d.p_close).to_numpy()
    rng = np.random.default_rng(11)
    out, rows = {}, []
    for name, m in (("2018-23", (d.season < "2023-24").to_numpy()), ("2023-26", (d.season >= "2023-24").to_numpy())):
        Xm = X[m]
        keep = Xm.std() > 0
        Xm = Xm.loc[:, keep]
        Xs = ((Xm - Xm.mean()) / Xm.std()).to_numpy()
        r = r_all[m]
        z = zscores(Xs, r)
        out[name] = pd.Series(z, index=Xm.columns)
        seasons = d.season.to_numpy()[m]
        null_share, null_max = [], []
        for _ in range(B):
            rp = r.copy()
            for s in np.unique(seasons):
                idx = np.where(seasons == s)[0]
                rp[idx] = rp[rng.permutation(idx)]
            zp = zscores(Xs, rp)
            null_share.append((np.abs(zp) > 1.96).mean())
            null_max.append(np.abs(zp).max())
        share = (np.abs(z) > 1.96).mean()
        rows.append(dict(period=name, n_games=int(m.sum()), n_features=len(z), share_abs_z_gt_1_96=share,
                         null_share_median=np.median(null_share), null_share_95=np.percentile(null_share, 95),
                         p_share=float((np.array(null_share) >= share).mean()), max_abs_z=np.abs(z).max(),
                         null_max_95=np.percentile(null_max, 95),
                         p_max=float((np.array(null_max) >= np.abs(z).max()).mean())))
    T = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== share of features with |z| > 1.96 vs the within-season permutation null ===")
    print(T.round(4).to_string(index=False))
    Z = pd.DataFrame(out).sort_values("2018-23", key=abs, ascending=False)
    print("\n=== top features by |z| in 2018-23, and the same feature in 2023-26 ===")
    print(Z.head(12).round(2).to_string())
    print(f"\ncorrelation of the z-profiles across features (2018-23 vs 2023-26): "
          f"{np.corrcoef(Z.dropna()['2018-23'], Z.dropna()['2023-26'])[0, 1]:+.3f}")
    r = pd.Series(r_all, index=d.index)
    per = {}
    for c in Z.index[:3]:
        per[c] = {s: np.corrcoef(X.loc[d.season == s, c], r[d.season == s])[0, 1] for s in sorted(d.season.unique())}
    print("\n=== per-season corr(feature, home_win - p_close) of the three top features ===")
    print(pd.DataFrame(per).T.round(3).to_string())
    print("\nsource of the closing price by season:")
    print(d.groupby("season").odds_src.agg(lambda s: s.value_counts().index[0]).to_string())
    Z.to_csv(RES / "decay_features.csv")
    T.to_csv(RES / "decay_summary.csv", index=False)


if __name__ == "__main__":
    main()

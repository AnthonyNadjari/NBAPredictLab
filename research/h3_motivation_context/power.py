"""Power of the walk-forward test for the main H3 features (closing line, spec B).

Outcomes are re-simulated for every game (all seasons) as  y ~ Bernoulli(expit(logit(close) + w * x)),
where x is the raw feature (e.g. po_fav = +1 for the home favourite / -1 for the road favourite in the
postseason, 0 otherwise). The full walk-forward test is then re-run and we record how often the variant
passes (a) the plain bar: 95% CI < 0 and >= 3 of 4 seasons negative, (b) the Bonferroni bar (33 variants).

    python research/h3_motivation_context/power.py     (~3 min; writes results/power.csv)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from h3data import build  # noqa: E402
from h3eval import boot_mean, ci, game_features, ll_vec, logit, walk_forward  # noqa: E402

warnings.filterwarnings("ignore")
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
BONF = 1 - 0.05 / 33


def main(n_sim=60):
    d0 = build()
    d0 = d0[d0.mkt_close.notna()].copy()
    F = game_features(d0, "mkt_close")
    m = logit(d0.mkt_close)
    rows = []
    rng = np.random.default_rng(2026)
    for feat, ws in (("po_fav", (-0.1, -0.2, -0.3)), ("locked_diff", (-0.2, -0.4, -0.6)),
                     ("tank_diff", (-0.1, -0.2, -0.3)), ("nostakes_diff", (-0.1, -0.2, -0.3))):
        x = F[feat].to_numpy()
        for w in ws:
            p_true = expit(m + w * x)
            # effect size in probability points on the games the feature touches
            act = x != 0
            pp = 100 * np.mean(np.abs(p_true[act] - d0.mkt_close.to_numpy()[act]))
            ok95 = okb = 0
            for _ in range(n_sim):
                d = d0.copy()
                d["home_win"] = (rng.random(len(d)) < p_true).astype(float)
                pr, _, _ = walk_forward(d, "mkt_close", SEASONS, True, variants=[feat])
                te = d.loc[pr[feat].index]
                y = te.home_win.to_numpy(float)
                delta = ll_vec(y, pr[feat].to_numpy()) - ll_vec(y, te.mkt_close.to_numpy())
                bs = boot_mean(delta, 1000, seed=int(rng.integers(1e9)))
                neg = sum(delta[(te.season == S).to_numpy()].mean() < 0 for S in SEASONS)
                ok95 += (ci(bs)[1] < 0) and neg >= 3
                okb += (ci(bs, BONF)[1] < 0) and neg >= 3
            rows.append({"feature": feat, "true_w_logit": w, "n_active_games": int(act.sum()),
                         "mean_shift_pp_on_active": round(pp, 2), "power_95": ok95 / n_sim, "power_bonf": okb / n_sim})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results/power.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

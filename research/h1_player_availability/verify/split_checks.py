import sys
from pathlib import Path
import numpy as np, pandas as pd
H1 = Path(r"C:\Users\nadja\NBAPredictLab\research\h1_player_availability")
sys.path.insert(0, str(H1)); sys.argv=[sys.argv[0]]
import run
g = run.load()
g["month"] = pd.to_datetime(g.game_date).dt.month
g["phase"] = np.select([g.season_type != "Regular Season", g.month.isin([10, 11, 12]), g.month.isin([1, 2])],
                       ["post", "Oct-Dec", "Jan-Feb"], "Mar-Apr")
f = "d_miss_min_prev"
for base in ("mkt_close", "mkt_open"):
    for seasons in (["2022-23", "2023-24"], ["2024-25", "2025-26"]):
        d = g[g.season.isin(seasons) & g[base].notna()]
        if not len(d): continue
        row = []
        for ph in ("Oct-Dec", "Jan-Feb", "Mar-Apr", "post"):
            x = d[d.phase == ph]
            mu, sd = d[f].mean(), d[f].std()
            z = ((x[f] - mu) / sd).to_numpy()
            tr = x.assign(**{f: z})
            _, c = run.fit_offset(tr, tr, base, [f]); b = c[1]
            # fit_offset re-standardizes; recompute se on its scale
            zz = ((tr[f] - tr[f].mean()) / tr[f].std()).to_numpy()
            p = 1 / (1 + np.exp(-(run.logit(x[base]) + b * zz)))
            se = 1 / np.sqrt(np.sum(p * (1 - p) * zz ** 2))
            row.append(f"{ph} n={len(x)} b={b:+.3f} z={b/se:+.2f}")
        print(base, seasons, " | ".join(row))
# share of absent_prev games: how big is the prev feature in recent seasons vs earlier (rest/load mgmt trend)
print(g.groupby("season")[["home_miss_min_prev", "home_miss_min_oracle"]].mean().round(2))

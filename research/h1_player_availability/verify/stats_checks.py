"""Adversarial stats checks on H1 (uses the cached h1_games.csv; no file in the repo is written)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

H1 = Path(r"C:\Users\nadja\NBAPredictLab\research\h1_player_availability")
sys.path.insert(0, str(H1))
sys.argv = [sys.argv[0]]
import run  # noqa

g = run.load()
rng = np.random.default_rng(7)


def day_boot(o, n=2000):
    diff = (o.ll_model - o.ll_mkt).to_numpy()
    days = pd.factorize(o.game_date)[0]
    nd = days.max() + 1
    s = np.bincount(days, diff, nd)
    c = np.bincount(days, None, nd)
    idx = rng.integers(0, nd, size=(n, nd))
    m = s[idx].sum(1) / c[idx].sum(1)
    return diff.mean(), np.quantile(m, .025), np.quantile(m, .975)


def ev(base, feats, tests, scheme, spec, data=None):
    data = g if data is None else data
    o, c = run.evaluate(data, base, feats, tests, scheme, spec)
    o = o.merge(data[["game_id", "game_date"]], on="game_id", how="left")
    return o, c


print("=== 1. day-clustered bootstrap, opening line, realistic + oracle_min ===")
for scheme, tests in (("wf", run.TEST_SEASONS), ("loso", ["2023-24", "2024-25", "2025-26"])):
    for spec in ("offset", "lr"):
        for name in ("prev_min", "prev_gs", "prev_oo", "oracle_min"):
            o, c = ev("mkt_open", run.PRIMARY[name], tests, scheme, spec)
            m, lo, hi = day_boot(o)
            print(f"open {scheme:4s} {spec:6s} {name:10s} delta {1e3*m:+.2f} dayCI [{1e3*lo:+.2f},{1e3*hi:+.2f}] coefs {np.round(c[:,1:].ravel(),3)}")
print("=== close, day-clustered ===")
for spec in ("offset", "lr"):
    for name in run.PRIMARY:
        o, c = ev("mkt_close", run.PRIMARY[name], run.TEST_SEASONS, "wf", spec)
        m, lo, hi = day_boot(o)
        print(f"close wf {spec:6s} {name:10s} delta {1e3*m:+.2f} dayCI [{1e3*lo:+.2f},{1e3*hi:+.2f}] coefs {np.round(c[:,1:].ravel(),3)}")

print("=== 2. permutation null for opening-line WF (feature shuffled within season) ===")
real = {}
for spec in ("offset", "lr"):
    for name in run.PRIMARY:
        o, _ = ev("mkt_open", run.PRIMARY[name], run.TEST_SEASONS, "wf", spec)
        real[(spec, name)] = (o.ll_model - o.ll_mkt).mean()
NP = 300
d0 = g[g.mkt_open.notna()].copy()
fcols = sorted({f for v in run.PRIMARY.values() for f in v})
null_rows = []
for k in range(NP):
    d = d0.copy()
    # shuffle the whole feature vector jointly (keeps correlation between variants)
    for s in d.season.unique():
        ix = d.index[d.season == s]
        perm = rng.permutation(ix)
        d.loc[ix, fcols] = d0.loc[perm, fcols].to_numpy()
    row = {}
    for spec in ("offset", "lr"):
        for name in run.PRIMARY:
            o, _ = run.evaluate(d, "mkt_open", run.PRIMARY[name], run.TEST_SEASONS, "wf", spec)
            row[(spec, name)] = (o.ll_model - o.ll_mkt).mean()
    null_rows.append(row)
null = pd.DataFrame(null_rows)
for key, v in real.items():
    p = (null[key] <= v).mean()
    print(f"{key}: real {1e3*v:+.2f}  null mean {1e3*null[key].mean():+.2f}  perm p (one-sided) {p:.3f}")
realistic = [k for k in real if k[1].startswith("prev")]
best_real = min(real[k] for k in realistic)
null_best = null[realistic].min(axis=1)
print(f"family-wise over 6 realistic variants: best real {1e3*best_real:+.2f}, P(min null <= best) = {(null_best <= best_real).mean():.3f}")
null_best12 = null.min(axis=1)
print(f"family-wise over 12 primary: best {1e3*min(real.values()):+.2f}, P = {(null_best12 <= min(real.values())).mean():.3f}")
null.to_csv(Path(__file__).with_name("perm_null_open_wf.csv"), index=False)

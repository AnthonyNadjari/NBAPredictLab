"""Shared loaders for H6 (timing): ESPN games + odds, Kalshi pre-game price path snapshots."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.engine.teams import from_espn  # noqa: E402

DATA = ROOT / "research/data/h6_timing"
KAL = DATA / "kalshi"
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)

MAX_SPREAD = 0.10  # ignore a Kalshi quote whose bid-ask spread is wider than 10 cents
HOURS = [48, 36, 24, 18, 12, 9, 6, 4, 3, 2, 1, 0.5, 0.25]  # hours before scheduled tip
CLOCK_UTC = list(range(9, 27))  # UTC hour on the (US Eastern) game date; 24-26 = 00:00-02:00 next day


def am_to_p(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def am_to_dec(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, 1 + 100 / -a, 1 + a / 100)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def espn_games():
    """All ESPN games 2021-22..2025-26 with outcome and de-vigged open/close main-book prices."""
    o = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "research/data/espn_20*.csv")))],
                  ignore_index=True)
    o["home"], o["away"] = o.home.map(from_espn), o.away.map(from_espn)
    o["tip"] = pd.to_datetime(o.date_utc, utc=True)
    o["game_date"] = o.tip.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    o = o[(o.completed == True) & o.home_score.notna()].copy()  # noqa: E712
    o["home_win"] = (pd.to_numeric(o.home_score) > pd.to_numeric(o.away_score)).astype(int)
    for k in ("_open", "_close"):
        h, a = am_to_p(o[f"home_ml{k}"]), am_to_p(o[f"away_ml{k}"])
        o[f"mkt{k}"] = h / (h + a)
        o[f"vig{k}"] = h + a - 1
    # 2021-22 has no explicit close: the post-game `current` line is the close (see H4 README)
    h, a = am_to_p(o.home_ml), am_to_p(o.away_ml)
    cur = h / (h + a)
    o["mkt_close"] = o.mkt_close.where(o.mkt_close.notna(), cur)
    o["home_ml_close"] = o.home_ml_close.where(o.home_ml_close.notna(), o.home_ml)
    o["away_ml_close"] = o.away_ml_close.where(o.away_ml_close.notna(), o.away_ml)
    bad = (o.vig_open <= 0) | (o.vig_open > 0.10)
    o.loc[bad, "mkt_open"] = np.nan
    return o


def _series(ticker, kind):
    f = KAL / "candles" / f"{ticker}_{kind}.json"
    if not f.exists():
        return None
    c = json.loads(f.read_text())
    if not c:
        return None
    t = np.array([x["t"] for x in c], dtype=np.int64)
    b = np.array([float(x["b"]) if x["b"] is not None else np.nan for x in c])
    a = np.array([float(x["a"]) if x["a"] is not None else np.nan for x in c])
    mid = (a + b) / 2
    ok = (b > 0) & (a < 1) & (a - b <= MAX_SPREAD + 1e-9) & (a >= b)
    mid[~ok] = np.nan
    return t, mid


def _mid_at(series, ts, tol):
    if series is None:
        return np.nan
    t, mid = series
    i = np.searchsorted(t, ts, side="right") - 1
    if i < 0 or ts - t[i] > tol:
        return np.nan
    return mid[i]


def _p_home(hs, as_, ts):
    """Home win prob at time ts from both team markets (average of the two when both quote)."""
    vals = []
    for s, flip in ((hs, False), (as_, True)):
        for kind, tol in (("m", 120), ("h", 3600)):
            m = _mid_at(s.get(kind), ts, tol)
            if not np.isnan(m):
                vals.append(1 - m if flip else m)
                break
    return float(np.mean(vals)) if vals else np.nan


def kalshi_snapshots(force=False):
    """One row per matched game: Kalshi home-win price at hours-before-tip and at UTC clock times."""
    out = DATA / "kalshi_snapshots.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    ev = pd.read_csv(KAL / "events_matched.csv")
    rows = []
    for g in ev.itertuples():
        tip = int(pd.Timestamp(g.tip).timestamp())
        hs = {k: _series(g.home_ticker, k) for k in ("h", "m")}
        as_ = {k: _series(g.away_ticker, k) for k in ("h", "m")}
        r = {"event_id": g.event_id, "event_ticker": g.event_ticker}
        r["k_close"] = _p_home(hs, as_, tip - 60)  # last minute before scheduled tip
        for h in HOURS:
            r[f"k_h{h:g}"] = _p_home(hs, as_, tip - int(h * 3600))
        day = pd.Timestamp(g.game_date, tz="UTC")
        for x in CLOCK_UTC:
            ts = int((day + pd.Timedelta(hours=x)).timestamp())
            r[f"k_utc{x:02d}"] = _p_home(hs, as_, ts) if ts <= tip - 60 else np.nan
        prev = int((day - pd.Timedelta(days=1) + pd.Timedelta(hours=9)).timestamp())
        r["k_prev_utc09"] = _p_home(hs, as_, prev)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def dataset(force=False):
    """Matched Kalshi x ESPN games with outcome, tip time and all price snapshots."""
    k = kalshi_snapshots(force)
    o = espn_games()
    d = o.merge(k, on="event_id", how="inner")
    d["tip_hour_utc"] = d.tip.dt.hour + d.tip.dt.minute / 60
    return d


def boot_diff(a, b, n=2000, seed=0):
    """Paired bootstrap of mean(a - b) over games -> (mean, lo95, hi95, se)."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(m.std())


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    acc = np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1)).mean()
    return {"n": len(y), "logloss": ll_vec(y, p).mean(), "brier": ((p - y) ** 2).mean(), "accuracy": acc}

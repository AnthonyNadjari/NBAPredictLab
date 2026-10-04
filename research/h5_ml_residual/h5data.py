"""H5 dataset: market prices + every leak-free pre-game engine feature, 2018-19 .. 2025-26.

Rows
  * 2022-23 .. 2025-26 (evaluation): exactly the 5,197 games of research/upsets.py load(), so the
    closing-line baseline is the protocol one. Closing odds for all, opening odds from 2023-24.
  * 2021-22 (training only): ESPN stores no explicit close; the stored "current" DraftKings line
    stands in for it (equal to the close in >99% of later games, see H2/H4).
  * 2018-19 .. 2020-21 (training only, optional): last pre-game line kept by ESPN for several books,
    downloaded by research/h3_motivation_context/fetch_older_odds.py into
    research/data/h3_motivation_context/espn_20*.csv. Book choice Caesars > Westgate > Wynn > Unibet >
    consensus, first one within 6 pp of the median of the real books (same rule as H3). If those
    files are absent the study simply trains from 2021-22.

Features: src/engine/features.build() on data/games_history.csv. Every engine feature is computed
from games strictly before the row (shift(1) / Elo state before the update). The game's own box score
columns (home_pts, home_fgm, ...) are dropped here so they can never reach a model.
"""
import glob
import json
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="divide by zero")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "research/data/h5_ml_residual"
OLD_ODDS = ROOT / "research/data/h3_motivation_context"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
OLD_PREF = ["Caesars", "Caesar's", "Westgate", "Wynn", "Unibet", "consensus"]
STATS = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov"]
# the game's own box score: outcome information, never a feature
BOX = [f"{s}_{k}" for s in ("home", "away") for k in STATS]


def am_num(a):
    s = pd.Series(a, dtype="object").astype(str).str.strip().str.upper().replace({"EVEN": "100"})
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def am_to_p(a):
    a = am_num(a)
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def am_to_dec(a):
    a = am_num(a)
    return np.where(a < 0, 1 + 100 / -a, 1 + a / 100)


def _espn_common(o):
    o = o.copy()
    o["home"], o["away"] = o.home.replace(ESPN_TO_NBA), o.away.replace(ESPN_TO_NBA)
    o["game_date"] = (pd.to_datetime(o.date_utc, utc=True).dt.tz_convert("America/New_York")
                      .dt.strftime("%Y-%m-%d"))
    return o[o.completed == True]  # noqa: E712


def espn_recent():
    """Main-book moneylines (open/close/current, vig-inclusive) for 2021-22 .. 2025-26."""
    o = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "research/data/espn_*.csv")))],
                  ignore_index=True)
    o = _espn_common(o)
    out = o[["game_date", "home", "away", "season", "neutral", "book"]].copy()
    for k, sfx in (("cur", ""), ("open", "_open"), ("close", "_close")):
        h, a = am_to_p(o[f"home_ml{sfx}"]), am_to_p(o[f"away_ml{sfx}"])
        out[f"p_{k}"] = h / (h + a)
        out[f"vig_{k}"] = h + a - 1
        out[f"dec_h_{k}"], out[f"dec_a_{k}"] = am_to_dec(o[f"home_ml{sfx}"]), am_to_dec(o[f"away_ml{sfx}"])
    out["spread_raw"] = pd.to_numeric(o.spread, errors="coerce")
    out["total_raw"] = pd.to_numeric(o.total, errors="coerce")
    # All-Star exhibitions etc. (duplicated keys, no odds) are not NBA games
    return out.drop_duplicates(["game_date", "home", "away"], keep=False)


def _p1(a):
    return -a / (-a + 100) if a < 0 else 100 / (a + 100)


def _dec1(a):
    return 1 + 100 / -a if a < 0 else 1 + a / 100


def _pick_old(raw):
    books = []
    for b in json.loads(raw) if isinstance(raw, str) else []:
        p, h, a = b.get("p"), b.get("h"), b.get("a")
        if p not in OLD_PREF or h is None or a is None:
            continue
        try:
            h, a = float(h), float(a)
        except (TypeError, ValueError):
            continue
        if abs(h) < 100 or abs(a) < 100 or (h > 0 and a > 0):
            continue
        ph, pa = _p1(h), _p1(a)
        if not (1.0 < ph + pa < 1.12):
            continue
        books.append((p, ph / (ph + pa), h, a, b.get("s"), b.get("t"), ph + pa - 1))
    if not books:
        return None
    med = float(np.median([x[1] for x in books]))
    for pref in OLD_PREF:
        for x in books:
            if x[0] == pref and abs(x[1] - med) <= 0.06:
                return {"book": x[0], "p_close": x[1], "dec_h_close": _dec1(x[2]), "dec_a_close": _dec1(x[3]),
                        "spread_raw": x[4], "total_raw": x[5], "vig_close": x[6]}
    return None


def espn_old():
    fs = sorted(glob.glob(str(OLD_ODDS / "espn_20*.csv")))
    if not fs:
        return pd.DataFrame()
    o = _espn_common(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True))
    picked = o.raw.map(_pick_old)
    keep = picked.notna()
    o = o[keep].copy()
    p = pd.DataFrame(list(picked[keep]), index=o.index)
    out = pd.concat([o[["game_date", "home", "away", "season", "neutral"]], p], axis=1)
    out = out.drop_duplicates(["game_date", "home", "away"], keep=False)
    for c in ("spread_raw", "total_raw"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build(cache=True):
    path = CACHE / "dataset.csv"
    if cache and path.exists():
        return pd.read_csv(path)
    from upsets import load  # protocol game set 2022-23..2025-26 (+ engine features)
    from src.engine import features, history

    base = load()
    recent = espn_recent()
    # --- evaluation seasons: the protocol rows, plus vig-inclusive odds for the betting view
    ev = base.merge(recent.drop(columns=["season"]), on=["game_date", "home", "away"], how="left",
                    validate="1:1")
    assert np.allclose(ev.p_close, ev.mkt_close), "close prob mismatch vs upsets.load()"
    ev["odds_src"] = "espn_close"

    # --- training-only seasons: engine features for every game, joined to older odds
    f = features.build(history.load())
    f = f[f.home_win.notna()].copy()
    r21 = recent[(recent.season == "2021-22") & recent.p_cur.notna()].copy()
    r21 = r21.assign(p_close=r21.p_cur, vig_close=r21.vig_cur, dec_h_close=r21.dec_h_cur, dec_a_close=r21.dec_a_cur,
                     odds_src="espn_current_dk")
    parts = [r21]
    old = espn_old()
    if len(old):
        parts.append(old.assign(odds_src="espn_old_" + old.book.astype(str)))
    oo = pd.concat(parts, ignore_index=True)
    tr = f[f.season.isin(sorted(oo.season.unique()))].drop(columns=["season"]).merge(
        oo, on=["game_date", "home", "away"], how="inner", validate="1:1")
    tr = tr[tr.p_close.notna()].copy()

    d = pd.concat([tr, ev], ignore_index=True, sort=False)
    d = d.drop(columns=[c for c in BOX if c in d.columns])
    # implied home margin from the (closing) spread; fall back to the ML-implied margin when missing
    from scipy.stats import norm
    d["spread"] = d.spread_raw
    d["total"] = d.total_raw
    ml_spread = -12.5 * norm.ppf(np.clip(d.p_close, 1e-4, 1 - 1e-4))
    d["spread_missing"] = d.spread.isna().astype(int)
    d["spread"] = d.spread.fillna(pd.Series(ml_spread, index=d.index))
    d["playoffs"] = (d.season_type != "Regular Season").astype(int)
    d["neutral"] = d.neutral.astype(str).str.lower().eq("true").astype(int)
    d["month"] = pd.to_datetime(d.game_date).dt.month
    d = d.sort_values(["game_date", "home"]).reset_index(drop=True)
    keep_cols = [c for c in d.columns if c not in ("p", "fav_home", "fav_p", "fav_won", "move", "mkt", "mkt_open",
                                                     "mkt_close", "vig", "mkt_spread", "espn_type", "date",
                                                     "source", "p_cur", "vig_cur", "dec_h_cur", "dec_a_cur",
                                                     "spread_raw", "total_raw", "bucket")]
    d = d[keep_cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False)
    return d


if __name__ == "__main__":
    d = build(cache=False)
    print(d.groupby("season").agg(n=("home_win", "size"), close=("p_close", "count"), open=("p_open", "count"),
                                  dec=("dec_h_close", "count"), spread=("spread_missing", lambda s: (1 - s).sum()),
                                  total=("total", "count"), src=("odds_src", lambda s: s.mode().iloc[0])))
    print(d.shape)

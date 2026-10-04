"""H3 evaluation dataset: odds + outcome + motivation/context features (home_* / away_*) + series state.

Game set
  * 2022-23 .. 2025-26: exactly the 5,197 games of research/upsets.py load() (cached CSV), so the closing
    line baseline matches (0.58162 over 2023-24..2025-26). Closing and (2023-24+) opening odds.
  * 2021-22 (training + playoff hold-out only): ESPN stores no explicit close for this season; the
    "current" DraftKings moneyline is used as a closing proxy (in later seasons it equals the close in
    99.6% of games, see research/h2_travel_schedule).
  * 2018-19 .. 2020-21 (training + playoff hold-out only): fetched by fetch_older_odds.py. ESPN keeps
    the last pre-game line of several books; we take Caesars > Westgate > Wynn > Unibet > consensus,
    keeping the first one within 6 pp (de-vigged) of the median of the real books. Projection sites
    (numberfire, teamrankings, accuscore) and the 'Caesars Sportsbook' feed (which sometimes stores
    spread juice as a moneyline) are ignored.
Vig-inclusive decimal odds are kept for the betting view.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "research/data/h3_motivation_context"
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from h3feat import build as build_context  # noqa: E402

ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS",
         "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"}
OLD_PREF = ["Caesars", "Caesar's", "Westgate", "Wynn", "Unibet", "consensus"]
CTX = ["W", "L", "gp", "rem", "win_pct", "conf_rank", "league_bottom_rank", "best_seed", "worst_seed",
       "margin_1", "margin_4", "margin_6", "margin_8", "margin_10", "elim_post", "clinch_post", "clinch_po6",
       "seed_locked", "no_stakes", "in_race", "tank", "last_week", "last_game", "days_to_rs_end"]
SER = ["gid", "game_no", "round", "home_series_w", "away_series_w", "home_lost_prev", "prev_margin_home",
       "home_elim", "away_elim", "game7", "home_down02"]


def am_to_p(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def am_to_dec(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, 1 + 100 / -a, 1 + a / 100)


def _espn_common(o):
    o["home"], o["away"] = o.home.replace(ESPN_TO_NBA), o.away.replace(ESPN_TO_NBA)
    o = o[o.home.isin(TEAMS) & o.away.isin(TEAMS)].copy()
    o["tip_utc"] = pd.to_datetime(o.date_utc, utc=True)
    o["game_date"] = o.tip_utc.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    o["espn_home_win"] = (pd.to_numeric(o.home_score, errors="coerce") >
                          pd.to_numeric(o.away_score, errors="coerce")).astype(float)
    o["espn_margin"] = pd.to_numeric(o.home_score, errors="coerce") - pd.to_numeric(o.away_score, errors="coerce")
    o["season_type"] = o.season_type.map({2: "Regular Season", 3: "Playoffs", 5: "PlayIn"})
    return o[o.completed == True]  # noqa: E712


def espn_recent():
    o = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "research/data/espn_*.csv")))],
                  ignore_index=True)
    o = _espn_common(o)
    for k in ("", "_open", "_close"):
        h, a = am_to_p(o[f"home_ml{k}"]), am_to_p(o[f"away_ml{k}"])
        o[f"mkt{k}"] = h / (h + a)
    for k in ("open", "close"):
        o[f"dec_home_{k}"], o[f"dec_away_{k}"] = am_to_dec(o[f"home_ml_{k}"]), am_to_dec(o[f"away_ml_{k}"])
    # current ("last stored") ML -> closing proxy for 2021-22
    o["dec_home_cur"], o["dec_away_cur"] = am_to_dec(o.home_ml), am_to_dec(o.away_ml)
    return o


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
        if abs(h) < 100 or abs(a) < 100 or (h > 0 and a > 0):  # not a valid American moneyline pair
            continue
        ph, pa = _p1(h), _p1(a)
        if not (1.0 < ph + pa < 1.12):
            continue
        books.append((p, ph / (ph + pa), h, a, b.get("s")))
    if not books:
        return None
    med = float(np.median([x[1] for x in books]))
    for pref in OLD_PREF:
        for x in books:
            if x[0] == pref and abs(x[1] - med) <= 0.06:
                return {"book": x[0], "mkt_close": x[1], "dec_home_close": _dec1(x[2]),
                        "dec_away_close": _dec1(x[3]), "spread": x[4], "n_real_books": len(books)}
    return None


def espn_old():
    fs = sorted(glob.glob(str(CACHE / "espn_20*.csv")))
    if not fs:
        return pd.DataFrame()
    o = _espn_common(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True))
    picked = o.raw.map(_pick_old)
    keep = picked.notna()
    o = o[keep].copy()
    p = pd.DataFrame(list(picked[keep]), index=o.index)
    return pd.concat([o.drop(columns=[c for c in p.columns if c in o.columns]), p], axis=1)


def build(cache=True):
    path = CACHE / "eval_dataset.csv"
    if cache and path.exists():
        return pd.read_csv(path)
    st, ss = build_context()
    o = espn_recent()
    base = pd.read_csv(ROOT / "research/data/upsets_dataset.csv",
                       usecols=["game_id", "game_date", "season", "season_type", "home", "away", "home_win",
                                "margin", "mkt_close", "mkt_open", "spread"])
    d = base.merge(o[["game_date", "home", "away", "tip_utc", "dec_home_close", "dec_away_close",
                      "dec_home_open", "dec_away_open"]],
                   on=["game_date", "home", "away"], how="left", validate="1:1")
    d["odds_src"] = "espn_close"
    # 2021-22: closing proxy = "current" ML
    o21 = o[(o.season == "2021-22") & o.mkt.notna()].copy()
    o21 = o21.assign(home_win=o21.espn_home_win, margin=o21.espn_margin, mkt_close=o21.mkt, mkt_open=np.nan,
                     game_id=np.nan, dec_home_close=o21.dec_home_cur, dec_away_close=o21.dec_away_cur,
                     dec_home_open=np.nan, dec_away_open=np.nan, odds_src="espn_current_dk")
    parts = [o21[d.columns], d]
    old = espn_old()
    if len(old):
        old = old.assign(home_win=old.espn_home_win, margin=old.espn_margin, mkt_open=np.nan, game_id=np.nan,
                         dec_home_open=np.nan, dec_away_open=np.nan, odds_src="espn_old_" + old.book)
        parts.insert(0, old[d.columns])
    d = pd.concat(parts, ignore_index=True)
    # context features for both sides (standings as of the morning of game_date)
    for side in ("home", "away"):
        sub = st[["season", "date", "team"] + CTX].rename(columns={c: f"{side}_{c}" for c in CTX})
        d = d.merge(sub.rename(columns={"date": "game_date", "team": side}), on=["season", "game_date", side],
                    how="left", validate="m:1")
    ss2 = ss[["season", "date", "home", "away"] + SER].rename(columns={"date": "game_date"})
    d = d.merge(ss2, on=["season", "game_date", "home", "away"], how="left", validate="m:1")
    n0 = len(d)
    d = d[d.home_W.notna() & d.away_W.notna()].copy()
    po = d.season_type != "Regular Season"
    print(f"[h3data] rows {n0} -> {len(d)} after context join; playoff/play-in rows {po.sum()}, "
          f"with series state {d.loc[po, 'game_no'].notna().sum()}")
    d = d.sort_values(["game_date", "home"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False)
    return d


if __name__ == "__main__":
    d = build(cache=False)
    print(d.groupby(["season", "season_type"]).agg(n=("home_win", "size"), close=("mkt_close", "count"),
                                                   open=("mkt_open", "count"), dec=("dec_home_close", "count"),
                                                   src=("odds_src", lambda s: s.mode().iloc[0])))

"""Evaluation dataset: ESPN odds + outcome + per-team schedule features (home_* / away_*).

Game set:
  * 2022-23 .. 2025-26: exactly the 5,197 games of research/upsets.py load() (cached CSV), so
    numbers are comparable with the closing-line baseline there.
  * 2021-22 (TRAINING ONLY): ESPN has no explicit close for this season, but the "current" moneyline
    stored after the game equals the closing line in 99-100% of later games, so we use it as a
    closing-line proxy. This lets 2022-23 be a test season (fit on 2021-22).
Vig-inclusive American odds are kept for the betting view (closing; opening where present).
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from arenas import ARENAS  # noqa: E402
from h2feat import body_clock_hour, build as build_team_features  # noqa: E402

ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}

FEATS = ["travel_km", "tz_shift", "rest_days", "b2b", "b2b_travel", "three_in_four", "four_in_six",
         "games_last7", "travel_7d", "road_trip_len", "home_stand_len", "days_since_home",
         "return_from_trip", "visiting_altitude", "after_altitude", "at_home", "first_game", "home_tz"]


def am_to_p(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def am_to_dec(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, 1 + 100 / -a, 1 + a / 100)


def espn():
    o = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / "research/data/espn_*.csv")))],
                  ignore_index=True)
    o["home"], o["away"] = o.home.replace(ESPN_TO_NBA), o.away.replace(ESPN_TO_NBA)
    o = o[o.home.isin(ARENAS) & o.away.isin(ARENAS)].copy()  # drop All-Star / exhibition events
    o["tip_utc"] = pd.to_datetime(o.date_utc, utc=True)
    o["game_date"] = o.tip_utc.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    for k in ("", "_open", "_close"):
        h, a = am_to_p(o[f"home_ml{k}"]), am_to_p(o[f"away_ml{k}"])
        o[f"mkt{k}"] = h / (h + a)
    # vig-inclusive decimal odds for the betting view
    o["dec_home_close"] = am_to_dec(o.home_ml_close)
    o["dec_away_close"] = am_to_dec(o.away_ml_close)
    o["dec_home_open"] = am_to_dec(o.home_ml_open)
    o["dec_away_open"] = am_to_dec(o.away_ml_open)
    o["espn_home_win"] = (o.home_score > o.away_score).astype(float).where(o.completed == True)  # noqa: E712
    return o


def build(cache=True):
    path = ROOT / "research/data/h2_travel_schedule/eval_dataset.csv"
    if cache and path.exists():
        return pd.read_csv(path)
    o = espn()
    tf = build_team_features()
    base = pd.read_csv(ROOT / "research/data/upsets_dataset.csv",
                       usecols=["game_id", "game_date", "season", "season_type", "home", "away",
                                "home_win", "margin", "mkt_close", "mkt_open", "spread"])
    # --- 2022-23+: the canonical game set
    d = base.merge(o[["game_date", "home", "away", "tip_utc", "mkt", "dec_home_close", "dec_away_close",
                      "dec_home_open", "dec_away_open", "espn_home_win"]],
                   on=["game_date", "home", "away"], how="left", validate="1:1")
    # --- 2021-22: training-only rows with closing proxy = "current" ML
    o21 = o[(o.season == "2021-22") & (o.completed == True) & o.mkt.notna()].copy()  # noqa: E712
    o21 = o21.assign(season="2021-22", home_win=o21.espn_home_win, margin=o21.home_score - o21.away_score,
                     mkt_close=o21.mkt, game_id=np.nan,
                     season_type=o21.season_type.map({2: "Regular Season", 3: "Playoffs", 5: "PlayIn"}))
    d = pd.concat([o21[d.columns], d], ignore_index=True)
    # attach team features by (date, team) for both sides
    tf = tf.rename(columns={"date": "game_date"})
    for side in ("home", "away"):
        sub = tf[["game_date", "team"] + FEATS].rename(columns={c: f"{side}_{c}" for c in FEATS})
        d = d.merge(sub.rename(columns={"team": side}), on=["game_date", side], how="left", validate="1:1")
    n0 = len(d)
    d = d[d.home_travel_km.notna() & d.away_travel_km.notna()].copy()  # drops All-Star etc. (2021-22)
    print(f"[h2data] rows {n0} -> {len(d)} after feature join (dropped non-NBA / unmatched)")
    tip = pd.to_datetime(d.tip_utc, utc=True)
    for side in ("home", "away"):
        d[f"{side}_body_clock"] = [body_clock_hour(t, z) for t, z in zip(tip, d[f"{side}_home_tz"])]
    d = d.sort_values(["game_date", "home"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False)
    return d


if __name__ == "__main__":
    d = build(cache=False)
    print(d.groupby("season").agg(n=("home_win", "size"), close=("mkt_close", lambda s: s.notna().mean()),
                                  open=("mkt_open", lambda s: s.notna().mean()),
                                  dec_close=("dec_home_close", lambda s: s.notna().mean())))
    chk = d[d.season > "2021-22"]
    print("espn result == dataset result:", (chk.espn_home_win == chk.home_win).mean())
    print("mkt == mkt_close (22-23+):", (np.abs(chk.mkt - chk.mkt_close) < 1e-9).mean())

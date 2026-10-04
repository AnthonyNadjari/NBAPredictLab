"""Data for the control panel (docs/index.html).

docs/dashboard.json  : track record, calibration, recent results, data health
docs/pending_games.json is enriched in place with per-game details + thread preview
"""
import json
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import espn, history
from .teams import to_code

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_PATH = ROOT / "docs" / "dashboard.json"
SEASON_START = "2026-10-01"   # v2 engine track record starts here
BUCKETS = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 1.01)]


def _resolved(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""SELECT game_date, home_team, away_team, predicted_winner, predicted_home_prob,
                               confidence, actual_winner, actual_home_score, actual_away_score, correct,
                               home_odds, away_odds
                        FROM predictions WHERE correct IS NOT NULL""", conn)
    conn.close()
    df["p_pick"] = df.predicted_home_prob.where(df.predicted_home_prob >= 0.5, 1 - df.predicted_home_prob)
    return df


def _record(df: pd.DataFrame) -> Dict:
    n = int(len(df))
    ok = int(df.correct.sum()) if n else 0
    brier = float(((df.predicted_home_prob - (df.actual_winner == df.home_team)) ** 2).mean()) if n else None
    return {"n": n, "correct": ok, "accuracy": ok / n if n else None, "brier": brier,
            "expected": float(df.p_pick.mean()) if n else None}


def _buckets(df: pd.DataFrame) -> List[Dict]:
    out = []
    for lo, hi in BUCKETS:
        b = df[(df.p_pick >= lo) & (df.p_pick < hi)]
        out.append({"label": f"{int(lo * 100)}–{min(int(hi * 100), 100)}%", "n": int(len(b)),
                    "expected": float(b.p_pick.mean()) if len(b) else (lo + min(hi, 1)) / 2,
                    "actual": float(b.correct.mean()) if len(b) else None})
    return out


def _flat_bets(df: pd.DataFrame) -> Optional[Dict]:
    """1 unit on every pick at the stored odds (only rows with real odds)."""
    pick_home = df.predicted_home_prob >= 0.5
    odds = df.home_odds.where(pick_home, df.away_odds)
    d = df[odds.notna() & (odds > 1)]
    if d.empty:
        return None
    o = odds[d.index]
    pnl = (o - 1).where(d.correct == 1, -1.0)
    return {"bets": int(len(d)), "units": round(float(pnl.sum()), 2), "roi": float(pnl.mean())}


def next_slate(today: date, horizon: int = 21) -> Optional[Dict]:
    for i in range(horizon):
        d = today + timedelta(days=i)
        try:
            games = [g for g in espn.scoreboard(d) if g["state"] == "pre"]
        except RuntimeError:
            return None
        if games:
            return {"date": d.isoformat(), "games": len(games),
                    "first_tipoff_utc": min(g["start_utc"] for g in games)}
    return None


def build_dashboard(db_path: str, today: date, slate: Optional[Dict] = None) -> Dict:
    df = _resolved(db_path)
    season = df[df.game_date >= SEASON_START]
    previous = df[(df.game_date >= "2025-10-01") & (df.game_date < SEASON_START)]

    daily = (season.groupby("game_date").agg(n=("correct", "size"), correct=("correct", "sum"))
             .reset_index().sort_values("game_date"))
    recent = season.sort_values("game_date", ascending=False).head(40)

    hist = history.load()
    model_info = json.loads((ROOT / "models" / "v2_model.json").read_text())
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "us_date": today.isoformat(),
        "season": {"label": espn.season_label(today), "since": SEASON_START, **_record(season),
                   "flat_bets": _flat_bets(season), "buckets": _buckets(season),
                   "daily": [{"date": r.game_date, "n": int(r.n), "correct": int(r.correct)}
                             for r in daily.itertuples()]},
        "previous_season": {"label": "2025-26 (ancien moteur)", **_record(previous), "buckets": _buckets(previous)},
        "recent": [{
            "date": r.game_date, "home": _code(r.home_team), "away": _code(r.away_team),
            "pick": _code(r.predicted_winner), "p": round(float(r.p_pick), 3), "correct": int(r.correct),
            "score": f"{int(r.actual_away_score)}–{int(r.actual_home_score)}"
            if pd.notna(r.actual_home_score) else None,
        } for r in recent.itertuples()],
        "data": {"games_in_history": int(len(hist)), "last_game_date": str(hist.game_date.max()),
                 "model_trained_at": model_info.get("trained_at"), "model_seasons": model_info.get("seasons", [])},
        "backtest": {"seasons": "2022-23 → 2025-26", "market_accuracy": 0.686, "market_brier": 0.203,
                     "model_accuracy": 0.655, "model_brier": 0.214},
        "next_slate": slate,
    }


def _code(name: str) -> str:
    try:
        return to_code(name)
    except KeyError:
        return name


def enrich_pending(json_path: Path, db_path: str, thread_preview: bool = True,
                   today: Optional[date] = None) -> int:
    """Add tipoff, model/market split and the exact thread (texts + card data) to each game,
    plus special posts (weekly recap) under 'specials'."""
    from src.social.thread import build_thread, record_line
    data = json.loads(json_path.read_text(encoding="utf-8"))
    previous_specials = {}
    try:
        previous_specials = {s_["id"]: s_ for s_ in json.loads(json_path.read_text(encoding="utf-8")).get("specials", [])}
    except Exception:
        pass
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rec = record_line(db_path, since=SEASON_START)
    threads, done = {}, 0
    for key in ("games", "games_today", "games_tomorrow"):
        for g in data.get(key, []):
            row = conn.execute("SELECT * FROM predictions WHERE game_date = ? AND home_team = ? AND away_team = ?",
                               (g["date"], g["home_team"], g["away_team"])).fetchone()
            f = json.loads(row["features_json"]) if row and row["features_json"] else {}
            g["home_code"], g["away_code"] = _code(g["home_team"]), _code(g["away_team"])
            g["start_utc"] = f.get("start_utc")
            g["model_home_prob"] = f.get("model_home_prob")
            g["market_home_prob"] = f.get("market_home_prob")
            g["probability_source"] = f.get("probability_source")
            g["market_books"] = f.get("market_books")
            g["key_out"] = {"home": f.get("home_key_out", []), "away": f.get("away_key_out", [])}
            if thread_preview and row is not None and g["id"] not in threads:
                try:
                    threads[g["id"]] = build_thread({**dict(row), "features": f}, rec)["tweets"]
                except Exception as e:  # preview is optional
                    log.warning("Thread preview failed for %s: %s", g["id"], e)
                    threads[g["id"]] = None
            g["thread"] = threads.get(g["id"])
            g.pop("thread_preview", None)
            done += 1
    conn.close()

    specials = []
    try:
        from src.social.weekly import build_weekly
        weekly = build_weekly(db_path, today or date.today(), SEASON_START)
        if weekly:
            prev = previous_specials.get(weekly["id"], {})
            if prev.get("published"):
                weekly.update({k: prev[k] for k in ("published", "published_at", "tweet_url") if k in prev})
            specials.append(weekly)
    except Exception as e:
        log.warning("Weekly recap failed: %s", e)
    data["specials"] = specials
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return done


def write_dashboard(db_path: str, today: date) -> None:
    slate = next_slate(today)
    DASHBOARD_PATH.write_text(json.dumps(build_dashboard(db_path, today, slate), indent=2, ensure_ascii=False),
                              encoding="utf-8")

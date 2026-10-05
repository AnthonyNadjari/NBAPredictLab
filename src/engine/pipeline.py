"""Daily v2 pipeline: refresh history, resolve results, predict a date."""
import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import espn, features, history, model, own_model, player_store
from .teams import full_name, to_code

log = logging.getLogger(__name__)

# Chart/thread keys copied from the feature row into features_json
_SIDE_KEYS = [
    "elo", "streak", "rest_days", "back_to_back", "weighted_recent_form", "form_acceleration",
    "last10_win_pct", "last10_net_rating", "last10_point_diff", "last10_ppg", "last10_fg_pct", "last10_pace",
    "last10_offensive_rating", "last10_defensive_rating", "last10_fg3_pct", "last10_opp_fg3_pct",
    "last10_ast", "last10_reb", "last10_tov", "last10_opp_ppg", "last10_three_point_rate",
    "last5_win_pct", "last5_net_rating", "last5_point_diff", "last5_ppg", "last5_fg_pct", "last5_pace",
    "last3_win_pct", "last3_net_rating", "last3_point_diff", "games_played", "season_win_pct",
    "season_diff", "games_last5d",
]
_GAME_KEYS = ["elo_diff", "elo_win_prob", "net_diff", "form_diff", "rest_diff", "b2b_diff", "g5_diff",
              "home_team_home_win_pct", "home_team_home_ppg", "home_team_home_point_diff", "home_team_home_fg_pct",
              "away_team_road_win_pct", "away_team_road_ppg", "away_team_road_point_diff", "away_team_road_fg_pct"]


def refresh_history(days_back: int = 7) -> pd.DataFrame:
    df = history.update_from_espn(history.load(), days_back=days_back)
    history.save(df)
    return history.load()


def refresh_players(days_back: int = 7) -> pd.DataFrame:
    """Add the player box scores of games finished in the last `days_back` days to
    data/player_games.csv (always written, so the file exists for the data commit)."""
    df = player_store.update_from_espn(player_store.load(), days_back=days_back)
    player_store.save(df)
    return df


def own_probabilities(games: List[Dict], hist: pd.DataFrame) -> Dict:
    """Own-model probabilities (own_model.predict) that never raise: {} if the model fails."""
    try:
        return own_model.predict(games, hist)
    except Exception as e:  # never block the published (market) probability
        log.error("Own model failed: %s", e, exc_info=True)
        return {}


def own_for_days(days: List[date], hist: pd.DataFrame) -> Dict:
    """Own-model probabilities for the not-started games of `days`, computed in one pass.
    Unfinished games of the day before serve as schedule context (back-to-backs)."""
    games = []
    for d in [min(days) - timedelta(days=1)] + sorted(days):
        try:
            games += [g for g in espn.scoreboard(d) if not g["completed"]]
        except RuntimeError as e:
            log.warning("%s", e)
    return own_probabilities(games, hist)


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def predict_date(day: date, hist: Optional[pd.DataFrame] = None, with_odds: bool = True,
                 own: Optional[Dict] = None) -> List[Dict]:
    """Predictions for games on US Eastern date `day` that have not started.

    Output dicts follow the legacy format used by _save_predictions_to_db,
    the exporter, the email and the Twitter thread.
    Published probability: the market's when there is one, else our own model's (own),
    else (own model failed) the stats logit. own: own_for_days() output, computed here if None.
    Raises RuntimeError if the schedule source is unreachable.
    """
    hist = hist if hist is not None else history.load()
    games = [g for g in espn.scoreboard(day) if g["state"] == "pre"]
    if not games:
        return []
    if own is None:
        own = own_for_days([day], hist)
    upcoming = pd.DataFrame([{
        "game_id": f"espn_{g['event_id']}", "game_date": g["game_date"], "season": g["season"],
        "season_type": g["season_type"], "home": g["home"], "away": g["away"], "source": "upcoming",
    } for g in games])
    hist = hist[~hist.set_index(["game_date", "home", "away"]).index.isin(
        upcoming.set_index(["game_date", "home", "away"]).index)]
    feats = features.build(pd.concat([hist, upcoming], ignore_index=True))
    feats = feats[feats.source == "upcoming"].set_index(["game_date", "home", "away"])
    m = model.load()

    out = []
    for g in games:
        row = feats.loc[(g["game_date"], g["home"], g["away"])]
        p_model = float(model.predict_proba(m, row.to_frame().T)[0])
        mk = espn.odds(g["event_id"]) if with_odds else None
        o = own.get((g["game_date"], g["home"], g["away"])) or {}
        p_home, source = model.final_probability(p_model, mk and mk["home_prob"], o.get("own_home_prob"))
        pick_home = p_home >= 0.5

        f = {f"{side}_{k}": _clean(row.get(f"{side}_{k}")) for side in ("home", "away") for k in _SIDE_KEYS}
        f.update({k: _clean(row.get(k)) for k in _GAME_KEYS})
        # Thread/chart code does arithmetic on these: no None (only possible for a brand-new team)
        f = {k: (0.0 if v is None else v) for k, v in f.items()}
        f.update({
            "model_home_prob": round(p_model, 4),
            "market_home_prob": round(mk["home_prob"], 4) if mk else None,
            "market_home_ml": mk["home_odds"] if mk else None,
            "market_away_ml": mk["away_odds"] if mk else None,
            "market_spread": mk["spread"] if mk else None,
            "market_books": len(mk["books"]) if mk else 0,
            "probability_source": source,
            # our own model (research H7 avg3, no odds): stored and shown next to the market
            "own_home_prob": o.get("own_home_prob"),
            "own_away_prob": o.get("own_away_prob"),
            "own_model_mode": o.get("own_model_mode", "unavailable"),
            "own_components": o.get("own_components"),
            "engine": "v2",
            "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "start_utc": g["start_utc"],
            "season_type": g["season_type"],
        })
        ctx = espn.game_context(g["event_id"]) if with_odds else None
        for side in ("home", "away"):
            c = (ctx or {}).get(side) or {}
            f[f"{side}_out"] = c.get("out", [])[:6]
            f[f"{side}_doubtful"] = c.get("doubtful", [])[:6]
            f[f"{side}_questionable"] = c.get("questionable", [])[:6]
            f[f"{side}_key_out"] = c.get("key_out", [])
            # legacy chart fields
            f[f"{side}_injured_starters"] = len(c.get("key_out", []))
            f[f"{side}_star_injured"] = int(bool(c.get("key_out")))
        out.append({
            "home_team": g["home"], "away_team": g["away"],
            "prediction": "home" if pick_home else "away",
            "predicted_winner": g["home"] if pick_home else g["away"],
            "home_win_probability": p_home, "away_win_probability": 1 - p_home,
            # confidence = probability of the picked side (calibrated), not a separate score
            "confidence": max(p_home, 1 - p_home),
            "home_odds": f["market_home_ml"], "away_odds": f["market_away_ml"],
            "features": f,
            "game_info": {"game_date": g["game_date"], "start_utc": g["start_utc"],
                          "event_id": g["event_id"], "season_type": g["season_type"]},
            "prediction_quality": _quality(max(p_home, 1 - p_home)),
        })
    return out


def _quality(conf: float) -> str:
    return "high" if conf >= 0.70 else "medium" if conf >= 0.60 else "low"


def resolve_predictions(db_path: str, hist: Optional[pd.DataFrame] = None) -> int:
    """Fill actual results for every pending prediction found in the history."""
    hist = hist if hist is not None else history.load()
    results = {(r.game_date, r.home, r.away): (int(r.home_pts), int(r.away_pts))
               for r in hist.dropna(subset=["home_pts", "away_pts"]).itertuples()}
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""SELECT id, game_date, home_team, away_team, predicted_home_prob
                           FROM predictions WHERE actual_winner IS NULL""").fetchall()
    updated = 0
    for pid, gdate, home, away, p_home in rows:
        try:
            h, a = to_code(home), to_code(away)
        except KeyError:
            continue
        res = None
        # older rows were sometimes stored with the Paris date (US date + 1)
        for d in (gdate, (datetime.strptime(gdate, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")):
            res = results.get((d, h, a))
            if res:
                break
        if not res:
            continue
        hs, as_ = res
        home_won = hs > as_
        conn.execute("""UPDATE predictions SET actual_winner = ?, actual_home_score = ?, actual_away_score = ?,
                        correct = ?, prediction_error = ?, calibration_error = ? WHERE id = ?""",
                     (home if home_won else away, hs, as_, int((p_home > 0.5) == home_won),
                      (p_home - home_won) ** 2, abs(p_home - home_won), pid))
        updated += 1
    conn.commit()
    conn.close()
    return updated


def track_record(db_path: str, since: Optional[str] = None) -> Dict:
    """Live accuracy of published predictions (resolved ones only)."""
    conn = sqlite3.connect(db_path)
    q = "SELECT COUNT(*), SUM(correct) FROM predictions WHERE correct IS NOT NULL"
    args = ()
    if since:
        q += " AND game_date >= ?"
        args = (since,)
    n, ok = conn.execute(q, args).fetchone()
    conn.close()
    return {"n": n or 0, "correct": ok or 0, "accuracy": (ok / n) if n else None}


def published_game_keys(json_path) -> set:
    """(game_date, home full name, away full name) of games whose thread is already out."""
    import json
    try:
        data = json.loads(open(json_path, encoding="utf-8").read())
    except (OSError, ValueError):
        return set()
    return {(g["date"], g["home_team"], g["away_team"]) for g in data.get("games", []) if g.get("published")}


def save_predictions(db_path: str, predictions: List[Dict], frozen: set = frozenset()) -> int:
    """Store predictions (full team names, same columns as the legacy writer).

    Pending rows for the same game are replaced; resolved rows are never touched, and
    neither are games in `frozen` (already published: the public record must grade
    the probability that was posted).
    """
    import json
    conn = sqlite3.connect(db_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved = 0
    for p in predictions:
        home, away = full_name(p["home_team"]), full_name(p["away_team"])
        gdate = p["game_info"]["game_date"]
        if (gdate, home, away) in frozen:
            continue
        conn.execute("DELETE FROM predictions WHERE game_date = ? AND home_team = ? AND away_team = ? "
                     "AND actual_winner IS NULL", (gdate, home, away))
        if conn.execute("SELECT 1 FROM predictions WHERE game_date = ? AND home_team = ? AND away_team = ?",
                        (gdate, home, away)).fetchone():
            continue  # already resolved
        conn.execute("""INSERT INTO predictions (prediction_date, game_date, home_team, away_team, predicted_winner,
                        predicted_home_prob, predicted_away_prob, confidence, features_json, home_odds, away_odds)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (now, gdate, home, away, full_name(p["predicted_winner"]), p["home_win_probability"],
                      p["away_win_probability"], p["confidence"], json.dumps(p["features"]),
                      p["home_odds"] or round(1 / max(p["home_win_probability"], 0.01), 2),
                      p["away_odds"] or round(1 / max(p["away_win_probability"], 0.01), 2)))
        saved += 1
    conn.commit()
    conn.close()
    return saved

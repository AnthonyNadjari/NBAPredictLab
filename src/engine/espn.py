"""ESPN public JSON endpoints: schedule, final scores, box scores, betting odds.

Chosen because they answer from GitHub Actions runners, where cdn.nba.com
returns 403 and stats.nba.com times out. No API key needed.
"""
import logging
import time
from datetime import date, datetime
from statistics import median
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

from .teams import from_espn

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}&limit=100"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds"

# ESPN season.type: 1 preseason, 2 regular, 3 playoffs, 5 play-in
SEASON_TYPES = {2: "Regular Season", 3: "Playoffs", 5: "PlayIn"}
PRESEASON = "Preseason"

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (compatible; NBAPredictLab/2.0)"


def _get(url: str, tries: int = 3, timeout: int = 20) -> Optional[dict]:
    for attempt in range(tries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            log.warning("ESPN %s -> HTTP %s", url, r.status_code)
        except requests.RequestException as e:
            log.warning("ESPN %s -> %s", url, e)
        time.sleep(2 * (attempt + 1))
    return None


def season_label(d: date) -> str:
    """'2026-27' for any date of that season (season rolls over in August)."""
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def scoreboard(day: date, include_preseason: bool = False) -> List[Dict]:
    """All NBA games (regular season, play-in, playoffs) on a US Eastern date.

    include_preseason=True also returns preseason games (season_type 'Preseason'): only the
    player box-score store wants them (they show who moved team over the summer).
    Raises RuntimeError if ESPN could not be reached, so callers can tell
    "no games" from "source down".
    """
    js = _get(SCOREBOARD.format(day.strftime("%Y%m%d")))
    if js is None:
        raise RuntimeError(f"ESPN scoreboard unavailable for {day}")
    types = {**SEASON_TYPES, 1: PRESEASON} if include_preseason else SEASON_TYPES
    games = []
    for ev in js.get("events", []):
        stype = (ev.get("season") or {}).get("type")
        if stype not in types:
            continue
        comp = ev["competitions"][0]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        status = comp["status"]["type"]
        start = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        games.append({
            "event_id": ev["id"],
            "game_date": start.astimezone(ET).date().isoformat(),
            "start_utc": start.isoformat(),
            "season": season_label(start.astimezone(ET).date()),
            "season_type": types[stype],
            "home": from_espn(sides["home"]["team"]["abbreviation"]),
            "away": from_espn(sides["away"]["team"]["abbreviation"]),
            "home_score": _int(sides["home"].get("score")),
            "away_score": _int(sides["away"].get("score")),
            "completed": bool(status.get("completed")),
            "state": status.get("state"),  # pre / in / post
            "status": status.get("name"),
        })
    return games


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


_STAT_KEYS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
    "offensiveRebounds": ("oreb",), "defensiveRebounds": ("dreb",), "totalRebounds": ("reb",),
    "assists": ("ast",), "steals": ("stl",), "blocks": ("blk",), "turnovers": ("tov",),
}


def boxscore(event_id: str) -> Optional[Dict[str, Dict[str, int]]]:
    """Team box score totals: {'home': {...}, 'away': {...}} or None."""
    js = _get(SUMMARY.format(event_id))
    if not js:
        return None
    out = {}
    for team in (js.get("boxscore") or {}).get("teams", []):
        stats = {}
        for s in team.get("statistics", []):
            keys = _STAT_KEYS.get(s.get("name"))
            if not keys:
                continue
            parts = str(s.get("displayValue", "")).split("-")
            if len(parts) != len(keys):
                continue
            for k, v in zip(keys, parts):
                stats[k] = _int(v)
        out[team.get("homeAway")] = stats
    if "home" not in out or "away" not in out or not out["home"].get("fga"):
        return None
    return out


def summary(event_id: str) -> Optional[dict]:
    """Raw ESPN game summary (header, box scores, injuries, ...) or None."""
    return _get(SUMMARY.format(event_id))


# Player box-score columns, by ESPN stat key (labels are the fallback for older payloads)
PLAYER_STATS = ["min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl",
                "blk", "tov", "pf", "plus_minus"]
_PLAYER_KEYS = {
    "minutes": ("min",), "points": ("pts",),
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("fg3m", "fg3a"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
    "offensiveRebounds": ("oreb",), "defensiveRebounds": ("dreb",), "rebounds": ("reb",),
    "assists": ("ast",), "steals": ("stl",), "blocks": ("blk",), "turnovers": ("tov",),
    "fouls": ("pf",), "plusMinus": ("plus_minus",),
}
_PLAYER_LABELS = {"MIN": "minutes", "PTS": "points", "FG": "fieldGoalsMade-fieldGoalsAttempted",
                  "3PT": "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
                  "FT": "freeThrowsMade-freeThrowsAttempted", "OREB": "offensiveRebounds",
                  "DREB": "defensiveRebounds", "REB": "rebounds", "AST": "assists", "STL": "steals",
                  "BLK": "blocks", "TO": "turnovers", "PF": "fouls", "+/-": "plusMinus"}
_ALL_TYPES = {**SEASON_TYPES, 1: PRESEASON}


def _signed_int(v) -> Optional[int]:
    s = str(v).strip().replace("+", "")
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def parse_player_box(js: dict) -> List[Dict]:
    """Player box score of a finished game from an ESPN summary payload.

    One dict per player who played (ESPN 'didNotPlay' players and empty lines are left out:
    no line = did not play). Keys: game_id ('espn_<event>'), game_date (US Eastern), season,
    season_type, team, opp, home (1/0), player_id (ESPN athlete id), player, starter (1/0)
    and PLAYER_STATS (minutes, box stats, +/-). Returns [] when the payload has no player lines.
    """
    header = js.get("header") or {}
    comp = (header.get("competitions") or [{}])[0]
    if not comp.get("date") or not comp.get("competitors"):
        return []
    start = datetime.fromisoformat(comp["date"].replace("Z", "+00:00")).astimezone(ET).date()
    stype = ((header.get("season") or {}).get("type"))
    side_of, code_of = {}, {}
    for c in comp["competitors"]:
        tid = str((c.get("team") or {}).get("id") or c.get("id"))
        side_of[tid] = c.get("homeAway")
        code_of[tid] = from_espn((c.get("team") or {}).get("abbreviation", ""))
    rows = []
    for block in (js.get("boxscore") or {}).get("players", []):
        tid = str((block.get("team") or {}).get("id"))
        team = code_of.get(tid) or from_espn((block.get("team") or {}).get("abbreviation", ""))
        opp = next((c for t, c in code_of.items() if t != tid), None)
        for st in block.get("statistics", [])[:1]:
            keys = st.get("keys") or [_PLAYER_LABELS.get(lb, lb) for lb in st.get("labels", [])]
            for ath in st.get("athletes", []):
                vals = ath.get("stats") or []
                if ath.get("didNotPlay") or len(vals) != len(keys):
                    continue
                row = {"game_id": f"espn_{header.get('id') or comp.get('id')}", "game_date": start.isoformat(),
                       "season": season_label(start), "season_type": _ALL_TYPES.get(stype, "Other"),
                       "team": team, "opp": opp, "home": int(side_of.get(tid) == "home"),
                       "player_id": str((ath.get("athlete") or {}).get("id")),
                       "player": (ath.get("athlete") or {}).get("displayName"),
                       "starter": int(bool(ath.get("starter")))}
                row.update({k: None for k in PLAYER_STATS})
                for key, v in zip(keys, vals):
                    cols = _PLAYER_KEYS.get(key)
                    if not cols:
                        continue
                    parts = [v] if len(cols) == 1 else str(v).split("-")
                    if len(parts) != len(cols):
                        continue
                    for c, x in zip(cols, parts):
                        row[c] = _signed_int(x)
                if row["min"] is None:
                    continue           # '--': listed but no minutes recorded
                rows.append(row)
    return rows


def player_box(event_id: str) -> Optional[List[Dict]]:
    """Player box score of a finished game (see parse_player_box); None if ESPN is unreachable."""
    js = summary(event_id)
    if js is None:
        return None
    return parse_player_box(js)


def _american_to_decimal(v) -> Optional[float]:
    try:
        a = float(v)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return 1 + (100 / -a if a < 0 else a / 100)


def odds(event_id: str) -> Optional[Dict]:
    """Consensus moneyline across books.

    Returns decimal odds (median across books) and the de-vigged home win
    probability (mean of each book's normalised implied probability).
    """
    js = _get(ODDS.format(event_id))
    if not js:
        return None
    probs, home_dec, away_dec, spreads, books = [], [], [], [], []
    for it in js.get("items", []):
        h = _american_to_decimal((it.get("homeTeamOdds") or {}).get("moneyLine"))
        a = _american_to_decimal((it.get("awayTeamOdds") or {}).get("moneyLine"))
        if not h or not a or h <= 1 or a <= 1:
            continue
        ih, ia = 1 / h, 1 / a
        probs.append(ih / (ih + ia))
        home_dec.append(h)
        away_dec.append(a)
        books.append((it.get("provider") or {}).get("name"))
        if it.get("spread") is not None:
            spreads.append(float(it["spread"]))
    if not probs:
        return None
    return {
        "home_prob": sum(probs) / len(probs),
        "home_odds": round(median(home_dec), 2),
        "away_odds": round(median(away_dec), 2),
        "spread": median(spreads) if spreads else None,
        "books": books,
    }


def game_context(event_id: str) -> Optional[Dict]:
    """Pre-game injury report and team leaders from the ESPN summary.

    {'home': {'out': [...], 'doubtful': [...], 'questionable': [...], 'leaders': [...],
              'key_out': [...]}, 'away': {...}}
    key_out = players listed Out/Doubtful who are among the team's leaders
    (points / assists / rebounds), i.e. absences that matter.
    """
    js = _get(SUMMARY.format(event_id))
    if not js:
        return None
    comp = ((js.get("header") or {}).get("competitions") or [{}])[0]
    side_of = {from_espn(c["team"]["abbreviation"]): c["homeAway"] for c in comp.get("competitors", [])}
    out = {s: {"out": [], "doubtful": [], "questionable": [], "leaders": [], "key_out": []} for s in ("home", "away")}
    for team in js.get("injuries", []):
        side = side_of.get(from_espn((team.get("team") or {}).get("abbreviation", "")))
        if not side:
            continue
        for inj in team.get("injuries", []):
            name = (inj.get("athlete") or {}).get("displayName")
            status = (inj.get("status") or "").lower()
            if not name:
                continue
            if status == "out":
                out[side]["out"].append(name)
            elif status == "doubtful":
                out[side]["doubtful"].append(name)
            elif status in ("questionable", "day-to-day", "game-time decision"):
                out[side]["questionable"].append(name)
    for team in js.get("leaders", []):
        side = side_of.get(from_espn((team.get("team") or {}).get("abbreviation", "")))
        if not side:
            continue
        for cat in team.get("leaders", []):
            for ld in cat.get("leaders", [])[:1]:
                name = (ld.get("athlete") or {}).get("displayName")
                if name and name not in out[side]["leaders"]:
                    out[side]["leaders"].append(name)
    for side in out:
        missing = set(out[side]["out"]) | set(out[side]["doubtful"])
        out[side]["key_out"] = [p for p in out[side]["leaders"] if p in missing]
    return out

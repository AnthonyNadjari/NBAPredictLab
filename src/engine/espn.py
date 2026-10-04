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


def scoreboard(day: date) -> List[Dict]:
    """All NBA games (regular season, play-in, playoffs) on a US Eastern date.

    Raises RuntimeError if ESPN could not be reached, so callers can tell
    "no games" from "source down".
    """
    js = _get(SCOREBOARD.format(day.strftime("%Y%m%d")))
    if js is None:
        raise RuntimeError(f"ESPN scoreboard unavailable for {day}")
    games = []
    for ev in js.get("events", []):
        stype = (ev.get("season") or {}).get("type")
        if stype not in SEASON_TYPES:
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
            "season_type": SEASON_TYPES[stype],
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

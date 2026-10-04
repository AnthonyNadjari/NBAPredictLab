"""
Verified NBA facts for a tweet, from this repo's own data.

The reply bot runs on a checkout of the predictor repo, so it can ground its
replies in data/games_history.csv (results), docs/pending_games.json (tonight's
games and win probabilities) and ESPN rosters (player -> team). The LLM may use
these facts; anything else it must not assert.
"""
from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import requests

from config import LOCAL_STATE_DIR, REPO_ROOT, TZ

HISTORY = REPO_ROOT / "data" / "games_history.csv"
PENDING = REPO_ROOT / "docs" / "pending_games.json"
ROSTERS_CACHE = LOCAL_STATE_DIR / "rosters.json"
ROSTERS_TTL_H = 20
MAX_FACTS = 6

TEAMS = {
    "ATL": ("Atlanta Hawks", ["hawks", "atlanta"]), "BOS": ("Boston Celtics", ["celtics", "boston"]),
    "BKN": ("Brooklyn Nets", ["nets", "brooklyn"]), "CHA": ("Charlotte Hornets", ["hornets", "charlotte"]),
    "CHI": ("Chicago Bulls", ["bulls", "chicago"]), "CLE": ("Cleveland Cavaliers", ["cavaliers", "cavs", "cleveland"]),
    "DAL": ("Dallas Mavericks", ["mavericks", "mavs", "dallas"]), "DEN": ("Denver Nuggets", ["nuggets", "denver"]),
    "DET": ("Detroit Pistons", ["pistons", "detroit"]), "GSW": ("Golden State Warriors", ["warriors", "golden state", "dubs"]),
    "HOU": ("Houston Rockets", ["rockets", "houston"]), "IND": ("Indiana Pacers", ["pacers", "indiana"]),
    "LAC": ("LA Clippers", ["clippers", "clips"]), "LAL": ("Los Angeles Lakers", ["lakers"]),
    "MEM": ("Memphis Grizzlies", ["grizzlies", "grizz", "memphis"]), "MIA": ("Miami Heat", ["miami heat", "heat"]),
    "MIL": ("Milwaukee Bucks", ["bucks", "milwaukee"]), "MIN": ("Minnesota Timberwolves", ["timberwolves", "wolves", "minnesota"]),
    "NOP": ("New Orleans Pelicans", ["pelicans", "new orleans"]), "NYK": ("New York Knicks", ["knicks"]),
    "OKC": ("Oklahoma City Thunder", ["thunder", "okc"]), "ORL": ("Orlando Magic", ["orlando magic", "orlando"]),
    "PHI": ("Philadelphia 76ers", ["76ers", "sixers", "philly", "philadelphia"]), "PHX": ("Phoenix Suns", ["suns", "phoenix"]),
    "POR": ("Portland Trail Blazers", ["trail blazers", "blazers", "portland"]), "SAC": ("Sacramento Kings", ["sacramento kings", "sacramento"]),
    "SAS": ("San Antonio Spurs", ["spurs", "san antonio"]), "TOR": ("Toronto Raptors", ["raptors", "toronto"]),
    "UTA": ("Utah Jazz", ["utah jazz", "utah"]), "WAS": ("Washington Wizards", ["wizards", "washington"]),
}
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
# Well-known short names that are unambiguous in NBA talk
NICKNAMES = {"lebron": "LeBron James", "giannis": "Giannis Antetokounmpo", "wemby": "Victor Wembanyama",
             "sga": "Shai Gilgeous-Alexander", "jokic": "Nikola Jokic", "luka": "Luka Doncic",
             "steph": "Stephen Curry", "kd": "Kevin Durant", "embiid": "Joel Embiid", "ja": "Ja Morant"}
# Last names that are also common words / shared by many players: never matched alone
AMBIGUOUS_LAST = {"james", "brown", "green", "white", "young", "johnson", "williams", "smith", "jones",
                  "davis", "allen", "harris", "jackson", "thompson", "walker", "porter", "murray", "holiday",
                  "love", "rose", "wall", "hill", "price", "bridges", "mitchell", "anderson", "martin"}


def _now():
    return datetime.now(TZ)


# ---------------------------------------------------------------- data loading
_cache: dict = {}


def _history() -> list[dict]:
    if "history" not in _cache:
        rows = []
        try:
            with open(HISTORY, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("home_pts") and r.get("away_pts"):
                        rows.append(r)
        except OSError:
            pass
        _cache["history"] = rows
    return _cache["history"]


def _pending() -> list[dict]:
    if "pending" not in _cache:
        try:
            _cache["pending"] = json.loads(PENDING.read_text(encoding="utf-8")).get("games", [])
        except Exception:
            _cache["pending"] = []
    return _cache["pending"]


def _rosters() -> dict:
    """{normalized full name: (display name, team code)} refreshed from ESPN at most every 20 h."""
    if "rosters" in _cache:
        return _cache["rosters"]
    data = {}
    try:
        cached = json.loads(ROSTERS_CACHE.read_text(encoding="utf-8"))
        if time.time() - cached.get("fetched", 0) < ROSTERS_TTL_H * 3600:
            data = cached["players"]
    except Exception:
        pass
    if not data:
        try:
            teams = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams",
                                 timeout=15).json()["sports"][0]["leagues"][0]["teams"]
            for t in teams:
                tid, abbr = t["team"]["id"], ESPN_TO_NBA.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
                r = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/roster",
                                 timeout=15).json()
                for a in r.get("athletes", []):
                    name = a.get("displayName") or a.get("fullName")
                    if name:
                        data[_norm(name)] = [name, abbr]
            ROSTERS_CACHE.write_text(json.dumps({"fetched": time.time(), "players": data}), encoding="utf-8")
        except Exception as e:
            print(f"    Context: rosters unavailable ({e})", flush=True)
    _cache["rosters"] = data
    return data


def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).strip()


# ---------------------------------------------------------------- detection
def detect(text: str) -> tuple[set[str], dict[str, str]]:
    """Teams (codes) and players ({display name: team code}) mentioned in a tweet."""
    t = " " + _norm(text) + " "
    teams = {code for code, (_, keys) in TEAMS.items() if any(f" {k} " in t for k in keys)}
    teams |= {code for code in TEAMS if re.search(rf"\b{code}\b", text)}
    players = {}
    rosters = _rosters()
    by_last = defaultdict(list)
    for key, (name, team) in rosters.items():
        by_last[key.split()[-1]].append((name, team))
    for key, (name, team) in rosters.items():
        if f" {key} " in t:
            players[name] = team
    for nick, full in NICKNAMES.items():
        if f" {nick} " in t and _norm(full) in rosters:
            players[rosters[_norm(full)][0]] = rosters[_norm(full)][1]
    for last, cands in by_last.items():
        if len(cands) == 1 and len(last) >= 5 and last not in AMBIGUOUS_LAST and f" {last} " in t:
            players.setdefault(cands[0][0], cands[0][1])
    return teams, players


# ---------------------------------------------------------------- facts
def _team_games(code: str) -> list[dict]:
    games = [g for g in _history() if code in (g["home"], g["away"])]
    return sorted(games, key=lambda g: g["game_date"])


def _result_line(code: str, g: dict) -> tuple[bool, str]:
    home = g["home"] == code
    pts, opp_pts = (int(float(g["home_pts"])), int(float(g["away_pts"]))) if home else \
        (int(float(g["away_pts"])), int(float(g["home_pts"])))
    opp = g["away"] if home else g["home"]
    won = pts > opp_pts
    return won, f"{'W' if won else 'L'} {pts}-{opp_pts} {'vs' if home else 'at'} {opp} ({g['game_date'][5:]})"


def team_facts(code: str) -> list[str]:
    name = TEAMS[code][0]
    games = _team_games(code)
    facts = []
    if games:
        season = games[-1]["season"]
        season_games = [g for g in games if g["season"] == season and g.get("season_type") == "Regular Season"]
        recent = games[-5:]
        results = [_result_line(code, g) for g in recent]
        if season_games:
            w = sum(_result_line(code, g)[0] for g in season_games)
            facts.append(f"{name} are {w}-{len(season_games) - w} in the {season} regular season")
        facts.append(f"{name} last game: {results[-1][1]}")
        streak, last = 0, results[-1][0]
        for won, _ in reversed(results):
            if won != last:
                break
            streak += 1
        if streak >= 2:
            facts.append(f"{name} have {'won' if last else 'lost'} {streak} straight")
        last_date = datetime.strptime(games[-1]["game_date"], "%Y-%m-%d").date()
        if (_now().date() - last_date).days > 60:
            facts.append(f"(no games since {games[-1]['game_date']}: offseason data)")
    for g in _pending():
        if code in (g.get("home_code"), g.get("away_code")):
            home = g.get("home_code") == code
            opp = g.get("away_code") if home else g.get("home_code")
            p = g.get("predicted_home_prob") if home else g.get("predicted_away_prob")
            when = ""
            if g.get("start_utc"):
                et = datetime.fromisoformat(g["start_utc"]).astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
                when = et.strftime("%a %b %d %I:%M %p ET").replace(" 0", " ")
            src = "betting market" if g.get("probability_source") == "market" else "our model"
            facts.append(f"{name} next: {'vs' if home else 'at'} {opp} {when}; {src} win chance {round(100 * p)}%")
            break
    return facts


def facts_for(tweet_text: str) -> list[str]:
    """Up to MAX_FACTS verified one-line facts relevant to the tweet (may be empty)."""
    try:
        teams, players = detect(tweet_text)
    except Exception as e:
        print(f"    Context: detection failed ({e})", flush=True)
        return []
    facts = [f"{p} plays for the {TEAMS[t][0]}" for p, t in list(players.items())[:2] if t in TEAMS]
    for code in list(teams | set(players.values()))[:2]:
        if code in TEAMS:
            facts += team_facts(code)
    return facts[:MAX_FACTS]


def today_line() -> str:
    """One line on where the season stands, for the system context."""
    games = _history()
    last = max((g["game_date"] for g in games), default=None)
    upcoming = sorted({g.get("date") for g in _pending() if g.get("date")})
    d = _now().strftime("%Y-%m-%d")
    if upcoming:
        return f"Today is {d}. Next NBA games on our slate: {', '.join(upcoming[:2])}. Latest result we have: {last}."
    return f"Today is {d}. Latest NBA result we have: {last}."

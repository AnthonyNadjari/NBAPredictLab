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
    "ATL": ("Atlanta Hawks", ["atlanta hawks"]), "BOS": ("Boston Celtics", ["celtics"]),
    "BKN": ("Brooklyn Nets", ["brooklyn nets"]), "CHA": ("Charlotte Hornets", ["hornets"]),
    "CHI": ("Chicago Bulls", ["chicago bulls"]), "CLE": ("Cleveland Cavaliers", ["cavaliers", "cavs"]),
    "DAL": ("Dallas Mavericks", ["mavericks", "mavs"]), "DEN": ("Denver Nuggets", ["nuggets"]),
    "DET": ("Detroit Pistons", ["pistons"]), "GSW": ("Golden State Warriors", ["warriors", "golden state", "dubs"]),
    "HOU": ("Houston Rockets", ["houston rockets"]), "IND": ("Indiana Pacers", ["pacers"]),
    "LAC": ("LA Clippers", ["clippers", "clips"]), "LAL": ("Los Angeles Lakers", ["lakers"]),
    "MEM": ("Memphis Grizzlies", ["grizzlies", "grizz"]), "MIA": ("Miami Heat", ["miami heat"]),
    "MIL": ("Milwaukee Bucks", ["bucks"]), "MIN": ("Minnesota Timberwolves", ["timberwolves"]),
    "NOP": ("New Orleans Pelicans", ["pelicans"]), "NYK": ("New York Knicks", ["knicks"]),
    "OKC": ("Oklahoma City Thunder", ["okc thunder", "okc"]), "ORL": ("Orlando Magic", ["orlando magic"]),
    "PHI": ("Philadelphia 76ers", ["76ers", "sixers"]), "PHX": ("Phoenix Suns", ["phoenix suns"]),
    "POR": ("Portland Trail Blazers", ["trail blazers", "blazers"]), "SAC": ("Sacramento Kings", ["sacramento kings"]),
    "SAS": ("San Antonio Spurs", ["san antonio spurs"]), "TOR": ("Toronto Raptors", ["raptors"]),
    "UTA": ("Utah Jazz", ["utah jazz"]), "WAS": ("Washington Wizards", ["wizards"]),
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
                        exp = (a.get("experience") or {}).get("years")
                        data[_norm(name)] = [name, abbr, exp, a.get("age")]
            ROSTERS_CACHE.write_text(json.dumps({"fetched": time.time(), "players": data}), encoding="utf-8")
        except Exception as e:
            print(f"    Context: rosters unavailable ({e})", flush=True)
    _cache["rosters"] = data
    return data


STANDINGS_CACHE = LOCAL_STATE_DIR / "standings.json"
STANDINGS_TTL_H = 6


def _standings(end_year: int) -> dict:
    """Official W-L per team code for the season ending in `end_year` (ESPN standings, cached 6 h).
    Counting our own history gave wrong records (missing games, the NBA Cup final which does
    not count): never again."""
    key = f"standings_{end_year}"
    if key in _cache:
        return _cache[key]
    data = {}
    try:
        cached = json.loads(STANDINGS_CACHE.read_text(encoding="utf-8"))
        if cached.get("year") == end_year and time.time() - cached.get("fetched", 0) < STANDINGS_TTL_H * 3600:
            data = cached["teams"]
    except Exception:
        pass
    if not data:
        try:
            js = requests.get("https://site.api.espn.com/apis/v2/sports/basketball/nba/standings",
                              params={"season": end_year, "seasontype": 2}, timeout=15).json()
            for conf in js.get("children", []):
                for e in conf["standings"]["entries"]:
                    st = {x["name"]: x.get("value") for x in e["stats"]}
                    abbr = ESPN_TO_NBA.get(e["team"]["abbreviation"], e["team"]["abbreviation"])
                    data[abbr] = [int(st.get("wins") or 0), int(st.get("losses") or 0)]
            STANDINGS_CACHE.write_text(json.dumps({"year": end_year, "fetched": time.time(), "teams": data}),
                                       encoding="utf-8")
        except Exception as e:
            print(f"    Context: standings unavailable ({e})", flush=True)
    _cache[key] = data
    return data


def champion_fact() -> str | None:
    """Last NBA Finals, from our history: the two teams of the season's last playoff games."""
    po = [g for g in _history() if g.get("season_type") == "Playoffs"]
    if not po:
        return None
    season = max(g["season"] for g in po)
    po = sorted([g for g in po if g["season"] == season], key=lambda g: g["game_date"])
    pair = {po[-1]["home"], po[-1]["away"]}
    finals = []
    for g in reversed(po):
        if {g["home"], g["away"]} != pair:
            break
        finals.append(g)
    wins = {t: sum(_result_line(t, g)[0] for g in finals) for t in pair}
    champ, loser = sorted(pair, key=lambda t: -wins[t])
    return (f"{TEAMS[champ][0]} won the {season[:2]}{season[-2:]} NBA title, beating the {TEAMS[loser][0]} "
            f"{wins[champ]}-{wins[loser]} in the Finals")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).strip()


# ---------------------------------------------------------------- detection
def detect(text: str) -> tuple[set[str], dict[str, str]]:
    """Teams (codes) and players ({display name: team code}) mentioned in a tweet."""
    t = " " + _norm(text) + " "
    teams = {code for code, (_, keys) in TEAMS.items() if any(f" {k} " in t for k in keys)}
    # Tricodes like MIN, DAL, DEN, MEM, SAC are ordinary words/abbreviations: only the unambiguous ones
    teams |= {code for code in ("LAL", "LAC", "GSW", "NYK", "OKC", "PHI", "BKN") if re.search(rf"\b{code}\b", text)}
    players = {}
    rosters = _rosters()
    by_last = defaultdict(list)
    for key, (name, team, *_rest) in rosters.items():
        by_last[key.split()[-1]].append((name, team))
    for key, (name, team, *_rest) in rosters.items():
        if f" {key} " in t:
            players[name] = team
    for nick, full in NICKNAMES.items():
        if f" {nick} " in t and _norm(full) in rosters:
            players[rosters[_norm(full)][0]] = rosters[_norm(full)][1]
    # words of the full names already found ("Cooper Flagg" must not also detect Carson Cooper)
    used = {w for n in players for w in _norm(n).split()}
    for last, cands in by_last.items():
        if (len(cands) == 1 and len(last) >= 5 and last not in AMBIGUOUS_LAST and last not in used
                and f" {last} " in t):
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
        today = _now().date()
        current = f"{today.year if today.month >= 8 else today.year - 1}-{str((today.year if today.month >= 8 else today.year - 1) + 1)[2:]}"
        season = games[-1]["season"]
        # official record (ESPN standings) of the current season if it has started, else last season's
        cur_end = int(current[:4]) + 1
        rec_season, wl = current, _standings(cur_end).get(code)
        if not wl or sum(wl) == 0:
            rec_season, wl = f"{cur_end - 2}-{str(cur_end - 1)[2:]}", _standings(cur_end - 1).get(code)
        if wl and sum(wl) > 0:
            if rec_season == current:
                facts.append(f"{name} are {wl[0]}-{wl[1]} so far this season ({rec_season})")
            else:
                # Offseason / before their first game: say it is LAST season, never "this season"
                facts.append(f"{name} finished last season ({rec_season}) {wl[0]}-{wl[1]}")
        last_date = datetime.strptime(games[-1]["game_date"], "%Y-%m-%d").date()
        if (today - last_date).days <= 10:   # recent form only when it is actually recent
            results = [_result_line(code, g) for g in games[-5:]]
            facts.append(f"{name} last game: {results[-1][1]}")
            streak, last = 0, results[-1][0]
            for won, _ in reversed(results):
                if won != last:
                    break
                streak += 1
            if streak >= 2:
                facts.append(f"{name} have {'won' if last else 'lost'} {streak} straight")
    for g in _pending():
        if code in (g.get("home_code"), g.get("away_code")):
            home = g.get("home_code") == code
            opp = g.get("away_code") if home else g.get("home_code")
            p = g.get("predicted_home_prob") if home else g.get("predicted_away_prob")
            when = ""
            if g.get("start_utc"):
                et = datetime.fromisoformat(g["start_utc"]).astimezone(__import__("zoneinfo").ZoneInfo("America/New_York"))
                when = et.strftime("%a %b %d %I:%M %p ET").replace(" 0", " ")
            src = {"market": "betting market", "blend": "our model + betting market"}.get(g.get("probability_source"), "our model")
            facts.append(f"{name} next: {'vs' if home else 'at'} {opp} {when}; {src} win chance {round(100 * p)}%")
            break
    return facts


# Records and form are only relevant when the tweet is about results/standing/form.
# Whole words only: "ring" must not match "during", "rank" not "Frank", "form" not "former".
# W-L records like "49-33" (a reply may quote one at most once per session)
RECORD_RE = re.compile(r"\b\d{1,2}-\d{1,2}\b")
PERFORMANCE_RE = re.compile(
    r"\b(record|wins?|won|loss(es)?|losing|lost|beat|beats|standings?|seed(ed)?|playoffs?|streak|"
    r"contenders?|washed|ranked|odds|favou?rites?|title|champions(hip)?|over \.500|under \.500|"
    r"last season|this season)\b", re.I)


def player_fact(name: str, team: str) -> str:
    """'Cooper Flagg plays for the Dallas Mavericks (2 NBA seasons of experience, age 19)': the model's
    own knowledge is older than the league (8 Oct: 'logo before a single NBA bucket' about a player in
    his second season got the account called a bot)."""
    row = _rosters().get(_norm(name)) or []
    exp, age = (row[2] if len(row) > 2 else None), (row[3] if len(row) > 3 else None)
    extra = []
    if exp == 0:
        extra.append("rookie, first NBA season")
    elif isinstance(exp, int):
        extra.append(f"{exp} NBA seasons of experience")
    if age:
        extra.append(f"age {age}")
    return f"{name} plays for the {TEAMS[team][0]}" + (f" ({', '.join(extra)})" if extra else "")


def facts_for(tweet_text: str) -> list[str]:
    """Up to MAX_FACTS verified one-line facts relevant to the tweet (may be empty)."""
    try:
        teams, players = detect(tweet_text)
    except Exception as e:
        print(f"    Context: detection failed ({e})", flush=True)
        return []
    facts = [player_fact(p, t) for p, t in list(players.items())[:2] if t in TEAMS]
    involved = [c for c in list(teams | set(players.values())) if c in TEAMS][:2]
    if PERFORMANCE_RE.search(tweet_text) or re.search(r"(?i)\b(champ|champions|title|finals|ring|swept|sweep)\b", tweet_text):
        champ = champion_fact()
        if champ and any(TEAMS[c][0] in champ for c in involved):
            facts.append(champ)
    if PERFORMANCE_RE.search(tweet_text):
        for code in involved:
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

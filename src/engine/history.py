"""Game history store: data/games_history.csv, one row per finished game.

Plain CSV (not the SQLite DB) so it diffs and merges cleanly in git.
"""
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from . import espn

log = logging.getLogger(__name__)

HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "games_history.csv"
STATS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov"]
COLUMNS = (["game_id", "game_date", "season", "season_type", "home", "away", "home_pts", "away_pts"]
           + [f"{s}_{k}" for s in ("home", "away") for k in STATS] + ["source"])


def load(path: Path = HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"game_id": str})
    return df[COLUMNS]


def save(df: pd.DataFrame, path: Path = HISTORY_PATH) -> None:
    df = df.drop_duplicates(["game_date", "home", "away"], keep="last")
    df = df.sort_values(["game_date", "home"]).reset_index(drop=True)
    df[COLUMNS].to_csv(path, index=False)


def from_nba_api_logs(logs: pd.DataFrame) -> pd.DataFrame:
    """Convert nba_api LeagueGameFinder rows (one per team) to history rows."""
    logs = logs.copy()
    logs["is_home"] = logs["MATCHUP"].str.contains(" vs. ")
    rename = {"PTS": "pts", "FGM": "fgm", "FGA": "fga", "FG3M": "fg3m", "FG3A": "fg3a", "FTM": "ftm",
              "FTA": "fta", "OREB": "oreb", "DREB": "dreb", "REB": "reb", "AST": "ast", "STL": "stl",
              "BLK": "blk", "TOV": "tov", "TEAM_ABBREVIATION": "team"}
    cols = list(rename)
    h = logs[logs.is_home][["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE"] + cols].rename(columns=rename)
    a = logs[~logs.is_home][["GAME_ID"] + cols].rename(columns=rename)
    g = h.merge(a, on="GAME_ID", suffixes=("_h", "_a"))
    out = pd.DataFrame({
        "game_id": g.GAME_ID.astype(str), "game_date": pd.to_datetime(g.GAME_DATE).dt.strftime("%Y-%m-%d"),
        "season": g.SEASON, "season_type": g.SEASON_TYPE, "home": g.team_h, "away": g.team_a,
        "home_pts": g.pts_h, "away_pts": g.pts_a, "source": "nba_api",
    })
    for side, suf in (("home", "_h"), ("away", "_a")):
        for k in STATS:
            out[f"{side}_{k}"] = g[k + suf]
    return out[COLUMNS]


def update_from_espn(df: pd.DataFrame, days_back: int = 7, today: Optional[date] = None) -> pd.DataFrame:
    """Add finished games of the last `days_back` US dates that are missing."""
    today = today or espn_today()
    known = set(zip(df.game_date, df.home, df.away))
    new_rows = []
    for i in range(1, days_back + 1):
        day = today - timedelta(days=i)
        try:
            games = espn.scoreboard(day)
        except RuntimeError as e:
            log.warning("%s", e)
            continue
        for g in games:
            if not g["completed"] or (g["game_date"], g["home"], g["away"]) in known:
                continue
            box = espn.boxscore(g["event_id"]) or {}
            row = {"game_id": f"espn_{g['event_id']}", "game_date": g["game_date"], "season": g["season"],
                   "season_type": g["season_type"], "home": g["home"], "away": g["away"],
                   "home_pts": g["home_score"], "away_pts": g["away_score"], "source": "espn"}
            for side in ("home", "away"):
                for k in STATS:
                    row[f"{side}_{k}"] = (box.get(side) or {}).get(k)
            new_rows.append(row)
    if new_rows:
        log.info("History: +%d finished games from ESPN", len(new_rows))
        df = pd.concat([df, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True)
    return df


def espn_today() -> date:
    """Current date in US Eastern time (the NBA's calendar)."""
    from datetime import datetime
    return datetime.now(espn.ET).date()

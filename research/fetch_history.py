"""Download team game logs (regular season + playoffs) for backtesting."""
import time
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

frames = []
for season in SEASONS:
    for stype in ["Regular Season", "Playoffs", "PlayIn"]:
        df = leaguegamefinder.LeagueGameFinder(
            season_nullable=season, league_id_nullable="00",
            season_type_nullable=stype, timeout=60,
        ).get_data_frames()[0]
        df["SEASON"] = season
        df["SEASON_TYPE"] = stype
        frames.append(df)
        print(season, stype, len(df))
        time.sleep(1)
logs = pd.concat(frames, ignore_index=True)
logs.to_csv(OUT / "team_logs.csv", index=False)
print("saved", len(logs))

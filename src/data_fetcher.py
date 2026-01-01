"""
src/data_fetcher.py - REAL NBA Data Fetching
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import requests
from typing import Optional, List, Dict, Tuple

# Rate limiting for NBA API
from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelogs,
    playergamelogs,
    scoreboardv2,
    commonteamroster,
    teamdashboardbygeneralsplits,
    leaguestandings,
    commonplayerinfo,
    playerdashboardbygeneralsplits,
    leaguedashplayerstats
)
from nba_api.stats.static import teams, players

# Import player cache system
from src.player_cache import PlayerStatsCache


class NBADataFetcher:
    """
    Fetches REAL NBA data from nba_api.
    Handles rate limiting, caching, and data processing.
    """
    
    # Team ID mapping
    TEAMS = {t['abbreviation']: t['id'] for t in teams.get_teams()}
    TEAM_NAMES = {t['id']: t['full_name'] for t in teams.get_teams()}
    
    def __init__(self, db_path: str = "data/nba_predictor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        # Initialize player stats cache
        self.player_cache = PlayerStatsCache()
        
    def _init_database(self):
        """Initialize SQLite database with proper schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Games table - historical results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                game_date DATE,
                season TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                home_win INTEGER,
                point_differential INTEGER,
                home_fgm INTEGER,
                home_fga INTEGER,
                home_fg_pct REAL,
                home_fg3m INTEGER,
                home_fg3a INTEGER,
                home_fg3_pct REAL,
                home_ftm INTEGER,
                home_fta INTEGER,
                home_ft_pct REAL,
                home_oreb INTEGER,
                home_dreb INTEGER,
                home_reb INTEGER,
                home_ast INTEGER,
                home_stl INTEGER,
                home_blk INTEGER,
                home_tov INTEGER,
                away_fgm INTEGER,
                away_fga INTEGER,
                away_fg_pct REAL,
                away_fg3m INTEGER,
                away_fg3a INTEGER,
                away_fg3_pct REAL,
                away_ftm INTEGER,
                away_fta INTEGER,
                away_ft_pct REAL,
                away_oreb INTEGER,
                away_dreb INTEGER,
                away_reb INTEGER,
                away_ast INTEGER,
                away_stl INTEGER,
                away_blk INTEGER,
                away_tov INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Team game logs - detailed stats per game
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_game_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                team_id INTEGER,
                game_date DATE,
                is_home INTEGER,
                win INTEGER,
                points INTEGER,
                fgm INTEGER,
                fga INTEGER,
                fg_pct REAL,
                fg3m INTEGER,
                fg3a INTEGER,
                fg3_pct REAL,
                ftm INTEGER,
                fta INTEGER,
                ft_pct REAL,
                oreb INTEGER,
                dreb INTEGER,
                reb INTEGER,
                ast INTEGER,
                stl INTEGER,
                blk INTEGER,
                tov INTEGER,
                pf INTEGER,
                plus_minus INTEGER,
                UNIQUE(game_id, team_id)
            )
        """)
        
        # Player game logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_game_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                player_id INTEGER,
                player_name TEXT,
                team_id INTEGER,
                game_date DATE,
                minutes REAL,
                points INTEGER,
                rebounds INTEGER,
                assists INTEGER,
                steals INTEGER,
                blocks INTEGER,
                turnovers INTEGER,
                fg_pct REAL,
                fg3_pct REAL,
                ft_pct REAL,
                plus_minus INTEGER,
                UNIQUE(game_id, player_id)
            )
        """)
        
        # Elo ratings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER,
                game_date DATE,
                elo_before REAL,
                elo_after REAL,
                game_id TEXT,
                UNIQUE(team_id, game_id)
            )
        """)
        
        # Current Elo
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS current_elo (
                team_id INTEGER PRIMARY KEY,
                elo REAL,
                last_updated DATE
            )
        """)

        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date DATE,
                game_date DATE,
                home_team TEXT,
                away_team TEXT,
                predicted_winner TEXT,
                predicted_home_prob REAL,
                predicted_away_prob REAL,
                confidence REAL,
                home_odds REAL,
                away_odds REAL,
                actual_winner TEXT,
                correct INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add unique index to prevent duplicates
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_game 
            ON predictions(game_date, home_team, away_team)
        """)
        
        conn.commit()
        conn.close()
        
    def _api_call_with_retry(self, func, max_retries=3, delay=1.0):
        """Make API call with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                time.sleep(delay)  # Rate limiting - REQUIRED
                return func()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API call failed, retrying in {delay * 2}s: {e}")
                    time.sleep(delay * 2)
                else:
                    raise e
                    
    def fetch_historical_games(self, seasons: List[str] = None) -> pd.DataFrame:
        """
        Fetch all historical games for specified seasons.
        
        Args:
            seasons: List of seasons like ['2023-24', '2022-23']
        """
        if seasons is None:
            seasons = ['2024-25', '2023-24', '2022-23', '2021-22', '2020-21']
            
        all_games = []
        
        for season in seasons:
            print(f"Fetching games for season {season}...")
            
            try:
                # Fetch games
                game_finder = self._api_call_with_retry(
                    lambda s=season: leaguegamefinder.LeagueGameFinder(
                        season_nullable=s,
                        league_id_nullable='00'  # NBA
                    )
                )
                
                games_df = game_finder.get_data_frames()[0]
                
                if games_df.empty:
                    continue
                    
                # Process games
                games_df['SEASON'] = season
                all_games.append(games_df)
                
                print(f"  Found {len(games_df)} game records for {season}")
                
            except Exception as e:
                print(f"  Error fetching {season}: {e}")
                continue
                
        if not all_games:
            raise ValueError("No games fetched. Check API connection.")
            
        combined = pd.concat(all_games, ignore_index=True)
        
        # Process into game-level data (each game appears twice, once per team)
        games = self._process_game_data(combined)
        
        # Save to database
        self._save_games_to_db(games)
        
        return games
        
    def _process_game_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw game data into clean format."""
        
        # Sort by game ID and home/away
        df = df.sort_values(['GAME_ID', 'MATCHUP'])
        
        games = []
        
        # Group by game ID
        for game_id, group in df.groupby('GAME_ID'):
            if len(group) != 2:
                continue
                
            # Determine home/away by MATCHUP (contains 'vs.' for home, '@' for away)
            home_row = group[group['MATCHUP'].str.contains(' vs. ')].iloc[0] if len(group[group['MATCHUP'].str.contains(' vs. ')]) > 0 else None
            away_row = group[group['MATCHUP'].str.contains(' @ ')].iloc[0] if len(group[group['MATCHUP'].str.contains(' @ ')]) > 0 else None
            
            if home_row is None or away_row is None:
                continue
                
            games.append({
                'game_id': game_id,
                'game_date': home_row['GAME_DATE'],
                'season': home_row.get('SEASON', 'Unknown'),
                'home_team_id': home_row['TEAM_ID'],
                'away_team_id': away_row['TEAM_ID'],
                'home_team': home_row['TEAM_ABBREVIATION'],
                'away_team': away_row['TEAM_ABBREVIATION'],
                'home_score': home_row['PTS'],
                'away_score': away_row['PTS'],
                'home_win': 1 if home_row['PTS'] > away_row['PTS'] else 0,
                'point_differential': home_row['PTS'] - away_row['PTS'],
                # Home team stats
                'home_fgm': home_row.get('FGM', 0),
                'home_fga': home_row.get('FGA', 0),
                'home_fg_pct': home_row.get('FG_PCT', 0),
                'home_fg3m': home_row.get('FG3M', 0),
                'home_fg3a': home_row.get('FG3A', 0),
                'home_fg3_pct': home_row.get('FG3_PCT', 0),
                'home_ftm': home_row.get('FTM', 0),
                'home_fta': home_row.get('FTA', 0),
                'home_ft_pct': home_row.get('FT_PCT', 0),
                'home_oreb': home_row.get('OREB', 0),
                'home_dreb': home_row.get('DREB', 0),
                'home_reb': home_row.get('REB', 0),
                'home_ast': home_row.get('AST', 0),
                'home_stl': home_row.get('STL', 0),
                'home_blk': home_row.get('BLK', 0),
                'home_tov': home_row.get('TOV', 0),
                # Away team stats
                'away_fgm': away_row.get('FGM', 0),
                'away_fga': away_row.get('FGA', 0),
                'away_fg_pct': away_row.get('FG_PCT', 0),
                'away_fg3m': away_row.get('FG3M', 0),
                'away_fg3a': away_row.get('FG3A', 0),
                'away_fg3_pct': away_row.get('FG3_PCT', 0),
                'away_ftm': away_row.get('FTM', 0),
                'away_fta': away_row.get('FTA', 0),
                'away_ft_pct': away_row.get('FT_PCT', 0),
                'away_oreb': away_row.get('OREB', 0),
                'away_dreb': away_row.get('DREB', 0),
                'away_reb': away_row.get('REB', 0),
                'away_ast': away_row.get('AST', 0),
                'away_stl': away_row.get('STL', 0),
                'away_blk': away_row.get('BLK', 0),
                'away_tov': away_row.get('TOV', 0),
            })
            
        return pd.DataFrame(games)
        
    def _save_games_to_db(self, games: pd.DataFrame):
        """Save games to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        
        # Save only core columns to games table
        core_cols = ['game_id', 'game_date', 'season', 'home_team_id', 'away_team_id',
                     'home_team', 'away_team', 'home_score', 'away_score', 
                     'home_win', 'point_differential', 'home_fgm', 'home_fga', 'home_fg_pct',
                     'home_fg3m', 'home_fg3a', 'home_fg3_pct', 'home_ftm', 'home_fta', 'home_ft_pct',
                     'home_oreb', 'home_dreb', 'home_reb', 'home_ast', 'home_stl', 'home_blk', 'home_tov',
                     'away_fgm', 'away_fga', 'away_fg_pct', 'away_fg3m', 'away_fg3a', 'away_fg3_pct',
                     'away_ftm', 'away_fta', 'away_ft_pct', 'away_oreb', 'away_dreb', 'away_reb',
                     'away_ast', 'away_stl', 'away_blk', 'away_tov']
        
        # Only include columns that exist
        available_cols = [col for col in core_cols if col in games.columns]
        games[available_cols].to_sql('games', conn, if_exists='replace', index=False)
        conn.close()
        
    def get_todays_games(self) -> pd.DataFrame:
        """Fetch today's scheduled games."""
        today = datetime.now().strftime('%Y-%m-%d')
        return self.get_games_for_date(today)

    def get_games_for_date(self, date_str: str) -> pd.DataFrame:
        """
        Fetch scheduled games for a specific date.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            DataFrame with games for the specified date
        """
        try:
            # Convert YYYY-MM-DD to datetime
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Use the requested date directly (no year adjustments)
            target_date_db = date_str
            date_str_api = date_obj.strftime('%m/%d/%Y')
            
            print(f"Fetching games for {date_str} (trying API format: {date_str_api})...")

            games_header = pd.DataFrame()
            try:
                scoreboard = self._api_call_with_retry(
                    lambda: scoreboardv2.ScoreboardV2(game_date=date_str_api)
                )
                games_header = scoreboard.get_data_frames()[0]
            except Exception as e:
                print(f"Scoreboard API error for {date_str_api}: {e}")

            # Check if we have valid games with team IDs
            # Scoreboard may return games but without team IDs for future dates
            has_valid_team_ids = False
            if not games_header.empty:
                if 'HOME_TEAM_ID' in games_header.columns and 'VISITOR_TEAM_ID' in games_header.columns:
                    # Check if any row has non-null team IDs
                    has_valid_team_ids = not games_header[['HOME_TEAM_ID', 'VISITOR_TEAM_ID']].isna().all().all()
            
            # If empty or missing team IDs, fall back to static CDN schedule
            if games_header.empty or not has_valid_team_ids:
                try:
                    # Try new structure first, then fallback to old URL if needed
                    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    # Convert date to match JSON format: MM/DD/YYYY 00:00:00
                    target_date_formatted = date_str_api + " 00:00:00"
                    
                    static_games = []
                    
                    # Try new structure: leagueSchedule.gameDates
                    league_schedule = data.get("leagueSchedule", {})
                    if league_schedule:
                        game_dates = league_schedule.get("gameDates", [])
                        for date_entry in game_dates:
                            if date_entry.get("gameDate") == target_date_formatted:
                                games = date_entry.get("games", [])
                                for game in games:
                                    home_team = game.get("homeTeam", {})
                                    away_team = game.get("awayTeam", {})
                                    static_games.append({
                                        "game_id": game.get("gameId"),
                                        "game_date": target_date_db,
                                        "home_team_id": home_team.get("teamId"),
                                        "away_team_id": away_team.get("teamId"),
                                        "home_team_tricode": home_team.get("teamTricode"),
                                        "away_team_tricode": away_team.get("teamTricode"),
                                        "game_status": game.get("gameStatusText", "").strip(),
                                        "game_status_code": game.get("gameStatus"),
                                        "tipoff_et": game.get("gameTimeEst"),
                                        "arena_name": game.get("arenaName"),
                                        "arena_city": game.get("arenaCity"),
                                    })
                                break
                    else:
                        # Fallback to old structure: league.standard
                        for g in data.get("league", {}).get("standard", []):
                            if g.get("startDateEastern") == target_date_db:
                                home_code = g.get("hTeam", {}).get("triCode")
                                away_code = g.get("vTeam", {}).get("triCode")
                                home_id = self.TEAMS.get(home_code)
                                away_id = self.TEAMS.get(away_code)
                                static_games.append({
                                    "game_id": g.get("gameId"),
                                    "game_date": target_date_db,
                                    "home_team_id": home_id,
                                    "away_team_id": away_id,
                                    "home_team_tricode": home_code,
                                    "away_team_tricode": away_code,
                                    "game_status": g.get("gameStatusText", g.get("statusNum", "")),
                                    "tipoff_et": g.get("startTimeEastern", "TBD"),
                                })
                    
                    if static_games:
                        print(f"Found {len(static_games)} games via static schedule for {target_date_db}")
                        # Print game details similar to scoreboard output
                        for game in static_games:
                            home_name = self.TEAM_NAMES.get(game.get('home_team_id'), 'Unknown')
                            away_name = self.TEAM_NAMES.get(game.get('away_team_id'), 'Unknown')
                            status = game.get('game_status', 'TBD')
                            print(f"  Game: {away_name} @ {home_name} - {status}")
                        return pd.DataFrame(static_games)
                    else:
                        print(f"No games scheduled for {target_date_db} (target format: {target_date_formatted})")
                        return pd.DataFrame()
                except Exception as e:
                    print(f"Static schedule fallback failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return pd.DataFrame()

            # Use scoreboard data
            print(f"Found {len(games_header)} games for {target_date_db}")

            games = []
            for _, row in games_header.iterrows():
                game_info = {
                    'game_id': row['GAME_ID'],
                    'game_date': target_date_db,
                    'home_team_id': row['HOME_TEAM_ID'],
                    'away_team_id': row['VISITOR_TEAM_ID'],
                    'game_status': row['GAME_STATUS_TEXT'],
                }
                games.append(game_info)
                print(f"  Game: {self.TEAM_NAMES.get(row['VISITOR_TEAM_ID'])} @ {self.TEAM_NAMES.get(row['HOME_TEAM_ID'])} - {row['GAME_STATUS_TEXT']}")

            return pd.DataFrame(games)

        except Exception as e:
            print(f"Error fetching games for {date_str}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def update_recent_games(self, days_back: int = 7) -> int:
        """
        Fetch and save recent game results to database.

        Args:
            days_back: Number of days back to fetch games

        Returns:
            Number of games fetched and saved
        """
        from datetime import timedelta

        games_fetched = 0
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        print(f"Updating games from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Fetch games for each day in the range
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            try:
                games_df = self.get_games_for_date(date_str)
                if not games_df.empty:
                    # Save to database
                    conn = sqlite3.connect(self.db_path)
                    for _, game in games_df.iterrows():
                        # Only save if game has results (not scheduled)
                        if 'home_score' in game and pd.notna(game['home_score']):
                            # Check if game already exists
                            cursor = conn.cursor()
                            cursor.execute("SELECT game_id FROM games WHERE game_id = ?", (game['game_id'],))
                            exists = cursor.fetchone()

                            if exists:
                                # Update existing game
                                cursor.execute("""
                                    UPDATE games SET
                                        home_score = ?, away_score = ?, home_win = ?
                                    WHERE game_id = ?
                                """, (game['home_score'], game['away_score'],
                                     1 if game['home_score'] > game['away_score'] else 0,
                                     game['game_id']))
                            else:
                                # Insert new game
                                cursor.execute("""
                                    INSERT OR IGNORE INTO games
                                    (game_id, game_date, home_team_id, away_team_id,
                                     home_score, away_score, home_win)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (game['game_id'], game['game_date'],
                                     game['home_team_id'], game['away_team_id'],
                                     game['home_score'], game['away_score'],
                                     1 if game['home_score'] > game['away_score'] else 0))

                            games_fetched += 1

                    conn.commit()
                    conn.close()

            except Exception as e:
                print(f"Error fetching games for {date_str}: {e}")

            current_date += timedelta(days=1)

        print(f"Updated {games_fetched} games")
        return games_fetched

    def get_team_id(self, team_abbrev: str) -> Optional[int]:
        """
        Get team ID from team abbreviation.

        Args:
            team_abbrev: Team abbreviation (e.g., 'LAL', 'BOS')

        Returns:
            Team ID or None if not found
        """
        return self.TEAMS.get(team_abbrev)

    def get_team_recent_games(self, team_id: int, n_games: int = 10) -> pd.DataFrame:
        """Get a team's most recent games."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM games 
            WHERE home_team_id = ? OR away_team_id = ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(team_id, team_id, n_games))
        conn.close()

        return df

    def get_team_roster(self, team_id: int, season: str = "2024-25") -> pd.DataFrame:
        """Get current roster for a team."""
        try:
            roster = self._api_call_with_retry(
                lambda: commonteamroster.CommonTeamRoster(
                    team_id=team_id,
                    season=season
                )
            )
            return roster.get_data_frames()[0]
        except Exception as e:
            print(f"Error fetching roster for team {team_id}: {e}")
            return pd.DataFrame()

    def get_player_recent_stats(self, player_id: int, n_games: int = 10) -> Dict:
        """Get a player's recent performance stats."""
        try:
            player_stats = self._api_call_with_retry(
                lambda: playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                    player_id=player_id,
                    season="2024-25"
                )
            )

            df = player_stats.get_data_frames()[0]

            if df.empty:
                return {}

            # Get recent games
            player_games = self._api_call_with_retry(
                lambda: playergamelogs.PlayerGameLogs(
                    player_id_nullable=player_id,
                    season_nullable="2024-25"
                )
            )

            games_df = player_games.get_data_frames()[0]

            if games_df.empty or len(games_df) == 0:
                return {}

            recent = games_df.head(n_games)

            return {
                'ppg': recent['PTS'].mean() if 'PTS' in recent.columns else 0,
                'rpg': recent['REB'].mean() if 'REB' in recent.columns else 0,
                'apg': recent['AST'].mean() if 'AST' in recent.columns else 0,
                'fg_pct': recent['FG_PCT'].mean() if 'FG_PCT' in recent.columns else 0,
                'games_played': len(recent),
                'minutes': recent['MIN'].mean() if 'MIN' in recent.columns else 0
            }

        except Exception as e:
            print(f"Error fetching player stats for {player_id}: {e}")
            return {}

    def get_league_player_stats(self, season: str = "2024-25") -> Dict[int, Dict]:
        """
        Fetch stats for ALL players in the league in one request.
        Returns a dictionary mapping player_id -> stats dict.
        """
        try:
            # Check cache first (global league stats cache)
            # We use a special key for the whole league
            cache_key = f"league_stats_{season}"
            cached = self.player_cache.get_team_aggregated_stats(0, season) # Use 0 as dummy team_id
            
            # Note: The cache system is designed for team stats, but we can potentially abuse it 
            # or just use a simple class-level cache if the persistent cache is too strict.
            # For now, let's just fetch if not in memory, as this is called once per session usually.
            
            print("Fetching league-wide player stats...")
            league_stats = self._api_call_with_retry(
                lambda: leaguedashplayerstats.LeagueDashPlayerStats(
                    season=season,
                    per_mode_detailed='PerGame'
                )
            )
            
            df = league_stats.get_data_frames()[0]
            
            player_stats_map = {}
            for _, row in df.iterrows():
                pid = row['PLAYER_ID']
                stats = {
                    'ppg': row['PTS'],
                    'rpg': row['REB'],
                    'apg': row['AST'],
                    'fg_pct': row['FG_PCT'],
                    'games_played': row['GP'],
                    'minutes': row['MIN']
                }
                player_stats_map[pid] = stats
                
                # Update individual player cache too!
                # This populates the cache for any subsequent individual lookups
                self.player_cache.set_player_stats(
                    pid, row['PLAYER_NAME'], row['TEAM_ID'], stats, ttl_hours=24
                )
                
            return player_stats_map
            
        except Exception as e:
            print(f"Error fetching league stats: {e}")
            return {}

    def get_team_player_aggregated_stats(self, team_id: int, season: str = "2024-25") -> Dict:
        """
        Get aggregated player statistics for a team with caching.
        Returns team-level metrics based on active roster performance.

        Uses bulk fetching to avoid N+1 API call problem.
        """
        try:
            # Check cache first
            cached = self.player_cache.get_team_aggregated_stats(team_id, season)
            if cached:
                return cached

            # Get roster (also cached)
            roster_cached = self.player_cache.get_team_roster(team_id, season)
            if roster_cached:
                roster = pd.DataFrame(roster_cached)
            else:
                roster = self.get_team_roster(team_id, season)
                if not roster.empty:
                    # Cache roster for 1 week
                    self.player_cache.set_team_roster(team_id, season, roster.to_dict('records'), ttl_hours=168)

            if roster.empty:
                print(f"No roster found for team {team_id}")
                return {}

            # OPTIMIZATION: Instead of fetching each player individually, 
            # check if we have them in cache. If many are missing, fetch the WHOLE LEAGUE.
            # This is a heuristic: if we need > 3 players, it's faster to fetch the league (1 call)
            # than 3+ individual calls (3+ calls).
            
            missing_players = []
            for _, player in roster.iterrows():
                pid = player.get('PLAYER_ID')
                if pid and not self.player_cache.get_player_stats(pid):
                    missing_players.append(pid)
            
            if len(missing_players) > 2:
                # Bulk fetch!
                self.get_league_player_stats(season)
            
            # Now proceed with aggregation (stats should be in cache now)
            total_ppg = 0
            total_rpg = 0
            total_apg = 0
            total_fg_pct = 0
            player_count = 0
            top_scorer_ppg = 0
            top_playmaker_apg = 0

            for _, player in roster.iterrows():
                player_id = player.get('PLAYER_ID')
                player_name = player.get('PLAYER', 'Unknown')

                if player_id:
                    # Stats should be in cache now (either from previous runs or bulk fetch)
                    player_stats = self.player_cache.get_player_stats(player_id)

                    # Fallback if still missing (e.g. new player not in league dash yet?)
                    if not player_stats:
                        player_stats = self.get_player_recent_stats(player_id, n_games=5)
                        if player_stats:
                            self.player_cache.set_player_stats(
                                player_id, player_name, team_id, player_stats, ttl_hours=24
                            )
                        time.sleep(0.6)

                    if player_stats and player_stats.get('games_played', 0) > 0:
                        total_ppg += player_stats.get('ppg', 0)
                        total_rpg += player_stats.get('rpg', 0)
                        total_apg += player_stats.get('apg', 0)
                        total_fg_pct += player_stats.get('fg_pct', 0)
                        player_count += 1

                        # Track top performers
                        if player_stats.get('ppg', 0) > top_scorer_ppg:
                            top_scorer_ppg = player_stats.get('ppg', 0)
                        if player_stats.get('apg', 0) > top_playmaker_apg:
                            top_playmaker_apg = player_stats.get('apg', 0)

            if player_count == 0:
                return {}

            aggregated_stats = {
                'team_ppg_from_players': total_ppg,
                'team_rpg_from_players': total_rpg,
                'team_apg_from_players': total_apg,
                'avg_fg_pct_from_players': total_fg_pct / player_count,
                'top_scorer_ppg': top_scorer_ppg,
                'top_playmaker_apg': top_playmaker_apg,
                'active_players': player_count
            }

            # Cache aggregated stats for 24h
            self.player_cache.set_team_aggregated_stats(team_id, season, aggregated_stats, ttl_hours=24)

            return aggregated_stats

        except Exception as e:
            print(f"Error aggregating player stats for team {team_id}: {e}")
            return {}


class EloRatingSystem:
    """
    Elo rating system for NBA teams.
    
    This is how you properly rate team strength.
    """
    
    K_FACTOR = 20  # How quickly ratings change
    HOME_ADVANTAGE = 100  # Elo points for home court
    INITIAL_ELO = 1500
    
    def __init__(self, db_path: str = "data/nba_predictor.db"):
        self.db_path = Path(db_path)
        self.ratings = {}  # team_id -> current elo
        self._load_or_init_ratings()
        
    def _load_or_init_ratings(self):
        """Load existing ratings or initialize new ones."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT team_id, elo FROM current_elo")
        rows = cursor.fetchall()
        
        if rows:
            self.ratings = {row[0]: row[1] for row in rows}
        else:
            # Initialize all teams at 1500
            for team in teams.get_teams():
                self.ratings[team['id']] = self.INITIAL_ELO
                
        conn.close()
        
    def expected_win_prob(self, team_elo: float, opponent_elo: float, 
                          is_home: bool = True) -> float:
        """
        Calculate expected win probability.
        
        Args:
            team_elo: Team's current Elo rating
            opponent_elo: Opponent's current Elo rating
            is_home: Whether team is playing at home
        """
        if is_home:
            team_elo += self.HOME_ADVANTAGE
            
        elo_diff = team_elo - opponent_elo
        return 1 / (1 + 10 ** (-elo_diff / 400))
        
    def update_ratings(self, home_team_id: int, away_team_id: int,
                       home_score: int, away_score: int, game_id: str = None):
        """
        Update Elo ratings after a game.

        Uses margin of victory adjustment for more accurate ratings.
        """
        # Initialize team ratings if not present
        if home_team_id not in self.ratings:
            self.ratings[home_team_id] = self.INITIAL_ELO
        if away_team_id not in self.ratings:
            self.ratings[away_team_id] = self.INITIAL_ELO

        home_elo = self.ratings[home_team_id]
        away_elo = self.ratings[away_team_id]

        # Expected probabilities
        home_expected = self.expected_win_prob(home_elo, away_elo, is_home=True)
        away_expected = 1 - home_expected

        # Actual result
        home_win = 1 if home_score > away_score else 0
        away_win = 1 - home_win

        # Margin of victory multiplier (prevents autocorrelation)
        point_diff = abs(home_score - away_score)
        mov_multiplier = np.log(point_diff + 1) * (2.2 / ((home_elo - away_elo) * 0.001 + 2.2))
        mov_multiplier = max(1.0, min(mov_multiplier, 3.0))  # Cap between 1 and 3

        # Update ratings
        home_new = home_elo + self.K_FACTOR * mov_multiplier * (home_win - home_expected)
        away_new = away_elo + self.K_FACTOR * mov_multiplier * (away_win - away_expected)

        # Store old ratings before update
        home_old = self.ratings[home_team_id]
        away_old = self.ratings[away_team_id]

        self.ratings[home_team_id] = home_new
        self.ratings[away_team_id] = away_new
        
        # Save to database
        self._save_rating_update(home_team_id, home_old, home_new, game_id)
        self._save_rating_update(away_team_id, away_old, away_new, game_id)
        
        return home_new, away_new
        
    def _save_rating_update(self, team_id: int, old_elo: float, 
                            new_elo: float, game_id: str):
        """Save rating update to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update current elo
        cursor.execute("""
            INSERT OR REPLACE INTO current_elo (team_id, elo, last_updated)
            VALUES (?, ?, ?)
        """, (team_id, new_elo, datetime.now().strftime('%Y-%m-%d')))
        
        # Save to history
        if game_id:
            cursor.execute("""
                INSERT OR REPLACE INTO elo_ratings (team_id, game_date, elo_before, elo_after, game_id)
                VALUES (?, ?, ?, ?, ?)
            """, (team_id, datetime.now().strftime('%Y-%m-%d'), old_elo, new_elo, game_id))
            
        conn.commit()
        conn.close()
        
    def calculate_all_historical(self, games_df: pd.DataFrame):
        """
        Calculate Elo ratings for all historical games.
        
        MUST be called after fetching historical data.
        Games must be sorted by date (oldest first).
        """
        # Reset all ratings
        for team in teams.get_teams():
            self.ratings[team['id']] = self.INITIAL_ELO
            
        # Sort by date
        games_df = games_df.sort_values('game_date')
        
        print("Calculating historical Elo ratings...")
        
        for idx, row in games_df.iterrows():
            self.update_ratings(
                home_team_id=row['home_team_id'],
                away_team_id=row['away_team_id'],
                home_score=row['home_score'],
                away_score=row['away_score'],
                game_id=row['game_id']
            )
            
        print(f"Elo ratings calculated for {len(games_df)} games")
        
    def get_rating(self, team_id: int) -> float:
        """Get current Elo rating for a team."""
        return self.ratings.get(team_id, self.INITIAL_ELO)

    def diagnose_elo_freshness(self) -> pd.DataFrame:
        """
        Check if ELO ratings are current.
        CRITICAL: Stale ELO ratings are a major source of prediction errors.

        Returns DataFrame with team ELO freshness status.
        """
        conn = sqlite3.connect(self.db_path)

        # Get last ELO update for each team
        query = """
        SELECT ce.team_id, ce.elo, ce.last_updated,
               t.full_name as team_name
        FROM current_elo ce
        LEFT JOIN (
            SELECT team_id, full_name FROM (
                SELECT home_team_id as team_id, home_team as full_name FROM games
                UNION
                SELECT away_team_id as team_id, away_team as full_name FROM games
            ) GROUP BY team_id
        ) t ON ce.team_id = t.team_id
        ORDER BY ce.last_updated DESC
        """

        try:
            df = pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error querying ELO: {e}")
            conn.close()
            return pd.DataFrame()

        conn.close()

        if df.empty:
            print("[WARNING] No ELO ratings found in database!")
            return df

        print("\n=== ELO FRESHNESS DIAGNOSTIC ===")

        # Check staleness
        today = datetime.now()
        stale_teams = []

        for _, row in df.iterrows():
            try:
                last_update = pd.to_datetime(row['last_updated'])
                days_old = (today - last_update).days
                team_name = row.get('team_name', f"Team {row['team_id']}")

                if days_old > 3:
                    stale_teams.append({
                        'team': team_name,
                        'days_old': days_old,
                        'elo': row['elo'],
                        'last_updated': row['last_updated']
                    })
            except Exception:
                continue

        if stale_teams:
            print(f"\n[WARNING] {len(stale_teams)} teams have stale ELO ratings (>3 days old):")
            for team in stale_teams[:10]:  # Show first 10
                print(f"  - {team['team']}: {team['days_old']} days old (ELO: {team['elo']:.0f})")
            print("\nRun update_elo_after_games() to refresh ratings!")
        else:
            print("[OK] All ELO ratings are current (updated within 3 days)")

        # Show top/bottom teams
        print("\nTop 5 ELO ratings:")
        for _, row in df.nlargest(5, 'elo').iterrows():
            team_name = row.get('team_name', f"Team {row['team_id']}")
            print(f"  {team_name}: {row['elo']:.0f}")

        print("\nBottom 5 ELO ratings:")
        for _, row in df.nsmallest(5, 'elo').iterrows():
            team_name = row.get('team_name', f"Team {row['team_id']}")
            print(f"  {team_name}: {row['elo']:.0f}")

        return df

    def update_elo_from_recent_games(self, days: int = 7):
        """
        Update ELO ratings from recent game results.
        Call this to ensure ELO is fresh before making predictions.
        """
        conn = sqlite3.connect(self.db_path)

        # Get recent games that haven't been used to update ELO
        query = f"""
        SELECT g.game_id, g.game_date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score
        FROM games g
        WHERE g.game_date >= date('now', '-{days} days')
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
        ORDER BY g.game_date ASC
        """

        games_df = pd.read_sql(query, conn)
        conn.close()

        if games_df.empty:
            print(f"No completed games found in last {days} days")
            return

        print(f"Updating ELO from {len(games_df)} recent games...")

        for _, row in games_df.iterrows():
            self.update_ratings(
                home_team_id=row['home_team_id'],
                away_team_id=row['away_team_id'],
                home_score=row['home_score'],
                away_score=row['away_score'],
                game_id=row['game_id']
            )

        print(f"[OK] ELO ratings updated from {len(games_df)} games")


class FeatureEngineer:
    """
    Creates features for ML model.

    ROCKSTAR VERSION with 95+ features:
    - Elo ratings
    - Recent form (balanced recency weighting)
    - Strength of Schedule (NEW - reduces recency bias)
    - Home/Away splits
    - Head-to-head
    - Rest days
    - Streaks
    - Travel distance & fatigue
    - Injuries
    - Betting lines (market wisdom)
    - Player stats
    """
    
    def __init__(self, db_path: str = "data/nba_predictor.db"):
        self.db_path = Path(db_path)
        self.elo_system = EloRatingSystem(db_path)
        
        # Import new modules
        try:
            from src.travel_calculator import get_travel_features
            from src.injury_tracker import InjuryTracker
            from src.betting_lines_fetcher import BettingLinesFetcher
            
            self.travel_calculator = get_travel_features
            self.injury_tracker = InjuryTracker(db_path)
            self.betting_lines_fetcher = BettingLinesFetcher(db_path)
            self.enhanced_features_available = True
        except ImportError as e:
            print(f"Warning: Enhanced features not available: {e}")
            self.enhanced_features_available = False
        
    def create_features_for_game(self, home_team_id: int, away_team_id: int,
                                  game_date: str = None, include_player_stats: bool = False) -> Dict:
        """
        Create complete feature set for a single game prediction.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Date of game (default: today)
            include_player_stats: Whether to fetch real-time player stats (slow, for predictions only)

        Returns dict with 60+ features (81 with player stats).
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
            
        features = {}
        
        conn = sqlite3.connect(self.db_path)
        
        # ═══════════════════════════════════════════════════════════════
        # 1. ELO RATINGS (4 features)
        # ═══════════════════════════════════════════════════════════════
        home_elo = self.elo_system.get_rating(home_team_id)
        away_elo = self.elo_system.get_rating(away_team_id)

        elo_diff = home_elo - away_elo

        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = elo_diff

        # ELO capped version (reduce over-reliance on extreme differences)
        # Analysis showed 50% error rate when |ELO diff| > 200
        features['elo_diff_capped'] = np.sign(elo_diff) * min(abs(elo_diff), 200)

        features['elo_win_prob'] = self.elo_system.expected_win_prob(
            home_elo, away_elo, is_home=True
        )
        
        # ═══════════════════════════════════════════════════════════════
        # 2. RECENT FORM - Last 10 games (30+ features with advanced metrics)
        # ═══════════════════════════════════════════════════════════════
        for prefix, team_id in [('home', home_team_id), ('away', away_team_id)]:
            recent = self._get_recent_stats(conn, team_id, game_date, n_games=10)

            # Traditional stats
            features[f'{prefix}_last10_win_pct'] = recent['win_pct']
            features[f'{prefix}_last10_ppg'] = recent['ppg']
            features[f'{prefix}_last10_opp_ppg'] = recent['opp_ppg']
            features[f'{prefix}_last10_point_diff'] = recent['point_diff']
            features[f'{prefix}_last10_fg_pct'] = recent['fg_pct']
            features[f'{prefix}_last10_fg3_pct'] = recent['fg3_pct']
            features[f'{prefix}_last10_ft_pct'] = recent['ft_pct']
            features[f'{prefix}_last10_reb'] = recent['reb']
            features[f'{prefix}_last10_ast'] = recent['ast']
            features[f'{prefix}_last10_tov'] = recent['tov']

            # Advanced metrics (NEW)
            features[f'{prefix}_last10_offensive_rating'] = recent['offensive_rating']
            features[f'{prefix}_last10_defensive_rating'] = recent['defensive_rating']
            features[f'{prefix}_last10_net_rating'] = recent['net_rating']
            features[f'{prefix}_last10_pace'] = recent['pace']
            features[f'{prefix}_last10_three_point_rate'] = recent['three_point_rate']
            features[f'{prefix}_last10_opp_fg3_pct'] = recent['opp_fg3_pct']  # 3pt defense

        # ═══════════════════════════════════════════════════════════════
        # 3. LAST 5 GAMES - More recent form (15+ features with advanced metrics)
        # ═══════════════════════════════════════════════════════════════
        for prefix, team_id in [('home', home_team_id), ('away', away_team_id)]:
            recent5 = self._get_recent_stats(conn, team_id, game_date, n_games=5)

            # Traditional stats
            features[f'{prefix}_last5_win_pct'] = recent5['win_pct']
            features[f'{prefix}_last5_ppg'] = recent5['ppg']
            features[f'{prefix}_last5_point_diff'] = recent5['point_diff']
            features[f'{prefix}_last5_fg_pct'] = recent5['fg_pct']
            features[f'{prefix}_last5_tov'] = recent5['tov']

            # Advanced metrics (NEW)
            features[f'{prefix}_last5_offensive_rating'] = recent5['offensive_rating']
            features[f'{prefix}_last5_defensive_rating'] = recent5['defensive_rating']
            features[f'{prefix}_last5_net_rating'] = recent5['net_rating']
            features[f'{prefix}_last5_pace'] = recent5['pace']

        # ═══════════════════════════════════════════════════════════════
        # 3b. ULTRA-RECENT FORM - LAST 3 GAMES (AGGRESSIVE RECENCY)
        # ═══════════════════════════════════════════════════════════════
        # These features give MAXIMUM weight to the most recent 3 games
        # Research shows recent form (especially last 3) is a better predictor than season-long stats
        for prefix, team_id in [('home', home_team_id), ('away', away_team_id)]:
            recent3 = self._get_recent_stats(conn, team_id, game_date, n_games=3)

            # Critical recent form indicators
            features[f'{prefix}_last3_win_pct'] = recent3['win_pct']  # Most important
            features[f'{prefix}_last3_point_diff'] = recent3['point_diff']  # Dominance indicator
            features[f'{prefix}_last3_net_rating'] = recent3['net_rating']  # True strength
            features[f'{prefix}_last3_offensive_rating'] = recent3['offensive_rating']
            features[f'{prefix}_last3_defensive_rating'] = recent3['defensive_rating']

        # ═══════════════════════════════════════════════════════════════
        # 3c. MOMENTUM & TREND FEATURES (ULTRA-AGGRESSIVE RECENCY)
        # ═══════════════════════════════════════════════════════════════
        # These features capture whether teams are getting BETTER or WORSE
        # "Hot hand" effect - teams on upward trajectory vs downward spiral
        for prefix in ['home', 'away']:
            # Form acceleration: Are they improving or declining?
            # Last3 > Last5 = getting better (positive momentum)
            # Last3 < Last5 = getting worse (negative momentum)
            features[f'{prefix}_form_acceleration'] = (
                features[f'{prefix}_last3_win_pct'] - features[f'{prefix}_last5_win_pct']
            ) * 2.0  # Amplify the signal

            # Scoring trend: Recent offensive explosion or slump?
            features[f'{prefix}_recent_scoring_surge'] = (
                features[f'{prefix}_last3_point_diff'] - features[f'{prefix}_last5_point_diff']
            )

            # Defensive trend: Getting stingier or leakier?
            features[f'{prefix}_defensive_trend'] = (
                features[f'{prefix}_last5_defensive_rating'] - features[f'{prefix}_last3_defensive_rating']
            ) / 10.0  # Lower is better for defense, so flip it

            # Net rating acceleration (most powerful single indicator)
            features[f'{prefix}_net_rating_surge'] = (
                features[f'{prefix}_last3_net_rating'] - features[f'{prefix}_last10_net_rating']
            )

        # ═══════════════════════════════════════════════════════════════
        # 3d. BALANCED RECENCY FEATURES (FIXED: Reduced over-weighting)
        # ═══════════════════════════════════════════════════════════════
        # Previous: 60% on last 3 games caused severe recency bias
        # Fixed: More balanced weighting to prevent random variance from dominating
        for prefix in ['home', 'away']:
            # Weighted win rate - REDUCED recency bias (was 0.60/0.25/0.15)
            weighted_win_pct = (
                0.35 * features[f'{prefix}_last3_win_pct'] +    # 35% weight on last 3 (was 60%)
                0.35 * features[f'{prefix}_last5_win_pct'] +    # 35% weight on last 5 (was 25%)
                0.30 * features[f'{prefix}_last10_win_pct']     # 30% weight on last 10 (was 15%)
            )
            features[f'{prefix}_weighted_recent_form'] = weighted_win_pct

            # Weighted net rating - more balanced
            weighted_net_rating = (
                0.35 * features[f'{prefix}_last3_net_rating'] +  # Reduced from 60%
                0.35 * features[f'{prefix}_last5_net_rating'] +  # Increased from 25%
                0.30 * features[f'{prefix}_last10_net_rating']   # Increased from 15%
            )
            features[f'{prefix}_weighted_net_rating'] = weighted_net_rating

            # REMOVED: Ultra-aggressive recency (was causing bad predictions)
            # Instead, let Elo and longer-term stats balance out recent noise

        # Differential of weighted forms (CRITICAL FEATURE)
        features['weighted_form_differential'] = (
            features['home_weighted_recent_form'] - features['away_weighted_recent_form']
        )

        features['weighted_net_rating_differential'] = (
            features['home_weighted_net_rating'] - features['away_weighted_net_rating']
        )

        # NOTE: Ultra-weighted form was REMOVED due to causing bad predictions
        # The SOS-adjusted features below now serve this purpose better

        # ═══════════════════════════════════════════════════════════════
        # 3e. STRENGTH OF SCHEDULE (NEW - reduces recency bias)
        # ═══════════════════════════════════════════════════════════════
        # A team's recent wins mean less if they beat weak opponents
        # A team's recent losses mean less if they faced strong opponents
        for prefix, team_id in [('home', home_team_id), ('away', away_team_id)]:
            sos = self._get_strength_of_schedule(conn, team_id, game_date, n_games=10)

            features[f'{prefix}_sos_normalized'] = sos['sos_normalized']
            features[f'{prefix}_avg_opponent_elo'] = sos['avg_opponent_elo']
            features[f'{prefix}_sos_adjusted_win_pct'] = sos['adjusted_win_pct']
            features[f'{prefix}_opponent_win_pct'] = sos['opponent_win_pct']

        # SOS differentials - who faced harder opponents?
        features['sos_differential'] = (
            features['home_sos_normalized'] - features['away_sos_normalized']
        )
        features['avg_opponent_elo_differential'] = (
            features['home_avg_opponent_elo'] - features['away_avg_opponent_elo']
        )

        # Adjusted form differential - THIS IS THE KEY FEATURE
        # Uses SOS-adjusted win percentages instead of raw win percentages
        features['sos_adjusted_form_differential'] = (
            features['home_sos_adjusted_win_pct'] - features['away_sos_adjusted_win_pct']
        )

        # ═══════════════════════════════════════════════════════════════
        # 4. HOME/AWAY SPLITS (8 features)
        # ═══════════════════════════════════════════════════════════════
        home_at_home = self._get_home_away_split(conn, home_team_id, game_date, is_home=True)
        away_on_road = self._get_home_away_split(conn, away_team_id, game_date, is_home=False)
        
        features['home_team_home_win_pct'] = home_at_home['win_pct']
        features['home_team_home_ppg'] = home_at_home['ppg']
        features['home_team_home_point_diff'] = home_at_home['point_diff']
        features['home_team_home_fg_pct'] = home_at_home['fg_pct']
        
        features['away_team_road_win_pct'] = away_on_road['win_pct']
        features['away_team_road_ppg'] = away_on_road['ppg']
        features['away_team_road_point_diff'] = away_on_road['point_diff']
        features['away_team_road_fg_pct'] = away_on_road['fg_pct']
        
        # ═══════════════════════════════════════════════════════════════
        # 5. HEAD-TO-HEAD (6 features)
        # ═══════════════════════════════════════════════════════════════
        h2h = self._get_head_to_head(conn, home_team_id, away_team_id, game_date)
        
        features['h2h_home_win_pct'] = h2h['home_win_pct']
        features['h2h_total_games'] = h2h['total_games']
        features['h2h_avg_point_diff'] = h2h['avg_point_diff']
        features['h2h_last3_home_wins'] = h2h['last3_home_wins']
        features['h2h_home_ppg'] = h2h['home_ppg']
        features['h2h_away_ppg'] = h2h['away_ppg']
        
        # ═══════════════════════════════════════════════════════════════
        # 6. REST DAYS & SCHEDULE FEATURES (expanded with special dates)
        # ═══════════════════════════════════════════════════════════════
        home_rest = self._get_rest_days(conn, home_team_id, game_date)
        away_rest = self._get_rest_days(conn, away_team_id, game_date)

        features['home_rest_days'] = home_rest
        features['away_rest_days'] = away_rest
        features['rest_advantage'] = home_rest - away_rest
        features['home_back_to_back'] = 1 if home_rest == 0 else 0
        features['away_back_to_back'] = 1 if away_rest == 0 else 0

        # Schedule features - special dates that affect performance
        schedule_features = self._get_schedule_features(game_date, conn, home_team_id, away_team_id)
        features.update(schedule_features)
        
        # ═══════════════════════════════════════════════════════════════
        # 7. STREAK (4 features)
        # ═══════════════════════════════════════════════════════════════
        home_streak = self._get_streak(conn, home_team_id, game_date)
        away_streak = self._get_streak(conn, away_team_id, game_date)
        
        features['home_streak'] = home_streak  # Positive = winning, negative = losing
        features['away_streak'] = away_streak
        features['home_on_win_streak'] = 1 if home_streak >= 3 else 0
        features['away_on_win_streak'] = 1 if away_streak >= 3 else 0
        
        # ═══════════════════════════════════════════════════════════════
        # 8. DIFFERENTIALS (10 features)
        # ═══════════════════════════════════════════════════════════════
        features['ppg_diff'] = features['home_last10_ppg'] - features['away_last10_ppg']
        features['point_diff_diff'] = features['home_last10_point_diff'] - features['away_last10_point_diff']
        features['fg_pct_diff'] = features['home_last10_fg_pct'] - features['away_last10_fg_pct']
        features['fg3_pct_diff'] = features['home_last10_fg3_pct'] - features['away_last10_fg3_pct']
        features['reb_diff'] = features['home_last10_reb'] - features['away_last10_reb']
        features['ast_diff'] = features['home_last10_ast'] - features['away_last10_ast']
        features['tov_diff'] = features['home_last10_tov'] - features['away_last10_tov']  # Lower is better
        features['win_pct_diff'] = features['home_last10_win_pct'] - features['away_last10_win_pct']
        features['home_split_diff'] = features['home_team_home_win_pct'] - features['away_team_road_win_pct']
        features['streak_diff'] = home_streak - away_streak

        # ═══════════════════════════════════════════════════════════════
        # 8b. ADVANCED EFFICIENCY METRICS (15+ features) - ENHANCED!
        # ═══════════════════════════════════════════════════════════════

        # Legacy net rating (kept for backward compatibility)
        home_off_eff = features['home_last10_ppg'] / 100 if features['home_last10_ppg'] > 0 else 1.1
        away_off_eff = features['away_last10_ppg'] / 100 if features['away_last10_ppg'] > 0 else 1.1
        home_def_eff = features['home_last10_opp_ppg'] / 100 if features['home_last10_opp_ppg'] > 0 else 1.1
        away_def_eff = features['away_last10_opp_ppg'] / 100 if features['away_last10_opp_ppg'] > 0 else 1.1

        features['home_net_rating'] = home_off_eff - home_def_eff
        features['away_net_rating'] = away_off_eff - away_def_eff
        features['net_rating_diff'] = features['home_net_rating'] - features['away_net_rating']

        # NEW: Advanced Rating Differentials
        features['offensive_rating_diff'] = features['home_last10_offensive_rating'] - features['away_last10_offensive_rating']
        features['defensive_rating_diff'] = features['home_last10_defensive_rating'] - features['away_last10_defensive_rating']
        features['pace_diff'] = features['home_last10_pace'] - features['away_last10_pace']
        features['three_point_rate_diff'] = features['home_last10_three_point_rate'] - features['away_last10_three_point_rate']

        # NEW: 3-point defense differential (lower opponent 3pt% is better)
        features['three_point_defense_diff'] = features['away_last10_opp_fg3_pct'] - features['home_last10_opp_fg3_pct']

        # NEW: Pace matchup indicator (significant if diff > 5 possessions)
        pace_diff_abs = abs(features['pace_diff'])
        features['pace_mismatch'] = 1 if pace_diff_abs > 5 else 0

        # Turnover efficiency (lower is better)
        features['home_tov_rate'] = features['home_last10_tov'] / features['home_last10_ppg'] if features['home_last10_ppg'] > 0 else 0.15
        features['away_tov_rate'] = features['away_last10_tov'] / features['away_last10_ppg'] if features['away_last10_ppg'] > 0 else 0.15
        features['tov_rate_diff'] = features['home_tov_rate'] - features['away_tov_rate']

        # ═══════════════════════════════════════════════════════════════
        # 8c. MOMENTUM INDICATORS - ENHANCED WITH ULTRA-RECENT DATA
        # ═══════════════════════════════════════════════════════════════
        # LEGACY momentum (kept for backward compatibility with trained model)
        features['home_momentum'] = features['home_last5_win_pct'] - features['home_last10_win_pct']
        features['away_momentum'] = features['away_last5_win_pct'] - features['away_last10_win_pct']
        features['momentum_diff'] = features['home_momentum'] - features['away_momentum']
        features['home_scoring_trend'] = features['home_last5_ppg'] - features['home_last10_ppg']

        # NEW: Ultra-recent momentum (last 3 vs last 10) - STRONGER SIGNAL
        features['home_ultra_momentum'] = features['home_last3_win_pct'] - features['home_last10_win_pct']
        features['away_ultra_momentum'] = features['away_last3_win_pct'] - features['away_last10_win_pct']
        features['ultra_momentum_diff'] = features['home_ultra_momentum'] - features['away_ultra_momentum']

        # Momentum clash: One team surging, other team collapsing
        features['momentum_clash'] = abs(features['ultra_momentum_diff']) * (
            1 if features['ultra_momentum_diff'] * features['elo_diff'] < 0 else 0
        )  # Amplify when momentum contradicts ELO

        # ═══════════════════════════════════════════════════════════════
        # 8d. CRITICAL INTERACTION FEATURES (from error analysis)
        # ═══════════════════════════════════════════════════════════════
        # ELO-Momentum Interaction: When hot team plays, ELO may underestimate
        home_streak = features['home_streak']
        away_streak = features['away_streak']

        # Positive when hot team is underrated by ELO
        features['elo_momentum_interaction'] = elo_diff * (home_streak - away_streak) / 10.0

        # Hot away team flag (historically wins 60% when streak >= 4 and |ELO diff| < 100)
        features['hot_away_underdog'] = 1 if (away_streak >= 4 and abs(elo_diff) < 100) else 0

        # Cold home team flag (historically loses 73% when streak <= -3 and |ELO diff| < 150)
        features['cold_home_favorite'] = 1 if (home_streak <= -3 and abs(elo_diff) < 150) else 0

        # ELO-Travel Interaction: Tired favorites lose more often
        # travel_fatigue multiplied by ELO advantage
        away_travel_fatigue = features.get('away_travel_fatigue_index', 0)
        if elo_diff > 0:  # Home team favored
            features['elo_travel_interaction'] = elo_diff * away_travel_fatigue / 100.0
        else:  # Away team favored
            features['elo_travel_interaction'] = -elo_diff * away_travel_fatigue / 100.0

        # ═══════════════════════════════════════════════════════════════
        # 9. PLAYER-LEVEL STATISTICS (14 features) - OPTIONAL
        # ═══════════════════════════════════════════════════════════════
        # Only fetch if explicitly enabled (slow due to API rate limiting)
        if include_player_stats:
            try:
                # Note: This uses NBA API which can be slow
                # Get aggregated player stats for both teams
                data_fetcher = NBADataFetcher(self.db_path)

                home_player_stats = data_fetcher.get_team_player_aggregated_stats(home_team_id)
                away_player_stats = data_fetcher.get_team_player_aggregated_stats(away_team_id)

                # Home team player stats
                features['home_team_ppg_players'] = home_player_stats.get('team_ppg_from_players', 0)
                features['home_team_rpg_players'] = home_player_stats.get('team_rpg_from_players', 0)
                features['home_team_apg_players'] = home_player_stats.get('team_apg_from_players', 0)
                features['home_avg_fg_pct_players'] = home_player_stats.get('avg_fg_pct_from_players', 0)
                features['home_top_scorer_ppg'] = home_player_stats.get('top_scorer_ppg', 0)
                features['home_top_playmaker_apg'] = home_player_stats.get('top_playmaker_apg', 0)
                features['home_active_players'] = home_player_stats.get('active_players', 0)

                # Away team player stats
                features['away_team_ppg_players'] = away_player_stats.get('team_ppg_from_players', 0)
                features['away_team_rpg_players'] = away_player_stats.get('team_rpg_from_players', 0)
                features['away_team_apg_players'] = away_player_stats.get('team_apg_from_players', 0)
                features['away_avg_fg_pct_players'] = away_player_stats.get('avg_fg_pct_from_players', 0)
                features['away_top_scorer_ppg'] = away_player_stats.get('top_scorer_ppg', 0)
                features['away_top_playmaker_apg'] = away_player_stats.get('top_playmaker_apg', 0)
                features['away_active_players'] = away_player_stats.get('active_players', 0)

            except Exception as e:
                print(f"Warning: Could not fetch player stats: {e}")
                # Fill with zeros if player stats unavailable
                for prefix in ['home', 'away']:
                    features[f'{prefix}_team_ppg_players'] = 0
                    features[f'{prefix}_team_rpg_players'] = 0
                    features[f'{prefix}_team_apg_players'] = 0
                    features[f'{prefix}_avg_fg_pct_players'] = 0
                    features[f'{prefix}_top_scorer_ppg'] = 0
                    features[f'{prefix}_top_playmaker_apg'] = 0
                    features[f'{prefix}_active_players'] = 0
        else:
            # Fill with zeros when player stats are disabled (for training)
            for prefix in ['home', 'away']:
                features[f'{prefix}_team_ppg_players'] = 0
                features[f'{prefix}_team_rpg_players'] = 0
                features[f'{prefix}_team_apg_players'] = 0
                features[f'{prefix}_avg_fg_pct_players'] = 0
                features[f'{prefix}_top_scorer_ppg'] = 0
                features[f'{prefix}_top_playmaker_apg'] = 0
                features[f'{prefix}_active_players'] = 0

        # ═══════════════════════════════════════════════════════════════
        # 10. TRAVEL & FATIGUE (3 features) - NEW!
        # ═══════════════════════════════════════════════════════════════
        if self.enhanced_features_available:
            try:
                # Get last game location for away team (simplified - assumes coming from home)
                travel_features = self.travel_calculator(
                    away_team_id=away_team_id,
                    home_team_id=home_team_id
                )
                features.update(travel_features)
            except Exception as e:
                print(f"Warning: Could not calculate travel features: {e}")
                features['away_travel_distance'] = 0.0
                features['away_time_zones_crossed'] = 0
                features['away_travel_fatigue_index'] = 0.0
        else:
            features['away_travel_distance'] = 0.0
            features['away_time_zones_crossed'] = 0
            features['away_travel_fatigue_index'] = 0.0

        # ═══════════════════════════════════════════════════════════════
        # 11. INJURIES (4 features) - NEW!
        # ═══════════════════════════════════════════════════════════════
        if self.enhanced_features_available:
            try:
                # For predictions (not training), force refresh to get latest injuries
                force_refresh = include_player_stats  # Refresh during prediction time

                injury_features = self.injury_tracker.get_injury_features(
                    home_team_id, away_team_id
                )

                # Enhanced injury impact calculation
                home_injury_impact = (
                    injury_features.get('home_injured_starters', 0) * 2.0 +
                    (3.0 if injury_features.get('home_star_injured', 0) else 0)
                )
                away_injury_impact = (
                    injury_features.get('away_injured_starters', 0) * 2.0 +
                    (3.0 if injury_features.get('away_star_injured', 0) else 0)
                )

                features.update(injury_features)
                features['home_injury_impact'] = home_injury_impact
                features['away_injury_impact'] = away_injury_impact
                features['injury_differential'] = away_injury_impact - home_injury_impact

                print(f"  Injury data: Home={home_injury_impact:.1f}, Away={away_injury_impact:.1f}")

            except Exception as e:
                print(f"Warning: Could not fetch injury features: {e}")
                features['home_injured_starters'] = 0
                features['away_injured_starters'] = 0
                features['home_star_injured'] = 0
                features['away_star_injured'] = 0
                features['home_injury_impact'] = 0
                features['away_injury_impact'] = 0
                features['injury_differential'] = 0
        else:
            features['home_injured_starters'] = 0
            features['away_injured_starters'] = 0
            features['home_star_injured'] = 0
            features['away_star_injured'] = 0
            features['home_injury_impact'] = 0
            features['away_injury_impact'] = 0
            features['injury_differential'] = 0

        # ═══════════════════════════════════════════════════════════════
        # 12. BETTING LINES - Market Wisdom (5 features) - NEW!
        # ═══════════════════════════════════════════════════════════════
        if self.enhanced_features_available:
            try:
                # Get team names for betting lines API
                home_team_name = self._get_team_name(conn, home_team_id)
                away_team_name = self._get_team_name(conn, away_team_id)
                
                betting_features = self.betting_lines_fetcher.get_betting_features(
                    home_team_name, away_team_name, game_date
                )
                features.update(betting_features)
            except Exception as e:
                print(f"Warning: Could not fetch betting line features: {e}")
                features['market_spread'] = 0.0
                features['market_total'] = 220.0
                features['market_home_ml'] = 2.0
                features['market_implied_prob'] = 0.5
                features['market_confidence'] = 0.0
        else:
            features['market_spread'] = 0.0
            features['market_total'] = 220.0
            features['market_home_ml'] = 2.0
            features['market_implied_prob'] = 0.5
            features['market_confidence'] = 0.0

        conn.close()

        return features
        
    def _get_team_name(self, conn, team_id: int) -> str:
        """Get team name from team ID."""
        from nba_api.stats.static import teams
        all_teams = teams.get_teams()
        for team in all_teams:
            if team['id'] == team_id:
                return team['full_name']
        return "Unknown Team"
    
    def _get_recent_stats(self, conn, team_id: int, before_date: str,
                          n_games: int = 10) -> Dict:
        """Get team's stats from recent games with advanced metrics."""
        query = """
            SELECT
                CASE WHEN home_team_id = ? THEN home_win ELSE 1 - home_win END as win,
                CASE WHEN home_team_id = ? THEN home_score ELSE away_score END as pts,
                CASE WHEN home_team_id = ? THEN away_score ELSE home_score END as opp_pts,
                CASE WHEN home_team_id = ? THEN home_fg_pct ELSE away_fg_pct END as fg_pct,
                CASE WHEN home_team_id = ? THEN home_fg3_pct ELSE away_fg3_pct END as fg3_pct,
                CASE WHEN home_team_id = ? THEN home_fg3a ELSE away_fg3a END as fg3a,
                CASE WHEN home_team_id = ? THEN home_fg3m ELSE away_fg3m END as fg3m,
                CASE WHEN home_team_id = ? THEN home_ft_pct ELSE away_ft_pct END as ft_pct,
                CASE WHEN home_team_id = ? THEN home_reb ELSE away_reb END as reb,
                CASE WHEN home_team_id = ? THEN home_ast ELSE away_ast END as ast,
                CASE WHEN home_team_id = ? THEN home_tov ELSE away_tov END as tov,
                CASE WHEN home_team_id = ? THEN home_fga ELSE away_fga END as fga,
                CASE WHEN home_team_id = ? THEN home_fgm ELSE away_fgm END as fgm,
                CASE WHEN home_team_id = ? THEN home_fta ELSE away_fta END as fta,
                CASE WHEN home_team_id = ? THEN home_oreb ELSE away_oreb END as oreb,
                CASE WHEN home_team_id = ? THEN home_dreb ELSE away_dreb END as dreb,
                CASE WHEN home_team_id = ? THEN away_oreb ELSE home_oreb END as opp_oreb,
                CASE WHEN home_team_id = ? THEN away_dreb ELSE home_dreb END as opp_dreb,
                CASE WHEN home_team_id = ? THEN away_fga ELSE home_fga END as opp_fga,
                CASE WHEN home_team_id = ? THEN away_fgm ELSE home_fgm END as opp_fgm,
                CASE WHEN home_team_id = ? THEN away_fta ELSE home_fta END as opp_fta,
                CASE WHEN home_team_id = ? THEN away_tov ELSE home_tov END as opp_tov,
                CASE WHEN home_team_id = ? THEN away_fg3_pct ELSE home_fg3_pct END as opp_fg3_pct,
                CASE WHEN home_team_id = ? THEN away_fg3a ELSE home_fg3a END as opp_fg3a
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        """

        params = [team_id] * 24 + [team_id, team_id, before_date, n_games]
        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return {
                'win_pct': 0.5, 'ppg': 110, 'opp_ppg': 110, 'point_diff': 0,
                'fg_pct': 0.45, 'fg3_pct': 0.35, 'fg3a': 30, 'opp_fg3_pct': 0.35, 'opp_fg3a': 30,
                'ft_pct': 0.78, 'reb': 44, 'ast': 25, 'tov': 14,
                'offensive_rating': 110, 'defensive_rating': 110, 'net_rating': 0,
                'pace': 100, 'three_point_rate': 0.35, 'opp_three_point_rate': 0.35
            }

        # Calculate advanced metrics
        pace_values = []
        ortg_values = []
        drtg_values = []

        for _, row in df.iterrows():
            # Estimate possessions using formula
            poss = self._estimate_possessions(
                row['fga'], row['fgm'], row['fta'], row['oreb'], row['dreb'],
                row['tov'], row['opp_fga'], row['opp_fgm'], row['opp_fta'],
                row['opp_oreb'], row['opp_dreb'], row['opp_tov']
            )

            if poss > 0:
                # Offensive Rating: points per 100 possessions
                ortg = (row['pts'] / poss) * 100
                # Defensive Rating: opponent points per 100 possessions
                drtg = (row['opp_pts'] / poss) * 100
                # Pace: possessions per 48 minutes (estimated)
                pace = poss  # Already calculated per game

                ortg_values.append(ortg)
                drtg_values.append(drtg)
                pace_values.append(pace)

        return {
            'win_pct': df['win'].mean(),
            'ppg': df['pts'].mean(),
            'opp_ppg': df['opp_pts'].mean(),
            'point_diff': (df['pts'] - df['opp_pts']).mean(),
            'fg_pct': df['fg_pct'].mean(),
            'fg3_pct': df['fg3_pct'].mean(),
            'fg3a': df['fg3a'].mean(),
            'opp_fg3_pct': df['opp_fg3_pct'].mean(),
            'opp_fg3a': df['opp_fg3a'].mean(),
            'ft_pct': df['ft_pct'].mean(),
            'reb': df['reb'].mean(),
            'ast': df['ast'].mean(),
            'tov': df['tov'].mean(),
            'offensive_rating': np.mean(ortg_values) if ortg_values else 110,
            'defensive_rating': np.mean(drtg_values) if drtg_values else 110,
            'net_rating': (np.mean(ortg_values) - np.mean(drtg_values)) if ortg_values and drtg_values else 0,
            'pace': np.mean(pace_values) if pace_values else 100,
            'three_point_rate': df['fg3a'].mean() / df['fga'].mean() if df['fga'].mean() > 0 else 0.35,
            'opp_three_point_rate': df['opp_fg3a'].mean() / df['opp_fga'].mean() if df['opp_fga'].mean() > 0 else 0.35
        }

    def _estimate_possessions(self, fga, fgm, fta, oreb, dreb, tov,
                              opp_fga, opp_fgm, opp_fta, opp_oreb, opp_dreb, opp_tov):
        """
        Estimate possessions using the standard NBA formula.
        Formula: 0.5 * ((Team Poss Estimate) + (Opp Poss Estimate))
        """
        # Team possessions estimate
        team_poss = fga + 0.4 * fta - 1.07 * (oreb / (oreb + opp_dreb + 1)) * (fga - fgm) + tov

        # Opponent possessions estimate
        opp_poss = opp_fga + 0.4 * opp_fta - 1.07 * (opp_oreb / (opp_oreb + dreb + 1)) * (opp_fga - opp_fgm) + opp_tov

        # Average of both estimates
        return 0.5 * (team_poss + opp_poss)
        
    def _get_home_away_split(self, conn, team_id: int, before_date: str,
                              is_home: bool, n_games: int = 15) -> Dict:
        """Get team's performance at home or on the road."""
        if is_home:
            query = """
                SELECT home_win as win, home_score as pts, away_score as opp_pts,
                       home_fg_pct as fg_pct
                FROM games
                WHERE home_team_id = ? AND game_date < ?
                ORDER BY game_date DESC
                LIMIT ?
            """
        else:
            query = """
                SELECT 1 - home_win as win, away_score as pts, home_score as opp_pts,
                       away_fg_pct as fg_pct
                FROM games
                WHERE away_team_id = ? AND game_date < ?
                ORDER BY game_date DESC
                LIMIT ?
            """
            
        df = pd.read_sql_query(query, conn, params=(team_id, before_date, n_games))
        
        if df.empty:
            return {'win_pct': 0.5, 'ppg': 110, 'point_diff': 0, 'fg_pct': 0.45}
            
        return {
            'win_pct': df['win'].mean(),
            'ppg': df['pts'].mean(),
            'point_diff': (df['pts'] - df['opp_pts']).mean(),
            'fg_pct': df['fg_pct'].mean()
        }
        
    def _get_head_to_head(self, conn, home_team_id: int, away_team_id: int,
                          before_date: str, n_games: int = 10) -> Dict:
        """Get head-to-head history between teams."""
        query = """
            SELECT home_win, home_score, away_score
            FROM games
            WHERE ((home_team_id = ? AND away_team_id = ?) 
                   OR (home_team_id = ? AND away_team_id = ?))
            AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        
        params = (home_team_id, away_team_id, away_team_id, home_team_id, 
                  before_date, n_games)
        df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            return {
                'home_win_pct': 0.5, 'total_games': 0, 'avg_point_diff': 0,
                'last3_home_wins': 1.5, 'home_ppg': 110, 'away_ppg': 110
            }
            
        return {
            'home_win_pct': df['home_win'].mean(),
            'total_games': len(df),
            'avg_point_diff': (df['home_score'] - df['away_score']).mean(),
            'last3_home_wins': df.head(3)['home_win'].sum() if len(df) >= 3 else df['home_win'].sum(),
            'home_ppg': df['home_score'].mean(),
            'away_ppg': df['away_score'].mean()
        }
        
    def _get_rest_days(self, conn, team_id: int, game_date: str) -> int:
        """Get number of rest days before game."""
        query = """
            SELECT game_date FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND game_date < ?
            ORDER BY game_date DESC
            LIMIT 1
        """
        
        df = pd.read_sql_query(query, conn, params=(team_id, team_id, game_date))
        
        if df.empty:
            return 3  # Default to well-rested
            
        last_game = pd.to_datetime(df['game_date'].iloc[0])
        current = pd.to_datetime(game_date)
        
        return (current - last_game).days - 1  # -1 because game_date counts
        
    def _get_streak(self, conn, team_id: int, before_date: str) -> int:
        """Get current win/loss streak. Positive = winning, negative = losing."""
        query = """
            SELECT 
                CASE WHEN home_team_id = ? THEN home_win ELSE 1 - home_win END as win
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND game_date < ?
            ORDER BY game_date DESC
            LIMIT 15
        """
        
        df = pd.read_sql_query(query, conn, params=(team_id, team_id, team_id, before_date))
        
        if df.empty:
            return 0
            
        streak = 0
        first_result = df['win'].iloc[0]
        
        for win in df['win']:
            if win == first_result:
                streak += 1
            else:
                break
                
        return streak if first_result == 1 else -streak

    def _get_schedule_features(self, game_date: str, conn, home_team_id: int, away_team_id: int) -> Dict:
        """
        Add schedule-related features that affect performance.
        Analysis showed December 26 (day after Christmas) had 33% accuracy.
        """
        features = {}

        # Parse date
        if isinstance(game_date, str):
            game_date_obj = datetime.strptime(game_date, '%Y-%m-%d')
        else:
            game_date_obj = game_date

        # Day of week (0=Monday, 6=Sunday)
        features['day_of_week'] = game_date_obj.weekday()
        features['is_weekend'] = 1 if game_date_obj.weekday() >= 5 else 0

        # Special dates that affect performance
        features['is_christmas'] = 1 if (game_date_obj.month == 12 and game_date_obj.day == 25) else 0
        features['is_day_after_christmas'] = 1 if (game_date_obj.month == 12 and game_date_obj.day == 26) else 0
        features['is_new_years_eve'] = 1 if (game_date_obj.month == 12 and game_date_obj.day == 31) else 0
        features['is_new_years_day'] = 1 if (game_date_obj.month == 1 and game_date_obj.day == 1) else 0

        # MLK weekend (third Monday of January)
        if game_date_obj.month == 1:
            # Find third Monday
            first_day = datetime(game_date_obj.year, 1, 1)
            days_until_monday = (7 - first_day.weekday()) % 7
            third_monday = first_day + timedelta(days=days_until_monday + 14)
            mlk_weekend = third_monday - timedelta(days=2) <= game_date_obj <= third_monday
            features['is_mlk_weekend'] = 1 if mlk_weekend else 0
        else:
            features['is_mlk_weekend'] = 0

        # Month of season (affects playoff push, tanking)
        features['month'] = game_date_obj.month
        features['is_late_season'] = 1 if game_date_obj.month in [3, 4] else 0  # Playoff push
        features['is_early_season'] = 1 if game_date_obj.month in [10, 11] else 0  # Still gelling

        # Games in last 7 days (schedule density)
        home_games_7d = self._count_recent_games(conn, home_team_id, game_date, days=7)
        away_games_7d = self._count_recent_games(conn, away_team_id, game_date, days=7)

        features['home_games_last_7d'] = home_games_7d
        features['away_games_last_7d'] = away_games_7d
        features['schedule_density_diff'] = home_games_7d - away_games_7d

        # Road trip length for away team
        features['away_road_trip_length'] = self._get_road_trip_length(conn, away_team_id, game_date)
        features['away_long_road_trip'] = 1 if features['away_road_trip_length'] >= 4 else 0

        return features

    def _count_recent_games(self, conn, team_id: int, game_date: str, days: int = 7) -> int:
        """Count games played in last N days"""
        query = f"""
        SELECT COUNT(*) as game_count
        FROM games
        WHERE (home_team_id = ? OR away_team_id = ?)
        AND game_date BETWEEN date(?, '-{days} days') AND date(?, '-1 day')
        """
        result = pd.read_sql_query(query, conn, params=(team_id, team_id, game_date, game_date))
        return int(result['game_count'].iloc[0]) if not result.empty else 0

    def _get_road_trip_length(self, conn, team_id: int, game_date: str) -> int:
        """Calculate current road trip length"""
        query = """
        SELECT game_date, home_team_id
        FROM games
        WHERE (home_team_id = ? OR away_team_id = ?)
        AND game_date < ?
        ORDER BY game_date DESC
        LIMIT 10
        """
        games = pd.read_sql_query(query, conn, params=(team_id, team_id, game_date))

        if games.empty:
            return 0

        road_games = 0
        for _, game in games.iterrows():
            if game['home_team_id'] != team_id:
                road_games += 1
            else:
                break  # Found a home game, road trip ended

        return road_games

    def _get_strength_of_schedule(self, conn, team_id: int, before_date: str,
                                   n_games: int = 10) -> Dict:
        """
        Calculate strength of schedule based on opponent Elo ratings.

        This is CRITICAL for reducing recency bias:
        - A 7-3 record against weak teams (avg Elo 1450) is worse than
        - A 5-5 record against strong teams (avg Elo 1550)

        Returns:
            Dict with:
            - avg_opponent_elo: Average Elo of recent opponents
            - sos_normalized: 0-1 scale (1 = faced strongest opponents)
            - adjusted_win_pct: Win rate adjusted for opponent strength
            - opponent_win_pct: Average win% of opponents faced
        """
        # Get recent games with opponent info
        query = """
            SELECT
                game_date,
                CASE WHEN home_team_id = ? THEN away_team_id ELSE home_team_id END as opponent_id,
                CASE WHEN home_team_id = ? THEN home_win ELSE 1 - home_win END as win
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND game_date < ?
            ORDER BY game_date DESC
            LIMIT ?
        """
        params = [team_id, team_id, team_id, team_id, before_date, n_games]
        games = pd.read_sql_query(query, conn, params=params)

        if games.empty:
            return {
                'avg_opponent_elo': 1500,
                'sos_normalized': 0.5,
                'adjusted_win_pct': 0.5,
                'opponent_win_pct': 0.5
            }

        # Get Elo ratings for each opponent
        opponent_elos = []
        opponent_win_pcts = []

        for _, game in games.iterrows():
            opp_id = int(game['opponent_id'])
            opp_elo = self.elo_system.get_rating(opp_id)
            opponent_elos.append(opp_elo)

            # Get opponent's win percentage (their recent form)
            opp_query = """
                SELECT
                    AVG(CASE WHEN home_team_id = ? THEN home_win ELSE 1 - home_win END) as win_pct
                FROM games
                WHERE (home_team_id = ? OR away_team_id = ?)
                AND game_date < ?
                ORDER BY game_date DESC
                LIMIT 10
            """
            opp_df = pd.read_sql_query(opp_query, conn, params=[opp_id, opp_id, opp_id, before_date])
            opp_win_pct = opp_df['win_pct'].iloc[0] if not opp_df.empty and opp_df['win_pct'].iloc[0] is not None else 0.5
            opponent_win_pcts.append(opp_win_pct)

        avg_opp_elo = np.mean(opponent_elos)
        avg_opp_win_pct = np.mean(opponent_win_pcts)
        actual_win_pct = games['win'].mean()

        # Normalize SOS: Elo typically ranges 1350-1650, center at 1500
        # sos_normalized: 0 = weakest opponents (Elo ~1350), 1 = strongest (Elo ~1650)
        sos_normalized = np.clip((avg_opp_elo - 1350) / 300, 0, 1)

        # Adjusted win percentage:
        # If you beat strong teams, your wins are worth more
        # If you beat weak teams, your wins are worth less
        # Formula: actual_win_pct * (0.5 + 0.5 * sos_normalized) + (1 - actual_win_pct) * (1 - sos_normalized) * 0.3
        # Simplified: adjust toward 0.5 based on inverse of SOS
        sos_adjustment = (sos_normalized - 0.5) * 0.3  # -0.15 to +0.15 adjustment
        adjusted_win_pct = actual_win_pct + sos_adjustment * (actual_win_pct - 0.5)
        adjusted_win_pct = np.clip(adjusted_win_pct, 0, 1)

        return {
            'avg_opponent_elo': avg_opp_elo,
            'sos_normalized': sos_normalized,
            'adjusted_win_pct': adjusted_win_pct,
            'opponent_win_pct': avg_opp_win_pct
        }

    def create_training_dataset(self, games_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Create features for all historical games with sample weights.

        This is how you create REAL training data.

        Returns:
            X: Features dataframe
            y: Target series
            sample_weights: Array of weights emphasizing recent games
        """
        print("Creating features for training dataset...")

        features_list = []
        targets = []
        game_dates = []

        # Sort by date to ensure proper historical features
        games_df = games_df.sort_values('game_date')

        # Skip first 30 games (not enough history)
        total_games = len(games_df.iloc[30:])
        print(f"Processing {total_games} games...")

        for i, (idx, row) in enumerate(games_df.iloc[30:].iterrows()):
            try:
                # Progress reporting every 100 games
                if (i + 1) % 100 == 0:
                    progress_pct = (i + 1) / total_games * 100
                    print(f"  Progress: {i+1}/{total_games} games ({progress_pct:.1f}%)")

                features = self.create_features_for_game(
                    home_team_id=row['home_team_id'],
                    away_team_id=row['away_team_id'],
                    game_date=row['game_date']
                )

                features_list.append(features)
                targets.append(row['home_win'])
                game_dates.append(pd.to_datetime(row['game_date']))

            except Exception as e:
                continue

        # Calculate sample weights based on recency
        # More recent games get higher weight using exponential decay
        game_dates_arr = np.array(game_dates)
        days_old = (game_dates_arr.max() - game_dates_arr).astype('timedelta64[D]').astype(int)

        # Exponential decay: weight = exp(-lambda * days_old)
        # Lambda = 0.001 means weight halves every ~700 days (2 seasons)
        # This gives recent games 2-3x more weight than oldest games
        decay_rate = 0.001
        sample_weights = np.exp(-decay_rate * days_old)

        # Normalize weights to sum to number of samples (maintains scale)
        sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

        print(f"Created features for {len(features_list)} games")
        print(f"Sample weights - Recent games: {sample_weights[-10:].mean():.2f}, Old games: {sample_weights[:10].mean():.2f}")

        X = pd.DataFrame(features_list)
        y = pd.Series(targets)

        return X, y, sample_weights

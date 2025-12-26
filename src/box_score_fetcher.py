"""
Historical Box Score Fetcher
Fetches and caches box scores for historical games to fix data leakage
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time
import json
from nba_api.stats.endpoints import boxscoretraditionalv2
import pandas as pd


class BoxScoreFetcher:
    """
    Fetches and caches historical box scores.
    Critical for avoiding data leakage in training.
    """
    
    def __init__(self, db_path: str = "data/nba_predictor.db"):
        self.db_path = Path(db_path)
        self._init_box_score_table()
        self.rate_limit_delay = 0.6  # 600ms between requests
        self.last_request_time = 0
    
    def _init_box_score_table(self):
        """Initialize box score cache table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS box_scores (
                game_id TEXT PRIMARY KEY,
                game_date DATE,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_team_stats TEXT,  -- JSON
                away_team_stats TEXT,  -- JSON
                home_player_stats TEXT,  -- JSON array
                away_player_stats TEXT,  -- JSON array
                fetched_at TIMESTAMP
            )
        """)
        
        # Index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_box_scores_date 
            ON box_scores(game_date)
        """)
        
        conn.commit()
        conn.close()
    
    def get_box_score(self, game_id: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get box score for a game.
        
        Args:
            game_id: NBA game ID (e.g., '0022100001')
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary with box score data or None if not found
        """
        # Check cache first
        if not force_refresh:
            cached = self._get_cached_box_score(game_id)
            if cached is not None:
                return cached
        
        # Fetch from NBA API
        try:
            box_score = self._fetch_box_score_from_api(game_id)
            
            if box_score:
                # Cache it
                self._cache_box_score(game_id, box_score)
                return box_score
            
            return None
            
        except Exception as e:
            print(f"Error fetching box score for game {game_id}: {e}")
            return None
    
    def _get_cached_box_score(self, game_id: str) -> Optional[Dict]:
        """Get box score from cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT game_date, home_team_id, away_team_id, 
                   home_team_stats, away_team_stats,
                   home_player_stats, away_player_stats
            FROM box_scores
            WHERE game_id = ?
        """, (game_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            'game_id': game_id,
            'game_date': row[0],
            'home_team_id': row[1],
            'away_team_id': row[2],
            'home_team_stats': json.loads(row[3]),
            'away_team_stats': json.loads(row[4]),
            'home_player_stats': json.loads(row[5]),
            'away_player_stats': json.loads(row[6])
        }
    
    def _fetch_box_score_from_api(self, game_id: str) -> Optional[Dict]:
        """
        Fetch box score from NBA API.
        
        Rate limited to avoid API throttling.
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        try:
            # Fetch box score
            box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            self.last_request_time = time.time()
            
            # Parse team stats
            team_stats = box_score.team_stats.get_data_frame()
            
            if team_stats.empty:
                return None
            
            home_team = team_stats[team_stats['TEAM_ID'] == team_stats.iloc[0]['TEAM_ID']].iloc[0]
            away_team = team_stats[team_stats['TEAM_ID'] == team_stats.iloc[1]['TEAM_ID']].iloc[0]
            
            # Parse player stats
            player_stats = box_score.player_stats.get_data_frame()
            
            home_players = player_stats[player_stats['TEAM_ID'] == home_team['TEAM_ID']]
            away_players = player_stats[player_stats['TEAM_ID'] == away_team['TEAM_ID']]
            
            return {
                'game_id': game_id,
                'game_date': home_team.get('GAME_DATE', ''),
                'home_team_id': int(home_team['TEAM_ID']),
                'away_team_id': int(away_team['TEAM_ID']),
                'home_team_stats': self._extract_team_stats(home_team),
                'away_team_stats': self._extract_team_stats(away_team),
                'home_player_stats': self._extract_player_stats(home_players),
                'away_player_stats': self._extract_player_stats(away_players)
            }
            
        except Exception as e:
            print(f"API error for game {game_id}: {e}")
            return None
    
    def _extract_team_stats(self, team_row) -> Dict:
        """Extract relevant team stats from box score row."""
        return {
            'pts': float(team_row.get('PTS', 0)),
            'fgm': float(team_row.get('FGM', 0)),
            'fga': float(team_row.get('FGA', 0)),
            'fg_pct': float(team_row.get('FG_PCT', 0)),
            'fg3m': float(team_row.get('FG3M', 0)),
            'fg3a': float(team_row.get('FG3A', 0)),
            'fg3_pct': float(team_row.get('FG3_PCT', 0)),
            'ftm': float(team_row.get('FTM', 0)),
            'fta': float(team_row.get('FTA', 0)),
            'ft_pct': float(team_row.get('FT_PCT', 0)),
            'oreb': float(team_row.get('OREB', 0)),
            'dreb': float(team_row.get('DREB', 0)),
            'reb': float(team_row.get('REB', 0)),
            'ast': float(team_row.get('AST', 0)),
            'stl': float(team_row.get('STL', 0)),
            'blk': float(team_row.get('BLK', 0)),
            'tov': float(team_row.get('TOV', 0)),
            'pf': float(team_row.get('PF', 0)),
        }
    
    def _extract_player_stats(self, players_df: pd.DataFrame) -> List[Dict]:
        """Extract player stats from dataframe."""
        players = []
        
        for _, player in players_df.iterrows():
            # Only include players who actually played
            if player.get('MIN') and player['MIN'] != '0:00':
                players.append({
                    'player_id': int(player['PLAYER_ID']),
                    'player_name': str(player['PLAYER_NAME']),
                    'start_position': str(player.get('START_POSITION', '')),
                    'minutes': str(player.get('MIN', '0')),
                    'pts': float(player.get('PTS', 0)),
                    'reb': float(player.get('REB', 0)),
                    'ast': float(player.get('AST', 0)),
                    'fg_pct': float(player.get('FG_PCT', 0) or 0),
                    'fg3_pct': float(player.get('FG3_PCT', 0) or 0),
                })
        
        return players
    
    def _cache_box_score(self, game_id: str, box_score: Dict):
        """Save box score to cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO box_scores
            (game_id, game_date, home_team_id, away_team_id,
             home_team_stats, away_team_stats, home_player_stats, away_player_stats,
             fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            box_score['game_date'],
            box_score['home_team_id'],
            box_score['away_team_id'],
            json.dumps(box_score['home_team_stats']),
            json.dumps(box_score['away_team_stats']),
            json.dumps(box_score['home_player_stats']),
            json.dumps(box_score['away_player_stats']),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def batch_fetch_box_scores(self, game_ids: List[str], 
                               progress_callback=None) -> Dict[str, Dict]:
        """
        Fetch multiple box scores with progress tracking.
        
        Args:
            game_ids: List of game IDs to fetch
            progress_callback: Optional callback(completed, total)
            
        Returns:
            Dictionary mapping game_id -> box_score
        """
        results = {}
        total = len(game_ids)
        
        for i, game_id in enumerate(game_ids):
            # Check cache first
            cached = self._get_cached_box_score(game_id)
            if cached:
                results[game_id] = cached
            else:
                # Fetch from API
                box_score = self.get_box_score(game_id)
                if box_score:
                    results[game_id] = box_score
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM box_scores")
        total_cached = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT MIN(game_date), MAX(game_date) 
            FROM box_scores
        """)
        date_range = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_cached': total_cached,
            'earliest_game': date_range[0] if date_range[0] else 'N/A',
            'latest_game': date_range[1] if date_range[1] else 'N/A'
        }


# Example usage
if __name__ == "__main__":
    fetcher = BoxScoreFetcher()
    
    # Example game ID
    game_id = "0022100001"
    
    print(f"Fetching box score for game {game_id}...")
    box_score = fetcher.get_box_score(game_id)
    
    if box_score:
        print(f"\nGame Date: {box_score['game_date']}")
        print(f"Home Team: {box_score['home_team_id']}")
        print(f"  Points: {box_score['home_team_stats']['pts']}")
        print(f"  Players: {len(box_score['home_player_stats'])}")
        print(f"\nAway Team: {box_score['away_team_id']}")
        print(f"  Points: {box_score['away_team_stats']['pts']}")
        print(f"  Players: {len(box_score['away_player_stats'])}")
    
    # Cache stats
    stats = fetcher.get_cache_stats()
    print(f"\nCache Stats:")
    print(f"  Total Cached: {stats['total_cached']}")
    print(f"  Date Range: {stats['earliest_game']} to {stats['latest_game']}")

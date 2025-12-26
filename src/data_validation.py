"""
Data Validation and Freshness Checking Module
Ensures data is current and valid before making predictions
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
import pandas as pd


class DataFreshnessValidator:
    """Validates data freshness and provides warnings."""
    
    def __init__(self, db_path: str = "data/nba_predictor.db"):
        self.db_path = db_path
        self.warnings = []
    
    def check_data_freshness(self) -> Dict[str, any]:
        """
        Check freshness of various data sources.
        
        Returns:
            Dict with freshness status and warnings
        """
        result = {
            'is_fresh': True,
            'warnings': [],
            'last_updates': {},
            'data_age_days': {}
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check last game date
            try:
                games_query = """
                    SELECT MAX(game_date) as last_game_date
                    FROM games
                """
                last_game = pd.read_sql_query(games_query, conn)
                
                if not last_game.empty and last_game['last_game_date'].iloc[0] is not None:
                    last_game_date_str = last_game['last_game_date'].iloc[0]
                    try:
                        last_game_date = pd.to_datetime(last_game_date_str)
                        # Handle timezone-naive datetime comparisons
                        if last_game_date.tzinfo is None:
                            now = datetime.now()
                        else:
                            now = datetime.now(last_game_date.tzinfo)
                        
                        days_old = (now - last_game_date).days
                        
                        result['last_updates']['games'] = last_game_date.strftime('%Y-%m-%d')
                        result['data_age_days']['games'] = days_old
                        
                        if days_old > 7:
                            result['is_fresh'] = False
                            result['warnings'].append(
                                f"⚠️ Game data is {days_old} days old (last update: {last_game_date.strftime('%Y-%m-%d')}). "
                                "Consider updating historical data."
                            )
                        elif days_old > 2:
                            result['warnings'].append(
                                f"ℹ️ Game data is {days_old} days old. Recent games may not be included."
                            )
                    except (ValueError, TypeError) as e:
                        result['warnings'].append(f"⚠️ Could not parse game date: {last_game_date_str}")
                else:
                    result['is_fresh'] = False
                    result['warnings'].append("⚠️ No game data found in database. Please fetch historical data.")
            except Exception as e:
                result['is_fresh'] = False
                result['warnings'].append(f"⚠️ Error checking game data: {str(e)}")
            
            # Check Elo ratings freshness
            # Try current_elo table first (has last_updated column)
            try:
                elo_query = """
                    SELECT MAX(last_updated) as last_elo_update
                    FROM current_elo
                """
                last_elo = pd.read_sql_query(elo_query, conn)
                
                if not last_elo.empty and last_elo['last_elo_update'].iloc[0] is not None:
                    try:
                        last_elo_date = pd.to_datetime(last_elo['last_elo_update'].iloc[0])
                        # Handle timezone-naive datetime comparisons
                        if last_elo_date.tzinfo is None:
                            now = datetime.now()
                        else:
                            now = datetime.now(last_elo_date.tzinfo)
                        
                        elo_days_old = (now - last_elo_date).days
                        
                        result['last_updates']['elo'] = last_elo_date.strftime('%Y-%m-%d')
                        result['data_age_days']['elo'] = elo_days_old
                        
                        if elo_days_old > 7:
                            result['warnings'].append(
                                f"⚠️ Elo ratings are {elo_days_old} days old. Predictions may be less accurate."
                            )
                    except (ValueError, TypeError) as e:
                        pass  # Skip if date parsing fails
            except Exception:
                # Fallback: use game_date from elo_ratings table (most recent game with elo update)
                try:
                    elo_query = """
                        SELECT MAX(game_date) as last_elo_update
                        FROM elo_ratings
                    """
                    last_elo = pd.read_sql_query(elo_query, conn)
                    
                    if not last_elo.empty and last_elo['last_elo_update'].iloc[0] is not None:
                        try:
                            last_elo_date = pd.to_datetime(last_elo['last_elo_update'].iloc[0])
                            # Handle timezone-naive datetime comparisons
                            if last_elo_date.tzinfo is None:
                                now = datetime.now()
                            else:
                                now = datetime.now(last_elo_date.tzinfo)
                            
                            elo_days_old = (now - last_elo_date).days
                            
                            result['last_updates']['elo'] = last_elo_date.strftime('%Y-%m-%d')
                            result['data_age_days']['elo'] = elo_days_old
                            
                            if elo_days_old > 7:
                                result['warnings'].append(
                                    f"⚠️ Elo ratings are {elo_days_old} days old. Predictions may be less accurate."
                                )
                        except (ValueError, TypeError) as e:
                            pass  # Skip if date parsing fails
                except Exception as e:
                    # Elo tables might not exist or have no data yet - silently skip
                    pass
            
            # Check player stats cache
            try:
                cache_query = """
                    SELECT COUNT(*) as cache_count,
                           MAX(last_updated) as last_cache_update
                    FROM player_stats_cache
                """
                cache_info = pd.read_sql_query(cache_query, conn)
                
                if not cache_info.empty:
                    cache_count = cache_info['cache_count'][0]
                    result['last_updates']['player_cache_size'] = cache_count
                    
                    if cache_info['last_cache_update'][0]:
                        last_cache = pd.to_datetime(cache_info['last_cache_update'][0])
                        cache_days_old = (datetime.now() - last_cache).days
                        
                        result['last_updates']['player_stats'] = last_cache.strftime('%Y-%m-%d')
                        result['data_age_days']['player_stats'] = cache_days_old
                        
                        if cache_days_old > 3:
                            result['warnings'].append(
                                f"ℹ️ Player stats cache is {cache_days_old} days old. "
                                "Fresh stats will be fetched during predictions."
                            )
            except Exception:
                # Player cache table might not exist yet
                pass
            
            conn.close()
            
        except Exception as e:
            result['is_fresh'] = False
            result['warnings'].append(f"❌ Error checking data freshness: {str(e)}")
        
        return result
    
    def get_data_status_summary(self) -> str:
        """Get a formatted summary of data status."""
        freshness = self.check_data_freshness()
        
        if freshness['is_fresh']:
            status = "✅ **Data Status: Fresh**\n\n"
        else:
            status = "⚠️ **Data Status: Needs Update**\n\n"
        
        # Add last update times
        if freshness['last_updates']:
            status += "**Last Updates:**\n"
            for source, date in freshness['last_updates'].items():
                age = freshness['data_age_days'].get(source, 'N/A')
                if isinstance(age, int):
                    status += f"- {source.title()}: {date} ({age} days ago)\n"
                else:
                    status += f"- {source.title()}: {date}\n"
            status += "\n"
        
        # Add warnings
        if freshness['warnings']:
            for warning in freshness['warnings']:
                status += f"{warning}\n\n"
        
        return status
    
    def needs_refresh(self, max_age_days: int = 2) -> Tuple[bool, str]:
        """
        Check if data needs refresh.
        
        Args:
            max_age_days: Maximum acceptable age in days
        
        Returns:
            Tuple of (needs_refresh: bool, reason: str)
        """
        freshness = self.check_data_freshness()
        
        if not freshness['is_fresh']:
            return True, "Data is stale or missing"
        
        # Check game data age
        game_age = freshness['data_age_days'].get('games', 0)
        if game_age > max_age_days:
            return True, f"Game data is {game_age} days old (max: {max_age_days})"
        
        return False, "Data is current"


def check_top_scorer_status(team_id: int, db_path: str = "data/nba_predictor.db") -> Optional[Dict]:
    """
    Check the status of a team's top scorer.
    
    Args:
        team_id: NBA team ID
        db_path: Path to database
    
    Returns:
        Dict with top scorer info or None
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Find top scorer from latest player stats
        query = """
            SELECT player_id, player_name, ppg, is_active
            FROM player_stats_cache
            WHERE team_id = ?
            ORDER BY ppg DESC
            LIMIT 1
        """
        
        cursor = conn.cursor()
        cursor.execute(query, (team_id,))
        row = cursor.fetchone()
        
        if row:
            player_id, player_name, ppg, is_active = row
            
            result = {
                'player_id': player_id,
                'player_name': player_name,
                'ppg': ppg,
                'is_active': bool(is_active),
                'status': '✅ Active' if is_active else '❌ Inactive',
                'impact': 'High' if ppg > 25 else 'Medium' if ppg > 20 else 'Low'
            }
            
            conn.close()
            return result
        
        conn.close()
        return None
        
    except Exception as e:
        print(f"Error checking top scorer: {e}")
        return None


def validate_prediction_data(home_team_id: int, away_team_id: int, 
                             db_path: str = "data/nba_predictor.db") -> Dict[str, any]:
    """
    Validate that necessary data exists for prediction.
    
    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        db_path: Path to database
    
    Returns:
        Dict with validation results
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check if teams have Elo ratings
        elo_query = """
            SELECT team_id, elo
            FROM elo_ratings
            WHERE team_id IN (?, ?)
        """
        elo_df = pd.read_sql_query(elo_query, conn, params=(home_team_id, away_team_id))
        
        if len(elo_df) < 2:
            result['valid'] = False
            result['errors'].append("Missing Elo ratings for one or both teams")
        
        # Check if teams have recent games
        games_query = """
            SELECT COUNT(*) as game_count
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
               OR (home_team_id = ? OR away_team_id = ?)
        """
        cursor = conn.cursor()
        cursor.execute(games_query, (home_team_id, home_team_id, away_team_id, away_team_id))
        game_counts = cursor.fetchone()
        
        if game_counts[0] < 10:
            result['warnings'].append(
                f"Limited historical data ({game_counts[0]} games). Predictions may be less reliable."
            )
        
        conn.close()
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Validation error: {str(e)}")
    
    return result

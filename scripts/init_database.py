"""
Initialize the database for NBA Predictor
Creates SQLite database and tables for storing game data
"""

import sqlite3
import os
from pathlib import Path


def init_database(db_path='data/nba_predictor.db'):
    """Initialize the database with required tables"""
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Use NBADataFetcher to initialize database with correct schema
    from src.data_fetcher import NBADataFetcher
    fetcher = NBADataFetcher(db_path)
    
    print(f"Database initialized successfully at {db_path}")
    print("Tables created: games, predictions, team_game_logs, player_game_logs, elo_ratings, current_elo")


if __name__ == "__main__":
    init_database()
    print("\nDatabase setup complete!")


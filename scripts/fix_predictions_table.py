"""
Fix Predictions Table Schema
Recreates the predictions table with the correct schema including 'correct' column
"""

import sqlite3
from pathlib import Path

def fix_predictions_table():
    """Recreate predictions table with correct schema"""
    db_path = Path(__file__).parent.parent / 'data' / 'nba_predictor.db'

    print("=" * 60)
    print("FIXING PREDICTIONS TABLE SCHEMA")
    print("=" * 60)
    print()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop existing table
    print("Dropping existing predictions table...")
    cursor.execute('DROP TABLE IF EXISTS predictions')
    print("OK - Table dropped")
    print()

    # Create new table with correct schema
    print("Creating new predictions table with correct schema...")
    cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            predicted_winner TEXT,
            predicted_home_prob REAL,
            predicted_away_prob REAL,
            confidence REAL,
            actual_winner TEXT,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            correct INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("OK - Table created with 'correct' column")
    print()

    conn.commit()
    conn.close()

    print("=" * 60)
    print("PREDICTIONS TABLE FIXED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("The predictions table now has the correct schema.")
    print("You can now use the Prediction Tracker tab in the app.")
    print()

if __name__ == "__main__":
    fix_predictions_table()

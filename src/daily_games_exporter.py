"""
Export daily games data for GitHub Pages publishing interface.
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DailyGamesExporter:
    """Export today's predictions to JSON for web publishing interface"""

    def __init__(self, db_path: str = 'data/nba_predictor.db'):
        """
        Initialize exporter.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

    def get_today_predictions(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get today's predictions from database.

        Args:
            date: Date string (YYYY-MM-DD). If None, uses today.

        Returns:
            List of prediction dictionaries
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                game_date,
                home_team,
                away_team,
                predicted_winner,
                predicted_home_prob,
                predicted_away_prob,
                confidence,
                home_odds,
                away_odds
            FROM predictions
            WHERE game_date = ? AND actual_winner IS NULL
            ORDER BY confidence DESC, home_team, away_team
        """, (date,))

        predictions = []
        for row in cursor.fetchall():
            game_date, home_team, away_team, predicted_winner, pred_home_prob, pred_away_prob, confidence, home_odds, away_odds = row

            # Use stored odds if available, otherwise calculate from probabilities
            if home_odds is None:
                home_odds = round(1 / pred_home_prob, 2) if pred_home_prob > 0 else 99.0
            if away_odds is None:
                away_odds = round(1 / pred_away_prob, 2) if pred_away_prob > 0 else 99.0

            predictions.append({
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': predicted_winner,
                'predicted_home_prob': pred_home_prob,
                'predicted_away_prob': pred_away_prob,
                'home_odds': home_odds,
                'away_odds': away_odds,
                'confidence': confidence
            })

        conn.close()
        return predictions

    def export_games_for_publishing(self, date: Optional[str] = None, output_path: str = 'docs/pending_games.json') -> bool:
        """
        Export today's games to JSON for the web interface.

        Args:
            date: Date string (YYYY-MM-DD). If None, uses today.
            output_path: Path to save JSON file

        Returns:
            True if export successful, False otherwise
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Exporting games for {date}...")

            # Get today's predictions
            predictions = self.get_today_predictions(date)
            logger.info(f"Found {len(predictions)} predictions for {date}")

            # Format for web interface
            games_data = []
            for pred in predictions:
                game_id = f"{pred['away_team']}_vs_{pred['home_team']}_{date}".replace(' ', '_')

                games_data.append({
                    'id': game_id,
                    'matchup': f"{pred['away_team']} @ {pred['home_team']}",
                    'home_team': pred['home_team'],
                    'away_team': pred['away_team'],
                    'predicted_winner': pred['predicted_winner'],
                    'predicted_home_prob': pred['predicted_home_prob'],
                    'predicted_away_prob': pred['predicted_away_prob'],
                    'home_odds': pred['home_odds'],
                    'away_odds': pred['away_odds'],
                    'confidence': pred['confidence'],
                    'date': date,
                    'published': False  # Track if already published
                })

            # Create output data
            output_data = {
                'date': date,
                'generated_at': datetime.now().isoformat(),
                'games': games_data
            }

            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Exported {len(games_data)} games to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export games: {e}", exc_info=True)
            return False


if __name__ == '__main__':
    # Test export
    import sys

    logging.basicConfig(level=logging.INFO)

    exporter = DailyGamesExporter()

    # Use today's date or provided date
    date = sys.argv[1] if len(sys.argv) > 1 else None

    success = exporter.export_games_for_publishing(date)

    if success:
        print("✓ Export completed successfully")
    else:
        print("✗ Export failed")
        sys.exit(1)

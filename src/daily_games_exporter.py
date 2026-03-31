"""
Export daily games data for GitHub Pages publishing interface.
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Tricode to full team name mapping
TRICODE_TO_NAME = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}


class DailyGamesExporter:
    """Export today's predictions to JSON for web publishing interface"""

    def __init__(self, db_path: str = 'data/nba_predictor.db'):
        """
        Initialize exporter.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

    def _to_full_name(self, team: str) -> str:
        """Convert tricode to full team name, or return as-is if already full name."""
        if team in TRICODE_TO_NAME:
            return TRICODE_TO_NAME[team]
        return team  # Already a full name

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

    def _format_predictions(self, predictions: List[Dict], date: str) -> List[Dict]:
        """Format raw prediction dicts into web-interface-ready dicts."""
        games_data = []
        for pred in predictions:
            home_team = self._to_full_name(pred['home_team'])
            away_team = self._to_full_name(pred['away_team'])
            predicted_winner = self._to_full_name(pred['predicted_winner'])
            game_date = pred.get('game_date', date)

            game_id = f"{away_team}_vs_{home_team}_{game_date}".replace(' ', '_')

            games_data.append({
                'id': game_id,
                'matchup': f"{away_team} @ {home_team}",
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': predicted_winner,
                'predicted_home_prob': pred['predicted_home_prob'],
                'predicted_away_prob': pred['predicted_away_prob'],
                'home_odds': pred['home_odds'],
                'away_odds': pred['away_odds'],
                'confidence': pred['confidence'],
                'date': game_date,
                'published': False
            })
        return games_data

    def export_games_for_publishing(self, date: Optional[str] = None, output_path: str = 'docs/pending_games.json') -> bool:
        """
        Export a single day's games to JSON for the web interface.

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

            predictions = self.get_today_predictions(date)
            logger.info(f"Found {len(predictions)} predictions for {date}")

            games_data = self._format_predictions(predictions, date)

            output_data = {
                'date': date,
                'generated_at': datetime.now().isoformat(),
                'games': games_data
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[OK] Exported {len(games_data)} games to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export games: {e}", exc_info=True)
            return False

    def export_today_and_tomorrow(self, today_str: str, tomorrow_str: str,
                                   output_path: str = 'docs/pending_games.json') -> bool:
        """
        Export today's AND tomorrow's predictions into a single JSON file.
        Today's games are listed first, tomorrow's games second.

        Args:
            today_str: Today's date (YYYY-MM-DD)
            tomorrow_str: Tomorrow's date (YYYY-MM-DD)
            output_path: Path to save JSON file

        Returns:
            True if export successful, False otherwise
        """
        try:
            logger.info(f"Exporting games for {today_str} and {tomorrow_str}...")

            today_preds = self.get_today_predictions(today_str)
            tomorrow_preds = self.get_today_predictions(tomorrow_str)

            logger.info(f"Found {len(today_preds)} predictions for today, "
                        f"{len(tomorrow_preds)} for tomorrow")

            today_games = self._format_predictions(today_preds, today_str)
            tomorrow_games = self._format_predictions(tomorrow_preds, tomorrow_str)

            output_data = {
                'date': today_str,
                'generated_at': datetime.now().isoformat(),
                'today': today_str,
                'tomorrow': tomorrow_str,
                'games': today_games + tomorrow_games,  # today first, then tomorrow
                'games_today': today_games,
                'games_tomorrow': tomorrow_games,
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[OK] Exported {len(today_games)}+{len(tomorrow_games)} "
                        f"games to {output_path}")
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
        print("[OK] Export completed successfully")
    else:
        print("[ERROR] Export failed")
        sys.exit(1)

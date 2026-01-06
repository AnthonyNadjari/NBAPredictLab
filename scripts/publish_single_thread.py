#!/usr/bin/env python3
"""
Publish a single game's prediction thread to Twitter.

This script is called by GitHub Actions when a user clicks the publish button.
It reads the game data, generates images, and posts to Twitter.

Usage:
    python scripts/publish_single_thread.py GAME_ID

Example:
    python scripts/publish_single_thread.py LAL_vs_BOS_2026-01-03
"""

import sys
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.twitter_integration import (
    create_fresh_twitter_client,
    create_twitter_thread,
    format_prediction_tweet
)
from src.predictor import NBAPredictor
from src.data_fetcher import NBADataFetcher
from daily_auto_prediction import DailyPredictionAutomation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_game_from_json(game_id: str) -> Optional[Dict]:
    """Load game data from pending_games.json"""
    try:
        with open('docs/pending_games.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        for game in data.get('games', []):
            if game['id'] == game_id:
                return game

        logger.error(f"Game {game_id} not found in pending_games.json")
        return None

    except Exception as e:
        logger.error(f"Failed to load game from JSON: {e}")
        return None


def get_prediction_from_db(home_team: str, away_team: str, game_date: str) -> Optional[Dict]:
    """Get full prediction data from database"""
    try:
        conn = sqlite3.connect('data/nba_predictor.db')
        cursor = conn.cursor()

        # First check which columns exist in the table
        cursor.execute("PRAGMA table_info(predictions)")
        columns = {row[1] for row in cursor.fetchall()}
        has_features = 'prediction_features' in columns

        # Build query based on available columns
        if has_features:
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
                    away_odds,
                    prediction_features
                FROM predictions
                WHERE home_team = ? AND away_team = ? AND game_date = ?
            """, (home_team, away_team, game_date))
        else:
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
                WHERE home_team = ? AND away_team = ? AND game_date = ?
            """, (home_team, away_team, game_date))

        row = cursor.fetchone()
        conn.close()

        if not row:
            logger.error(f"No prediction found in database for {away_team} @ {home_team} on {game_date}")
            return None

        # Unpack based on whether features column exists
        if has_features:
            game_date, home_team, away_team, predicted_winner, pred_home_prob, pred_away_prob, \
            confidence, home_odds, away_odds, prediction_features = row
            features = json.loads(prediction_features) if prediction_features else {}
        else:
            game_date, home_team, away_team, predicted_winner, pred_home_prob, pred_away_prob, \
            confidence, home_odds, away_odds = row
            features = {}

        return {
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': predicted_winner,
            'predicted_home_prob': pred_home_prob,
            'predicted_away_prob': pred_away_prob,
            'confidence': confidence,
            'home_odds': home_odds if home_odds else round(1 / pred_home_prob, 2),
            'away_odds': away_odds if away_odds else round(1 / pred_away_prob, 2),
            'features': features
        }

    except Exception as e:
        logger.error(f"Failed to get prediction from database: {e}", exc_info=True)
        return None


def format_thread_tweets_full(prediction: Dict) -> tuple:
    """
    Format prediction data into full Twitter thread format using DailyPredictionAutomation.
    This matches the same rich format used by Streamlit.

    Returns:
        Tuple of (texts, image_paths) for the thread
    """
    try:
        # Create a temporary DailyPredictionAutomation instance to use its format_twitter_thread method
        temp_daily = DailyPredictionAutomation(
            db_path="data/nba_predictor.db",
            model_dir="models",
            dry_run=False
        )

        # Prepare prediction dict in the format expected by format_twitter_thread
        # Need to convert from DB format to the format expected by format_twitter_thread
        home = prediction['home_team']
        away = prediction['away_team']

        # Determine prediction direction
        if prediction['predicted_winner'] == home:
            pred_direction = 'home'
        else:
            pred_direction = 'away'

        prediction_for_thread = {
            'home_team': home,
            'away_team': away,
            'prediction': pred_direction,
            'confidence': prediction['confidence'],
            'home_win_probability': prediction['predicted_home_prob'],
            'away_win_probability': prediction['predicted_away_prob'],
            'home_odds': prediction.get('home_odds'),  # Pass actual betting odds
            'away_odds': prediction.get('away_odds'),  # Pass actual betting odds
            'features': prediction.get('features', {}),
            'pattern_adjustments': prediction.get('pattern_adjustments', []),
        }

        # Use the same format_twitter_thread method as daily prediction
        thread_texts, thread_image_paths = temp_daily.format_twitter_thread(prediction_for_thread)

        logger.info(f"Generated full thread with {len(thread_texts)} tweets and {len([p for p in thread_image_paths if p])} images")

        return thread_texts, thread_image_paths

    except Exception as e:
        logger.error(f"Failed to format full thread, falling back to simple format: {e}", exc_info=True)
        # Fallback to simple format
        return format_thread_tweets_simple(prediction), []


def format_thread_tweets_simple(prediction: Dict) -> list:
    """Simple fallback format for Twitter thread"""
    try:
        home = prediction['home_team']
        away = prediction['away_team']
        winner = prediction['predicted_winner']
        confidence = prediction['confidence'] * 100
        home_prob = prediction['predicted_home_prob'] * 100
        away_prob = prediction['predicted_away_prob'] * 100
        home_odds = prediction['home_odds']
        away_odds = prediction['away_odds']

        tweet1 = f"""🏀 NBA Prediction
{away} @ {home}

🎯 Prediction: {winner}
📊 Confidence: {confidence:.1f}%

Probabilities:
{home}: {home_prob:.1f}%
{away}: {away_prob:.1f}%

Odds: {away_odds:.2f} / {home_odds:.2f}"""

        return [tweet1]

    except Exception as e:
        logger.error(f"Failed to format thread tweets: {e}")
        return [f"🏀 {prediction['away_team']} @ {prediction['home_team']}\nPrediction: {prediction['predicted_winner']}"]


def publish_thread(game_id: str) -> bool:
    """
    Publish a Twitter thread for a specific game.

    Args:
        game_id: Game identifier (e.g., "LAL_vs_BOS_2026-01-03")

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"=" * 60)
        logger.info(f"Publishing thread for game: {game_id}")
        logger.info(f"=" * 60)

        # Load game from JSON
        game = load_game_from_json(game_id)
        if not game:
            logger.error("Failed to load game data")
            return False

        logger.info(f"Game: {game['matchup']}")
        logger.info(f"Date: {game['date']}")
        logger.info(f"Predicted winner: {game['predicted_winner']}")

        # Get full prediction from database
        prediction = get_prediction_from_db(
            game['home_team'],
            game['away_team'],
            game['date']
        )

        if not prediction:
            logger.error("Failed to get prediction from database")
            return False

        # Format tweets using the full thread format (same as Streamlit)
        tweets, image_paths = format_thread_tweets_full(prediction)
        logger.info(f"Formatted {len(tweets)} tweets for thread with {len([p for p in image_paths if p])} images")

        # Create Twitter client
        logger.info("Creating Twitter client...")

        # Debug: Log environment variables for Twitter credentials
        import os
        tw_api_key = os.getenv('TW_API_KEY', '')
        tw_access_token = os.getenv('TW_ACCESS_TOKEN', '')
        logger.info(f"🔍 DEBUG - Environment variables check:")
        logger.info(f"   TW_API_KEY present: {bool(tw_api_key)} (length: {len(tw_api_key)}, starts with: {tw_api_key[:10] if tw_api_key else 'N/A'}...)")
        logger.info(f"   TW_ACCESS_TOKEN present: {bool(tw_access_token)} (length: {len(tw_access_token)}, starts with: {tw_access_token[:20] if tw_access_token else 'N/A'}...)")
        logger.info(f"   TW_API_SECRET present: {bool(os.getenv('TW_API_SECRET'))}")
        logger.info(f"   TW_ACCESS_SECRET present: {bool(os.getenv('TW_ACCESS_SECRET'))}")
        logger.info(f"   TW_DRY_RUN value: {os.getenv('TW_DRY_RUN', 'not set')}")

        twitter_clients = create_fresh_twitter_client()

        # Post thread
        logger.info("Posting thread to Twitter...")
        responses = create_twitter_thread(
            twitter_clients,
            tweets,
            image_paths=image_paths if image_paths else None,
            dry_run=False  # Actually post to Twitter
        )

        logger.info(f"✓ Thread posted successfully!")
        logger.info(f"✓ Posted {len(responses)} tweets")

        # Cleanup temp images
        for img_path in (image_paths or []):
            if img_path and Path(img_path).exists():
                Path(img_path).unlink()
                logger.info(f"✓ Cleaned up temp image: {img_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to publish thread: {e}", exc_info=True)
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/publish_single_thread.py GAME_ID")
        print("Example: python scripts/publish_single_thread.py LAL_vs_BOS_2026-01-03")
        sys.exit(1)

    game_id = sys.argv[1]

    logger.info(f"NBA Predictor - Single Thread Publisher")
    logger.info(f"Started at: {datetime.now()}")

    success = publish_thread(game_id)

    if success:
        logger.info("✓ SUCCESS: Thread published successfully")
        sys.exit(0)
    else:
        logger.error("✗ FAILED: Could not publish thread")
        sys.exit(1)


if __name__ == '__main__':
    main()

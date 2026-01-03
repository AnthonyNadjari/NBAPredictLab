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
from src.explainability_viz import create_game_visualization
from src.predictor import NBAPredictor
from src.data_fetcher import NBADataFetcher

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

        row = cursor.fetchone()
        conn.close()

        if not row:
            logger.error(f"No prediction found in database for {away_team} @ {home_team} on {game_date}")
            return None

        game_date, home_team, away_team, predicted_winner, pred_home_prob, pred_away_prob, \
        confidence, home_odds, away_odds, prediction_features = row

        # Parse features JSON
        features = json.loads(prediction_features) if prediction_features else {}

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


def format_thread_tweets(prediction: Dict) -> list:
    """Format prediction data into Twitter thread format"""
    try:
        # Tweet 1: Main prediction
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

        # Tweet 2: Key factors (if features available)
        features = prediction.get('features', {})
        if features:
            key_factors = []

            # Extract relevant features
            if 'home_win_pct' in features:
                key_factors.append(f"Home W%: {features['home_win_pct']*100:.1f}%")
            if 'away_win_pct' in features:
                key_factors.append(f"Away W%: {features['away_win_pct']*100:.1f}%")
            if 'home_avg_points' in features:
                key_factors.append(f"Home PPG: {features['home_avg_points']:.1f}")
            if 'away_avg_points' in features:
                key_factors.append(f"Away PPG: {features['away_avg_points']:.1f}")

            if key_factors:
                tweet2 = "📈 Key Stats:\n" + "\n".join(f"• {f}" for f in key_factors[:4])
                return [tweet1, tweet2]

        return [tweet1]

    except Exception as e:
        logger.error(f"Failed to format thread tweets: {e}")
        return [f"🏀 {prediction['away_team']} @ {prediction['home_team']}\nPrediction: {prediction['predicted_winner']}"]


def generate_prediction_image(prediction: Dict) -> Optional[str]:
    """Generate visualization image for prediction"""
    try:
        logger.info("Generating prediction visualization...")

        # Create image using existing visualization code
        image_path = f"temp_{prediction['home_team']}_{prediction['away_team']}.png"

        # Use the explainability_viz module to create image
        fig = create_game_visualization(prediction)

        if fig:
            fig.write_image(image_path, width=1200, height=800)
            logger.info(f"✓ Image saved to {image_path}")
            return image_path
        else:
            logger.warning("No visualization generated")
            return None

    except Exception as e:
        logger.error(f"Failed to generate image: {e}", exc_info=True)
        return None


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

        # Generate image
        image_path = generate_prediction_image(prediction)

        # Format tweets
        tweets = format_thread_tweets(prediction)
        logger.info(f"Formatted {len(tweets)} tweets for thread")

        # Create Twitter client
        logger.info("Creating Twitter client...")
        twitter_clients = create_fresh_twitter_client()

        # Post thread
        logger.info("Posting thread to Twitter...")
        responses = create_twitter_thread(
            twitter_clients,
            tweets,
            image_paths=[image_path] if image_path else None,
            dry_run=False  # Actually post to Twitter
        )

        logger.info(f"✓ Thread posted successfully!")
        logger.info(f"✓ Posted {len(responses)} tweets")

        # Cleanup temp image
        if image_path and Path(image_path).exists():
            Path(image_path).unlink()
            logger.info(f"✓ Cleaned up temp image: {image_path}")

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

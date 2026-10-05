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

import os
import sys
import time
import json
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Legacy modules (xgboost, plotly, tweepy...) are imported lazily: the server that posts
# through the browser doesn't have them.

# Full name to tricode mapping (reverse of TRICODE_TO_NAME)
NAME_TO_TRICODE = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',
}


def to_tricode(team: str) -> str:
    """Convert full team name to tricode, or return as-is if already tricode."""
    return NAME_TO_TRICODE.get(team, team)

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

        for game in data.get('games', []) + data.get('specials', []):
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

        # Try both full names (Streamlit format) and tricodes (legacy format)
        # First try with full names as-is
        home_team_db = home_team
        away_team_db = away_team
        logger.info(f"DEBUG - DB lookup (full names): home='{home_team_db}', away='{away_team_db}'")

        # First check which columns exist in the table
        cursor.execute("PRAGMA table_info(predictions)")
        columns = {row[1] for row in cursor.fetchall()}
        # Check for features_json column (the actual column name in the database)
        has_features = 'features_json' in columns

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
                    features_json
                FROM predictions
                WHERE home_team = ? AND away_team = ? AND game_date = ?
            """, (home_team_db, away_team_db, game_date))
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
            """, (home_team_db, away_team_db, game_date))

        row = cursor.fetchone()

        # If not found with full names, try with tricodes
        if not row:
            home_team_db = to_tricode(home_team)
            away_team_db = to_tricode(away_team)
            logger.info(f"DEBUG - Retrying with tricodes: home='{home_team_db}', away='{away_team_db}'")

            if has_features:
                cursor.execute("""
                    SELECT
                        game_date, home_team, away_team, predicted_winner,
                        predicted_home_prob, predicted_away_prob, confidence,
                        home_odds, away_odds, features_json
                    FROM predictions
                    WHERE home_team = ? AND away_team = ? AND game_date = ?
                """, (home_team_db, away_team_db, game_date))
            else:
                cursor.execute("""
                    SELECT
                        game_date, home_team, away_team, predicted_winner,
                        predicted_home_prob, predicted_away_prob, confidence,
                        home_odds, away_odds
                    FROM predictions
                    WHERE home_team = ? AND away_team = ? AND game_date = ?
                """, (home_team_db, away_team_db, game_date))
            row = cursor.fetchone()

        conn.close()

        if not row:
            logger.error(f"No prediction found in database for {away_team} @ {home_team} on {game_date}")
            return None

        # Unpack based on whether features column exists
        if has_features:
            game_date, home_team, away_team, predicted_winner, pred_home_prob, pred_away_prob, \
            confidence, home_odds, away_odds, features_json = row
            features = json.loads(features_json) if features_json else {}
            logger.info(f"✓ Loaded {len(features)} features from database")
            # Debug: Log key features to verify they're not zero
            if features:
                logger.info(f"   DEBUG - home_elo: {features.get('home_elo', 'MISSING')}")
                logger.info(f"   DEBUG - away_elo: {features.get('away_elo', 'MISSING')}")
                logger.info(f"   DEBUG - home_last10_offensive_rating: {features.get('home_last10_offensive_rating', 'MISSING')}")
                logger.info(f"   DEBUG - away_last10_offensive_rating: {features.get('away_last10_offensive_rating', 'MISSING')}")
            else:
                logger.warning("   DEBUG - features_json was empty or null!")
        else:
            game_date, home_team, away_team, predicted_winner, pred_home_prob, pred_away_prob, \
            confidence, home_odds, away_odds = row
            features = {}
            logger.warning("⚠ No features column found in database - charts will be empty")

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


def format_thread_tweets_full(prediction: Dict, with_images: bool = True) -> tuple:
    """
    Format prediction data into full Twitter thread format using DailyPredictionAutomation.
    This matches the same rich format used by Streamlit.

    Returns:
        Tuple of (texts, image_paths) for the thread
    """
    try:
        # Create a temporary DailyPredictionAutomation instance to use its format_twitter_thread method
        from daily_auto_prediction import DailyPredictionAutomation
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
            'predicted_winner': prediction['predicted_winner'],
            'confidence': prediction['confidence'],
            'home_win_probability': prediction['predicted_home_prob'],
            'away_win_probability': prediction['predicted_away_prob'],
            'home_odds': prediction.get('home_odds'),
            'away_odds': prediction.get('away_odds'),
            'features': prediction.get('features', {}),
            'pattern_adjustments': prediction.get('pattern_adjustments', []),
        }

        # Use the same format_twitter_thread method as daily prediction
        thread_texts, thread_image_paths = temp_daily.format_twitter_thread(prediction_for_thread, with_images=with_images)

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


def build_posts(game: Dict, prediction: Optional[Dict]) -> list:
    """[{text, card}] from the thread factory (or the pre-built weekly recap)."""
    if game.get('type') in ('weekly', 'announcement'):
        return game['thread']
    from src.social.thread import build_thread, record_line
    return build_thread(prediction, record_line('data/nba_predictor.db', since='2026-10-01'))['tweets']


def apply_text_overrides(posts: list) -> list:
    """Texts edited in the control panel arrive as THREAD_TEXTS_JSON (one string per tweet).

    An empty string removes that tweet (and its card)."""
    import os
    raw = os.getenv('THREAD_TEXTS_JSON', '').strip()
    if not raw:
        return posts
    texts = json.loads(raw)
    if not isinstance(texts, list) or len(texts) != len(posts) or not all(isinstance(t, str) for t in texts):
        raise ValueError('THREAD_TEXTS_JSON must be a list of strings, one per tweet')
    out = []
    for post, text in zip(posts, texts):
        text = text.strip()
        if not text:
            continue
        if len(text) > 280:
            raise ValueError(f'Edited tweet over 280 characters: {text[:40]}...')
        out.append({**post, 'text': text})
    if not out:
        raise ValueError('All tweets were removed')
    logger.info(f"Using {len(out)} edited tweet text(s) from the control panel")
    return out


def post_thread_browser(posts: list, image_paths: list, dry_run: bool) -> list:
    """Post through the logged-in browser (vision/publisher.py): no API, no per-post cost."""
    sys.path.insert(0, str(PROJECT_ROOT / 'vision'))
    from auth import launch_and_auth
    from publisher import post_thread as browser_post_thread
    res = launch_and_auth()
    if res[0] is None:
        raise RuntimeError(f"X login failed in the browser ({res[-1]})")
    pw, _, context, _page = res
    try:
        ids, error = browser_post_thread(
            context, [{'text': p['text'], 'image': img} for p, img in zip(posts, image_paths)], dry_run)
    finally:
        try:
            context.close()
            pw.stop()
        except Exception:
            pass
    if error:
        logger.error(f"Browser posting stopped: {error}")
        os.environ['THREAD_ERROR'] = error
    return [f"dry{i}" for i in range(len(posts))] if dry_run and not error else ids


def post_thread(posts: list, image_paths: list, dry_run: bool) -> list:
    """Post tweet by tweet; returns posted tweet ids (stops at the first failure)."""
    if os.getenv('PUBLISH_BACKEND', 'api') == 'browser':
        return post_thread_browser(posts, image_paths, dry_run)
    if dry_run:
        for i, p in enumerate(posts):
            logger.info(f"[DRY RUN] tweet {i + 1}/{len(posts)} ({len(p['text'])} chars, "
                        f"image={bool(image_paths[i])}): {p['text']!r}")
        return [f"dry{i}" for i in range(len(posts))]
    from src.twitter_integration import create_fresh_twitter_client, _upload_media
    clients = create_fresh_twitter_client()
    client_v2, api_v1 = clients.get('client_v2'), clients.get('api_v1')
    if not client_v2:
        raise RuntimeError(f"Twitter client unavailable: {clients.get('auth_status')}")
    ids, prev = [], None
    for i, p in enumerate(posts):
        kwargs = {'text': p['text']}
        if image_paths[i] and api_v1:
            media_id = _upload_media(api_v1, image_paths[i])
            if media_id:
                kwargs['media_ids'] = [media_id]
            else:
                logger.warning(f"Image upload failed for tweet {i + 1}: posting text only")
        if prev:
            kwargs['in_reply_to_tweet_id'] = prev
        try:
            resp = client_v2.create_tweet(**kwargs)
        except Exception as e:
            logger.error(f"Tweet {i + 1}/{len(posts)} failed: {e}")
            break
        prev = (resp.data or {}).get('id')
        ids.append(prev)
        logger.info(f"Posted tweet {i + 1}/{len(posts)}: {prev}")
        time.sleep(1.5)
    return ids


def publish_thread(game_id: str) -> bool:
    """Publish the thread for one game (or a special post such as the weekly recap)."""
    import os
    game = load_game_from_json(game_id)
    if not game:
        return False
    if game.get('published'):
        logger.error(f"Already published at {game.get('published_at')} - refusing to post twice")
        return False
    logger.info(f"Publishing: {game.get('matchup') or game.get('title') or game_id}")

    prediction = None
    if game.get('type') not in ('weekly', 'announcement'):
        prediction = get_prediction_from_db(game['home_team'], game['away_team'], game['date'])
        if not prediction:
            logger.error("No prediction in the database for this game")
            return False

    try:
        posts = apply_text_overrides(build_posts(game, prediction))
        from src.social.render import render_cards
        image_paths = render_cards([p.get('card') for p in posts])
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Thread factory failed ({e}); falling back to the legacy format", exc_info=True)
        texts, imgs = format_thread_tweets_full(prediction)
        posts = [{'text': t} for t in texts]
        image_paths = [(imgs[i] if imgs and i < len(imgs) else None) for i in range(len(texts))]

    logger.info(f"{len(posts)} tweets, {sum(1 for p in image_paths if p)} images")
    if os.getenv('PUBLISH_BACKEND', 'api') != 'browser':
        for var in ('TW_API_KEY', 'TW_API_SECRET', 'TW_ACCESS_TOKEN', 'TW_ACCESS_SECRET'):
            logger.info(f"   {var}: {'set' if os.getenv(var) else 'MISSING'}")
    dry = os.getenv('TW_DRY_RUN', 'false').lower() in ('1', 'true', 'yes')
    ids = post_thread(posts, image_paths, dry)

    result = {'posted': len(ids), 'total': len(posts), 'first_id': ids[0] if ids else None,
              'dry_run': dry, 'error': os.getenv('THREAD_ERROR') or None}
    Path(os.getenv('THREAD_RESULT_FILE', 'thread_result.json')).write_text(json.dumps(result))
    if not ids:
        logger.error("Nothing was posted")
        return False
    if len(ids) < len(posts):
        # The opener is live: report success so the game is marked published and nobody re-posts it.
        logger.warning(f"Partial thread: {len(ids)}/{len(posts)} tweets posted")
    return True


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

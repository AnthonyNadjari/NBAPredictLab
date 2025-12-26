#!/usr/bin/env python3
"""
Daily NBA Prediction Automation Script

This script runs independently of the Streamlit app and:
1. Fetches today's NBA matches
2. Generates predictions for all matches
3. Filters predictions with odds > 1.3
4. Selects the highest confidence prediction
5. Posts to Twitter as a thread

Usage:
    python daily_auto_prediction.py [--dry-run] [--verbose]

Arguments:
    --dry-run: Test mode - no actual Twitter posting
    --verbose: Enable debug logging
    --date YYYY-MM-DD: Override date (default: today)
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.predictor import NBAPredictor
from src.data_fetcher import NBADataFetcher
from src.twitter_integration import (
    create_fresh_twitter_client,
    load_credentials_from_env,
    create_twitter_thread,
    format_prediction_tweet
)
from src.betting_odds import calculate_betting_odds, get_fair_odds
from src.model_feedback_system import ModelFeedbackSystem
from src.email_reporter import EmailReporter
import sqlite3
import time
import pandas as pd


class DailyPredictionAutomation:
    """Automated daily NBA prediction and Twitter posting system"""

    def __init__(
        self,
        db_path: str = 'data/nba_predictor.db',
        model_dir: str = 'models',
        log_dir: str = 'logs',
        dry_run: bool = False
    ):
        """
        Initialize the automation system

        Args:
            db_path: Path to SQLite database
            model_dir: Directory containing trained models
            log_dir: Directory for log files
            dry_run: If True, simulate posting without actual Twitter API calls
        """
        self.db_path = db_path
        self.model_dir = model_dir
        self.log_dir = Path(log_dir)
        self.dry_run = dry_run

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)

        # Initialize logger
        self.logger = self._setup_logger()

        # Initialize components (lazy loading)
        self.predictor: Optional[NBAPredictor] = None
        self.fetcher: Optional[NBADataFetcher] = None
        self.api_clients: Optional[Dict] = None

    def _setup_logger(self) -> logging.Logger:
        """Configure logging with both file and console handlers"""
        logger = logging.getLogger('DailyPredictionBot')
        logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # File handler - daily rotation (UTF-8 for unicode characters)
        log_file = self.log_dir / f"daily_predictions_{datetime.now().strftime('%Y%m')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Console handler with UTF-8 encoding (Windows fix)
        import io
        if sys.platform == 'win32':
            # On Windows, wrap stdout with UTF-8 encoding to handle special characters
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter with timestamps
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def initialize_components(self) -> bool:
        """
        Initialize predictor, data fetcher, and Twitter client

        Returns:
            True if all components initialized successfully, False otherwise
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("Initializing Daily NBA Prediction Automation")
            self.logger.info("=" * 80)

            # Initialize predictor
            self.logger.info("Loading NBA predictor model...")
            self.predictor = NBAPredictor(
                db_path=self.db_path,
                model_dir=self.model_dir
            )
            self.predictor.load_model()
            self.logger.info("✓ Predictor model loaded successfully")

            # Initialize data fetcher
            self.logger.info("Initializing data fetcher...")
            self.fetcher = NBADataFetcher(db_path=self.db_path)
            self.logger.info("✓ Data fetcher initialized")

            # Initialize Twitter client
            if not self.dry_run:
                self.logger.info("Connecting to Twitter API...")
                self.api_clients = create_fresh_twitter_client()

                # Check if client was created successfully
                if not self.api_clients or not self.api_clients.get('client_v2'):
                    self.logger.error("✗ Twitter client creation failed")
                    return False

                # Check auth status (verified key, not authenticated)
                auth_status = self.api_clients.get('auth_status', {})
                if not auth_status.get('verified', False):
                    self.logger.error("✗ Twitter authentication failed")
                    self.logger.error(f"Auth error: {auth_status.get('error', 'Unknown')}")
                    return False

                self.logger.info("✓ Twitter client authenticated")
            else:
                self.logger.info("ℹ Dry-run mode: Skipping Twitter authentication")

            return True

        except Exception as e:
            self.logger.error(f"✗ Component initialization failed: {e}", exc_info=True)
            return False

    def fetch_todays_games(self, target_date: Optional[str] = None) -> List[Dict]:
        """
        Fetch NBA games for the target date
        First tries API, then falls back to database (which was refreshed in Step 0.5)

        Args:
            target_date: Date string in 'YYYY-MM-DD' format (default: today)

        Returns:
            List of game dictionaries with team IDs and metadata
        """
        try:
            date_str = target_date or datetime.now().strftime('%Y-%m-%d')
            self.logger.info(f"Fetching NBA games for {date_str}...")

            # First, try API method
            games_df = self.fetcher.get_games_for_date(date_str)

            # If API returns nothing, check database (games were refreshed in Step 0.5)
            if games_df is None or games_df.empty:
                self.logger.info(f"API returned no games, checking database for {date_str}...")
                conn = sqlite3.connect(self.db_path)
                
                # Query database for games on this date (including scheduled games without scores)
                db_games = pd.read_sql_query("""
                    SELECT DISTINCT
                        game_id,
                        game_date,
                        home_team_id,
                        away_team_id,
                        home_team,
                        away_team,
                        home_score,
                        away_score
                    FROM games
                    WHERE game_date = ?
                    ORDER BY game_id
                """, conn, params=(date_str,))
                conn.close()
                
                if not db_games.empty:
                    self.logger.info(f"✓ Found {len(db_games)} game(s) in database for {date_str}")
                    # Convert to expected format
                    games = []
                    for _, row in db_games.iterrows():
                        # Only include games without scores (scheduled games) or with scores (finished games)
                        # Skip games that are finished (we only want scheduled games for predictions)
                        if pd.isna(row.get('home_score')) or row.get('home_score') == 0:
                            games.append({
                                'game_id': row['game_id'],
                                'game_date': row['game_date'],
                                'home_team_id': row['home_team_id'],
                                'away_team_id': row['away_team_id'],
                                'home_team_tricode': row['home_team'],
                                'away_team_tricode': row['away_team'],
                            })
                    
                    if games:
                        self.logger.info(f"✓ Found {len(games)} scheduled game(s) for {date_str}")
                        for i, game in enumerate(games, 1):
                            self.logger.debug(
                                f"  Game {i}: {game.get('away_team_tricode', 'N/A')} @ "
                                f"{game.get('home_team_tricode', 'N/A')}"
                            )
                        return games
                    else:
                        self.logger.warning(f"No scheduled games found for {date_str} (all games may be finished)")
                        return []
                else:
                    self.logger.warning(f"No NBA games scheduled for {date_str} (checked API and database)")
                    return []

            games = games_df.to_dict('records')
            self.logger.info(f"✓ Found {len(games)} game(s) scheduled")

            for i, game in enumerate(games, 1):
                self.logger.debug(
                    f"  Game {i}: {game.get('home_team_tricode', 'N/A')} vs "
                    f"{game.get('away_team_tricode', 'N/A')}"
                )

            return games

        except Exception as e:
            self.logger.error(f"✗ Failed to fetch games: {e}", exc_info=True)
            return []

    def generate_predictions(self, games: List[Dict]) -> List[Dict]:
        """
        Generate predictions for all games

        Args:
            games: List of game dictionaries

        Returns:
            List of prediction dictionaries with confidence and odds
        """
        predictions = []

        self.logger.info(f"Generating predictions for {len(games)} game(s)...")

        for i, game in enumerate(games, 1):
            try:
                # Get team identifiers - preferably tricode, fallback to ID
                home_team_tricode = game.get('home_team_tricode')
                away_team_tricode = game.get('away_team_tricode')
                home_team_id = game.get('home_team_id')
                away_team_id = game.get('away_team_id')
                game_date = game.get('game_date')

                # Map team ID to tricode if tricode is missing
                if not home_team_tricode and home_team_id:
                    # Reverse lookup: ID -> tricode
                    for tricode, team_id in self.fetcher.TEAMS.items():
                        if team_id == home_team_id:
                            home_team_tricode = tricode
                            break
                if not away_team_tricode and away_team_id:
                    # Reverse lookup: ID -> tricode
                    for tricode, team_id in self.fetcher.TEAMS.items():
                        if team_id == away_team_id:
                            away_team_tricode = tricode
                            break

                # Fallback to 'Unknown' if still missing
                home_team = home_team_tricode or 'Unknown'
                away_team = away_team_tricode or 'Unknown'

                self.logger.info(f"Predicting game {i}/{len(games)}: {home_team} vs {away_team}")

                # Skip if we don't have valid team identifiers
                if home_team == 'Unknown' or away_team == 'Unknown':
                    self.logger.warning(f"  ✗ Missing team information for game")
                    continue

                # Generate prediction
                result = self.predictor.predict_game(
                    home_team=home_team,
                    away_team=away_team,
                    game_date=game_date
                )

                if not result:
                    self.logger.warning(f"  ✗ Prediction failed for {home_team} vs {away_team}")
                    continue

                # Calculate odds for the predicted winner
                predicted_team = result['prediction']
                if predicted_team == 'home':
                    win_probability = result['home_win_probability']
                    predicted_team_name = home_team
                else:
                    win_probability = result['away_win_probability']
                    predicted_team_name = away_team

                # Calculate fair odds (no bookmaker margin)
                fair_odds = get_fair_odds(win_probability)

                # Add calculated fields to result
                result['predicted_team_name'] = predicted_team_name
                result['win_probability'] = win_probability
                result['odds'] = fair_odds
                result['game_info'] = game

                predictions.append(result)

                self.logger.info(
                    f"  ✓ Prediction: {predicted_team_name} | "
                    f"Confidence: {result['confidence']:.1%} | "
                    f"Win Prob: {win_probability:.1%} | "
                    f"Odds: {fair_odds:.2f}"
                )

            except Exception as e:
                self.logger.error(
                    f"  ✗ Error predicting {game.get('home_team_tricode')} vs "
                    f"{game.get('away_team_tricode')}: {e}",
                    exc_info=True
                )
                continue

        self.logger.info(f"✓ Generated {len(predictions)} prediction(s)")
        return predictions

    def filter_and_select_best(
        self,
        predictions: List[Dict],
        min_odds: float = 1.3
    ) -> Optional[Dict]:
        """
        Filter predictions by minimum odds and select the highest confidence

        Args:
            predictions: List of prediction dictionaries
            min_odds: Minimum odds threshold (default: 1.3)

        Returns:
            Best prediction dict or None if no predictions meet criteria
        """
        self.logger.info(f"Filtering predictions with odds > {min_odds}...")

        # Filter by odds
        filtered = [p for p in predictions if p['odds'] > min_odds]

        self.logger.info(f"✓ {len(filtered)} prediction(s) meet odds criteria")

        if not filtered:
            self.logger.warning("⚠ No predictions meet the minimum odds threshold")
            return None

        # Sort by confidence (descending) and select the best
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        best = filtered[0]

        self.logger.info("=" * 60)
        self.logger.info("SELECTED PREDICTION:")
        self.logger.info(f"  Matchup: {best['home_team']} vs {best['away_team']}")
        self.logger.info(f"  Pick: {best['predicted_team_name']}")
        self.logger.info(f"  Confidence: {best['confidence']:.1%}")
        self.logger.info(f"  Win Probability: {best['win_probability']:.1%}")
        self.logger.info(f"  Odds: {best['odds']:.2f}")
        self.logger.info("=" * 60)

        return best

    def format_twitter_thread(self, prediction: Dict) -> tuple:
        """
        Format prediction as a Twitter thread (matches manual Streamlit format exactly)

        Args:
            prediction: Prediction dictionary

        Returns:
            Tuple of (texts, image_paths) for the thread
        """
        from src.twitter_integration import format_prediction_tweet, create_chart_image
        from src.explainability_viz import create_comprehensive_dashboard_charts, format_injury_tweet
        import tempfile
        import os

        home = prediction['home_team']
        away = prediction['away_team']
        features = prediction.get('features', {})

        # Tweet 1: Main prediction (using the exact same format as Streamlit)
        tweet1 = format_prediction_tweet(prediction, features)

        # Extract metrics for subsequent tweets
        home_elo = features.get('home_elo', 0)
        away_elo = features.get('away_elo', 0)
        home_ortg = features.get('home_last10_offensive_rating', 0)
        away_ortg = features.get('away_last10_offensive_rating', 0)
        home_drtg = features.get('home_last10_defensive_rating', 0)
        away_drtg = features.get('away_last10_defensive_rating', 0)
        home_net = home_ortg - home_drtg
        away_net = away_ortg - away_drtg
        home_3pt = features.get('home_last10_fg3_pct', 0) * 100
        away_3pt = features.get('away_last10_fg3_pct', 0) * 100
        home_opp_3pt = features.get('home_last10_opp_fg3_pct', 0) * 100
        away_opp_3pt = features.get('away_last10_opp_fg3_pct', 0) * 100
        home_pace = features.get('home_last10_pace', 0)
        away_pace = features.get('away_last10_pace', 0)
        elo_prob = features.get('elo_win_prob', 0) * 100
        home_rest = features.get('home_rest_days', 1)
        away_rest = features.get('away_rest_days', 1)
        home_streak = features.get('home_streak', 0)
        away_streak = features.get('away_streak', 0)
        home_home_win = features.get('home_team_home_win_pct', 0) * 100
        away_road_win = features.get('away_team_road_win_pct', 0) * 100
        home_home_ppg = features.get('home_team_home_ppg', 0)
        away_road_ppg = features.get('away_team_road_ppg', 0)

        thread_texts = [tweet1]

        # Tweet 2: Injury Report (if applicable)
        home_team_id = prediction.get('home_team_id', self.fetcher.get_team_id(home) if self.fetcher else None)
        away_team_id = prediction.get('away_team_id', self.fetcher.get_team_id(away) if self.fetcher else None)
        injury_tweet = format_injury_tweet(home, away, features, home_team_id, away_team_id)
        if injury_tweet:
            thread_texts.append(injury_tweet)

        # Tweet 3 (or 2 if no injuries): Elo Ratings (compact, orange-themed)
        thread_texts.append(
            f"🔶 ELO RATING\n\n"
            f"Overall team strength:\n\n"
            f"🏠 {home}: {home_elo:.0f}\n"
            f"✈️ {away}: {away_elo:.0f}\n"
            f"🔸 Gap: {abs(home_elo - away_elo):.0f} pts\n\n"
            f"Higher Elo = stronger team historically"
        )

        # Tweet 3: Elo Win Probability (compact, orange-themed)
        thread_texts.append(
            f"🔥 ELO WIN PROBABILITY\n\n"
            f"{home} has {elo_prob:.0f}% chance to win at home based on Elo.\n\n"
            f"This baseline doesn't include:\n"
            f"🟠 Recent form\n"
            f"🟠 Injuries\n"
            f"🟠 Matchup factors"
        )

        # Tweet 4: Offensive/Defensive Ratings (compact, orange-themed)
        thread_texts.append(
            f"🟠 RATINGS (L10)\n\n"
            f"Per 100 possessions:\n\n"
            f"🏠 {home}:\n"
            f"🔥 Off: {home_ortg:.1f}\n"
            f"🛡️ Def: {home_drtg:.1f}\n"
            f"📊 Net: {home_net:+.1f}\n\n"
            f"✈️ {away}:\n"
            f"🔥 Off: {away_ortg:.1f}\n"
            f"🛡️ Def: {away_drtg:.1f}\n"
            f"📊 Net: {away_net:+.1f}"
        )

        # Tweet 5: Shooting & Perimeter Defense (compact, orange-themed)
        thread_texts.append(
            f"🎯 SHOOTING (L10)\n\n"
            f"🏠 {home}:\n"
            f"🔶 3PT%: {home_3pt:.1f}%\n"
            f"🛡️ Opp 3PT: {home_opp_3pt:.1f}%\n"
            f"📊 FG%: {features.get('home_last10_fg_pct', 0)*100:.1f}%\n\n"
            f"✈️ {away}:\n"
            f"🔶 3PT%: {away_3pt:.1f}%\n"
            f"🛡️ Opp 3PT: {away_opp_3pt:.1f}%\n"
            f"📊 FG%: {features.get('away_last10_fg_pct', 0)*100:.1f}%"
        )

        # Tweet 6: Pace (compact, orange-themed)
        thread_texts.append(
            f"⏱️ PACE (L10)\n\n"
            f"Possessions per game:\n\n"
            f"🏠 {home}: {home_pace:.1f}\n"
            f"✈️ {away}: {away_pace:.1f}\n"
            f"🔸 Gap: {abs(home_pace - away_pace):.1f}\n\n"
            f"Higher pace = faster tempo\n"
            f"Similar pace = predictable scoring"
        )

        # Tweet 7: Home/Away Splits (compact, orange-themed)
        thread_texts.append(
            f"🏠 HOME/AWAY SPLITS\n\n"
            f"🏠 {home} at Home:\n"
            f"🔶 Win%: {home_home_win:.1f}%\n"
            f"🔶 PPG: {home_home_ppg:.1f}\n\n"
            f"✈️ {away} on Road:\n"
            f"🔶 Win%: {away_road_win:.1f}%\n"
            f"🔶 PPG: {away_road_ppg:.1f}\n\n"
            f"Home court = ~3pt advantage"
        )

        # Tweet 8: Situational Factors (compact, orange-themed)
        thread_texts.append(
            f"📅 SITUATIONAL\n\n"
            f"Rest & momentum:\n\n"
            f"🏠 {home}:\n"
            f"🔶 Rest: {home_rest} day(s)\n"
            f"🔥 Streak: {home_streak:+d}\n\n"
            f"✈️ {away}:\n"
            f"🔶 Rest: {away_rest} day(s)\n"
            f"🔥 Streak: {away_streak:+d}\n\n"
            f"More rest = better energy"
        )

        # Generate chart images for tweets 2-8
        image_paths = []

        try:
            self.logger.info("Generating chart images...")

            # Create all charts using the same function as Streamlit
            all_charts = create_comprehensive_dashboard_charts(prediction, features, home, away)

            # Define chart order matching tweets 2-8
            chart_order = ['elo', 'elo_gauge', 'ratings_l10', 'shooting_l10', 'pace', 'splits', 'situational']

            # Create temporary directory for chart images
            temp_dir = tempfile.mkdtemp()

            # Export each chart as PNG
            for chart_name in chart_order:
                if chart_name in all_charts:
                    chart_path = os.path.join(temp_dir, f"{chart_name}.png")
                    try:
                        create_chart_image(all_charts[chart_name], chart_path)
                        image_paths.append(chart_path)
                        self.logger.debug(f"Created chart: {chart_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create chart {chart_name}: {e}")
                        image_paths.append(None)  # Use None if chart creation fails
                else:
                    image_paths.append(None)

            self.logger.info(f"✓ Generated {len([p for p in image_paths if p])} chart images")

        except Exception as e:
            self.logger.warning(f"Chart generation failed: {e}")
            # Return empty image list if chart generation fails
            image_paths = []

        # Return texts and image paths (first tweet has no image)
        return thread_texts, image_paths

    def post_to_twitter(self, prediction: Dict) -> bool:
        """
        Post prediction to Twitter as a thread (exact same format as manual Streamlit posting)

        Args:
            prediction: Prediction dictionary

        Returns:
            True if posted successfully, False otherwise
        """
        try:
            self.logger.info("Preparing Twitter thread...")

            # Format thread (matches Streamlit format)
            tweets, image_paths = self.format_twitter_thread(prediction)

            self.logger.info(f"Thread formatted: {len(tweets)} tweets with {len([p for p in image_paths if p])} images")
            for i, tweet in enumerate(tweets, 1):
                has_image = i > 1 and i <= len(image_paths) + 1 and image_paths[i-2] is not None
                img_status = "📷 [with image]" if has_image else ""
                self.logger.debug(f"Tweet {i} ({len(tweet)} chars) {img_status}:\n{tweet}\n")

            if self.dry_run:
                self.logger.info("=" * 60)
                self.logger.info("DRY RUN MODE - Would post the following thread:")
                self.logger.info("=" * 60)
                for i, tweet in enumerate(tweets, 1):
                    self.logger.info(f"\n--- TWEET {i} ({len(tweet)} chars) ---\n{tweet}\n")
                self.logger.info("=" * 60)
                return True

            # Post thread (first tweet has no image, others can have images)
            self.logger.info("Posting to Twitter...")

            # Create image_paths list with None for first tweet
            image_paths_with_none = [None] + image_paths if image_paths else [None] * len(tweets)

            responses = create_twitter_thread(
                api_clients=self.api_clients,
                texts=tweets,
                image_paths=image_paths_with_none[:len(tweets)],  # Ensure same length
                dry_run=False
            )

            # Check if all tweets posted successfully
            success = all(r.get('success', False) for r in responses)

            if success:
                tweet_ids = [r.get('tweet_id') for r in responses if r.get('tweet_id')]
                self.logger.info(f"✓ Thread posted successfully! {len(tweet_ids)} tweets")
                self.logger.info(f"First tweet ID: {tweet_ids[0] if tweet_ids else 'N/A'}")

                # Save posted prediction to log
                self._save_posted_prediction(prediction, tweet_ids)

                return True
            else:
                self.logger.error("✗ Some tweets failed to post")
                for i, resp in enumerate(responses, 1):
                    if not resp.get('success'):
                        self.logger.error(f"Tweet {i} error: {resp.get('error', 'Unknown')}")
                return False

        except Exception as e:
            self.logger.error(f"✗ Failed to post to Twitter: {e}", exc_info=True)
            return False

    def _save_posted_prediction(self, prediction: Dict, tweet_ids: List[str]):
        """Save posted prediction details to a JSON log file"""
        try:
            log_file = self.log_dir / "posted_predictions.jsonl"

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'matchup': f"{prediction['home_team']} vs {prediction['away_team']}",
                'pick': prediction['predicted_team_name'],
                'confidence': prediction['confidence'],
                'win_probability': prediction['win_probability'],
                'odds': prediction['odds'],
                'tweet_ids': tweet_ids,
                'first_tweet_url': f"https://twitter.com/user/status/{tweet_ids[0]}" if tweet_ids else None
            }

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')

            self.logger.debug(f"Saved posted prediction to {log_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save posted prediction log: {e}")

    def check_and_update_previous_predictions(self, lookback_days: int = 7) -> Dict:
        """
        Check and update previous pending predictions with actual game results

        This follows the same 2-step process as the Streamlit Performance tab:
        1. Fetch game data from NBA API into games table
        2. Match predictions with game results

        Args:
            lookback_days: How many days back to check (default: 7)

        Returns:
            Dictionary with update statistics
        """
        try:
            self.logger.info(f"Updating previous predictions (last {lookback_days} days)...")

            # Step 1: Fetch game data from NBA API (like "Refresh Game Data" button)
            self.logger.info("  Step 1: Fetching game results from NBA API...")

            if not self.fetcher:
                self.fetcher = NBADataFetcher(self.db_path)

            games_fetched = self.fetcher.update_recent_games(days_back=lookback_days)
            self.logger.info(f"  ✓ Fetched {games_fetched} games from NBA API")

            # Step 2: Update predictions with results from database (like "Update Results" button)
            self.logger.info("  Step 2: Matching predictions with game results...")

            feedback_system = ModelFeedbackSystem(self.db_path)
            updated_predictions = feedback_system.update_predictions_with_results(
                lookback_days=lookback_days,
                use_api=False  # Use database (already fetched in step 1)
            )
            feedback_system.close()

            self.logger.info(f"  ✓ Updated {updated_predictions} predictions")

            # Get stats for reporting
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count correct, wrong, and pending predictions
            cursor.execute("""
                SELECT
                    COUNT(CASE WHEN correct = 1 THEN 1 END) as correct_count,
                    COUNT(CASE WHEN correct = 0 THEN 1 END) as wrong_count,
                    COUNT(CASE WHEN actual_winner IS NULL THEN 1 END) as pending_count
                FROM predictions
                WHERE game_date >= date('now', '-' || ? || ' days')
            """, (lookback_days,))

            result = cursor.fetchone()
            conn.close()

            stats = {
                'games_fetched': games_fetched,
                'predictions_updated': updated_predictions,
                'correct': result[0] if result else 0,
                'wrong': result[1] if result else 0,
                'pending': result[2] if result else 0
            }

            # Log summary
            if stats['predictions_updated'] > 0:
                self.logger.info(f"✓ Prediction update complete:")
                self.logger.info(f"  - Correct: {stats['correct']}")
                self.logger.info(f"  - Wrong: {stats['wrong']}")
                self.logger.info(f"  - Still pending: {stats['pending']}")

                if stats['correct'] + stats['wrong'] > 0:
                    accuracy = stats['correct'] / (stats['correct'] + stats['wrong']) * 100
                    self.logger.info(f"  - Accuracy: {accuracy:.1f}%")

            return stats

        except Exception as e:
            self.logger.error(f"✗ Failed to update predictions: {e}", exc_info=True)
            return {'error': str(e)}

    def run(
        self,
        target_date: Optional[str] = None,
        lookback_days: int = 7,
        skip_prediction_check: bool = False
    ) -> bool:
        """
        Execute the full daily automation workflow

        Args:
            target_date: Optional date override (YYYY-MM-DD)
            lookback_days: Days to check for prediction results (default: 7)
            skip_prediction_check: Skip updating previous predictions (default: False)

        Returns:
            True if workflow completed successfully, False otherwise
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"Starting daily automation workflow at {start_time}")
            self.logger.info("=" * 80)

            # Step 0: Update previous predictions with results
            if not skip_prediction_check:
                self.logger.info("STEP 0: Updating previous predictions...")
                self.check_and_update_previous_predictions(lookback_days=lookback_days)
                self.logger.info("")
            else:
                self.logger.info("STEP 0: Skipping prediction updates (--skip-prediction-check)")
                self.logger.info("")

            # Step 1: Initialize components
            self.logger.info("STEP 1: Initializing components...")
            if not self.initialize_components():
                self.logger.error("✗ Workflow aborted: Component initialization failed")
                return False
            self.logger.info("")

            # Step 2: Fetch today's games
            self.logger.info("STEP 2: Fetching today's games...")
            games = self.fetch_todays_games(target_date)
            if not games:
                self.logger.info("ℹ No games today - workflow complete (nothing to post)")
                return True  # Not a failure - just no games
            self.logger.info("")

            # Step 3: Generate predictions
            self.logger.info("STEP 3: Generating predictions...")
            predictions = self.generate_predictions(games)
            if not predictions:
                self.logger.warning("⚠ No predictions generated - workflow complete (nothing to post)")
                return False
            self.logger.info("")

            # Step 4: Filter and select best prediction
            self.logger.info("STEP 4: Filtering and selecting best prediction...")
            best_prediction = self.filter_and_select_best(predictions, min_odds=1.3)
            if not best_prediction:
                self.logger.info("ℹ No predictions meet criteria - workflow complete (nothing to post)")
                return True  # Not a failure - just no qualifying predictions
            self.logger.info("")

            # Step 5: Post to Twitter
            self.logger.info("STEP 5: Posting to Twitter...")
            twitter_success = self.post_to_twitter(best_prediction)
            self.logger.info("")
            
            # Step 6: Send daily email report
            self.logger.info("STEP 6: Sending daily email report...")
            try:
                email_reporter = EmailReporter(db_path=self.db_path)
                email_success = email_reporter.send_daily_report(test_mode=False)
                if email_success:
                    self.logger.info("✓ Email report sent successfully")
                else:
                    self.logger.warning("⚠ Email report failed (non-critical)")
            except Exception as e:
                self.logger.warning(f"⚠ Email report error (non-critical): {e}")
            self.logger.info("")
            
            success = twitter_success  # Twitter success determines overall success

            # Workflow summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.logger.info("=" * 80)
            if success:
                self.logger.info(f"✓ WORKFLOW COMPLETED SUCCESSFULLY in {duration:.1f}s")
            else:
                self.logger.error(f"✗ WORKFLOW FAILED in {duration:.1f}s")
            self.logger.info("=" * 80)

            return success

        except Exception as e:
            self.logger.error(f"✗ Workflow failed with unexpected error: {e}", exc_info=True)
            return False


def main():
    """Main entry point for the daily automation script"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Daily NBA Prediction Automation - Updates predictions, generates new ones, posts to Twitter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python daily_auto_prediction.py                            # Run full workflow
  python daily_auto_prediction.py --dry-run                  # Test mode (no Twitter posting)
  python daily_auto_prediction.py --verbose                  # Debug logging
  python daily_auto_prediction.py --date 2025-12-25          # Run for specific date
  python daily_auto_prediction.py --skip-prediction-check    # Skip updating old predictions (faster)
  python daily_auto_prediction.py --lookback-days 14         # Update predictions from last 14 days

Features:
  - Updates previous pending predictions with actual results
  - Fetches game data from NBA API
  - Generates predictions for today's games
  - Posts best prediction to Twitter as 8-tweet thread
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test mode - simulate posting without calling Twitter API'
    )
    
    # Also check .env file for TW_DRY_RUN if --dry-run not specified
    from dotenv import load_dotenv
    import os
    load_dotenv()  # Load .env file
    env_dry_run = os.getenv("TW_DRY_RUN", "false").lower() in ("1", "true", "yes")

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging to console'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Override target date (format: YYYY-MM-DD)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='data/nba_predictor.db',
        help='Path to database (default: data/nba_predictor.db)'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Path to model directory (default: models)'
    )

    parser.add_argument(
        '--lookback-days',
        type=int,
        default=7,
        help='Days to look back for updating prediction results (default: 7)'
    )

    parser.add_argument(
        '--skip-prediction-check',
        action='store_true',
        help='Skip updating previous predictions with results (faster startup)'
    )

    args = parser.parse_args()

    # Create automation instance
    # Use command line arg if provided, otherwise use .env value
    dry_run_mode = args.dry_run if args.dry_run else env_dry_run
    
    automation = DailyPredictionAutomation(
        db_path=args.db_path,
        model_dir=args.model_dir,
        log_dir='logs',
        dry_run=dry_run_mode
    )

    # Adjust console logging level if verbose
    if args.verbose:
        for handler in automation.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)

    # Run the workflow
    success = automation.run(
        target_date=args.date,
        lookback_days=args.lookback_days,
        skip_prediction_check=args.skip_prediction_check
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

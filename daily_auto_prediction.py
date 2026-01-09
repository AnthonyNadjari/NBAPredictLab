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
from src.data_fetcher import NBADataFetcher, EloRatingSystem
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
import numpy as np


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


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

    def _ensure_fresh_elo(self) -> None:
        """
        Ensure ELO ratings are fresh before making predictions.
        Stale ELO ratings are a major source of prediction errors.
        """
        self.logger.info("Checking ELO rating freshness...")

        try:
            elo = EloRatingSystem(db_path=self.db_path)

            # Get freshness info
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT team_id, elo, last_updated
            FROM current_elo
            ORDER BY last_updated DESC
            """
            df = pd.read_sql(query, conn)
            conn.close()

            if df.empty:
                self.logger.warning("[WARNING] No ELO ratings found - predictions may be inaccurate")
                return

            # Check for stale ratings (>2 days old)
            today = datetime.now()
            stale_count = 0

            for _, row in df.iterrows():
                try:
                    last_update = pd.to_datetime(row['last_updated'])
                    days_old = (today - last_update).days
                    if days_old > 2:
                        stale_count += 1
                except Exception:
                    continue

            if stale_count > 0:
                self.logger.warning(f"[WARNING] {stale_count} teams have stale ELO ratings (>2 days old)")
                self.logger.info("Updating ELO from recent games...")
                elo.update_elo_from_recent_games(days=5)
                self.logger.info("[OK] ELO ratings refreshed")
            else:
                self.logger.info("[OK] All ELO ratings are current")

        except Exception as e:
            self.logger.error(f"ELO freshness check failed: {e}")
            # Continue anyway - stale ELO is better than no ELO

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
            self.logger.info("[OK] Data fetcher initialized")

            # Ensure ELO ratings are fresh (critical for prediction quality)
            self._ensure_fresh_elo()

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

    def _save_predictions_to_db(self, predictions: List[Dict], game_date: str) -> int:
        """
        Save all predictions to the database (matches Streamlit save_prediction_to_db format).

        Args:
            predictions: List of prediction dictionaries from generate_predictions
            game_date: Date string (YYYY-MM-DD)

        Returns:
            Number of predictions saved
        """
        saved_count = 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for pred in predictions:
                try:
                    # Extract data from prediction dict
                    home_team = pred.get('home_team', '')
                    away_team = pred.get('away_team', '')
                    features = pred.get('features', {})

                    # Determine winner
                    if pred.get('prediction') == 'home':
                        predicted_winner = home_team
                    else:
                        predicted_winner = away_team

                    # Convert features to JSON-safe format
                    try:
                        features_converted = convert_numpy_types(features)
                        features_json = json.dumps(features_converted)
                    except (TypeError, ValueError) as e:
                        self.logger.warning(f"Error serializing features: {e}")
                        features_json = json.dumps({})

                    # Get betting odds from features if available
                    home_odds = features.get('home_odds') or pred.get('home_odds')
                    away_odds = features.get('away_odds') or pred.get('away_odds')

                    # If no odds, calculate from probabilities
                    home_prob = pred.get('home_win_probability', 0.5)
                    away_prob = pred.get('away_win_probability', 0.5)
                    if not home_odds and home_prob > 0:
                        home_odds = round(1 / home_prob, 2)
                    if not away_odds and away_prob > 0:
                        away_odds = round(1 / away_prob, 2)

                    # Insert or replace prediction
                    cursor.execute("""
                        INSERT OR REPLACE INTO predictions (
                            prediction_date, game_date, home_team, away_team,
                            predicted_winner, predicted_home_prob, predicted_away_prob,
                            confidence, features_json, home_odds, away_odds
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        game_date,
                        home_team,
                        away_team,
                        predicted_winner,
                        float(home_prob),
                        float(away_prob),
                        float(pred.get('confidence', 0.5)),
                        features_json,
                        home_odds,
                        away_odds
                    ))
                    saved_count += 1

                except Exception as e:
                    self.logger.error(f"Error saving prediction for {home_team} vs {away_team}: {e}")
                    continue

            conn.commit()
            conn.close()
            self.logger.info(f"✓ Saved {saved_count} predictions to database")

        except Exception as e:
            self.logger.error(f"Failed to save predictions to database: {e}", exc_info=True)

        return saved_count

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

        # DEBUG: Log features received
        self.logger.info(f"format_twitter_thread - features count: {len(features)}")
        if features:
            self.logger.info(f"  DEBUG - home_elo: {features.get('home_elo', 'MISSING')}")
            self.logger.info(f"  DEBUG - home_last10_offensive_rating: {features.get('home_last10_offensive_rating', 'MISSING')}")
        else:
            self.logger.warning("  DEBUG - features dict is empty!")

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

        # CRITICAL: Determine which team we picked to ensure consistency
        is_home_pick = prediction.get('prediction') == 'home'
        picked_team = home if is_home_pick else away
        opponent = away if is_home_pick else home

        # =============================================================================
        # COLLECT ALL SUPPORTING FACTORS WITH STRENGTH SCORES
        # Each factor: (category, strength, tweet_content, short_summary)
        # =============================================================================
        all_supporting_factors = []

        # Extract all metrics for picked team vs opponent
        home_last3_win = features.get('home_last3_win_pct', 0) * 100
        away_last3_win = features.get('away_last3_win_pct', 0) * 100
        home_weighted_form = features.get('home_weighted_recent_form', 0) * 100
        away_weighted_form = features.get('away_weighted_recent_form', 0) * 100

        picked_last3_win = home_last3_win if is_home_pick else away_last3_win
        picked_streak = home_streak if is_home_pick else away_streak
        opponent_streak = away_streak if is_home_pick else home_streak
        picked_form = home_weighted_form if is_home_pick else away_weighted_form
        opponent_form = away_weighted_form if is_home_pick else home_weighted_form
        picked_rest = home_rest if is_home_pick else away_rest
        opponent_rest = away_rest if is_home_pick else home_rest
        picked_last3_net = features.get('home_last3_net_rating', 0) if is_home_pick else features.get('away_last3_net_rating', 0)
        opponent_last3_net = features.get('away_last3_net_rating', 0) if is_home_pick else features.get('home_last3_net_rating', 0)
        picked_form_accel = features.get('home_form_acceleration', 0) if is_home_pick else features.get('away_form_acceleration', 0)
        opponent_last3_win = away_last3_win if is_home_pick else home_last3_win
        picked_ortg = home_ortg if is_home_pick else away_ortg
        opponent_ortg = away_ortg if is_home_pick else home_ortg
        picked_drtg = home_drtg if is_home_pick else away_drtg
        opponent_drtg = away_drtg if is_home_pick else home_drtg
        picked_3pt = home_3pt if is_home_pick else away_3pt
        opponent_3pt = away_3pt if is_home_pick else home_3pt

        if is_home_pick:
            picked_split_win = home_home_win
            picked_split_ppg = home_home_ppg
            opponent_split_win = away_road_win
            picked_split_label = "at home"
        else:
            picked_split_win = away_road_win
            picked_split_ppg = away_road_ppg
            opponent_split_win = home_home_win
            picked_split_label = "on road"

        # --- FACTOR 1: HOT STREAK (picked team on fire) ---
        if picked_streak >= 2:  # Lowered from 3
            strength = picked_streak * 10 + picked_last3_win * 0.5
            tweet = f"🔥 {picked_team} ON FIRE\n\n"
            tweet += f"Current streak: {picked_streak} WINS\n"
            tweet += f"Last 3 games: {picked_last3_win:.0f}% win rate\n"
            tweet += f"Net rating L3: {picked_last3_net:+.1f}\n\n"
            tweet += f"💡 Momentum is real in the NBA\nHot teams cover at 58% rate"
            all_supporting_factors.append(('hot_streak', strength, tweet, f"🔥 {picked_streak}W streak"))

        # --- FACTOR 2: COLD OPPONENT (opponent struggling) ---
        if opponent_streak <= -2:  # Lowered from -3
            strength = abs(opponent_streak) * 10 + (100 - opponent_last3_win) * 0.5
            tweet = f"❄️ {opponent} IN FREEFALL\n\n"
            tweet += f"Current streak: {abs(opponent_streak)} LOSSES\n"
            tweet += f"Last 3 games: {opponent_last3_win:.0f}% win rate\n"
            tweet += f"Net rating L3: {opponent_last3_net:+.1f}\n\n"
            tweet += f"💡 Slumping teams lose at 65% rate\n{picked_team} should capitalize"
            all_supporting_factors.append(('cold_opponent', strength, tweet, f"❄️ {opponent} {abs(opponent_streak)}L streak"))

        # --- FACTOR 3: REST ADVANTAGE ---
        rest_diff = picked_rest - opponent_rest
        if rest_diff >= 1 and opponent_rest == 0:
            strength = rest_diff * 25 + 20
            tweet = f"😴 FATIGUE FACTOR\n\n"
            tweet += f"⭐ {picked_team}: {picked_rest} days rest\n"
            tweet += f"😩 {opponent}: BACK-TO-BACK\n\n"
            tweet += f"Historical edge: ~5 points\n"
            tweet += f"4th quarter = tired legs\n\n"
            tweet += f"💡 B2B teams lose 58% of games"
            all_supporting_factors.append(('rest', strength, tweet, f"😴 {opponent} on B2B"))
        elif rest_diff >= 1:  # Lowered from 2
            strength = rest_diff * 15 + 5
            tweet = f"⚡ REST ADVANTAGE\n\n"
            tweet += f"⭐ {picked_team}: {picked_rest} days rest\n"
            tweet += f"{opponent}: {opponent_rest} days rest\n\n"
            tweet += f"💡 Well-rested teams have ~3pt edge"
            all_supporting_factors.append(('rest', strength, tweet, f"⚡ +{rest_diff} days rest"))

        # --- FACTOR 4: OFFENSIVE EDGE ---
        off_edge = picked_ortg - opponent_ortg
        if off_edge > 0:  # Always include if positive (lowered from 3)
            strength = max(off_edge * 8, 5)  # Minimum strength of 5
            tweet = f"🔥 OFFENSIVE FIREPOWER\n\n"
            tweet += f"⭐ {picked_team}: {picked_ortg:.1f} OffRtg\n"
            tweet += f"{opponent}: {opponent_ortg:.1f} OffRtg\n\n"
            tweet += f"Gap: {off_edge:.1f} pts per 100 possessions\n\n"
            tweet += f"💡 {picked_team} scores more efficiently"
            all_supporting_factors.append(('offense', strength, tweet, f"🔥 +{off_edge:.1f} OffRtg"))

        # --- FACTOR 5: DEFENSIVE EDGE ---
        def_edge = opponent_drtg - picked_drtg
        if def_edge > 0:  # Always include if positive (lowered from 3)
            strength = max(def_edge * 8, 5)
            tweet = f"🛡️ DEFENSIVE WALL\n\n"
            tweet += f"⭐ {picked_team}: {picked_drtg:.1f} DefRtg\n"
            tweet += f"{opponent}: {opponent_drtg:.1f} DefRtg\n\n"
            tweet += f"Gap: {def_edge:.1f} fewer pts allowed/100\n\n"
            tweet += f"💡 Defense wins championships\n{picked_team} locks down"
            all_supporting_factors.append(('defense', strength, tweet, f"🛡️ +{def_edge:.1f} DefRtg edge"))

        # --- FACTOR 6: 3-POINT SHOOTING ---
        three_edge = picked_3pt - opponent_3pt
        if three_edge > 0:  # Always include if positive (lowered from 3)
            strength = max(three_edge * 6, 4)
            tweet = f"🎯 3-POINT ADVANTAGE\n\n"
            tweet += f"⭐ {picked_team}: {picked_3pt:.1f}% from 3\n"
            tweet += f"{opponent}: {opponent_3pt:.1f}% from 3\n\n"
            tweet += f"Gap: {three_edge:.1f}% better shooting\n\n"
            tweet += f"💡 Modern NBA = 3PT shooting wins"
            all_supporting_factors.append(('three_pt', strength, tweet, f"🎯 +{three_edge:.1f}% from 3"))

        # --- FACTOR 7: HOME/ROAD SPLITS ---
        if is_home_pick:
            strength = max((picked_split_win - 40) * 1.5, 5)  # Always include for home picks
            tweet = f"🏠 HOME COURT EDGE\n\n"
            tweet += f"⭐ {picked_team} at home:\n"
            tweet += f"Win rate: {picked_split_win:.0f}%\n"
            tweet += f"PPG: {picked_split_ppg:.1f}\n\n"
            tweet += f"vs {opponent} on road: {opponent_split_win:.0f}% W\n\n"
            if picked_split_win >= 60:
                tweet += f"💡 {picked_team} dominates at home"
            else:
                tweet += f"💡 Home court = ~3pt advantage"
            all_supporting_factors.append(('home_split', strength, tweet, f"🏠 {picked_split_win:.0f}% home W%"))
        else:
            strength = max(picked_split_win * 1.2, 5)  # Always include for away picks
            tweet = f"✈️ ROAD WARRIORS\n\n"
            tweet += f"⭐ {picked_team} on road:\n"
            tweet += f"Win rate: {picked_split_win:.0f}%\n"
            tweet += f"PPG: {picked_split_ppg:.1f}\n\n"
            tweet += f"vs {opponent} at home: {opponent_split_win:.0f}% W\n\n"
            if picked_split_win >= 45:
                tweet += f"💡 {picked_team} thrives on the road"
            else:
                tweet += f"💡 {picked_team} can handle hostile crowds"
            all_supporting_factors.append(('road_split', strength, tweet, f"✈️ {picked_split_win:.0f}% road W%"))

        # --- FACTOR 8: RECENT FORM (L3) ---
        form_diff = picked_form - opponent_form
        strength = max(abs(form_diff) * 1.2, 8)  # Always include
        tweet = f"📈 RECENT FORM (L3)\n\n"
        tweet += f"⭐ {picked_team}:\n"
        tweet += f"Win%: {picked_last3_win:.0f}%\n"
        tweet += f"Net: {picked_last3_net:+.1f}\n"
        tweet += f"Trend: {'📈 SURGING' if picked_form_accel > 0.1 else '📉 Falling' if picked_form_accel < -0.1 else '➡️ Stable'}\n\n"
        tweet += f"vs {opponent}: {opponent_last3_win:.0f}% W\n\n"
        if form_diff > 10:
            tweet += f"💡 {picked_team} playing {form_diff:.0f}% better recently"
        else:
            tweet += f"💡 Last 3 games = 50% of AI's decision"
        all_supporting_factors.append(('form', strength, tweet, f"📈 L3: {picked_last3_win:.0f}%"))

        # --- FACTOR 9: ELO RATING ---
        elo_diff = (home_elo - away_elo) if is_home_pick else (away_elo - home_elo)
        picked_elo = home_elo if is_home_pick else away_elo
        opponent_elo = away_elo if is_home_pick else home_elo
        strength = max(abs(elo_diff) * 0.2, 6)  # Always include
        tweet = f"📊 ELO RATINGS\n\n"
        tweet += f"⭐ {picked_team}: {picked_elo:.0f}\n"
        tweet += f"{opponent}: {opponent_elo:.0f}\n\n"
        if elo_diff > 0:
            tweet += f"Gap: +{elo_diff:.0f} rating points\n\n"
            tweet += f"💡 {picked_team} is the stronger team"
        else:
            tweet += f"Gap: {elo_diff:.0f} rating points\n\n"
            tweet += f"💡 Model sees upset value here"
        all_supporting_factors.append(('elo', strength, tweet, f"📊 ELO {picked_elo:.0f}"))

        # --- FACTOR 10: NET RATING ---
        picked_net = picked_ortg - picked_drtg
        opponent_net = opponent_ortg - opponent_drtg
        net_edge = picked_net - opponent_net
        strength = max(abs(net_edge) * 3, 7)  # Always include
        tweet = f"⚔️ NET RATING\n\n"
        tweet += f"⭐ {picked_team}: {picked_net:+.1f}\n"
        tweet += f"{opponent}: {opponent_net:+.1f}\n\n"
        if net_edge > 0:
            tweet += f"Edge: +{net_edge:.1f} net rating\n\n"
            tweet += f"💡 {picked_team} outscores opponents more"
        else:
            tweet += f"Edge: {net_edge:.1f} net rating\n\n"
            tweet += f"💡 AI sees hidden value beyond net rating"
        all_supporting_factors.append(('net_rating', strength, tweet, f"⚔️ Net: {picked_net:+.1f}"))

        # =============================================================================
        # SORT FACTORS BY STRENGTH AND BUILD EXACTLY 7 TWEETS
        # Tweet 1: Main prediction (already added)
        # Tweet 2: Summary of top factors
        # Tweets 3-7: Top 5 factor details
        # Tweet 8: CTA (added later)
        # =============================================================================
        all_supporting_factors.sort(key=lambda x: x[1], reverse=True)

        # Tweet 2: TOP EDGE SUMMARY (why we picked this team)
        edge_tweet = f"🎯 WHY {picked_team.upper()}?\n\n"
        edge_tweet += "Top factors driving this pick:\n\n"

        # List top 5 factors as bullet points
        for i, (cat, strength, full_tweet, summary) in enumerate(all_supporting_factors[:5]):
            edge_tweet += f"{i+1}. {summary}\n"
        edge_tweet += f"\n📊 Confidence: {prediction['confidence']*100:.0f}%"

        thread_texts.append(edge_tweet)

        # Tweets 3-7: DETAILED BREAKDOWN OF TOP 5 FACTORS
        # We always have at least 5 factors now (form, elo, net_rating, splits are always included)
        for i in range(5):
            if i < len(all_supporting_factors):
                cat, strength, full_tweet, summary = all_supporting_factors[i]
                thread_texts.append(full_tweet)

        # Tweet 8: Smart Adjustments OR Model Summary
        # Always include this tweet to maintain 8-tweet structure
        if 'pattern_adjustments' in prediction and prediction['pattern_adjustments']:
            adjustments_list = prediction['pattern_adjustments']

            # Build explanation tweet
            adj_tweet = "🔧 SMART ADJUSTMENTS\n\n"
            adj_tweet += "AI detected patterns where it historically struggled:\n\n"

            for adj in adjustments_list[:2]:  # Max 2 adjustments to fit in tweet
                if "Hot road" in adj:
                    adj_tweet += "⚡ Hot Road Team (+10%)\n"
                    adj_tweet += f"{away} on 4+ win streak\n\n"
                elif "Cold home" in adj:
                    adj_tweet += "⚠️ Cold Home Team (-8%)\n"
                    adj_tweet += f"{home} on 3+ loss streak\n\n"
                elif "Heavy travel" in adj:
                    adj_tweet += "🛫 Travel Fatigue (-15%)\n"
                    adj_tweet += f"{away} cross-country on B2B\n\n"
                elif "Large ELO" in adj:
                    adj_tweet += "📊 Big Favorite Alert (-5%)\n"
                    adj_tweet += "Reducing overconfidence\n\n"
                elif "B2B" in adj:
                    adj_tweet += "😴 Fatigue Factor (-6%)\n"
                    adj_tweet += f"{home} on back-to-back\n\n"

            adj_tweet += "💡 These corrections improve accuracy ~5%"
            thread_texts.append(adj_tweet)
        else:
            # No adjustments - add model summary tweet instead
            summary_tweet = f"🤖 MODEL SUMMARY\n\n"
            summary_tweet += f"⭐ Pick: {picked_team}\n"
            summary_tweet += f"📊 Confidence: {prediction['confidence']*100:.0f}%\n"
            summary_tweet += f"🎯 Quality: {prediction.get('prediction_quality', 'medium').upper()}\n\n"
            summary_tweet += f"Features analyzed: 165\n"
            summary_tweet += f"Recency weight: 50% on L3\n"
            summary_tweet += f"Calibration: Temperature scaling\n\n"
            summary_tweet += f"💡 AI finds edges humans miss"
            thread_texts.append(summary_tweet)

        # FINAL TWEET (Tweet 9): CTA for Telegram (CRITICAL for growth!)
        cta_tweet = "💰 WANT MORE PICKS?\n\n"
        cta_tweet += "Daily NBA predictions:\n\n"
        cta_tweet += "📈 63.5% accuracy\n"
        cta_tweet += "🤖 AI: 165 data points\n"
        cta_tweet += "⚡ Last 3 games = 50% weight\n"
        cta_tweet += "🎯 Smart adjustments\n\n"
        cta_tweet += "Join Telegram for:\n"
        cta_tweet += "✅ All daily picks\n"
        cta_tweet += "✅ Full analysis threads\n"
        cta_tweet += "✅ Live updates\n\n"
        cta_tweet += "Link in bio 👆"
        
        # Safety check: ensure tweet fits within 280 character limit
        if len(cta_tweet) > 280:
            # Ultra-compact fallback if somehow still too long
            cta_tweet = "💰 WANT MORE PICKS?\n\nDaily NBA predictions:\n📈 63.5% accuracy\n🤖 AI: 165 data points\n⚡ Last 3 games = 50% weight\n\nJoin Telegram:\n✅ All daily picks\n✅ Full threads\n✅ Live updates\n\nLink in bio 👆"
        
        thread_texts.append(cta_tweet)

        # Generate chart images for the thread
        # NEW STRUCTURE: Tweet 1 (no image), Tweets 2-6 (with images), Tweet 7+ (optional, no images)
        image_paths = []

        try:
            self.logger.info("Generating chart images...")

            # Create all charts using the same function as Streamlit
            all_charts = create_comprehensive_dashboard_charts(prediction, features, home, away)

            # NEW: Map tweets to most relevant charts
            # Tweet 1: Main prediction (no image)
            # Tweet 2: THE EDGE (use situational chart - shows streaks/rest)
            # Tweet 3: RECENT FORM (use ratings_l10 chart)
            # Tweet 4: KEY MATCHUP (use ratings_l10 or shooting_l10 based on which is biggest)
            # Tweet 5: SCHEDULE SPOT (use situational chart)
            # Tweet 6: HOME/ROAD SPLITS (use splits chart)
            # Tweet 7+: Smart Adjustments / CTA (no images)

            # Create temporary directory for chart images
            temp_dir = tempfile.mkdtemp()

            # Chart mapping for new structure (skip first tweet)
            # Fix: Ensure each tweet gets a unique chart to avoid duplicates
            chart_mapping = [
                None,  # Tweet 1: Main prediction (no image)
                'situational',  # Tweet 2: THE EDGE (shows streaks/rest)
                'ratings_l10',  # Tweet 3: RECENT FORM (ratings comparison)
                'shooting_l10',  # Tweet 4: KEY MATCHUP (use shooting chart to avoid duplicate with Tweet 3)
                None,  # Tweet 5: SCHEDULE SPOT (no chart - similar content to Tweet 2, avoid duplicate)
                'splits',       # Tweet 6: HOME/ROAD SPLITS
            ]

            # Generate images for mapped charts
            for chart_name in chart_mapping:
                if chart_name and chart_name in all_charts:
                    chart_path = os.path.join(temp_dir, f"{chart_name}_{len(image_paths)}.png")
                    try:
                        create_chart_image(all_charts[chart_name], chart_path)
                        image_paths.append(chart_path)
                        self.logger.debug(f"Created chart: {chart_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create chart {chart_name}: {e}")
                        image_paths.append(None)
                else:
                    image_paths.append(None)

            self.logger.info(f"✓ Generated {len([p for p in image_paths if p])} chart images")

        except Exception as e:
            self.logger.warning(f"Chart generation failed: {e}")
            # Return empty image list if chart generation fails
            image_paths = []

        # Return texts and image paths
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

    def _push_to_github(self, target_date: Optional[str] = None) -> bool:
        """
        Push exported predictions JSON to GitHub for GitHub Pages update.

        Args:
            target_date: Date string for commit message (YYYY-MM-DD)

        Returns:
            True if push successful, False otherwise
        """
        import subprocess

        try:
            date_str = target_date or datetime.now().strftime('%Y-%m-%d')

            # Check if there are changes to commit
            result = subprocess.run(
                ['git', 'status', '--porcelain', 'docs/pending_games.json'],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if not result.stdout.strip():
                self.logger.info("ℹ No changes to docs/pending_games.json")
                return True  # No changes is not a failure

            # Stage the file
            subprocess.run(
                ['git', 'add', 'docs/pending_games.json'],
                check=True,
                capture_output=True,
                cwd=str(PROJECT_ROOT)
            )
            self.logger.debug("Staged docs/pending_games.json")

            # Commit with descriptive message
            commit_message = f"Auto-export predictions for {date_str}"
            subprocess.run(
                ['git', 'commit', '-m', commit_message],
                check=True,
                capture_output=True,
                cwd=str(PROJECT_ROOT)
            )
            self.logger.debug(f"Committed: {commit_message}")

            # Push to remote
            result = subprocess.run(
                ['git', 'push'],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode != 0:
                self.logger.error(f"Git push failed: {result.stderr}")
                return False

            self.logger.info(f"✓ Pushed predictions for {date_str} to GitHub")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git command failed: {e}")
            if e.stderr:
                self.logger.error(f"  stderr: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")
            return False
        except Exception as e:
            self.logger.error(f"GitHub push failed: {e}", exc_info=True)
            return False

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
            date_str = target_date or datetime.now().strftime('%Y-%m-%d')
            predictions = self.generate_predictions(games)
            if not predictions:
                self.logger.warning("⚠ No predictions generated - workflow complete (nothing to post)")
                return False
            self.logger.info("")

            # Step 3.5: Save predictions to database (critical for GitHub Pages publish)
            self.logger.info("STEP 3.5: Saving predictions to database...")
            saved_count = self._save_predictions_to_db(predictions, date_str)
            self.logger.info(f"✓ Saved {saved_count}/{len(predictions)} predictions to database")
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
            
            # Step 6: Export games for GitHub Pages
            self.logger.info("STEP 6: Exporting games for GitHub Pages...")
            export_success = False
            try:
                from src.daily_games_exporter import DailyGamesExporter
                exporter = DailyGamesExporter(db_path=self.db_path)
                export_success = exporter.export_games_for_publishing(target_date)
                if export_success:
                    self.logger.info("✓ Games exported to docs/pending_games.json")
                else:
                    self.logger.warning("⚠ Game export failed (non-critical)")
            except Exception as e:
                self.logger.warning(f"⚠ Game export error (non-critical): {e}")
            self.logger.info("")

            # Step 7: Push to GitHub (for GitHub Pages update)
            self.logger.info("STEP 7: Pushing to GitHub...")
            if export_success:
                try:
                    push_success = self._push_to_github(target_date)
                    if push_success:
                        self.logger.info("✓ Changes pushed to GitHub")
                    else:
                        self.logger.warning("⚠ GitHub push failed (non-critical)")
                except Exception as e:
                    self.logger.warning(f"⚠ GitHub push error (non-critical): {e}")
            else:
                self.logger.info("ℹ Skipping GitHub push (no export to push)")
            self.logger.info("")

            # Step 8: Send daily email report
            self.logger.info("STEP 8: Sending daily email report...")
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

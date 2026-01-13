"""
Model Feedback & Continuous Learning System
Tracks predictions, fetches actual results, and provides insights for retraining
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from nba_api.stats.endpoints import scoreboardv2, leaguegamefinder
import time
from nba_api.stats.static import teams


class ModelFeedbackSystem:
    """
    Manages the feedback loop between predictions and actual results.
    Enables continuous improvement by tracking model performance.
    """

    def __init__(self, db_path: str = "data/nba_predictor.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._ensure_schema()
        
        # Create team name normalization maps (full name <-> abbreviation)
        all_teams = teams.get_teams()
        self.team_full_to_abbrev = {t['full_name']: t['abbreviation'] for t in all_teams}
        self.team_abbrev_to_full = {t['abbreviation']: t['full_name'] for t in all_teams}
        self.team_name_variations = {}
        for t in all_teams:
            # Create variations: full name, abbreviation, lowercase versions
            full = t['full_name']
            abbrev = t['abbreviation']
            self.team_name_variations[full.lower()] = full
            self.team_name_variations[abbrev.lower()] = full
            self.team_name_variations[full] = full
            self.team_name_variations[abbrev] = full

            # Add common alternative names (e.g., "LA Clippers" for "Los Angeles Clippers")
            if 'Los Angeles' in full:
                la_short = full.replace('Los Angeles', 'LA')
                self.team_name_variations[la_short.lower()] = full
                self.team_name_variations[la_short] = full

    def _ensure_schema(self):
        """Ensure predictions table has all necessary columns"""
        cursor = self.conn.cursor()

        # Create predictions table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT,
                game_date TEXT,
                game_id TEXT,
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
                prediction_error REAL,
                calibration_error REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create model performance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_date TEXT,
                period_start TEXT,
                period_end TEXT,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                accuracy REAL,
                brier_score REAL,
                calibration_score REAL,
                avg_confidence REAL,
                high_conf_accuracy REAL,
                low_conf_accuracy REAL,
                home_bias REAL,
                favorite_bias REAL,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()

    def save_prediction(self, prediction_data: Dict):
        """
        Save a prediction to the database

        Args:
            prediction_data: Dict with keys:
                - game_date, game_id, home_team, away_team
                - predicted_winner, predicted_home_prob, predicted_away_prob
                - confidence
        """
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO predictions (
                prediction_date, game_date, game_id, home_team, away_team,
                predicted_winner, predicted_home_prob, predicted_away_prob, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            prediction_data['game_date'],
            prediction_data.get('game_id', ''),
            prediction_data['home_team'],
            prediction_data['away_team'],
            prediction_data['predicted_winner'],
            prediction_data['predicted_home_prob'],
            prediction_data['predicted_away_prob'],
            prediction_data['confidence']
        ))

        self.conn.commit()
        return cursor.lastrowid

    def update_predictions_with_results(self, lookback_days: int = 30, use_api: bool = False) -> int:
        """
        Fetch actual game results and update predictions.
        First checks database, then NBA API if needed.

        Args:
            lookback_days: How many days back to check for results (default: 30)

        Returns:
            Number of predictions updated
        """
        cursor = self.conn.cursor()

        # Get unverified predictions from last N days
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT id, game_date, home_team, away_team,
                   predicted_home_prob, predicted_away_prob
            FROM predictions
            WHERE actual_winner IS NULL
            AND game_date >= ?
            AND game_date <= ?
        ''', (cutoff_date, today))

        pending_predictions = cursor.fetchall()
        print(f"Found {len(pending_predictions)} pending predictions to update (date range: {cutoff_date} to {today})")
        if len(pending_predictions) == 0:
            # Check if there are any predictions at all
            cursor.execute('SELECT COUNT(*) FROM predictions WHERE actual_winner IS NULL')
            total_pending = cursor.fetchone()[0]
            if total_pending > 0:
                cursor.execute('SELECT MIN(game_date), MAX(game_date) FROM predictions WHERE actual_winner IS NULL')
                row = cursor.fetchone()
                if row and row[0]:
                    min_date, max_date = row
                    print(f"  Note: There are {total_pending} pending predictions total (dates: {min_date} to {max_date})")
                    if max_date and max_date > today:
                        print(f"  [WARN]  Some predictions are in the future (max: {max_date}, today: {today})")
        updated_count = 0
        db_updated = 0
        api_updated = 0
        skipped_future = 0
        skipped_not_found = 0

        for pred_id, game_date, home_team, away_team, pred_home_prob, pred_away_prob in pending_predictions:
            # Normalize game_date format (handle different formats)
            try:
                # Try to parse and reformat date
                if isinstance(game_date, str):
                    # Handle different date formats
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            date_obj = datetime.strptime(game_date, fmt).date()
                            game_date = date_obj.strftime('%Y-%m-%d')  # Normalize to YYYY-MM-DD
                            break
                        except:
                            continue
                    else:
                        # If no format matched, try to parse as date object
                        date_obj = pd.to_datetime(game_date).date()
                        game_date = date_obj.strftime('%Y-%m-%d')
                else:
                    date_obj = pd.to_datetime(game_date).date()
                    game_date = date_obj.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"  [WARN]  Warning: Could not parse game_date {game_date}: {e}")
                continue
            
            # Check if game date is today or in the future
            try:
                date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()
                today_obj = datetime.now().date()
                if date_obj > today_obj:
                    skipped_future += 1
                    print(f"  [SKIP] Skipping {home_team} vs {away_team} on {game_date} - future game")
                    continue
                if date_obj == today_obj:
                    # Today's game - might not be played yet, but try anyway
                    print(f"  [TODAY] Checking today's game: {home_team} vs {away_team} (might not be finished yet)")
            except Exception as e:
                print(f"  [WARN] Could not parse game_date {game_date}: {e}")
                continue
            
            # First, try to get result from database
            result = self._fetch_game_result_from_db(game_date, home_team, away_team)

            if not result:
                if use_api:
                    # Not found in DB - try NBA API (slow)
                    print(f"  [API] Not in DB, trying NBA API for {home_team} vs {away_team} on {game_date}...")
                    result = self._fetch_game_result(game_date, home_team, away_team)
                    if result:
                        api_updated += 1
                        print(f"  [OK] Found result via API: {home_team} vs {away_team} on {game_date}")
                    else:
                        skipped_not_found += 1
                        print(f"  [MISS] Not found in API for {game_date}")
                    # Rate limit API calls
                    time.sleep(0.6)
                else:
                    # DB-only mode - just count as not found
                    skipped_not_found += 1
                    # Quick debug info
                    cursor.execute('SELECT COUNT(*) FROM games WHERE game_date = ?', (game_date,))
                    games_on_date = cursor.fetchone()[0]
                    if games_on_date > 0:
                        print(f"  [MISS] No match in DB for {home_team} vs {away_team} on {game_date} ({games_on_date} games exist)")
                    else:
                        print(f"  [MISS] No games in DB for {game_date}")
            else:
                db_updated += 1
                print(f"  [OK] Found in DB: {home_team} vs {away_team} on {game_date}")

            if result:
                home_score, away_score, actual_winner = result

                # Normalize team names for comparison
                home_full, home_abbrev = self._normalize_team_name(home_team)
                away_full, away_abbrev = self._normalize_team_name(away_team)
                winner_full, winner_abbrev = self._normalize_team_name(actual_winner)
                
                # Calculate if prediction was correct
                predicted_winner = home_team if pred_home_prob > pred_away_prob else away_team
                pred_winner_full, pred_winner_abbrev = self._normalize_team_name(predicted_winner)
                
                # Match winner names (handles different formats)
                actual_is_home = (winner_full == home_full or winner_abbrev == home_abbrev or 
                                 actual_winner.lower() == home_team.lower())
                prediction_was_home = (pred_winner_full == home_full or pred_winner_abbrev == home_abbrev or
                                      predicted_winner.lower() == home_team.lower())
                
                predicted_winner_correct = 1 if actual_is_home == prediction_was_home else 0

                # Calculate prediction error (Brier score component)
                actual_home_outcome = 1 if actual_is_home else 0
                prediction_error = (pred_home_prob - actual_home_outcome) ** 2

                # Calculate calibration error
                calibration_error = abs(pred_home_prob - actual_home_outcome)

                # Update prediction record
                cursor.execute('''
                    UPDATE predictions
                    SET actual_winner = ?,
                        actual_home_score = ?,
                        actual_away_score = ?,
                        correct = ?,
                        prediction_error = ?,
                        calibration_error = ?
                    WHERE id = ?
                ''', (actual_winner, home_score, away_score, predicted_winner_correct,
                      prediction_error, calibration_error, pred_id))

                updated_count += 1
                print(f"  [OK] Updated {home_team} vs {away_team} on {game_date}: {actual_winner} won")

        self.conn.commit()
        print(f"[OK] Update complete: {updated_count} updated ({db_updated} from DB, {api_updated} from API)")
        if skipped_future > 0:
            print(f"  [SKIP]  Skipped {skipped_future} future games")
        if skipped_not_found > 0:
            print(f"  [WARN]  Could not find results for {skipped_not_found} games")
        return updated_count
    
    def _normalize_team_name(self, team_name: str) -> Tuple[str, str]:
        """
        Normalize team name to handle both full names and abbreviations.
        Returns (full_name, abbreviation) tuple.
        """
        if not team_name:
            return ("", "")
        
        team_lower = team_name.lower().strip()
        
        # Check if it's already in our variations map
        if team_lower in self.team_name_variations:
            full = self.team_name_variations[team_lower]
            abbrev = self.team_full_to_abbrev.get(full, "")
            return (full, abbrev)
        
        # Try direct lookup
        if team_name in self.team_full_to_abbrev:
            return (team_name, self.team_full_to_abbrev[team_name])
        if team_name in self.team_abbrev_to_full:
            return (self.team_abbrev_to_full[team_name], team_name)
        
        # Try fuzzy matching - check if team name contains or is contained in known names
        for full_name, abbrev in self.team_full_to_abbrev.items():
            if team_lower in full_name.lower() or full_name.lower() in team_lower:
                return (full_name, abbrev)
            if team_lower == abbrev.lower():
                return (full_name, abbrev)
        
        # Return as-is if no match found
        return (team_name, team_name)
    
    def _fetch_game_result_from_db(self, game_date: str, home_team: str, away_team: str) -> Optional[Tuple[int, int, str]]:
        """
        Try to get game result from local database first (faster than API).

        Returns:
            (home_score, away_score, winning_team) or None if not found
        """
        try:
            cursor = self.conn.cursor()

            # Normalize team names to handle both full names and abbreviations
            home_full, home_abbrev = self._normalize_team_name(home_team)
            away_full, away_abbrev = self._normalize_team_name(away_team)

            # Build list of all possible team name combinations to try
            home_variants = list(set([home_team, home_full, home_abbrev, home_team.upper(), home_abbrev.upper()]))
            away_variants = list(set([away_team, away_full, away_abbrev, away_team.upper(), away_abbrev.upper()]))

            # Also try just the city/mascot parts
            if ' ' in home_full:
                home_variants.extend([home_full.split()[-1], home_full.split()[0]])  # "Celtics", "Boston"
            if ' ' in away_full:
                away_variants.extend([away_full.split()[-1], away_full.split()[0]])

            # Special handling for "LA" vs "Los Angeles" teams
            # "Los Angeles Clippers" -> also try "LA Clippers"
            # "Los Angeles Lakers" -> also try "LA Lakers"
            if 'Los Angeles' in home_full:
                la_short = home_full.replace('Los Angeles', 'LA')
                home_variants.append(la_short)
            if 'Los Angeles' in away_full:
                la_short = away_full.replace('Los Angeles', 'LA')
                away_variants.append(la_short)

            # Remove empty strings and duplicates
            home_variants = list(set([h for h in home_variants if h]))
            away_variants = list(set([a for a in away_variants if a]))

            # Strategy 1: Try all combinations of name variants
            for h in home_variants:
                for a in away_variants:
                    cursor.execute('''
                        SELECT home_score, away_score, home_team, away_team
                        FROM games
                        WHERE game_date = ?
                        AND home_team = ? AND away_team = ?
                        AND home_score IS NOT NULL AND away_score IS NOT NULL
                        LIMIT 1
                    ''', (game_date, h, a))

                    result = cursor.fetchone()
                    if result:
                        home_score, away_score, db_home, db_away = result
                        winner = db_home if home_score > away_score else db_away
                        # Normalize winner name to match prediction format
                        winner_full, _ = self._normalize_team_name(winner)
                        return (int(home_score), int(away_score), winner_full)

            # Strategy 2: Try using team IDs (most reliable)
            # Get team IDs from NBA API
            team_map = {t['full_name']: t['id'] for t in teams.get_teams()}
            home_id = team_map.get(home_full) or team_map.get(home_team)
            away_id = team_map.get(away_full) or team_map.get(away_team)

            if home_id and away_id:
                cursor.execute('''
                    SELECT home_score, away_score, home_team, away_team
                    FROM games
                    WHERE game_date = ?
                    AND home_team_id = ? AND away_team_id = ?
                    AND home_score IS NOT NULL AND away_score IS NOT NULL
                    LIMIT 1
                ''', (game_date, home_id, away_id))

                result = cursor.fetchone()
                if result:
                    home_score, away_score, db_home, db_away = result
                    winner = db_home if home_score > away_score else db_away
                    winner_full, _ = self._normalize_team_name(winner)
                    return (int(home_score), int(away_score), winner_full)

            # Debug info when not found
            cursor.execute('SELECT COUNT(*) FROM games WHERE game_date = ?', (game_date,))
            games_on_date = cursor.fetchone()[0]
            if games_on_date > 0:
                # Show what games are actually in the DB for this date
                cursor.execute('''
                    SELECT home_team, away_team, home_team_id, away_team_id
                    FROM games WHERE game_date = ? LIMIT 10
                ''', (game_date,))
                db_games = cursor.fetchall()
                print(f"  [NOTFOUND] No match for {home_full} vs {away_full} on {game_date}")
                print(f"     DB has {games_on_date} games. Sample: {db_games[:3]}")
            return None
        except Exception as e:
            print(f"Error checking database for game result: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _fetch_game_result(self, game_date: str, home_team: str, away_team: str) -> Tuple[int, int, str]:
        """
        Fetch actual game result from NBA API using leaguegamefinder (more reliable for historical games)

        Returns:
            (home_score, away_score, winning_team) or None if not found
        """
        try:
            # Parse game date
            date_obj = datetime.strptime(game_date, '%Y-%m-%d')
            
            # Don't try to fetch results for future games
            if date_obj.date() > datetime.now().date():
                print(f"  ⏩ Skipping future game: {home_team} vs {away_team} on {game_date}")
                return None

            # Normalize team names
            home_full, home_abbrev = self._normalize_team_name(home_team)
            away_full, away_abbrev = self._normalize_team_name(away_team)
            
            print(f"  🔍 Fetching from NBA API: {home_full} ({home_abbrev}) vs {away_full} ({away_abbrev}) on {game_date}")
            
            # Get team IDs
            team_map = {t['full_name']: t['id'] for t in teams.get_teams()}
            team_map_abbrev = {t['abbreviation']: t['id'] for t in teams.get_teams()}
            
            home_id = team_map.get(home_full) or team_map_abbrev.get(home_abbrev)
            away_id = team_map.get(away_full) or team_map_abbrev.get(away_abbrev)
            
            if not home_id or not away_id:
                print(f"    ❌ Could not find team IDs: home={home_id}, away={away_id}")
                return None
            
            # Use leaguegamefinder which is more reliable for historical data
            # Fetch games for the home team on this date
            time.sleep(0.6)  # Rate limit
            finder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=home_id,
                date_from_nullable=game_date.replace('-', '/'),
                date_to_nullable=game_date.replace('-', '/'),
                season_type_nullable='Regular Season'
            )
            
            games_df = finder.get_data_frames()[0]
            
            if games_df.empty:
                print(f"    ⚠️ No games found for {home_full} on {game_date}")
                # Try fetching for away team instead
                time.sleep(0.6)
                finder = leaguegamefinder.LeagueGameFinder(
                    team_id_nullable=away_id,
                    date_from_nullable=game_date.replace('-', '/'),
                    date_to_nullable=game_date.replace('-', '/'),
                    season_type_nullable='Regular Season'
                )
                games_df = finder.get_data_frames()[0]
                
                if games_df.empty:
                    print(f"    ❌ No games found for either team on {game_date}")
                    return None
            
            print(f"    📋 Found {len(games_df)} game entries")
            
            # Find the game between these two teams
            for _, row in games_df.iterrows():
                matchup = row.get('MATCHUP', '')  # Format: "BOS vs. NYK" or "BOS @ NYK"
                team_abbrev = row.get('TEAM_ABBREVIATION', '')
                
                # Check if this is our game
                # The matchup string contains both teams
                opponent_in_matchup = (
                    away_abbrev in matchup or 
                    home_abbrev in matchup or
                    away_full.split()[-1] in matchup or  # Last word of team name
                    home_full.split()[-1] in matchup
                )
                
                if not opponent_in_matchup:
                    continue
                
                # Get game details
                pts = row.get('PTS')
                wl = row.get('WL')  # 'W' or 'L'
                plus_minus = row.get('PLUS_MINUS', 0)
                
                if pts is None or wl is None:
                    print(f"    ⚠️ Missing data for game: {matchup}")
                    continue
                
                print(f"    🎯 Found: {matchup} - {team_abbrev} scored {pts} ({wl})")
                
                # Determine home/away from matchup
                # "BOS vs. NYK" = BOS is home, "BOS @ NYK" = BOS is away
                is_home = 'vs.' in matchup or 'vs' in matchup.lower()
                
                if team_abbrev == home_abbrev:
                    home_score = int(pts)
                    # Calculate away score from plus_minus
                    if wl == 'W':
                        away_score = home_score - abs(int(plus_minus))
                    else:
                        away_score = home_score + abs(int(plus_minus))
                    winner = home_full if wl == 'W' else away_full
                elif team_abbrev == away_abbrev:
                    away_score = int(pts)
                    if wl == 'W':
                        home_score = away_score - abs(int(plus_minus))
                    else:
                        home_score = away_score + abs(int(plus_minus))
                    winner = away_full if wl == 'W' else home_full
                else:
                    # This team's game but we're looking at opponent's stats
                    # Try to match by context
                    continue
                
                print(f"    🏆 Result: {home_full} {home_score} - {away_full} {away_score}, Winner: {winner}")
                return (int(home_score), int(away_score), winner)
            
            print(f"    ❌ No matching game found between {home_full} and {away_full}")
            return None

        except Exception as e:
            print(f"    ❌ API Error for {home_team} vs {away_team} on {game_date}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def evaluate_model_performance(self, period_days: int = 30) -> Dict:
        """
        Evaluate model performance over a period

        Args:
            period_days: Number of days to evaluate

        Returns:
            Dict with performance metrics
        """
        cursor = self.conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')

        # Get verified predictions
        cursor.execute('''
            SELECT predicted_home_prob, predicted_away_prob, confidence,
                   correct, prediction_error, calibration_error,
                   predicted_winner, actual_winner
            FROM predictions
            WHERE actual_winner IS NOT NULL
            AND game_date >= ?
        ''', (cutoff_date,))

        predictions = cursor.fetchall()

        if not predictions:
            return {
                'total_predictions': 0,
                'accuracy': 0.0,
                'brier_score': 0.0,
                'calibration_score': 0.0,
                'error': 'No verified predictions in period'
            }

        df = pd.DataFrame(predictions, columns=[
            'pred_home_prob', 'pred_away_prob', 'confidence',
            'correct', 'pred_error', 'calib_error',
            'predicted_winner', 'actual_winner'
        ])

        # Calculate metrics
        total = len(df)
        accuracy = df['correct'].mean()
        brier_score = df['pred_error'].mean()
        calibration_score = df['calib_error'].mean()
        avg_confidence = df['confidence'].mean()

        # High vs low confidence accuracy
        high_conf = df[df['confidence'] >= 0.70]
        low_conf = df[df['confidence'] < 0.70]

        high_conf_acc = high_conf['correct'].mean() if len(high_conf) > 0 else 0.0
        low_conf_acc = low_conf['correct'].mean() if len(low_conf) > 0 else 0.0

        # Bias analysis
        home_predicted = df[df['pred_home_prob'] > 0.5]
        home_bias = home_predicted['correct'].mean() if len(home_predicted) > 0 else 0.0

        # Favorite bias (predicted higher probability team)
        favorite_correct = df[df['confidence'] >= 0.60]['correct'].mean() if len(df[df['confidence'] >= 0.60]) > 0 else 0.0

        metrics = {
            'total_predictions': total,
            'correct_predictions': int(df['correct'].sum()),
            'accuracy': round(accuracy, 4),
            'brier_score': round(brier_score, 4),
            'calibration_score': round(calibration_score, 4),
            'avg_confidence': round(avg_confidence, 4),
            'high_conf_accuracy': round(high_conf_acc, 4),
            'low_conf_accuracy': round(low_conf_acc, 4),
            'home_bias': round(home_bias, 4),
            'favorite_bias': round(favorite_correct, 4),
            'period_days': period_days,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d')
        }

        # Save performance snapshot
        self._save_performance_snapshot(metrics)

        return metrics

    def _save_performance_snapshot(self, metrics: Dict):
        """Save performance metrics to database"""
        cursor = self.conn.cursor()

        period_start = (datetime.now() - timedelta(days=metrics['period_days'])).strftime('%Y-%m-%d')
        period_end = datetime.now().strftime('%Y-%m-%d')

        cursor.execute('''
            INSERT INTO model_performance (
                evaluation_date, period_start, period_end,
                total_predictions, correct_predictions, accuracy,
                brier_score, calibration_score, avg_confidence,
                high_conf_accuracy, low_conf_accuracy,
                home_bias, favorite_bias
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['evaluation_date'], period_start, period_end,
            metrics['total_predictions'], metrics['correct_predictions'],
            metrics['accuracy'], metrics['brier_score'], metrics['calibration_score'],
            metrics['avg_confidence'], metrics['high_conf_accuracy'],
            metrics['low_conf_accuracy'], metrics['home_bias'], metrics['favorite_bias']
        ))

        self.conn.commit()

    def get_retraining_recommendations(self) -> Dict:
        """
        Analyze model performance and recommend retraining

        Returns:
            Dict with recommendation and reasoning
        """
        # Get recent performance (last 30 days)
        recent_perf = self.evaluate_model_performance(period_days=30)

        # Get longer-term performance (last 90 days)
        longterm_perf = self.evaluate_model_performance(period_days=90)

        recommendations = {
            'should_retrain': False,
            'urgency': 'low',  # low, medium, high
            'reasons': [],
            'recent_performance': recent_perf,
            'longterm_performance': longterm_perf
        }

        # Check if accuracy dropped significantly
        if recent_perf['accuracy'] < longterm_perf['accuracy'] - 0.05:
            recommendations['should_retrain'] = True
            recommendations['urgency'] = 'high'
            recommendations['reasons'].append('Accuracy dropped >5% in last 30 days')

        # Check if Brier score worsened
        if recent_perf['brier_score'] > longterm_perf['brier_score'] * 1.2:
            recommendations['should_retrain'] = True
            recommendations['urgency'] = 'medium'
            recommendations['reasons'].append('Brier score worsened by 20%')

        # Check if we have enough data for meaningful evaluation
        if recent_perf['total_predictions'] < 20:
            recommendations['reasons'].append('Insufficient recent predictions (<20)')

        # Check calibration
        if recent_perf['calibration_score'] > 0.15:
            recommendations['should_retrain'] = True
            recommendations['urgency'] = 'medium'
            recommendations['reasons'].append('Poor calibration (error > 0.15)')

        # Time-based recommendation
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(evaluation_date) FROM model_performance')
        last_eval = cursor.fetchone()[0]

        if last_eval:
            days_since_eval = (datetime.now() - datetime.strptime(last_eval, '%Y-%m-%d')).days
            if days_since_eval > 7:
                recommendations['should_retrain'] = True
                recommendations['urgency'] = 'low'
                recommendations['reasons'].append(f'Last training was {days_since_eval} days ago')

        return recommendations

    def close(self):
        """Close database connection"""
        self.conn.close()


if __name__ == "__main__":
    # Test the feedback system
    feedback = ModelFeedbackSystem()

    print("=" * 60)
    print("MODEL FEEDBACK SYSTEM TEST")
    print("=" * 60)
    print()

    # Update predictions with results
    print("Updating predictions with actual results...")
    updated = feedback.update_predictions_with_results(lookback_days=7)
    print(f"Updated {updated} predictions")
    print()

    # Evaluate performance
    print("Evaluating model performance (last 30 days)...")
    perf = feedback.evaluate_model_performance(period_days=30)
    print(f"Total Predictions: {perf['total_predictions']}")
    print(f"Accuracy: {perf['accuracy']:.1%}")
    print(f"Brier Score: {perf['brier_score']:.4f}")
    print(f"Calibration: {perf['calibration_score']:.4f}")
    print()

    # Get retraining recommendations
    print("Checking retraining recommendations...")
    recs = feedback.get_retraining_recommendations()
    print(f"Should Retrain: {recs['should_retrain']}")
    print(f"Urgency: {recs['urgency']}")
    if recs['reasons']:
        print("Reasons:")
        for reason in recs['reasons']:
            print(f"  - {reason}")

    feedback.close()

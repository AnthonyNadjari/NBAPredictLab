#!/usr/bin/env python3
"""
Morning Routine Script for NBA Predictor
=========================================
This script automates the morning tasks:
1. Fetch today's predictions (generate predictions for today's games)
2. Refresh game data from NBA API (get yesterday's scores)
3. Update prediction results (verify yesterday's predictions)
4. Send email report (today's predictions + yesterday's results)

Usage:
    python scripts/morning_routine.py [--skip-email] [--skip-predictions]
"""

import sys
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / f'morning_routine_{datetime.now().strftime("%Y%m")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = PROJECT_ROOT / 'data' / 'nba_predictor.db'


def refresh_game_data(lookback_days: int = 7) -> bool:
    """
    Refresh game data from NBA API (equivalent to "Refresh Game Data" button).
    Fetches recent game scores to update the database.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Refreshing game data from NBA API")
    logger.info("=" * 60)

    try:
        import sqlite3
        from nba_api.stats.endpoints import leaguegamefinder
        import time

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        date_from = start_date.strftime('%m/%d/%Y')
        date_to = end_date.strftime('%m/%d/%Y')

        logger.info(f"Fetching games from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

        # Fetch games using leaguegamefinder (with timeout to avoid hanging forever)
        time.sleep(1)  # Rate limiting
        games_df = None
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lambda: leaguegamefinder.LeagueGameFinder(
                        date_from_nullable=date_from,
                        date_to_nullable=date_to,
                        league_id_nullable='00'  # NBA
                    ).get_data_frames()[0]
                )
                games_df = future.result(timeout=45)
        except FuturesTimeoutError:
            logger.warning("[WARN] LeagueGameFinder timed out (45s), skipping game refresh for this run")
            return True
        except Exception as e:
            logger.warning(f"[WARN] LeagueGameFinder failed: {e}")
            return True

        if games_df.empty:
            logger.warning("No games found in date range")
            return True

        # Filter to completed games only
        games_df = games_df[games_df['WL'].notna()].copy()

        if games_df.empty:
            logger.info("No completed games in date range")
            return True

        # Process and insert games
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Group by game to get both teams
        game_ids = games_df['GAME_ID'].unique()
        games_inserted = 0
        games_updated = 0

        for game_id in game_ids:
            game_rows = games_df[games_df['GAME_ID'] == game_id]
            if len(game_rows) != 2:
                continue

            # Determine home vs away
            home_row = game_rows[game_rows['MATCHUP'].str.contains(' vs. ')].iloc[0] if len(game_rows[game_rows['MATCHUP'].str.contains(' vs. ')]) > 0 else None
            away_row = game_rows[game_rows['MATCHUP'].str.contains(' @ ')].iloc[0] if len(game_rows[game_rows['MATCHUP'].str.contains(' @ ')]) > 0 else None

            if home_row is None or away_row is None:
                continue

            game_date = home_row['GAME_DATE']
            home_team = home_row['TEAM_NAME']
            away_team = away_row['TEAM_NAME']
            home_score = int(home_row['PTS'])
            away_score = int(away_row['PTS'])
            home_team_id = int(home_row['TEAM_ID'])
            away_team_id = int(away_row['TEAM_ID'])
            home_win = 1 if home_score > away_score else 0

            # Extract box score stats (these come from LeagueGameFinder)
            def safe_int(val, default=0):
                try:
                    return int(val) if val is not None else default
                except (TypeError, ValueError):
                    return default

            def safe_float(val, default=0.0):
                try:
                    return float(val) if val is not None else default
                except (TypeError, ValueError):
                    return default

            box_stats = {
                'home_fgm': safe_int(home_row.get('FGM')), 'home_fga': safe_int(home_row.get('FGA')),
                'home_fg_pct': safe_float(home_row.get('FG_PCT')),
                'home_fg3m': safe_int(home_row.get('FG3M')), 'home_fg3a': safe_int(home_row.get('FG3A')),
                'home_fg3_pct': safe_float(home_row.get('FG3_PCT')),
                'home_ftm': safe_int(home_row.get('FTM')), 'home_fta': safe_int(home_row.get('FTA')),
                'home_ft_pct': safe_float(home_row.get('FT_PCT')),
                'home_oreb': safe_int(home_row.get('OREB')), 'home_dreb': safe_int(home_row.get('DREB')),
                'home_reb': safe_int(home_row.get('REB')), 'home_ast': safe_int(home_row.get('AST')),
                'home_stl': safe_int(home_row.get('STL')), 'home_blk': safe_int(home_row.get('BLK')),
                'home_tov': safe_int(home_row.get('TOV')),
                'away_fgm': safe_int(away_row.get('FGM')), 'away_fga': safe_int(away_row.get('FGA')),
                'away_fg_pct': safe_float(away_row.get('FG_PCT')),
                'away_fg3m': safe_int(away_row.get('FG3M')), 'away_fg3a': safe_int(away_row.get('FG3A')),
                'away_fg3_pct': safe_float(away_row.get('FG3_PCT')),
                'away_ftm': safe_int(away_row.get('FTM')), 'away_fta': safe_int(away_row.get('FTA')),
                'away_ft_pct': safe_float(away_row.get('FT_PCT')),
                'away_oreb': safe_int(away_row.get('OREB')), 'away_dreb': safe_int(away_row.get('DREB')),
                'away_reb': safe_int(away_row.get('REB')), 'away_ast': safe_int(away_row.get('AST')),
                'away_stl': safe_int(away_row.get('STL')), 'away_blk': safe_int(away_row.get('BLK')),
                'away_tov': safe_int(away_row.get('TOV')),
            }

            # Check if game exists (by game_id first, then by date+teams)
            cursor.execute("SELECT game_id FROM games WHERE game_id = ?", (game_id,))
            existing = cursor.fetchone()

            if not existing:
                cursor.execute("""
                    SELECT game_id FROM games
                    WHERE game_date = ? AND home_team = ? AND away_team = ?
                """, (game_date, home_team, away_team))
                existing = cursor.fetchone()

            if existing:
                # Update existing game with scores AND box score stats
                cursor.execute("""
                    UPDATE games SET
                        home_score = ?, away_score = ?, home_win = ?,
                        home_team_id = ?, away_team_id = ?,
                        home_fgm = ?, home_fga = ?, home_fg_pct = ?,
                        home_fg3m = ?, home_fg3a = ?, home_fg3_pct = ?,
                        home_ftm = ?, home_fta = ?, home_ft_pct = ?,
                        home_oreb = ?, home_dreb = ?, home_reb = ?,
                        home_ast = ?, home_stl = ?, home_blk = ?, home_tov = ?,
                        away_fgm = ?, away_fga = ?, away_fg_pct = ?,
                        away_fg3m = ?, away_fg3a = ?, away_fg3_pct = ?,
                        away_ftm = ?, away_fta = ?, away_ft_pct = ?,
                        away_oreb = ?, away_dreb = ?, away_reb = ?,
                        away_ast = ?, away_stl = ?, away_blk = ?, away_tov = ?
                    WHERE game_id = ?
                """, (home_score, away_score, home_win,
                      home_team_id, away_team_id,
                      box_stats['home_fgm'], box_stats['home_fga'], box_stats['home_fg_pct'],
                      box_stats['home_fg3m'], box_stats['home_fg3a'], box_stats['home_fg3_pct'],
                      box_stats['home_ftm'], box_stats['home_fta'], box_stats['home_ft_pct'],
                      box_stats['home_oreb'], box_stats['home_dreb'], box_stats['home_reb'],
                      box_stats['home_ast'], box_stats['home_stl'], box_stats['home_blk'], box_stats['home_tov'],
                      box_stats['away_fgm'], box_stats['away_fga'], box_stats['away_fg_pct'],
                      box_stats['away_fg3m'], box_stats['away_fg3a'], box_stats['away_fg3_pct'],
                      box_stats['away_ftm'], box_stats['away_fta'], box_stats['away_ft_pct'],
                      box_stats['away_oreb'], box_stats['away_dreb'], box_stats['away_reb'],
                      box_stats['away_ast'], box_stats['away_stl'], box_stats['away_blk'], box_stats['away_tov'],
                      existing[0]))
                games_updated += 1
            else:
                # Insert new game with full box score data
                cursor.execute("""
                    INSERT INTO games
                    (game_id, game_date, home_team_id, away_team_id,
                     home_team, away_team, home_score, away_score, home_win,
                     home_fgm, home_fga, home_fg_pct, home_fg3m, home_fg3a, home_fg3_pct,
                     home_ftm, home_fta, home_ft_pct,
                     home_oreb, home_dreb, home_reb, home_ast, home_stl, home_blk, home_tov,
                     away_fgm, away_fga, away_fg_pct, away_fg3m, away_fg3a, away_fg3_pct,
                     away_ftm, away_fta, away_ft_pct,
                     away_oreb, away_dreb, away_reb, away_ast, away_stl, away_blk, away_tov)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?)
                """, (game_id, game_date, home_team_id, away_team_id,
                      home_team, away_team, home_score, away_score, home_win,
                      box_stats['home_fgm'], box_stats['home_fga'], box_stats['home_fg_pct'],
                      box_stats['home_fg3m'], box_stats['home_fg3a'], box_stats['home_fg3_pct'],
                      box_stats['home_ftm'], box_stats['home_fta'], box_stats['home_ft_pct'],
                      box_stats['home_oreb'], box_stats['home_dreb'], box_stats['home_reb'],
                      box_stats['home_ast'], box_stats['home_stl'], box_stats['home_blk'], box_stats['home_tov'],
                      box_stats['away_fgm'], box_stats['away_fga'], box_stats['away_fg_pct'],
                      box_stats['away_fg3m'], box_stats['away_fg3a'], box_stats['away_fg3_pct'],
                      box_stats['away_ftm'], box_stats['away_fta'], box_stats['away_ft_pct'],
                      box_stats['away_oreb'], box_stats['away_dreb'], box_stats['away_reb'],
                      box_stats['away_ast'], box_stats['away_stl'], box_stats['away_blk'], box_stats['away_tov']))
                games_inserted += 1

        conn.commit()
        conn.close()

        logger.info(f"[OK] Inserted {games_inserted} new games, updated {games_updated} games")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Failed to refresh game data: {e}", exc_info=True)
        return False


def update_prediction_results(lookback_days: int = 7) -> bool:
    """
    Update prediction results (equivalent to "Update Results" button).
    Matches predictions to actual game outcomes.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Updating prediction results")
    logger.info("=" * 60)

    try:
        from daily_auto_prediction import DailyPredictionAutomation

        automation = DailyPredictionAutomation(
            db_path=str(DB_PATH),
            model_dir=str(PROJECT_ROOT / 'models'),
            dry_run=True
        )

        # Initialize components
        if not automation.initialize_components():
            logger.error("Failed to initialize automation components")
            return False

        # Update previous predictions with results
        result = automation.check_and_update_previous_predictions(lookback_days=lookback_days)

        correct = result.get('correct', 0)
        incorrect = result.get('incorrect', 0)
        total_updated = correct + incorrect

        if total_updated > 0:
            logger.info(f"[OK] Updated {total_updated} predictions with results ({correct} correct, {incorrect} incorrect)")
        else:
            logger.info("[OK] No predictions to update (all already have results or no matching games)")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Failed to update prediction results: {e}", exc_info=True)
        return False


def fetch_todays_predictions() -> bool:
    """
    Fetch and generate today's AND tomorrow's predictions.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching today's and tomorrow's predictions")
    logger.info("=" * 60)

    try:
        from daily_auto_prediction import DailyPredictionAutomation

        automation = DailyPredictionAutomation(
            db_path=str(DB_PATH),
            model_dir=str(PROJECT_ROOT / 'models'),
            dry_run=True  # Don't post to Twitter
        )

        # Initialize components first
        if not automation.initialize_components():
            logger.error("Failed to initialize automation components")
            return False

        # Fetch today's games
        today_str = datetime.now().strftime('%Y-%m-%d')
        tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching games for today ({today_str})...")
        games_today = automation.fetch_todays_games(target_date=today_str)
        
        logger.info(f"Fetching games for tomorrow ({tomorrow_str})...")
        games_tomorrow = automation.fetch_todays_games(target_date=tomorrow_str)
        
        # Combine both days
        all_games = games_today + games_tomorrow if games_today and games_tomorrow else (games_today or games_tomorrow or [])
        
        if not all_games:
            logger.warning("[WARN] No games today or tomorrow")
            return True  # Not an error

        logger.info(f"Found {len(games_today) if games_today else 0} games today, {len(games_tomorrow) if games_tomorrow else 0} games tomorrow")
        
        # Generate predictions for all games
        predictions = automation.generate_predictions(all_games)

        # Save predictions to database (for both today and tomorrow)
        if predictions:
            # Each prediction carries its own game_date via game_info
            automation._save_predictions_to_db(predictions, today_str)
            logger.info(f"[OK] Generated {len(predictions)} predictions ({len(games_today) if games_today else 0} for today, {len(games_tomorrow) if games_tomorrow else 0} for tomorrow)")

            # Export to JSON for publishing interface (both today and tomorrow in one file)
            from src.daily_games_exporter import DailyGamesExporter
            exporter = DailyGamesExporter(str(DB_PATH))
            today_str = datetime.now().strftime('%Y-%m-%d')
            tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            # Single export with today first, tomorrow second — no overwrite
            export_success = exporter.export_today_and_tomorrow(today_str, tomorrow_str)

            if export_success:
                logger.info("[OK] Exported predictions to pending_games.json")

                # Git commit and push with robust conflict resolution
                import subprocess
                import json

                def validate_json_file(filepath):
                    """Check if JSON file is valid and has no git conflict markers."""
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Check for git conflict markers
                        if '<<<<<<<' in content or '=======' in content or '>>>>>>>' in content:
                            return False, "Git conflict markers found"
                        # Try to parse as JSON
                        json.loads(content)
                        return True, "Valid JSON"
                    except json.JSONDecodeError as e:
                        return False, f"Invalid JSON: {e}"
                    except FileNotFoundError:
                        return False, "File not found"

                def ensure_clean_json():
                    """Re-export JSON if it has conflicts or is invalid."""
                    json_path = PROJECT_ROOT / 'docs' / 'pending_games.json'
                    is_valid, msg = validate_json_file(json_path)
                    if not is_valid:
                        logger.info(f"[INFO] JSON file invalid ({msg}), re-exporting...")
                        exporter.export_games_for_publishing(today_str)
                        return True
                    return False

                try:
                    # Configure git for CI environment
                    if os.environ.get('GITHUB_ACTIONS'):
                        subprocess.run(['git', 'config', 'user.name', 'GitHub Actions Bot'], capture_output=True, cwd=str(PROJECT_ROOT))
                        subprocess.run(['git', 'config', 'user.email', 'actions@github.com'], capture_output=True, cwd=str(PROJECT_ROOT))

                    # First, check if we're in a broken rebase state and abort it
                    rebase_check = subprocess.run(
                        ['git', 'status'],
                        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
                    )
                    if 'rebase in progress' in rebase_check.stdout:
                        logger.info("[INFO] Found stuck rebase, aborting...")
                        subprocess.run(['git', 'rebase', '--abort'], capture_output=True, cwd=str(PROJECT_ROOT))
                        ensure_clean_json()

                    # Pull remote changes first with merge (not rebase) to simplify conflict handling
                    pull_first = subprocess.run(
                        ['git', 'pull', '--no-rebase'],
                        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
                    )

                    # Check if pull caused conflicts
                    if pull_first.returncode != 0 or 'CONFLICT' in pull_first.stdout + pull_first.stderr:
                        logger.info("[INFO] Conflict during pull, resolving...")
                        # For pending_games.json, always use our fresh export
                        ensure_clean_json()
                        subprocess.run(['git', 'add', 'docs/pending_games.json'], capture_output=True, cwd=str(PROJECT_ROOT))
                        # Try to complete the merge
                        subprocess.run(['git', 'commit', '-m', 'Resolve merge conflict by using fresh predictions'],
                                       capture_output=True, cwd=str(PROJECT_ROOT))

                    # Always validate JSON after any git operation
                    ensure_clean_json()

                    # Now add and commit our new predictions
                    subprocess.run(['git', 'add', 'docs/pending_games.json'], check=True, capture_output=True, cwd=str(PROJECT_ROOT))
                    subprocess.run(['git', 'add', 'data/nba_predictor.db'], check=True, capture_output=True, cwd=str(PROJECT_ROOT))
                    commit_result = subprocess.run(
                        ['git', 'commit', '-m', f'Auto-export predictions for {today_str}'],
                        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
                    )
                    if commit_result.returncode == 0:
                        # Push
                        push_result = subprocess.run(['git', 'push'], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                        if push_result.returncode != 0:
                            logger.info("[INFO] Push failed, trying pull and retry...")
                            # Pull with merge strategy
                            subprocess.run(['git', 'pull', '--no-rebase'], capture_output=True, cwd=str(PROJECT_ROOT))
                            # Always ensure JSON is clean after pull
                            ensure_clean_json()
                            subprocess.run(['git', 'add', 'docs/pending_games.json'], capture_output=True, cwd=str(PROJECT_ROOT))
                            subprocess.run(['git', 'commit', '-m', 'Ensure clean JSON after merge'],
                                           capture_output=True, cwd=str(PROJECT_ROOT))
                            # Final push attempt
                            retry_push = subprocess.run(['git', 'push'], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                            if retry_push.returncode == 0:
                                logger.info("[OK] Pushed predictions + database to GitHub")
                            else:
                                logger.warning(f"[WARN] Final push failed: {retry_push.stderr}")
                        else:
                            logger.info("[OK] Pushed predictions + database to GitHub")
                    else:
                        logger.info("[OK] No changes to commit (predictions already exported)")

                    # Final validation - always ensure the JSON is valid before finishing
                    is_valid, msg = validate_json_file(PROJECT_ROOT / 'docs' / 'pending_games.json')
                    if not is_valid:
                        logger.warning(f"[WARN] Final JSON validation failed: {msg}, re-exporting...")
                        exporter.export_games_for_publishing(today_str)
                        subprocess.run(['git', 'add', 'docs/pending_games.json'], capture_output=True, cwd=str(PROJECT_ROOT))
                        subprocess.run(['git', 'commit', '-m', 'Fix invalid JSON'], capture_output=True, cwd=str(PROJECT_ROOT))
                        subprocess.run(['git', 'push'], capture_output=True, cwd=str(PROJECT_ROOT))

                except subprocess.CalledProcessError as git_error:
                    logger.warning(f"[WARN] Git push failed: {git_error}")
            else:
                logger.warning("[WARN] Failed to export predictions to JSON")

            return True
        else:
            logger.warning("[WARN] No predictions generated (no games today?)")
            return True  # Not an error if no games

    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch predictions: {e}", exc_info=True)
        return False


def send_email_report() -> bool:
    """
    Send the daily email report.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Sending email report")
    logger.info("=" * 60)

    try:
        from src.email_reporter import EmailReporter

        email_reporter = EmailReporter(db_path=str(DB_PATH))
        success = email_reporter.send_daily_report(test_mode=False)

        if success:
            logger.info("[OK] Email report sent successfully")
        else:
            logger.warning("[WARN] Email report failed to send")

        return success

    except Exception as e:
        logger.error(f"[ERROR] Failed to send email: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='NBA Predictor Morning Routine')
    parser.add_argument('--skip-email', action='store_true', help='Skip sending email')
    parser.add_argument('--skip-predictions', action='store_true', help='Skip fetching today\'s predictions')
    parser.add_argument('--lookback', type=int, default=7, help='Days to look back for game data (default: 7)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NBA Predictor - Morning Routine")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Create logs directory if needed
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)

    all_success = True

    # CORRECT ORDER: Predictions -> Games -> Results -> Email

    # Step 1: Fetch today's predictions (so they're ready for email)
    if not args.skip_predictions:
        if not fetch_todays_predictions():
            all_success = False
    else:
        logger.info("\n[SKIP] Skipping predictions (--skip-predictions)")

    # Step 2: Refresh game data (get yesterday's scores BEFORE updating results)
    if not refresh_game_data(lookback_days=args.lookback):
        all_success = False

    # Step 3: Update prediction results (verify yesterday's predictions with fresh game data)
    if not update_prediction_results(lookback_days=args.lookback):
        all_success = False

    # Step 4: Send email (today's predictions + yesterday's results - now updated!)
    if not args.skip_email:
        if not send_email_report():
            all_success = False
    else:
        logger.info("\n[SKIP] Skipping email (--skip-email)")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    if all_success:
        logger.info("[OK] Morning routine completed successfully!")
    else:
        logger.info("[WARN] Morning routine completed with some warnings")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

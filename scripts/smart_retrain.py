#!/usr/bin/env python3
"""
Smart Retraining System
Automatically retrains the model based on:
1. New games played since last training
2. Model performance degradation
3. Prediction accuracy on recent games

Usage:
    python scripts/smart_retrain.py --check    # Check if retraining needed
    python scripts/smart_retrain.py --force    # Force retrain now
    python scripts/smart_retrain.py --auto     # Auto retrain if needed
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_fetcher import NBADataFetcher
from src.box_score_fetcher import BoxScoreFetcher
from src.models import ROCKSTARPredictor
from src.model_feedback_system import ModelFeedbackSystem
from src.elo_calculator import calculate_elo_ratings


def check_retraining_need():
    """
    Check if model needs retraining based on:
    1. Performance metrics
    2. Number of new games
    3. Time since last training
    """
    print("=" * 60)
    print("CHECKING IF RETRAINING IS NEEDED")
    print("=" * 60)

    reasons = []
    should_retrain = False
    urgency = "low"

    # 1. Check model feedback system
    try:
        db_path = project_root / 'data' / 'nba_predictor.db'
        feedback = ModelFeedbackSystem(str(db_path))

        # Update predictions with actual results
        print("\n[1/4] Updating predictions with actual results...")
        updated = feedback.update_predictions_with_results(lookback_days=7)
        print(f"  Updated {updated} predictions")

        # Get performance metrics
        print("\n[2/4] Evaluating model performance...")
        perf_7d = feedback.evaluate_model_performance(period_days=7)
        perf_30d = feedback.evaluate_model_performance(period_days=30)

        print(f"\n  7-day performance:")
        print(f"    Accuracy: {perf_7d.get('accuracy', 0)*100:.1f}%")
        print(f"    Predictions: {perf_7d.get('total_predictions', 0)}")
        print(f"    Brier Score: {perf_7d.get('brier_score', 0):.3f}")

        print(f"\n  30-day performance:")
        print(f"    Accuracy: {perf_30d.get('accuracy', 0)*100:.1f}%")
        print(f"    Predictions: {perf_30d.get('total_predictions', 0)}")
        print(f"    Brier Score: {perf_30d.get('brier_score', 0):.3f}")

        # Check if performance is degrading
        if perf_30d.get('accuracy', 1.0) < 0.55:
            reasons.append(f"Accuracy below 55% ({perf_30d.get('accuracy', 0)*100:.1f}%)")
            should_retrain = True
            urgency = "high"
        elif perf_30d.get('accuracy', 1.0) < 0.58:
            reasons.append(f"Accuracy below 58% ({perf_30d.get('accuracy', 0)*100:.1f}%)")
            should_retrain = True
            urgency = "medium"

        feedback.close()
    except Exception as e:
        print(f"  Warning: Could not check performance metrics: {e}")

    # 2. Check for new games
    print("\n[3/4] Checking for new games...")
    try:
        # Check when model was last trained
        model_info_path = project_root / 'models' / 'model_info.json'
        last_train_date = None

        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
                last_train_date = datetime.fromisoformat(model_info.get('trained_at', '2020-01-01'))
                print(f"  Last trained: {last_train_date.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("  No training info found - first training needed")
            reasons.append("No previous training found")
            should_retrain = True
            urgency = "high"

        # Count games since last training
        if last_train_date:
            fetcher = NBADataFetcher()
            current_season = fetcher.get_current_season()
            all_games = fetcher.fetch_historical_games(seasons=[current_season])

            # Filter games after last training
            all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
            new_games = all_games[all_games['GAME_DATE'] > last_train_date]

            print(f"  New games since last training: {len(new_games)}")

            if len(new_games) > 50:
                reasons.append(f"{len(new_games)} new games available")
                should_retrain = True
                if urgency == "low":
                    urgency = "medium"
            elif len(new_games) > 20:
                reasons.append(f"{len(new_games)} new games (consider retraining)")
                if urgency == "low":
                    urgency = "low"
    except Exception as e:
        print(f"  Warning: Could not check new games: {e}")

    # 3. Check time since last training
    print("\n[4/4] Checking time since last training...")
    if last_train_date:
        days_since = (datetime.now() - last_train_date).days
        print(f"  Days since last training: {days_since}")

        if days_since > 14:
            reasons.append(f"{days_since} days since last training")
            should_retrain = True
            if urgency == "low":
                urgency = "medium"
        elif days_since > 7:
            reasons.append(f"{days_since} days since last training (consider retraining)")

    # Summary
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"\nShould retrain: {'YES' if should_retrain else 'NO'}")
    print(f"Urgency: {urgency.upper()}")

    if reasons:
        print("\nReasons:")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("\nModel performance is good. No retraining needed yet.")

    print("\n" + "=" * 60)

    return should_retrain, urgency, reasons


def retrain_model(incremental=True):
    """
    Retrain the model

    Args:
        incremental: If True, only fetch new games since last training
    """
    print("\n" + "=" * 60)
    print("STARTING MODEL RETRAINING")
    print("=" * 60)

    start_time = datetime.now()

    try:
        # [1/5] Determine date range
        print("\n[1/5] Determining training data range...")

        if incremental:
            model_info_path = project_root / 'models' / 'model_info.json'
            if model_info_path.exists():
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    last_train_date = datetime.fromisoformat(model_info.get('trained_at', '2020-01-01'))
                    print(f"  Last training: {last_train_date.strftime('%Y-%m-%d')}")
                    print(f"  Will fetch NEW games only")
            else:
                incremental = False
                print("  No previous training found - doing full training")

        # [2/5] Fetch games
        print("\n[2/5] Fetching NBA games...")
        fetcher = NBADataFetcher()

        if incremental:
            # Fetch current season only
            seasons = [fetcher.get_current_season()]
        else:
            # Full training: last 3 seasons
            current_year = datetime.now().year
            seasons = [
                f"{current_year}-{str(current_year+1)[2:]}",
                f"{current_year-1}-{str(current_year)[2:]}",
                f"{current_year-2}-{str(current_year-1)[2:]}"
            ]

        print(f"  Seasons: {', '.join(seasons)}")
        games = fetcher.fetch_historical_games(seasons=seasons)
        print(f"  OK - Fetched {len(games)} games")

        # [3/5] Fetch box scores (uses cache!)
        print("\n[3/5] Fetching box scores...")
        print("  NOTE: This uses cache - should be fast if already fetched")
        box_score_fetcher = BoxScoreFetcher()
        game_ids = games['GAME_ID'].unique().tolist()
        box_scores = box_score_fetcher.batch_fetch_box_scores(game_ids)
        print(f"  OK - Got {len(box_scores)} box scores")

        # [4/5] Train model
        print("\n[4/5] Training ROCKSTAR model...")
        print("  This will take 5-15 minutes...")

        model = ROCKSTARPredictor()

        # Calculate Elo ratings
        elo_ratings = calculate_elo_ratings(games)

        # Prepare training data (model will engineer 79 features)
        X, y = model.prepare_training_data(games, box_scores, elo_ratings)

        # Train with 5-fold CV
        model.train(X, y, n_splits=5)

        # [5/5] Save model and metadata
        print("\n[5/5] Saving model...")
        model.save()

        # Save training metadata
        model_info = {
            'trained_at': datetime.now().isoformat(),
            'training_games': len(games),
            'seasons': seasons,
            'incremental': incremental,
            'training_duration_seconds': (datetime.now() - start_time).total_seconds()
        }

        with open(project_root / 'models' / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

        print("\n" + "=" * 60)
        print("RETRAINING COMPLETE!")
        print("=" * 60)
        duration = datetime.now() - start_time
        print(f"\nTotal time: {str(duration).split('.')[0]}")
        print(f"Games used: {len(games)}")
        print(f"Model saved to: models/")

        return True

    except Exception as e:
        print(f"\n\nERROR during retraining: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Smart Model Retraining System')
    parser.add_argument('--check', action='store_true',
                       help='Check if retraining is needed')
    parser.add_argument('--force', action='store_true',
                       help='Force full retraining now')
    parser.add_argument('--auto', action='store_true',
                       help='Auto retrain if needed (based on --check)')
    parser.add_argument('--incremental', action='store_true',
                       help='Incremental retraining (new games only)')

    args = parser.parse_args()

    if args.check or args.auto:
        should_retrain, urgency, reasons = check_retraining_need()

        if args.auto and should_retrain:
            print("\nAuto-retraining triggered...")
            retrain_model(incremental=args.incremental)
        elif not args.auto:
            if should_retrain:
                print("\nTo retrain now, run:")
                print("  python scripts/smart_retrain.py --force")

    elif args.force:
        print("Force retraining requested...")
        retrain_model(incremental=args.incremental)

    else:
        parser.print_help()


if __name__ == "__main__":
    import pandas as pd
    main()

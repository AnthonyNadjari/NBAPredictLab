"""
Automated Retraining Script
Run this script weekly to keep your model up-to-date
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_feedback_system import ModelFeedbackSystem
from src.data_fetcher import NBADataFetcher, EloRatingSystem, FeatureEngineer
from src.models import StackedEnsembleModel
from src.box_score_fetcher import BoxScoreFetcher


def main():
    print("=" * 70)
    print("                  AUTOMATED MODEL RETRAINING")
    print("=" * 70)
    print()

    # Step 1: Update predictions with actual results
    print("[1/6] Fetching actual game results...")
    feedback = ModelFeedbackSystem()
    updated = feedback.update_predictions_with_results(lookback_days=14)
    print(f"  OK - Updated {updated} predictions with actual results")
    print()

    # Step 2: Evaluate current model performance
    print("[2/6] Evaluating model performance...")
    perf = feedback.evaluate_model_performance(period_days=30)
    print(f"  Accuracy: {perf['accuracy']:.1%}")
    print(f"  Brier Score: {perf['brier_score']:.4f}")
    print(f"  Calibration: {perf['calibration_score']:.4f}")
    print()

    # Step 3: Check if retraining is needed
    print("[3/6] Checking retraining recommendations...")
    recs = feedback.get_retraining_recommendations()

    print(f"  Should Retrain: {recs['should_retrain']}")
    print(f"  Urgency: {recs['urgency'].upper()}")

    if recs['reasons']:
        print("  Reasons:")
        for reason in recs['reasons']:
            print(f"    - {reason}")
    print()

    if not recs['should_retrain']:
        print("=" * 70)
        print("  Model performance is good. No retraining needed.")
        print("=" * 70)
        feedback.close()
        return

    # Step 4: Fetch latest data
    print("[4/6] Fetching latest NBA data...")
    fetcher = NBADataFetcher()

    try:
        games_df = fetcher.fetch_historical_games(
            seasons=['2024-25', '2023-24', '2022-23']  # Last 3 seasons
        )
        print(f"  OK - Fetched {len(games_df)} games")
    except Exception as e:
        print(f"  ERROR - {e}")
        feedback.close()
        return
    print()

    # Step 5: Fetch box scores (will use cache)
    print("[5/6] Loading box scores (from cache)...")
    box_score_fetcher = BoxScoreFetcher()

    cache_stats = box_score_fetcher.get_cache_stats()
    print(f"  Cache: {cache_stats['total_cached']} box scores cached")

    game_ids = games_df['game_id'].unique().tolist()
    print(f"  Need: {len(game_ids)} games")

    start_time = time.time()

    def progress_callback(completed, total):
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        print(f"\r  Progress: {completed}/{total} | ETA: {eta/60:.0f} min", end='')

    box_scores = box_score_fetcher.batch_fetch_box_scores(
        game_ids,
        progress_callback=progress_callback
    )

    print(f"\n  OK - Box scores ready")
    print()

    # Step 6: Retrain model
    print("[6/6] Retraining model...")

    # Calculate Elo
    elo = EloRatingSystem()
    elo.calculate_all_historical(games_df)

    # Create features
    engineer = FeatureEngineer()
    X, y = engineer.create_training_dataset(games_df)

    print(f"  Features: {len(X.columns)}")
    print(f"  Samples: {len(X)}")

    # Train model
    model = StackedEnsembleModel()
    results = model.train(X, y, n_splits=5)

    # Save model
    model.save("models")

    # Final summary
    print()
    print("=" * 70)
    print("                  RETRAINING COMPLETE!")
    print("=" * 70)
    print(f"  New Accuracy: {results['mean_cv_accuracy']:.1%} +/- {results['std_cv_accuracy']:.1%}")
    print(f"  Previous Accuracy: {perf['accuracy']:.1%}")

    improvement = results['mean_cv_accuracy'] - perf['accuracy']
    if improvement > 0:
        print(f"  Improvement: +{improvement:.1%}")
    else:
        print(f"  Change: {improvement:.1%}")

    print(f"  Model saved to: models/")
    print("=" * 70)
    print()

    feedback.close()


if __name__ == "__main__":
    main()

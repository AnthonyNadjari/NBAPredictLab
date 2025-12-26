"""
scripts/train_model.py - PROPER Training Pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import NBADataFetcher, EloRatingSystem, FeatureEngineer
from src.models import StackedEnsembleModel


def main():
    print("="*60)
    print("NBA PREDICTOR - TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Fetch historical data
    print("\n[1/4] Fetching historical NBA data...")
    fetcher = NBADataFetcher()
    
    try:
        # Using 4 recent seasons for optimal balance:
        # - Enough data for robust model (~4,900 games)
        # - Recent enough to be relevant (no COVID-era data)
        # - Captures multi-year patterns and trends
        games_df = fetcher.fetch_historical_games(
            seasons=['2024-25', '2023-24', '2022-23', '2021-22']
        )
        print(f"  Fetched {len(games_df)} games")
    except Exception as e:
        print(f"  Error fetching data: {e}")
        print("  Make sure you have internet connection and nba_api is working")
        return
        
    # Step 2: Calculate Elo ratings
    print("\n[2/4] Calculating Elo ratings...")
    elo = EloRatingSystem()
    elo.calculate_all_historical(games_df)
    print("  Elo ratings calculated")
    
    # Step 3: Create features with sample weights
    print("\n[3/4] Engineering features...")
    engineer = FeatureEngineer()
    X, y, sample_weights = engineer.create_training_dataset(games_df)
    print(f"  Created {len(X)} samples with {len(X.columns)} features")

    # Step 4: Train model with recency weighting
    print("\n[4/4] Training stacked ensemble with recency weighting...")
    model = StackedEnsembleModel()
    results = model.train(X, y, sample_weights=sample_weights, n_splits=5)
    
    # Save model
    model.save("models")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Samples: {results['n_samples']}")
    print(f"  Features: {results['n_features']}")
    print(f"  CV Accuracy: {results['mean_cv_accuracy']:.1%} ± {results['std_cv_accuracy']:.1%}")
    print(f"  Model saved to: models/")
    print("="*60)
    


if __name__ == "__main__":
    main()

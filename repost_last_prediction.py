#!/usr/bin/env python3
"""
Repost the last prediction that should have been posted to Twitter
"""

import sys
from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from daily_auto_prediction import DailyPredictionAutomation
from src.twitter_integration import create_fresh_twitter_client, create_twitter_thread

def get_last_prediction_from_logs():
    """Extract the last prediction details from logs"""
    log_file = PROJECT_ROOT / "logs" / "daily_predictions_202512.log"
    
    if not log_file.exists():
        print("❌ Log file not found")
        return None
    
    # Read last 200 lines
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        last_lines = lines[-200:]
    
    # Find the last SELECTED PREDICTION section
    prediction_info = {}
    in_section = False
    
    for line in last_lines:
        if "SELECTED PREDICTION:" in line:
            in_section = True
            continue
        if in_section and "Matchup:" in line:
            matchup = line.split("Matchup:")[1].strip()
            parts = matchup.split(" vs ")
            if len(parts) == 2:
                prediction_info['home_team'] = parts[1].strip()
                prediction_info['away_team'] = parts[0].strip()
        elif in_section and "Pick:" in line:
            prediction_info['predicted_team'] = line.split("Pick:")[1].strip()
        elif in_section and "Confidence:" in line:
            conf_str = line.split("Confidence:")[1].strip().replace('%', '')
            prediction_info['confidence'] = float(conf_str) / 100
        elif in_section and "Win Probability:" in line:
            prob_str = line.split("Win Probability:")[1].strip().replace('%', '')
            prediction_info['win_probability'] = float(prob_str) / 100
        elif in_section and "Odds:" in line:
            odds_str = line.split("Odds:")[1].strip()
            prediction_info['odds'] = float(odds_str)
        elif in_section and "=" * 60 in line:
            break
    
    return prediction_info if prediction_info else None

def get_prediction_from_db(home_team, away_team):
    """Get full prediction data from database"""
    db_path = PROJECT_ROOT / "data" / "nba_predictor.db"
    
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return None
    
    conn = sqlite3.connect(str(db_path))
    
    # Get the most recent prediction for this matchup
    query = """
        SELECT * FROM predictions 
        WHERE home_team = ? AND away_team = ?
        ORDER BY created_at DESC
        LIMIT 1
    """
    
    df = pd.read_sql_query(query, conn, params=(home_team, away_team))
    conn.close()
    
    if df.empty:
        print(f"❌ No prediction found in database for {away_team} @ {home_team}")
        return None
    
    # Convert to dict and parse features_json
    pred = df.iloc[0].to_dict()
    if pred.get('features_json'):
        import json
        pred['features'] = json.loads(pred['features_json'])
    else:
        pred['features'] = {}
    
    # Add missing fields
    pred['prediction'] = 'home' if pred.get('predicted_winner') == home_team else 'away'
    pred['predicted_team_name'] = pred.get('predicted_winner', home_team)
    pred['home_win_probability'] = pred.get('predicted_home_prob', 0.5)
    pred['away_win_probability'] = pred.get('predicted_away_prob', 0.5)
    pred['confidence'] = pred.get('confidence', 0.5)
    pred['win_probability'] = pred['home_win_probability'] if pred['prediction'] == 'home' else pred['away_win_probability']
    pred['odds'] = 1.0 / pred['win_probability'] if pred['win_probability'] > 0 else 99.0
    
    return pred

def main():
    print("=" * 80)
    print("Reposting Last Prediction to Twitter")
    print("=" * 80)
    
    # Get prediction info from logs
    print("\n1. Extracting last prediction from logs...")
    log_info = get_last_prediction_from_logs()
    
    if not log_info:
        print("❌ Could not extract prediction from logs")
        return False
    
    print(f"   ✓ Found: {log_info['away_team']} @ {log_info['home_team']}")
    print(f"   Pick: {log_info['predicted_team']}")
    print(f"   Confidence: {log_info['confidence']:.1%}")
    
    # Get full prediction from database
    print(f"\n2. Loading full prediction data from database...")
    prediction = get_prediction_from_db(log_info['home_team'], log_info['away_team'])
    
    if not prediction:
        print("❌ Could not load prediction from database")
        return False
    
    print(f"   ✓ Loaded prediction data")
    
    # Initialize automation to use its format_twitter_thread method
    print(f"\n3. Initializing automation system...")
    automation = DailyPredictionAutomation(
        db_path=str(PROJECT_ROOT / "data" / "nba_predictor.db"),
        model_dir=str(PROJECT_ROOT / "models"),
        log_dir=str(PROJECT_ROOT / "logs"),
        dry_run=False  # Real posting
    )
    
    # Initialize components
    if not automation.initialize_components():
        print("   ❌ Failed to initialize components")
        return False
    
    print(f"   ✓ Components initialized")
    
    print(f"\n4. Formatting Twitter thread...")
    
    # Format thread
    try:
        tweets, image_paths = automation.format_twitter_thread(prediction)
        print(f"   ✓ Formatted {len(tweets)} tweets with {len([p for p in image_paths if p])} images")
    except Exception as e:
        print(f"   ❌ Error formatting thread: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Show preview
    print(f"\n4. Thread preview:")
    for i, tweet in enumerate(tweets, 1):
        print(f"\n--- Tweet {i} ---")
        print(tweet[:200] + "..." if len(tweet) > 200 else tweet)
    
    # Post directly (no confirmation needed)
    print(f"\n5. Posting to Twitter...")
    
    # Create Twitter client
    try:
        api_clients = create_fresh_twitter_client()
        print("   ✓ Twitter client created")
    except Exception as e:
        print(f"   ❌ Failed to create Twitter client: {e}")
        return False
    
    # Post thread
    try:
        # First tweet has no image, others can have images
        image_paths_with_none = [None] + image_paths if image_paths else [None] * len(tweets)
        
        responses = create_twitter_thread(
            api_clients=api_clients,
            texts=tweets,
            image_paths=image_paths_with_none[:len(tweets)],
            dry_run=False
        )
        
        # Check success
        success = all(r.get('success', False) for r in responses)
        
        if success:
            tweet_ids = [r.get('tweet_id') for r in responses if r.get('tweet_id')]
            print(f"\n✅ Successfully posted thread with {len(tweet_ids)} tweets!")
            print(f"   First tweet ID: {tweet_ids[0] if tweet_ids else 'N/A'}")
            print(f"   View at: https://twitter.com/user/status/{tweet_ids[0] if tweet_ids else ''}")
            return True
        else:
            print(f"\n❌ Some tweets failed to post")
            for i, resp in enumerate(responses, 1):
                if not resp.get('success'):
                    print(f"   Tweet {i} error: {resp.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error posting to Twitter: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


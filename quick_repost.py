#!/usr/bin/env python3
"""Quick script to repost the last prediction thread to Twitter"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from daily_auto_prediction import DailyPredictionAutomation
from src.twitter_integration import create_fresh_twitter_client, create_twitter_thread
import sqlite3
import pandas as pd
import json

# Get last prediction from database (SAC vs HOU from logs)
db_path = PROJECT_ROOT / "data" / "nba_predictor.db"
conn = sqlite3.connect(str(db_path))

query = """
    SELECT * FROM predictions 
    WHERE home_team = 'SAC' AND away_team = 'HOU'
    ORDER BY created_at DESC
    LIMIT 1
"""

df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("❌ No prediction found")
    sys.exit(1)

pred = df.iloc[0].to_dict()
pred['features'] = json.loads(pred['features_json']) if pred.get('features_json') else {}
pred['prediction'] = 'away'  # HOU is away
pred['predicted_team_name'] = 'HOU'
pred['home_win_probability'] = pred.get('predicted_home_prob', 0.0)
pred['away_win_probability'] = pred.get('predicted_away_prob', 1.0)
pred['confidence'] = pred.get('confidence', 0.962)
pred['win_probability'] = 1.0
pred['odds'] = 99.0

# Initialize automation
automation = DailyPredictionAutomation(
    db_path=str(db_path),
    model_dir=str(PROJECT_ROOT / "models"),
    log_dir=str(PROJECT_ROOT / "logs"),
    dry_run=False
)

if not automation.initialize_components():
    print("❌ Failed to initialize")
    sys.exit(1)

# Format thread
tweets, image_paths = automation.format_twitter_thread(pred)
print(f"✓ Formatted {len(tweets)} tweets")

# Post to Twitter
api_clients = create_fresh_twitter_client()
image_paths_with_none = [None] + image_paths if image_paths else [None] * len(tweets)

responses = create_twitter_thread(
    api_clients=api_clients,
    texts=tweets,
    image_paths=image_paths_with_none[:len(tweets)],
    dry_run=False
)

if all(r.get('success', False) for r in responses):
    tweet_ids = [r.get('tweet_id') for r in responses if r.get('tweet_id')]
    print(f"✅ Posted! First tweet: https://twitter.com/user/status/{tweet_ids[0] if tweet_ids else ''}")
else:
    print("❌ Failed to post")
    for i, resp in enumerate(responses, 1):
        if not resp.get('success'):
            print(f"Tweet {i} error: {resp.get('error', 'Unknown')}")


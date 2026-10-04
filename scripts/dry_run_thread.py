#!/usr/bin/env python3
"""End-to-end dry run of the bot on a real slate, without posting.

Predicts a US date with the v2 engine into a temporary copy of the DB,
exports the publishing JSON, then builds the thread + card images with the
same factory publish_single_thread.py uses. Checks every tweet fits 280 chars.

Usage: python scripts/dry_run_thread.py 2026-10-20 [game_index]
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main():
    day = date.fromisoformat(sys.argv[1])
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tmp = Path(tempfile.mkdtemp())
    db = tmp / 'nba_predictor.db'
    shutil.copy2(ROOT / 'data' / 'nba_predictor.db', db)

    from src.engine import pipeline
    preds = pipeline.predict_date(day)
    if not preds:
        print(f'No upcoming games on {day}')
        return 1
    pipeline.save_predictions(str(db), preds)

    from src.daily_games_exporter import DailyGamesExporter
    out = tmp / 'pending_games.json'
    DailyGamesExporter(str(db)).export_today_and_tomorrow(day.isoformat(), day.isoformat(), output_path=str(out))
    game = json.loads(out.read_text(encoding='utf-8'))['games'][idx]

    # Route the publisher's hardcoded DB path to the temp copy
    real_connect = sqlite3.connect
    sqlite3.connect = lambda p, *a, **k: real_connect(str(db) if 'nba_predictor.db' in str(p) else p, *a, **k)
    import scripts.publish_single_thread as pub
    from src.social.render import render_cards
    prediction = pub.get_prediction_from_db(game['home_team'], game['away_team'], game['date'])
    posts = pub.build_posts(game, prediction)
    texts = [p['text'] for p in posts]
    images = render_cards([p.get('card') for p in posts], out_dir=str(tmp / 'cards'))

    ok = True
    for i, t in enumerate(texts):
        img = images[i] if images and i < len(images) else None
        print(f'--- tweet {i + 1}/{len(texts)} ({len(t)} chars){" + image" if img else ""}\n{t}')
        ok &= len(t) <= 280
    made = [p for p in images or [] if p]
    print(f'\n{len(texts)} tweets, {len(made)} images in {Path(made[0]).parent if made else "-"}')
    if len(texts) < 3 or not made or not ok:
        print('FAIL: thread incomplete (fallback format, no images, or tweet over 280)')
        return 1
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())

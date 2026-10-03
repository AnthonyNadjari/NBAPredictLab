#!/usr/bin/env python3
"""
Morning Routine (v2 engine)
===========================
1. Refresh finished games from ESPN into data/games_history.csv
2. Resolve every pending prediction that now has a result
3. Predict today's and tomorrow's games (US Eastern dates)
4. Export docs/pending_games.json for the publishing page
5. Send the email report
6. Commit + push data (CI only, or with --push)

Usage:
    python scripts/morning_routine.py [--skip-email] [--skip-predictions] [--lookback 7] [--push] [--no-push]
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # legacy modules use paths relative to the repo root

(PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / f'morning_routine_{datetime.now().strftime("%Y%m")}.log',
                            encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / 'data' / 'nba_predictor.db'
JSON_PATH = PROJECT_ROOT / 'docs' / 'pending_games.json'
DATA_FILES = ['data/nba_predictor.db', 'data/games_history.csv', 'docs/dashboard.json']


def step(title: str) -> None:
    logger.info('')
    logger.info('=' * 60)
    logger.info(title)
    logger.info('=' * 60)


def export_json(today: str, tomorrow: str) -> bool:
    from src.daily_games_exporter import DailyGamesExporter
    from src.engine.dashboard import enrich_pending
    if not DailyGamesExporter(str(DB_PATH)).export_today_and_tomorrow(today, tomorrow, output_path=str(JSON_PATH)):
        return False
    try:
        enrich_pending(JSON_PATH, str(DB_PATH))
    except Exception as e:
        logger.warning(f'[WARN] Could not enrich pending games: {e}')
    return True


def to_email_format(predictions: list, target_date: str) -> list:
    from src.engine.teams import full_name
    out = []
    for p in predictions:
        if p['game_info']['game_date'] != target_date:
            continue
        out.append({
            'game_date': target_date,
            'home_team': full_name(p['home_team']),
            'away_team': full_name(p['away_team']),
            'predicted_winner': full_name(p['predicted_winner']),
            'predicted_home_prob': p['home_win_probability'],
            'predicted_away_prob': p['away_win_probability'],
            'home_odds': float(p['home_odds'] or round(1 / max(p['home_win_probability'], 0.01), 2)),
            'away_odds': float(p['away_odds'] or round(1 / max(p['away_win_probability'], 0.01), 2)),
            'confidence': p['confidence'],
        })
    return out


def git(*args, check=False):
    r = subprocess.run(['git', *args], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def push_data(today: str, tomorrow: str, message: str) -> bool:
    """Commit data + JSON and push.

    The DB and history are only written by this job; pending_games.json is
    also written by the publish workflow. On a rejected push we reset to the
    remote, restore our data files and re-export the JSON, which carries the
    remote's 'published' flags over, so nothing is lost on either side.
    """
    if os.environ.get('GITHUB_ACTIONS'):
        git('config', 'user.name', 'GitHub Actions Bot')
        git('config', 'user.email', 'actions@github.com')
    for attempt in range(1, 4):
        git('add', 'docs/pending_games.json', *DATA_FILES)
        if git('diff', '--cached', '--quiet').returncode == 0:
            logger.info('[OK] Nothing to commit')
            return True
        git('commit', '-m', message, check=True)
        if git('push', 'origin', 'HEAD:main').returncode == 0:
            logger.info('[OK] Pushed data to GitHub')
            return True
        logger.info(f'[INFO] Push rejected (attempt {attempt}), re-applying on top of remote')
        backup = Path(tempfile.mkdtemp())
        for f in DATA_FILES:
            shutil.copy2(PROJECT_ROOT / f, backup / Path(f).name)
        git('fetch', 'origin', 'main', check=True)
        git('reset', '--hard', 'origin/main', check=True)
        for f in DATA_FILES:
            shutil.copy2(backup / Path(f).name, PROJECT_ROOT / f)
        export_json(today, tomorrow)
    logger.error('[ERROR] Could not push data after 3 attempts')
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description='NBA Predictor - Morning Routine')
    parser.add_argument('--skip-email', action='store_true')
    parser.add_argument('--skip-predictions', action='store_true')
    parser.add_argument('--lookback', type=int, default=7, help='days of results to refresh')
    parser.add_argument('--push', action='store_true', help='commit + push data (default in CI)')
    parser.add_argument('--no-push', action='store_true')
    args = parser.parse_args()

    from src.engine import pipeline
    from src.engine.history import espn_today

    today_d = espn_today()
    today, tomorrow = today_d.isoformat(), (today_d + timedelta(days=1)).isoformat()
    logger.info('=' * 60)
    logger.info(f'NBA Predictor - Morning Routine (US date {today})')
    logger.info('=' * 60)
    problems = []

    step('STEP 1: Refreshing results from ESPN')
    try:
        hist = pipeline.refresh_history(days_back=args.lookback)
        logger.info(f'[OK] History: {len(hist)} games, last {hist.game_date.max()}')
    except Exception as e:
        logger.error(f'[ERROR] History refresh failed: {e}', exc_info=True)
        problems.append('history')
        from src.engine import history
        hist = history.load()

    step('STEP 2: Resolving pending predictions')
    try:
        n = pipeline.resolve_predictions(str(DB_PATH), hist)
        logger.info(f'[OK] Resolved {n} prediction(s)')
        rec = pipeline.track_record(str(DB_PATH), since='2026-10-01')
        if rec['n']:
            logger.info(f"Season record: {rec['correct']}/{rec['n']} ({rec['accuracy']:.1%})")
    except Exception as e:
        logger.error(f'[ERROR] Resolving failed: {e}', exc_info=True)
        problems.append('resolve')

    predictions = []
    if not args.skip_predictions:
        step(f'STEP 3: Predicting {today} and {tomorrow}')
        for d in (today_d, today_d + timedelta(days=1)):
            try:
                preds = pipeline.predict_date(d, hist)
                predictions += preds
                logger.info(f'{d}: {len(preds)} game(s)')
            except Exception as e:
                logger.error(f'[ERROR] Prediction for {d} failed: {e}', exc_info=True)
                problems.append(f'predict {d}')
        for p in predictions:
            f = p['features']
            logger.info(f"  {p['away_team']} @ {p['home_team']} ({p['game_info']['game_date']}): "
                        f"{p['predicted_winner']} {p['confidence']:.1%} [{f['probability_source']}] "
                        f"model={f['model_home_prob']:.3f} market={f['market_home_prob']}")
        if predictions:
            saved = pipeline.save_predictions(str(DB_PATH), predictions)
            logger.info(f'[OK] Saved {saved} prediction(s)')

    step('STEP 4: Exporting publishing JSON + dashboard')
    if not export_json(today, tomorrow):
        problems.append('export')
    try:
        from src.engine.dashboard import write_dashboard
        write_dashboard(str(DB_PATH), today_d)
        logger.info('[OK] Dashboard data written')
    except Exception as e:
        logger.error(f'[ERROR] Dashboard failed: {e}', exc_info=True)
        problems.append('dashboard')

    if not args.skip_email:
        step('STEP 5: Email report')
        try:
            from src.email_reporter import EmailReporter
            ok = EmailReporter(db_path=str(DB_PATH)).send_daily_report(
                test_mode=False,
                today_predictions_override=to_email_format(predictions, today) if predictions else None,
                tomorrow_predictions_override=to_email_format(predictions, tomorrow) if predictions else None,
            )
            logger.info('[OK] Email step done' if ok else '[WARN] Email not sent')
        except Exception as e:
            logger.error(f'[ERROR] Email failed: {e}', exc_info=True)

    if (os.environ.get('GITHUB_ACTIONS') or args.push) and not args.no_push:
        step('STEP 6: Pushing data')
        if not push_data(today, tomorrow, f'Auto-export predictions for {today}'):
            problems.append('push')

    logger.info('')
    if problems:
        logger.error(f"[FAIL] Morning routine finished with problems: {', '.join(problems)}")
        return 1
    logger.info('[OK] Morning routine completed')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Autopilot: queue tonight's best threads (and the weekly recap) for the server, with no click.

Runs in the daily workflow right after the 21:00 UTC refresh (research/h6_timing: the
evening price captures ~80% of the accuracy gain over the morning one). It only adds
requests to docs/vision/publish_queue.json, exactly like the panel's "Publier" button:
the server then posts them through the browser, with the same checks (request guard,
tip-off expiry, never twice). Settings in docs/autopilot.json, edited from the panel:

    {"enabled": false, "threads_per_day": 2, "min_odds": 1.25, "weekly_recap": true}

Picks: games not yet published, starting >= 45 min from now, with real market odds,
ranked by win probability among picks priced at >= min_odds (very short favourites
make dull posts). Off until switched on in the panel.
"""
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(os.getenv('AUTOPILOT_ROOT') or Path(__file__).resolve().parents[1])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger('autopilot')

CONFIG = ROOT / 'docs' / 'autopilot.json'
PENDING = ROOT / 'docs' / 'pending_games.json'
QUEUE = ROOT / 'docs' / 'vision' / 'publish_queue.json'
MIN_LEAD = timedelta(minutes=45)
MAX_THREADS = 5
DEFAULTS = {"enabled": False, "threads_per_day": 2, "min_odds": 1.25, "weekly_recap": True}


def load_config() -> dict:
    try:
        return {**DEFAULTS, **json.loads(CONFIG.read_text(encoding='utf-8'))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def _start(g: dict):
    try:
        t = datetime.fromisoformat(str(g.get('start_utc')).replace('Z', '+00:00'))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def choose(data: dict, cfg: dict, now: datetime) -> list:
    """Tonight's candidates, best first."""
    seen, candidates = set(), []
    for g in data.get('games', []):
        if g['id'] in seen or g.get('published'):
            continue
        seen.add(g['id'])
        start = _start(g)
        if g.get('probability_source') not in ('market', 'blend') or start is None or start - now < MIN_LEAD:
            continue
        if start - now > timedelta(hours=12):        # tonight only, not tomorrow's slate
            continue
        pick_home = g['predicted_home_prob'] >= 0.5
        odds = g.get('home_odds') if pick_home else g.get('away_odds')
        if not odds or float(odds) < float(cfg['min_odds']):
            continue
        candidates.append((max(g['predicted_home_prob'], g['predicted_away_prob']), g))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [g for _, g in candidates]


def plan(data: dict, cfg: dict, now: datetime, queued: set) -> list:
    """Game/special ids to queue tonight (never one already queued)."""
    ids = []
    if cfg.get('weekly_recap'):
        ids += [sp['id'] for sp in data.get('specials', [])
                if sp.get('type') == 'weekly' and not sp.get('published') and sp.get('thread')]
    n = max(0, min(int(cfg.get('threads_per_day') or 0), MAX_THREADS))
    ids += [g['id'] for g in choose(data, cfg, now)[:n]]
    return [i for i in ids if i not in queued]


def add_to_queue(ids: list, now: datetime) -> int:
    q = json.loads(QUEUE.read_text(encoding='utf-8')) if QUEUE.exists() else {'requests': []}
    have = {r.get('id') for r in q['requests'] if isinstance(r, dict)}
    added = 0
    for gid in ids:
        rid = f"autopilot-{now:%Y%m%d}-{gid}"
        if rid in have:
            continue
        q['requests'].append({'id': rid, 'game_id': gid, 'texts': None, 'dry': False,
                              'requested_at': now.isoformat(timespec='seconds'), 'source': 'autopilot'})
        added += 1
    q['requests'] = q['requests'][-20:]
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding='utf-8')
    return added


def already_queued(now: datetime) -> set:
    try:
        q = json.loads(QUEUE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return set()
    return {r.get('game_id') for r in q.get('requests', []) if isinstance(r, dict)}


def _git(*args) -> int:
    return subprocess.run(['git', *args], cwd=ROOT).returncode


def main() -> int:
    for attempt in range(3):
        cfg = load_config()
        if not cfg['enabled']:
            log.info('Autopilot is off (switch it on in the control panel)')
            return 0
        data = json.loads(PENDING.read_text(encoding='utf-8')) if PENDING.exists() else {}
        now = datetime.now(timezone.utc)
        ids = plan(data, cfg, now, already_queued(now))
        if not ids:
            log.info('Nothing to queue tonight')
            return 0
        added = add_to_queue(ids, now)
        log.info(f'Queued {added} post(s) for the server: {", ".join(ids)}')
        if os.getenv('AUTOPILOT_NO_PUSH') or not added:
            return 0
        _git('add', 'docs/vision/publish_queue.json')
        _git('-c', 'user.name=github-actions[bot]', '-c', 'user.email=github-actions[bot]@users.noreply.github.com',
             'commit', '-q', '-m', f'Autopilot: queue {len(ids)} post(s)')
        if _git('push', '-q', 'origin', 'HEAD:main') == 0:
            return 0
        # someone pushed meanwhile: start again from the remote state
        _git('fetch', '-q', 'origin', 'main')
        _git('reset', '-q', '--hard', 'origin/main')
    log.error('Could not push the autopilot queue')
    return 1


if __name__ == '__main__':
    sys.exit(main())

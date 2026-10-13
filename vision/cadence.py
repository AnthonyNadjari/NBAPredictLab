#!/usr/bin/env python3
"""
Automatic cadence of the reply bot, around the clock, shaped like one person's day (no fixed slots).

What it decides, from docs/vision/runs.json (when each session ended and how many replies it posted):
  * a quiet stretch each day while the US sleeps (08:30-13:30 Paris, edges moved by up to 40 min
    a day): measured on 450 replies, 09:00-15:00 Paris gets the fewest views, and an account that
    replies at every hour of the day is what bot-spotters look for;
  * a daily total that changes from day to day (15-20 replies; 60-80 got every reply hidden by X
    from 12 Oct, 2 views under 100k-view tweets), spread over the waking hours with
    twice the weight on US prime time (21:00-06:00 Paris, the most views and likes per view), so the
    budget is not spent by the afternoon; never more than 4 in an hour;
  * a random pause between sessions (shorter during US prime time, 21:00-06:00 Paris) and a random
    session size (2-4 replies), so the posting pattern never repeats.
Random draws are seeded by the date / the previous session, so every 5-min tick sees the same plan.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

DAILY = (15, 20)            # replies per 24 h
HOURLY_MAX = 4
QUIET = ((8, 30), (13, 30))   # Paris time, US night
QUIET_JITTER_MIN = 40
SESSION_SIZE = (2, 4)
GAP_PEAK = (30, 60)           # minutes between sessions, 21:00-06:00 Paris
GAP_DAY = (60, 120)
PEAK_HOURS = set(range(21, 24)) | set(range(0, 6))


def _t(s) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def quiet_window(day, tz) -> tuple[datetime, datetime]:
    rng = random.Random(f"quiet-{day.isoformat()}")
    (h0, m0), (h1, m1) = QUIET
    start = datetime(day.year, day.month, day.day, h0, m0, tzinfo=tz) + timedelta(minutes=rng.randint(-QUIET_JITTER_MIN, QUIET_JITTER_MIN))
    end = datetime(day.year, day.month, day.day, h1, m1, tzinfo=tz) + timedelta(minutes=rng.randint(-QUIET_JITTER_MIN, QUIET_JITTER_MIN))
    return start, end


def daily_target(day) -> int:
    return random.Random(f"target-{day.isoformat()}").randint(*DAILY)


def _weight_minutes(a: datetime, b: datetime) -> float:
    """Minutes between a and b, US prime-time minutes counting double (5-min steps)."""
    w, t = 0.0, a
    while t < b:
        w += 2.0 if t.hour in PEAK_HOURS else 1.0
        t += timedelta(minutes=5)
    return w


def decide(now: datetime, runs: list[dict]) -> dict:
    """{"due": bool, "why": str, "max": int, "next_after": iso or None, "done": int, "target": int}"""
    real = [r for r in runs if not r.get("dry_run") and r.get("auth") == "ok" and _t(r.get("end") or r.get("at"))]
    ends = sorted(((_t(r.get("end") or r.get("at")), int(r.get("replied") or 0), str(r.get("run_id"))) for r in real),
                  key=lambda x: x[0])
    q0, q1 = quiet_window(now.date(), now.tzinfo)
    # the bot's "day" runs from the end of one quiet stretch to the start of the next
    day = now.date() if now >= q1 else now.date() - timedelta(days=1)
    start = quiet_window(day, now.tzinfo)[1]
    stop = quiet_window(day + timedelta(days=1), now.tzinfo)[0]
    target = daily_target(day)
    done = sum(n for t, n, _ in ends if t >= start)
    last_1h = sum(n for t, n, _ in ends if now - t < timedelta(hours=1))
    out = {"due": False, "max": 0, "next_after": None, "done": done, "target": target}
    if q0 <= now < q1:
        return out | {"why": "quiet hours", "next_after": q1.isoformat(timespec="minutes")}
    if done >= target:
        return out | {"why": f"daily total reached ({done}/{target})"}
    allowed = target * _weight_minutes(start, now) / max(1.0, _weight_minutes(start, stop)) + SESSION_SIZE[1]
    if done >= allowed:
        return out | {"why": f"ahead of the day's pace ({done}/{allowed:.0f})"}
    if last_1h >= HOURLY_MAX:
        return out | {"why": f"hourly cap ({last_1h})"}
    if ends:
        t_end, _, rid = ends[-1]
        gap = random.Random(f"gap-{rid}").randint(*(GAP_PEAK if t_end.hour in PEAK_HOURS else GAP_DAY))
        nxt = t_end + timedelta(minutes=gap)
        if now < nxt:
            return out | {"why": f"pause after last session ({gap} min)", "next_after": nxt.isoformat(timespec="minutes")}
    size = random.Random(f"size-{ends[-1][2] if ends else now.date()}").randint(*SESSION_SIZE)
    size = max(1, min(size, target - done, HOURLY_MAX - last_1h))
    return out | {"due": True, "why": "go", "max": size}


def load_runs(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

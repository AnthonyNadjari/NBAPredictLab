#!/usr/bin/env python3
"""Decide whether a scheduled reply session is due now (Paris time).

The vision workflow ticks every 15 minutes; this prints due=true|false and the
slot to $GITHUB_OUTPUT. Slots come from docs/vision/schedule.json (editable from
the control panel). A slot is due for 45 minutes after its time (GitHub's cron
often runs late) and only once: slots already in docs/vision/runs.json are done.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RUNS_FILE, SCHEDULE_FILE, TZ  # noqa: E402

WINDOW = timedelta(minutes=45)


def due_slot(now: datetime, schedule: dict) -> str | None:
    for s in schedule.get("schedules", []):
        if not s.get("enabled"):
            continue
        h, m = map(int, s["time"].split(":"))
        for day in (now.date(), now.date() - timedelta(days=1)):
            slot = datetime(day.year, day.month, day.day, h, m, tzinfo=TZ)
            if timedelta(0) <= now - slot < WINDOW:
                return slot.isoformat()
    return None


def done_slots() -> set:
    try:
        return {r.get("slot") for r in json.loads(RUNS_FILE.read_text(encoding="utf-8")) if r.get("slot")}
    except Exception:
        return set()


def main() -> int:
    schedule = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    now = datetime.now(TZ)
    slot = due_slot(now, schedule)
    due = bool(slot) and schedule.get("enabled", True) and slot not in done_slots()
    print(f"now={now.isoformat(timespec='minutes')} slot={slot} enabled={schedule.get('enabled', True)} due={due}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"due={'true' if due else 'false'}\nslot={slot or ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Decide whether a scheduled reply session is due now (Paris time).

The vision workflow ticks every 15 minutes on the self-hosted runner; this
prints due=true|false to $GITHUB_OUTPUT. Slots come from docs/vision/schedule.json
(editable from the control panel). A slot fires once, in the 15 minutes after it.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LOCAL_STATE_DIR, SCHEDULE_FILE, TZ  # noqa: E402

WINDOW = timedelta(minutes=15)


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


def main() -> int:
    schedule = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    now = datetime.now(TZ)
    slot = due_slot(now, schedule)
    last_file = LOCAL_STATE_DIR / "last_slot.txt"
    last = last_file.read_text().strip() if last_file.exists() else ""
    due = bool(slot) and slot != last and schedule.get("enabled", True)
    if due:
        last_file.write_text(slot)
    print(f"now={now.isoformat(timespec='minutes')} slot={slot} last={last or '-'} due={due}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"due={'true' if due else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

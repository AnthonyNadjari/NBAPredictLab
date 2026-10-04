"""Compact run history for the control panel (docs/vision/runs.json)."""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime

from config import RUNS_FILE, TZ, DRY_RUN

KEEP_RUNS = 300
KEEP_REPLIES_PER_RUN = 25


def _load() -> list:
    try:
        data = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def record_run(entry: dict) -> None:
    entry = {"at": datetime.now(TZ).isoformat(timespec="seconds"), "dry_run": DRY_RUN, **entry}
    runs = _load() + [entry]
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs[-KEEP_RUNS:], indent=1, ensure_ascii=False), encoding="utf-8")


def summarize_session(log: dict, auth: str = "ok") -> dict:
    skips = Counter(log.get("skip_reasons") or {})
    return {
        "run_id": log.get("run_id"),
        "auth": auth,
        "start": log.get("start_time"),
        "end": log.get("end_time"),
        "scraped": log.get("total_scraped", 0),
        "candidates": log.get("total_filtered", 0),
        "llm_calls": log.get("total_llm_calls", 0),
        "replied": log.get("total_replied", 0),
        "skipped": log.get("total_skipped", 0),
        "top_skips": dict(skips.most_common(6)),
        "replies": (log.get("replies_posted") or [])[-KEEP_REPLIES_PER_RUN:],
    }

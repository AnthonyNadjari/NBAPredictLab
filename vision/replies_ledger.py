"""
Ledger of every reply the bot posts, and how it did.

One JSON file per month in the server's state dir (the source of truth: survives git
resets), copied to docs/vision/replies/ for the control panel and the periodic review.

Each entry: the tweet we answered, our reply, why the model replied, then metric
snapshots taken by vision/tracker.py (h24 / h72 / last), who answered us, whether the
author or a replier followed us, and an automatic grade. Grades given by hand in the
panel live separately in docs/vision/reply_grades.json (never overwritten by the server).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import LOCAL_STATE_DIR

LEDGER_DIR = LOCAL_STATE_DIR / "replies"
TRACK_DAYS = 7                       # metrics stop being refreshed after a week
GRADES = ("bad", "normal", "good", "good_follow", "wrong")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str | None) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _month_file(ts: datetime) -> Path:
    return LEDGER_DIR / f"{ts:%Y-%m}.json"


def _read(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def add(reply: dict, run_id: str | None = None) -> dict:
    """Record a reply that was just posted (engine's replies_posted entry + reply id)."""
    posted = _parse(reply.get("posted_at")) or _now()
    tweet_id = (reply.get("tweet_url") or "").rstrip("/").split("/")[-1]
    entry = {
        "id": reply.get("reply_id") or f"pending-{tweet_id}",
        "reply_id": reply.get("reply_id"),
        "reply_url": f"https://x.com/NBAPredictLab/status/{reply['reply_id']}" if reply.get("reply_id") else None,
        "posted_at": posted.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "tweet_url": reply.get("tweet_url"),
        "tweet_id": tweet_id,
        "author": reply.get("author"),
        "tweet_text": reply.get("tweet_text"),
        "tweet_likes": reply.get("tweet_likes"),
        "reply_text": reply.get("reply_text"),
        "reason": reply.get("reason"),
        "fact_used": reply.get("fact_used") or None,
        "metrics": {},
        "repliers": [],
        "author_replied": None,
        "follow": None,
        "auto_grade": None,
    }
    path = _month_file(posted)
    entries = [e for e in _read(path) if e.get("id") != entry["id"]]
    _write(path, entries + [entry])
    return entry


def recent(days: int = TRACK_DAYS, now: datetime | None = None) -> dict[Path, list[dict]]:
    """{month file: entries} for the files that may hold replies of the last `days` days."""
    now = now or _now()
    months = {_month_file(now - timedelta(days=d)) for d in range(days + 1)}
    return {p: _read(p) for p in sorted(months) if p.exists()}


def save(files: dict[Path, list[dict]]) -> None:
    for path, entries in files.items():
        _write(path, entries)


def backfill_from_runs(runs_file: Path, days: int = TRACK_DAYS, now: datetime | None = None) -> int:
    """Add replies listed in runs.json (posted before the ledger existed, or missed) to the ledger."""
    now = now or _now()
    try:
        runs = json.loads(runs_file.read_text(encoding="utf-8"))
    except Exception:
        return 0
    known = {e.get("tweet_url") for es in recent(days, now).values() for e in es}
    added = 0
    for run in runs if isinstance(runs, list) else []:
        for r in run.get("replies") or []:
            posted = _parse(r.get("posted_at"))
            if (r.get("dry_run") or not r.get("tweet_url") or r["tweet_url"] in known
                    or not posted or now - posted > timedelta(days=days)):
                continue
            add(r, run_id=run.get("run_id"))
            known.add(r["tweet_url"])
            added += 1
    return added


# ------------------------------------------------------------------ metrics and grades
_METRIC_RE = re.compile(r"([\d.,]+)\s*([KkMm]?)\s+(repl(?:y|ies)|reposts?|likes?|bookmarks?|views?)", re.I)


def parse_metrics(label: str) -> dict:
    """'3 replies, 1 repost, 12 likes, 2 bookmarks, 1234 views' -> {'replies': 3, ...}"""
    out = {"replies": 0, "reposts": 0, "likes": 0, "bookmarks": 0, "views": 0}
    for num, mult, kind in _METRIC_RE.findall(label or ""):
        n = float(num.replace(",", ""))
        n *= {"k": 1e3, "m": 1e6}.get(mult.lower(), 1)
        key = kind.lower()
        key = "replies" if key.startswith("repl") else key if key.endswith("s") else key + "s"
        out[key] = int(n)
    return out


def snapshot(entry: dict, metrics: dict, now: datetime | None = None) -> None:
    """Store `metrics` as the latest reading, and as the h24 / h72 checkpoint when due."""
    now = now or _now()
    age = now - (_parse(entry.get("posted_at")) or now)
    m = {**metrics, "at": now.isoformat(timespec="seconds")}
    snaps = entry.setdefault("metrics", {})
    snaps["last"] = m
    if age >= timedelta(hours=24) and "h24" not in snaps:
        snaps["h24"] = m
    if age >= timedelta(hours=72) and "h72" not in snaps:
        snaps["h72"] = m


def grade(entry: dict) -> str | None:
    """Automatic grade once the reply is 24 h old (None before)."""
    m = (entry.get("metrics") or {}).get("h24")
    if not m:
        return None
    last = entry["metrics"].get("last") or m
    likes, replies, views = last.get("likes", 0), last.get("replies", 0), last.get("views", 0)
    if entry.get("follow"):
        return "good_follow"
    if entry.get("author_replied") or likes >= 5 or (replies >= 1 and likes >= 2):
        return "good"
    if likes >= 1 or replies >= 1 or views >= 100:
        return "normal"
    return "bad"


def attribute_follows(entries: list[dict], new_followers: set[str], now: datetime | None = None) -> int:
    """Mark replies whose author (or someone who answered us) has just followed us."""
    now = now or _now()
    hits = 0
    low = {h.lower().lstrip("@") for h in new_followers}
    for e in entries:
        posted = _parse(e.get("posted_at"))
        if not posted or now - posted > timedelta(days=TRACK_DAYS) or e.get("follow"):
            continue
        who = [(e.get("author") or "").lower()] + [r.lower() for r in e.get("repliers") or []]
        match = next((h for h in who if h and h in low), None)
        if match:
            e["follow"] = {"by": match, "seen_at": now.isoformat(timespec="seconds")}
            hits += 1
    return hits


def export(dest: Path, months: int = 3) -> list[Path]:
    """Copy the last `months` month files to the repo (docs/vision/replies) for the panel."""
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(LEDGER_DIR.glob("????-??.json"))[-months:]
    out = []
    for f in files:
        target = dest / f.name
        target.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
        out.append(target)
    (dest / "index.json").write_text(json.dumps([f.name for f in files]), encoding="utf-8")
    return out

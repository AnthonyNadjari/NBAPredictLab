#!/usr/bin/env python3
"""Compact digest of the posted replies for the periodic review (keeps the reviewer's reading small).

    python vision/tools/review_digest.py [--days N]

Reads docs/vision/replies/*.json and docs/vision/reply_grades.json; prints the key numbers,
engagement per 1,000 views broken down by reply type / fact / length / account size / hour,
then one line per reply in the window. Window: since the previous review
(docs/vision/reviews/latest.json), at least 48 h, at most N days (default 7).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2] / "docs" / "vision"
PARIS = ZoneInfo("Europe/Paris")


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _ts(s):
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _rate(num, den):
    return f"{1000 * num / den:.1f}" if den else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    last = _ts((_load(ROOT / "reviews" / "latest.json", {}) or {}).get("date"))
    start = max(now - timedelta(days=args.days), min(last or now, now - timedelta(hours=48)))

    grades = _load(ROOT / "reply_grades.json", {}) or {}
    rows = []
    for name in _load(ROOT / "replies" / "index.json", []):
        rows += _load(ROOT / "replies" / name, [])
    rows = [r for r in rows if (_ts(r.get("posted_at")) or now) >= start]
    rows.sort(key=lambda r: r.get("posted_at") or "")
    for r in rows:
        r["_grade"] = (grades.get(r.get("id")) or {}).get("grade") or r.get("auto_grade")
        r["_by_owner"] = r.get("id") in grades
        r["_m"] = (r.get("metrics") or {}).get("last") or {}

    def summary(group):
        m = [r["_m"] for r in group if r["_m"]]
        views = sum(x.get("views", 0) for x in m)
        likes = sum(x.get("likes", 0) for x in m)
        replies = sum(x.get("replies", 0) for x in m)
        good = sum(1 for r in group if r["_grade"] in ("good", "good_follow"))
        graded = sum(1 for r in group if r["_grade"])
        return (f"n={len(group)} measured={len(m)} views={views} likes/1k={_rate(likes, views)} "
                f"replies/1k={_rate(replies, views)} good={round(100 * good / graded) if graded else '-'}%")

    print(f"# Reply digest {start:%Y-%m-%d %H:%M} -> {now:%Y-%m-%d %H:%M} UTC\n")
    split = defaultdict(int)
    for r in rows:
        split[r["_grade"] or "ungraded"] += 1
    print(f"ALL: {summary(rows)}")
    print(f"grades: {dict(split)} | graded by owner: {sum(r['_by_owner'] for r in rows)} | "
          f"follows won: {sum(1 for r in rows if r.get('follow'))} | removed: {sum(1 for r in rows if r.get('removed'))}\n")

    keys = {
        "reason": lambda r: r.get("reason") or "unknown",
        "fact": lambda r: "fact" if r.get("fact_used") else "no fact",
        "length": lambda r: "<70" if len(r.get("reply_text") or "") < 70 else "70-120" if len(r.get("reply_text") or "") < 120 else "120+",
        "account": lambda r: "<50" if (r.get("tweet_likes") or 0) < 50 else "50-500" if (r.get("tweet_likes") or 0) < 500 else "500+",
        "hour_paris": lambda r: ["night 0-6", "morning 6-12", "afternoon 12-18", "evening 18-24"][
            _ts(r["posted_at"]).astimezone(PARIS).hour // 6],
    }
    for name, key in keys.items():
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        print(f"## by {name}")
        for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"- {k}: {summary(g)}")
        print()

    print("## replies (grade [owner*] | likes/views | replies | reason | fact | @author (tweet likes) | tweet | OUR REPLY | answers)")
    for r in rows:
        m = r["_m"]
        said = " / ".join(f"@{s.get('user')}: {(s.get('text') or '')[:80]}" for s in (r.get("reply_samples") or [])[:2])
        print(" | ".join([
            f"{r.get('id')}", f"{r['_grade'] or '-'}{'*' if r['_by_owner'] else ''}",
            f"{m.get('likes', '-')}/{m.get('views', '-')}", f"{m.get('replies', '-')}",
            r.get("reason") or "-", "fact" if r.get("fact_used") else "-",
            f"@{r.get('author')} ({r.get('tweet_likes', 0)})",
            (r.get("tweet_text") or "").replace("\n", " ")[:110],
            f"REPLY: {r.get('reply_text')}", said or "-",
        ]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Measure how the posted replies did, from the logged-in browser (run by the server tick
every few hours, never during a reply session).

1. our "Replies" timeline: likes / replies / reposts / views of every recent reply in one
   scroll (cheap: one page instead of one per reply)
2. the replies that got answers: who answered (did the original author come back?)
3. our newest followers: a reply whose author or replier just followed us = a follow it won
Then the automatic grade, all written to the ledger (vision/replies_ledger.py).
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import replies_ledger as L
from config import LOCAL_STATE_DIR

HANDLE = "NBAPredictLab"
FOLLOWERS_SEEN = LOCAL_STATE_DIR / "followers_seen.json"
MAX_STATUS_VISITS = 25
MAX_CONVERSATIONS = 20

_ARTICLES_JS = """() => [...document.querySelectorAll('article')].map(a => {
  const t = a.querySelector('a[href*="/status/"] time');
  const link = t ? t.closest('a').getAttribute('href') : null;
  const grp = a.querySelector('[role="group"][aria-label]');
  const user = a.querySelector('[data-testid="User-Name"] a[href^="/"]');
  const text = a.querySelector('[data-testid="tweetText"]');
  return {link, at: t ? t.getAttribute('datetime') : null, metrics: grp ? grp.getAttribute('aria-label') : '',
          user: user ? user.getAttribute('href').slice(1) : null, text: text ? text.innerText.slice(0, 280) : ''};
})"""


def _sid(link: str | None) -> str | None:
    m = re.search(r"/status/(\d+)", link or "")
    return m.group(1) if m else None


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()[:80]


def _pause(a=1.5, b=3.0):
    time.sleep(random.uniform(a, b))


def _open(page, url: str, selector: str = "article") -> bool:
    """Go to `url` and wait until X has rendered `selector` (the SPA can take 10-20 s)."""
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    try:
        page.wait_for_selector(selector, timeout=30000)
        _pause(1.5, 2.5)
        return True
    except Exception:
        return False


def timeline(page, oldest: datetime, max_scrolls: int = 80) -> dict[str, dict]:
    """{reply id: {"metrics", "text", "at"}} for our replies newer than `oldest`."""
    if not _open(page, f"https://x.com/{HANDLE}/with_replies"):
        print("Tracker: our Replies timeline did not load", flush=True)
        return {}
    found: dict[str, dict] = {}
    stale = 0
    for _ in range(max_scrolls):
        before = len(found)
        reached_old = False
        for a in page.evaluate(_ARTICLES_JS):
            sid = _sid(a.get("link"))
            if not sid or not (a.get("link") or "").lower().startswith(f"/{HANDLE.lower()}/status/"):
                continue
            found[sid] = {"metrics": L.parse_metrics(a.get("metrics")), "text": a.get("text"), "at": a.get("at")}
            at = L._parse(a.get("at"))
            if at and at < oldest:
                reached_old = True
        stale = stale + 1 if len(found) == before else 0
        if reached_old or stale >= 5:
            break
        page.mouse.wheel(0, random.randint(1800, 2600))
        _pause()
    return found


def conversation(page, reply_id: str) -> dict:
    """Metrics of one reply and the accounts that answered it (status page)."""
    _open(page, f"https://x.com/{HANDLE}/status/{reply_id}")
    arts = page.evaluate(_ARTICLES_JS)
    idx = next((i for i, a in enumerate(arts) if _sid(a.get("link")) == reply_id), None)
    if idx is None:
        body = page.inner_text("body")[:2000].lower()
        gone = "doesn't exist" in body or "unavailable" in body or "deleted" in body
        return {"missing": True, "removed": gone}
    below = [a for a in arts[idx + 1:] if a.get("user") and a["user"].lower() != HANDLE.lower()]
    return {"metrics": L.parse_metrics(arts[idx].get("metrics")),
            "repliers": list(dict.fromkeys(a["user"] for a in below))[:10],
            "samples": [{"user": a["user"], "text": a.get("text", "")[:200]} for a in below[:3]]}


def newest_followers(page, screens: int = 6) -> list[str]:
    if not _open(page, f"https://x.com/{HANDLE}/followers", '[data-testid="UserCell"]'):
        raise RuntimeError("followers page did not load")
    handles: list[str] = []
    for _ in range(screens):
        for h in page.evaluate("""() => [...document.querySelectorAll('[data-testid="UserCell"]')].map(c => {
              const a = c.querySelector('a[href^="/"]'); return a ? a.getAttribute('href').slice(1) : null; })"""):
            if h and h not in handles:
                handles.append(h)
        page.mouse.wheel(0, 2200)
        _pause()
    return handles


def track(page, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    files = L.recent(now=now)
    entries = [e for es in files.values() for e in es]
    window = [e for e in entries if (L._parse(e.get("posted_at")) or now) > now - timedelta(days=L.TRACK_DAYS)]
    stats = {"tracked": len(window), "timeline": 0, "status_visits": 0, "conversations": 0, "new_followers": 0, "follows": 0}

    if window:
        oldest = min(L._parse(e["posted_at"]) for e in window) - timedelta(hours=1)
        seen = timeline(page, oldest)
        by_text = {_norm(v["text"]): k for k, v in seen.items()}
        for e in window:
            if not e.get("reply_id"):            # id not captured at posting time: match by text
                rid = by_text.get(_norm(e.get("reply_text")))
                if rid:
                    e["reply_id"], e["id"] = rid, rid
                    e["reply_url"] = f"https://x.com/{HANDLE}/status/{rid}"
            if e.get("reply_id") in seen:
                L.snapshot(e, seen[e["reply_id"]]["metrics"], now)
                stats["timeline"] += 1

        # due for a 24 h / 72 h checkpoint but not on the timeline: open the reply itself
        visits = 0
        for e in window:
            age = now - L._parse(e["posted_at"])
            snaps = e.get("metrics") or {}
            due = (age >= timedelta(hours=24) and "h24" not in snaps) or (age >= timedelta(hours=72) and "h72" not in snaps)
            if not e.get("reply_id") or not due or e["reply_id"] in seen or visits >= MAX_STATUS_VISITS:
                continue
            res = conversation(page, e["reply_id"])
            visits += 1
            if res.get("missing"):
                e["removed"] = bool(res.get("removed"))
            else:
                L.snapshot(e, res["metrics"], now)
                e["repliers"], e["reply_samples"] = res["repliers"], res["samples"]
                e["author_replied"] = (e.get("author") or "").lower() in {r.lower() for r in res["repliers"]}
                e["replies_checked"] = res["metrics"].get("replies", 0)
            _pause(2, 4)
        stats["status_visits"] = visits

        # new answers since the last look: who are they from?
        convs = 0
        for e in sorted(window, key=lambda x: -((x.get("metrics") or {}).get("last", {}).get("replies", 0))):
            n = (e.get("metrics") or {}).get("last", {}).get("replies", 0)
            if not e.get("reply_id") or n <= e.get("replies_checked", 0) or convs >= MAX_CONVERSATIONS:
                continue
            res = conversation(page, e["reply_id"])
            convs += 1
            if not res.get("missing"):
                e["repliers"], e["reply_samples"] = res["repliers"], res["samples"]
                e["author_replied"] = (e.get("author") or "").lower() in {r.lower() for r in res["repliers"]}
                e["replies_checked"] = n
            _pause(2, 4)
        stats["conversations"] = convs

    # who followed us since last time?
    try:
        current = newest_followers(page)
        try:
            known = set(json.loads(FOLLOWERS_SEEN.read_text(encoding="utf-8")))
        except Exception:
            known = set()
        new = set(current) - known if known else set()      # first run = baseline only
        FOLLOWERS_SEEN.write_text(json.dumps(sorted(known | set(current))[-5000:]), encoding="utf-8")
        stats["new_followers"] = len(new)
        stats["follows"] = L.attribute_follows(entries, new, now)
    except Exception as e:
        print(f"Tracker: followers unavailable ({e})", flush=True)

    for e in window:
        e["auto_grade"] = L.grade(e)
    L.save(files)
    return stats


def main() -> int:
    from auth import launch_and_auth
    res = launch_and_auth()
    if res[0] is None:
        print(f"Tracker: not logged in ({res[-1]})", flush=True)
        return 1
    pw, _, context, page = res
    try:
        from config import RUNS_FILE
        added = L.backfill_from_runs(RUNS_FILE)
        if added:
            print(f"Tracker: {added} earlier repl(ies) added to the ledger from runs.json", flush=True)
        stats = track(page)
        print("Tracker: " + json.dumps(stats), flush=True)
        return 0
    finally:
        try:
            context.close()
            pw.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Answer people who reply to our replies (Notifications > Mentions), at the start of each session.

A conversation is what turns a view into a profile visit: until now nobody answered them. Rules:
  * only answers to one of OUR replies (the tweet just above theirs is ours), less than 6 h old;
  * each answer at most once, each person once a day, MAX_PER_SESSION per session, MAX_PER_DAY a day;
  * the model skips trolls, "bot" accusations and empty answers ("lol", "facts", emojis);
  * same checks as normal replies: banned tics, repeated phrases, the fact-check pass.
State: <state dir>/conversations.json {"answered": [tweet ids], "day": date, "people": {user: n}, "count": n}.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timedelta, timezone

from config import LOCAL_STATE_DIR, TZ
from tracker import HANDLE, _ARTICLES_JS, _sid

MAX_PER_SESSION = 3
MAX_PER_DAY = 15
MAX_AGE = timedelta(hours=6)
STATE = LOCAL_STATE_DIR / "conversations.json"


def _load() -> dict:
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        st = {}
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if st.get("day") != today:
        st.update(day=today, people={}, count=0)
    st.setdefault("answered", [])
    return st


def _save(st: dict) -> None:
    st["answered"] = st["answered"][-1000:]
    STATE.write_text(json.dumps(st), encoding="utf-8")


def _articles(page, url: str) -> list[dict]:
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    try:
        page.wait_for_selector("article", timeout=20000)
    except Exception:
        return []
    time.sleep(random.uniform(2, 4))
    return page.evaluate(_ARTICLES_JS)


def mentions(page, st: dict, now: datetime) -> list[dict]:
    """Recent answers to us, newest first, not handled yet."""
    out = []
    for a in _articles(page, "https://x.com/notifications/mentions"):
        sid, user = _sid(a.get("link")), a.get("user") or ""
        try:
            at = datetime.fromisoformat((a.get("at") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        if (not sid or not user or user.lower() == HANDLE.lower() or sid in st["answered"]
                or now - at > MAX_AGE or st["people"].get(user.lower(), 0) >= 1 or len((a.get("text") or "").split()) < 3):
            continue
        out.append({"id": sid, "user": user, "text": a.get("text") or "", "at": at})
    return out


def thread(page, m: dict) -> dict | None:
    """{"origin": {user, text}, "ours": text} when the tweet answered is one of our replies."""
    arts = _articles(page, f"https://x.com/{m['user']}/status/{m['id']}")
    idx = next((i for i, a in enumerate(arts) if _sid(a.get("link")) == m["id"]), None)
    if not idx or (arts[idx - 1].get("user") or "").lower() != HANDLE.lower():
        return None
    origin = arts[idx - 2] if idx >= 2 else {}
    return {"origin": {"user": origin.get("user") or "", "text": origin.get("text") or ""}, "ours": arts[idx - 1].get("text") or "",
            "ours_id": _sid(arts[idx - 1].get("link"))}


def answer_mentions(page, *, recent: list[str], dry_run: bool = False, run_id: str = "") -> list[dict]:
    """Answer up to MAX_PER_SESSION people. Returns the posted answers (reply dicts like engine's)."""
    from llm_client import CONVO_PROMPT, call_llm, verify_reply
    from poster import post_reply, wait_before_next_tweet
    import poster
    from reply_validator import humanize, repeats_recent, validate_reply

    st = _load()
    if st["count"] >= MAX_PER_DAY:
        return []
    now = datetime.now(timezone.utc)
    posted: list[dict] = []
    for m in mentions(page, st, now)[:MAX_PER_SESSION * 3]:
        if len(posted) >= MAX_PER_SESSION or st["count"] >= MAX_PER_DAY:
            break
        st["answered"].append(m["id"])          # looked at once, answered or not
        t = thread(page, m)
        if not t:
            continue
        convo = (f"Original tweet by @{t['origin']['user'] or 'unknown'}:\n{t['origin']['text']}\n\n"
                 f"Our reply (@{HANDLE}):\n{t['ours']}\n\n@{m['user']} answered us:\n{m['text']}")
        res = call_llm(convo, m["user"], system=CONVO_PROMPT)
        text = ((res or {}).get("response") or "").strip()
        if not res or "SKIP" in (res.get("decision") or "").upper() or not text:
            print(f"  convo @{m['user']}: skip ({(res or {}).get('reason')})", flush=True)
            continue
        ok, why = validate_reply(text, [], tweet_text=convo + "\n" + "\n".join(res.get("facts") or []))
        rep = repeats_recent(text, recent + [p["reply_text"] for p in posted]) if ok else None
        if ok and not rep:
            ok, why = verify_reply(convo, list(res.get("facts") or []), text)
        if not ok or rep:
            print(f"  convo @{m['user']}: rejected ({rep or why})", flush=True)
            continue
        text = humanize(text)
        url = f"https://x.com/{m['user']}/status/{m['id']}"
        print(f"  convo @{m['user']}: {text!r}", flush=True)
        if not dry_run:
            success, err = post_reply(page, url, text)
            if not success:
                print(f"  convo post failed: {err}", flush=True)
                continue
        st["people"][m["user"].lower()] = st["people"].get(m["user"].lower(), 0) + 1
        st["count"] += 1
        p = {"tweet_url": url, "author": m["user"], "tweet_text": m["text"][:280], "tweet_likes": 0, "reply_text": text,
             "reason": "conversation", "fact_used": None, "context_used": False, "dry_run": dry_run,
             "reply_id": None if dry_run else poster.LAST_POSTED_ID, "posted_at": datetime.now(TZ).isoformat()}
        posted.append(p)
        if not dry_run:
            try:
                import replies_ledger
                replies_ledger.add(p, run_id=run_id)
            except Exception as e:
                print(f"  Ledger: could not record the answer ({e})", flush=True)
            wait_before_next_tweet()
    _save(st)
    return posted

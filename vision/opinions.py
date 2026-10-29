#!/usr/bin/env python3
"""
Our own takes between the replies: a profile made only of replies is what people check before they
write "bot account" (11 Oct). Up to MAX_PER_DAY standalone posts, in the evening (Paris), each on the
most-liked NBA tweet of the cycle, written in our voice, never quoting or tagging it.

Same checks as the replies: banned tics, repeated phrases, the fact-check pass.
State: <state dir>/opinions.json {"day": date, "count": n, "last": iso, "topics": [tweet ids]}.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from config import LOCAL_STATE_DIR, TZ

MAX_PER_DAY = 1   # 1 since the spam label (Oct 2026)
MIN_GAP = timedelta(hours=4)
HOURS = set(range(17, 24)) | {0, 1}      # Paris: US afternoon and evening
MIN_LIKES = 500
STATE = LOCAL_STATE_DIR / "opinions.json"


def _load() -> dict:
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        st = {}
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if st.get("day") != today:
        st.update(day=today, count=0)
    st.setdefault("topics", [])
    return st


def due(now: datetime | None = None, st: dict | None = None) -> bool:
    now = now or datetime.now(TZ)
    st = st if st is not None else _load()
    if now.hour not in HOURS or st["count"] >= MAX_PER_DAY:
        return False
    try:
        return now - datetime.fromisoformat(st["last"]) >= MIN_GAP
    except (KeyError, ValueError):
        return True


def maybe_post(context, tweets: list[dict], *, recent: list[str], dry_run: bool = False) -> str | None:
    """Post one take on the biggest fresh topic if one is due. Returns the text posted (or None)."""
    st = _load()
    if not due(st=st):
        return None
    pool = sorted((t for t in tweets if (t.get("likes") or 0) >= MIN_LIKES and t.get("tweet_id") not in st["topics"]),
                  key=lambda t: -(t.get("likes") or 0))
    if not pool:
        return None
    from llm_client import OPINION_PROMPT, call_llm, verify_reply
    from reply_validator import humanize, repeats_recent, validate_reply
    for t in pool[:3]:
        st["topics"].append(t["tweet_id"])
        res = call_llm(t.get("text") or "", t.get("username") or "", system=OPINION_PROMPT)
        text = ((res or {}).get("response") or "").strip()
        if not res or "SKIP" in (res.get("decision") or "").upper() or not text or "@" in text or "http" in text:
            continue
        facts = list(res.get("facts") or [])
        ok, why = validate_reply(text, [], tweet_text=(t.get("text") or "") + "\n" + "\n".join(facts))
        if ok and repeats_recent(text, recent):
            ok = False
        if ok:
            ok, why = verify_reply(t.get("text") or "", facts, text)
        if not ok:
            print(f"  Take on @{t.get('username')}: rejected ({why})", flush=True)
            continue
        text = humanize(text)
        print(f"  Take ({'dry run, not posted' if dry_run else 'posting'}): {text!r}", flush=True)
        if not dry_run:
            from publisher import post_thread
            ids, err = post_thread(context, [{"text": text, "image": None}])
            if not ids:
                print(f"  Take not posted: {err}", flush=True)
                break
        st["count"] += 1
        st["last"] = datetime.now(TZ).isoformat()
        st["topics"] = st["topics"][-300:]
        STATE.write_text(json.dumps(st), encoding="utf-8")
        return text
    st["topics"] = st["topics"][-300:]
    STATE.write_text(json.dumps(st), encoding="utf-8")
    return None

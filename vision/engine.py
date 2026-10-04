"""
NBAVision Engine — Orchestration: scrape -> filter -> score -> LLM -> validate -> post.
Batches LLM calls concurrently for speed.
"""
import os
import random
import time
from datetime import datetime
from filter_tweets import filter_tweets, minutes_since_post
from scorer import rank_and_top
from llm_client import call_llm
from reply_validator import validate_reply
from poster import post_reply, wait_before_next_tweet
from session_log import write_session_log, build_session_log
from auth import save_session_state, check_session_alive
from notify import notify_session_summary
from config import (
    MAX_REPLIES,
    MAX_REPLIES_PER_AUTHOR,
    CYCLE_INTERVAL_MINUTES,
    CYCLE_INTERVAL_JITTER_SEC,
    MAX_CONSECUTIVE_ERRORS,
    MAX_POSTING_FAILURES,
    DRY_RUN,
    TZ,
)

# Serial LLM calls with delay to stay under Groq rate limit (avoids 429)
LLM_SERIAL_DELAY_SEC = (3.0, 4.5)

SESSION_HEALTH_CHECK_INTERVAL = 5


def _engagement_velocity(tweet: dict) -> float:
    minutes = max(1, minutes_since_post(tweet.get("timestamp") or ""))
    likes = tweet.get("likes") or 0
    replies = tweet.get("replies") or 0
    retweets = tweet.get("retweets") or 0
    return (likes + 2 * replies + 2 * retweets) / minutes


def _event(step: str, detail: dict | None = None) -> dict:
    at = datetime.now(TZ).isoformat()
    return {"step": step, "at": at, "detail": detail or {}}


def run_session(page, context, *, browser, playwright_instance):
    """
    Run one full session: cycles until max_replies or stop conditions.
    Returns dict of session stats for logging.
    """
    run_id = os.environ.get("NBAVISION_RUN_ID")
    start_time = datetime.now(TZ).isoformat()
    seen_tweet_ids: set[str] = set()
    replied_author_count: dict[str, int] = {}
    session_replies: list[str] = []
    replies_posted: list[dict] = []
    total_scraped = 0
    total_filtered = 0
    total_scored = 0
    total_llm_calls = 0
    total_replied = 0
    total_skipped = 0
    skip_reasons: dict[str, int] = {}
    response_lengths: list[int] = []
    engagement_velocities: list[float] = []
    cycles: list[dict] = []
    events = [_event("session_start", {"max_replies": MAX_REPLIES, "dry_run": DRY_RUN})]

    consecutive_errors = 0
    posting_failures = 0
    cycle_index = 0

    if DRY_RUN:
        print("*** DRY RUN MODE — no replies will be posted ***", flush=True)

    while total_replied < MAX_REPLIES and posting_failures < MAX_POSTING_FAILURES:
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            events.append(_event("session_stop", {"reason": "max_consecutive_errors"}))
            print(f"Stopping: {consecutive_errors} consecutive errors reached limit", flush=True)
            break

        # Periodic session health check
        if cycle_index > 0 and cycle_index % SESSION_HEALTH_CHECK_INTERVAL == 0:
            print(f"[Cycle {cycle_index}] Session health check...", flush=True)
            if not check_session_alive(page):
                print("Session appears dead — cookies may have been invalidated", flush=True)
                events.append(_event("session_stop", {"reason": "session_dead"}))
                break
            print("  Session still alive.", flush=True)

        events.append(_event("cycle_start", {"cycle_index": cycle_index}))
        print(f"[Cycle {cycle_index}] Scraping...", flush=True)
        try:
            from scraper import scrape_all_keywords
            raw = scrape_all_keywords(page, context, cycle_index=cycle_index)
            total_scraped += len(raw)
            events.append(_event("scrape_done", {"raw_count": len(raw)}))
            consecutive_errors = max(0, consecutive_errors - 1)
        except Exception as e:
            consecutive_errors += 1
            events.append(_event("scrape_error", {"error": str(e)}))
            print(f"[Cycle {cycle_index}] Scrape error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}", flush=True)
            backoff = min(300, 60 * (2 ** (consecutive_errors - 1)))
            time.sleep(backoff + random.uniform(0, 30))
            cycle_index += 1
            continue

        if len(raw) == 0:
            print(f"[Cycle {cycle_index}] No tweets scraped — skipping cycle", flush=True)
            events.append(_event("cycle_empty", {"cycle_index": cycle_index}))
            cycle_index += 1
            time.sleep(random.uniform(30, 60))
            continue

        accepted, cycle_skip = filter_tweets(raw, seen_tweet_ids)
        total_filtered += len(accepted)
        skip_detail = ", ".join(f"{k}={v}" for k, v in sorted(cycle_skip.items())) if cycle_skip else "none"
        print(f"[Cycle {cycle_index}] Scraped {len(raw)}, accepted {len(accepted)} (filter out: {skip_detail})", flush=True)
        for k, v in cycle_skip.items():
            skip_reasons[k] = skip_reasons.get(k, 0) + v
        events.append(_event("filter_done", {"accepted_count": len(accepted), "cycle_skip_reasons": cycle_skip}))

        top = rank_and_top(accepted)
        total_scored += len(top)
        cycles.append({
            "cycle_index": cycle_index,
            "scraped_count": len(raw),
            "accepted_count": len(accepted),
            "top_count": len(top),
            "at": datetime.now(TZ).isoformat(),
        })
        events.append(_event("rank_done", {"top_count": len(top)}))
        top_preview = [f"@{t.get('username')}(L{t.get('likes') or 0})" for t in top[:5]]
        print(f"[Cycle {cycle_index}] Top {len(top)} to evaluate: {', '.join(top_preview)}{'...' if len(top) > 5 else ''}", flush=True)

        # Filter candidates (author limit, already seen)
        candidates = []
        for tweet in top:
            if len(candidates) + total_replied >= MAX_REPLIES + 10:
                break
            tweet_id = tweet.get("tweet_id")
            author = tweet.get("username") or ""
            if replied_author_count.get(author, 0) >= MAX_REPLIES_PER_AUTHOR:
                total_skipped += 1
                skip_reasons["max_per_author"] = skip_reasons.get("max_per_author", 0) + 1
                events.append(_event("skip", {"tweet_id": tweet_id, "reason": "max_per_author", "author": author}))
                continue
            seen_tweet_ids.add(tweet_id)
            candidates.append(tweet)

        # Serial LLM calls with delay to avoid Groq 429 (no concurrent requests)
        llm_results: dict[str, dict | None] = {}
        for i, tweet in enumerate(candidates):
            if i > 0:
                time.sleep(random.uniform(*LLM_SERIAL_DELAY_SEC))
            total_llm_calls += 1
            tid = tweet.get("tweet_id")
            try:
                llm_results[tid] = call_llm(tweet.get("text") or "", tweet.get("username") or "")
            except Exception:
                llm_results[tid] = None

        # Process results in order and post
        for tweet in candidates:
            if total_replied >= MAX_REPLIES:
                break
            tweet_id = tweet.get("tweet_id")
            author = tweet.get("username") or ""
            tweet_url = f"https://x.com/{author}/status/{tweet_id}"

            events.append(_event("llm_call", {"tweet_id": tweet_id, "author": author}))
            llm_result = llm_results.get(tweet_id)
            if llm_result is None:
                total_skipped += 1
                skip_reasons["llm_timeout"] = skip_reasons.get("llm_timeout", 0) + 1
                consecutive_errors += 1
                print(f"  @{author}: Skip (LLM timeout)", flush=True)
                events.append(_event("llm_skip", {"tweet_id": tweet_id, "reason": "llm_timeout"}))
                continue
            decision = (llm_result.get("decision") or "").upper()
            reason = llm_result.get("reason") or ""
            response = (llm_result.get("response") or "").strip()

            if "SKIP" in decision or not response:
                total_skipped += 1
                skip_reasons[reason or "skip"] = skip_reasons.get(reason or "skip", 0) + 1
                consecutive_errors = 0
                print(f"  @{author}: Skip — {reason or '(no reason)'}", flush=True)
                events.append(_event("llm_skip", {"tweet_id": tweet_id, "reason": reason or "skip", "decision": decision}))
                continue

            response = response[:180].strip()
            reply_preview = (response[:60] + "...") if len(response) > 60 else response
            print(f"  @{author}: Reply ({len(response)} chars): {reply_preview!r}", flush=True)
            valid, fail_reason = validate_reply(response, session_replies, tweet_text=tweet.get("text") or "")
            if not valid:
                total_skipped += 1
                skip_reasons[fail_reason or "validation"] = skip_reasons.get(fail_reason or "validation", 0) + 1
                print(f"  @{author}: Skip (validation — {fail_reason})", flush=True)
                events.append(_event("validation_fail", {"tweet_id": tweet_id, "reason": fail_reason}))
                continue

            print(f"  Posting to {tweet_url}...", flush=True)
            events.append(_event("post_attempt", {"tweet_id": tweet_id, "tweet_url": tweet_url}))
            success, err = post_reply(page, tweet_url, response)
            if not success:
                posting_failures += 1
                print(f"  Post failed ({posting_failures}/{MAX_POSTING_FAILURES}): {err}", flush=True)
                if posting_failures >= MAX_POSTING_FAILURES:
                    events.append(_event("post_fail", {"tweet_id": tweet_id, "error": str(err)}))
                    break
                time.sleep(random.uniform(10, 20))
                success, err = post_reply(page, tweet_url, response)
                if not success:
                    total_skipped += 1
                    skip_reasons["post_failed"] = skip_reasons.get("post_failed", 0) + 1
                    events.append(_event("post_fail", {"tweet_id": tweet_id, "error": str(err)}))
                    continue

            total_replied += 1
            replied_author_count[author] = replied_author_count.get(author, 0) + 1
            session_replies.append(response)
            replies_posted.append({
                "tweet_url": tweet_url,
                "reply_text": response,
                "posted_at": datetime.now(TZ).isoformat(),
            })
            response_lengths.append(len(response))
            engagement_velocities.append(_engagement_velocity(tweet))
            consecutive_errors = 0
            posting_failures = 0
            print(f"  Posted. Session: {total_replied}/{MAX_REPLIES}", flush=True)
            events.append(_event("post_ok", {"tweet_id": tweet_id, "tweet_url": tweet_url, "reply_length": len(response)}))

            wait_before_next_tweet()

        # Save session state after each cycle so cookies stay fresh
        try:
            save_session_state(context)
        except Exception:
            pass

        cycle_index += 1
        if total_replied < MAX_REPLIES:
            interval_sec = CYCLE_INTERVAL_MINUTES * 60 + random.uniform(-CYCLE_INTERVAL_JITTER_SEC, CYCLE_INTERVAL_JITTER_SEC)
            interval_sec = max(15, interval_sec)
            print(f"[Cycle {cycle_index}] Sleeping {int(interval_sec)}s until next cycle", flush=True)
            time.sleep(interval_sec)

    end_time = datetime.now(TZ).isoformat()
    events.append(_event("session_end", {"total_replied": total_replied, "total_skipped": total_skipped}))
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    avg_engagement_velocity = sum(engagement_velocities) / len(engagement_velocities) if engagement_velocities else 0

    print("--- Session summary ---", flush=True)
    print(f"  Replied: {total_replied}, skipped: {total_skipped}, LLM calls: {total_llm_calls}", flush=True)
    print(f"  Scraped: {total_scraped}, passed filter: {total_filtered}, scored: {total_scored}", flush=True)
    print(f"  Cycles: {cycle_index}, posting failures: {posting_failures}", flush=True)
    if skip_reasons:
        print(f"  Skip reasons: {dict(skip_reasons)}", flush=True)
    if DRY_RUN:
        print("  *** DRY RUN — no replies were actually posted ***", flush=True)
    print("---", flush=True)

    notify_session_summary(total_replied, total_skipped, skip_reasons if skip_reasons else None)

    log_data = build_session_log(
        start_time=start_time,
        end_time=end_time,
        total_scraped=total_scraped,
        total_filtered=total_filtered,
        total_scored=total_scored,
        total_llm_calls=total_llm_calls,
        total_replied=total_replied,
        total_skipped=total_skipped,
        skip_reasons=skip_reasons,
        avg_response_length=avg_response_length,
        avg_engagement_velocity=avg_engagement_velocity,
        replies_posted=replies_posted,
        run_id=run_id,
        cycles=cycles,
        events=events,
    )
    write_session_log(log_data)
    return log_data

"""
NBAVision Engine — Reply generation.
Without LLM API key: uses template replies (no Groq, no setup).
With LLM API key: uses Groq for AI-generated replies. Handles 429 with backoff.
"""
from __future__ import annotations
import json
import os
import random
import re
import time
from datetime import datetime
import requests
from config import get_llm_api_key, get_llm_model, LLM_TIMEOUT_SECONDS, LLM_RETRY_MAX, TZ

# Used when the configured Groq model is retired or misspelled
FALLBACK_MODEL = "openai/gpt-oss-20b"

# Template replies when no LLM key — no API, no credentials
TEMPLATE_REPLIES = [
    "Tough matchup. Defense will decide it.",
    "Key is who shows up in the 4th.",
    "Can't sleep on the role players in this one.",
    "Matchup to watch: the paint.",
    "Coaching will matter more than people think.",
    "Bench depth could swing this.",
    "Expect a physical game.",
    "The X-factor is health.",
    "Clutch time will tell.",
    "Rebounding battle will be huge.",
]

SYSTEM_PROMPT = """You reply to tweets as @NBAPredictLab: a sharp NBA fan account that knows the numbers. Goal: a reply people like and that makes them check the profile. Most tweets deserve no reply.

Return exactly one JSON object: {"decision": "REPLY" or "SKIP", "reason": "...", "response": "...", "fact_used": "..." or ""}

SKIP (put the reason in "reason") when:
- not clearly about NBA basketball (birthdays, politics, crypto, betting tips, other sports, vague memes) -> "not_about_basketball"
- text under ~35 characters that is probably a caption for media you cannot see -> "likely_media_caption"
- death, crime, health tragedy, heavy politics, harassment -> "sensitive"
- you cannot add anything specific -> "nothing_to_add"
- live score / lineup / "first bucket" / stat-line posts where you would only repeat the numbers, and generic preseason hype -> "nothing_to_add"

TRUTH RULES (most important):
- You only know two things: the tweet, and the VERIFIED FACTS block (today's data). Your training memory about rosters, trades, injuries and stats is OUT OF DATE: never use it.
- Never state a team, player, trade, injury, record, score or stat unless it is written in the tweet or in VERIFIED FACTS. If VERIFIED FACTS contradict what you remember, the facts win.
- Facts are OPTIONAL. Most good replies use none. Use one only when it directly answers or sharpens what the tweet says; never bolt a record onto a reply about something else. If you use one, keep its exact numbers and its tense ("finished last season 45-37" is last season, not now) and copy it word for word into "fact_used".
- Don't predict outcomes as certainties. A win chance from the facts may be quoted as a percentage.
- Don't derive new claims from the facts or the tweet: no rankings ("best in the league", "topped the West"), no home/away or schedule details, no "tonight" unless stated. Never correct or contradict a factual claim of the tweet (records, signings, trades, waivers, injuries, who plays where, titles): the tweet is newer than our facts; if a fact disagrees with the tweet, SKIP with "facts_conflict". Never assert a stat, comparison or trend that is in neither the tweet nor the facts (no "he has more series wins than", no "they've been flirting with the play-in lately"). A record means only what its numbers say: more wins than losses is a winning record, fewer is a losing one (32-50 is well below .500, never "cracked .500").

A GOOD REPLY:
- reacts to what THIS tweet says (agree + add one specific, push back with a reason, or a dry joke)
- never just restates or summarises the tweet: if you have no take of your own, SKIP with "nothing_to_add"
- 40-100 characters, usually ONE punchy sentence (short replies get the most likes and answers; a sharp take beats a summary), no hashtags, no links, no self-promotion, at most one emoji (usually none)
- no em dashes (—); plain punctuation
- sounds like a fan typing on a phone, not an analyst: no "I'd argue", "speaks volumes", "at the end of the day", "only time will tell", "it will be interesting", "key factor", "moving forward", "narrative", "chemistry", "resilience", "upside", "hope", "hype", "vibes", "solid", "curious to see", "rough", "nice to see"

Examples
Tweet: "Knicks have been the best team in the East since January, no debate"
Facts: New York Knicks are 52-29 so far this season (2025-26)
Good: {"decision":"REPLY","reason":"agree_with_number","response":"52-29 and nobody wants that matchup in May. Hard to argue.","fact_used":"New York Knicks are 52-29 so far this season (2025-26)"}
Tweet: "LeBron still looks smooth in practice at 41"
Facts: LeBron James plays for the Philadelphia 76ers; Philadelphia 76ers finished last season (2025-26) 45-37
Good: {"decision":"REPLY","reason":"agree","response":"41 and still moving like he's got a Finals to win. Wild.","fact_used":""}
Tweet: "this man is HIM"  (no names, likely a video)
Good: {"decision":"SKIP","reason":"likely_media_caption","response":"","fact_used":""}
Tweet: "Spurs fans acting like they already won a ring"
Facts: San Antonio Spurs next: vs DAL Wed Oct 21 8:30 PM ET; betting market win chance 71%
Good: {"decision":"REPLY","reason":"pushback","response":"Books still have them at 71% tomorrow night. Confidence is earned, rings aren't.","fact_used":"San Antonio Spurs next: vs DAL Wed Oct 21 8:30 PM ET; betting market win chance 71%"}

Return only the JSON."""


GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Tokens used by this process (read by the replay lab to compare settings)
USAGE = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0}


def _reasoning_args(model: str, env: str = "LLM_REASONING") -> dict:
    """gpt-oss (Groq): low effort. DeepSeek thinks at HIGH effort by default, which is most of the
    cost (output tokens): LLM_REASONING = low (default) | high | none (thinking disabled)."""
    if "gpt-oss" in model:
        return {"reasoning_effort": "low"}
    if "deepseek" in model:
        level = os.getenv(env) or os.getenv("LLM_REASONING", "high")
        return {"thinking": {"type": "disabled"}} if level == "none" else {"reasoning_effort": level}
    return {}


def _count(data: dict) -> None:
    u = (data or {}).get("usage") or {}
    USAGE["calls"] += 1
    USAGE["prompt"] += int(u.get("prompt_tokens") or 0)
    USAGE["cached"] += int(u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    USAGE["completion"] += int(u.get("completion_tokens") or 0)


def switch_to_fallback(why: str) -> bool:
    """DeepSeek over budget, out of credit or down: continue on Groq (free tier) instead of stopping.
    Returns True if the switch happened (False if already on Groq or no Groq key)."""
    groq = os.getenv("GROQ_API_KEY", "").strip()
    if not groq or "groq" in os.getenv("LLM_BASE_URL", GROQ_BASE_URL):
        return False
    os.environ.update({"LLM_API_KEY": groq, "LLM_BASE_URL": GROQ_BASE_URL, "LLM_MODEL": "openai/gpt-oss-120b",
                       "LLM_FALLBACK_ACTIVE": why})
    print(f"    LLM: switching to Groq ({why})", flush=True)
    return True


def _extract_json(text: str):
    """Try to parse JSON from LLM output (allow markdown code block)."""
    text = (text or "").strip()
    # Try raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try first { ... }
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _should_skip_template(text: str) -> bool:
    """Simple skip: avoid toxic/sensitive topics."""
    t = (text or "").lower()
    skip_words = ["death", "died", "kill", "crime", "war", "scandal", "arrest"]
    return any(w in t for w in skip_words)


def call_llm(tweet_text: str, tweet_author: str = ""):
    """
    Returns {"decision": "REPLY"|"SKIP", "reason": "...", "response": "..."}.
    Without API key: uses template replies. With key: calls Groq.
    """
    api_key = get_llm_api_key()
    if not api_key:
        # Template mode — no Groq, no credentials
        if _should_skip_template(tweet_text):
            print("    LLM: template mode — skip (sensitive)", flush=True)
            return {"decision": "SKIP", "reason": "template_skip", "response": ""}
        reply = random.choice(TEMPLATE_REPLIES)
        print(f"    LLM: template reply ({len(reply)} chars)", flush=True)
        return {
            "decision": "REPLY",
            "reason": "template",
            "response": reply,
        }

    model = get_llm_model()
    print(f"    LLM: calling {model}...", flush=True)
    # Any OpenAI-compatible endpoint (Groq by default; e.g. http://localhost:11434/v1 for Ollama tests)
    url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        import nba_context
        facts = nba_context.facts_for(tweet_text or "")
        today_line = nba_context.today_line()
    except Exception as e:  # context is a bonus, never a blocker
        print(f"    Context unavailable: {e}", flush=True)
        facts, today_line = [], f"Today is {datetime.now(TZ).strftime('%Y-%m-%d')}."
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(none for this tweet)"
    user_content = (f"{today_line}\n\nVERIFIED FACTS:\n{facts_block}\n\n"
                    f"Tweet by @{tweet_author or 'unknown'}:\n{tweet_text or ''}")

    max_attempts = LLM_RETRY_MAX + 1
    max_429_backoffs = 5
    last_err = None

    for attempt in range(max_attempts + max_429_backoffs):
        try:
            r = requests.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    # reasoning models spend tokens thinking before the JSON: keep it short
                    "max_tokens": 1200,
                    "temperature": 0.5,
                    **_reasoning_args(model),
                    "response_format": {"type": "json_object"},
                },
                timeout=LLM_TIMEOUT_SECONDS,
            )

            if r.status_code == 429:
                retry_after = 60
                if "Retry-After" in r.headers:
                    try:
                        retry_after = int(r.headers["Retry-After"])
                    except ValueError:
                        pass
                retry_after = min(120, max(retry_after, 60))
                if attempt < max_attempts + max_429_backoffs - 1:
                    print(f"    LLM: 429 rate limit — waiting {retry_after}s then retry", flush=True)
                    time.sleep(retry_after)
                    continue
                last_err = "429 Too Many Requests"
                print(f"    LLM: 429 — all backoffs exhausted", flush=True)
                break

            r.raise_for_status()
            data = r.json()
            _count(data)
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = _extract_json(content)
            if parsed and isinstance(parsed.get("decision"), str):
                dec = (parsed.get("decision") or "").upper()
                reason = (parsed.get("reason") or "")[:80]
                fact = (parsed.get("fact_used") or "").strip()
                # a quoted fact must be one we supplied, verbatim
                if fact and fact not in facts:
                    print("    LLM: cited a fact we did not supply -> SKIP", flush=True)
                    return {"decision": "SKIP", "reason": "unverified_fact", "response": ""}
                parsed["context_used"] = bool(fact)
                parsed["facts"] = facts
                print(f"    LLM: {dec} — {reason}{' [fact]' if fact else ''}", flush=True)
                return parsed
            print("    LLM: invalid output -> SKIP", flush=True)
            return {"decision": "SKIP", "reason": "invalid_llm_output", "response": ""}
        except requests.Timeout:
            last_err = "timeout"
            print(f"    LLM: timeout (attempt {attempt + 1})", flush=True)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if (code in (401, 402, 403) or code >= 500) and switch_to_fallback(f"HTTP {code}"):
                api_key, model = get_llm_api_key(), get_llm_model()
                url = os.getenv("LLM_BASE_URL", GROQ_BASE_URL).rstrip("/") + "/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                continue
            if (e.response is not None and e.response.status_code in (400, 404)
                    and "model" in (e.response.text or "").lower() and model != FALLBACK_MODEL):
                print(f"    LLM: model {model!r} unavailable -> {FALLBACK_MODEL}", flush=True)
                model = FALLBACK_MODEL
                continue
            if e.response is not None and e.response.status_code == 429:
                retry_after = 60
                if e.response.headers.get("Retry-After"):
                    try:
                        retry_after = int(e.response.headers["Retry-After"])
                    except ValueError:
                        pass
                retry_after = min(120, max(retry_after, 60))
                if attempt < max_attempts + max_429_backoffs - 1:
                    print(f"    LLM: 429 — waiting {retry_after}s then retry", flush=True)
                    time.sleep(retry_after)
                    continue
            last_err = str(e)
            print(f"    LLM: error — {e}", flush=True)
        except Exception as e:
            last_err = str(e)
            print(f"    LLM: error — {e}", flush=True)

    print("    LLM: all attempts failed -> None", flush=True)
    return None


VERIFY_PROMPT = """You check a reply an NBA fan account is about to post on X, before it goes out.
List every FACTUAL claim in the reply (numbers, records, ages, dates, schedules, who plays for which team,
titles, injuries, stats, comparisons, places, what someone did or said). Opinions, jokes, predictions phrased
as opinions ("could be", "should", "let's see") and plain reactions are NOT claims.
A claim is supported only if it is stated in the TWEET or in the VERIFIED FACTS, or follows from them
directly (same numbers, same team, same event). General knowledge does not count: our data may be newer than
yours (trades, signings, titles) and the tweet is newer than both.
Also unsupported: anything that contradicts or "corrects" the tweet, and misleading framing (e.g. "still no
ring" for a player who has one, "at home" when the tweet says the game is away).
Return JSON only: {"claims": [{"claim": "...", "supported": true/false, "why": "short"}], "verdict": "ok" or "reject"}
verdict is "reject" if any claim is unsupported."""


def verify_reply(tweet_text: str, facts: list, reply: str) -> tuple[bool, str]:
    """Second pass: every factual claim of `reply` must come from the tweet or our verified facts.
    Returns (ok, reason). Fails closed only on a clear 'reject'; an unreadable check lets the reply
    through (the rule-based validator already ran)."""
    api_key = get_llm_api_key()
    if not api_key:
        return True, "no_key"
    url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/") + "/chat/completions"
    model = get_llm_model()
    facts_block = "\n".join(f"- {f}" for f in (facts or [])) or "(none)"
    last = "rate limited"
    user = f"TWEET:\n{tweet_text}\n\nVERIFIED FACTS:\n{facts_block}\n\nREPLY TO CHECK:\n{reply}"
    for attempt in range(4):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                              json={"model": model, "temperature": 0, "max_tokens": 900,
                                    **_reasoning_args(model, "VERIFY_REASONING"),
                                    "response_format": {"type": "json_object"},
                                    "messages": [{"role": "system", "content": VERIFY_PROMPT},
                                                 {"role": "user", "content": user}]},
                              timeout=LLM_TIMEOUT_SECONDS + 10)
            if r.status_code == 429:
                time.sleep(min(120, max(30, int(r.headers.get("Retry-After", "60") or 60))))
                continue
            if (r.status_code in (401, 402, 403) or r.status_code >= 500) and switch_to_fallback(f"HTTP {r.status_code}"):
                api_key, model = get_llm_api_key(), get_llm_model()
                url = os.getenv("LLM_BASE_URL", GROQ_BASE_URL).rstrip("/") + "/chat/completions"
                continue
            r.raise_for_status()
            data = r.json()
            _count(data)
            js = _extract_json(data["choices"][0]["message"].get("content") or "") or {}
            bad = [c for c in js.get("claims", []) if isinstance(c, dict) and c.get("supported") is False]
            if (js.get("verdict") or "").lower() == "reject" or bad:
                why = "; ".join(f"{c.get('claim')} ({c.get('why')})" for c in bad)[:240] or "rejected"
                return False, why
            return True, "ok"
        except Exception as e:
            last = str(e)
            time.sleep(5)
    return True, f"check_failed: {last[:80]}"

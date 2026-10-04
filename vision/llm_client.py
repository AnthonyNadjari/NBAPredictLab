"""
NBAVision Engine — Reply generation.
Without LLM API key: uses template replies (no Groq, no setup).
With LLM API key: uses Groq for AI-generated replies. Handles 429 with backoff.
"""
import json
import random
import re
import time
from datetime import datetime
import requests
from config import get_llm_api_key, get_llm_model, LLM_TIMEOUT_SECONDS, LLM_RETRY_MAX, TZ

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

SYSTEM_PROMPT = """You are an NBA/basketball fan on X. You ONLY reply to tweets that are clearly about NBA basketball.

DEFAULT: SKIP. Only REPLY when your reply adds ZERO claims (who wins, who plays, what will happen, roster, availability) that are not explicitly in the tweet. When in doubt, SKIP.

OUTPUT FORMAT (STRICT)

Return exactly one JSON object:

{
  "decision": "REPLY" or "SKIP",
  "reason": "...",
  "response": "..."
}

NBA-ONLY RULE (CRITICAL)

REPLY only when the tweet is clearly and primarily about NBA/basketball. You MUST SKIP with reason "not_about_basketball" when the tweet is about:
- Birthdays, anniversaries, personal life (unless directly about an NBA player's career/game)
- Politics, crypto, NFT, betting tips, unrelated sports, memes not about basketball
- Generic or vague text that could be about anything
- Anything where the main point is not NBA/basketball

If the tweet text is very short (under ~35 characters) it may be a caption for an image or video we cannot see. SKIP with reason "likely_media_caption" unless the short text explicitly mentions NBA/basketball (e.g. "LeBron 40 points").

If REPLY:
- ≤180 characters
- No hashtags, no links, no promotion
- Max one emoji (most replies should have zero)
- Specific to THIS tweet's basketball content

ALSO SKIP if:
- Death, serious crime, heavy politics
- Tweet is in a language you can't reply to naturally
- Tweet is literally just a link with no text

FACTS / HALLUCINATION (CRITICAL)

NEVER invent specific facts. Your reply must ONLY:
- Comment on what the tweet actually says, or
- Use generic opinions (e.g. "defense matters", "tough matchup") that do not assert teams, trades, or seasons.
Do NOT claim "X played at Y last year", "he was with the Z in 2023", or any specific team/season/trade fact unless the tweet states it explicitly. When in doubt, SKIP with reason "unsure_or_invented".

ACCURACY (AS OF TODAY)

Everything you state must be accurate as of today. Do not state old-dated or outdated information (e.g. past seasons, old rosters, trades, or events that may have changed). If you are unsure whether a fact is still current, either keep your reply generic or SKIP. Prefer skipping over replying with anything that could be read as outdated.

CONTEXT (CRITICAL)

Your reply must directly address the tweet's topic. If your reply would be generic, unrelated, or could apply to any tweet, SKIP with reason "off_topic". The reply must be clearly about the same subject as the tweet.

STRICT REPLY SCOPE (CRITICAL — AVOID OUT-OF-CONTEXT REPLIES)

Your reply must ONLY respond to what the tweet explicitly says. Do NOT add claims about games, wins, losses, who is playing, or roster/availability unless the tweet itself states them.
- If the tweet is skeptical or about "hype" (e.g. "until then it's just hype", "let's see them win without X and Y"), respond ONLY to that skepticism or hype — do NOT add predictions about who will win, whether they will win close games, or who is playing.
- Do NOT say "they" or "them" winning/losing games unless the tweet is clearly about that. Do NOT introduce specific players, teams, or seasons the tweet does not mention.
- When in doubt, SKIP with reason "unsure_or_invented" or "off_topic". Prefer skipping over replying with anything that could be read as a claim the tweet didn't make.

EXAMPLE — tweet is skeptical/hype (e.g. "But let's see them win one of those close games without George and Williams playing together. Until then, it's just hype"):
- BAD (SKIP): Any reply that talks about "them" winning, close games, who is playing, or adds roster/availability. Your reply must NOT mention winning games, who plays, or "until then" unless you are only agreeing with the skepticism.
- GOOD: Reply ONLY to the skepticism/hype point in short form, e.g. "Fair. Prove it first." or "Yeah the hype is real until they do it." Do NOT add "see them win", "close games", "without X and Y", or any new claim.

VOICE

Write like a real fan on their phone:
- Slightly opinionated
- Comfortable being blunt
- Uses short sentences naturally
- Sometimes agrees and adds a detail
- Sometimes pushes back
- Occasionally funny or dry
- Never sounds like a TV analyst or a blog

BANNED PHRASES

Never use: "I'd argue", "speaks volumes", "at the end of the day", "it will be interesting", "cannot let", "that type of", "key factor", "moving forward", "the question is", "only time will tell"

Never use abstract filler: chemistry, resilience, upside, value, efficiency, narrative

REPLY TYPES (mix these)

- Quick agree + detail: "Yeah his midrange has been different since January"
- Pushback: "Nah that's a regular season take. Playoffs he's getting trapped"
- Observation: "Watch his feet on that closeout. That's why he's elite"
- Prediction: "This team is a second-round exit and everyone knows it"
- Vibe: "They just look flat out there"
- Humor: "Bro got cooked so bad the arena went quiet"

LENGTH: 60–150 characters. Short and sharp > long and safe.

Return only the JSON."""


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
    print(f"    LLM: calling Groq ({model})...", flush=True)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    user_content = f"Today (UTC+1): {today}. All facts in your reply must be accurate as of this date — do not state old-dated or outdated information.\n\nTweet:\n{tweet_text or ''}\n\nAuthor:\n{tweet_author or 'unknown'}"

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
                    "max_tokens": 256,
                    "temperature": 0.35,
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
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = _extract_json(content)
            if parsed and isinstance(parsed.get("decision"), str):
                dec = (parsed.get("decision") or "").upper()
                reason = (parsed.get("reason") or "")[:80]
                print(f"    LLM: {dec} — {reason}", flush=True)
                return parsed
            print("    LLM: invalid output -> SKIP", flush=True)
            return {"decision": "SKIP", "reason": "invalid_llm_output", "response": ""}
        except requests.Timeout:
            last_err = "timeout"
            print(f"    LLM: timeout (attempt {attempt + 1})", flush=True)
        except requests.HTTPError as e:
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

"""
NBAVision Engine — Vérification avancée des réponses (Spec Section 9).
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (
    MAX_RESPONSES_SAME_FIRST_WORD,
    MAX_EMOJIS_IN_SESSION,
    MAX_SENTENCES,
    TFIDF_SIMILARITY_THRESHOLD,
)


def _first_word(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    return t.split()[0].lower() if t.split() else ""


def _count_emojis(text: str) -> int:
    # Simple emoji count: common Unicode ranges + common symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"  # misc symbols and pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(text or ""))


def _count_sentences(text: str) -> int:
    if not (text or "").strip():
        return 0
    return max(1, len(re.split(r"[.!?]+", text.strip())))


# Phrases that often signal invented team/season facts — reject reply if present
INVENTED_PHRASE_BLOCKLIST = [
    "last year at",
    "last season at",
    "when he was with the",
    "when he played for",
    "back when he was on",
    "in 2023",
    "in 2022",
    "in 2021",
    "see them win",
    "win one of those close games",
    "close games without",
    "without george",
    "without williams",
    "george and williams",
    "williams playing",
    "playing together",
    "until then it's just hype",
    "win without",
]


def _reply_has_invented_phrase(text: str) -> bool:
    """True if reply contains a blocklisted phrase suggesting invented facts."""
    t = (text or "").lower()
    return any(phrase in t for phrase in INVENTED_PHRASE_BLOCKLIST)


# Entities that must not appear in reply unless they appear in the tweet (avoids adding roster/game claims)
_REPLY_ENTITY_WHITELIST_REQUIRED = (
    "george", "williams", "james", "curry", "durant", "jokic", "giannis", "tatum", "embiid",
    "doncic", "lakers", "celtics", "warriors", "bucks", "suns", "nuggets", "clippers", "heat",
    "knicks", "mavericks", "thunder", "cavaliers", "grizzlies", "pelicans", "kings", "pacers",
)


def _reply_adds_entity_not_in_tweet(reply: str, tweet_text: str) -> bool:
    """True if reply contains a listed entity (name/team) that the tweet does not mention."""
    if not (reply or "").strip() or not (tweet_text or "").strip():
        return False
    reply_lower = reply.lower()
    tweet_lower = tweet_text.lower()
    for entity in _REPLY_ENTITY_WHITELIST_REQUIRED:
        if entity in reply_lower and entity not in tweet_lower:
            return True
    return False


def _meaningful_word_overlap(tweet_text: str, reply_text: str) -> bool:
    """True if reply shares at least one meaningful word (3+ chars, not common stop) with tweet."""
    stop = {"the", "and", "for", "was", "but", "not", "you", "all", "can", "had", "her", "his", "its", "may", "now", "our", "out", "see", "has", "him", "how", "man", "new", "old", "any", "day", "get", "got", "let", "put", "say", "too", "use", "who", "did", "get", "got", "let", "put", "say", "too", "use"}
    tweet_words = {w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", tweet_text or "") if w.lower() not in stop}
    reply_words = {w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", reply_text or "")}
    return bool(tweet_words & reply_words)


def _tfidf_similarity(new_text: str, previous_texts: list[str]) -> float:
    """Return max similarity between new_text and any of previous_texts (0..1)."""
    if not previous_texts or not (new_text or "").strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        all_texts = previous_texts + [new_text]
        matrix = vectorizer.fit_transform(all_texts)
        from sklearn.metrics.pairwise import cosine_similarity
        new_vec = matrix[-1:]
        prev_matrix = matrix[:-1]
        sims = cosine_similarity(new_vec, prev_matrix)[0]
        return float(sims.max()) if len(sims) else 0.0
    except Exception:
        return 0.0


def validate_reply(
    response: str,
    session_replies: list[str],
    tweet_text: str | None = None,
) -> tuple[bool, str | None]:
    """
    Vérifier (Section 9):
    - Moins de 3 réponses commençant par même mot
    - Moins de 3 emojis dans session (total)
    - Pas plus de 2 phrases
    - Similarité TF-IDF < 0.7 avec toutes réponses précédentes
    - Optional: reply must share at least one meaningful word with tweet (if tweet_text given)
    - Optional: blocklist of phrases that suggest invented facts
    Returns (valid, reason). valid=True means OK to post.
    """
    if not (response or "").strip():
        return False, "empty"

    if _reply_has_invented_phrase(response):
        return False, "invented_phrase_blocklist"

    if tweet_text and (tweet_text or "").strip():
        if _reply_adds_entity_not_in_tweet(response, tweet_text):
            return False, "reply_adds_entity_not_in_tweet"

    if tweet_text and (tweet_text or "").strip() and len((response or "").strip()) > 40:
        if not _meaningful_word_overlap(tweet_text, response):
            return False, "no_relevance_to_tweet"

    first = _first_word(response)
    if first:
        same_first = sum(1 for r in session_replies if _first_word(r) == first)
        if same_first >= MAX_RESPONSES_SAME_FIRST_WORD:
            return False, "too_many_same_first_word"

    total_emojis = _count_emojis(response) + sum(_count_emojis(r) for r in session_replies)
    if total_emojis >= MAX_EMOJIS_IN_SESSION:
        return False, "too_many_emojis"

    if _count_sentences(response) > MAX_SENTENCES:
        return False, "too_many_sentences"

    if session_replies and _tfidf_similarity(response, session_replies) >= TFIDF_SIMILARITY_THRESHOLD:
        return False, "too_similar"

    return True, None

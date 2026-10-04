"""
NBAVision Engine — Scoring des tweets (Spec Section 6).
"""
import math
from filter_tweets import minutes_since_post
from config import TOP_N_SCORED


def compute_score(tweet: dict) -> float:
    """
    engagement_velocity = (likes + 2*replies + 2*retweets) / max(minutes, 1)
    freshness_score = 1 - (minutes / 180)   (matched to MAX_MINUTES_SINCE_POST)
    text_quality = log(len(text) + 1) — prefer meatier tweets
    score = 0.6 * velocity + 0.25 * freshness + 0.15 * text_quality
    """
    minutes = minutes_since_post(tweet.get("timestamp") or "")
    likes = tweet.get("likes") or 0
    replies = tweet.get("replies") or 0
    retweets = tweet.get("retweets") or 0
    text = (tweet.get("text") or "").strip()

    engagement_velocity = (likes + 2 * replies + 2 * retweets) / max(minutes, 1)
    freshness_score = max(0.0, min(1.0, 1.0 - (minutes / 180.0)))
    text_quality = math.log(max(len(text), 1) + 1)

    score = (
        0.7 * engagement_velocity
        + 0.25 * freshness_score
        + 0.05 * text_quality
    )
    return score


def rank_and_top(tweets: list[dict], top_n: int = TOP_N_SCORED) -> list[dict]:
    """
    Compute score for each tweet, sort descending, return top N.
    """
    for t in tweets:
        t["_score"] = compute_score(t)
    sorted_tweets = sorted(tweets, key=lambda x: x["_score"], reverse=True)
    return sorted_tweets[:top_n]

"""The request guard decides, before anything leaves the browser, whether a post matches the preview."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))
pytest.importorskip("playwright")


class FakeRoute:
    def __init__(self, variables):
        self.request = type("R", (), {"post_data_json": {"variables": variables}})()
        self.outcome = None

    def abort(self):
        self.outcome = "aborted"

    def continue_(self):
        self.outcome = "sent"


def _run(guard, variables):
    r = FakeRoute(variables)
    guard.handle(r)
    return r.outcome, guard.refused


def test_matching_opener_and_reply_go_out():
    from publisher import _Guard
    g = _Guard()
    g.arm("🏀 Pick: Rockets\n\nThe case 🧵", None, True)
    assert _run(g, {"tweet_text": "🏀 Pick: Rockets\nThe case 🧵",
                    "media": {"media_entities": [{"media_id": "1", "tagged_users": []}]}})[0] == "sent"
    g.arm("⚠️ THE RISK", "111", False)
    assert _run(g, {"tweet_text": "⚠️ THE RISK", "reply": {"in_reply_to_tweet_id": "111"}})[0] == "sent"


def test_wrong_reply_target_missing_image_or_changed_text_are_aborted():
    from publisher import _Guard
    g = _Guard()
    g.arm("tweet 3", "222", False)
    out, why = _run(g, {"tweet_text": "tweet 3", "reply": {"in_reply_to_tweet_id": "111"}})   # forked thread
    assert out == "aborted" and "111" in why
    g.arm("tweet 2", "111", True)
    assert _run(g, {"tweet_text": "tweet 2", "reply": {"in_reply_to_tweet_id": "111"}})[0] == "aborted"
    g.arm("Lakers #NBA\nPick: LAL", None, False)
    assert _run(g, {"tweet_text": "Lakers #NBA Pick: LAL"})[0] == "aborted"            # lost line break
    # a second request for the same armed tweet (double click) never goes out
    g.arm("x", None, False)
    assert _run(g, {"tweet_text": "x"})[0] == "sent"
    assert _run(g, {"tweet_text": "x"})[0] == "aborted"

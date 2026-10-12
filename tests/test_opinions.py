import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))
import opinions  # noqa: E402
from config import TZ  # noqa: E402


def test_due_window_quota_and_gap():
    evening = datetime(2026, 10, 12, 21, 0, tzinfo=TZ)
    assert opinions.due(evening, {"count": 0, "topics": []})
    assert not opinions.due(evening.replace(hour=10), {"count": 0, "topics": []})
    assert not opinions.due(evening, {"count": 2, "topics": []})
    assert not opinions.due(evening, {"count": 1, "topics": [], "last": (evening - timedelta(hours=2)).isoformat()})
    assert opinions.due(evening, {"count": 1, "topics": [], "last": (evening - timedelta(hours=5)).isoformat()})


def test_prompts_build():
    from llm_client import OPINION_PROMPT, SHORT_PROMPT
    assert "short style" in SHORT_PROMPT and "TRUTH RULES" in SHORT_PROMPT
    assert "A GOOD POST" in OPINION_PROMPT and "@handle" in OPINION_PROMPT

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

NOW = datetime(2026, 10, 21, 21, 10, tzinfo=timezone.utc)


def _game(gid, hours, p_home, home_odds=1.4, away_odds=3.0, source="market", published=False):
    return {"id": gid, "start_utc": (NOW + timedelta(hours=hours)).isoformat(), "probability_source": source,
            "predicted_home_prob": p_home, "predicted_away_prob": 1 - p_home,
            "home_odds": home_odds, "away_odds": away_odds, "published": published}


def test_plan_picks_tonights_best_market_games():
    import autopilot as ap
    data = {"games": [
        _game("A", 2, 0.70), _game("B", 3, 0.62), _game("C", 2.5, 0.80, home_odds=1.10),   # C too short a price
        _game("D", 0.5, 0.75),                         # tips off in 30 min: too late
        _game("E", 20, 0.78),                          # tomorrow night
        _game("F", 2, 0.66, source="model"),           # no market odds
        _game("G", 2, 0.90, published=True),
    ], "specials": [{"id": "weekly-1", "type": "weekly", "thread": [{"text": "x"}], "published": False}]}
    cfg = {**ap.DEFAULTS, "enabled": True, "threads_per_day": 2}
    assert ap.plan(data, cfg, NOW, set()) == ["weekly-1", "A", "B"]
    assert ap.plan(data, cfg, NOW, {"A", "weekly-1"}) == ["B"]                # never queued twice


def test_off_by_default_and_queue_entries(tmp_path, monkeypatch):
    import autopilot as ap
    assert ap.DEFAULTS["enabled"] is False
    monkeypatch.setattr(ap, "QUEUE", tmp_path / "publish_queue.json")
    assert ap.add_to_queue(["A", "B"], NOW) == 2
    assert ap.add_to_queue(["A"], NOW) == 0
    q = json.loads((tmp_path / "publish_queue.json").read_text(encoding="utf-8"))
    assert [r["game_id"] for r in q["requests"]] == ["A", "B"]
    assert all(r["dry"] is False and r["texts"] is None for r in q["requests"])

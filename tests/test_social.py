import sqlite3

import pytest

from src.social import angles as A
from src.social.thread import build_thread
from src.social.weekly import build_weekly, last_week


def _features(**over):
    f = {
        "start_utc": "2026-10-22T00:30:00+00:00", "probability_source": "market",
        "model_home_prob": 0.66, "market_home_prob": 0.72,
        "home_elo": 1600, "away_elo": 1450, "home_last10_net_rating": 6.0, "away_last10_net_rating": -4.0,
        "home_last10_win_pct": 0.7, "away_last10_win_pct": 0.3,
        "home_last10_offensive_rating": 118.0, "away_last10_offensive_rating": 110.0,
        "home_last10_defensive_rating": 109.0, "away_last10_defensive_rating": 114.0,
        "home_last10_fg3_pct": 0.38, "away_last10_fg3_pct": 0.34,
        "home_rest_days": 2, "away_rest_days": 0, "home_back_to_back": 0, "away_back_to_back": 1,
        "home_games_last5d": 2, "away_games_last5d": 3, "home_streak": 4, "away_streak": -3,
        "home_team_home_win_pct": 0.7, "away_team_road_win_pct": 0.3,
        "home_games_played": 8, "away_games_played": 8, "home_season_win_pct": 0.75, "away_season_win_pct": 0.25,
        "home_season_diff": 7.5, "away_season_diff": -6.0,
        "home_key_out": [], "away_key_out": ["Star Player"],
    }
    f.update(over)
    return f


def _pred(ph=0.72, **over):
    return {"home_team": "Houston Rockets", "away_team": "Dallas Mavericks", "predicted_home_prob": ph,
            "home_odds": 1.32, "away_odds": 3.4, "features": _features(**over)}


def test_thread_shape_and_lengths():
    th = build_thread(_pred(), "Our probabilities hit 68.6% of winners over 4 seasons of backtests.")
    tw = th["tweets"]
    assert len(tw) == 6
    assert all(len(t["text"]) <= 280 for t in tw)
    assert tw[0]["card"]["type"] == "match" and tw[0]["card"]["pick_name"] == "Rockets"
    assert "Houston Rockets (72%)" in tw[0]["text"]
    assert tw[4]["text"].startswith("⚠️ THE RISK")


def test_reasons_always_favour_the_pick():
    # away pick: the Mavericks are favoured; every reason tweet must argue for them
    th = build_thread(_pred(ph=0.30, model_home_prob=0.35, market_home_prob=0.30,
                            home_elo=1400, away_elo=1650, home_last10_net_rating=-5.0, away_last10_net_rating=7.0),
                      "record")
    assert th["pick"] == "DAL"
    pick = A.Side(_features(), "away", "DAL", "Mavericks")
    opp = A.Side(_features(), "home", "HOU", "Rockets")
    for t in th["tweets"][1:4]:
        assert "Rockets" not in t["text"].split("\n")[0]  # headline never praises the opponent
    assert pick and opp


def test_angle_signs():
    f = _features()
    P = A.Side(f, "home", "HOU", "Rockets")
    O = A.Side(f, "away", "DAL", "Mavericks")
    got = {a.key: a.strength for a in A.build(f, P, O, pick_home=True)}
    assert got["elo"] > 0 and got["form"] > 0 and got["defense"] > 0
    assert got["rest"] > 0 and got["injuries"] > 0 and got["streak"] > 0


def test_weekly_recap(tmp_path):
    db = tmp_path / "p.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE predictions (game_date TEXT, home_team TEXT, away_team TEXT, "
              "predicted_home_prob REAL, correct INT)")
    rows = [("2026-10-26", "Boston Celtics", "New York Knicks", 0.7, 1),
            ("2026-10-27", "Boston Celtics", "Miami Heat", 0.8, 1),
            ("2026-10-28", "Utah Jazz", "Denver Nuggets", 0.3, 1),
            ("2026-10-29", "Detroit Pistons", "Chicago Bulls", 0.55, 0),
            ("2026-10-30", "LA Clippers", "Phoenix Suns", 0.62, 1),
            ("2026-11-01", "Orlando Magic", "Atlanta Hawks", 0.51, 1)]
    c.executemany("INSERT INTO predictions VALUES (?,?,?,?,?)", rows)
    c.commit()
    from datetime import date
    today = date(2026, 11, 2)  # Monday
    assert last_week(today)[0].isoformat() == "2026-10-26"
    w = build_weekly(str(db), today, "2026-10-01")
    assert w["id"] == "weekly-2026-10-26"
    assert "5-1 on our picks" in w["thread"][0]["text"]
    assert w["thread"][0]["card"]["type"] == "recap"
    assert all(len(t["text"]) <= 280 for t in w["thread"])
    assert build_weekly(str(db), date(2026, 10, 26), "2026-10-01") is None  # too few picks


def test_text_overrides(monkeypatch):
    import scripts.publish_single_thread as pub
    posts = [{"text": "a", "card": {"type": "match"}}, {"text": "b", "card": None}]
    monkeypatch.setenv("THREAD_TEXTS_JSON", '["edited", ""]')
    assert pub.apply_text_overrides(posts) == [{"text": "edited", "card": {"type": "match"}}]
    monkeypatch.setenv("THREAD_TEXTS_JSON", '["x" ]')
    with pytest.raises(ValueError):
        pub.apply_text_overrides(posts)

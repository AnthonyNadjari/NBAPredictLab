import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.engine import espn, features, history, model, pipeline
from src.engine.teams import full_name, to_code


def _hist(n_days=30, seed=0):
    """Synthetic league: 4 teams, one game a day, deterministic scores."""
    rng = np.random.default_rng(seed)
    teams = ["BOS", "NYK", "LAL", "GSW"]
    rows = []
    for i in range(n_days):
        h, a = rng.choice(teams, 2, replace=False)
        hp, ap = int(rng.integers(95, 125)), int(rng.integers(95, 125))
        if hp == ap:
            hp += 1
        row = {"game_id": str(i),
               "game_date": (pd.Timestamp("2025-11-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
               "season": "2025-26", "season_type": "Regular Season", "home": h, "away": a,
               "home_pts": hp, "away_pts": ap, "source": "test"}
        for side in ("home", "away"):
            for k in history.STATS:
                row[f"{side}_{k}"] = int(rng.integers(5, 90))
        rows.append(row)
    return pd.DataFrame(rows, columns=history.COLUMNS)


def test_features_do_not_use_future_results():
    h = _hist()
    base = features.build(h)
    changed = h.copy()
    changed.loc[20:, "home_pts"] = changed.loc[20:, "home_pts"] + 50
    after = features.build(changed)
    cols = features.MODEL_FEATURES + ["home_elo", "away_elo", "home_last10_net_rating", "home_streak"]
    pd.testing.assert_frame_equal(base.loc[:20, cols], after.loc[:20, cols])
    assert not base.loc[21:, "home_elo"].equals(after.loc[21:, "home_elo"])


def test_upcoming_game_gets_features_without_result():
    h = _hist()
    up = pd.DataFrame([{"game_id": "x", "game_date": "2025-12-15", "season": "2025-26",
                        "season_type": "Regular Season", "home": "BOS", "away": "LAL", "source": "upcoming"}])
    g = features.build(pd.concat([h, up], ignore_index=True))
    row = g[g.source == "upcoming"].iloc[0]
    assert np.isnan(row.home_win)
    assert row[features.MODEL_FEATURES].notna().all()
    assert 0 < row.elo_win_prob < 1


def test_model_roundtrip_and_market_priority(tmp_path):
    g = features.build(_hist(200))
    m = model.fit(g.dropna(subset=["home_win"]))
    model.save(m, tmp_path / "m.json")
    m2 = model.load(tmp_path / "m.json")
    p = model.predict_proba(m2, g.tail(5))
    assert ((p > 0) & (p < 1)).all()
    assert model.final_probability(0.7, 0.4) == (0.4, "market")
    assert model.final_probability(0.7, None) == (0.7, "model")


def test_team_codes():
    assert to_code("GS") == "GSW" and to_code("UTAH") == "UTA"
    assert to_code("LA Clippers") == "LAC" and to_code("Los Angeles Clippers") == "LAC"
    assert full_name("NYK") == "New York Knicks"
    with pytest.raises(KeyError):
        to_code("Seattle SuperSonics")


def _db(path):
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_date TEXT,
        game_date TEXT, home_team TEXT, away_team TEXT, predicted_winner TEXT, predicted_home_prob REAL,
        predicted_away_prob REAL, confidence REAL, actual_winner TEXT, actual_home_score INT,
        actual_away_score INT, correct INT, created_at TEXT, home_odds REAL, away_odds REAL,
        prediction_error REAL, calibration_error REAL, features_json TEXT)""")
    return conn


def test_resolve_predictions_handles_paris_dates(tmp_path):
    db = tmp_path / "p.db"
    conn = _db(db)
    conn.execute("INSERT INTO predictions (game_date, home_team, away_team, predicted_home_prob) "
                 "VALUES ('2025-11-02', 'Boston Celtics', 'New York Knicks', 0.6)")
    # stored with the Paris date (US date + 1), as some older rows were
    conn.execute("INSERT INTO predictions (game_date, home_team, away_team, predicted_home_prob) "
                 "VALUES ('2025-11-03', 'LA Clippers', 'Golden State Warriors', 0.3)")
    conn.commit()
    h = pd.DataFrame([
        {"game_date": "2025-11-02", "home": "BOS", "away": "NYK", "home_pts": 110, "away_pts": 100},
        {"game_date": "2025-11-02", "home": "LAC", "away": "GSW", "home_pts": 99, "away_pts": 101},
    ], columns=history.COLUMNS)
    assert pipeline.resolve_predictions(str(db), h) == 2
    rows = conn.execute("SELECT actual_winner, correct FROM predictions ORDER BY id").fetchall()
    assert rows == [("Boston Celtics", 1), ("Golden State Warriors", 1)]
    assert pipeline.track_record(str(db))["accuracy"] == 1.0


def test_save_predictions_never_overwrites_resolved(tmp_path):
    db = tmp_path / "p.db"
    conn = _db(db)
    conn.execute("INSERT INTO predictions (game_date, home_team, away_team, predicted_home_prob, actual_winner) "
                 "VALUES ('2025-11-02', 'Boston Celtics', 'New York Knicks', 0.6, 'Boston Celtics')")
    conn.commit()
    pred = {"home_team": "BOS", "away_team": "NYK", "predicted_winner": "NYK", "home_win_probability": 0.4,
            "away_win_probability": 0.6, "confidence": 0.6, "home_odds": None, "away_odds": None,
            "features": {}, "game_info": {"game_date": "2025-11-02"}}
    assert pipeline.save_predictions(str(db), [pred]) == 0
    pred["game_info"]["game_date"] = "2025-11-05"
    assert pipeline.save_predictions(str(db), [pred]) == 1
    assert pipeline.save_predictions(str(db), [pred]) == 1  # replaces its own pending row
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2


def test_exporter_keeps_published_flags(tmp_path):
    from src.daily_games_exporter import DailyGamesExporter
    out = tmp_path / "pending.json"
    gid = "New_York_Knicks_vs_Boston_Celtics_2025-11-05"
    out.write_text(json.dumps({"games": [{"id": gid, "published": True, "published_at": "t0"}]}))
    games = [{"id": gid, "published": False}, {"id": "other", "published": False}]
    res = DailyGamesExporter._carry_published(games, str(out))
    assert res[0]["published"] is True and res[0]["published_at"] == "t0"
    assert res[1]["published"] is False


SCOREBOARD = {"events": [
    {"id": "1", "date": "2026-10-21T23:30Z", "season": {"type": 2},
     "competitions": [{"status": {"type": {"completed": False, "state": "pre", "name": "STATUS_SCHEDULED"}},
                       "competitors": [
                           {"homeAway": "home", "team": {"abbreviation": "NY"}, "score": "0"},
                           {"homeAway": "away", "team": {"abbreviation": "GS"}, "score": "0"}]}]},
    {"id": "2", "date": "2026-10-10T23:30Z", "season": {"type": 1},  # preseason: ignored
     "competitions": [{"status": {"type": {}}, "competitors": []}]},
]}


def test_espn_scoreboard_parsing(monkeypatch):
    monkeypatch.setattr(espn, "_get", lambda url, **k: SCOREBOARD)
    games = espn.scoreboard(pd.Timestamp("2026-10-21").date())
    assert len(games) == 1
    g = games[0]
    assert (g["home"], g["away"], g["game_date"], g["season"], g["state"]) == \
        ("NYK", "GSW", "2026-10-21", "2026-27", "pre")


def test_espn_down_is_an_error(monkeypatch):
    monkeypatch.setattr(espn, "_get", lambda url, **k: None)
    with pytest.raises(RuntimeError):
        espn.scoreboard(pd.Timestamp("2026-10-21").date())


def test_espn_odds_consensus(monkeypatch):
    items = {"items": [
        {"provider": {"name": "A"}, "spread": -3.5,
         "homeTeamOdds": {"moneyLine": -150}, "awayTeamOdds": {"moneyLine": 130}},
        {"provider": {"name": "B"}, "spread": -4.0,
         "homeTeamOdds": {"moneyLine": -160}, "awayTeamOdds": {"moneyLine": 140}},
        {"provider": {"name": "C"}, "homeTeamOdds": {}, "awayTeamOdds": {}},
    ]}
    monkeypatch.setattr(espn, "_get", lambda url, **k: items)
    o = espn.odds("1")
    assert len(o["books"]) == 2
    assert 0.57 < o["home_prob"] < 0.60
    assert o["home_odds"] == pytest.approx(1.65, abs=0.01)


def test_published_predictions_are_frozen(tmp_path):
    db = tmp_path / "p.db"
    conn = _db(db)
    conn.close()
    pred = {"home_team": "BOS", "away_team": "NYK", "predicted_winner": "BOS", "home_win_probability": 0.62,
            "away_win_probability": 0.38, "confidence": 0.62, "home_odds": 1.6, "away_odds": 2.4,
            "features": {}, "game_info": {"game_date": "2026-10-25"}}
    assert pipeline.save_predictions(str(db), [pred]) == 1
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps({"games": [{"date": "2026-10-25", "home_team": "Boston Celtics",
                                              "away_team": "New York Knicks", "published": True}]}))
    frozen = pipeline.published_game_keys(pending)
    flipped = {**pred, "predicted_winner": "NYK", "home_win_probability": 0.45, "away_win_probability": 0.55}
    assert pipeline.save_predictions(str(db), [flipped], frozen=frozen) == 0
    row = sqlite3.connect(db).execute("SELECT predicted_home_prob FROM predictions").fetchone()
    assert row[0] == 0.62


def test_final_probability_blends_own_model_with_market_only_in_full_mode():
    from src.engine import model
    p, src = model.final_probability(0.55, 0.70, own_prob=0.60, own_mode="full")
    assert src == "blend" and 0.68 < p < 0.70                     # mostly the market, pulled toward ours
    assert model.final_probability(0.55, 0.70, 0.60, "team_only") == (0.70, "market")
    assert model.final_probability(0.55, None, 0.60, "team_only") == (0.60, "own")
    assert model.final_probability(0.55, None, None) == (0.55, "model")

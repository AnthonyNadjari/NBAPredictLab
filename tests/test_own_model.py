"""Own model (src/engine/own_model.py): research parity, no look-ahead, team-only fallback,
ESPN player box-score parser, player store, pipeline wiring."""
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine import espn, history, model, own_model, pipeline, player_store

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROD_PARAMS = own_model.load_params()
# the synthetic league is short (~35 games per team at the test day): lower the history depth
# required for 'full' mode (test_thin_player_history_is_team_only checks the production value)
PARAMS = {**PROD_PARAMS, "mode": {**PROD_PARAMS["mode"], "min_player_games": 20}}
TEAMS = ["BOS", "NYK", "LAL", "GSW", "DEN", "MIA", "CHI", "PHX"]


# ---------------------------------------------------------------------------- synthetic league
def _league(n_days=75, start="2025-10-21", season="2025-26", seed=3, n_players=11):
    """8 teams, 4 games on most days, consistent player and team box scores."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, rng.normal(0, 5, len(TEAMS))))
    games, lines = [], []
    gid = 0
    for day in range(n_days):
        if day % 4 == 3:
            continue
        d = (pd.Timestamp(start) + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        order = list(rng.permutation(TEAMS))
        for h, a in zip(order[::2], order[1::2]):
            gid += 1
            box = {}
            for team in (h, a):
                plays = [i for i in range(n_players) if rng.random() < 0.88][:10]
                w = rng.gamma(2.0, 1.0, len(plays)) * np.linspace(2, 1, len(plays))
                mins = np.floor(240 * w / w.sum()).astype(int)
                mins[0] += 240 - mins.sum()
                rows = []
                for i, m in zip(plays, mins):
                    fga = int(rng.poisson(m * 0.35 + 0.5))
                    fgm = int(rng.binomial(fga, 0.47 + strength[team] / 200))
                    fg3a = int(rng.binomial(fga, 0.4))
                    fg3m = int(rng.binomial(min(fg3a, fgm), 0.6))
                    fta = int(rng.poisson(m * 0.08))
                    ftm = int(rng.binomial(fta, 0.78))
                    rows.append({"player_id": f"{team}{i}", "player": f"{team} player {i}", "min": int(m),
                                 "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta,
                                 "pts": 2 * fgm + fg3m + ftm, "oreb": int(rng.poisson(m * 0.03)),
                                 "dreb": int(rng.poisson(m * 0.12)), "ast": int(rng.poisson(m * 0.08)),
                                 "stl": int(rng.poisson(m * 0.03)), "blk": int(rng.poisson(m * 0.02)),
                                 "tov": int(rng.poisson(m * 0.05)), "pf": int(rng.poisson(m * 0.07))})
                box[team] = rows
            if sum(r["pts"] for r in box[h]) == sum(r["pts"] for r in box[a]):
                box[h][0]["pts"] += 1
                box[h][0]["ftm"] += 1
                box[h][0]["fta"] += 1
            tot = {t: {k: sum(r[k] for r in box[t]) for k in box[t][0] if k not in ("player_id", "player")}
                   for t in (h, a)}
            margin = tot[h]["pts"] - tot[a]["pts"]
            row = {"game_id": f"g{gid}", "game_date": d, "season": season, "season_type": "Regular Season",
                   "home": h, "away": a, "home_pts": tot[h]["pts"], "away_pts": tot[a]["pts"], "source": "test"}
            for side, t in (("home", h), ("away", a)):
                for k in history.STATS:
                    row[f"{side}_{k}"] = tot[t]["oreb"] + tot[t]["dreb"] if k == "reb" else tot[t][k]
            games.append(row)
            for t, o, home in ((h, a, 1), (a, h, 0)):
                sign = 1 if home else -1
                for r in box[t]:
                    lines.append({**r, "game_id": f"espn_{gid}", "game_date": d, "season": season,
                                  "season_type": "Regular Season", "team": t, "opp": o, "home": home,
                                  "starter": 0, "reb": r["oreb"] + r["dreb"],
                                  "plus_minus": int(round(sign * margin * r["min"] / 48 + rng.normal(0, 4)))})
    return (pd.DataFrame(games, columns=history.COLUMNS),
            pd.DataFrame(lines, columns=player_store.COLUMNS))


def _upcoming(hist, day):
    return hist[hist.game_date == day][["game_id", "game_date", "season", "season_type", "home", "away"]].assign(
        game_id=lambda x: "up_" + x.game_id)


@pytest.fixture(scope="module")
def league():
    return _league()


@pytest.fixture(scope="module")
def day_and_base(league):
    hist, players = league
    day = sorted(hist.game_date.unique())[45]
    up = _upcoming(hist, day)
    base = own_model.compute(hist[hist.game_date < day], players[players.game_date < day], PARAMS, upcoming=up)
    return day, up, base


COMPONENTS = ["p_elo_plus", "p_elo_team", "p_margin_avail", "p_margin_team", "p_player", "own_home_prob"]


# ---------------------------------------------------------------------------- no look-ahead
def test_base_prediction_is_full_mode(day_and_base):
    _, up, base = day_and_base
    assert len(base) == len(up) == 4
    assert (base.own_model_mode == own_model.FULL).all()
    assert base[COMPONENTS].notna().all().all()
    assert ((base.own_home_prob > 0.05) & (base.own_home_prob < 0.95)).all()
    z = np.mean([own_model._logit(base[c]) for c in ("p_elo_plus", "p_margin_avail", "p_player")], 0)
    np.testing.assert_allclose(base.own_home_prob, 1 / (1 + np.exp(-z)), atol=1e-12)


def test_future_games_do_not_change_a_prediction(league, day_and_base):
    hist, players = league
    day, up, base = day_and_base
    # every game after the day (scores + player lines) added to the history
    later = own_model.compute(hist[hist.game_date != day], players[players.game_date != day], PARAMS,
                              upcoming=up)
    # ...and even the day's own results and player lines
    same_day = own_model.compute(hist, players, PARAMS, upcoming=up)
    for other in (later, same_day):
        pd.testing.assert_frame_equal(base[["home", "away", "own_model_mode"]], other[["home", "away", "own_model_mode"]])
        np.testing.assert_allclose(base[COMPONENTS].to_numpy(float), other[COMPONENTS].to_numpy(float), atol=1e-7)


def test_changing_future_results_changes_later_predictions_only(league):
    """Sanity of the leak test: the history is really used (a past change moves the number)."""
    hist, players = league
    day = sorted(hist.game_date.unique())[45]
    up = _upcoming(hist, day)
    h2 = hist.copy()
    before = h2.game_date < day
    h2.loc[before, "home_pts"] = h2.loc[before, "home_pts"] + 15
    a = own_model.compute(hist[hist.game_date < day], players[players.game_date < day], PARAMS, upcoming=up)
    b = own_model.compute(h2[h2.game_date < day], players[players.game_date < day], PARAMS, upcoming=up)
    assert np.abs(a.p_elo_team - b.p_elo_team).max() > 0.01


# ---------------------------------------------------------------------------- fallback
def test_team_only_fallback_when_a_team_lacks_recent_player_lines(league, day_and_base):
    hist, players = league
    day, up, base = day_and_base
    past_h, past_p = hist[hist.game_date < day], players[players.game_date < day]
    team = up.home.iloc[0]
    last = past_h[(past_h.home == team) | (past_h.away == team)].game_date.max()
    stale = past_p[~((past_p.team == team) & (past_p.game_date == last))]
    r = own_model.compute(past_h, stale, PARAMS, upcoming=up)
    hit = (r.home == team) | (r.away == team)
    assert (r.own_model_mode[hit] == own_model.TEAM_ONLY).all()
    assert (r.own_model_mode[~hit] == own_model.FULL).all()
    z = (own_model._logit(r.p_elo_team) + own_model._logit(r.p_margin_team)) / 2
    np.testing.assert_allclose(r.own_home_prob[hit], (1 / (1 + np.exp(-z)))[hit], atol=1e-12)
    # the team-only parts never depend on player data
    np.testing.assert_allclose(r.p_elo_team, base.p_elo_team, atol=1e-12)
    np.testing.assert_allclose(r.p_margin_team, base.p_margin_team, atol=1e-12)


def test_thin_player_history_is_team_only(league, day_and_base):
    """A store with only a few weeks of player lines (daily run, backfill not run) must not
    switch to 'full': its player ratings are mostly prior (worse than team-only on 2025-26)."""
    hist, players = league
    day, up, base = day_and_base
    past_h, past_p = hist[hist.game_date < day], players[players.game_date < day]
    assert PROD_PARAMS["mode"]["min_player_games"] >= 82
    r = own_model.compute(past_h, past_p, PROD_PARAMS, upcoming=up)
    assert (r.own_model_mode == own_model.TEAM_ONLY).all()
    recent = past_p[past_p.game_date >= sorted(past_h.game_date.unique())[-8]]
    r = own_model.compute(past_h, recent, PARAMS, upcoming=up)
    assert (r.own_model_mode == own_model.TEAM_ONLY).all()
    np.testing.assert_allclose(r.p_elo_team, base.p_elo_team, atol=1e-12)


def test_team_only_without_any_player_data(league, day_and_base):
    hist, players = league
    day, up, base = day_and_base
    r = own_model.compute(hist[hist.game_date < day], players.iloc[:0], PARAMS, upcoming=up)
    assert (r.own_model_mode == own_model.TEAM_ONLY).all()
    assert r.p_player.isna().all()
    np.testing.assert_allclose(r.p_elo_team, base.p_elo_team, atol=1e-12)
    out = own_model.predict([{"event_id": "1", "game_date": day, "season": "2025-26",
                              "season_type": "Regular Season", "home": up.home.iloc[0], "away": up.away.iloc[0]}],
                            hist[hist.game_date < day], players=players.iloc[:0], params=PARAMS)
    o = out[(day, up.home.iloc[0], up.away.iloc[0])]
    assert o["own_model_mode"] == "team_only"
    assert o["own_home_prob"] + o["own_away_prob"] == pytest.approx(1.0)


def test_all_star_games_are_ignored(league, day_and_base):
    """ESPN lists All-Star games as regular season (teams STARS / WORLD / STRIPES)."""
    hist, players = league
    day, up, base = day_and_base
    past_h, past_p = hist[hist.game_date < day], players[players.game_date < day]
    d0 = past_h.game_date.max()
    star = past_h.iloc[[0]].assign(game_id="asg", game_date=d0, home="STARS", away="WORLD")
    lines = past_p[past_p.game_date == d0].head(10).assign(game_id="espn_asg", team="STARS", opp="WORLD")
    r = own_model.compute(pd.concat([past_h, star]), pd.concat([past_p, lines]), PARAMS, upcoming=up)
    np.testing.assert_allclose(r.own_home_prob, base.own_home_prob, atol=1e-12)
    games = [{"event_id": str(i), "game_date": day, "season": "2025-26", "season_type": "Regular Season",
              "home": h, "away": a} for i, (h, a) in enumerate(zip(up.home, up.away))]
    out = own_model.predict(games + [{**games[0], "event_id": "x", "home": "STRIPES", "away": "STARS"}],
                            past_h, players=past_p, params=PARAMS)
    assert len(out) == len(up) and all(k[1] in TEAMS for k in out)


def test_published_probability_market_first_then_own():
    assert model.final_probability(0.7, 0.4, 0.55) == (0.4, "market")
    assert model.final_probability(0.7, None, 0.55) == (0.55, "own")
    assert model.final_probability(0.7, None, None) == (0.7, "model")


def test_predict_date_stores_own_next_to_market(monkeypatch, league):
    hist, players = league
    day = pd.Timestamp(sorted(hist.game_date.unique())[45])
    games = [{"event_id": "1", "game_date": day.strftime("%Y-%m-%d"), "start_utc": "2025-12-10T00:00:00+00:00",
              "season": "2025-26", "season_type": "Regular Season", "home": "BOS", "away": "NYK",
              "home_score": None, "away_score": None, "completed": False, "state": "pre", "status": "x"},
             {**{"event_id": "2", "game_date": day.strftime("%Y-%m-%d"), "start_utc": "2025-12-10T00:00:00+00:00",
                 "season": "2025-26", "season_type": "Regular Season", "home": "LAL", "away": "GSW",
                 "home_score": None, "away_score": None, "completed": False, "state": "pre", "status": "x"}}]
    monkeypatch.setattr(espn, "scoreboard", lambda d, **k: games if d == day.date() else [])
    monkeypatch.setattr(espn, "odds", lambda eid: {"home_prob": 0.61, "home_odds": 1.6, "away_odds": 2.4,
                                                   "spread": -3.5, "books": ["A"]} if eid == "1" else None)
    monkeypatch.setattr(espn, "game_context", lambda eid: None)
    monkeypatch.setattr(own_model, "predict", lambda g, h, **k: {
        (x["game_date"], x["home"], x["away"]): {"own_home_prob": 0.52, "own_away_prob": 0.48,
                                                 "own_model_mode": "full", "own_components": {}} for x in g})
    out = {p["home_team"]: p for p in pipeline.predict_date(day.date(), hist[hist.game_date < day.strftime("%Y-%m-%d")])}
    bos, lal = out["BOS"], out["LAL"]
    # market price + own model in full mode: the published number is the blend (mostly the market)
    assert 0.59 < bos["home_win_probability"] < 0.61 and bos["features"]["probability_source"] == "blend"
    assert bos["features"]["own_home_prob"] == 0.52 and bos["features"]["market_home_prob"] == 0.61
    assert lal["home_win_probability"] == 0.52 and lal["features"]["probability_source"] == "own"
    assert lal["features"]["own_model_mode"] == "full"


def test_own_model_failure_falls_back_to_the_logit(monkeypatch, league):
    hist, _ = league
    monkeypatch.setattr(own_model, "predict", lambda *a, **k: 1 / 0)
    assert pipeline.own_probabilities([{"x": 1}], hist) == {}


# ---------------------------------------------------------------------------- ESPN player box score
def test_parse_player_box_from_saved_summary():
    js = json.loads((FIXTURES / "espn_summary_401810791.json").read_text(encoding="utf-8"))
    rows = espn.parse_player_box(js)
    d = pd.DataFrame(rows)
    assert len(d) == 20 and set(d.team) == {"PHI", "MEM"}
    assert (d.game_id == "espn_401810791").all() and (d.game_date == "2026-03-10").all()
    assert (d.season == "2025-26").all() and (d.season_type == "Regular Season").all()
    assert set(zip(d.team, d.home, d.opp)) == {("PHI", 1, "MEM"), ("MEM", 0, "PHI")}
    tot = d.groupby("team")[["min", "pts", "tov", "plus_minus"]].sum()
    assert tot.loc["PHI"].tolist() == [240, 139, 10, 50] and tot.loc["MEM"].tolist() == [241, 129, 20, -50]
    p = d[d.player == "Olivier-Maxence Prosper"].iloc[0]
    assert (p.player_id, p.starter, p["min"], p.pts, p.fgm, p.fga, p.fg3m, p.fg3a, p.ftm, p.fta, p.oreb, p.dreb,
            p.reb, p.ast, p.tov, p.stl, p.blk, p.pf, p.plus_minus) == \
        ("4595400", 1, 21, 11, 4, 6, 1, 1, 2, 3, 2, 4, 6, 0, 2, 2, 1, 2, 8)
    assert "Brandon Clarke" not in set(d.player)          # didNotPlay: no line
    assert d[espn.PLAYER_STATS].notna().all().all()
    assert set(player_store.COLUMNS) <= set(d.columns)


def test_parse_player_box_without_players():
    assert espn.parse_player_box({}) == []
    js = json.loads((FIXTURES / "espn_summary_401810791.json").read_text(encoding="utf-8"))
    js["boxscore"]["players"] = []
    assert espn.parse_player_box(js) == []


def test_player_store_skips_known_games(monkeypatch, tmp_path):
    js = json.loads((FIXTURES / "espn_summary_401810791.json").read_text(encoding="utf-8"))
    rows = espn.parse_player_box(js)
    sb = [{"event_id": "401810791", "game_date": "2026-03-10", "home": "PHI", "away": "MEM", "completed": True},
          {"event_id": "999", "game_date": "2026-03-10", "home": "BOS", "away": "NYK", "completed": False}]
    calls = []
    monkeypatch.setattr(espn, "scoreboard", lambda d, include_preseason=False: sb)
    monkeypatch.setattr(espn, "player_box", lambda eid: calls.append(eid) or rows)
    df = player_store.fetch_days(player_store.load(tmp_path / "none.csv"), [pd.Timestamp("2026-03-10").date()])
    assert len(df) == 20 and calls == ["401810791"]
    player_store.save(df, tmp_path / "p.csv")
    again = player_store.fetch_days(player_store.load(tmp_path / "p.csv"), [pd.Timestamp("2026-03-10").date()])
    assert len(again) == 20 and calls == ["401810791"]


# ---------------------------------------------------------------------------- parity with the research
RESEARCH = Path(os.environ.get("NBA_RESEARCH_DIR", ROOT / "research"))
H7 = RESEARCH / "h7_own_model"
_HAVE_RESEARCH = (any((RESEARCH / "data" / "h1_player_availability").glob("player_logs_*.csv"))
                  and all((H7 / f / "preds.csv").exists() for f in ("elo_plus", "margin_ratings", "player_impact")))


def research_players(rd: Path) -> pd.DataFrame:
    """The research's nba_api player logs in the player_games.csv format (nba_api ids)."""
    p = pd.concat([pd.read_csv(f) for f in sorted((rd / "data" / "h1_player_availability").glob("player_logs_*.csv"))],
                  ignore_index=True).drop_duplicates(["PLAYER_ID", "GAME_ID"])
    out = pd.DataFrame({"game_id": p.GAME_ID.astype(str), "game_date": p.GAME_DATE, "season": p.SEASON,
                        "season_type": p.SEASON_TYPE, "team": p.TEAM_ABBREVIATION, "opp": None,
                        "home": p.MATCHUP.str.contains(" vs. ").astype(int), "player_id": p.PLAYER_ID.astype(str),
                        "player": p.PLAYER_NAME, "starter": 0})
    for c in espn.PLAYER_STATS:
        out[c] = p[{"plus_minus": "PLUS_MINUS"}.get(c, c.upper())]
    return out


@pytest.mark.skipif(not _HAVE_RESEARCH, reason="research data not available (set NBA_RESEARCH_DIR)")
def test_parity_with_research_2024_25():
    """Same inputs as the research (its game table = data/games_history.csv up to 2025-26, its
    nba_api player logs, its 2024-25 walk-forward parameters): every 2024-25 game of the
    research evaluation set, each family and avg3 within 0.01 of research/h7_own_model."""
    params = json.loads((FIXTURES / "own_model_params_2024_25.json").read_text(encoding="utf-8"))
    hist = history.load()
    hist = hist[hist.season <= "2025-26"]
    res = own_model.compute(hist, research_players(RESEARCH), params, rows=lambda g: (g.season == "2024-25").to_numpy())
    res["GAME_ID"] = res.GAME_ID.astype(int)
    ref = None
    for fam, col in (("elo_plus", "r_elo"), ("margin_ratings", "r_margin"), ("player_impact", "r_player")):
        x = pd.read_csv(H7 / fam / "preds.csv")[["GAME_ID", "p"]].rename(columns={"p": col})
        ref = x if ref is None else ref.merge(x, on="GAME_ID")
    m = res.merge(ref, on="GAME_ID", how="inner")
    assert len(m) > 1200
    m["avg3"] = own_model._sigm(np.mean([own_model._logit(m[c]) for c in ("p_elo_plus", "p_margin_avail", "p_player")], 0))
    m["r_avg3"] = own_model._sigm(np.mean([own_model._logit(m[c]) for c in ("r_elo", "r_margin", "r_player")], 0))
    diffs = {k: float(np.abs(m[a] - m[b]).max()) for k, a, b in (
        ("elo_plus", "p_elo_plus", "r_elo"), ("margin_ratings", "p_margin_avail", "r_margin"),
        ("player_impact", "p_player", "r_player"), ("avg3", "avg3", "r_avg3"))}
    print("\nparity 2024-25, max |diff| over", len(m), "games:", {k: round(v, 5) for k, v in diffs.items()},
          "| mean |diff| avg3:", round(float(np.abs(m.avg3 - m.r_avg3).mean()), 6),
          "| own_model_mode:", m.own_model_mode.value_counts().to_dict())
    ens = H7 / "ensemble" / "preds.csv"
    if ens.exists():  # the ensemble's published preds.csv is stack3 (fitted weights), not avg3
        e = m.merge(pd.read_csv(ens)[["GAME_ID", "p"]], on="GAME_ID")
        print("vs ensemble/preds.csv (stack3): max |avg3 - stack3| =", round(float(np.abs(e.avg3 - e.p).max()), 5),
              "mean =", round(float(np.abs(e.avg3 - e.p).mean()), 5))
    for k, v in diffs.items():
        assert v < 0.01, (k, v)
    assert (m.own_model_mode == own_model.FULL).all()     # the published avg3 is the full mode
    np.testing.assert_allclose(m.own_home_prob, m.avg3, atol=1e-12)


def test_backfill_rescans_days_with_missing_games(monkeypatch):
    """Resumable backfill: a day where one game's summary failed is scanned again; complete
    days are skipped (except the last covered one)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("backfill_players", ROOT / "scripts" / "backfill_players.py")
    bf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bf)
    h = pd.DataFrame({"game_date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"], "season": "2024-25",
                      "home": ["BOS", "LAL", "BOS", "MIA"], "away": ["NYK", "GSW", "MIA", "STARS"]})
    monkeypatch.setattr(bf.history, "load", lambda: h)
    store = pd.DataFrame({"game_id": ["e1", "e1", "e3", "e4"], "season_type": "Regular Season",
                          "game_date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"]})
    todo = bf.days_to_scan(["2024-25"], store, date(2025, 9, 26))
    days = [d.isoformat() for d in todo]
    assert "2025-01-01" in days          # 1 of 2 games in the store
    assert "2025-01-02" not in days      # complete
    assert "2025-01-03" in days          # last covered day (All-Star 'STARS' game not counted)

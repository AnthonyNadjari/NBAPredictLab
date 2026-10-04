"""Shared data loading + evaluation protocol for H4 (market microstructure).

Game table: one row per completed ESPN event 2021-22..2025-26 with
  - outcome (home_win, margin) from ESPN final scores
  - main-book closing moneyline (raw American odds incl. vig) and de-vigged prob p_close
    (2021-22 has no explicit 'close' field: the post-game 'current' line is used, which for
     a finished game is the last pre-game line; same for <1% of later games)
  - main-book opening moneyline (2023-24..2025-26 only) -> p_open
  - spread / total of the main book
Book table: one row per (event, bookmaker) from the ESPN odds feed (live-odds feeds and
non-bookmaker sources removed), only 2021-22..2023-24 have more than one real book.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research/data"
CACHE = DATA / "h4_market_microstructure"
BOOKS_DIR = CACHE / "books"
RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)

ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TEST_SEASONS = SEASONS[1:]
# feeds that are not pre-game bookmaker prices
NON_BOOKS = ("live", "consensus", "teamrankings", "accuscore", "betegy")
# multiple-testing bar: number of log-loss variants / betting rules tested in this study
# (run.py prints the realised counts; Bonferroni CIs use a normal approximation with the
#  bootstrap standard error because 2000 resamples cannot resolve a 0.05/K tail directly)
BONF_K_LL = 37
BONF_K_BET = 70


# ----------------------------------------------------------------------------- odds helpers
def am_num(a):
    s = pd.Series(a, dtype="object").astype(str).str.strip().str.upper().replace({"EVEN": "100"})
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def am_to_p(a):
    a = am_num(a)
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def am_to_dec(a):
    a = am_num(a)
    return np.where(a < 0, 1 + 100 / -a, 1 + a / 100)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ----------------------------------------------------------------------------- game table
def load_games():
    f = CACHE / "games.csv"
    if f.exists():
        return pd.read_csv(f)
    o = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(str(DATA / "espn_*.csv")))],
                  ignore_index=True)
    o = o[o.completed.astype(bool) & o.home_score.notna()].copy()
    o["home"] = o.home.map(lambda x: ESPN_TO_NBA.get(x, x))
    o["away"] = o.away.map(lambda x: ESPN_TO_NBA.get(x, x))
    o["tip"] = pd.to_datetime(o.date_utc, utc=True)
    o["game_date"] = o.tip.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    o["home_win"] = (o.home_score > o.away_score).astype(int)
    o["margin"] = o.home_score - o.away_score
    # closing ML: explicit close, else post-game current (= last pre-game line)
    o["h_close"] = np.where(o.home_ml_close.notna(), am_num(o.home_ml_close), am_num(o.home_ml))
    o["a_close"] = np.where(o.away_ml_close.notna(), am_num(o.away_ml_close), am_num(o.away_ml))
    o["h_open"], o["a_open"] = am_num(o.home_ml_open), am_num(o.away_ml_open)
    for k in ("close", "open"):
        h, a = am_to_p(o[f"h_{k}"]), am_to_p(o[f"a_{k}"])
        o[f"p_{k}"] = h / (h + a)
        o[f"vig_{k}"] = h + a - 1
    o["spread"] = pd.to_numeric(o.spread, errors="coerce")
    o["total"] = pd.to_numeric(o.total, errors="coerce")
    o["playoffs"] = (o.season_type != 2).astype(int)
    keep = ["event_id", "season", "season_type", "playoffs", "neutral", "tip", "game_date", "home",
            "away", "home_score", "away_score", "home_win", "margin", "book", "h_close", "a_close",
            "p_close", "vig_close", "h_open", "a_open", "p_open", "vig_open", "spread", "total"]
    g = o[keep].copy()
    # sanity: drop games without a usable closing price or with absurd vig (live-odds leak)
    g = g[g.p_close.notna() & (g.vig_close > 0) & (g.vig_close < 0.12)]
    g.loc[~((g.vig_open > 0) & (g.vig_open < 0.12)), "p_open"] = np.nan
    g = g.sort_values("tip").reset_index(drop=True)
    g = g.merge(team_context(), on=["game_date", "home", "away"], how="left")
    g.to_csv(f, index=False)
    return g


def team_context():
    """Leak-free per-team context from nba_api team logs: previous game margin, streak
    entering the game, days since previous game. Only games strictly before the current
    one are used (shift(1) within team, ordered by date)."""
    t = pd.read_csv(DATA / "team_logs.csv", dtype={"GAME_ID": str})
    t = t[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PLUS_MINUS"]].copy()
    t.columns = ["team", "gid", "game_date", "matchup", "wl", "pm"]
    t = t.sort_values(["team", "game_date", "gid"])
    t["win"] = (t.wl == "W").astype(int)
    grp = t.groupby("team", group_keys=False)
    t["prev_margin"] = grp.pm.shift(1)
    # streak entering the game (+n wins in a row / -n losses), across seasons reset
    t["season_key"] = t.gid.str[3:5]
    streaks = []
    for _, s in t.groupby(["team", "season_key"], sort=False):
        cur, out = 0, []
        for w in s.win:
            out.append(cur)
            cur = (cur + 1 if cur >= 0 else 1) if w else (cur - 1 if cur <= 0 else -1)
        streaks.append(pd.Series(out, index=s.index))
    t["streak_in"] = pd.concat(streaks)
    t.loc[t.groupby(["team", "season_key"]).cumcount() == 0, "prev_margin"] = np.nan
    t["is_home"] = t.matchup.str.contains("vs.")
    h = t[t.is_home][["game_date", "team", "prev_margin", "streak_in"]]
    a = t[~t.is_home][["game_date", "team", "prev_margin", "streak_in"]]
    h.columns = ["game_date", "home", "home_prev_margin", "home_streak"]
    a.columns = ["game_date", "away", "away_prev_margin", "away_streak"]
    # need home/away pairing: join through the game id
    hh = t[t.is_home][["gid"]].join(h)
    aa = t[~t.is_home][["gid"]].join(a)
    m = hh.merge(aa.drop(columns="game_date"), on="gid")
    return m.drop(columns="gid").drop_duplicates(["game_date", "home", "away"])


# ----------------------------------------------------------------------------- book table
def _book_rows_from_cache():
    rows = []
    files = list(BOOKS_DIR.glob("*.json")) if BOOKS_DIR.exists() else []
    for fp in files:
        eid = int(fp.stem)
        for it in json.loads(fp.read_text()):
            h, a = it["h"], it["a"]
            rows.append({"event_id": eid, "book": it["p"], "spread": it["s"], "total": it["t"],
                         "h_ml": h["ml"], "a_ml": a["ml"], "h_so": h["so"], "a_so": a["so"],
                         "h_ml_open": h["ml_open"], "a_ml_open": a["ml_open"],
                         "h_ml_close": h["ml_close"], "a_ml_close": a["ml_close"],
                         "h_sp_open": h["sp_open"], "h_sp_close": h["sp_close"],
                         "h_so_open": h["so_open"], "a_so_open": a["so_open"],
                         "h_so_close": h["so_close"], "a_so_close": a["so_close"],
                         "t_open": it.get("t_open"), "t_close": it.get("t_close")})
    return pd.DataFrame(rows), len(files)


def load_books(games=None, source="auto"):
    """Long table event x book with de-vigged prob p, vig, decimal odds.
    source: 'cache' (full refetch, has spread juice), 'raw' (original CSV column), 'auto'."""
    f = CACHE / "books.csv"
    if f.exists() and source == "auto":
        return pd.read_csv(f)
    games = load_games() if games is None else games
    b, nfiles = _book_rows_from_cache() if source in ("auto", "cache") else (pd.DataFrame(), 0)
    complete = nfiles >= 0.98 * len(games)
    if not complete:
        rows = []
        for _, fp in enumerate(sorted(glob.glob(str(DATA / "espn_*.csv")))):
            o = pd.read_csv(fp, usecols=["event_id", "raw"])
            for eid, raw in zip(o.event_id, o.raw):
                if isinstance(raw, str):
                    for x in json.loads(raw):
                        rows.append({"event_id": eid, "book": x["p"], "spread": x["s"],
                                     "h_ml": x["h"], "a_ml": x["a"]})
        b = pd.DataFrame(rows)
    b = b[~b.book.str.lower().str.contains("|".join(NON_BOOKS))].copy()
    b["h_ml"], b["a_ml"] = am_num(b.h_ml), am_num(b.a_ml)
    hp, ap = am_to_p(b.h_ml), am_to_p(b.a_ml)
    b["vig"] = hp + ap - 1
    b["p"] = hp / (hp + ap)
    b["h_dec"], b["a_dec"] = am_to_dec(b.h_ml), am_to_dec(b.a_ml)
    # sanity: drop rows without both sides, crossed/negative vig, or absurd vig
    b = b[b.p.notna() & (b.vig > 0.0) & (b.vig < 0.10)]
    b = b[b.event_id.isin(games.event_id)].drop_duplicates(["event_id", "book"])
    # Leak guard: a stored 'current' line far from the main-book close is likely in-play.
    g = games.set_index("event_id")
    b["dev_main"] = logit(b.p) - logit(g.loc[b.event_id, "p_close"].to_numpy())
    b["suspect_live"] = b.dev_main.abs() > 0.75
    b = b[~b.suspect_live].drop(columns="suspect_live")
    if complete:
        b.to_csv(f, index=False)
    return b.reset_index(drop=True)


# ----------------------------------------------------------------------------- protocol
def walk_forward(df, make_X, base="p_close", test_seasons=TEST_SEASONS, C=1e4, model=None,
                 min_train=200):
    """Fit logistic regression on [logit(base), make_X(df)] on seasons < S, predict S.
    Returns a Series of out-of-sample probabilities indexed like df (NaN outside test)."""
    out = pd.Series(np.nan, index=df.index)
    X_all = make_X(df) if make_X is not None else None
    for s in test_seasons:
        tr = (df.season < s).to_numpy()
        te = (df.season == s).to_numpy()
        if tr.sum() < min_train or te.sum() == 0:
            continue
        cols = [logit(df[base].to_numpy())]
        if X_all is not None:
            X_all_ = np.asarray(X_all, float)
            cols += [X_all_] if X_all_.ndim == 1 else [X_all_[:, j] for j in range(X_all_.shape[1])]
        X = np.column_stack(cols)
        ok = np.isfinite(X).all(1) & df.home_win.notna().to_numpy()
        m = (model() if model else LogisticRegression(C=C, max_iter=10000))
        m.fit(X[tr & ok], df.home_win.to_numpy()[tr & ok])
        pr = np.full(len(df), np.nan)
        pr[te & ok] = m.predict_proba(X[te & ok])[:, 1]
        out[te] = pr[te]
    return out


def compare(df, p_model, p_base, label, n_boot=2000, seed=0, n_variants=None):
    """Per-game log-loss difference model - baseline on rows where both exist.
    Bootstrap CI (paired, over games); Bonferroni-adjusted CI when n_variants > 1."""
    m = np.isfinite(p_model) & np.isfinite(p_base)
    d = df[m]
    diff = ll_vec(d.home_win, p_model[m]) - ll_vec(d.home_win, p_base[m])
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.empty(n_boot)
    for i in range(0, n_boot, 200):
        idx = rng.integers(0, n, (min(200, n_boot - i), n))
        boots[i:i + idx.shape[0]] = diff[idx].mean(1)
    from scipy.stats import norm
    k = BONF_K_LL if n_variants is None else n_variants
    zb = norm.ppf(1 - 0.025 / k)
    lo, hi = diff.mean() - zb * boots.std(), diff.mean() + zb * boots.std()
    lo95, hi95 = np.quantile(boots, [0.025, 0.975])
    per = pd.Series(diff, index=d.index).groupby(d.season).mean()
    res = {"variant": label, "n": int(n), "ll_base": float(ll_vec(d.home_win, p_base[m]).mean()),
           "ll_model": float(ll_vec(d.home_win, p_model[m]).mean()), "delta": float(diff.mean()),
           "ci_lo": float(lo95), "ci_hi": float(hi95), "ci_lo_bonf": float(lo), "ci_hi_bonf": float(hi),
           "seasons_neg": int((per < 0).sum()), "seasons": int(len(per))}
    for s, v in per.items():
        res[f"d_{s}"] = float(v)
    res["passes"] = bool(hi95 < 0 and res["seasons_neg"] >= min(3, len(per)) and len(per) >= 3)
    res["passes_bonf"] = bool(hi < 0 and res["passes"])
    return res


def bet_roi(df, p_model, threshold, h_odds="h_close", a_odds="a_close", n_boot=2000, seed=0,
            label=""):
    """Flat 1-unit bets at the given American odds (incl. vig) when p_model exceeds the raw
    implied probability (with vig) by > threshold. At most one side per game."""
    p = np.asarray(p_model, float)
    hd, ad = am_to_dec(df[h_odds]), am_to_dec(df[a_odds])
    hi, ai = 1 / hd, 1 / ad
    eh, ea = p - hi, (1 - p) - ai
    bet_h = np.isfinite(eh) & (eh > threshold) & (eh >= ea)
    bet_a = np.isfinite(ea) & (ea > threshold) & ~bet_h
    y = df.home_win.to_numpy()
    pnl = np.concatenate([np.where(y[bet_h] == 1, hd[bet_h] - 1, -1.0),
                          np.where(y[bet_a] == 0, ad[bet_a] - 1, -1.0)])
    seas = np.concatenate([df.season.to_numpy()[bet_h], df.season.to_numpy()[bet_a]])
    res = {"rule": label, "threshold": threshold, "bets": int(len(pnl))}
    if len(pnl) == 0:
        return res
    rng = np.random.default_rng(seed)
    boots = rng.choice(pnl, (n_boot, len(pnl))).mean(1)
    res.update(_roi_stats(pnl, boots))
    for s in sorted(set(seas)):
        res[f"roi_{s}"] = float(pnl[seas == s].mean())
    return res


def pnl_roi(pnl, seasons=None, n_boot=2000, seed=0, label=""):
    pnl = np.asarray(pnl, float)
    res = {"rule": label, "bets": int(len(pnl))}
    if len(pnl) == 0:
        return res
    rng = np.random.default_rng(seed)
    boots = rng.choice(pnl, (n_boot, len(pnl))).mean(1)
    res.update(_roi_stats(pnl, boots))
    if seasons is not None:
        seasons = np.asarray(seasons)
        for s in sorted(set(seasons)):
            res[f"roi_{s}"] = float(pnl[seasons == s].mean())
    return res


def _roi_stats(pnl, boots):
    from scipy.stats import norm
    zb = norm.ppf(1 - 0.025 / BONF_K_BET)
    return {"roi": float(pnl.mean()), "roi_lo": float(np.quantile(boots, .025)),
            "roi_hi": float(np.quantile(boots, .975)), "units": float(pnl.sum()),
            "roi_lo_bonf": float(pnl.mean() - zb * boots.std()),
            "roi_hi_bonf": float(pnl.mean() + zb * boots.std())}

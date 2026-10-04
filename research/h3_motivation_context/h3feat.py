"""H3 features: motivation and competitive context, as of the MORNING of each game.

Everything is computed from research/data/team_logs.csv (nba_api team game logs) using only games
played on dates strictly before the game date (standings "as of the previous day"). The only forward
looking inputs are schedule facts known before the season (planned number of games, the date of the
last regular-season day), see `planned_games` / `rs_end` below.

Per team and game date (regular season):
  W, L, rem (planned games left), conf_rank (approx. NBA tiebreakers), league_bottom_rank (1 = worst)
  best_seed / worst_seed  : best / worst conference seed still mathematically reachable (approximation,
                            see `_reach`), from which:
  elim_post   : eliminated from the postseason (seed > 10 in the play-in era, > 8 before 2020-21)
  clinch_post : clinched a postseason spot; clinch_po6: clinched a direct playoff spot (top 6; top 8 pre)
  seed_locked : best_seed == worst_seed (and in the postseason) -> nothing left to play for, rest risk
  no_stakes   : elim_post or seed_locked
  margin_<b>  : signed games relative to boundary b (rank b vs b+1): >0 = cushion above, <0 = games back
  in_race     : a seed boundary that matters (1|2, 4|5 home court, 6|7 direct playoffs, 8|9, 10|11) is
                still reachable on both sides AND within 2 games, with <= 20 games left
  tank        : one of the 6 worst records in the league with <= 25 games left (lottery odds are flat
                for the bottom 3 since 2019; positions 4-6 still gain odds by losing)
  last_week   : game within 7 days of the last scheduled regular-season day
  last_game   : the team's final regular-season game

Per playoff game (series state before the game, from earlier games of the same series):
  game_no, home_series_w, away_series_w, home_lost_prev (+1 home lost previous series game, -1 away
  lost it, 0 in game 1), prev_margin_home (home team's margin in the previous series game),
  home_elim (home team faces elimination), away_elim, game7, home_down02 (+1 home trails 0-2, -1 home
  leads 2-0).
Play-in: elimination for the loser of 9v10 and of the '8th-seed' game; 7v8 loser gets a second chance.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "research/data/h3_motivation_context"

EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
WEST = {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"}
CONF = {**{t: "E" for t in EAST}, **{t: "W" for t in WEST}}


def planned_games(season, date, actual_total):
    """Games a team was scheduled to play, as known on `date`."""
    if season == "2020-21":
        return 72
    if season == "2019-20":
        # season suspended on 2020-03-11; the bubble restart schedule (8 seeding games for 22 teams) was
        # announced in June 2020. Before that the planned total was 82.
        return 82 if date < "2020-06-05" else actual_total
    return 82


def post_line(season):
    """Last seed that reaches the postseason (play-in era: 10) and last direct playoff seed."""
    return (8, 8) if season in ("2018-19", "2019-20") else (10, 6)


def team_games(path=ROOT / "research/data/team_logs.csv"):
    t = pd.read_csv(path)
    t["home"] = t.MATCHUP.str.contains("vs.")
    t["opp"] = t.MATCHUP.str.split(" ").str[-1]
    t["win"] = (t.WL == "W").astype(int)
    t["margin"] = t.PLUS_MINUS.astype(float)
    t = t.rename(columns={"TEAM_ABBREVIATION": "team", "GAME_DATE": "date", "SEASON": "season",
                          "SEASON_TYPE": "stype", "GAME_ID": "gid"})
    return t[["season", "stype", "gid", "date", "team", "opp", "home", "win", "margin", "PTS"]]


# ----------------------------------------------------------------------------- standings
def _rank_conf(teams, W, L, h2h_w, h2h_g, conf_w, conf_g, diff):
    """Order teams of one conference with an approximation of NBA tiebreakers:
    win% > head-to-head win% among the tied teams > conference win% > point differential."""
    pct = {t: W[t] / max(W[t] + L[t], 1) for t in teams}
    order = sorted(teams, key=lambda t: -pct[t])
    out, i = [], 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(pct[order[j + 1]] - pct[order[i]]) < 1e-12:
            j += 1
        grp = order[i:j + 1]
        if len(grp) > 1:
            def key(t):
                w = sum(h2h_w.get((t, o), 0) for o in grp if o != t)
                g = sum(h2h_g.get((t, o), 0) for o in grp if o != t)
                return (-(w / g if g else 0.5), -(conf_w[t] / max(conf_g[t], 1)), -diff[t])
            grp = sorted(grp, key=key)
        out += grp
        i = j + 1
    return out


def _reach(t, teams, W, rem):
    """Best and worst reachable seed for team t (counting-based approximation).
    best : 1 + #teams whose CURRENT wins already exceed t's maximum possible wins
    worst: 1 + #teams whose maximum possible wins reach t's current wins (ties counted against t)
    Ignores remaining head-to-head games (teams playing each other cannot both win) and tiebreakers,
    so 'eliminated'/'locked' flags are conservative (sometimes set later than the true clinch date)."""
    mx = W[t] + rem[t]
    best = 1 + sum(1 for o in teams if o != t and W[o] > mx)
    worst = 1 + sum(1 for o in teams if o != t and W[o] + rem[o] >= W[t])
    return best, worst


def standings(tg):
    """Team context on every date the team plays (any season type), using regular-season games before it."""
    rows = []
    rs_all = tg[tg.stype == "Regular Season"]
    for season, g in tg.groupby("season"):
        rs = rs_all[rs_all.season == season].sort_values("date")
        rs_end = rs.date.max()
        actual = rs.groupby("team").size().to_dict()
        n_post, n_po = post_line(season)
        teams_all = sorted(rs.team.unique())
        W = {t: 0 for t in teams_all}
        L = {t: 0 for t in teams_all}
        diff = {t: 0.0 for t in teams_all}
        conf_w = {t: 0 for t in teams_all}
        conf_g = {t: 0 for t in teams_all}
        h2h_w, h2h_g = {}, {}
        dates = sorted(g.date.unique())
        rs_by_date = {d: x for d, x in rs.groupby("date")}
        played_dates = sorted(rs_by_date)
        k = 0
        for d in dates:
            # add every regular-season game strictly before d
            while k < len(played_dates) and played_dates[k] < d:
                for r in rs_by_date[played_dates[k]].itertuples():
                    W[r.team] += r.win
                    L[r.team] += 1 - r.win
                    diff[r.team] += r.margin
                    h2h_w[(r.team, r.opp)] = h2h_w.get((r.team, r.opp), 0) + r.win
                    h2h_g[(r.team, r.opp)] = h2h_g.get((r.team, r.opp), 0) + 1
                    if CONF[r.team] == CONF[r.opp]:
                        conf_w[r.team] += r.win
                        conf_g[r.team] += 1
                k += 1
            rem = {t: max(planned_games(season, d, actual[t]) - W[t] - L[t], 0) for t in teams_all}
            league_pct = {t: W[t] / max(W[t] + L[t], 1) for t in teams_all}
            # league bottom rank: 1 = worst record (ties broken by point differential, worse first)
            bottom = sorted(teams_all, key=lambda t: (league_pct[t], diff[t]))
            bottom_rank = {t: i + 1 for i, t in enumerate(bottom)}
            for conf in ("E", "W"):
                teams = [t for t in teams_all if CONF[t] == conf]
                order = _rank_conf(teams, W, L, h2h_w, h2h_g, conf_w, conf_g, diff)
                rank = {t: i + 1 for i, t in enumerate(order)}

                def gb(a, b):  # games team a is ahead of team b
                    return ((W[a] - W[b]) + (L[b] - L[a])) / 2

                for t in teams:
                    best, worst = _reach(t, teams, W, rem)
                    r = rank[t]
                    row = {"season": season, "date": d, "team": t, "W": W[t], "L": L[t], "gp": W[t] + L[t],
                           "rem": rem[t], "win_pct": league_pct[t], "diff_pg": diff[t] / max(W[t] + L[t], 1),
                           "conf_rank": r, "league_bottom_rank": bottom_rank[t], "best_seed": best,
                           "worst_seed": worst, "rs_end": rs_end,
                           "days_to_rs_end": (pd.Timestamp(rs_end) - pd.Timestamp(d)).days}
                    for b in (1, 4, 6, 8, 10):
                        # signed distance to boundary b|b+1
                        if r <= b:
                            row[f"margin_{b}"] = gb(t, order[b]) if b < len(order) else 99.0
                        else:
                            row[f"margin_{b}"] = -gb(order[b - 1], t)
                    rows.append(row)
    s = pd.DataFrame(rows)
    n_post = s.season.map(lambda x: post_line(x)[0])
    n_po = s.season.map(lambda x: post_line(x)[1])
    s["elim_post"] = (s.best_seed > n_post).astype(int)
    s["clinch_post"] = (s.worst_seed <= n_post).astype(int)
    s["clinch_po6"] = (s.worst_seed <= n_po).astype(int)
    s["seed_locked"] = ((s.best_seed == s.worst_seed) & (s.worst_seed <= n_post)).astype(int)
    s["no_stakes"] = ((s.elim_post == 1) | (s.seed_locked == 1)).astype(int)
    # live boundaries: the team can still finish on either side of b|b+1 and is within 2 games of it
    race = np.zeros(len(s), bool)
    for b in (1, 4, 6, 8, 10):
        live = (s.best_seed <= b) & (s.worst_seed > b)
        if b == 10:
            live &= n_post == 10
        if b == 6:
            live &= n_po == 6
        race |= live & (s[f"margin_{b}"].abs() <= 2)
    s["in_race"] = (race & (s.rem <= 20)).astype(int)
    s["tank"] = ((s.league_bottom_rank <= 6) & (s.rem <= 25)).astype(int)
    s["last_week"] = (s.days_to_rs_end <= 7).astype(int)
    s["last_game"] = (s.rem == 1).astype(int)
    return s


# ----------------------------------------------------------------------------- playoffs
def series_state(tg):
    """One row per playoff / play-in game (keyed by date + home + away) with the series state before it."""
    po = tg[tg.stype.isin(["Playoffs", "PlayIn"]) & tg.home].copy().sort_values(["date", "gid"])
    rows = []
    for (season, stype), g in po.groupby(["season", "stype"]):
        hist = {}  # frozenset pair -> list of (winner, margin_for_team_a, home)
        for r in g.itertuples():
            key = frozenset((r.team, r.opp))
            prev = hist.get(key, [])
            hw = sum(1 for p in prev if p[0] == r.team)
            aw = sum(1 for p in prev if p[0] == r.opp)
            row = {"season": season, "date": r.date, "home": r.team, "away": r.opp, "stype": stype,
                   "gid": r.gid, "game_no": len(prev) + 1, "home_series_w": hw, "away_series_w": aw}
            if stype == "Playoffs":
                gid = str(r.gid).zfill(10)
                row["gid_game_no"] = int(gid[-1])
                row["round"] = int(gid[-3])
                if prev:
                    last_w, last_m_for, last_team = prev[-1]
                    row["home_lost_prev"] = 1 if last_w != r.team else -1
                    row["prev_margin_home"] = last_m_for if last_team == r.team else -last_m_for
                else:
                    row["home_lost_prev"], row["prev_margin_home"] = 0, 0.0
                row["home_elim"] = int(aw == 3)
                row["away_elim"] = int(hw == 3)
                row["game7"] = int(hw == 3 and aw == 3)
                row["home_down02"] = 1 if (hw, aw) == (0, 2) else (-1 if (hw, aw) == (2, 0) else 0)
            else:  # play-in: game ids 0052X00101.. ; elimination logic by seeding is approximated below
                row["round"] = 0
            rows.append(row)
            hist.setdefault(key, []).append((r.team if r.win else r.opp, r.margin, r.team))
    s = pd.DataFrame(rows)
    return s


def playin_elim(tg, stand):
    """Play-in elimination flags: 9v10 game -> loser out (both 'face elimination'); 7v8 game -> nobody out;
    the second-round game (loser of 7v8 vs winner of 9v10) -> both face elimination.
    Identified from end-of-season seeds (standings on the play-in date)."""
    pi = tg[(tg.stype == "PlayIn") & tg.home].copy()
    out = []
    for r in pi.itertuples():
        sh = stand[(stand.season == r.season) & (stand.date == r.date) & (stand.team == r.team)]
        sa = stand[(stand.season == r.season) & (stand.date == r.date) & (stand.team == r.opp)]
        if sh.empty or sa.empty:
            out.append((r.date, r.team, r.opp, 1, 1))
            continue
        seeds = {int(sh.conf_rank.iloc[0]), int(sa.conf_rank.iloc[0])}
        both = 0 if seeds == {7, 8} else 1
        out.append((r.date, r.team, r.opp, both, both))
    return pd.DataFrame(out, columns=["date", "home", "away", "home_elim", "away_elim"])


def build(cache=True):
    path = CACHE / "context_features.csv"
    spath = CACHE / "series_state.csv"
    if cache and path.exists() and spath.exists():
        return pd.read_csv(path), pd.read_csv(spath)
    tg = team_games()
    st = standings(tg)
    ss = series_state(tg)
    pe = playin_elim(tg, st)
    ss = ss.merge(pe, on=["date", "home", "away"], how="left", suffixes=("", "_pi"))
    for c in ("home_elim", "away_elim"):
        ss[c] = ss[c].fillna(ss[f"{c}_pi"]).fillna(0).astype(int)
    ss = ss.drop(columns=["home_elim_pi", "away_elim_pi"])
    CACHE.mkdir(parents=True, exist_ok=True)
    st.to_csv(path, index=False)
    ss.to_csv(spath, index=False)
    return st, ss


if __name__ == "__main__":
    st, ss = build(cache=False)
    print(st.shape, ss.shape)
    last = st[st.date == st.groupby("season").date.transform("max")]
    print(st.groupby("season")[["elim_post", "seed_locked", "no_stakes", "in_race", "tank", "last_week"]].mean())
    po = ss[ss.stype == "Playoffs"]
    print("game_no == id game_no:", (po.game_no == po.gid_game_no).mean())

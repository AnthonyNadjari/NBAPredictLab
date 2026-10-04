"""Talking points for a game, scored by how strongly they favour one side.

Each angle compares the picked team (P) with its opponent (O) on one dimension,
normalised by the league-wide spread of that difference (2024-26 seasons), so
"strength" is comparable across angles: +1 = a one-standard-deviation edge for
the pick, negative = the angle argues against the pick (used for the risk tweet).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Standard deviation of (home - away) for each stat over 2024-25..2025-26 games
SCALE = {"elo": 197.0, "last10_net_rating": 10.6, "last10_win_pct": 0.30, "offense": 6.8, "defense": 6.8,
         "fg3": 0.040, "season_diff": 6.0, "rest": 1.0, "g5": 0.72, "streak": 4.8, "venue": 0.18}
LEAGUE_HOME_WIN, LEAGUE_ROAD_WIN = 0.547, 0.455


@dataclass
class Angle:
    key: str
    strength: float            # >0 supports the pick, <0 argues against it
    title: str                 # card title / tweet header
    text: str                  # tweet body (header excluded)
    rows: List[Dict] = field(default_factory=list)  # card metric rows
    sub: str = ""

    @property
    def supports(self) -> bool:
        return self.strength > 0


class Side:
    """Feature accessor for one team of the game."""

    def __init__(self, f: Dict, prefix: str, code: str, name: str):
        self.f, self.p, self.code, self.name = f, prefix, code, name

    def g(self, key: str, default=None):
        v = self.f.get(f"{self.p}_{key}")
        return default if v is None else v


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _row(label, p: Side, pv, pl, o: Side, ov, ol, lower_is_better=False):
    return {"label": label, "lower_is_better": lower_is_better,
            "a": {"team": p.code, "value": pv, "label": pl}, "b": {"team": o.code, "value": ov, "label": ol}}


def _record(s: Side) -> Optional[str]:
    gp, w = _num(s.g("games_played")), s.g("season_win_pct")
    if gp >= 3 and w is not None:
        wins = round(_num(w) * gp)
        return f"{wins}-{int(gp) - wins}"
    return None


def _l10(s: Side) -> str:
    w = round(_num(s.g("last10_win_pct")) * 10)
    return f"{w}-{10 - w}"


def build(f: Dict, pick: Side, opp: Side, pick_home: bool) -> List[Angle]:
    out: List[Angle] = []
    P, O = pick, opp

    # Power rating (Elo)
    pe, oe = _num(P.g("elo"), 1500), _num(O.g("elo"), 1500)
    out.append(Angle("elo", (pe - oe) / SCALE["elo"], "Power rating",
                     f"Elo has {P.name} at {pe:.0f}, {O.name} at {oe:.0f}. "
                     + ("A real gap in team strength." if pe - oe > 60 else "Closer than the record says." if pe > oe else
                        f"{O.name} rate higher on paper."),
                     [_row("Elo rating", P, pe, f"{pe:.0f}", O, oe, f"{oe:.0f}")]))

    # Season point differential
    if P.g("season_diff") is not None and O.g("season_diff") is not None and _num(P.g("games_played")) >= 3:
        pd_, od = _num(P.g("season_diff")), _num(O.g("season_diff"))
        rec_p, rec_o = _record(P), _record(O)
        out.append(Angle("season", (pd_ - od) / SCALE["season_diff"], "Season so far",
                         f"{P.name}: {pd_:+.1f} points per game{f' ({rec_p})' if rec_p else ''}. "
                         f"{O.name}: {od:+.1f}{f' ({rec_o})' if rec_o else ''}.",
                         [_row("Point diff / game", P, pd_, f"{pd_:+.1f}", O, od, f"{od:+.1f}")]))

    # Recent form (last 10)
    pn, on = _num(P.g("last10_net_rating")), _num(O.g("last10_net_rating"))
    out.append(Angle("form", (pn - on) / SCALE["last10_net_rating"], "Recent form",
                     f"Last 10 games: {P.name} {_l10(P)}, net rating {pn:+.1f}. {O.name} {_l10(O)}, {on:+.1f}.",
                     [_row("Net rating, last 10", P, pn, f"{pn:+.1f}", O, on, f"{on:+.1f}"),
                      _row("Record, last 10", P, _num(P.g("last10_win_pct")), _l10(P),
                           O, _num(O.g("last10_win_pct")), _l10(O))]))

    # Offense / defense
    po, oo = _num(P.g("last10_offensive_rating")), _num(O.g("last10_offensive_rating"))
    if po and oo:
        out.append(Angle("offense", (po - oo) / SCALE["offense"], "Offense",
                         f"{P.name} score {po:.1f} points per 100 possessions over their last 10, "
                         f"{O.name} {oo:.1f}.",
                         [_row("Offensive rating", P, po, f"{po:.1f}", O, oo, f"{oo:.1f}")]))
    pdr, odr = _num(P.g("last10_defensive_rating")), _num(O.g("last10_defensive_rating"))
    if pdr and odr:
        out.append(Angle("defense", (odr - pdr) / SCALE["defense"], "Defense",
                         f"{P.name} allow {pdr:.1f} per 100 possessions lately, {O.name} {odr:.1f}.",
                         [_row("Defensive rating (lower is better)", P, pdr, f"{pdr:.1f}", O, odr, f"{odr:.1f}",
                               lower_is_better=True)]))

    # Three-point shooting
    p3, o3 = _num(P.g("last10_fg3_pct")), _num(O.g("last10_fg3_pct"))
    if p3 and o3:
        out.append(Angle("shooting", (p3 - o3) / SCALE["fg3"], "From deep",
                         f"{P.name} are hitting {p3 * 100:.1f}% from three over their last 10, {O.name} {o3 * 100:.1f}%.",
                         [_row("3P%, last 10", P, p3, f"{p3 * 100:.1f}%", O, o3, f"{o3 * 100:.1f}%")]))

    # Rest and schedule
    pr, orr = _num(P.g("rest_days"), 2), _num(O.g("rest_days"), 2)
    pb, ob = int(_num(P.g("back_to_back"))), int(_num(O.g("back_to_back")))
    if pb != ob or abs(pr - orr) >= 1:
        strength = (pr - orr) / SCALE["rest"] + 1.2 * (ob - pb)
        if ob and not pb:
            txt = f"{O.name} are on the second night of a back-to-back. {P.name} come in with {pr:.0f} day{'s' if pr != 1 else ''} of rest."
        elif pb and not ob:
            txt = f"{P.name} are on a back-to-back; {O.name} had {orr:.0f} day{'s' if orr != 1 else ''} off."
        else:
            txt = f"Rest: {P.name} {pr:.0f} day{'s' if pr != 1 else ''}, {O.name} {orr:.0f}."
        out.append(Angle("rest", min(strength, 2.5) if strength > 0 else max(strength, -2.5), "Fresh legs", txt,
                         [_row("Days of rest", P, pr, f"{pr:.0f}", O, orr, f"{orr:.0f}")]))
    pg5, og5 = P.g("games_last5d"), O.g("games_last5d")
    if pg5 is not None and og5 is not None and abs(_num(pg5) - _num(og5)) >= 1:
        out.append(Angle("schedule", (_num(og5) - _num(pg5)) / SCALE["g5"], "Schedule load",
                         f"Games in the last 5 days: {P.name} {int(_num(pg5))}, {O.name} {int(_num(og5))}.",
                         [_row("Games in last 5 days", P, _num(pg5), str(int(_num(pg5))), O, _num(og5),
                               str(int(_num(og5))), lower_is_better=True)]))

    # Home / road splits (last 20 home or road games)
    hv, rv = f.get("home_team_home_win_pct"), f.get("away_team_road_win_pct")
    if hv is not None and rv is not None:
        if pick_home:
            s = ((_num(hv) - LEAGUE_HOME_WIN) - (_num(rv) - LEAGUE_ROAD_WIN)) / SCALE["venue"]
            txt = f"{P.name} have won {_num(hv) * 100:.0f}% at home lately; {O.name} {_num(rv) * 100:.0f}% on the road."
            rows = [_row("Win % (home vs road)", P, _num(hv), f"{_num(hv) * 100:.0f}%", O, _num(rv), f"{_num(rv) * 100:.0f}%")]
        else:
            s = ((_num(rv) - LEAGUE_ROAD_WIN) - (_num(hv) - LEAGUE_HOME_WIN)) / SCALE["venue"]
            txt = f"{P.name} have won {_num(rv) * 100:.0f}% of recent road games; {O.name} {_num(hv) * 100:.0f}% at home."
            rows = [_row("Win % (road vs home)", P, _num(rv), f"{_num(rv) * 100:.0f}%", O, _num(hv), f"{_num(hv) * 100:.0f}%")]
        out.append(Angle("venue", s, "Home and away", txt, rows))

    # Streaks
    ps, os_ = int(_num(P.g("streak"))), int(_num(O.g("streak")))
    if ps >= 3 or os_ <= -3 or ps <= -3 or os_ >= 3:
        def streak_txt(n):
            return f"won {n} straight" if n > 0 else f"lost {-n} straight" if n < 0 else "no streak"
        out.append(Angle("streak", (ps - os_) / SCALE["streak"], "Momentum",
                         f"{P.name} have {streak_txt(ps)}. {O.name} have {streak_txt(os_)}.",
                         [_row("Current streak", P, ps, f"{ps:+d}", O, os_, f"{os_:+d}")]))

    # Key absences (ESPN injury report: team leaders listed Out/Doubtful)
    pk, ok = P.g("key_out") or [], O.g("key_out") or []
    if ok and not pk:
        out.append(Angle("injuries", 1.2 * len(ok), "Missing pieces",
                         f"{O.name} are listed without {', '.join(ok[:2])}. {P.name} have no key absence on the report."))
    elif pk:
        out.append(Angle("injuries", -1.2 * len(pk) + 1.2 * len(ok), "Missing pieces",
                         f"{P.name} are listed without {', '.join(pk[:2])}."
                         + (f" {O.name} are missing {', '.join(ok[:2])} too." if ok else "")))

    # Stats model vs market
    mk, md = f.get("market_home_prob"), f.get("model_home_prob")
    if mk is not None and md is not None:
        mp = _num(md) if pick_home else 1 - _num(md)
        out.append(Angle("model", (mp - 0.5) / 0.15, "Model check",
                         f"Our stats model (Elo, form, rest) gives {P.name} {mp * 100:.0f}%"
                         + (", in line with the books." if mp >= 0.5 else f". It leans {O.name}: a contrarian signal.")))
    return out


def ranked(angles: List[Angle]) -> List[Angle]:
    return sorted(angles, key=lambda a: a.strength, reverse=True)

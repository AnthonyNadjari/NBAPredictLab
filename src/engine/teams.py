"""NBA team identifiers. Canonical code = NBA tricode (as used by nba_api)."""

TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# ESPN abbreviations that differ from NBA tricodes
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}

_ALIASES = {"Los Angeles Clippers": "LAC", "L.A. Clippers": "LAC", "L.A. Lakers": "LAL"}
_FULL_TO_CODE = {v: k for k, v in TEAMS.items()}
_FULL_TO_CODE.update(_ALIASES)


def from_espn(abbr: str) -> str:
    return ESPN_TO_NBA.get(abbr, abbr)


def to_code(name: str) -> str:
    """Tricode from a tricode, ESPN abbreviation or full name. Raises if unknown."""
    if name in TEAMS:
        return name
    if name in ESPN_TO_NBA:
        return ESPN_TO_NBA[name]
    if name in _FULL_TO_CODE:
        return _FULL_TO_CODE[name]
    raise KeyError(f"Unknown team: {name!r}")


def full_name(code: str) -> str:
    return TEAMS.get(code, code)

"""Static venue data: arena coordinates, IANA time zones, elevation, neutral sites.

All values are hardcoded (approximate to <1 km, elevation to ~10 m). No team relocated
between 2018-19 and 2025-26; name changes (e.g. arena sponsors) do not matter.
The only venue move inside one metro area is the Clippers (Crypto.com Arena -> Intuit Dome,
2024-25), which is ~13 km and is handled for completeness.
"""

# code: (lat, lon, IANA tz, elevation m)
ARENAS = {
    "ATL": (33.7573, -84.3963, "America/New_York", 320),     # State Farm Arena
    "BOS": (42.3662, -71.0621, "America/New_York", 6),       # TD Garden
    "BKN": (40.6826, -73.9754, "America/New_York", 10),      # Barclays Center
    "CHA": (35.2251, -80.8392, "America/New_York", 230),     # Spectrum Center
    "CHI": (41.8807, -87.6742, "America/Chicago", 180),      # United Center
    "CLE": (41.4965, -81.6882, "America/New_York", 200),     # Rocket Mortgage FieldHouse
    "DAL": (32.7905, -96.8103, "America/Chicago", 140),      # American Airlines Center
    "DEN": (39.7487, -105.0077, "America/Denver", 1609),     # Ball Arena (5,280 ft)
    "DET": (42.3411, -83.0553, "America/Detroit", 190),      # Little Caesars Arena
    "GSW": (37.7680, -122.3877, "America/Los_Angeles", 5),   # Chase Center
    "HOU": (29.7508, -95.3621, "America/Chicago", 15),       # Toyota Center
    "IND": (39.7640, -86.1555, "America/Indiana/Indianapolis", 220),  # Gainbridge Fieldhouse
    "LAC": (34.0430, -118.2673, "America/Los_Angeles", 90),  # Crypto.com Arena (to 2023-24)
    "LAL": (34.0430, -118.2673, "America/Los_Angeles", 90),  # Crypto.com Arena
    "MEM": (35.1382, -90.0506, "America/Chicago", 80),       # FedExForum
    "MIA": (25.7814, -80.1870, "America/New_York", 2),       # Kaseya Center
    "MIL": (43.0451, -87.9172, "America/Chicago", 190),      # Fiserv Forum
    "MIN": (44.9795, -93.2761, "America/Chicago", 255),      # Target Center
    "NOP": (29.9490, -90.0821, "America/Chicago", 2),        # Smoothie King Center
    "NYK": (40.7505, -73.9934, "America/New_York", 10),      # Madison Square Garden
    "OKC": (35.4634, -97.5151, "America/Chicago", 365),      # Paycom Center
    "ORL": (28.5392, -81.3839, "America/New_York", 30),      # Kia Center
    "PHI": (39.9012, -75.1720, "America/New_York", 10),      # Wells Fargo Center
    "PHX": (33.4457, -112.0712, "America/Phoenix", 330),     # Footprint Center (no DST)
    "POR": (45.5316, -122.6668, "America/Los_Angeles", 15),  # Moda Center
    "SAC": (38.5802, -121.4997, "America/Los_Angeles", 8),   # Golden 1 Center
    "SAS": (29.4270, -98.4375, "America/Chicago", 200),      # Frost Bank Center
    "TOR": (43.6435, -79.3791, "America/Toronto", 80),       # Scotiabank Arena
    "UTA": (40.7683, -111.9011, "America/Denver", 1288),     # Delta Center (4,226 ft)
    "WAS": (38.8981, -77.0209, "America/New_York", 20),      # Capital One Arena
}
LAC_INTUIT = (33.9447, -118.3414, "America/Los_Angeles", 30)  # Intuit Dome, from 2024-25

NEUTRAL = {
    "MEX": (19.4040, -99.0960, "America/Mexico_City", 2240),  # Arena CDMX (higher than Denver)
    "LAS": (36.1028, -115.1784, "America/Los_Angeles", 610),  # T-Mobile Arena (NBA Cup KO)
    "PAR": (48.8386, 2.3786, "Europe/Paris", 35),             # Accor Arena
    "BER": (52.5079, 13.4436, "Europe/Berlin", 35),           # Uber Arena
    "LON": (51.5030, 0.0032, "Europe/London", 5),             # The O2
}

# Neutral-site regular-season games 2021-22 .. 2025-26, keyed by (US-Eastern date, frozenset teams).
# (No NBA neutral-site games in 2021-22. Earlier seasons are not used: features reset each season.)
NEUTRAL_GAMES = {
    ("2022-12-17", frozenset({"SAS", "MIA"})): "MEX",
    ("2023-01-19", frozenset({"DET", "CHI"})): "PAR",
    ("2023-11-09", frozenset({"ORL", "ATL"})): "MEX",
    ("2023-12-07", frozenset({"MIL", "IND"})): "LAS",  # In-Season Tournament semis
    ("2023-12-07", frozenset({"LAL", "NOP"})): "LAS",
    ("2023-12-09", frozenset({"LAL", "IND"})): "LAS",  # final (not in team logs, added below)
    ("2024-01-11", frozenset({"CLE", "BKN"})): "PAR",
    ("2024-11-02", frozenset({"WAS", "MIA"})): "MEX",
    ("2024-12-14", frozenset({"MIL", "ATL"})): "LAS",
    ("2024-12-14", frozenset({"OKC", "HOU"})): "LAS",
    ("2024-12-17", frozenset({"OKC", "MIL"})): "LAS",
    ("2025-01-23", frozenset({"IND", "SAS"})): "PAR",
    ("2025-01-25", frozenset({"IND", "SAS"})): "PAR",
    ("2025-11-01", frozenset({"DET", "DAL"})): "MEX",
    ("2025-12-13", frozenset({"ORL", "NYK"})): "LAS",
    ("2025-12-13", frozenset({"OKC", "SAS"})): "LAS",
    ("2025-12-16", frozenset({"NYK", "SAS"})): "LAS",
    ("2026-01-15", frozenset({"ORL", "MEM"})): "BER",
    ("2026-01-18", frozenset({"MEM", "ORL"})): "LON",
}

# NBA Cup finals do not count in the standings, so nba_api team logs omit them, but the two
# teams still played (and travelled). Added to the schedule for fatigue features only.
CUP_FINALS = [
    # (date ET, season, home, away)
    ("2023-12-09", "2023-24", "LAL", "IND"),
    ("2024-12-17", "2024-25", "OKC", "MIL"),
    ("2025-12-16", "2025-26", "NYK", "SAS"),
]


def home_arena(team, season):
    if team == "LAC" and season >= "2024-25":
        return LAC_INTUIT
    return ARENAS[team]

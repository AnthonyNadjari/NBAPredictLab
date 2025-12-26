"""
Travel Distance Calculator for NBA Teams
Calculates travel distance, time zones crossed, and fatigue index
"""

from typing import Dict, Tuple
from math import radians, cos, sin, asin, sqrt

# NBA Arena Coordinates (Latitude, Longitude)
NBA_ARENAS = {
    # Eastern Conference
    1610612738: (42.3662, -71.0621),   # Boston Celtics - TD Garden
    1610612751: (40.7505, -73.9934),   # Brooklyn Nets - Barclays Center
    1610612752: (40.7505, -73.9934),   # New York Knicks - Madison Square Garden
    1610612755: (39.9012, -75.1720),   # Philadelphia 76ers - Wells Fargo Center
    1610612761: (43.6435, -79.3791),   # Toronto Raptors - Scotiabank Arena
    1610612741: (41.8807, -87.6742),   # Chicago Bulls - United Center
    1610612739: (41.4965, -81.6882),   # Cleveland Cavaliers - Rocket Mortgage FieldHouse
    1610612765: (42.3409, -83.0553),   # Detroit Pistons - Little Caesars Arena
    1610612754: (39.8283, -86.1781),   # Indiana Pacers - Gainbridge Fieldhouse
    1610612749: (43.0435, -87.9170),   # Milwaukee Bucks - Fiserv Forum
    1610612737: (33.7573, -84.3963),   # Atlanta Hawks - State Farm Arena
    1610612766: (35.2251, -80.8392),   # Charlotte Hornets - Spectrum Center
    1610612748: (25.7814, -80.1870),   # Miami Heat - FTX Arena
    1610612753: (28.5392, -81.3839),   # Orlando Magic - Amway Center
    1610612764: (38.8980, -77.0209),   # Washington Wizards - Capital One Arena
    
    # Western Conference
    1610612743: (39.7485, -105.0076),  # Denver Nuggets - Ball Arena
    1610612750: (44.9795, -93.2761),   # Minnesota Timberwolves - Target Center
    1610612760: (35.4634, -97.5151),   # Oklahoma City Thunder - Paycom Center
    1610612757: (45.5316, -122.6668),  # Portland Trail Blazers - Moda Center
    1610612762: (40.7683, -111.9011),  # Utah Jazz - Vivint Arena
    1610612744: (37.7680, -122.3878),  # Golden State Warriors - Chase Center
    1610612746: (34.0430, -118.2673),  # LA Clippers - Crypto.com Arena
    1610612747: (34.0430, -118.2673),  # LA Lakers - Crypto.com Arena
    1610612756: (33.4457, -112.0712),  # Phoenix Suns - Footprint Center
    1610612758: (38.5802, -121.4997),  # Sacramento Kings - Golden 1 Center
    1610612742: (32.7905, -96.8103),   # Dallas Mavericks - American Airlines Center
    1610612745: (29.7500, -95.3621),   # Houston Rockets - Toyota Center
    1610612763: (35.1382, -90.0506),   # Memphis Grizzlies - FedExForum
    1610612740: (29.4270, -98.4375),   # San Antonio Spurs - AT&T Center
    1610612759: (29.9511, -90.0812),   # New Orleans Pelicans - Smoothie King Center
}

# Time Zones (EST offset)
NBA_TIMEZONES = {
    # Eastern (UTC-5)
    1610612738: 0, 1610612751: 0, 1610612752: 0, 1610612755: 0, 1610612761: 0,
    1610612741: 0, 1610612739: 0, 1610612765: 0, 1610612754: 0, 1610612749: 0,
    1610612737: 0, 1610612766: 0, 1610612748: 0, 1610612753: 0, 1610612764: 0,
    
    # Central (UTC-6) = -1 from EST
    1610612743: -1, 1610612750: -1, 1610612760: -1, 1610612763: -1, 1610612740: -1,
    1610612745: -1, 1610612759: -1, 1610612742: -1,
    
    # Mountain (UTC-7) = -2 from EST
    1610612762: -2, 1610612756: -2,
    
    # Pacific (UTC-8) = -3 from EST
    1610612757: -3, 1610612744: -3, 1610612746: -3, 1610612747: -3, 1610612758: -3,
}


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in km
    r = 6371
    
    return c * r


def get_travel_features(away_team_id: int, home_team_id: int, 
                        away_last_game_team_id: int = None) -> Dict[str, float]:
    """
    Calculate travel-related features for the away team.
    
    Args:
        away_team_id: ID of the away team
        home_team_id: ID of the home team (destination)
        away_last_game_team_id: ID of team where away team played last game
                               If None, assumes they're traveling from home
        
    Returns:
        Dictionary with travel features:
        - away_travel_distance: Distance traveled in km
        - away_time_zones_crossed: Number of time zones crossed
        - away_travel_fatigue_index: Composite fatigue score (0-1)
    """
    # Determine origin (where away team is coming from)
    if away_last_game_team_id is not None:
        origin_id = away_last_game_team_id
    else:
        origin_id = away_team_id  # Coming from home
    
    # Get coordinates
    if origin_id not in NBA_ARENAS or home_team_id not in NBA_ARENAS:
        # Fallback if team not found
        return {
            'away_travel_distance': 0.0,
            'away_time_zones_crossed': 0,
            'away_travel_fatigue_index': 0.0
        }
    
    origin_lat, origin_lon = NBA_ARENAS[origin_id]
    dest_lat, dest_lon = NBA_ARENAS[home_team_id]
    
    # Calculate distance
    distance_km = calculate_distance(origin_lat, origin_lon, dest_lat, dest_lon)
    
    # Calculate time zones crossed
    origin_tz = NBA_TIMEZONES.get(origin_id, 0)
    dest_tz = NBA_TIMEZONES.get(home_team_id, 0)
    tz_crossed = abs(dest_tz - origin_tz)
    
    # Calculate fatigue index (0-1 scale)
    # Based on distance and time zones
    # Long distance + time zones = high fatigue
    distance_factor = min(distance_km / 4000, 1.0)  # Max at 4000km
    tz_factor = tz_crossed / 3.0  # Max 3 time zones
    
    fatigue_index = 0.6 * distance_factor + 0.4 * tz_factor
    
    return {
        'away_travel_distance': round(distance_km, 1),
        'away_time_zones_crossed': tz_crossed,
        'away_travel_fatigue_index': round(fatigue_index, 3)
    }


def get_team_location(team_id: int) -> Tuple[float, float]:
    """Get arena coordinates for a team."""
    return NBA_ARENAS.get(team_id, (0.0, 0.0))


def get_team_timezone(team_id: int) -> int:
    """Get timezone offset for a team (relative to EST)."""
    return NBA_TIMEZONES.get(team_id, 0)


# Example usage
if __name__ == "__main__":
    # Example: Lakers traveling to Boston
    lakers_id = 1610612747
    celtics_id = 1610612738
    
    features = get_travel_features(
        away_team_id=lakers_id,
        home_team_id=celtics_id
    )
    
    print("Lakers @ Celtics Travel Analysis:")
    print(f"  Distance: {features['away_travel_distance']} km")
    print(f"  Time Zones: {features['away_time_zones_crossed']}")
    print(f"  Fatigue Index: {features['away_travel_fatigue_index']}")

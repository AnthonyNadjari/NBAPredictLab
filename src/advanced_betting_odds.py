"""
Advanced Betting Odds Calculator
Sophisticated odds model with market simulation, implied probability calibration,
and multiple bookmaker scenarios
"""

from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class BookmakerProfile:
    """Different bookmaker profiles with varying margins"""
    name: str
    margin: float  # Overround percentage
    reputation: str


# Realistic bookmaker profiles
BOOKMAKERS = {
    'pinnacle': BookmakerProfile('Pinnacle', 0.02, 'Sharp - Lowest margins'),
    'bet365': BookmakerProfile('Bet365', 0.045, 'Recreational - Average margins'),
    'betway': BookmakerProfile('Betway', 0.05, 'Recreational - Standard margins'),
    'draftkings': BookmakerProfile('DraftKings', 0.055, 'US Market - Higher margins'),
}


def calibrate_probability(raw_probability: float, confidence: float) -> float:
    """
    Calibrate raw model probability using confidence level.
    High confidence predictions are more reliable, low confidence need adjustment.

    Args:
        raw_probability: Raw probability from model (0-1)
        confidence: Model confidence level (0-1)

    Returns:
        Calibrated probability
    """
    # Regression to mean based on confidence
    # Low confidence -> pull towards 50%
    # High confidence -> keep original probability
    mean_probability = 0.5
    calibrated = raw_probability * confidence + mean_probability * (1 - confidence)

    return np.clip(calibrated, 0.01, 0.99)


def apply_overround(prob_home: float, prob_away: float, margin: float) -> Tuple[float, float]:
    """
    Apply bookmaker overround (margin) to probabilities.
    The overround creates the bookmaker's profit margin.

    Args:
        prob_home: Home team win probability
        prob_away: Away team win probability
        margin: Bookmaker margin (e.g., 0.05 = 5%)

    Returns:
        (implied_prob_home, implied_prob_away) with margin applied
    """
    # Normalize probabilities to sum to 1
    total = prob_home + prob_away
    prob_home = prob_home / total
    prob_away = prob_away / total

    # Apply overround
    # Method: Proportional margin addition
    implied_home = prob_home * (1 + margin)
    implied_away = prob_away * (1 + margin)

    # Normalize to ensure sum > 1 (overround)
    total_implied = implied_home + implied_away
    implied_home = implied_home / total_implied * (1 + margin)
    implied_away = implied_away / total_implied * (1 + margin)

    return np.clip(implied_home, 0.01, 0.99), np.clip(implied_away, 0.01, 0.99)


def probability_to_decimal_odds(probability: float) -> float:
    """Convert probability to decimal odds"""
    return 1 / max(probability, 0.01)


def decimal_to_american(decimal_odds: float) -> str:
    """Convert decimal odds to American format"""
    if decimal_odds >= 2.0:
        # Underdog
        american = int((decimal_odds - 1) * 100)
        return f"+{american}"
    else:
        # Favorite
        american = int(-100 / (decimal_odds - 1))
        return f"{american}"


def decimal_to_fractional(decimal_odds: float) -> str:
    """
    Convert decimal odds to fractional format (UK style).
    Uses proper fraction simplification.
    """
    fractional_value = decimal_odds - 1

    # Common betting fractions
    common_fractions = {
        0.05: "1/20", 0.10: "1/10", 0.125: "1/8", 0.167: "1/6",
        0.20: "1/5", 0.25: "1/4", 0.333: "1/3", 0.40: "2/5",
        0.50: "1/2", 0.60: "3/5", 0.667: "2/3", 0.75: "3/4",
        0.80: "4/5", 1.00: "Evens", 1.20: "6/5", 1.25: "5/4",
        1.333: "4/3", 1.50: "3/2", 1.667: "5/3", 1.75: "7/4",
        2.00: "2/1", 2.50: "5/2", 3.00: "3/1", 3.50: "7/2",
        4.00: "4/1", 4.50: "9/2", 5.00: "5/1", 6.00: "6/1",
        7.00: "7/1", 8.00: "8/1", 9.00: "9/1", 10.00: "10/1"
    }

    # Find closest common fraction
    closest = min(common_fractions.keys(), key=lambda x: abs(x - fractional_value))
    if abs(closest - fractional_value) < 0.02:
        return common_fractions[closest]

    # Otherwise, create simplified fraction
    numerator = int(fractional_value * 100)
    denominator = 100

    # Simple GCD to simplify
    from math import gcd
    divisor = gcd(numerator, denominator)
    numerator //= divisor
    denominator //= divisor

    return f"{numerator}/{denominator}"


def calculate_expected_value(probability: float, decimal_odds: float, stake: float = 100) -> Dict:
    """
    Calculate expected value of a bet.
    EV = (probability * profit) - ((1-probability) * stake)

    Args:
        probability: True probability of winning (0-1)
        decimal_odds: Odds offered by bookmaker
        stake: Amount wagered

    Returns:
        Dict with EV metrics
    """
    profit_if_win = stake * (decimal_odds - 1)
    ev = (probability * profit_if_win) - ((1 - probability) * stake)
    ev_percentage = (ev / stake) * 100

    # Kelly Criterion - optimal bet sizing
    kelly = (probability * decimal_odds - 1) / (decimal_odds - 1)
    kelly_percentage = max(0, kelly * 100)  # Kelly as % of bankroll

    return {
        'expected_value': round(ev, 2),
        'ev_percentage': round(ev_percentage, 2),
        'kelly_criterion': round(kelly_percentage, 2),
        'recommendation': 'VALUE BET' if ev > 0 else 'NO VALUE'
    }


def calculate_advanced_odds(
    home_win_prob: float,
    away_win_prob: float,
    confidence: float,
    bookmaker: str = 'bet365',
    home_elo: float = 1500,
    away_elo: float = 1500
) -> Dict:
    """
    Calculate advanced betting odds.
    
    CRITICAL UPDATE:
    We use Elo ratings to simulate the "Market" (Bookmaker) view.
    Bookmakers rely heavily on long-term strength (Elo).
    Our Model relies on specific match features.
    
    This creates realistic "Model vs Market" disagreements (Value Bets).
    """
    # Get bookmaker profile
    bookie = BOOKMAKERS.get(bookmaker, BOOKMAKERS['bet365'])

    # 1. Calculate Market Probability using Elo (The "Bookmaker's View")
    # Elo formula: 1 / (1 + 10^((opp_elo - team_elo)/400))
    # Add 100 points for home court advantage in the market view
    elo_diff = (home_elo + 100) - away_elo
    market_home_prob = 1 / (1 + 10 ** (-elo_diff / 400))
    market_away_prob = 1 - market_home_prob
    
    # Add small random noise to simulate market inefficiencies (±2%)
    noise = np.random.uniform(-0.02, 0.02)
    market_home_prob = np.clip(market_home_prob + noise, 0.05, 0.95)
    market_away_prob = np.clip(market_away_prob - noise, 0.05, 0.95)
    
    # Normalize market probs
    total_market = market_home_prob + market_away_prob
    market_home_prob /= total_market
    market_away_prob /= total_market

    # 2. Apply Bookmaker Margin (Overround) to Market Probs
    implied_home, implied_away = apply_overround(market_home_prob, market_away_prob, bookie.margin)

    # 3. Calculate Bookmaker Odds (what the user sees)
    decimal_home = probability_to_decimal_odds(implied_home)
    decimal_away = probability_to_decimal_odds(implied_away)

    # 4. Calibrate OUR Model Probability
    calibrated_home = calibrate_probability(home_win_prob, confidence)
    calibrated_away = calibrate_probability(away_win_prob, confidence)

    # 5. Calculate Fair Odds based on OUR Model
    fair_decimal_home = probability_to_decimal_odds(calibrated_home)
    fair_decimal_away = probability_to_decimal_odds(calibrated_away)

    # 6. Calculate Expected Value (EV)
    # EV = (Our Prob * Bookmaker Odds) - 1
    ev_home = calculate_expected_value(calibrated_home, decimal_home)
    ev_away = calculate_expected_value(calibrated_away, decimal_away)

    return {
        'bookmaker': bookie.name,
        'bookmaker_margin': bookie.margin * 100,
        'home_odds': {
            'probability': round(calibrated_home * 100, 1),
            'implied_probability': round(implied_home * 100, 1),
            'decimal': round(decimal_home, 2),
            'american': decimal_to_american(decimal_home),
            'fractional': decimal_to_fractional(decimal_home),
            'fair_decimal': round(fair_decimal_home, 2),
            'value': round((fair_decimal_home - decimal_home) / decimal_home * 100, 1),
            'expected_value': ev_home
        },
        'away_odds': {
            'probability': round(calibrated_away * 100, 1),
            'implied_probability': round(implied_away * 100, 1),
            'decimal': round(decimal_away, 2),
            'american': decimal_to_american(decimal_away),
            'fractional': decimal_to_fractional(decimal_away),
            'fair_decimal': round(fair_decimal_away, 2),
            'value': round((fair_decimal_away - decimal_away) / decimal_away * 100, 1),
            'expected_value': ev_away
        },
        'market_analysis': {
            'total_overround': round((implied_home + implied_away - 1) * 100, 2),
            'confidence_level': round(confidence * 100, 1),
            'calibration_adjustment': round((calibrated_home - home_win_prob) * 100, 1)
        }
    }


def compare_bookmakers(
    home_win_prob: float, 
    away_win_prob: float, 
    confidence: float,
    home_elo: float = 1500,
    away_elo: float = 1500,
    home_team: str = None,
    away_team: str = None,
    use_real_odds: bool = True
) -> List[Dict]:
    """
    Compare odds across multiple bookmakers.
    
    Args:
        home_win_prob: Model's home win probability
        away_win_prob: Model's away win probability  
        confidence: Model confidence
        home_elo: Home team Elo rating
        away_elo: Away team Elo rating
        home_team: Home team name (for scraping)
        away_team: Away team name (for scraping)
        use_real_odds: Whether to try scraping real odds first

    Returns:
        List of odds for each bookmaker
    """
    # Try to get real odds first
    real_odds = None
    if use_real_odds and home_team and away_team:
        try:
            from src.odds_api_client import OddsAPIClient
            client = OddsAPIClient()  # Will auto-load from .env
            real_odds = client.find_game_odds(home_team, away_team)
            if real_odds:
                print(f"✅ Real odds fetched for {away_team} @ {home_team}")
            else:
                print(f"⚠️ No real odds found for {away_team} @ {home_team} - using simulation")
        except Exception as e:
            print(f"❌ Error fetching real odds for {away_team} @ {home_team}: {e}")
            real_odds = None
    
    # If we got real odds, use them
    if real_odds and real_odds.get('bookmakers'):
        comparisons = []
        for bookie_name, odds_data in real_odds['bookmakers'].items():
            # Calculate EV using our model vs real market odds
            home_decimal = odds_data['home']
            away_decimal = odds_data['away']
            
            # Calibrate our model probabilities
            calibrated_home = calibrate_probability(home_win_prob, confidence)
            calibrated_away = calibrate_probability(away_win_prob, confidence)
            
            # Calculate fair odds from our model
            fair_home = probability_to_decimal_odds(calibrated_home)
            fair_away = probability_to_decimal_odds(calibrated_away)
            
            # Calculate EV
            ev_home = calculate_expected_value(calibrated_home, home_decimal)
            ev_away = calculate_expected_value(calibrated_away, away_decimal)
            
            # Calculate implied probability from market odds
            implied_home = 1 / home_decimal
            implied_away = 1 / away_decimal
            
            comparisons.append({
                'bookmaker': bookie_name.title(),
                'bookmaker_margin': ((implied_home + implied_away - 1) * 100),
                'source': 'real_market',
                'home_odds': {
                    'probability': round(calibrated_home * 100, 1),
                    'implied_probability': round(implied_home * 100, 1),
                    'decimal': round(home_decimal, 2),
                    'american': decimal_to_american(home_decimal),
                    'fractional': decimal_to_fractional(home_decimal),
                    'fair_decimal': round(fair_home, 2),
                    'value': round((fair_home - home_decimal) / home_decimal * 100, 1),
                    'expected_value': ev_home
                },
                'away_odds': {
                    'probability': round(calibrated_away * 100, 1),
                    'implied_probability': round(implied_away * 100, 1),
                    'decimal': round(away_decimal, 2),
                    'american': decimal_to_american(away_decimal),
                    'fractional': decimal_to_fractional(away_decimal),
                    'fair_decimal': round(fair_away, 2),
                    'value': round((fair_away - away_decimal) / away_decimal * 100, 1),
                    'expected_value': ev_away
                },
                'market_analysis': {
                    'total_overround': round((implied_home + implied_away - 1) * 100, 2),
                    'confidence_level': round(confidence * 100, 1),
                    'calibration_adjustment': round((calibrated_home - home_win_prob) * 100, 1)
                }
            })
        
        return comparisons
    
    # Fallback to Elo-based simulation
    comparisons = []
    for bookie_key in ['bet365', 'betclic', 'unibet', 'winamax']:
        odds_data = calculate_advanced_odds(
            home_win_prob, away_win_prob, confidence, bookie_key, home_elo, away_elo
        )
        odds_data['source'] = 'elo_simulation'
        comparisons.append(odds_data)

    return comparisons


def get_best_odds(
    home_win_prob: float, 
    away_win_prob: float, 
    confidence: float,
    home_elo: float = 1500,
    away_elo: float = 1500,
    home_team: str = None,
    away_team: str = None,
    use_real_odds: bool = True
) -> Dict:
    """
    Find the best odds across all bookmakers for each outcome.

    Returns:
        Dict with best odds for home and away
    """
    comparisons = compare_bookmakers(
        home_win_prob, away_win_prob, confidence, 
        home_elo, away_elo, home_team, away_team, use_real_odds
    )

    best_home_decimal = max(comp['home_odds']['decimal'] for comp in comparisons)
    best_away_decimal = max(comp['away_odds']['decimal'] for comp in comparisons)

    best_home_bookie = next(comp['bookmaker'] for comp in comparisons
                           if comp['home_odds']['decimal'] == best_home_decimal)
    best_away_bookie = next(comp['bookmaker'] for comp in comparisons
                           if comp['away_odds']['decimal'] == best_away_decimal)

    return {
        'home_best': {
            'bookmaker': best_home_bookie,
            'decimal': best_home_decimal
        },
        'away_best': {
            'bookmaker': best_away_bookie,
            'decimal': best_away_decimal
        }
    }


if __name__ == "__main__":
    # Test with example prediction
    home_prob = 0.65
    away_prob = 0.35
    confidence = 0.72

    print("=" * 80)
    print("ADVANCED BETTING ODDS ANALYSIS")
    print("=" * 80)
    print(f"\nModel Prediction:")
    print(f"  Home Win: {home_prob*100:.1f}%")
    print(f"  Away Win: {away_prob*100:.1f}%")
    print(f"  Confidence: {confidence*100:.1f}%")
    print()

    # Calculate for Bet365
    odds = calculate_advanced_odds(home_prob, away_prob, confidence, 'bet365')

    print(f"Bookmaker: {odds['bookmaker']} (Margin: {odds['bookmaker_margin']:.1f}%)")
    print()
    print("HOME TEAM:")
    print(f"  Decimal: {odds['home_odds']['decimal']}")
    print(f"  American: {odds['home_odds']['american']}")
    print(f"  Fractional: {odds['home_odds']['fractional']}")
    print(f"  Fair Value: {odds['home_odds']['fair_decimal']} ({odds['home_odds']['value']:+.1f}% value)")
    print(f"  EV: {odds['home_odds']['expected_value']['ev_percentage']:+.2f}%")
    print()
    print("AWAY TEAM:")
    print(f"  Decimal: {odds['away_odds']['decimal']}")
    print(f"  American: {odds['away_odds']['american']}")
    print(f"  Fractional: {odds['away_odds']['fractional']}")
    print(f"  Fair Value: {odds['away_odds']['fair_decimal']} ({odds['away_odds']['value']:+.1f}% value)")
    print(f"  EV: {odds['away_odds']['expected_value']['ev_percentage']:+.2f}%")

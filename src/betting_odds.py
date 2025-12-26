"""
Betting Odds Calculator
Calculates proper betting odds from probabilities
"""

def calculate_betting_odds(probability: float, margin: float = 0.05) -> dict:
    """
    Calculate betting odds from win probability.

    Args:
        probability: Win probability (0-1)
        margin: Bookmaker margin/vig (default 5%)

    Returns:
        dict with decimal, fractional, and American odds
    """
    if probability <= 0 or probability >= 1:
        return {
            'decimal': 99.00,
            'fractional': '98/1',
            'american': '+9900'
        }

    # Apply bookmaker margin (vig)
    # Real bookmakers reduce payouts by adding margin
    implied_prob = probability * (1 + margin)
    if implied_prob > 0.95:
        implied_prob = 0.95  # Cap at 95%

    # Decimal odds (European format) - what you get for 1€ bet
    decimal = 1 / implied_prob

    # Fractional odds (UK format)
    fractional_value = decimal - 1
    if fractional_value < 1:
        # e.g., 1/2, 2/5
        denominator = round(1 / fractional_value)
        fractional = f"1/{denominator}"
    else:
        # e.g., 2/1, 5/2
        numerator = round(fractional_value * 2)
        fractional = f"{numerator}/2"

    # American odds (US format)
    if decimal >= 2.0:
        # Underdog: positive odds
        american_value = round((decimal - 1) * 100)
        american = f"+{american_value}"
    else:
        # Favorite: negative odds
        american_value = round(-100 / (decimal - 1))
        american = f"{american_value}"

    return {
        'decimal': round(decimal, 2),
        'fractional': fractional,
        'american': american,
        'implied_probability': round(implied_prob * 100, 1)
    }


def get_fair_odds(probability: float) -> float:
    """
    Calculate fair decimal odds without bookmaker margin.

    This is what the TRUE odds should be based on probability.
    """
    if probability <= 0 or probability >= 1:
        return 99.00

    return round(1 / probability, 2)


def compare_with_market(our_probability: float, market_decimal_odds: float) -> dict:
    """
    Compare our model's probability with market odds.

    Returns value bet analysis.
    """
    # Market's implied probability
    market_prob = 1 / market_decimal_odds

    # Our fair odds
    our_odds = get_fair_odds(our_probability)

    # Value calculation
    # Positive = we think it's more likely than market
    # Negative = we think it's less likely than market
    value = our_probability - market_prob

    has_value = value > 0.05  # 5% edge required for value bet

    return {
        'our_probability': round(our_probability * 100, 1),
        'market_probability': round(market_prob * 100, 1),
        'our_odds': our_odds,
        'market_odds': market_decimal_odds,
        'value_diff': round(value * 100, 1),
        'has_value': has_value,
        'recommendation': 'BET' if has_value else 'PASS'
    }

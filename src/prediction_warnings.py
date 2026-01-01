"""
Prediction Warning System

Generates warnings when predictions may be unreliable.
Helps users identify when to trust or question model predictions.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class PredictionWarning:
    """A warning about prediction reliability"""
    level: str  # "INFO", "WARNING", "CRITICAL"
    icon: str   # Emoji for UI display
    message: str
    details: str


def generate_warnings(
    model_home_prob: float,
    model_confidence: float,
    market_home_prob: float = None,
    home_team_elo: float = 1500,
    away_team_elo: float = 1500,
    home_recent_form: float = 0.5,
    away_recent_form: float = 0.5,
    home_team: str = "",
    away_team: str = ""
) -> List[PredictionWarning]:
    """
    Generate all applicable warnings for a prediction.

    Args:
        model_home_prob: Model's home win probability (0-1)
        model_confidence: Model's confidence (0-1)
        market_home_prob: Market's home win probability from odds (0-1, optional)
        home_team_elo: Home team Elo rating
        away_team_elo: Away team Elo rating
        home_recent_form: Home team recent win rate (last 10 games)
        away_recent_form: Away team recent win rate (last 10 games)
        home_team: Home team name
        away_team: Away team name

    Returns:
        List of warnings ordered by severity (CRITICAL first)
    """
    warnings = []

    # ========================================================================
    # WARNING 1: Major market disagreement (CRITICAL)
    # ========================================================================
    if market_home_prob is not None:
        discrepancy = abs(model_home_prob - market_home_prob)

        if discrepancy > 0.25:
            # 25%+ disagreement - model is almost certainly wrong
            model_favorite = home_team if model_home_prob > 0.5 else away_team
            market_favorite = home_team if market_home_prob > 0.5 else away_team

            warnings.append(PredictionWarning(
                level="CRITICAL",
                icon="🚨",
                message=f"Model disagrees with market by {discrepancy:.0%}",
                details=(
                    f"Model favors {model_favorite} ({model_home_prob:.0%}) but "
                    f"market favors {market_favorite} ({market_home_prob:.0%}). "
                    f"Check for injuries, lineup changes, or recent news. "
                    f"Market is usually more accurate."
                )
            ))

        elif discrepancy > 0.15:
            # 15-25% disagreement - significant mismatch
            warnings.append(PredictionWarning(
                level="WARNING",
                icon="⚠️",
                message=f"Model disagrees with market by {discrepancy:.0%}",
                details=(
                    f"Moderate disagreement (Model: {model_home_prob:.0%}, "
                    f"Market: {market_home_prob:.0%}). "
                    f"Verify team news and injury reports before betting."
                )
            ))

    # ========================================================================
    # WARNING 2: High confidence on market underdog (WARNING)
    # ========================================================================
    if market_home_prob is not None and model_confidence > 0.70:
        # Model is confident (>70%) but market disagrees
        model_favorite_home = model_home_prob > 0.5
        market_favorite_home = market_home_prob > 0.5

        if model_favorite_home != market_favorite_home:
            # Model and market pick different teams
            model_pick = home_team if model_favorite_home else away_team
            market_pick = home_team if market_favorite_home else away_team

            warnings.append(PredictionWarning(
                level="WARNING",
                icon="🤔",
                message=f"High confidence ({model_confidence:.0%}) on market underdog",
                details=(
                    f"Model strongly favors {model_pick} but market favors {market_pick}. "
                    f"This could be a value bet OR model error. "
                    f"Research team news and matchup carefully."
                )
            ))

    # ========================================================================
    # WARNING 3: Recent form contradicts season strength (INFO)
    # ========================================================================
    elo_diff = home_team_elo - away_team_elo
    form_diff = home_recent_form - away_recent_form

    # Elo says one team is better, recent form says opposite
    elo_favors_home = elo_diff > 30  # Home team stronger
    form_favors_home = form_diff > 0.15  # Home team hot

    if elo_favors_home != form_favors_home and abs(elo_diff) > 50:
        stronger_team = home_team if elo_diff > 0 else away_team
        hot_team = home_team if form_diff > 0 else away_team

        warnings.append(PredictionWarning(
            level="INFO",
            icon="ℹ️",
            message="Recent form contradicts season strength",
            details=(
                f"{stronger_team} is the stronger team overall (Elo {abs(elo_diff):.0f} advantage) "
                f"but {hot_team} has better recent form. "
                f"Model must decide between long-term quality vs short-term momentum."
            )
        ))

    # ========================================================================
    # WARNING 4: Low model confidence - coin flip (INFO)
    # ========================================================================
    if model_confidence < 0.55:
        warnings.append(PredictionWarning(
            level="INFO",
            icon="🪙",
            message=f"Low model confidence ({model_confidence:.0%})",
            details=(
                "Base models disagree significantly. "
                "Prediction is close to a coin flip. "
                "Consider skipping this game or betting small."
            )
        ))

    # ========================================================================
    # WARNING 5: Very close game - toss-up (INFO)
    # ========================================================================
    if 0.45 <= model_home_prob <= 0.55:
        warnings.append(PredictionWarning(
            level="INFO",
            icon="⚖️",
            message="Very close game (near 50-50)",
            details=(
                f"Model sees this as nearly even ({model_home_prob:.0%} vs {1-model_home_prob:.0%}). "
                "Close games are unpredictable. Small edges can decide outcome."
            )
        ))

    # ========================================================================
    # WARNING 6: Extreme confidence without market data (WARNING)
    # ========================================================================
    if market_home_prob is None and (model_home_prob > 0.80 or model_home_prob < 0.20):
        favorite = home_team if model_home_prob > 0.5 else away_team
        warnings.append(PredictionWarning(
            level="WARNING",
            icon="⚡",
            message=f"Extreme confidence ({model_home_prob:.0%}) without market validation",
            details=(
                f"Model strongly favors {favorite} but we have no real odds to verify. "
                "Add betting API key to see if market agrees."
            )
        ))

    # ========================================================================
    # WARNING 7: Perfect Elo match - uncertainty (INFO)
    # ========================================================================
    if abs(elo_diff) < 20:
        warnings.append(PredictionWarning(
            level="INFO",
            icon="🎲",
            message="Teams are evenly matched by Elo",
            details=(
                f"Elo difference is only {abs(elo_diff):.0f} points (very close). "
                "Outcome depends heavily on recent form, injuries, and home court."
            )
        ))

    # Sort by severity: CRITICAL -> WARNING -> INFO
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    warnings.sort(key=lambda w: severity_order[w.level])

    return warnings


def get_warning_color(level: str) -> str:
    """Get Streamlit color for warning level"""
    return {
        "CRITICAL": "#dc2626",  # Red
        "WARNING": "#ea580c",   # Orange
        "INFO": "#2563eb"       # Blue
    }.get(level, "#64748b")


def format_warning_for_display(warning: PredictionWarning) -> str:
    """Format warning as HTML for Streamlit"""
    color = get_warning_color(warning.level)

    return f"""
    <div style="
        background: {color}15;
        border-left: 4px solid {color};
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    ">
        <div style="font-weight: 600; color: {color}; margin-bottom: 4px;">
            {warning.icon} {warning.message}
        </div>
        <div style="font-size: 0.9em; color: #64748b;">
            {warning.details}
        </div>
    </div>
    """


# Example usage
if __name__ == "__main__":
    # Test case: Detroit vs Miami scenario
    warnings = generate_warnings(
        model_home_prob=0.80,  # Model says Detroit 80%
        model_confidence=0.75,
        market_home_prob=0.22,  # Market says Detroit 22%
        home_team_elo=1480,  # Detroit weaker
        away_team_elo=1580,  # Miami stronger
        home_recent_form=0.70,  # Detroit hot (7-3)
        away_recent_form=0.40,  # Miami cold (4-6)
        home_team="Detroit Pistons",
        away_team="Miami Heat"
    )

    print("Warnings for Detroit vs Miami:")
    for warning in warnings:
        print(f"\n[{warning.level}] {warning.icon} {warning.message}")
        print(f"  {warning.details}")

    # Expected output:
    # [CRITICAL] 🚨 Model disagrees with market by 58%
    #   Model favors Detroit Pistons (80%) but market favors Miami Heat (78%). ...
    # [WARNING] 🤔 High confidence (75%) on market underdog
    #   Model strongly favors Detroit Pistons but market favors Miami Heat. ...
    # [INFO] ℹ️ Recent form contradicts season strength
    #   Miami Heat is the stronger team overall (Elo 100 advantage) ...

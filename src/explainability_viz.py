"""
Explainability Visualization Module
Creates rich visualizations for model predictions and feature importance
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Orange color scheme for Twitter branding
ORANGE_PRIMARY = '#f97316'      # Main orange
ORANGE_SECONDARY = '#fb923c'    # Lighter orange
ORANGE_LIGHT = '#fdba74'        # Light orange
ORANGE_DARK = '#ea580c'         # Dark orange
ORANGE_COMPLEMENT = '#fbbf24'   # Amber/orange complement
GRAY_NEUTRAL = '#64748b'        # Neutral gray
GRAY_LIGHT = '#cbd5e1'          # Light gray

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def create_shap_waterfall(top_factors: List[Dict], home_team: str, away_team: str) -> go.Figure:
    """
    Create a SHAP waterfall plot showing feature contributions.
    
    Args:
        top_factors: List of dicts with 'feature', 'impact', 'direction'
        home_team: Name of home team
        away_team: Name of away team
    
    Returns:
        Plotly figure object
    """
    if not top_factors:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Prepare data
    features = [f['feature'].replace('_', ' ').title() for f in top_factors]
    impacts = [f['impact'] for f in top_factors]
    
    # Create color mapping (positive = home advantage, negative = away advantage) - Orange theme
    colors = [ORANGE_PRIMARY if imp > 0 else ORANGE_DARK for imp in impacts]
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Feature Impact",
        orientation="h",
        measure=["relative"] * len(features),
        y=features[::-1],  # Reverse to show most important at top
        x=impacts[::-1],
        connector={"mode": "between", "line": {"width": 1, "color": "#e2e8f0"}},
        decreasing={"marker": {"color": ORANGE_DARK}},
        increasing={"marker": {"color": ORANGE_PRIMARY}},
        totals={"marker": {"color": ORANGE_SECONDARY}}
    ))
    
    fig.update_layout(
        title={
            'text': f"Feature Impact Analysis<br><sub>{away_team} @ {home_team}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Impact on Home Win Probability",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=80, b=40),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor=GRAY_LIGHT,
            gridcolor='#f1f5f9'
        ),
        yaxis=dict(
            gridcolor='#f1f5f9'
        )
    )
    
    return fig


def create_feature_importance_chart(top_factors: List[Dict], home_team: str = "", away_team: str = "") -> go.Figure:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        top_factors: List of dicts with 'feature', 'impact', 'direction'
    
    Returns:
        Plotly figure object
    """
    if not top_factors:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Prepare data
    df = pd.DataFrame(top_factors)
    df['feature_name'] = df['feature'].apply(lambda x: x.replace('_', ' ').title())
    df['abs_impact'] = df['impact'].abs()
    df = df.sort_values('abs_impact', ascending=True)
    
    # Color by direction
    df['color'] = df['impact'].apply(lambda x: ORANGE_PRIMARY if x > 0 else ORANGE_DARK)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['feature_name'],
        x=df['abs_impact'],
        orientation='h',
        marker=dict(
            color=df['color'],
            line=dict(width=0)
        ),
        text=[f"{x*100:+.2f}%" for x in df['impact']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: %{text}<extra></extra>'
    ))
    
    # Add legend showing what colors mean
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=12, color=ORANGE_PRIMARY),
        name='Favors Home Team',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=12, color=ORANGE_DARK),
        name='Favors Away Team',
        showlegend=True
    ))
    
    fig.update_layout(
        title={
            'text': "Top 10 Feature Importance",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Impact on Home Win Probability",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(tickformat=".2%", gridcolor='#f1f5f9'),
        yaxis=dict(gridcolor='#f1f5f9')
    )
    
    return fig


def create_model_consensus_chart(base_model_predictions: Dict[str, float], 
                                  final_prob: float) -> go.Figure:
    """
    Create a chart showing predictions from different models.
    
    Args:
        base_model_predictions: Dict mapping model name to home win probability
        final_prob: Final ensemble probability
    
    Returns:
        Plotly figure object
    """
    if not base_model_predictions:
        fig = go.Figure()
        fig.add_annotation(
            text="No model consensus data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Prepare data
    models = list(base_model_predictions.keys())
    probs = [base_model_predictions[m] * 100 for m in models]
    
    # Add ensemble prediction
    models.append('Ensemble')
    probs.append(final_prob * 100)
    
    # Color coding
    colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#059669']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=probs,
        marker=dict(
            color=colors[:len(models)],
            line=dict(width=0)
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Home Win Prob: %{y:.1f}%<extra></extra>'
    ))
    
    # Add 50% reference line
    fig.add_hline(
        y=50, 
        line_dash="dash", 
        line_color="#cbd5e1",
        annotation_text="50% (Toss-up)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title={
            'text': "Model Consensus",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="",
        yaxis_title="Home Win Probability (%)",
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor='white',
        showlegend=False,
        yaxis=dict(
            range=[0, 100],
            gridcolor='#f1f5f9'
        ),
        xaxis=dict(gridcolor='#f1f5f9')
    )
    
    return fig


def create_player_contribution_chart(player_stats: List[Dict]) -> go.Figure:
    """
    Create a chart showing player contributions to team metrics.
    
    Args:
        player_stats: List of dicts with player names and stats
    
    Returns:
        Plotly figure object
    """
    if not player_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="No player data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    df = pd.DataFrame(player_stats)
    
    # Get top 10 scorers
    df = df.nlargest(10, 'ppg')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['player_name'],
        x=df['ppg'],
        orientation='h',
        name='PPG',
        marker=dict(color='#2563eb'),
        text=[f"{x:.1f}" for x in df['ppg']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title={
            'text': "Top 10 Scorers",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Points Per Game",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(gridcolor='#f1f5f9'),
        yaxis=dict(gridcolor='#f1f5f9')
    )
    
    return fig


def format_feature_name(feature: str) -> str:
    """Format feature name for display."""
    # Common abbreviations
    replacements = {
        'ppg': 'PPG',
        'rpg': 'RPG',
        'apg': 'APG',
        'fg': 'FG',
        'pct': '%',
        'elo': 'Elo',
        'last10': 'L10',
        'last5': 'L5',
        'win': 'Win',
        'home': 'Home',
        'away': 'Away',
        'opp': 'Opp',
        'diff': 'Diff',
        'h2h': 'H2H',
        'reb': 'Reb',
        'ast': 'Ast',
        'tov': 'TOV'
    }
    
    parts = feature.split('_')
    formatted = []
    
    for part in parts:
        if part.lower() in replacements:
            formatted.append(replacements[part.lower()])
        else:
            formatted.append(part.capitalize())
    
    return ' '.join(formatted)


def create_feature_comparison_chart(features: Dict, home_team: str, away_team: str) -> go.Figure:
    """
    Create a radar/spider chart comparing key features between teams.
    
    Args:
        features: Dictionary of feature values
        home_team: Name of home team
        away_team: Name of away team
    
    Returns:
        Plotly figure object
    """
    # Select key features to compare - UPDATED FOR ADVANCED METRICS
    key_features = [
        ('home_elo', 'away_elo', 'Elo Rating'),
        ('home_last10_offensive_rating', 'away_last10_offensive_rating', 'Offensive Rating'),
        ('home_last10_defensive_rating', 'away_last10_defensive_rating', 'Defensive Rating'),
        ('home_last10_win_pct', 'away_last10_win_pct', 'Win % (L10)'),
        ('home_last10_fg3_pct', 'away_last10_fg3_pct', '3P%'),
        ('home_last10_pace', 'away_last10_pace', 'Pace'),
        ('home_team_home_win_pct', 'away_team_road_win_pct', 'Home/Road Win %'),
    ]
    
    categories = []
    home_values = []
    away_values = []
    
    # Collect all values first for normalization
    all_values = []
    for home_feat, away_feat, label in key_features:
        if home_feat in features and away_feat in features:
            home_val = features[home_feat]
            away_val = features[away_feat]
            if home_val is not None and away_val is not None:
                all_values.extend([float(home_val), float(away_val)])
    
    # Calculate normalization range if we have values
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        val_range = max_val - min_val if max_val != min_val else 1
    else:
        min_val, max_val, val_range = 0, 1, 1
    
    for home_feat, away_feat, label in key_features:
        if home_feat in features and away_feat in features:
            categories.append(label)
            home_val = features[home_feat]
            away_val = features[away_feat]
            
            # Normalize values to 0-100 scale for radar chart
            if home_val is not None and away_val is not None:
                home_float = float(home_val)
                away_float = float(away_val)
                
                # Normalize to 0-100 scale
                home_norm = ((home_float - min_val) / val_range * 100) if val_range > 0 else 50
                away_norm = ((away_float - min_val) / val_range * 100) if val_range > 0 else 50
                
                home_values.append(max(0, min(100, home_norm)))
                away_values.append(max(0, min(100, away_norm)))
            else:
                home_values.append(50)  # Neutral value
                away_values.append(50)
    
    if not categories:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature data available for comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=home_values,
        theta=categories,
        fill='toself',
        name=home_team,
        line_color=ORANGE_PRIMARY,
        fillcolor='rgba(249, 115, 22, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=away_values,
        theta=categories,
        fill='toself',
        name=away_team,
        line_color=ORANGE_DARK,
        fillcolor='rgba(234, 88, 12, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#e2e8f0',
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        title={
            'text': f"Team Comparison: {home_team} vs {away_team}<br><sub>Values normalized to 0-100 scale</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=500,
        margin=dict(t=100, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_advanced_metrics_chart(features: Dict, home_team: str, away_team: str) -> go.Figure:
    """
    Create chart showing advanced metrics: Offensive/Defensive Ratings, Pace, 3pt defense.
    Twitter-friendly and focuses on meaningful analytics.

    Args:
        features: Dictionary of feature values
        home_team: Name of home team
        away_team: Name of away team

    Returns:
        Plotly figure object with advanced metrics comparison
    """
    from plotly.subplots import make_subplots

    # Define advanced metric groups
    stat_groups = {
        'Efficiency Ratings': [
            ('home_last10_offensive_rating', 'away_last10_offensive_rating', 'Off Rating'),
            ('home_last10_defensive_rating', 'away_last10_defensive_rating', 'Def Rating'),
            ('home_last10_net_rating', 'away_last10_net_rating', 'Net Rating'),
        ],
        '3-Point Game': [
            ('home_last10_fg3_pct', 'away_last10_fg3_pct', '3P%'),
            ('home_last10_opp_fg3_pct', 'away_last10_opp_fg3_pct', 'Opp 3P% (Defense)'),
            ('home_last10_three_point_rate', 'away_last10_three_point_rate', '3P Rate'),
        ],
        'Pace & Tempo': [
            ('home_last10_pace', 'away_last10_pace', 'Pace (L10)'),
            ('home_last5_pace', 'away_last5_pace', 'Pace (L5)'),
        ]
    }

    # Create subplots: 2 rows x 2 columns (3rd column space for pace)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(stat_groups.keys()),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    subplot_positions = [
        (1, 1),  # Efficiency Ratings
        (1, 2),  # 3-Point Game
        (2, 1),  # Pace & Tempo
    ]

    for idx, (group_name, stats) in enumerate(stat_groups.items()):
        if idx >= len(subplot_positions):
            break

        row, col = subplot_positions[idx]

        stat_labels = []
        home_values = []
        away_values = []

        for home_feat, away_feat, label in stats:
            if home_feat in features and away_feat in features:
                home_val = float(features[home_feat]) if features[home_feat] is not None else 0
                away_val = float(features[away_feat]) if features[away_feat] is not None else 0
                stat_labels.append(label)
                home_values.append(home_val)
                away_values.append(away_val)

        if stat_labels:
            # Home team bars
            fig.add_trace(
                go.Bar(
                    name=home_team,
                    x=stat_labels,
                    y=home_values,
                    marker_color=ORANGE_PRIMARY,
                    text=[f"{v:.1f}" for v in home_values],
                    textposition='outside',
                    textfont=dict(color=ORANGE_PRIMARY, size=11),
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )

            # Away team bars
            fig.add_trace(
                go.Bar(
                    name=away_team,
                    x=stat_labels,
                    y=away_values,
                    marker_color=ORANGE_DARK,
                    text=[f"{v:.1f}" for v in away_values],
                    textposition='outside',
                    textfont=dict(color=ORANGE_DARK, size=11),
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )

            # Update axes
            fig.update_xaxes(tickangle=-45, row=row, col=col)
            max_val = max(max(home_values) if home_values else 0, max(away_values) if away_values else 0)
            min_val = min(min(home_values) if home_values else 0, min(away_values) if away_values else 0)
            padding = (max_val - min_val) * 0.25 if max_val != min_val else max_val * 0.15

            fig.update_yaxes(
                title_text="Value",
                row=row,
                col=col,
                range=[min_val - padding, max_val + padding * 1.5]  # Extra padding for text
            )

    fig.update_layout(
        title={
            'text': f"Advanced Metrics: {home_team} vs {away_team}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1e293b'}
        },
        barmode='group',
        height=650,
        margin=dict(t=90, b=60, l=50, r=50),
        showlegend=True,
        plot_bgcolor='#f8fafc',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )

    return fig


def create_stats_comparison_bars(features: Dict, home_team: str, away_team: str) -> go.Figure:
    """
    Create side-by-side bar chart comparing team statistics.
    Uses subplots to handle different scales properly.
    NOW FOCUSES ON: Win%, FG%, Rebounds, Assists (no PPG!)

    Args:
        features: Dictionary of feature values
        home_team: Name of home team
        away_team: Name of away team

    Returns:
        Plotly figure object with subplots for different scales
    """
    # Group stats by similar scales - REMOVED PPG, ADDED MORE RELEVANT STATS
    stat_groups_scaled = {
        'Percentages': [
            ('home_last10_win_pct', 'away_last10_win_pct', 'Win % (L10)'),
            ('home_last10_fg_pct', 'away_last10_fg_pct', 'FG% (L10)'),
            ('home_last10_fg3_pct', 'away_last10_fg3_pct', '3P% (L10)'),
            ('home_team_home_win_pct', 'away_team_road_win_pct', 'Home/Road Win %'),
        ],
        'Team Play': [
            ('home_last10_reb', 'away_last10_reb', 'Rebounds (L10)'),
            ('home_last10_ast', 'away_last10_ast', 'Assists (L10)'),
            ('home_last10_tov', 'away_last10_tov', 'Turnovers (L10)'),
        ],
        'Recent Form': [
            ('home_last10_point_diff', 'away_last10_point_diff', 'Point Diff (L10)'),
            ('home_streak', 'away_streak', 'Current Streak'),
        ],
        'Elo Rating': [
            ('home_elo', 'away_elo', 'Elo Rating'),
        ]
    }
    
    # Create subplots: 2 rows x 2 columns
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(stat_groups_scaled.keys()),
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    subplot_positions = [
        (1, 1),  # Percentages
        (1, 2),  # Scoring
        (2, 1),  # Differentials
        (2, 2),  # Ratings
    ]
    
    for idx, (group_name, stats) in enumerate(stat_groups_scaled.items()):
        row, col = subplot_positions[idx]
        
        group_stats = []
        stat_labels = []
        home_values = []
        away_values = []
        
        for home_feat, away_feat, label in stats:
            if home_feat in features and away_feat in features:
                home_val = float(features[home_feat]) if features[home_feat] is not None else 0
                away_val = float(features[away_feat]) if features[away_feat] is not None else 0
                group_stats.append((home_val, away_val))
                stat_labels.append(label)
                home_values.append(home_val)
                away_values.append(away_val)
        
        if group_stats:
            # Home team bars - use 'inside' text position when values are large
            fig.add_trace(
                go.Bar(
                    name=home_team,
                    x=stat_labels,
                    y=home_values,
                    marker_color=ORANGE_PRIMARY,
                    text=[f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}" for v in home_values],
                    textposition='inside',
                    textfont=dict(color='white', size=10),
                    insidetextanchor='middle',
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Away team bars
            fig.add_trace(
                go.Bar(
                    name=away_team,
                    x=stat_labels,
                    y=away_values,
                    marker_color=ORANGE_DARK,
                    text=[f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}" for v in away_values],
                    textposition='inside',
                    textfont=dict(color='white', size=10),
                    insidetextanchor='middle',
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Update subplot layout - add padding for text labels
            fig.update_xaxes(tickangle=-45, row=row, col=col)
            
            # Calculate max value and add MORE padding to prevent text clipping
            max_val = max(max(home_values) if home_values else 0, max(away_values) if away_values else 0)
            min_val = min(min(home_values) if home_values else 0, min(away_values) if away_values else 0)
            range_val = max_val - min_val if max_val != min_val else max(abs(max_val), abs(min_val)) or 1
            padding = range_val * 0.30  # 30% padding for text labels (increased from 15%)
            
            fig.update_yaxes(
                title_text="Value", 
                row=row, 
                col=col,
                range=[min_val - padding, max_val + padding] if min_val < 0 else [0, max_val + padding]
            )
    
    fig.update_layout(
        title={
            'text': f"Statistics Comparison: {home_team} vs {away_team}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        barmode='group',
        height=700,
        margin=dict(t=80, b=60, l=40, r=40),
        showlegend=True,
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def create_ev_comparison_chart(comparisons: list, home_team: str, away_team: str) -> go.Figure:
    """
    Create a bar chart comparing Expected Value across bookmakers.

    Args:
        comparisons: List of bookmaker comparison dicts from compare_bookmakers()
        home_team: Name of home team
        away_team: Name of away team

    Returns:
        Plotly figure object
    """
    if not comparisons:
        fig = go.Figure()
        fig.add_annotation(
            text="No odds data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    # Prepare data
    bookmakers = [c['bookmaker'] for c in comparisons]
    home_ev = [c['home_odds']['expected_value']['ev_percentage'] for c in comparisons]
    away_ev = [c['away_odds']['expected_value']['ev_percentage'] for c in comparisons]

    fig = go.Figure()

    # Home team EV bars
    fig.add_trace(go.Bar(
        name=f'{home_team} (Home)',
        x=bookmakers,
        y=home_ev,
        marker_color=['#059669' if ev > 0 else '#dc2626' for ev in home_ev],
        text=[f'{ev:+.1f}%' for ev in home_ev],
        textposition='outside'
    ))

    # Away team EV bars
    fig.add_trace(go.Bar(
        name=f'{away_team} (Away)',
        x=bookmakers,
        y=away_ev,
        marker_color=['#3b82f6' if ev > 0 else '#f59e0b' for ev in away_ev],
        text=[f'{ev:+.1f}%' for ev in away_ev],
        textposition='outside'
    ))

    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="#374151",
        line_width=2
    )

    # Add value threshold
    fig.add_hline(
        y=3,
        line_dash="dash",
        line_color="#059669",
        annotation_text="Value Threshold (+3%)",
        annotation_position="right"
    )

    fig.update_layout(
        title={
            'text': "Expected Value by Bookmaker",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Bookmaker",
        yaxis_title="Expected Value (%)",
        barmode='group',
        height=400,
        margin=dict(t=60, b=80, l=40, r=40),
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#374151',
            gridcolor='#f1f5f9'
        )
    )

    return fig


def create_model_vs_market_chart(model_prob: float, market_prob: float,
                                  home_team: str, away_team: str) -> go.Figure:
    """
    Create a comparison chart showing model probability vs market implied probability.

    Args:
        model_prob: Model's home win probability
        market_prob: Market implied home win probability
        home_team: Name of home team
        away_team: Name of away team

    Returns:
        Plotly figure object
    """
    categories = [f'{home_team} Win', f'{away_team} Win']
    model_probs = [model_prob * 100, (1 - model_prob) * 100]
    market_probs = [market_prob * 100, (1 - market_prob) * 100]

    fig = go.Figure()

    # Model probabilities
    fig.add_trace(go.Bar(
        name='Our Model',
        x=categories,
        y=model_probs,
        marker_color='#2563eb',
        text=[f'{p:.1f}%' for p in model_probs],
        textposition='outside'
    ))

    # Market probabilities
    fig.add_trace(go.Bar(
        name='Market Implied',
        x=categories,
        y=market_probs,
        marker_color='#9ca3af',
        text=[f'{p:.1f}%' for p in market_probs],
        textposition='outside'
    ))

    # Add 50% reference
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="#cbd5e1",
        annotation_text="50%",
        annotation_position="right"
    )

    fig.update_layout(
        title={
            'text': "Model vs Market Probability",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="",
        yaxis_title="Probability (%)",
        barmode='group',
        height=350,
        margin=dict(t=60, b=40, l=40, r=40),
        plot_bgcolor='white',
        showlegend=True,
        yaxis=dict(
            range=[0, 100],
            gridcolor='#f1f5f9'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_feature_categories_table(features: Dict, home_team: str, away_team: str) -> pd.DataFrame:
    """
    Create a comprehensive feature comparison table organized by categories.
    
    Args:
        features: Dictionary of feature values
        home_team: Name of home team
        away_team: Name of away team
    
    Returns:
        DataFrame with feature comparisons
    """
    categories = {
        'Elo & Ratings': [
            ('home_elo', 'away_elo', 'Elo Rating (Team Strength)'),
            ('elo_diff', None, 'Elo Difference (Home - Away)'),
            ('elo_win_prob', None, 'Elo-Based Win Probability'),
        ],
        'Recent Form (Last 10)': [
            ('home_last10_win_pct', 'away_last10_win_pct', 'Win Percentage'),
            ('home_last10_offensive_rating', 'away_last10_offensive_rating', 'Offensive Rating (Points per 100 Possessions)'),
            ('home_last10_defensive_rating', 'away_last10_defensive_rating', 'Defensive Rating (Opp Points per 100 Poss)'),
            ('home_last10_net_rating', 'away_last10_net_rating', 'Net Rating (Off - Def)'),
            ('home_last10_pace', 'away_last10_pace', 'Pace (Possessions per Game)'),
            ('home_last10_fg3_pct', 'away_last10_fg3_pct', '3-Point Shooting Percentage'),
            ('home_last10_opp_fg3_pct', 'away_last10_opp_fg3_pct', 'Opponent 3PT% Allowed (Perimeter Defense)'),
            ('home_last10_ppg', 'away_last10_ppg', 'Points Per Game'),
            ('home_last10_opp_ppg', 'away_last10_opp_ppg', 'Opponent Points Per Game'),
            ('home_last10_fg_pct', 'away_last10_fg_pct', 'Field Goal Percentage'),
            ('home_last10_reb', 'away_last10_reb', 'Rebounds Per Game'),
            ('home_last10_ast', 'away_last10_ast', 'Assists Per Game'),
            ('home_last10_tov', 'away_last10_tov', 'Turnovers Per Game'),
        ],
        'Recent Form (Last 5)': [
            ('home_last5_win_pct', 'away_last5_win_pct', 'Win Percentage'),
            ('home_last5_ppg', 'away_last5_ppg', 'Points Per Game'),
            ('home_last5_point_diff', 'away_last5_point_diff', 'Average Point Differential'),
            ('home_last5_fg_pct', 'away_last5_fg_pct', 'Field Goal Percentage'),
        ],
        'Home/Away Splits': [
            ('home_team_home_win_pct', 'away_team_road_win_pct', 'Win Percentage (Home vs Road)'),
            ('home_team_home_ppg', 'away_team_road_ppg', 'Points Per Game (Home vs Road)'),
            ('home_team_home_point_diff', 'away_team_road_point_diff', 'Point Differential (Home vs Road)'),
            ('home_team_home_fg_pct', 'away_team_road_fg_pct', 'Field Goal % (Home vs Road)'),
        ],
        'Situational': [
            ('home_rest_days', 'away_rest_days', 'Days of Rest'),
            ('rest_advantage', None, 'Rest Advantage (Home - Away Days)'),
            ('home_streak', 'away_streak', 'Current Win/Loss Streak'),
            ('home_back_to_back', 'away_back_to_back', 'Playing Back-to-Back?'),
        ],
        'Head-to-Head': [
            ('h2h_home_win_pct', None, 'Home Team H2H Win Percentage'),
            ('h2h_total_games', None, 'Total Head-to-Head Games Played'),
            ('h2h_avg_point_diff', None, 'Average Point Differential in H2H'),
            ('h2h_home_ppg', 'h2h_away_ppg', 'Points Per Game in H2H Matchups'),
        ]
    }
    
    rows = []
    for category, feature_list in categories.items():
        for home_feat, away_feat, label in feature_list:
            row = {'Category': category, 'Feature': label}
            
            # Helper function to format values consistently
            def format_value(val, feat_name):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                
                val_float = float(val) if isinstance(val, (int, float)) else 0
                
                # Percentages (values between 0-1 that should be displayed as %)
                if any(x in feat_name for x in ['win_pct', 'fg_pct', 'fg3_pct', 'ft_pct', 'prob', '_pct']):
                    # Check if value is already a percentage (0-1) or already in 0-100 range
                    if 0 <= val_float <= 1:
                        return f"{val_float*100:.1f}%"
                    elif 0 <= val_float <= 100:
                        return f"{val_float:.1f}%"
                    else:
                        return f"{val_float:.1f}"
                
                # Ratings (offensive_rating, defensive_rating, net_rating)
                elif 'rating' in feat_name:
                    return f"{val_float:.1f}"
                
                # Pace
                elif 'pace' in feat_name:
                    return f"{val_float:.1f}"
                
                # Elo ratings
                elif 'elo' in feat_name:
                    return f"{val_float:.0f}"
                
                # Counts (ppg, reb, ast, tov, etc.)
                elif any(x in feat_name for x in ['ppg', 'reb', 'ast', 'tov', 'point_diff', 'streak', 'rest_days', 'total_games']):
                    return f"{val_float:.1f}"
                
                # Boolean
                elif 'back_to_back' in feat_name:
                    return "Yes" if val_float == 1 else "No"
                
                # Default
                else:
                    return f"{val_float:.2f}"
            
            if home_feat and home_feat in features:
                row[home_team] = format_value(features[home_feat], home_feat)
            else:
                row[home_team] = "N/A"
            
            if away_feat and away_feat in features:
                row[away_team] = format_value(features[away_feat], away_feat)
            elif home_feat in features and not away_feat:  # Differential feature (like elo_diff, rest_advantage)
                row[away_team] = format_value(features.get(home_feat), home_feat)
            else:
                row[away_team] = "N/A"
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def create_advanced_metrics_comparison(features: Dict, home_team: str, away_team: str) -> go.Figure:
    """
    Create a visualization comparing advanced metrics (Offensive/Defensive Ratings, 3PT%, Pace).
    Replaces tables with graphs for Twitter-friendly display.
    
    Args:
        features: Dictionary of feature values
        home_team: Name of home team
        away_team: Name of away team
    
    Returns:
        Plotly figure object
    """
    # Extract advanced metrics
    metrics_data = {
        'Offensive Rating': {
            'home': features.get('home_last10_offensive_rating', 0),
            'away': features.get('away_last10_offensive_rating', 0),
            'label': 'Points per 100 possessions'
        },
        'Defensive Rating': {
            'home': features.get('home_last10_defensive_rating', 0),
            'away': features.get('away_last10_defensive_rating', 0),
            'label': 'Opponent points per 100'
        },
        'Net Rating': {
            'home': features.get('home_last10_net_rating', 0),
            'away': features.get('away_last10_net_rating', 0),
            'label': 'Off Rating - Def Rating'
        },
        '3PT%': {
            'home': features.get('home_last10_fg3_pct', 0),
            'away': features.get('away_last10_fg3_pct', 0),
            'label': 'Three-point percentage'
        },
        'Opp 3PT%': {
            'home': features.get('home_last10_opp_fg3_pct', 0),
            'away': features.get('away_last10_opp_fg3_pct', 0),
            'label': 'Opponent 3PT% allowed'
        },
        'Pace': {
            'home': features.get('home_last10_pace', 0),
            'away': features.get('away_last10_pace', 0),
            'label': 'Possessions per game'
        }
    }
    
    # Prepare data for grouped bar chart
    categories = list(metrics_data.keys())
    home_values = []
    away_values = []
    
    for metric in categories:
        home_val = metrics_data[metric]['home']
        away_val = metrics_data[metric]['away']
        
        # Convert percentages to display format
        if '3PT%' in metric or 'Opp 3PT%' in metric:
            home_values.append(float(home_val) * 100 if home_val else 0)
            away_values.append(float(away_val) * 100 if away_val else 0)
        else:
            home_values.append(float(home_val) if home_val else 0)
            away_values.append(float(away_val) if away_val else 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=home_team,
        x=categories,
        y=home_values,
        marker_color=ORANGE_PRIMARY,
        text=[f"{v:.1f}" for v in home_values],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name=away_team,
        x=categories,
        y=away_values,
        marker_color=ORANGE_SECONDARY,
        text=[f"{v:.1f}" for v in away_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title={
            'text': f"Advanced Metrics Comparison<br><sub>Last 10 Games</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="",
        yaxis_title="Rating / Percentage",
        barmode='group',
        height=400,
        margin=dict(t=80, b=40, l=40, r=40),
        plot_bgcolor='white',
        showlegend=True,
        yaxis=dict(gridcolor='#f1f5f9'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def format_injury_tweet(home_team: str, away_team: str, features: Dict, home_team_id: int = None, away_team_id: int = None) -> Optional[str]:
    """
    Generate a tweet about injuries if there are any significant injuries.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        features: Features dictionary with injury data
        home_team_id: Optional home team ID for fetching live injury data
        away_team_id: Optional away team ID for fetching live injury data

    Returns:
        Tweet text or None if no significant injuries
    """
    # Try to get injury features from features dict first
    home_injured_starters = features.get('home_injured_starters', 0)
    away_injured_starters = features.get('away_injured_starters', 0)
    home_star_injured = features.get('home_star_injured', 0)
    away_star_injured = features.get('away_star_injured', 0)

    # If we have team IDs, try to get detailed injury info
    injury_details = None
    if home_team_id and away_team_id:
        try:
            from src.injury_tracker import InjuryTracker
            tracker = InjuryTracker()
            home_injuries = tracker.get_team_injuries(home_team_id)
            away_injuries = tracker.get_team_injuries(away_team_id)
            injury_details = {'home': home_injuries, 'away': away_injuries}
        except Exception:
            pass

    # Check if there are any significant injuries
    has_injuries = (home_injured_starters > 0 or away_injured_starters > 0 or
                   home_star_injured or away_star_injured)

    if not has_injuries and not injury_details:
        return None

    lines = [f"🏥 Injury Report: {away_team} @ {home_team}", ""]

    # Home team injuries
    if home_star_injured or home_injured_starters > 0:
        lines.append(f"🔴 {home_team}:")
        if home_star_injured:
            lines.append(f"   ⭐ Star player OUT")
        if home_injured_starters > 0:
            lines.append(f"   👤 {int(home_injured_starters)} starter(s) OUT")

        # Add detailed injury list if available
        if injury_details and injury_details['home']['injuries']:
            for inj in injury_details['home']['injuries'][:3]:  # Max 3 to fit in tweet
                emoji = "⭐" if inj.get('is_star') else "👤"
                lines.append(f"   {emoji} {inj['player']} - {inj['injury']}")
        lines.append("")

    # Away team injuries
    if away_star_injured or away_injured_starters > 0:
        lines.append(f"🔴 {away_team}:")
        if away_star_injured:
            lines.append(f"   ⭐ Star player OUT")
        if away_injured_starters > 0:
            lines.append(f"   👤 {int(away_injured_starters)} starter(s) OUT")

        # Add detailed injury list if available
        if injury_details and injury_details['away']['injuries']:
            for inj in injury_details['away']['injuries'][:3]:  # Max 3 to fit in tweet
                emoji = "⭐" if inj.get('is_star') else "👤"
                lines.append(f"   {emoji} {inj['player']} - {inj['injury']}")
        lines.append("")

    # If no injuries, return None
    if len(lines) <= 2:
        return None

    lines.append("⚠️ Injuries factored into prediction")

    return "\n".join(lines)


def format_prediction_for_twitter(prediction: Dict, features: Dict, include_injuries: bool = False) -> str:
    """
    Format prediction data as Twitter-friendly text with key metrics.
    Designed for easy copy-paste to Twitter.

    Args:
        prediction: Prediction dictionary with winner, confidence, etc.
        features: Features dictionary with team stats
        include_injuries: Whether to include injury info in the main tweet

    Returns:
        Formatted string ready for Twitter
    """
    home_team = prediction.get('home_team', 'Home')
    away_team = prediction.get('away_team', 'Away')
    winner = prediction.get('prediction', 'home')
    predicted_winner = home_team if winner == 'home' else away_team
    confidence = prediction.get('confidence', 0)
    home_prob = prediction.get('home_win_probability', 0)

    # Get key metrics
    home_ortg = features.get('home_last10_offensive_rating', 0)
    away_ortg = features.get('away_last10_offensive_rating', 0)
    home_drtg = features.get('home_last10_defensive_rating', 0)
    away_drtg = features.get('away_last10_defensive_rating', 0)
    home_3pt = features.get('home_last10_fg3_pct', 0) * 100
    away_3pt = features.get('away_last10_fg3_pct', 0) * 100
    home_opp_3pt = features.get('home_last10_opp_fg3_pct', 0) * 100
    away_opp_3pt = features.get('away_last10_opp_fg3_pct', 0) * 100
    home_pace = features.get('home_last10_pace', 0)
    away_pace = features.get('away_last10_pace', 0)

    # Check if close matchup (difference < 3 points in rating)
    elo_diff = abs(features.get('elo_diff', 0))
    is_close = elo_diff < 100  # Less than ~64% win probability difference

    # Build Twitter post
    lines = [
        f"🏀 NBA Prediction: {away_team} @ {home_team}",
        f"",
        f"📊 Predicted Winner: {predicted_winner} ({confidence:.0%} confidence)",
        f"",
        f"Key Metrics (Last 10 Games):",
        f"",
        f"⚡ Offensive Rating:",
        f"   {home_team}: {home_ortg:.1f}",
        f"   {away_team}: {away_ortg:.1f}",
        f"",
        f"🛡️ Defensive Rating:",
        f"   {home_team}: {home_drtg:.1f}",
        f"   {away_team}: {away_drtg:.1f}",
        f"",
        f"🎯 3PT%:",
        f"   {home_team}: {home_3pt:.1f}%",
        f"   {away_team}: {away_3pt:.1f}%",
        f"",
        f"🔒 Opponent 3PT% Allowed:",
        f"   {home_team}: {home_opp_3pt:.1f}%",
        f"   {away_team}: {away_opp_3pt:.1f}%",
        f"",
        f"⏱️ Pace:",
        f"   {home_team}: {home_pace:.1f} poss/game",
        f"   {away_team}: {away_pace:.1f} poss/game",
    ]

    if is_close:
        lines.append("")
        lines.append("⚠️ Close matchup - Home court advantage less significant")

    # Add injury note if requested and injuries are present
    if include_injuries:
        home_injured_starters = features.get('home_injured_starters', 0)
        away_injured_starters = features.get('away_injured_starters', 0)
        home_star_injured = features.get('home_star_injured', 0)
        away_star_injured = features.get('away_star_injured', 0)

        if home_injured_starters > 0 or away_injured_starters > 0 or home_star_injured or away_star_injured:
            lines.append("")
            lines.append("🏥 Key injuries factored in - see thread for details")

    return "\n".join(lines)


def get_matchup_context(features: Dict, home_team: str, away_team: str) -> str:
    """
    Generate context about the matchup, especially for close games.
    
    Args:
        features: Features dictionary
        home_team: Home team name
        away_team: Away team name
    
    Returns:
        Context string
    """
    elo_diff = abs(features.get('elo_diff', 0))
    home_ortg = features.get('home_last10_offensive_rating', 110)
    away_ortg = features.get('away_last10_offensive_rating', 110)
    
    # Determine if close matchup
    if elo_diff < 100:  # Less than ~3-4 point difference in expected score
        context = f"⚠️ **Matchup très serré** (différence Elo: {elo_diff:.0f} pts)\n\n"
        context += "Dans ce type de match serré, l'avantage du terrain est moins significatif. "
        context += "Les facteurs clés sont la forme récente et les performances en fin de match."
        
        # Add specific insights
        if abs(home_ortg - away_ortg) < 2:
            context += "\n\nLes deux équipes ont un niveau offensif très similaire."
        
        return context
    
    return ""


def create_comprehensive_dashboard_charts(prediction: Dict, features: Dict, home_team: str, away_team: str) -> Dict[str, go.Figure]:
    """
    Create all comprehensive visualization charts for Twitter.
    Converts ALL table data into visual charts.
    
    Args:
        prediction: Prediction dictionary
        features: Features dictionary
        home_team: Home team name
        away_team: Away team name
    
    Returns:
        Dictionary mapping chart names to Plotly figures
    """
    charts = {}
    
    # 1. Elo Ratings Comparison
    home_elo = features.get('home_elo', 1500)
    away_elo = features.get('away_elo', 1500)
    fig_elo = go.Figure()
    fig_elo.add_trace(go.Bar(
        x=[away_team, home_team],
        y=[away_elo, home_elo],
        marker_color=[ORANGE_DARK, ORANGE_PRIMARY],
        text=[f"{away_elo:.0f}", f"{home_elo:.0f}"],
        textposition='outside'
    ))
    fig_elo.update_layout(
        title="Elo Ratings",
        yaxis_title="Elo Rating",
        height=350,
        plot_bgcolor='white'
    )
    charts['elo'] = fig_elo
    
    # 2. Elo Win Probability Gauge
    elo_prob = features.get('elo_win_prob', 0.5) * 100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=elo_prob,
        number={'suffix': '%'},
        title={'text': f"{home_team} Win Probability (Elo)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': ORANGE_PRIMARY},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(t=60, b=40, l=40, r=40))
    charts['elo_gauge'] = fig_gauge
    
    # 3. Recent Form - Ratings (Last 10)
    fig_ratings = go.Figure()
    metrics = ['Offensive Rating', 'Defensive Rating', 'Net Rating']
    home_vals = [
        features.get('home_last10_offensive_rating', 0),
        features.get('home_last10_defensive_rating', 0),
        features.get('home_last10_net_rating', 0)
    ]
    away_vals = [
        features.get('away_last10_offensive_rating', 0),
        features.get('away_last10_defensive_rating', 0),
        features.get('away_last10_net_rating', 0)
    ]
    fig_ratings.add_trace(go.Bar(name=home_team, x=metrics, y=home_vals, marker_color=ORANGE_PRIMARY))
    fig_ratings.add_trace(go.Bar(name=away_team, x=metrics, y=away_vals, marker_color=ORANGE_SECONDARY))
    fig_ratings.update_layout(
        title="Offensive/Defensive Ratings (Last 10)",
        yaxis_title="Rating",
        barmode='group',
        height=400,
        plot_bgcolor='white'
    )
    charts['ratings_l10'] = fig_ratings
    
    # 4. Shooting & Pace (Last 10)
    fig_shooting = go.Figure()
    shoot_metrics = ['3PT%', 'Opp 3PT% Allowed', 'FG%']
    home_shoot = [
        features.get('home_last10_fg3_pct', 0) * 100,
        features.get('home_last10_opp_fg3_pct', 0) * 100,
        features.get('home_last10_fg_pct', 0) * 100
    ]
    away_shoot = [
        features.get('away_last10_fg3_pct', 0) * 100,
        features.get('away_last10_opp_fg3_pct', 0) * 100,
        features.get('away_last10_fg_pct', 0) * 100
    ]
    fig_shooting.add_trace(go.Bar(name=home_team, x=shoot_metrics, y=home_shoot, marker_color=ORANGE_PRIMARY))
    fig_shooting.add_trace(go.Bar(name=away_team, x=shoot_metrics, y=away_shoot, marker_color=ORANGE_SECONDARY))
    fig_shooting.update_layout(
        title="Shooting & Defense (Last 10)",
        yaxis_title="Percentage (%)",
        barmode='group',
        height=400,
        plot_bgcolor='white'
    )
    charts['shooting_l10'] = fig_shooting
    
    # 5. Pace Comparison
    home_pace = features.get('home_last10_pace', 100)
    away_pace = features.get('away_last10_pace', 100)
    fig_pace = go.Figure()
    fig_pace.add_trace(go.Bar(
        x=[away_team, home_team],
        y=[away_pace, home_pace],
        marker_color=[ORANGE_DARK, ORANGE_PRIMARY],
        text=[f"{away_pace:.1f}", f"{home_pace:.1f}"],
        textposition='outside'
    ))
    fig_pace.update_layout(
        title="Pace (Possessions per Game) - Last 10",
        yaxis_title="Possessions",
        height=350,
        plot_bgcolor='white'
    )
    charts['pace'] = fig_pace
    
    # 6. Home/Away Splits
    fig_splits = go.Figure()
    split_metrics = ['Win %', 'PPG', 'Point Diff']
    home_split = [
        features.get('home_team_home_win_pct', 0) * 100,
        features.get('home_team_home_ppg', 0),
        features.get('home_team_home_point_diff', 0)
    ]
    away_split = [
        features.get('away_team_road_win_pct', 0) * 100,
        features.get('away_team_road_ppg', 0),
        features.get('away_team_road_point_diff', 0)
    ]
    fig_splits.add_trace(go.Bar(name=f"{home_team} (Home)", x=split_metrics, y=home_split, marker_color=ORANGE_PRIMARY))
    fig_splits.add_trace(go.Bar(name=f"{away_team} (Road)", x=split_metrics, y=away_split, marker_color=ORANGE_SECONDARY))
    fig_splits.update_layout(
        title="Home/Away Splits",
        barmode='group',
        height=400,
        plot_bgcolor='white'
    )
    charts['splits'] = fig_splits
    
    # 7. Situational Factors
    home_rest = features.get('home_rest_days', 1)
    away_rest = features.get('away_rest_days', 1)
    home_streak = features.get('home_streak', 0)
    away_streak = features.get('away_streak', 0)
    
    fig_situational = go.Figure()
    fig_situational.add_trace(go.Bar(
        name="Rest Days",
        x=[away_team, home_team],
        y=[away_rest, home_rest],
        marker_color=[ORANGE_DARK, ORANGE_PRIMARY],
        text=[f"{away_rest}d", f"{home_rest}d"],
        textposition='outside'
    ))
    fig_situational.update_layout(
        title="Rest Days & Streaks",
        yaxis_title="Days / Games",
        height=350,
        plot_bgcolor='white'
    )
    # Add streak as second y-axis
    fig_situational.add_trace(go.Scatter(
        name="Current Streak",
        x=[away_team, home_team],
        y=[away_streak, home_streak],
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='black', width=3),
        marker=dict(size=10)
    ))
    fig_situational.update_layout(
        yaxis2=dict(
            title="Streak",
            overlaying='y',
            side='right',
            range=[-5, 5]
        )
    )
    charts['situational'] = fig_situational

    return charts


def create_game_visualization(prediction: Dict) -> Optional[go.Figure]:
    """
    Create a single unified visualization for a game prediction.
    Used by GitHub Actions to generate Twitter thread images.

    Args:
        prediction: Dictionary containing prediction data with keys:
            - home_team, away_team, predicted_winner
            - predicted_home_prob, predicted_away_prob, confidence
            - features (optional): Dict of feature values

    Returns:
        Plotly figure object or None if creation fails
    """
    try:
        home_team = prediction.get('home_team', 'Home')
        away_team = prediction.get('away_team', 'Away')
        winner = prediction.get('predicted_winner', home_team)
        home_prob = prediction.get('predicted_home_prob', 0.5) * 100
        away_prob = prediction.get('predicted_away_prob', 0.5) * 100
        confidence = prediction.get('confidence', 0.5) * 100
        features = prediction.get('features', {})

        # Create subplot figure with key metrics
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar", "colspan": 2}, None]],
            subplot_titles=["", "", "Win Probability"],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # Row 1, Col 1: Winner prediction gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={'suffix': '%', 'font': {'size': 40}},
                title={'text': f"🎯 {winner}", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': ORANGE_PRIMARY},
                    'bgcolor': 'white',
                    'borderwidth': 2,
                    'bordercolor': GRAY_LIGHT,
                    'steps': [
                        {'range': [0, 50], 'color': GRAY_LIGHT},
                        {'range': [50, 75], 'color': ORANGE_LIGHT},
                        {'range': [75, 100], 'color': ORANGE_SECONDARY}
                    ],
                }
            ),
            row=1, col=1
        )

        # Row 1, Col 2: Matchup indicator
        home_odds = prediction.get('home_odds', round(1 / max(prediction.get('predicted_home_prob', 0.5), 0.01), 2))
        away_odds = prediction.get('away_odds', round(1 / max(prediction.get('predicted_away_prob', 0.5), 0.01), 2))

        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=home_odds,
                number={'prefix': f"{home_team}: ", 'font': {'size': 24}},
                delta={'reference': away_odds, 'relative': False, 'prefix': f"{away_team}: "},
                title={'text': "Odds", 'font': {'size': 18}},
            ),
            row=1, col=2
        )

        # Row 2: Win probability bar chart
        fig.add_trace(
            go.Bar(
                x=[away_team, home_team],
                y=[away_prob, home_prob],
                marker_color=[ORANGE_SECONDARY, ORANGE_PRIMARY],
                text=[f"{away_prob:.1f}%", f"{home_prob:.1f}%"],
                textposition='outside',
                textfont={'size': 18, 'color': 'black'}
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title={
                'text': f"🏀 {away_team} @ {home_team}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 28, 'color': ORANGE_DARK}
            },
            height=800,
            width=1200,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=40, r=40, t=100, b=60)
        )

        # Update y-axis for bar chart
        fig.update_yaxes(range=[0, 100], title_text="Probability (%)", row=2, col=1)

        return fig

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to create game visualization: {e}")
        return None

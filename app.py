"""
NBA Predictor - Enhanced Interface
Full-featured: Predictions, Portfolio Tracking, Analytics, Performance
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Check for required dependencies
try:
    from src.predictor import NBAPredictor
    from src.data_fetcher import NBADataFetcher, FeatureEngineer, EloRatingSystem
    from src.odds_scraper import generate_bookmaker_odds, odds_to_american
    from src.model_feedback_system import ModelFeedbackSystem
    from src.injury_tracker import InjuryTracker
    from nba_api.stats.static import teams
    DEPS_OK = True
except ImportError as e:
    DEPS_OK = False
    IMPORT_ERR = str(e)

# Page config
st.set_page_config(page_title="NBA Predictor", page_icon="🏀", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# CSS - Compact & Clean
# =============================================================================
st.markdown("""
<style>
    .main {background: #f8fafc;}
    .block-container {max-width: 1400px; padding: 0.5rem 1rem;}
    h1 {font-size: 1.4rem !important; font-weight: 800 !important; color: #0f172a !important;}
    h2 {font-size: 1.1rem !important; font-weight: 700 !important; color: #1e293b !important;}
    h3 {font-size: 0.95rem !important; font-weight: 600 !important; color: #334155 !important;}

    /* Fix tab styling */
    .stTabs [data-baseweb="tab-list"] {gap: 4px; background-color: transparent;}
    .stTabs [data-baseweb="tab"] {
        background-color: #e2e8f0;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #475569;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-panel"] {background-color: transparent; padding-top: 1rem;}

    /* Selectbox styling - remove blue highlight */
    .stSelectbox > div > div {
        background-color: white !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background-color: white !important;
    }

    .game-card {
        background: white;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 1px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }

    .stat-box {
        background: white;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
    }
    .stat-value {font-size: 1.4rem; font-weight: 700; color: #0f172a;}
    .stat-label {font-size: 0.7rem; color: #64748b; text-transform: uppercase;}
    .stat-value.green {color: #059669;}
    .stat-value.blue {color: #2563eb;}
    .stat-value.orange {color: #d97706;}

    .factor-home {
        background: linear-gradient(90deg, #f0fdf4 0%, white 100%);
        border-left: 3px solid #22c55e;
        border-radius: 0 6px 6px 0;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
    }
    .factor-away {
        background: linear-gradient(90deg, #fef2f2 0%, white 100%);
        border-left: 3px solid #ef4444;
        border-radius: 0 6px 6px 0;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
    }
    .factor-text {font-weight: 600; color: #1e293b; font-size: 0.85rem;}
    .factor-impact {font-size: 0.7rem; color: #64748b;}

    .conf-high {background: #dcfce7; color: #166534; padding: 3px 10px; border-radius: 15px; font-size: 0.75rem; font-weight: 600;}
    .conf-med {background: #fef3c7; color: #92400e; padding: 3px 10px; border-radius: 15px; font-size: 0.75rem; font-weight: 600;}
    .conf-low {background: #fee2e2; color: #991b1b; padding: 3px 10px; border-radius: 15px; font-size: 0.75rem; font-weight: 600;}

    .portfolio-card {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin-bottom: 0.75rem;
    }
    .portfolio-loss {background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);}

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .parlay-info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }

    /* Compact sidebar */
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 1rem !important;
    }
    
    /* Improve expander behavior - prevent auto-scroll to bottom */
    .streamlit-expanderHeader {
        scroll-margin-top: 80px;
    }
    .streamlit-expanderContent {
        scroll-margin-top: 80px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def safe_rerun():
    """Rerun the app - compatible with different Streamlit versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except:
            pass  # Silent fail - page will update on next interaction

def safe_dataframe(df, **kwargs):
    """Display dataframe compatible with all Streamlit versions"""
    try:
        st.dataframe(df, use_container_width=True, **kwargs)
    except TypeError:
        try:
            st.dataframe(df, **kwargs)
        except:
            st.table(df)

def safe_plotly_chart(fig, **kwargs):
    """Display plotly chart compatible with all Streamlit versions"""
    try:
        st.plotly_chart(fig, use_container_width=True, **kwargs)
    except TypeError:
        try:
            st.plotly_chart(fig, **kwargs)
        except:
            st.write("Chart display error - please upgrade Streamlit")

# =============================================================================
# CONSTANTS
# =============================================================================
RELIABLE_BOOKMAKERS = ['Pinnacle', 'Bet365', 'DraftKings']

# =============================================================================
# DATABASE HELPER FUNCTIONS
# =============================================================================
def init_database(db_path: str):
    """Initialize all database tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            bet_type TEXT,
            games TEXT,
            stake REAL,
            potential_payout REAL,
            combined_odds REAL,
            status TEXT DEFAULT 'pending',
            actual_payout REAL DEFAULT 0,
            settled_at TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_balance (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance REAL DEFAULT 1000.0,
            total_wagered REAL DEFAULT 0,
            total_won REAL DEFAULT 0,
            total_lost REAL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("INSERT OR IGNORE INTO user_balance (id, balance) VALUES (1, 1000.0)")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            game_date TEXT,
            game_id TEXT,
            home_team TEXT,
            away_team TEXT,
            predicted_winner TEXT,
            predicted_home_prob REAL,
            predicted_away_prob REAL,
            confidence REAL,
            actual_winner TEXT,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            correct INTEGER,
            prediction_error REAL,
            calibration_error REAL,
            features_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add missing columns for existing DBs
    for col in ['prediction_error REAL', 'calibration_error REAL', 'features_json TEXT']:
        try:
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
        except:
            pass

    conn.commit()
    conn.close()

def get_user_balance(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT balance, total_wagered, total_won, total_lost FROM user_balance WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    return {'balance': row[0], 'wagered': row[1], 'won': row[2], 'lost': row[3]} if row else {'balance': 1000.0, 'wagered': 0, 'won': 0, 'lost': 0}

def update_user_balance(db_path, amount, bet_type='wager'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if bet_type == 'wager':
        cursor.execute("UPDATE user_balance SET balance = balance - ?, total_wagered = total_wagered + ? WHERE id = 1", (amount, amount))
    elif bet_type == 'win':
        cursor.execute("UPDATE user_balance SET balance = balance + ?, total_won = total_won + ? WHERE id = 1", (amount, amount))
    elif bet_type == 'reset':
        cursor.execute("UPDATE user_balance SET balance = 1000.0, total_wagered = 0, total_won = 0, total_lost = 0 WHERE id = 1")
    conn.commit()
    conn.close()

def place_bet(db_path, games_info, stake, combined_odds, bet_type='single'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO portfolio (bet_type, games, stake, potential_payout, combined_odds, status)
        VALUES (?, ?, ?, ?, ?, 'pending')
    """, (bet_type, json.dumps(games_info), stake, stake * combined_odds, combined_odds))
    conn.commit()
    conn.close()
    update_user_balance(db_path, stake, 'wager')

def get_portfolio_history(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM portfolio ORDER BY created_at DESC LIMIT 50", conn)
    conn.close()
    return df

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    Handles numpy int64, float64, int32, float32, bool_, ndarray, etc.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    else:
        return obj

def save_prediction_to_db(db_path, prediction_data):
    """Save or update prediction in database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Convert features to JSON-safe format
    features = prediction_data.get('features', {})
    try:
        # Convert numpy types to native Python types
        features_converted = convert_numpy_types(features)
        features_json = json.dumps(features_converted)
    except (TypeError, ValueError) as e:
        print(f"⚠️ Error serializing features to JSON: {e}")
        # If serialization fails, save empty dict but log the error
        features_json = json.dumps({})
        print(f"   Saving prediction without features. Feature types: {[type(v).__name__ for v in list(features.values())[:5]]}")

    # Use INSERT OR REPLACE to handle duplicates
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO predictions (
                prediction_date, game_date, home_team, away_team,
                predicted_winner, predicted_home_prob, predicted_away_prob,
                confidence, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            prediction_data['game_date'],
            prediction_data['home_team'],
            prediction_data['away_team'],
            prediction_data['predicted_winner'],
            float(prediction_data['predicted_home_prob']),
            float(prediction_data['predicted_away_prob']),
            float(prediction_data['confidence']),
            features_json
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ Error saving prediction to database: {e}")
        raise
    finally:
        conn.close()

# =============================================================================
# SETUP
# =============================================================================
if not DEPS_OK:
    st.error(f"Missing dependencies: {IMPORT_ERR}")
    st.stop()

NBA_TEAMS = sorted([team['full_name'] for team in teams.get_teams()])
TEAM_IDS = {team['full_name']: team['id'] for team in teams.get_teams()}
ID_TO_TEAM = {team['id']: team['full_name'] for team in teams.get_teams()}
db_path = project_root / "data" / "nba_predictor.db"
models_path = project_root / "models"

init_database(str(db_path))

if 'predictor' not in st.session_state:
    st.session_state.predictor = NBAPredictor(db_path=str(db_path), model_dir=str(models_path))
if 'todays_predictions' not in st.session_state:
    st.session_state.todays_predictions = None
if 'selected_bets' not in st.session_state:
    st.session_state.selected_bets = []

def get_conf_badge(conf):
    if conf >= 0.7: return "High", "conf-high"
    if conf >= 0.55: return "Medium", "conf-med"
    return "Low", "conf-low"

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_feature_importance_chart(top_factors, home_team, away_team):
    if not top_factors:
        return None

    features, impacts, colors, impact_pct = [], [], [], []
    for factor in top_factors[:10]:
        nn = factor['feature'].replace('_', ' ').replace('home ', '').replace('away ', '').title()
        nn = nn.replace('Last10', 'L10').replace('Last5', 'L5').replace('Pct', '%')
        features.append(nn)
        impact_val = factor['impact']
        impacts.append(impact_val)
        # Convert to percentage impact on win probability
        impact_pct.append(abs(impact_val) * 100)
        colors.append('#22c55e' if impact_val > 0 else '#ef4444')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features, 
        x=impacts, 
        orientation='h', 
        marker_color=colors,
        text=[f"{abs(i)*100:.2f}%" for i in impacts], 
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impact: %{text} on win probability<extra></extra>'
    ))
    fig.update_layout(
        title=f"Top Feature Impacts<br><sub>Values show change in home win probability (scale: ±{max(impact_pct):.2f}%)</sub>",
        xaxis_title="Impact on Win Probability (%)",
        yaxis_title="",
        height=400,
        showlegend=False,
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#cbd5e1',
            gridcolor='#f1f5f9',
            tickformat='.1%'
        ),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=80, b=40)
    )
    return fig

def create_feature_radar_chart(features, home_team, away_team):
    """
    Create radar chart with ADVANCED METRICS (no PPG!)
    Shows: Win%, Off Rating, Def Rating, 3P%, Rebounds, Elo
    """
    if not features:
        return None

    categories = ['Win %', 'Off Rtg', 'Def Rtg', '3P%', 'Rebounds', 'Elo']
    home_values = [
        features.get('home_last10_win_pct', 0.5) * 100,
        (features.get('home_last10_offensive_rating', 110) - 90) / 0.4,  # Normalize 90-130 to 0-100
        100 - ((features.get('home_last10_defensive_rating', 110) - 90) / 0.4),  # Lower is better, so invert
        features.get('home_last10_fg3_pct', 0.35) * 250,  # Normalize ~0.35 to ~87
        features.get('home_last10_reb', 44),
        (features.get('home_elo', 1500) - 1400) / 3
    ]
    away_values = [
        features.get('away_last10_win_pct', 0.5) * 100,
        (features.get('away_last10_offensive_rating', 110) - 90) / 0.4,
        100 - ((features.get('away_last10_defensive_rating', 110) - 90) / 0.4),
        features.get('away_last10_fg3_pct', 0.35) * 250,
        features.get('away_last10_reb', 44),
        (features.get('away_elo', 1500) - 1400) / 3
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_values, theta=categories, fill='toself', name=home_team, line_color='#22c55e'))
    fig.add_trace(go.Scatterpolar(r=away_values, theta=categories, fill='toself', name=away_team, line_color='#ef4444'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=350)
    return fig

# =============================================================================
# SIDEBAR - Compact
# =============================================================================
with st.sidebar:
    st.markdown("### 🏀 NBA Predictor")

    balance_info = get_user_balance(str(db_path))
    st.markdown(f"""
    <div class="portfolio-card">
        <div style="font-size: 0.8rem; opacity: 0.8;">Balance</div>
        <div style="font-size: 1.5rem; font-weight: 800;">${balance_info['balance']:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    profit = balance_info['won'] - balance_info['lost']
    c1, c2 = st.columns(2)
    c1.metric("Wagered", f"${balance_info['wagered']:,.0f}")
    c2.metric("P/L", f"${profit:+,.0f}")

    st.markdown("---")
    
    # =============================================================================
    # HELP SECTION - Metric Definitions
    # =============================================================================
    with st.expander("📊 Metric Definitions", expanded=False):
        st.markdown("""
        **Win Probability**: The model's estimated chance (0-100%) that the home team will win this game.

        **Confidence Level**:
        - **High (≥70%)**: Model strongly agrees across all base models, prediction is far from 50/50
        - **Medium (55-69%)**: Moderate agreement, some uncertainty
        - **Low (<55%)**: Low agreement between models, close to coin flip

        **Feature Impact**: Shows how much each feature changes the predicted win probability.
        Values like 0.006 mean a 0.6% shift in win probability - small individual impacts add up.

        **Model Consensus**: Individual predictions from each base model (XGBoost, LightGBM, Random Forest, Logistic Regression).
        When all models agree, confidence is higher.
        """)

        st.markdown("---")
        st.markdown("### 📈 Understanding Feature Impact")
        st.markdown("""
        - **Impact Value**: Shows how much each feature changes the predicted win probability
        - **Scale**: Small values (0.002-0.006 = 0.2%-0.6%) are normal - many features contribute, and their combined effect determines the prediction
        - **Positive Values**: Favor the home team winning (green bars)
        - **Negative Values**: Favor the away team winning (red bars)
        - **Why Small?**: With 100+ features, each contributes a small amount. A single feature changing win probability by 1-2% is significant when combined with others
        """)

    with st.expander("📚 Feature Definitions & Categories", expanded=False):
        st.markdown("""
        **1. Elo Ratings**
        - **Elo Rating**: Team strength rating (1500 = average). Higher = stronger team
        - **Elo Difference**: Home Elo - Away Elo. Positive = home team advantage
        - **Elo Win Probability**: Expected win probability based purely on Elo ratings
        
        **2. Recent Form (Last 10 Games)**
        - **Win %**: Percentage of games won in last 10
        - **PPG**: Points per game (offensive output)
        - **Opp PPG**: Opponent points per game (defensive quality)
        - **Point Diff**: Average point differential (PPG - Opp PPG)
        - **FG% / 3P% / FT%**: Shooting percentages
        - **Rebounds / Assists / Turnovers**: Key team stats
        
        **3. Recent Form (Last 5 Games)**
        - Same metrics as Last 10, but more recent (captures current momentum)
        
        **4. Home/Away Splits**
        - **Home Win %**: Team's win percentage when playing at home
        - **Road Win %**: Team's win percentage when playing away
        - Teams often perform differently at home vs on the road
        
        **5. Head-to-Head**
        - **H2H Win %**: Historical win percentage between these teams
        - **Last 3 Results**: Recent matchups (psychological factor)
        - **Avg Point Diff**: Average margin in previous matchups
        
        **6. Situational Factors**
        - **Rest Days**: Days since last game (affects fatigue)
        - **Rest Advantage**: Home rest days - Away rest days
        - **Streak**: Current win/loss streak (positive = wins, negative = losses)
        - **Back-to-Back**: Playing on consecutive days (fatigue indicator)
        
        **7. Advanced Metrics**
        - **Net Rating**: Offensive efficiency - Defensive efficiency
        - **TOV Rate**: Turnover rate (lower is better)
        - **Momentum**: Comparison of Last 5 vs Last 10 (improving vs declining)
        
        **8. Player-Level Stats** (when available)
        - **Top Scorer PPG**: Leading scorer's points per game
        - **Top Playmaker APG**: Leading assist man's assists per game
        - **Active Players**: Number of active roster members
        """)

    st.markdown("---")
    st.markdown("#### 🎫 Bet Slip")

    if st.session_state.selected_bets:
        combined_odds = 1.0
        for bet in st.session_state.selected_bets:
            combined_odds *= bet['odds']
            st.caption(f"**{bet['team']}** @ {bet['odds']:.2f}")

        if len(st.session_state.selected_bets) > 1:
            st.markdown(f'<div class="parlay-info"><strong>PARLAY</strong> - Combined: {combined_odds:.2f}</div>', unsafe_allow_html=True)

        stake = st.number_input("Stake ($)", min_value=1.0, max_value=float(balance_info['balance']), value=10.0, step=5.0)
        st.caption(f"Potential: **${stake * combined_odds:.2f}**")

        c1, c2 = st.columns(2)
        if c1.button("Place Bet"):
            if stake <= balance_info['balance']:
                games_info = [{'team': b['team'], 'game': b['game'], 'odds': b['odds']} for b in st.session_state.selected_bets]
                place_bet(str(db_path), games_info, stake, combined_odds, 'parlay' if len(st.session_state.selected_bets) > 1 else 'single')
                st.success("Bet placed!")
                st.session_state.selected_bets = []
                safe_rerun()
        if c2.button("Clear"):
            st.session_state.selected_bets = []
            safe_rerun()
    else:
        st.caption("Add bets from Today's Games")

    st.markdown("---")
    if st.button("Reset Balance"):
        update_user_balance(str(db_path), 0, 'reset')
        safe_rerun()

# =============================================================================
# DISPLAY GAME
# =============================================================================
def display_game(pred, expanded=False):
    ht, at = pred['home_team'], pred['away_team']
    winner = ht if pred['prediction'] == 'home' else at
    win_prob = pred['home_win_probability'] if pred['prediction'] == 'home' else pred['away_win_probability']
    conf = pred['confidence']
    conf_text, conf_class = get_conf_badge(conf)
    features = pred.get('features', {})
    odds = generate_bookmaker_odds(pred['home_win_probability'])

    st.markdown(f"""
    <div class="game-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 1rem; font-weight: 600;">{at} @ {ht}</div>
                <div style="margin-top: 0.3rem;">
                    <span style="color: #059669; font-weight: 700;">{winner}</span>
                    <span style="color: #64748b; margin-left: 0.5rem;">{win_prob:.1%}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="{conf_class}">{conf_text}</span>
                <div style="font-size: 0.8rem; color: #64748b;">{conf:.0%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add tooltip explaining "@" notation
    st.caption(f"💡 **Note**: '{at} @ {ht}' means {at} is playing at {ht}'s home stadium")
    # Note: We don't use expanded=True to prevent auto-scroll to bottom
    with st.expander(f"📊 Details: {at} @ {ht}", expanded=False):
        # Bet buttons
        c1, c2 = st.columns(2)
        home_odds = odds['bookmakers'].get('Pinnacle', {}).get('home', 2.0)
        away_odds = odds['bookmakers'].get('Pinnacle', {}).get('away', 2.0)

        if c1.button(f"Bet {ht} @ {home_odds:.2f}", key=f"h_{ht}_{at}"):
            st.session_state.selected_bets.append({'team': ht, 'game': f"{at} @ {ht}", 'odds': home_odds})
            safe_rerun()
        if c2.button(f"Bet {at} @ {away_odds:.2f}", key=f"a_{ht}_{at}"):
            st.session_state.selected_bets.append({'team': at, 'game': f"{at} @ {ht}", 'odds': away_odds})
            safe_rerun()

        # Odds
        st.markdown("**Odds (Top 3 Books)**")
        oc1, oc2 = st.columns(2)
        with oc1:
            for book in RELIABLE_BOOKMAKERS:
                if book in odds['bookmakers']:
                    st.caption(f"{book}: {odds['bookmakers'][book]['home']:.2f}")
        with oc2:
            for book in RELIABLE_BOOKMAKERS:
                if book in odds['bookmakers']:
                    st.caption(f"{book}: {odds['bookmakers'][book]['away']:.2f}")

        # Metric definitions removed - now in sidebar for all matches
        
        # Metric and Feature definitions removed - now in sidebar for all matches
        
        # Charts
        if 'top_factors' in pred and pred['top_factors']:
            st.markdown("### 📈 Feature Impact Analysis")
            # Explanation removed - now in sidebar for all matches
            fig = create_feature_importance_chart(pred['top_factors'], ht, at)
            if fig:
                safe_plotly_chart(fig)
            
            # Show detailed breakdown of top factors
            st.markdown("#### 🔍 Top Features Breakdown")
            top_factors_df = pd.DataFrame(pred['top_factors'][:10])
            top_factors_df['Feature'] = top_factors_df['feature'].apply(lambda x: x.replace('_', ' ').title())
            top_factors_df['Impact %'] = (top_factors_df['impact'] * 100).round(3)
            top_factors_df['Effect'] = top_factors_df['impact'].apply(lambda x: "Favors Home" if x > 0 else "Favors Away")
            display_df = top_factors_df[['Feature', 'Impact %', 'Effect']]
            safe_dataframe(display_df, hide_index=True)

        radar_fig = create_feature_radar_chart(features, ht, at)
        if radar_fig:
            st.markdown("### 🎯 Team Comparison")
            st.markdown("""
            <div style="background: #f0fdf4; border-left: 4px solid #059669; padding: 12px; border-radius: 4px; margin-bottom: 15px;">
            <b>📊 Team Comparison Radar Chart:</b> This visualization compares key team metrics side-by-side. Values are normalized to a 0-100 scale for comparison.
            </div>
            """, unsafe_allow_html=True)
            safe_plotly_chart(radar_fig)

            # NEW: Advanced Metrics Chart (Off/Def Rating, Pace, 3pt defense)
            st.markdown("#### 🎯 Advanced Metrics")
            st.info("💡 **Key Analytics**: Offensive/Defensive Ratings show efficiency per 100 possessions (better than PPG). Pace shows game tempo. 3P% Defense shows perimeter defense quality.")
            from src.explainability_viz import create_advanced_metrics_comparison, create_stats_comparison_bars, create_feature_categories_table, format_prediction_for_twitter, get_matchup_context
            advanced_fig = create_advanced_metrics_comparison(features, ht, at)
            if advanced_fig:
                safe_plotly_chart(advanced_fig)
            
            # Close matchup context
            matchup_context = get_matchup_context(features, ht, at)
            if matchup_context:
                st.markdown("---")
                st.markdown(matchup_context)
            
            # Twitter Post Button
            st.markdown("---")
            st.markdown("### 📱 Post to Twitter")
            
            try:
                from src.twitter_integration import (
                    load_credentials_from_env, setup_twitter_api,
                    format_prediction_tweet,
                    create_chart_image, post_tweet_with_image, create_twitter_thread
                )
                from src.explainability_viz import (
                    create_advanced_metrics_comparison,
                    create_comprehensive_dashboard_charts,
                    format_injury_tweet
                )
                import tempfile
                
                # Force reload from .env file FIRST (before loading credentials)
                import os
                from dotenv import load_dotenv
                load_dotenv(override=True)  # Force reload .env
                
                # Load credentials fresh from env (after reload)
                creds = load_credentials_from_env()
                
                # Get dry_run value directly from env (most current)
                dry_run = os.getenv("TW_DRY_RUN", "true").lower() in ("1", "true", "yes")
                
                # Generate tweet text (needed for both preview and posting)
                twitter_text = format_prediction_tweet(pred, features)
                
                # Preview section (using columns instead of expander to avoid nesting)
                preview_col1, preview_col2 = st.columns([3, 1])
                with preview_col1:
                    st.markdown("**📝 Preview Tweet**")
                with preview_col2:
                    show_preview = st.checkbox("Show Preview", key=f"show_preview_{ht}_{at}", value=False)
                
                if show_preview:
                    st.text_area("Tweet Text", twitter_text, height=150, key=f"preview_{ht}_{at}")
                    
                    # Generate preview chart
                    preview_fig = create_advanced_metrics_comparison(features, ht, at)
                    if preview_fig:
                        st.markdown("**Preview Chart:**")
                        st.plotly_chart(preview_fig, use_container_width=True)
                
                st.markdown("---")
                
                # Posting options
                # Use a stable key and store value in session_state to prevent expander collapse
                post_mode_key = f"post_mode_value_{ht}_{at}"
                if post_mode_key not in st.session_state:
                    st.session_state[post_mode_key] = "Single Tweet (Composite Image)"
                
                post_mode = st.radio(
                    "Post Format:",
                    ["Single Tweet (Composite Image)", "Thread (Multiple Charts)"],
                    index=0 if st.session_state[post_mode_key] == "Single Tweet (Composite Image)" else 1,
                    key=f"post_mode_{ht}_{at}"
                )
                st.session_state[post_mode_key] = post_mode
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🚀 Post to Twitter", key=f"post_twitter_{ht}_{at}"):
                        with st.spinner("Generating charts and posting..."):
                            try:
                                # Verify credentials before proceeding
                                if not all([creds.get('api_key'), creds.get('access_token')]):
                                    st.error("❌ Twitter credentials not fully loaded. Check your .env file.")
                                    return
                                
                                # Setup API clients - Use fresh client to avoid Streamlit caching issues
                                # This ensures we always use the latest credentials from .env
                                from src.twitter_integration import create_fresh_twitter_client
                                try:
                                    api_clients = create_fresh_twitter_client()
                                except Exception as e:
                                    st.error(f"❌ Failed to create Twitter client: {e}")
                                    return
                                
                                # Quick test: verify client can authenticate (test read permission)
                                test_client = api_clients.get("client_v2")
                                if not test_client:
                                    st.error("❌ Failed to create Twitter API client. Check your credentials.")
                                    return
                                
                                # Test if we can read (verifies credentials work)
                                try:
                                    me = test_client.get_me()
                                    st.info(f"✓ Authenticated as: @{me.data.username}")
                                    
                                    # DEBUG: Show which credentials are being used (first 10 chars only)
                                    st.caption(f"🔑 Using API Key: {creds.get('api_key', 'MISSING')[:10]}... | Access Token: {creds.get('access_token', 'MISSING')[:10]}...")
                                    
                                except Exception as auth_error:
                                    st.error(f"❌ Authentication failed: {auth_error}")
                                    return
                                
                                # Generate all charts
                                temp_dir = tempfile.mkdtemp()
                                chart_paths = []
                                
                                if post_mode == "Single Tweet (Composite Image)":
                                    # Create comprehensive dashboard
                                    all_charts = create_comprehensive_dashboard_charts(pred, features, ht, at)
                                    
                                    # Export key charts
                                    key_charts = ['elo', 'ratings_l10', 'shooting_l10']
                                    for chart_name in key_charts:
                                        if chart_name in all_charts:
                                            chart_path = os.path.join(temp_dir, f"{chart_name}.png")
                                            create_chart_image(all_charts[chart_name], chart_path)
                                            chart_paths.append(chart_path)
                                    
                                    # Create composite
                                    if chart_paths:
                                        composite_path = os.path.join(temp_dir, "composite.png")
                                        from src.twitter_integration import create_composite_image
                                        create_composite_image(chart_paths, composite_path, rows=2, cols=2)
                                        
                                        # Post
                                        response = post_tweet_with_image(
                                            api_clients, twitter_text, composite_path,
                                            alt_text=f"NBA prediction charts: {at} @ {ht}",
                                            dry_run=dry_run
                                        )
                                        
                                        if dry_run:
                                            st.info("✅ [DRY RUN] Would post to Twitter. Set TW_DRY_RUN=false in .env and restart Streamlit to actually post.")
                                        else:
                                            st.success("✅ Posted to Twitter successfully!")
                                else:
                                    # Thread mode - Include ALL available charts
                                    all_charts = create_comprehensive_dashboard_charts(pred, features, ht, at)
                                    
                                    # Export ALL charts available
                                    chart_order = ['elo', 'elo_gauge', 'ratings_l10', 'shooting_l10', 'pace', 'splits', 'situational']
                                    exported_charts = []
                                    chart_names_exported = []
                                    
                                    for chart_name in chart_order:
                                        if chart_name in all_charts:
                                            chart_path = os.path.join(temp_dir, f"{chart_name}.png")
                                            try:
                                                create_chart_image(all_charts[chart_name], chart_path)
                                                exported_charts.append(chart_path)
                                                chart_names_exported.append(chart_name)
                                            except Exception as e:
                                                logger.warning(f"Failed to export chart {chart_name}: {e}")
                                    
                                    # Use the same format_twitter_thread method as daily_auto_prediction.py
                                    # Import the DailyPredictionAutomation class to use its format_twitter_thread method
                                    from daily_auto_prediction import DailyPredictionAutomation

                                    # Create a temporary DailyPredictionAutomation instance to use its format_twitter_thread method
                                    temp_daily = DailyPredictionAutomation(
                                        db_path=str(db_path),
                                        model_dir="models",
                                        dry_run=False
                                    )
                                    
                                    # Prepare prediction dict in the format expected by format_twitter_thread
                                    prediction_for_thread = {
                                        'home_team': ht,
                                        'away_team': at,
                                        'prediction': pred['prediction'],  # 'home' or 'away'
                                        'predicted_winner': ht if pred['prediction'] == 'home' else at,
                                        'predicted_team_name': ht if pred['prediction'] == 'home' else at,
                                        'confidence': pred['confidence'],
                                        'home_win_probability': pred['home_win_probability'],
                                        'away_win_probability': pred['away_win_probability'],
                                        'features': features,
                                        'home_team_id': st.session_state.predictor._get_team_id(ht),
                                        'away_team_id': st.session_state.predictor._get_team_id(at),
                                    }
                                    
                                    # Use the same format_twitter_thread method as daily prediction
                                    thread_texts_from_daily, thread_image_paths_from_daily = temp_daily.format_twitter_thread(prediction_for_thread)
                                    
                                    # Use the texts from daily format (which matches the exact template)
                                    # But use our own charts (exported_charts) instead of the ones from daily
                                    thread_texts = thread_texts_from_daily
                                    
                                    # Map thread_texts to our exported charts
                                    # The format_twitter_thread returns tweets in this order:
                                    # [tweet1 (prediction), injury (if any), elo, elo_gauge, ratings, shooting, pace, splits, situational]
                                    # Our chart_order is: ['elo', 'elo_gauge', 'ratings_l10', 'shooting_l10', 'pace', 'splits', 'situational']
                                    
                                    # Check if injury tweet is present (second tweet usually contains "INJURY" or similar)
                                    has_injury_tweet = len(thread_texts) > 1 and ('injury' in thread_texts[1].lower() or 'injured' in thread_texts[1].lower() or 'status' in thread_texts[1].lower())
                                    injury_offset = 1 if has_injury_tweet else 0
                                    
                                    # Map chart names to their positions in thread_texts
                                    # After tweet1 (index 0) and injury (index 1 if exists), charts start
                                    chart_positions = {
                                        'elo': 1 + injury_offset,
                                        'elo_gauge': 2 + injury_offset,
                                        'ratings_l10': 3 + injury_offset,
                                        'shooting_l10': 4 + injury_offset,
                                        'pace': 5 + injury_offset,
                                        'splits': 6 + injury_offset,
                                        'situational': 7 + injury_offset,
                                    }
                                    
                                    # Build filtered lists - only include non-empty tweets
                                    filtered_texts = []
                                    filtered_charts_with_none = []
                                    
                                    for i, tweet_text in enumerate(thread_texts):
                                        if tweet_text.strip():  # Only include non-empty tweets
                                            filtered_texts.append(tweet_text)
                                            
                                            # First tweet (main prediction) and injury tweet have no chart
                                            if i == 0 or (i == 1 and has_injury_tweet):
                                                filtered_charts_with_none.append(None)
                                            else:
                                                # Find which chart corresponds to this tweet index
                                                chart_name = None
                                                for cn, pos in chart_positions.items():
                                                    if i == pos:
                                                        chart_name = cn
                                                        break
                                                
                                                if chart_name and chart_name in chart_names_exported:
                                                    # Find the index in chart_order
                                                    chart_idx_in_order = chart_order.index(chart_name)
                                                    if chart_idx_in_order < len(exported_charts):
                                                        filtered_charts_with_none.append(exported_charts[chart_idx_in_order])
                                                    else:
                                                        filtered_charts_with_none.append(None)
                                                else:
                                                    filtered_charts_with_none.append(None)
                                    
                                    # Post thread (filtered_charts_with_none has None for first tweet, charts for others)
                                    response = create_twitter_thread(
                                        api_clients, filtered_texts, filtered_charts_with_none, dry_run=dry_run
                                    )
                                    
                                    if dry_run:
                                        st.info("✅ [DRY RUN] Would post thread to Twitter. Set TW_DRY_RUN=false in .env and restart Streamlit to actually post.")
                                    else:
                                        # Extract tweet IDs from response
                                        tweet_ids = []
                                        tweet_urls = []
                                        for r in response:
                                            if isinstance(r, dict) and r.get('success'):
                                                tweet_id = r.get('tweet_id')
                                                if tweet_id:
                                                    tweet_ids.append(str(tweet_id))
                                                    tweet_urls.append(f"https://twitter.com/user/status/{tweet_id}")
                                        
                                        if tweet_ids:
                                            st.success(f"✅ Posted thread with {len(response)} tweets to Twitter!")
                                            st.markdown(f"**Tweet IDs:** {', '.join(tweet_ids)}")
                                            if len(tweet_urls) > 0:
                                                st.markdown(f"**View tweets:**")
                                                for i, url in enumerate(tweet_urls, 1):
                                                    st.markdown(f"- Tweet {i}: {url}")
                                        else:
                                            st.warning("⚠️ Thread posted but could not extract tweet IDs. Check your Twitter account.")
                                            st.json(response)
                            
                            except Exception as e:
                                st.error(f"❌ Error posting to Twitter: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                
                with col2:
                    if st.button("💾 Download Text Only", key=f"download_twitter_{ht}_{at}"):
                        twitter_text = format_prediction_tweet(pred, features)
                        st.download_button(
                            "⬇️ Download",
                            data=twitter_text,
                            file_name=f"prediction_{ht.replace(' ', '_')}_{at.replace(' ', '_')}_{pred.get('game_date', 'today')}.txt",
                            mime="text/plain"
                        )
                
                # Show dry_run status (check actual value)
                actual_dry_run = os.getenv("TW_DRY_RUN", "true").lower() in ("1", "true", "yes")
                if actual_dry_run:
                    st.info("⚠️ **Dry Run Mode**: Tweets won't be posted. Set `TW_DRY_RUN=false` in your `.env` file and restart Streamlit to post.")
                else:
                    st.success("✅ **Live Mode**: Tweets will be posted to Twitter!")
            
            except Exception as e:
                import traceback
                st.error(f"⚠️ Twitter integration error: {str(e)}")
                # Fallback to download
                try:
                    twitter_text = format_prediction_for_twitter(pred, features)
                    st.download_button(
                        "📱 Download for Twitter (Text Only)",
                        data=twitter_text,
                        file_name=f"prediction_{ht.replace(' ', '_')}_{at.replace(' ', '_')}_{pred.get('game_date', 'today')}.txt",
                        mime="text/plain"
                    )
                except:
                    pass

            # Add detailed statistics table
            st.markdown("#### 📋 Detailed Statistics Comparison")
            
            # Show feature categories table in a more compact way
            feat_table = create_feature_categories_table(features, ht, at)
            if not feat_table.empty:
                # Group by category and show in expandable sections
                categories = feat_table['Category'].unique()
                
                # Create tabs for major categories
                if len(categories) > 3:
                    tab_names = ['Elo & Form', 'Splits & Situational', 'Head-to-Head']
                    tab1, tab2, tab3 = st.tabs(tab_names)
                    
                    with tab1:
                        # Elo, Recent Form (Last 10), Recent Form (Last 5)
                        display_df = feat_table[feat_table['Category'].isin(['Elo & Ratings', 'Recent Form (Last 10)', 'Recent Form (Last 5)'])]
                        if not display_df.empty:
                            # Remove Category column for cleaner display
                            display_df = display_df.drop(columns=['Category'])
                            safe_dataframe(display_df, hide_index=True)
                    
                    with tab2:
                        # Home/Away Splits, Situational
                        display_df = feat_table[feat_table['Category'].isin(['Home/Away Splits', 'Situational'])]
                        if not display_df.empty:
                            display_df = display_df.drop(columns=['Category'])
                            safe_dataframe(display_df, hide_index=True)
                    
                    with tab3:
                        # Head-to-Head with warning
                        st.warning("⚠️ **H2H Limitations**: (1) Limited data: only ~4 games/year between teams. (2) Games alternate home/away, reducing home-specific relevance. Use with caution.")
                        display_df = feat_table[feat_table['Category'] == 'Head-to-Head']
                        if not display_df.empty:
                            display_df = display_df.drop(columns=['Category'])
                            safe_dataframe(display_df, hide_index=True)
                else:
                    # If few categories, show all at once but more compact
                    # Use st.expander for each category
                    for category in categories:
                        with st.expander(f"📊 {category}", expanded=True):
                            cat_df = feat_table[feat_table['Category'] == category].copy()
                            cat_df = cat_df.drop(columns=['Category'])
                            safe_dataframe(cat_df, hide_index=True)
            
            # Show bar chart comparison
            bars_fig = create_stats_comparison_bars(features, ht, at)
            if bars_fig:
                st.markdown("#### 📊 Side-by-Side Statistics")
                safe_plotly_chart(bars_fig)

        if 'base_model_predictions' in pred:
            st.markdown("### 🤖 Model Consensus")
            st.info("💡 **Individual home win probabilities from each base model.** These show how each model independently predicts the home team's chance to win. The final prediction combines these with weighted averaging. Low percentages mean most models predict the away team will win - this is normal when the away team is favored.")
            cols = st.columns(4)
            model_names = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'random_forest': 'Random Forest', 'logistic': 'Logistic'}
            final_home_prob = pred.get('home_win_probability', 0)
            for i, (name, prob) in enumerate(pred['base_model_predictions'].items()):
                display_name = model_names.get(name, name.title())
                # Show the home win probability with context
                cols[i].metric(
                    display_name, 
                    f"{prob:.1%}",
                    help=f"Home win probability: {prob:.1%}. Final ensemble prediction: {final_home_prob:.1%}"
                )

# =============================================================================
# HEADER & TABS
# =============================================================================
st.markdown("# 🏀 NBA Predictor")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Today", "Portfolio", "Analytics", "Train", "Performance", "Settings", "Twitter Status"])

# =============================================================================
# TAB 1: GAMES PREDICTIONS (with date picker)
# =============================================================================
with tab1:
    st.markdown("### 🗓️ Select Date")
    col_date, col_btn = st.columns([2, 1])
    with col_date:
        selected_date = st.date_input(
            "Game Date",
            value=datetime.now().date(),
            min_value=datetime.now().date() - timedelta(days=7),
            max_value=datetime.now().date() + timedelta(days=14),
            help="Select a date to fetch games for that day"
        )
    with col_btn:
        st.write("")
        st.write("")
        fetch_button = st.button("🔄 Fetch Games")

    if fetch_button:
        with st.spinner(f"Loading games for {selected_date.strftime('%Y-%m-%d')}..."):
            try:
                fetcher = NBADataFetcher(str(db_path))
                # Convert selected_date to string format
                date_str = selected_date.strftime('%Y-%m-%d')
                todays = fetcher.get_games_for_date(date_str)

                if todays.empty:
                    st.info(f"No games found for {date_str}")
                    st.session_state.todays_predictions = None
                else:
                    # Create list with game dates
                    games_with_dates = []
                    for _, g in todays.iterrows():
                        home = ID_TO_TEAM.get(g['home_team_id'])
                        away = ID_TO_TEAM.get(g['away_team_id'])
                        if home and away:
                            game_date = g.get('game_date', datetime.now().strftime('%Y-%m-%d'))
                            # Ensure game_date is a string in YYYY-MM-DD format
                            if isinstance(game_date, pd.Timestamp):
                                game_date = game_date.strftime('%Y-%m-%d')
                            elif not isinstance(game_date, str):
                                game_date = datetime.now().strftime('%Y-%m-%d')
                            games_with_dates.append((home, away, game_date))

                    if games_with_dates:
                        prog = st.progress(0)
                        status = st.empty()
                        preds = []

                        for i, (h, a, game_date) in enumerate(games_with_dates):
                            status.text(f"Analyzing {a} @ {h}...")
                            try:
                                p = st.session_state.predictor.predict_game(h, a, game_date=game_date)
                                if p and p.get('prediction'):
                                    p['home_team'] = h
                                    p['away_team'] = a
                                    p['game_date'] = game_date  # Store game date in prediction
                                    preds.append(p)
                                    save_prediction_to_db(str(db_path), {
                                        'game_date': game_date,  # Use actual game date, not today
                                        'home_team': h, 'away_team': a,
                                        'predicted_winner': h if p['prediction'] == 'home' else a,
                                        'predicted_home_prob': p['home_win_probability'],
                                        'predicted_away_prob': p['away_win_probability'],
                                        'confidence': p['confidence'],
                                        'features': p.get('features', {})
                                    })
                            except Exception as e:
                                error_msg = str(e)
                                st.warning(f"Error on {a} @ {h}: {error_msg}")
                                # Print full error for debugging
                                print(f"❌ Error on {a} @ {h}: {error_msg}")
                                import traceback
                                traceback.print_exc()
                            prog.progress((i + 1) / len(games_with_dates))

                        prog.empty()
                        status.empty()
                        st.session_state.todays_predictions = preds
                        if preds:
                            st.success(f"✅ Analyzed {len(preds)} games! ({len(games_with_dates)} total games found)")
                            # Debug: Show which games were successfully predicted
                            if len(preds) < len(games_with_dates):
                                missing = len(games_with_dates) - len(preds)
                                st.warning(f"⚠️ {missing} game(s) failed to generate predictions. Check console for errors.")
                            
                            # Email report button
                            st.markdown("---")
                            col_email1, col_email2 = st.columns([3, 1])
                            with col_email1:
                                st.markdown("**📧 Envoyer le rapport par email**")
                            with col_email2:
                                if st.button("📧 Envoyer Email", key="send_email_report"):
                                    with st.spinner("Envoi du rapport par email..."):
                                        try:
                                            from src.email_reporter import EmailReporter
                                            email_reporter = EmailReporter(db_path=str(db_path))
                                            success = email_reporter.send_daily_report(test_mode=False)
                                            if success:
                                                st.success("✅ Email envoyé avec succès!")
                                            else:
                                                st.error("❌ Erreur lors de l'envoi de l'email. Vérifiez les logs.")
                                        except Exception as e:
                                            st.error(f"❌ Erreur: {str(e)[:100]}")
                        else:
                            st.warning("No predictions generated. Check if model is trained.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.todays_predictions:
        preds = st.session_state.todays_predictions
        st.markdown("### 📊 Prediction Summary")
        c1, c2, c3 = st.columns(3)
        high_conf_count = sum(1 for p in preds if p['confidence'] >= 0.7)
        c1.metric("High Confidence", high_conf_count)
        avg_conf = np.mean([p['confidence'] for p in preds])
        c2.metric("Average Confidence", f"{avg_conf:.1%}")
        c3.metric("Total Games", len(preds))

        st.markdown("---")
        st.markdown("### 🏀 Game Predictions")
        if len(preds) == 0:
            st.warning("No predictions to display. Check if predictions were generated successfully.")
        else:
            for i, p in enumerate(sorted(preds, key=lambda x: x['confidence'], reverse=True)):
                try:
                    display_game(p, expanded=False)  # Never auto-expand to prevent scroll issues
                except Exception as e:
                    # If one game fails to display, show error but continue with others
                    game_name = f"{p.get('away_team', '?')} @ {p.get('home_team', '?')}"
                    st.error(f"Error displaying game {i+1} ({game_name}): {str(e)[:100]}")
                    import traceback
                    with st.expander(f"Error details for game {i+1}"):
                        st.code(traceback.format_exc())
                    continue  # Continue to next game
    elif st.session_state.todays_predictions is None:
        st.info("👆 Click '🔄 Fetch Today's Games' above to load predictions")

# =============================================================================
# TAB 2: PORTFOLIO
# =============================================================================
with tab2:
    st.markdown("## Portfolio")
    balance_info = get_user_balance(str(db_path))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balance", f"${balance_info['balance']:,.2f}")
    c2.metric("Wagered", f"${balance_info['wagered']:,.2f}")
    c3.metric("Won", f"${balance_info['won']:,.2f}")
    profit = balance_info['won'] - balance_info['lost']
    roi = (profit / balance_info['wagered'] * 100) if balance_info['wagered'] > 0 else 0
    c4.metric("ROI", f"{roi:+.1f}%")

    st.markdown("---")
    history = get_portfolio_history(str(db_path))
    if not history.empty:
        for _, bet in history.iterrows():
            games = json.loads(bet['games']) if isinstance(bet['games'], str) else bet['games']
            icon = {'pending': '⏳', 'won': '✅', 'lost': '❌'}.get(bet['status'], '⏳')
            st.markdown(f"{icon} **{bet['bet_type'].upper()}**: {' + '.join([g['team'] for g in games])} | ${bet['stake']:.2f} @ {bet['combined_odds']:.2f}")
    else:
        st.info("No bets yet")

# =============================================================================
# TAB 3: TEAM ANALYTICS
# =============================================================================
with tab3:
    st.markdown("## Team Analytics")
    c1, c2 = st.columns(2)
    selected_team = c1.selectbox("Team", NBA_TEAMS)
    analysis = c2.selectbox("Type", ["Overview", "Injuries", "Head-to-Head"])

    if st.button("Load"):
        team_id = TEAM_IDS.get(selected_team)
        if team_id:
            with st.spinner("Loading..."):
                try:
                    fetcher = NBADataFetcher(str(db_path))
                    fe = FeatureEngineer(str(db_path))

                    if analysis == "Overview":
                        recent = fetcher.get_team_recent_games(team_id, n_games=15)
                        if not recent.empty:
                            is_home = recent['home_team_id'] == team_id
                            wins = sum((is_home & (recent['home_win'] == 1)) | (~is_home & (recent['home_win'] == 0)))
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Record", f"{wins}-{15-wins}")
                            c2.metric("Win %", f"{wins/15*100:.0f}%")
                            c3.metric("Elo", f"{fe.elo_system.get_rating(team_id):.0f}")
                            ppg = recent.apply(lambda x: x['home_score'] if x['home_team_id'] == team_id else x['away_score'], axis=1).mean()
                            c4.metric("PPG", f"{ppg:.1f}")

                            display_df = recent[['game_date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
                            display_df['W/L'] = recent.apply(lambda x: 'W' if ((x['home_team_id'] == team_id and x['home_win']) or (x['away_team_id'] == team_id and not x['home_win'])) else 'L', axis=1)
                            safe_dataframe(display_df)

                    elif analysis == "Injuries":
                        try:
                            tracker = InjuryTracker(str(db_path))
                            inj = tracker.get_team_injuries(team_id, force_refresh=True)
                            c1, c2 = st.columns(2)
                            c1.metric("Injured", inj['total_injured'])
                            c2.metric("Stars Out", "Yes" if inj['star_injured'] else "No")
                            if inj['injuries']:
                                for i in inj['injuries']:
                                    st.write(f"- {i['player']}: {i['injury']} ({i['status']})")
                            else:
                                st.success("No injuries reported")
                        except Exception as e:
                            st.warning(f"Could not fetch injuries: {e}")

                    elif analysis == "Head-to-Head":
                        opp = st.selectbox("Opponent", [t for t in NBA_TEAMS if t != selected_team], key="h2h_opponent_select")
                        opp_id = TEAM_IDS.get(opp)
                        if opp_id:
                            conn = sqlite3.connect(db_path)
                            h2h = pd.read_sql_query("""
                                SELECT game_date, home_team, away_team, home_score, away_score FROM games
                                WHERE (home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?)
                                ORDER BY game_date DESC LIMIT 10
                            """, conn, params=(team_id, opp_id, opp_id, team_id))
                            conn.close()
                            if not h2h.empty:
                                safe_dataframe(h2h)
                            else:
                                st.info("No H2H games found")
                except Exception as e:
                    st.error(f"Error: {e}")

# =============================================================================
# TAB 4: TRAIN MODEL
# =============================================================================
with tab4:
    st.markdown("## Train Model")

    # Check if model exists by looking for required files
    models_path_check = project_root / "models"
    required_files = ['meta_model.pkl', 'scaler.pkl', 'feature_names.json']
    model_files_exist = all((models_path_check / f).exists() for f in required_files)
    
    if model_files_exist:
        import os
        # Get the most recent file modification time as the training date
        file_times = []
        for f in required_files:
            file_path = models_path_check / f
            if file_path.exists():
                file_times.append(os.path.getmtime(file_path))
        if file_times:
            mt = datetime.fromtimestamp(max(file_times))
            st.success(f"✅ Model found! Last trained: {mt.strftime('%Y-%m-%d %H:%M')}")
            
            # Verify model can actually be loaded
            try:
                test_predictor = NBAPredictor(db_path=str(db_path), model_dir=str(models_path))
                if test_predictor.load_model():
                    st.info(f"✓ Model successfully loaded and ready for predictions")
                else:
                    st.warning("⚠️ Model files exist but could not be loaded. May need retraining.")
            except Exception as e:
                st.warning(f"⚠️ Model files exist but error loading: {str(e)[:100]}")
    else:
        st.warning("No model found - train first!")
        st.info("💡 The model consists of multiple files. Make sure to train using the button below.")

    c1, c2 = st.columns(2)
    start_yr = c1.selectbox("Start", [2022, 2023, 2024], index=2)  # Default to 2024
    end_yr = c2.selectbox("End", [2023, 2024, 2025], index=2)

    st.info("💡 **Tip**: Using only recent seasons (2024-2025) trains faster and focuses on current team dynamics. Older data may be less relevant due to roster changes.")

    if st.button("🚀 Train Model"):
        st.warning("This takes 5-15 minutes...")
        prog = st.progress(0)
        status = st.empty()
        log = st.empty()
        t0 = time.time()

        try:
            from src.data_fetcher import NBADataFetcher, EloRatingSystem, FeatureEngineer
            from src.models import StackedEnsembleModel

            # Step 1: Fetch
            status.text("📥 Step 1/4: Fetching games...")
            fetcher = NBADataFetcher(str(db_path))
            seasons = [f"{yr}-{str(yr+1)[-2:]}" for yr in range(start_yr, end_yr + 1)]
            log.write(f"Seasons: {seasons}")
            games = fetcher.fetch_historical_games(seasons)
            log.write(f"✅ Fetched {len(games)} games ({time.time()-t0:.0f}s)")
            prog.progress(25)

            # Step 2: Elo
            status.text("📊 Step 2/4: Calculating Elo ratings...")
            elo = EloRatingSystem(str(db_path))
            games_sorted = games.sort_values('game_date')
            total = len(games_sorted)

            for team in teams.get_teams():
                elo.ratings[team['id']] = elo.INITIAL_ELO

            elo_prog = st.progress(0)
            for idx, (_, row) in enumerate(games_sorted.iterrows()):
                elo.update_ratings(row['home_team_id'], row['away_team_id'], row['home_score'], row['away_score'], row['game_id'])
                if idx % 200 == 0:
                    elo_prog.progress(idx / total)
                    log.write(f"Elo: {idx}/{total} games... ({time.time()-t0:.0f}s)")
            elo_prog.empty()
            log.write(f"✅ Elo complete ({time.time()-t0:.0f}s)")
            prog.progress(45)

            # Step 3: Features
            status.text("🔧 Step 3/4: Engineering features...")
            log.write("Creating features (no injuries/odds for speed)...")
            log.write("⏳ This may take 1-3 minutes depending on data size...")
            fe = FeatureEngineer(str(db_path))
            fe.enhanced_features_available = False  # Disable slow external APIs
            X, y, sample_weights = fe.create_training_dataset(games)
            log.write(f"✅ {len(X)} samples, {len(X.columns)} features")
            log.write(f"📊 Sample weights calculated (recent games weighted {sample_weights[-100:].mean():.2f}x vs old {sample_weights[:100].mean():.2f}x)")
            log.write(f"⏱️ Feature engineering: {time.time()-t0:.0f}s")
            prog.progress(65)

            # Step 4: Train
            status.text("🤖 Step 4/4: Training model with recency weighting...")
            log.write("⏳ Training ensemble (XGBoost, LightGBM, RandomForest, Logistic)...")
            log.write("📈 Using 5-fold time-series cross-validation...")
            model = StackedEnsembleModel()
            results = model.train(X, y, sample_weights=sample_weights, n_splits=5)
            acc = results['mean_cv_accuracy']
            log.write(f"✅ Trained! CV Accuracy: {acc:.1%} ± {results['std_cv_accuracy']:.1%} ({time.time()-t0:.0f}s)")
            prog.progress(90)

            # Save
            model.save(str(project_root / "models"))
            prog.progress(100)
            st.success(f"✅ Done! Accuracy: {acc:.1%} ± {results['std_cv_accuracy']:.1%} | Time: {(time.time()-t0)/60:.1f} min")

            st.session_state.predictor = NBAPredictor(db_path=str(db_path), model_dir=str(models_path))

        except Exception as e:
            st.error(f"Failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# =============================================================================
# TAB 5: PERFORMANCE
# =============================================================================
with tab5:
    st.markdown("## Performance")

    col1, col2 = st.columns([3, 1])
    with col1:
        lookback = st.slider("Lookback Days", min_value=0, max_value=30, value=7, help="How many days back to check for game results")
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing

    st.markdown("---")

    col_update1, col_update2 = st.columns(2)
    with col_update1:
        if st.button("🔄 1. Refresh Game Data", key="perf_refresh_games"):
            with st.spinner("Fetching recent games from NBA API..."):
                try:
                    from nba_api.stats.endpoints import leaguegamefinder
                    from nba_api.stats.static import teams as nba_teams
                    
                    # Fetch recent games using leaguegamefinder (more reliable)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                    
                    date_from = start_date.strftime('%m/%d/%Y')
                    date_to = end_date.strftime('%m/%d/%Y')
                    
                    status_text = st.empty()
                    status_text.text(f"Fetching games from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
                    
                    # Fetch all recent games
                    time.sleep(0.6)
                    finder = leaguegamefinder.LeagueGameFinder(
                        date_from_nullable=date_from,
                        date_to_nullable=date_to,
                        season_type_nullable='Regular Season',
                        league_id_nullable='00'  # NBA
                    )
                    
                    games_df = finder.get_data_frames()[0]
                    
                    if games_df.empty:
                        st.warning("No games found from NBA API")
                    else:
                        st.info(f"📋 Found {len(games_df)} game entries from API")
                        
                        # Process and save games to database
                        conn_temp = sqlite3.connect(str(db_path))
                        fetched = 0
                        
                        # Group by GAME_ID to get both teams' stats
                        team_map = {t['id']: t for t in nba_teams.get_teams()}
                        
                        for game_id, game_group in games_df.groupby('GAME_ID'):
                            if len(game_group) != 2:
                                continue  # Need both teams
                                
                            # Determine home and away from matchup string
                            row1, row2 = game_group.iloc[0], game_group.iloc[1]
                            
                            # "vs." means home, "@" means away
                            if 'vs.' in str(row1.get('MATCHUP', '')):
                                home_row, away_row = row1, row2
                            else:
                                home_row, away_row = row2, row1
                            
                            home_pts = home_row.get('PTS')
                            away_pts = away_row.get('PTS')
                            
                            if pd.isna(home_pts) or pd.isna(away_pts) or home_pts == 0:
                                continue  # Game not finished
                            
                            game_date_str = str(home_row.get('GAME_DATE', ''))[:10]  # Get YYYY-MM-DD
                            home_team_id = home_row.get('TEAM_ID')
                            away_team_id = away_row.get('TEAM_ID')
                            
                            home_team_info = team_map.get(home_team_id, {})
                            away_team_info = team_map.get(away_team_id, {})
                            
                            try:
                                conn_temp.execute("""
                                    INSERT OR REPLACE INTO games (
                                        game_id, game_date, home_team_id, away_team_id,
                                        home_team, away_team, home_score, away_score, home_win
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    game_id,
                                    game_date_str,
                                    home_team_id,
                                    away_team_id,
                                    home_team_info.get('abbreviation', str(home_team_id)),
                                    away_team_info.get('abbreviation', str(away_team_id)),
                                    int(home_pts),
                                    int(away_pts),
                                    1 if home_pts > away_pts else 0
                                ))
                                fetched += 1
                            except Exception as e:
                                print(f"Error saving game {game_id}: {e}")
                                continue
                        
                        conn_temp.commit()
                        conn_temp.close()
                        status_text.empty()
                        
                        if fetched > 0:
                            st.success(f"✅ Fetched {fetched} games from last 7 days!")
                            safe_rerun()
                        else:
                            st.info("No new finished games found")
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col_update2:
        if st.button("✅ 2. Update Results", key="perf_update_results"):
            with st.spinner("Matching predictions to game results..."):
                try:
                    fb = ModelFeedbackSystem(str(db_path))
                    updated = fb.update_predictions_with_results(lookback_days=lookback, use_api=False)
                    fb.close()
                    
                    if updated > 0:
                        st.success(f"✅ Updated {updated} predictions!")
                        safe_rerun()
                    else:
                        st.warning("No matches found. Make sure you clicked 'Refresh Game Data' first.")
                except Exception as e:
                    st.error(f"Error: {e}")

    try:
        fb = ModelFeedbackSystem(str(db_path))
        try:
            perf = fb.evaluate_model_performance(period_days=30)
            if perf and perf.get('total_predictions', 0) > 0:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{perf.get('accuracy',0)*100:.1f}%")
                c2.metric("Predictions", perf.get('total_predictions', 0))
                c3.metric("Correct", perf.get('correct_predictions', 0))
                c4.metric("Brier Score", f"{perf.get('brier_score', 0):.4f}")
            else:
                st.info("No verified predictions yet")
        except:
            st.info("Make predictions and update results to see performance")

        # Show all recent predictions with better date filtering
        conn = sqlite3.connect(str(db_path))
        cutoff_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        today_str = datetime.now().strftime('%Y-%m-%d')

        recent = pd.read_sql_query("""
            SELECT game_date, home_team, away_team, predicted_winner, confidence,
                   CASE WHEN correct = 1 THEN '✅' WHEN correct = 0 THEN '❌' ELSE '⏳' END as result,
                   prediction_date, actual_winner
                FROM predictions
            WHERE game_date >= ? AND game_date <= ?
            ORDER BY game_date DESC, prediction_date DESC
            LIMIT 50
        """, conn, params=(cutoff_date, today_str))
        conn.close()

        if not recent.empty:
            st.markdown(f"### Recent Predictions (Last {lookback} Days)")
            # Add helpful columns
            recent['Days Ago'] = recent['game_date'].apply(lambda x: (datetime.now().date() - datetime.strptime(x, '%Y-%m-%d').date()).days if pd.notna(x) else None)
            
            # Reorder columns for better readability
            display_cols = ['game_date', 'Days Ago', 'home_team', 'away_team', 'predicted_winner', 
                          'confidence', 'result', 'actual_winner']
            display_cols = [c for c in display_cols if c in recent.columns]
            safe_dataframe(recent[display_cols].rename(columns={
                'game_date': 'Game Date',
                'Days Ago': 'Days Ago',
                'home_team': 'Home',
                'away_team': 'Away',
                'predicted_winner': 'Predicted Winner',
                'confidence': 'Confidence',
                'result': 'Result',
                'actual_winner': 'Actual Winner'
            }), hide_index=True)
            
            # Summary stats
            total = len(recent)
            pending = len(recent[recent['result'] == '⏳'])
            correct = len(recent[recent['result'] == '✅'])
            wrong = len(recent[recent['result'] == '❌'])
            
            st.markdown(f"**Summary**: {total} predictions | ✅ {correct} correct | ❌ {wrong} wrong | ⏳ {pending} pending")
            
            # Show diagnostic info for pending predictions older than today
            if pending > 0:
                past_pending = recent[(recent['result'] == '⏳') & (recent['Days Ago'] > 0)]
                if len(past_pending) > 0:
                    with st.expander(f"⚠️ {len(past_pending)} past games still pending - click to diagnose"):
                        st.markdown("""
                        **These predictions are for games that should have finished but results weren't found.**
                        
                        **Possible reasons:**
                        1. **Wrong matchup**: The predicted game might not exist on that date
                        2. **Team name mismatch**: Database uses different team names than predictions
                        3. **Game data not in DB**: Need to refresh game data first
                        
                        **Try these steps:**
                        1. Click **"🔄 Refresh Game Data"** to fetch recent games
                        2. Then click **"🔄 Update Results from NBA API"** again
                        """)
                        
                        # Show what games exist in DB for these dates
                        st.markdown("**Games in database for these dates:**")
                        conn_diag = sqlite3.connect(str(db_path))
                        for _, row in past_pending.iterrows():
                            gdate = row['game_date']
                            games_on_date = pd.read_sql_query("""
                                SELECT home_team, away_team, home_score, away_score 
                                FROM games WHERE game_date = ?
                            """, conn_diag, params=(gdate,))
                            if not games_on_date.empty:
                                st.write(f"**{gdate}**: {len(games_on_date)} games found")
                                st.caption(", ".join([f"{r['away_team']}@{r['home_team']}" for _, r in games_on_date.iterrows()]))
                            else:
                                st.write(f"**{gdate}**: ❌ No games in database")
                        conn_diag.close()
                        
                        # Option to clear invalid predictions
                        if st.button("🗑️ Clear Old Pending Predictions", key="clear_invalid_preds"):
                            try:
                                conn_del = sqlite3.connect(str(db_path))
                                # Delete predictions where game_date < today and result is still pending
                                today_str = datetime.now().strftime('%Y-%m-%d')
                                conn_del.execute("""
                                    DELETE FROM predictions 
                                    WHERE game_date < ? AND actual_winner IS NULL
                                """, (today_str,))
                                conn_del.commit()
                                deleted = conn_del.total_changes
                                conn_del.close()
                                st.success(f"✅ Cleared {deleted} old pending predictions. Refresh to see changes.")
                                safe_rerun()
                            except Exception as e:
                                st.error(f"Error clearing predictions: {e}")
        else:
            # Check if there are ANY predictions at all
            conn = sqlite3.connect(str(db_path))
            total_count = pd.read_sql_query("SELECT COUNT(*) as count FROM predictions", conn)
            conn.close()
            if total_count.iloc[0]['count'] > 0:
                st.info(f"No predictions found in the last {lookback} days. Try increasing the lookback period or check if predictions are being saved with the correct game dates.")
            else:
                st.info("No predictions found in database. Make some predictions first!")
        fb.close()
    except Exception as e:
        st.error(f"Error: {e}")

# =============================================================================
# TAB 6: SETTINGS
# =============================================================================
with tab6:
    st.markdown("## Settings")

    c1, c2 = st.columns(2)
    if c1.button("Init Database"):
        init_database(str(db_path))
        st.success("Done!")

    if c2.button("Update Results", key="settings_update_results"):
        with st.spinner("Updating predictions..."):
            try:
                fb = ModelFeedbackSystem(str(db_path))
                n = fb.update_predictions_with_results(lookback_days=30)
                fb.close()
                if n > 0:
                    st.success(f"✅ Updated {n} predictions")
                else:
                    st.info(f"ℹ️ No predictions updated. Check if games have been played and are in the database.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("""
    **Model:** XGBoost + LightGBM + Random Forest + Logistic Regression

    **Features (84+):** Elo, Recent Form, H2H, Rest Days, Streaks, Travel, Injuries

    **Feedback Loop:** Predictions saved → Results fetched → Performance tracked → Retrain when needed
    """)

# =============================================================================
# TAB 7: TWITTER STATUS (Rate Limits)
# =============================================================================
with tab7:
    st.markdown("## 📊 Twitter API Rate Limits & Status")
    
    # Try to load Twitter credentials and get rate limits
    try:
        from src.twitter_integration import create_fresh_twitter_client
        
        if st.button("🔄 Refresh Rate Limits"):
            with st.spinner("Fetching Twitter API rate limits..."):
                try:
                    # Create fresh client
                    api_clients = create_fresh_twitter_client()
                    api_v1 = api_clients.get("api_v1")
                    client_v2 = api_clients.get("client_v2")
                    
                    if not api_v1:
                        st.error("❌ Could not create Twitter API v1.1 client (needed for rate limits).")
                        st.stop()
                    
                    # Note: rate_limit_status() itself uses an API call and may trigger rate limit waiting
                    # Create a temporary client without wait_on_rate_limit for this call
                    import tweepy
                    from src.twitter_integration import load_credentials_from_env
                    creds = load_credentials_from_env()
                    
                    # Create API client without wait_on_rate_limit to avoid long waits
                    temp_auth = tweepy.OAuth1UserHandler(
                        creds.get('api_key'),
                        creds.get('api_key_secret'),
                        creds.get('access_token'),
                        creds.get('access_token_secret')
                    )
                    temp_api = tweepy.API(temp_auth, wait_on_rate_limit=False)  # Don't wait when checking limits
                    
                    try:
                        # Get rate limit status (uses v1.1 API)
                        # Note: This call itself consumes 1 request from /application/rate_limit_status endpoint
                        rate_limits = temp_api.rate_limit_status()['resources']
                    except tweepy.errors.TooManyRequests:
                        st.warning("⚠️ Rate limit check temporarily unavailable due to API limits. Please wait a few minutes and try again.")
                        st.stop()
                    
                    st.success("✅ Rate limits fetched successfully!")
                    
                    # Display rate limits organized by endpoint category
                    st.markdown("### Rate Limits by Endpoint Category")
                    
                    # Helper function to format time until reset
                    def format_reset_time(reset_timestamp):
                        if reset_timestamp == 0:
                            return "N/A"
                        reset_dt = datetime.fromtimestamp(reset_timestamp)
                        now = datetime.now()
                        if reset_dt > now:
                            delta = reset_dt - now
                            hours = int(delta.total_seconds() // 3600)
                            minutes = int((delta.total_seconds() % 3600) // 60)
                            return f"{hours}h {minutes}m ({reset_dt.strftime('%H:%M:%S')})"
                        return "Available now"
                    
                    # Helper function to get status color
                    def get_status_color(remaining, limit):
                        pct = (remaining / limit) * 100 if limit > 0 else 0
                        if pct >= 50:
                            return "🟢"
                        elif pct >= 25:
                            return "🟡"
                        else:
                            return "🔴"
                    
                    # Organize by category
                    categories = {
                        'statuses': 'Tweets (Statuses)',
                        'users': 'Users',
                        'search': 'Search',
                        'followers': 'Followers',
                        'friends': 'Friends',
                        'trends': 'Trends',
                        'lists': 'Lists',
                        'account': 'Account',
                        'application': 'Application',
                        'help': 'Help',
                        'media': 'Media Upload'
                    }
                    
                    # Display each category
                    for category_key, category_name in categories.items():
                        if category_key in rate_limits:
                            st.markdown(f"#### {category_name}")
                            
                            endpoints = rate_limits[category_key]
                            rate_data = []
                            
                            for endpoint, limit_info in endpoints.items():
                                limit = limit_info['limit']
                                remaining = limit_info['remaining']
                                reset = limit_info['reset']
                                
                                status = get_status_color(remaining, limit)
                                reset_time = format_reset_time(reset)
                                pct = (remaining / limit) * 100 if limit > 0 else 0
                                
                                rate_data.append({
                                    'Endpoint': endpoint.replace('/', ' / '),
                                    'Status': status,
                                    'Remaining': remaining,
                                    'Limit': limit,
                                    'Used': limit - remaining,
                                    '% Remaining': f"{pct:.1f}%",
                                    'Resets At': reset_time
                                })
                            
                            if rate_data:
                                df = pd.DataFrame(rate_data)
                                # Sort by remaining percentage (lowest first)
                                df = df.sort_values('% Remaining', ascending=True)
                                safe_dataframe(df, hide_index=True)
                    
                    # Show key endpoints summary
                    st.markdown("---")
                    st.markdown("### 📌 Key Endpoints Summary")
                    
                    key_endpoints = {
                        '/statuses/update': 'Post Tweet',
                        '/statuses/update/:id': 'Reply to Tweet',
                        '/media/upload': 'Upload Media',
                        '/statuses/home_timeline': 'Get Timeline',
                        '/users/show': 'Get User Info',
                        '/account/verify_credentials': 'Verify Credentials'
                    }
                    
                    summary_data = []
                    for endpoint, description in key_endpoints.items():
                        # Find in statuses or users or account or media
                        found = False
                        for category in ['statuses', 'users', 'account', 'media']:
                            if category in rate_limits:
                                full_endpoint = endpoint.replace(':', '/')
                                # Try exact match or partial
                                for ep_key, ep_info in rate_limits[category].items():
                                    if endpoint in ep_key or ep_key.endswith(endpoint.split('/')[-1]):
                                        limit = ep_info['limit']
                                        remaining = ep_info['remaining']
                                        reset = ep_info['reset']
                                        
                                        status = get_status_color(remaining, limit)
                                        reset_time = format_reset_time(reset)
                                        pct = (remaining / limit) * 100 if limit > 0 else 0
                                        
                                        summary_data.append({
                                            'Endpoint': description,
                                            'Status': status,
                                            'Remaining': f"{remaining} / {limit}",
                                            '% Available': f"{pct:.1f}%",
                                            'Resets At': reset_time
                                        })
                                        found = True
                                        break
                            if found:
                                break
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        safe_dataframe(summary_df, hide_index=True)
                    
                    # Authentication status - DO NOT make API calls here
                    # Free tier only allows 25 /users/me calls per 24 hours!
                    st.markdown("---")
                    st.markdown("### 🔐 Authentication Status")
                    auth_status = api_clients.get("auth_status", {})

                    if auth_status.get("skip_verification") or auth_status.get("verified"):
                        # We skipped verification to preserve rate limits
                        st.success("✅ **Twitter Client Ready**")
                        st.info("💡 Credentials configured. Verification skipped to preserve rate limits.")
                        st.markdown("""
                        **Free Tier Limits (per 24 hours):**
                        - `/users/me` (verify auth): **25 requests**
                        - `POST /tweets`: **17 tweets**
                        - `DELETE /tweets`: **17 deletes**

                        Your credentials will be validated when you post a tweet.
                        """)
                    elif auth_status.get("error"):
                        st.error(f"❌ Authentication failed: {auth_status['error']}")
                    else:
                        st.warning("⚠️ Authentication status unknown")
                    
                except Exception as e:
                    st.error(f"❌ Error fetching rate limits: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.info("👆 Click '🔄 Refresh Rate Limits' to fetch current Twitter API rate limits and status.")
            st.markdown("""
            This tab shows:
            - **Rate limits** for all Twitter API endpoints
            - **Remaining requests** before hitting limits
            - **Reset times** for each endpoint
            - **Authentication status**
            
            Rate limits are important to avoid hitting API quotas when posting tweets or fetching data.
            """)
            
    except ImportError:
        st.warning("⚠️ Twitter integration not available. Install required packages:")
        st.code("pip install tweepy")
    except Exception as e:
        st.error(f"❌ Error loading Twitter integration: {e}")

st.caption("NBA Predictor v2.1")

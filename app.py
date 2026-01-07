"""
NBA Predictor - Enhanced Interface
Full-featured: Predictions, Analytics, Performance
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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # dotenv not installed, rely on system env vars

# Check for required dependencies
try:
    from src.predictor import NBAPredictor
    from src.data_fetcher import NBADataFetcher, FeatureEngineer, EloRatingSystem
    from src.odds_scraper import generate_bookmaker_odds, odds_to_american
    from src.real_odds_fetcher import RealOddsFetcher
    from src.model_feedback_system import ModelFeedbackSystem
    from src.injury_tracker import InjuryTracker
    from src.prediction_warnings import generate_warnings, format_warning_for_display, PredictionWarning
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
                confidence, features_json, home_odds, away_odds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            prediction_data['game_date'],
            prediction_data['home_team'],
            prediction_data['away_team'],
            prediction_data['predicted_winner'],
            float(prediction_data['predicted_home_prob']),
            float(prediction_data['predicted_away_prob']),
            float(prediction_data['confidence']),
            features_json,
            prediction_data.get('home_odds'),
            prediction_data.get('away_odds')
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
# SIDEBAR - API Key Management
# =============================================================================
def get_api_quota(api_key: str) -> dict:
    """Check remaining API quota for The Odds API"""
    if not api_key:
        return {'error': 'No API key'}
    try:
        import requests
        response = requests.get(
            "https://api.the-odds-api.com/v4/sports",
            params={'apiKey': api_key},
            timeout=5
        )
        if response.status_code == 200:
            return {
                'remaining': response.headers.get('x-requests-remaining', 'Unknown'),
                'used': response.headers.get('x-requests-used', 'Unknown'),
                'status': 'OK'
            }
        elif response.status_code == 401:
            return {'error': 'Invalid API key', 'status': 'ERROR'}
        else:
            return {'error': f'HTTP {response.status_code}', 'status': 'ERROR'}
    except Exception as e:
        return {'error': str(e), 'status': 'ERROR'}

with st.sidebar:
    st.markdown("### 🏀 NBA Predictor")
    st.markdown("---")

    # =============================================================================
    # API KEY MANAGEMENT
    # =============================================================================
    st.markdown("#### 🔑 API Keys")

    # Load existing API keys from environment/session
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
        # Load from environment
        odds_key = os.getenv('ODDS_API_KEY', '')
        if odds_key:
            st.session_state.api_keys['The Odds API'] = odds_key

    # Add new API key
    with st.expander("➕ Add API Key", expanded=False):
        api_name = st.selectbox("API Provider", ["The Odds API"], key="api_provider_select")
        new_key = st.text_input("API Key", type="password", key="new_api_key_input")
        if st.button("Save Key"):
            if new_key:
                st.session_state.api_keys[api_name] = new_key
                # Also set in environment for current session
                if api_name == "The Odds API":
                    os.environ['ODDS_API_KEY'] = new_key
                st.success(f"✅ {api_name} key saved!")
                safe_rerun()
            else:
                st.warning("Please enter a key")

    # Display existing keys and their quotas
    st.markdown("#### 📊 API Quotas")

    if st.session_state.api_keys:
        for api_name, api_key in st.session_state.api_keys.items():
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"

            if api_name == "The Odds API":
                quota = get_api_quota(api_key)

                if quota.get('status') == 'OK':
                    remaining = int(quota.get('remaining', 0))
                    used = int(quota.get('used', 0))
                    total = remaining + used
                    pct_remaining = (remaining / total * 100) if total > 0 else 0

                    # Color based on remaining
                    if pct_remaining > 50:
                        color = "#22c55e"  # Green
                    elif pct_remaining > 20:
                        color = "#f59e0b"  # Orange
                    else:
                        color = "#ef4444"  # Red

                    st.markdown(f"""
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin: 8px 0;">
                        <div style="font-weight: 600; color: #1e293b; margin-bottom: 4px;">🎯 {api_name}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Key: {masked_key}</div>
                        <div style="margin-top: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                                <span style="color: {color}; font-weight: 600;">{remaining} remaining</span>
                                <span style="color: #64748b;">{used} used</span>
                            </div>
                            <div style="background: #e2e8f0; border-radius: 4px; height: 8px; margin-top: 4px;">
                                <div style="background: {color}; width: {pct_remaining}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Resets monthly</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px; margin: 8px 0;">
                        <div style="font-weight: 600; color: #991b1b;">❌ {api_name}</div>
                        <div style="font-size: 0.8rem; color: #dc2626;">{quota.get('error', 'Unknown error')}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No API keys configured. Add one above to get real bookmaker odds.")

    # Refresh quota button
    if st.button("🔄 Refresh Quotas"):
        safe_rerun()

    st.markdown("---")
    st.markdown("#### ℹ️ Quick Help")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #64748b;">
    <b>Get free API key:</b><br>
    <a href="https://the-odds-api.com/" target="_blank">the-odds-api.com</a><br>
    Free tier: 500 requests/month
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# REAL ODDS FETCHING
# =============================================================================
# Use st.cache for older Streamlit versions, st.cache_data for newer
try:
    _cache_decorator = st.cache_data(ttl=300)
except AttributeError:
    _cache_decorator = st.cache(ttl=300, suppress_st_warning=True)

@_cache_decorator
def get_real_or_simulated_odds(home_team, away_team, model_home_prob):
    """
    Try to fetch real bookmaker odds. If unavailable, simulate from model.

    Returns:
        (odds_dict, is_real: bool, market_disagrees: bool)
    """
    try:
        real_odds_fetcher = RealOddsFetcher()
        real_odds = real_odds_fetcher.get_game_odds(home_team, away_team)

        if real_odds and real_odds.get('bookmakers'):
            # Successfully got real odds
            market_home_prob = real_odds['market_home_prob']

            # Check if market significantly disagrees with model (>15% difference)
            market_disagrees = abs(market_home_prob - model_home_prob) > 0.15

            # Format for compatibility with existing code
            formatted_odds = {
                'bookmakers': real_odds['bookmakers'],
                'best_home': real_odds['best_home_odds'],
                'best_away': real_odds['best_away_odds'],
                'avg_home_odds': real_odds['avg_home_odds'],
                'avg_away_odds': real_odds['avg_away_odds'],
                'market_home_prob': market_home_prob,
                'market_away_prob': real_odds['market_away_prob'],
                'source': 'real_bookmakers'
            }

            return formatted_odds, True, market_disagrees
    except Exception as e:
        print(f"Could not fetch real odds: {e}")

    # Fallback to simulated odds
    simulated = generate_bookmaker_odds(model_home_prob)
    simulated['source'] = 'simulated'
    return simulated, False, False


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

    # Try to get real odds, fallback to simulated
    odds, is_real_odds, market_disagrees = get_real_or_simulated_odds(
        ht, at, pred['home_win_probability']
    )

    # Extract odds values for display
    if is_real_odds:
        home_odds_display = odds.get('avg_home_odds', 2.0)
        away_odds_display = odds.get('avg_away_odds', 2.0)
    else:
        home_odds_display = odds['bookmakers'].get('Pinnacle', {}).get('home', 2.0)
        away_odds_display = odds['bookmakers'].get('Pinnacle', {}).get('away', 2.0)

    # Get prediction quality from calibrated model
    prediction_quality = pred.get('prediction_quality', 'medium')
    should_predict = pred.get('should_predict', True)

    # Add quality badge for close games
    quality_badge = ""
    if prediction_quality == "low" or not should_predict:
        quality_badge = '<span style="background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem; margin-left: 0.5rem;">Close Game</span>'
    elif prediction_quality == "high":
        quality_badge = '<span style="background: #dcfce7; color: #166534; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem; margin-left: 0.5rem;">Strong Edge</span>'

    # Build HTML as single concatenated string to avoid any newline parsing issues
    game_card_html = (
        '<div class="game-card">'
        '<div style="display: flex; justify-content: space-between; align-items: center;">'
        '<div>'
        f'<div style="font-size: 1rem; font-weight: 600;">{at} @ {ht}</div>'
        '<div style="margin-top: 0.3rem;">'
        f'<span style="color: #059669; font-weight: 700;">{winner}</span>'
        f'<span style="color: #64748b; margin-left: 0.5rem;">{win_prob:.1%}</span>'
        f'{quality_badge}'
        '</div>'
        '</div>'
        '<div style="text-align: right;">'
        f'<span class="{conf_class}">{conf_text}</span>'
        f'<div style="font-size: 0.8rem; color: #64748b;">{conf:.0%}</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(game_card_html, unsafe_allow_html=True)

    # Add tooltip explaining "@" notation
    st.caption(f"💡 **Note**: '{at} @ {ht}' means {at} is playing at {ht}'s home stadium")

    # Show odds source warning
    if not is_real_odds:
        st.warning("⚠️ **Simulated Odds**: No real bookmaker odds available. Showing calculated odds from model probabilities. Get free API key at https://the-odds-api.com/")
    elif market_disagrees:
        st.error("🚨 **Market Disagrees**: Bookmakers have significantly different odds than model prediction. Review carefully before betting!")

    # Note: We don't use expanded=True to prevent auto-scroll to bottom
    # Include odds in expander label for quick reference
    odds_label = f"{at} ({away_odds_display:.2f}) @ {ht} ({home_odds_display:.2f})"
    with st.expander(f"📊 {odds_label}", expanded=False):
        # Show Model vs Market comparison if real odds available
        if is_real_odds and 'market_home_prob' in odds:
            st.markdown("### 🔄 Model vs Market Comparison")
            comp_col1, comp_col2, comp_col3 = st.columns(3)

            model_home_prob = pred['home_win_probability']
            market_home_prob = odds['market_home_prob']
            diff = abs(model_home_prob - market_home_prob)

            with comp_col1:
                st.metric("Model Home Win", f"{model_home_prob:.1%}")
            with comp_col2:
                st.metric("Market Home Win", f"{market_home_prob:.1%}")
            with comp_col3:
                if diff > 0.15:
                    st.metric("Difference", f"{diff:.1%}", delta=None, delta_color="inverse")
                    st.caption("⚠️ Large gap!")
                else:
                    st.metric("Difference", f"{diff:.1%}")

            if market_home_prob > model_home_prob + 0.10:
                st.info(f"📊 **Market favors {ht}** more than our model does ({market_home_prob:.1%} vs {model_home_prob:.1%})")
            elif model_home_prob > market_home_prob + 0.10:
                st.info(f"🤖 **Model favors {ht}** more than market does ({model_home_prob:.1%} vs {market_home_prob:.1%})")

            st.markdown("---")

        # ═══════════════════════════════════════════════════════════════
        # PREDICTION WARNINGS SECTION
        # ═══════════════════════════════════════════════════════════════
        market_prob = odds.get('market_home_prob') if is_real_odds else None
        warnings_list = generate_warnings(
            model_home_prob=pred['home_win_probability'],
            model_confidence=conf,
            market_home_prob=market_prob,
            home_team_elo=features.get('home_elo', 1500),
            away_team_elo=features.get('away_elo', 1500),
            home_recent_form=features.get('home_last10_win_pct', 0.5),
            away_recent_form=features.get('away_last10_win_pct', 0.5),
            home_team=ht,
            away_team=at
        )

        if warnings_list:
            # Show critical warnings prominently
            critical_warnings = [w for w in warnings_list if w.level == "CRITICAL"]
            other_warnings = [w for w in warnings_list if w.level != "CRITICAL"]

            if critical_warnings:
                st.markdown("### 🚨 Critical Warnings")
                for warning in critical_warnings:
                    st.markdown(format_warning_for_display(warning), unsafe_allow_html=True)

            # Show other warnings inline (no nested expander - Streamlit doesn't support it)
            if other_warnings:
                st.markdown("### ⚠️ Additional Warnings")
                for warning in other_warnings:
                    st.markdown(format_warning_for_display(warning), unsafe_allow_html=True)

            st.markdown("---")

        # Get odds for display
        if is_real_odds:
            home_odds = odds.get('avg_home_odds', 2.0)
            away_odds = odds.get('avg_away_odds', 2.0)
        else:
            home_odds = odds['bookmakers'].get('Pinnacle', {}).get('home', 2.0)
            away_odds = odds['bookmakers'].get('Pinnacle', {}).get('away', 2.0)

        # Odds display
        if is_real_odds:
            st.markdown("**📊 Real Bookmaker Odds**")
        else:
            st.markdown("**🤖 Simulated Odds** (from model)")
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

        # Pattern-Based Adjustments (NEW - from error analysis)
        if 'pattern_adjustments' in pred and pred['pattern_adjustments']:
            st.markdown("### 🔧 Smart Adjustments Applied")
            st.markdown("""
            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; margin-bottom: 15px;">
            <b>⚡ What are Smart Adjustments?</b><br>
            The AI detected patterns where the base prediction might be wrong. These adjustments are based on analysis of 94+ historical games where the model struggled.<br>
            <em>Think of it as the model saying: "I've seen this situation before and I was wrong - let me correct it."</em>
            </div>
            """, unsafe_allow_html=True)

            for adjustment in pred['pattern_adjustments']:
                if "Hot road team" in adjustment:
                    st.success(f"""
                    **✅ {adjustment}**

                    **What it means**: The away team is on a 4+ game winning streak and the ELO rating gap is small (<100 points).

                    **Why it matters**: Historically, hot road teams in this situation win 60% of the time, even when the model slightly favors the home team. The model boosted the away team's win probability by 10% to account for momentum.

                    **Translation**: The away team is playing with confidence and momentum trumps slight rating disadvantages.
                    """)
                elif "Cold home team" in adjustment:
                    st.warning(f"""
                    **⚠️ {adjustment}**

                    **What it means**: The home team is on a 3+ game losing streak and the ELO rating gap is moderate (<150 points).

                    **Why it matters**: Teams in free fall lose 73% of games even at home. The model reduced the home team's win probability by 8% to account for poor form.

                    **Translation**: The home team is struggling badly and home court advantage doesn't overcome that slump.
                    """)
                elif "Heavy travel" in adjustment:
                    st.error(f"""
                    **🛫 {adjustment}**

                    **What it means**: The away team traveled 2000+ miles (cross-country) and is playing on zero rest (back-to-back game).

                    **Why it matters**: This brutal schedule combination has a 100% historical loss rate. The model reduced the away team's win probability by 15% - a massive penalty.

                    **Translation**: Extreme fatigue from cross-country travel + no rest = expect a tired, sluggish performance.
                    """)
                elif "Large ELO" in adjustment:
                    st.info(f"""
                    **📊 {adjustment}**

                    **What it means**: The ELO rating difference between teams is very large (>200 points) - a big mismatch on paper.

                    **Why it matters**: The model historically had a 50% error rate in these "obvious favorite" games. It was overconfident. The adjustment reduces the favorite's probability by 5% to avoid overconfidence.

                    **Translation**: Even heavy favorites can lose - the model is being more cautious about blowout predictions.
                    """)
                elif "B2B penalty" in adjustment:
                    st.warning(f"""
                    **😴 {adjustment}**

                    **What it means**: The home team is playing on zero rest (back-to-back) while the away team is rested.

                    **Why it matters**: Fatigue matters even at home. The model reduced the home team's win probability by 6% to account for tired legs.

                    **Translation**: The home team might be sluggish in the 4th quarter when fresh legs matter most.
                    """)

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
                import tweepy
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
                    st.session_state[post_mode_key] = "Thread (Multiple Charts)"  # Default to Thread

                post_mode = st.radio(
                    "Post Format:",
                    ["Thread (Multiple Charts)", "Single Tweet (Composite Image)"],  # Thread first
                    index=0 if st.session_state[post_mode_key] == "Thread (Multiple Charts)" else 1,
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
                                    st.info("💡 If you just updated your .env file, **restart Streamlit** to load new credentials.")
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
                                    
                                except tweepy.Unauthorized as auth_error:
                                    error_msg = (
                                        f"❌ 401 Unauthorized: Invalid credentials.\n\n"
                                        f"This usually means:\n"
                                        f"1. Access Token doesn't match the API Key/Secret\n"
                                        f"2. Tokens were copied incorrectly (check for extra spaces)\n"
                                        f"3. Tokens belong to a different Twitter App\n\n"
                                        f"🔧 Fix:\n"
                                        f"1. Verify your API Key and Secret in .env match the app that generated the Access Token\n"
                                        f"2. In Twitter Developer Portal, go to your app → Keys and tokens\n"
                                        f"3. Make sure you're using tokens from the SAME app\n"
                                        f"4. Regenerate Access Token and Secret if needed\n"
                                        f"5. Copy tokens carefully (no extra spaces)\n"
                                        f"6. Restart Streamlit after updating .env"
                                    )
                                    st.error(error_msg)
                                    st.error(f"Technical error: {auth_error}")
                                    return
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

                                    # NEW STRUCTURE (Dec 2024):
                                    # format_twitter_thread now returns:
                                    # Tweet 1: Main prediction (no image)
                                    # Tweet 2: THE EDGE (situational chart)
                                    # Tweet 3: RECENT FORM (ratings_l10 chart)
                                    # Tweet 4: KEY MATCHUP (ratings_l10 or shooting_l10, dynamic)
                                    # Tweet 5: SCHEDULE SPOT (situational chart)
                                    # Tweet 6: HOME/ROAD SPLITS (splits chart)
                                    # Tweet 7+: Smart Adjustments (no image) / CTA (no image)

                                    # The method already returns the correct image_paths, so we can use them directly!
                                    thread_texts = thread_texts_from_daily

                                    # Use the image paths from format_twitter_thread
                                    # If they're empty or failed, use our exported charts as fallback
                                    if thread_image_paths_from_daily and any(thread_image_paths_from_daily):
                                        # Use the paths from format_twitter_thread
                                        filtered_texts = thread_texts
                                        filtered_charts_with_none = thread_image_paths_from_daily
                                    else:
                                        # Fallback: map our exported charts to the new structure
                                        filtered_texts = thread_texts
                                        filtered_charts_with_none = []

                                        # NEW MAPPING for new thread structure:
                                        new_chart_mapping = [
                                            None,  # Tweet 1: Main prediction (no image)
                                            'situational',  # Tweet 2: THE EDGE
                                            'ratings_l10',  # Tweet 3: RECENT FORM
                                            'ratings_l10',  # Tweet 4: KEY MATCHUP (default to ratings, could be shooting)
                                            'situational',  # Tweet 5: SCHEDULE SPOT
                                            'splits',       # Tweet 6: HOME/ROAD SPLITS
                                        ]

                                        for i, tweet_text in enumerate(thread_texts):
                                            if i < len(new_chart_mapping):
                                                chart_name = new_chart_mapping[i]
                                                if chart_name and chart_name in chart_names_exported:
                                                    chart_idx_in_order = chart_order.index(chart_name)
                                                    if chart_idx_in_order < len(exported_charts):
                                                        filtered_charts_with_none.append(exported_charts[chart_idx_in_order])
                                                    else:
                                                        filtered_charts_with_none.append(None)
                                                else:
                                                    filtered_charts_with_none.append(None)
                                            else:
                                                # Tweets 7+ (Smart Adjustments, CTA) have no images
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Today", "Features Explorer", "Analytics", "Train", "Performance", "Settings", "Twitter Status"])

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

                                    # Fetch odds for this game
                                    odds, is_real_odds, _ = get_real_or_simulated_odds(h, a, p['home_win_probability'])
                                    if is_real_odds:
                                        home_odds = odds.get('avg_home_odds', 2.0)
                                        away_odds = odds.get('avg_away_odds', 2.0)
                                    else:
                                        home_odds = odds['bookmakers'].get('Pinnacle', {}).get('home', 2.0)
                                        away_odds = odds['bookmakers'].get('Pinnacle', {}).get('away', 2.0)

                                    preds.append(p)
                                    save_prediction_to_db(str(db_path), {
                                        'game_date': game_date,  # Use actual game date, not today
                                        'home_team': h, 'away_team': a,
                                        'predicted_winner': h if p['prediction'] == 'home' else a,
                                        'predicted_home_prob': p['home_win_probability'],
                                        'predicted_away_prob': p['away_win_probability'],
                                        'confidence': p['confidence'],
                                        'features': p.get('features', {}),
                                        'home_odds': home_odds,
                                        'away_odds': away_odds
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

                            # Auto-export to pending_games.json and push to GitHub
                            with st.spinner("📤 Exporting to GitHub Pages..."):
                                try:
                                    from src.daily_games_exporter import DailyGamesExporter
                                    import subprocess

                                    # Export predictions to JSON
                                    exporter = DailyGamesExporter(str(db_path))
                                    export_success = exporter.export_games_for_publishing(date_str)

                                    if export_success:
                                        # Git commit and push (include database for GitHub Actions)
                                        try:
                                            subprocess.run(['git', 'add', 'docs/pending_games.json'], check=True, capture_output=True)
                                            subprocess.run(['git', 'add', 'data/nba_predictor.db'], check=True, capture_output=True)
                                            commit_result = subprocess.run(
                                                ['git', 'commit', '-m', f'Auto-export predictions for {date_str}'],
                                                capture_output=True,
                                                text=True
                                            )
                                            # Only push if there was something to commit
                                            if commit_result.returncode == 0:
                                                subprocess.run(['git', 'push'], check=True, capture_output=True)
                                                st.success("✅ Predictions + database pushed to GitHub!")
                                            else:
                                                st.info("ℹ️ Predictions exported (no changes to commit)")
                                        except subprocess.CalledProcessError as git_error:
                                            st.warning(f"⚠️ Predictions exported but git push failed: {git_error}")
                                    else:
                                        st.warning("⚠️ Failed to export predictions to GitHub Pages")
                                except Exception as export_error:
                                    st.warning(f"⚠️ Auto-export failed: {export_error}")
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

        # Email report button (persistent - always visible when predictions exist)
        st.markdown("---")
        col_email_label, col_email_btn = st.columns([3, 1])
        with col_email_label:
            st.markdown("**📧 Daily Email Report**")
            st.caption("Send today's predictions + yesterday's results")
        with col_email_btn:
            st.write("")
            if st.button("📧 Send Email", key="send_email_persistent"):
                with st.spinner("Sending email report..."):
                    try:
                        from src.email_reporter import EmailReporter
                        email_reporter = EmailReporter(db_path=str(db_path))
                        success = email_reporter.send_daily_report(test_mode=False)
                        if success:
                            st.success("✅ Email sent successfully!")
                        else:
                            st.error("❌ Failed to send email. Check logs.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)[:100]}")

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
# TAB 2: FEATURES EXPLORER
# =============================================================================
# Feature definitions with categories, descriptions, and computation details
FEATURE_DEFINITIONS = {
    # Elo Ratings
    "home_elo": {
        "category": "Elo Ratings",
        "name": "Home Team Elo",
        "description": "Elo rating of the home team. 1500 is average, higher is better.",
        "computation": "Calculated using the Elo rating system: after each game, the winner gains points and the loser loses points. The amount exchanged depends on the expected outcome - beating a stronger team gives more points.",
        "range": "Typically 1350-1650",
        "impact": "Higher Elo = stronger team historically"
    },
    "away_elo": {
        "category": "Elo Ratings",
        "name": "Away Team Elo",
        "description": "Elo rating of the away team.",
        "computation": "Same as home_elo but for the away team.",
        "range": "Typically 1350-1650",
        "impact": "Higher Elo = stronger team historically"
    },
    "elo_diff": {
        "category": "Elo Ratings",
        "name": "Elo Difference",
        "description": "Home Elo minus Away Elo. Positive means home team is stronger.",
        "computation": "Simply home_elo - away_elo",
        "range": "-300 to +300",
        "impact": "Each 100 points ≈ 64% expected win rate for the higher-rated team"
    },
    "elo_win_prob": {
        "category": "Elo Ratings",
        "name": "Elo Win Probability",
        "description": "Expected home win probability based purely on Elo ratings.",
        "computation": "1 / (1 + 10^((away_elo - home_elo - home_advantage) / 400))",
        "range": "0.0 to 1.0",
        "impact": "Direct probability estimate from Elo system"
    },

    # Strength of Schedule (NEW)
    "home_sos_normalized": {
        "category": "Strength of Schedule",
        "name": "Home SOS (Normalized)",
        "description": "Strength of schedule for home team's last 10 games. Higher = faced tougher opponents.",
        "computation": "Average opponent Elo normalized to 0-1 scale: (avg_opp_elo - 1350) / 300",
        "range": "0.0 to 1.0",
        "impact": "Helps distinguish between good records vs weak teams and bad records vs strong teams"
    },
    "away_sos_normalized": {
        "category": "Strength of Schedule",
        "name": "Away SOS (Normalized)",
        "description": "Strength of schedule for away team's last 10 games.",
        "computation": "Same as home_sos_normalized but for away team.",
        "range": "0.0 to 1.0",
        "impact": "Higher = faced tougher opponents recently"
    },
    "home_sos_adjusted_win_pct": {
        "category": "Strength of Schedule",
        "name": "Home SOS-Adjusted Win %",
        "description": "Home team's win percentage adjusted for opponent strength.",
        "computation": "Raw win% adjusted based on SOS. Beating strong teams → higher adjusted win%. Beating weak teams → lower adjusted win%.",
        "range": "0.0 to 1.0",
        "impact": "More accurate representation of team quality than raw win%"
    },
    "away_sos_adjusted_win_pct": {
        "category": "Strength of Schedule",
        "name": "Away SOS-Adjusted Win %",
        "description": "Away team's win percentage adjusted for opponent strength.",
        "computation": "Same as home but for away team.",
        "range": "0.0 to 1.0",
        "impact": "Reduces recency bias from easy/hard schedules"
    },
    "sos_adjusted_form_differential": {
        "category": "Strength of Schedule",
        "name": "SOS-Adjusted Form Differential",
        "description": "Difference in SOS-adjusted win percentages. THE KEY ANTI-RECENCY-BIAS FEATURE.",
        "computation": "home_sos_adjusted_win_pct - away_sos_adjusted_win_pct",
        "range": "-1.0 to 1.0",
        "impact": "Positive = home team has better quality-adjusted form"
    },

    # Recent Form
    "home_last10_win_pct": {
        "category": "Recent Form",
        "name": "Home Last 10 Win %",
        "description": "Home team's win percentage over last 10 games.",
        "computation": "Count of wins / 10",
        "range": "0.0 to 1.0",
        "impact": "Captures current momentum"
    },
    "away_last10_win_pct": {
        "category": "Recent Form",
        "name": "Away Last 10 Win %",
        "description": "Away team's win percentage over last 10 games.",
        "computation": "Count of wins / 10",
        "range": "0.0 to 1.0",
        "impact": "Captures current momentum"
    },
    "home_last10_ppg": {
        "category": "Recent Form",
        "name": "Home Last 10 PPG",
        "description": "Home team's points per game over last 10 games.",
        "computation": "Average of points scored in last 10 games",
        "range": "Typically 100-130",
        "impact": "Offensive output indicator"
    },
    "home_last10_net_rating": {
        "category": "Recent Form",
        "name": "Home Last 10 Net Rating",
        "description": "Home team's offensive rating minus defensive rating (last 10 games).",
        "computation": "(Points scored per 100 possessions) - (Points allowed per 100 possessions)",
        "range": "-20 to +20",
        "impact": "Best single measure of recent team quality"
    },
    "weighted_form_differential": {
        "category": "Recent Form",
        "name": "Weighted Form Differential",
        "description": "Difference in weighted recent win percentages.",
        "computation": "Uses 35% last-3, 35% last-5, 30% last-10 weighting for each team, then takes difference",
        "range": "-1.0 to 1.0",
        "impact": "Balances recency with stability"
    },

    # Home/Away Splits
    "home_team_home_win_pct": {
        "category": "Home/Away Splits",
        "name": "Home Team's Home Win %",
        "description": "How well the home team performs when playing at home.",
        "computation": "Win % from last 15 home games",
        "range": "0.0 to 1.0",
        "impact": "Some teams have strong home court advantage"
    },
    "away_team_road_win_pct": {
        "category": "Home/Away Splits",
        "name": "Away Team's Road Win %",
        "description": "How well the away team performs when playing on the road.",
        "computation": "Win % from last 15 road games",
        "range": "0.0 to 1.0",
        "impact": "Some teams struggle on the road"
    },

    # Rest & Schedule
    "home_rest_days": {
        "category": "Rest & Schedule",
        "name": "Home Rest Days",
        "description": "Days since home team's last game.",
        "computation": "Current date - last game date",
        "range": "0-7+ days",
        "impact": "More rest generally helps, but too much can cause rust"
    },
    "away_rest_days": {
        "category": "Rest & Schedule",
        "name": "Away Rest Days",
        "description": "Days since away team's last game.",
        "computation": "Current date - last game date",
        "range": "0-7+ days",
        "impact": "0 days = back-to-back (fatigue)"
    },
    "rest_advantage": {
        "category": "Rest & Schedule",
        "name": "Rest Advantage",
        "description": "Home rest days minus away rest days.",
        "computation": "home_rest_days - away_rest_days",
        "range": "-5 to +5",
        "impact": "Positive = home team more rested"
    },
    "home_b2b": {
        "category": "Rest & Schedule",
        "name": "Home Back-to-Back",
        "description": "Is home team playing on consecutive days?",
        "computation": "1 if last game was yesterday, 0 otherwise",
        "range": "0 or 1",
        "impact": "B2B typically hurts performance by 2-3%"
    },

    # Streaks
    "home_streak": {
        "category": "Streaks",
        "name": "Home Team Streak",
        "description": "Current win/loss streak for home team.",
        "computation": "Positive = consecutive wins, Negative = consecutive losses",
        "range": "-10 to +10",
        "impact": "Hot streaks may indicate confidence/momentum"
    },
    "away_streak": {
        "category": "Streaks",
        "name": "Away Team Streak",
        "description": "Current win/loss streak for away team.",
        "computation": "Positive = consecutive wins, Negative = consecutive losses",
        "range": "-10 to +10",
        "impact": "Cold streaks may indicate problems"
    },

    # Head-to-Head
    "h2h_home_win_pct": {
        "category": "Head-to-Head",
        "name": "H2H Home Win %",
        "description": "Home team's historical win % against this opponent.",
        "computation": "Wins / Total games between these teams",
        "range": "0.0 to 1.0",
        "impact": "Some matchups favor certain teams"
    },
    "h2h_total_games": {
        "category": "Head-to-Head",
        "name": "H2H Total Games",
        "description": "Total games played between these teams.",
        "computation": "Count of historical matchups",
        "range": "0-50+",
        "impact": "More games = more reliable H2H stats"
    },
}

with tab2:
    st.markdown("## 📚 Features Explorer")
    st.markdown("Explore all the features used by the prediction model. Select a feature to learn how it's computed and why it matters.")

    # Get categories
    categories = sorted(set(f["category"] for f in FEATURE_DEFINITIONS.values()))

    # Category filter
    selected_category = st.selectbox("📂 Filter by Category", ["All Categories"] + categories)

    # Filter features by category
    if selected_category == "All Categories":
        filtered_features = FEATURE_DEFINITIONS
    else:
        filtered_features = {k: v for k, v in FEATURE_DEFINITIONS.items() if v["category"] == selected_category}

    # Feature dropdown
    feature_names = {k: v["name"] for k, v in filtered_features.items()}
    selected_feature_key = st.selectbox(
        "🔍 Select Feature",
        options=list(feature_names.keys()),
        format_func=lambda x: f"{feature_names[x]} ({x})"
    )

    if selected_feature_key:
        feature = FEATURE_DEFINITIONS[selected_feature_key]

        st.markdown("---")

        # Feature details card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); border-radius: 12px; padding: 20px; color: white; margin-bottom: 20px;">
            <div style="font-size: 0.8rem; opacity: 0.8; text-transform: uppercase;">{feature['category']}</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin: 8px 0;">{feature['name']}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{feature['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Details in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧮 How It's Computed")
            st.info(feature['computation'])

        with col2:
            st.markdown("### 📊 Typical Range")
            st.info(feature['range'])

        st.markdown("### 💡 Impact on Predictions")
        st.success(feature['impact'])

        # Show example values if we can compute them
        st.markdown("---")
        st.markdown("### 📈 Live Example")

        try:
            fe = FeatureEngineer(str(db_path))
            # Use a sample game (Lakers vs Celtics)
            sample_features = fe.create_features_for_game(1610612747, 1610612738)

            if selected_feature_key in sample_features:
                value = sample_features[selected_feature_key]

                # Create a simple gauge visualization
                if isinstance(value, (int, float)):
                    st.metric(
                        label=f"Current Value (Lakers vs Celtics example)",
                        value=f"{value:.3f}" if isinstance(value, float) else str(value)
                    )

                    # For normalized features (0-1), show a progress bar
                    if 'pct' in selected_feature_key or 'normalized' in selected_feature_key or 'prob' in selected_feature_key:
                        st.progress(min(max(float(value), 0), 1))
            else:
                st.caption(f"Feature '{selected_feature_key}' not available in current computation")

        except Exception as e:
            st.caption(f"Could not compute example: {e}")

    # Category summary at bottom
    st.markdown("---")
    st.markdown("### 📋 Feature Categories Summary")

    cat_counts = {}
    for f in FEATURE_DEFINITIONS.values():
        cat = f["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    cols = st.columns(len(cat_counts))
    for i, (cat, count) in enumerate(sorted(cat_counts.items())):
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 1.2rem; font-weight: 700; color: #1e3a5f;">{count}</div>
                <div style="font-size: 0.75rem; color: #64748b;">{cat}</div>
            </div>
            """, unsafe_allow_html=True)

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

    st.markdown("#### Training Date Range")
    c1, c2, c3, c4 = st.columns(4)

    # Month names for display
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Start date selection
    start_yr = c1.selectbox("Start Year", [2022, 2023, 2024, 2025], index=2)  # Default to 2024
    start_month = c2.selectbox("Start Month", months, index=9)  # Default to October (start of season)

    # End date - default to current month/year
    current_year = datetime.now().year
    current_month_idx = datetime.now().month - 1  # 0-indexed
    end_yr = c3.selectbox("End Year", [2023, 2024, 2025, 2026], index=3)  # Default to 2026
    end_month = c4.selectbox("End Month", months, index=current_month_idx)  # Default to current month

    # Convert month names to numbers
    start_month_num = months.index(start_month) + 1
    end_month_num = months.index(end_month) + 1

    st.info("💡 **Tip**: Using only recent data (e.g., Oct 2024 - today) trains faster and focuses on current team dynamics. Older data may be less relevant due to roster changes.")

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

            # Filter by date range (using month selection)
            start_date = f"{start_yr}-{start_month_num:02d}-01"
            # End date: last day of selected month
            if end_month_num == 12:
                end_date = f"{end_yr}-12-31"
            else:
                end_date = f"{end_yr}-{end_month_num + 1:02d}-01"  # First of next month

            games_before = len(games)
            games = games[(games['game_date'] >= start_date) & (games['game_date'] < end_date)]
            log.write(f"📅 Filtered to {start_month} {start_yr} - {end_month} {end_yr}: {len(games)} games (was {games_before})")
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
# TAB 7: TWITTER STATUS (24-Hour Rate Limits)
# =============================================================================
with tab7:
    st.markdown("## 🐦 Twitter Rate Limits")
    st.caption("Free Tier: 17 tweets per 24 hours")

    if st.button("🔄 Check Status"):
        try:
            from src.tweet_counter import get_tweet_count_24h

            with st.spinner("Counting your tweets from the last 24 hours..."):
                # Get tweet count from local log
                tweet_count_result = get_tweet_count_24h()

                if 'error' in tweet_count_result:
                    st.warning(f"⚠️ Error loading tweet count: {tweet_count_result['error']}")
                else:
                    st.success("✅ Retrieved tweet count from local history")

            # Display the count prominently
            if tweet_count_result:
                count = tweet_count_result['count']
                limit = tweet_count_result['limit']
                remaining = tweet_count_result['remaining']

                # Overall status
                if remaining > 0:
                    st.success("✅ Can post tweets")
                else:
                    st.error("🔴 Daily tweet limit reached")

                # Main metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Tweets Posted (24h)",
                        f"{count}",
                        delta=f"-{count} from limit",
                        delta_color="inverse"
                    )

                with col2:
                    st.metric(
                        "Remaining",
                        f"{remaining}",
                        delta=f"out of {limit}",
                        delta_color="normal"
                    )

                with col3:
                    # Show percentage used
                    pct_used = (count / limit * 100) if limit > 0 else 0
                    st.metric("Usage", f"{pct_used:.1f}%")

                # Progress bar
                st.progress(min(count / limit, 1.0) if limit > 0 else 0)

                # Show recent tweets if available
                if tweet_count_result.get('tweets'):
                    with st.expander("📝 Recent tweets (last 24h)", expanded=False):
                        for tweet in tweet_count_result['tweets']:
                            posted_at = tweet.get('posted_at', '')
                            text = tweet.get('text', '')
                            age_hours = tweet.get('age_hours', 0)
                            st.caption(f"**{posted_at}** ({age_hours:.1f}h ago): {text}")

                # Show API rate limit info if available
                if tweet_count_result.get('api_rate_limit'):
                    api_limit = tweet_count_result['api_rate_limit']
                    if api_limit.get('remaining') is not None:
                        # Show reset time if limit exhausted
                        if api_limit.get('remaining') == 0 and api_limit.get('reset'):
                            from datetime import datetime, timezone
                            reset_ts = api_limit['reset']
                            reset_time = datetime.fromtimestamp(int(reset_ts), tz=timezone.utc)
                            reset_local = reset_time.astimezone()
                            hours_until = (reset_time - datetime.now(timezone.utc)).total_seconds() / 3600

                            st.error(f"🔴 Twitter API says: {api_limit['limit']}/{api_limit['limit']} tweets used (LIMIT EXHAUSTED)")
                            st.error(f"⏰ Resets at: {reset_local.strftime('%Y-%m-%d %H:%M:%S')} ({hours_until:.1f} hours from now)")
                        else:
                            st.info(f"🔄 API rate limit: {api_limit['remaining']}/{api_limit['limit']} remaining")

                st.caption("💡 This tracks tweets posted through this app")
                st.caption("⚠️ Note: If you posted tweets outside this app (via Twitter.com or other tools), they won't appear in the local count above, but Twitter still counts them toward your 17/24h limit.")

                # Show time until next tweet is available (if at limit)
                if remaining == 0 and tweet_count_result.get('oldest_tweet_age_hours'):
                    hours_until_available = 24 - tweet_count_result['oldest_tweet_age_hours']
                    if hours_until_available > 0:
                        h = int(hours_until_available)
                        m = int((hours_until_available - h) * 60)
                        st.info(f"⏰ Next tweet available in: {h}h {m}m (when oldest tweet turns 24h old)")

            else:
                st.warning("⚠️ Could not retrieve tweet count")
                st.info("""
                **📝 Note:**

                This feature tracks tweets posted through this app in the last 24 hours.

                Free tier allows **17 tweets per 24 hours** (excludes retweets and replies).

                The counter automatically logs each tweet you post and cleans up tweets older than 48 hours.
                """)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👆 Click to check your current Twitter API rate limits")

st.caption("NBA Predictor v2.1")

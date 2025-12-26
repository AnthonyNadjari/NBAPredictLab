"""
NBA Game Predictor
Main prediction logic for the application
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import StackedEnsembleModel
from src.data_fetcher import NBADataFetcher, FeatureEngineer
from nba_api.stats.static import teams


class NBAPredictor:
    """Main predictor class"""

    def __init__(self, db_path='data/nba_predictor.db', model_dir='models'):
        self.model = StackedEnsembleModel()
        self.data_fetcher = NBADataFetcher(db_path)
        self.feature_engineer = FeatureEngineer(db_path)
        # Handle case where db_path is passed as model_dir (backwards compatibility)
        if model_dir.endswith('.db'):
            self.model_dir = 'models'
        else:
            self.model_dir = model_dir
        self.model_loaded = False
        
        # Team name to ID mapping
        self.team_map = {t['full_name']: t['id'] for t in teams.get_teams()}
        self.team_abbrev_map = {t['abbreviation']: t['id'] for t in teams.get_teams()}
        self.id_to_name = {t['id']: t['full_name'] for t in teams.get_teams()}
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model.load(self.model_dir)
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _get_team_id(self, team_name):
        """Convert team name to team ID"""
        # Try full name first
        if team_name in self.team_map:
            return self.team_map[team_name]
        # Try abbreviation
        if team_name in self.team_abbrev_map:
            return self.team_abbrev_map[team_name]
        # Try partial match
        for name, tid in self.team_map.items():
            if team_name.lower() in name.lower() or name.lower() in team_name.lower():
                return tid
        return None
    
    def predict_game(self, home_team, away_team, game_date=None):
        """Predict the outcome of a game using real features"""
        if not self.model_loaded:
            if not self.load_model():
                return None
        
        # Get team IDs
        home_team_id = self._get_team_id(home_team)
        away_team_id = self._get_team_id(away_team)
        
        if home_team_id is None or away_team_id is None:
            return None
        
        # Create features using FeatureEngineer
        try:
            features = self.feature_engineer.create_features_for_game(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                game_date=game_date,
                include_player_stats=True  # Now enabled with robust caching
            )
            
            # Ensure features is a dictionary (even if empty)
            if not isinstance(features, dict):
                print(f"Warning: Features not a dict, got {type(features)}")
                features = {}
            
            # Debug: Check if features were created
            if not features or len(features) == 0:
                print(f"Warning: Features dictionary is empty for {home_team} vs {away_team}")
                print(f"Home team ID: {home_team_id}, Away team ID: {away_team_id}")
            else:
                print(f"Created {len(features)} features for {home_team} vs {away_team}")
            
            # Ensure features is ALWAYS a dict before prediction
            if not isinstance(features, dict):
                print(f"WARNING: Features is not a dict, type: {type(features)}, value: {features}")
                features = {}
            
            # Create a SAFE copy of features that will definitely persist
            import copy
            features_backup = {}
            try:
                features_backup = copy.deepcopy(features) if features else {}
            except:
                # If deepcopy fails, use regular dict copy
                features_backup = dict(features) if features else {}
            
            print(f"📊 Features before prediction: {len(features_backup)} features")
            
            # Make prediction
            result = self.model.predict_single(features)
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                print(f"Error: Model predict_single returned non-dict: {type(result)}")
                return None
            
            print(f"📋 Result keys after predict_single: {list(result.keys())}")
            
            # Add team names FIRST
            result['home_team'] = home_team
            result['away_team'] = away_team
            
            # FORCE features to be included - do this BEFORE any other operations
            result['features'] = features_backup  # Always include, even if empty
            
            print(f"📋 Result keys after adding features: {list(result.keys())}")
            print(f"📋 Features count: {len(result.get('features', {}))}")
            
            # CRITICAL: Verify features were added
            if 'features' not in result:
                print(f"❌ CRITICAL ERROR: Features still not in result! Force adding...")
                result['features'] = features_backup
            else:
                feat_count = len(result['features']) if isinstance(result['features'], dict) else 0
                print(f"✅ Features confirmed in result: {feat_count} features")
            
            # Final verification - ensure features is always a dict
            if not isinstance(result.get('features'), dict):
                print(f"⚠️ WARNING: Features is not a dict, fixing... Type: {type(result.get('features'))}")
                result['features'] = features_backup if isinstance(features_backup, dict) else {}
            
            # Print final verification
            final_keys = list(result.keys())
            has_features = 'features' in result
            feat_type = type(result.get('features'))
            feat_len = len(result.get('features', {})) if isinstance(result.get('features'), dict) else 0
            
            print(f"📋 FINAL VERIFICATION:")
            print(f"   - Result keys: {final_keys}")
            print(f"   - Has 'features' key: {has_features}")
            print(f"   - Features type: {feat_type}")
            print(f"   - Features count: {feat_len}")
            
            if not has_features:
                print(f"❌❌❌ FEATURES MISSING FROM RESULT! This should never happen! ❌❌❌")
                result['features'] = features_backup
            
            return result
            
        except Exception as e:
            import traceback
            print(f"Error making prediction: {e}")
            print(traceback.format_exc())
            # Return a result with empty features so the UI can still display something
            return {
                'prediction': None,
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'confidence': 0.0,
                'top_factors': [],
                'base_model_predictions': {},
                'home_team': home_team,
                'away_team': away_team,
                'features': {},  # Always include features, even if empty
                'error': str(e)
            }
    
    def predict_game_batch(self, games_list, progress_callback=None, max_workers=4):
        """
        Predict multiple games in parallel using multi-threading.

        Args:
            games_list: List of tuples (home_team, away_team, game_date)
            progress_callback: Optional callback function(completed, total) for progress updates
            max_workers: Number of parallel threads (default: 4)

        Returns:
            List of prediction results
        """
        if not self.model_loaded:
            if not self.load_model():
                return []

        predictions = []
        total_games = len(games_list)
        completed = 0
        lock = Lock()

        def predict_single_game(game_info):
            """Thread worker function"""
            home_team, away_team, game_date = game_info
            result = self.predict_game(home_team, away_team, game_date)
            return result

        # Use ThreadPoolExecutor for parallel predictions
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_game = {
                executor.submit(predict_single_game, game): game
                for game in games_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_game):
                try:
                    result = future.result()
                    if result:
                        with lock:
                            predictions.append(result)
                            completed += 1
                            if progress_callback:
                                progress_callback(completed, total_games)
                except Exception as e:
                    print(f"Error predicting game: {e}")
                    with lock:
                        completed += 1
                        if progress_callback:
                            progress_callback(completed, total_games)

        return predictions

    def predict_upcoming_games(self):
        """Predict outcomes for upcoming games"""
        upcoming = self.data_fetcher.get_todays_games()

        if upcoming.empty:
            return pd.DataFrame()

        predictions = []
        for _, game in upcoming.iterrows():
            home_team_id = game['home_team_id']
            away_team_id = game['away_team_id']

            home_team = self.id_to_name.get(home_team_id, f"Team {home_team_id}")
            away_team = self.id_to_name.get(away_team_id, f"Team {away_team_id}")

            result = self.predict_game(home_team, away_team, game['game_date'])

            if result:
                predictions.append(result)

        return pd.DataFrame(predictions)

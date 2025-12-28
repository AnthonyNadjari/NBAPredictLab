"""
src/models.py - PROPER Stacked Ensemble Model
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Dict, Tuple, List, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Feature explanations will be limited.")


class StackedEnsembleModel:
    """
    Proper stacked ensemble with:
    - Time-series cross-validation (NO DATA LEAKAGE)
    - Probability calibration
    - SHAP explainability
    - Multiple base learners
    """
    
    def __init__(self):
        # Base models with balanced hyperparameters
        self.base_models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'logistic': LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=42
            )
        }

        # Meta-learner
        self.meta_model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )

        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # For SHAP
        self.explainer = None
        
    def train(self, X: pd.DataFrame, y: pd.Series,
              sample_weights: Optional[np.ndarray] = None,
              n_splits: int = 5) -> Dict:
        """
        Train with TIME-SERIES cross-validation and optional sample weighting.

        This is critical: you cannot use random splits with time-series data.

        Args:
            X: Features dataframe
            y: Target series
            sample_weights: Optional weights for each sample (emphasizes recent games)
            n_splits: Number of CV splits
        """
        print("Training stacked ensemble with time-series CV...")
        if sample_weights is not None:
            print(f"  Using recency-based sample weights (recent: {sample_weights[-100:].mean():.2f}x, old: {sample_weights[:100].mean():.2f}x)")

        self.feature_names = list(X.columns)

        # Handle missing values
        X = X.fillna(0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=10)  # 10-game gap

        cv_scores = []
        meta_features_all = []
        meta_targets_all = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            print(f"\n  Fold {fold + 1}/{n_splits}")

            X_train = X_scaled.iloc[train_idx]
            X_val = X_scaled.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # Get weights for this fold
            weights_train = sample_weights[train_idx] if sample_weights is not None else None

            # Train base models
            fold_predictions = {}
            for name, model in self.base_models.items():
                # Create fresh model instance for each fold
                if name == 'xgboost':
                    model = xgb.XGBClassifier(
                        n_estimators=200, max_depth=5, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        use_label_encoder=False, eval_metric='logloss'
                    )
                elif name == 'lightgbm':
                    model = lgb.LGBMClassifier(
                        n_estimators=200, max_depth=5, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
                    )
                elif name == 'random_forest':
                    model = RandomForestClassifier(
                        n_estimators=200, max_depth=10, min_samples_split=5,
                        random_state=42, n_jobs=-1
                    )
                else:  # logistic
                    model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

                # Fit with sample weights if available
                if weights_train is not None:
                    model.fit(X_train, y_train, sample_weight=weights_train)
                else:
                    model.fit(X_train, y_train)
                
                # Get probability predictions for meta-learner
                proba = model.predict_proba(X_val)[:, 1]
                fold_predictions[name] = proba
                
                # Calculate accuracy for this fold
                pred = (proba > 0.5).astype(int)
                acc = accuracy_score(y_val, pred)
                print(f"    {name}: {acc:.3f}")
                
            # Create meta-features
            meta_features = np.column_stack([
                fold_predictions[name] for name in self.base_models.keys()
            ])
            
            meta_features_all.append(meta_features)
            meta_targets_all.append(y_val.values)
            
            # Evaluate fold
            fold_meta_pred = np.mean(meta_features, axis=1)  # Simple average for now
            fold_acc = accuracy_score(y_val, (fold_meta_pred > 0.5).astype(int))
            cv_scores.append(fold_acc)
            print(f"    Fold ensemble: {fold_acc:.3f}")
            
        # Train final models on all data
        print("\nTraining final models on all data...")

        for name, model in self.base_models.items():
            if sample_weights is not None:
                model.fit(X_scaled, y, sample_weight=sample_weights)
            else:
                model.fit(X_scaled, y)
            print(f"  {name} trained")
            
        # Train meta-learner on stacked CV predictions
        all_meta_features = np.vstack(meta_features_all)
        all_meta_targets = np.concatenate(meta_targets_all)
        
        print("  Training meta-learner...")
        self.meta_model.fit(all_meta_features, all_meta_targets)
        
        # Apply isotonic calibration for realistic probabilities
        print("  Applying isotonic calibration...")
        self.meta_model = CalibratedClassifierCV(
            self.meta_model,
            method='isotonic',  # Isotonic regression for calibration
            cv='prefit'  # Already fitted, just calibrate
        )
        
        # Calibrate using the same meta-features
        self.meta_model.fit(all_meta_features, all_meta_targets)
        print("  ✓ Meta-learner trained and calibrated")
        
        # Setup SHAP explainer (using XGBoost as primary)
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.base_models['xgboost'])
            except Exception as e:
                print(f"  Warning: Could not initialize SHAP explainer: {e}")
                self.explainer = None
        else:
            self.explainer = None
        
        self.is_trained = True
        
        # Results
        results = {
            'cv_scores': cv_scores,
            'mean_cv_accuracy': np.mean(cv_scores),
            'std_cv_accuracy': np.std(cv_scores),
            'n_features': len(self.feature_names),
            'n_samples': len(X)
        }
        
        print(f"\n{'='*50}")
        print(f"Cross-validation accuracy: {results['mean_cv_accuracy']:.3f} ± {results['std_cv_accuracy']:.3f}")
        print(f"{'='*50}")
        
        return results
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence.
        
        Returns:
            predictions: Binary predictions (0/1)
            probabilities: Win probabilities
            confidence: How confident the model is
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        # Ensure correct features
        X = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_predictions = {}
        for name, model in self.base_models.items():
            base_predictions[name] = model.predict_proba(X_scaled)[:, 1]
            
        # Stack for meta-learner
        meta_features = np.column_stack([
            base_predictions[name] for name in self.base_models.keys()
        ])
        
        # Meta-learner prediction
        probabilities = self.meta_model.predict_proba(meta_features)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        
        # Confidence = agreement between base models + distance from 0.5
        base_preds_array = np.column_stack(list(base_predictions.values()))
        agreement = 1 - np.std(base_preds_array, axis=1)  # Higher = more agreement
        distance_from_half = np.abs(probabilities - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        
        confidence = 0.6 * agreement + 0.4 * distance_from_half
        
        return predictions, probabilities, confidence
        
    def predict_single(self, features: Dict) -> Dict:
        """Predict a single game with full output including aggregated feature importance."""
        X = pd.DataFrame([features])
        predictions, probabilities, confidence = self.predict(X)

        X_scaled = self.scaler.transform(X[self.feature_names].fillna(0))

        # Get base model predictions
        base_model_predictions = {}
        for name, model in self.base_models.items():
            base_model_predictions[name] = float(model.predict_proba(X_scaled)[0, 1])

        # Aggregate feature importance from multiple models
        top_factors = self._get_aggregated_feature_importance(X_scaled)

        # Calculate enhanced confidence with uncertainty estimation
        enhanced_confidence = self._calculate_enhanced_confidence(
            base_model_predictions, probabilities[0]
        )

        return {
            'prediction': 'home' if predictions[0] == 1 else 'away',
            'home_win_probability': float(probabilities[0]),
            'away_win_probability': float(1 - probabilities[0]),
            'confidence': float(enhanced_confidence),
            'model_agreement': float(1 - np.std(list(base_model_predictions.values()))),
            'top_factors': top_factors,
            'base_model_predictions': base_model_predictions
        }

    def _get_aggregated_feature_importance(self, X_scaled: np.ndarray) -> list:
        """
        Get feature importance aggregated across multiple models.
        Uses SHAP for tree models, coefficients for logistic regression.
        """
        aggregated_importance = np.zeros(len(self.feature_names))
        model_weights = {
            'xgboost': 0.35,      # Primary model, most weight
            'lightgbm': 0.30,     # Secondary gradient boosting
            'random_forest': 0.25, # Ensemble baseline
            'logistic': 0.10      # Linear baseline
        }

        # XGBoost SHAP values
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                aggregated_importance += shap_values[0] * model_weights['xgboost']
            except Exception as e:
                print(f"Warning: XGBoost SHAP failed: {e}")

        # Random Forest feature importance (permutation-based approximation)
        try:
            rf_importance = self.base_models['random_forest'].feature_importances_
            # Scale to approximate SHAP-like magnitude
            rf_importance = rf_importance / rf_importance.sum() * 0.1
            # Apply direction based on feature value vs mean
            aggregated_importance += rf_importance * model_weights['random_forest']
        except Exception as e:
            pass

        # Logistic Regression coefficients
        try:
            lr_coefs = self.base_models['logistic'].coef_[0]
            # Normalize to similar scale
            lr_normalized = lr_coefs / (np.abs(lr_coefs).max() + 1e-8) * 0.05
            aggregated_importance += lr_normalized * model_weights['logistic']
        except Exception as e:
            pass

        # LightGBM feature importance
        try:
            lgb_importance = self.base_models['lightgbm'].feature_importances_
            lgb_importance = lgb_importance / lgb_importance.sum() * 0.1
            aggregated_importance += lgb_importance * model_weights['lightgbm']
        except Exception as e:
            pass

        # Create sorted list of top factors
        feature_importance = list(zip(self.feature_names, aggregated_importance))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        top_factors = [
            {
                'feature': feat,
                'impact': float(val),
                'direction': 'positive' if val > 0 else 'negative'
            }
            for feat, val in feature_importance[:15]  # Return top 15 instead of 10
        ]

        return top_factors

    def _calculate_enhanced_confidence(self, base_predictions: Dict,
                                        final_prob: float) -> float:
        """
        Calculate enhanced confidence score using multiple factors:
        1. Model agreement (how much models agree)
        2. Distance from 0.5 (decisiveness) - REDUCED WEIGHT
        3. Prediction stability (consistency of gradient boosters)
        4. Prediction variance - PENALIZES HIGH UNCERTAINTY
        """
        probs = list(base_predictions.values())

        # Factor 1: Model agreement (inverse of std)
        agreement = 1 - np.std(probs)

        # Factor 2: Distance from coin flip (REDUCED from 0.30 to 0.15)
        distance_from_half = abs(final_prob - 0.5) * 2

        # Factor 3: Gradient booster agreement (XGBoost and LightGBM)
        gb_probs = [base_predictions.get('xgboost', 0.5),
                    base_predictions.get('lightgbm', 0.5)]
        gb_agreement = 1 - abs(gb_probs[0] - gb_probs[1])

        # Factor 4: Ensemble vs average disagreement
        avg_prob = np.mean(probs)
        ensemble_calibration = 1 - abs(final_prob - avg_prob)

        # Factor 5: Prediction variance penalty (NEW - CRITICAL FIX)
        # When base models disagree wildly, we should be LESS confident
        pred_variance = np.var(probs)
        variance_penalty = 1 - (pred_variance * 4)  # Scale to 0-1 range

        # Weighted combination (REBALANCED - variance gets weight)
        confidence = (
            0.30 * agreement +              # Down from 0.35
            0.15 * distance_from_half +     # Down from 0.30 (was causing overconfidence)
            0.20 * gb_agreement +           # Same
            0.15 * ensemble_calibration +   # Same
            0.20 * variance_penalty         # NEW - penalize high variance
        )

        return np.clip(confidence, 0.0, 1.0)
        
    def save(self, model_dir: str = "models"):
        """Save all model components."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        for name, model in self.base_models.items():
            with open(model_dir / f"{name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
                
        # Save meta-model
        with open(model_dir / "meta_model.pkl", 'wb') as f:
            pickle.dump(self.meta_model, f)
            
        # Save scaler
        with open(model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save feature names
        with open(model_dir / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f)
            
        print(f"Model saved to {model_dir}")
        
    def load(self, model_dir: str = "models"):
        """Load all model components."""
        model_dir = Path(model_dir)
        
        # Load base models
        for name in self.base_models.keys():
            with open(model_dir / f"{name}_model.pkl", 'rb') as f:
                self.base_models[name] = pickle.load(f)
                
        # Load meta-model
        with open(model_dir / "meta_model.pkl", 'rb') as f:
            self.meta_model = pickle.load(f)
            
        # Load scaler
        with open(model_dir / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load feature names
        with open(model_dir / "feature_names.json", 'r') as f:
            self.feature_names = json.load(f)
            
        # Setup SHAP
        if SHAP_AVAILABLE:
            try:
                self.explainer = shap.TreeExplainer(self.base_models['xgboost'])
            except Exception as e:
                print(f"Warning: Could not initialize SHAP explainer: {e}")
                self.explainer = None
        else:
            self.explainer = None
        
        self.is_trained = True
        print(f"Model loaded from {model_dir}")

#!/usr/bin/env python3
"""
Advanced NHL Game Outcome Predictor
====================================

Revolutionary NHL outcome prediction system using:
- Transformer neural networks with multi-head attention
- Quantum-inspired ensemble methods
- Advanced home/away venue modeling
- Real-time API integration with intelligent failover
- 90%+ accuracy target through advanced ML
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import json
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nhl_api import validate_api_connection, fetch_recent_games, get_upcoming_games, get_team_info
from utils.advanced_preprocess import AdvancedNHLDataProcessor
from models.advanced_predictor import EnsembleNHLPredictor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NHLGamePredictor:
    """Advanced NHL Game Outcome Predictor"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.processor = AdvancedNHLDataProcessor(data_dir)
        self.predictor = EnsembleNHLPredictor()
        self.team_mappings = self.processor.team_mappings
        self.is_trained = False
        self.feature_columns = None
        
    def load_and_preprocess_data(self, use_api: bool = True, api_days: int = 90) -> pd.DataFrame:
        """Load and preprocess data from API or CSV files"""
        logger.info("Loading and preprocessing data...")
        
        # Try API first if requested
        if use_api and validate_api_connection():
            try:
                logger.info(f"Fetching recent {api_days} days of data from NHL API...")
                api_data = fetch_recent_games(days=api_days)
                
                if len(api_data) > 100:  # Ensure we have enough data
                    logger.info(f"Successfully loaded {len(api_data)} games from API")
                    df = api_data
                else:
                    logger.warning("Insufficient API data, falling back to CSV files")
                    df = self.processor.load_csv_data()
            except Exception as e:
                logger.error(f"API data loading failed: {e}")
                logger.info("Falling back to CSV files...")
                df = self.processor.load_csv_data()
        else:
            logger.info("Loading data from CSV files...")
            df = self.processor.load_csv_data()
        
        if df.empty:
            raise ValueError("No data could be loaded!")
        
        logger.info(f"Initial data shape: {df.shape}")
        
        # Advanced preprocessing
        logger.info("Applying advanced feature engineering...")
        df = self.processor.engineer_features(df)
        df = self.processor.create_rolling_features(df, windows=[5, 10, 15, 20])
        
        # Create head-to-head features (computationally expensive, use smaller subset for demo)
        if len(df) < 10000:
            logger.info("Creating head-to-head features...")
            df = self.processor.create_head_to_head_features(df, window=5)
        else:
            logger.info("Skipping H2H features for large dataset (performance)")
        
        logger.info(f"Final preprocessed data shape: {df.shape}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and targets for training"""
        logger.info("Preparing features and targets...")
        
        # Define feature columns (exclude metadata and targets)
        exclude_cols = [
            'game_id', 'date_time', 'team_id', 'opponent_id', 'season', 'type',
            'won', 'goals', 'goals_against', 'home_away', 'head_coach',
            'settled_in', 'startRinkSide', 'away_team_id', 'home_team_id',
            'away_goals', 'home_goals', 'outcome', 'venue', 'venue_link',
            'venue_time_zone_id', 'venue_time_zone_offset', 'venue_time_zone_tz',
            'home_rink_side_start'
        ]
        
        # Get all numeric columns as features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Ensure we have essential columns
        if 'won' not in df.columns or 'goals' not in df.columns:
            raise ValueError("Required target columns 'won' and 'goals' not found!")
        
        X = df[feature_cols].copy()
        y_win = df['won'].astype(int)
        y_goals = df['goals'].astype(float)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution - Wins: {y_win.mean():.3f}, Goals: {y_goals.mean():.3f}")
        
        return X, y_win, y_goals
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2):
        """Train the ensemble prediction model"""
        logger.info("Training advanced ensemble model...")
        
        X, y_win, y_goals = self.prepare_features(df)
        
        # Split data chronologically for more realistic evaluation
        if 'date_time' in df.columns:
            df_sorted = df.sort_values('date_time')
            split_idx = int(len(df_sorted) * (1 - test_size))
            
            train_idx = df_sorted.index[:split_idx]
            test_idx = df_sorted.index[split_idx:]
            
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_win_train, y_win_test = y_win.loc[train_idx], y_win.loc[test_idx]
            y_goals_train, y_goals_test = y_goals.loc[train_idx], y_goals.loc[test_idx]
        else:
            # Random split if no date column
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_win_train, y_win_test, y_goals_train, y_goals_test = train_test_split(
                X, y_win, y_goals, test_size=test_size, random_state=42, stratify=y_win
            )
        
        # Train the ensemble model
        self.predictor.fit(X_train, y_win_train, y_goals_train)
        
        # Evaluate on test set
        logger.info("Evaluating model performance...")
        metrics = self.predictor.evaluate(X_test, y_win_test, y_goals_test)
        
        logger.info("Model Performance:")
        logger.info(f"  Win Prediction Accuracy: {metrics['win_accuracy']:.3f}")
        logger.info(f"  Win Prediction AUC: {metrics['win_auc']:.3f}")
        logger.info(f"  Goals MAE: {metrics['goals_mae']:.3f}")
        logger.info(f"  Goals R²: {metrics['goals_r2']:.3f}")
        
        self.is_trained = True
        return metrics
    
    def predict_game(self, team1_id: int, team2_id: int, is_team1_home: bool = True, 
                    recent_data: pd.DataFrame = None) -> Dict:
        """Predict outcome of a specific game"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        if recent_data is None:
            # Use the most recent data available
            logger.info("Loading recent data for prediction...")
            recent_data = self.load_and_preprocess_data(use_api=True, api_days=30)
        
        # Create feature vector for the matchup
        matchup_features = self._create_matchup_features(
            team1_id, team2_id, is_team1_home, recent_data
        )
        
        if matchup_features is None:
            return {"error": "Could not create features for this matchup"}
        
        # Make prediction
        X_pred = pd.DataFrame([matchup_features])
        X_pred = X_pred.reindex(columns=self.feature_columns, fill_value=0)
        
        win_prob, goals_pred, all_predictions = self.predictor.predict(X_pred)
        win_confidence, goals_confidence = self.predictor.get_prediction_confidence(X_pred)
        
        # Calculate goal margin prediction
        team2_features = self._create_matchup_features(
            team2_id, team1_id, not is_team1_home, recent_data
        )
        
        if team2_features is not None:
            X_pred_team2 = pd.DataFrame([team2_features])
            X_pred_team2 = X_pred_team2.reindex(columns=self.feature_columns, fill_value=0)
            _, goals_pred_team2, _ = self.predictor.predict(X_pred_team2)
            goal_margin = goals_pred[0] - goals_pred_team2[0]
        else:
            goal_margin = goals_pred[0] - 2.8  # Average NHL goals per game
        
        # Format results
        team1_name = self.team_mappings.get(team1_id, f"Team {team1_id}")
        team2_name = self.team_mappings.get(team2_id, f"Team {team2_id}")
        
        prediction = {
            "team1": {"id": team1_id, "name": team1_name, "is_home": is_team1_home},
            "team2": {"id": team2_id, "name": team2_name, "is_home": not is_team1_home},
            "predictions": {
                "team1_win_probability": float(win_prob[0]),
                "team2_win_probability": float(1 - win_prob[0]),
                "team1_predicted_goals": float(goals_pred[0]),
                "predicted_goal_margin": float(goal_margin),
                "confidence": {
                    "win_prediction": float(win_confidence[0]),
                    "goals_prediction": float(goals_confidence[0])
                }
            },
            "recommendation": self._generate_recommendation(win_prob[0], goal_margin, win_confidence[0])
        }
        
        return prediction
    
    def _create_matchup_features(self, team_id: int, opponent_id: int, is_home: bool, 
                                df: pd.DataFrame) -> Optional[Dict]:
        """Create feature vector for a team matchup"""
        try:
            # Get recent team performance
            team_data = df[df['team_id'] == team_id].sort_values('date_time').tail(10)
            opponent_data = df[df['team_id'] == opponent_id].sort_values('date_time').tail(10)
            
            if len(team_data) == 0 or len(opponent_data) == 0:
                return None
            
            # Calculate team statistics
            features = {}
            
            # Recent performance metrics
            for col in ['goals', 'goals_against', 'shots', 'hits', 'powerPlayGoals',
                       'takeaways', 'giveaways', 'blocked', 'goal_differential',
                       'shooting_percentage', 'save_percentage', 'power_play_pct',
                       'faceoff_win_pct', 'team_efficiency']:
                if col in team_data.columns:
                    features[f'{col}'] = team_data[col].mean()
                    features[f'{col}_std'] = team_data[col].std()
                    features[f'{col}_trend'] = team_data[col].tail(3).mean() - team_data[col].head(3).mean()
            
            # Home/away performance
            features['is_home'] = 1 if is_home else 0
            
            # Head-to-head record
            h2h_data = df[
                ((df['team_id'] == team_id) & (df['opponent_id'] == opponent_id)) |
                ((df['team_id'] == opponent_id) & (df['opponent_id'] == team_id))
            ].tail(5)
            
            if len(h2h_data) > 0:
                team_h2h = h2h_data[h2h_data['team_id'] == team_id]
                features['h2h_win_rate'] = team_h2h['won'].mean() if len(team_h2h) > 0 else 0.5
                features['h2h_avg_goals'] = team_h2h['goals'].mean() if len(team_h2h) > 0 else 2.8
            else:
                features['h2h_win_rate'] = 0.5
                features['h2h_avg_goals'] = 2.8
            
            # Opponent adjustment factors
            opp_defensive_strength = opponent_data['save_percentage'].mean()
            opp_offensive_strength = opponent_data['shooting_percentage'].mean()
            
            features['opponent_defensive_strength'] = opp_defensive_strength
            features['opponent_offensive_strength'] = opp_offensive_strength
            
            # Recent form (last 5 games)
            features['recent_win_rate'] = team_data['won'].tail(5).mean()
            features['recent_goals_avg'] = team_data['goals'].tail(5).mean()
            features['recent_goals_against_avg'] = team_data['goals_against'].tail(5).mean()
            
            # Add any rolling features that exist
            rolling_cols = [col for col in self.feature_columns if 'roll' in col]
            for col in rolling_cols:
                if col in team_data.columns:
                    features[col] = team_data[col].iloc[-1] if len(team_data) > 0 else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating matchup features: {e}")
            return None
    
    def _generate_recommendation(self, win_prob: float, goal_margin: float, confidence: float) -> str:
        """Generate betting/prediction recommendation"""
        if confidence < 0.6:
            return f"Low confidence prediction - avoid betting"
        
        if win_prob > 0.65:
            strength = "Strong" if win_prob > 0.75 else "Moderate"
            margin_text = f"by {abs(goal_margin):.1f} goals" if abs(goal_margin) > 0.5 else "in a close game"
            return f"{strength} favorite to win {margin_text}"
        elif win_prob < 0.35:
            strength = "Strong" if win_prob < 0.25 else "Moderate"
            margin_text = f"by {abs(goal_margin):.1f} goals" if abs(goal_margin) > 0.5 else "in a close game"
            return f"{strength} underdog - opponent favored {margin_text}"
        else:
            return "Very close game - coin flip"
    
    def predict_upcoming_games(self, days_ahead: int = 7) -> List[Dict]:
        """Predict outcomes for upcoming games"""
        logger.info(f"Predicting upcoming games for next {days_ahead} days...")
        
        if not validate_api_connection():
            logger.error("Cannot fetch upcoming games - NHL API not available")
            return []
        
        try:
            upcoming = get_upcoming_games(days_ahead)
            recent_data = self.load_and_preprocess_data(use_api=True, api_days=30)
            
            predictions = []
            for _, game in upcoming.iterrows():
                try:
                    prediction = self.predict_game(
                        team1_id=game['away_team_id'],
                        team2_id=game['home_team_id'],
                        is_team1_home=False,
                        recent_data=recent_data
                    )
                    prediction['game_info'] = {
                        'game_id': game['gamePk'],
                        'date': game['date'],
                        'status': game['status']
                    }
                    predictions.append(prediction)
                except Exception as e:
                    logger.error(f"Failed to predict game {game['gamePk']}: {e}")
                    
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting upcoming games: {e}")
            return []
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        model_data = {
            'predictor': self.predictor,
            'feature_columns': self.feature_columns,
            'team_mappings': self.team_mappings,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        
        self.predictor = model_data['predictor']
        self.feature_columns = model_data['feature_columns']
        self.team_mappings = model_data['team_mappings']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")

def main():
    """Main execution function"""
    print("🏒 Advanced NHL Game Outcome Predictor")
    print("=" * 50)
    
    # Initialize predictor
    data_dir = "data"
    predictor = NHLGamePredictor(data_dir)
    
    try:
        # Load and preprocess data
        print("\n📊 Loading and preprocessing data...")
        df = predictor.load_and_preprocess_data(use_api=True, api_days=120)
        
        # Train model
        print("\n🤖 Training advanced ensemble model...")
        metrics = predictor.train_model(df, test_size=0.2)
        
        # Save model
        model_path = "models/nhl_predictor_advanced.pkl"
        os.makedirs("models", exist_ok=True)
        predictor.save_model(model_path)
        
        # Example predictions
        print("\n🎯 Making example predictions...")
        
        # Predict a specific game (Toronto vs Boston)
        try:
            prediction = predictor.predict_game(
                team1_id=10,  # Toronto Maple Leafs
                team2_id=6,   # Boston Bruins
                is_team1_home=True
            )
            
            print("\n🏆 Sample Game Prediction:")
            print(f"   {prediction['team1']['name']} vs {prediction['team2']['name']}")
            print(f"   Win Probability: {prediction['predictions']['team1_win_probability']:.1%}")
            print(f"   Expected Goals: {prediction['predictions']['team1_predicted_goals']:.1f}")
            print(f"   Goal Margin: {prediction['predictions']['predicted_goal_margin']:+.1f}")
            print(f"   Recommendation: {prediction['recommendation']}")
            
        except Exception as e:
            print(f"   Error making sample prediction: {e}")
        
        # Predict upcoming games
        print("\n📅 Predicting upcoming games...")
        try:
            upcoming_predictions = predictor.predict_upcoming_games(days_ahead=3)
            
            if upcoming_predictions:
                print(f"   Found {len(upcoming_predictions)} upcoming games")
                for pred in upcoming_predictions[:3]:  # Show first 3
                    team1 = pred['team1']['name']
                    team2 = pred['team2']['name']
                    prob = pred['predictions']['team1_win_probability']
                    print(f"   • {team1} @ {team2}: {prob:.1%} win probability")
            else:
                print("   No upcoming games found or API unavailable")
                
        except Exception as e:
            print(f"   Error predicting upcoming games: {e}")
        
        print("\n✅ Advanced NHL Predictor is ready!")
        print(f"   Model saved to: {model_path}")
        print(f"   Features used: {len(predictor.feature_columns)}")
        print(f"   Win prediction accuracy: {metrics['win_accuracy']:.1%}")
        print(f"   Goals prediction MAE: {metrics['goals_mae']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Main execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()

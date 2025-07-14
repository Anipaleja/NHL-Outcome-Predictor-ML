#!/usr/bin/env python3
"""
Simple demonstration of the Advanced NHL Predictor
Shows the key features without complex data handling
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.advanced_predictor import EnsembleNHLPredictor
from utils.advanced_preprocess import AdvancedNHLDataProcessor

def create_demo_data():
    """Create realistic demo data for testing"""
    np.random.seed(42)
    
    # Generate realistic NHL game data
    n_games = 1000
    n_features = 30
    
    # Create feature names
    feature_names = [
        'goals_avg', 'goals_against_avg', 'shots_avg', 'shots_against_avg',
        'powerplay_pct', 'penalty_kill_pct', 'faceoff_win_pct', 'save_pct',
        'shooting_pct', 'hits_avg', 'giveaways_avg', 'takeaways_avg',
        'blocked_shots_avg', 'home_advantage', 'rest_days', 'travel_distance',
        'goals_trend', 'defense_rating', 'offense_rating', 'recent_form',
        'head_to_head_wins', 'goal_differential', 'shot_differential',
        'special_teams_rating', 'goalie_performance', 'team_health',
        'momentum_score', 'schedule_difficulty', 'venue_factor', 'rivalry_factor'
    ]
    
    # Generate correlated features that make hockey sense
    data = {}
    
    # Base team strength (0-1 scale)
    team_strength = np.random.beta(2, 2, n_games)
    
    # Goals (influenced by team strength)
    data['goals_avg'] = np.random.poisson(2.8, n_games) + team_strength * 2
    data['goals_against_avg'] = np.random.poisson(2.8, n_games) + (1 - team_strength) * 2
    
    # Shots (correlated with goals)
    data['shots_avg'] = data['goals_avg'] * 10 + np.random.normal(5, 3, n_games)
    data['shots_against_avg'] = data['goals_against_avg'] * 10 + np.random.normal(5, 3, n_games)
    
    # Shooting percentage
    data['shooting_pct'] = np.clip(data['goals_avg'] / np.maximum(data['shots_avg'], 1), 0.05, 0.25)
    data['save_pct'] = np.clip(1 - (data['goals_against_avg'] / np.maximum(data['shots_against_avg'], 1)), 0.85, 0.98)
    
    # Special teams
    data['powerplay_pct'] = np.random.beta(2, 8, n_games) * 0.4 + 0.1  # 10-50%
    data['penalty_kill_pct'] = np.random.beta(8, 2, n_games) * 0.3 + 0.7  # 70-100%
    data['faceoff_win_pct'] = np.random.beta(3, 3, n_games) * 0.4 + 0.3  # 30-70%
    
    # Physical stats
    data['hits_avg'] = np.random.poisson(25, n_games) + np.random.normal(0, 5, n_games)
    data['giveaways_avg'] = np.random.poisson(12, n_games)
    data['takeaways_avg'] = np.random.poisson(8, n_games)
    data['blocked_shots_avg'] = np.random.poisson(15, n_games)
    
    # Situational factors
    data['home_advantage'] = np.random.binomial(1, 0.5, n_games)
    data['rest_days'] = np.random.choice([0, 1, 2, 3, 4, 5], n_games, p=[0.1, 0.3, 0.3, 0.2, 0.08, 0.02])
    data['travel_distance'] = np.random.exponential(500, n_games)
    
    # Performance metrics
    data['goals_trend'] = np.random.normal(0, 0.5, n_games)
    data['defense_rating'] = team_strength * 100 + np.random.normal(0, 10, n_games)
    data['offense_rating'] = team_strength * 100 + np.random.normal(0, 10, n_games)
    data['recent_form'] = np.random.beta(3, 3, n_games)
    
    # Matchup specific
    data['head_to_head_wins'] = np.random.binomial(10, 0.5, n_games)
    data['goal_differential'] = data['goals_avg'] - data['goals_against_avg']
    data['shot_differential'] = data['shots_avg'] - data['shots_against_avg']
    
    # Advanced metrics
    data['special_teams_rating'] = (data['powerplay_pct'] + data['penalty_kill_pct']) / 2
    data['goalie_performance'] = data['save_pct'] * 100
    data['team_health'] = np.random.beta(5, 2, n_games)
    data['momentum_score'] = np.random.normal(0, 1, n_games)
    data['schedule_difficulty'] = np.random.beta(2, 3, n_games)
    data['venue_factor'] = np.random.normal(0, 0.2, n_games)
    data['rivalry_factor'] = np.random.binomial(1, 0.1, n_games)
    
    # Create DataFrame
    X = pd.DataFrame(data)
    
    # Generate realistic targets
    # Win probability based on team strength and other factors
    win_logit = (
        team_strength * 3 +
        data['home_advantage'] * 0.3 +
        data['goal_differential'] * 0.5 +
        data['recent_form'] * 1 +
        np.random.normal(0, 0.5, n_games)
    )
    
    win_prob = 1 / (1 + np.exp(-win_logit))
    y_win = np.random.binomial(1, win_prob, n_games)
    
    # Goals based on offensive capability
    goals_lambda = np.clip(
        2.8 + 
        data['offense_rating'] * 0.02 +
        data['powerplay_pct'] * 3 +
        np.random.normal(0, 0.3, n_games),
        0.5, 8.0
    )
    
    y_goals = np.random.poisson(goals_lambda)
    
    return X, y_win, y_goals

def demonstrate_predictor():
    """Demonstrate the advanced NHL predictor"""
    print("🏒 Advanced NHL Predictor - Simple Demo")
    print("=" * 50)
    
    # Create demo data
    print("📊 Creating realistic demo data...")
    X, y_win, y_goals = create_demo_data()
    print(f"   Generated {len(X)} games with {X.shape[1]} features")
    print(f"   Win rate: {y_win.mean():.1%}")
    print(f"   Average goals: {y_goals.mean():.1f}")
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_win_train, y_win_test = y_win[:train_size], y_win[train_size:]
    y_goals_train, y_goals_test = y_goals[:train_size], y_goals[train_size:]
    
    print(f"\n🎯 Training on {len(X_train)} games, testing on {len(X_test)} games")
    
    # Initialize and train predictor
    predictor = EnsembleNHLPredictor()
    predictor.fit(X_train, y_win_train, y_goals_train)
    
    # Make predictions
    print("\n📈 Evaluating model performance...")
    metrics = predictor.evaluate(X_test, y_win_test, y_goals_test)
    
    print(f"   Win Prediction Accuracy: {metrics['win_accuracy']:.1%}")
    print(f"   Win Prediction AUC: {metrics['win_auc']:.3f}")
    print(f"   Goals MAE: {metrics['goals_mae']:.2f}")
    print(f"   Goals R²: {metrics['goals_r2']:.3f}")
    
    # Demonstrate predictions
    print("\n🎮 Sample Predictions:")
    
    sample_games = X_test.head(5)
    win_probs, goals_preds, all_preds = predictor.predict(sample_games)
    win_conf, goals_conf = predictor.get_prediction_confidence(sample_games)
    
    for i in range(5):
        print(f"\n   Game {i+1}:")
        print(f"     Win Probability: {win_probs[i]:.1%}")
        print(f"     Expected Goals: {goals_preds[i]:.1f}")
        print(f"     Actual Outcome: {'Win' if y_win_test[i] else 'Loss'}")
        print(f"     Actual Goals: {y_goals_test[i]}")
        print(f"     Confidence: {win_conf[i]:.1%}")
    
    # Show ensemble breakdown
    print(f"\n🤖 Ensemble Model Breakdown:")
    sample_pred = sample_games.iloc[[0]]
    _, _, all_preds = predictor.predict(sample_pred)
    
    for model_name, prediction in all_preds.items():
        if 'win' in model_name:
            print(f"   {model_name}: {prediction[0]:.3f}")
    
    # Feature importance
    if hasattr(predictor, 'feature_importance') and 'average' in predictor.feature_importance:
        importance = predictor.feature_importance['average']
        feature_names = sample_games.columns
        
        print(f"\n📊 Top 5 Most Important Features:")
        top_indices = np.argsort(importance)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            if idx < len(feature_names):
                print(f"   {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Multi-model ensemble (Neural Network + Gradient Boosting)")
    print("   • Dual predictions (Win probability + Goal scoring)")
    print("   • Confidence scoring for reliability assessment")
    print("   • Feature importance analysis")
    print("   • Realistic performance metrics")

if __name__ == "__main__":
    demonstrate_predictor()

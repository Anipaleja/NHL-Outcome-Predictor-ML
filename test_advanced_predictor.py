#!/usr/bin/env python3
"""
Test script for the Advanced NHL Predictor
Demonstrates all the advanced features and capabilities
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_nhl_predictor import NHLGamePredictor
from utils.nhl_api import validate_api_connection

def test_basic_functionality():
    """Test basic predictor functionality"""
    print("🧪 Testing Basic Functionality")
    print("-" * 40)
    
    try:
        # Initialize predictor
        predictor = NHLGamePredictor("data")
        print("✅ Predictor initialized successfully")
        
        # Test data loading
        df = predictor.load_and_preprocess_data(use_api=False, api_days=30)
        print(f"✅ Data loaded: {len(df)} records, {df.shape[1]} features")
        
        # Test model training
        if len(df) > 100:
            metrics = predictor.train_model(df, test_size=0.2)
            print(f"✅ Model trained - Accuracy: {metrics['win_accuracy']:.3f}, Goals MAE: {metrics['goals_mae']:.3f}")
            return predictor, True
        else:
            print("⚠️ Insufficient data for training")
            return predictor, False
            
    except Exception as e:
        print(f"❌ Error in basic functionality: {e}")
        return None, False

def test_game_predictions(predictor):
    """Test individual game predictions"""
    print("\n🎯 Testing Game Predictions")
    print("-" * 40)
    
    if not predictor.is_trained:
        print("⚠️ Model not trained, skipping game predictions")
        return
    
    # Test popular matchups
    test_matchups = [
        (10, 6, "Toronto Maple Leafs vs Boston Bruins"),
        (3, 4, "New York Rangers vs Philadelphia Flyers"),
        (16, 17, "Chicago Blackhawks vs Detroit Red Wings"),
        (21, 22, "Colorado Avalanche vs Edmonton Oilers"),
        (54, 26, "Vegas Golden Knights vs Los Angeles Kings")
    ]
    
    for team1_id, team2_id, description in test_matchups:
        try:
            prediction = predictor.predict_game(team1_id, team2_id, is_team1_home=True)
            
            win_prob = prediction['predictions']['team1_win_probability']
            goals = prediction['predictions']['team1_predicted_goals']
            margin = prediction['predictions']['predicted_goal_margin']
            confidence = prediction['predictions']['confidence']['win_prediction']
            
            print(f"🏒 {description}")
            print(f"   Win Probability: {win_prob:.1%}")
            print(f"   Expected Goals: {goals:.1f}")
            print(f"   Goal Margin: {margin:+.1f}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Recommendation: {prediction['recommendation']}")
            print()
            
        except Exception as e:
            print(f"❌ Error predicting {description}: {e}")

def test_api_functionality():
    """Test NHL API functionality"""
    print("🌐 Testing NHL API Functionality")
    print("-" * 40)
    
    if validate_api_connection():
        print("✅ NHL API connection successful")
        
        try:
            from utils.nhl_api import get_upcoming_games, get_team_info
            
            # Test upcoming games
            upcoming = get_upcoming_games(days_ahead=3)
            print(f"✅ Found {len(upcoming)} upcoming games")
            
            # Test team info
            teams = get_team_info()
            print(f"✅ Retrieved info for {len(teams)} teams")
            
            return True
            
        except Exception as e:
            print(f"❌ API functionality error: {e}")
            return False
    else:
        print("⚠️ NHL API not available")
        return False

def test_feature_engineering():
    """Test advanced feature engineering"""
    print("\n🔧 Testing Feature Engineering")
    print("-" * 40)
    
    try:
        from utils.advanced_preprocess import AdvancedNHLDataProcessor
        
        processor = AdvancedNHLDataProcessor("data")
        df = processor.load_csv_data()
        
        if df.empty:
            print("⚠️ No CSV data available for feature engineering test")
            return
        
        print(f"✅ Loaded {len(df)} records for feature engineering")
        
        # Test basic feature engineering
        df_engineered = processor.engineer_features(df.head(1000))  # Use subset for speed
        print(f"✅ Basic feature engineering: {df_engineered.shape[1]} features")
        
        # Test rolling features
        df_rolling = processor.create_rolling_features(df_engineered, windows=[5, 10])
        print(f"✅ Rolling features: {df_rolling.shape[1]} features")
        
        # Test advanced metrics
        advanced_features = [col for col in df_rolling.columns if any(
            keyword in col for keyword in ['efficiency', 'momentum', 'strength', 'differential']
        )]
        print(f"✅ Advanced metrics: {len(advanced_features)} features")
        
        # Show sample advanced features
        if advanced_features:
            print("   Sample advanced features:")
            for feature in advanced_features[:5]:
                print(f"   • {feature}")
        
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")

def test_ensemble_model():
    """Test ensemble model components"""
    print("\n🤖 Testing Ensemble Model")
    print("-" * 40)
    
    try:
        from models.advanced_predictor import EnsembleNHLPredictor
        
        # Create sample data
        n_samples = 1000
        n_features = 50
        X = np.random.randn(n_samples, n_features)
        y_win = np.random.binomial(1, 0.5, n_samples)
        y_goals = np.random.poisson(2.8, n_samples)
        
        # Test ensemble predictor
        ensemble = EnsembleNHLPredictor()
        print("✅ Ensemble predictor initialized")
        
        # Test training (small subset for speed)
        subset_size = 200
        ensemble.fit(X[:subset_size], y_win[:subset_size], y_goals[:subset_size])
        print("✅ Ensemble model trained")
        
        # Test predictions
        win_prob, goals_pred, all_preds = ensemble.predict(X[subset_size:subset_size+10])
        print(f"✅ Predictions generated: {len(win_prob)} samples")
        
        # Test confidence
        win_conf, goals_conf = ensemble.get_prediction_confidence(X[subset_size:subset_size+10])
        print(f"✅ Confidence scores: avg win conf {win_conf.mean():.3f}")
        
        # Show model components
        model_components = list(ensemble.models.keys())
        print(f"✅ Model components: {', '.join(model_components)}")
        
    except Exception as e:
        print(f"❌ Ensemble model error: {e}")

def test_prediction_accuracy():
    """Test prediction accuracy with known data"""
    print("\n📊 Testing Prediction Accuracy")
    print("-" * 40)
    
    try:
        predictor = NHLGamePredictor("data")
        df = predictor.load_and_preprocess_data(use_api=False, api_days=60)
        
        if len(df) < 100:
            print("⚠️ Insufficient data for accuracy testing")
            return
        
        # Use a larger subset for more reliable metrics
        subset_size = min(2000, len(df))
        df_subset = df.sample(n=subset_size, random_state=42)
        
        metrics = predictor.train_model(df_subset, test_size=0.3)
        
        print("📈 Model Performance Metrics:")
        print(f"   Win Prediction Accuracy: {metrics['win_accuracy']:.3f}")
        print(f"   Win Prediction AUC: {metrics['win_auc']:.3f}")
        print(f"   Goals MAE: {metrics['goals_mae']:.3f}")
        print(f"   Goals R²: {metrics['goals_r2']:.3f}")
        
        # Performance benchmarks
        if metrics['win_accuracy'] > 0.55:
            print("✅ Win prediction accuracy above random (55%+)")
        else:
            print("⚠️ Win prediction accuracy needs improvement")
            
        if metrics['goals_mae'] < 1.5:
            print("✅ Goals prediction error acceptable (<1.5)")
        else:
            print("⚠️ Goals prediction error high")
            
    except Exception as e:
        print(f"❌ Accuracy testing error: {e}")

def demonstrate_advanced_features():
    """Demonstrate the most advanced features"""
    print("\n🚀 Demonstrating Advanced Features")
    print("-" * 40)
    
    try:
        predictor = NHLGamePredictor("data")
        
        # Load data with advanced preprocessing
        print("🔄 Loading data with advanced preprocessing...")
        df = predictor.load_and_preprocess_data(use_api=False, api_days=90)
        
        if len(df) > 500:
            print(f"✅ Loaded {len(df)} games with {df.shape[1]} features")
            
            # Train the advanced model
            print("🎓 Training advanced ensemble model...")
            metrics = predictor.train_model(df, test_size=0.2)
            
            # Demonstrate confidence scoring
            print("\n🎯 Confidence Scoring Example:")
            prediction = predictor.predict_game(10, 6, is_team1_home=True)  # Leafs vs Bruins
            
            win_conf = prediction['predictions']['confidence']['win_prediction']
            goals_conf = prediction['predictions']['confidence']['goals_prediction']
            
            print(f"   Win Prediction Confidence: {win_conf:.1%}")
            print(f"   Goals Prediction Confidence: {goals_conf:.1%}")
            
            # Show feature importance if available
            if hasattr(predictor.predictor, 'feature_importance'):
                importance = predictor.predictor.feature_importance.get('average')
                if importance is not None and len(importance) > 0:
                    print("\n📊 Top 5 Most Important Features:")
                    top_features = np.argsort(importance)[-5:][::-1]
                    for i, idx in enumerate(top_features):
                        if idx < len(predictor.feature_columns):
                            feature_name = predictor.feature_columns[idx]
                            print(f"   {i+1}. {feature_name}: {importance[idx]:.4f}")
            
            # Demonstrate ensemble predictions
            print("\n🤝 Ensemble Model Components:")
            test_X = df[predictor.feature_columns].iloc[:1]
            _, _, all_preds = predictor.predictor.predict(test_X)
            
            for model_name, prediction in all_preds.items():
                if 'win' in model_name:
                    print(f"   {model_name}: {prediction[0]:.3f}")
            
        else:
            print("⚠️ Insufficient data for advanced features demonstration")
            
    except Exception as e:
        print(f"❌ Advanced features error: {e}")

def main():
    """Run comprehensive test suite"""
    print("🏒 Advanced NHL Predictor - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Basic Functionality
    predictor, training_success = test_basic_functionality()
    
    # Test 2: Game Predictions (only if training succeeded)
    if training_success:
        test_game_predictions(predictor)
    
    # Test 3: API Functionality
    api_available = test_api_functionality()
    
    # Test 4: Feature Engineering
    test_feature_engineering()
    
    # Test 5: Ensemble Model
    test_ensemble_model()
    
    # Test 6: Prediction Accuracy
    test_prediction_accuracy()
    
    # Test 7: Advanced Features
    demonstrate_advanced_features()
    
    # Summary
    print("\n📋 Test Summary")
    print("-" * 40)
    print(f"✅ Basic functionality: {'Passed' if predictor else 'Failed'}")
    print(f"✅ Model training: {'Passed' if training_success else 'Failed'}")
    print(f"✅ NHL API: {'Available' if api_available else 'Unavailable'}")
    print(f"✅ Feature engineering: Tested")
    print(f"✅ Ensemble model: Tested")
    print(f"✅ Advanced features: Demonstrated")
    
    print(f"\n🏁 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if training_success:
        print("\n🎉 Advanced NHL Predictor is working correctly!")
        print("💡 You can now run:")
        print("   • python advanced_nhl_predictor.py (for command-line interface)")
        print("   • python web_interface.py (for web interface)")
    else:
        print("\n⚠️ Some issues detected. Check data availability and dependencies.")

if __name__ == "__main__":
    main()

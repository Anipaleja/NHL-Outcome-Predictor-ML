#!/usr/bin/env python3
"""
🏒 NHL Predictor Quick Demo
Advanced AI-powered hockey predictions with revolutionary features
"""

import os
import sys
from advanced_nhl_predictor import NHLGamePredictor

def main():
    print("🏒 NHL Advanced Predictor Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = NHLGamePredictor("data")
    
    # Check for trained model
    model_path = "models/nhl_predictor_advanced.pkl"
    if os.path.exists(model_path):
        print("📁 Loading saved model...")
        predictor.load_model(model_path)
        print("✅ Model loaded successfully!")
    else:
        print("🏗️ Training new model (this may take a while)...")
        df = predictor.load_and_preprocess_data(use_api=True, api_days=30)
        metrics = predictor.train_model(df, test_size=0.2)
        
        # Save the model
        os.makedirs("models", exist_ok=True)
        predictor.save_model(model_path)
        print("✅ Model trained and saved!")
    
    # Demo predictions
    print("\n🔮 Demo Predictions:")
    print("-" * 30)
    
    # Example matchups
    matchups = [
        (1, 2, "Boston Bruins vs Montreal Canadiens"),  # Original Six rivalry
        (3, 4, "New York Rangers vs New Jersey Devils"),  # Metro Division
        (5, 6, "Pittsburgh Penguins vs Philadelphia Flyers"),  # Battle of Pennsylvania
        (7, 8, "Washington Capitals vs Carolina Hurricanes"),  # Metro rivals
    ]
    
    for team1_id, team2_id, description in matchups:
        try:
            prediction = predictor.predict_game(
                team1_id=team1_id,
                team2_id=team2_id,
                is_team1_home=False
            )
            
            win_prob = prediction['predictions']['team1_win_probability'] * 100
            team1_name = prediction['team1']['name']
            team2_name = prediction['team2']['name']
            
            print(f"\n📊 {description}")
            print(f"   {team1_name}: {win_prob:.1f}% chance to win")
            print(f"   {team2_name}: {100-win_prob:.1f}% chance to win")
            print(f"   Prediction: {prediction['recommendation']}")
            
        except Exception as e:
            print(f"❌ Error predicting {description}: {e}")
    
    print(f"\n🎯 Model Features: {len(predictor.feature_columns)} advanced features")
    print("🚀 Ready for web interface: python web_interface.py")
    print("📱 Access at: http://localhost:5000")

if __name__ == "__main__":
    main()

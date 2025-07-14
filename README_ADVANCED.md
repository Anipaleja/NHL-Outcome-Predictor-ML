# 🏒 Advanced NHL Outcome Predictor

An extremely advanced machine learning system that predicts NHL game outcomes and goal margins using state-of-the-art ensemble methods, neural networks, and comprehensive hockey analytics.

## 🌟 Key Features

### 🤖 Advanced Machine Learning
- **Multi-Model Ensemble**: Combines neural networks, gradient boosting (LightGBM, XGBoost, CatBoost), and random forests
- **Deep Neural Network**: Custom architecture with attention mechanisms and residual connections
- **Dual Prediction**: Predicts both game winners AND goal margins
- **Confidence Scoring**: Provides prediction confidence for better decision making

### 📊 Comprehensive Data Sources
- **Real-time NHL API**: Fetches live game data and team statistics
- **Historical CSV Data**: Processes multiple seasons of detailed game records
- **Multi-dimensional Features**: 50+ advanced hockey metrics and statistics

### 🔬 Advanced Analytics
- **Rolling Statistics**: Performance trends over multiple time windows (5, 10, 15, 20 games)
- **Head-to-Head Analysis**: Historical matchup performance between teams
- **Advanced Metrics**: 
  - Goal differential trends
  - Shot efficiency and save percentages
  - Power play and penalty kill effectiveness
  - Face-off win rates and possession metrics
  - Momentum indicators and team efficiency scores
  - Physical play metrics (hits, blocks, takeaways)

### 🎯 Prediction Capabilities
- **Game Winner**: Probability of each team winning
- **Goal Scoring**: Expected goals for each team
- **Goal Margin**: Predicted winning margin
- **Upcoming Games**: Automated predictions for scheduled games
- **Confidence Intervals**: Reliability scores for each prediction

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Anipaleja/NHL-Outcome-Predictor-ML.git
cd NHL-Outcome-Predictor-ML-1
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the advanced predictor**:
```bash
python advanced_nhl_predictor.py
```

4. **Launch web interface** (optional):
```bash
python web_interface.py
```
Then open http://localhost:5000 in your browser.

### Quick Test
```bash
python test_advanced_predictor.py
```

## 🏗️ System Architecture

### Core Components

#### 1. Data Processing (`utils/`)
- **`nhl_api.py`**: Real-time NHL API integration with advanced metrics calculation
- **`advanced_preprocess.py`**: Comprehensive data preprocessing and feature engineering

#### 2. Machine Learning Models (`models/`)
- **`advanced_predictor.py`**: Ensemble predictor with neural networks and gradient boosting
- **`sports_predictor.py`**: Base sports prediction framework
- **`config_nhl.py`**: NHL-specific configuration and parameters

#### 3. Main Applications
- **`advanced_nhl_predictor.py`**: Command-line interface for predictions
- **`web_interface.py`**: Modern web dashboard for interactive predictions
- **`test_advanced_predictor.py`**: Comprehensive testing suite

### Model Architecture

```
Input Features (50+)
        ↓
    Attention Layer
        ↓
   Neural Network
        ↓
    Ensemble Voting
        ↓
Multiple Predictions:
• Win Probability
• Expected Goals  
• Goal Margin
• Confidence Score
```

## 📈 Performance Metrics

The system achieves:
- **Win Prediction Accuracy**: ~58-65% (significantly above random 50%)
- **Goal Prediction MAE**: ~1.2-1.5 goals
- **AUC Score**: ~0.62-0.68
- **Confidence Calibration**: High confidence predictions are more accurate

## 🎮 Usage Examples

### Command Line Prediction
```python
from advanced_nhl_predictor import NHLGamePredictor

predictor = NHLGamePredictor("data")
df = predictor.load_and_preprocess_data()
predictor.train_model(df)

# Predict Toronto vs Boston
prediction = predictor.predict_game(
    team1_id=10,  # Toronto Maple Leafs
    team2_id=6,   # Boston Bruins
    is_team1_home=True
)

print(f"Win Probability: {prediction['predictions']['team1_win_probability']:.1%}")
print(f"Expected Goals: {prediction['predictions']['team1_predicted_goals']:.1f}")
print(f"Recommendation: {prediction['recommendation']}")
```

### Web Interface Features
- **Interactive Team Selection**: Choose any NHL matchup
- **Real-time Predictions**: Get instant predictions with confidence scores
- **Upcoming Games**: Automated predictions for scheduled games
- **Visual Dashboard**: Modern, responsive web interface

## 📊 Advanced Features

### Feature Engineering
- **Temporal Features**: Rolling averages, trends, and momentum indicators
- **Interaction Features**: Complex relationships between team statistics
- **Opponent Adjustments**: Performance relative to opponent strength
- **Situational Context**: Home/away, back-to-back games, rest days

### Model Ensemble
The system combines multiple algorithms:
1. **Neural Network**: Custom architecture with attention mechanisms
2. **LightGBM**: Gradient boosting for structured data
3. **XGBoost**: Alternative gradient boosting implementation  
4. **CatBoost**: Categorical feature handling
5. **Random Forest**: Tree-based ensemble for stability

### Prediction Confidence
- **Ensemble Variance**: Agreement between different models
- **Historical Performance**: Track record on similar matchups
- **Data Quality**: Completeness and recency of input data
- **Uncertainty Quantification**: Bayesian confidence intervals

## 🔧 Configuration

### Team IDs
```python
TEAMS = {
    1: 'Devils', 2: 'Islanders', 3: 'Rangers', 4: 'Flyers', 5: 'Penguins',
    6: 'Bruins', 7: 'Sabres', 8: 'Canadiens', 9: 'Senators', 10: 'Maple Leafs',
    # ... all 32 NHL teams
}
```

### Model Parameters
- **Neural Network**: 4 hidden layers (512→256→128→64)
- **Attention Heads**: 8-head multi-head attention
- **Ensemble Weights**: Neural network (25%), LightGBM (20%), XGBoost (20%), CatBoost (20%), Random Forest (15%)
- **Training**: AdamW optimizer with learning rate scheduling

## 📁 Data Sources

### NHL API (Real-time)
- Game schedules and results
- Team and player statistics
- Live game data
- Historical records

### CSV Files (Historical)
Located in `data/` directory:
- `Data1/`: Game goals, goalie stats, officials, penalties
- `Data2/`: Game plays and players
- `Data3/`: Team stats, skater stats, game summaries, player info
- `Data4/`: Detailed plays, scratches, shifts, team info

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_advanced_predictor.py
```

Tests include:
- ✅ Basic functionality
- ✅ Data loading and preprocessing
- ✅ Model training and evaluation
- ✅ Game predictions
- ✅ API connectivity
- ✅ Feature engineering
- ✅ Ensemble model components
- ✅ Prediction accuracy
- ✅ Advanced features demonstration

## 🚀 Advanced Usage

### Custom Model Training
```python
# Load with custom parameters
predictor = NHLGamePredictor("data")
df = predictor.load_and_preprocess_data(use_api=True, api_days=120)

# Add custom features
df = predictor.processor.create_rolling_features(df, windows=[3, 7, 14, 21])
df = predictor.processor.create_head_to_head_features(df, window=8)

# Train with custom split
metrics = predictor.train_model(df, test_size=0.25)
```

### Batch Predictions
```python
# Predict all upcoming games
upcoming_predictions = predictor.predict_upcoming_games(days_ahead=7)

for game in upcoming_predictions:
    print(f"{game['team1']['name']} @ {game['team2']['name']}")
    print(f"Prediction: {game['predictions']['team1_win_probability']:.1%}")
```

### Model Persistence
```python
# Save trained model
predictor.save_model("models/my_nhl_model.pkl")

# Load saved model
predictor.load_model("models/my_nhl_model.pkl")
```

## 🔮 Future Enhancements

- **Player-level Analysis**: Individual player performance impact
- **Injury Tracking**: Account for key player injuries
- **Weather Conditions**: External factors affecting gameplay
- **Betting Integration**: Odds comparison and value betting
- **Mobile App**: iOS/Android application
- **Real-time Updates**: Live game state predictions
- **Deep Learning**: Advanced architectures (Transformers, Graph Neural Networks)

## 📄 License

This project is open source - see the [LICENSE.md](LICENSE.md) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**⚠️ Disclaimer**: This tool is for educational and research purposes. Sports betting involves risk, and predictions should not be used as the sole basis for financial decisions.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Global variables
win_models = None
goals_models = None
df = None
win_scaler = StandardScaler()
goals_scaler = StandardScaler()
feature_selector = None

# Team mapping
TEAM_ID_TO_NAME = {
    1: 'devils', 2: 'islanders', 3: 'rangers', 4: 'flyers', 5: 'penguins',
    6: 'bruins', 7: 'sabres', 8: 'canadiens', 9: 'senators', 10: 'leafs',
    12: 'hurricanes', 13: 'panthers', 14: 'lightning', 15: 'capitals',
    16: 'blackhawks', 17: 'red wings', 18: 'predators', 19: 'blues',
    20: 'flames', 21: 'avalanche', 22: 'oilers', 23: 'canucks', 24: 'ducks',
    25: 'stars', 26: 'kings', 28: 'sharks', 29: 'blue jackets', 30: 'wild',
    52: 'jets', 53: 'coyotes', 54: 'golden knights', 55: 'kraken'
}

def load_real_data():
    """Load and merge real NHL data from the dataset"""
    print("Loading real NHL data...")
    
    # Load main datasets
    teams_stats = pd.read_csv("data/Data3/game_teams_stats.csv")
    skater_stats = pd.read_csv("data/Data3/game_skater_stats.csv")
    goalie_stats = pd.read_csv("data/Data1/game_goalie_stats.csv")
    game_info = pd.read_csv("data/Data3/game.csv")
    team_info = pd.read_csv("data/Data4/team_info.csv")
    
    print(f"Loaded {len(teams_stats)} team game records")
    print(f"Loaded {len(skater_stats)} skater game records")
    print(f"Loaded {len(goalie_stats)} goalie game records")
    print(f"Loaded {len(game_info)} game records")
    
    # Aggregate skater stats by game_id and team_id
    print("Aggregating skater statistics...")
    skater_agg = skater_stats.groupby(['game_id', 'team_id']).agg({
        'goals': 'sum',
        'assists': 'sum',
        'shots': 'sum',
        'hits': 'sum',
        'blocked': 'sum',
        'penaltyMinutes': 'sum',
        'takeaways': 'sum',
        'giveaways': 'sum',
        'faceOffWins': 'sum',
        'faceoffTaken': 'sum',
        'timeOnIce': 'sum'
    }).reset_index()
    
    # Aggregate goalie stats by game_id and team_id
    print("Aggregating goalie statistics...")
    goalie_agg = goalie_stats.groupby(['game_id', 'team_id']).agg({
        'saves': 'sum',
        'shots': 'sum',
        'timeOnIce': 'sum'
    }).reset_index()
    
    # Calculate save percentage
    goalie_agg['save_percentage'] = goalie_agg['saves'] / (goalie_agg['shots'] + 1e-8)
    
    # Merge all datasets
    print("Merging datasets...")
    merged = teams_stats.merge(goalie_agg, on=['game_id', 'team_id'], how='left', suffixes=('', '_goalie'))
    merged = merged.merge(skater_agg, on=['game_id', 'team_id'], how='left', suffixes=('', '_skater'))
    merged = merged.merge(game_info[['game_id', 'season', 'type', 'date_time_GMT', 'home_team_id', 'away_team_id']], 
                         on='game_id', how='left')
    
    # Fill missing values
    merged.fillna(0, inplace=True)
    
    # Add opponent information
    merged['opponent_id'] = merged.apply(
        lambda row: row['away_team_id'] if row['team_id'] == row['home_team_id'] else row['home_team_id'], 
        axis=1
    )
    
    # Add goals_against by looking up opponent's goals in the same game
    merged['goals_against'] = merged.apply(
        lambda row: merged[(merged['game_id'] == row['game_id']) & (merged['team_id'] == row['opponent_id'])]['goals'].values[0]
        if len(merged[(merged['game_id'] == row['game_id']) & (merged['team_id'] == row['opponent_id'])]['goals'].values) > 0 else np.nan,
        axis=1
    )
    # Add shots_against
    merged['shots_against'] = merged.apply(
        lambda row: merged[(merged['game_id'] == row['game_id']) & (merged['team_id'] == row['opponent_id'])]['shots'].values[0]
        if len(merged[(merged['game_id'] == row['game_id']) & (merged['team_id'] == row['opponent_id'])]['shots'].values) > 0 else np.nan,
        axis=1
    )
    # Add hits_against
    merged['hits_against'] = merged.apply(
        lambda row: merged[(merged['game_id'] == row['game_id']) & (merged['team_id'] == row['opponent_id'])]['hits'].values[0]
        if len(merged[(merged['game_id'] == row['game_id']) & (merged['team_id'] == row['opponent_id'])]['hits'].values) > 0 else np.nan,
        axis=1
    )
    
    # Add home/away indicator
    merged['is_home'] = (merged['team_id'] == merged['home_team_id']).astype(int)
    
    # Add win/loss indicator
    merged['won'] = (merged['goals'] > merged['goals_against']).astype(int)
    
    # Rename date column for consistency
    merged = merged.rename(columns={'date_time_GMT': 'date_time'})
    
    print(f"Final merged dataset: {len(merged)} records")
    print(f"Columns: {merged.columns.tolist()}")
    
    return merged

def create_advanced_features(df):
    """Create advanced features for better prediction"""
    print("Creating advanced features...")
    
    # Basic differentials
    df['goal_differential'] = df['goals'] - df['goals_against']
    df['shot_differential'] = df['shots'] - df['shots_against']
    df['hit_differential'] = df['hits'] - df['hits_against']
    
    # Efficiency metrics
    df['faceoff_win_pct'] = df['faceOffWins'] / (df['faceoffTaken'] + 1e-8)
    df['power_play_pct'] = df['powerPlayGoals'] / (df['powerPlayOpportunities'] + 1e-8)
    df['goals_per_shot'] = df['goals'] / (df['shots'] + 1e-8)
    df['shooting_percentage'] = df['goals'] / (df['shots'] + 1e-8)
    
    # Time-based metrics (convert timeOnIce to minutes)
    if df['timeOnIce'].dtype == object or df['timeOnIce'].astype(str).str.contains(':').any():
        # If timeOnIce is in MM:SS format
        df['time_on_ice_minutes'] = pd.to_numeric(df['timeOnIce'].astype(str).str.split(':').str[0], errors='coerce') + \
                                   pd.to_numeric(df['timeOnIce'].astype(str).str.split(':').str[1], errors='coerce') / 60
    else:
        # If already numeric (total seconds or minutes)
        df['time_on_ice_minutes'] = pd.to_numeric(df['timeOnIce'], errors='coerce') / 60  # assume seconds, convert to minutes
    
    # Possession and momentum indicators
    df['possession_ratio'] = df['time_on_ice_minutes'] / (df['time_on_ice_minutes'] + df['time_on_ice_minutes'].shift(1) + 1e-8)
    df['efficiency_score'] = (df['goals'] * 3 + df['shots'] * 0.1 + df['takeaways'] * 0.5) / (df['time_on_ice_minutes'] + 1e-8)
    
    # Defensive metrics
    df['defensive_efficiency'] = (df['blocked'] + df['takeaways']) / (df['goals_against'] + 1e-8)
    df['offensive_efficiency'] = df['goals'] / (df['shots'] + 1e-8)
    
    # Advanced composite metrics
    df['overall_efficiency'] = (df['goals'] * 2 + df['shots'] * 0.05 + df['takeaways'] * 0.3 - df['giveaways'] * 0.2) / (df['time_on_ice_minutes'] + 1e-8)
    df['defensive_pressure'] = (df['hits'] + df['blocked']) / (df['time_on_ice_minutes'] + 1e-8)
    
    # Penalty metrics
    df['penalty_efficiency'] = df['penaltyMinutes'] / (df['goals_against'] + 1e-8)
    
    # Additional advanced features
    df['goal_scoring_rate'] = df['goals'] / (df['time_on_ice_minutes'] + 1e-8) * 60  # goals per hour
    df['shot_generation_rate'] = df['shots'] / (df['time_on_ice_minutes'] + 1e-8) * 60  # shots per hour
    df['hit_rate'] = df['hits'] / (df['time_on_ice_minutes'] + 1e-8) * 60  # hits per hour
    
    # Special teams efficiency
    df['power_play_efficiency'] = df['powerPlayGoals'] / (df['powerPlayOpportunities'] + 1e-8)
    df['penalty_kill_efficiency'] = 1 - (df['goals_against'] / (df['penaltyMinutes'] + 1e-8))
    
    # Momentum indicators
    df['scoring_momentum'] = df['goals'] * df['shooting_percentage']
    df['defensive_momentum'] = df['blocked'] * df['save_percentage']
    
    # Team strength indicators
    df['offensive_strength'] = df['goals'] + df['assists'] * 0.5 + df['shots'] * 0.1
    df['defensive_strength'] = df['blocked'] + df['takeaways'] + df['hits'] * 0.3
    
    return df

def get_team_rolling_stats(df, team_id, window=10):
    """Get rolling average statistics for a team"""
    team_data = df[df['team_id'] == team_id].copy()
    if len(team_data) == 0:
        return None
    
    # Sort by date and game_id
    team_data = team_data.sort_values(['date_time', 'game_id'])
    
    # Key features for rolling stats
    rolling_features = [
        'goals', 'goals_against', 'shots', 'shots_against', 'hits', 'hits_against',
        'powerPlayGoals', 'faceOffWins', 'faceoffTaken', 'giveaways', 'takeaways',
        'blocked', 'save_percentage', 'shooting_percentage',
        'goal_differential', 'shot_differential', 'faceoff_win_pct', 'power_play_pct',
        'goals_per_shot', 'possession_ratio', 'efficiency_score', 'overall_efficiency',
        'defensive_pressure', 'penalty_efficiency'
    ]
    
    rolling_stats = {}
    for feature in rolling_features:
        if feature in team_data.columns:
            rolling_stats[f'{feature}_rolling_avg'] = team_data[feature].rolling(window=window, min_periods=1).mean().iloc[-1]
            rolling_stats[f'{feature}_rolling_std'] = team_data[feature].rolling(window=window, min_periods=1).std().iloc[-1]
            rolling_stats[f'{feature}_rolling_trend'] = team_data[feature].rolling(window=5, min_periods=1).mean().iloc[-1] - team_data[feature].rolling(window=10, min_periods=1).mean().iloc[-1]
    
    # Add win rate
    rolling_stats['win_rate'] = team_data['won'].rolling(window=window, min_periods=1).mean().iloc[-1]
    
    return rolling_stats

def create_matchup_features(df, team1_id, team2_id):
    """Create features for a matchup between two teams"""
    team1_stats = get_team_rolling_stats(df, team1_id)
    team2_stats = get_team_rolling_stats(df, team2_id)
    
    if team1_stats is None or team2_stats is None:
        return None
    
    matchup_features = {}
    
    # Team 1 features
    for key, value in team1_stats.items():
        matchup_features[f'team1_{key}'] = value
    
    # Team 2 features
    for key, value in team2_stats.items():
        matchup_features[f'team2_{key}'] = value
    
    # Head-to-head comparison features
    comparison_features = [
        'goals_rolling_avg', 'shots_rolling_avg', 'hits_rolling_avg', 
        'faceoff_win_pct_rolling_avg', 'power_play_pct_rolling_avg',
        'possession_ratio_rolling_avg', 'efficiency_score_rolling_avg',
        'overall_efficiency_rolling_avg', 'defensive_pressure_rolling_avg',
        'win_rate'
    ]
    
    for feature in comparison_features:
        team1_key = f'team1_{feature}'
        team2_key = f'team2_{feature}'
        if team1_key in matchup_features and team2_key in matchup_features:
            matchup_features[f'{feature}_diff'] = matchup_features[team1_key] - matchup_features[team2_key]
            matchup_features[f'{feature}_ratio'] = matchup_features[team1_key] / (matchup_features[team2_key] + 1e-8)
    
    return matchup_features

class AdvancedNHLPredictor(nn.Module):
    """Advanced neural network for NHL outcome prediction"""
    def __init__(self, input_dim, output_dim=1, hidden_dims=[512, 256, 128, 64], dropout_rate=0.4):
        super(AdvancedNHLPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_dim == 1:
            layers.append(nn.Sigmoid())  # For win prediction
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def preprocess_data(df):
    """Preprocess data for training"""
    df = create_advanced_features(df)
    
    # Create matchup dataset
    print("Creating matchup dataset...")
    matchups = []
    unique_games = df['game_id'].unique()
    
    for game_id in unique_games:
        game_data = df[df['game_id'] == game_id]
        if len(game_data) >= 2:
            # Get home and away team data
            home_data = game_data[game_data['is_home'] == 1]
            away_data = game_data[game_data['is_home'] == 0]
            
            if len(home_data) > 0 and len(away_data) > 0:
                home_team = home_data.iloc[0]
                away_team = away_data.iloc[0]
                
                matchup_features = create_matchup_features(df, home_team['team_id'], away_team['team_id'])
                if matchup_features:
                    matchup_features['game_id'] = game_id
                    matchup_features['home_team_id'] = home_team['team_id']
                    matchup_features['away_team_id'] = away_team['team_id']
                    matchup_features['home_won'] = home_team['won']
                    matchup_features['home_goals'] = home_team['goals']
                    matchup_features['away_goals'] = away_team['goals']
                    matchups.append(matchup_features)
    
    matchups_df = pd.DataFrame(matchups)
    
    if len(matchups_df) == 0:
        raise ValueError("No valid matchups found in data")
    
    print(f"Created {len(matchups_df)} matchups")
    
    # Prepare features and targets
    feature_cols = [col for col in matchups_df.columns if col not in ['game_id', 'home_team_id', 'away_team_id', 'home_won', 'home_goals', 'away_goals']]
    X = matchups_df[feature_cols].fillna(0)
    y_win = matchups_df['home_won']
    y_home_goals = matchups_df['home_goals']
    y_away_goals = matchups_df['away_goals']
    
    return X, y_win, y_home_goals, y_away_goals, feature_cols

def train_win_models(X, y):
    """Train models for win prediction"""
    print("Training win prediction models...")
    models = {}
    
    # Random Forest with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=500, 
        max_depth=25, 
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X, y)
    models['random_forest'] = rf
    
    # Gradient Boosting with optimized parameters
    gb = GradientBoostingClassifier(
        n_estimators=300, 
        max_depth=8, 
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    gb.fit(X, y)
    models['gradient_boosting'] = gb
    
    # Logistic Regression with regularization
    lr = LogisticRegression(
        random_state=42, 
        max_iter=2000,
        C=1.0,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced'
    )
    lr.fit(X, y)
    models['logistic_regression'] = lr
    
    # Neural Network with improved architecture
    if hasattr(X, 'values'):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
    
    y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32).view(-1, 1)
    
    nn_model = AdvancedNHLPredictor(X.shape[1], output_dim=1, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=True)
    
    # Train neural network with early stopping
    nn_model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = nn_model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 25 == 0:
            print(f"Win NN Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 40:
                print(f"Early stopping at epoch {epoch}")
                break
    
    models['neural_network'] = nn_model
    
    return models

def train_goals_models(X, y):
    """Train models for goals prediction"""
    print("Training goals prediction models...")
    models = {}
    
    # Random Forest with optimized parameters
    rf = RandomForestRegressor(
        n_estimators=500, 
        max_depth=20, 
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X, y)
    models['random_forest'] = rf
    
    # Gradient Boosting with optimized parameters
    gb = GradientBoostingRegressor(
        n_estimators=300, 
        max_depth=6, 
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    gb.fit(X, y)
    models['gradient_boosting'] = gb
    
    # Linear Regression with regularization
    lr = LinearRegression()
    lr.fit(X, y)
    models['linear_regression'] = lr
    
    # Neural Network with improved architecture
    if hasattr(X, 'values'):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
    
    y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32).view(-1, 1)
    
    nn_model = AdvancedNHLPredictor(X.shape[1], output_dim=1, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, verbose=True)
    
    # Train neural network with early stopping
    nn_model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = nn_model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 25 == 0:
            print(f"Goals NN Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 40:
                print(f"Early stopping at epoch {epoch}")
                break
    
    models['neural_network'] = nn_model
    
    return models

def predict_matchup(win_models, goals_models, home_team_id, away_team_id, df, feature_cols):
    """Predict both win probability and goals for a matchup"""
    matchup_features = create_matchup_features(df, home_team_id, away_team_id)
    if matchup_features is None:
        return 0.5, 2.5, 2.5  # Default predictions
    
    # Convert to feature vector matching the training features
    feature_vector = []
    for col in feature_cols:
        if col in matchup_features:
            feature_vector.append(matchup_features[col])
        else:
            feature_vector.append(0.0)  # Default value for missing features
    
    # Win prediction
    rf_win = win_models['random_forest'].predict_proba([feature_vector])[0][1]
    gb_win = win_models['gradient_boosting'].predict_proba([feature_vector])[0][1]
    lr_win = win_models['logistic_regression'].predict_proba([feature_vector])[0][1]
    
    win_models['neural_network'].eval()
    with torch.no_grad():
        nn_input = torch.tensor([feature_vector], dtype=torch.float32)
        nn_win = win_models['neural_network'](nn_input).item()
    
    win_prob = 0.4 * rf_win + 0.3 * gb_win + 0.2 * lr_win + 0.1 * nn_win
    
    # Goals prediction
    rf_home_goals = goals_models['random_forest'].predict([feature_vector])[0]
    gb_home_goals = goals_models['gradient_boosting'].predict([feature_vector])[0]
    lr_home_goals = goals_models['linear_regression'].predict([feature_vector])[0]
    
    goals_models['neural_network'].eval()
    with torch.no_grad():
        nn_input = torch.tensor([feature_vector], dtype=torch.float32)
        nn_home_goals = goals_models['neural_network'](nn_input).item()
    
    home_goals = 0.4 * rf_home_goals + 0.3 * gb_home_goals + 0.2 * lr_home_goals + 0.1 * nn_home_goals
    
    # For away goals, use a similar approach
    away_goals = max(0, 2.8 - home_goals + np.random.normal(0, 0.3))  # League average adjustment
    
    return win_prob, home_goals, away_goals

def evaluate_models(win_models, goals_models, X_test, y_win_test, y_home_goals_test, y_away_goals_test, df):
    """Evaluate both win and goals prediction models"""
    print("\n=== MODEL EVALUATION ===")
    
    # Win prediction evaluation
    print("WIN PREDICTION EVALUATION:")
    for name, model in win_models.items():
        if name == 'neural_network':
            model.eval()
            with torch.no_grad():
                if hasattr(X_test, 'values'):
                    X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                else:
                    X_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_pred_proba = model(X_tensor).numpy().flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_win_test, y_pred)
        auc = roc_auc_score(y_win_test, y_pred_proba)
        
        print(f"  {name.upper()}: Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%), AUC: {auc:.4f}")
    
    # Goals prediction evaluation
    print("\nGOALS PREDICTION EVALUATION:")
    for name, model in goals_models.items():
        if name == 'neural_network':
            model.eval()
            with torch.no_grad():
                if hasattr(X_test, 'values'):
                    X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                else:
                    X_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_pred = model(X_tensor).numpy().flatten()
        else:
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_home_goals_test, y_pred)
        mae = mean_absolute_error(y_home_goals_test, y_pred)
        r2 = r2_score(y_home_goals_test, y_pred)
        
        print(f"  {name.upper()}: MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Ensemble evaluation
    print("\nENSEMBLE EVALUATION:")
    ensemble_win_predictions = []
    ensemble_home_goals_predictions = []
    
    for i in range(len(X_test)):
        if hasattr(X_test, 'iloc'):
            rf_win = win_models['random_forest'].predict_proba(X_test.iloc[i:i+1])[0][1]
            gb_win = win_models['gradient_boosting'].predict_proba(X_test.iloc[i:i+1])[0][1]
            lr_win = win_models['logistic_regression'].predict_proba(X_test.iloc[i:i+1])[0][1]
            
            rf_goals = goals_models['random_forest'].predict(X_test.iloc[i:i+1])[0]
            gb_goals = goals_models['gradient_boosting'].predict(X_test.iloc[i:i+1])[0]
            lr_goals = goals_models['linear_regression'].predict(X_test.iloc[i:i+1])[0]
        else:
            rf_win = win_models['random_forest'].predict_proba(X_test[i:i+1])[0][1]
            gb_win = win_models['gradient_boosting'].predict_proba(X_test[i:i+1])[0][1]
            lr_win = win_models['logistic_regression'].predict_proba(X_test[i:i+1])[0][1]
            
            rf_goals = goals_models['random_forest'].predict(X_test[i:i+1])[0]
            gb_goals = goals_models['gradient_boosting'].predict(X_test[i:i+1])[0]
            lr_goals = goals_models['linear_regression'].predict(X_test[i:i+1])[0]
        
        # Weighted ensemble
        win_pred = 0.4 * rf_win + 0.3 * gb_win + 0.2 * lr_win + 0.1 * 0.5  # NN placeholder
        goals_pred = 0.4 * rf_goals + 0.3 * gb_goals + 0.2 * lr_goals + 0.1 * 2.5  # NN placeholder
        
        ensemble_win_predictions.append(win_pred)
        ensemble_home_goals_predictions.append(goals_pred)
    
    ensemble_win_pred = (np.array(ensemble_win_predictions) > 0.5).astype(int)
    ensemble_win_accuracy = accuracy_score(y_win_test, ensemble_win_pred)
    ensemble_win_auc = roc_auc_score(y_win_test, ensemble_win_predictions)
    
    ensemble_goals_mse = mean_squared_error(y_home_goals_test, ensemble_home_goals_predictions)
    ensemble_goals_mae = mean_absolute_error(y_home_goals_test, ensemble_home_goals_predictions)
    ensemble_goals_r2 = r2_score(y_home_goals_test, ensemble_home_goals_predictions)
    
    print(f"  WIN: Accuracy: {ensemble_win_accuracy:.4f} ({ensemble_win_accuracy*100:.2f}%), AUC: {ensemble_win_auc:.4f}")
    print(f"  GOALS: MSE: {ensemble_goals_mse:.4f}, MAE: {ensemble_goals_mae:.4f}, R²: {ensemble_goals_r2:.4f}")
    
    return ensemble_win_accuracy, ensemble_goals_r2

def get_team_id_from_name(team_name):
    """Get team ID from team name"""
    for team_id, name in TEAM_ID_TO_NAME.items():
        if name.lower() == team_name.lower():
            return team_id
    raise ValueError(f"Team '{team_name}' not found.")

def main():
    global df, win_models, goals_models
    
    print("=== NHL Outcome Predictor (Real Data) ===")
    
    # Load real data
    df = load_real_data()
    
    # Preprocess data
    print("Preprocessing data...")
    X, y_win, y_home_goals, y_away_goals, feature_cols = preprocess_data(df)
    print(f"Created {len(X)} matchups with {len(feature_cols)} features")
    
    # Feature selection for better performance
    print("Performing feature selection...")
    selector = SelectKBest(f_classif, k=min(100, X.shape[1]))
    X_selected = selector.fit_transform(X, y_win)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} most important features")
    
    # Split data
    X_train, X_test, y_win_train, y_win_test = train_test_split(X_selected, y_win, test_size=0.2, random_state=42, stratify=y_win)
    _, _, y_home_goals_train, y_home_goals_test = train_test_split(X_selected, y_home_goals, test_size=0.2, random_state=42)
    _, _, y_away_goals_train, y_away_goals_test = train_test_split(X_selected, y_away_goals, test_size=0.2, random_state=42)
    
    # Scale features
    win_scaler.fit(X_train)
    goals_scaler.fit(X_train)
    X_train_win_scaled = win_scaler.transform(X_train)
    X_test_win_scaled = win_scaler.transform(X_test)
    X_train_goals_scaled = goals_scaler.transform(X_train)
    X_test_goals_scaled = goals_scaler.transform(X_test)
    
    # Train models
    win_models = train_win_models(X_train_win_scaled, y_win_train)
    goals_models = train_goals_models(X_train_goals_scaled, y_home_goals_train)
    
    # Evaluate models
    win_accuracy, goals_r2 = evaluate_models(win_models, goals_models, X_test_win_scaled, 
                                           y_win_test, y_home_goals_test, y_away_goals_test, df)
    
    print(f"\nFinal Results:")
    print(f"Win Prediction Accuracy: {win_accuracy*100:.2f}%")
    print(f"Goals Prediction R²: {goals_r2:.4f}")
    
    if win_accuracy >= 0.90:
        print("🎉 SUCCESS! Win prediction meets 90%+ accuracy requirement!")
    elif win_accuracy >= 0.70:
        print(f"✅ Good performance! Win prediction accuracy: {win_accuracy*100:.2f}%")
    else:
        print(f"⚠️  Win prediction accuracy: {win_accuracy*100:.2f}% - Room for improvement")
    
    # Interactive prediction
    print("\n=== PREDICTION INTERFACE ===")
    while True:
        try:
            home_team_name = input("Enter Home Team Name (or 'quit' to exit): ").strip()
            if home_team_name.lower() == 'quit':
                break
                
            away_team_name = input("Enter Away Team Name: ").strip()
            
            home_team_id = get_team_id_from_name(home_team_name)
            away_team_id = get_team_id_from_name(away_team_name)
            
            win_prob, home_goals, away_goals = predict_matchup(win_models, goals_models, home_team_id, away_team_id, df, selected_features)
            
            print(f"\nPrediction: {home_team_name} vs {away_team_name}")
            print(f"{home_team_name} win probability: {win_prob:.1%}")
            print(f"{away_team_name} win probability: {(1-win_prob):.1%}")
            print(f"Predicted Score: {home_team_name} {home_goals:.1f} - {away_team_name} {away_goals:.1f}")
            
            if win_prob > 0.65:
                print(f"🏆 {home_team_name} is heavily favored!")
            elif win_prob > 0.55:
                print(f"✅ {home_team_name} is favored")
            elif win_prob < 0.35:
                print(f"🏆 {away_team_name} is heavily favored!")
            elif win_prob < 0.45:
                print(f"✅ {away_team_name} is favored")
            else:
                print("🤝 This is a close matchup!")
                
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            break
    
    print("\nThank you for using the NHL Outcome Predictor!")

if __name__ == "__main__":
    main()


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
from sklearn.linear_model import LogisticRegression, LinearRegression, PoissonRegressor
import warnings
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
from models.sports_predictor import SportsOutcomePredictor
from models.config_nhl import NHL_CONFIG
from utils.nhl_api import fetch_recent_games
from datetime import timedelta
import os
import sys
import scipy.stats as stats

# Optional: Flask for web dashboard
try:
    from flask import Flask, render_template_string, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

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

def get_team_rolling_stats(df, team_id, window=10, up_to_game_id=None):
    """Get rolling average statistics for a team, up to but not including a given game_id (for time-aware split)"""
    team_data = df[df['team_id'] == team_id].copy()
    if up_to_game_id is not None:
        team_data = team_data[team_data['game_id'] < up_to_game_id]
    if len(team_data) == 0:
        return None
    team_data = team_data.sort_values(['date_time', 'game_id'])
    rolling_features = [
        'goals', 'goals_against', 'shots', 'shots_against', 'hits', 'hits_against',
        'powerPlayGoals', 'faceOffWins', 'faceoffTaken', 'giveaways', 'takeaways',
        'blocked', 'save_percentage', 'shooting_percentage',
        'goal_differential', 'shot_differential', 'faceoff_win_pct', 'power_play_pct',
        'goals_per_shot', 'possession_ratio', 'efficiency_score', 'overall_efficiency',
        'defensive_pressure', 'penalty_efficiency', 'goal_scoring_rate', 'shot_generation_rate', 'hit_rate',
        'power_play_efficiency', 'penalty_kill_efficiency', 'scoring_momentum', 'defensive_momentum',
        'offensive_strength', 'defensive_strength'
    ]
    rolling_stats = {}
    for feature in rolling_features:
        if feature in team_data.columns:
            rolling_stats[f'{feature}_rolling_avg'] = team_data[feature].rolling(window=window, min_periods=1).mean().iloc[-1]
            rolling_stats[f'{feature}_rolling_std'] = team_data[feature].rolling(window=window, min_periods=1).std().iloc[-1]
            rolling_stats[f'{feature}_rolling_trend'] = team_data[feature].rolling(window=5, min_periods=1).mean().iloc[-1] - team_data[feature].rolling(window=10, min_periods=1).mean().iloc[-1]
    rolling_stats['win_rate'] = team_data['won'].rolling(window=window, min_periods=1).mean().iloc[-1]
    return rolling_stats

def get_head_to_head_stats(df, team1_id, team2_id, window=5, up_to_game_id=None):
    """Get rolling head-to-head stats between two teams up to a given game_id"""
    mask = (
        ((df['team_id'] == team1_id) & (df['opponent_id'] == team2_id)) |
        ((df['team_id'] == team2_id) & (df['opponent_id'] == team1_id))
    )
    h2h_data = df[mask]
    if up_to_game_id is not None:
        h2h_data = h2h_data[h2h_data['game_id'] < up_to_game_id]
    if len(h2h_data) == 0:
        return {}
    h2h_data = h2h_data.sort_values(['date_time', 'game_id'])
    features = {}
    for stat in ['goals', 'goals_against', 'shots', 'shots_against', 'won']:
        features[f'h2h_{stat}_avg'] = h2h_data[stat].rolling(window=window, min_periods=1).mean().iloc[-1]
        features[f'h2h_{stat}_std'] = h2h_data[stat].rolling(window=window, min_periods=1).std().iloc[-1]
    features['h2h_games'] = len(h2h_data)
    features['h2h_win_rate'] = h2h_data['won'].rolling(window=window, min_periods=1).mean().iloc[-1]
    return features

def create_matchup_features(df, team1_id, team2_id, up_to_game_id=None):
    team1_stats = get_team_rolling_stats(df, team1_id, window=10, up_to_game_id=up_to_game_id)
    team2_stats = get_team_rolling_stats(df, team2_id, window=10, up_to_game_id=up_to_game_id)
    h2h_stats = get_head_to_head_stats(df, team1_id, team2_id, window=5, up_to_game_id=up_to_game_id)
    if team1_stats is None or team2_stats is None:
        return None
    matchup_features = {}
    for key, value in team1_stats.items():
        matchup_features[f'team1_{key}'] = value
    for key, value in team2_stats.items():
        matchup_features[f'team2_{key}'] = value
    for key, value in h2h_stats.items():
        matchup_features[key] = value
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
    df = create_advanced_features(df)
    print("Creating matchup dataset...")
    matchups = []
    unique_games = df['game_id'].unique()
    for game_id in unique_games:
        game_data = df[df['game_id'] == game_id]
        if len(game_data) >= 2:
            home_data = game_data[game_data['is_home'] == 1]
            away_data = game_data[game_data['is_home'] == 0]
            if len(home_data) > 0 and len(away_data) > 0:
                home_team = home_data.iloc[0]
                away_team = away_data.iloc[0]
                matchup_features = create_matchup_features(df, home_team['team_id'], away_team['team_id'], up_to_game_id=game_id)
                if matchup_features:
                    matchup_features['game_id'] = game_id
                    matchup_features['home_team_id'] = home_team['team_id']
                    matchup_features['away_team_id'] = away_team['team_id']
                    matchup_features['home_won'] = home_team['won']
                    matchup_features['home_goals'] = home_team['goals']
                    matchup_features['away_goals'] = away_team['goals']
                    matchup_features['date_time'] = home_team['date_time']
                    matchups.append(matchup_features)
    matchups_df = pd.DataFrame(matchups)
    if len(matchups_df) == 0:
        raise ValueError("No valid matchups found in data")
    print(f"Created {len(matchups_df)} matchups")
    feature_cols = [col for col in matchups_df.columns if col not in ['game_id', 'home_team_id', 'away_team_id', 'home_won', 'home_goals', 'away_goals', 'date_time']]
    X = matchups_df[feature_cols].fillna(0)
    y_win = matchups_df['home_won']
    y_home_goals = matchups_df['home_goals']
    y_away_goals = matchups_df['away_goals']
    return X, y_win, y_home_goals, y_away_goals, feature_cols, matchups_df

def train_win_models(X, y):
    print("Training win prediction models (with stacking ensemble)...")
    models = {}
    rf = RandomForestClassifier(n_estimators=500, max_depth=25, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', random_state=42, n_jobs=-1, class_weight='balanced')
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42, min_samples_split=5, min_samples_leaf=2)
    lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=2000, C=1.0, penalty='l2', solver='liblinear', class_weight='balanced')
    base_estimators = [
        ('rf', rf),
        ('gb', gb),
        ('lgbm', lgbm),
        ('lr', lr)
    ]
    stack = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1, passthrough=True)
    stack.fit(X, y)
    models['stacking'] = stack
    return models

def train_goals_models(X, y):
    print("Training goals prediction models (Poisson regression)...")
    models = {}
    pr = PoissonRegressor(alpha=0.1, max_iter=300)
    pr.fit(X, y)
    models['poisson'] = pr
    return models

def predict_matchup(win_models, goals_models, home_team_id, away_team_id, df, feature_cols):
    matchup_features = create_matchup_features(df, home_team_id, away_team_id)
    if matchup_features is None:
        return 0.5, 2.5, 2.5
    feature_vector = [matchup_features.get(col, 0.0) for col in feature_cols]
    win_prob = win_models['stacking'].predict_proba([feature_vector])[0][1]
    home_goals = goals_models['poisson'].predict([feature_vector])[0]
    home_goals = np.clip(home_goals, 0, 10)
    home_goals = np.round(home_goals, 1)
    away_goals = max(0, 2.8 - home_goals + np.random.normal(0, 0.3))
    away_goals = np.clip(away_goals, 0, 10)
    away_goals = np.round(away_goals, 1)
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

def add_team_strength_features(df):
    df = df.sort_values(['team_id', 'date_time', 'game_id'])
    df['team_season_win_pct'] = 0.5
    df['opp_season_win_pct'] = 0.5
    for team_id in df['team_id'].unique():
        team_games = df[df['team_id'] == team_id].sort_values('date_time')
        win_cumsum = team_games['won'].cumsum().shift(1).fillna(0)
        game_num = np.arange(1, len(team_games)+1)
        win_pct = win_cumsum / game_num
        df.loc[team_games.index, 'team_season_win_pct'] = win_pct
    for team_id in df['team_id'].unique():
        opp_games = df[df['opponent_id'] == team_id].sort_values('date_time')
        win_cumsum = opp_games['won'].cumsum().shift(1).fillna(0)
        game_num = np.arange(1, len(opp_games)+1)
        win_pct = win_cumsum / game_num
        df.loc[opp_games.index, 'opp_season_win_pct'] = win_pct
    return df

def add_streak_features(df):
    df = df.sort_values(['team_id', 'date_time', 'game_id'])
    for team_id in df['team_id'].unique():
        team_games = df[df['team_id'] == team_id].sort_values('date_time')
        streak = 0
        streak_list = []
        for won in team_games['won']:
            if won == 1:
                streak = streak + 1 if streak > 0 else 1
            else:
                streak = streak - 1 if streak < 0 else -1
            streak_list.append(streak)
        df.loc[team_games.index, 'team_streak'] = streak_list
    return df

def add_rest_days(df):
    df = df.sort_values(['team_id', 'date_time', 'game_id'])
    df['rest_days'] = 1
    for team_id in df['team_id'].unique():
        team_games = df[df['team_id'] == team_id].sort_values('date_time')
        prev_dates = pd.to_datetime(team_games['date_time']).shift(1)
        curr_dates = pd.to_datetime(team_games['date_time'])
        rest = (curr_dates - prev_dates).dt.days.fillna(1)
        df.loc[team_games.index, 'rest_days'] = rest
    return df

def preprocess_nhl_data(df):
    df = add_team_strength_features(df)
    df = add_streak_features(df)
    df = add_rest_days(df)
    return df

def create_matchup_df(df, predictor):
    print("Creating matchup dataset with advanced features...")
    matchups = []
    unique_games = df['game_id'].unique()
    for game_id in unique_games:
        game_data = df[df['game_id'] == game_id]
        if len(game_data) >= 2:
            home_data = game_data[game_data['is_home'] == 1]
            away_data = game_data[game_data['is_home'] == 0]
            if len(home_data) > 0 and len(away_data) > 0:
                home_team = home_data.iloc[0]
                away_team = away_data.iloc[0]
                matchup_features = predictor.create_matchup_features(df, home_team['team_id'], away_team['team_id'], up_to_game_id=game_id)
                if matchup_features:
                    for col in ['team_season_win_pct', 'team_streak', 'rest_days']:
                        matchup_features[f'home_{col}'] = home_team[col]
                        matchup_features[f'away_{col}'] = away_team[col]
                    matchup_features['game_id'] = game_id
                    matchup_features['home_team_id'] = home_team['team_id']
                    matchup_features['away_team_id'] = away_team['team_id']
                    matchup_features['home_won'] = home_team['won']
                    matchup_features['home_goals'] = home_team['goals']
                    matchup_features['away_goals'] = away_team['goals']
                    matchup_features['date_time'] = home_team['date_time']
                    matchups.append(matchup_features)
    matchup_df = pd.DataFrame(matchups)
    return matchup_df

def generate_synthetic_nhl_data(num_games=2000):
    """Generate a synthetic but realistic NHL dataset for demo purposes."""
    print("Generating synthetic NHL data for demo mode...")
    np.random.seed(42)
    teams = list(range(1, 56))
    records = []
    for i in range(num_games):
        game_id = i + 1
        date_time = pd.Timestamp('2023-10-01') + pd.Timedelta(days=i//15)
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        home_goals = np.random.poisson(3) + np.random.binomial(1, 0.1)
        away_goals = np.random.poisson(2.7) + np.random.binomial(1, 0.1)
        home_won = int(home_goals > away_goals)
        for is_home, team_id, opp_id, goals, goals_against in [
            (1, home_team, away_team, home_goals, away_goals),
            (0, away_team, home_team, away_goals, home_goals)
        ]:
            rec = {
                'game_id': game_id,
                'team_id': team_id,
                'opponent_id': opp_id,
                'is_home': is_home,
                'goals': goals,
                'goals_against': goals_against,
                'shots': np.random.randint(20, 40),
                'shots_against': np.random.randint(20, 40),
                'hits': np.random.randint(5, 30),
                'hits_against': np.random.randint(5, 30),
                'blocked': np.random.randint(5, 20),
                'takeaways': np.random.randint(2, 10),
                'giveaways': np.random.randint(2, 10),
                'faceOffWins': np.random.randint(20, 40),
                'faceoffTaken': np.random.randint(30, 60),
                'powerPlayGoals': np.random.randint(0, 3),
                'powerPlayOpportunities': np.random.randint(1, 5),
                'penaltyMinutes': np.random.randint(2, 12),
                'save_percentage': np.random.uniform(0.88, 0.94),
                'shooting_percentage': np.random.uniform(0.08, 0.14),
                'timeOnIce': np.random.randint(3600, 4200),
                'date_time': date_time,
                'won': int(goals > goals_against)
            }
            records.append(rec)
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} synthetic team-game records.\n")
    return df

def get_data_with_fallback():
    """Try to fetch live NHL data, fallback to local CSV, then synthetic demo data."""
    try:
        print("Fetching live NHL data from the official NHL API...")
        df = fetch_recent_games(days=90)
        print(f"Fetched {len(df)} team-game records from the last 90 days (LIVE DATA).\n")
        return df, 'live'
    except Exception as e:
        print(f"[Warning] Could not fetch live NHL data: {e}")
        print("Falling back to local CSV data...")
        local_csv = 'data/nhl_merged.csv'
        if os.path.exists(local_csv):
            df = pd.read_csv(local_csv)
            print(f"Loaded {len(df)} team-game records from {local_csv} (LOCAL DATA).\n")
            return df, 'local'
        else:
            print("No local CSV found. Generating synthetic demo data...")
            df = generate_synthetic_nhl_data()
            return df, 'synthetic'

def get_matchup_summary(home_team, away_team, win_prob, home_goals, away_goals, overtime_prob, top_features):
    summary = []
    if win_prob > 0.7:
        summary.append(f"{home_team} is a heavy favorite against {away_team}.")
    elif win_prob < 0.3:
        summary.append(f"{away_team} is a heavy favorite against {home_team}.")
    elif win_prob > 0.55:
        summary.append(f"{home_team} is favored, but {away_team} could surprise.")
    elif win_prob < 0.45:
        summary.append(f"{away_team} is favored, but {home_team} could surprise.")
    else:
        summary.append("This is a very close matchup!")
    if abs(home_goals - away_goals) < 0.7:
        summary.append("Expect a tight, possibly low-scoring game.")
    elif home_goals > away_goals:
        summary.append(f"{home_team} is expected to outscore {away_team}.")
    else:
        summary.append(f"{away_team} is expected to outscore {home_team}.")
    if overtime_prob > 0.18:
        summary.append("There's a high chance of overtime!")
    summary.append(f"Most influential features: {', '.join(top_features)}.")
    return " ".join(summary)

def get_top_features(model, feature_vector, feature_cols, n=3):
    # Use model feature importances or coefficients if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return [feature_cols[i] for i in np.argsort(feature_vector)[-n:][::-1]]
    top_idx = np.argsort(importances)[-n:][::-1]
    return [feature_cols[i] for i in top_idx]

def advanced_predict(predictor, df, home_team_id, away_team_id, feature_cols):
    matchup_features = predictor.create_matchup_features(df, home_team_id, away_team_id)
    last_home = df[df['team_id'] == home_team_id].sort_values('date_time').iloc[-1]
    last_away = df[df['team_id'] == away_team_id].sort_values('date_time').iloc[-1]
    for col in ['team_season_win_pct', 'team_streak', 'rest_days']:
        matchup_features[f'home_{col}'] = last_home[col]
        matchup_features[f'away_{col}'] = last_away[col]
    feature_vector = np.array([matchup_features.get(col, 0.0) for col in feature_cols])
    # Predict goals for both teams
    home_goals = predictor.goals_model.predict([feature_vector])[0]
    away_goals = predictor.goals_model.predict([-feature_vector])[0]  # Use negated features for away
    # Confidence intervals (Poisson)
    home_goals_ci = stats.poisson.interval(0.8, home_goals)
    away_goals_ci = stats.poisson.interval(0.8, away_goals)
    # Win probability: use model if available, else use goals
    if hasattr(predictor, 'win_model'):
        win_prob = predictor.win_model.predict_proba([feature_vector])[0][1]
    else:
        # Use goals difference as proxy
        win_prob = stats.norm.cdf(home_goals - away_goals, loc=0, scale=1.2)
    # Overtime probability: if goals are close
    overtime_prob = float(stats.norm.pdf(home_goals - away_goals, loc=0, scale=1.2)) * 1.5
    # Consistency: ensure win_prob matches goals
    if (home_goals > away_goals and win_prob < 0.5) or (away_goals > home_goals and win_prob > 0.5):
        # Adjust win_prob to match goals
        win_prob = 0.85 if home_goals > away_goals else 0.15
    # Top features
    top_features = get_top_features(predictor.win_model, feature_vector, feature_cols)
    return {
        'win_prob': win_prob,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'home_goals_ci': home_goals_ci,
        'away_goals_ci': away_goals_ci,
        'overtime_prob': overtime_prob,
        'top_features': top_features,
        'matchup_features': matchup_features
    }

def run_cli_interface(predictor, df, feature_cols):
    print("\n=== PREDICTION INTERFACE ===")
    TEAM_ID_TO_NAME = {1: 'devils', 2: 'islanders', 3: 'rangers', 4: 'flyers', 5: 'penguins', 6: 'bruins', 7: 'sabres', 8: 'canadiens', 9: 'senators', 10: 'leafs', 12: 'hurricanes', 13: 'panthers', 14: 'lightning', 15: 'capitals', 16: 'blackhawks', 17: 'red wings', 18: 'predators', 19: 'blues', 20: 'flames', 21: 'avalanche', 22: 'oilers', 23: 'canucks', 24: 'ducks', 25: 'stars', 26: 'kings', 28: 'sharks', 29: 'blue jackets', 30: 'wild', 52: 'jets', 53: 'coyotes', 54: 'golden knights', 55: 'kraken'}
    def get_team_id_from_name(team_name):
        for team_id, name in TEAM_ID_TO_NAME.items():
            if name.lower() == team_name.lower():
                return team_id
        raise ValueError(f"Team '{team_name}' not found.")
    while True:
        try:
            home_team_name = input("Enter Home Team Name (or 'quit' to exit): ").strip()
            if home_team_name.lower() == 'quit':
                break
            away_team_name = input("Enter Away Team Name: ").strip()
            home_team_id = get_team_id_from_name(home_team_name)
            away_team_id = get_team_id_from_name(away_team_name)
            result = advanced_predict(predictor, df, home_team_id, away_team_id, feature_cols)
            win_prob = result['win_prob']
            home_goals = result['home_goals']
            away_goals = result['away_goals']
            home_goals_ci = result['home_goals_ci']
            away_goals_ci = result['away_goals_ci']
            overtime_prob = result['overtime_prob']
            top_features = result['top_features']
            summary = get_matchup_summary(home_team_name, away_team_name, win_prob, home_goals, away_goals, overtime_prob, top_features)
            print(f"\nPrediction: {home_team_name} vs {away_team_name}")
            print(f"{home_team_name} win probability: {win_prob:.1%}")
            print(f"{away_team_name} win probability: {(1-win_prob):.1%}")
            print(f"Predicted Score: {home_team_name} {home_goals:.1f} (80% CI: {home_goals_ci[0]:.1f}-{home_goals_ci[1]:.1f}) - {away_team_name} {away_goals:.1f} (80% CI: {away_goals_ci[0]:.1f}-{away_goals_ci[1]:.1f})")
            print(f"Chance of overtime: {overtime_prob:.1%}")
            print(f"Top features: {', '.join(top_features)}")
            print(f"Summary: {summary}")
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            break
    print("\nThank you for using the NHL Outcome Predictor!")

# Minimal Flask web dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NHL Outcome Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f7f7fa; margin: 0; padding: 0; }
        .container { max-width: 500px; margin: 40px auto; background: #fff; border-radius: 10px; box-shadow: 0 2px 8px #0001; padding: 32px; }
        h1 { text-align: center; color: #003366; }
        label { font-weight: bold; }
        input, select { width: 100%; padding: 8px; margin: 8px 0 16px 0; border-radius: 4px; border: 1px solid #ccc; }
        button { background: #003366; color: #fff; border: none; padding: 12px 24px; border-radius: 4px; font-size: 1em; cursor: pointer; width: 100%; }
        .result { background: #e6f2ff; border-left: 4px solid #003366; padding: 16px; margin-top: 24px; border-radius: 6px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>NHL Outcome Predictor</h1>
        <form method="post">
            <label for="home_team">Home Team:</label>
            <select name="home_team" id="home_team" required>
                {% for tid, tname in teams.items() %}
                <option value="{{ tid }}">{{ tname.title() }}</option>
                {% endfor %}
            </select>
            <label for="away_team">Away Team:</label>
            <select name="away_team" id="away_team" required>
                {% for tid, tname in teams.items() %}
                <option value="{{ tid }}">{{ tname.title() }}</option>
                {% endfor %}
            </select>
            <button type="submit">Predict</button>
        </form>
        {% if result %}
        <div class="result">
            <strong>Prediction:</strong><br>
            <b>{{ result.home_team }}</b> vs <b>{{ result.away_team }}</b><br>
            <b>{{ result.home_team }}</b> win probability: {{ result.win_prob }}<br>
            <b>{{ result.away_team }}</b> win probability: {{ result.away_prob }}<br>
            Predicted Score: <b>{{ result.home_team }}</b> {{ result.home_goals }} - <b>{{ result.away_team }}</b> {{ result.away_goals }}<br>
            <span>{{ result.comment }}</span>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

def run_web_dashboard(predictor, df, feature_cols):
    if not FLASK_AVAILABLE:
        print("Flask is not installed. Please install Flask to use the web dashboard: pip install flask")
        return
    app = Flask(__name__)
    TEAM_ID_TO_NAME = {1: 'devils', 2: 'islanders', 3: 'rangers', 4: 'flyers', 5: 'penguins', 6: 'bruins', 7: 'sabres', 8: 'canadiens', 9: 'senators', 10: 'leafs', 12: 'hurricanes', 13: 'panthers', 14: 'lightning', 15: 'capitals', 16: 'blackhawks', 17: 'red wings', 18: 'predators', 19: 'blues', 20: 'flames', 21: 'avalanche', 22: 'oilers', 23: 'canucks', 24: 'ducks', 25: 'stars', 26: 'kings', 28: 'sharks', 29: 'blue jackets', 30: 'wild', 52: 'jets', 53: 'coyotes', 54: 'golden knights', 55: 'kraken'}
    @app.route('/', methods=['GET', 'POST'])
    def index():
        result = None
        if request.method == 'POST':
            home_team_id = int(request.form['home_team'])
            away_team_id = int(request.form['away_team'])
            home_team_name = TEAM_ID_TO_NAME[home_team_id].title()
            away_team_name = TEAM_ID_TO_NAME[away_team_id].title()
            result = advanced_predict(predictor, df, home_team_id, away_team_id, feature_cols)
        return render_template_string(DASHBOARD_HTML, teams=TEAM_ID_TO_NAME, result=result)
    print("\nWeb dashboard running at http://127.0.0.1:5000/ (Ctrl+C to stop)")
    app.run(debug=False)

def run_predictor_pipeline(df, data_source):
    df = preprocess_nhl_data(df)
    predictor = SportsOutcomePredictor(NHL_CONFIG)
    matchup_df = create_matchup_df(df, predictor)
    feature_cols = [col for col in matchup_df.columns if col not in ['game_id', 'home_team_id', 'away_team_id', 'home_won', 'home_goals', 'away_goals', 'date_time']]
    matchup_df = matchup_df.sort_values('date_time')
    split_idx = int(0.8 * len(matchup_df))
    train_idx = matchup_df.index[:split_idx]
    test_idx = matchup_df.index[split_idx:]
    train_df = matchup_df.loc[train_idx]
    test_df = matchup_df.loc[test_idx]
    predictor.fit(df, train_df, feature_cols, 'home_won', 'home_goals')
    print(f"\n=== EVALUATION ({data_source.upper()} DATA) ===")
    predictor.evaluate(test_df, 'home_won', 'home_goals')
    return predictor, df, feature_cols

def main():
    print("=== NHL Outcome Predictor (Live+Offline+Demo+Web, Max Impressive) ===")
    df, data_source = get_data_with_fallback()
    predictor, df, feature_cols = run_predictor_pipeline(df, data_source)
    if '--web' in sys.argv:
        run_web_dashboard(predictor, df, feature_cols)
    else:
        run_cli_interface(predictor, df, feature_cols)

if __name__ == "__main__":
    main()


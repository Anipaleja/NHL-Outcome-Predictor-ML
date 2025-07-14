import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.web.nhl.com/api/v1"

# --- API Fetchers ---
def get_teams():
    resp = requests.get(f"{BASE_URL}/teams").json()
    return pd.DataFrame(resp['teams'])

def get_schedule(start_date=None, end_date=None):
    params = {}
    if start_date: params['startDate'] = start_date
    if end_date: params['endDate'] = end_date
    resp = requests.get(f"{BASE_URL}/schedule", params=params).json()
    games = []
    for day in resp.get('dates', []):
        for game in day['games']:
            games.append({
                'gamePk': game['gamePk'],
                'date': day['date'],
                'home_team_id': game['teams']['home']['team']['id'],
                'away_team_id': game['teams']['away']['team']['id'],
                'status': game['status']['detailedState']
            })
    return pd.DataFrame(games)

def get_game_boxscore(game_id):
    resp = requests.get(f"{BASE_URL}/game/{game_id}/boxscore").json()
    return resp

def get_game_linescore(game_id):
    resp = requests.get(f"{BASE_URL}/game/{game_id}/linescore").json()
    return resp

# --- Data Preprocessing ---
def fetch_recent_games(days=60):
    today = datetime.utcnow().date()
    start = today - timedelta(days=days)
    schedule = get_schedule(start_date=start.isoformat(), end_date=today.isoformat())
    all_games = []
    for _, row in schedule.iterrows():
        game_id = row['gamePk']
        box = get_game_boxscore(game_id)
        line = get_game_linescore(game_id)
        # Only completed games
        if line.get('currentPeriod', 0) == 0:
            continue
        for side in ['home', 'away']:
            team = box['teams'][side]
            stats = team['teamStats']['teamSkaterStats'] if 'teamSkaterStats' in team['teamStats'] else {}
            team_id = team['team']['id']
            opp_id = box['teams']['away']['team']['id'] if side == 'home' else box['teams']['home']['team']['id']
            goals = team['goals']
            goals_against = box['teams']['away']['goals'] if side == 'home' else box['teams']['home']['goals']
            record = {
                'game_id': game_id,
                'date_time': line['startTime'][:10],
                'team_id': team_id,
                'opponent_id': opp_id,
                'is_home': 1 if side == 'home' else 0,
                'goals': goals,
                'goals_against': goals_against,
                'won': int(goals > goals_against),
                'shots': stats.get('shots', 0),
                'hits': stats.get('hits', 0),
                'powerPlayGoals': stats.get('powerPlayGoals', 0),
                'powerPlayOpportunities': stats.get('powerPlayOpportunities', 0),
                'faceOffWinPercentage': float(stats.get('faceOffWinPercentage', 0)),
                'blocked': stats.get('blocked', 0),
                'takeaways': stats.get('takeaways', 0),
                'giveaways': stats.get('giveaways', 0),
                'pim': stats.get('pim', 0),
                'shots_against': box['teams']['away']['teamStats']['teamSkaterStats']['shots'] if side == 'home' else box['teams']['home']['teamStats']['teamSkaterStats']['shots'],
                'hits_against': box['teams']['away']['teamStats']['teamSkaterStats']['hits'] if side == 'home' else box['teams']['home']['teamStats']['teamSkaterStats']['hits'],
            }
            all_games.append(record)
    
    df = pd.DataFrame(all_games)
    # Fill missing columns for compatibility
    for col in ['faceOffWins', 'faceoffTaken', 'save_percentage', 'shooting_percentage', 'missedShots', 'penaltyMinutes', 'power_play_pct', 'goals_per_shot', 'efficiency_score', 'overall_efficiency', 'defensive_pressure', 'penalty_efficiency', 'goal_scoring_rate', 'shot_generation_rate', 'hit_rate', 'power_play_efficiency', 'penalty_kill_efficiency', 'scoring_momentum', 'defensive_momentum', 'offensive_strength', 'defensive_strength']:
        if col not in df.columns:
            df[col] = 0.0
    
    # Calculate advanced metrics
    df = calculate_advanced_metrics(df)
    return df

def calculate_advanced_metrics(df):
    """Calculate advanced hockey metrics for better predictions"""
    # Goal differential
    df['goal_differential'] = df['goals'] - df['goals_against']
    
    # Shot differential and percentages
    df['shot_differential'] = df['shots'] - df['shots_against']
    df['shooting_percentage'] = np.where(df['shots'] > 0, df['goals'] / df['shots'], 0)
    df['save_percentage'] = np.where(df['shots_against'] > 0, 1 - (df['goals_against'] / df['shots_against']), 0)
    
    # Power play efficiency
    df['power_play_pct'] = np.where(df['powerPlayOpportunities'] > 0, 
                                    df['powerPlayGoals'] / df['powerPlayOpportunities'], 0)
    
    # Face-off metrics
    df['faceOffWins'] = df['faceOffWinPercentage'] * df.get('faceoffTaken', 50) / 100
    df['faceoffTaken'] = df.get('faceoffTaken', 50)
    
    # Advanced efficiency metrics
    df['goals_per_shot'] = df['shooting_percentage']
    df['possession_ratio'] = df['faceOffWinPercentage'] / 100
    df['efficiency_score'] = (df['goals'] + df['takeaways'] - df['giveaways']) / (df['shots'] + 1)
    
    # Physical play metrics
    df['hit_rate'] = df['hits'] / (df['shots'] + df['hits'] + 1)
    df['defensive_pressure'] = (df['hits'] + df['blocked'] + df['takeaways']) / 3
    
    # Momentum indicators
    df['scoring_momentum'] = df['goals'] * df['shooting_percentage']
    df['defensive_momentum'] = df['blocked'] + df['takeaways'] - df['giveaways']
    
    # Overall strength indicators
    df['offensive_strength'] = (df['goals'] + df['shots'] + df['powerPlayGoals']) / 3
    df['defensive_strength'] = (df['blocked'] + df['takeaways'] + df['save_percentage']) / 3
    
    # Penalty metrics
    df['penalty_efficiency'] = np.where(df['pim'] > 0, df['powerPlayGoals'] / df['pim'], 0)
    
    return df

def get_upcoming_games(days_ahead=7):
    """Fetch upcoming NHL games for prediction"""
    today = datetime.utcnow().date()
    end_date = today + timedelta(days=days_ahead)
    
    schedule = get_schedule(start_date=today.isoformat(), end_date=end_date.isoformat())
    upcoming = schedule[schedule['status'].isin(['Preview', 'Scheduled'])].copy()
    
    return upcoming

def get_team_info():
    """Get comprehensive team information"""
    teams_df = get_teams()
    return teams_df

def validate_api_connection():
    """Test if NHL API is accessible"""
    try:
        response = requests.get(f"{BASE_URL}/teams", timeout=10)
        if response.status_code == 200:
            logger.info("NHL API connection successful")
            return True
        else:
            logger.warning(f"NHL API returned status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"NHL API connection failed: {e}")
        return False 
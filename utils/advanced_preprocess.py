import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedNHLDataProcessor:
    """Advanced NHL data processor that handles both CSV and API data"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir
        self.team_mappings = self._create_team_mappings()
        
    def _create_team_mappings(self) -> Dict:
        """Create team ID to name mappings"""
        return {
            1: 'Devils', 2: 'Islanders', 3: 'Rangers', 4: 'Flyers', 5: 'Penguins',
            6: 'Bruins', 7: 'Sabres', 8: 'Canadiens', 9: 'Senators', 10: 'Maple Leafs',
            12: 'Hurricanes', 13: 'Panthers', 14: 'Lightning', 15: 'Capitals', 16: 'Blackhawks',
            17: 'Red Wings', 18: 'Predators', 19: 'Blues', 20: 'Flames', 21: 'Avalanche',
            22: 'Oilers', 23: 'Canucks', 24: 'Ducks', 25: 'Stars', 26: 'Kings',
            28: 'Sharks', 29: 'Blue Jackets', 30: 'Wild', 52: 'Jets', 53: 'Coyotes',
            54: 'Golden Knights', 55: 'Kraken'
        }
    
    def load_csv_data(self) -> pd.DataFrame:
        """Load and combine all CSV data sources"""
        try:
            # Load main game data
            games_df = pd.read_csv(os.path.join(self.data_dir, 'Data3/game.csv'))
            teams_stats_df = pd.read_csv(os.path.join(self.data_dir, 'Data3/game_teams_stats.csv'))
            
            # Clean column names (remove quotes)
            games_df.columns = games_df.columns.str.strip('"')
            teams_stats_df.columns = teams_stats_df.columns.str.strip('"')
            
            # Optional data files
            try:
                skater_stats_df = pd.read_csv(os.path.join(self.data_dir, 'Data3/game_skater_stats.csv'))
                skater_stats_df.columns = skater_stats_df.columns.str.strip('"')
            except FileNotFoundError:
                skater_stats_df = None
                
            try:
                goalie_stats_df = pd.read_csv(os.path.join(self.data_dir, 'Data1/game_goalie_stats.csv'))
                goalie_stats_df.columns = goalie_stats_df.columns.str.strip('"')
            except FileNotFoundError:
                goalie_stats_df = None
                
            try:
                goals_df = pd.read_csv(os.path.join(self.data_dir, 'Data1/game_goals.csv'))
                goals_df.columns = goals_df.columns.str.strip('"')
            except FileNotFoundError:
                goals_df = None
                
            try:
                penalties_df = pd.read_csv(os.path.join(self.data_dir, 'Data1/game_penalties.csv'))
                penalties_df.columns = penalties_df.columns.str.strip('"')
            except FileNotFoundError:
                penalties_df = None
                
            try:
                plays_df = pd.read_csv(os.path.join(self.data_dir, 'Data4/game_plays.csv'))
                plays_df.columns = plays_df.columns.str.strip('"')
            except FileNotFoundError:
                plays_df = None
            
            # Merge data
            combined_df = self._merge_csv_data(games_df, teams_stats_df, skater_stats_df, 
                                               goalie_stats_df, goals_df, penalties_df, plays_df)
            
            logger.info(f"Loaded CSV data: {len(combined_df)} records")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            return pd.DataFrame()
    
    def _merge_csv_data(self, games_df, teams_stats_df, skater_stats_df=None, 
                        goalie_stats_df=None, goals_df=None, penalties_df=None, plays_df=None) -> pd.DataFrame:
        """Merge all CSV data sources into unified format"""
        
        # Start with team stats as base
        merged_df = teams_stats_df.copy()
        
        # Add game information
        game_cols = ['game_id', 'date_time_GMT', 'season', 'type', 'away_team_id', 'home_team_id', 'away_goals', 'home_goals']
        available_game_cols = [col for col in game_cols if col in games_df.columns]
        
        merged_df = merged_df.merge(
            games_df[available_game_cols],
            on='game_id', how='left'
        )
        
        # Add opponent information
        merged_df['opponent_id'] = np.where(
            merged_df['HoA'] == 'home', 
            merged_df['away_team_id'], 
            merged_df['home_team_id']
        )
        
        # Calculate goals against
        merged_df['goals_against'] = np.where(
            merged_df['HoA'] == 'home',
            merged_df['away_goals'],
            merged_df['home_goals']
        )
        
        # Rename columns for consistency
        merged_df = merged_df.rename(columns={
            'date_time_GMT': 'date_time',
            'HoA': 'home_away',
            'powerPlayOpportunities': 'powerPlayOpportunities',
            'faceOffWinPercentage': 'faceOffWinPercentage'
        })
        
        # Convert boolean columns
        if 'won' in merged_df.columns:
            merged_df['won'] = merged_df['won'].astype(str).str.upper()
            merged_df['won'] = (merged_df['won'] == 'TRUE').astype(int)
        
        # Add is_home indicator
        merged_df['is_home'] = (merged_df['home_away'] == 'home').astype(int)
        
        # Add missing shots_against column by matching with opponent
        if 'shots_against' not in merged_df.columns and 'shots' in merged_df.columns:
            # Create a mapping of game_id to shots for opponent calculation
            shots_by_game_team = merged_df.set_index(['game_id', 'team_id'])['shots'].to_dict()
            
            def get_opponent_shots(row):
                game_id = row['game_id']
                opponent_id = row['opponent_id']
                return shots_by_game_team.get((game_id, opponent_id), 0)
            
            merged_df['shots_against'] = merged_df.apply(get_opponent_shots, axis=1)
        
        # Add missing hits_against column similarly
        if 'hits_against' not in merged_df.columns and 'hits' in merged_df.columns:
            hits_by_game_team = merged_df.set_index(['game_id', 'team_id'])['hits'].to_dict()
            
            def get_opponent_hits(row):
                game_id = row['game_id']
                opponent_id = row['opponent_id']
                return hits_by_game_team.get((game_id, opponent_id), 0)
            
            merged_df['hits_against'] = merged_df.apply(get_opponent_hits, axis=1)
        
        # Add aggregated skater stats if available
        if skater_stats_df is not None and len(skater_stats_df) > 0:
            try:
                skater_agg = self._aggregate_skater_stats(skater_stats_df)
                if len(skater_agg) > 0:
                    merged_df = merged_df.merge(skater_agg, on=['game_id', 'team_id'], how='left')
            except Exception as e:
                logger.warning(f"Could not aggregate skater stats: {e}")
        
        # Add goalie stats if available
        if goalie_stats_df is not None and len(goalie_stats_df) > 0:
            try:
                goalie_agg = self._aggregate_goalie_stats(goalie_stats_df)
                if len(goalie_agg) > 0:
                    merged_df = merged_df.merge(goalie_agg, on=['game_id', 'team_id'], how='left')
            except Exception as e:
                logger.warning(f"Could not aggregate goalie stats: {e}")
        
        # Add penalty stats if available
        if penalties_df is not None and len(penalties_df) > 0:
            try:
                penalty_agg = self._aggregate_penalty_stats(penalties_df)
                if len(penalty_agg) > 0:
                    merged_df = merged_df.merge(penalty_agg, on=['game_id', 'team_id'], how='left')
            except Exception as e:
                logger.warning(f"Could not aggregate penalty stats: {e}")
        
        # Calculate advanced metrics
        merged_df = self._calculate_advanced_metrics(merged_df)
        
        # Clean and format
        merged_df = self._clean_data(merged_df)
        
        return merged_df
    
    def _aggregate_skater_stats(self, skater_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate individual skater stats to team level"""
        agg_dict = {
            'assists': 'sum',
            'goals': 'sum',
            'shots': 'sum',
            'hits': 'sum',
            'powerPlayGoals': 'sum',
            'powerPlayAssists': 'sum',
            'penaltyMinutes': 'sum',
            'faceOffWins': 'sum',
            'faceoffTaken': 'sum',
            'takeaways': 'sum',
            'giveaways': 'sum',
            'shortHandedGoals': 'sum',
            'shortHandedAssists': 'sum',
            'blocked': 'sum',
            'plusMinus': 'sum',
            'evenTimeOnIce': 'sum',
            'powerPlayTimeOnIce': 'sum',
            'shortHandedTimeOnIce': 'sum'
        }
        
        # Only aggregate columns that exist
        existing_cols = {k: v for k, v in agg_dict.items() if k in skater_df.columns}
        
        if not existing_cols:
            return pd.DataFrame()
            
        skater_agg = skater_df.groupby(['game_id', 'team_id']).agg(existing_cols).reset_index()
        
        # Flatten column names
        skater_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in skater_agg.columns.values]
        skater_agg = skater_agg.rename(columns=lambda x: x.replace('_sum', '_team'))
        
        return skater_agg
    
    def _aggregate_goalie_stats(self, goalie_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate goalie stats to team level"""
        agg_dict = {
            'saves': 'sum',
            'powerPlaySaves': 'sum',
            'shortHandedSaves': 'sum',
            'evenSaves': 'sum',
            'shortHandedShotsAgainst': 'sum',
            'evenShotsAgainst': 'sum',
            'powerPlayShotsAgainst': 'sum',
            'decision': 'first',
            'savePercentage': 'mean',
            'powerPlaySavePercentage': 'mean',
            'evenStrengthSavePercentage': 'mean'
        }
        
        existing_cols = {k: v for k, v in agg_dict.items() if k in goalie_df.columns}
        
        if not existing_cols:
            return pd.DataFrame()
            
        goalie_agg = goalie_df.groupby(['game_id', 'team_id']).agg(existing_cols).reset_index()
        goalie_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in goalie_agg.columns.values]
        goalie_agg = goalie_agg.rename(columns=lambda x: x.replace('_sum', '_total').replace('_mean', '_avg'))
        
        return goalie_agg
    
    def _aggregate_penalty_stats(self, penalties_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate penalty stats to team level"""
        penalty_agg = penalties_df.groupby(['game_id', 'team_id']).agg({
            'penaltyMinutes': 'sum',
            'penaltyType': 'count'
        }).reset_index()
        
        penalty_agg = penalty_agg.rename(columns={
            'penaltyMinutes': 'total_penalty_minutes',
            'penaltyType': 'penalty_count'
        })
        
        return penalty_agg
    
    def _calculate_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced hockey analytics"""
        # Basic differentials
        df['goal_differential'] = df['goals'] - df['goals_against']
        
        # Handle shots_against if not present
        if 'shots_against' not in df.columns:
            # Try to calculate from opponent data or use default
            df['shots_against'] = df.get('shots', 0) * 0.95  # Approximate
        
        df['shot_differential'] = df.get('shots', 0) - df.get('shots_against', 0)
        
        # Efficiency metrics
        df['shooting_percentage'] = np.where(df.get('shots', 0) > 0, 
                                            df['goals'] / df.get('shots', 1), 0)
        df['save_percentage'] = np.where(df.get('shots_against', 0) > 0,
                                        1 - (df['goals_against'] / df.get('shots_against', 1)), 0)
        
        # Power play metrics
        df['power_play_pct'] = np.where(df.get('powerPlayOpportunities', 0) > 0,
                                       df.get('powerPlayGoals', 0) / df.get('powerPlayOpportunities', 1), 0)
        
        # Face-off metrics
        df['faceoff_win_pct'] = df.get('faceOffWinPercentage', 50) / 100
        
        # Physical play
        df['hits_per_shot'] = np.where(df.get('shots', 0) > 0,
                                      df.get('hits', 0) / df.get('shots', 1), 0)
        
        # Possession metrics
        df['possession_ratio'] = df['faceoff_win_pct']
        df['puck_control'] = (df.get('takeaways', 0) - df.get('giveaways', 0)) / (df.get('takeaways', 0) + df.get('giveaways', 0) + 1)
        
        # Defensive metrics
        df['defensive_efficiency'] = (df.get('blocked', 0) + df.get('takeaways', 0)) / (df.get('shots_against', 1) + df.get('blocked', 0))
        
        # Momentum indicators
        df['offensive_momentum'] = df['goals'] * df['shooting_percentage'] * df['power_play_pct']
        df['defensive_momentum'] = df['save_percentage'] * df['defensive_efficiency']
        
        # Overall efficiency
        df['team_efficiency'] = (df['shooting_percentage'] + df['save_percentage'] + df['power_play_pct'] + df['faceoff_win_pct']) / 4
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Convert date column
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
        
        # Remove invalid records
        df = df[df['goals'].notna() & df['goals_against'].notna()]
        
        # Add season indicators
        if 'season' in df.columns:
            df['season_numeric'] = df['season'].astype(str).str[:4].astype(int)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Create rolling window features for teams"""
        df = df.sort_values(['team_id', 'date_time']).reset_index(drop=True)
        
        feature_cols = [
            'goals', 'goals_against', 'shots', 'hits', 'powerPlayGoals',
            'faceOffWinPercentage', 'takeaways', 'giveaways', 'blocked',
            'goal_differential', 'shooting_percentage', 'save_percentage',
            'power_play_pct', 'team_efficiency', 'won'
        ]
        
        for window in windows:
            for col in feature_cols:
                if col in df.columns:
                    df[f'{col}_roll_{window}'] = df.groupby('team_id')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df[f'{col}_roll_std_{window}'] = df.groupby('team_id')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
        
        return df
    
    def create_head_to_head_features(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Create head-to-head historical features"""
        h2h_features = []
        
        for _, row in df.iterrows():
            team_id = row['team_id']
            opponent_id = row['opponent_id']
            game_date = row['date_time']
            
            # Get historical H2H games
            h2h_mask = (
                ((df['team_id'] == team_id) & (df['opponent_id'] == opponent_id)) |
                ((df['team_id'] == opponent_id) & (df['opponent_id'] == team_id))
            ) & (df['date_time'] < game_date)
            
            h2h_games = df[h2h_mask].tail(window)
            
            if len(h2h_games) > 0:
                # Calculate H2H metrics
                team_h2h = h2h_games[h2h_games['team_id'] == team_id]
                
                h2h_record = {
                    'h2h_games_count': len(h2h_games),
                    'h2h_win_rate': team_h2h['won'].mean() if len(team_h2h) > 0 else 0,
                    'h2h_avg_goals': team_h2h['goals'].mean() if len(team_h2h) > 0 else 0,
                    'h2h_avg_goals_against': team_h2h['goals_against'].mean() if len(team_h2h) > 0 else 0,
                    'h2h_avg_goal_diff': team_h2h['goal_differential'].mean() if len(team_h2h) > 0 else 0
                }
            else:
                h2h_record = {
                    'h2h_games_count': 0,
                    'h2h_win_rate': 0,
                    'h2h_avg_goals': 0,
                    'h2h_avg_goals_against': 0,
                    'h2h_avg_goal_diff': 0
                }
            
            h2h_features.append(h2h_record)
        
        h2h_df = pd.DataFrame(h2h_features)
        return pd.concat([df.reset_index(drop=True), h2h_df], axis=1)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature engineering"""
        # Basic features already calculated in _calculate_advanced_metrics
        
        # Interaction features
        df['goals_shots_interaction'] = df['goals'] * df.get('shots', 0)
        df['defense_interaction'] = df.get('blocked', 0) * df.get('takeaways', 0)
        df['pp_efficiency_interaction'] = df.get('powerPlayGoals', 0) * df['faceoff_win_pct']
        
        # Time-based features
        if 'date_time' in df.columns:
            df['month'] = df['date_time'].dt.month
            df['day_of_week'] = df['date_time'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Home/Away streaks (simplified)
        df['home_game'] = df['is_home']
        
        # Lag features (previous game performance)
        lag_cols = ['goals', 'goals_against', 'won', 'goal_differential']
        for col in lag_cols:
            if col in df.columns:
                df[f'{col}_lag1'] = df.groupby('team_id')[col].shift(1)
                df[f'{col}_lag2'] = df.groupby('team_id')[col].shift(2)
        
        return df

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

model = None
df = None


scaler = StandardScaler()


TEAM_ID_TO_NAME = {
    1: 'devils',
    2: 'islanders',
    3: 'rangers',
    4: 'flyers',
    5: 'penguins',
    6: 'bruins',
    7: 'sabres',
    8: 'canadiens',
    9: 'senators',
    10: 'leafs',
    12: 'hurricanes',
    13: 'panthers',
    14: 'lightning',
    15: 'capitals',
    16: 'blackhawks',
    17: 'red wings',
    18: 'predators',
    19: 'blues',
    20: 'flames',
    21: 'avalanche',
    22: 'oilers',
    23: 'canucks',
    24: 'ducks',
    25: 'stars',
    26: 'kings',
    28: 'sharks',
    29: 'blue jackets',
    30: 'wild',
    52: 'jets',
    53: 'coyotes',
    54: 'golden knights',
    55: 'kraken'
}


def get_team_id_from_name(team_name):
    for team_id, name in TEAM_ID_TO_NAME.items():
        if name.lower() == team_name.lower():
            return team_id
    raise ValueError(f"Team '{team_name}' not found.")


def get_team_features(df, team_id):
    team_data = df[df['team_id'] == team_id].copy()

    # Drop same columns as in preprocess_data
    cols_to_drop = ["team_id", "game_id", "HoA", "won", "settled_in", "head_coach", "is_home"]
    team_data = team_data.drop(columns=[col for col in cols_to_drop if col in team_data.columns], errors='ignore')

   

    # Label encode any object columns just like preprocess_data
    for col in team_data.select_dtypes(include='object').columns:
        team_data[col] = LabelEncoder().fit_transform(team_data[col].astype(str))

    # Drop inf/nan values
    team_data = team_data.replace([np.inf, -np.inf], np.nan).dropna()

    # Get mean of numeric features
    team_features = team_data.mean(numeric_only=True)

    return team_features.values


def create_team_features(df, team_id):
    team_data = df[df['team_id'] == team_id].copy()

    cols_to_drop = ["team_id", "game_id", "HoA", "won", "settled_in", "head_coach", "is_home", "date"]
    team_data = team_data.drop(columns=[col for col in cols_to_drop if col in team_data.columns], errors='ignore')

    for col in team_data.select_dtypes(include='object').columns:
        team_data[col] = LabelEncoder().fit_transform(team_data[col].astype(str))

    team_data = team_data.replace([np.inf, -np.inf], np.nan).dropna()

    team_avg = team_data.mean(numeric_only=True)

    # Reindex to match training features exactly
    expected_cols = scaler.feature_names_in_  # This attribute exists in sklearn >= 1.0
    team_avg = team_avg.reindex(expected_cols).fillna(0)

    print("Number of features in team_avg:", team_avg.shape[0])
    expected_features = len(expected_cols)
    assert team_avg.shape[0] == expected_features, f"Expected {expected_features} features, got {team_avg.shape[0]}"

    team_avg_df = pd.DataFrame([team_avg], columns=expected_cols)

    team_scaled = scaler.transform(team_avg_df)

    return torch.tensor(team_scaled, dtype=torch.float32)


def create_matchup_features(df, team1_id, team2_id):
    team1_features = create_team_features(df, team1_id)
    team2_features = create_team_features(df, team2_id)
    matchup_features = torch.cat((team1_features, team2_features), dim=1)
    return matchup_features

def predict_matchup(team_id):
    team_tensor = create_team_features(df, team_id)
    model.eval()
    with torch.no_grad():
        predicted_goals = model(team_tensor).item()
    return predicted_goals


def load_and_merge_data():
    # Load datasets
    teams_stats = pd.read_csv("data/game_teams_stats.csv")
    goalie_stats = pd.read_csv("data/game_goalie_stats.csv")
    skater_stats = pd.read_csv("data/game_skater_stats.csv")

    # Aggregate skater stats by game_id and team_id (sum numeric stats)
    skater_agg = skater_stats.groupby(['game_id', 'team_id']).sum(numeric_only=True).reset_index()

    # Aggregate goalie stats by game_id and team_id (sum numeric stats)
    goalie_agg = goalie_stats.groupby(['game_id', 'team_id']).sum(numeric_only=True).reset_index()

    # Merge all on game_id and team_id
    merged = teams_stats.merge(goalie_agg, on=['game_id', 'team_id'], how='left', suffixes=('', '_goalie'))
    merged = merged.merge(skater_agg, on=['game_id', 'team_id'], how='left', suffixes=('', '_skater'))

    # Fill NaNs (if goalie or skater data missing)
    merged.fillna(0, inplace=True)

    return merged

def preprocess_data(df):
    cols_to_drop = ["team_id", "game_id", "HoA", "won", "settled_in", "head_coach", "is_home", "date"]
    # Drop columns that are not needed for training
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    print("Columns after drop (preprocess_data):", df.columns.tolist())


    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df.drop(columns=["goals"])
    y = df["goals"]

    return X, y  # Don't scale here

def build_model(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return model

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        losses = []
        for batch_X, batch_y in test_loader:
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            losses.append(loss.item())
    print(f"Test MSE: {np.mean(losses):.4f}")

def main():
    global df, model
    df = load_and_merge_data()
    X, y = preprocess_data(df)

    # Fit global scaler
    X_scaled = scaler.fit_transform(X)
    print("Features used for training:", X.columns.tolist())
    print("Number of features for scaler:", X.shape[1])


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Torch datasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Build and train model
    model = build_model(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epochs=10)
    evaluate_model(model, test_loader, criterion)

    # User prediction input
    team1_name = input("Enter Team 1 Name: ")
    team2_name = input("Enter Team 2 Name: ")

    try:
        team1_id = get_team_id_from_name(team1_name)
        team2_id = get_team_id_from_name(team2_name)
    except ValueError as e:
        print(e)
        return

    goals_team1 = predict_matchup(team1_id)
    goals_team2 = predict_matchup(team2_id)

    print(f"Predicted goals - {team1_name}: {goals_team1:.1f}, {team2_name}: {goals_team2:.1f}")
    if goals_team1 > goals_team2:
        print(f"{team1_name} is predicted to WIN!")
    elif goals_team2 > goals_team1:
        print(f"{team2_name} is predicted to WIN!")
    else:
        print("It's predicted to be a DRAW!")


if __name__ == "__main__":
    main()


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def get_team_id_from_name(df, team_name):
    name_map = df[['team_id', 'team_name']].drop_duplicates()
    match = name_map[name_map['team_name'].str.lower() == team_name.lower()]
    if match.empty:
        raise ValueError(f"Team '{team_name}' not found.")
    return int(match['team_id'].values[0])

def get_team_features(df, team_id):
    # Pull aggregated stats for given team_id (you can choose recent games or season averages)
    team_data = df[df['team_id'] == team_id].mean(numeric_only=True)  # example: average stats
    # Drop non-feature columns
    team_features = team_data.drop(['team_id', 'game_id', 'goals'], errors='ignore')
    return team_features.values

def create_matchup_features(df, team1_id, team2_id):
    # Get features for both teams
    team1_feats = get_team_features(df, team1_id)
    team2_feats = get_team_features(df, team2_id)

    # Combine features (how you combine depends on your model input)
    # For example, concatenate team1 and team2 features side-by-side
    matchup_features = np.concatenate([team1_feats, team2_feats])

    # Scale features (use the scaler you fit during training)
    matchup_features_scaled = scaler.transform(matchup_features.reshape(1, -1))

    return torch.tensor(matchup_features_scaled, dtype=torch.float32)

def predict_matchup(team1_id, team2_id):
    df = load_and_merge_data()
    # Use your existing scaler object (you might need to save it globally or return from preprocess_data)
    matchup_tensor = create_matchup_features(df, team1_id, team2_id)
    model.eval()
    with torch.no_grad():
        pred_goals = model(matchup_tensor).item()
    return pred_goals

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
    # Drop columns that are not useful or leak label
    cols_to_drop = ["team_id", "game_id", "HoA", "won", "settled_in", "head_coach", "is_home"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Encode categorical columns if any remain
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Remove infinite and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Separate features and target (goals)
    X = df.drop(columns=["goals"])
    y = df["goals"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

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
    df = load_and_merge_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = build_model(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epochs=10)
    evaluate_model(model, test_loader, criterion)
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
    # Convert X_test (numpy) to torch tensor with float32 dtype
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_test_tensor).squeeze().numpy()


    for i in range(10):
        print(f"Predicted: {preds[i]:.3f}, Actual: {y_test.values[i]:.3f}")

    df = load_and_merge_data()

team1_name = input("Enter Team 1 Name: ")
team2_name = input("Enter Team 2 Name: ")

try:
    team1_id = get_team_id_from_name(df, team1_name)
    team2_id = get_team_id_from_name(df, team2_name)
except ValueError as e:
    print(e)
    exit()

goals_team1 = predict_matchup(team1_id, team2_id)
goals_team2 = predict_matchup(team2_id, team1_id)

print(f"Predicted goals - {team1_name}: {goals_team1:.1f}, {team2_name}: {goals_team2:.1f}")
if goals_team1 > goals_team2:
    print(f"{team1_name} is predicted to WIN!")
elif goals_team2 > goals_team1:
    print(f"{team2_name} is predicted to WIN!")
else:
    print("It's predicted to be a DRAW!")

if __name__ == "__main__":
    main()


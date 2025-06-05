from utils import map_team_name_to_code
from fetch_api import fetch_team_stats
from features import prepare_features
from predict import predict_score

def main():
    team1 = input("Enter Team 1 Name: ")
    team2 = input("Enter Team 2 Name: ")

    code1 = map_team_name_to_code(team1)
    code2 = map_team_name_to_code(team2)

    if not code1 or not code2:
        print("Invalid team names.")
        return

    stats1 = fetch_team_stats(code1)
    stats2 = fetch_team_stats(code2)

    df1 = prepare_features(stats1)
    df2 = prepare_features(stats2)

    score1, score2 = predict_score(df1, df2)

    print(f"Predicted goals - {team1.title()}: {score1:.1f}, {team2.title()}: {score2:.1f}")
    winner = team1 if score1 > score2 else team2
    print(f"{winner.title()} is predicted to WIN!")

if __name__ == "__main__":
    main()

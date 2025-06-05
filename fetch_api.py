import requests

def fetch_team_stats(team_code):
    url = f"https://api-web.nhle.com/v1/club-stats/{team_code}/now"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("stats", {})
    else:
        raise Exception(f"Failed to fetch data for team code {team_code}")

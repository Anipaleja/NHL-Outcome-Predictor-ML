import pandas as pd

def prepare_features(stats):
    # Choose features your model expects (46 from training)
    selected_keys = [
        "shots", "hits", "pim", "powerPlayOpportunities", "powerPlayGoals",
        "faceOffWinPercentage", "giveaways", "takeaways", "blocked", "timeOnIce",
        "assists", "goals"
        # Include the rest from your training list...
    ]
    data = {key: stats.get(key, 0) for key in selected_keys}
    return pd.DataFrame([data])

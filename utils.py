team_name_to_code = {
    "oilers": "EDM",
    "panthers": "FLA",
    "maple leafs": "TOR",
    "canadiens": "MTL",
    "bruins": "BOS",
    "rangers": "NYR",
    "blackhawks": "CHI",
    # ... add more as needed
}

def map_team_name_to_code(name):
    name = name.lower().strip()
    return team_name_to_code.get(name, None)

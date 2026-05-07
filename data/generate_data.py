"""
Premier League 2025/26 Season Dataset Generator
Realistic player stats based on current season form and squads.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(2026)

PLAYERS = [
    {"name": "Erling Haaland",       "team": "Man City",       "position": "ST",  "skill": 0.93},
    {"name": "Phil Foden",            "team": "Man City",       "position": "AM",  "skill": 0.86},
    {"name": "Bernardo Silva",        "team": "Man City",       "position": "AM",  "skill": 0.83},
    {"name": "Mohamed Salah",         "team": "Liverpool",      "position": "RW",  "skill": 0.89},
    {"name": "Darwin Nunez",          "team": "Liverpool",      "position": "ST",  "skill": 0.76},
    {"name": "Luis Diaz",             "team": "Liverpool",      "position": "LW",  "skill": 0.80},
    {"name": "Bukayo Saka",           "team": "Arsenal",        "position": "RW",  "skill": 0.87},
    {"name": "Gabriel Martinelli",    "team": "Arsenal",        "position": "LW",  "skill": 0.78},
    {"name": "Kai Havertz",           "team": "Arsenal",        "position": "ST",  "skill": 0.79},
    {"name": "Cole Palmer",           "team": "Chelsea",        "position": "AM",  "skill": 0.88},
    {"name": "Nicolas Jackson",       "team": "Chelsea",        "position": "ST",  "skill": 0.74},
    {"name": "Pedro Neto",            "team": "Chelsea",        "position": "RW",  "skill": 0.76},
    {"name": "Son Heung-min",         "team": "Spurs",          "position": "LW",  "skill": 0.81},
    {"name": "Dominic Solanke",       "team": "Spurs",          "position": "ST",  "skill": 0.74},
    {"name": "Brennan Johnson",       "team": "Spurs",          "position": "RW",  "skill": 0.73},
    {"name": "Marcus Rashford",       "team": "Man Utd",        "position": "LW",  "skill": 0.76},
    {"name": "Rasmus Hojlund",        "team": "Man Utd",        "position": "ST",  "skill": 0.74},
    {"name": "Bruno Fernandes",       "team": "Man Utd",        "position": "AM",  "skill": 0.82},
    {"name": "Ollie Watkins",         "team": "Aston Villa",    "position": "ST",  "skill": 0.82},
    {"name": "Leon Bailey",           "team": "Aston Villa",    "position": "LW",  "skill": 0.74},
    {"name": "Morgan Rogers",         "team": "Aston Villa",    "position": "AM",  "skill": 0.72},
    {"name": "Alexander Isak",        "team": "Newcastle",      "position": "ST",  "skill": 0.85},
    {"name": "Anthony Gordon",        "team": "Newcastle",      "position": "LW",  "skill": 0.78},
    {"name": "Jarrod Bowen",          "team": "West Ham",       "position": "RW",  "skill": 0.75},
    {"name": "Joao Pedro",            "team": "Brighton",       "position": "ST",  "skill": 0.76},
    {"name": "Bryan Mbeumo",          "team": "Brentford",      "position": "RW",  "skill": 0.78},
    {"name": "Yoane Wissa",           "team": "Brentford",      "position": "ST",  "skill": 0.74},
    {"name": "Raul Jimenez",          "team": "Fulham",         "position": "ST",  "skill": 0.71},
    {"name": "Matheus Cunha",         "team": "Wolves",         "position": "ST",  "skill": 0.76},
    {"name": "Eberechi Eze",          "team": "Crystal Palace", "position": "AM",  "skill": 0.79},
    {"name": "Jean-Philippe Mateta",  "team": "Crystal Palace", "position": "ST",  "skill": 0.74},
    {"name": "Dominic Calvert-Lewin", "team": "Everton",        "position": "ST",  "skill": 0.71},
    {"name": "Chris Wood",            "team": "Nott'm Forest",  "position": "ST",  "skill": 0.72},
    {"name": "Evanilson",             "team": "Bournemouth",    "position": "ST",  "skill": 0.72},
    {"name": "Liam Delap",            "team": "Ipswich",        "position": "ST",  "skill": 0.70},
    {"name": "Jamie Vardy",           "team": "Leicester",      "position": "ST",  "skill": 0.69},
]

UNIQUE_TEAMS = list(dict.fromkeys(p["team"] for p in PLAYERS))

OPPONENT_STRENGTH = {
    "Man City": 0.91, "Liverpool": 0.89, "Arsenal": 0.87, "Chelsea": 0.83,
    "Aston Villa": 0.80, "Newcastle": 0.79, "Man Utd": 0.76, "Spurs": 0.75,
    "Brighton": 0.72, "Fulham": 0.70, "Brentford": 0.69, "West Ham": 0.68,
    "Crystal Palace": 0.67, "Wolves": 0.66, "Everton": 0.65, "Bournemouth": 0.64,
    "Nott'm Forest": 0.66, "Leicester": 0.60, "Ipswich": 0.58,
}


def generate_match_records(n_records: int = 3000) -> pd.DataFrame:
    records = []
    for _ in range(n_records):
        player = np.random.choice(PLAYERS)
        opponent = np.random.choice([t for t in UNIQUE_TEAMS if t != player["team"]])
        opp_strength = OPPONENT_STRENGTH.get(opponent, 0.65)
        is_home = np.random.choice([0, 1])

        shot_mean = {"ST": 4.2, "RW": 3.0, "LW": 3.0, "AM": 2.2}.get(player["position"], 2.0)
        shots = max(0, int(np.random.normal(shot_mean, 1.6)))
        shots_on_target = max(0, min(shots, int(np.random.normal(shots * 0.42, 0.7))))
        xg = round(np.clip(np.random.normal(shots_on_target * 0.30, 0.12), 0, 3.5), 2)
        minutes_played = int(np.clip(np.random.normal(76, 14), 20, 90))
        key_passes = max(0, int(np.random.normal(1.8, 1.3)))
        dribbles = max(0, int(np.random.normal(2.1, 1.4)))
        touches_in_box = max(0, int(np.random.normal(4.5, 2.1)))
        form_last_5 = round(np.clip(np.random.normal(player["skill"] * 2.8, 0.9), 0, 5), 1)
        position_encoded = {"ST": 3, "RW": 2, "LW": 2, "AM": 1, "CM": 0}.get(player["position"], 0)

        base_prob = (
            player["skill"] * 0.28
            + (shots_on_target / max(shots + 1, 1)) * 0.22
            + xg * 0.20
            + (1 - opp_strength) * 0.12
            + is_home * 0.05
            + (touches_in_box / 12) * 0.07
            + (form_last_5 / 5) * 0.06
        )
        base_prob = np.clip(base_prob + np.random.normal(0, 0.05), 0.02, 0.95)
        scored = int(np.random.random() < base_prob)

        records.append({
            "player_name": player["name"], "team": player["team"],
            "opponent": opponent, "season": "2025/26",
            "position": player["position"], "position_encoded": position_encoded,
            "is_home": is_home, "minutes_played": minutes_played,
            "shots": shots, "shots_on_target": shots_on_target, "xg": xg,
            "key_passes": key_passes, "dribbles_completed": dribbles,
            "touches_in_box": touches_in_box, "form_last_5_goals": form_last_5,
            "opponent_strength": opp_strength, "player_skill": player["skill"],
            "scored": scored,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_match_records(3000)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/pl_player_match_data.csv", index=False)
    print(f"✅ 2025/26 Season Dataset Generated")
    print(f"   Records : {len(df)}")
    print(f"   Players : {df['player_name'].nunique()}")
    print(f"   Teams   : {df['team'].nunique()}")
    print(f"   Goal rate: {df['scored'].mean():.2%}")
    print(f"\nTop scorers (by goal probability):")
    print(df.groupby("player_name")["scored"].mean().sort_values(ascending=False).head(10).to_string())

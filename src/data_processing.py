import pandas as pd
import numpy as np
from src.add_features import add_features, duplicate_entries

KEEP_COLUMNS = [
    "winner_id", "loser_id", "tourney_level", "points_diff", "rank_diff", 
    "age_diff", "h2h_diff", "best_of", "surface", "round", "win_loss"
]

KEY_FEATURES = [
    'surface', 'winner_rank', 'loser_rank',
    'winner_rank_points', 'loser_rank_points'
]

def load_data(file_list, data_dir="data"):
    # Loads tennis match data from a list of CSV files and combines them into one DataFrame.
    dfs = []

    for filename in file_list:
        path = f"{data_dir}/{filename}"
        print(f"Loading {filename}...")
        df = pd.read_csv(path)
        dfs.append(df)

    # Combine indexing for rows into one big data frame
    combined = pd.concat(dfs, ignore_index=True) 

    # Output final data information
    print(f"Loaded {combined.shape[0]} matches from {len(file_list)} files")

    return combined

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Adds new columns of data and prepares previous columns for processing and features

    # Convert tourney_date to datetime
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

    # Encode round labels to numeric
    round_map = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF': 5, 'SF': 6, 'F': 7, 'RR': 3, 'BR': 6
    }
    df['round'] = (
        df['round']
        .map(round_map)
        .fillna(0)
        .astype(int)
    )

    # Encodes categorical features
    df['surface'] = df['surface'].map({'Hard': 0, 'Clay': 1, 'Grass': 2})

    tourney_level_map = {'D': 1, 'A': 2, 'M': 3, 'F': 4, 'O': 5, 'G': 6}
    df['tourney_level'] = (
        df['tourney_level']
        .map(tourney_level_map)
        .fillna(0)
        .astype(int)
    )

    return df

def process_data(df: pd.DataFrame) -> pd.DataFrame:

    # Remove rows missing key features
    df = df.dropna(subset=KEY_FEATURES)

    # Add features
    df = add_features(df)

    # Keep key columns
    df = df[KEEP_COLUMNS]

    # Duplicate entries
    df = duplicate_entries(df)

    return df
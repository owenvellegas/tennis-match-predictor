import pandas as pd

def load_data(file_list, data_dir="data"):

    # Loads ATP tennis match data from a list of CSV files and combines them into one DataFrame.
    dfs = []

    for filename in file_list:
        path = f"{data_dir}/{filename}"
        print(f"Loading {filename}...")
        df = pd.read_csv(path)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True) # Combine indexing for rows into one big data frame
    print(f"Loaded {combined.shape[0]} matches from {len(file_list)} files") # Output final data information

    return combined

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    keep_cols = [
        "tourney_name", "surface", "tourney_date", 
        "tourney_level", "best_of", "round",
        "winner_name", "winner_rank", "winner_ht", "winner_age",
        "loser_name", "loser_rank", "loser_ht", "loser_age"
    ]

    df = df[keep_cols]
    df = df.dropna(subset=["winner_name", "loser_name", "tourney_date"])

    # Convert tourney_date to datetime
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

    # Coerce numeric columns
    numeric_cols_to_coerce = [
        "winner_rank", "loser_rank", "winner_age", "loser_age", 
        "winner_ht", "loser_ht", "best_of", "round"
    ]
    for c in numeric_cols_to_coerce:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
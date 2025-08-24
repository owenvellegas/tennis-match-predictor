import pandas as pd
import numpy as np
from collections import defaultdict

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Adds engineered features to the tennis matches dataset.

    # Computes match importance feature
    df['match_importance'] = df['tourney_level'] * df['round'] * df['best_of']

    # Computes difference features
    df['rank_diff'] = -(df['winner_rank'] - df['loser_rank'])
    df['points_diff'] = (np.log1p(np.abs(df['winner_rank_points'] - df['loser_rank_points'])) * np.sign(df['winner_rank_points'] - df['loser_rank_points'])).round(4)
    df['age_diff'] = (df['winner_age'] - df['loser_age']).round(2)
    df['win_loss'] = 1

    df = add_h2h_stats(df)

    return df


def add_h2h_stats(df: pd.DataFrame) -> pd.DataFrame:

    # h2h[winner][loser] = (winner's wins vs loser) - (loser's wins vs winner)
    h2h = defaultdict(lambda: defaultdict(int))
    h2h_diffs = []

    for _, row in df.iterrows():
        winner = row['winner_id']      # winner
        loser = row['loser_id']  # loser

        # h2h diff from winner's perspective
        diff = h2h[winner][loser]
        h2h_diffs.append(diff)

        # Update h2h map after this match
        h2h[winner][loser] += 1
        h2h[loser][winner] -= 1

    df['h2h_diff'] = h2h_diffs
    return df


def duplicate_entries(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Define column pairs to swap between player and opponent
    swap_pairs = [
        ('winner_id', 'loser_id'),
    ]

    # 2. Create a mirrored copy of the DataFrame
    mirrored = df.copy()

    # 3. Swap values for each pair of columns
    for col_a, col_b in swap_pairs:
        mirrored[[col_a, col_b]] = mirrored[[col_b, col_a]]

    # 4. Flip sign for difference columns in mirrored DataFrame
    diff_cols = ['rank_diff', 'points_diff', 'age_diff', 'h2h_diff']
    for col in diff_cols:
        if col in mirrored.columns:
            mirrored[col] = -mirrored[col]

    mirrored['win_loss'] = 1 - mirrored['win_loss']

    # 5. Interleave original and mirrored entries
    n = len(df)
    df.index = np.arange(0, 2 * n, 2)
    mirrored.index = np.arange(1, 2 * n, 2)

    # 6. Concatenate and reset index
    combined_df = pd.concat([df, mirrored]).sort_index().reset_index(drop=True)
    
    return combined_df
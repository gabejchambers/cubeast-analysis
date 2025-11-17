import pandas as pd


def load_data(path):
    """
    Load CSV exactly as-is with no column assumptions or conversions.
    """
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """
    Clean the DataFrame and return a cleaned copy.

    Behavior:
    - If the DataFrame has a 'dnf' column, drop any rows where its
      value is the literal string 'true' (case-insensitive).
    - If 'dnf' column is not present, return the DataFrame unchanged.
    """
    if df is None:
        return df

    # If there's no 'dnf' column, nothing to do
    if 'dnf' not in df.columns:
        return df

    # Convert values to string, compare lowercase to 'true' to be robust
    mask = df['dnf'].astype(str).str.lower() == 'true'

    # Keep rows where mask is False (i.e., not a DNF)
    cleaned_df = df.loc[~mask].copy()
    return cleaned_df

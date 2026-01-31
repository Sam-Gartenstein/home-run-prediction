from pybaseball import playerid_reverse_lookup
import pandas as pd

def add_batter_names(
    df: pd.DataFrame,
    id_col: str = "batter_id",
    key_type: str = "mlbam",
) -> pd.DataFrame:
    """
    Merge batter names into a Statcast DataFrame using player IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a batter ID column.
    id_col : str, default 'batter_id'
        Column in df containing batter IDs.
    key_type : str, default 'mlbam'
        The type of player ID to use for the lookup (e.g., 'mlbam').

    Returns
    -------
    pd.DataFrame
        DataFrame with batter name columns merged in.
    """
    # Get unique batter IDs
    batter_ids = df[id_col].unique()

    # Look up player names
    batter_names_df = playerid_reverse_lookup(batter_ids.tolist(), key_type=key_type)

    # Merge names into the DataFrame
    df = df.merge(batter_names_df, how="left", left_on=id_col, right_on=f"key_{key_type}")

    return df

def add_batter_full_name(
    df: pd.DataFrame,
    first_col: str = "name_first",
    last_col: str = "name_last",
    new_col: str = "batter_name",
) -> pd.DataFrame:
    """
    Capitalize first and last names and create a full batter name column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing player first and last name columns.
    first_col : str, default "name_first"
        Column containing first names.
    last_col : str, default "name_last"
        Column containing last names.
    new_col : str, default "batter_name"
        Name of the new full-name column.

    Returns
    -------
    pd.DataFrame
        DataFrame with capitalized first/last names and a new full name column.
    """
    df[first_col] = df[first_col].str.capitalize()
    df[last_col] = df[last_col].str.capitalize()
    df[new_col] = df[first_col] + " " + df[last_col]

    return df

### ADD prepare_barrels here

def get_latest_batter_pitcher_data(df, batters=None, pitchers=None, window=40):
    """
    Returns the latest batter and pitcher rows with current (non-shifted) rolling barrel rates.

    batter_df columns:
      game_date, home_team, away_team, batter_name, batter_id, rolling_batter_barrel_rate, batter_idx
    pitcher_df columns:
      game_date, home_team, away_team, pitcher_name, rolling_pitcher_barrel_rate, pitcher_idx, hand_num
    """
    df = df.copy()

    # Remove any old rolling columns to avoid duplicates (handles single/double underscore variants)
    df = df.drop(
        columns=[
            "rolling_batter_barrel_rate",
            "rolling_pitcher_barrel_rate",
            "rolling__batter_barrel_rate",
            "rolling__pitcher_barrel_rate",
        ],
        errors="ignore",
    )

    # Rolling barrel rate for batters (by batter_id)
    df = df.sort_values(["batter_id", "game_date"])
    df["rolling_batter_barrel_rate"] = (
        df.groupby("batter_id")["barrel"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    # Rolling barrel rate for pitchers (by pitcher_name)
    df = df.sort_values(["pitcher_name", "game_date"])
    df["rolling_pitcher_barrel_rate"] = (
        df.groupby("pitcher_name")["barrel"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    # Prepare empty outputs with required columns (in case batters/pitchers is None/empty)
    batter_df = pd.DataFrame(columns=[
        "game_date","home_team","away_team","batter_name","batter_id","rolling_batter_barrel_rate","batter_idx"
    ])
    pitcher_df = pd.DataFrame(columns=[
        "game_date","home_team","away_team","pitcher_name","rolling_pitcher_barrel_rate","pitcher_idx","hand_num"
    ])

    # Latest per batter (supports multiple names)
    if batters:
        if isinstance(batters, str):
            batters = [batters]
        tmp = (
            df[df["batter_name"].isin(batters)]
            .sort_values("game_date", ascending=False)
            .drop_duplicates(subset="batter_name", keep="first")
        )
        batter_df = tmp[[
            "game_date","home_team","away_team","batter_name","batter_id","rolling_batter_barrel_rate","batter_idx"
        ]].reset_index(drop=True)

    # Latest per pitcher (supports multiple names)
    if pitchers:
        if isinstance(pitchers, str):
            pitchers = [pitchers]
        tmp = (
            df[df["pitcher_name"].isin(pitchers)]
            .sort_values("game_date", ascending=False)
            .drop_duplicates(subset="pitcher_name", keep="first")
        )
        pitcher_df = tmp[[
            "game_date","home_team","away_team","pitcher_name","rolling_pitcher_barrel_rate","pitcher_idx","hand_num"
        ]].reset_index(drop=True)

    return batter_df, pitcher_df


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


def prepare_barrels(
    df: pd.DataFrame,
    barrel_col: str = "launch_speed_angle",
) -> pd.DataFrame:
    """
    Prepare an at-bat–level DataFrame by:
      1) Sorting by batter and game context,
      2) Keeping only the final pitch of each at-bat,
      3) Creating a 'barrel' indicator column (1 if barrel_col == 6, else 0).

    Parameters
    ----------
    df : pd.DataFrame
        Input Statcast DataFrame at the pitch level.
    barrel_col : str, default 'launch_speed_angle'
        Column used to determine if a pitch is a barrel.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per at-bat and a 'barrel' indicator.
    """
    # Sort for correct sequence
    df = df.sort_values(
        by=["batter_name", "game_date", "inning", "at_bat_number"]
    )

    # Keep only the final pitch of each at-bat
    df = (
        df.groupby(["game_date", "at_bat_number", "batter_name"], as_index=False)
          .tail(1)
    )

    # Create 'barrel' indicator
    df["barrel"] = (df[barrel_col] == 6).fillna(False).astype(int)

    return df

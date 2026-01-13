# -------------------- standard library --------------------
import re
import time
from io import StringIO

# -------------------- third-party --------------------
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from tqdm import tqdm


####################################### ESPN ########################################

def espn_doubleheaders_textparse(url: str) -> pd.DataFrame:
    """
    Parse an ESPN MLB doubleheaders page into a DataFrame with columns:
    DATE | TEAMS | GAME 1 | GAME 2

    Args:
        url (str): full ESPN URL, e.g. "https://www.espn.com/mlb/stats/doubleheaders/_/year/2025"
    """
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()

    # Pull visible text (works even if it's an ARIA/virtual table)
    soup = BeautifulSoup(resp.text, "lxml")
    txt = soup.get_text("\n", strip=True)

    # Regex: Month Day | Teams | Game1 | Game2
    pat = re.compile(
        r"([A-Z][a-z]+ \d{1,2})\s+(.+?)\s+([A-Z]{2,3},\s*\d+-\d+|Postponed)\s+([A-Z]{2,3},\s*\d+-\d+|Postponed)",
        flags=re.MULTILINE
    )

    rows = []
    for m in pat.finditer(txt):
        date, teams, g1, g2 = m.groups()
        rows.append({"DATE": date, "TEAMS": teams, "GAME 1": g1, "GAME 2": g2})

    if not rows:
        raise RuntimeError("Could not parse any doubleheader rows from ESPN text.")

    df = pd.DataFrame(rows, columns=["DATE", "TEAMS", "GAME 1", "GAME 2"])
    return df.reset_index(drop=True)

def normalize_doubleheader_dates(df: pd.DataFrame, year: int = 2025) -> pd.DataFrame:
    out = df.copy()
    out["DATE"] = (out["DATE"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
                   + f" {year}")
    out["DATE"] = pd.to_datetime(out["DATE"], format="%B %d %Y", errors="coerce").dt.strftime("%Y-%m-%d")
    return out

def drop_postponed_doubleheaders(df):
    """
    Remove rows from a double-header DataFrame where either GAME 1 or GAME 2
    is postponed.

    Assumes columns 'GAME 1' and 'GAME 2' exist.

    Parameters
    ----------
    df : pd.DataFrame
        Double-header DataFrame with columns 'GAME 1' and 'GAME 2'.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with postponed double headers removed.
    """
    # Convert to string in case of NaNs or non-string entries
    game1_postponed = df["GAME 1"].astype(str).str.contains("Postponed", case=False, na=False)
    game2_postponed = df["GAME 2"].astype(str).str.contains("Postponed", case=False, na=False)

    # Keep only rows where neither game is postponed
    mask_keep = ~(game1_postponed | game2_postponed)
    return df.loc[mask_keep].reset_index(drop=True)


####################################### Abbreviation ########################################

# Short-name → abbreviation map (your scheme)
team_to_abbr_short = {
    "Arizona": "AZ", "Athletics": "ATH", "Atlanta": "ATL", "Baltimore": "BAL",
    "Boston": "BOS", "Chicago Cubs": "CHC", "Chicago Sox": "CWS", "Cincinnati": "CIN",
    "Cleveland": "CLE", "Colorado": "COL", "Detroit": "DET", "Houston": "HOU",
    "Kansas City": "KC", "LA Angels": "LAA", "LA Dodgers": "LAD", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "NY Mets": "NYM", "NY Yankees": "NYY",
    "Philadelphia": "PHI", "Pittsburgh": "PIT", "San Diego": "SD", "San Francisco": "SF",
    "Seattle": "SEA", "St. Louis": "STL", "Tampa Bay": "TB", "Texas": "TEX",
    "Toronto": "TOR", "Washington": "WSH",
}

def _normalize_team_label(s: str) -> str:
    if s is None: return None
    s = str(s).strip()
    rules = {
        r"(?i)^chicago\s+white\s+sox$": "Chicago Sox",
        r"(?i)^chi(?:\.|cago)?\s*(white\s*)?sox$": "Chicago Sox",
        r"(?i)^chi(?:\.|cago)?\s*cubs$": "Chicago Cubs",
        r"(?i)^(los\s+angeles\s+angels|la\s+angels)$": "LA Angels",
        r"(?i)^(los\s+angeles\s+dodgers|la\s+dodgers)$": "LA Dodgers",
        r"(?i)^(new\s+york\s+yankees)$": "NY Yankees",
        r"(?i)^(new\s+york\s+mets)$": "NY Mets",
        r"(?i)^(cleveland\s+guardians)$": "Cleveland",
        r"(?i)^(arizona\s+diamondbacks)$": "Arizona",
        r"(?i)^(tampa\s+bay\s+rays)$": "Tampa Bay",
        r"(?i)^(oakland\s+athletics|a's|athletic[s]?)$": "Athletics",
        r"(?i)^(st\.?\s*louis\s*cardinals)$": "St. Louis",
        r"(?i)^(san\s+francisco\s+giants)$": "San Francisco",
        r"(?i)^(san\s+diego\s+padres)$": "San Diego",
        r"(?i)^(seattle\s+mariners)$": "Seattle",
        r"(?i)^(texas\s+rangers)$": "Texas",
        r"(?i)^(toronto\s+blue\s+jays)$": "Toronto",
        r"(?i)^(washington\s+nationals)$": "Washington",
        r"(?i)^(philadelphia\s+phillies)$": "Philadelphia",
        r"(?i)^(pittsburgh\s+pirates)$": "Pittsburgh",
        r"(?i)^(cincinnati\s+reds)$": "Cincinnati",
        r"(?i)^(colorado\s+rockies)$": "Colorado",
        r"(?i)^(detroit\s+tigers)$": "Detroit",
        r"(?i)^(milwaukee\s+brewers)$": "Milwaukee",
        r"(?i)^(minnesota\s+twins)$": "Minnesota",
        r"(?i)^(atlanta\s+braves)$": "Atlanta",
        r"(?i)^(baltimore\s+orioles)$": "Baltimore",
        r"(?i)^(boston\s+red\s+sox)$": "Boston",
        r"(?i)^(miami\s+marlins)$": "Miami",
    }
    for pat, repl in rules.items():
        if re.match(pat, s):
            return repl
    return s

def replace_away_home_with_abbr(df: pd.DataFrame, teams_col: str = "TEAMS") -> pd.DataFrame:
    """
    From a 'TEAMS' column like 'Away at Home', create away/home and REPLACE them
    with abbreviations using team_to_abbr_short. Returns a new DataFrame.
    """
    if teams_col not in df.columns:
        raise KeyError(f"'{teams_col}' column not found.")

    out = df.copy()

    # Parse '<away> at <home>' (case-insensitive, tolerant spacing)
    pairs = (out[teams_col].astype(str)
             .str.extract(r"^(.*?)\s+at\s+(.*)$", flags=re.IGNORECASE))
    away = pairs[0].str.strip().map(_normalize_team_label)
    home = pairs[1].str.strip().map(_normalize_team_label)

    # Map to abbreviations and REPLACE away_team/home_team with the abbrs
    out["away_team"] = away.map(team_to_abbr_short)
    out["home_team"] = home.map(team_to_abbr_short)

    return out

#########################################################################################


def expand_double_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a double-header DataFrame (with GAME 1 / GAME 2 columns) and return a long
    DataFrame with one row per game, sorted consistently by:
    DATE, away_team, home_team, game_number.

    Required columns:
      - DATE, TEAMS, away_team, home_team, GAME 1, GAME 2

    Output columns include:
      - DATE, TEAMS, away_team, home_team, game_label, raw_result, game_number
    """
    required_cols = ["DATE", "TEAMS", "away_team", "home_team", "GAME 1", "GAME 2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"expand_double_headers: missing required columns: {missing}")

    long_df = df.melt(
        id_vars=["DATE", "TEAMS", "away_team", "home_team"],
        value_vars=["GAME 1", "GAME 2"],
        var_name="game_label",
        value_name="raw_result",
    )

    # Ensure DATE is datetime (matches sort_doubleheaders behavior)
    long_df["DATE"] = pd.to_datetime(long_df["DATE"])

    # Extract game number: "GAME 1" -> 1, "GAME 2" -> 2
    long_df["game_number"] = long_df["game_label"].str.extract(r"(\d+)").astype(int)

    # Sort like sort_doubleheaders, but keep Game 1 before Game 2 within matchup/day
    long_df = (
        long_df.sort_values(["DATE", "away_team", "home_team", "game_number"])
              .reset_index(drop=True)
    )

    return long_df


def build_team_name_to_abbrev_map(df: pd.DataFrame, teams_col: str = "TEAMS") -> dict:
    out = df.copy()

    # split "Arizona at Washington" (or "Arizona vs Washington")
    parts = out[teams_col].astype(str).str.split(r"\s+(?:at|vs)\s+", expand=True)

    out["away_name"] = parts[0].str.strip()
    out["home_name"] = parts[1].str.strip()

    # build mapping from names -> abbreviations
    away_map = dict(zip(out["away_name"], out["away_team"]))
    home_map = dict(zip(out["home_name"], out["home_team"]))

    # merge (they should agree if a name appears in both roles)
    name_to_abbrev = {**away_map, **home_map}
    return name_to_abbrev


def add_target_team_codes(df: pd.DataFrame, season: int,
                          away_col: str = "away_team",
                          home_col: str = "home_team",
                          new_away_col: str = "away_team_code",
                          new_home_col: str = "home_team_code") -> pd.DataFrame:
    """
    Adds target team-code columns (BRef/Retrosheet-style) to a DH dataframe.
    Oakland caveat: uses ATH in 2025, OAK otherwise.
    """
    out = df.copy()

    mapping = ABBR_TO_TARGET_2025 if season == 2025 else ABBR_TO_TARGET_NON_2025

    out[new_away_col] = out[away_col].map(mapping)
    out[new_home_col] = out[home_col].map(mapping)

    # Optional safety check: surface any unmapped abbreviations
    unmapped = sorted(set(pd.concat([out[away_col], out[home_col]])).difference(mapping.keys()))
    if unmapped:
        raise ValueError(f"Unmapped team abbreviations found for season {season}: {unmapped}")

    return out

####################################### Web Scraping Baseball Reference ########################################


def add_bref_boxscore_urls(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ensure datetime
    out["DATE"] = pd.to_datetime(out["DATE"])

    # create YYYYMMDD string
    out["bref_date"] = out["DATE"].dt.strftime("%Y%m%d")

    # ensure game_number is 1/2 as a string
    out["bref_game_number"] = out["game_number"].astype(int).astype(str)

    # build url
    out["bref_url"] = (
        "https://www.baseball-reference.com/boxes/"
        + out["home_team_code"] + "/"
        + out["home_team_code"] + out["bref_date"] + out["bref_game_number"]
        + ".shtml"
    )

    return out


# ---------- parsing helpers ----------

def table_to_df_from_comments(soup, table_id: str) -> pd.DataFrame:
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        commented = BeautifulSoup(c, "html.parser")
        table = commented.find("table", {"id": table_id})
        if table is not None:
            return pd.read_html(StringIO(str(table)))[0]
    raise ValueError(f"Table id '{table_id}' not found in comments.")

def fix_mojibake(s: str) -> str:
    try:
        return s.encode("latin1").decode("utf-8")
    except UnicodeError:
        return s

def clean_pitcher_name(raw: str) -> str:
    # "Zack Wheeler, W (1-1)" -> "Zack Wheeler"
    raw = fix_mojibake(raw)
    return re.sub(r",\s*[A-Z]{1,3}\s*\([^)]*\)\s*$", "", raw).strip()

def pitchers_who_pitched(pitch_df: pd.DataFrame) -> list[str]:
    name_col = pitch_df.columns[0]
    s = pitch_df[name_col].astype(str).str.strip()
    s = s[~s.str.contains("Team Totals", case=False, na=False)]
    s = s[s.ne("")]

    # Clean + preserve order, de-dupe just in case
    out, seen = [], set()
    for x in s.tolist():
        x = clean_pitcher_name(x)
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def bref_team_slug(team_name: str) -> str:
    # "St. Louis" -> "StLouis", "NY Yankees" -> "NYYankees"
    return re.sub(r"[^A-Za-z0-9]", "", str(team_name))

# ---------- BRef fetch + extraction ----------

'''
FIGURE OUT WHERE TO PUT THIS
'''
# Abbreviations to Target
ABBR_TO_TARGET_2025 = {
    "AZ":  "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN",   # Cubs
    "CWS": "CHA",   # White Sox
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC":  "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "ATH": "ATH",   # 2025 ONLY
    "PHI": "PHI",
    "PIT": "PIT",
    "SD":  "SDN",
    "SF":  "SFN",
    "SEA": "SEA",
    "STL": "SLN",
    "TB":  "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WAS",
}

# For non-2025 seasons, Oakland should be OAK
ABBR_TO_TARGET_NON_2025 = ABBR_TO_TARGET_2025.copy()
ABBR_TO_TARGET_NON_2025["ATH"] = "OAK"


def build_bref_boxscore_url(home_team_code: str, date, game_number: int | str) -> str:
    """
    home_team_code like 'BOS', date like '2025-04-06' or Timestamp, game_number 1/2
    """
    dt = pd.to_datetime(date)
    yyyymmdd = dt.strftime("%Y%m%d")
    g = str(int(game_number))
    return f"https://www.baseball-reference.com/boxes/{home_team_code}/{home_team_code}{yyyymmdd}{g}.shtml"

def fetch_soup(url: str, session=None, sleep_s: float = 1.5) -> BeautifulSoup:
    sess = session or requests.Session()
    headers = {"User-Agent": "personal-research (contact: you@example.com)"}
    resp = sess.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(sleep_s)  # polite
    return BeautifulSoup(resp.text, "html.parser")

def get_pitching_table_ids_from_row(row) -> tuple[str, str]:
    """
    Returns (away_pitching_id, home_pitching_id) using away_team_code/home_team_code
    (e.g., SLN/BOS) -> (StLouisCardinalspitching, BostonRedSoxpitching)
    """
    away_slug = ABBR_TO_BREF_SLUG[row["away_team_code"]]
    home_slug = ABBR_TO_BREF_SLUG[row["home_team_code"]]
    return f"{away_slug}pitching", f"{home_slug}pitching"

def get_pitchers_for_game(row, session=None) -> dict:
    """
    row must have: DATE, TEAMS, game_number, home_team_code, away_team_code
    Returns dict with date + url + home/away pitchers
    """
    url = build_bref_boxscore_url(
        home_team_code=row["home_team_code"],
        date=row["DATE"],
        game_number=row["game_number"],
    )

    soup = fetch_soup(url, session=session)

    away_pid, home_pid = get_pitching_table_ids_from_row(row)

    away_pit = table_to_df_from_comments(soup, away_pid)
    home_pit = table_to_df_from_comments(soup, home_pid)

    return {
        "date": pd.to_datetime(row["DATE"]).date(),
        "teams": row.get("TEAMS", None),

        # add abbreviations here
        "away_team": row["away_team"],
        "home_team": row["home_team"],

        "game_number": int(row["game_number"]),
        "bref_url": url,
        "away_pitchers": pitchers_who_pitched(away_pit),
        "home_pitchers": pitchers_who_pitched(home_pit),
    }

# ---------------------------------------------

# Figure out where to put this!
ABBR_TO_BREF_SLUG = {
    "ARI": "ArizonaDiamondbacks",
    "ATL": "AtlantaBraves",
    "BAL": "BaltimoreOrioles",
    "BOS": "BostonRedSox",
    "CHN": "ChicagoCubs",
    "CHA": "ChicagoWhiteSox",
    "CIN": "CincinnatiReds",
    "CLE": "ClevelandGuardians",
    "COL": "ColoradoRockies",
    "DET": "DetroitTigers",
    "HOU": "HoustonAstros",
    "KCA": "KansasCityRoyals",
    "ANA": "LosAngelesAngels",
    "LAN": "LosAngelesDodgers",
    "MIA": "MiamiMarlins",
    "MIL": "MilwaukeeBrewers",
    "MIN": "MinnesotaTwins",
    "NYN": "NewYorkMets",
    "NYA": "NewYorkYankees",
    "OAK": "OaklandAthletics",
    "ATH": "OaklandAthletics",   # table slug is still OaklandAthletics
    "PHI": "PhiladelphiaPhillies",
    "PIT": "PittsburghPirates",
    "SDN": "SanDiegoPadres",
    "SFN": "SanFranciscoGiants",
    "SEA": "SeattleMariners",
    "SLN": "StLouisCardinals",
    "TBA": "TampaBayRays",
    "TEX": "TexasRangers",
    "TOR": "TorontoBlueJays",
    "WAS": "WashingtonNationals",
}


def collect_pitchers_for_season(dh_df: pd.DataFrame, sleep_s: float = 1.5) -> pd.DataFrame:
    """
    Loops over a DH dataframe (one row per game) and collects BRef pitchers for each game.
    Returns a DataFrame with one row per game (pitchers are list columns).
    """
    out_rows = []
    sess = requests.Session()

    dh_iter = dh_df.reset_index(drop=True).iterrows()

    for i, row in tqdm(dh_iter, total=len(dh_df), desc="Scraping BRef boxscores"):
        try:
            res = get_pitchers_for_game(row, session=sess)
            out_rows.append(res)
        except Exception as e:
            out_rows.append({
                "date": pd.to_datetime(row["DATE"]).date(),
                "teams": row.get("TEAMS", None),
                "away_team": row.get("away_team", None),
                "home_team": row.get("home_team", None),
                "game_number": int(row.get("game_number", 0)) if pd.notna(row.get("game_number", None)) else None,
                "bref_url": build_bref_boxscore_url(row["home_team_code"], row["DATE"], row["game_number"]),
                "away_pitchers": None,
                "home_pitchers": None,
                "error": str(e),
            })

        time.sleep(sleep_s)

    return pd.DataFrame(out_rows)

####################################### Pitcher Procressing ########################################

def explode_pitchers_long(
    df: pd.DataFrame,
    away_col: str = "away_pitchers",
    home_col: str = "home_pitchers",
    pitcher_col: str = "pitcher",
    side_col: str = "side",
    side_labels: tuple[str, str] = ("away", "home"),
    drop_empty: bool = False,   # default: keep everything (use NA for missing)
    sort_output: bool = True,   # sort by date/game_number/side
    date_col: str = "date",
    game_number_col: str = "game_number",
) -> pd.DataFrame:
    """
    Convert game-level pitcher lists (or comma-separated strings) into long format:
    one pitcher per row + an indicator for away/home, with all other columns repeated.

    - Supports list/tuple cells (your current format) and comma-separated strings.
    - If drop_empty=False, missing pitcher lists become one row with pitcher=<NA>.
    - Sorting (optional): date, game_number, then side (away then home).
    """

    def split_pitchers(s):
        # If it's already a list/tuple, clean it directly (avoid pd.isna on lists)
        if isinstance(s, (list, tuple)):
            lst = [str(p).strip() for p in s if p is not None and str(p).strip() != ""]
            return lst if lst else ([] if drop_empty else [pd.NA])

        # Scalar missing
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return [] if drop_empty else [pd.NA]

        s_str = str(s).strip()
        if s_str == "":
            return [] if drop_empty else [pd.NA]

        # Fallback: comma-separated string
        lst = [p.strip() for p in s_str.split(",") if p.strip()]
        return lst if lst else ([] if drop_empty else [pd.NA])

    game_cols = [c for c in df.columns if c not in [away_col, home_col]]

    away_long = (
        df[game_cols + [away_col]]
        .assign(**{
            side_col: side_labels[0],
            pitcher_col: lambda d: d[away_col].map(split_pitchers)
        })
        .drop(columns=[away_col])
        .explode(pitcher_col)
    )

    home_long = (
        df[game_cols + [home_col]]
        .assign(**{
            side_col: side_labels[1],
            pitcher_col: lambda d: d[home_col].map(split_pitchers)
        })
        .drop(columns=[home_col])
        .explode(pitcher_col)
    )

    out = pd.concat([away_long, home_long], ignore_index=True)

    if drop_empty:
        out[pitcher_col] = out[pitcher_col].astype("string")
        out = out.dropna(subset=[pitcher_col])
        out = out[out[pitcher_col].str.strip() != ""]

    if sort_output:
        side_order = {side_labels[0]: 0, side_labels[1]: 1}
        out["_side_order"] = out[side_col].map(side_order)

        sort_cols = []
        if date_col in out.columns:
            sort_cols.append(date_col)
        if game_number_col in out.columns:
            sort_cols.append(game_number_col)
        sort_cols.append("_side_order")

        out = out.sort_values(sort_cols, kind="mergesort").drop(columns=["_side_order"])

    return out.reset_index(drop=True)


def first_last_to_last_first(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    if s == "" or s.lower() == "nan":
        return pd.NA
    parts = s.split()
    if len(parts) == 1:
        return s  # e.g., "Cher" (rare, but safe)
    last = parts[-1]
    first = " ".join(parts[:-1])
    return f"{last}, {first}"


def add_game_id(df):
    df = df.copy()

    # YYYYMMDD
    df["game_date_temp"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")

    # clean pitcher name:
    # - remove commas
    # - collapse whitespace
    # - replace spaces with underscores
    df["pitcher_id"] = (
        df["pitcher"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    df["game_id"] = (
        df["game_date_temp"]
        + "_"
        + df["away_team"].astype(str)
        + "@"
        + df["home_team"].astype(str)
        + "_"
        + df["pitcher_id"]
    )

    return df.drop(columns=["game_date_temp", "pitcher_id"])



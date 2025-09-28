from __future__ import annotations
import numpy as np
import arviz as az
import xarray as xr
import matplotlib.pyplot as plt

def _flatten(x: xr.DataArray) -> np.ndarray:
    """Flatten chain and draw dims to 1D."""
    if {"chain", "draw"}.issubset(set(x.dims)):
        x = x.stack(sample=("chain", "draw"))
    return np.asarray(x).ravel()

def _sel_idx(da: xr.DataArray, dim_name: str, idx: int) -> xr.DataArray:
    """Select index along a named dim if present; otherwise assume last dim is the index axis."""
    return da.sel({dim_name: idx}) if dim_name in da.dims else da.isel({da.dims[-1]: idx})

def posterior_hr_prob(
    trace: az.InferenceData,
    *,
    batter_idx: int,
    pitcher_idx: int,
    ballpark_idx: int | None = None,
    # Back-compat explicit features
    batter_feature_value: float | None = None,        # e.g., rolling batter barrel
    pitcher_feature_value: float | None = None,       # e.g., rolling pitcher barrel allowed
    batter_beta_name: str = "beta_batter_barrel",
    pitcher_beta_name: str = "beta_pitcher_barrel",
    # Arbitrary fixed effects
    feature_values: dict[str, float] | None = None,   # e.g., {"beta_prev_hr": 1, "beta_pitcher_hand": 1}
    feature_map: dict[str, str] | None = None,        # map feature col -> beta name if you pass feature col names
    hdi_prob: float = 0.95,
    return_array: bool = False,
    plot: bool = False,
    title: str | None = None,                         # used for plot title if provided
):
    """
    Compute posterior HR probability for a single PA; optionally plot posterior of the probability.
    """
    post = trace.posterior

    # Intercept & random effects
    alpha = _flatten(post["alpha"])
    b_eff = _flatten(_sel_idx(post["batter_effect"],  "batter",  batter_idx))
    p_eff = _flatten(_sel_idx(post["pitcher_effect"], "pitcher", pitcher_idx))
    if "ballpark_effect" in post and ballpark_idx is not None:
        bp_eff = _flatten(_sel_idx(post["ballpark_effect"], "ballpark", ballpark_idx))
    else:
        bp_eff = np.zeros_like(alpha)

    # Fixed-effect contribution
    beta_sum = np.zeros_like(alpha)

    # Explicit batter/pitcher features
    if batter_feature_value is not None and batter_beta_name in post:
        beta_b = _flatten(post[batter_beta_name])
        beta_sum += beta_b * float(batter_feature_value)

    if pitcher_feature_value is not None and pitcher_beta_name in post:
        beta_p = _flatten(post[pitcher_beta_name])
        beta_sum += beta_p * float(pitcher_feature_value)

    # Arbitrary fixed effects via dict
    if feature_values:
        for key, x in feature_values.items():
            param = feature_map[key] if (feature_map and key in feature_map) else key
            if param in post:
                beta_draws = _flatten(post[param])
                beta_sum += beta_draws * float(x)

    # Logit -> prob
    logits = alpha + b_eff + p_eff + bp_eff + beta_sum
    logits = np.clip(logits, -30, 30)
    probs = 1.0 / (1.0 + np.exp(-logits))

    if plot:
        if title:
            az.plot_posterior({title: probs}, hdi_prob=hdi_prob)
        else:
            az.plot_posterior(probs, hdi_prob=hdi_prob)
        plt.show()

    if return_array:
        return probs

    hdi = az.hdi(probs, hdi_prob=hdi_prob)
    hdi_vals = hdi.to_array().values if hasattr(hdi, "to_array") else np.asarray(hdi).ravel()
    return float(probs.mean()), float(hdi_vals[0]), float(hdi_vals[1])

def plot_hr_prob(
    trace,
    *,
    batter_df, pitcher_df,
    batter_name: str, pitcher_name: str,
    ballpark_idx: int,
    prev_hr_flag: int = 0,
    hdi_prob: float = 0.95,
    plot: bool = True,
):
    """
    Compute HR probability for (batter_name vs pitcher_name) using rows from
    batter_df/pitcher_df (Path B only).

    Expects these columns:
      batter_df:  game_date, home_team, away_team, batter_name, batter_id,
                  rolling_batter_barrel_rate, batter_idx
      pitcher_df: game_date, home_team, away_team, pitcher_name,
                  rolling_pitcher_barrel_rate, pitcher_idx, hand_num
    """
    # Latest per batter
    b_sel = (
        batter_df.loc[batter_df["batter_name"] == batter_name,
                      ["game_date","home_team","away_team","batter_name","batter_id",
                       "rolling_batter_barrel_rate","batter_idx"]]
        .sort_values("game_date", ascending=False)
    )
    if b_sel.empty:
        raise ValueError(f"No row found for batter '{batter_name}' in batter_df.")
    b_row = b_sel.iloc[0]

    # Latest per pitcher
    p_sel = (
        pitcher_df.loc[pitcher_df["pitcher_name"] == pitcher_name,
                       ["game_date","home_team","away_team","pitcher_name",
                        "rolling_pitcher_barrel_rate","pitcher_idx","hand_num"]]
        .sort_values("game_date", ascending=False)
    )
    if p_sel.empty:
        raise ValueError(f"No row found for pitcher '{pitcher_name}' in pitcher_df.")
    p_row = p_sel.iloc[0]

    # Scalars from rows
    batter_idx   = int(b_row["batter_idx"])
    pitcher_idx  = int(p_row["pitcher_idx"])
    batter_rate  = float(b_row["rolling_batter_barrel_rate"])
    pitcher_rate = float(p_row["rolling_pitcher_barrel_rate"])
    hand_num     = int(p_row["hand_num"])

    # Call your posterior helper with a title
    return posterior_hr_prob(
        trace=trace,
        batter_idx=batter_idx,
        pitcher_idx=pitcher_idx,
        ballpark_idx=ballpark_idx,
        batter_feature_value=batter_rate,
        pitcher_feature_value=pitcher_rate,
        feature_values={"beta_prev_hr": prev_hr_flag, "beta_pitcher_hand": hand_num},
        hdi_prob=hdi_prob,
        plot=plot,
        title=f"Home Run Probability for {batter_name}",
    )






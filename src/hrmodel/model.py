import numpy as np
import pandas as pd
import pymc as pm
from typing import Dict, Optional


def remove_fixed_effects(fixed_map: dict, priors: dict, to_drop):
    """
    Remove fixed effects by DataFrame column name(s) (e.g., 'prev_game_hr')
    and/or coefficient name(s) (e.g., 'beta_prev_hr').

    Returns:
        fixed_reduced (dict): filtered {col: beta_name}
        priors_reduced (dict): filtered {"beta_name": prior_dict}
    """
    if isinstance(to_drop, str):
        to_drop = [to_drop]

    # Separate beta names vs. column names
    drop_betas = {x for x in to_drop if x.startswith("beta_")}
    drop_cols  = set(to_drop) - drop_betas

    # Normalize: map betas -> cols and cols -> betas so both are pruned
    drop_cols  |= {col for col, beta in fixed_map.items() if beta in drop_betas}
    drop_betas |= {beta for col, beta in fixed_map.items() if col in drop_cols}

    fixed_reduced  = {col: beta for col, beta in fixed_map.items() if col not in drop_cols}
    priors_reduced = {beta: p   for beta, p   in priors.items()     if beta not in drop_betas}
    return fixed_reduced, priors_reduced


def build_and_sample_hr_model(
    df: pd.DataFrame,
    feature_map: Dict[str, str] | None = None,          # {df_col: coef_name}
    priors: Dict[str, Dict[str, float]] | None = None,  # fixed-effect priors: {"beta_name": {"mu":0,"sigma":1}}
    priors_global: Dict[str, Dict[str, float]] | None = None,  # overrides for alpha/sigmas
    standardize: bool = False,                           # z-score selected features
    idx_cols: Dict[str, str] | None = None,             # mapping of index cols
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
):
    """
    Build & sample a Bayesian hierarchical HR model with selectable fixed effects.

    Pure function: no logging or file I/O. Returns (model, idata).

    Parameters
    ----------
    df : DataFrame
        Must contain binary 'home_run' and index columns for batter/pitcher/ballpark.
    feature_map : dict or None
        Mapping {df_column_name: coefficient_name}. Only these columns are used as fixed effects.
    priors : dict or None
        Per-coefficient Normal priors, e.g. {"park_factor": {"mu":0.0, "sigma":0.5}}.
        Any coefficient not specified defaults to Normal(0, 1).
    priors_global : dict or None
        Overrides for global priors:
            alpha:          {"mu": ..., "sigma": ...}         # Normal
            sigma_batter:   {"sigma": ...}                     # HalfNormal
            sigma_pitcher:  {"sigma": ...}                     # HalfNormal
            sigma_ballpark: {"sigma": ...}                     # HalfNormal
        Omitted keys fall back to sensible defaults shown below.
    standardize : bool
        If True, z-score the fixed-effect columns in `feature_map` using df means/stds.
    idx_cols : dict or None
        Names for index columns in df, defaults to:
        {"batter": "batter_idx", "pitcher": "pitcher_idx", "ballpark": "ballpark_idx"}.
        These columns must be integer-coded from 0..N-1.
    draws, tune, chains, target_accept, random_seed : sampling controls.

    Returns
    -------
    model : pm.Model
    idata : arviz.InferenceData
    """
    # ---- indices / counts
    if idx_cols is None:
        idx_cols = {"batter": "batter_idx", "pitcher": "pitcher_idx", "ballpark": "ballpark_idx"}

    b_idx = df[idx_cols["batter"]].to_numpy(dtype=int)
    p_idx = df[idx_cols["pitcher"]].to_numpy(dtype=int)
    bp_idx = df[idx_cols["ballpark"]].to_numpy(dtype=int)

    num_batters   = int(np.max(b_idx)) + 1
    num_pitchers  = int(np.max(p_idx)) + 1
    num_ballparks = int(np.max(bp_idx)) + 1

    # ---- outcome
    y = df["home_run"].to_numpy()

    # ---- fixed-effects design
    feature_map = feature_map or {}
    X_cols = list(feature_map.keys())
    X = None
    if X_cols:
        X = df[X_cols].to_numpy(dtype=float)
        if standardize:
            means = X.mean(axis=0)
            stds = X.std(axis=0, ddof=0)
            stds_safe = np.where(stds == 0, 1.0, stds)
            X = (X - means) / stds_safe

    priors = priors or {}

    # ---- global priors: defaults + optional overrides
    GLOBAL_DEFAULTS = {
        "alpha":          {"mu": 0.0, "sigma": 1.0},  # Normal
        "sigma_batter":   {"sigma": 1.0},             # HalfNormal
        "sigma_pitcher":  {"sigma": 1.0},             # HalfNormal
        "sigma_ballpark": {"sigma": 1.0},             # HalfNormal
    }
    gpri = {**GLOBAL_DEFAULTS, **(priors_global or {})}

    # ---- model
    with pm.Model() as model:
        # intercept & group-level scales (non-centered)
        alpha = pm.Normal("alpha", mu=gpri["alpha"]["mu"], sigma=gpri["alpha"]["sigma"])

        sigma_batter   = pm.HalfNormal("sigma_batter",  sigma=gpri["sigma_batter"]["sigma"])
        sigma_pitcher  = pm.HalfNormal("sigma_pitcher", sigma=gpri["sigma_pitcher"]["sigma"])
        sigma_ballpark = pm.HalfNormal("sigma_ballpark",sigma=gpri["sigma_ballpark"]["sigma"])

        b_nc  = pm.Normal("batter_non_centered",   0, 1, shape=num_batters)
        p_nc  = pm.Normal("pitcher_non_centered",  0, 1, shape=num_pitchers)
        bp_nc = pm.Normal("ballpark_non_centered", 0, 1, shape=num_ballparks)

        b_eff  = pm.Deterministic("batter_effect",   b_nc  * sigma_batter)
        p_eff  = pm.Deterministic("pitcher_effect",  p_nc  * sigma_pitcher)
        bp_eff = pm.Deterministic("ballpark_effect", bp_nc * sigma_ballpark)

        # linear predictor
        logits = alpha + b_eff[b_idx] + p_eff[p_idx] + bp_eff[bp_idx]

        # fixed effects (Normal priors)
        if X is not None:
            for j, col in enumerate(X_cols):
                beta_name = feature_map[col]
                prior = priors.get(beta_name, {"mu": 0.0, "sigma": 1.0})
                beta = pm.Normal(beta_name, mu=prior["mu"], sigma=prior["sigma"])
                logits = logits + beta * X[:, j]

        # likelihood
        pm.Bernoulli("y_obs", logit_p=logits, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


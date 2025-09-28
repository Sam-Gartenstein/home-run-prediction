#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pymc as pm


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
    feature_map: dict[str, str] | None = None,   # {df_col: coef_name}
    priors: dict[str, dict] | None = None,       # {"beta_name": {"mu":0,"sigma":1}, ...}
    standardize: bool = False,                   # z-score selected features
    idx_cols: dict[str, str] | None = None,      # mapping of index cols
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
):
    """
    Build & sample a Bayesian hierarchical HR model with selectable fixed effects.
    """
    if idx_cols is None:
        idx_cols = {"batter": "batter_idx", "pitcher": "pitcher_idx", "ballpark": "ballpark_idx"}

    # Indices and counts
    b_idx = df[idx_cols["batter"]].to_numpy(dtype=int)
    p_idx = df[idx_cols["pitcher"]].to_numpy(dtype=int)
    bp_idx = df[idx_cols["ballpark"]].to_numpy(dtype=int)

    num_batters   = int(np.max(b_idx)) + 1
    num_pitchers  = int(np.max(p_idx)) + 1
    num_ballparks = int(np.max(bp_idx)) + 1

    # Outcome
    y = df["home_run"].to_numpy()

    # Fixed effects assembly
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

    with pm.Model() as model:
        # Intercept
        alpha = pm.Normal("alpha", 0, 1)

        # Random effects (non-centered)
        sigma_batter   = pm.HalfNormal("sigma_batter", 1)
        sigma_pitcher  = pm.HalfNormal("sigma_pitcher", 1)
        sigma_ballpark = pm.HalfNormal("sigma_ballpark", 1)

        b_nc  = pm.Normal("batter_non_centered",   0, 1, shape=num_batters)
        p_nc  = pm.Normal("pitcher_non_centered",  0, 1, shape=num_pitchers)
        bp_nc = pm.Normal("ballpark_non_centered", 0, 1, shape=num_ballparks)

        b_eff  = pm.Deterministic("batter_effect",   b_nc  * sigma_batter)
        p_eff  = pm.Deterministic("pitcher_effect",  p_nc  * sigma_pitcher)
        bp_eff = pm.Deterministic("ballpark_effect", bp_nc * sigma_ballpark)

        # Baseline linear predictor
        logits = alpha + b_eff[b_idx] + p_eff[p_idx] + bp_eff[bp_idx]

        # Add selected fixed effects
        if X is not None:
            for j, col in enumerate(X_cols):
                beta_name = feature_map[col]
                prior = priors.get(beta_name, {"mu": 0, "sigma": 1})
                beta = pm.Normal(beta_name, mu=prior["mu"], sigma=prior["sigma"])
                logits = logits + beta * X[:, j]

        # Likelihood
        pm.Bernoulli("y_obs", logit_p=logits, observed=y)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True}
        )

    return model, trace


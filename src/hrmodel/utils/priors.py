#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

# Central place for your default global priors
PRIORS_GLOBAL = {
    "alpha":          {"dist": "Normal",     "mu": 0.0, "sigma": 1.0},
    "sigma_batter":   {"dist": "HalfNormal", "sigma": 1.0},
    "sigma_pitcher":  {"dist": "HalfNormal", "sigma": 1.0},
    "sigma_ballpark": {"dist": "HalfNormal", "sigma": 1.0},
}

def make_prior_spec(feature_map: dict,
                    priors_fixed: dict | None = None,
                    priors_global: dict | None = None) -> dict:
    """Return a YAML/JSON-friendly spec of priors (global + fixed effects)."""
    priors_fixed = priors_fixed or {}
    priors_global = priors_global or PRIORS_GLOBAL
    fixed = {}
    for coef_name in feature_map.values():
        # default Normal(0,1), overridden by priors_fixed if provided
        fixed[coef_name] = {"dist": "Normal", "mu": 0.0, "sigma": 1.0}
        fixed[coef_name].update(priors_fixed.get(coef_name, {}))
    return {"global": priors_global, "fixed_effects": fixed}


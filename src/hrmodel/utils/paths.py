#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import re
from typing import Optional, Tuple
import arviz as az
import numpy as np

# ---------- internal helpers ----------

_RUN_PAT = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<idx>\d+)(?:-(?P<sfx>\d+))?$")

def _sorted_runs(dir_: Path):
    runs = []
    for p in Path(dir_).iterdir():
        if p.is_dir():
            m = _RUN_PAT.match(p.name)
            if m:
                runs.append((m.group("date"), int(m.group("idx")), int(m.group("sfx") or 0), p))
    runs.sort(key=lambda t: (t[0], t[1], t[2]))
    return [p for *_ignore, p in runs]

def _pick_idata_file(run_dir: Path, require_ppc: bool = True) -> Path:
    ppc = run_dir / "idata_ppc.nc"
    if ppc.exists():
        return ppc
    plain = run_dir / "idata.nc"
    if plain.exists() and not require_ppc:
        return plain
    raise FileNotFoundError(f"No idata_ppc.nc (or idata.nc) in {run_dir}")

def _resolve_run_dir(variant_dir: Path, run_id: Optional[str]) -> Path:
    """
    If run_id is provided:
      - exact match if folder exists
      - else treat as prefix (e.g., '2025-10-01' -> choose latest '2025-10-01_XX[-N]')
    If run_id is None:
      - choose globally latest by date/idx/suffix
    """
    variant_dir = Path(variant_dir)
    runs = _sorted_runs(variant_dir)
    if not runs:
        raise FileNotFoundError(f"No run folders under {variant_dir}")

    if run_id is None:
        return runs[-1]

    # exact folder name given?
    exact = variant_dir / run_id
    if exact.exists() and exact.is_dir():
        return exact

    # treat as prefix
    pref = str(run_id)
    candidates = [r for r in runs if r.name.startswith(pref)]
    if not candidates:
        raise FileNotFoundError(f"No runs matching prefix '{pref}' under {variant_dir}")
    return candidates[-1]

# ---------- public API ----------

def list_runs(variant: str, base_models_dir: Path = Path("artifacts/models")) -> list[str]:
    """List available run folder names for a variant (sorted ascending)."""
    return [p.name for p in _sorted_runs(base_models_dir / variant)]

def load_variant(variant: str,
                 run_id: Optional[str] = None,
                 base_models_dir: Path = Path("artifacts/models"),
                 var_name: str = "y_obs",
                 require_ppc: bool = True):
    """
    Load (idata, y_true, pred_prob) for a variant.
    - If run_id is None → pick latest run.
    - If run_id is 'YYYY-MM-DD' → pick latest from that date.
    - If run_id is 'YYYY-MM-DD_05' (or with -2 suffix) → use that folder.
    - require_ppc=True → prefer/require idata_ppc.nc (error if missing).
    """
    variant_dir = base_models_dir / variant
    run_dir = _resolve_run_dir(variant_dir, run_id)
    nc_path = _pick_idata_file(run_dir, require_ppc=require_ppc)

    idata = az.from_netcdf(nc_path)

    if var_name not in idata.observed_data:
        raise KeyError(f"'{var_name}' not in observed_data for {nc_path}")

    y_true = np.asarray(idata.observed_data[var_name].values, dtype=int).reshape(-1)

    if "posterior_predictive" not in idata.groups() or var_name not in idata.posterior_predictive:
        if require_ppc:
            raise ValueError(f"posterior_predictive['{var_name}'] not found in {nc_path}")
        pred_prob = None
    else:
        pred_prob = idata.posterior_predictive[var_name].mean(dim=("chain","draw")).values
        pred_prob = np.asarray(pred_prob, dtype=float).reshape(-1)

    return idata, y_true, pred_prob, run_dir


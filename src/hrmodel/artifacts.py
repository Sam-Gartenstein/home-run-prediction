
import os
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm

def sample_ppc_and_save(model,
                        idata,
                        spec_name,                 # e.g. "full", "noprev"
                        base_dir,                  # e.g. "/content/drive/.../artifacts"
                        run_id=None,               # reuse the same run_id across specs
                        subdir="models",           # will create base_dir/models/<spec>/<run_id>/
                        var_names=("y_obs",),
                        random_seed=42,
                        attrs=None,
                        filename="idata.nc"):      # file saved inside the run dir
    """
    Runs posterior predictive, appends to `idata`, saves to:
      <base_dir>/<subdir>/<spec_name>/<run_id>/<filename>

    Returns:
      updated_idata, saved_file_path, run_dir
    """
    # 1) Posterior predictive
    updated = pm.sample_posterior_predictive(
        trace=idata,
        model=model,
        var_names=list(var_names),
        extend_inferencedata=True,
        random_seed=random_seed,
    )

    # 2) Tag optional attrs
    if attrs:
        for k, v in attrs.items():
            updated.posterior.attrs[k] = str(v)

    # 3) Build run dir internally
    if run_id is None:
        run_id = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, subdir, spec_name, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 4) Save file
    path = os.path.join(run_dir, filename)
    az.to_netcdf(updated, path)

    return updated, path, run_dir

def load_idata_and_preds(nc_path, var_name="y_obs"):
    """
    Load an ArviZ NetCDF and return:
      (idata, y_true: 1D int array, pred_prob: 1D float array)
    where pred_prob = mean over chain/draw of posterior_predictive[var_name].
    """
    idata = az.from_netcdf(Path(nc_path))

    if var_name not in idata.observed_data:
        raise KeyError(f"'{var_name}' not found in observed_data.")

    # Observed labels
    y = idata.observed_data[var_name].values
    y_true = np.asarray(y, dtype=int).reshape(-1)

    # Posterior predictive mean P(y=1)
    pp = idata.posterior_predictive[var_name]
    reduce_dims = [d for d in ("chain", "draw") if d in pp.dims]
    p = pp.mean(dim=reduce_dims).values
    pred_prob = np.asarray(p, dtype=float).reshape(-1)

    return idata, y_true, pred_prob


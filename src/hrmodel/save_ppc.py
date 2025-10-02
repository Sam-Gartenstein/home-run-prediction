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


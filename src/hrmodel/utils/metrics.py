from pathlib import Path
import re
import pandas as pd
import arviz as az
from typing import Optional, Dict, Any
from .logging_utils import RunLogger

_RUN_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}(?:_\d+)?(?:-\d+)?$")

def _latest_logs_run_dir(base: Path = Path("artifacts/logs")) -> Path:
    runs = [d for d in base.iterdir() if d.is_dir() and _RUN_PAT.match(d.name)]
    if not runs:
        raise FileNotFoundError(f"No run folders under {base}")
    return sorted(runs)[-1]

def log_and_save_posterior_summary(
    runlog: Optional[RunLogger],
    summary_df: pd.DataFrame,
    variant: str,                         # "full" or "noprev"
    base_logs: Path = Path("artifacts/logs"),
    csv_name: str = "posterior_summary.csv",
) -> Path:
    """
    Save an already-computed posterior summary DataFrame to:
      artifacts/logs/<latest_run>/<variant>/<csv_name>

    If runlog is None, it will auto-attach to the latest logs run.
    """
    # attach or infer the run folder
    if runlog is None:
        run_dir = _latest_logs_run_dir(base_logs)
        runlog = RunLogger.open(run_dir)

    # ensure variant folder exists
    variant_dir = runlog.dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    # write CSV
    csv_path = variant_dir / csv_name
    summary_df.to_csv(csv_path, index=True)
    return csv_path

def log_and_save_model_comparison(
    runlog: Optional[RunLogger],
    *,
    comp_df: Optional[pd.DataFrame] = None,          # if you already computed az.compare(...)
    comparables: Optional[Dict[str, Any]] = None,    # e.g. {"noprev": loo_n, "full": loo_f}
    ic: str = "loo",
    base_logs: Path = Path("artifacts/logs"),
    csv_name: Optional[str] = None,
) -> tuple[Path, pd.DataFrame]:
    """
    Save a model comparison table under the latest (or given) run.
    - If `comp_df` is provided, it is used directly.
    - Else, computes `az.compare(comparables, ic=ic)`.
    - If `runlog` is None, attaches to the latest run under artifacts/logs.

    Returns: (csv_path, comparison_dataframe)
    """
    # find latest run if none given
    if runlog is None:
        run_dir = _latest_logs_run_dir(base_logs)
        runlog = RunLogger.open(run_dir)

    # Make /comparison folder
    cmp_dir = runlog.dir / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    # Compute or use provided comparison
    if comp_df is None:
        if comparables is None:
            raise ValueError("Provide either `comp_df` or `comparables` (+ ic).")
        comp_df = az.compare(comparables, ic=ic)

    # Choose filename
    if csv_name is None:
        csv_name = f"comparison_{ic}.csv"

    csv_path = cmp_dir / csv_name
    comp_df.to_csv(csv_path, index=True)

    return csv_path, comp_df

def log_and_save_threshold_tables(
    runlog: Optional[RunLogger],
    *,
    sweep_df: pd.DataFrame,
    choices_df: Optional[pd.DataFrame] = None,
    # NEW: where to save
    variant: Optional[str] = None,              # e.g. "noprev" or "full"
    target_dir: Optional[Path] = None,          # explicit folder wins over variant
    base_logs: Path = Path("artifacts/logs"),
    # filenames
    sweep_csv: str = "threshold_sweep.csv",
    choices_csv: str = "threshold_choices.csv",
) -> Dict[str, Path]:
    """
    Save threshold sweep and (optionally) choices as CSVs into a variant folder.

    Destination resolution (in order):
      1) if target_dir is given → use it
      2) else require variant; save under <run_dir>/<variant>, where:
         - if runlog is None, the latest logs run is opened automatically
         - else use runlog.dir

    Returns dict with keys present: {"sweep": Path, "choices": Path}
    """
    # Resolve destination directory
    if target_dir is None:
        if variant is None:
            raise ValueError("Provide either `target_dir` or `variant`.")
        if runlog is None:
            run_dir = _latest_logs_run_dir(base_logs)
            runlog = RunLogger.open(run_dir)
        target_dir = runlog.dir / variant

    target_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Path] = {}

    # Save CSVs
    sweep_path = target_dir / sweep_csv
    sweep_df.to_csv(sweep_path, index=True)
    out["sweep"] = sweep_path

    if choices_df is not None:
        choices_path = target_dir / choices_csv
        choices_df.to_csv(choices_path, index=True)
        out["choices"] = choices_path

    return out

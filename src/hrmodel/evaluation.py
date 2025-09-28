import numpy as np
import pandas as pd

def sweep_thresholds(y_true, probs, thresholds):
    """
    Sweep thresholds and compute precision, recall, alerts for each model.
    Returns a DataFrame indexed by (model, threshold).
    """
    y_true = np.asarray(y_true, int).ravel()
    items = list(probs.items()) if isinstance(probs, dict) else [("model_1", probs)]
    rows = []

    for name, p in items:
        p = np.asarray(p, float).ravel()
        for th in thresholds:
            y_pred = (p >= th).astype(int)
            tp = int(((y_pred==1)&(y_true==1)).sum())
            fp = int(((y_pred==1)&(y_true==0)).sum())
            fn = int(((y_pred==0)&(y_true==1)).sum())
            precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
            recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
            rows.append({
                "model": name, "threshold": float(th),
                "precision": precision, "recall": recall, "alerts": int(tp+fp),
                "tp": tp, "fp": fp, "fn": fn
            })

    return (pd.DataFrame(rows)
              .sort_values(["model","threshold"])
              .set_index(["model","threshold"]))

def pick_by_precision_floor(sweep_df, floor):
    """
    From a sweep DataFrame (indexed by model,threshold), pick the threshold with
    max recall among rows where precision ≥ floor. Falls back to max precision if none meet the floor.
    Returns threshold, precision, recall, alerts, tp, fp, fn, and whether the floor was met.
    """
    out = []
    for model, g in sweep_df.groupby(level=0):
        g = g.reset_index()
        ok = g["precision"] >= float(floor)
        if ok.any():
            best = g.loc[ok].sort_values(["recall","precision","threshold"],
                                         ascending=[False,False,True]).iloc[0]
            met = True
        else:
            best = g.sort_values(["precision","recall","threshold"],
                                 ascending=[False,False,True]).iloc[0]
            met = False

        out.append({
            "model": model,
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "alerts": int(best["alerts"]),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "floor": float(floor),
            "met_floor": met,
        })

    return pd.DataFrame(out).set_index("model")


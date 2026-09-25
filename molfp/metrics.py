"""Metrics shared by the GCN and the RF baseline, so both are scored identically."""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def masked_roc_auc(y_true, y_prob):
    """Per-task ROC-AUC over labelled rows only (NaN = missing, invariant 7).

    A task whose labelled rows are all one class has no defined AUC: reported as NaN and
    excluded from the mean.
    """
    per_task = []
    for t in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, t])
        yt = y_true[m, t]
        ok = m.any() and 0 < yt.sum() < len(yt)
        per_task.append(float(roc_auc_score(yt, y_prob[m, t])) if ok else float("nan"))
    return {"roc_auc": float(np.nanmean(per_task)), "roc_auc_per_task": per_task}

"""Evaluation metrics for encoding models."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def pearson_per_column(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Pearson correlation between matching columns of two ``(n, p)`` arrays.

    Columns with zero variance in either input yield NaN.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    true_centered = y_true - y_true.mean(axis=0, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=0, keepdims=True)
    true_norm = np.linalg.norm(true_centered, axis=0)
    pred_norm = np.linalg.norm(pred_centered, axis=0)
    denominator = true_norm * pred_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (true_centered * pred_centered).sum(axis=0) / denominator
    r[denominator == 0] = np.nan
    return np.asarray(r, dtype=np.float64)


def summarize(r: npt.NDArray[np.float64]) -> dict[str, float]:
    """Summary statistics over per-parcel correlations (NaNs excluded)."""
    valid = r[~np.isnan(r)]
    if valid.size == 0:
        raise ValueError("No valid (non-NaN) correlations to summarize")
    return {
        "n_parcels": float(r.size),
        "n_valid": float(valid.size),
        "mean_r": float(valid.mean()),
        "median_r": float(np.median(valid)),
        "max_r": float(valid.max()),
        "p95_r": float(np.percentile(valid, 95)),
        "frac_positive": float((valid > 0).mean()),
    }

"""Stimulus feature construction aligned to fMRI TRs."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def word_rate(onsets: Sequence[float], n_trs: int, tr_seconds: float) -> npt.NDArray[np.float64]:
    """Count words with onset inside each TR bin.

    Parameters
    ----------
    onsets:
        Word onset times in seconds from run start.
    n_trs:
        Number of TRs in the run; defines the output length.
    tr_seconds:
        Repetition time in seconds.

    Returns
    -------
    Array of shape ``(n_trs,)`` with the word count per TR.
    """
    counts = np.zeros(n_trs, dtype=np.float64)
    if len(onsets) == 0:
        return counts
    idx = np.floor(np.asarray(onsets, dtype=np.float64) / tr_seconds).astype(np.int64)
    valid = (idx >= 0) & (idx < n_trs)
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.debug("Dropping %d word onsets outside the scanned interval", n_dropped)
    np.add.at(counts, idx[valid], 1.0)
    return counts


def fir_lags(features: npt.NDArray[np.float64], lags: Sequence[int]) -> npt.NDArray[np.float64]:
    """Build a finite-impulse-response design by time-shifting features.

    The BOLD response lags the stimulus by several seconds, so the feature
    value at TR ``t - lag`` is used to predict brain activity at TR ``t``.
    Rows before a lag's reach are zero-padded.

    Parameters
    ----------
    features:
        Array of shape ``(n_trs,)`` or ``(n_trs, k)``.
    lags:
        Non-negative TR shifts, e.g. ``(2, 3, 4, 5)`` is roughly 3-7.5 s at TR=1.49 s.

    Returns
    -------
    Array of shape ``(n_trs, k * len(lags))``.
    """
    if len(lags) == 0:
        raise ValueError("lags must be non-empty")
    x = features.reshape(features.shape[0], -1)
    columns: list[npt.NDArray[np.float64]] = []
    for lag in lags:
        if lag < 0:
            raise ValueError(f"lags must be non-negative, got {lag}")
        shifted = np.zeros_like(x)
        if lag == 0:
            shifted[:] = x
        else:
            shifted[lag:] = x[:-lag]
        columns.append(shifted)
    return np.concatenate(columns, axis=1)


def zscore(x: npt.NDArray[np.float64], eps: float = 1e-8) -> npt.NDArray[np.float64]:
    """Z-score each column; constant columns become zeros instead of NaN.

    Applied per run, so no statistics leak between training and test runs.
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return np.asarray((x - mean) / np.maximum(std, eps), dtype=np.float64)

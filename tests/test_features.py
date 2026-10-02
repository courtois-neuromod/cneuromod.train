"""Tests for stimulus feature construction."""

import numpy as np
import pytest

from cneuromod.train.features import fir_lags, word_rate, zscore


def test_word_rate_bins_onsets() -> None:
    # TR = 2 s; bins are [0,2), [2,4), [4,6)
    rate = word_rate([0.0, 1.9, 2.0, 5.99], n_trs=3, tr_seconds=2.0)
    assert rate.tolist() == [2.0, 1.0, 1.0]


def test_word_rate_drops_out_of_range_onsets() -> None:
    rate = word_rate([-0.5, 1.0, 100.0], n_trs=2, tr_seconds=2.0)
    assert rate.tolist() == [1.0, 0.0]


def test_word_rate_empty() -> None:
    assert word_rate([], n_trs=4, tr_seconds=1.5).tolist() == [0.0] * 4


def test_fir_lags_shifts_forward_in_time() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    design = fir_lags(x, lags=(0, 2))
    assert design.shape == (4, 2)
    assert design[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]  # lag 0
    assert design[:, 1].tolist() == [0.0, 0.0, 1.0, 2.0]  # lag 2, zero-padded


def test_fir_lags_rejects_negative_lag() -> None:
    with pytest.raises(ValueError):
        fir_lags(np.zeros(3), lags=(-1,))


def test_zscore_standardizes_and_guards_constant_columns() -> None:
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    z = zscore(x)
    assert z[:, 0].mean() == pytest.approx(0.0)
    assert z[:, 0].std() == pytest.approx(1.0)
    assert z[:, 1].tolist() == [0.0, 0.0, 0.0]  # constant column, no NaN

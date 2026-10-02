"""Tests for evaluation metrics."""

import numpy as np
import pytest

from cneuromod.train.evaluation import pearson_per_column, summarize


def test_pearson_matches_numpy_corrcoef() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal((50, 3))
    y_pred = rng.standard_normal((50, 3))
    r = pearson_per_column(y_true, y_pred)
    for j in range(3):
        expected = np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]
        assert r[j] == pytest.approx(expected)


def test_pearson_perfect_and_anticorrelation() -> None:
    y = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    assert pearson_per_column(y, y * 3.0 + 1.0)[0] == pytest.approx(1.0)
    assert pearson_per_column(y, -y)[0] == pytest.approx(-1.0)


def test_pearson_zero_variance_gives_nan() -> None:
    y_true = np.ones((10, 1))
    y_pred = np.random.default_rng(0).standard_normal((10, 1))
    assert np.isnan(pearson_per_column(y_true, y_pred)[0])


def test_summarize_excludes_nans() -> None:
    r = np.array([0.5, np.nan, -0.1])
    stats = summarize(r)
    assert stats["n_parcels"] == 3.0
    assert stats["n_valid"] == 2.0
    assert stats["mean_r"] == pytest.approx(0.2)
    assert stats["frac_positive"] == pytest.approx(0.5)

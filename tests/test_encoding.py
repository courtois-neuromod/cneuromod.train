"""Tests for ridge encoding with leave-one-group-out CV, on synthetic data."""

import numpy as np
import pytest

from cneuromod.train.encoding import leave_one_group_out
from cneuromod.train.features import fir_lags, zscore


def test_recovers_signal_and_not_noise() -> None:
    """Targets built from lagged word-rate must be predictable; noise must not be."""
    rng = np.random.default_rng(42)
    n_trs = 200
    lags = (2, 3)

    features = []
    targets = []
    groups = []
    for movie in ("bourne", "wolf", "life", "figures"):
        rate = rng.poisson(2.0, size=n_trs).astype(np.float64)
        design = zscore(fir_lags(rate, lags))
        signal = design @ np.array([1.0, 0.5])  # parcel 0: driven by lagged rate
        noise = rng.standard_normal(n_trs)  # parcel 1: pure noise
        y = np.stack([signal + 0.3 * rng.standard_normal(n_trs), noise], axis=1)
        features.append(design)
        targets.append(zscore(y))
        groups.append(movie)

    folds = leave_one_group_out(features, targets, groups)
    assert [f.held_out for f in folds] == ["bourne", "figures", "life", "wolf"]
    for fold in folds:
        assert fold.n_test_trs == n_trs
        assert fold.n_train_trs == 3 * n_trs
        assert fold.r[0] > 0.8, f"signal parcel not recovered ({fold.held_out})"
        assert abs(fold.r[1]) < 0.25, f"noise parcel spuriously predicted ({fold.held_out})"


def test_length_mismatch_raises() -> None:
    x = [np.zeros((10, 2))]
    y = [np.zeros((10, 3))]
    with pytest.raises(ValueError):
        leave_one_group_out(x, y, ["a", "b"])

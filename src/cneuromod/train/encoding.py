"""Ridge encoding model with leave-one-group-out cross-validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import RidgeCV

from cneuromod.train.evaluation import pearson_per_column

#: Default ridge penalty grid searched (by efficient LOO-CV) on each fold's
#: training data.
DEFAULT_ALPHAS: tuple[float, ...] = tuple(float(a) for a in np.logspace(-1, 6, 8))


@dataclass(frozen=True)
class FoldResult:
    """Metrics for one held-out group (e.g. one movie)."""

    held_out: str
    n_train_trs: int
    n_test_trs: int
    alpha: float
    r: npt.NDArray[np.float64]  # per-column (parcel) Pearson r on the test set


def leave_one_group_out(
    features: Sequence[npt.NDArray[np.float64]],
    targets: Sequence[npt.NDArray[np.float64]],
    groups: Sequence[str],
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> list[FoldResult]:
    """Fit ridge on all-but-one group and evaluate on the held-out group.

    Parameters
    ----------
    features / targets:
        Per-run design matrices ``(n_trs_i, k)`` and targets ``(n_trs_i, p)``;
        both lists parallel to *groups*.
    groups:
        Group label per run (the movie name for leave-one-movie-out).
    alphas:
        Ridge penalty grid; the best value is selected on training data only.
    """
    if not (len(features) == len(targets) == len(groups)):
        raise ValueError("features, targets, and groups must have equal length")
    results: list[FoldResult] = []
    for held_out in sorted(set(groups)):
        train_idx = [i for i, g in enumerate(groups) if g != held_out]
        test_idx = [i for i, g in enumerate(groups) if g == held_out]
        train_x = np.concatenate([features[i] for i in train_idx], axis=0)
        train_y = np.concatenate([targets[i] for i in train_idx], axis=0)
        test_x = np.concatenate([features[i] for i in test_idx], axis=0)
        test_y = np.concatenate([targets[i] for i in test_idx], axis=0)

        model = RidgeCV(alphas=np.asarray(alphas), fit_intercept=True)
        model.fit(train_x, train_y)
        predictions = np.asarray(model.predict(test_x), dtype=np.float64)

        results.append(
            FoldResult(
                held_out=held_out,
                n_train_trs=int(train_x.shape[0]),
                n_test_trs=int(test_x.shape[0]),
                alpha=float(model.alpha_),
                r=pearson_per_column(test_y, predictions),
            )
        )
    return results

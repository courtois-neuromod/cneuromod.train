"""Tests for upstream compatibility shims.

Skipped automatically when the optional ``neuralfetch-cneuromod`` data layer
is not installed (e.g. in the default CI environment).
"""

from pathlib import Path

import pytest

from cneuromod.train.compat import patch_neuralfetch_cneuromod


def test_patch_neuralfetch_cneuromod(tmp_path: Path) -> None:
    pytest.importorskip("neuralfetch_cneuromod")

    patch_neuralfetch_cneuromod()
    patch_neuralfetch_cneuromod()  # idempotent

    from neuralfetch_cneuromod.studies.movie10 import Movie10

    root = tmp_path / "cneuromod"
    root.mkdir()
    study = Movie10(path=root, subjects=["01"])
    assert study.TASK == "movie10"
    assert study.bids_dir == root / "Movie10" / "bids"

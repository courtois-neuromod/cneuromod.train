"""Data-layer drift detection against neuralfetch-cneuromod (marie_dev).

Skipped automatically when the optional ``[data]`` extra is not installed
(e.g. in the default CI environment). Run locally before bumping any
neuralset / neuralfetch version bound to catch upstream API drift early.

Known constraints encoded here:

* Constructing a Study datalad-clones its repos when the expected
  subdirectories are missing — the fixture pre-creates them to keep tests
  offline.
* ``infra`` (an exca cache backend) must be provided: ``model_post_init``
  reads ``self.infra.folder``.
* neuralset's ``run()`` rejects explicitly-set Study fields ("class
  parameters"), so restriction like ``subjects=`` is only usable with
  ``download()`` / ``iter_timelines()``.
"""

from pathlib import Path

import pytest

from cneuromod.train.compat import patch_neuralfetch_cneuromod

SPACE = "MNI152NLin2009cAsym"


def _make_synthetic_movie10(root: Path) -> None:
    """Miniature movie10 layout following the real per-movie task naming."""
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")

    for sub in ("bids", "fmriprep", "stimuli", "annotations"):
        (root / "movie10" / sub).mkdir(parents=True)

    func = root / "movie10" / "fmriprep" / "sub-01" / "ses-001" / "func"
    func.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for task in ("bourne01", "bourne02"):
        data = rng.standard_normal((4, 4, 4, 20)).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms((2.0, 2.0, 2.0, 1.49))
        nib.save(
            img,
            str(func / f"sub-01_ses-001_task-{task}_space-{SPACE}_desc-preproc_bold.nii.gz"),
        )


def test_patch_is_idempotent_and_harmless() -> None:
    pytest.importorskip("neuralfetch_cneuromod")
    import neuralfetch_cneuromod.base as base

    patch_neuralfetch_cneuromod()
    patch_neuralfetch_cneuromod()
    assert hasattr(base.CNeuroModStudy, "infra_timelines")


def test_movie10_discovers_synthetic_runs(tmp_path: Path) -> None:
    pytest.importorskip("neuralfetch_cneuromod")
    from exca.steps.backends import Cached
    from neuralfetch_cneuromod.studies.movie10 import Movie10

    root = tmp_path / "cneuromod"
    root.mkdir()
    _make_synthetic_movie10(root)
    cache = tmp_path / "cache"
    cache.mkdir()

    study = Movie10(path=root, subjects=["01"], infra=Cached(folder=cache))
    assert study.TASK == "movie10"
    assert study.MOVIES == ["bourne", "figures", "life", "wolf"]

    timelines = list(study.iter_timelines())
    assert len(timelines) == 2
    assert {t["task"] for t in timelines} == {"bourne01", "bourne02"}
    assert {t["subject"] for t in timelines} == {"01"}

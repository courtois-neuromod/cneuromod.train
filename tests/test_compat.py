"""Tests for upstream compatibility shims and data-layer drift detection.

Skipped automatically when the optional ``neuralfetch-cneuromod`` data layer
is not installed (e.g. in the default CI environment). Run locally with the
``[data]`` extra installed to detect breaking changes in upstream
neuralset / neuralfetch releases early (the fast-moving Meta FAIR stack).
"""

from pathlib import Path

import pytest

from cneuromod.train.compat import patch_neuralfetch_cneuromod


def _make_synthetic_movie10(root: Path) -> None:
    """Create a miniature movie10 layout: one BOLD run + one events.tsv."""
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    func_fmriprep = root / "Movie10" / "fmriprep" / "sub-01" / "ses-001" / "func"
    func_bids = root / "Movie10" / "bids" / "sub-01" / "ses-001" / "func"
    func_fmriprep.mkdir(parents=True)
    func_bids.mkdir(parents=True)

    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 4, 4, 20)).astype(np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0, 1.49))
    stem = "sub-01_ses-001_task-movie10_run-01"
    nib.save(
        img,
        str(func_fmriprep / f"{stem}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"),
    )

    events = pd.DataFrame(
        {
            "onset": [0.0, 10.0],
            "duration": [10.0, 10.0],
            "trial_type": ["Stimulus", "Stimulus"],
        }
    )
    events.to_csv(func_bids / f"{stem}_events.tsv", sep="\t", index=False)


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


def test_movie10_run_end_to_end(tmp_path: Path) -> None:
    """Full iter_timelines() + run() pipeline on a synthetic dataset.

    This is the upstream drift detector: it exercises the same neuralset code
    paths as a real experiment, minus DataLad downloads. Two known upstream
    constraints are encoded here:

    * ``run()`` only works with an all-default Study parametrization —
      neuralset rejects explicitly-set fields ("class parameters"); use a
      separate restricted instance for ``download()``/``iter_timelines()``.
    * Events must map to neuralset's canonical event types; file-backed types
      (``Video``) require the stimulus files to exist, which the CNeuroMod
      download patterns intentionally skip.
    """
    pytest.importorskip("neuralfetch_cneuromod")

    patch_neuralfetch_cneuromod()
    from neuralfetch_cneuromod.studies.movie10 import Movie10

    root = tmp_path / "cneuromod"
    root.mkdir()
    _make_synthetic_movie10(root)

    restricted = Movie10(path=root, subjects=["01"])
    timelines = list(restricted.iter_timelines())
    assert timelines == [{"subject": "01", "session": "001", "run": "01", "task": "movie10"}]

    events = Movie10(path=root).run()
    fmri = events[events["type"] == "Fmri"]
    assert len(fmri) == 1
    assert float(fmri["duration"].iloc[0]) == pytest.approx(20 * 1.49)
    assert float(fmri["frequency"].iloc[0]) == pytest.approx(1 / 1.49)
    stimulus = events[events["type"] == "Stimulus"]
    assert len(stimulus) == 2
    assert set(events["subject"]) == {"Movie10/01"}

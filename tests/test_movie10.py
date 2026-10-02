"""Tests for the movie10 file adapter, using a synthetic on-disk layout."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cneuromod.train.movie10 import (
    has_transcript,
    load_runs,
    load_transcript,
    segment_to_movie,
)


@pytest.fixture()
def movie10_root(tmp_path: Path) -> Path:
    root = tmp_path / "Movie10"
    h5_path = (
        root
        / "timeseries/timeseries/cneuromod2026/sub-01"
        / "sub-01_task-movie10_space-MNI152NLin2009cAsym"
        "_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
    )
    h5_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(0)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "ses-001/ses-001_task-bourne01_timeseries",
            data=rng.standard_normal((30, 7)).astype(np.float32),
        )
        f.create_dataset(
            "ses-002/ses-002_task-wolf01_timeseries",
            data=rng.standard_normal((25, 7)).astype(np.float32),
        )

    transcript_dir = root / "annotations/annotations/transcripts/bourne"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "movie10_bourne01_model-AA_transcript.json").write_text(
        json.dumps(
            {
                "transcript": "hello world",
                "words": [
                    {"word": "hello", "start": 1.0, "end": 1.4, "confidence": 0.99},
                    {"word": "world", "start": 2.0, "end": 2.5, "confidence": 0.98},
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_segment_to_movie() -> None:
    assert segment_to_movie("bourne01") == "bourne"
    assert segment_to_movie("figures12") == "figures"


def test_load_runs(movie10_root: Path) -> None:
    runs = load_runs(movie10_root, "01")
    assert [(r.session, r.segment, r.movie) for r in runs] == [
        ("ses-001", "bourne01", "bourne"),
        ("ses-002", "wolf01", "wolf"),
    ]
    assert runs[0].n_trs == 30
    assert runs[0].n_parcels == 7
    assert runs[0].bold.dtype == np.float32


def test_load_runs_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path, "01")


def test_transcript_loading(movie10_root: Path) -> None:
    assert has_transcript(movie10_root, "bourne01")
    assert not has_transcript(movie10_root, "wolf01")
    words = load_transcript(movie10_root, "bourne01")
    assert [w.text for w in words] == ["hello", "world"]
    assert words[0].start == pytest.approx(1.0)

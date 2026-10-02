"""Thin data adapter for CNeuroMod movie10 files on disk.

Reads the files materialized by the ``neuralfetch-cneuromod`` Study class
(pre-extracted parcel timeseries HDF5 + AssemblyAI transcript JSONs) without
re-implementing data acquisition. Reading files directly is the sanctioned
bypass while ``Study.run()`` requires stimulus videos we do not download;
switching to the neuralset events table is a follow-up once that upstream
constraint is resolved.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: movie10 BOLD repetition time in seconds (CNeuroMod acquisition parameter;
#: not stored in the timeseries HDF5, so pinned here).
TR_SECONDS = 1.49

_TIMESERIES_H5 = (
    "timeseries/timeseries/cneuromod2026/sub-{subject}/"
    "sub-{subject}_task-movie10_space-MNI152NLin2009cAsym"
    "_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
)
_TRANSCRIPT = (
    "annotations/annotations/transcripts/{movie}/movie10_{segment}_model-AA_transcript.json"
)

_SEGMENT_RE = re.compile(r"task-([a-z]+\d+)")


@dataclass(frozen=True)
class Word:
    """One transcribed word with onset/offset in seconds from segment start."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Run:
    """One BOLD run: a subject watching one ~10-minute movie segment."""

    subject: str
    session: str
    segment: str  # e.g. "bourne01"
    movie: str  # e.g. "bourne"
    bold: npt.NDArray[np.float32]  # (n_trs, n_parcels)

    @property
    def n_trs(self) -> int:
        return int(self.bold.shape[0])

    @property
    def n_parcels(self) -> int:
        return int(self.bold.shape[1])


def segment_to_movie(segment: str) -> str:
    """``"bourne01"`` -> ``"bourne"``."""
    return segment.rstrip("0123456789")


def load_runs(root: Path, subject: str) -> list[Run]:
    """Load every parcel-timeseries run for *subject* from the movie10 HDF5.

    Parameters
    ----------
    root:
        The movie10 study directory (containing ``timeseries/`` and
        ``annotations/``), e.g. ``data/cneuromod/Movie10``.
    subject:
        Subject label without the ``sub-`` prefix, e.g. ``"01"``.
    """
    h5_path = root / _TIMESERIES_H5.format(subject=subject)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Timeseries file not found: {h5_path}\n"
            "Fetch it with: datalad -C <root>/timeseries get "
            "timeseries/cneuromod2026/sub-{subject}/..."
        )
    runs: list[Run] = []
    with h5py.File(h5_path, "r") as f:
        for session in sorted(f):
            group = f[session]
            for name in sorted(group):
                match = _SEGMENT_RE.search(name)
                if match is None:
                    logger.warning("Skipping dataset with unrecognized name: %s/%s", session, name)
                    continue
                segment = match.group(1)
                bold = np.asarray(group[name], dtype=np.float32)
                runs.append(
                    Run(
                        subject=subject,
                        session=session,
                        segment=segment,
                        movie=segment_to_movie(segment),
                        bold=bold,
                    )
                )
    return runs


def transcript_path(root: Path, segment: str) -> Path:
    return root / _TRANSCRIPT.format(movie=segment_to_movie(segment), segment=segment)


def has_transcript(root: Path, segment: str) -> bool:
    return transcript_path(root, segment).exists()


def load_transcript(root: Path, segment: str) -> list[Word]:
    """Load the word-level transcript for one movie segment."""
    path = transcript_path(root, segment)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Word(text=str(w["word"]), start=float(w["start"]), end=float(w["end"]))
        for w in payload["words"]
    ]

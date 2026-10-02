"""First encoding baseline: word-rate -> movie10 parcel timeseries (Milestone A1).

For one subject: count transcribed words per TR, FIR-lag the counts to cover
the hemodynamic delay, fit ridge regression to the 1134-parcel cneuromod2026
timeseries with leave-one-movie-out cross-validation, and report parcel-wise
Pearson r on held-out movies. Results (per-fold and per-parcel) plus full
provenance are written as JSON.

Run from WSL (the timeseries HDF5 is a git-annex file only readable there)::

    .venv-wsl/bin/python scripts/run_movie10_wordrate.py --root data/cneuromod/Movie10 --subject 01
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np

from cneuromod.train import __version__ as cneuromod_train_version
from cneuromod.train.encoding import DEFAULT_ALPHAS, leave_one_group_out
from cneuromod.train.evaluation import summarize
from cneuromod.train.features import fir_lags, word_rate, zscore
from cneuromod.train.movie10 import TR_SECONDS, has_transcript, load_runs, load_transcript

logger = logging.getLogger(__name__)

DEFAULT_LAGS = (2, 3, 4, 5)  # TR shifts, roughly 3-7.5 s at TR=1.49 s (HRF delay)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/cneuromod/Movie10"))
    parser.add_argument("--subject", default="01")
    parser.add_argument("--lags", type=int, nargs="*", default=list(DEFAULT_LAGS))
    parser.add_argument("--seed", type=int, default=0, help="recorded for provenance")
    parser.add_argument("--out", type=Path, default=None, help="output JSON path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    np.random.seed(args.seed)

    runs = load_runs(args.root, args.subject)
    usable = [r for r in runs if has_transcript(args.root, r.segment)]
    skipped = sorted({r.segment for r in runs} - {r.segment for r in usable})
    if skipped:
        logger.warning("Skipping runs without transcript: %s", skipped)

    features = []
    targets = []
    groups = []
    for run in usable:
        onsets = [w.start for w in load_transcript(args.root, run.segment)]
        rate = word_rate(onsets, run.n_trs, TR_SECONDS)
        features.append(zscore(fir_lags(rate, args.lags)))
        targets.append(zscore(run.bold.astype(np.float64)))
        groups.append(run.movie)

    logger.info(
        "Fitting on %d runs (%d TRs total), movies: %s",
        len(usable),
        sum(f.shape[0] for f in features),
        sorted(set(groups)),
    )
    folds = leave_one_group_out(features, targets, groups)

    fold_payload = []
    for fold in folds:
        stats = summarize(fold.r)
        logger.info(
            "held-out %-8s  mean r=%.4f  median r=%.4f  max r=%.4f  p95 r=%.4f",
            fold.held_out,
            stats["mean_r"],
            stats["median_r"],
            stats["max_r"],
            stats["p95_r"],
        )
        fold_payload.append(
            {
                "held_out_movie": fold.held_out,
                "n_train_trs": fold.n_train_trs,
                "n_test_trs": fold.n_test_trs,
                "alpha": fold.alpha,
                "summary": stats,
                "r_per_parcel": [None if np.isnan(v) else float(v) for v in fold.r],
            }
        )
    mean_r = np.nanmean(np.stack([f.r for f in folds]), axis=0)
    overall = summarize(mean_r)
    logger.info("ACROSS FOLDS   mean r=%.4f  max r=%.4f", overall["mean_r"], overall["max_r"])

    payload = {
        "experiment": "movie10_wordrate_ridge",
        "config": {
            "subject": args.subject,
            "root": str(args.root),
            "tr_seconds": TR_SECONDS,
            "lags": list(args.lags),
            "alphas": list(DEFAULT_ALPHAS),
            "split": "leave-one-movie-out",
            "features": "word_rate (per-run z-scored, FIR-lagged)",
            "targets": "cneuromod2026 1134-parcel timeseries (per-run z-scored)",
            "seed": args.seed,
        },
        "data": {
            "n_runs_used": len(usable),
            "segments_skipped_no_transcript": skipped,
            "runs": [
                {"session": r.session, "segment": r.segment, "movie": r.movie, "n_trs": r.n_trs}
                for r in usable
            ],
        },
        "provenance": {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "git_commit": _git_commit(),
            "cneuromod_train": cneuromod_train_version,
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "h5py": version("h5py"),
        },
        "results": {
            "overall_summary_mean_r_across_folds": overall,
            "folds": fold_payload,
        },
    }
    out = args.out or Path("outputs") / f"movie10_wordrate_sub-{args.subject}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results written to %s", out)


if __name__ == "__main__":
    main()

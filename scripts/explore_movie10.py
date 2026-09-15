"""Explore the CNeuroMod movie10 dataset through its NeuralSet Study class.

First step toward Milestone A1 (movie10 encoding baseline): download a
subject's worth of events + preprocessed BOLD through the existing
``neuralfetch-cneuromod`` Study class and summarize what it exposes.

Usage (bash, with datalad and git-annex installed)::

    set -a; source .env; set +a    # AWS S3 credentials for annex content
    python scripts/explore_movie10.py --path data/cneuromod --subjects 01 --download

Without ``--download`` the script only inspects data already on disk.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cneuromod.train.compat import patch_neuralfetch_cneuromod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="data/cneuromod", help="root data directory")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=["01"],
        help="subject labels without the sub- prefix (default: 01)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="clone the DataLad repos and fetch events/BOLD content "
        "(requires datalad, git-annex, and S3 credentials in the environment)",
    )
    parser.add_argument("--datalad-jobs", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    patch_neuralfetch_cneuromod()
    from neuralfetch_cneuromod.studies.movie10 import Movie10

    Path(args.path).mkdir(parents=True, exist_ok=True)
    study = Movie10(path=args.path, subjects=args.subjects, datalad_jobs=args.datalad_jobs)
    print(f"study        : {study.dataset_name}")
    print(f"bids dir     : {study.bids_dir}")
    print(f"fmriprep dir : {study.fmriprep_dir}")
    print(f"space / res  : {study.space} / {study.resolution}")

    if args.download:
        study.download()

    try:
        timelines = list(study.iter_timelines())
    except FileNotFoundError as err:
        print(f"\nNo data on disk yet ({err}).\nRe-run with --download to fetch it.")
        return

    print(f"\n{len(timelines)} timelines (subject / session / run):")
    for timeline in timelines[:10]:
        print("   ", timeline)
    if len(timelines) > 10:
        print(f"    ... and {len(timelines) - 10} more")
    if not timelines:
        print("Directories exist but contain no matching BOLD runs.")
        return

    events = study.run()
    print("\nevents DataFrame:")
    print(f"  shape   : {events.shape}")
    print(f"  columns : {list(events.columns)}")
    print("  event types:")
    print(events["type"].value_counts().to_string())


if __name__ == "__main__":
    main()

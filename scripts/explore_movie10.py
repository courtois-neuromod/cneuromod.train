"""Explore the CNeuroMod movie10 dataset through its NeuralSet Study class.

First step toward Milestone A1 (movie10 encoding baseline): download a
subject's worth of data through the ``neuralfetch-cneuromod`` (marie_dev)
Movie10 Study class and summarize what it exposes.

Notes on the data layer (verified 2026-09-16):

* Constructing a Study datalad-clones its repos when the expected
  subdirectories are missing — construction alone touches the network.
* ``infra`` (exca cache backend) is required; pass a cache folder.
* neuralset's ``run()`` rejects explicitly-set Study fields ("class
  parameters"), so ``subjects=`` is only used for download/discovery here
  and the events DataFrame is filtered afterwards.
* Annotations are speech-to-text transcripts in the movie10.annotations
  DataLad repo; pre-extracted timeseries (``--timeseries``) avoid raw BOLD.

Usage (bash, with datalad and git-annex installed)::

    set -a; source .env; set +a    # AWS S3 credentials for annex content
    python scripts/explore_movie10.py --path data/cneuromod --subjects 01 --download

Without ``--download`` the script only inspects data already on disk (but
construction still clones the repo skeletons on first use).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


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
        help="fetch events/BOLD/stimuli/transcript content via DataLad "
        "(requires git-annex and S3 credentials in the environment)",
    )
    parser.add_argument(
        "--timeseries",
        default=None,
        help="load pre-extracted timeseries instead of raw BOLD "
        "(e.g. cneuromod2026, schaefer1000, voxel_mni)",
    )
    parser.add_argument("--cache", default=None, help="exca cache folder (default: {path}/.cache)")
    parser.add_argument("--datalad-jobs", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from exca.steps.backends import Cached
    from neuralfetch_cneuromod.studies.movie10 import Movie10

    Path(args.path).mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else Path(args.path) / ".cache"
    cache.mkdir(parents=True, exist_ok=True)

    study = Movie10(
        path=args.path,
        subjects=args.subjects,
        infra=Cached(folder=cache),
        datalad_jobs=args.datalad_jobs,
        timeseries=args.timeseries,
    )
    print(f"study        : {study.dataset_name}")
    print(f"bids dir     : {study.bids_dir}")
    print(f"fmriprep dir : {study.fmriprep_dir}")
    print(f"movies       : {study.MOVIES}")
    print(f"timeseries   : {study.timeseries}")

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
        print("Directories exist but contain no matching BOLD runs — download first.")
        return

    # run() needs an all-default instance (neuralset rejects set fields);
    # filter the result to the requested subjects instead.
    events = Movie10(path=args.path, infra=Cached(folder=cache)).run()
    events = events[events["subject"].isin([f"Movie10/{s}" for s in args.subjects])]
    print("\nevents DataFrame:")
    print(f"  shape   : {events.shape}")
    print(f"  columns : {list(events.columns)}")
    print("  event types:")
    print(events["type"].value_counts().to_string())


if __name__ == "__main__":
    main()

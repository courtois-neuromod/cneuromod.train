"""Smoke tests for the installed package."""

from importlib.metadata import version

import cneuromod.train


def test_import() -> None:
    assert cneuromod.train.__doc__ is not None


def test_version_matches_metadata() -> None:
    assert cneuromod.train.__version__ == version("cneuromod.train")

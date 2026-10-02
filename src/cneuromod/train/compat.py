"""Compatibility shims for upstream NeuroAI packages.

These work around version drift between ``neuralfetch-cneuromod`` and the
``neuralset`` / ``neuralfetch`` releases on PyPI. Each shim must be removed
once the corresponding fix lands upstream.
"""

from __future__ import annotations

import typing as tp

__all__ = ["patch_neuralfetch_cneuromod"]


class _LegacyInfraShim:
    """Stand-in for the removed ``Study.infra_timelines`` field.

    ``neuralfetch-cneuromod`` 0.1.0 runs ``self.infra_timelines.cluster = None``
    (meaning: load timelines locally) during ``model_post_init``. neuralset
    >= 0.2 renamed the field to ``Study.timelines.infra``, whose default
    ``None`` already means local execution, so accepting the assignment as a
    no-op preserves the intended behavior.
    """

    cluster: tp.Any = None


def patch_neuralfetch_cneuromod() -> None:
    """Make pre-merge ``neuralfetch-cneuromod`` installs importable.

    Only the pre-2026-10 stale ``main`` needed this (it targeted neuralset
    <= 0.2.2, whose ``infra_timelines`` field was renamed in 0.2.3). The
    merged ``main`` — whose commit the ``[data]`` extra now pins — uses the
    current API, making the patch a harmless idempotent no-op; it can be
    removed once we trust there are no pre-merge installs left.
    """
    import neuralfetch_cneuromod.base as base

    if not hasattr(base.CNeuroModStudy, "infra_timelines"):
        base.CNeuroModStudy.infra_timelines = _LegacyInfraShim()  # type: ignore[attr-defined]

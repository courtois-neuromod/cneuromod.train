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
    """Make ``neuralfetch-cneuromod``'s stale *main* branch importable.

    Only the ``main`` branch needs this (it targets neuralset <= 0.2.2, whose
    ``infra_timelines`` field was renamed in 0.2.3); the actively developed
    ``marie_dev`` branch — which the ``[data]`` extra installs — already uses
    the current API, and the patch is a harmless no-op there. Idempotent.
    """
    import neuralfetch_cneuromod.base as base

    if not hasattr(base.CNeuroModStudy, "infra_timelines"):
        base.CNeuroModStudy.infra_timelines = _LegacyInfraShim()  # type: ignore[attr-defined]

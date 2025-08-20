"""Lightweight package init for core.learn.

Avoid importing heavy, optional dependencies (e.g., torch) at package import
time so tests that only need utility modules (like policy_utils) can run
without requiring the full ML stack to be installed.
"""

try:
    from .ac_network import ACNetwork  # type: ignore
    _HAS_ACNETWORK = True
except Exception:  # pragma: no cover - absence of torch or other deps
    ACNetwork = None  # type: ignore
    _HAS_ACNETWORK = False

from .ac_player import ACPlayer
from .recording_ac_player import (
    RecordingACPlayer,
    ExperienceBuffer,
    RecordingHeuristicACPlayer,
)

__all__ = [
    "ACPlayer",
    "RecordingACPlayer",
    "RecordingHeuristicACPlayer",
    "ExperienceBuffer",
]

if _HAS_ACNETWORK:
    __all__.append("ACNetwork")



"""Deterministic signal primitives."""

from .g7_chen import (
    SELECTED_D3_INDICES,
    chen_signature_level3,
    cross_section_average_tie_rank,
    daily_d3,
    trailing_complete_means,
)
from .g7_builder import G7BuildResult, build_g7_77
from .full_builder import SignalPanel, build_signal_panel

__all__ = [
    "SELECTED_D3_INDICES",
    "chen_signature_level3",
    "cross_section_average_tie_rank",
    "daily_d3",
    "trailing_complete_means",
    "G7BuildResult",
    "build_g7_77",
    "SignalPanel",
    "build_signal_panel",
]

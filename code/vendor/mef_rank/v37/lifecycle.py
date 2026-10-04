"""Daily anchor identity and v3.7 Memory lifecycle contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class MemoryLifecycle(str, Enum):
    OOF_CREATED = "OOF_CREATED"
    PENDING_LABEL = "PENDING_LABEL"
    LABEL_RESOLVED = "LABEL_RESOLVED"
    DIRECTION_WRITER_SCHEDULED = "DIRECTION_WRITER_SCHEDULED"
    DIRECTION_WRITER_COMPLETED = "DIRECTION_WRITER_COMPLETED"
    MAGNITUDE_CALIBRATION_COMPLETED = "MAGNITUDE_CALIBRATION_COMPLETED"
    OUTER_OOB_COMPLETED = "OUTER_OOB_COMPLETED"
    FULL_DIRECTION_COMPLETED = "FULL_DIRECTION_COMPLETED"
    MEMORY_ELIGIBLE = "MEMORY_ELIGIBLE"
    TERMINAL_INVALID_LABEL = "TERMINAL_INVALID_LABEL"
    TERMINAL_REJECT = "TERMINAL_REJECT"


_ALLOWED_TRANSITIONS = {
    MemoryLifecycle.OOF_CREATED: {MemoryLifecycle.PENDING_LABEL},
    MemoryLifecycle.PENDING_LABEL: {
        MemoryLifecycle.LABEL_RESOLVED,
        MemoryLifecycle.TERMINAL_INVALID_LABEL,
    },
    MemoryLifecycle.LABEL_RESOLVED: {
        MemoryLifecycle.DIRECTION_WRITER_SCHEDULED
    },
    MemoryLifecycle.DIRECTION_WRITER_SCHEDULED: {
        MemoryLifecycle.DIRECTION_WRITER_COMPLETED,
        MemoryLifecycle.TERMINAL_REJECT,
    },
    MemoryLifecycle.DIRECTION_WRITER_COMPLETED: {
        MemoryLifecycle.MAGNITUDE_CALIBRATION_COMPLETED,
        MemoryLifecycle.TERMINAL_REJECT,
    },
    MemoryLifecycle.MAGNITUDE_CALIBRATION_COMPLETED: {
        MemoryLifecycle.OUTER_OOB_COMPLETED,
        MemoryLifecycle.TERMINAL_REJECT,
    },
    MemoryLifecycle.OUTER_OOB_COMPLETED: {
        MemoryLifecycle.FULL_DIRECTION_COMPLETED,
        MemoryLifecycle.TERMINAL_REJECT,
    },
    MemoryLifecycle.FULL_DIRECTION_COMPLETED: {
        MemoryLifecycle.MEMORY_ELIGIBLE,
        MemoryLifecycle.TERMINAL_REJECT,
    },
}


def assert_transition(current: MemoryLifecycle, target: MemoryLifecycle) -> None:
    if target not in _ALLOWED_TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal v3.7 Memory transition: {current} -> {target}")


@dataclass(frozen=True)
class DailyAnchor:
    checkpoint_id: str
    feature_date: str
    label_maturity_timestamp: str
    memory_eligible_timestamp: str
    episode_id: str
    anchor_id: str

    def __post_init__(self) -> None:
        expected = f"{self.checkpoint_id}-{self.feature_date}"
        if self.anchor_id != expected:
            raise ValueError("v3.7 anchor_id must be checkpoint_id-feature_date")
        if self.memory_eligible_timestamp < self.label_maturity_timestamp:
            raise ValueError("Memory cannot become eligible before H5 maturity")
        if not self.feature_date.isdigit() or len(self.feature_date) != 8:
            raise ValueError("feature_date must be YYYYMMDD")


def daily_mature_anchors(
    *,
    checkpoint_id: str,
    episode_id: str,
    feature_dates: Iterable[str],
    label_maturity_timestamps: Iterable[str],
    asof_timestamp: str,
) -> tuple[DailyAnchor, ...]:
    dates = tuple(str(value) for value in feature_dates)
    maturity = tuple(str(value) for value in label_maturity_timestamps)
    if len(dates) != len(maturity) or len(set(dates)) != len(dates):
        raise ValueError("daily anchor dates/maturity values do not align")
    rows = tuple(
        DailyAnchor(
            checkpoint_id=str(checkpoint_id),
            feature_date=date,
            label_maturity_timestamp=timestamp,
            memory_eligible_timestamp=timestamp,
            episode_id=str(episode_id),
            anchor_id=f"{checkpoint_id}-{date}",
        )
        for date, timestamp in zip(dates, maturity, strict=True)
        if timestamp < str(asof_timestamp)
    )
    return tuple(sorted(rows, key=lambda row: row.feature_date))

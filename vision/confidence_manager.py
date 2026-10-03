"""Confidence bookkeeping for every perception subsystem.

PRISM never reports a number it is not sure about.  Each subsystem (OCR keys,
minimap detections, ROI searches) records its observations with a confidence
score; the manager decides whether a value is *accepted*, *tentative* or
*unavailable* and always keeps the strongest reading seen so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.logger import get_logger

logger = get_logger("confidence")

STATUS_ACCEPTED = "accepted"
STATUS_LOW = "confidence too low"
STATUS_UNAVAILABLE = "value unavailable"
STATUS_INVALID = "value unusable"


@dataclass
class ConfidenceRecord:
    """Aggregated view over all readings captured for one key."""

    key: str
    value: str = ""
    confidence: float = 0.0
    attempts: int = 0
    readings: List[float] = field(default_factory=list)
    status: str = STATUS_UNAVAILABLE
    floor: float = 0.0
    best_value: str = ""


class ConfidenceManager:
    """Collect readings and gate them behind configurable confidence floors."""

    def __init__(self, default_threshold: float = 0.55) -> None:
        self.default_threshold = float(default_threshold)
        self._records: Dict[str, ConfidenceRecord] = {}

    def reset(self) -> None:
        self._records.clear()

    def record(
        self,
        key: str,
        value: str,
        confidence: float,
        threshold: Optional[float] = None,
    ) -> ConfidenceRecord:
        """Store one reading and re-evaluate acceptance for ``key``."""
        confidence = float(min(1.0, max(0.0, confidence)))
        record = self._records.setdefault(key, ConfidenceRecord(key=key))
        record.attempts += 1
        record.readings.append(confidence)

        floor = self.default_threshold if threshold is None else float(threshold)
        record.floor = floor
        if confidence > max(record.readings[:-1], default=0.0):
            record.best_value = value  # best text seen, accepted or not
        if confidence >= floor and confidence >= record.confidence:
            record.value = value
            record.confidence = confidence

        if record.confidence >= floor and record.value:
            record.status = STATUS_ACCEPTED
        elif record.status == STATUS_ACCEPTED:
            # Never downgrade an already accepted key: the strongest reading
            # seen so far stays available (documented behaviour).
            pass
        elif floor > 1.0:
            # Sentinelled reading (confident text that failed validation):
            # the text exists but is unusable - not a confidence problem.
            record.status = STATUS_INVALID
        else:
            record.status = STATUS_LOW if record.readings else STATUS_UNAVAILABLE
        return record

    def get(self, key: str) -> Optional[ConfidenceRecord]:
        return self._records.get(key)

    def value(self, key: str) -> Optional[str]:
        record = self._records.get(key)
        if record and record.status == STATUS_ACCEPTED:
            return record.value
        return None

    def confidence(self, key: str) -> float:
        record = self._records.get(key)
        return record.confidence if record else 0.0

    def status(self, key: str) -> str:
        record = self._records.get(key)
        return record.status if record else STATUS_UNAVAILABLE

    def summary(self) -> Dict[str, Dict[str, object]]:
        """Human readable dump used by the report and the logs."""
        out: Dict[str, Dict[str, object]] = {}
        for key, record in self._records.items():
            out[key] = {
                "value": record.value or STATUS_UNAVAILABLE,
                "confidence": round(record.confidence, 3),
                "attempts": record.attempts,
                "status": record.status,
            }
        return out

    def note_failures(self) -> None:
        """Log every key that never reached its acceptance threshold."""
        for key, record in self._records.items():
            if record.status != STATUS_ACCEPTED:
                best = max(record.readings) if record.readings else 0.0
                logger.warning(
                    "OCR key '%s' rejected (%s, best=%.2f >= floor %.2f? over %d "
                    "attempts), best text=%r",
                    key, record.status, best, record.floor, record.attempts,
                    record.best_value or record.value or "(none)",
                )

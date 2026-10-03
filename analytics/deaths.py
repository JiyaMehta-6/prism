"""Vision-only death inference (no telemetry, no death feed).

A champion that dies vanishes from the minimap and reappears at its own
fountain after the respawn timer.  PRISM watches the *marker* instead of the
kill feed:

1. the tracker loses the marker somewhere away from base,
2. the first re-appearance sits inside the own-base radius,
3. optionally the OCR scoreboard shows a team-kill delta around the gap.

Every gate is deliberately conservative - a short unconfirmed gap (a recall
channel or a tracker blip) is discarded rather than reported.  With the
scoreboard available a real death is almost always confirmed, because some
team's kill counter must increase while the marker is gone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from core.models import PositionSample

# Minimum absence: shorter gaps are tracker dropouts or the 8 s recall
# channel, not deaths.  Without scoreboard corroboration the bar is higher
# (a blind recall reappears at the fountain too).
DEATH_MIN_GAP_SEC = 6.0
DEATH_MAX_GAP_SEC = 55.0
DEATH_UNCONFIRMED_GAP_SEC = 10.0

# Distance from the own-fountain centre (normalised map units) that still
# counts as "at base" for both the vanish and the respawn end.  Matches
# RegionMapper's ``base_radius`` (0.10) so a fountain visit recognised as the
# Base region is also recognised here - ROI scale varies between videos and
# the respawn marker can sit ~0.09 from the corner anchor.
DEATH_BASE_RADIUS = 0.10

# The scoreboard is read only every ~30 s, so the corroborating kill can be
# stamped well after (or slightly before) the marker vanished.
KILL_CONFIRM_PRE_SEC = 15.0
KILL_CONFIRM_POST_SEC = 35.0

_BASE_CONFIDENCE = 0.55
_CONFIRM_BONUS = 0.25
_LONG_GAP_BONUS = 0.10
_MAX_CONFIDENCE = 0.90


@dataclass(frozen=True)
class DeathInference:
    """One inferred death."""

    time: float  # last sighting - the inferred death moment (effective time)
    respawn_time: float  # first reappearance at the fountain
    off_map_sec: float
    confirmed: bool  # a scoreboard kill delta overlapped the gap
    confidence: float


def _near(x: float, y: float, base_xy: Tuple[float, float], radius: float) -> bool:
    return math.hypot(x - base_xy[0], y - base_xy[1]) <= radius


def detect_deaths(
    samples: Sequence[PositionSample],
    base_xy: Tuple[float, float],
    kill_times: Sequence[float] = (),
    min_gap: float = DEATH_MIN_GAP_SEC,
    max_gap: float = DEATH_MAX_GAP_SEC,
    base_radius: float = DEATH_BASE_RADIUS,
) -> List[DeathInference]:
    """Infer deaths from marker dropouts that end at the own fountain.

    Args:
        samples: Position samples (interpolated or not; only ``detected``
            observations are used as gap anchors).
        base_xy: Centre of the player's own fountain.
        kill_times: Effective times of scoreboard kill events (either team).
        min_gap/max_gap: Accepted absence duration in seconds.
        base_radius: Normalised radius that counts as "at base".

    Returns:
        One :class:`DeathInference` per accepted gap, in time order.
    """
    anchors = [sample for sample in samples if sample.detected]
    deaths: List[DeathInference] = []
    for prev, curr in zip(anchors, anchors[1:]):
        gap = curr.effective_time - prev.effective_time
        if gap < min_gap or gap > max_gap:
            continue
        if _near(prev.x, prev.y, base_xy, base_radius):
            continue  # vanished at base: fountain/tracker artifact
        if not _near(curr.x, curr.y, base_xy, base_radius):
            continue  # reappeared elsewhere: tracker blip, not a death
        confirmed = any(
            prev.effective_time - KILL_CONFIRM_PRE_SEC
            <= kill_time
            <= curr.effective_time + KILL_CONFIRM_POST_SEC
            for kill_time in kill_times
        )
        if not confirmed and gap < DEATH_UNCONFIRMED_GAP_SEC:
            continue  # short + blind: a recall looks exactly like this
        confidence = _BASE_CONFIDENCE + (_CONFIRM_BONUS if confirmed else 0.0)
        if gap >= 12.0:
            confidence += _LONG_GAP_BONUS
        deaths.append(
            DeathInference(
                time=prev.effective_time,
                respawn_time=curr.effective_time,
                off_map_sec=gap,
                confirmed=confirmed,
                confidence=min(confidence, _MAX_CONFIDENCE),
            )
        )
    return deaths

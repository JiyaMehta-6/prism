"""Vision- and event-aware micro-metrics.

Two small measurements that replay-telemetry tools cannot produce because
they require *watching* the minimap over time:

* :func:`unwarded_forward_fraction` - how much of the player's forward time
  has no recent vision bloom anywhere nearby ("pushing without vision"),
* :func:`reaction_after_kills` - how quickly the player changes what they are
  doing (speed or heading) after a scoreboard kill event appears.

Both are pure functions over already-tracked samples so they stay trivially
testable and cost O(samples x wards) at worst.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from core.models import PositionSample

# A vision bloom covers this normalised map radius and stays "active" in the
# metric for this long (yellow trinket wards last ~90-120 s; the lookback is
# intentionally conservative so coverage is never overstated).
WARD_VISION_RADIUS = 0.10
WARD_ACTIVE_LOOKBACK_SEC = 120.0

# Reaction probe: movement is compared over this window on each side of the
# kill event; a response is a big enough *sustained* speed change or a heading
# change.  Windows are medians/displacements so single-sample jitter at 1 Hz
# cannot fake a reaction.
REACTION_PRE_SEC = 5.0
REACTION_POST_SEC = 8.0
REACTION_MIN_SIDE = 3
REACTION_ANGLE_DEG = 75.0
REACTION_SPEED_UP = 1.7
REACTION_SLOW_DOWN = 0.4
REACTION_STANDSTILL_SPEED = 0.015  # map units / second: "basically stopped"
REACTION_MIN_DISPLACEMENT = 0.006  # below this a heading is meaningless


def unwarded_forward_fraction(
    samples: Sequence[PositionSample],
    forward: Sequence[bool],
    wards: Sequence[Tuple[float, float, float]],
    radius: float = WARD_VISION_RADIUS,
    lookback: float = WARD_ACTIVE_LOOKBACK_SEC,
) -> Optional[float]:
    """Share of forward samples with no recent ward bloom within ``radius``.

    Args:
        samples: Position samples in effective (game) time.
        forward: Boolean mask aligned with ``samples`` (e.g. forward bias
            above the analysis threshold).
        wards: ``(time, x, y)`` tuples of confirmed vision blooms, same clock
            as the samples.
        radius: Normalised map radius covered by one bloom.
        lookback: How long after placement a bloom still counts.

    Returns:
        Fraction in ``[0, 1]``, or ``None`` when there are no forward
        samples (nothing to judge).
    """
    points = [
        (sample.effective_time, sample.x, sample.y)
        for sample, is_forward in zip(samples, forward)
        if is_forward
    ]
    if not points:
        return None
    unwarded = 0
    for time, x, y in points:
        covered = any(
            0.0 <= time - ward_time <= lookback
            and math.hypot(x - wx, y - wy) <= radius
            for ward_time, wx, wy in wards
        )
        if not covered:
            unwarded += 1
    return unwarded / len(points)


def _median_speed(group: Sequence[PositionSample]) -> Optional[float]:
    """Median per-second pace inside the group (jitter-robust)."""
    if len(group) < 2:
        return None
    speeds = []
    for left, right in zip(group, group[1:]):
        dt = right.effective_time - left.effective_time
        if dt <= 1e-6:
            continue
        speeds.append(math.hypot(right.x - left.x, right.y - left.y) / dt)
    if not speeds:
        return None
    speeds.sort()
    return speeds[len(speeds) // 2]


def _displacement(
    group: Sequence[PositionSample],
) -> Optional[Tuple[float, float]]:
    """Net movement vector of the group (first -> last sample)."""
    if len(group) < 2:
        return None
    dt = group[-1].effective_time - group[0].effective_time
    if dt <= 1e-6:
        return None
    return (group[-1].x - group[0].x, group[-1].y - group[0].y)


def _heading_changed(pre_vec: Tuple[float, float], post_vec: Tuple[float, float]) -> bool:
    pre_mag = math.hypot(pre_vec[0], pre_vec[1])
    post_mag = math.hypot(post_vec[0], post_vec[1])
    if pre_mag < REACTION_MIN_DISPLACEMENT or post_mag < REACTION_MIN_DISPLACEMENT:
        return False
    dot = pre_vec[0] * post_vec[0] + pre_vec[1] * post_vec[1]
    cosine = max(-1.0, min(1.0, dot / (pre_mag * post_mag)))
    return math.degrees(math.acos(cosine)) >= REACTION_ANGLE_DEG


def _speed_changed(pre_speed: float, post_speed: float) -> bool:
    if pre_speed < REACTION_STANDSTILL_SPEED:
        # Was standing still: starting to move is the response.
        return post_speed >= REACTION_STANDSTILL_SPEED
    if post_speed >= pre_speed * REACTION_SPEED_UP:
        return True
    return post_speed <= pre_speed * REACTION_SLOW_DOWN


def reaction_after_kills(
    samples: Sequence[PositionSample],
    kill_times: Sequence[float],
    window_end: float,
    pre_sec: float = REACTION_PRE_SEC,
    post_sec: float = REACTION_POST_SEC,
) -> Tuple[int, List[float]]:
    """Measure how fast movement changes after each scoreboard kill.

    A kill is *eligible* when both sides of the event have at least
    ``REACTION_MIN_SIDE`` samples.  A kill *responds* when, within
    ``post_sec`` after the event, either the sustained pace differs from the
    pre-event median pace by ``REACTION_SPEED_UP`` / ``REACTION_SLOW_DOWN``
    (or the player starts from standstill) or the net heading swings at least
    ``REACTION_ANGLE_DEG``.  The first sample where that holds gives the
    latency.

    Returns:
        ``(eligible_count, response_latencies_seconds)`` - latencies only
        contain responded events, so ``len(latencies) <= eligible_count``.
    """
    ordered = sorted(samples, key=lambda sample: sample.effective_time)
    eligible = 0
    latencies: List[float] = []
    for kill in kill_times:
        if kill < 0.0 or kill > window_end:
            continue
        pre = [
            s for s in ordered
            if kill - pre_sec <= s.effective_time <= kill - 1.0
        ]
        post = [
            s for s in ordered
            if kill + 1.0 <= s.effective_time <= kill + post_sec
        ]
        if len(pre) < REACTION_MIN_SIDE or len(post) < REACTION_MIN_SIDE:
            continue
        pre_speed = _median_speed(pre)
        pre_vec = _displacement(pre)
        if pre_speed is None or pre_vec is None:
            continue
        eligible += 1
        for index in range(REACTION_MIN_SIDE - 1, len(post)):
            window = post[: index + 1]
            post_speed = _median_speed(window)
            post_vec = _displacement(window)
            if post_speed is None or post_vec is None:
                continue
            if _speed_changed(pre_speed, post_speed) or _heading_changed(pre_vec, post_vec):
                latencies.append(window[-1].effective_time - kill)
                break
    return eligible, latencies

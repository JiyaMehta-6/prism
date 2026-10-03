"""Roaming analytics.

Turns the raw roam events detected by the behaviour engine into a strategic
picture: how often the player leaves lane, when they do it, where they go and
which routes they repeat (e.g. ``Mid -> River -> Top``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from core.logger import get_logger
from core.models import Fingerprint, PlayerProfile, RoamEvent, VideoAnalysis

logger = get_logger("roaming")

BUCKET_EDGES = (0.0, 3.0, 7.0, 12.0, 15.0)


@dataclass
class RoamingAnalysis:
    """Roam frequency, timing and route statistics."""

    score: float
    roam_rate: float
    total_roams: int
    destinations: List[Tuple[str, int]] = field(default_factory=list)
    routes: List[Tuple[str, int]] = field(default_factory=list)
    timing: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.5
    summary: str = ""
    notes: List[str] = field(default_factory=list)


def _bucket_label(minutes: float, edges: Sequence[float]) -> str:
    if minutes < edges[0]:
        return f"<{edges[0]:.0f} min"
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        if index == len(edges) - 2:
            if low <= minutes <= high:
                return f"{low:.0f}-{high:.0f} min"
        elif low <= minutes < high:
            return f"{low:.0f}-{high:.0f} min"
    return f">{edges[-1]:.0f} min"


def analyze_roaming(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    analyses: Sequence[VideoAnalysis] = (),
    edges: Optional[Sequence[float]] = None,
) -> RoamingAnalysis:
    """Build the roaming section of the report.

    ``edges`` are the timeline window boundaries in minutes so roam timing
    buckets always match the configured ``window_plan``.
    """
    bucket_edges: Tuple[float, ...] = (
        tuple(float(v) for v in edges) if edges and len(edges) >= 2 else BUCKET_EDGES
    )
    events: List[RoamEvent] = []
    timing: Dict[str, int] = {f"{bucket_edges[i]:.0f}-{bucket_edges[i + 1]:.0f} min": 0
                              for i in range(len(bucket_edges) - 1)}

    for analysis in analyses:
        for event in analysis.roam_events:
            events.append(event)
            start_minutes = event.start_time / 60.0
            label = _bucket_label(start_minutes, bucket_edges)
            timing[label] = timing.get(label, 0) + 1

    destinations = sorted(profile.roam_destinations.items(), key=lambda kv: kv[1], reverse=True)
    routes = sorted(profile.roam_paths.items(), key=lambda kv: kv[1], reverse=True)

    score = fingerprint.get("Roaming")
    confidence = fingerprint.confidence.get("Roaming", 0.5)

    notes: List[str] = []
    if destinations:
        top_region, top_count = destinations[0]
        notes.append(
            f"Most frequent roam destination: {top_region} ({top_count} of {len(events)} rotations)."
        )
    if routes:
        notes.append(f"Recurring route: {routes[0][0]} (used {routes[0][1]} times).")
    if timing:
        busiest = max(timing, key=timing.get)  # type: ignore[arg-type]
        count = timing[busiest]
        if count >= 2:
            notes.append(f"Rotations cluster in the {busiest} window ({count} events).")
        elif count == 1:
            notes.append(f"The single recorded rotation falls in the {busiest} window.")
    if not events:
        notes.append("No sustained lane exits detected inside the analysed window.")

    top_destination = destinations[0][0] if destinations else None
    summary = _summarise(
        score,
        profile.roam_rate,
        profile.average_metrics.get("lane_fraction", 0.0),
        top_destination,
    )
    analysis = RoamingAnalysis(
        score=score,
        roam_rate=profile.roam_rate,
        total_roams=len(events),
        destinations=destinations[:6],
        routes=routes[:6],
        timing=timing,
        confidence=confidence,
        summary=summary,
        notes=notes,
    )
    logger.debug("Roaming analysis: %d roams, %.2f/min", len(events), profile.roam_rate)
    return analysis


def _summarise(
    score: float, rate: float, lane_fraction: float, top_destination: Optional[str]
) -> str:
    if score >= 7.5:
        if rate >= 0.35:
            tail = f", most often toward {top_destination}" if top_destination else ""
            return (
                f"Aggressive roaming profile ({score:.1f}/10): {rate:.2f} rotations per minute, "
                "with the player repeatedly leaving lane to influence other areas of the map"
                f"{tail}."
            )
        return (
            f"Aggressive roaming profile ({score:.1f}/10): the score is driven by time outside "
            f"the assigned lane ({(1 - lane_fraction) * 100:.0f}% of samples) rather than "
            f"rotation count ({rate:.2f}/min)."
        )
    if score >= 5.0:
        tail = f", most often toward {top_destination}" if top_destination else ""
        return (
            f"Active roaming profile ({score:.1f}/10): rotations happen regularly "
            f"({rate:.2f}/min){tail}."
        )
    if score >= 3.0:
        if rate <= 0.0:
            return (
                f"Situational roaming ({score:.1f}/10): no sustained lane exits were detected "
                "- the score reflects time spent outside the assigned lane."
            )
        return (
            f"Situational roaming ({score:.1f}/10): the player leaves lane only when the "
            f"situation clearly warrants it ({rate:.2f} rotations per minute)."
        )
    if lane_fraction >= 0.55:
        return (
            f"Lane-anchored profile ({score:.1f}/10): rotations are rare ({rate:.2f}/min) and "
            "the player derives most early-game value from staying in lane."
        )
    return (
        f"Lane-anchored profile ({score:.1f}/10): rotations are rare ({rate:.2f}/min) despite "
        f"limited lane presence ({lane_fraction * 100:.0f}%)."
    )

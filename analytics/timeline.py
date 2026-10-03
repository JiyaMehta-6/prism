"""Tactical timeline of the early game.

The 0-15 minute window is sliced into configurable segments (default
0-3, 3-7, 7-12, 12-15).  For every segment PRISM measures the region
distribution, the forward-positioning index and the rotation count, then
labels the segment with the behavioural mode that best explains those numbers:

* Conservative Laning
* Aggressive Trading
* Frequent Roaming
* Objective Preparation
* Transitional Play

Labels always carry a confidence derived from the amount of evidence in the
segment, so thin segments never masquerade as conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from core.logger import get_logger
from core.models import PositionSample, TimelineWindow, VideoAnalysis
from vision.region_mapper import RegionMapper

logger = get_logger("timeline")

DEFAULT_EDGES: Tuple[float, ...] = (0.0, 3.0, 7.0, 12.0, 15.0)


@dataclass
class TimelineResult:
    """Timeline plus per-window statistics."""

    windows: List[TimelineWindow]
    confidence: float
    summary: str


def build_timeline(
    analyses: Sequence[VideoAnalysis],
    edges: Sequence[float] = DEFAULT_EDGES,
    sample_floor: int = 6,
    own_base: str = "Blue",
    anchors: Dict[str, object] | None = None,
) -> List[TimelineWindow]:
    """Build the behavioural timeline across all recordings."""
    try:
        cleaned = sorted({float(value) for value in edges})
    except (TypeError, ValueError):
        cleaned = list(DEFAULT_EDGES)
    edges = tuple(cleaned) if len(cleaned) >= 3 else DEFAULT_EDGES
    windows: List[TimelineWindow] = []
    last_index = len(edges) - 2
    video_count = max(1, len(analyses))

    for index in range(len(edges) - 1):
        start, end = edges[index], edges[index + 1]
        is_last = index == last_index
        window_parts: List[Tuple[List[PositionSample], str]] = []
        roams = 0
        kills = 0
        for analysis in analyses:
            base = analysis_own_base(analysis, own_base)
            if is_last:
                in_window = lambda s: start * 60 <= s.effective_time <= end * 60  # noqa: E731
                in_roam = lambda e: start * 60 <= e.start_time <= end * 60  # noqa: E731
                in_event = lambda e: start * 60 <= e.time <= end * 60  # noqa: E731
            else:
                in_window = lambda s: start * 60 <= s.effective_time < end * 60  # noqa: E731
                in_roam = lambda e: start * 60 <= e.start_time < end * 60  # noqa: E731
                in_event = lambda e: start * 60 <= e.time < end * 60  # noqa: E731
            window_parts.append(([s for s in analysis.samples if in_window(s)], base))
            roams += sum(1 for event in analysis.roam_events if in_roam(event))
            kills += sum(
                event.count
                for event in getattr(analysis, "events", [])
                if event.kind == "kill" and in_event(event)
            )

        samples = [s for part, _ in window_parts for s in part]
        distribution = _distribution(samples)
        avg_roams = roams / video_count
        aggression_index = _aggression_index(window_parts, avg_roams, anchors)
        label, summary, confidence = _label_segment(
            distribution, aggression_index, roams, avg_roams, video_count,
            len(samples), sample_floor, start, kills
        )

        windows.append(
            TimelineWindow(
                start=start * 60,
                end=end * 60,
                label=label,
                summary=summary,
                confidence=confidence,
                region_distribution=distribution,
                aggression_index=aggression_index,
                kills=kills,
            )
        )

    logger.info(
        "Timeline built: %s",
        " | ".join(f"{w.start_label}-{w.end_label}: {w.label}" for w in windows),
    )
    return windows


def _distribution(samples: Sequence[PositionSample]) -> Dict[str, float]:
    if not samples:
        return {}
    counts: Dict[str, float] = {}
    for sample in samples:
        counts[sample.region] = counts.get(sample.region, 0.0) + 1.0
    total = sum(counts.values())
    return {region: value / total for region, value in counts.items()}


def analysis_own_base(analysis: VideoAnalysis, fallback: str) -> str:
    """Per-video own base (mixed-side batches must not mirror each other)."""
    value = analysis.metrics.get("own_base_blue")
    if value is None:
        return fallback
    try:
        return "Blue" if float(value) >= 0.5 else "Red"
    except (TypeError, ValueError):
        return fallback


def _aggression_index(
    parts: Sequence[Tuple[List[PositionSample], str]],
    avg_roams: float,
    anchors: Dict[str, object] | None = None,
) -> float:
    mapper = RegionMapper(anchors)
    bias_sum = 0.0
    forward_count = 0
    total = 0
    for samples, base in parts:
        for sample in samples:
            bias = mapper.forward_bias(sample.x, sample.y, base)
            bias_sum += bias
            if bias > 0.3:
                forward_count += 1
            total += 1
    if total == 0:
        return 0.0
    bias = bias_sum / total
    forward_share = forward_count / total
    roam_share = min(1.0, avg_roams / 3.0)
    raw = 0.45 * ((bias + 1.0) / 2.0) + 0.35 * forward_share + 0.20 * roam_share
    return round(max(0.0, min(1.0, raw)) * 10.0, 2)


def _label_segment(
    distribution: Dict[str, float],
    aggression_index: float,
    roams: int,
    avg_roams: float,
    video_count: int,
    sample_count: int,
    sample_floor: int,
    start: float,
    kills: int = 0,
) -> Tuple[str, str, float]:
    label, summary, confidence = _label_segment_core(
        distribution, aggression_index, roams, avg_roams, video_count,
        sample_count, sample_floor, start,
    )
    if kills > 0:
        noun = "kill" if kills == 1 else "kills"
        summary = f"{summary} {kills} team {noun} observed in this window."
    return label, summary, confidence


def _label_segment_core(
    distribution: Dict[str, float],
    aggression_index: float,
    roams: int,
    avg_roams: float,
    video_count: int,
    sample_count: int,
    sample_floor: int,
    start: float,
) -> Tuple[str, str, float]:
    if sample_count < sample_floor:
        return (
            "Insufficient Data",
            f"Only {sample_count} position samples in this segment - conclusion withheld.",
            0.30,
        )

    lane_share = sum(distribution.get(region, 0.0) for region in ("Top Lane", "Mid Lane", "Bot Lane"))
    river_share = distribution.get("River", 0.0)
    jungle_share = distribution.get("Blue Jungle", 0.0) + distribution.get("Red Jungle", 0.0)
    objective_share = distribution.get("Dragon Area", 0.0) + distribution.get("Herald Area", 0.0)
    evidence = min(1.0, sample_count / 60.0)
    # Lower intercept than before: a handful of samples can no longer reach
    # "conclusion" confidence (0.45+) - thin segments start near 0.37 instead
    # of 0.50 and grow continuously with evidence.
    base_confidence = round(0.30 + 0.65 * evidence, 3)

    if avg_roams >= 2.0 and lane_share < 0.90:
        video_word = "video" if video_count == 1 else "videos"
        return (
            "Frequent Roaming",
            f"{roams} rotations in {video_count} {video_word} "
            f"({avg_roams:.1f} per video) left the lane while regional spread shows "
            f"{lane_share * 100:.0f}% lane presence - the player is actively "
            "transferring pressure around the map.",
            min(0.93, base_confidence + 0.08),
        )
    if objective_share >= 0.12 or (start >= 7.0 and river_share >= 0.18):
        return (
            "Objective Preparation",
            f"{(objective_share + river_share) * 100:.0f}% of time inside objective or river "
            "zones indicates setup play before neutral objectives.",
            min(0.92, base_confidence + 0.05),
        )
    if aggression_index >= 6.0:
        return (
            "Aggressive Trading",
            f"Forward-positioning index {aggression_index:.1f}/10 with "
            f"{lane_share * 100:.0f}% lane presence - the player presses from "
            "forward positions.",
            base_confidence,
        )
    if lane_share >= 0.55 and aggression_index <= 4.5:
        return (
            "Conservative Laning",
            f"Lane presence {lane_share * 100:.0f}% with a low forward index "
            f"({aggression_index:.1f}/10) - a stable laning pattern.",
            base_confidence,
        )
    if river_share + jungle_share > 0.15:
        summary = (
            f"Movement mixes lanes ({lane_share * 100:.0f}%), river "
            f"({river_share * 100:.0f}%) and jungle ({jungle_share * 100:.0f}%) "
            "without a single dominant pattern."
        )
    else:
        top_region, top_share = (
            max(distribution.items(), key=lambda kv: kv[1]) if distribution else ("-", 0.0)
        )
        summary = (
            f"No dominant pattern: largest shares are {top_region} "
            f"{top_share * 100:.0f}% and lanes {lane_share * 100:.0f}%."
        )
    return (
        "Transitional Play",
        summary,
        max(0.40, base_confidence - 0.08),
    )

"""Match-to-match consistency analytics.

Consistency answers a simple question: *does this player behave the same way
in every recording?*  Two complementary measurements are combined:

1. **Metric dispersion** - coefficient of variation of the core behavioural
   indicators across videos (aggression, lane share, roaming rate, ...).
2. **Regional overlap** - per-video region distributions are compared pairwise
   with the Jensen-Shannon divergence; low divergence means the player
   repeats the same map pattern.

The final score is corroborated by the fingerprint's consistency axis so the
GUI, the PDF and the JSON export always agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from core.fingerprint_engine import CONSISTENCY_METRICS, metric_variability
from core.logger import get_logger
from core.models import Fingerprint, PlayerProfile, VideoAnalysis

logger = get_logger("consistency")


@dataclass
class ConsistencyAnalysis:
    """Stability of behaviour across recordings."""

    score: float
    metric_dispersion: float
    regional_overlap: float
    pairwise: List[Tuple[str, str, float]] = field(default_factory=list)
    confidence: float = 0.5
    summary: str = ""
    notes: List[str] = field(default_factory=list)


def analyze_consistency(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    _analyses: Sequence[VideoAnalysis] = (),
) -> ConsistencyAnalysis:
    """Build the consistency section of the report."""
    score = fingerprint.get("Consistency")
    confidence = fingerprint.confidence.get("Consistency", 0.5)

    # Same unweighted CV computation as the fingerprint's consistency axis so
    # the two reported numbers can never disagree.
    dispersion = metric_variability(profile, CONSISTENCY_METRICS)

    distributions = _extract_regions(profile)
    pairwise: List[Tuple[str, str, float]] = []
    overlaps: List[float] = []
    names = list(distributions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = distributions[names[i]]
            b = distributions[names[j]]
            similarity = 1.0 - _jensen_shannon(a, b)
            pairwise.append((names[i], names[j], similarity))
            overlaps.append(similarity)
    overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    notes: List[str] = []
    if len(names) < 2:
        notes.append(
            "Only one recording analysed - consistency will be validated once more "
            "videos of the same player are added."
        )
    else:
        notes.append(
            f"Mean regional overlap between recordings (Jensen-Shannon): "
            f"{overlap * 100:.0f}% over {len(pairwise)} pairs."
        )
        notes.append(
            f"Average metric dispersion: {dispersion:.2f} "
            "(coefficient of variation, lower is steadier)."
        )
        best = max(pairwise, key=lambda item: item[2]) if pairwise else None
        worst = min(pairwise, key=lambda item: item[2]) if pairwise else None
        if best:
            notes.append(f"Most similar pair: {best[0]} & {best[1]} ({best[2] * 100:.0f}%).")
        if worst and best and len(pairwise) > 1 and worst[2] < best[2]:
            notes.append(f"Most different pair: {worst[0]} & {worst[1]} ({worst[2] * 100:.0f}%).")

    summary = _summarise(score, overlap, len(names))
    analysis = ConsistencyAnalysis(
        score=score,
        metric_dispersion=dispersion,
        regional_overlap=overlap,
        pairwise=pairwise,
        confidence=confidence,
        summary=summary,
        notes=notes,
    )
    _log(profile, fingerprint, dispersion, overlap)
    return analysis


def _extract_regions(profile: PlayerProfile) -> Dict[str, Dict[str, float]]:
    regions: Dict[str, Dict[str, float]] = {}
    for name, metrics in profile.per_video_metrics.items():
        regions[name] = {
            key[len("region_") :]: value
            for key, value in metrics.items()
            if key.startswith("region_")
        }
    return regions


def _jensen_shannon(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 1.0
    p_vals = [max(0.0, p.get(k, 0.0)) for k in keys]
    q_vals = [max(0.0, q.get(k, 0.0)) for k in keys]
    p_sum = sum(p_vals) or 1.0
    q_sum = sum(q_vals) or 1.0
    p_vals = [v / p_sum for v in p_vals]
    q_vals = [v / q_sum for v in q_vals]
    divergence = 0.0
    for pi, qi in zip(p_vals, q_vals):
        m = 0.5 * (pi + qi)
        if pi > 0:
            divergence += 0.5 * pi * math.log2(pi / m)
        if qi > 0:
            divergence += 0.5 * qi * math.log2(qi / m)
    return max(0.0, min(1.0, divergence))


def _summarise(score: float, overlap: float, video_count: int) -> str:
    if video_count < 2:
        return (
            "Consistency cannot be fully validated yet - add more recordings of the same "
            "player to measure match-to-match stability."
        )
    if score >= 7.0:
        return (
            f"Highly consistent player ({score:.1f}/10): the core metrics barely move "
            f"between games; mean regional overlap {overlap * 100:.0f}%."
        )
    if score >= 4.5:
        return (
            f"Moderately consistent ({score:.1f}/10): the core identity holds while "
            f"individual metrics drift between games; mean regional overlap "
            f"{overlap * 100:.0f}%."
        )
    return (
        f"Variable performances ({score:.1f}/10): the core metrics shift noticeably "
        f"between recordings; mean regional overlap {overlap * 100:.0f}%."
    )


def _log(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    dispersion: float,
    overlap: float,
) -> None:
    logger.info(
        "Consistency: score=%.1f (conf %.2f), dispersion=%.2f, overlap=%.0f%%, videos=%d",
        fingerprint.get("Consistency"),
        fingerprint.confidence.get("Consistency", 0.0),
        dispersion,
        overlap * 100,
        profile.video_count,
    )

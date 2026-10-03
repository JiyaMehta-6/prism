"""Aggression analytics.

Quantifies how confrontationally a player operates during the first 15
minutes from four movement-derived indicators:

* **forward positioning frequency** - share of samples on the enemy side of the
  mid-line diagonal,
* **engagement tendency** - share of samples spent in high-speed bursts and
  rapid direction changes (proxies for trades / all-ins),
* **risk exposure** - time spent inside enemy territory and in the river,
* **roaming frequency** - rotations leaving the assigned lane.

Everything is computed from tracked minimap positions, so no telemetry is
required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from core.logger import get_logger
from core.models import Fingerprint, PlayerProfile, VideoAnalysis

logger = get_logger("aggression")


@dataclass
class AggressionAnalysis:
    """Explainable aggression breakdown."""

    score: float
    confidence: float
    summary: str
    notes: List[str] = field(default_factory=list)


def analyze_aggression(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    _analyses: Sequence[VideoAnalysis] = (),
) -> AggressionAnalysis:
    """Build the aggression section of the report."""
    metrics = profile.average_metrics
    forward = metrics.get("forward_high_fraction", 0.0)
    engagement = 0.55 * metrics.get("burst_fraction", 0.0) + 0.45 * min(
        1.0, metrics.get("erratic_turn_rate", 0.0) / 0.4
    )
    risk = metrics.get("enemy_territory_fraction", 0.0)
    roam = profile.roam_rate

    score = fingerprint.get("Aggression")
    confidence = fingerprint.confidence.get("Aggression", 0.5)

    notes: List[str] = []
    if forward > 0.40:
        notes.append(
            f"Frequent forward positioning: {forward * 100:.0f}% of tracked samples hold "
            "a clearly forward position (forward bias > 0.35)."
        )
    elif forward < 0.15:
        notes.append(
            f"Conservative positioning: only {forward * 100:.0f}% of samples reach a "
            "clearly forward position (forward bias > 0.35)."
        )
    else:
        notes.append(
            f"Balanced forward positioning at {forward * 100:.0f}% of samples "
            "(forward bias > 0.35)."
        )

    if risk > 0.30:
        notes.append(f"High risk exposure: {risk * 100:.0f}% of time inside enemy territory.")
    if roam >= 0.35:
        notes.append(f"Rotations occur {roam:.2f} times per minute of early game.")
    if engagement > 0.5:
        notes.append("Burst movement and sharp heading changes suggest frequent short trades.")

    summary = _summarise(score, forward, roam)
    analysis = AggressionAnalysis(
        score=score,
        confidence=confidence,
        summary=summary,
        notes=notes,
    )
    logger.debug("Aggression analysis: score=%.1f conf=%.2f", score, confidence)
    return analysis


def _summarise(score: float, forward: float, roam: float) -> str:
    if forward > 0.40:
        forward_phrase = f"repeatedly takes forward positions ({forward * 100:.0f}% of samples)"
    elif forward >= 0.15:
        forward_phrase = f"holds forward positions selectively ({forward * 100:.0f}% of samples)"
    else:
        forward_phrase = f"rarely holds clearly forward positions ({forward * 100:.0f}% of samples)"
    if roam >= 0.35:
        rotation_phrase = f"rotates regularly ({roam:.2f} rotations per minute)"
    else:
        rotation_phrase = f"rotations stay rare ({roam:.2f} per minute)"
    if score >= 7.5:
        return (
            f"Highly aggressive early profile ({score:.1f}/10): the player {forward_phrase} "
            f"and {rotation_phrase}, accepting trades and territory risk to create pressure."
        )
    if score >= 5.5:
        return (
            f"Moderately aggressive profile ({score:.1f}/10): the player {forward_phrase} "
            f"and {rotation_phrase}, balanced by safe moments in own territory."
        )
    if score >= 3.5:
        return (
            f"Controlled aggression ({score:.1f}/10): the player {forward_phrase} "
            f"and {rotation_phrase}, favouring stable farm-oriented positions."
        )
    return (
        f"Passive early profile ({score:.1f}/10): the player {forward_phrase} "
        f"and {rotation_phrase}, prioritising safety over map pressure."
    )

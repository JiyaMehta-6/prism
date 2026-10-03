"""Objective-control analytics.

Estimates how actively the player participates in objective-related play by
measuring *where* they stand:

* dwell time inside the Dragon and Herald pits,
* rotations whose destination is an objective area,
* proximity to outer turrets (turret plates / early sieges),
* river presence, the approach corridor for every neutral objective.

Because PRISM only sees the minimap, involvement is a well-founded estimate
rather than a kill/assist fact - the confidence score reflects that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from core.logger import get_logger
from core.models import Fingerprint, PlayerProfile, VideoAnalysis

logger = get_logger("objectives")


@dataclass
class ObjectivesAnalysis:
    """Objective involvement estimate for the early game."""

    score: float
    dragon_fraction: float
    herald_fraction: float
    tower_fraction: float
    river_fraction: float
    objective_rotations: int = 0
    confidence: float = 0.5
    summary: str = ""
    notes: List[str] = field(default_factory=list)


def analyze_objectives(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    analyses: Sequence[VideoAnalysis] = (),
) -> ObjectivesAnalysis:
    """Build the objectives section of the report."""
    objective_rotations = 0

    for analysis in analyses:
        objective_rotations += sum(
            1
            for event in analysis.roam_events
            if event.destination_region in ("Dragon Area", "Herald Area")
        )

    analyzed_minutes = sum(
        float(analysis.metrics.get("analyzed_minutes", 0.0)) for analysis in analyses
    )
    metrics = profile.average_metrics
    score = fingerprint.get("Objectives")
    confidence = fingerprint.confidence.get("Objectives", 0.5)

    notes: List[str] = []
    dragon_fraction = profile.region_distribution.get("Dragon Area", 0.0)
    herald_fraction = profile.region_distribution.get("Herald Area", 0.0)
    tower_fraction = metrics.get("tower_presence_fraction", 0.0)
    river_fraction = profile.region_distribution.get("River", 0.0)

    if dragon_fraction + herald_fraction < 0.02:
        notes.append(
            "Almost no time inside the dragon or herald pits during the analysed window."
        )
    else:
        notes.append(
            f"Objective-pit presence: dragon {dragon_fraction * 100:.1f}%, "
            f"herald {herald_fraction * 100:.1f}% of tracked time."
        )
    if objective_rotations:
        notes.append(f"{objective_rotations} rotations ended directly at an objective pit.")
    if tower_fraction > 0.25:
        notes.append(
            f"Spends {tower_fraction * 100:.0f}% of samples near outer turrets, "
            "consistent with plate pressure or early sieges."
        )
    elif tower_fraction < 0.08:
        notes.append("Limited presence around outer turrets in the first 15 minutes.")
    if river_fraction > 0.15:
        notes.append(
            f"Strong river presence ({river_fraction * 100:.0f}%) - the approach corridor "
            "for dragon and herald contests."
        )

    summary = _summarise(score, dragon_fraction + herald_fraction, tower_fraction)
    analysis = ObjectivesAnalysis(
        score=score,
        dragon_fraction=dragon_fraction,
        herald_fraction=herald_fraction,
        tower_fraction=tower_fraction,
        river_fraction=river_fraction,
        objective_rotations=objective_rotations,
        confidence=confidence,
        summary=summary,
        notes=notes,
    )
    logger.debug(
        "Objectives: dragon=%.3f herald=%.3f tower=%.3f (from %s analyzed minutes)",
        dragon_fraction, herald_fraction, tower_fraction, round(analyzed_minutes, 1),
    )
    return analysis


def _summarise(score: float, pit_fraction: float, tower_fraction: float) -> str:
    if score >= 7.0:
        return (
            f"Objective-minded player ({score:.1f}/10): visible around neutral pits "
            f"({pit_fraction * 100:.1f}% of time) and turret lines ({tower_fraction * 100:.0f}%), "
            "indicating proactive setup before objectives spawn."
        )
    if score >= 4.5:
        return (
            f"Moderate objective involvement ({score:.1f}/10): the player shows up for "
            "contested areas when fights break out but does not always pre-position."
        )
    return (
        f"Low objective focus ({score:.1f}/10): limited time around dragon, herald and "
        "turret zones in the first 15 minutes."
    )

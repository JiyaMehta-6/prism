"""Explainable player-archetype classification.

PRISM never prints a label without justification.  Every archetype is a
transparent rule evaluated against the fingerprint and the region statistics;
when a rule fires, the report shows *which numbers* triggered it and by how
much.  Several archetypes may apply at once - most players are a blend - and
they are ordered by strength.

The confidence of a classification combines:

* how far the measurements sit past the rule threshold (margin),
* how trustworthy the underlying vision pipeline was (detection rate),
* how much evidence was available (sample and video counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from core.logger import get_logger
from core.models import Archetype, Fingerprint, PlayerProfile

logger = get_logger("archetypes")

RuleFn = Callable[[PlayerProfile, Fingerprint], Tuple[bool, List[str], float]]


@dataclass(frozen=True)
class ArchetypeRule:
    label: str
    description: str
    evaluate: RuleFn


def _margin(value: float, threshold: float, scale: float) -> float:
    return max(0.0, min(1.0, (value - threshold) / scale))


def _aggressive_roamer(profile: PlayerProfile, fingerprint: Fingerprint) -> Tuple[bool, List[str], float]:
    aggression = fingerprint.get("Aggression")
    roaming = fingerprint.get("Roaming")
    matched = aggression >= 6.3 and roaming >= 6.3
    reasons = [
        f"Aggression {aggression:.1f}/10 (threshold 6.3) and Roaming {roaming:.1f}/10 (threshold 6.3)",
        f"{profile.roam_rate:.2f} rotations per minute with "
        f"{profile.average_metrics.get('enemy_territory_fraction', 0) * 100:.0f}% enemy-side presence",
    ]
    strength = 0.5 * _margin(aggression, 6.3, 3.0) + 0.5 * _margin(roaming, 6.3, 3.0)
    return matched, reasons, strength


def _objective_controller(profile: PlayerProfile, fingerprint: Fingerprint) -> Tuple[bool, List[str], float]:
    objectives = fingerprint.get("Objectives")
    matched = objectives >= 6.5
    dragon = profile.region_distribution.get("Dragon Area", 0.0)
    herald = profile.region_distribution.get("Herald Area", 0.0)
    reasons = [
        f"Objectives {objectives:.1f}/10 (threshold 6.5)",
        f"Dragon presence {dragon * 100:.1f}%, Herald presence {herald * 100:.1f}%",
        f"Outer-turret proximity {profile.average_metrics.get('tower_presence_fraction', 0) * 100:.0f}%",
    ]
    return matched, reasons, _margin(objectives, 6.5, 3.0)


def _lane_dominator(profile: PlayerProfile, fingerprint: Fingerprint) -> Tuple[bool, List[str], float]:
    lane = profile.average_metrics.get("lane_fraction", 0.0)
    roaming = fingerprint.get("Roaming")
    matched = lane >= 0.55 and roaming <= 4.8
    reasons = [
        f"Lane presence {lane * 100:.0f}% (threshold 55%)",
        f"Roaming {roaming:.1f}/10 stays below 4.8 - priority is converted into lane pressure",
    ]
    strength = 0.5 * _margin(lane, 0.55, 0.30) + 0.5 * _margin(4.8 - roaming, 0.0, 3.0)
    return matched, reasons, strength


def _safe_farmer(profile: PlayerProfile, fingerprint: Fingerprint) -> Tuple[bool, List[str], float]:
    risk = fingerprint.get("Risk")
    aggression = fingerprint.get("Aggression")
    matched = risk <= 4.2 and aggression <= 5.2
    reasons = [
        f"Risk {risk:.1f}/10 (ceiling 4.2) and Aggression {aggression:.1f}/10 (ceiling 5.2)",
        f"Enemy-territory exposure {profile.average_metrics.get('enemy_territory_fraction', 0) * 100:.0f}%",
    ]
    strength = 0.5 * _margin(4.2 - risk, 0.0, 3.0) + 0.5 * _margin(5.2 - aggression, 0.0, 3.0)
    return matched, reasons, strength


def _team_strategist(profile: PlayerProfile, fingerprint: Fingerprint) -> Tuple[bool, List[str], float]:
    objectives = fingerprint.get("Objectives")
    consistency = fingerprint.get("Consistency")
    aggression = fingerprint.get("Aggression")
    # Consistency cannot be verified from a single recording - the score is
    # pinned to ~6.0 in that case, so requiring it would be vacuous.
    matched = (
        profile.video_count >= 2
        and objectives >= 5.8
        and consistency >= 6.0
        and aggression <= 6.8
    )
    reasons = [
        f"Objectives {objectives:.1f}/10, Consistency {consistency:.1f}/10",
        f"Aggression {aggression:.1f}/10 stays controlled - decisions follow the team plan",
        f"Stable across {profile.video_count} recordings ({profile.detection_rate * 100:.0f}% detection)",
    ]
    strength = (
        0.4 * _margin(objectives, 5.8, 3.0)
        + 0.4 * _margin(consistency, 6.0, 3.0)
        + 0.2 * _margin(6.8 - aggression, 0.0, 3.0)
    )
    return matched, reasons, strength


def _vision_focused(profile: PlayerProfile, fingerprint: Fingerprint) -> Tuple[bool, List[str], float]:
    vision = fingerprint.get("Vision")
    matched = vision >= 6.3
    reasons = [
        f"Vision proxy {vision:.1f}/10 (threshold 6.3)",
        f"River presence {profile.region_distribution.get('River', 0) * 100:.0f}% and "
        f"corridor presence {profile.average_metrics.get('vision_proxy_fraction', 0) * 100:.0f}%",
        "Vision is estimated from movement around warding corridors (no telemetry used)",
    ]
    return matched, reasons, _margin(vision, 6.3, 3.0)


def _high_risk_playmaker(profile: PlayerProfile, fingerprint: Fingerprint) -> Tuple[bool, List[str], float]:
    risk = fingerprint.get("Risk")
    aggression = fingerprint.get("Aggression")
    matched = risk >= 6.8 and aggression >= 6.2
    reasons = [
        f"Risk {risk:.1f}/10 (threshold 6.8) with Aggression {aggression:.1f}/10 (threshold 6.2)",
        f"Late-window enemy exposure {profile.average_metrics.get('late_enemy_fraction', 0) * 100:.0f}%",
        f"Erratic heading changes {profile.average_metrics.get('erratic_turn_rate', 0):.2f} per step",
    ]
    strength = 0.5 * _margin(risk, 6.8, 3.0) + 0.5 * _margin(aggression, 6.2, 3.0)
    return matched, reasons, strength


RULES: Tuple[ArchetypeRule, ...] = (
    ArchetypeRule(
        "Aggressive Roamer",
        "Leaves lane early and often to create numbers advantages elsewhere.",
        _aggressive_roamer,
    ),
    ArchetypeRule(
        "Objective Controller",
        "Times movement around dragon, herald and turret pressure.",
        _objective_controller,
    ),
    ArchetypeRule(
        "Lane Dominator",
        "Converts lane priority into plates and CS instead of roaming.",
        _lane_dominator,
    ),
    ArchetypeRule(
        "Safe Farmer",
        "Minimises risk and avoids unnecessary early engagements.",
        _safe_farmer,
    ),
    ArchetypeRule(
        "Team-Oriented Strategist",
        "Aligns personal behaviour with team objectives and repeats it every game.",
        _team_strategist,
    ),
    ArchetypeRule(
        "Vision-Focused Player",
        "Repeats warding corridors and river control patterns.",
        _vision_focused,
    ),
    ArchetypeRule(
        "High-Risk Playmaker",
        "Trades safety for game-changing moments.",
        _high_risk_playmaker,
    ),
)


def classify_archetypes(
    profile: PlayerProfile, fingerprint: Fingerprint
) -> List[Archetype]:
    """Return every matching archetype, strongest first, each with reasons."""
    matches: List[Archetype] = []
    if not fingerprint.scores:
        logger.warning("Fingerprint has no scores; returning the fallback archetype")
        return [_balanced_profile(profile, fingerprint)]

    for rule in RULES:
        try:
            matched, reasons, strength = rule.evaluate(profile, fingerprint)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Archetype rule '%s' failed: %s", rule.label, exc)
            continue
        if not matched:
            continue
        confidence = _confidence(profile, strength)
        matches.append(Archetype(label=rule.label, confidence=confidence, reasons=reasons))
        logger.debug("Archetype matched: %s (conf=%.2f)", rule.label, confidence)

    matches.sort(key=lambda a: a.confidence, reverse=True)

    if not matches:
        matches.append(_balanced_profile(profile, fingerprint))
    return matches


def _confidence(profile: PlayerProfile, strength: float) -> float:
    evidence = min(1.0, profile.total_samples / 350.0)
    quality = min(1.0, profile.detection_rate)
    videos = min(1.0, profile.video_count / 3.0)
    raw = 0.45 + 0.30 * strength + 0.15 * evidence + 0.10 * quality
    raw *= 0.75 + 0.25 * (0.5 * videos + 0.5 * quality)
    return round(max(0.35, min(0.96, raw)), 3)


def _balanced_profile(profile: PlayerProfile, fingerprint: Fingerprint) -> Archetype:
    reasons = [
        "No archetype threshold was crossed with enough margin to justify a label",
        "Add more recordings for a sharper classification",
    ]
    if fingerprint.scores:
        strongest = max(fingerprint.scores, key=lambda m: fingerprint.scores[m])
        weakest = min(fingerprint.scores, key=lambda m: fingerprint.scores[m])
        reasons.insert(
            1,
            f"Strongest axis: {strongest} ({fingerprint.scores[strongest]:.1f}/10)",
        )
        reasons.insert(
            2,
            f"Weakest axis: {weakest} ({fingerprint.scores[weakest]:.1f}/10)",
        )
    return Archetype(
        label="Balanced All-Rounder",
        confidence=round(max(0.35, 0.6 * profile.detection_rate), 3),
        reasons=reasons,
    )

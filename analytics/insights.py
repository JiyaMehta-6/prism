"""Improvement-insights engine.

Generates constructive, evidence-backed findings.  Each insight contains:

* **finding** - what the movement data shows,
* **confidence** - derived from fingerprint confidence, detection quality and
  how far the measurement sits from the rule threshold,
* **suggestion** - one concrete, actionable correction,
* **evidence** - the raw numbers a coach can verify.

Rules are deliberately simple and auditable; an unsupported claim is never
emitted.  When no weakness is found, positive findings are reported instead so
the report always has something meaningful to say.
"""

from __future__ import annotations

from typing import List, Sequence

from analytics.awareness import WARD_ACTIVE_LOOKBACK_SEC, WARD_VISION_RADIUS
from core.logger import get_logger
from core.models import Fingerprint, Insight, PlayerProfile

logger = get_logger("insights")


def generate_insights(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    archetypes: Sequence = (),
) -> List[Insight]:
    """Produce an ordered list of actionable findings."""
    insights: List[Insight] = []
    quality = max(0.55, min(1.0, profile.detection_rate))
    archetype_labels = {getattr(a, "label", "") for a in archetypes}

    overextension = None if "Safe Farmer" in archetype_labels else _overextension(
        profile, fingerprint, quality
    )
    if overextension:
        insights.append(overextension)

    kill_conversion = _kill_conversion(profile, fingerprint, quality)
    if kill_conversion:
        insights.append(kill_conversion)

    objectives = _objective_gap(profile, fingerprint, quality)
    if objectives:
        insights.append(objectives)

    vision = _vision_gap(profile, fingerprint, quality)
    if vision:
        insights.append(vision)

    unwarded = _unwarded_aggression(profile, fingerprint, quality)
    if unwarded:
        insights.append(unwarded)

    death_pattern = _death_pattern(profile, fingerprint, quality)
    if death_pattern:
        insights.append(death_pattern)

    slow_reaction = _slow_reaction(profile, fingerprint, quality)
    if slow_reaction:
        insights.append(slow_reaction)

    pressure = _pressure_gap(profile, fingerprint, quality)
    if pressure:
        insights.append(pressure)

    consistency = _consistency_gap(profile, fingerprint, quality)
    if consistency:
        insights.append(consistency)

    # "Under-trading in lane" and the "Lane Dominator" archetype describe the
    # same lane numbers in opposite ways - never emit both.
    passivity = None if "Lane Dominator" in archetype_labels else _passivity_gap(
        profile, fingerprint, quality
    )
    if passivity:
        insights.append(passivity)

    if not insights:
        # Positive findings are reported *instead* of weaknesses (docstring).
        insights.extend(_strengths(profile, fingerprint, quality))

    if not insights:
        insights.append(
            Insight(
                title="Insufficient evidence for targeted advice",
                detail=(
                    "The tracked positions do not deviate enough from neutral behaviour to "
                    "support a specific improvement claim."
                ),
                suggestion="Record two or three more full games and re-run PRISM.",
                confidence=round(0.4 * quality, 3),
                severity="info",
                evidence=[f"Samples: {profile.total_samples}", f"Videos: {profile.video_count}"],
            )
        )

    insights.sort(key=lambda i: i.confidence, reverse=True)
    logger.info("Generated %d insights (top: %s)", len(insights), insights[0].title)
    return insights


def _conf(base: float, fingerprint_conf: float, quality: float) -> float:
    return round(max(0.30, min(0.96, base * (0.6 + 0.4 * fingerprint_conf) * quality)), 3)


def _overextension(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    enemy = profile.average_metrics.get("enemy_territory_fraction", 0.0)
    roam_rate = profile.roam_rate
    late_enemy = profile.average_metrics.get("late_enemy_fraction", 0.0)
    if enemy < 0.28 or roam_rate < 0.25:
        return None
    strength = min(1.0, (enemy - 0.28) / 0.30)
    confidence = _conf(0.62 + 0.25 * strength, fingerprint.confidence.get("Aggression", 0.6), quality)
    return Insight(
        title="Frequent overextension during roams",
        detail=(
            f"{enemy * 100:.0f}% of tracked time is spent inside the enemy half while "
            f"rotating at {roam_rate:.2f} times per minute"
            + (f", and exposure stays at {late_enemy * 100:.0f}%" if late_enemy > 0.3 else "")
            + ". Deep positioning without lane priority turns roam attempts into free kills "
            "for the opponent."
        ),
        suggestion="Push the wave to the enemy tower before roaming so the lane punishes "
        "their roam, and retreat once the wave bounces back.",
        confidence=confidence,
        severity="risk",
        evidence=[
            f"Enemy-side presence: {enemy * 100:.1f}%",
            f"Roam rate: {roam_rate:.2f}/min",
            f"Late-window enemy-side presence: {late_enemy * 100:.1f}%",
        ],
    )


def _kill_conversion(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    """Roams that never overlap a team kill are time spent without payoff."""
    metrics = profile.average_metrics
    if "team_kills_for" not in metrics:
        # OCR scoreboard was never read: no evidence, no claim.
        return None
    kills_for = metrics.get("team_kills_for", 0.0)
    during = metrics.get("kills_during_roams", 0.0)
    roam_rate = profile.roam_rate
    if kills_for < 2.0 or roam_rate < 0.25:
        return None
    share = max(0.0, min(1.0, during / kills_for))
    if share >= 0.34:
        return None
    strength = min(1.0, (0.34 - share) / 0.34)
    confidence = _conf(
        0.55 + 0.20 * strength, fingerprint.confidence.get("Roaming", 0.6), quality
    )
    return Insight(
        title="Roams rarely convert into kills",
        detail=(
            f"Your team scored {kills_for:.1f} kills on average but only "
            f"{during:.1f} of them ({share * 100:.0f}%) landed while you were rotating "
            f"between lanes (roam rate {roam_rate:.2f}/min). Rotations that arrive "
            "after the fight resolves cost lane pressure without changing the map."
        ),
        suggestion="Ping your jungler about 20 seconds before leaving lane and tie the "
        "roam to a wave crash, so the numbers advantage exists on arrival.",
        confidence=confidence,
        severity="info",
        evidence=[
            f"Team kills in the analysed match: {kills_for:.1f}",
            f"Team kills during your roams: {during:.1f}",
            f"Roam rate: {roam_rate:.2f}/min",
        ],
    )


def _objective_gap(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    objectives = fingerprint.get("Objectives")
    if objectives >= 5.5:
        return None
    pit = profile.region_distribution.get("Dragon Area", 0.0) + profile.region_distribution.get(
        "Herald Area", 0.0
    )
    strength = min(1.0, (5.5 - objectives) / 4.0)
    confidence = _conf(0.60 + 0.20 * strength, fingerprint.confidence.get("Objectives", 0.6), quality)
    return Insight(
        title="Low objective participation",
        detail=(
            f"Only {pit * 100:.1f}% of analysed time is spent near the dragon or herald pits "
            f"and the objectives axis scores {objectives:.1f}/10. Early objective fights are "
            "decided by who arrives first, so late arrivals cost neutral objectives."
        ),
        suggestion="Start moving towards the pit roughly 30 seconds before the objective "
        "spawns and help establish river control with the jungler.",
        confidence=confidence,
        severity="risk",
        evidence=[
            f"Objectives axis: {objectives:.1f}/10",
            f"Dragon + Herald presence: {pit * 100:.2f}%",
            f"River presence: {profile.region_distribution.get('River', 0) * 100:.1f}%",
        ],
    )


def _vision_gap(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    vision = fingerprint.get("Vision")
    if vision >= 5.5:
        return None
    river = profile.region_distribution.get("River", 0.0)
    corridor = profile.average_metrics.get("vision_proxy_fraction", 0.0)
    strength = min(1.0, (5.5 - vision) / 4.0)
    confidence = _conf(0.55 + 0.20 * strength, fingerprint.confidence.get("Vision", 0.5), quality)
    confidence = round(max(0.30, confidence * 0.92), 3)
    evidence = [
        f"Vision proxy axis: {vision:.1f}/10 (movement-based estimate)",
        f"River presence: {river * 100:.1f}%",
        f"River + half-jungle corridor presence: {corridor * 100:.1f}%",
    ]
    if "ward_blooms_per_min" in profile.average_metrics:
        evidence.append(
            "Vision blooms (ward-like, corridor): "
            f"{profile.average_metrics.get('ward_blooms', 0.0):.0f} "
            f"({profile.average_metrics['ward_blooms_per_min']:.2f}/min)"
        )
    return Insight(
        title="Limited control of vision corridors",
        detail=(
            f"River presence is {river * 100:.1f}% and the vision proxy scores "
            f"{vision:.1f}/10. The player spends little time in the corridors where "
            "defensive and offensive wards are typically placed."
        ),
        suggestion="On every recall and rotation, walk through the river entrance and drop "
        "a ward at the enemy jungle mouth before returning to lane.",
        confidence=confidence,
        severity="info",
        evidence=evidence,
    )


def _unwarded_aggression(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    """Forward time with no recent vision bloom is measurable ambush exposure."""
    metrics = profile.average_metrics
    if "unwarded_forward_fraction" not in metrics:
        return None  # ward tracking off: no evidence, no claim
    frac = metrics.get("unwarded_forward_fraction", 0.0)
    forward = metrics.get("forward_high_fraction", 0.0)
    if frac < 0.45 or forward < 0.10:
        return None
    strength = min(1.0, (frac - 0.45) / 0.40)
    confidence = _conf(0.58 + 0.26 * strength, fingerprint.confidence.get("Vision", 0.6), quality)
    return Insight(
        title="Pushing forward without nearby vision",
        detail=(
            f"{frac * 100:.0f}% of your forward time (position bias > 0.35) has no vision "
            f"bloom within {WARD_VISION_RADIUS:.2f} map units placed in the previous "
            f"{WARD_ACTIVE_LOOKBACK_SEC:.0f} seconds, out of "
            f"{forward * 100:.0f}% forward-positioned samples overall. Advancing on "
            "unwarded ground removes the few seconds of warning the minimap can give "
            "before a collapse arrives."
        ),
        suggestion="Drop a ward in the river or enemy jungle mouth *before* the wave "
        "passes half-way, and retreat to the warded side once the bloom expires.",
        confidence=confidence,
        severity="risk",
        evidence=[
            f"Forward samples without recent nearby vision: {frac * 100:.0f}%",
            f"Forward-positioned samples: {forward * 100:.0f}%",
            f"Vision bloom coverage model: radius {WARD_VISION_RADIUS:.2f}, "
            f"lookback {WARD_ACTIVE_LOOKBACK_SEC:.0f}s",
        ],
    )


def _death_pattern(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    """Repeated inferred deaths - counted from the marker alone - deserve attention."""
    metrics = profile.average_metrics
    deaths = metrics.get("deaths", 0.0)
    rate = metrics.get("death_rate_per_min", 0.0)
    if deaths < 1.2 or rate < 0.10:
        return None
    off_map = metrics.get("death_off_map_median", 0.0)
    confirmed = metrics.get("death_confirmed_fraction", 0.0)
    unwarded = metrics.get("unwarded_forward_fraction")
    strength = min(1.0, (rate - 0.10) / 0.35)
    confidence = _conf(0.60 + 0.25 * strength, fingerprint.confidence.get("Risk", 0.6), quality)
    detail = (
        f"PRISM infers {deaths:.1f} deaths per video ({rate:.2f}/min) purely from the "
        f"minimap marker vanishing away from base and reappearing at your fountain - "
        f"median {off_map:.0f}s off the map, {confirmed * 100:.0f}% corroborated by "
        "scoreboard kill deltas. Deaths at this rate hand the enemy free tempo every "
        "few minutes of the analysed game."
    )
    if unwarded is not None and unwarded >= 0.50:
        detail += (
            f" {unwarded * 100:.0f}% of forward time had no nearby vision, so most of "
            "these deaths happen while pushing blind."
        )
    evidence = [
        f"Inferred deaths per video: {deaths:.1f} ({rate:.2f}/min)",
        f"Median time off the map: {off_map:.0f}s",
        f"Scoreboard-confirmed: {confirmed * 100:.0f}%",
    ]
    if unwarded is not None:
        evidence.append(f"Forward time without nearby vision: {unwarded * 100:.0f}%")
    return Insight(
        title="Deaths visible in the movement data",
        detail=detail,
        suggestion="Before each death window, ask whether a ward covers the side you "
        "are pushing from - trade one CS for staying on the warded side.",
        confidence=confidence,
        severity="risk",
        evidence=evidence,
    )


def _slow_reaction(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    """Kills that never change the player's pace are fights not converted."""
    metrics = profile.average_metrics
    eligible = metrics.get("reaction_events_eligible", 0.0)
    if eligible < 4:
        return None  # too few observed kills to judge tempo honestly
    rate = metrics.get("reaction_response_rate", 1.0)
    if rate >= 0.60:
        return None
    latency = metrics.get("reaction_latency_median", 0.0)
    strength = min(1.0, (0.60 - rate) / 0.60)
    confidence = _conf(0.55 + 0.25 * strength, fingerprint.confidence.get("Aggression", 0.6), quality)
    return Insight(
        title="Slow response to kill events",
        detail=(
            f"Only {rate * 100:.0f}% of the {eligible:.0f} scoreboard kills visible in "
            f"the footage were followed by a clear change of pace or direction "
            f"within 8 seconds (median response {latency:.0f}s). Kills elsewhere on the "
            "map are temporary numbers advantages - hesitation lets them expire."
        ),
        suggestion="After any kill appears, pick one of three moves within ~3 seconds: "
        "crash the wave, rotate to the nearest objective, or recall - standing still "
        "is the only wrong answer.",
        confidence=confidence,
        severity="info",
        evidence=[
            f"Kills with measurable response: {rate * 100:.0f}% of {eligible:.0f}",
            f"Median response latency: {latency:.0f}s",
            "Response = heading change >= 75 deg or speed change >= 1.7x within 8s",
        ],
    )


def _pressure_gap(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    stability = fingerprint.get("Pressure Stability")
    if stability >= 5.5:
        return None
    metrics = profile.average_metrics
    drift = metrics.get("window_forward_std", 0.0)
    growth = metrics.get("window_burst_growth", 0.0)
    speed_mean = max(metrics.get("speed_mean", 0.0), 1e-6)
    speed_cv = metrics.get("window_speed_std", 0.0) / speed_mean
    turns = metrics.get("erratic_turn_rate", 0.0)
    strength = min(1.0, (5.5 - stability) / 4.0)
    confidence = _conf(0.58 + 0.22 * strength, fingerprint.confidence.get("Pressure Stability", 0.6), quality)
    if drift > 0.18:
        title = "Positioning destabilises under pressure"
        detail = (
            f"Forward-positioning drift is {drift:.2f} between phases"
            + (f" and burst activity escalates {growth * 100:+.0f}%" if growth > 0.2 else "")
            + f". The pressure-stability axis scores {stability:.1f}/10, meaning behaviour "
            "changes sharply when fights break out."
        )
    else:
        title = "Movement tempo destabilises under pressure"
        detail = (
            f"Positioning drift stays low ({drift:.2f}), but movement speed varies by "
            f"{speed_cv:.2f} (coefficient of variation) and headings change "
            f"{turns:.2f}/step. The pressure-stability axis scores {stability:.1f}/10 "
            "because pace and pathing destabilise rather than the lane position itself."
        )
    return Insight(
        title=title,
        detail=detail,
        suggestion="After every skirmish, reset to the standard lane position for one full "
        "wave instead of chasing - this rebuilds a stable baseline.",
        confidence=confidence,
        severity="risk",
        evidence=[
            f"Pressure stability axis: {stability:.1f}/10",
            f"Positioning drift: {drift:.3f}",
            f"Speed variation (CV): {speed_cv:.2f}",
            f"Erratic turns: {turns:.2f}/step",
            f"Burst growth: {max(-100.0, min(1000.0, growth * 100)):+.1f}%",
        ],
    )


def _consistency_gap(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    consistency = fingerprint.get("Consistency")
    if profile.video_count < 2 or consistency >= 5.5:
        return None
    strength = min(1.0, (5.5 - consistency) / 4.0)
    confidence = _conf(0.60 + 0.20 * strength, fingerprint.confidence.get("Consistency", 0.55), quality)
    return Insight(
        title="Inconsistent pattern between matches",
        detail=(
            f"The consistency axis is {consistency:.1f}/10 across {profile.video_count} "
            "recordings. Region occupancy and forward positioning vary enough match to "
            "match that opponents cannot be given a single reliable scouting read - and "
            "the player themselves cannot rely on a repeatable game plan."
        ),
        suggestion="Write down a fixed first-15-minutes plan (trade windows, roam triggers, "
        "objective timers) and follow it for five straight games, then re-run PRISM.",
        confidence=confidence,
        severity="info",
        evidence=[
            f"Consistency axis: {consistency:.1f}/10",
            f"Videos compared: {profile.video_count}",
            f"Detection quality: {profile.detection_rate * 100:.0f}%",
        ],
    )


def _passivity_gap(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> Insight | None:
    aggression = fingerprint.get("Aggression")
    if aggression >= 4.5:
        return None
    forward = profile.average_metrics.get("forward_high_fraction", 0.0)
    strength = min(1.0, (4.5 - aggression) / 4.0)
    confidence = _conf(0.58 + 0.22 * strength, fingerprint.confidence.get("Aggression", 0.6), quality)
    return Insight(
        title="Under-trading in lane",
        detail=(
            f"The aggression axis is {aggression:.1f}/10 and the player only holds a "
            f"clearly forward position (forward bias > 0.35) on {forward * 100:.0f}% of "
            "samples. Passing up even trades gives the opponent free priority, which "
            "then removes roam and objective options."
        ),
        suggestion="Punish every CS the opponent walks up to last-hit with one auto or "
        "ability, starting from level 1, to establish lane priority early.",
        confidence=confidence,
        severity="info",
        evidence=[
            f"Aggression axis: {aggression:.1f}/10",
            f"Forward positioning share: {forward * 100:.1f}%",
            f"Lane presence: {profile.average_metrics.get('lane_fraction', 0) * 100:.1f}%",
        ],
    )


def _strengths(
    profile: PlayerProfile, fingerprint: Fingerprint, quality: float
) -> List[Insight]:
    """Positive findings reported when nothing else needs fixing."""
    results: List[Insight] = []
    if fingerprint.get("Consistency") >= 7.0:
        results.append(
            Insight(
                title="Highly repeatable playstyle",
                detail=(
                    f"Consistency scores {fingerprint.get('Consistency'):.1f}/10 across "
                    f"{profile.video_count} recordings - the player executes the same "
                    "identity every game."
                ),
                suggestion="Keep the current routine and layer one new champion or route "
                "on top of the stable base.",
                confidence=_conf(0.70, fingerprint.confidence.get("Consistency", 0.7), quality),
                severity="strength",
                evidence=[f"Consistency axis: {fingerprint.get('Consistency'):.1f}/10"],
            )
        )
    if fingerprint.get("Objectives") >= 7.0:
        results.append(
            Insight(
                title="Strong objective discipline",
                detail=(
                    f"Objectives score {fingerprint.get('Objectives'):.1f}/10 with "
                    f"{profile.region_distribution.get('Dragon Area', 0) * 100:.1f}% dragon "
                    "presence - the player is already around when neutral objectives matter."
                ),
                suggestion="Maintain the timer discipline and extend it to herald setups "
                "before minute 8.",
                confidence=_conf(0.68, fingerprint.confidence.get("Objectives", 0.7), quality),
                severity="strength",
                evidence=[f"Objectives axis: {fingerprint.get('Objectives'):.1f}/10"],
            )
        )
    if fingerprint.get("Pressure Stability") >= 7.0:
        results.append(
            Insight(
                title="Steady behaviour in chaotic moments",
                detail=(
                    f"Pressure stability of {fingerprint.get('Pressure Stability'):.1f}/10 "
                    "shows the player keeps the same positional discipline during "
                    "high-risk engagements."
                ),
                suggestion="Use this composure to shot-call objective setups for the team.",
                confidence=_conf(0.66, fingerprint.confidence.get("Pressure Stability", 0.7), quality),
                severity="strength",
                evidence=[f"Pressure stability axis: {fingerprint.get('Pressure Stability'):.1f}/10"],
            )
        )
    return results

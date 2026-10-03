"""Pressure-stability analytics.

Estimates how steadily a player behaves when the game state becomes volatile.
Since PRISM cannot read kill scores without OCR, volatility is inferred from
movement itself:

* **stress episodes** - short bursts of high-speed, direction-changing motion
  performed while inside enemy territory or the river (proxies for trades,
  skirmishes and team fights),
* **positional drift** - standard deviation of forward bias across time slices,
* **late-window escalation** - whether burst activity grows towards minute 15,
* **recovery** - how quickly the player returns to a stable lane pattern after
  a stress episode.

A player whose metrics stay flat under stress scores high on
*Pressure Stability*; a player whose movement becomes erratic scores low.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Sequence

from analytics.timeline import analysis_own_base
from core.logger import get_logger
from core.models import Fingerprint, PlayerProfile, PositionSample, VideoAnalysis
from vision.region_mapper import RegionMapper

logger = get_logger("pressure")


@dataclass
class PressureAnalysis:
    """Pressure stability breakdown."""

    score: float
    drift: float
    burst_growth: float
    stress_episodes: int
    recovery_ratio: float
    confidence: float
    summary: str
    notes: List[str] = field(default_factory=list)


def analyze_pressure(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    analyses: Sequence[VideoAnalysis] = (),
    anchors: dict | None = None,
) -> PressureAnalysis:
    """Build the pressure section of the report."""
    metrics = profile.average_metrics
    mapper = RegionMapper(anchors)

    drift = metrics.get("window_forward_std", 0.0)
    growth = metrics.get("window_burst_growth", 0.0)

    episodes = 0
    recoveries = 0
    for analysis in analyses:
        # Each recording may be on either side; never mirror a minority-side
        # video with the majority base.
        base = analysis_own_base(analysis, profile.own_base)
        ep, rec = _stress_episodes(analysis.samples, mapper, base)
        episodes += ep
        recoveries += rec

    score = fingerprint.get("Pressure Stability")
    confidence = fingerprint.confidence.get("Pressure Stability", 0.5)
    recovery_ratio = (recoveries / episodes) if episodes else 0.0

    notes: List[str] = []
    if episodes == 0:
        if drift <= 0.18:
            notes.append("No high-risk engagement clusters detected - play stayed controlled.")
        else:
            notes.append(
                "No high-risk engagement clusters detected, but forward positioning "
                f"still drifts {drift:.2f} between phases."
            )
    else:
        notes.append(f"{episodes} high-risk engagement clusters detected across all recordings.")
        notes.append(
            f"Player returned to a stable lane pattern after {recovery_ratio * 100:.0f}% "
            "of those clusters."
        )
    if drift > 0.18:
        notes.append(
            f"Forward positioning swings substantially between phases (drift {drift:.2f}), "
            "suggesting emotional reactivity to game state."
        )
    elif drift < 0.08:
        notes.append(f"Positioning stays steady across phases (drift {drift:.2f}).")
    if growth > 0.5:
        notes.append(
            f"Burst activity escalates {growth * 100:+.0f}% late in the window - "
            "pressure handling weakens as the game speeds up."
        )
    elif growth < -0.1:
        notes.append(
            f"Burst activity eases {growth * 100:+.0f}% late in the window, "
            "indicating composure as tension rises."
        )

    summary = _summarise(score, drift, recovery_ratio, episodes)
    analysis = PressureAnalysis(
        score=score,
        drift=drift,
        burst_growth=growth,
        stress_episodes=episodes,
        recovery_ratio=recovery_ratio,
        confidence=confidence,
        summary=summary,
        notes=notes,
    )
    logger.debug("Pressure: score=%.1f episodes=%d recovery=%.0f%%", score, episodes, recovery_ratio * 100)
    return analysis


def _stress_episodes(
    samples: Sequence[PositionSample], mapper: RegionMapper, own_base: str
) -> tuple[int, int]:
    """Count high-risk motion clusters and successful recoveries."""
    if len(samples) < 6:
        return 0, 0

    episodes = 0
    recoveries = 0
    active = False
    episode_len = 0
    burst_threshold = _burst_threshold(samples)

    for index in range(1, len(samples)):
        prev, curr = samples[index - 1], samples[index]
        dt = max(1e-6, curr.effective_time - prev.effective_time)
        speed = ((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2) ** 0.5 / dt
        contested = curr.region in ("River", "Dragon Area", "Herald Area") or mapper.is_enemy_side(
            curr.x, curr.y, own_base
        )
        stressed = speed >= burst_threshold and contested

        if stressed and not active:
            active = True
            episode_len = 1
        elif stressed and active:
            episode_len += 1
        elif active and not stressed:
            active = False
            if episode_len >= 2:
                episodes += 1
                if _stable_after(samples, index):
                    recoveries += 1
            episode_len = 0

    if active and episode_len >= 2:
        episodes += 1
        if _stable_after(samples, len(samples) - 1):
            recoveries += 1
    return episodes, recoveries


def _stable_after(samples: Sequence[PositionSample], index: int) -> bool:
    """True when the following samples show settling movement.

    Without at least three trailing samples the outcome is *unknown* - which
    must not be counted as a successful recovery.
    """
    window = samples[index : min(len(samples), index + 5)]
    if len(window) < 3:
        return False
    speeds = []
    for i in range(1, len(window)):
        dt = max(1e-6, window[i].effective_time - window[i - 1].effective_time)
        speeds.append(
            ((window[i].x - window[i - 1].x) ** 2 + (window[i].y - window[i - 1].y) ** 2) ** 0.5 / dt
        )
    return statistics.pstdev(speeds) < 0.05


def _burst_threshold(samples: Sequence[PositionSample]) -> float:
    speeds: List[float] = []
    for i in range(1, len(samples)):
        dt = max(1e-6, samples[i].effective_time - samples[i - 1].effective_time)
        speeds.append(
            ((samples[i].x - samples[i - 1].x) ** 2 + (samples[i].y - samples[i - 1].y) ** 2) ** 0.5 / dt
        )
    if not speeds:
        return 0.05
    speeds.sort()
    # Floor keeps a mostly-stationary recording (85% of speeds == 0) from
    # treating "not moving" as a high-risk burst.
    return max(speeds[int(0.85 * (len(speeds) - 1))], 0.05)


def _summarise(score: float, drift: float, recovery: float, episodes: int) -> str:
    if episodes == 0:
        if score >= 7.0:
            return (
                f"Composed under pressure ({score:.1f}/10): positioning drift stays at "
                f"{drift:.2f} and no high-risk engagement clusters were detected."
            )
        if score >= 4.5:
            return (
                f"Average pressure stability ({score:.1f}/10): no high-risk engagement "
                f"clusters were detected; positioning drift {drift:.2f} between phases."
            )
        if drift > 0.18:
            return (
                f"Reactive under pressure ({score:.1f}/10): positional drift {drift:.2f} - "
                "positioning shifts between phases even though no engagement clusters "
                "were detected."
            )
        return (
            f"Reactive under pressure ({score:.1f}/10): no engagement clusters and only "
            f"{drift:.2f} positioning drift - the score reflects movement-speed and turn "
            "variability rather than detected fights."
        )
    if score >= 7.0:
        if recovery >= 0.5:
            return (
                f"Composed under pressure ({score:.1f}/10): positioning drift stays at "
                f"{drift:.2f} and the player recovers structure after {recovery * 100:.0f}% of "
                f"the {episodes} detected stress clusters."
            )
        return (
            f"Composed under pressure ({score:.1f}/10): positioning drift stays at "
            f"{drift:.2f}, though the player settles after only {recovery * 100:.0f}% of "
            f"the {episodes} detected stress clusters."
        )
    if score >= 4.5:
        if recovery >= 0.5:
            return (
                f"Average pressure stability ({score:.1f}/10): behaviour shifts during "
                f"volatile moments but usually re-stabilises (drift {drift:.2f}, "
                f"recovery after {recovery * 100:.0f}% of {episodes} clusters)."
            )
        return (
            f"Average pressure stability ({score:.1f}/10): behaviour shifts during "
            f"volatile moments and rarely settles quickly afterwards (drift {drift:.2f}, "
            f"recovery after {recovery * 100:.0f}% of {episodes} clusters)."
        )
    return (
        f"Reactive under pressure ({score:.1f}/10): positional drift {drift:.2f} and "
        f"{episodes} stress clusters indicate behaviour changes sharply when the game "
        "becomes contested."
    )

"""Behavioural fingerprint generation.

The fingerprint is a seven-axis, normalised (0-10) behavioural identity.
Each axis is a weighted blend of *explainable* indicators computed by the
analytics modules; every axis records

* the final score,
* a confidence value driven by detection quality and evidence volume,
* a human-readable contribution string describing what moved the number.

No black-box model is involved: every weight and saturation point is declared
in :data:`COMPOSITIONS` so the result can be audited and defended in a report.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from core.logger import get_logger
from core.models import FINGERPRINT_METRICS, Fingerprint, PlayerProfile

logger = get_logger("fingerprint")

Indicator = Tuple[
    str,
    float,
    Callable[[Dict[str, float]], Optional[float]],
    Callable[[Dict[str, float]], str],
]


CONSISTENCY_METRICS: Tuple[str, ...] = (
    "forward_high_fraction",
    "enemy_territory_fraction",
    "lane_fraction",
    "river_fraction",
    "roam_rate_per_min",
)


def _saturation(key: str, ceiling: float, offset: float = 0.0) -> Callable[[Dict[str, float]], float]:
    def mapper(metrics: Dict[str, float]) -> float:
        value = metrics.get(key, 0.0) - offset
        return max(0.0, min(1.0, value / ceiling)) if ceiling > 0 else 0.0

    return mapper


def _region_stability(profile: PlayerProfile) -> float:
    """1 - mean absolute deviation of region fractions across videos."""
    regions = [key for key in profile.average_metrics if key.startswith("region_")]
    if len(regions) < 2 or profile.video_count < 2:
        return 0.0
    if sum(max(0.0, profile.average_metrics.get(key, 0.0)) for key in regions) <= 1e-9:
        # No regional occupancy at all - stability would be vacuously perfect.
        return 0.0
    diffs: List[float] = []
    for region in regions:
        values = [
            metrics[region]
            for metrics in profile.per_video_metrics.values()
            if region in metrics
        ]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        diffs.append(sum(abs(v - mean) for v in values) / len(values))
    if not diffs:
        return 0.0
    mean_diff = sum(diffs) / len(diffs)
    return max(0.0, min(1.0, 1.0 - mean_diff / 0.22))


def metric_variability(profile: PlayerProfile, keys: Tuple[str, ...]) -> float:
    """Mean coefficient of variation across videos.

    Mean and standard deviation are both computed *unweighted* from the
    per-video values so the ratio compares like with like (mixing a
    time-weighted mean with an unweighted spread would inflate or deflate
    the CV depending on video lengths).

    The mean is floored at ``0.05`` so a metric hovering around zero cannot
    explode the ratio, and each key's contribution is clamped at ``3.0`` so a
    single near-zero metric cannot dominate the average.
    """
    coefficients: List[float] = []
    for key in keys:
        values = [
            metrics[key]
            for metrics in profile.per_video_metrics.values()
            if key in metrics
        ]
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = math.sqrt(variance)
        else:
            # Fall back to the pre-computed fields when per-video data is absent.
            mean = profile.average_metrics.get(key)
            std = profile.metric_variability.get(key)
            if mean is None or std is None:
                continue
        scale = max(abs(mean), 0.05)
        coefficients.append(min(std / scale, 3.0))
    if not coefficients:
        return 0.0
    return sum(coefficients) / len(coefficients)


COMPOSITIONS: Dict[str, List[Indicator]] = {
    "Aggression": [
        ("forward", 0.35, _saturation("forward_high_fraction", 0.45),
         lambda m: (
             f"clearly forward position (bias > 0.35) on "
             f"{m.get('forward_high_fraction', 0) * 100:.0f}% of samples"
         )),
        ("enemy", 0.25, _saturation("enemy_territory_fraction", 0.35),
         lambda m: f"spent {m.get('enemy_territory_fraction', 0) * 100:.0f}% of time in enemy territory"),
        ("burst", 0.20, _saturation("burst_fraction", 0.30),
         lambda m: f"burst-movement share {m.get('burst_fraction', 0) * 100:.0f}%"),
        ("roam", 0.20, _saturation("roam_rate_per_min", 0.60),
         lambda m: f"{m.get('roam_rate_per_min', 0):.2f} rotations per minute"),
    ],
    "Roaming": [
        ("rate", 0.45, _saturation("roam_rate_per_min", 0.60),
         lambda m: f"{m.get('roam_rate_per_min', 0):.2f} roams/min"),
        ("spread", 0.25, _saturation("roam_destinations_unique", 3.0),
         lambda m: f"{m.get('roam_destinations_unique', 0):.0f} distinct roam destinations"),
        ("leaving", 0.30, lambda m: max(0.0, min(1.0, (1.0 - m.get("lane_fraction", 1.0)) / 0.70)),
         lambda m: f"{(1 - m.get('lane_fraction', 1.0)) * 100:.0f}% of time outside the assigned lane"),
    ],
    "Vision": [
        ("proxy", 0.60, _saturation("vision_proxy_fraction", 0.45),
         lambda m: f"river/jungle corridor presence {m.get('vision_proxy_fraction', 0) * 100:.0f}%"),
        ("river", 0.40, _saturation("river_fraction", 0.20),
         lambda m: f"river presence {m.get('river_fraction', 0) * 100:.0f}%"),
        # Optional: only active when ward-bloom tracking produced data, so a
        # run with tracking disabled scores exactly as before.
        ("wards", 0.25,
         lambda m: (
             None if "ward_blooms_per_min" not in m
             else max(0.0, min(1.0, m.get("ward_blooms_per_min", 0.0) / 1.5))
         ),
         lambda m: (
             f"{m.get('ward_blooms', 0):.0f} vision blooms "
             f"({m.get('ward_blooms_per_min', 0):.2f}/min)"
         )),
    ],
    "Objectives": [
        ("pits", 0.45, _saturation("objective_fraction", 0.12),
         lambda m: f"{m.get('objective_fraction', 0) * 100:.1f}% of time at dragon/herald areas"),
        ("towers", 0.30, _saturation("tower_presence_fraction", 0.35),
         lambda m: f"{m.get('tower_presence_fraction', 0) * 100:.0f}% of samples near outer turrets"),
        ("river", 0.25, _saturation("river_fraction", 0.18),
         lambda m: f"river presence {m.get('river_fraction', 0) * 100:.0f}% (objective setups)"),
    ],
    "Risk": [
        ("enemy", 0.35, _saturation("enemy_territory_fraction", 0.35),
         lambda m: f"{m.get('enemy_territory_fraction', 0) * 100:.0f}% exposure in enemy half"),
        ("late", 0.25, _saturation("late_enemy_fraction", 0.40),
         lambda m: (
             f"enemy-side exposure holds at "
             f"{m.get('late_enemy_fraction', 0) * 100:.0f}% late in the window"
         )),
        ("burst", 0.20, _saturation("burst_fraction", 0.32),
         lambda m: f"high-speed burst share {m.get('burst_fraction', 0) * 100:.0f}%"),
        ("erratic", 0.20, _saturation("erratic_turn_rate", 0.40),
         lambda m: f"erratic heading changes {m.get('erratic_turn_rate', 0):.2f}/step"),
        # Optional: active only when ward-bloom tracking produced data, so a
        # run without vision tracking scores exactly as before.  This is the
        # vision-adjusted risk: time spent forward with no recent bloom.
        ("unwarded", 0.25,
         lambda m: (
             None if "unwarded_forward_fraction" not in m
             else max(0.0, min(1.0, m.get("unwarded_forward_fraction", 0.0)))
         ),
         lambda m: (
             f"{m.get('unwarded_forward_fraction', 0.0) * 100:.0f}% of forward time "
             "with no nearby vision bloom"
         )),
    ],
    "Consistency": [],
    "Pressure Stability": [],
}


class FingerprintEngine:
    """Build the normalised behavioural fingerprint for a profile."""

    def compute(self, profile: PlayerProfile) -> Fingerprint:
        metrics = profile.average_metrics
        fingerprint = Fingerprint()

        for metric in FINGERPRINT_METRICS:
            if metric in ("Consistency", "Pressure Stability"):
                score, confidence, note = self._special_metric(metric, profile)
            else:
                score, confidence, note = self._standard_metric(metric, metrics, profile)
            fingerprint.scores[metric] = round(score, 2)
            fingerprint.confidence[metric] = round(confidence, 3)
            fingerprint.contributions[metric] = note

        logger.info(
            "Fingerprint: %s",
            ", ".join(f"{m}={fingerprint.scores.get(m, 0):.1f}" for m in FINGERPRINT_METRICS),
        )
        return fingerprint

    def _standard_metric(
        self, metric: str, metrics: Dict[str, float], profile: PlayerProfile
    ) -> Tuple[float, float, str]:
        indicators = COMPOSITIONS[metric]
        active: List[Tuple[str, float, float, str]] = []
        for name, weight, mapper, describer in indicators:
            value = mapper(metrics)
            if value is None:
                # Optional indicator with no data this run (e.g. ward blooms
                # with tracking disabled): drop it and renormalise the rest
                # so the axis keeps its full 0-10 range.
                continue
            active.append((name, weight, value, describer(metrics)))
        weight_sum = sum(weight for _, weight, _, _ in active)
        if weight_sum <= 0:
            return 0.0, self._confidence(profile, metric), "no supporting indicators"
        total = sum(weight * value for _, weight, value, _ in active) / weight_sum
        score = max(0.0, min(10.0, total * 10.0))
        confidence = self._confidence(profile, metric)
        note = "; ".join(f"{name}: {desc}" for name, _, _, desc in active)
        return score, confidence, note or "no supporting indicators"

    def _special_metric(
        self, metric: str, profile: PlayerProfile
    ) -> Tuple[float, float, str]:
        if metric == "Consistency":
            return self._consistency(profile)
        return self._pressure_stability(profile)

    def _consistency(self, profile: PlayerProfile) -> Tuple[float, float, str]:
        variability = metric_variability(profile, CONSISTENCY_METRICS)
        stability = _region_stability(profile)
        spread_penalty = min(1.0, variability / 0.55)
        score = max(0.0, min(10.0, (0.6 * (1.0 - spread_penalty) + 0.4 * stability) * 10.0))

        if profile.video_count < 2:
            confidence = 0.30
            note = "single recording: match-to-match stability cannot be verified"
        else:
            confidence = min(0.95, 0.45 + 0.12 * profile.video_count) * self._quality(profile)
            note = (
                f"metric spread {variability:.2f} (lower is steadier); "
                f"regional stability (deviation-based) {stability * 100:.0f}% "
                f"across {profile.video_count} videos"
            )
        return score, confidence, note

    def _pressure_stability(self, profile: PlayerProfile) -> Tuple[float, float, str]:
        metrics = profile.average_metrics
        forward_std = metrics.get("window_forward_std", 0.0)
        speed_cv = 0.0
        if metrics.get("speed_mean", 0.0) > 1e-9:
            speed_cv = metrics.get("window_speed_std", 0.0) / max(metrics["speed_mean"], 1e-9)
        growth = metrics.get("window_burst_growth", 0.0)
        turns = metrics.get("erratic_turn_rate", 0.0)

        forward_score = max(0.0, min(1.0, 1.0 - forward_std / 0.28))
        speed_score = max(0.0, min(1.0, 1.0 - speed_cv / 1.40))
        growth_score = max(0.0, min(1.0, 1.0 - max(0.0, growth) / 1.20))
        turn_score = max(0.0, min(1.0, 1.0 - turns / 0.45))

        score = (0.35 * forward_score + 0.30 * speed_score + 0.20 * growth_score + 0.15 * turn_score) * 10.0
        confidence = self._confidence(profile, "Pressure Stability")
        note = (
            f"positional drift {forward_std:.2f}; speed variation {speed_cv:.2f}; "
            f"late-window burst growth {growth * 100:+.0f}%; erratic turns {turns:.2f}/step"
        )
        return max(0.0, min(10.0, score)), confidence, note

    def _confidence(self, profile: PlayerProfile, metric: str) -> float:
        base = 0.42 + 0.58 * max(0.0, min(1.0, profile.detection_rate))
        volume = min(1.0, profile.total_samples / 350.0)
        videos = min(1.0, profile.video_count / 3.0)
        confidence = base * (0.55 + 0.25 * volume + 0.20 * videos)
        confidence *= self._quality(profile)
        if metric == "Vision":
            confidence = min(confidence, 0.68)
            confidence *= 0.9
        return round(max(0.15, min(0.97, confidence)), 3)

    @staticmethod
    def _quality(profile: PlayerProfile) -> float:
        penalty = 1.0
        for note in profile.quality_notes:
            penalty *= 0.92
        return max(0.7, penalty)

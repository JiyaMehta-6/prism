"""Cross-video profile aggregation.

A single video yields noisy statistics.  The profiler merges several
``VideoAnalysis`` records belonging to the same player into one stable
:class:`~core.models.PlayerProfile` containing:

* time-weighted region distribution,
* merged region transition matrix,
* roam rate, destinations and recurring roam paths,
* mean / standard-deviation of every per-video metric (the basis of the
  consistency score),
* a coarse position histogram used for heatmap rendering,
* quality notes describing how trustworthy the underlying detections were.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.logger import get_logger
from core.models import PlayerProfile, VideoAnalysis
from vision.champion_identifier import CLAIM_GATE
from vision.region_mapper import REGIONS

logger = get_logger("profiler")

HIST_BINS = 24


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_profile(analyses: Sequence[VideoAnalysis], player_label: str) -> PlayerProfile:
    """Aggregate several :class:`VideoAnalysis` records into one profile."""
    region_time_total: Dict[str, float] = {region: 0.0 for region in REGIONS}
    transition_total: Dict[str, Dict[str, int]] = {region: {} for region in REGIONS}
    all_samples = 0
    detected = 0
    destination_counter: Dict[str, int] = {}
    path_counter: Dict[str, int] = {}
    per_video: Dict[str, Dict[str, float]] = {}
    quality_notes: List[str] = []
    histogram = np.zeros((HIST_BINS, HIST_BINS), dtype=np.float64)
    blue_votes, red_votes = 0, 0
    champion_votes: Dict[str, List[float]] = {}

    for analysis in analyses:
        all_samples += len(analysis.samples)
        detected += sum(1 for s in analysis.samples if s.detected)

        for region, seconds in analysis.region_time.items():
            region_time_total[region] = region_time_total.get(region, 0.0) + seconds

        for source, targets in analysis.transitions.items():
            transition_total.setdefault(source, {})
            for target, count in targets.items():
                transition_total[source][target] = transition_total[source].get(target, 0) + count

        for roam in analysis.roam_events:
            destination_counter[roam.destination_region] = (
                destination_counter.get(roam.destination_region, 0) + 1
            )
            key = " -> ".join(roam.path[:4])
            path_counter[key] = path_counter.get(key, 0) + 1

        # Key by a collision-free label so identical filenames from different
        # folders cannot silently overwrite each other's metrics.
        video_key = analysis.info.name
        duplicate = 2
        while video_key in per_video:
            video_key = f"{analysis.info.name} ({duplicate})"
            duplicate += 1
        per_video[video_key] = dict(analysis.metrics)

        if analysis.metrics.get("own_base_blue", 1.0) >= 0.5:
            blue_votes += 1
        else:
            red_votes += 1

        if analysis.detection_rate < 0.55:
            quality_notes.append(
                f"{analysis.info.name}: low marker detection ({analysis.detection_rate:.0%})"
            )
        if analysis.game_time_offset is None and analysis.samples:
            quality_notes.append(f"{analysis.info.name}: match clock unavailable, video time used")
        if analysis.champion and analysis.champion_confidence >= CLAIM_GATE:
            champion_votes.setdefault(analysis.champion, []).append(
                analysis.champion_confidence
            )

        for sample in analysis.samples:
            row = min(HIST_BINS - 1, max(0, int(sample.y * HIST_BINS)))
            col = min(HIST_BINS - 1, max(0, int(sample.x * HIST_BINS)))
            histogram[row, col] += 1.0

    total_time = sum(region_time_total.values()) or 1.0
    region_distribution = {r: seconds / total_time for r, seconds in region_time_total.items()}

    metric_keys = set()
    for metrics in per_video.values():
        metric_keys.update(metrics.keys())

    # Time-weighted means: a 15-minute VOD should not be averaged equally
    # with a 1-minute clip when both contribute metrics.
    weights = {
        key: max(0.0, metrics.get("analyzed_minutes", 0.0))
        for key, metrics in per_video.items()
    }
    weighted = sum(weights.values()) > 0

    average_metrics: Dict[str, float] = {}
    variability: Dict[str, float] = {}
    for key in sorted(metric_keys):
        pairs = [
            (weights.get(name, 0.0), metrics[key])
            for name, metrics in per_video.items()
            if key in metrics
        ]
        if not pairs:
            continue
        if weighted:
            total_weight = sum(weight for weight, _ in pairs) or 1.0
            average_metrics[key] = sum(w * v for w, v in pairs) / total_weight
        else:
            average_metrics[key] = _mean([value for _, value in pairs])
        values = [value for _, value in pairs]
        variability[key] = statistics.stdev(values) if len(values) > 1 else 0.0

    analyzed_minutes = total_time / 60.0
    total_roams = sum(destination_counter.values())

    # Champion of the profile: the most-voted confident per-video identity,
    # reported with the median confidence of the agreeing videos.
    profile_champion: Optional[str] = None
    profile_champion_confidence = 0.0
    if champion_votes:
        name = max(
            champion_votes,
            key=lambda key: (len(champion_votes[key]), sum(champion_votes[key])),
        )
        confidences = sorted(champion_votes[name])
        profile_champion = name
        profile_champion_confidence = confidences[len(confidences) // 2]

    # One detection-rate definition everywhere: attempts weighted, matching
    # the per-video value that feeds quality notes.
    attempts_total = sum(
        max(0.0, m.get("tracking_attempts", 0.0)) for m in per_video.values()
    )
    if attempts_total > 0:
        profile_detection = (
            sum(
                m.get("detection_rate", 0.0) * max(0.0, m.get("tracking_attempts", 0.0))
                for m in per_video.values()
            )
            / attempts_total
        )
    else:
        profile_detection = (detected / all_samples) if all_samples else 0.0

    profile = PlayerProfile(
        player_label=player_label,
        video_count=len(analyses),
        total_samples=all_samples,
        detection_rate=profile_detection,
        region_distribution=region_distribution,
        transition_matrix=transition_total,
        roam_rate=(total_roams / analyzed_minutes) if analyzed_minutes else 0.0,
        roam_destinations=destination_counter,
        roam_paths=path_counter,
        average_metrics=average_metrics,
        per_video_metrics=per_video,
        metric_variability=variability,
        own_base="Blue" if blue_votes >= red_votes else "Red",
        quality_notes=quality_notes,
        histogram=histogram.tolist(),
        champion=profile_champion,
        champion_confidence=profile_champion_confidence,
    )

    logger.info(
        "Profile built: %d videos, %d samples, %.0f%% detection, roam rate %.2f/min",
        profile.video_count, profile.total_samples,
        profile.detection_rate * 100, profile.roam_rate,
    )
    return profile

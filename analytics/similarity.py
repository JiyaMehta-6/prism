"""Behavioural similarity between player profiles.

Each profile is projected onto a single dense vector:

* the seven fingerprint axes,
* the nine region-presence fractions,
* the core movement indicators.

Vectors are stored next to the profile (``data/profiles/*.json``) and compared
with cosine similarity, which is scale invariant - a player analysed in one
short clip and another in five long clips remain comparable.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

import numpy as np

from core.logger import get_logger
from core.models import (
    FINGERPRINT_METRICS,
    Fingerprint,
    PlayerProfile,
    SimilarityResult,
    load_json,
    save_json,
)
from vision.region_mapper import REGIONS

logger = get_logger("similarity")

VECTOR_METRICS: Sequence[str] = (
    "forward_high_fraction",
    "enemy_territory_fraction",
    "lane_fraction",
    "river_fraction",
    "jungle_fraction",
    "roam_rate_per_min",
    "burst_fraction",
    "tower_presence_fraction",
)


def build_behavioral_vector(profile: PlayerProfile, fingerprint: Fingerprint) -> np.ndarray:
    """Project a profile onto a normalised behavioural vector."""
    parts: List[float] = []
    for metric in FINGERPRINT_METRICS:
        parts.append(float(fingerprint.scores.get(metric, 0.0)) / 10.0)
    for region in REGIONS:
        parts.append(float(profile.region_distribution.get(region, 0.0)))
    for metric in VECTOR_METRICS:
        parts.append(float(profile.average_metrics.get(metric, 0.0)))
    vector = np.asarray(parts, dtype=np.float64)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        vector = vector / norm
    return vector


def vector_to_list(vector: np.ndarray) -> List[float]:
    return [round(float(v), 6) for v in vector.tolist()]


def save_profile_snapshot(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    directory: str,
    vector: Optional[np.ndarray] = None,
) -> str:
    """Persist a profile snapshot for later similarity comparisons.

    An identical (label + vector) snapshot is reused instead of duplicated so
    repeated runs cannot flood the comparison library with self-matches.
    """
    vector = vector if vector is not None else build_behavioral_vector(profile, fingerprint)
    if float(np.linalg.norm(vector)) < 1e-9:
        logger.warning(
            "Empty behavioural vector for %s - snapshot not stored", profile.player_label
        )
        return ""
    safe_label = "".join(c if c.isalnum() or c in "-_ " else "_" for c in profile.player_label)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(directory, f"{safe_label.strip() or 'player'}-{stamp}.json")
    payload = {
        "player_label": profile.player_label,
        "saved_at": stamp,
        "video_count": profile.video_count,
        "fingerprint": fingerprint.scores,
        "region_distribution": profile.region_distribution,
        "metrics": {
            key: profile.average_metrics.get(key, 0.0) for key in VECTOR_METRICS
        },
        "vector": vector_to_list(vector),
    }

    if os.path.isdir(directory):
        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".json"):
                continue
            existing = load_json(os.path.join(directory, filename))
            if not isinstance(existing, dict):
                continue
            if (
                existing.get("player_label") == profile.player_label
                and existing.get("vector") == payload["vector"]
            ):
                logger.info("Identical snapshot already stored: %s", filename)
                return os.path.join(directory, filename)

    save_json(path, payload)
    logger.info("Profile snapshot saved: %s", path)
    return path


def compare_profiles(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    directory: str,
    top_n: int = 5,
) -> List[SimilarityResult]:
    """Compare the current profile against every stored snapshot."""
    current = build_behavioral_vector(profile, fingerprint)
    results: List[SimilarityResult] = []

    if float(np.linalg.norm(current)) < 1e-9:
        logger.warning("Current behavioural vector is empty - skipping similarity comparison")
        return results

    if not os.path.isdir(directory):
        return results

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        payload = load_json(os.path.join(directory, filename))
        # One corrupt snapshot must never abort the whole analysis.
        if not isinstance(payload, dict) or not payload.get("vector"):
            continue
        try:
            stored = np.asarray(payload["vector"], dtype=np.float64)
        except (TypeError, ValueError):
            logger.warning("Skipping unreadable vector in %s", filename)
            continue
        if stored.shape != current.shape:
            logger.warning("Skipping incompatible vector in %s", filename)
            continue
        if not bool(np.all(np.isfinite(stored))):
            logger.warning("Skipping non-finite vector in %s", filename)
            continue
        if float(np.linalg.norm(stored)) < 1e-9:
            logger.warning("Skipping empty vector in %s", filename)
            continue
        label = str(payload.get("player_label") or filename)
        similarity = _cosine(current, stored)
        note = _interpret(similarity)
        if label == profile.player_label:
            note = "Previous analysis of this profile - " + note[0].lower() + note[1:]
        results.append(SimilarityResult(profile_name=label, similarity=similarity, note=note))

    results.sort(key=lambda r: r.similarity, reverse=True)
    # One row per stored label: repeated snapshots of the same player must not
    # flood the top-N table with duplicate self-matches.
    deduped: List[SimilarityResult] = []
    seen: set = set()
    for result in results:
        if result.profile_name in seen:
            continue
        seen.add(result.profile_name)
        deduped.append(result)
    logger.info("Similarity computed against %d stored profiles", len(deduped))
    return deduped[: max(0, top_n)]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, 0.0, 1.0))


def _interpret(similarity: float) -> str:
    if similarity >= 0.85:
        return "Near-identical behavioural fingerprint"
    if similarity >= 0.75:
        return "Very similar strategic identity"
    if similarity >= 0.65:
        return "Comparable playstyle with notable differences"
    if similarity >= 0.50:
        return "Loosely related behaviour pattern"
    return "Distinctly different player profile"

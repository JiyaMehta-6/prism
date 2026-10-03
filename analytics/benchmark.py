"""Percentile benchmarking of the current fingerprint against stored profiles.

Every analysis stores a profile snapshot (``data/profiles/*.json``) carrying
the seven fingerprint axes.  With enough *other* players on disk we can say
where the current player sits on each axis - e.g. "Aggression P80 among 5
stored profiles".  A minimum of ``MIN_BENCHMARK_N`` comparable snapshots is
required; below that the benchmark is simply omitted (an N of 1-2 would
manufacture misleading percentiles).

Snapshots sharing the current profile's label are excluded: benchmarking a
player against their own previous runs answers a different question (and the
similarity section already covers self-matches).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.logger import get_logger
from core.models import FINGERPRINT_METRICS, Fingerprint, PlayerProfile, load_json

logger = get_logger("benchmark")

MIN_BENCHMARK_N = 3


@dataclass
class BenchmarkSection:
    """Report section (``summary``/``notes`` shape) for percentile standings."""

    summary: str
    notes: List[str] = field(default_factory=list)
    n: int = 0
    percentiles: Dict[str, float] = field(default_factory=dict)


def _percentile(values: np.ndarray, current: float) -> float:
    """Mid-rank percentile of ``current`` within ``values`` (0-100)."""
    below = float(np.sum(values < current))
    equal = float(np.sum(values == current))
    return 100.0 * (below + 0.5 * equal) / len(values)


def load_fingerprint_snapshots(
    directory: str, exclude_label: str = ""
) -> List[Dict[str, float]]:
    """Load stored fingerprint score dicts, skipping corrupt/self entries."""
    loaded: List[Dict[str, float]] = []
    if not os.path.isdir(directory):
        return loaded
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        payload = load_json(os.path.join(directory, filename))
        if not isinstance(payload, dict):
            continue
        if exclude_label and payload.get("player_label") == exclude_label:
            continue
        scores = payload.get("fingerprint")
        if not isinstance(scores, dict):
            continue
        try:
            parsed = {str(k): float(v) for k, v in scores.items()}
        except (TypeError, ValueError):
            logger.warning("Skipping unreadable fingerprint in %s", filename)
            continue
        if any(metric not in parsed for metric in FINGERPRINT_METRICS):
            continue  # incomplete snapshot cannot be compared axis by axis
        loaded.append(parsed)
    return loaded


def compute_benchmark(
    profile: PlayerProfile,
    fingerprint: Fingerprint,
    directory: str,
    min_n: int = MIN_BENCHMARK_N,
) -> Optional[BenchmarkSection]:
    """Return percentile standings, or ``None`` when ``n < min_n``."""
    stored = load_fingerprint_snapshots(directory, exclude_label=profile.player_label)
    if len(stored) < max(1, min_n):
        logger.info(
            "Benchmark skipped: %d comparable stored profile(s) (need >= %d)",
            len(stored),
            max(1, min_n),
        )
        return None

    notes: List[str] = []
    percentiles: Dict[str, float] = {}
    ranked: List[tuple] = []
    for metric in FINGERPRINT_METRICS:
        current = float(fingerprint.scores.get(metric, 0.0))
        values = np.asarray([scores[metric] for scores in stored], dtype=np.float64)
        pct = _percentile(values, current)
        percentiles[metric] = pct
        median = float(np.median(values))
        ranked.append((metric, pct, current, median))
        notes.append(
            f"{metric}: {current:.1f}/10 - percentile {pct:.0f} "
            f"(stored median {median:.1f}, n={len(stored)})"
        )

    ranked.sort(key=lambda item: item[1], reverse=True)
    top = ranked[0]
    bottom = ranked[-1]
    summary = (
        f"Compared against {len(stored)} stored profiles of other players: "
        f"{top[0]} is highest at P{top[1]:.0f} ({top[2]:.1f}/10), "
        f"{bottom[0]} lowest at P{bottom[1]:.0f} ({bottom[2]:.1f}/10)."
    )
    logger.info("Benchmark computed against %d stored profiles", len(stored))
    return BenchmarkSection(
        summary=summary, notes=notes, n=len(stored), percentiles=percentiles
    )

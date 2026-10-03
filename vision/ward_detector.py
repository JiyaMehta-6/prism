"""Ward-like vision bloom detection on the minimap crop.

A placed allied ward renders as a small *blue three-pointed star* that stays
pixel-stationary on the minimap for 90+ seconds (enemy wards are red and
mostly invisible without oracle).  PRISM detects it heuristically - fully
offline, classical CV only:

1. a slowly adapting background (EMA) models the static minimap,
2. pixels whose *blue channel* is notably brighter than the background AND
   are themselves bright and blue-dominant (cyan-blue star glyph) form
   candidate blobs - dark fog-boundary transitions and white minimap dots
   fail this gate, while jungle-camp icons and structures are gold/red;
3. a candidate only qualifies when it stays *stationary* (centroid drift
   below a tolerance AND negligible net travel from where it first
   appeared), sits away from the crop border, is not on the tracked player,
   is compact enough to be a star rather than a portrait or recall swirl,
   and falls inside a corridor region (river / jungle / objective pits).

Moving markers (champions, minions, ping alerts, skillshots) either travel
between samples - which resets the stationarity counter - or, when they creep
slowly enough to stay under the per-sample drift tolerance, accumulate net
travel from their birth point that a genuine ward never does, so they still
fail to confirm.  Even so this remains a *proxy*: PRISM reports "vision
blooms (ward-like)" and never claims an exact ward inventory, which would
require telemetry PRISM does not use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


@dataclass
class WardDetection:
    """One confirmed, stationary vision bloom."""

    time: float
    x: float
    y: float
    age: int


@dataclass
class _Track:
    x: float
    y: float
    last_x: float
    last_y: float
    age: int
    missed: int
    emitted: bool
    birth_x: float
    birth_y: float


class WardDetector:
    """Track stationary bright blobs across sampled minimap frames."""

    def __init__(
        self,
        sample_fps: float = 1.0,
        brightness: float = 58.0,
        match_radius: float = 0.045,
        drift: float = 0.030,
        net_tolerance: float = 0.020,
        min_persist_sec: float = 10.0,
        miss_grace_sec: float = 3.0,
        min_area_frac: float = 0.0002,
        max_area_frac: float = 0.0011,
        min_blue: float = 140.0,
        blue_dominance: float = 30.0,
        player_radius: float = 0.11,
        background_alpha: float = 0.03,
        warmup_sec: float = 10.0,
        exclusions: Sequence[Tuple[Point, float]] = (),
    ) -> None:
        fps = max(0.1, float(sample_fps))
        self.brightness = float(brightness)
        self.match_radius = float(match_radius)
        self.drift = float(drift)
        self.net_tolerance = float(net_tolerance)
        self.min_persist = max(2, int(round(min_persist_sec * fps)))
        self.miss_grace = max(1, int(round(miss_grace_sec * fps)))
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.min_blue = float(min_blue)
        self.blue_dominance = float(blue_dominance)
        self.player_radius = float(player_radius)
        self.background_alpha = float(background_alpha)
        self.warmup_sec = float(warmup_sec)

        self.exclusions: List[Tuple[Point, float]] = [
            (tuple(point), float(radius)) for point, radius in exclusions  # type: ignore[misc]
        ]
        self._background: Optional[np.ndarray] = None
        self._tracks: List[_Track] = []
        self._kernel = np.ones((3, 3), np.uint8)

    def process(
        self,
        crop: np.ndarray,
        player_xy: Optional[Point],
        timestamp: float,
    ) -> List[WardDetection]:
        """Feed one minimap frame; return blooms confirmed on this sample."""
        if crop is None or crop.size == 0:
            return []
        # Work purely on the blue channel: ward stars are blue (see module
        # docstring) and this single channel separates them from gold camp
        # icons, structures and red portraits without a second threshold.
        blue = crop[:, :, 2].astype(np.float32)
        if self._background is None or self._background.shape != blue.shape:
            # First frame only seeds the background; nothing can be transient yet.
            self._background = blue.copy()
            return []
        if timestamp < self.warmup_sec:
            # Warm-up: the minimap is still revealing itself (fog lift, HUD
            # fade-in) and the background has not settled.  Reseed to the
            # current frame so that *after* warm-up every difference is a
            # transient object, never part of the initial map render.
            self._background = blue.copy()
            return []

        diff = blue - self._background
        # A real ward star is *bright and blue-dominant* (cyan-blue glyph).
        # Fog-boundary transitions are dark (low absolute blue) and white
        # minimap dots/pings are bright but neutral (no blue dominance),
        # so requiring both properties removes them before tracking.
        red = crop[:, :, 0].astype(np.float32)
        mask = (
            (diff > self.brightness)
            & (blue >= self.min_blue)
            & ((blue - red) >= self.blue_dominance)
        ).astype(np.uint8)
        # Open removes single-pixel noise; without it compression sparkle
        # would spawn a new track on every frame.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        candidates = self._candidates(mask, player_xy)
        # Background adapts after masking so a brand-new bloom is compared
        # against the pre-bloom state for as long as possible.
        self._background = (
            (1.0 - self.background_alpha) * self._background
            + self.background_alpha * blue
        )
        return self._update_tracks(candidates, timestamp)

    def _candidates(self, mask: np.ndarray, player_xy: Optional[Point]) -> List[Point]:
        height, width = mask.shape[:2]
        area = height * width
        min_area = max(4.0, self.min_area_frac * area)
        max_area = self.max_area_frac * area
        edge_margin_x = 0.04 * width
        edge_margin_y = 0.04 * height
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        found: List[Point] = []
        for index in range(1, count):
            blob_area = stats[index, cv2.CC_STAT_AREA]
            if blob_area < min_area or blob_area > max_area:
                continue
            # Blobs hugging the crop border are minimap frame / structure
            # artifacts, not wards tucked into a brush.
            left = float(stats[index, cv2.CC_STAT_LEFT])
            top = float(stats[index, cv2.CC_STAT_TOP])
            right = left + float(stats[index, cv2.CC_STAT_WIDTH])
            bottom = top + float(stats[index, cv2.CC_STAT_HEIGHT])
            if (
                left < edge_margin_x
                or top < edge_margin_y
                or right > width - edge_margin_x
                or bottom > height - edge_margin_y
            ):
                continue
            cx, cy = float(centroids[index][0]), float(centroids[index][1])
            x, y = cx / width, cy / height
            if player_xy is not None and math.hypot(
                x - player_xy[0], y - player_xy[1]
            ) <= self.player_radius:
                continue
            if any(
                math.hypot(x - ex, y - ey) <= radius
                for (ex, ey), radius in self.exclusions
            ):
                continue
            # One bloom often fragments into several components: keep the
            # first of any cluster so it cannot double-count as wards.
            if any(math.hypot(x - ox, y - oy) <= self.drift for ox, oy in found):
                continue
            found.append((x, y))
        return found

    def _update_tracks(
        self, candidates: List[Point], timestamp: float
    ) -> List[WardDetection]:
        confirmed: List[WardDetection] = []
        matched: set = set()

        for track in self._tracks:
            best_index: Optional[int] = None
            best_distance = self.match_radius
            for index, (x, y) in enumerate(candidates):
                if index in matched:
                    continue
                distance = math.hypot(x - track.last_x, y - track.last_y)
                if distance <= best_distance:
                    best_index, best_distance = index, distance
            if best_index is None:
                track.missed += 1
                continue

            matched.add(best_index)
            x, y = candidates[best_index]
            if math.hypot(x - track.last_x, y - track.last_y) > self.drift:
                # The blob moved: restart the stationarity counter.  Clashing
                # minion waves jitter past this tolerance; wards never do.
                track.x, track.y = x, y
                track.age = 1
                track.emitted = False
            else:
                track.age += 1
                track.x = 0.5 * (track.x + x)
                track.y = 0.5 * (track.y + y)
            track.last_x, track.last_y = x, y
            track.missed = 0
            if (
                track.age >= self.min_persist
                and not track.emitted
                # A ward is placed once and never moves: if the blob has
                # travelled from where it first appeared it is a creeping
                # champion/minion, not a bloom, even when it pauses briefly.
                and math.hypot(
                    track.x - track.birth_x, track.y - track.birth_y
                )
                <= self.net_tolerance
            ):
                track.emitted = True
                confirmed.append(WardDetection(timestamp, track.x, track.y, track.age))

        for index, (x, y) in enumerate(candidates):
            if index in matched:
                continue
            self._tracks.append(
                _Track(
                    x, y, x, y, age=1, missed=0, emitted=False, birth_x=x, birth_y=y
                )
            )

        self._tracks = [t for t in self._tracks if t.missed <= self.miss_grace]
        return confirmed

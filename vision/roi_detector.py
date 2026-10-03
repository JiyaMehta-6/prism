"""Minimap region-of-interest detection.

Locates the square minimap panel inside a full gameplay frame without any
template matching or external assets.  The detector:

1. builds candidate squares anchored to the bottom-left and bottom-right
   corners at several side lengths,
2. scores each candidate with a *map-likeness* heuristic (Summoner's Rift is
   rich in green / brown / blue terrain hues while HUD chrome is dark and
   desaturated),
3. refines the winner by extracting the bounding box of terrain-coloured
   pixels so the HUD frame and border are cropped away,
4. caches the result but re-validates it against every live frame: the cached
   rectangle must keep looking map-like, otherwise (loading screen, edited
   transition, HUD scale change) the detector re-runs from scratch.

Manual overrides (fixed side + fixed normalised rectangle) are supported via
the settings object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger("roi_detector")


@dataclass
class ROIResult:
    """Detected minimap rectangle in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    side: str
    score: float
    found: bool

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    """Guarantee a 3-channel BGR frame (grayscale input cannot crash cvtColor)."""
    if frame is None or frame.size == 0:
        return frame
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _terrain_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that look like Summoner's Rift terrain."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= 25) & (hue <= 75) & (sat > 40) & (val > 30)
    brown = ((hue <= 20) | (hue >= 160)) & (sat > 35) & (val > 30)
    blue = (hue >= 85) & (hue <= 135) & (sat > 40) & (val > 30)
    return green | brown | blue


class ROIDetector:
    """Find and cache the minimap rectangle inside gameplay frames."""

    CANDIDATE_SIDES = (0.09, 0.12, 0.15, 0.18)
    MARGIN = 0.004
    # When several side lengths score almost equally, prefer the larger one:
    # ratio-based scoring slightly favours small crops.
    SIZE_TIE = 0.04
    # Failed detections are cached (the HUD does not change mid-video) but
    # retried periodically in case detection ran during a loading screen.
    RETRY_AFTER_FAILURES = 60
    # A cached rectangle must still look like terrain on the live frame.
    # Consecutive misses are required so a single transient overlay (recall
    # flash, edited transition) never throws away a good cache.
    LIVE_MIN_SCORE = 0.16
    LIVE_MIN_RATIO = 0.70
    VALIDATION_STRIKES = 3

    def __init__(self, side: str = "auto", fixed_roi: Optional[list] = None) -> None:
        self.side_pref = side if side in ("auto", "left", "right") else "auto"
        self.fixed_roi = fixed_roi or [0.0, 0.0, 0.0, 0.0]
        self._cache: Optional[ROIResult] = None
        self._cache_signature: Optional[Tuple[int, int]] = None
        self._failure_age = 0
        self._strikes = 0

    def reset(self) -> None:
        self._cache = None
        self._cache_signature = None
        self._failure_age = 0
        self._strikes = 0

    def detect(self, frame: np.ndarray, force: bool = False) -> ROIResult:
        """Return the minimap ROI for ``frame`` (cached when possible)."""
        frame = _as_bgr(frame)
        height, width = frame.shape[:2]
        signature = (width, height)

        if not force and self._cache is not None and self._cache_signature == signature:
            if self._cache.found:
                if self._cache.side == "fixed":
                    return self._cache  # manual override is authoritative
                if not self._validate_cache(frame):
                    logger.info(
                        "Cached minimap ROI no longer matches frame content -> %s",
                        self._cache.as_tuple(),
                    )
                    # fall through and re-detect on this frame
                else:
                    return self._cache
            else:
                self._failure_age += 1
                if self._failure_age < self.RETRY_AFTER_FAILURES:
                    return self._cache

        was_failing = (
            self._cache is not None
            and not self._cache.found
            and self._cache_signature == signature
        )
        result = self._detect_fixed(frame, width, height) or self._search(frame)
        if result is None or not result.found:
            result = self._fallback(width, height)
            if not was_failing:
                # Warn once per signature instead of on every frame.
                logger.warning("Minimap ROI not detected; using layout fallback %s", result.as_tuple())
            self._failure_age = 0
        else:
            if self._cache is None or not self._cache.found or self._cache.as_tuple() != result.as_tuple():
                logger.info(
                    "Minimap ROI detected on %s side (score=%.3f) -> %s",
                    result.side, result.score, result.as_tuple(),
                )

        self._cache = result
        self._cache_signature = signature
        if result.found:
            self._failure_age = 0
            self._strikes = 0
        return result

    def _validate_cache(self, frame: np.ndarray) -> bool:
        """Check the cached rectangle still looks like terrain on ``frame``.

        Returns ``True`` while the cache is healthy.  Requires
        ``VALIDATION_STRIKES`` consecutive bad frames before invalidating so a
        single flash/transition cannot discard a good detection.
        """
        roi = self._cache
        if roi is None or not roi.found:
            return False
        height, width = frame.shape[:2]
        x0 = max(0, min(width - 1, roi.x))
        y0 = max(0, min(height - 1, roi.y))
        x1 = max(x0 + 1, min(width, roi.x + roi.w))
        y1 = max(y0 + 1, min(height, roi.y + roi.h))
        patch = frame[y0:y1, x0:x1]
        live = self._score_patch(patch)
        threshold = max(self.LIVE_MIN_SCORE, self.LIVE_MIN_RATIO * roi.score)
        if live >= threshold:
            self._strikes = 0
            return True
        self._strikes += 1
        logger.debug(
            "ROI cache validation %d/%d: live score %.3f < %.3f",
            self._strikes, self.VALIDATION_STRIKES, live, threshold,
        )
        return self._strikes < self.VALIDATION_STRIKES

    def crop(self, frame: np.ndarray, roi: Optional[ROIResult] = None) -> np.ndarray:
        """Crop the minimap out of ``frame`` with safe bounds handling."""
        roi = roi or self.detect(frame)
        height, width = frame.shape[:2]
        x0 = max(0, min(width - 1, roi.x))
        y0 = max(0, min(height - 1, roi.y))
        x1 = max(x0 + 1, min(width, roi.x + roi.w))
        y1 = max(y0 + 1, min(height, roi.y + roi.h))
        return frame[y0:y1, x0:x1].copy()

    def _detect_fixed(
        self, frame: np.ndarray, width: int, height: int
    ) -> Optional[ROIResult]:
        if any(v != 0 for v in self.fixed_roi[:4]):
            vals = list(self.fixed_roi[:4])
            valid = (
                len(vals) == 4
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
                and 0.0 <= vals[0] <= 1.0
                and 0.0 <= vals[1] <= 1.0
                and vals[2] > 0.0
                and vals[3] > 0.0
                and vals[0] + vals[2] <= 1.0
                and vals[1] + vals[3] <= 1.0
            )
            if valid:
                x = int(vals[0] * width)
                y = int(vals[1] * height)
                w = int(vals[2] * width)
                h = int(vals[3] * height)
                if w > 4 and h > 4:
                    return ROIResult(x, y, w, h, "fixed", 1.0, True)
            else:
                logger.warning("Ignoring invalid fixed ROI override %s", vals)
        if self.side_pref in ("left", "right"):
            side_px = int(min(self.CANDIDATE_SIDES[2] * width, 0.9 * height))
            margin = int(self.MARGIN * width)
            x = margin if self.side_pref == "left" else width - side_px - margin
            y = height - side_px - margin
            if x < 0 or y < 0 or side_px <= 0:
                return None
            square = ROIResult(x, y, side_px, side_px, self.side_pref, 0.5, True)
            patch = frame[y : y + side_px, x : x + side_px]
            score = self._score_patch(patch) if patch.size else square.score
            if score < 0.18:
                # Same quality gate as auto-search: an explicit side preference
                # must not promote a non-terrain rectangle to a valid ROI.
                logger.debug("Explicit %s-side ROI score too low (%.3f)", self.side_pref, score)
                return None
            # Apply the terrain refinement just like auto-detected squares.
            refined = self._refine(frame, square)
            return ROIResult(refined.x, refined.y, refined.w, refined.h,
                             self.side_pref, score, True)
        return None

    def _search(self, frame: np.ndarray) -> Optional[ROIResult]:
        height, width = frame.shape[:2]
        margin = int(self.MARGIN * width)
        sides = [int(s * width) for s in self.CANDIDATE_SIDES]
        sides = [s for s in sides if s > 20 and s < min(width, height) * 0.6]
        if not sides:
            return None

        sides_wanted = ("left", "right") if self.side_pref == "auto" else (self.side_pref,)
        best: Optional[ROIResult] = None

        for side in sides_wanted:
            for side_px in sides:
                if side == "left":
                    x = margin
                else:
                    x = width - side_px - margin
                y = height - side_px - margin
                if x < 0 or y < 0:
                    continue
                patch = frame[y : y + side_px, x : x + side_px]
                score = self._score_patch(patch)
                if best is None or score > best.score:
                    best = ROIResult(x, y, side_px, side_px, side, score, True)
                elif (
                    side == best.side
                    and side_px > best.w
                    and score >= best.score - self.SIZE_TIE
                ):
                    # Near-tied scores *within one corner*: prefer the larger,
                    # more complete square (never jump to the opposite corner).
                    best = ROIResult(x, y, side_px, side_px, side, score, True)

        if best is None:
            return None
        if best.score < 0.18:
            logger.debug("Best ROI score too low (%.3f)", best.score)
            return None
        return self._refine(frame, best)

    @staticmethod
    def _score_patch(patch: np.ndarray) -> float:
        """Heuristic map-likeness in [0, 1]."""
        if patch.size == 0:
            return 0.0
        small = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
        mask = _terrain_mask(small)
        terrain_ratio = float(mask.mean())

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        saturation_mean = float(hsv[..., 1].mean()) / 255.0
        value_mean = float(hsv[..., 2].mean()) / 255.0

        edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 60, 140)
        edge_ratio = float((edges > 0).mean())

        dark_ratio = float((value_mean < 0.18))
        score = (
            0.60 * terrain_ratio
            + 0.20 * saturation_mean
            + 0.15 * min(edge_ratio * 6.0, 1.0)
            + 0.05 * (1.0 - dark_ratio)
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _refine(frame: np.ndarray, roi: ROIResult) -> ROIResult:
        """Shrink the square to the terrain-coloured content inside it."""
        patch = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        if patch.size == 0:
            return roi
        mask = _terrain_mask(patch).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.15 * patch.shape[0] * patch.shape[1]:
            return roi
        x0, y0, w, h = cv2.boundingRect(largest)
        if w < 8 or h < 8:
            return roi
        return ROIResult(roi.x + x0, roi.y + y0, w, h, roi.side, roi.score, True)

    def _fallback(self, width: int, height: int) -> ROIResult:
        side = int(min(0.14 * width, 0.26 * height))
        margin = int(self.MARGIN * width)
        if self.side_pref == "right":
            x = width - side - margin
        else:
            x = margin
        y = height - side - margin
        return ROIResult(max(0, x), max(0, y), side, side, self.side_pref, 0.0, False)

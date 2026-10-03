"""Player-marker tracking on the minimap.

This is the behavioural backbone of PRISM: every downstream insight is derived
from the tracked player position.  The tracker combines three independent
signals so that it keeps working when any single cue degrades:

1. **Motion cue** - a running background model isolates pixels that differ from
   the static terrain (champions, minions, wards, pings, camera rectangle).
2. **Appearance cue** - a configurable HSV gate captures the player's marker
   (bright / low-saturation arrow by default, team colours optionally).
3. **Tracking prior** - an alpha-beta motion model predicts where the marker
   should be and boosts candidates near that prediction while penalising
   implausible jumps.

Candidates are extracted with connected-component analysis, scored, and the
winner is converted to a confidence value in ``[0, 1]``.  When no candidate
clears the quality bar the tracker reports a *miss*; short misses are later
interpolated by the behaviour engine, so OCR/vision failures never abort a run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger("minimap_tracker")


@dataclass
class MarkerDetection:
    """A single player-position observation inside the minimap crop."""

    x: float
    y: float
    confidence: float
    score: float
    method: str
    velocity: Tuple[float, float] = (0.0, 0.0)


class MinimapTracker:
    """Detect and temporally smooth the player marker on the minimap."""

    WORK_SIZE = 160
    MIN_AREA = 3
    MAX_AREA_FRAC = 0.06
    BG_ALPHA = 0.04
    # Background must keep adapting *under* foreground pixels, otherwise the
    # frame-1 marker (or any minion/ward parked on frame 1) leaves a permanent
    # "ghost" blob that can hijack tracking after a single occlusion.
    BG_ALPHA_FOREGROUND = 0.06
    SEED_FRAMES = 5
    SEED_ALPHA = 0.25
    # Reject a low-evidence candidate that would teleport the marker.
    MAX_JUMP = 0.25
    JUMP_MIN_SCORE = 0.72

    def __init__(
        self,
        min_confidence: float = 0.30,
        marker_hsv: Optional[Sequence[Tuple[int, int, int]]] = None,
        prediction_gain: float = 0.35,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.prediction_gain = float(prediction_gain)
        self.marker_hsv = self._validated_hsv(marker_hsv)
        self.background: Optional[np.ndarray] = None
        self.prev_pos: Optional[Tuple[float, float]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.miss_streak = 0
        self.detections = 0
        self.misses = 0
        self._seed_frames = 0

    @staticmethod
    def _validated_hsv(
        marker_hsv: Optional[Sequence[Tuple[int, int, int]]],
    ) -> List[Tuple[int, int, int]]:
        """Clamp/validate user HSV gates so bad ranges cannot crash inRange."""
        validated: List[Tuple[int, int, int]] = []
        for entry in marker_hsv or []:
            try:
                low, high, weight = entry
            except (TypeError, ValueError):
                continue
            lo = (
                max(0, min(180, int(low[0]))),
                max(0, min(255, int(low[1]))),
                max(0, min(255, int(low[2]))),
            )
            hi = (
                max(0, min(180, int(high[0]))),
                max(0, min(255, int(high[1]))),
                max(0, min(255, int(high[2]))),
            )
            if any(lo[i] > hi[i] for i in range(3)):
                continue  # inverted range: unusable, skip
            validated.append((lo, hi, max(0.0, min(1.0, float(weight)))))
        return validated

    def reset(self) -> None:
        """Clear all temporal state (call between videos)."""
        self.background = None
        self.prev_pos = None
        self.velocity = (0.0, 0.0)
        self.miss_streak = 0
        self.detections = 0
        self.misses = 0
        self._seed_frames = 0

    @property
    def reliability(self) -> float:
        """Fraction of processed frames that produced a usable detection."""
        total = self.detections + self.misses
        return self.detections / total if total else 0.0

    def process(self, minimap_crop: np.ndarray) -> Optional[MarkerDetection]:
        """Track the marker inside one minimap crop.

        Returns ``None`` when the marker cannot be located reliably.
        """
        if minimap_crop is None or minimap_crop.size == 0:
            self._record_miss()
            return None

        work = cv2.resize(
            minimap_crop, (self.WORK_SIZE, self.WORK_SIZE), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)

        if self.background is None:
            self.background = gray.copy()

        motion = self._motion_mask(gray)
        appearance = self._appearance_mask(hsv, motion)
        self._update_background(gray, motion, appearance)

        candidates = self._collect_candidates(motion, appearance, gray)
        best = self._select_candidate(candidates)

        if best is None:
            self._record_miss()
            return None

        x, y, score, method = best
        confidence = self._score_to_confidence(score, x, y)

        if confidence < self.min_confidence:
            # Treat a low-confidence candidate exactly like a miss so the
            # prediction decay and miss counters keep working.
            self._record_miss()
            return None

        if self.prev_pos is not None:
            jump = math.hypot(x - self.prev_pos[0], y - self.prev_pos[1])
            # Back off the jump gate while the marker is missing: after a few
            # misses the prediction is stale, so a genuinely relocated marker
            # must be allowed back in (otherwise one rejection locks tracking
            # out for the rest of the video).
            tolerance = min(0.6, self.MAX_JUMP * (1.0 + 0.25 * self.miss_streak))
            if jump > tolerance and score < self.JUMP_MIN_SCORE:
                # A distant, low-evidence blob (stale background ghost) must
                # not steal the track; treat it as a miss so the real marker
                # can be re-acquired on the next frame.
                logger.debug(
                    "Rejected teleported candidate at (%.2f, %.2f): jump %.2f score %.2f",
                    x, y, jump, score,
                )
                self._record_miss()
                return None

        predicted = self._predict()

        if predicted is not None and self.prev_pos is not None:
            delta = (x - self.prev_pos[0], y - self.prev_pos[1])
            self.velocity = (
                0.6 * self.velocity[0] + 0.4 * delta[0],
                0.6 * self.velocity[1] + 0.4 * delta[1],
            )

        self.prev_pos = (x, y)
        self.miss_streak = 0
        self.detections += 1

        return MarkerDetection(
            x=x, y=y, confidence=confidence, score=score, method=method, velocity=self.velocity
        )

    def _record_miss(self) -> None:
        """Shared miss bookkeeping (counters, streak, velocity decay)."""
        self.misses += 1
        self.miss_streak += 1
        if self.miss_streak > 3:
            self.velocity = (0.0, 0.0)

    def _motion_mask(self, gray: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(gray, self.background)  # type: ignore[arg-type]
        threshold = max(10.0, float(np.percentile(diff, 92)) * 0.6)
        mask = (diff > threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return mask

    def _appearance_mask(self, hsv: np.ndarray, motion: np.ndarray) -> np.ndarray:
        """Bright-marker gate plus team-coloured dots restricted to motion."""
        sat, val = hsv[..., 1], hsv[..., 2]
        bright = ((val >= 165) & (sat <= 115)).astype(np.uint8)

        if self.marker_hsv:
            custom = np.zeros_like(bright)
            for low, high, _weight in self.marker_hsv:
                in_range = cv2.inRange(
                    hsv, np.array(low, np.uint8), np.array(high, np.uint8)
                )
                custom |= (in_range > 0).astype(np.uint8)
            bright = np.maximum(bright, custom)

        team_blue = cv2.inRange(hsv, np.array([95, 120, 120]), np.array([130, 255, 255]))
        team_red = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
        team_red_alt = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
        team = ((team_blue | team_red | team_red_alt) > 0).astype(np.uint8)
        combined = np.maximum(bright, team & motion)

        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def _update_background(self, gray: np.ndarray, motion: np.ndarray, appearance: np.ndarray) -> None:
        if self._seed_frames < self.SEED_FRAMES:
            # Fast wash-in: the very first frame often contains the marker or
            # minions that are not terrain; converge before tracking starts.
            alpha = np.full_like(gray, self.SEED_ALPHA, dtype=np.float32)
            self._seed_frames += 1
        else:
            static = ((motion == 0) & (appearance == 0)).astype(np.float32)
            alpha = self.BG_ALPHA * static + self.BG_ALPHA_FOREGROUND * (1.0 - static)
        self.background = (1.0 - alpha) * self.background + alpha * gray  # type: ignore[operator]

    def _collect_candidates(
        self, motion: np.ndarray, appearance: np.ndarray, gray: np.ndarray
    ) -> List[Tuple[float, float, float, str]]:
        combined = np.maximum(motion, appearance)
        if combined.sum() < 4:
            return []

        max_area = self.MAX_AREA_FRAC * self.WORK_SIZE * self.WORK_SIZE
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[float, float, float, str]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_AREA or area > max_area:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-6:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            x0, y0, w, h = cv2.boundingRect(contour)
            roi_mask = combined[y0 : y0 + h, x0 : x0 + w]
            motion_ratio = float((motion[y0 : y0 + h, x0 : x0 + w] > 0).mean())
            appearance_ratio = float((appearance[y0 : y0 + h, x0 : x0 + w] > 0).mean())
            if roi_mask.any():
                brightness = float(gray[y0 : y0 + h, x0 : x0 + w][roi_mask > 0].mean()) / 255.0
            else:
                brightness = 0.0

            aspect = w / max(1, h)
            aspect_penalty = 0.25 if (aspect > 3.5 or aspect < 0.28) else 1.0

            size_prior = math.exp(-((area - 45.0) ** 2) / (2.0 * 55.0**2))
            touches_border = (
                x0 <= 1
                or y0 <= 1
                or x0 + w >= self.WORK_SIZE - 1
                or y0 + h >= self.WORK_SIZE - 1
            )
            border_penalty = 0.75 if touches_border else 1.0

            score = (
                0.34 * appearance_ratio
                + 0.30 * motion_ratio
                + 0.16 * brightness
                + 0.20 * size_prior
            ) * aspect_penalty * border_penalty

            method = "bright" if appearance_ratio >= motion_ratio else "motion"
            candidates.append((cx / self.WORK_SIZE, cy / self.WORK_SIZE, float(score), method))

        return candidates

    def _select_candidate(
        self, candidates: List[Tuple[float, float, float, str]]
    ) -> Optional[Tuple[float, float, float, str]]:
        if not candidates:
            return None
        predicted = self._predict()
        if predicted is None:
            return max(candidates, key=lambda c: c[2])

        best_score = -1.0
        best = None
        for cx, cy, score, method in candidates:
            dist = math.hypot(cx - predicted[0], cy - predicted[1])
            prior = math.exp(-(dist**2) / (2 * 0.16**2))
            combined = score * (0.72 + self.prediction_gain * prior)
            if combined > best_score:
                best_score = combined
                # Rank by the prior-boosted score but report the candidate's
                # own (raw) score so confidence measures detection quality.
                best = (cx, cy, score, method)
        return best

    def _predict(self) -> Optional[Tuple[float, float]]:
        if self.prev_pos is None:
            return None
        decay = 0.55 if self.miss_streak == 0 else max(0.1, 0.55 - 0.15 * self.miss_streak)
        px = self.prev_pos[0] + self.velocity[0] * decay
        py = self.prev_pos[1] + self.velocity[1] * decay
        return (min(1.0, max(0.0, px)), min(1.0, max(0.0, py)))

    def _score_to_confidence(self, score: float, x: float, y: float) -> float:
        confidence = (score - 0.16) / 0.42
        if x <= 0.01 or y <= 0.01 or x >= 0.99 or y >= 0.99:
            confidence *= 0.55
        return float(min(1.0, max(0.0, confidence)))

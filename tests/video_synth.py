"""Synthetic gameplay-video generator used by the test-suite.

The generator produces a LoL-like frame: dark HUD chrome, a terrain-coloured
minimap square anchored bottom-left, a bright player marker and a handful of
team-coloured minion dots.  The marker follows a scripted route so tests can
verify that minimap tracking, region mapping and the analytics pipeline
recover a known behaviour.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720
MINIMAP_SIZE = 153
MINIMAP_X = 5
MINIMAP_Y = HEIGHT - MINIMAP_SIZE - 5

WAYPOINTS: Sequence[Tuple[float, float, float]] = (
    (0.00, 0.50, 0.50),
    (0.28, 0.50, 0.50),
    (0.40, 0.42, 0.42),
    (0.55, 0.28, 0.32),
    (0.70, 0.12, 0.18),
    (0.85, 0.10, 0.16),
    (1.00, 0.50, 0.48),
)


def marker_position(t: float) -> Tuple[float, float]:
    """Return the normalised marker position at normalised time ``t`` in [0, 1]."""
    t = max(0.0, min(1.0, t))
    for index in range(len(WAYPOINTS) - 1):
        start, end = WAYPOINTS[index], WAYPOINTS[index + 1]
        if start[0] <= t <= end[0]:
            span = end[0] - start[0] or 1.0
            ratio = (t - start[0]) / span
            smooth = ratio * ratio * (3 - 2 * ratio)
            x = start[1] + (end[1] - start[1]) * smooth
            y = start[2] + (end[2] - start[2]) * smooth
            jitter = 0.008 * math.sin(t * 40.0)
            return max(0.02, min(0.98, x + jitter)), max(0.02, min(0.98, y - jitter))
    return WAYPOINTS[-1][1], WAYPOINTS[-1][2]


def draw_hud(frame: np.ndarray) -> None:
    """Draw cheap HUD chrome so the ROI detector has realistic competition."""
    frame[:] = (24, 26, 30)
    for row in range(0, HEIGHT, 40):
        shade = 26 + (row // 40) % 3
        frame[row : row + 2, :] = (shade, shade, shade + 4)

    cv2.rectangle(frame, (430, HEIGHT - 60), (850, HEIGHT - 12), (45, 48, 54), -1)
    for index in range(5):
        x = 450 + index * 78
        cv2.rectangle(frame, (x, HEIGHT - 52), (x + 64, HEIGHT - 20), (70, 95, 130), -1)
    cv2.rectangle(frame, (900, 8), (1272, 34), (18, 20, 24), -1)
    cv2.putText(frame, "12:34", (1180, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)


def draw_minimap(
    frame: np.ndarray,
    marker: Tuple[float, float],
    others: Sequence[Tuple[float, float]] = (),
) -> None:
    """Render the minimap square with terrain, marker and minion dots."""
    roi = frame[MINIMAP_Y : MINIMAP_Y + MINIMAP_SIZE, MINIMAP_X : MINIMAP_X + MINIMAP_SIZE]
    roi[:] = (70, 150, 60)
    cv2.rectangle(roi, (0, 0), (MINIMAP_SIZE - 1, MINIMAP_SIZE - 1), (45, 88, 132), 12)
    cv2.line(roi, (6, 6), (MINIMAP_SIZE - 6, MINIMAP_SIZE - 6), (165, 115, 45), 14)
    cv2.line(roi, (6, MINIMAP_SIZE - 6), (MINIMAP_SIZE - 6, 6), (95, 95, 95), 6)

    for ox, oy in others:
        cv2.circle(roi, (int(ox * MINIMAP_SIZE), int(oy * MINIMAP_SIZE)), 3, (200, 90, 60), -1)

    mx = MINIMAP_X + int(marker[0] * (MINIMAP_SIZE - 1))
    my = MINIMAP_Y + int(marker[1] * (MINIMAP_SIZE - 1))
    cv2.circle(frame, (mx, my), 5, (255, 255, 255), -1)
    cv2.circle(frame, (mx, my), 5, (40, 40, 40), 1)


def make_synthetic_video(path: str, seconds: int = 100, fps: int = 20) -> str:
    """Write a synthetic gameplay recording to ``path`` and return the path."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {path}")

    total = seconds * fps
    for index in range(total):
        t = index / max(1, total - 1)
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        draw_hud(frame)
        marker = marker_position(t)
        others = [
            (0.5 + 0.05 * math.sin(t * 6 + phase), 0.5 + 0.05 * math.cos(t * 5 + phase))
            for phase in (0.0, 1.7, 3.4, 5.1)
        ]
        draw_minimap(frame, marker, others)
        writer.write(frame)

    writer.release()
    return path


def make_unreadable_video(path: str) -> str:
    """Write a file with a video extension but no decodable frames."""
    with open(path, "wb") as handle:
        handle.write(b"NOT-A-VIDEO" * 64)
    return path

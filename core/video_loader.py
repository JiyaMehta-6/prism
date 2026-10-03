"""Video discovery and validation.

Responsibilities:
* accept only MP4 / MKV / AVI / MOV containers,
* probe technical metadata with OpenCV,
* reject unreadable or corrupted files with a descriptive error instead of
  raising exceptions deeper inside the pipeline.
"""

from __future__ import annotations

import math
import os

import cv2

from core.config import SUPPORTED_EXTENSIONS
from core.logger import get_logger
from core.models import VideoInfo

logger = get_logger("video_loader")


def is_supported(path: str) -> bool:
    """Return ``True`` when ``path`` has a supported video extension."""
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def _finite_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _finite_int(value) -> int:
    number = _finite_float(value)
    try:
        return int(number)
    except (ValueError, OverflowError):
        return 0


def probe_video(path: str) -> VideoInfo:
    """Probe a video file and return its :class:`VideoInfo`.

    Never raises: invalid files are returned with ``valid=False`` and a
    human readable ``error`` message.
    """
    name = os.path.basename(path)
    container = os.path.splitext(path)[1].lstrip(".").lower() or "unknown"

    if not os.path.isfile(path):
        return VideoInfo(path, name, container, 0, 0, 0, 0, 0.0, False, "File not found")
    if not is_supported(path):
        return VideoInfo(
            path, name, container, 0, 0, 0, 0, 0.0, False, f"Unsupported format .{container}"
        )

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return VideoInfo(
                path, name, container, 0, 0, 0, 0, 0.0, False, "Unable to open (corrupted file)"
            )

        fps = _finite_float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = _finite_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = _finite_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _finite_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 1e-6 or frame_count <= 0 or width <= 0 or height <= 0:
            grab_ok, _ = capture.read()
            if not grab_ok:
                return VideoInfo(
                    path, name, container, fps, frame_count, width, height, 0.0,
                    False, "No decodable frames found",
                )
            if fps <= 1e-6:
                fps = 25.0
            if frame_count <= 0:
                # Some containers do not report a frame count.  A decodable
                # frame proves there is content - seek to the end for a
                # duration estimate instead of rejecting the file.
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(1 << 30))
                end_msec = _finite_float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if end_msec > 0:
                    frame_count = int(end_msec / 1000.0 * fps)
                logger.warning(
                    "%s did not report a frame count; estimated %d frames",
                    name, frame_count,
                )

        duration = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
        if not math.isfinite(duration) or duration < 0:
            duration = 0.0

        info = VideoInfo(path, name, container, fps, max(0, frame_count), width, height, duration)
        logger.info(
            "Probed %s: %sx%s @ %.2f fps, %s", name, width, height, fps,
            info.duration_label if duration > 0 else "unknown duration",
        )
        return info
    except Exception as exc:  # noqa: BLE001 - probing must never raise
        logger.error("Probe failed for %s: %s", name, exc)
        return VideoInfo(
            path, name, container, 0, 0, 0, 0, 0.0, False, f"Probe failed: {exc}"
        )
    finally:
        capture.release()

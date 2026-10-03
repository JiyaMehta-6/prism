"""Memory-efficient frame sampling.

Frames are streamed one at a time; skipped frames use ``VideoCapture.grab()``
which advances the decoder without running full deinterlacing/decompression of
the colour planes, keeping RAM usage constant regardless of video length.

Hardware decode (optional, benchmark-gated): on the first video of a session
the FFmpeg backend is asked to decode with the platform's hardware accelerator
(D3D11 on Windows).  The hardware path is only *kept* when it decodes at least
1.25x faster than software for the same file; otherwise (unsupported build,
slow driver, odd codec) software decoding is used for the rest of the session.
Set ``PRISM_HW_DECODE=0`` to skip the benchmark and force software decoding.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from core.logger import get_logger

logger = get_logger("frame_extractor")

FrameTuple = Tuple[int, float, np.ndarray]

# Session-wide hardware-decode decision (None = not yet benchmarked).
# The lock serialises the one-off benchmark when videos run in parallel.
_HW_LOCK = threading.Lock()
_HW_DECISION: Optional[bool] = None
_BENCH_FRAMES = 120
_MIN_SPEEDUP = 1.25


def _hw_available() -> bool:
    return (
        hasattr(cv2, "CAP_PROP_HW_ACCELERATION")
        and hasattr(cv2, "VIDEO_ACCELERATION_D3D11")
        and os.environ.get("PRISM_HW_DECODE", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )


def _hw_open(path: str) -> Optional[cv2.VideoCapture]:
    """Open ``path`` with hardware acceleration; verify a few frames decode."""
    if not _hw_available():
        return None
    params = [
        cv2.CAP_PROP_HW_ACCELERATION,
        cv2.VIDEO_ACCELERATION_D3D11,
        cv2.CAP_PROP_HW_DEVICE,
        0,
    ]
    try:
        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
    except cv2.error:  # backend rejected the params entirely
        return None
    if not capture.isOpened():
        return None
    for _ in range(5):  # this specific file must actually decode
        if not capture.grab():
            capture.release()
            return None
    capture.release()
    try:
        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
    except cv2.error:
        return None
    return capture if capture.isOpened() else None


def _decode_rate(capture: cv2.VideoCapture) -> Tuple[int, float]:
    """Grab up to ``_BENCH_FRAMES`` frames; return (frames, frames/second)."""
    frames = 0
    started = time.perf_counter()
    while frames < _BENCH_FRAMES:
        if not capture.grab():
            break
        frames += 1
    elapsed = max(1e-6, time.perf_counter() - started)
    return frames, frames / elapsed


def _benchmark_hw(path: str) -> bool:
    """True when hardware decode beats software by at least ``_MIN_SPEEDUP``."""
    hw = _hw_open(path)
    if hw is None:
        logger.info("Hardware decode unavailable for this file/build - using software")
        return False
    hw_frames, hw_rate = _decode_rate(hw)
    hw.release()

    sw = cv2.VideoCapture(path)
    if not sw.isOpened():
        return False
    sw_frames, sw_rate = _decode_rate(sw)
    sw.release()

    if sw_frames == 0 or sw_rate <= 0.0:
        return False
    if hw_frames < min(_BENCH_FRAMES, sw_frames):
        logger.info(
            "Hardware decode stopped early (%d/%d frames) - using software",
            hw_frames, sw_frames,
        )
        return False
    speedup = hw_rate / sw_rate
    decision = speedup >= _MIN_SPEEDUP
    logger.info(
        "Decode benchmark: hw %.0f fps vs sw %.0f fps (%.2fx) -> %s",
        hw_rate, sw_rate, speedup, "hardware" if decision else "software",
    )
    return decision


def open_capture(path: str) -> cv2.VideoCapture:
    """Open ``path`` for sequential decoding, preferring benchmarked hardware."""
    global _HW_DECISION

    if not _hw_available():
        return cv2.VideoCapture(path)
    with _HW_LOCK:
        if _HW_DECISION is None:
            try:
                _HW_DECISION = _benchmark_hw(path)
            except cv2.error as exc:  # pragma: no cover - backend quirks
                logger.warning("Hardware decode benchmark failed: %s", exc)
                _HW_DECISION = False
        want_hw = _HW_DECISION
    if want_hw:
        capture = _hw_open(path)
        if capture is not None:
            return capture
        with _HW_LOCK:
            _HW_DECISION = False  # this file did not decode in hardware
        logger.info("Hardware decode rejected for %s - using software", path)
    return cv2.VideoCapture(path)


class FrameExtractor:
    """Yield ``(frame_index, timestamp_seconds, frame)`` at a target rate."""

    def __init__(self, sample_fps: float = 1.0) -> None:
        if not (sample_fps > 0) or not math.isfinite(sample_fps):
            raise ValueError("sample_fps must be positive")
        self.sample_fps = float(sample_fps)
        self.stride = max(1, int(round(30.0 / self.sample_fps)))

    def iter_frames(
        self, path: str, max_frames: Optional[int] = None
    ) -> Generator[FrameTuple, None, None]:
        """Stream sampled frames from ``path``.

        Args:
            path: Video file path.
            max_frames: Optional hard cap on yielded frames.

        Yields:
            Tuples of ``(frame_index, timestamp_seconds, bgr_frame)``.
        """
        capture = open_capture(path)
        if not capture.isOpened():
            logger.error("FrameExtractor cannot open %s", path)
            return

        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(native_fps) or native_fps <= 1e-6:
            native_fps = 30.0
            logger.warning("Unknown FPS for %s, assuming 30", path)

        stride = max(1, int(round(native_fps / self.sample_fps)))
        self.stride = stride
        emitted = 0
        index = 0

        try:
            while True:
                if not capture.grab():
                    break
                if index % stride == 0:
                    ok, frame = capture.retrieve()
                    if not ok or frame is None:
                        index += 1
                        continue
                    yield index, index / native_fps, frame
                    emitted += 1
                    if max_frames is not None and emitted >= max_frames:
                        break
                index += 1
        finally:
            capture.release()

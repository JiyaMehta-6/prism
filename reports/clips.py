"""Export short MP4 clips around roam events from the source recordings.

Fully offline and free: clips are re-encoded with OpenCV's ``mp4v`` codec
(no ffmpeg CLI, no paid tooling).  For every :class:`RoamEvent` the exporter
seeks to ``start - CLIP_PRE_SEC``, decodes until ``end + CLIP_POST_SEC`` and
writes one self-contained clip into ``outputs/clips`` so each roam can be
watched next to the timeline entry in the report.

Game-time vs video-time: roam timestamps follow
:meth:`PositionSample.effective_time`, i.e. they are shifted by the OCR
game-clock offset when one was found.  ``VideoAnalysis.game_time_offset``
holds that shift, so video time = game time - offset (identity when the
offset is unknown).
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional, Sequence

import cv2

from core import config
from core.logger import get_logger
from core.models import RoamEvent, VideoAnalysis

logger = get_logger("clips")

CLIP_PRE_SEC = 5.0
CLIP_POST_SEC = 5.0
# A hard cap keeps a full replay from producing hundreds of files;
# the most interesting roams are exported first (earliest in the match).
MAX_CLIPS_PER_VIDEO = 6


class ClipExportCancelled(Exception):
    """Raised when the user cancels an in-progress clip export."""


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return cleaned or "video"


def _video_time(analysis: VideoAnalysis, timestamp: float) -> float:
    """Convert a game-time timestamp back to seconds within the video file."""
    offset = analysis.game_time_offset
    if offset:
        return timestamp - offset
    return timestamp


def export_clips(
    analyses: Sequence[VideoAnalysis],
    output_dir: Optional[str] = None,
    pre_sec: float = CLIP_PRE_SEC,
    post_sec: float = CLIP_POST_SEC,
    max_per_video: int = MAX_CLIPS_PER_VIDEO,
    should_cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """Write roam clips for every analysed video; return the created paths.

    ``analyses`` is normally ``report.video_analyses``.  The output directory
    defaults to ``PRISM_CLIPS_DIR`` (test hook) then ``outputs/clips``.
    """
    out_dir = (
        output_dir
        or os.environ.get("PRISM_CLIPS_DIR")
        or config.CLIPS_DIR
    )
    os.makedirs(out_dir, exist_ok=True)

    with_events = [a for a in analyses if a.roam_events]
    # Recordings from different folders can share a basename (game.mp4);
    # assign each video a unique stem up-front so one video's export can
    # never silently overwrite another video's clips.
    stems: Dict[int, str] = {}
    used: set[str] = set()
    for analysis in with_events:
        base = _safe_name(os.path.splitext(analysis.info.name)[0])
        stem = base
        suffix = 2
        while stem in used:
            stem = f"{base}_{suffix}"
            suffix += 1
        used.add(stem)
        stems[id(analysis)] = stem
    total = sum(min(len(a.roam_events), max(1, max_per_video)) for a in with_events)
    done = 0
    written: List[str] = []
    for analysis in with_events:
        if should_cancel and should_cancel():
            raise ClipExportCancelled()
        events = sorted(analysis.roam_events, key=lambda r: r.start_time)
        events = events[: max(1, max_per_video)]
        written.extend(
            _export_video_clips(
                analysis, events, out_dir, pre_sec, post_sec, should_cancel,
                stem=stems[id(analysis)],
            )
        )
        done += len(events)
        if progress:
            progress(done, max(1, total))
    return written


def _export_video_clips(
    analysis: VideoAnalysis,
    events: Sequence[RoamEvent],
    out_dir: str,
    pre_sec: float,
    post_sec: float,
    should_cancel: Optional[Callable[[], bool]],
    stem: Optional[str] = None,
) -> List[str]:
    info = analysis.info
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        logger.warning("Clip export: cannot open %s", info.path)
        return []

    fps = info.fps if info.fps and info.fps > 1e-3 else 30.0
    size = (int(info.width) or 1280, int(info.height) or 720)
    if not stem:
        stem = _safe_name(os.path.splitext(info.name)[0])
    written: List[str] = []
    try:
        for index, roam in enumerate(events, start=1):
            if should_cancel and should_cancel():
                raise ClipExportCancelled()
            t0 = max(0.0, _video_time(analysis, roam.start_time) - pre_sec)
            t1 = _video_time(analysis, roam.end_time) + post_sec
            if info.duration_sec > 0.0:
                t1 = min(t1, info.duration_sec)
            if t1 - t0 < 1.0:
                continue
            minutes, seconds = divmod(int(t0), 60)
            name = f"{stem}_roam_{index:02d}_{minutes:02d}m{seconds:02d}s.mp4"
            path = os.path.join(out_dir, name)
            if _write_clip(cap, t0, t1, path, fps, size):
                written.append(path)
                logger.info("Clip written: %s", path)
    finally:
        cap.release()
    return written


def _write_clip(
    cap: cv2.VideoCapture,
    t0: float,
    t1: float,
    path: str,
    fps: float,
    size: tuple,
) -> bool:
    """Decode ``[t0, t1]`` from an open capture into ``path`` (mp4v)."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    writer: Optional[cv2.VideoWriter] = None
    frames = 0
    # Safety bound: some backends report a lagging POS_MSEC; never decode
    # more than 1.5x the requested span (plus slack) even if the position
    # reading misbehaves.
    max_frames = int((t1 - t0) * fps * 1.5) + int(fps) + 30
    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pos_sec > t1:
                break
            if writer is None:
                writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size
                )
                if not writer.isOpened():
                    logger.warning("Clip export: writer failed for %s", path)
                    return False
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
            frames += 1
    finally:
        if writer is not None:
            writer.release()
    if frames == 0:
        # A failed or empty export must not leave a 0-byte stub behind.
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        logger.warning("Clip export: no frames decoded for %s", path)
        return False
    return True

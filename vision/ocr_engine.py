"""Local OCR subsystem built exclusively on EasyOCR.

OCR is *supplementary* in PRISM: behavioural analysis is driven by movement,
so a failed OCR pass degrades confidence scores but never aborts a run.

Pipeline::

    ROI selection -> OpenCV preprocessing variants -> EasyOCR (offline)
                   -> confidence evaluation -> accept / retry / unavailable

The engine is lazy-loaded (EasyOCR + torch are only imported on first use) so
the GUI starts instantly even when OCR is disabled.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.logger import get_logger
from vision.confidence_manager import ConfidenceManager
from vision.ocr_preprocessor import OCRPreprocessor

logger = get_logger("ocr_engine")

# The EasyOCR reader is created once per process and shared by every
# OCREngine instance.  When videos are analysed in parallel worker threads,
# both lazy initialisation and inference must be serialised: the reader is
# not thread-safe and two threads must never construct two readers.
_READER_LOCK = threading.RLock()

DEFAULT_ROIS: Dict[str, List[List[float]]] = {
    # Ordered cheapest-first: the common layout (replay HUD with a centred
    # clock) is read first so ``read_best`` early-exits instead of burning
    # variants on the legacy spectator corners.  Extra entries only cost
    # time when the leading ones fail, so alternative HUD layouts still work.
    "game_timer": [
        [0.472, 0.057, 0.062, 0.034],
        [0.870, 0.004, 0.125, 0.045],
        [0.440, 0.004, 0.120, 0.045],
        [0.900, 0.030, 0.095, 0.035],
    ],
    "kda": [
        [0.010, 0.855, 0.140, 0.045],
        [0.380, 0.930, 0.150, 0.050],
    ],
    "objective_feed": [
        [0.380, 0.180, 0.240, 0.070],
    ],
    # Top-centre team scoreboard: blue kills are printed left, red kills
    # right, separated by a sword glyph - one tight ROI per side keeps the
    # two digits from being merged into a single ambiguous number.
    "kill_score_blue": [[0.477, 0.005, 0.030, 0.043]],
    "kill_score_red": [[0.513, 0.005, 0.030, 0.043]],
}

_TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")
_SCORE_PATTERN = re.compile(r"^\d{1,2}$")


@dataclass
class OcrResult:
    """A recognised string with its confidence."""

    text: str
    confidence: float
    variant: str
    elapsed: float


def parse_clock(text: str) -> Optional[int]:
    """Parse ``MM:SS`` / ``H:MM:SS`` into total seconds."""
    cleaned = text.strip().replace(" ", "")
    if not _TIME_PATTERN.match(cleaned):
        return None
    parts = [int(p) for p in cleaned.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        if seconds > 59:
            return None
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        if minutes > 59 or seconds > 59:
            return None
        return hours * 3600 + minutes * 60 + seconds
    return None


def sanitize_scoreboard(text: str) -> str:
    """Extract a KDA-style triple, falling back to safe characters only."""
    match = re.search(r"\d{1,3}\s*/\s*\d{1,3}\s*/\s*\d{1,3}", text)
    if match:
        return re.sub(r"\s+", "", match.group(0))
    return re.sub(r"[^0-9/:.\-]", "", text)


def parse_team_kills(text: str) -> Optional[int]:
    """Parse a team-kill count (one or two digits, 0-80); ``None`` otherwise.

    Anything but a bare digit string is rejected, so clock-like text
    (``"00:05"``) or KDA fragments can never be mistaken for a score.
    """
    cleaned = (text or "").strip()
    if not _SCORE_PATTERN.match(cleaned):
        return None
    value = int(cleaned)
    return value if 0 <= value <= 80 else None


_SHARED_READER = None


def _allow_download() -> bool:
    return os.environ.get("PRISM_ALLOW_DOWNLOAD", "0").strip().lower() in ("1", "true", "yes")


def _models_present() -> bool:
    """True when the expected EasyOCR model files already exist locally."""
    detector_names = ("craft_mlt_25k.pth", "craft_mlt.pth")
    recognizer_names = ("english_g2.pth", "english.pth", "latin.pth")
    roots: List[str] = []
    for var in ("EASYOCR_MODULE_PATH", "MODULE_PATH"):
        base = os.environ.get(var)
        if base:
            roots.append(os.path.join(base, "model"))
            roots.append(base)
    roots.append(os.path.join(os.path.expanduser("~"), ".EasyOCR", "model"))
    for root in roots:
        try:
            names = set(os.listdir(root))
        except OSError:
            continue
        if any(d in names for d in detector_names) and any(
            r in names for r in recognizer_names
        ):
            return True
    return False


class OCREngine:
    """Offline OCR wrapper with variant retry and confidence gating."""

    def __init__(
        self,
        confidence_manager: ConfidenceManager,
        min_confidence: float = 0.55,
        rois: Optional[Dict[str, List[List[float]]]] = None,
        enabled: bool = True,
    ) -> None:
        self.confidence_manager = confidence_manager
        self.min_confidence = float(min_confidence)
        self.rois = rois or DEFAULT_ROIS
        self.enabled = bool(enabled)
        self.preprocessor = OCRPreprocessor(upscale=3)
        self._reader = None
        self._reader_failed = False
        self._init_failures = 0
        self.calls = 0
        self.failures = 0
        self.exceptions = 0
        self._last_error = ""
        self._warned_keys: set = set()

    @property
    def available(self) -> bool:
        """True when EasyOCR has been imported successfully."""
        return self._reader is not None or self._init_reader()

    def _init_reader(self) -> bool:
        global _SHARED_READER

        with _READER_LOCK:
            if self._reader is not None:
                return True
            if _SHARED_READER is not None:
                self._reader = _SHARED_READER
                return True
            if self._reader_failed or not self.enabled:
                return False

            download = _allow_download()
            if not download and not _models_present():
                # Never touch the network silently: PRISM is an offline tool.
                # Missing models are permanent for this session (latched), but a
                # transient initialisation error is retried on the next call.
                self._reader_failed = True
                logger.warning(
                    "OCR models are not installed locally and downloads are disabled. "
                    "Run the prefetch command in docs/INSTALLATION.md (or set "
                    "PRISM_ALLOW_DOWNLOAD=1 once while online). OCR is disabled; "
                    "analysis continues with video time."
                )
                return False
            try:
                import easyocr  # noqa: PLC0415 - lazy heavy import

                logger.info("Initialising EasyOCR (local, CPU%s)...",
                            "" if download else ", no downloads")
                started = time.perf_counter()
                reader = easyocr.Reader(
                    ["en"], gpu=False, verbose=False, download_enabled=download
                )
                _SHARED_READER = reader
                self._reader = reader
                logger.info("EasyOCR ready in %.1fs", time.perf_counter() - started)
                return True
            except Exception as exc:  # pragma: no cover - depends on host machine
                # Do NOT latch: a one-off import/init failure (missing torch CUDA,
                # locked file) must not disable OCR for the whole process.
                self._init_failures += 1
                if self._init_failures >= 3:
                    self._reader_failed = True
                logger.warning("EasyOCR initialisation failed (attempt %d): %s",
                               self._init_failures, exc)
                return False

    def extract_rois(self, frame: np.ndarray, key: str) -> List[np.ndarray]:
        """Cut every configured ROI registered under ``key`` (clamped)."""
        height, width = frame.shape[:2]
        crops: List[np.ndarray] = []
        for rel in self.rois.get(key, []):
            if len(rel) != 4:
                continue
            try:
                rx, ry, rw, rh = (float(v) for v in rel[:4])
            except (TypeError, ValueError):
                continue
            if not all(map(math.isfinite, (rx, ry, rw, rh))):
                continue
            # Clamp fractions into the frame so negative/oversized ROIs can
            # never read the wrong edge or raise.
            rx, ry = min(max(rx, 0.0), 1.0), min(max(ry, 0.0), 1.0)
            rw = min(max(rw, 0.0), 1.0 - rx)
            rh = min(max(rh, 0.0), 1.0 - ry)
            x, y = int(rx * width), int(ry * height)
            w, h = int(rw * width), int(rh * height)
            if w <= 0 or h <= 0:
                continue
            crop = frame[y : y + h, x : x + w]
            if crop.size:
                crops.append(crop)
        return crops

    def read_best(
        self,
        frame: np.ndarray,
        key: str,
        allowlist: str = "",
        sanitizer=None,
        validator=None,
    ) -> Optional[OcrResult]:
        """Run the full variant loop for ``key`` and keep the best reading.

        ``validator`` filters readings by *content* (e.g. parseable clock) so
        that a confident but unusable string can never shadow a usable one
        found in another ROI or variant.
        """
        if not self.enabled or not self.available:
            return None

        crops = self.extract_rois(frame, key)
        if not crops:
            return None

        best: Optional[OcrResult] = None
        top_any: Optional[OcrResult] = None
        early_exit = max(0.90, self.min_confidence)
        for crop in crops:
            for variant_name, prepared in self.preprocessor.variants(crop):
                for result in self._recognise(prepared, allowlist, variant_name):
                    if sanitizer:
                        result.text = sanitizer(result.text)
                        if not result.text:
                            continue
                    if top_any is None or result.confidence > top_any.confidence:
                        top_any = result
                    if validator is not None and not validator(result.text):
                        continue
                    # On a confidence tie prefer the longer text: a two-digit
                    # score split into boxes ("1" + "0") must lose to the
                    # merged "10", or 10+ kills would be under-counted.
                    if best is None or result.confidence > best.confidence or (
                        abs(result.confidence - best.confidence) <= 1e-9
                        and len(result.text) > len(best.text)
                    ):
                        best = result
                    if best.confidence >= early_exit:
                        break
                if best is not None and best.confidence >= early_exit:
                    break
            if best is not None and best.confidence >= early_exit:
                break

        if best is None:
            self.failures += 1
            if self.exceptions and top_any is None:
                # OCR itself blew up (not just rejected text): record a
                # synthetic zero-confidence reading so the key shows up in the
                # confidence report instead of disappearing silently.
                self.confidence_manager.record(key, "", 0.0, self.min_confidence)
                if key not in self._warned_keys:
                    self._warned_keys.add(key)
                    logger.warning(
                        "EasyOCR raised errors reading '%s' (first: %s)",
                        key, self._last_error,
                    )
            elif top_any is not None:
                # Confident but unusable text: record it with a floor above 1.0
                # so it is always rejected, yet still shows up in the
                # confidence report instead of failing silently.
                self.confidence_manager.record(key, top_any.text, top_any.confidence, 1.01)
                logger.debug(
                    "OCR '%s' produced no usable reading (best=%r conf=%.2f)",
                    key, top_any.text, top_any.confidence,
                )
            return None
        # Record every attempt (including rejected ones) so the confidence
        # manager can report low-confidence keys instead of staying silent.
        self.confidence_manager.record(key, best.text, best.confidence, self.min_confidence)
        if best.confidence < self.min_confidence:
            self.failures += 1
            logger.debug("OCR '%s' rejected (conf=%.2f): %r", key, best.confidence, best.text)
            return None
        return best

    def _recognise(
        self, image: np.ndarray, allowlist: str, variant: str
    ) -> List[OcrResult]:
        """Recognise one variant; return one candidate per text box.

        Validating per-box (instead of a single concatenation) lets a usable
        clock string survive when neighbouring boxes hold junk.
        """
        started = time.perf_counter()
        try:
            # detail=1 returns (bbox, text, confidence); detail=0 would return
            # bare strings with no confidence and gate every reading out.
            kwargs: Dict[str, object] = {"detail": 1}
            if allowlist:
                kwargs["allowlist"] = allowlist
            with _READER_LOCK:  # shared reader: serialise across worker threads
                output = self._reader.readtext(image, **kwargs)  # type: ignore[union-attr]
            self.calls += 1
        except Exception as exc:
            self.failures += 1
            self.exceptions += 1
            self._last_error = str(exc)
            logger.debug("EasyOCR failure on variant %s: %s", variant, exc)
            return []

        if not output:
            return []

        candidates: List[OcrResult] = []
        texts: List[str] = []
        confidences: List[float] = []
        for entry in output:
            if isinstance(entry, (tuple, list)) and len(entry) >= 3:
                text = str(entry[1]).strip()
                try:
                    conf = float(entry[2])
                except (TypeError, ValueError):
                    conf = 0.5
            else:
                # Backend returned a bare string: no confidence available, use
                # a neutral value so the configured floor still applies.
                text = str(entry).strip()
                conf = 0.5
            if not text:
                continue
            texts.append(text)
            confidences.append(conf)
            candidates.append(
                OcrResult(
                    text=text,
                    confidence=conf,
                    variant=variant,
                    elapsed=time.perf_counter() - started,
                )
            )

        # Also offer the concatenation (some engines split a clock across
        # boxes); per-box candidates already in the list keep it usable.
        merged = "".join(texts).strip()
        if merged and len(texts) > 1:
            candidates.append(
                OcrResult(
                    text=merged,
                    confidence=float(sum(confidences) / len(confidences)),
                    variant=variant,
                    elapsed=time.perf_counter() - started,
                )
            )
        return candidates

    def read_game_clock(self, frame: np.ndarray, video_time: float) -> Optional[Tuple[int, float]]:
        """Best-effort match timer extraction; returns ``(seconds, confidence)``.

        Only readings that actually parse as a clock are considered, so a
        confident garbage string from one ROI can never beat a parseable
        reading from another.
        """
        result = self.read_best(
            frame,
            "game_timer",
            allowlist="0123456789:",
            validator=lambda text: parse_clock(text) is not None,
        )
        if result is None:
            return None
        seconds = parse_clock(result.text)
        if seconds is None:
            return None
        logger.debug("Game clock %s (%.2f) at video %.1fs", result.text, result.confidence, video_time)
        return seconds, result.confidence

    def read_kill_score(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Read the top-centre team scoreboard; ``(blue, red, confidence)``.

        Both sides must be readable - a half-known score can never produce
        trustworthy kill deltas, so the pair is discarded as a whole.
        """
        if not self.enabled or not self.available:
            return None
        scores: List[int] = []
        confidence = 1.0
        for key in ("kill_score_blue", "kill_score_red"):
            if not self.rois.get(key):
                continue
            result = self.read_best(
                frame,
                key,
                allowlist="0123456789",
                sanitizer=None,
                validator=lambda text: parse_team_kills(text) is not None,
            )
            if result is None:
                return None
            value = parse_team_kills(result.text)
            if value is None:
                return None
            scores.append(value)
            confidence = min(confidence, result.confidence)
        if len(scores) != 2:
            return None
        return scores[0], scores[1], confidence

"""Runtime configuration for PRISM.

Settings are stored as JSON in ``data/settings.json`` so that preferences
survive restarts. Every field has a safe default; a corrupt settings file is
discarded silently and replaced with defaults.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PROFILE_DIR = os.path.join(DATA_DIR, "profiles")
CHAMPION_DIR = os.path.join(DATA_DIR, "champions")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
LOG_PATH = os.path.join(LOG_DIR, "analysis.log")

SUPPORTED_EXTENSIONS: List[str] = [".mp4", ".mkv", ".avi", ".mov"]

# Scalar region anchors users may override (see vision.region_mapper).
# Values are clamped here so a hand-edited settings file can never produce a
# degenerate geometry (negative radius, lane inset past the map centre, ...).
ANCHOR_LIMITS: Dict[str, tuple] = {
    "lane_inset": (0.01, 0.25),
    "lane_threshold": (0.01, 0.25),
    "base_radius": (0.02, 0.30),
    "dragon_radius": (0.02, 0.20),
    "herald_radius": (0.02, 0.20),
    "river_threshold": (0.01, 0.20),
    "mid_threshold": (0.01, 0.20),
    "river_start": (0.0, 0.60),
    "river_end": (0.40, 1.0),
    "tower_radius": (0.01, 0.20),
}



def _num(value: Any, low: float, high: float, default: float) -> float:
    """Coerce ``value`` to a float clamped into ``[low, high]`` (never raises)."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if number != number or number in (float("inf"), float("-inf")):
        return float(default)
    return float(min(high, max(low, number)))


def _int(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)
    return int(min(high, max(low, number)))


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(default)


@dataclass
class Settings:
    """User configurable analysis parameters."""

    sample_fps: float = 1.0
    early_game_minutes: float = 15.0
    minimap_side: str = "auto"
    minimap_roi: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    marker_min_confidence: float = 0.30
    max_interpolate_gap: int = 4
    roam_min_duration_sec: float = 8.0
    roam_max_lane_absence_sec: float = 45.0
    window_plan: List[float] = field(default_factory=lambda: [0.0, 3.0, 7.0, 12.0, 15.0])
    ocr_enabled: bool = True
    ocr_interval_sec: float = 30.0
    # 40 reads x 30 s covers the whole 15-minute window (30 reads needed) for
    # both the match clock and the kill-scoreboard event tracking.
    ocr_max_calls_per_video: int = 40
    ocr_min_confidence: float = 0.55
    use_game_clock: bool = True
    # Heuristic ward-like bloom detection on the minimap (pure CV, offline).
    ward_tracking: bool = True
    similarity_top_n: int = 5
    player_label: str = "Player"
    video_titles: Dict[str, str] = field(default_factory=dict)
    # Partial override of the scalar region anchors (HUD scaling / map re-skins);
    # unknown keys and out-of-range values are dropped by validated().
    region_anchors: Dict[str, float] = field(default_factory=dict)

    def validated(self) -> "Settings":
        """Clamp/coerce every field so bad input can never crash the app."""
        self.sample_fps = _num(self.sample_fps, 0.1, 30.0, 1.0)
        # PRISM's scope is the 0-15 min early game; never accept more.
        self.early_game_minutes = _num(self.early_game_minutes, 1.0, 15.0, 15.0)
        self.marker_min_confidence = _num(self.marker_min_confidence, 0.0, 0.95, 0.30)
        self.max_interpolate_gap = _int(self.max_interpolate_gap, 0, 30, 4)
        self.roam_min_duration_sec = _num(self.roam_min_duration_sec, 1.0, 300.0, 8.0)
        self.roam_max_lane_absence_sec = _num(
            self.roam_max_lane_absence_sec, self.roam_min_duration_sec, 1800.0, 45.0
        )
        self.ocr_interval_sec = _num(self.ocr_interval_sec, 5.0, 3600.0, 30.0)
        self.ocr_max_calls_per_video = _int(self.ocr_max_calls_per_video, 0, 1000, 40)
        self.ocr_min_confidence = _num(self.ocr_min_confidence, 0.0, 1.0, 0.55)
        self.similarity_top_n = _int(self.similarity_top_n, 1, 50, 5)
        self.ocr_enabled = _bool(self.ocr_enabled, True)
        self.use_game_clock = _bool(self.use_game_clock, True)
        self.ward_tracking = _bool(self.ward_tracking, True)
        self.player_label = str(self.player_label or "Player").strip() or "Player"
        self.minimap_side = self.minimap_side if self.minimap_side in (
            "auto", "left", "right"
        ) else "auto"

        roi = self.minimap_roi
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            coerced = [_num(roi[i], 0.0, 1.0, 0.0) for i in range(4)]
        else:
            coerced = [0.0, 0.0, 0.0, 0.0]
        if not all(v > 0 for v in coerced):
            coerced = [0.0, 0.0, 0.0, 0.0]
        self.minimap_roi = coerced

        plan: List[float] = []
        if isinstance(self.window_plan, (list, tuple)):
            for item in self.window_plan:
                try:
                    plan.append(float(item))
                except (TypeError, ValueError, OverflowError):
                    continue
        plan = sorted({round(value, 3) for value in plan if value >= 0})
        # Phase boundaries must stay inside the analysis window so no bucket
        # lies beyond ``early_game_minutes``.
        window = round(self.early_game_minutes, 3)
        plan = [value for value in plan if value <= window]
        # Re-anchor to the window: the first bucket must start at 0 and the
        # last must end exactly at the window edge, otherwise timeline/roam
        # bucketing silently drops every sample after the last edge (e.g. a
        # stale 15-minute plan under a 10-minute window).
        if plan and plan[0] > 0.0:
            plan.insert(0, 0.0)
        if plan and plan[-1] < window:
            plan.append(window)
        if len(plan) < 3:
            plan = [0.0, round(window / 3.0, 3), round(2.0 * window / 3.0, 3), window]
        self.window_plan = plan

        if not isinstance(self.video_titles, dict):
            self.video_titles = {}
        else:
            self.video_titles = {str(k): str(v) for k, v in self.video_titles.items()}

        anchors: Dict[str, float] = {}
        if isinstance(self.region_anchors, dict):
            for key, raw in self.region_anchors.items():
                limits = ANCHOR_LIMITS.get(str(key))
                if limits is None:
                    continue
                try:
                    number = float(raw)
                except (TypeError, ValueError, OverflowError):
                    continue
                if number != number or number in (float("inf"), float("-inf")):
                    continue
                low, high = limits
                anchors[str(key)] = float(min(high, max(low, number)))
        # The river is modelled as the segment (start, start)-(end, end): a
        # degenerate or reversed segment would classify nothing as river.
        if anchors.get("river_start", 0.20) >= anchors.get("river_end", 0.80):
            anchors.pop("river_start", None)
            anchors.pop("river_end", None)
        self.region_anchors = anchors
        return self


def ensure_directories() -> None:
    """Create every directory the application writes to."""
    for path in (DATA_DIR, LOG_DIR, OUTPUT_DIR, PROFILE_DIR, CHART_DIR, CLIPS_DIR):
        os.makedirs(path, exist_ok=True)


def load_settings() -> Settings:
    """Load settings from disk, falling back to defaults on any problem."""
    settings = Settings()
    try:
        ensure_directories()
        import json

        with open(SETTINGS_PATH, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            known = {f.name for f in fields(Settings)}
            for key, value in raw.items():
                if key in known:
                    setattr(settings, key, value)
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        settings = Settings()
    try:
        return settings.validated()
    except Exception:  # noqa: BLE001 - validation must never block startup
        return Settings()


def save_settings(settings: Settings) -> bool:
    """Persist settings to disk. Returns True on success, never raises."""
    import json

    tmp_path = SETTINGS_PATH + ".tmp"
    try:
        ensure_directories()
        payload = asdict(settings.validated())
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, SETTINGS_PATH)
        return True
    except Exception as exc:  # noqa: BLE001 - persistence must never crash a slot
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        from core.logger import get_logger

        get_logger("config").error("Could not save settings: %s", exc)
        return False

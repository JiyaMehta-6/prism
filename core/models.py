"""Data models shared across the PRISM pipeline.

Every stage of the pipeline exchanges plain, serialisable dataclasses so that
results can be persisted to JSON, rendered in the GUI and embedded in PDF
reports without additional translation layers.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

REGIONS: List[str] = [
    "Base",
    "Top Lane",
    "Mid Lane",
    "Bot Lane",
    "River",
    "Blue Jungle",
    "Red Jungle",
    "Dragon Area",
    "Herald Area",
]

FINGERPRINT_METRICS: List[str] = [
    "Aggression",
    "Roaming",
    "Vision",
    "Objectives",
    "Risk",
    "Consistency",
    "Pressure Stability",
]


@dataclass
class VideoInfo:
    """Technical metadata describing a single gameplay recording."""

    path: str
    name: str
    container: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    valid: bool = True
    error: str = ""

    @property
    def duration_label(self) -> str:
        total = self.duration_sec
        if not isinstance(total, (int, float)) or not math.isfinite(float(total)):
            total = 0.0
        minutes, seconds = divmod(int(max(0.0, total)), 60)
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class PositionSample:
    """One timestamped player-position observation on the minimap."""

    video_time: float
    frame_index: int
    x: float
    y: float
    confidence: float
    region: str
    game_time: Optional[float] = None
    detected: bool = True

    @property
    def effective_time(self) -> float:
        return self.game_time if self.game_time is not None else self.video_time


@dataclass
class RoamEvent:
    """A detected excursion out of the player's assigned lane."""

    start_time: float
    end_time: float
    origin_region: str
    destination_region: str
    path: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class OcrReading:
    """A single OCR observation together with its confidence."""

    key: str
    value: str
    confidence: float
    video_time: float


@dataclass
class GameEvent:
    """A HUD-observed match event (currently team-kill score changes).

    ``time`` is expressed in game time whenever a clock offset is known,
    matching :attr:`PositionSample.effective_time` so events, roams and
    timeline windows share one timeline.
    """

    time: float
    kind: str
    team: str
    detail: str
    confidence: float = 1.0
    count: int = 1


@dataclass
class VideoAnalysis:
    """Full analysis output for one video."""

    info: VideoInfo
    samples: List[PositionSample] = field(default_factory=list)
    roam_events: List[RoamEvent] = field(default_factory=list)
    region_time: Dict[str, float] = field(default_factory=dict)
    transitions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    ocr_readings: List[OcrReading] = field(default_factory=list)
    events: List[GameEvent] = field(default_factory=list)
    game_time_offset: Optional[float] = None
    detection_rate: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    champion: Optional[str] = None
    champion_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["info"] = asdict(self.info)
        return data


@dataclass
class Fingerprint:
    """Normalised behavioural identity of a player (0-10 per metric)."""

    scores: Dict[str, float] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    contributions: Dict[str, str] = field(default_factory=dict)

    def get(self, metric: str) -> float:
        return float(self.scores.get(metric, 0.0))


@dataclass
class Archetype:
    """A labelled playstyle category with human readable justification."""

    label: str
    confidence: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class Insight:
    """An actionable, confidence-scored improvement finding."""

    title: str
    detail: str
    suggestion: str
    confidence: float
    severity: str = "info"
    evidence: List[str] = field(default_factory=list)


@dataclass
class TimelineWindow:
    """Behavioural summary for one slice of the early game."""

    start: float
    end: float
    label: str
    summary: str
    confidence: float
    region_distribution: Dict[str, float] = field(default_factory=dict)
    aggression_index: float = 0.0
    kills: int = 0

    @property
    def start_label(self) -> str:
        return _clock(self.start)

    @property
    def end_label(self) -> str:
        return _clock(self.end)


@dataclass
class SimilarityResult:
    """Behavioural similarity against previously stored profiles."""

    profile_name: str
    similarity: float
    note: str = ""


@dataclass
class PlayerProfile:
    """Aggregate behavioural statistics across several videos of one player."""

    player_label: str
    video_count: int
    total_samples: int
    detection_rate: float
    region_distribution: Dict[str, float] = field(default_factory=dict)
    transition_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    roam_rate: float = 0.0
    roam_destinations: Dict[str, int] = field(default_factory=dict)
    roam_paths: Dict[str, int] = field(default_factory=dict)
    average_metrics: Dict[str, float] = field(default_factory=dict)
    per_video_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metric_variability: Dict[str, float] = field(default_factory=dict)
    own_base: str = "Blue"
    quality_notes: List[str] = field(default_factory=list)
    histogram: List[List[float]] = field(default_factory=list)
    champion: Optional[str] = None
    champion_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    """Everything the GUI and the PDF exporter need to present results."""

    profile: PlayerProfile
    fingerprint: Fingerprint
    archetypes: List[Archetype]
    insights: List[Insight]
    timeline: List[TimelineWindow]
    similarity: List[SimilarityResult]
    chart_paths: Dict[str, str] = field(default_factory=dict)
    video_analyses: List[VideoAnalysis] = field(default_factory=list)
    sections: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    settings_snapshot: Dict[str, Any] = field(default_factory=dict)

    @property
    def headline(self) -> str:
        if self.archetypes:
            return self.archetypes[0].label
        return "Unclassified Profile"


def _clock(seconds: float) -> str:
    total = int(max(0.0, seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def save_json(path: str, payload: Any) -> None:
    """Serialise ``payload`` to ``path`` atomically, creating parent dirs."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
    except OSError:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def load_json(path: str) -> Any:
    """Load a JSON document, returning ``None`` when it cannot be read."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return None

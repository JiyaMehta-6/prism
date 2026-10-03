"""Unit tests for the deterministic parts of the PRISM pipeline."""

from __future__ import annotations

from analytics.events import (
    extract_kill_events,
    kills_during_roams,
    own_team_kills,
    parse_score_pair,
)
from core.behavior_engine import BehaviorEngine
from core.config import Settings, load_settings
from core.fingerprint_engine import FingerprintEngine
from core.frame_extractor import FrameExtractor
from core.models import FINGERPRINT_METRICS, OcrReading, PlayerProfile, RoamEvent
from core.video_loader import is_supported, probe_video
from tests.video_synth import make_synthetic_video, make_unreadable_video
from vision.confidence_manager import ConfidenceManager
from vision.ocr_engine import parse_clock, parse_team_kills, sanitize_scoreboard
from vision.region_mapper import RegionMapper


def test_supported_extensions() -> None:
    assert is_supported("clip.mp4")
    assert is_supported("clip.MKV")
    assert is_supported("clip.avi")
    assert is_supported("clip.mov")
    assert not is_supported("clip.gif")
    assert not is_supported("replay.json")


def test_probe_rejects_corrupted_video(tmp_path) -> None:
    bad = make_unreadable_video(str(tmp_path / "broken.mp4"))
    info = probe_video(bad)
    assert not info.valid
    assert info.error


def test_probe_missing_file(tmp_path) -> None:
    info = probe_video(str(tmp_path / "nope.mp4"))
    assert not info.valid
    assert "not found" in info.error.lower()


def test_frame_extractor_sampling_rate(tmp_path) -> None:
    path = make_synthetic_video(str(tmp_path / "sample.mp4"), seconds=6, fps=20)
    frames = list(FrameExtractor(sample_fps=2.0).iter_frames(path))
    assert 10 <= len(frames) <= 15
    assert all(frame is not None for _, _, frame in frames)
    times = [t for _, t, _ in frames]
    assert times == sorted(times)


def test_region_mapper_known_positions() -> None:
    mapper = RegionMapper()
    assert mapper.classify(0.05, 0.95) == "Base"
    assert mapper.classify(0.95, 0.05) == "Base"
    assert mapper.classify(0.66, 0.70) == "Dragon Area"
    assert mapper.classify(0.34, 0.28) == "Herald Area"
    assert mapper.classify(0.60, 0.40) == "Mid Lane"
    assert mapper.classify(0.60, 0.60) == "River"
    assert mapper.classify(0.03, 0.30) == "Top Lane"
    assert mapper.classify(0.90, 0.95) == "Bot Lane"
    assert mapper.classify(0.25, 0.50) == "Blue Jungle"
    assert mapper.classify(0.75, 0.50) == "Red Jungle"


def test_forward_bias_sign() -> None:
    mapper = RegionMapper()
    near_blue = mapper.forward_bias(0.15, 0.85, "Blue")
    near_red = mapper.forward_bias(0.85, 0.15, "Blue")
    assert near_blue < 0.0 < near_red


def test_clock_parsing() -> None:
    assert parse_clock("12:34") == 754
    assert parse_clock("1:02:33") == 3753
    assert parse_clock("00:00") == 0
    assert parse_clock("12:99") is None
    assert parse_clock("abc") is None
    assert parse_clock("12-34") is None
    assert sanitize_scoreboard("K/D/A: 5/2/7!") == "5/2/7"


def test_confidence_manager_gates_values() -> None:
    manager = ConfidenceManager(default_threshold=0.6)
    manager.record("game_timer", "10:00", 0.42)
    assert manager.value("game_timer") is None
    assert manager.status("game_timer") != "accepted"

    manager.record("game_timer", "10:05", 0.91)
    assert manager.value("game_timer") == "10:05"
    assert manager.status("game_timer") == "accepted"
    assert manager.confidence("game_timer") >= 0.9

    summary = manager.summary()
    assert summary["game_timer"]["attempts"] == 2


def _stub_profile() -> PlayerProfile:
    return PlayerProfile(
        player_label="Tester",
        video_count=3,
        total_samples=600,
        detection_rate=0.9,
        region_distribution={
            "Base": 0.05,
            "Top Lane": 0.10,
            "Mid Lane": 0.40,
            "Bot Lane": 0.05,
            "River": 0.15,
            "Blue Jungle": 0.10,
            "Red Jungle": 0.05,
            "Dragon Area": 0.05,
            "Herald Area": 0.05,
        },
        average_metrics={
            "region_Top Lane": 0.35,
            "region_Mid Lane": 0.41,
            "region_River": 0.14,
            "forward_high_fraction": 0.45,
            "enemy_territory_fraction": 0.30,
            "lane_fraction": 0.60,
            "river_fraction": 0.15,
            "jungle_fraction": 0.15,
            "roam_rate_per_min": 0.55,
            "roam_destinations_unique": 3.0,
            "burst_fraction": 0.25,
            "erratic_turn_rate": 0.20,
            "vision_proxy_fraction": 0.30,
            "objective_fraction": 0.10,
            "tower_presence_fraction": 0.30,
            "late_enemy_fraction": 0.35,
            "window_forward_std": 0.10,
            "window_speed_std": 0.02,
            "window_burst_growth": 0.05,
            "speed_mean": 0.03,
        },
        metric_variability={
            "forward_high_fraction": 0.05,
            "lane_fraction": 0.08,
            "river_fraction": 0.03,
            "enemy_territory_fraction": 0.06,
            "roam_rate_per_min": 0.08,
        },
        per_video_metrics={
            "a.mp4": {
                "region_Top Lane": 0.35, "region_Mid Lane": 0.40, "region_River": 0.15,
                "forward_high_fraction": 0.45, "lane_fraction": 0.60, "river_fraction": 0.15,
                "enemy_territory_fraction": 0.30, "roam_rate_per_min": 0.55,
            },
            "b.mp4": {
                "region_Top Lane": 0.40, "region_Mid Lane": 0.38, "region_River": 0.12,
                "forward_high_fraction": 0.42, "lane_fraction": 0.63, "river_fraction": 0.13,
                "enemy_territory_fraction": 0.27, "roam_rate_per_min": 0.50,
            },
            "c.mp4": {
                "region_Top Lane": 0.30, "region_Mid Lane": 0.44, "region_River": 0.16,
                "forward_high_fraction": 0.48, "lane_fraction": 0.58, "river_fraction": 0.17,
                "enemy_territory_fraction": 0.33, "roam_rate_per_min": 0.60,
            },
        },
    )


def test_fingerprint_ranges_and_confidence() -> None:
    fingerprint = FingerprintEngine().compute(_stub_profile())
    for metric in FINGERPRINT_METRICS:
        assert 0.0 <= fingerprint.scores[metric] <= 10.0, metric
        assert 0.0 < fingerprint.confidence[metric] <= 1.0, metric
        assert fingerprint.contributions[metric], metric
    assert fingerprint.scores["Aggression"] > 5.0
    assert fingerprint.scores["Consistency"] > 5.0


def test_settings_roundtrip(tmp_path, monkeypatch) -> None:
    settings = Settings(sample_fps=2.5, player_label="CoachMode")
    settings.validated()

    import core.config as config_module

    original = config_module.SETTINGS_PATH
    try:
        config_module.SETTINGS_PATH = str(tmp_path / "settings.json")
        config_module.save_settings(settings)
        loaded = config_module.load_settings()
        assert loaded.sample_fps == 2.5
        assert loaded.player_label == "CoachMode"
    finally:
        config_module.SETTINGS_PATH = original


def test_default_settings_are_sane() -> None:
    settings = load_settings()
    assert 0.1 <= settings.sample_fps <= 10.0
    assert settings.early_game_minutes >= 1.0
    assert settings.minimap_side in ("auto", "left", "right")


def test_profile_histogram_shape() -> None:
    profile = _stub_profile()
    assert profile.video_count == 3
    assert abs(sum(profile.region_distribution.values()) - 1.0) < 1e-6


def test_parse_team_kills_only_accepts_bare_digits() -> None:
    assert parse_team_kills("4") == 4
    assert parse_team_kills("10") == 10
    assert parse_team_kills(" 7 ") == 7
    assert parse_team_kills("0") == 0
    assert parse_team_kills("00:05") is None   # clock-shaped text never scores
    assert parse_team_kills("1/0/1") is None   # KDA fragment
    assert parse_team_kills("abc") is None
    assert parse_team_kills("012") is None     # three digits are not a score
    assert parse_team_kills("99") is None      # above the plausible range
    assert parse_team_kills("") is None


def test_parse_score_pair() -> None:
    assert parse_score_pair("4:6") == (4, 6)
    assert parse_score_pair("0:0") == (0, 0)
    assert parse_score_pair("4-6") is None
    assert parse_score_pair("4:90") is None
    assert parse_score_pair("4:6:7") is None
    assert parse_score_pair("") is None
    assert parse_score_pair("x:y") is None


def _score_readings() -> list:
    return [
        OcrReading("kill_score", "0:0", 0.90, 10.0),
        OcrReading("kill_score", "1:0", 0.90, 40.0),
        OcrReading("kill_score", "1:3", 0.90, 70.0),
        OcrReading("game_timer", "70", 0.90, 70.0),   # ignored (not a score)
        OcrReading("kill_score", "0:5", 0.90, 100.0),  # impossible drop -> rebase
        OcrReading("kill_score", "2:5", 0.90, 130.0),
    ]


def test_kill_event_extraction_deltas_and_rebase() -> None:
    events = extract_kill_events(
        _score_readings(), own_base="Blue", offset=5.0, window_end=900.0
    )
    assert [(e.team, e.count, e.time) for e in events] == [
        ("Blue", 1, 45.0),
        ("Red", 3, 75.0),
        ("Blue", 2, 135.0),
    ]
    assert all(e.kind == "kill" for e in events)
    assert own_team_kills(events, "Blue") == 3
    assert own_team_kills(events, "Red") == 3

    roams = [RoamEvent(40.0, 60.0, "Mid Lane", "River", ["Mid Lane", "River"])]
    assert kills_during_roams(events, roams, "Blue") == 1
    assert kills_during_roams(events, roams, "Red") == 0


def test_kill_events_respect_window_and_gap_limits() -> None:
    outside = [
        OcrReading("kill_score", "0:0", 0.9, 10.0),
        OcrReading("kill_score", "1:0", 0.9, 1000.0),
    ]
    assert extract_kill_events(outside, "Blue", offset=0.0, window_end=900.0) == []

    too_far = [
        OcrReading("kill_score", "0:0", 0.9, 10.0),
        OcrReading("kill_score", "1:0", 0.9, 300.0),
    ]
    assert extract_kill_events(too_far, "Blue", offset=0.0, window_end=900.0) == []

    assert extract_kill_events([], "Blue") == []
    assert (
        extract_kill_events(
            [OcrReading("kill_score", "junk", 0.9, 10.0)], "Blue"
        )
        == []
    )


def test_region_anchor_overrides_are_validated() -> None:
    settings = Settings()
    settings.region_anchors = {
        "lane_inset": 5.0,           # clamped to the upper bound
        "base_radius": 0.15,         # kept
        "mid_threshold": -1.0,       # clamped to the lower bound
        "unknown_anchor": 0.5,       # unknown key dropped
        "blue_base": [0.1, 0.9],     # non-scalar anchor dropped
        "river_start": 0.9,          # reversed river segment -> both dropped
        "river_end": 0.1,
        "bad_value": "x",            # unparsable value dropped
    }
    validated = settings.validated()
    assert validated.region_anchors["lane_inset"] == 0.25
    assert validated.region_anchors["base_radius"] == 0.15
    assert validated.region_anchors["mid_threshold"] == 0.01
    assert "unknown_anchor" not in validated.region_anchors
    assert "blue_base" not in validated.region_anchors
    assert "river_start" not in validated.region_anchors
    assert "river_end" not in validated.region_anchors


def test_engine_applies_region_anchor_overrides() -> None:
    default = RegionMapper()
    assert default.classify(0.15, 0.85) != "Base"

    settings = Settings()
    settings.region_anchors = {"base_radius": 0.30}
    engine = BehaviorEngine(settings.validated())
    assert engine.mapper.classify(0.15, 0.85) == "Base"
    # Untouched anchors keep their defaults.
    assert engine.mapper.a["lane_inset"] == 0.07


def _blank_minimap(size: int = 160):
    import numpy as np

    return np.full((size, size, 3), 40, dtype=np.uint8)


def _with_blob(frame, x: float, y: float, radius: int = 2):
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = int(x * w), int(y * h)
    out[cy - radius : cy + radius, cx - radius : cx + radius] = (60, 60, 230)
    return out


def test_ward_detector_confirms_stationary_bloom_once() -> None:
    from vision.ward_detector import WardDetector

    detector = WardDetector(sample_fps=1.0, warmup_sec=0.0)
    blank = _blank_minimap()
    assert detector.process(blank, None, 0.0) == []  # seeds the background

    confirmed = []
    for step in range(1, 15):
        frame = _with_blob(blank, 0.30, 0.60)
        confirmed.extend(detector.process(frame, None, float(step)))

    assert confirmed, "stationary bloom was never confirmed"
    first = confirmed[0]
    assert abs(first.x - 0.30) < 0.06
    assert abs(first.y - 0.60) < 0.06
    assert first.age >= 10
    assert len(confirmed) == 1, "one placement must emit exactly one event"


def test_ward_detector_ignores_moving_blobs_and_player_position() -> None:
    from vision.ward_detector import WardDetector

    detector = WardDetector(sample_fps=1.0, warmup_sec=0.0)
    blank = _blank_minimap()
    detector.process(blank, None, 0.0)

    # Travelling blob: stationarity counter must reset every sample.
    for step in range(1, 16):
        detector.process(_with_blob(blank, 0.2 + 0.05 * step, 0.5), None, float(step))
    # Blob sitting exactly on the tracked player is excluded.
    for step in range(16, 26):
        detector.process(_with_blob(blank, 0.30, 0.60), (0.30, 0.60), float(step))

    # A fresh detector on the same sequence must not report player blooms
    # either; here we only assert no confirmation happened in this instance.
    assert all(track.emitted is False for track in detector._tracks)


def test_fingerprint_vision_uses_optional_ward_indicator() -> None:
    without = FingerprintEngine().compute(_stub_profile())

    rich = _stub_profile()
    rich.average_metrics["ward_blooms"] = 9.0
    rich.average_metrics["ward_blooms_per_min"] = 1.5
    with_wards = FingerprintEngine().compute(rich)

    assert 0.0 <= without.scores["Vision"] <= 10.0
    assert 0.0 <= with_wards.scores["Vision"] <= 10.0
    assert with_wards.scores["Vision"] >= without.scores["Vision"]
    assert "vision blooms" in with_wards.contributions["Vision"].lower()
    assert "bloom" not in without.contributions["Vision"].lower()


def test_phash_stable_and_discriminating() -> None:
    import numpy as np

    from vision.champion_identifier import hamming, phash

    yy, xx = np.mgrid[0:64, 0:64]
    first = (128 + 90 * np.sin(xx / 4.0) * np.cos(yy / 6.0)).astype(np.uint8)
    second = (128 + 90 * np.sin(xx / 13.0 + 1.5) * np.cos(yy / 2.0)).astype(np.uint8)
    assert phash(first) == phash(first)
    assert hamming(phash(first), phash(first)) == 0
    assert hamming(phash(first), phash(second)) >= 8
    darker = (first.astype(np.float32) * 0.55 + 40.0).astype(np.uint8)
    assert hamming(phash(first), phash(darker)) <= 10



def test_champion_identifier_offline_without_cache(tmp_path, monkeypatch) -> None:
    import numpy as np

    from vision.champion_identifier import ChampionIdentifier

    monkeypatch.delenv("PRISM_ALLOW_DOWNLOAD", raising=False)
    ident = ChampionIdentifier(str(tmp_path / "champions"))
    assert ident.available is False
    assert ident.ensure_assets() is False  # never downloads without opt-in
    assert ident.identify(np.zeros((720, 1280, 3), np.uint8)) is None


def test_champion_identify_from_local_cache(tmp_path) -> None:
    import json

    import cv2
    import numpy as np

    from vision.champion_identifier import PORTRAIT_ROI, ChampionIdentifier, phash

    cache = tmp_path / "champions"
    cache.mkdir(parents=True)

    def _icon(seed: int):
        y, x = np.mgrid[0:120, 0:120]
        img = np.zeros((120, 120, 3), np.uint8)
        img[..., 0] = (seed * 40 + x * 2) % 256
        img[..., 1] = (seed * 60 + y * 3) % 256
        img[..., 2] = (seed * 90 + (x + y)) % 256
        cv2.circle(img, (60, 60), 15 + 7 * seed, (255, 255, 255), -1)
        return img

    icons = {}
    for name, seed in (("Alpha", 1), ("Beta", 2), ("Gamma", 3)):
        icon = _icon(seed)
        cv2.imwrite(str(cache / f"{name}.png"), icon)
        icons[name] = {"file": f"{name}.png", "phash": f"{phash(icon):016x}"}
    (cache / "manifest.json").write_text(
        json.dumps({"version": "test", "champions": icons}), encoding="utf-8"
    )
    ident = ChampionIdentifier(str(cache))
    assert ident.available

    # Paste an icon exactly where portrait_crop() will sample it.
    frame = np.full((1080, 1920, 3), 30, np.uint8)
    x, y, w, h = PORTRAIT_ROI
    x0, y0 = int(x * 1920), int(y * 1080)
    bw, bh = max(8, int(w * 1920)), max(8, int(h * 1080))
    mx, my = int(bw * 0.10), int(bh * 0.10)
    inner = frame[y0 + my : y0 + bh - my, x0 + mx : x0 + bw - mx]
    icon = _icon(1)
    frame[y0 + my : y0 + bh - my, x0 + mx : x0 + bw - mx] = cv2.resize(
        icon, (inner.shape[1], inner.shape[0])
    )


    match = ident.identify(frame)
    assert match is not None
    assert match.name == "Alpha"
    assert match.confidence >= 0.65


def test_champion_aggregate_and_claim_gate() -> None:
    from vision.champion_identifier import CLAIM_GATE, ChampionIdentifier, ChampionMatch

    agree = ChampionIdentifier.aggregate(
        [ChampionMatch("Zed", 0.90), ChampionMatch("Zed", 0.80)]
    )
    assert agree is not None and agree.name == "Zed"
    assert agree.confidence == 0.9  # median of the agreeing frames
    split = ChampionIdentifier.aggregate(
        [ChampionMatch("Zed", 0.90), ChampionMatch("Akali", 0.85)]
    )
    assert split is not None
    assert split.name == "Zed"  # plurality wins
    assert split.confidence < 0.90  # dissent tempers the claim
    assert ChampionIdentifier.aggregate([]) is None
    assert CLAIM_GATE == 0.65


def test_profile_picks_most_voted_champion() -> None:
    from core.models import VideoAnalysis, VideoInfo
    from core.profiler import build_profile
    from vision.champion_identifier import CLAIM_GATE

    def _analysis(name: str, champion: str, confidence: float) -> VideoAnalysis:
        info = VideoInfo(
            path=name, name=name, container="mp4", fps=20.0,
            frame_count=100, width=1920, height=1080, duration_sec=100.0,
        )
        return VideoAnalysis(info=info, champion=champion, champion_confidence=confidence)

    profile = build_profile(
        [
            _analysis("a.mp4", "Zed", 0.90),
            _analysis("b.mp4", "Zed", 0.80),
            _analysis("c.mp4", "Akali", 0.70),
            _analysis("d.mp4", "Ghost", CLAIM_GATE - 0.10),
        ],
        "Player",
    )
    assert profile.champion == "Zed"
    assert profile.champion_confidence == 0.9

    solo = build_profile([_analysis("x.mp4", "Ghost", CLAIM_GATE - 0.10)], "Player")
    assert solo.champion is None  # sub-gate evidence never becomes a claim

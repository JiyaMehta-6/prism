"""End-to-end pipeline test on a synthetic gameplay recording.

Covers: video validation -> minimap tracking -> region mapping -> profile ->
fingerprint -> analytics -> archetypes -> insights -> charts -> PDF.
OCR is disabled so the test stays hermetic (no model downloads).
"""

from __future__ import annotations

import os

import pytest

from analytics.aggression import analyze_aggression
from analytics.archetypes import classify_archetypes
from analytics.consistency import analyze_consistency
from analytics.insights import generate_insights
from analytics.objectives import analyze_objectives
from analytics.pressure import analyze_pressure
from analytics.roaming import analyze_roaming
from analytics.similarity import compare_profiles, save_profile_snapshot
from analytics.timeline import build_timeline
from core.behavior_engine import BehaviorEngine
from core.config import Settings
from core.fingerprint_engine import FingerprintEngine
from core.models import FINGERPRINT_METRICS, save_json
from core.profiler import build_profile
from core.video_loader import probe_video
from reports.pdf_export import export_pdf
from tests.video_synth import make_synthetic_video
from visualizations import charts


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("video") / "match.mp4")
    return make_synthetic_video(path, seconds=90, fps=20)


def _settings() -> Settings:
    settings = Settings()
    settings.sample_fps = 1.0
    settings.early_game_minutes = 15.0
    settings.ocr_enabled = False
    settings.use_game_clock = False
    settings.player_label = "SyntheticTester"
    return settings.validated()


def test_full_pipeline(tmp_path, synthetic_video) -> None:
    settings = _settings()
    info = probe_video(synthetic_video)
    assert info.valid, info.error

    progress_log: list[tuple[int, str]] = []
    engine = BehaviorEngine(settings, on_progress=lambda p, s: progress_log.append((p, s)))
    analysis = engine.analyze(info)

    assert analysis.samples, "no positions were tracked"
    assert analysis.detection_rate > 0.35, f"detection too low: {analysis.detection_rate}"
    assert len(analysis.samples) >= 40
    assert progress_log, "no progress callbacks"
    assert progress_log[-1][0] == 100

    regions = {sample.region for sample in analysis.samples}
    assert len(regions) >= 2, f"only regions seen: {regions}"
    assert abs(sum(analysis.region_time.values()) - 90.0) < 30.0

    profile = build_profile([analysis], settings.player_label)
    assert profile.video_count == 1
    assert profile.total_samples == len(analysis.samples)
    assert abs(sum(profile.region_distribution.values()) - 1.0) < 1e-6

    fingerprint = FingerprintEngine().compute(profile)
    for metric in FINGERPRINT_METRICS:
        assert 0.0 <= fingerprint.scores[metric] <= 10.0

    sections = {
        "Aggression": analyze_aggression(profile, fingerprint, [analysis]),
        "Roaming": analyze_roaming(profile, fingerprint, [analysis]),
        "Objectives": analyze_objectives(profile, fingerprint, [analysis]),
        "Pressure": analyze_pressure(profile, fingerprint, [analysis]),
        "Consistency": analyze_consistency(profile, fingerprint, [analysis]),
    }
    for name, section in sections.items():
        assert section.summary, f"{name} produced no summary"
        assert 0.0 <= section.confidence <= 1.0

    archetypes = classify_archetypes(profile, fingerprint)
    assert archetypes, "no archetype produced"
    assert archetypes[0].reasons, "archetype without justification"

    timeline = build_timeline(analyses=[analysis], edges=settings.window_plan)
    assert len(timeline) == 4
    assert all(window.label for window in timeline)

    insights = generate_insights(profile, fingerprint)
    assert insights
    assert all(0.0 < insight.confidence <= 1.0 for insight in insights)
    assert all(insight.suggestion for insight in insights)

    charts_dir = str(tmp_path / "charts")
    os.environ["PRISM_CHART_DIR"] = charts_dir
    chart_paths = charts.generate_all(profile, fingerprint, timeline, [])
    assert "fingerprint" in chart_paths
    assert "regions" in chart_paths
    for path in chart_paths.values():
        assert os.path.dirname(path) == charts_dir, f"chart outside override: {path}"
        assert os.path.isfile(path) and os.path.getsize(path) > 1000

    snapshot = save_profile_snapshot(profile, fingerprint, str(tmp_path / "profiles"))
    assert os.path.isfile(snapshot)
    comparisons = compare_profiles(profile, fingerprint, str(tmp_path / "profiles"))
    assert comparisons and comparisons[0].similarity > 0.9

    from core.models import AnalysisReport

    report = AnalysisReport(
        profile=profile,
        fingerprint=fingerprint,
        archetypes=archetypes,
        insights=insights,
        timeline=timeline,
        similarity=comparisons,
        chart_paths=chart_paths,
        video_analyses=[analysis],
        sections=sections,
        generated_at="test",
        settings_snapshot={"sample_fps": settings.sample_fps},
    )
    pdf_path = export_pdf(report, str(tmp_path / "report.pdf"))
    assert os.path.isfile(pdf_path)
    assert os.path.getsize(pdf_path) > 4000

    save_json(str(tmp_path / "export.json"), {"fingerprint": fingerprint.scores})
    assert os.path.isfile(str(tmp_path / "export.json"))


def test_invalid_video_raises_value_error(tmp_path) -> None:
    from tests.video_synth import make_unreadable_video

    settings = _settings()
    info = probe_video(make_unreadable_video(str(tmp_path / "junk.mp4")))
    engine = BehaviorEngine(settings)
    with pytest.raises(ValueError):
        engine.analyze(info)


def test_progress_is_monotonic(synthetic_video) -> None:
    settings = _settings()
    info = probe_video(synthetic_video)
    seen: list[int] = []
    engine = BehaviorEngine(settings, on_progress=lambda p, s: seen.append(p))
    engine.analyze(info)
    assert seen == sorted(seen), "progress went backwards"
    assert seen[-1] == 100


class _StubReader:
    """Duck-typed EasyOCR reader: no models, no network, deterministic.

    Reads are told apart by ``allowlist`` - the clock pass allows ``:`` while
    the kill-scoreboard pass does not - mirroring how the real engine calls
    the backend, so score reads cannot desynchronise the clock sequence.
    """

    def __init__(
        self,
        conf: float = 0.92,
        start_seconds: int = 60,
        step_seconds: int = 10,
        broken: bool = False,
    ) -> None:
        self.conf = conf
        self.start = start_seconds
        self.step = step_seconds
        self.broken = broken
        self.calls = 0
        self.score_calls = 0

    def readtext(self, image, detail=0, allowlist=None):  # noqa: ANN001
        if self.broken:
            raise RuntimeError("stub OCR backend unavailable")
        if allowlist and ":" in str(allowlist):
            seconds = self.start + self.step * self.calls
            self.calls += 1
            return [(None, f"{seconds // 60:02d}:{seconds % 60:02d}", self.conf)]
        # Scoreboard side: a plain, monotonically growing digit.
        self.score_calls += 1
        return [(None, str(self.score_calls % 12), self.conf)]


def _ocr_settings() -> Settings:
    settings = Settings()
    settings.sample_fps = 1.0
    settings.early_game_minutes = 15.0
    settings.ocr_enabled = True
    settings.use_game_clock = True
    settings.ocr_interval_sec = 10.0
    settings.ocr_max_calls_per_video = 12
    settings.player_label = "SyntheticTester"
    return settings.validated()


def _install_stub_reader(monkeypatch, stub: _StubReader) -> None:
    from vision.ocr_engine import OCREngine

    def _fake_init(self):  # noqa: ANN001
        if self._reader_failed or not self.enabled:
            return False
        self._reader = stub
        return True

    monkeypatch.setattr(OCREngine, "_init_reader", _fake_init)


def test_ocr_clock_path_aligns_game_time(monkeypatch, synthetic_video) -> None:
    """OCR runs hermetically (stub backend) and shifts game time to the clock."""
    stub = _StubReader(conf=0.92, start_seconds=60, step_seconds=10)
    _install_stub_reader(monkeypatch, stub)

    settings = _ocr_settings()
    info = probe_video(synthetic_video)
    seen: list[int] = []
    analysis = BehaviorEngine(settings, on_progress=lambda p, s: seen.append(p)).analyze(info)

    assert analysis.samples, "tracking failed while OCR was enabled"
    assert analysis.detection_rate > 0.35
    assert analysis.ocr_readings, "no clock readings were captured"
    clock_readings = [r for r in analysis.ocr_readings if r.key == "game_timer"]
    score_readings = [r for r in analysis.ocr_readings if r.key == "kill_score"]
    assert len(clock_readings) >= 3
    assert all(r.confidence >= settings.ocr_min_confidence for r in clock_readings)

    # Kill-scoreboard deltas must surface as team-kill events.
    assert score_readings, "kill scoreboard readings were not captured"
    assert analysis.events, "scoreboard deltas produced no events"
    assert all(e.kind == "kill" and e.count >= 1 for e in analysis.events)

    assert analysis.game_time_offset is not None
    assert abs(analysis.game_time_offset - 60.0) < 1e-6
    for sample in analysis.samples:
        assert sample.game_time is not None
        assert abs(sample.game_time - (sample.video_time + 60.0)) < 1e-6

    assert seen == sorted(seen)
    assert seen[-1] == 100


def test_ocr_low_confidence_is_rejected(monkeypatch, synthetic_video) -> None:
    """Sub-threshold clock reads never become readings or offsets."""
    stub = _StubReader(conf=0.40)
    _install_stub_reader(monkeypatch, stub)

    settings = _ocr_settings()
    analysis = BehaviorEngine(settings).analyze(probe_video(synthetic_video))

    assert analysis.ocr_readings == []
    assert analysis.game_time_offset is None
    assert analysis.samples, "analysis must survive a failing OCR pass"
    assert all(sample.game_time is None for sample in analysis.samples)


def test_ocr_backend_failure_never_aborts(monkeypatch, synthetic_video) -> None:
    """A crashing OCR backend degrades to video time, never raises."""
    stub = _StubReader(broken=True)
    _install_stub_reader(monkeypatch, stub)

    settings = _ocr_settings()
    analysis = BehaviorEngine(settings).analyze(probe_video(synthetic_video))

    assert analysis.ocr_readings == []
    assert analysis.game_time_offset is None
    assert analysis.detection_rate > 0.35
    regions = {sample.region for sample in analysis.samples}
    assert len(regions) >= 2

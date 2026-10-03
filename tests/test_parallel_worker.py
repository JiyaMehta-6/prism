from __future__ import annotations

import os


def test_parallel_worker_analyses_two_videos(tmp_path, monkeypatch) -> None:
    """The AnalysisWorker parallel path must preserve order and results."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("PRISM_CHART_DIR", str(tmp_path / "charts"))

    from core import config

    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(config, "PROFILE_DIR", str(tmp_path / "profiles"))

    from core.config import Settings
    from gui.worker import AnalysisWorker
    from tests.video_synth import make_synthetic_video

    paths = [
        make_synthetic_video(str(tmp_path / f"video{i}.mp4"), seconds=40, fps=20)
        for i in range(2)
    ]
    settings = Settings()
    settings.ocr_enabled = False
    settings.player_label = "ParallelTester"

    worker = AnalysisWorker(paths, settings)
    report = worker._execute()

    assert report.profile.video_count == 2
    assert report.profile.total_samples > 0
    names = [analysis.info.name for analysis in report.video_analyses]
    # Result order must match input order even when tasks finish out of order.
    assert names == [os.path.basename(path) for path in paths]
    assert len(report.chart_paths) >= 6
    assert "Benchmark" not in report.sections  # fresh PROFILE_DIR: n < 3

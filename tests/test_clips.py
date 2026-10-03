from __future__ import annotations

import os


def test_export_clips_writes_roam_clip(tmp_path) -> None:
    from core.models import RoamEvent, VideoAnalysis
    from core.video_loader import probe_video
    from reports.clips import export_clips
    from tests.video_synth import make_synthetic_video

    source = make_synthetic_video(str(tmp_path / "src.mp4"), seconds=100, fps=20)
    info = probe_video(source)
    assert info.valid

    analysis = VideoAnalysis(info=info)
    analysis.roam_events = [
        RoamEvent(40.0, 60.0, "Mid Lane", "River", ["Mid Lane", "River"]),
        RoamEvent(80.0, 95.0, "River", "Dragon Area", ["River", "Dragon Area"]),
    ]

    out_dir = str(tmp_path / "clips")
    written = export_clips([analysis], output_dir=out_dir)

    assert len(written) == 2
    for path in written:
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
        assert path.endswith(".mp4")
    # First clip: roam 40-60s plus 5s padding on each side => 30s window.
    first = probe_video(written[0])
    assert first.valid
    assert 24.0 <= first.duration_sec <= 36.0


def test_export_clips_respects_game_clock_offset(tmp_path) -> None:
    from core.models import RoamEvent, VideoAnalysis
    from core.video_loader import probe_video
    from reports.clips import export_clips
    from tests.video_synth import make_synthetic_video

    source = make_synthetic_video(str(tmp_path / "src.mp4"), seconds=100, fps=20)
    info = probe_video(source)

    analysis = VideoAnalysis(info=info)
    # Game time 65-70 with a +30s offset => video seconds 35-40.
    analysis.game_time_offset = 30.0
    analysis.roam_events = [RoamEvent(65.0, 70.0, "Base", "Mid Lane", ["Base", "Mid Lane"])]

    written = export_clips([analysis], output_dir=str(tmp_path / "clips"))
    assert len(written) == 1
    clip = probe_video(written[0])
    assert clip.valid
    # Video window 30s..45s => 15s of footage.
    assert 11.0 <= clip.duration_sec <= 21.0


def test_export_clips_skips_video_without_events(tmp_path) -> None:
    from core.models import VideoAnalysis
    from core.video_loader import probe_video
    from reports.clips import export_clips
    from tests.video_synth import make_synthetic_video

    source = make_synthetic_video(str(tmp_path / "src.mp4"), seconds=20, fps=20)
    info = probe_video(source)
    written = export_clips([VideoAnalysis(info=info)], output_dir=str(tmp_path / "clips"))
    assert written == []

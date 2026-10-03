"""Offline / zero-cost guarantees.

PRISM must complete its whole pipeline without ever opening a network socket,
must not download OCR models implicitly, and must not depend on paid services.
"""

from __future__ import annotations

import os
import socket

import pytest

from analytics.similarity import compare_profiles, save_profile_snapshot
from core.behavior_engine import BehaviorEngine
from core.config import Settings
from core.fingerprint_engine import FingerprintEngine
from core.models import FINGERPRINT_METRICS, AnalysisReport
from core.profiler import build_profile
from core.video_loader import probe_video
from tests.video_synth import make_synthetic_video
from visualizations import charts


class NetworkBlocked(RuntimeError):
    """Raised whenever the code under test tries to reach the network."""


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("video") / "match.mp4")
    return make_synthetic_video(path, seconds=60, fps=20)


@pytest.fixture()
def network_guard(monkeypatch) -> None:
    """Fail loudly on any outbound connection attempt."""

    def _blocked(*args, **kwargs):  # noqa: ANN002, ANN003
        raise NetworkBlocked("network access attempted by the code under test")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.delenv("PRISM_ALLOW_DOWNLOAD", raising=False)


def test_pipeline_runs_with_network_blocked(tmp_path, synthetic_video, network_guard) -> None:
    """Video -> tracking -> fingerprint -> charts -> snapshot, all offline."""
    settings = Settings()
    settings.sample_fps = 1.0
    settings.ocr_enabled = True
    settings.use_game_clock = True
    settings.player_label = "OfflineTester"
    settings = settings.validated()

    analysis = BehaviorEngine(settings).analyze(probe_video(synthetic_video))
    profile = build_profile([analysis], settings.player_label)
    fingerprint = FingerprintEngine().compute(profile)
    for metric in FINGERPRINT_METRICS:
        assert 0.0 <= fingerprint.scores[metric] <= 10.0

    charts_dir = str(tmp_path / "charts")
    os.environ["PRISM_CHART_DIR"] = charts_dir
    try:
        chart_paths = charts.generate_all(profile, fingerprint, [], [])
    finally:
        os.environ.pop("PRISM_CHART_DIR", None)
    assert chart_paths

    profiles_dir = str(tmp_path / "profiles")
    snapshot = save_profile_snapshot(profile, fingerprint, profiles_dir)
    assert os.path.isfile(snapshot)
    assert compare_profiles(profile, fingerprint, profiles_dir)

    report = AnalysisReport(
        profile=profile,
        fingerprint=fingerprint,
        archetypes=[],
        insights=[],
        timeline=[],
        similarity=[],
        chart_paths=chart_paths,
        video_analyses=[analysis],
        sections={},
        generated_at="offline-test",
        settings_snapshot={},
    )
    from reports.pdf_export import export_pdf

    pdf_path = export_pdf(report, str(tmp_path / "offline_report"))
    assert pdf_path.endswith(".pdf")
    assert os.path.getsize(pdf_path) > 1000


def test_ocr_model_download_is_disabled_by_default(network_guard, monkeypatch) -> None:
    """EasyOCR is never allowed to fetch models unless explicitly opted in."""
    from vision import ocr_engine

    assert ocr_engine._allow_download() is False

    monkeypatch.setenv("PRISM_ALLOW_DOWNLOAD", "1")
    assert ocr_engine._allow_download() is True


def test_no_paid_or_cloud_dependencies() -> None:
    """requirements.txt must not reference paid APIs or cloud SDKs."""
    forbidden = (
        "stripe", "paypal", "braintree", "openai", "anthropic", "azure-",
        "boto3", "google-cloud", "aws-sdk", "replicate", "huggingface_hub[hub]",
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements = os.path.join(root, "requirements.txt")
    with open(requirements, encoding="utf-8") as handle:
        content = handle.read().lower()
    for name in forbidden:
        assert name not in content, f"paid/cloud dependency found: {name}"

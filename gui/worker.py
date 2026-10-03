"""Background workers for the GUI.

All heavy work (video decoding, minimap tracking, OCR, chart rendering, PDF
writing) runs on :class:`QThread` subclasses so the Qt main loop never blocks.

Signals
-------
``progress(int)``   overall percentage 0-100
``stage(str)``      human readable phase description
``log(str)``        line appended to the GUI log pane
``finished(object)`` analysis report payload
``failed(str)``     fatal error message
"""

from __future__ import annotations

import copy
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence

from PySide6.QtCore import QThread, Signal

from analytics.aggression import analyze_aggression
from analytics.archetypes import classify_archetypes
from analytics.benchmark import compute_benchmark
from analytics.consistency import analyze_consistency
from analytics.insights import generate_insights
from analytics.objectives import analyze_objectives
from analytics.pressure import analyze_pressure
from analytics.roaming import analyze_roaming
from analytics.similarity import compare_profiles, save_profile_snapshot
from analytics.timeline import build_timeline
from core import config
from core.behavior_engine import AnalysisCancelled, BehaviorEngine
from core.config import Settings
from core.fingerprint_engine import FingerprintEngine
from core.logger import get_logger
from core.models import AnalysisReport, VideoAnalysis, VideoInfo, save_json
from core.profiler import build_profile
from core.video_loader import probe_video
from reports.clips import ClipExportCancelled, export_clips
from reports.pdf_export import PdfCancelled, export_pdf
from visualizations import charts

logger = get_logger("worker")


class AnalysisWorker(QThread):
    """Runs the full PRISM pipeline for a batch of videos."""

    progress = Signal(int)
    stage = Signal(str)
    log = Signal(str)
    finished_report = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        video_paths: Sequence[str],
        settings: Settings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_paths = list(video_paths)
        self.settings = settings
        self._cancelled = False
        self._last_progress = 0

    def cancel(self) -> None:
        """Request cooperative cancellation between stages."""
        self._cancelled = True

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            report = self._execute()
            if report is not None:
                self.finished_report.emit(report)
        except AnalysisCancelled:
            self._emit(self._last_progress, "Cancelled")
            self.log.emit("Analysis cancelled by user")
        except Exception as exc:  # pragma: no cover - surfaced in GUI
            logger.error("Analysis failed: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    def _emit(self, percent: int, message: str) -> None:
        # Progress must never move backwards in the UI.
        self._last_progress = max(self._last_progress, max(0, min(100, int(percent))))
        self.progress.emit(self._last_progress)
        self.stage.emit(message)
        if message:
            logger.info("[%3d%%] %s", self._last_progress, message)

    def _execute(self) -> Optional[AnalysisReport]:
        started = time.perf_counter()
        if not self.video_paths:
            raise ValueError("No videos selected. Add at least one gameplay recording.")

        self._emit(1, "Validating video files")
        total = len(self.video_paths)

        # Probe first (cheap, sequential) so invalid files are reported
        # before any decoding starts, then analyse in parallel when there is
        # more than one video.
        infos: List[VideoInfo] = []
        for path in self.video_paths:
            info: VideoInfo = probe_video(path)
            if not info.valid:
                self.log.emit(f"Skipped {info.name}: {info.error}")
                continue
            self.log.emit(
                f"Analysing {info.name} ({info.width}x{info.height}, "
                f"{info.duration_label}, {info.fps:.1f} fps)"
            )
            infos.append(info)

        # Cap the analysed span to the longest recording (rounded up to the
        # next 5 minutes), so timeline windows always land inside observed
        # data even when the configured cap exceeds a shorter clip.
        self.settings = self._effective_settings(infos)

        results: dict = {}
        if len(infos) > 1:
            workers = max(1, min(len(infos), os.cpu_count() or 2, 4))
            self.log.emit(f"Analysing {len(infos)} videos on {workers} parallel worker(s)")

            def _analyse_one(index: int, video_info: VideoInfo) -> VideoAnalysis:
                # One engine per task: the engine carries per-run mutable
                # state (cancel flag, last stride) that must not be shared
                # by concurrent analyse() calls.
                engine = BehaviorEngine(
                    self.settings, on_progress=self._on_video_progress
                )
                if self._cancelled:
                    raise AnalysisCancelled()
                return engine.analyze(
                    video_info,
                    base_percent=int(68.0 * index / total),
                    span=int(68.0 / total),
                    should_cancel=lambda: self._cancelled,
                )

            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prism-video")
            try:
                futures = {
                    pool.submit(_analyse_one, index, video_info): index
                    for index, video_info in enumerate(infos)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        results[index] = future.result()
                    except AnalysisCancelled:
                        raise
                    except Exception as exc:  # noqa: BLE001 - one video must not kill the batch
                        logger.error("Video analysis failed for %s: %s", infos[index].name, exc)
                        self.log.emit(f"Failed {infos[index].name}: {exc}")
            except AnalysisCancelled:
                # Stop queued videos immediately; running ones abort at their
                # next cancellation check.
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
        else:
            engine = BehaviorEngine(self.settings, on_progress=self._on_video_progress)
            for index, video_info in enumerate(infos):
                if self._cancelled:
                    raise AnalysisCancelled()
                try:
                    results[index] = engine.analyze(
                        video_info,
                        base_percent=int(68.0 * index / total),
                        span=int(68.0 / total),
                        should_cancel=lambda: self._cancelled,
                    )
                except AnalysisCancelled:
                    raise
                except Exception as exc:
                    logger.error("Video analysis failed for %s: %s", video_info.name, exc)
                    self.log.emit(f"Failed {video_info.name}: {exc}")

        analyses: List[VideoAnalysis] = [results[i] for i in sorted(results)]

        if self._cancelled:
            raise AnalysisCancelled()

        if not analyses:
            raise ValueError(
                "No video could be analysed. Check that the files are valid, "
                "supported formats (MP4/MKV/AVI/MOV) and contain a visible minimap."
            )

        self._emit(70, "Aggregating behavioural profile")
        self._check_cancel()
        profile = build_profile(analyses, self.settings.player_label or "Player")

        self._emit(74, "Computing behavioural fingerprint")
        fingerprint = FingerprintEngine().compute(profile)

        self._emit(78, "Running behavioural analytics")
        self._check_cancel()
        aggression = analyze_aggression(profile, fingerprint, analyses)
        roaming = analyze_roaming(profile, fingerprint, analyses, edges=self.settings.window_plan)
        objectives = analyze_objectives(profile, fingerprint, analyses)
        pressure = analyze_pressure(
            profile, fingerprint, analyses, anchors=self.settings.region_anchors or None
        )
        consistency = analyze_consistency(profile, fingerprint, analyses)
        self._log_analytics(aggression, roaming, objectives, pressure, consistency)

        self._emit(82, "Classifying player archetypes")
        self._check_cancel()
        archetypes = classify_archetypes(profile, fingerprint)
        for archetype in archetypes:
            self.log.emit(
                f"Archetype: {archetype.label} ({archetype.confidence * 100:.0f}% confidence)"
            )

        self._emit(85, "Building tactical timeline")
        self._check_cancel()
        timeline = build_timeline(
            analyses,
            edges=self.settings.window_plan,
            own_base=profile.own_base,
            anchors=self.settings.region_anchors or None,
            phases=self.settings.phase_boundaries,
        )

        self._emit(87, "Generating improvement insights")
        self._check_cancel()
        insights = generate_insights(profile, fingerprint, archetypes=archetypes)
        self.log.emit(f"{len(insights)} insight(s) generated")

        self._emit(89, "Comparing against stored profiles")
        self._check_cancel()
        comparison = compare_profiles(
            profile, fingerprint, config.PROFILE_DIR, self.settings.similarity_top_n
        )
        # Percentile standings need >=3 snapshots of *other* players; below
        # that compute_benchmark returns None and the section is omitted.
        benchmark = compute_benchmark(profile, fingerprint, config.PROFILE_DIR)
        if benchmark is not None:
            self.log.emit(benchmark.summary)

        self._emit(92, "Rendering charts")
        self._check_cancel()
        chart_paths = charts.generate_all(profile, fingerprint, timeline, comparison)
        self.log.emit(f"{len(chart_paths)} charts rendered")

        self._emit(96, "Saving profile snapshot")
        self._check_cancel()
        # Persistence is best-effort: a read-only disk or full volume must
        # never discard an analysis that already ran to completion.
        try:
            save_profile_snapshot(profile, fingerprint, config.PROFILE_DIR)
        except Exception as exc:  # noqa: BLE001 - report must survive
            logger.warning("Profile snapshot not saved: %s", exc)
            self.log.emit(f"Warning: profile snapshot not saved ({exc})")
        self._check_cancel()

        report = AnalysisReport(
            profile=profile,
            fingerprint=fingerprint,
            archetypes=archetypes,
            insights=insights,
            timeline=timeline,
            similarity=comparison,
            chart_paths=chart_paths,
            video_analyses=analyses,
            sections={
                "Aggression": aggression,
                "Roaming": roaming,
                "Objectives": objectives,
                "Pressure": pressure,
                "Consistency": consistency,
                **({"Benchmark": benchmark} if benchmark is not None else {}),
            },
            generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            settings_snapshot=self._settings_snapshot(),
        )

        self._emit(99, "Writing JSON export")
        try:
            save_json(
                os.path.join(config.OUTPUT_DIR, "last_analysis.json"),
                {
                    "profile": profile.to_dict(),
                    "fingerprint": fingerprint.scores,
                    "confidence": fingerprint.confidence,
                    "archetypes": [a.__dict__ for a in archetypes],
                    "insights": [i.__dict__ for i in insights],
                    "timeline": [t.__dict__ for t in timeline],
                    "generated_at": report.generated_at,
                },
            )
        except Exception as exc:  # noqa: BLE001 - report must survive
            logger.warning("JSON export not written: %s", exc)
            self.log.emit(f"Warning: JSON export not written ({exc})")

        elapsed = time.perf_counter() - started
        self._emit(100, f"Analysis complete in {elapsed:.1f}s")
        return report

    def _check_cancel(self) -> None:
        if self._cancelled:
            raise AnalysisCancelled()

    def _on_video_progress(self, percent: int, message: str) -> None:
        self._last_progress = max(self._last_progress, max(0, min(100, int(percent))))
        self.progress.emit(self._last_progress)
        if message:
            self.stage.emit(message)
            logger.debug(message)

    def _log_analytics(self, *sections) -> None:
        for section in sections:
            summary = getattr(section, "summary", "")
            if summary:
                self.log.emit(summary)

    def _effective_settings(self, infos: Sequence[VideoInfo]) -> Settings:
        """Copy of the settings with the analysis cap anchored to real data."""
        settings = copy.copy(self.settings)
        longest = max((info.duration_sec for info in infos), default=0.0)
        if longest > 0:
            span_minutes = max(5.0, math.ceil(longest / 300.0) * 5.0)
            settings.max_analysis_minutes = min(
                settings.max_analysis_minutes, span_minutes
            )
        return settings.validated()

    def _settings_snapshot(self) -> dict:
        return {
            "sample_fps": self.settings.sample_fps,
            "max_analysis_minutes": self.settings.max_analysis_minutes,
            "phase_boundaries": list(self.settings.phase_boundaries),
            "minimap_side": self.settings.minimap_side,
            "ocr_enabled": self.settings.ocr_enabled,
            "use_game_clock": self.settings.use_game_clock,
            "marker_min_confidence": self.settings.marker_min_confidence,
            "max_interpolate_gap": self.settings.max_interpolate_gap,
            "region_anchors": dict(self.settings.region_anchors or {}),
        }


class PdfExportWorker(QThread):
    """Writes the PDF report in the background."""

    finished_path = Signal(str)
    failed = Signal(str)

    def __init__(self, report: AnalysisReport, output_path: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self.report = report
        self.output_path = output_path
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation before the file is written."""
        self._cancelled = True

    def run(self) -> None:  # noqa: D102
        try:
            path = export_pdf(
                self.report, self.output_path, should_cancel=lambda: self._cancelled
            )
            self.finished_path.emit(path)
        except PdfCancelled:
            logger.info("PDF export cancelled")
        except Exception as exc:  # pragma: no cover
            logger.error("PDF export failed: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ClipExportWorker(QThread):
    """Extracts roam clips from the source videos in the background."""

    finished_paths = Signal(object)
    failed = Signal(str)

    def __init__(self, report: AnalysisReport, parent=None) -> None:
        super().__init__(parent)
        self.report = report
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation between clips."""
        self._cancelled = True

    def run(self) -> None:  # noqa: D102
        try:
            paths = export_clips(
                self.report.video_analyses, should_cancel=lambda: self._cancelled
            )
            self.finished_paths.emit(paths)
        except ClipExportCancelled:
            logger.info("Clip export cancelled")
        except Exception as exc:  # pragma: no cover
            logger.error("Clip export failed: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")

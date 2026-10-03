"""Tabbed report presentation.

``ReportView`` renders an :class:`~core.models.AnalysisReport` into seven
tabs: Overview, Fingerprint, Positioning, Timeline, Insights, Similarity and
Videos.  Every fact has exactly one owner tab - Overview carries identity and
verdict, Fingerprint the scores and their interpretation, Positioning the
geography, Timeline the chronology, Insights the advice, Similarity the
standing against other profiles and Videos the provenance - so nothing is
printed twice while browsing.  Charts are displayed from the PNG files
produced by :mod:`visualizations.charts`, so the GUI and the PDF always show
identical figures.
"""

from __future__ import annotations

import os
from html import escape as html_escape
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core.models import FINGERPRINT_METRICS, AnalysisReport, fresh_evidence, strip_axis_scores

ACCENT = "#C8AA6E"  # hextech gold
PRIMARY = "#5B9CFF"  # electric blue
MUTED = "#8B98AD"


class ChartLabel(QLabel):
    """Chart image that re-fits to the available width on every resize.

    Charts used to be pre-scaled to a fixed height, so a 1025 px-wide figure
    shown in a 740 px report area silently lost its right edge. This label
    scales the source pixmap width-first (capped by ``max_height``), so a
    chart is never cut off - it just renders smaller in a narrow window.
    """

    def __init__(self, pixmap: QPixmap, max_height: int) -> None:
        super().__init__()
        self._source = pixmap
        self._max_height = max_height
        self._fit_width = -1
        self.setAlignment(Qt.AlignCenter)
        # Ignored (not Preferred): a QLabel's minimum width normally equals
        # its pixmap width, so the pixmap would keep the widget wide and the
        # label could never receive a narrower resize event to shrink into.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        # Sensible hint before the first layout pass (resizeEvent refits).
        self._fit_to(min(pixmap.width(), 640))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Hidden widgets do not receive resize events, so refit on first show.
        super().showEvent(event)
        if self.width() != self._fit_width:
            self._fit_to(self.width())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if event.size().width() != self._fit_width:
            self._fit_to(event.size().width())

    def _fit_to(self, width: int) -> None:
        if self._source.isNull() or width <= 0:
            return
        target_w = width
        target_h = round(self._source.height() * width / self._source.width())
        if target_h > self._max_height:
            target_h = self._max_height
            target_w = round(self._source.width() * target_h / self._source.height())
        self._fit_width = width
        scaled = self._source.scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)
        # Height must follow the scale or the layout keeps stale tall rows.
        self.setFixedHeight(scaled.height())


class ReportView(QTabWidget):
    """Container for every report tab."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.report: Optional[AnalysisReport] = None
        self.setTabPosition(QTabWidget.North)
        self.setMovable(False)

        self._pages: Dict[str, QWidget] = {}
        for name in (
            "Overview",
            "Fingerprint",
            "Positioning",
            "Timeline",
            "Insights",
            "Similarity",
            "Videos",
        ):
            page = self._empty_page("No analysis yet. Add videos and click Analyse Player.")
            self._pages[name] = page
            self.addTab(page, name)

    def clear_report(self) -> None:
        """Reset every tab to its placeholder state."""
        self.report = None
        for name, page in self._pages.items():
            self._replace(page, self._empty_page("No analysis yet."))

    def set_report(self, report: AnalysisReport) -> None:
        """Populate all tabs from a finished analysis."""
        self.report = report
        self._render_overview(report)
        self._render_fingerprint(report)
        self._render_positioning(report)
        self._render_timeline(report)
        self._render_insights(report)
        self._render_similarity(report)
        self._render_videos(report)
        self.setCurrentIndex(0)

    @staticmethod
    def _empty_page(message: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("color: #6B7A92; font-size: 14px; padding: 40px;")
        layout.addWidget(label)
        return widget

    @staticmethod
    def _replace(page: QWidget, content: QWidget) -> None:
        """Swap the contents of a tab page in place."""
        parent = page.parentWidget()
        layout = page.layout()
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
            layout.addWidget(content)
        content.setParent(page)
        if parent is not None:
            parent.update()

    @staticmethod
    def _scroll(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _render_overview(self, report: AnalysisReport) -> None:
        # Owner of: identity, aggregate KPIs, playstyle verdict, data quality.
        # Scores live in Fingerprint, advice in Insights - nothing here repeats.
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        profile = report.profile
        meta: List[str] = []
        if profile.champion:
            meta.append(
                f"champion: {html_escape(profile.champion)} "
                f"({profile.champion_confidence * 100:.0f}%)"
            )
        if report.generated_at:
            meta.append(f"analysed {html_escape(report.generated_at)}")
        meta_html = (
            f"<br><span style='font-size:12px; color:{MUTED}'>{' | '.join(meta)}</span>"
            if meta
            else ""
        )
        header = QLabel(
            f"<span style='font-size:20px; font-weight:bold; color:{PRIMARY}'>"
            f"{html_escape(profile.player_label)}</span>{meta_html}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        kpi_host = QWidget()
        grid = QGridLayout(kpi_host)
        grid.setSpacing(10)
        tiles = (
            (str(profile.video_count), "videos analysed"),
            (str(profile.total_samples), "position samples"),
            (f"{profile.detection_rate * 100:.0f}%", "marker detection"),
            (profile.own_base, "own base"),
        )
        for index, (value, caption) in enumerate(tiles):
            grid.addWidget(self._kpi_tile(value, caption), 0, index)
        layout.addWidget(kpi_host)

        if report.archetypes:
            layout.addWidget(self._caption("Playstyle verdict"))
            grid_host = QWidget()
            arch_grid = QGridLayout(grid_host)
            arch_grid.setSpacing(10)
            for index, archetype in enumerate(report.archetypes[:4]):
                arch_grid.addWidget(self._archetype_card(archetype), index // 2, index % 2)
            layout.addWidget(grid_host)

        if profile.quality_notes:
            quality = QLabel(
                "<b>Data quality</b><br>"
                + "<br>".join(f"• {html_escape(n)}" for n in profile.quality_notes)
            )
            quality.setWordWrap(True)
            quality.setStyleSheet("color:#F5C26B; background:#231B0E; padding:8px; border-radius:4px;")
            layout.addWidget(quality)

        layout.addStretch(1)
        self._replace(self._pages["Overview"], self._scroll(container))

    @staticmethod
    def _kpi_tile(value: str, caption: str) -> QFrame:
        tile = QFrame()
        tile.setFrameShape(QFrame.StyledPanel)
        tile.setStyleSheet(
            "QFrame { background: #121A2B; border: 1px solid #26324A; border-radius: 6px; }"
        )
        box = QVBoxLayout(tile)
        box.setSpacing(2)
        number = QLabel(value)
        number.setAlignment(Qt.AlignCenter)
        number.setStyleSheet(f"color:{ACCENT}; font-size:20px; font-weight:bold;")
        box.addWidget(number)
        label = QLabel(caption)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        box.addWidget(label)
        return tile

    @staticmethod
    def _section_card(name: str, section) -> QWidget:
        # Score badges and "(x.x/10)" citations are stripped: the score table
        # directly above owns every number; cards explain what it means.
        summary = strip_axis_scores(getattr(section, "summary", "") or "")
        notes = [strip_axis_scores(n) for n in (getattr(section, "notes", []) or [])]
        html = (
            f"<div style='font-size:14px; font-weight:bold; color:{PRIMARY}'>"
            f"{html_escape(name)}</div>"
        )
        if summary:
            html += f"<p style='margin-top:4px;'>{html_escape(summary)}</p>"
        if notes:
            html += "<ul>" + "".join(f"<li>{html_escape(n)}</li>" for n in notes) + "</ul>"
        label = QLabel(html)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        label.setStyleSheet(
            "background: #121A2B; border-left: 4px solid "
            f"{PRIMARY}; border-radius: 4px; padding: 10px;"
        )
        return label

    @staticmethod
    def _archetype_card(archetype) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame { background: #121A2B; border: 1px solid #26324A; border-radius: 6px; }"
        )
        layout = QVBoxLayout(card)
        title = QLabel(
            f"<b style='font-size:15px; color:{PRIMARY}'>{html_escape(archetype.label)}</b>"
            f"<br><span style='color:{ACCENT}'>confidence "
            f"{archetype.confidence * 100:.0f}%</span>"
        )
        title.setWordWrap(True)
        layout.addWidget(title)
        reasons_html = "<ul style='margin:0'>" + "".join(
            f"<li>{html_escape(r)}</li>" for r in archetype.reasons
        ) + "</ul>"
        reasons = QLabel(reasons_html)
        reasons.setWordWrap(True)
        reasons.setTextFormat(Qt.RichText)
        reasons.setStyleSheet("color:#B7C2D4; font-size:12px;")
        layout.addWidget(reasons)
        card.setMinimumWidth(260)  # two cards + gap still fit a 1000 px window
        return card

    def _render_fingerprint(self, report: AnalysisReport) -> None:
        # Owner of: the seven scores (chart shape + exact table) and the
        # prose interpretation of each axis. The bars chart is deliberately
        # not shown - it plots the same seven scores the table already lists.
        container = QWidget()
        layout = QVBoxLayout(container)

        chart = report.chart_paths.get("fingerprint")
        if chart and os.path.isfile(chart):
            layout.addWidget(self._image(chart, 460))

        table = QTableWidget(len(FINGERPRINT_METRICS), 4)
        table.setHorizontalHeaderLabels(["Metric", "Score", "Confidence", "Evidence"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        for row, metric in enumerate(FINGERPRINT_METRICS):
            score = report.fingerprint.scores.get(metric, 0.0)
            conf = report.fingerprint.confidence.get(metric, 0.0)
            evidence = report.fingerprint.contributions.get(metric, "")
            table.setItem(row, 0, QTableWidgetItem(metric))
            score_item = QTableWidgetItem(f"{score:.1f} / 10")
            score_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 1, score_item)
            conf_item = QTableWidgetItem(f"{conf * 100:.0f}%")
            conf_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 2, conf_item)
            evidence_item = QTableWidgetItem(evidence)
            evidence_item.setToolTip(evidence)  # cells elide when narrow
            table.setItem(row, 3, evidence_item)
        table.setMinimumHeight(280)
        layout.addWidget(table)

        named = [(name, section) for name, section in report.sections.items()]
        if named:
            layout.addWidget(self._caption("What each axis means"))
            for name, section in named:
                if name == "Benchmark":
                    continue  # percentile standing belongs to Similarity
                layout.addWidget(self._section_card(name, section))
        layout.addStretch(1)
        self._replace(self._pages["Fingerprint"], self._scroll(container))

    def _render_positioning(self, report: AnalysisReport) -> None:
        # Owner of: geography. Captions state the takeaway of each figure,
        # never a rephrase of its embedded title, so nothing reads twice.
        container = QWidget()
        layout = QVBoxLayout(container)
        profile = report.profile

        regions = report.chart_paths.get("regions")
        if regions and os.path.isfile(regions):
            takeaway = self._regions_takeaway(profile)
            if takeaway:
                layout.addWidget(self._caption(takeaway))
            layout.addWidget(self._image(regions, 560))

        transitions = report.chart_paths.get("transitions")
        if transitions and os.path.isfile(transitions):
            takeaway = self._transition_takeaway(profile)
            if takeaway:
                layout.addWidget(self._caption(takeaway))
            layout.addWidget(self._image(transitions, 560))

        consistency = report.chart_paths.get("consistency")
        if consistency and os.path.isfile(consistency):
            layout.addWidget(self._caption(
                "Each group of bars is one recording - flatter groups mean steadier play."
            ))
            layout.addWidget(self._image(consistency, 560))

        if not any(
            report.chart_paths.get(key)
            for key in ("regions", "transitions", "consistency")
        ):
            layout.addWidget(QLabel("No positioning charts were produced for this analysis."))
        layout.addStretch(1)
        self._replace(self._pages["Positioning"], self._scroll(container))

    @staticmethod
    def _regions_takeaway(profile) -> str:
        ranked = sorted(
            (r for r, v in profile.region_distribution.items() if v > 0),
            key=lambda r: profile.region_distribution.get(r, 0.0),
            reverse=True,
        )
        if not ranked:
            return ""
        if len(ranked) == 1:
            return f"All tracked time spent in {ranked[0]}."
        return f"Most-visited areas: {ranked[0]} first, {ranked[1]} second."

    @staticmethod
    def _transition_takeaway(profile) -> str:
        best = (0, "", "")
        for source, row in profile.transition_matrix.items():
            for destination, count in row.items():
                if count > best[0]:
                    best = (count, source, destination)
        if not best[1]:
            return ""
        return f"Most frequent rotation: {best[1]} \u2192 {best[2]} ({best[0]}\u00d7)."

    def _render_timeline(self, report: AnalysisReport) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        timeline = report.timeline or []
        chart = report.chart_paths.get("timeline")
        if chart and os.path.isfile(chart):
            layout.addWidget(self._image(chart, 340))

        if not timeline:
            layout.addWidget(
                QLabel("No timeline windows were produced for this analysis.")
            )
            layout.addStretch(1)
            self._replace(self._pages["Timeline"], self._scroll(container))
            return

        # Window labels, aggression and confidence are annotated directly on
        # the chart above; the table owns the game phase, kills and summary.
        table = QTableWidget(len(timeline), 5)
        table.setHorizontalHeaderLabels(
            ["Window", "Game phase", "Behaviour", "Kills", "Summary"]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

        for row, window in enumerate(timeline):
            values = [
                f"{window.start_label}-{window.end_label}",
                window.phase or "-",
                window.label,
                str(window.kills),
                window.summary,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if column in (0, 1, 3):
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, column, item)
        table.setMinimumHeight(240)
        layout.addWidget(table)
        layout.addStretch(1)
        self._replace(self._pages["Timeline"], self._scroll(container))

    def _render_insights(self, report: AnalysisReport) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        intro = QLabel(
            f"{len(report.insights)} findings, ordered by confidence. "
            "Confidence reflects how much of the tracked evidence supports the claim."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        layout.addWidget(intro)

        colors = {"risk": "#FF4655", "strength": "#37D399", "info": PRIMARY}
        for insight in report.insights or []:
            color = colors.get(insight.severity, PRIMARY)
            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card.setStyleSheet(
                f"QFrame {{ background: #121A2B; border-left: 5px solid {color}; "
                "border-radius: 4px; }"
            )
            box = QVBoxLayout(card)
            title = QLabel(
                f"<b style='font-size:14px'>{html_escape(insight.title)}</b>"
                f"<span style='color:{ACCENT}'>&nbsp;&nbsp;confidence "
                f"{insight.confidence * 100:.0f}%</span>"
            )
            title.setWordWrap(True)
            box.addWidget(title)

            body = QTextBrowser()
            body.setOpenExternalLinks(False)
            body.setMaximumHeight(150)
            body.setStyleSheet("border:none; background: transparent;")
            html = f"<p>{html_escape(insight.detail)}</p>"
            # Only bullets with numbers the narrative does not already state.
            evidence = fresh_evidence(insight.detail, insight.evidence)
            if evidence:
                html += (
                    "<p style='color:#9AA7BC; font-size:11px'><b>Evidence:</b> "
                    + " | ".join(html_escape(item) for item in evidence)
                    + "</p>"
                )
            html += (
                f"<p style='color:{color}'><b>Suggestion:</b> "
                f"{html_escape(insight.suggestion)}</p>"
            )
            body.setHtml(html)
            box.addWidget(body)
            layout.addWidget(card)

        layout.addStretch(1)
        self._replace(self._pages["Insights"], self._scroll(container))

    def _render_similarity(self, report: AnalysisReport) -> None:
        # Owner of: standing against other profiles - pairwise similarity and
        # benchmark percentiles. The similarity chart is not shown; the table
        # already carries every percentage with its interpretation.
        container = QWidget()
        layout = QVBoxLayout(container)
        benchmark = report.sections.get("Benchmark")
        if report.similarity:
            table = QTableWidget(len(report.similarity), 3)
            table.setHorizontalHeaderLabels(["Profile", "Similarity", "Interpretation"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            for row, result in enumerate(report.similarity):
                name_item = QTableWidgetItem(result.profile_name)
                name_item.setToolTip(result.profile_name)
                table.setItem(row, 0, name_item)
                item = QTableWidgetItem(f"{result.similarity * 100:.1f}%")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, 1, item)
                note_item = QTableWidgetItem(result.note)
                note_item.setToolTip(result.note)  # last column elides
                table.setItem(row, 2, note_item)
            table.setMinimumHeight(200)
            layout.addWidget(table)
        elif benchmark is None:
            label = QLabel(
                "No comparable profiles stored yet. Analyse other players to build a "
                "behavioural comparison library."
            )
            label.setWordWrap(True)
            layout.addWidget(label)

        if benchmark is not None:
            layout.addWidget(self._caption("Standing vs stored profiles"))
            layout.addWidget(self._section_card("Benchmark", benchmark))
        layout.addStretch(1)
        self._replace(self._pages["Similarity"], self._scroll(container))

    def _render_videos(self, report: AnalysisReport) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        analyses = report.video_analyses or []
        table = QTableWidget(len(analyses), 8)
        table.setHorizontalHeaderLabels(
            ["Video", "Champion", "Duration", "Samples", "Detection", "Roams", "Clock", "Notes"]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

        for row, analysis in enumerate(analyses):
            metrics = analysis.metrics
            clock = (
                f"{analysis.game_time_offset:+.0f}s"
                if analysis.game_time_offset is not None
                else "video time"
            )
            notes = "OK" if analysis.detection_rate >= 0.55 else "low detection"
            values = [
                analysis.info.name,
                analysis.champion or "-",
                analysis.info.duration_label,
                str(len(analysis.samples)),
                f"{analysis.detection_rate * 100:.0f}%",
                f"{metrics.get('roam_count', 0):.0f}",
                clock,
                notes,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)  # video names/notes elide when narrow
                if column in (2, 3, 4, 5, 6):
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, column, item)

        table.setMinimumHeight(220)
        layout.addWidget(table)

        if report.settings_snapshot:
            settings_text = "<b>Run configuration</b><ul>" + "".join(
                f"<li>{html_escape(k.replace('_', ' ').title())}: "
                f"{html_escape(str(v))}</li>"
                for k, v in report.settings_snapshot.items()
            ) + "</ul>"
            label = QLabel(settings_text)
            label.setTextFormat(Qt.RichText)
            label.setWordWrap(True)  # otherwise long values force a horizontal scrollbar
            label.setStyleSheet(f"color:{MUTED}; font-size:12px;")
            layout.addWidget(label)

        layout.addStretch(1)
        self._replace(self._pages["Videos"], self._scroll(container))

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color:{MUTED}; font-size:12px; font-weight:bold;")
        return label

    @staticmethod
    def _image(path: str, max_height: int) -> QLabel:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            label = QLabel(f"(chart unavailable: {os.path.basename(path)})")
            label.setStyleSheet("color:#FF6B7A;")
            return label
        return ChartLabel(pixmap, max_height)

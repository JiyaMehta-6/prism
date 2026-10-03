"""Tabbed report presentation.

``ReportView`` renders an :class:`~core.models.AnalysisReport` into seven
tabs: Overview, Fingerprint, Heatmaps, Timeline, Insights, Similarity and
Videos.  Charts are displayed from the PNG files produced by
:mod:`visualizations.charts`, so the GUI and the PDF always show identical
figures.
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

from core.models import FINGERPRINT_METRICS, AnalysisReport

ACCENT = "#E8833A"
PRIMARY = "#2F6FB2"
MUTED = "#5D6D7E"


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
            "Heatmaps",
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
        self._render_heatmaps(report)
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
        label.setStyleSheet("color: #7F8C8D; font-size: 14px; padding: 40px;")
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
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        profile = report.profile
        champion_part = ""
        if profile.champion:
            champion_part = (
                f" &nbsp;|&nbsp; champion: {html_escape(profile.champion)} "
                f"({profile.champion_confidence * 100:.0f}%)"
            )
        header = QLabel(
            f"<span style='font-size:20px; font-weight:bold; color:{PRIMARY}'>"
            f"{html_escape(profile.player_label)}</span><br>"
            f"<span style='font-size:13px; color:{MUTED}'>"
            f"{html_escape(report.headline)} &nbsp;|&nbsp; {profile.video_count} videos &nbsp;|&nbsp; "
            f"{profile.total_samples} samples &nbsp;|&nbsp; "
            f"{profile.detection_rate * 100:.0f}% detection &nbsp;|&nbsp; "
            f"own base: {html_escape(profile.own_base)}"
            f"{champion_part}</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setSpacing(10)
        for index, archetype in enumerate(report.archetypes[:4]):
            grid.addWidget(self._archetype_card(archetype), index // 2, index % 2)
        layout.addWidget(grid_host)

        for name, section in report.sections.items():
            summary = getattr(section, "summary", "")
            notes: List[str] = getattr(section, "notes", [])
            score = getattr(section, "score", None)
            conf = getattr(section, "confidence", None)
            badge = ""
            if score is not None:
                badge = (
                    f"<span style='color:{ACCENT}; font-weight:bold;'>{score:.1f}/10</span>"
                )
            if conf is not None:
                badge += f" &nbsp; <span style='color:{MUTED}'>confidence {conf * 100:.0f}%</span>"
            html = f"<div style='font-size:14px; font-weight:bold; color:{PRIMARY}'>{name} {badge}</div>"
            html += f"<p style='margin-top:4px;'>{html_escape(summary)}</p>"
            if notes:
                html += "<ul>" + "".join(
                    f"<li>{html_escape(n)}</li>" for n in notes
                ) + "</ul>"
            label = QLabel(html)
            label.setWordWrap(True)
            label.setTextFormat(Qt.RichText)
            label.setStyleSheet(
                "background: #F8F9F9; border-left: 4px solid "
                f"{PRIMARY}; border-radius: 4px; padding: 10px;"
            )
            layout.addWidget(label)

        if profile.quality_notes:
            quality = QLabel(
                "<b>Data quality</b><br>"
                + "<br>".join(f"• {html_escape(n)}" for n in profile.quality_notes)
            )
            quality.setWordWrap(True)
            quality.setStyleSheet("color:#B9770E; background:#FEF9E7; padding:8px; border-radius:4px;")
            layout.addWidget(quality)

        layout.addStretch(1)
        self._replace(self._pages["Overview"], self._scroll(container))

    @staticmethod
    def _archetype_card(archetype) -> QWidget:
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame { background: white; border: 1px solid #D5D8DC; border-radius: 6px; }"
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
        reasons.setStyleSheet("color:#34495E; font-size:12px;")
        layout.addWidget(reasons)
        card.setMinimumWidth(320)
        return card

    def _render_fingerprint(self, report: AnalysisReport) -> None:
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
            table.setItem(row, 3, QTableWidgetItem(evidence))
        table.setMinimumHeight(280)
        layout.addWidget(table)

        bars = report.chart_paths.get("insights")
        if bars and os.path.isfile(bars):
            layout.addWidget(self._image(bars, 360))
        layout.addStretch(1)
        self._replace(self._pages["Fingerprint"], self._scroll(container))

    def _render_heatmaps(self, report: AnalysisReport) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        for key, caption in (
            ("heatmap", "Position heatmap over the 0-15 minute window"),
            ("regions", "Share of time per region"),
            ("transitions", "Region transition matrix"),
            ("consistency", "Match-to-match behavioural variation"),
        ):
            path = report.chart_paths.get(key)
            if path and os.path.isfile(path):
                layout.addWidget(self._caption(caption))
                layout.addWidget(self._image(path, 560))
        layout.addStretch(1)
        self._replace(self._pages["Heatmaps"], self._scroll(container))

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

        table = QTableWidget(len(timeline), 5)
        table.setHorizontalHeaderLabels(
            ["Window", "Label", "Aggression", "Confidence", "Summary"]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

        for row, window in enumerate(timeline):
            values = [
                f"{window.start_label}-{window.end_label}",
                window.label,
                f"{window.aggression_index:.1f}",
                f"{window.confidence * 100:.0f}%",
                window.summary,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (2, 3):
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

        colors = {"risk": "#C4453C", "strength": "#3FA46A", "info": PRIMARY}
        for insight in report.insights or []:
            color = colors.get(insight.severity, PRIMARY)
            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            card.setStyleSheet(
                f"QFrame {{ background: white; border-left: 5px solid {color}; "
                "border-radius: 4px; }}"
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
            if insight.evidence:
                html += (
                    "<p style='color:#5D6D7E; font-size:11px'><b>Evidence:</b> "
                    + " | ".join(html_escape(item) for item in insight.evidence)
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
        container = QWidget()
        layout = QVBoxLayout(container)
        if report.similarity:
            table = QTableWidget(len(report.similarity), 3)
            table.setHorizontalHeaderLabels(["Profile", "Similarity", "Interpretation"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            for row, result in enumerate(report.similarity):
                table.setItem(row, 0, QTableWidgetItem(result.profile_name))
                item = QTableWidgetItem(f"{result.similarity * 100:.1f}%")
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, 1, item)
                table.setItem(row, 2, QTableWidgetItem(result.note))
            table.setMinimumHeight(200)
            layout.addWidget(table)
        else:
            layout.addWidget(
                QLabel(
                    "No comparable profiles stored yet. Analyse other players to build a "
                    "behavioural comparison library."
                )
            )
        chart = report.chart_paths.get("similarity")
        if chart and os.path.isfile(chart):
            layout.addWidget(self._image(chart, 320))
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
            label.setStyleSheet("color:#C0392B;")
            return label
        scaled = pixmap.scaledToHeight(max_height, Qt.SmoothTransformation)
        label = QLabel()
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignCenter)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return label

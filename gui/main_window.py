"""PRISM main window.

Layout
------
Left rail: video list + actions + progress + live log.
Right side: tabbed behavioural report (see :class:`gui.report_view.ReportView`).

All heavy work is delegated to :class:`gui.worker.AnalysisWorker`; the window
only reacts to signals, so the UI never freezes.
"""

from __future__ import annotations

import copy
import os
from typing import List

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core import config
from core.config import Settings, load_settings, save_settings
from core.logger import get_logger
from core.models import AnalysisReport
from core.video_loader import is_supported, probe_video
from gui.report_view import ReportView
from gui.settings_dialog import SettingsDialog
from gui.storage_dialog import StorageDialog
from gui.worker import AnalysisWorker, ClipExportWorker, PdfExportWorker

logger = get_logger("main_window")

VIDEO_FILTER = "Gameplay Videos (*.mp4 *.mkv *.avi *.mov);;All Files (*)"


def _safe_filename(label: str) -> str:
    """Make a player label safe for use inside a file path.

    Also capped: a very long label would produce a >260-char default PDF
    path that the save dialog / filesystem cannot handle.
    """
    cleaned = "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(label or ""))
    cleaned = cleaned.strip(" ._")[:60]
    return cleaned or "player"

STYLESHEET = """
QWidget {
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
    color: #E6ECF7;
}
QMainWindow, QWidget#root, QDialog { background: #0B0F1A; }
QToolTip {
    background: #182238;
    color: #E6ECF7;
    border: 1px solid #26324A;
    padding: 4px 6px;
}
QLabel#brand {
    font-size: 26px;
    font-weight: 800;
    color: #C8AA6E;
    letter-spacing: 3px;
}
QLabel#tagline { color: #8B98AD; font-size: 12px; }
QListWidget {
    background: #121A2B;
    border: 1px solid #26324A;
    border-radius: 8px;
    padding: 4px;
}
QListWidget::item { padding: 7px; border-radius: 5px; color: #E6ECF7; }
QListWidget::item:hover { background: #182238; }
QListWidget::item:selected { background: #1D3557; color: #EAF2FF; }
QLineEdit, QTextEdit {
    background: #0F1626;
    color: #E6ECF7;
    border: 1px solid #26324A;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: #2563EB;
    selection-color: #FFFFFF;
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #3B82F6;
}
QLineEdit:disabled, QTextEdit:disabled {
    background: #101828;
    color: #5A6779;
}
QComboBox QAbstractItemView {
    background: #121A2B;
    color: #E6ECF7;
    border: 1px solid #26324A;
    selection-background-color: #1D3557;
    selection-color: #EAF2FF;
}
QPushButton {
    background: #182238;
    color: #E6ECF7;
    border: 1px solid #26324A;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #1E2A42; border-color: #3B82F6; }
QPushButton:pressed { background: #22314F; }
QPushButton:disabled { color: #5A6779; background: #101828; border-color: #1C2740; }
QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #3B82F6, stop:1 #2563EB);
    color: white;
    border: none;
    border-radius: 7px;
    font-size: 15px;
    padding: 11px;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #5B9CFF, stop:1 #2E76E0);
}
QPushButton#primary:pressed { background: #1D4ED8; }
QPushButton#primary:disabled { background: #2B3B55; color: #8B98AD; }
QPushButton#danger { color: #FF6B7A; }
QProgressBar {
    background: #182238;
    border: 1px solid #26324A;
    border-radius: 6px;
    height: 18px;
    text-align: center;
    color: #E6ECF7;
    font-size: 11px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 #37D399, stop:1 #2BB985);
    border-radius: 5px;
}
QPlainTextEdit {
    background: #0D1420;
    color: #9EE7B8;
    border: 1px solid #1C2740;
    border-radius: 8px;
    font-family: Consolas, monospace;
    font-size: 11.5px;
    padding: 6px;
}
QTabWidget::pane { border: 1px solid #26324A; background: #121A2B; border-radius: 8px; }
QTabBar::tab {
    background: #101828;
    /* Compact: the old 9x18 padding made the 7 tabs need 1033 px, so on
       1280-1366 laptops most tabs hid behind scroll arrows. */
    padding: 7px 10px;
    margin-right: 2px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 600;
    font-size: 11px;
    color: #8B98AD;
}
QTabBar::tab:selected { background: #121A2B; color: #C8AA6E; border-top: 2px solid #C8AA6E; }
QTabBar::tab:hover:!selected { background: #182238; color: #C8AA6E; }
QGroupBox {
    background: #121A2B;
    color: #E6ECF7;
    border: 1px solid #26324A;
    border-radius: 8px;
    margin-top: 12px;
    padding: 16px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #C8AA6E;
}
QCheckBox { color: #E6ECF7; spacing: 6px; }
QCheckBox:disabled { color: #5A6779; }
QHeaderView::section {
    background: #1D4ED8;
    color: white;
    padding: 6px;
    border: none;
    font-weight: 600;
}
QTableWidget {
    background: #121A2B;
    color: #E6ECF7;
    gridline-color: #1E2A42;
    alternate-background-color: #16203A;
}
QStatusBar { background: #0F1626; color: #8B98AD; }
QScrollBar:vertical { background: #0B0F1A; width: 12px; }
QScrollBar::handle:vertical { background: #26324A; border-radius: 6px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #3B4A66; }
QScrollBar:horizontal { background: #0B0F1A; height: 12px; }
QScrollBar::handle:horizontal { background: #26324A; border-radius: 6px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #3B4A66; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; background: transparent; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QMenu {
    background: #121A2B;
    color: #E6ECF7;
    border: 1px solid #26324A;
}
QMenu::item { padding: 6px 24px 6px 14px; }
QMenu::item:selected { background: #1D3557; color: #EAF2FF; }
QMenu::separator { height: 1px; background: #26324A; margin: 4px 8px; }
"""


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self) -> None:
        super().__init__()
        self.settings: Settings = load_settings()
        self.report: AnalysisReport | None = None
        self.worker: AnalysisWorker | None = None
        self.pdf_worker: PdfExportWorker | None = None
        self.clip_worker: ClipExportWorker | None = None
        self._closing = False

        self.setWindowTitle("PRISM - Player Replay Intelligence and Strategic Modeling")
        self.setAcceptDrops(True)

        self._build_ui()
        self._fit_window_to_screen()
        self.setStatusMessage("Ready - add 5 gameplay videos of the same player, then analyse.")
        logger.info("PRISM window initialised")

    def _fit_window_to_screen(self) -> None:
        """Open fully on-screen at a size the current display can hold.

        The window used to hardcode 1480x900 with a 1120x700 floor: on a
        1366x768 laptop (about 728 px of usable height plus the title bar)
        the bottom of the side panel - progress, cancel, log - was cut off,
        and on 1280x720 the default size did not fit at all.
        """
        floor = self.minimumSizeHint()
        # Explicit floor >= every layout minimum (incl. report cards), so a
        # shrunken window can never clip content; height tracks the layout.
        self.setMinimumSize(max(floor.width(), 1000), floor.height())
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        if avail is None:
            self.resize(1480, 900)
            return
        width = max(min(1480, avail.width() - 24), self.minimumWidth())
        height = max(min(900, avail.height() - 48), self.minimumHeight())
        self.resize(width, height)
        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        self.move(frame.topLeft())

    # ----------------------------------------------------------------- UI ---
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        splitter.addWidget(self._build_side_panel())
        report_host = QWidget()
        report_layout = QVBoxLayout(report_host)
        report_layout.setContentsMargins(0, 0, 0, 0)
        self.report_view = ReportView()
        report_layout.addWidget(self.report_view)
        splitter.addWidget(report_host)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1120])

        self.setStatusBar(QStatusBar())

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        # Resizable in the splitter (it used to be fixed at 360, which made
        # the handle immovable) but bounded so the report area keeps room.
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(520)
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)

        brand = QLabel("PRISM")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        tagline = QLabel("Player Replay Intelligence<br>and Strategic Modeling")
        tagline.setObjectName("tagline")
        tagline.setWordWrap(True)
        layout.addWidget(tagline)

        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QListWidget.ExtendedSelection)
        # Minimums kept low so the whole panel fits a 720p laptop (the old
        # 170/150 floors pushed the window minimum past its usable height).
        self.video_list.setMinimumHeight(100)
        layout.addWidget(self.video_list, stretch=3)

        file_buttons = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        self.add_btn.setToolTip("Add gameplay videos (MP4, MKV, AVI, MOV)")
        self.add_btn.clicked.connect(self.add_videos)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setProperty("class", "danger")
        self.remove_btn.clicked.connect(self.remove_selected)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_videos)
        for button in (self.add_btn, self.remove_btn, self.clear_btn):
            file_buttons.addWidget(button)
        layout.addLayout(file_buttons)

        hint = QLabel("Supported: MP4, MKV, AVI, MOV - drag & drop works too.")
        hint.setWordWrap(True)  # unwrapped this needed 594 px in a 342 px panel
        hint.setStyleSheet("color:#8B98AD; font-size:11px;")
        layout.addWidget(hint)

        self.analyse_btn = QPushButton("Analyse Player")
        self.analyse_btn.setObjectName("primary")
        self.analyse_btn.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyse_btn)

        # Two per row: Settings+Export PDF need 431 px side by side, which
        # overflowed the 342 px panel and clipped the Outputs button.
        action_grid = QGridLayout()
        action_grid.setSpacing(8)
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        self.export_btn = QPushButton("Export PDF")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_pdf)
        open_btn = QPushButton("Outputs")
        open_btn.clicked.connect(self.open_outputs)
        self.data_btn = QPushButton("Manage Data")
        self.data_btn.setToolTip(
            "Inspect and delete stored profiles, reports, charts, clips and logs"
        )
        self.data_btn.clicked.connect(self.open_storage)
        action_grid.addWidget(self.settings_btn, 0, 0)
        action_grid.addWidget(self.export_btn, 0, 1)
        action_grid.addWidget(open_btn, 1, 0)
        action_grid.addWidget(self.data_btn, 1, 1)
        layout.addLayout(action_grid)

        self.clips_btn = QPushButton("Export Roam Clips")
        self.clips_btn.setEnabled(False)
        self.clips_btn.clicked.connect(self.export_clips)
        layout.addWidget(self.clips_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.stage_label = QLabel("Idle")
        self.stage_label.setWordWrap(True)
        self.stage_label.setStyleSheet("color:#8B98AD; font-size:12px;")
        layout.addWidget(self.stage_label)

        self.cancel_btn = QPushButton("Cancel analysis")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        layout.addWidget(self.cancel_btn)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        self.log.setMinimumHeight(90)
        layout.addWidget(self.log, stretch=2)

        return panel

    # ------------------------------------------------------------- videos ---
    def add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select gameplay videos", "", VIDEO_FILTER
        )
        if paths:
            self._append_paths(paths)

    def _append_paths(self, paths: List[str]) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.log_line("Videos cannot be changed while an analysis is running.")
            return
        added = 0
        # Probing opens the container (metadata read) - keep the cursor honest
        # and let the event loop breathe between files so a large batch of
        # big recordings cannot freeze the window.
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                path = os.path.abspath(path)
                if not is_supported(path):
                    self.log_line(f"Unsupported format skipped: {path}")
                    continue
                if self._is_listed(path):
                    continue
                info = probe_video(path)
                label = info.name if info.valid else f"{info.name}  [INVALID: {info.error}]"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, path)
                item.setToolTip(
                    path if info.valid else f"{path}\n{info.error}"
                )
                if not info.valid:
                    item.setForeground(Qt.red)
                self.video_list.addItem(item)
                added += 1
                QApplication.processEvents()
        finally:
            QGuiApplication.restoreOverrideCursor()
        if added:
            self.setStatusMessage(f"{self.video_list.count()} video(s) queued.")
            self.log_line(f"Added {added} video(s); total {self.video_list.count()}.")

    def _is_listed(self, path: str) -> bool:
        # Windows paths are case-insensitive (and separators may differ):
        # compare normalized forms so C:\Game.mp4 and c:\game.mp4 can never
        # queue the same recording twice.
        key = os.path.normcase(os.path.normpath(path))
        for row in range(self.video_list.count()):
            listed = self.video_list.item(row).data(Qt.UserRole)
            if not listed:
                continue
            if os.path.normcase(os.path.normpath(str(listed))) == key:
                return True
        return False

    def remove_selected(self) -> None:
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))

    def clear_videos(self) -> None:
        self.video_list.clear()

    def selected_paths(self) -> List[str]:
        """Paths to analyse: the current selection, or every listed video."""
        items = self.video_list.selectedItems()
        if not items:
            items = [
                self.video_list.item(row) for row in range(self.video_list.count())
            ]
        return [item.data(Qt.UserRole) for item in items if item is not None]

    # ------------------------------------------------------------ actions ---
    def start_analysis(self) -> None:
        paths = self.selected_paths()
        if not paths:
            QMessageBox.information(
                self, "No videos", "Add at least one gameplay video before analysing."
            )
            return
        if self.worker is not None and self.worker.isRunning():
            return
        if self.pdf_worker is not None and self.pdf_worker.isRunning():
            self.log_line("Wait for the current PDF export to finish first.")
            return
        if self.clip_worker is not None and self.clip_worker.isRunning():
            self.log_line("Wait for the current clip export to finish first.")
            return

        # A new run invalidates the previous report (charts on disk are
        # regenerated too) - never let a stale report be exported.
        self.report = None
        self.report_view.clear_report()
        self.export_btn.setEnabled(False)
        self.clips_btn.setEnabled(False)

        self.log_line(f"Starting analysis of {len(paths)} video(s)...")
        self.analyse_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.settings_btn.setEnabled(False)
        self.data_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.video_list.setEnabled(False)
        self.progress.setValue(0)

        # Deep copy so settings edited during the run cannot race the worker.
        self.worker = AnalysisWorker(paths, copy.deepcopy(self.settings), parent=self)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.stage.connect(self.stage_label.setText)
        self.worker.log.connect(self.log_line)
        self.worker.finished_report.connect(self.on_report_ready)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def cancel_analysis(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.log_line("Cancellation requested...")

    def on_report_ready(self, report) -> None:
        self.report = report
        self.report_view.set_report(report)
        # Export stays disabled until the worker thread actually finishes
        # (on_worker_finished) so a click can never race thread teardown.
        self.stage_label.setText(
            f"Complete - {report.headline} "
            f"({report.profile.video_count} videos, {report.profile.total_samples} samples)"
        )
        self.log_line(
            f"Report ready: {report.headline} | "
            f"{len(report.insights)} insights | {len(report.chart_paths)} charts"
        )
        self.setStatusMessage("Analysis complete. Review the tabs or export the PDF.")

    def on_analysis_failed(self, message: str) -> None:
        self.progress.setValue(0)
        self.stage_label.setText("Failed")
        self.log_line(f"ERROR: {message}")
        if self._closing:
            return  # never block application shutdown with a modal dialog
        self._show_box(QMessageBox.Critical, "Analysis failed", message)

    def on_worker_finished(self) -> None:
        self.analyse_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.settings_btn.setEnabled(True)
        self.data_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.video_list.setEnabled(True)
        # Keep exporting possible when an earlier report is still on screen,
        # but never during an active PDF render.
        pdf_busy = self.pdf_worker is not None and self.pdf_worker.isRunning()
        clip_busy = self.clip_worker is not None and self.clip_worker.isRunning()
        self.export_btn.setEnabled(self.report is not None and not pdf_busy)
        self.clips_btn.setEnabled(self.report is not None and not clip_busy)

    def open_settings(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.apply_to_settings()
            if save_settings(self.settings):
                self.log_line("Settings saved.")
                self.setStatusMessage("Settings updated.")
            else:
                self.log_line(
                    "Settings applied for this session, but the settings file "
                    "could not be written."
                )
                self.setStatusMessage("Settings applied (file not written).")

    def open_storage(self) -> None:
        """Manage the data PRISM stored on this machine (delete history)."""
        if self.worker is not None and self.worker.isRunning():
            return
        if self.pdf_worker is not None and self.pdf_worker.isRunning():
            self.log_line("Wait for the PDF export to finish before managing data.")
            return
        if self.clip_worker is not None and self.clip_worker.isRunning():
            self.log_line("Wait for the clip export to finish before managing data.")
            return
        dialog = StorageDialog(self)
        dialog.exec()
        cleared = set(dialog.cleared_keys)
        if not cleared:
            return
        if "settings" in cleared:
            self.settings = load_settings()
            self.log_line("Settings reset to defaults.")
        if cleared & {"charts", "last_report"} and self.report is not None:
            # The PDF embeds the chart PNGs that were just removed; the next
            # analysis regenerates them and re-enables the button.
            self.export_btn.setEnabled(False)
            self.log_line(
                "Charts removed - Export PDF is disabled until the next analysis."
            )
        self.log_line(f"Deleted stored data: {', '.join(sorted(cleared))}.")
        self.setStatusMessage("Stored data deleted.")

    def export_pdf(self) -> None:
        if self.report is None:
            QMessageBox.information(self, "No report", "Run an analysis first.")
            return
        if self.worker is not None and self.worker.isRunning():
            self.log_line("Wait for the analysis to finish before exporting.")
            return
        if self.pdf_worker is not None and self.pdf_worker.isRunning():
            return

        default_name = os.path.join(
            config.OUTPUT_DIR, f"{_safe_filename(self.report.profile.player_label)}_report.pdf"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF report", default_name, "PDF Documents (*.pdf)"
        )
        if not path:
            return

        self.export_btn.setEnabled(False)
        self.stage_label.setText("Rendering PDF...")
        self.pdf_worker = PdfExportWorker(self.report, path, parent=self)
        self.pdf_worker.finished_path.connect(self.on_pdf_ready)
        self.pdf_worker.failed.connect(self.on_pdf_failed)
        self.pdf_worker.finished.connect(self._on_pdf_worker_finished)
        self.pdf_worker.start()

    def _on_pdf_worker_finished(self) -> None:
        analysis_busy = self.worker is not None and self.worker.isRunning()
        self.export_btn.setEnabled(self.report is not None and not analysis_busy)

    def on_pdf_ready(self, path: str) -> None:
        self.stage_label.setText(f"PDF written: {path}")
        self.log_line(f"PDF exported -> {path}")
        if self._closing:
            return  # closing: never raise a modal dialog over shutdown
        answer = self._show_box(
            QMessageBox.Information,
            "PDF exported",
            f"Report written to:\n{path}\n\nOpen it now?",
            QMessageBox.Open | QMessageBox.Close,
        )
        if answer == QMessageBox.Open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def on_pdf_failed(self, message: str) -> None:
        self.stage_label.setText("PDF export failed")
        self.log_line(f"ERROR: {message}")
        if self._closing:
            return
        self._show_box(QMessageBox.Critical, "PDF export failed", message)

    def export_clips(self) -> None:
        if self.report is None:
            QMessageBox.information(self, "No report", "Run an analysis first.")
            return
        if self.worker is not None and self.worker.isRunning():
            self.log_line("Wait for the analysis to finish before exporting clips.")
            return
        if self.clip_worker is not None and self.clip_worker.isRunning():
            return

        self.clips_btn.setEnabled(False)
        self.stage_label.setText("Exporting roam clips...")
        self.log_line("Exporting roam clips to outputs/clips ...")
        self.clip_worker = ClipExportWorker(self.report, parent=self)
        self.clip_worker.finished_paths.connect(self.on_clips_ready)
        self.clip_worker.failed.connect(self.on_clips_failed)
        self.clip_worker.finished.connect(self._on_clip_worker_finished)
        self.clip_worker.start()

    def _on_clip_worker_finished(self) -> None:
        analysis_busy = self.worker is not None and self.worker.isRunning()
        self.clips_btn.setEnabled(self.report is not None and not analysis_busy)

    def on_clips_ready(self, paths) -> None:
        count = len(paths)
        self.stage_label.setText(f"{count} clip(s) exported")
        self.log_line(f"{count} roam clip(s) -> {config.CLIPS_DIR}")
        if self._closing:
            return  # closing: never raise a modal dialog over shutdown
        answer = self._show_box(
            QMessageBox.Information,
            "Clips exported",
            f"{count} clip(s) written to:\n{config.CLIPS_DIR}\n\nOpen the folder now?",
            QMessageBox.Open | QMessageBox.Close,
        )
        if answer == QMessageBox.Open:
            self.open_clips_dir()

    def on_clips_failed(self, message: str) -> None:
        self.stage_label.setText("Clip export failed")
        self.log_line(f"ERROR: {message}")
        if self._closing:
            return
        self._show_box(QMessageBox.Critical, "Clip export failed", message)

    def open_clips_dir(self) -> None:
        config.ensure_directories()
        if hasattr(os, "startfile"):
            os.startfile(config.CLIPS_DIR)  # noqa: S606 - Windows desktop app
        else:  # pragma: no cover - non-Windows fallback
            QDesktopServices.openUrl(QUrl.fromLocalFile(config.CLIPS_DIR))

    def open_outputs(self) -> None:
        config.ensure_directories()
        if hasattr(os, "startfile"):
            os.startfile(config.OUTPUT_DIR)  # noqa: S606 - Windows desktop app
        else:  # pragma: no cover - non-Windows fallback
            QDesktopServices.openUrl(QUrl.fromLocalFile(config.OUTPUT_DIR))

    # ------------------------------------------------------------- events ---
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self._append_paths(paths)
            event.acceptProposedAction()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        # Decide whether the close is accepted *before* touching the analysis
        # worker, so an ignored close never leaves the app with a dead run.
        if self.clip_worker is not None and self.clip_worker.isRunning():
            # Cooperative cancel lands between clips, so the wait is short.
            self.clip_worker.cancel()
            if not self.clip_worker.wait(8000):
                self.log_line("Clip export still running - close cancelled.")
                self.stage_label.setText("Waiting for clip export to finish...")
                event.ignore()
                return
        if self.pdf_worker is not None and self.pdf_worker.isRunning():
            # QThread must never be destroyed while running. Ask the export
            # to stop first; it only cancels before the file is written, so
            # the wait is short in practice (still bounded as a safety net).
            self.pdf_worker.cancel()
            if not self.pdf_worker.wait(8000):
                self.log_line("PDF export still running - close cancelled.")
                self.stage_label.setText("Waiting for PDF export to finish...")
                event.ignore()
                return
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            if not self.worker.wait(8000):
                answer = QMessageBox.question(
                    self,
                    "Analysis still running",
                    "The analysis did not stop in time. Force quit anyway?\n"
                    "(Partial results will be lost.)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    self.log_line("Close cancelled - analysis still stopping.")
                    event.ignore()
                    return
                self.worker.terminate()
                self.worker.wait(2000)
        self._closing = True
        event.accept()

    # ------------------------------------------------------------ helpers ---
    def log_line(self, message: str) -> None:
        self.log.appendPlainText(message)
        logger.info(message)

    def setStatusMessage(self, message: str) -> None:  # noqa: N802 - Qt API
        self.statusBar().showMessage(message)

    def _show_box(
        self,
        icon: QMessageBox.Icon,
        title: str,
        text: str,
        buttons: QMessageBox.StandardButton = QMessageBox.Ok,
    ) -> int:
        """Show a message box whose text wraps instead of widening.

        Failure messages embed absolute paths; a 150-character unbroken path
        made the static helpers size the box at 1623 px - wider than common
        laptops, pushing the default button off-screen.
        """
        box = QMessageBox(icon, title, text, buttons, self)
        for label in box.findChildren(QLabel):
            label.setWordWrap(True)
        return int(box.exec())

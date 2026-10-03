"""PRISM application entry point.

Usage::

    python main.py

The script configures logging, ensures the output directories exist, launches
the PySide6 event loop and hands control to :class:`gui.main_window.MainWindow`.
"""

from __future__ import annotations

import os
import sys
import traceback


def _bootstrap_paths() -> None:
    """Make sure the project root is importable when launched from anywhere."""
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> int:
    """Start the PRISM desktop application."""
    _bootstrap_paths()

    from core import config
    from core.logger import get_logger, setup_logging

    setup_logging()
    config.ensure_directories()
    logger = get_logger("main")

    from PySide6.QtCore import QLockFile, Qt
    from PySide6.QtGui import QColor, QFont, QPalette
    from PySide6.QtWidgets import QApplication, QMessageBox

    from gui.main_window import STYLESHEET, MainWindow

    def _excepthook(exc_type, exc, tb) -> None:  # noqa: ANN001 - sys.excepthook API
        """Never let an exception in a Qt slot die silently."""
        logger.error(
            "Unhandled exception: %s", exc, exc_info=(exc_type, exc, tb)
        )
        traceback.print_exception(exc_type, exc, tb)
        try:
            QMessageBox.critical(
                None,
                "PRISM - unexpected error",
                f"{exc_type.__name__}: {exc}\n\n"
                "Details were written to logs/analysis.log.",
            )
        except Exception:  # pragma: no cover - GUI may be gone
            pass

    sys.excepthook = _excepthook

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("PRISM")
    app.setOrganizationName("PRISM")
    app.setStyle("Fusion")

    # PRISM is a light-themed app: without this, Windows dark mode hands Qt a
    # dark palette and the light stylesheet turns into dark-on-dark text
    # (the Settings dialog was unreadable). Force a light scheme first, then
    # pin an explicit palette so no widget can inherit system-dark colors.
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    except AttributeError:  # pragma: no cover - older Qt
        pass
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#F3F5F9"))
    palette.setColor(QPalette.WindowText, QColor("#1E2A32"))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.AlternateBase, QColor("#F7F9FB"))
    palette.setColor(QPalette.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ToolTipText, QColor("#1E2A32"))
    palette.setColor(QPalette.Text, QColor("#1E2A32"))
    palette.setColor(QPalette.PlaceholderText, QColor("#96A2AE"))
    palette.setColor(QPalette.Button, QColor("#FFFFFF"))
    palette.setColor(QPalette.ButtonText, QColor("#1E2A32"))
    palette.setColor(QPalette.Highlight, QColor("#2E6DB4"))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Link, QColor("#2E6DB4"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#AAB4BD"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#AAB4BD"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#AAB4BD"))
    app.setPalette(palette)

    app.setStyleSheet(STYLESHEET)
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Single instance: two copies would race on outputs/*.tmp chart and
    # settings files. QLockFile auto-releases if a previous run crashed.
    lock = QLockFile(os.path.join(config.DATA_DIR, "prism.lock"))
    if not lock.tryLock(100):
        QMessageBox.warning(
            None,
            "PRISM already running",
            "Another PRISM instance is already open. Close it first.",
        )
        return 0

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(0)
    except Exception:  # pragma: no cover - last-resort guard
        traceback.print_exc()
        sys.exit(1)

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
    from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

    from gui.main_window import STYLESHEET, MainWindow

    def _excepthook(exc_type, exc, tb) -> None:  # noqa: ANN001 - sys.excepthook API
        """Never let an exception in a Qt slot die silently."""
        logger.error(
            "Unhandled exception: %s", exc, exc_info=(exc_type, exc, tb)
        )
        traceback.print_exception(exc_type, exc, tb)
        try:
            # Instance + word wrap: exception strings can embed long absolute
            # paths, which made static helpers size the box past the screen.
            box = QMessageBox(
                QMessageBox.Critical,
                "PRISM - unexpected error",
                f"{exc_type.__name__}: {exc}\n\n"
                "Details were written to logs/analysis.log.",
            )
            for label in box.findChildren(QLabel):
                label.setWordWrap(True)
            box.exec()
        except Exception:  # pragma: no cover - GUI may be gone
            pass

    sys.excepthook = _excepthook

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("PRISM")
    app.setOrganizationName("PRISM")
    app.setStyle("Fusion")

    # PRISM uses a dark "Hextech esports" theme: pin the Qt scheme to Dark and
    # an explicit palette so a Windows light-mode host cannot hand Qt a light
    # palette (dark-on-light in the Settings dialog) and a Windows dark-mode
    # host cannot override our widget colors with system grays.
    try:
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    except AttributeError:  # pragma: no cover - older Qt
        pass
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0B0F1A"))
    palette.setColor(QPalette.WindowText, QColor("#E6ECF7"))
    palette.setColor(QPalette.Base, QColor("#121A2B"))
    palette.setColor(QPalette.AlternateBase, QColor("#16203A"))
    palette.setColor(QPalette.ToolTipBase, QColor("#182238"))
    palette.setColor(QPalette.ToolTipText, QColor("#E6ECF7"))
    palette.setColor(QPalette.Text, QColor("#E6ECF7"))
    palette.setColor(QPalette.PlaceholderText, QColor("#5A6779"))
    palette.setColor(QPalette.Button, QColor("#182238"))
    palette.setColor(QPalette.ButtonText, QColor("#E6ECF7"))
    palette.setColor(QPalette.Highlight, QColor("#2563EB"))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Link, QColor("#3B82F6"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#5A6779"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#5A6779"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5A6779"))
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

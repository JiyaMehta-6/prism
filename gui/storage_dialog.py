"""Modal manager for the data PRISM stores on disk.

Shows one row per stored category (profiles, last report, charts, PDFs,
clips, log, settings) with item counts and sizes; the user ticks what to
remove. Deletion is confirmed and delegated to :mod:`core.storage`, which
only ever touches those exact paths.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from core.storage import clear_entries, format_size, storage_entries


class StorageDialog(QDialog):
    """Inspect and delete locally stored analysis data."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage stored data")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMinimumHeight(320)
        # Keys actually removed during this dialog session; the parent uses
        # it to reload settings / disable PDF export when needed.
        self.cleared_keys: set = set()

        root = QVBoxLayout(self)
        root.setSpacing(10)

        intro = QLabel(
            "PRISM keeps everything on this machine only. Tick the categories "
            "you want to delete - your videos are never touched, and "
            "'Settings' only resets preferences to their defaults."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#8B98AD; font-size:11px;")
        root.addWidget(intro)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "Category", "Items", "Size"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        root.addWidget(self.table, 1)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color:#8B98AD; font-size:11px;")
        root.addWidget(self.summary)

        actions = QHBoxLayout()
        self.select_all_btn = QPushButton("Select all")
        self.select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        self.select_none_btn = QPushButton("Select none")
        self.select_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.setObjectName("primary")
        self.delete_btn.clicked.connect(self._delete_selected)
        actions.addWidget(self.select_all_btn)
        actions.addWidget(self.select_none_btn)
        actions.addStretch(1)
        actions.addWidget(self.delete_btn)
        root.addLayout(actions)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.rejected.connect(self.reject)
        close_box.accepted.connect(self.accept)
        root.addWidget(close_box)
        self._close_box = close_box

        self._entries: list = []
        self.refresh()

    # ------------------------------------------------------------- helpers ---
    def refresh(self) -> None:
        """Re-read the stored categories from disk and rebuild the table."""
        self._entries = storage_entries()
        self.table.setRowCount(len(self._entries))
        total_files = 0
        total_bytes = 0
        for row, entry in enumerate(self._entries):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            check.setCheckState(Qt.Unchecked)
            if entry.count == 0:
                # Nothing to delete: keep the row readable but uncheckable.
                check.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(row, 0, check)

            name = QTableWidgetItem(entry.label)
            name.setToolTip(f"{entry.detail}\nAction: {entry.action}")
            self.table.setItem(row, 1, name)

            count = QTableWidgetItem(str(entry.count) if entry.count else "-")
            count.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, count)

            size = QTableWidgetItem(
                format_size(entry.size_bytes) if entry.count else "-"
            )
            size.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 3, size)

            total_files += entry.count
            total_bytes += entry.size_bytes

        empty = total_files == 0
        self.summary.setText(
            "Nothing stored yet - all categories are empty."
            if empty
            else f"{total_files} item(s), {format_size(total_bytes)} in total."
        )
        self.delete_btn.setEnabled(not empty)
        self.select_all_btn.setEnabled(not empty)
        self.select_none_btn.setEnabled(not empty)

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(state)

    def _checked_keys(self) -> list:
        keys = []
        for row, entry in enumerate(self._entries):
            item = self.table.item(row, 0)
            if item is not None and item.checkState() == Qt.Checked:
                keys.append(entry.key)
        return keys

    # ------------------------------------------------------------- actions ---
    def _delete_selected(self) -> None:
        keys = self._checked_keys()
        if not keys:
            return
        labels = [
            entry.label for entry in self._entries if entry.key in set(keys)
        ]
        box = QMessageBox(
            QMessageBox.Warning,
            "Delete stored data",
            "Delete the following categories from this machine?\n\n- "
            + "\n- ".join(labels)
            + "\n\nThis cannot be undone. Videos are not affected.",
            QMessageBox.Yes | QMessageBox.No,
            self,
        )
        for label in box.findChildren(QLabel):
            label.setWordWrap(True)
        if box.exec() != QMessageBox.Yes:
            return

        errors = clear_entries(keys)
        done = [key for key in keys if key not in errors]
        self.cleared_keys.update(done)
        self.refresh()
        if errors:
            details = "\n".join(f"{key}: {reason}" for key, reason in errors.items())
            self._show(
                QMessageBox.Warning,
                "Some data could not be deleted",
                f"Removed {len(done)} categories.\n\nFailed:\n{details}",
            )
        else:
            self._show(
                QMessageBox.Information,
                "Data deleted",
                f"Removed {len(done)} categories from this machine.",
            )

    def _show(self, icon: QMessageBox.Icon, title: str, text: str) -> None:
        box = QMessageBox(icon, title, text, QMessageBox.Ok, self)
        for label in box.findChildren(QLabel):
            label.setWordWrap(True)
        box.exec()

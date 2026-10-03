"""Settings dialog for PRISM.

Exposes every analysis parameter with sensible bounds and safe defaults.
Values are written to ``data/settings.json`` through :mod:`core.config` so the
next launch restores them.
"""

from __future__ import annotations

import copy
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from core.config import Settings
from vision.region_mapper import DEFAULT_ANCHORS


def _finite(value: object, default: float) -> float:
    """Coerce a setting to a finite float (never raises)."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class SettingsDialog(QDialog):
    """Modal editor for :class:`~core.config.Settings`."""

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("PRISM Settings")
        self.setMinimumWidth(460)
        self.setModal(True)

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.sample_fps = QDoubleSpinBox()
        # Ranges must mirror Settings.validated() or round-tripping through
        # the dialog would silently clamp user values.
        self.sample_fps.setRange(0.1, 30.0)
        self.sample_fps.setSingleStep(0.1)
        self.sample_fps.setDecimals(4)
        self.sample_fps.setSuffix(" fps")
        form.addRow("Frame sampling rate", self.sample_fps)

        rate_note = QLabel(
            "Higher rates improve smoothness but take longer "
            "(1 fps ≈ 900 samples over a 15-minute window)."
        )
        rate_note.setWordWrap(True)
        rate_note.setStyleSheet("color:#6B7A88; font-size:11px;")
        form.addRow("", rate_note)

        # QDoubleSpinBox: early_game_minutes is a float in Settings - a
        # QSpinBox would silently truncate 7.5 to 7.
        self.window_minutes = QDoubleSpinBox()
        self.window_minutes.setRange(1.0, 15.0)
        self.window_minutes.setSingleStep(0.5)
        self.window_minutes.setDecimals(1)
        self.window_minutes.setSuffix(" min")
        form.addRow("Early game window", self.window_minutes)

        self.minimap_side = QComboBox()
        self.minimap_side.addItems(["auto", "left", "right"])
        form.addRow("Minimap screen side", self.minimap_side)

        self.minimap_roi = QLineEdit()
        self.minimap_roi.setPlaceholderText("auto (or x,y,w,h as 0-1 fractions)")
        form.addRow("Minimap ROI override", self.minimap_roi)

        self.marker_conf = QDoubleSpinBox()
        self.marker_conf.setRange(0.0, 0.95)
        self.marker_conf.setSingleStep(0.05)
        self.marker_conf.setDecimals(2)
        form.addRow("Minimum marker confidence", self.marker_conf)

        self.max_gap = QSpinBox()
        self.max_gap.setRange(0, 30)
        self.max_gap.setSuffix(" frames")
        form.addRow("Max interpolation gap", self.max_gap)

        self.ward_tracking = QCheckBox("Detect vision blooms (ward-like) on the minimap")
        form.addRow(self.ward_tracking)

        root.addLayout(form)
        root.addWidget(self._group("Optical Character Recognition"))
        root.addWidget(self._region_group())

        general = QFormLayout()
        self.player_label = QLineEdit()
        general.addRow("Player label", self.player_label)

        self.similarity_top = QSpinBox()
        self.similarity_top.setRange(1, 50)
        general.addRow("Similarity results", self.similarity_top)
        root.addLayout(general)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        reset = QPushButton("Restore defaults")
        reset.clicked.connect(self._restore_defaults)
        buttons.addButton(reset, QDialogButtonBox.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.load_from_settings()

    def _group(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        layout = QFormLayout(box)
        layout.setLabelAlignment(Qt.AlignRight)

        self.ocr_enabled = QCheckBox("Enable OCR (EasyOCR, runs locally)")
        layout.addRow(self.ocr_enabled)

        self.use_clock = QCheckBox("Align video time to the in-game match clock")
        layout.addRow(self.use_clock)

        self.ocr_interval = QDoubleSpinBox()
        self.ocr_interval.setRange(5.0, 3600.0)
        self.ocr_interval.setSingleStep(5.0)
        self.ocr_interval.setSuffix(" s")
        layout.addRow("OCR sampling interval", self.ocr_interval)

        self.ocr_conf = QDoubleSpinBox()
        self.ocr_conf.setRange(0.0, 1.0)
        self.ocr_conf.setSingleStep(0.05)
        self.ocr_conf.setDecimals(2)
        layout.addRow("Minimum OCR confidence", self.ocr_conf)

        note = QLabel(
            "OCR reads the match clock and the kill scoreboard. If confidence is too "
            "low, PRISM falls back to video time and continues normally."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6B7A88; font-size:11px;")
        layout.addRow(note)
        return box

    # Only scalar anchors are exposed; point/dict anchors (bases, towers) stay
    # in vision.region_mapper.DEFAULT_ANCHORS for code-level overrides.
    _REGION_FIELDS = (
        ("lane_inset", "Lane inset", 0.01, 0.25),
        ("base_radius", "Base radius", 0.02, 0.30),
        ("river_threshold", "River band width", 0.01, 0.20),
        ("mid_threshold", "Mid-lane band width", 0.01, 0.20),
    )

    def _region_group(self) -> QGroupBox:
        box = QGroupBox("Map Regions")
        layout = QFormLayout(box)
        layout.setLabelAlignment(Qt.AlignRight)
        self._region_widgets = {}
        for key, label, low, high in self._REGION_FIELDS:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setSingleStep(0.005)
            spin.setDecimals(3)
            layout.addRow(label, spin)
            self._region_widgets[key] = spin
        note = QLabel(
            "Geometry of the minimap regions (lanes, river, bases) as polygon bands "
            "and circles. Defaults fit the standard Summoner's Rift minimap."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#6B7A88; font-size:11px;")
        layout.addRow(note)
        return box

    def load_from_settings(self) -> None:
        """Populate widgets from the current settings object."""
        # Work on a copy: validating must never mutate the caller's Settings,
        # and Cancel has to leave the original untouched.
        s = self.settings = copy.deepcopy(self.settings).validated()
        self.sample_fps.setValue(_finite(s.sample_fps, 1.0))
        self.window_minutes.setValue(_finite(s.early_game_minutes, 15.0))
        self.minimap_side.setCurrentText(str(s.minimap_side or "auto"))
        roi = [v for v in (s.minimap_roi or []) if isinstance(v, (int, float))]
        self.minimap_roi.setText(
            ",".join(f"{float(v):.6f}" for v in roi[:4]) if any(roi) else ""
        )
        self.marker_conf.setValue(_finite(s.marker_min_confidence, 0.4))
        self.max_gap.setValue(int(_finite(s.max_interpolate_gap, 6)))
        self.ward_tracking.setChecked(bool(s.ward_tracking))
        self.ocr_enabled.setChecked(bool(s.ocr_enabled))
        self.use_clock.setChecked(bool(s.use_game_clock))
        self.ocr_interval.setValue(_finite(s.ocr_interval_sec, 20.0))
        self.ocr_conf.setValue(_finite(s.ocr_min_confidence, 0.5))
        self.player_label.setText(str(s.player_label or "Player"))
        self.similarity_top.setValue(int(_finite(s.similarity_top_n, 5)))
        merged_anchors = dict(DEFAULT_ANCHORS)
        merged_anchors.update(
            {k: v for k, v in (s.region_anchors or {}).items() if k in self._region_widgets}
        )
        for key, widget in self._region_widgets.items():
            widget.setValue(_finite(merged_anchors.get(key), 0.0))

    def apply_to_settings(self) -> Settings:
        """Write widget values back into the settings object."""
        s = self.settings
        s.sample_fps = float(self.sample_fps.value())
        s.early_game_minutes = float(self.window_minutes.value())
        s.minimap_side = self.minimap_side.currentText()
        roi_text = self.minimap_roi.text().strip()
        if roi_text:
            try:
                values = [float(part) for part in roi_text.replace(";", ",").split(",")]
            except ValueError:
                values = []
            if len(values) == 4 and all(math.isfinite(v) for v in values) and all(
                v > 0 for v in values
            ):
                s.minimap_roi = values
            else:
                # Malformed override: keep the previous value and say so
                # instead of silently wiping the user's configuration.
                QMessageBox.warning(
                    self,
                    "Invalid minimap ROI",
                    "The ROI override must be four positive numbers "
                    "(x,y,w,h as 0-1 fractions), e.g. 0.005,0.78,0.21,0.21.\n"
                    "The previous value was kept.",
                )
        else:
            s.minimap_roi = [0.0, 0.0, 0.0, 0.0]
        s.marker_min_confidence = float(self.marker_conf.value())
        s.max_interpolate_gap = int(self.max_gap.value())
        s.ward_tracking = self.ward_tracking.isChecked()
        s.ocr_enabled = self.ocr_enabled.isChecked()
        s.use_game_clock = self.use_clock.isChecked()
        s.ocr_interval_sec = float(self.ocr_interval.value())
        s.ocr_min_confidence = float(self.ocr_conf.value())
        s.player_label = self.player_label.text().strip() or "Player"
        s.similarity_top_n = int(self.similarity_top.value())
        # Persist only deviations from the built-in geometry so future default
        # improvements still apply to keys the user never touched.
        overrides = {}
        for key, widget in self._region_widgets.items():
            value = float(widget.value())
            if abs(value - float(DEFAULT_ANCHORS[key])) > 1e-9:
                overrides[key] = value
        s.region_anchors = overrides
        return s.validated()

    def _restore_defaults(self) -> None:
        self.settings = Settings()
        self.load_from_settings()

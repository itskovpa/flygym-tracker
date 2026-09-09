"""Dockable two-face activity heatmap, painted directly with Qt.

Elapsed time runs left to right; each face has sixteen locally-labelled vial rows. Numeric cells
share one colour scale. Missing data is blank, while a period measured on the opposite face is
hatched, so neither can be mistaken for a measured zero.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QLabel, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from flygym_tracker.gui import theme
from flygym_tracker.gui.behaviour_series import (AREA_NORMALIZED_FIELDS, BIN_CHOICES, FACES,
                                                 HEATMAP_OTHER_FACE, HeatmapSnapshot,
                                                 VIALS_PER_FACE, BehaviourSeries)
from flygym_tracker.gui.flow_layout import FlowLayout
from flygym_tracker.gui.plot_dock import SAMPLE_CAPS, _bin_label, _hms, _tick

HEATMAP_KEY = "activity_heatmap"
HEATMAP_LABEL = "activity heatmap (both faces)"
HEATMAP_FIELDS = (
    ("active_fraction_mean", "active fraction"),
    ("motion_px_sum", "motion (px)"),
)

LEFT = 52
RIGHT = 10
TOP = 18
FACE_GAP = 24
AXIS_HEIGHT = 25
LEGEND_HEIGHT = 42


def _heat_colour(value: float, span) -> QColor:
    """A perceptually ordered blue-to-cyan ramp; amber remains reserved for sensor control."""
    low, high = span
    fraction = max(0.0, min(1.0, (value - low) / max(1e-12, high - low)))
    stops = ((0.0, QColor("#17243A")), (0.5, QColor("#287F9E")),
             (1.0, QColor("#D8FAF4")))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if fraction <= b:
            f = (fraction - a) / (b - a)
            return QColor(round(ca.red() + f * (cb.red() - ca.red())),
                          round(ca.green() + f * (cb.green() - ca.green())),
                          round(ca.blue() + f * (cb.blue() - ca.blue())))
    return stops[-1][1]


class ActivityHeatmapWidget(QWidget):
    """Painter surface for one immutable :class:`HeatmapSnapshot`."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.snapshot = HeatmapSnapshot("active_fraction_mean", 10.0, (), {}, set(), None, None)
        self.setMinimumSize(360, 390)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def configure(self, snapshot: HeatmapSnapshot) -> None:
        self.snapshot = snapshot
        self.update()

    def _geometry(self):
        heat_w = max(1.0, self.width() - LEFT - RIGHT)
        usable_h = max(32.0, self.height() - TOP - FACE_GAP - AXIS_HEIGHT - LEGEND_HEIGHT)
        row_h = usable_h / (2 * VIALS_PER_FACE)
        block_h = row_h * VIALS_PER_FACE
        return heat_w, row_h, block_h

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.INK_1))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        font = QFont(painter.font())
        font.setPointSize(theme.PT_TINY)
        painter.setFont(font)
        heat_w, row_h, block_h = self._geometry()
        snap = self.snapshot

        for face_i, face in enumerate(FACES):
            y0 = TOP + face_i * (block_h + FACE_GAP)
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(4, round(y0 + 10), "FACE %s" % face)
            painter.fillRect(QRectF(LEFT, y0, heat_w, block_h), QColor(theme.INK_0))
            if snap.time_range is not None:
                t0, t1 = snap.time_range
                time_span = max(1e-12, t1 - t0)
                other_brush = QBrush(QColor(theme.INK_4), Qt.BrushStyle.Dense4Pattern)
                for bucket in snap.buckets:
                    if snap.state(face, 0, bucket) != HEATMAP_OTHER_FACE:
                        continue
                    x = LEFT + heat_w * (bucket * snap.bin_seconds - t0) / time_span
                    w = max(1.0, heat_w * snap.bin_seconds / time_span)
                    painter.fillRect(QRectF(x, y0, w, block_h), other_brush)
                for (cell_face, vial, bucket), value in snap.values.items():
                    if cell_face != face:
                        continue
                    x = LEFT + heat_w * (bucket * snap.bin_seconds - t0) / time_span
                    w = max(1.0, heat_w * snap.bin_seconds / time_span)
                    painter.fillRect(QRectF(x, y0 + vial * row_h, w, row_h),
                                     _heat_colour(value, snap.value_range))
            painter.setPen(QPen(QColor(theme.RULE), 1))
            for vial in range(VIALS_PER_FACE + 1):
                y = y0 + vial * row_h
                painter.drawLine(round(LEFT), round(y), round(LEFT + heat_w), round(y))
            painter.setPen(QColor(theme.TEXT_FAINT))
            for vial in range(VIALS_PER_FACE):
                painter.drawText(30, round(y0 + (vial + .72) * row_h), str(vial + 1))
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(4, round(y0 + block_h - 2), "vial")

        axis_y = TOP + 2 * block_h + FACE_GAP
        painter.setPen(QColor(theme.TEXT_FAINT))
        if snap.time_range is not None:
            t0, t1 = snap.time_range
            painter.drawText(LEFT, round(axis_y + 15), _hms(t0))
            mid = _hms((t0 + t1) / 2)
            mid_w = painter.fontMetrics().horizontalAdvance(mid)
            painter.drawText(round(LEFT + heat_w / 2 - mid_w / 2), round(axis_y + 15), mid)
            end = _hms(t1)
            painter.drawText(round(LEFT + heat_w - painter.fontMetrics().horizontalAdvance(end)),
                             round(axis_y + 15), end)
        painter.drawText(LEFT, round(axis_y + AXIS_HEIGHT), "elapsed time")
        self._paint_legend(painter, axis_y + AXIS_HEIGHT, heat_w)

    def _paint_legend(self, painter: QPainter, y: float, heat_w: float) -> None:
        snap = self.snapshot
        legend_w = min(180.0, max(90.0, heat_w * .38))
        if snap.value_range is not None:
            gradient = QLinearGradient(LEFT, y + 5, LEFT + legend_w, y + 5)
            gradient.setColorAt(0.0, _heat_colour(snap.value_range[0], snap.value_range))
            gradient.setColorAt(0.5, _heat_colour(sum(snap.value_range) / 2, snap.value_range))
            gradient.setColorAt(1.0, _heat_colour(snap.value_range[1], snap.value_range))
            painter.fillRect(QRectF(LEFT, y + 4, legend_w, 9), QBrush(gradient))
            painter.setPen(QColor(theme.TEXT_FAINT))
            painter.drawText(LEFT, round(y + 27), _tick(snap.value_range[0]))
            high = _tick(snap.value_range[1])
            painter.drawText(round(LEFT + legend_w - painter.fontMetrics().horizontalAdvance(high)),
                             round(y + 27), high)
        x = LEFT + legend_w + 16
        painter.fillRect(QRectF(x, y + 4, 12, 9), QColor(theme.INK_0))
        painter.setPen(QColor(theme.TEXT_FAINT))
        painter.drawText(round(x + 16), round(y + 13), "missing")
        x += 76
        painter.fillRect(QRectF(x, y + 4, 12, 9),
                         QBrush(QColor(theme.INK_4), Qt.BrushStyle.Dense4Pattern))
        painter.drawText(round(x + 16), round(y + 13), "other face visible")


class ActivityHeatmapPanel(QWidget):
    """Controls and the two-face heatmap."""

    def __init__(self, series: BehaviourSeries, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.series = series
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        controls = FlowLayout(margin=0, spacing=8)

        controls.addWidget(QLabel("metric"))
        self.metric_box = QComboBox()
        for field, label in HEATMAP_FIELDS:
            self.metric_box.addItem(label, field)
        self.metric_box.currentIndexChanged.connect(self._metric_changed)
        controls.addWidget(self.metric_box)

        controls.addWidget(QLabel("display bin"))
        self.bin_box = QComboBox()
        for seconds in BIN_CHOICES:
            if seconds > 0:
                self.bin_box.addItem(_bin_label(seconds), seconds)
        self.bin_box.setCurrentIndex(self.bin_box.findData(10))
        self.bin_box.currentIndexChanged.connect(self.refresh)
        self.bin_box.setToolTip("Median display grouping. It never creates measurements finer than "
                                "the activity bins recorded in the file.")
        controls.addWidget(self.bin_box)

        controls.addWidget(QLabel("show"))
        self.samples_box = QComboBox()
        for cap in SAMPLE_CAPS:
            self.samples_box.addItem("all" if cap is None else "last %d" % cap, cap)
        self.samples_box.setCurrentIndex(self.samples_box.findData(500))
        self.samples_box.currentIndexChanged.connect(self.refresh)
        self.samples_box.setToolTip("Limit occupied time columns drawn from the retained live history. "
                                    "The CSV and shared history limit are unchanged.")
        controls.addWidget(self.samples_box)

        self.area_box = QCheckBox("per ROI area")
        self.area_box.toggled.connect(self.refresh)
        controls.addWidget(self.area_box)
        layout.addLayout(controls)

        self.range_label = QLabel("")
        self.range_label.setProperty("role", "note")
        self.range_label.setWordWrap(True)
        self.resolution_label = QLabel("")
        self.resolution_label.setProperty("role", "note")
        self.resolution_label.setWordWrap(True)
        layout.addWidget(self.range_label)
        layout.addWidget(self.resolution_label)
        self.heatmap = ActivityHeatmapWidget()
        layout.addWidget(self.heatmap, 1)
        self._metric_changed()

    def field(self) -> str:
        return str(self.metric_box.currentData())

    def selected_bin_seconds(self) -> float:
        return float(self.bin_box.currentData())

    def effective_bin_seconds(self) -> float:
        widths = self.series.activity_bin_widths
        return (max([self.selected_bin_seconds()] + list(widths)) if widths
                else self.selected_bin_seconds())

    def _metric_changed(self) -> None:
        applies = self.field() in AREA_NORMALIZED_FIELDS
        self.area_box.setEnabled(applies)
        self.area_box.setChecked(applies)
        self.area_box.setToolTip("Rescale motion pixel sums to the median vial ROI area." if applies
                                 else "Active fraction is already normalized by lit ROI area.")
        self.refresh()

    def refresh(self) -> None:
        selected = self.selected_bin_seconds()
        effective = self.effective_bin_seconds()
        snapshot = self.series.activity_heatmap(
            self.field(), bin_seconds=effective, max_columns=self.samples_box.currentData(),
            normalize_area=self.area_box.isEnabled() and self.area_box.isChecked())
        self.heatmap.configure(snapshot)
        widths = sorted(self.series.activity_bin_widths)
        if widths:
            recorded = ", ".join(_bin_label(w) for w in widths)
            if selected < max(widths):
                resolution = ("Recorded activity bins: %s. Selected display bin: %s. Heatmap cells "
                              "remain %s; a smaller display bin cannot recover finer measurements."
                              % (recorded, _bin_label(selected), _bin_label(effective)))
            else:
                resolution = ("Recorded activity bins: %s. Display cells: %s; each cell is the "
                              "median of available recorded rows." % (recorded, _bin_label(effective)))
        else:
            resolution = "Waiting for activity rows; the normal recording resolution is 10 s."
        self.resolution_label.setText(resolution)
        if snapshot.value_range is None:
            summary = "no measured activity yet"
        else:
            summary = ("shared colour scale %.3g to %.3g across both faces" % snapshot.value_range)
            if snapshot.time_range:
                summary += "   -   showing %s to %s" % tuple(_hms(t) for t in snapshot.time_range)
        if self.series.dropped_rows:
            summary += "   -   %s older rows omitted from live history; CSV unchanged" % format(
                self.series.dropped_rows, ",")
        self.range_label.setText(summary)


class ActivityHeatmapDock(QDockWidget):
    def __init__(self, series: BehaviourSeries, parent: Optional[QWidget] = None) -> None:
        super().__init__("Activity heatmap", parent)
        self.field = HEATMAP_KEY
        self.setObjectName("plot-%s" % HEATMAP_KEY)
        self.panel = ActivityHeatmapPanel(series)
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setWidget(scroll)
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                         | QDockWidget.DockWidgetFeature.DockWidgetClosable)

    def refresh(self) -> None:
        self.panel.refresh()

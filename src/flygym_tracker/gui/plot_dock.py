"""One dockable panel per behavioural parameter: 8x2 vials of face A, 8x2 of face B.

LAID OUT AS THE RIG IS. Sixteen vials per face in two rows of eight, in vial-id order, so a dead
column on screen is a dead column on the drum without anyone having to map an index to a tube. The
two faces are stacked in the same window because the interesting comparison on this rig is usually
the same physical position on opposite faces.

PAINTED, NOT CHARTED. Thirty-two live plots is exactly where a general charting widget stops being
free: QtCharts would build 32 scenes, 32 axis pairs and 32 series objects and re-lay them on every
update, for cells that are 120 px wide and show a line and nothing else. A `paintEvent` over
polylines is a few hundred microseconds and gives the grid the same visual language as the rest of
the app.

ONE Y RANGE ACROSS ALL 32 CELLS, and this is a measurement decision rather than a drawing one.
Per-cell autoscaling would draw an empty vial's noise at the same amplitude as a busy vial's real
signal -- and the whole point of a grid is that vials can be compared with each other at a glance.
The range is printed on the panel so the scale is never a guess.

A DOCK PER PARAMETER, because the operator asked to watch more than one at a time and to be able
to put each where they want it. Qt gives floating, re-docking and tabbing for free, and closing one
costs nothing -- the data lives in the shared `BehaviourSeries`, so a dock reopened later draws the
whole run, not just what arrived after it was reopened.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QHBoxLayout, QLabel, QSizePolicy,
                               QVBoxLayout, QWidget)

from flygym_tracker.gui import theme
from flygym_tracker.gui.behaviour_series import (BIN_CHOICES, FACES, PLOT_LABELS, PLOTTABLE,
                                                 VIALS_PER_FACE, BehaviourSeries)

#: Columns of the grid. 8x2 per face, as the rig is built.
COLUMNS = 8
ROWS = VIALS_PER_FACE // COLUMNS

#: One colour per FACE, shared by all sixteen of its vials -- the same rule as the track overlay,
#: so a colour means the same thing wherever it appears in this app.
FACE_COLORS = {"A": QColor(theme.FOCUS), "B": QColor(theme.DEFAULT_GREEN)}
FALLBACK_COLOR = QColor(theme.TEXT_DIM)


class VialPlotGrid(QWidget):
    """8x2 sparklines for one face, sharing one y range with the rest of the panel."""

    def __init__(self, face: str, series: BehaviourSeries,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.face = face
        self.series = series
        self.field = PLOTTABLE[0][0]
        self.bin_seconds = 10.0
        self.cumulative = False
        self._range = None
        self._time = None
        self._points = {}
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def configure(self, *, field, bin_seconds, cumulative, value_range, time_range,
                  points) -> None:
        """Adopt ONE SNAPSHOT: the points AND the range they were measured from, together.

        THE BUG THIS FIXES, reported from the rig: a line ran outside its cell. `paintEvent` used
        to re-read the shared store, while the y range had been computed a moment earlier in
        `refresh` -- and behaviour rows arrive from the run thread in between. So the painter drew
        points the range had never seen, and anything above the old maximum was plotted above the
        top of the cell. Range and points must come from the same read or the axis is a claim
        about different data than the line.
        """
        self.field = field
        self.bin_seconds = bin_seconds
        self.cumulative = cumulative
        self._range = value_range
        self._time = time_range
        self._points = points or {}
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.INK_1))
        cell_w = self.width() / COLUMNS
        cell_h = self.height() / ROWS
        colour = FACE_COLORS.get(self.face, FALLBACK_COLOR)
        font = QFont(painter.font())
        font.setPointSize(theme.PT_TINY)
        painter.setFont(font)

        points = self._points
        for index in range(VIALS_PER_FACE):
            row, col = divmod(index, COLUMNS)
            x0, y0 = col * cell_w, row * cell_h
            painter.setPen(QPen(QColor(theme.RULE), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(int(x0), int(y0), int(cell_w) - 1, int(cell_h) - 1)

            painter.setPen(QPen(QColor(theme.TEXT_FAINT), 1))
            painter.drawText(int(x0) + 4, int(y0) + 12, "%s%d" % (self.face, index + 1))

            painter.save()
            # CLIPPED TO ITS OWN CELL. The snapshot above makes an out-of-range point impossible;
            # this makes it impossible to DRAW one, so a future bug of the same kind is visible as
            # a clipped line inside the right cell rather than a stray line across a neighbour.
            painter.setClipRect(int(x0) + 1, int(y0) + 1, int(cell_w) - 2, int(cell_h) - 2)
            self._paint_cell(painter, points.get(index) or [], colour,
                             x0 + 4, y0 + 16, cell_w - 8, cell_h - 22)
            painter.restore()

    def _paint_cell(self, painter, points, colour, x, y, w, h) -> None:
        if w <= 2 or h <= 2:
            return
        if not points or self._range is None or self._time is None:
            # NOTHING IS DRAWN FOR A VIAL WITH NO DATA. A flat line at the bottom would be this
            # panel claiming a measurement of zero where there was no measurement at all.
            return
        low, high = self._range
        t0, t1 = self._time
        span_t = max(1e-9, t1 - t0)
        span_v = max(1e-9, high - low)
        polygon = QPolygonF()
        for t, value in points:
            px = x + w * (t - t0) / span_t
            py = y + h * (1.0 - (value - low) / span_v)
            polygon.append(QPointF(px, py))
        painter.setPen(QPen(colour, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if len(polygon) == 1:
            painter.drawEllipse(polygon[0], 1.6, 1.6)
        else:
            painter.drawPolyline(polygon)


class BehaviourPlotPanel(QWidget):
    """The controls plus both faces' grids, for ONE parameter."""

    closed = Signal(str)

    def __init__(self, series: BehaviourSeries, field: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.series = series
        self.field = field
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("bin"))
        self.bin_box = QComboBox()
        for seconds in BIN_CHOICES:
            self.bin_box.addItem(_bin_label(seconds), seconds)
        self.bin_box.setCurrentIndex(BIN_CHOICES.index(10))
        self.bin_box.currentIndexChanged.connect(self.refresh)
        self.bin_box.setToolTip(
            "Group the rows for display. RE-BINNING IS NOT RE-MEASURING: behaviour.csv keeps one "
            "row per vial per dwell whatever this says, and each point here is the median of the "
            "rows in its window.")
        controls.addWidget(self.bin_box)

        self.cumulative_box = QCheckBox("cumulative")
        self.cumulative_box.setToolTip(
            "Running sum of the binned values. A genuine total-so-far for a rate-like parameter "
            "such as path length; not meaningful for a level such as mean height.")
        self.cumulative_box.toggled.connect(self.refresh)
        controls.addWidget(self.cumulative_box)

        self.range_label = QLabel("")
        self.range_label.setProperty("role", "note")
        self.range_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        controls.addWidget(self.range_label, 1)
        layout.addLayout(controls)

        self.grids = {}
        for face in FACES:
            label = QLabel("FACE %s" % face)
            label.setProperty("role", "grouptitle")
            layout.addWidget(label)
            grid = VialPlotGrid(face, series)
            self.grids[face] = grid
            layout.addWidget(grid, 1)
        self.refresh()

    def bin_seconds(self) -> float:
        return float(self.bin_box.currentData() or 10)

    def cumulative(self) -> bool:
        return self.cumulative_box.isChecked()

    def refresh(self) -> None:
        """Re-read the shared store. Cheap enough to call on every completed dwell."""
        kwargs = {"bin_seconds": self.bin_seconds(), "cumulative": self.cumulative()}
        # ONE READ of the store for the whole panel: the points every cell draws and the range
        # every cell is scaled by come from the same snapshot. See `VialPlotGrid.configure`.
        points = {face: self.series.face_series(self.field, face, **kwargs) for face in FACES}
        value_range = _range_of(points)
        time_range = self.series.time_range()
        for face, grid in self.grids.items():
            grid.configure(field=self.field, value_range=value_range, time_range=time_range,
                           points=points.get(face) or {}, **kwargs)
        if value_range is None:
            self.range_label.setText("no data yet")
        else:
            low, high = value_range
            span = "%s   -   y %.3g to %.3g" % (PLOT_LABELS.get(self.field, self.field), low, high)
            if time_range is not None:
                span += "   -   %s of run" % _hms(time_range[1])
            # THE RANGE IS PRINTED because every cell shares it: without the numbers, a tall line
            # in one cell and a flat one in another are not comparable, which is the whole point.
            self.range_label.setText(span)


class BehaviourPlotDock(QDockWidget):
    """A `BehaviourPlotPanel` in a dock: floatable, closable, tabbable with its siblings."""

    def __init__(self, series: BehaviourSeries, field: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(PLOT_LABELS.get(field, field), parent)
        self.field = field
        self.setObjectName("plot-%s" % field)
        self.panel = BehaviourPlotPanel(series, field)
        self.setWidget(self.panel)
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                         | QDockWidget.DockWidgetFeature.DockWidgetClosable)

    def refresh(self) -> None:
        self.panel.refresh()


def _range_of(points_by_face) -> Optional[tuple]:
    """`(low, high)` over every point of every cell, or None. Computed from the SAME snapshot the
    cells will draw, which is the whole point -- see `VialPlotGrid.configure`."""
    low = high = None
    for cells in (points_by_face or {}).values():
        for points in (cells or {}).values():
            for _t, value in points or ():
                low = value if low is None else min(low, value)
                high = value if high is None else max(high, value)
    if low is None:
        return None
    if high - low < 1e-9:
        return (low - 0.5, high + 0.5)
    return (low, high)


def _bin_label(seconds: int) -> str:
    if seconds < 60:
        return "%d s" % seconds
    if seconds < 3600:
        return "%d min" % (seconds // 60)
    return "%d h" % (seconds // 3600)


def _hms(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    return "%d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)

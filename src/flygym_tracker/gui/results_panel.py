"""The measurement, on screen, while it is being made.

WHAT WAS MISSING. The window showed frames, elapsed time, fps, a rotation count and a strip of
cells whose brightness tracked activity. Every one of those says THE MACHINE IS RUNNING. None of
them is a measurement. An operator could watch a three-day experiment from start to finish without
once seeing a number that would end up in the results -- so a threshold set too high, a vial that
never reports, or a face that is never identified all look exactly like a healthy run until the CSV
is opened afterwards, by which time the flies are gone.

TWO THINGS ARE SHOWN, AND THEY ARE DIFFERENT KINDS OF THING.

  LIVE (per frame, throttled to 5 Hz): what the pipeline is measuring RIGHT NOW, per vial. This is
  a working value -- it has not been binned, it is not in any file, and it changes faster than it
  can be read. It is here to answer "is this vial reporting at all, and is the threshold sane".

  RECORDED (per bin): the rows that have just been WRITTEN to activity.csv, verbatim. This is the
  result. It arrives every `bin_seconds` and it is what the analysis will see.

THEY ARE LABELLED AS DIFFERENT THINGS ON PURPOSE. Showing a live per-frame number next to a binned
one without saying which is which invites reading the fast-moving one as the measurement -- and the
fast one is the only one that is NOT in the output. Same rule as "delivered fps" in the camera
caption and "recorded, not the camera" in the vial selector: never present a number as more than it
is.

NOTHING HERE COMPUTES ANYTHING. Every figure is passed through from the pipeline, which counted it.
A panel that derived its own summary would be a second implementation of the measurement, on the
GUI thread, capable of disagreeing with the file.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from flygym_tracker.gui import theme

#: The drum: two faces of sixteen.
VIALS_PER_FACE = 16
N_FACES = 2

#: Columns of the recorded-rows table, and the `ActivityRecord` field each one reads. The field
#: names are the CSV's own, so what is on screen and what is in the file cannot drift apart.
COLUMNS = (
    ("elapsed", "elapsed_s"),
    ("face", "face"),
    ("vial", "vial_id"),
    ("motion px", "motion_px_sum"),
    ("active", "active_fraction_mean"),
    ("still", "n_stationary_frames"),
    ("rot", "n_rotating_frames"),
    ("lit px", "lit_area_px"),
)

#: How many recorded rows to keep on screen. A 3-day run at 10 s bins writes ~26000 rows per vial;
#: keeping them all in a widget would be a slow memory leak for no gain, because the file has them
#: all and this panel is for watching, not for analysis.
MAX_ROWS = 400


def _fmt(value) -> str:
    if isinstance(value, float):
        return "%.3f" % value if abs(value) < 100 else "%.1f" % value
    return str(value)


class VialActivityGrid(QWidget):
    """One cell per vial: the live activity number, coloured by magnitude.

    THE NUMBER IS THE POINT, not the colour. The strip this replaces showed brightness only, and
    was documented as "a presence check, not a measurement display" -- which was honest, and left
    the operator with no way to tell 3 from 300 without opening the file afterwards.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.cells: List[QLabel] = []
        for face in range(N_FACES):
            tag = QLabel("ABCDEFGH"[face])
            tag.setProperty("role", "grouptitle")
            layout.addWidget(tag, face, 0)
            for vial in range(VIALS_PER_FACE):
                cell = QLabel("")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # SMALL MINIMUM, and the grid lives in a scroll area (see `ResultsPanel`).
                # 16 cells at a comfortable fixed width made this widget demand 584 px,
                # which the splitter then took from the PICTURE -- squeezing the pane the
                # run is actually watched in down to its own 320 px minimum. A strip of
                # cells must never be the thing that decides how wide a pane is; same
                # rule the button strips learned in `flow_layout`.
                cell.setMinimumSize(22, 20)
                cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                cell.setToolTip("face %s, vial %d" % ("AB"[face], vial + 1))
                layout.addWidget(cell, face, vial + 1)
                self.cells.append(cell)
        self.clear()

    def clear(self) -> None:
        for cell in self.cells:
            cell.setText("")
            cell.setStyleSheet(self._style(theme.INK_2, theme.TEXT_FAINT))

    def _style(self, fill: str, text: str) -> str:
        return ("background: %s; color: %s; border: 1px solid %s; border-radius: 2px; "
                "font-family: %s; font-size: %dpt;" % (fill, text, theme.RULE, theme.FONT_MONO,
                                                       theme.PT_TINY))

    def set_activity(self, vial_results: dict) -> None:
        """`vial_results` is ``{global_vial_id: (motion_px, lit_area_px, active_fraction)}``.

        A vial the pipeline did not report is left BLANK rather than drawn as zero: "no reading"
        and "a reading of zero" are different facts, and drawing them the same way would be the
        display inventing a measurement. On this rig that distinction is load-bearing -- only one
        drum face is visible at a time, so the other face's sixteen SHOULD be blank.
        """
        for index, cell in enumerate(self.cells):
            result = vial_results.get(index)
            if result is None:
                cell.setText("")
                cell.setStyleSheet(self._style(theme.INK_2, theme.TEXT_FAINT))
                continue
            try:
                motion = float(result[0])
            except (TypeError, IndexError, ValueError):
                motion = 0.0
            cell.setText("%d" % int(motion) if motion < 10000 else "%dk" % int(motion / 1000))
            weight = max(0.0, min(1.0, motion / 400.0))
            fill = theme.INK_2 if weight <= 0.01 else "rgba(127, 178, 217, %.2f)" % (
                0.12 + 0.88 * weight)
            cell.setStyleSheet(self._style(fill, theme.TEXT if weight > 0.35 else theme.TEXT_DIM))


class ResultsPanel(QWidget):
    """Live per-vial activity, and the rows as they are written to activity.csv."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows_written = 0
        self._bins = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        live_head = QHBoxLayout()
        title = QLabel("MEASURING NOW")
        title.setProperty("role", "grouptitle")
        live_head.addWidget(title)
        self.live_note = QLabel("per frame, not yet binned - not in the file")
        self.live_note.setProperty("role", "note")
        live_head.addWidget(self.live_note, 1)
        layout.addLayout(live_head)

        self.grid = VialActivityGrid()
        # SCROLLED SIDEWAYS RATHER THAN FORCING THE PANE WIDER. At a narrow pane the operator can
        # scroll to the far vials; the alternative was the picture shrinking to make room for them.
        grid_scroll = QScrollArea()
        grid_scroll.setWidget(self.grid)
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        grid_scroll.setFixedHeight(56)
        layout.addWidget(grid_scroll)

        rec_head = QHBoxLayout()
        title2 = QLabel("WRITTEN TO activity.csv")
        title2.setProperty("role", "grouptitle")
        rec_head.addWidget(title2)
        self.recorded_note = QLabel("no bins finished yet")
        self.recorded_note.setProperty("role", "note")
        rec_head.addWidget(self.recorded_note, 1)
        layout.addLayout(rec_head)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([label for label, _ in COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(False)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table.setMinimumHeight(120)
        layout.addWidget(self.table, 1)

    # -- live ------------------------------------------------------------------------------------
    def set_progress(self, payload: dict) -> None:
        """One throttled per-frame snapshot. Every figure was counted by the pipeline."""
        self.grid.set_activity(payload.get("vial_results") or {})
        face = payload.get("face") or "?"
        threshold = payload.get("pixel_threshold")
        bits = ["face %s" % face, "per frame, not yet binned - not in the file"]
        if threshold is not None:
            # THE THRESHOLD IS SHOWN BESIDE THE NUMBERS IT PRODUCED. It is the one setting that
            # decides what counts as motion at all, it is live-adjustable mid-run, and reading
            # these values without it is reading them without their units.
            bits.insert(1, "pixel threshold %.1f" % float(threshold))
        self.live_note.setText("   -   ".join(bits))

    # -- recorded --------------------------------------------------------------------------------
    def add_bin(self, payload: dict) -> None:
        """One completed bin: append its rows, exactly as they went to the file."""
        records = payload.get("records") or []
        if not records:
            return
        self._bins += 1
        self._rows_written += len(records)
        for record in records:
            self._append(record)
        self._trim()
        self.table.scrollToBottom()
        self.recorded_note.setText(
            "%d bin(s), %d row(s) written   -   showing the last %d"
            % (self._bins, self._rows_written, min(self._rows_written, MAX_ROWS)))

    def _append(self, record: Dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, (_label, field) in enumerate(COLUMNS):
            item = QTableWidgetItem(_fmt(record.get(field, "")))
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            # A vial the calibration marks as absent is dimmed rather than hidden: it still has a
            # row in the file, and a reader comparing screen to CSV has to find it in both.
            if not record.get("present", True):
                item.setForeground(QColor(theme.TEXT_FAINT))
            self.table.setItem(row, column, item)

    def _trim(self) -> None:
        while self.table.rowCount() > MAX_ROWS:
            self.table.removeRow(0)

    def clear(self) -> None:
        self.grid.clear()
        self.table.setRowCount(0)
        self._rows_written = 0
        self._bins = 0
        self.live_note.setText("per frame, not yet binned - not in the file")
        self.recorded_note.setText("no bins finished yet")

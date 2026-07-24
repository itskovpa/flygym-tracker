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

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

from flygym_tracker.gui import theme

#: The drum: two faces of sixteen.
VIALS_PER_FACE = 16
N_FACES = 2

# `COLUMNS`, `MAX_ROWS` and `_fmt` USED TO LIVE HERE, for the recorded-rows table. The table is
# gone (see `ResultsPanel`), and with it the only reason this module knew the CSV's column names.


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

        THE CELL SHOWS `active_fraction` (motion / lit area) AS A PERCENT, not the raw motion-pixel
        count. The raw count grows with a bigger hand-drawn ROI, so two vials at the same real
        activity would read differently and could not be compared at a glance -- which is the whole
        job of this strip. active_fraction is already per-area, so the cells ARE comparable across
        vials whose ROIs were drawn to different sizes.

        A vial the pipeline did not report is left BLANK rather than drawn as zero: "no reading"
        and "a reading of zero" are different facts, and drawing them the same way would be the
        display inventing a measurement. On this rig that distinction is load-bearing -- only one
        drum face is visible at a time, so the other face's sixteen SHOULD be blank.
        """
        for index, cell in enumerate(self.cells):
            # GLOBAL VIAL IDS ARE 1-BASED: `pipeline` builds them as `face_index * 16 + v.id` with
            # v.id running 1..16, so face A is 1..16 and face B is 17..32. Cell 0 is A1, which is
            # gvid 1.
            #
            # THE BUG THIS FIXES, seen on the rig: reading `vial_results[index]` shifted every
            # reading one cell to the right -- A1 was always blank, A2 showed A1's number, and
            # face A's vial 16 appeared in the cell labelled B1. On screen that read as "15 vials
            # on face A and 1 on face B being measured at the same time", which is impossible:
            # only one drum face is in front of the camera at a time.
            result = vial_results.get(index + 1)
            if result is None:
                cell.setText("")
                cell.setStyleSheet(self._style(theme.INK_2, theme.TEXT_FAINT))
                continue
            try:
                active_fraction = float(result[2])
            except (TypeError, IndexError, ValueError):
                active_fraction = 0.0
            # SHOWN AS A PERCENT OF THE VIAL'S LIT AREA that moved this frame -- the area-normalised
            # number, so cells are comparable across differently-sized ROIs. One decimal is plenty
            # for a live "is this vial reporting and is the threshold sane" glance; the exact,
            # binned value is in the file and the plots.
            percent = max(0.0, 100.0 * active_fraction)
            cell.setText("%.1f" % percent if percent < 99.95 else "100")
            weight = max(0.0, min(1.0, active_fraction / 0.1))
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
        # TIGHT, because this is now a band under the picture and every pixel it takes is one the
        # drum does not get. Measured: 8/6 margins at 6 spacing put this band's minimum height at
        # 104 px, against a picture that wants 280.
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(3)

        live_head = QHBoxLayout()
        title = QLabel("MEASURING NOW")
        title.setProperty("role", "grouptitle")
        live_head.addWidget(title)
        self.live_note = QLabel("per frame, not yet binned - not in the file")
        self.live_note.setProperty("role", "note")
        # IGNORED WIDTH, like the other two notes, and this was measured. Its text grows once a run
        # starts posting to it ("face A - pixel threshold 15.0 - per frame, not yet binned - not in
        # the file"), and a plain QLabel's minimum width is its whole sentence: the window's minimum
        # went to 1740 px on a 1440 px desktop the moment a run began. The layout tests never caught
        # it because they build the window but never push a progress payload through it.
        self.live_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        live_head.addWidget(self.live_note, 1)
        layout.addLayout(live_head)

        self.grid = VialActivityGrid()
        # SCROLLED SIDEWAYS RATHER THAN FORCING THE PANE WIDER. At a narrow pane the operator can
        # scroll to the far vials; the alternative was the picture shrinking to make room for them.
        grid_scroll = QScrollArea()
        grid_scroll.setWidget(self.grid)
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # VERTICAL SCROLLBAR AS-NEEDED, NOT ALWAYS-OFF, AND A FEW PIXELS MORE ROOM. Two 20 px rows
        # fit 56, but when the pane is dragged narrow a HORIZONTAL scrollbar appears and eats ~15 px
        # of that fixed height -- enough to clip the second vial row (face B) with no way to reach
        # it. 64 px leaves room for the horizontal bar, and as-needed vertical scrolling is the
        # backstop for a larger UI font. Still fixed and short, so it never steals from the picture.
        grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        grid_scroll.setFixedHeight(64)
        layout.addWidget(grid_scroll)

        # THE TABLE IS GONE. It listed every row as it was written -- 32 rows per bin, scrolling
        # past faster than anyone reads -- and the rig owner's verdict was that it "should just go
        # as a graph". They were right: a wall of numbers answers "is a row being written", which
        # the counter below already answers in one line, while the question actually being asked of
        # a running experiment is "is this vial's number going up or down", which is a shape.
        # Everything the table showed is in the plot docks and in the file.
        # WHAT THE TABLE IS REPLACED BY, at this end: one line saying how much has been written.
        # That is the only question the table answered that a graph does not -- "is the file
        # actually growing" -- and it takes a line rather than a scrolling wall.
        self.recorded_note = QLabel("no bins finished yet")
        self.recorded_note.setProperty("role", "note")
        # NOT WRAPPED, now that this panel is a short band UNDER the picture rather than a tall
        # column beside it. A wrapping label's minimum height grows with its text, and these two
        # notes between them were demanding ~150 px of minimum -- enough that on a short screen
        # Qt gave the band its minimum and squeezed the PICTURE, which is the one thing the layout
        # exists to protect. One line each, and the full text stays in the tooltip.
        self.recorded_note.setWordWrap(False)
        self.recorded_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.recorded_note)

        # THE RECORDING, IF THERE IS ONE. Shown HERE rather than only beside its tick box because
        # that tick box lives in a section that collapses -- and a recorder quietly dropping frames
        # to a slow or filling disk looks exactly like one that is keeping up unless it is asked.
        # The answer is only worth anything while there is still time to lower the rate.
        self.recording_note = QLabel("")
        self.recording_note.setProperty("role", "note")
        self.recording_note.setWordWrap(False)
        self.recording_note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.recording_note.setVisible(False)
        layout.addWidget(self.recording_note)
        layout.addStretch(1)

    # -- live ------------------------------------------------------------------------------------
    def set_progress(self, payload: dict) -> None:
        """One throttled per-frame snapshot. Every figure was counted by the pipeline."""
        # On a ROTATING frame the drum is mid-flip: the pipeline passes rotation placeholders
        # (motion 0) only so the bin can count rotating frames. Drawing those as "0" makes the grid
        # invent a reading of zero where there is none -- the very thing its own blank-vs-zero rule
        # forbids, and the zeros the operator sees "on the flips". Blank the grid during a flip; a
        # genuine still-fly zero on a stationary frame still shows.
        state = str(payload.get("state") or "")
        vial_results = {} if "ROTATING" in state else (payload.get("vial_results") or {})
        self.grid.set_activity(vial_results)
        face = payload.get("face") or "?"
        threshold = payload.get("pixel_threshold")
        bits = ["face %s" % face, "active fraction: % of ROI area moving",
                "per frame, not yet binned - not in the file"]
        if threshold is not None:
            # THE THRESHOLD IS SHOWN BESIDE THE NUMBERS IT PRODUCED. It is the one setting that
            # decides what counts as motion at all, it is live-adjustable mid-run, and reading
            # these values without it is reading them without their units.
            bits.insert(1, "pixel threshold %.1f" % float(threshold))
        text = "   -   ".join(bits)
        self.live_note.setText(text)
        self.live_note.setToolTip(text)
        self.set_recording(payload.get("video"))

    def set_recording(self, stats: Optional[dict]) -> None:
        """One line about the video, or nothing at all when no video was asked for.

        DROPS ARE NAMED AS DROPS, and separately from the frames deliberately skipped. They are
        different facts: skipped frames are the sampling rate the operator chose, dropped ones are
        the encoder failing to keep up with it. Rolling them into one "frames not recorded" figure
        would hide a disk problem inside a setting.
        """
        if not stats:
            self.recording_note.setVisible(False)
            return
        self.recording_note.setVisible(True)
        error = stats.get("error")
        if error:
            self.recording_note.setText("VIDEO STOPPED: %s   -   the measurement is unaffected"
                                        % error)
            return
        written = int(stats.get("frames_written") or 0)
        dropped = int(stats.get("frames_dropped") or 0)
        megabytes = float(stats.get("bytes") or 0) / (1024.0 * 1024.0)
        bits = ["recording: %d frame(s)" % written, "%.0f MB" % megabytes]
        # A MEASURED PROJECTION, not an estimate from the settings. How many bytes a frame costs
        # depends entirely on what is in it -- a still drum compresses to nothing, thirty-two vials
        # of moving flies do not -- so the only honest way to say what a three-day run will cost is
        # to divide what THIS run has already written. It is worth saying because the answer is
        # frequently hundreds of gigabytes, and hour 50 is a bad time to find that out.
        fps = float(stats.get("fps") or 0.0)
        if written > 20 and fps > 0 and megabytes > 0:
            bits.append("about %.0f GB/day at this rate" % (
                megabytes / written * fps * 86400.0 / 1024.0))
        if dropped:
            # NOT FOLDED INTO A PERCENTAGE. "1.2% dropped" reads as a rounding error; a count that
            # keeps climbing reads as what it is.
            bits.append("%d frame(s) DROPPED - the disk or the codec is behind the camera" % dropped)
        text = "   -   ".join(bits)
        self.recording_note.setText(text)
        self.recording_note.setToolTip(text)

    # -- recorded --------------------------------------------------------------------------------
    def add_bin(self, payload: dict) -> None:
        """One completed bin. COUNTED, not listed -- the rows themselves are in the plot docks."""
        records = payload.get("records") or []
        if not records:
            return
        self._bins += 1
        self._rows_written += len(records)
        text = "%d bin(s), %d row(s) written to activity.csv" % (self._bins, self._rows_written)
        self.recorded_note.setText(text)
        self.recorded_note.setToolTip(text)      # the line no longer wraps; the tooltip is the rest

    def clear(self) -> None:
        self.grid.clear()
        self._rows_written = 0
        self._bins = 0
        self.live_note.setText("per frame, not yet binned - not in the file")
        self.recorded_note.setText("no bins finished yet")
        self.recording_note.setText("")
        self.recording_note.setVisible(False)

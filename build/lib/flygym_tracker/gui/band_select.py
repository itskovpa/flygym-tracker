"""Let the operator draw WHERE THE MARKER BAND IS, instead of the detector inferring it every frame.

WHY THIS EXISTS, and it is a rig owner's call rather than a programmer's. `MarkerBandDetector` finds
the band per frame from the row-wise lit-pixel profile inside `params.search_frac` (20%-80% of the
frame height). It never knows where the LED slots physically are -- it re-derives it from brightness
on every single frame. That is genuinely robust to exposure changes and to the drum's 1.35 degree
tilt, and it has one failure mode that matters:

    MEASURED ON THIS RIG, backlight unplugged: the two strips stopped being two runs. The row
    profile collapsed to a SINGLE run of 141 rows, which was rejected for exceeding
    `max_strip_h=110`, so the band came back "not found" on 661 of 661 stationary frames and only
    one drum face was ever learned.

Nothing in that chain is a bug. There was nothing to see. But it shows what the automatic search
costs: the band's LOCATION -- the one thing about this rig that does not change between experiments,
because it is bolted to the rotation axis -- was being re-guessed from the very signal that had
degraded. Drawing it once removes the guess.

WHAT IS DRAWN IS TWO ROWS, NOT A RECTANGLE. The band is a horizontal slice: `band_rows=(r0, r1)`
replaces the automatic search window and nothing else. Columns are not part of it -- the x extent is
still derived per frame from the lit fraction (`_x_window`), which is what keeps the profiles
comparable as the drum shifts slightly. Offering a full rectangle would imply a control the detector
does not have.

THE STRIPS ARE STILL FOUND INSIDE THE DRAWN ROWS. This does not pin the two strips to fixed rows --
that would break under the tilt, which moves them by a few pixels between faces. It pins the WINDOW
they are searched in, so `min_strip_h` / `max_strip_h` / `max_strip_gap` still do their work on a
region that contains the band and nothing else.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen

from flygym_tracker.gui import theme

#: The drawn band. Amber, which in this app always means "the operator is imposing this value".
COLOR_BAND = QColor(theme.IMPOSED)
#: What the detector currently finds inside it, drawn so the two can be compared at a glance.
COLOR_STRIPS = QColor(theme.DEFAULT_GREEN)
#: The rows the automatic search would have used, for reference.
COLOR_AUTO = QColor(theme.TEXT_FAINT)

#: A drag shorter than this is a click, not a band. Prevents a stray click collapsing the
#: selection to a zero-height band that would then find nothing at all.
MIN_BAND_ROWS = 8


class BandOverlay:
    """Paints the drawn band, the automatic window it replaces, and the strips found inside it."""

    def __init__(self, session: "BandSelectSession") -> None:
        self.session = session

    def paint(self, painter, view) -> None:
        rect = view.image_rect()
        if rect.width() <= 0:
            return
        width, height = view.frame_size
        painter.setRenderHint(painter.RenderHint.Antialiasing, False)

        # The automatic window this replaces -- so the operator can see what they are overriding.
        auto = self.session.auto_rows
        if auto is not None and not self.session.has_band:
            self._band(painter, view, auto, COLOR_AUTO, "automatic search window", fill=False)

        if self.session.has_band:
            rows = self.session.rows
            self._band(painter, view, rows, COLOR_BAND,
                       "marker band  rows %d-%d" % rows, fill=True)
            for index, strip in enumerate(self.session.strips):
                self._band(painter, view, strip, COLOR_STRIPS,
                           "strip %d  rows %d-%d" % (index + 1, strip[0], strip[1]), fill=False)

    def _band(self, painter, view, rows, colour, label, fill: bool) -> None:
        rect = view.image_rect()
        top = view.to_widget(0, rows[0]).y()
        bottom = view.to_widget(0, rows[1]).y()
        band = QRectF(rect.x(), top, rect.width(), max(1.0, bottom - top))
        if fill:
            painter.setBrush(QBrush(QColor(colour.red(), colour.green(), colour.blue(), 36)))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(colour, 2)
        if not fill:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(band)
        font = QFont(painter.font())
        font.setPointSize(theme.PT_SMALL)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
        painter.drawText(band.adjusted(6, 2, 0, 0), int(Qt.AlignmentFlag.AlignLeft), label)
        painter.setPen(QPen(colour, 1))
        painter.drawText(band.adjusted(6, 2, 0, 0), int(Qt.AlignmentFlag.AlignLeft), label)


class BandSelectSession(QObject):
    """One marker-band selection: drag two rows, see what the detector finds, save it."""

    changed = Signal()
    finished = Signal(dict)

    def __init__(self, *, out_dir: str, frame_height: int = 0,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        from flygym_tracker.marker_band import MarkerBandDetector

        self.out_dir = out_dir
        self.overlay = BandOverlay(self)
        self._detector = MarkerBandDetector()
        self._anchor: Optional[float] = None
        self._rows: Optional[Tuple[int, int]] = None
        self.last_image = None
        self.frame_height = int(frame_height)
        self.message = ""

    # -- what is drawn --------------------------------------------------------------------------
    @property
    def has_band(self) -> bool:
        return self._rows is not None

    @property
    def rows(self) -> Tuple[int, int]:
        return self._rows or (0, 0)

    @property
    def auto_rows(self) -> Optional[Tuple[int, int]]:
        """The window the automatic search would use on this frame -- what is being replaced."""
        if self.last_image is None:
            return None
        try:
            return self._detector._search_window(self.last_image)
        except Exception:
            return None

    @property
    def strips(self):
        """The strips the detector finds INSIDE the drawn rows. The whole point of the preview:
        a band that contains no two strips is a band that will identify nothing, and the operator
        can see that before saving rather than during an experiment."""
        if self.last_image is None or not self.has_band:
            return []
        try:
            self._detector.band_rows = self._rows
            return list(self._detector.find_strips(self.last_image))
        except Exception:
            return []
        finally:
            self._detector.band_rows = None

    # -- input ----------------------------------------------------------------------------------
    def on_press(self, _x: float, y: float) -> None:
        self._anchor = float(y)
        self._rows = None
        self.changed.emit()

    def on_drag(self, _x: float, y: float) -> None:
        if self._anchor is None:
            return
        self._set_rows(self._anchor, y)

    def on_release(self, _x: float, y: float) -> None:
        if self._anchor is None:
            return
        self._set_rows(self._anchor, y)
        self._anchor = None
        if self.has_band and len(self.strips) < 2:
            # SAID NOW, NOT AT SAVE TIME. A band with fewer than two strips in it cannot identify
            # anything, and the operator is looking straight at the picture that shows why.
            self.message = ("only %d lit strip(s) found in these rows - the band needs two. "
                            "Drag a band that covers BOTH bright slots." % len(self.strips))
        elif self.has_band:
            self.message = "two strips found - this band will work"
        self.changed.emit()

    def _set_rows(self, y0: float, y1: float) -> None:
        top, bottom = sorted((int(round(y0)), int(round(y1))))
        if self.frame_height:
            top = max(0, min(top, self.frame_height - 1))
            bottom = max(0, min(bottom, self.frame_height - 1))
        if bottom - top < MIN_BAND_ROWS:
            self._rows = None
            self.message = ""
        else:
            self._rows = (top, bottom)
            self.message = ""
        self.changed.emit()

    def on_frame(self, image) -> None:
        self.last_image = image
        if not self.frame_height:
            self.frame_height = int(image.shape[0])

    def clear(self) -> None:
        self._rows = None
        self._anchor = None
        self.message = ""
        self.changed.emit()

    # -- the end ---------------------------------------------------------------------------------
    def status(self) -> str:
        if not self.has_band:
            return ("drag down the picture across BOTH bright LED slots to mark the band "
                    "(the dashed box is what the software guesses today)")
        head = "band rows %d-%d   -   %d strip(s) found inside it" % (
            self.rows[0], self.rows[1], len(self.strips))
        return "%s   -   %s" % (head, self.message) if self.message else head

    def cancel(self) -> None:
        self.finished.emit({"saved": False, "message": "marker band not changed"})

    def save(self) -> None:
        """Write the band into the bundle. Never raises at the caller."""
        from flygym_tracker.calibration import attach_band_rows

        if not self.has_band:
            self.finished.emit({"saved": False, "message": "no band was drawn - nothing saved"})
            return
        strips = self.strips
        try:
            written = attach_band_rows(self.out_dir, self.rows)
        except Exception as exc:
            self.finished.emit({"saved": False, "rows": self.rows,
                                "message": "the marker band could not be saved: %s" % exc})
            return
        note = ("" if len(strips) >= 2 else
                "  WARNING: only %d lit strip(s) were found in it, so faces will NOT be "
                "identifiable until this is redrawn over both slots." % len(strips))
        self.finished.emit({
            "saved": True, "rows": self.rows, "faces": written, "strips": strips,
            "message": "marker band rows %d-%d saved to %s for face(s) %s%s"
                       % (self.rows[0], self.rows[1], self.out_dir, ", ".join(written), note)})

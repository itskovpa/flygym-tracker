"""Drawing the vial positions on the picture in the window, instead of in a cv2 window.

WHAT MOVED AND WHAT DID NOT. The KEYMAP and all the geometry bookkeeping did not move: this drives
`live_vial_selector.SelectorState` and `handle_key`, which are pure, already tested, and already
the thing the operator has learnt at this rig. What moved is the surface -- the clicking, the
drawing and the instructions -- from an OpenCV window in a child process into `PreviewWidget`.

THREE PROBLEMS THAT WENT AWAY WITH THE cv2 WINDOW, none of which had a fix inside it:

  1. THE FRAME COULD RUN OFF THE SCREEN. A `WINDOW_AUTOSIZE` window is laid out at the frame's own
     pixel size, so on the rig laptop (2880x1800 at 200%, i.e. a 1440x900 desktop) the bottom ~130
     rows of a 1280x1024 frame sat below the screen edge -- not visible and impossible to click,
     which is exactly the part of the tube the operator most needs to enclose.
     `screen_view_limit` exists to measure around that. A letterboxed widget inside a layout
     cannot do it at all: it is handed a rectangle and fits the frame into it, at any window size.
  2. THE PANEL COMPETED WITH THE PICTURE. Instructions had to be painted into the same canvas, so
     they either covered the tubes or ate 380 px of width from them. They are widgets beside the
     picture now and cost the picture nothing.
  3. THE PROCESS BOUNDARY. The selector ran as a child process, so it could not see the settings
     pane, could not use the camera this window already had open, and reported what it had done by
     printing to a console nobody was looking at.

THE FRAME UNDERNEATH IS NEVER MODIFIED. Polygons are painted by this overlay onto the widget, over
the drawn frame, at paint time -- the image the operator is clicking on is the image the camera
sent, at full resolution, with the mapping between click and pixel exact in both directions.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPolygonF

from flygym_tracker.gui import theme

#: Finished vials. Green means "this is settled" here exactly as it does on a settings row.
COLOR_DONE = QColor(theme.DEFAULT_GREEN)
#: The vial being clicked right now.
COLOR_CURRENT = QColor(theme.IMPOSED)
#: The first vertex, drawn larger -- it is the one the closing edge runs back to.
COLOR_FIRST = QColor(theme.FOCUS)
#: The border that marks a held picture. Impossible to miss on purpose: an operator who does not
#: notice the picture is frozen will draw the second half of a face onto a stale frame.
COLOR_FROZEN = QColor(theme.WARN)


class VialOverlay:
    """Paints a `SelectorState` over a `PreviewWidget`. Owns nothing and mutates nothing."""

    def __init__(self, state) -> None:
        self.state = state

    def paint(self, painter, view) -> None:
        painter.setRenderHint(painter.RenderHint.Antialiasing, True)
        self._paint_done(painter, view)
        self._paint_current(painter, view)
        if self.state.frozen:
            rect = view.image_rect().adjusted(1, 1, -2, -2)
            painter.setPen(QPen(COLOR_FROZEN, 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

    def _paint_done(self, painter, view) -> None:
        font = QFont(painter.font())
        font.setPointSize(theme.PT_BASE)
        font.setBold(True)
        painter.setFont(font)
        for index, polygon in enumerate(self.state.polygons):
            points = [view.to_widget(x, y) for x, y in polygon]
            if len(points) < 2:
                continue
            painter.setPen(QPen(COLOR_DONE, 2))
            # A 12% wash inside the outline. The vial NUMBER is what the operator checks against
            # the rig, and a bare outline over a grey IR tube is genuinely hard to count along a
            # row of sixteen; a faint fill makes "which ones have I done" a glance.
            painter.setBrush(QBrush(QColor(COLOR_DONE.red(), COLOR_DONE.green(),
                                           COLOR_DONE.blue(), 30)))
            painter.drawPolygon(QPolygonF(points))
            centre = QPointF(sum(p.x() for p in points) / len(points),
                             sum(p.y() for p in points) / len(points))
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            painter.drawText(centre, str(index + 1))
            painter.setPen(QPen(COLOR_DONE, 1))
            painter.drawText(centre, str(index + 1))

            # THE CORNERS ARE DRAWN AS HANDLES, because they are draggable and a control that does
            # not look like one is a control nobody tries. Only while editing: on a fresh session
            # every finished vial is exactly where it was just clicked, and sixteen rows of dots
            # over the tubes would be noise competing with the vial being drawn.
            if getattr(self.state, "editing", False):
                for point in points:
                    painter.setPen(QPen(QColor(0, 0, 0, 180), 2))
                    painter.setBrush(QBrush(COLOR_DONE))
                    painter.drawEllipse(point, 3.5, 3.5)

    def _paint_current(self, painter, view) -> None:
        current = self.state.current
        if not current:
            return
        points = [view.to_widget(x, y) for x, y in current]
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if len(points) >= 2:
            painter.setPen(QPen(COLOR_CURRENT, 2))
            painter.drawPolyline(QPolygonF(points))
            # The closing edge, thin and dashed: it shows the SHAPE that ENTER would store without
            # drawing it as though the polygon were already finished.
            pen = QPen(COLOR_CURRENT, 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(points[-1], points[0])
        for index, point in enumerate(points):
            colour = COLOR_FIRST if index == 0 else COLOR_CURRENT
            painter.setPen(QPen(colour, 1))
            painter.setBrush(QBrush(colour))
            painter.drawEllipse(point, 5.0 if index == 0 else 4.0, 5.0 if index == 0 else 4.0)


class VialDrawSession(QObject):
    """One drawing session: the state, the clicks, the keys, and the save at the end.

    NOTHING IS WRITTEN UNTIL THE OPERATOR IS DONE, and then the polygons are written BEFORE
    anything else is attempted -- the same order `load_or_select_vials` uses, and for the same
    reason: a face-learning step or a mask failure must never be able to lose a clicking session.
    """

    #: The state changed and the picture needs repainting (also carries the status line).
    changed = Signal()
    #: The operator finished. ``{"saved": bool, "n_vials": int, "out_dir": str, "message": str}``.
    finished = Signal(dict)

    def __init__(self, *, n_vials: int = 16, face: str = "A", source_label: str = "",
                 out_dir: str = "calib_faces", faces: Sequence[str] = ("A", "B"),
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        from flygym_tracker.live_vial_selector import SelectorState

        self.state = SelectorState(n_vials=n_vials, face=face, source_label=source_label)
        self.overlay = VialOverlay(self.state)
        self.out_dir = out_dir
        self.faces: List[str] = list(faces)
        #: The most recent frame that was actually shown. The illumination mask and the overlay PNG
        #: are built from THIS image, so they match the polygons drawn on top of it.
        self.last_image = None
        #: ``(vial, vertex)`` currently being dragged, or None. See `on_press`.
        self._drag = None

    # -- input ---------------------------------------------------------------------------------
    #: How near (in IMAGE pixels) a press must be to a saved corner to grab it instead of placing
    #: a new one. Generous, because the operator is aiming at a 4 px dot on a scaled-down picture:
    #: at the rig's 1280x1024 in a ~700 px wide pane the image is shown at ~0.55x, so 18 image
    #: pixels is ~10 screen pixels -- about the size of the dot they can see.
    GRAB_RADIUS = 18.0

    def on_press(self, x: float, y: float) -> None:
        """Mouse down. Grabs a saved corner if there is one under the pointer.

        RUNS BEFORE `on_click`, which is why `preview.PreviewWidget` emits `pressed` first: the
        decision "is this a drag of an existing corner or a new corner being placed" has to be made
        before anything is placed, or every attempt to adjust a vial would add a vertex to it.
        """
        self._drag = self.state.find_vertex(x, y, self.GRAB_RADIUS) if not self.state.current \
            else None
        if self._drag is not None:
            self.changed.emit()

    def on_drag(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        self.state.move_vertex(self._drag[0], self._drag[1], x, y)
        self.changed.emit()

    def on_release(self, x: float, y: float) -> None:
        if self._drag is None:
            return
        self.state.move_vertex(self._drag[0], self._drag[1], x, y)
        self._drag = None
        self.changed.emit()

    def on_click(self, x: float, y: float) -> None:
        """A click on the picture, in image pixels. Places a corner of the vial being drawn.

        Suppressed while a saved corner is being dragged -- see `on_press`.
        """
        if self.state.done or self._drag is not None:
            return
        self.state.add_vertex(x, y)
        self.changed.emit()

    def on_key(self, name: str) -> None:
        """One keystroke, by `handle_key` name. The keymap is that module's, not a second one."""
        from flygym_tracker.live_vial_selector import handle_key

        if handle_key(self.state, name) == "done":
            self.changed.emit()
            self.finish()
            return
        self.changed.emit()

    def on_frame(self, image) -> None:
        """A newly delivered frame. Ignored while the picture is frozen, which is the whole point
        of freezing: the drum turns, and clicking a moving tube is hopeless."""
        if not self.state.frozen:
            self.last_image = image

    # -- the buttons (the same actions as the keys, for people who do not know the keys) --------
    def finish_vial(self) -> None:
        self.state.finish_vial()
        self.changed.emit()
        if self.state.done:
            self.finish()

    def undo_vertex(self) -> None:
        self.state.undo_vertex()
        self.changed.emit()

    def undo_vial(self) -> None:
        self.state.undo_vial()
        self.changed.emit()

    def clear_vial(self) -> None:
        self.state.clear()
        self.changed.emit()

    def toggle_freeze(self) -> bool:
        frozen = self.state.toggle_freeze()
        self.changed.emit()
        return frozen

    def load(self, polygons) -> int:
        """Show saved vial positions, ready to edit. Returns how many were loaded."""
        n = self.state.load(polygons)
        self.changed.emit()
        return n

    def start_over(self) -> None:
        """Throw away every vial and draw from nothing."""
        self.state.start_over()
        self._drag = None
        self.changed.emit()

    def cancel(self) -> None:
        """Abandon the session without writing anything."""
        self.finished.emit({"saved": False, "n_vials": len(self.state.polygons),
                            "out_dir": self.out_dir,
                            "message": "drawing cancelled - nothing was saved"})

    # -- the end -------------------------------------------------------------------------------
    def status(self) -> str:
        """The one line under the picture: where the session is, and anything it just said."""
        if self.state.editing and not self.state.current:
            head = ("%d vial(s) loaded - drag a corner to adjust, or click to start vial %d"
                    % (len(self.state.polygons), self.state.vial_number))
        else:
            head = "vial %d of %d   -   %d point(s) clicked" % (
                min(self.state.vial_number, self.state.n_vials), self.state.n_vials,
                len(self.state.current))
        if self.state.frozen:
            head += "   -   PICTURE HELD"
        return "%s   -   %s" % (head, self.state.message) if self.state.message else head

    def finish(self) -> None:
        """Save what was drawn and report it. Never raises at the caller.

        A failure here is reported as a sentence with the polygon count in it, because the thing
        that has just gone wrong happened AFTER somebody spent several minutes clicking and the
        one thing they need to know is whether that work survived.
        """
        polygons = self.state.polygons
        if not polygons:
            self.finished.emit({"saved": False, "n_vials": 0, "out_dir": self.out_dir,
                                "message": "no vials were drawn - nothing saved"})
            return
        if self.last_image is None:
            self.finished.emit({
                "saved": False, "n_vials": len(polygons), "out_dir": self.out_dir,
                "message": "no frame was ever received, so the %d drawn vial(s) could not be "
                           "saved - the mask and overlay are built from the picture they were "
                           "drawn on" % len(polygons)})
            return
        try:
            calib = self._save(polygons)
        except Exception as exc:
            self.finished.emit({
                "saved": False, "n_vials": len(polygons), "out_dir": self.out_dir,
                "message": "the %d drawn vial(s) could NOT be saved: %s" % (len(polygons), exc)})
            return
        self.finished.emit({
            "saved": True, "n_vials": len(polygons), "out_dir": self.out_dir,
            "faces": sorted(calib.faces), "calibration": calib,
            "message": "saved %d vial(s) on face(s) %s to %s"
                       % (len(polygons), ", ".join(sorted(calib.faces)), self.out_dir)})

    def _save(self, polygons):
        """Write the bundle. The same two calls `load_or_select_vials` makes, in the same order."""
        from flygym_tracker.calibration import (build_two_face_calibration_from_polygons,
                                                save_calibration)

        frame = self.last_image
        height, width = frame.shape[:2]
        calib, masks, overlays = build_two_face_calibration_from_polygons(
            polygons, frame, (width, height), faces=self.faces)
        save_calibration(calib, masks, self.out_dir, overlay=overlays)
        # Saved with RELATIVE mask paths so the bundle stays movable, handed back RESOLVED so it
        # can go straight to the pipeline -- exactly what `load_calibration` returns.
        calib.resolve_mask_paths(os.path.abspath(self.out_dir))
        return calib

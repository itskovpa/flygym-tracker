"""The vials the run is measuring, drawn on the run's own picture.

WHY THIS IS NOT OPTIONAL DECORATION. The polygons ARE the measurement: `activity.csv` reports one
row per vial per bin, and which pixels went into each row is decided entirely by the shape that was
drawn on the rig weeks or minutes earlier. Watching a run without them, the operator sees a picture
of a drum and a table of numbers with no way to tell that vial 7's outline is sitting half on the
tube next door -- which produces a perfectly healthy-looking column of plausible values that belong
to the wrong fly.

The same shapes are on screen while they are DRAWN (`vial_draw.VialOverlay`) and while they are
MEASURED (here), so a mistake in one is visible in the other.

THIS OVERLAY IS READ-ONLY, and deliberately a different class from the drawing one. The drawing
overlay paints a `SelectorState` -- a live, mutable object with a vial in progress, grab handles and
a freeze border, none of which mean anything during a run. Reusing it would put an editing surface's
affordances on a picture nobody can edit.

TINTED BY WHAT EACH VIAL IS REPORTING, which is the point of drawing them here rather than just
listing numbers beside the picture: a vial whose outline has slipped off its tube reads near zero
while its neighbours do not, and that is visible in one glance at the drum instead of by scanning a
column of a table. The tint is the SAME per-frame value the results pane calls "not yet binned",
and it is not labelled with a number here -- a number over a tube would be read as that tube's
result, and the binned value is the result.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPolygonF

from flygym_tracker.gui import theme

#: The outline. Steel blue rather than the drawing overlay's green: green means "settled, you drew
#: this" on an editing surface, and during a run nothing is being settled.
COLOR_OUTLINE = QColor(theme.FOCUS)
#: A vial the calibration marks as absent -- it still has rows in the file, so it is still drawn.
COLOR_ABSENT = QColor(theme.TEXT_FAINT)

#: The drum: sixteen vials per face. Global ids are `face_index * 16 + local`.
VIALS_PER_FACE = 16

#: Motion (px) at which the tint saturates. Matches `results_panel.VialActivityGrid` so the picture
#: and the grid agree about what "bright" means.
FULL_SCALE = 400.0


class RunVialOverlay:
    """Paints saved vial polygons over the run's picture, tinted by live activity."""

    def __init__(self, polygons: Sequence[Sequence[Sequence[float]]],
                 present: Optional[Sequence[bool]] = None) -> None:
        self.polygons: List[List[List[float]]] = [[list(p) for p in poly] for poly in polygons]
        self.present = list(present) if present is not None else [True] * len(self.polygons)
        #: ``{vial_index: motion_px}`` from the latest progress snapshot. Empty until one arrives,
        #: and an ABSENT key is drawn untinted rather than as zero -- "no reading" and "a reading of
        #: zero" are different facts, and only one drum face is visible at a time.
        self.activity: Dict[int, float] = {}

    def set_activity(self, vial_results: dict) -> None:
        """Adopt the latest per-frame per-vial results, keyed by GLOBAL vial id.

        The polygons are one face's worth (16); the results are global (0..31, face A then B). The
        modulo is what maps the visible face's ids onto the shapes -- both faces share identical
        coordinates on this rig, which is exactly why one set of polygons can serve either.
        """
        activity: Dict[int, float] = {}
        for gvid, result in (vial_results or {}).items():
            try:
                # 1-BASED GLOBAL IDS (`face_index * 16 + v.id`, v.id 1..16), so face A is 1..16 and
                # face B is 17..32, and the -1 is what makes gvid 1 the FIRST polygon. Without it
                # gvid 16 wrapped to polygon 0 and every vial's tint sat one tube to the right.
                #
                # MODULO 16, NOT len(polygons): sixteen is the drum's geometry, while the polygon
                # count is however many vials this bundle marks PRESENT. On a bundle with a vial
                # missing the two differ, and dividing by the wrong one would silently re-map every
                # tint on that face.
                index = (int(gvid) - 1) % VIALS_PER_FACE
                if index >= len(self.polygons):
                    continue
                activity[index] = float(result[0])
            except (TypeError, ValueError, IndexError):
                continue
        self.activity = activity

    def clear_activity(self) -> None:
        self.activity = {}

    def paint(self, painter, view) -> None:
        if not self.polygons:
            return
        painter.setRenderHint(painter.RenderHint.Antialiasing, True)
        font = QFont(painter.font())
        font.setPointSize(theme.PT_TINY)
        painter.setFont(font)
        for index, polygon in enumerate(self.polygons):
            points = [view.to_widget(x, y) for x, y in polygon]
            if len(points) < 3:
                continue
            present = self.present[index] if index < len(self.present) else True
            colour = COLOR_OUTLINE if present else COLOR_ABSENT
            motion = self.activity.get(index)
            if motion is None:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                weight = max(0.0, min(1.0, motion / FULL_SCALE))
                painter.setBrush(QBrush(QColor(colour.red(), colour.green(), colour.blue(),
                                               int(18 + 90 * weight))))
            pen = QPen(colour, 2 if present else 1)
            if not present:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawPolygon(QPolygonF(points))

            centre = QPointF(sum(p.x() for p in points) / len(points),
                             sum(p.y() for p in points) / len(points))
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            painter.drawText(centre, str(index + 1))
            painter.setPen(QPen(colour, 1))
            painter.drawText(centre, str(index + 1))


def overlay_from_calibration(calib) -> Optional[RunVialOverlay]:
    """Build the run overlay from a loaded bundle, or None if it carries no usable shapes.

    Takes ONE face's vials: both faces share identical polygon coordinates on this rig (that is the
    whole premise of `build_two_face_calibration_from_polygons`), so drawing both would draw every
    shape twice in the same place.

    Never raises. A missing or odd bundle costs the overlay, not the run.
    """
    try:
        faces = getattr(calib, "faces", None) or {}
        if not faces:
            return None
        name = "A" if "A" in faces else sorted(faces)[0]
        polygons, present = [], []
        for vial in faces[name].vials:
            shape = getattr(vial, "polygon", None) or getattr(vial, "quad", None)
            if shape is None:
                continue
            polygons.append([[float(x), float(y)] for x, y in shape])
            present.append(bool(getattr(vial, "present", True)))
        return RunVialOverlay(polygons, present) if polygons else None
    except Exception:
        return None

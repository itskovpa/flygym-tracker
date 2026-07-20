"""Fly trajectories drawn on the run's picture, accumulating until the operator clears them.

WHAT IT IS FOR. `behaviour.csv` says a vial's median track length was 39 px; it cannot say whether
that came from flies walking or from the detector following a tube edge. The trajectories can. This
is the surface that answers "is the tracker finding real animals", and it is the one thing that
makes every behavioural number in this program checkable rather than merely plausible.

ACCUMULATED UNTIL CLEAR, which is the rig owner's call and NOT the same rule the analysis uses.
A `VialTracker` lives for one dwell -- across a rotation the flies are shaken and every identity is
lost, so linking through it would be fiction, and `behaviour.csv` therefore resets at every flip.
The PICTURE has no such constraint: the operator wants to watch paths build up over a whole
session. So the analysis resets and the drawing does not, and the two disagree on purpose.

    live      the current dwell's fragments, replaced on every update
    frozen    every earlier dwell's fragments, kept until `clear()`

The dwell number from the pipeline is what separates them. Without it the live fragments -- which
GROW point by point as a fly walks -- would either be appended over and over (one path drawn fifty
times) or replaced wholesale at each rotation (nothing ever accumulating).

ONE COLOUR PER FACE, all sixteen vials of a face sharing it, as asked. It is the same rule the
plot grids use, so a colour means the same thing wherever it appears in this app -- and on a drum
that flips every couple of seconds it is the only way to see at a glance which paths came from
which side.

OLDER PATHS FADE. A three-day session would otherwise end as a solid block of colour with no
information in it. The fade is by AGE, not by importance, and nothing is ever silently discarded
below the cap -- see `MAX_FROZEN_PATHS`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPen, QPolygonF

from flygym_tracker.gui import theme

Path = List[Tuple[float, float]]

#: One colour per drum face, shared by all sixteen of its vials. Matches `plot_dock.FACE_COLORS`.
FACE_COLORS = {"A": QColor(theme.FOCUS), "B": QColor(theme.DEFAULT_GREEN)}
FALLBACK_COLOR = QColor(theme.TEXT_DIM)

#: How many frozen paths are kept. A three-day session at ~2 s dwells produces tens of thousands;
#: past a few thousand the picture is a solid block and the painting cost is real. The OLDEST are
#: dropped, and `dropped` counts them, so the display never quietly claims to show everything.
MAX_FROZEN_PATHS = 4000

#: Opacity of the oldest kept path versus the newest. Fading by age is what keeps a long session
#: readable; it is a drawing choice and carries no claim about the data.
OLDEST_ALPHA = 60
NEWEST_ALPHA = 230


class TrackOverlay:
    """Accumulated fly trajectories, per face, drawn over the run's frames."""

    def __init__(self) -> None:
        #: ``[(face, path)]`` from dwells that have ended.
        self._frozen: List[Tuple[str, Path]] = []
        #: ``{(face, vial): path}`` for the dwell being tracked right now.
        self._live: Dict[Tuple[str, int], List[Path]] = {}
        self._dwell: Optional[int] = None
        self.dropped = 0
        self.enabled = True

    # -- state ------------------------------------------------------------------------------------
    def clear(self) -> None:
        """Throw every path away and start accumulating again from nothing.

        THE LIVE FRAGMENTS GO TOO. A Clear that left the current dwell on screen would look like
        it had half worked, and the operator presses this precisely when they want a clean slate
        to judge the next few minutes against.
        """
        self._frozen = []
        self._live = {}
        self.dropped = 0

    @property
    def n_paths(self) -> int:
        return len(self._frozen) + sum(len(paths) for paths in self._live.values())

    def update(self, payload: Optional[dict]) -> None:
        """Adopt one `pipeline.fly_tracks()` snapshot.

        A CHANGE OF DWELL FREEZES what was live. That is the whole mechanism: within a dwell the
        fragments grow, so they are REPLACED on every update; once the dwell ends they can never
        change again, so they are moved to the pile that only Clear empties.
        """
        if not payload:
            return
        dwell = payload.get("dwell")
        face = payload.get("face") or "?"
        tracks = payload.get("tracks") or {}

        if self._dwell is not None and dwell != self._dwell:
            self._freeze_live()
        self._dwell = dwell

        live: Dict[Tuple[str, int], List[Path]] = {}
        for vial, paths in tracks.items():
            usable = [list(path) for path in (paths or []) if path and len(path) >= 2]
            if usable:
                live[(face, int(vial))] = usable
        self._live = live

    def _freeze_live(self) -> None:
        for (face, _vial), paths in self._live.items():
            for path in paths:
                self._frozen.append((face, path))
        self._live = {}
        overflow = len(self._frozen) - MAX_FROZEN_PATHS
        if overflow > 0:
            del self._frozen[:overflow]
            self.dropped += overflow

    # -- painting ---------------------------------------------------------------------------------
    def paint(self, painter, view) -> None:
        if not self.enabled:
            return
        painter.setRenderHint(painter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        total = max(1, len(self._frozen))
        for index, (face, path) in enumerate(self._frozen):
            # Oldest faintest. `index` is age order because paths are appended as dwells end.
            alpha = OLDEST_ALPHA + (NEWEST_ALPHA - OLDEST_ALPHA) * index // total
            self._draw(painter, view, path, _colour(face), alpha, 1.0)
        for (face, _vial), paths in self._live.items():
            # The current dwell at full strength: it is what is happening now, and it is the part
            # the operator is judging the tracker on.
            for path in paths:
                self._draw(painter, view, path, _colour(face), 255, 1.6)

    @staticmethod
    def _draw(painter, view, path: Sequence[Tuple[float, float]], colour: QColor,
              alpha: int, width: float) -> None:
        polygon = QPolygonF()
        for x, y in path:
            point = view.to_widget(x, y)
            polygon.append(QPointF(point.x(), point.y()))
        pen = QPen(QColor(colour.red(), colour.green(), colour.blue(), int(alpha)), width)
        painter.setPen(pen)
        painter.drawPolyline(polygon)


def _colour(face: str) -> QColor:
    return FACE_COLORS.get(str(face), FALLBACK_COLOR)


class CompositeOverlay:
    """Draws several overlays in order. The vial outlines first, the tracks on top of them.

    Exists because `PreviewWidget` holds ONE overlay and both of these want the run's picture:
    the outlines say where each vial is, the tracks say what moved inside it, and either alone
    answers half the question.
    """

    def __init__(self, *overlays) -> None:
        self.overlays = [o for o in overlays if o is not None]

    def paint(self, painter, view) -> None:
        for overlay in self.overlays:
            painter.save()
            try:
                overlay.paint(painter, view)
            except Exception:
                # One overlay must never take the other down, nor the window: this is painted on
                # every frame of a run that is watched for days.
                pass
            finally:
                painter.restore()

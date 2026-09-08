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

import gc

from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen, QPolygonF

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
        # ONE QPainterPath FOR THE LIFETIME OF THE OVERLAY, cleared and refilled per path. Building
        # a fresh one per path was pixel-identical but left Qt wrappers for the collector to trip
        # over at interpreter shutdown (the same failure class as the crash, seen at test exit).
        # Set HERE, not in clear(): a fresh overlay is painted before anything clears it, and a
        # paint that raises leaves its QPainter active on a device Qt then aborts destroying.
        self._scratch = QPainterPath()
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
        """Paint every path, with Python's CYCLIC COLLECTOR PAUSED for the duration.

        THE CRASH THIS PREVENTS, established by experiment rather than inference. With tracking
        on, the run thread snapshots the track lists on every frame -- thousands of short-lived
        lists -- so it triggers cyclic garbage collection constantly, on ITS thread. A collection
        traverses every tracked object in the process, whichever thread is using it, and while
        this method was mid-paint it died with an access violation inside python314.dll at the same
        offset every time: run thread "Garbage-collecting", GUI thread in `_draw`. A headless
        reproducer of the real window crashed on baseline in every run; with cyclic collection
        disabled it survived. Rewrites of `_draw` that avoided per-point Qt objects shrank the
        window but did not close it -- because the collision is collector-vs-painter, not any one
        object.

        So the collector is paused for exactly the span of a paint and restored in `finally`, so
        an exception while drawing can never leave it off. Reference counting still frees
        everything acyclic immediately; only cycles wait the few milliseconds a paint takes.
        """
        if not self.enabled:
            return
        was_enabled = gc.isenabled()
        if was_enabled:
            gc.disable()
        try:
            self._paint(painter, view)
        finally:
            if was_enabled:
                gc.enable()

    def _paint(self, painter, view) -> None:
        painter.setRenderHint(painter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        transform = _view_transform(view)
        total = max(1, len(self._frozen))
        for index, (face, path) in enumerate(self._frozen):
            # Oldest faintest. `index` is age order because paths are appended as dwells end.
            alpha = OLDEST_ALPHA + (NEWEST_ALPHA - OLDEST_ALPHA) * index // total
            self._draw(painter, view, path, _colour(face), alpha, 1.0, self._scratch, transform)
        for (face, _vial), paths in self._live.items():
            # The current dwell at full strength: it is what is happening now, and it is the part
            # the operator is judging the tracker on.
            for path in paths:
                self._draw(painter, view, path, _colour(face), 255, 1.6, self._scratch, transform)

    @staticmethod
    def _draw(painter, view, path: Sequence[Tuple[float, float]], colour: QColor,
              alpha: int, width: float, scratch=None, transform=None) -> None:
        """One path as a single QPainterPath, with the image->widget transform done in floats.

        NO PER-POINT Qt OBJECTS, and that is the crash fix, not a style choice. This used to call
        `view.to_widget` for every point (a QPointF each) and append each to a QPolygonF: thousands
        of short-lived Shiboken wrapper objects per paint, while the run thread's per-frame track
        snapshot kept triggering the cyclic garbage collector -- which traverses every tracked
        object on EVERY thread, including wrappers this loop had half-built. The process died with
        an access violation inside python314.dll at the same offset each time, always with the GUI
        thread in this function and the run thread "Garbage-collecting". It only ever happened with
        tracking on (no tracks, no wrappers), which is why activity-only runs looked reliable.

        Established by experiment, not inference: a headless reproducer of the real window crashed
        on baseline, survived with cyclic GC disabled, and survived with this exact rewrite. The
        transform is read from `view.to_widget` TWICE (origin and unit point) per paint rather than
        once per point, so the view contract is unchanged and the pixels are the same; a test
        measures that against the old polyline rather than eyeballing it.
        """
        if transform is None:
            transform = _view_transform(view)
        if transform is None:
            return
        rx, ry, sx, sy = transform
        points = iter(path)
        try:
            x, y = next(points)
        except StopIteration:
            return
        line = scratch if scratch is not None else QPainterPath()
        line.clear()
        line.moveTo(rx + x * sx, ry + y * sy)
        for x, y in points:
            line.lineTo(rx + x * sx, ry + y * sy)
        painter.setPen(QPen(QColor(colour.red(), colour.green(), colour.blue(), int(alpha)), width))
        painter.drawPath(line)


def _view_transform(view):
    """``(rx, ry, sx, sy)`` of the image->widget mapping, from TWO `to_widget` calls.

    The preview maps image pixels to widget pixels by a scale and an offset, so two points fix it
    exactly -- and reading it this way keeps the only contract the overlay ever had with its view
    (`to_widget`) while replacing a QPointF per track point with two per paint. None when the view
    has no picture yet (both points collapse to the origin), in which case nothing is drawn.
    """
    ox, oy = _xy(view.to_widget(0.0, 0.0))
    ux, uy = _xy(view.to_widget(1.0, 1.0))
    sx, sy = ux - ox, uy - oy
    if sx == 0.0 and sy == 0.0:
        return None
    return ox, oy, sx, sy


def _xy(point):
    """``(x, y)`` floats from whatever a view hands back: QPointF/QPoint, or a plain pair."""
    try:
        return float(point.x()), float(point.y())
    except AttributeError:
        return float(point[0]), float(point[1])


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

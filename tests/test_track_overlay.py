"""Fly trajectories accumulate on the picture until Clear, and the analysis still resets at a flip.

THE TWO RULES DISAGREE ON PURPOSE. A `VialTracker` lives for ONE dwell -- across a rotation the
flies are shaken and every identity is lost, so `behaviour.csv` resets at every flip. The PICTURE
has no such constraint: the rig owner asked for paths that build up over a whole session. So the
measurement resets and the drawing does not, and the dwell number from the pipeline is what lets
one surface do both.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from flygym_tracker.gui.track_overlay import (FACE_COLORS, MAX_FROZEN_PATHS,   # noqa: E402
                                              CompositeOverlay, TrackOverlay)


def _path(x=0.0, n=5):
    return [(x + i, 10.0 + i) for i in range(n)]


def _snapshot(dwell, face, **vials):
    return {"dwell": dwell, "face": face,
            "tracks": {int(k.lstrip("v")): v for k, v in vials.items()}}


# =============================================================================================
# Accumulating
# =============================================================================================
def test_paths_survive_a_rotation_even_though_the_analysis_does_not():
    """The whole point. `behaviour.csv` starts a new row at every flip; the picture keeps going."""
    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[_path(0)]))
    overlay.update(_snapshot(1, "B", v1=[_path(50)]))
    overlay.update(_snapshot(2, "A", v1=[_path(100)]))
    assert overlay.n_paths == 3, "a rotation threw paths away"


def test_a_growing_path_is_replaced_not_appended_within_a_dwell():
    """Fragments GROW point by point as a fly walks, and the pipeline re-sends the whole fragment
    every update. Appending would draw one path fifty times and make a two-second dwell look like
    a hundred flies."""
    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[_path(0, n=3)]))
    overlay.update(_snapshot(0, "A", v1=[_path(0, n=4)]))
    overlay.update(_snapshot(0, "A", v1=[_path(0, n=5)]))
    assert overlay.n_paths == 1, "the same fragment was accumulated %d times" % overlay.n_paths


def test_clear_wipes_everything_including_the_current_dwell():
    """A Clear that left the live fragments on screen would look like it had half worked -- and it
    is pressed precisely when the operator wants a clean slate to judge the next minutes against."""
    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[_path(0)]))
    overlay.update(_snapshot(1, "A", v1=[_path(9)]))
    assert overlay.n_paths == 2
    overlay.clear()
    assert overlay.n_paths == 0


def test_accumulation_restarts_from_nothing_after_a_clear():
    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[_path(0)]))
    overlay.clear()
    overlay.update(_snapshot(1, "A", v1=[_path(5)]))
    assert overlay.n_paths == 1


def test_a_single_point_fragment_is_not_a_trajectory():
    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[[(1.0, 2.0)]]))
    assert overlay.n_paths == 0


def test_an_empty_or_missing_snapshot_changes_nothing():
    """A plot or an overlay must never be able to take down a running experiment."""
    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[_path(0)]))
    overlay.update(None)
    overlay.update({})
    assert overlay.n_paths == 1


# =============================================================================================
# Colour and capping
# =============================================================================================
def test_each_face_has_its_own_colour():
    """One colour per face, shared by all sixteen of its vials -- the same rule the plot grids
    use, so a colour means the same thing wherever it appears."""
    assert FACE_COLORS["A"] != FACE_COLORS["B"]


def test_the_face_is_remembered_per_path():
    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[_path(0)]))
    overlay.update(_snapshot(1, "B", v1=[_path(5)]))
    overlay.update(_snapshot(2, "A", v1=[_path(9)]))       # freezes the face-B dwell
    faces = {face for face, _path_points in overlay._frozen}
    assert faces == {"A", "B"}, "a path lost the face it came from"


def test_the_oldest_paths_are_dropped_and_counted_rather_than_silently_lost():
    """A three-day session produces tens of thousands. A picture that stopped accumulating without
    saying so would be read as a rig that stopped moving."""
    overlay = TrackOverlay()
    for dwell in range(MAX_FROZEN_PATHS + 50):
        overlay.update(_snapshot(dwell, "A", v1=[_path(dwell % 100)]))
    overlay.update(_snapshot(MAX_FROZEN_PATHS + 100, "A"))   # freeze the last live one
    assert len(overlay._frozen) <= MAX_FROZEN_PATHS
    assert overlay.dropped > 0, "paths were dropped without being counted"


# =============================================================================================
# Drawing
# =============================================================================================
class _View:
    """Just enough of `PreviewWidget` for the painter: image -> widget is the identity here."""

    def to_widget(self, x, y):
        from PySide6.QtCore import QPointF

        return QPointF(float(x), float(y))


def test_a_disabled_overlay_draws_nothing(qapp):
    """The tick box is drawing only: it must never touch what is measured or accumulated."""
    from PySide6.QtGui import QImage, QPainter

    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[_path(0)]))
    overlay.enabled = False

    image = QImage(60, 60, QImage.Format.Format_RGB32)
    image.fill(0)
    painter = QPainter(image)
    overlay.paint(painter, _View())
    painter.end()

    assert all(image.pixelColor(x, y).red() == 0 for x in range(60) for y in range(60)), \
        "a hidden overlay drew on the picture"
    assert overlay.n_paths == 1, "hiding the tracks discarded them"


def test_drawing_actually_puts_ink_on_the_picture(qapp):
    from PySide6.QtGui import QImage, QPainter

    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[[(5.0, 5.0), (50.0, 50.0)]]))

    image = QImage(60, 60, QImage.Format.Format_RGB32)
    image.fill(0)
    painter = QPainter(image)
    overlay.paint(painter, _View())
    painter.end()

    lit = sum(1 for x in range(60) for y in range(60) if image.pixelColor(x, y).blue() > 20)
    assert lit > 10, "the track was not drawn"


def test_one_broken_overlay_does_not_stop_the_other(qapp):
    """Both the vial outlines and the tracks want the run's picture, and the view holds one
    overlay. Neither may take the other down -- this paints on every frame of a run watched for
    days."""
    from PySide6.QtGui import QImage, QPainter

    class Explodes:
        def paint(self, painter, view):
            raise RuntimeError("boom")

    overlay = TrackOverlay()
    overlay.update(_snapshot(0, "A", v1=[[(5.0, 5.0), (50.0, 50.0)]]))
    composite = CompositeOverlay(Explodes(), overlay)

    image = QImage(60, 60, QImage.Format.Format_RGB32)
    image.fill(0)
    painter = QPainter(image)
    composite.paint(painter, _View())          # must not raise
    painter.end()

    lit = sum(1 for x in range(60) for y in range(60) if image.pixelColor(x, y).blue() > 20)
    assert lit > 10, "a raising overlay stopped the healthy one drawing"


# =============================================================================================
# In the window
# =============================================================================================
def test_the_tick_box_is_on_by_default(qapp):
    """The tracks are the only surface that shows whether the tracker is following flies or tube
    edges, so the useful default is that an operator sees them without knowing the box exists."""
    from flygym_tracker.gui.video_stage import VideoStage

    class Box:
        stats = (0, 0)

        def take(self):
            return None

    class Session:
        latest = Box()
        is_open = True
        measured_fps = 0.0
        tap = None

        def attach_tap(self, job):
            return False

        def detach_tap(self):
            pass

    stage = VideoStage(Session())
    assert stage.tracks_box.isChecked()
    assert stage.tracks.enabled

    stage.tracks_box.setChecked(False)
    assert not stage.tracks.enabled
    assert "hidden" in stage.tracks_note.text()

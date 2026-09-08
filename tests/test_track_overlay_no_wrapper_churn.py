"""The track overlay must not create a Qt object per point, and must draw the same pixels.

THE CRASH THIS PREVENTS. `TrackOverlay._draw` built a QPointF per point and appended each to a
QPolygonF -- thousands of short-lived Shiboken wrappers per paint. The cyclic garbage collector,
triggered on the RUN thread by the per-frame track snapshot, traverses every tracked object on
every thread, wrappers this loop was mid-way through building included. The result was an access
violation inside python314.dll at the same offset every time: the GUI thread always in `_draw`, the
run thread always "Garbage-collecting". Only with tracking on, only when the overlay painted, which
is why activity-only runs looked reliable and the headless CLI (no overlay) never crashed.

Established by experiment: a headless reproducer of the real window crashed on baseline, survived
with cyclic GC disabled, and survived with the wrapper-free rewrite. This file keeps it that way,
and checks the rewrite draws what the polyline drew -- measured, not eyeballed.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRect
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from flygym_tracker.gui.track_overlay import TrackOverlay


class _View:
    """The two things `_draw` needs from the preview: the letterboxed rect and the image size."""

    def __init__(self, w: int, h: int, rect: QRect):
        self._image = QImage(w, h, QImage.Format.Format_Grayscale8)
        self._rect = rect

    def image_rect(self):
        return self._rect

    def to_widget(self, x, y):                 # the reference transform, as Preview defines it
        r = self._rect
        return QPointF(r.x() + x * r.width() / float(self._image.width()),
                       r.y() + y * r.height() / float(self._image.height()))


def test_draw_creates_no_per_point_qt_objects():
    """THE INVARIANT, measured: the number of `to_widget` calls -- each a QPointF wrapper -- must
    not scale with the path length. Two per paint (origin and unit point) fix the transform."""
    calls = []

    class _CountingView(_View):
        def to_widget(self, x, y):
            calls.append((x, y))
            return super().to_widget(x, y)

    view = _CountingView(1280, 1024, QRect(15, 10, 170, 136))
    path = [(float(i * 30), float(i * 20)) for i in range(40)]
    img = _render(TrackOverlay._draw, view, path)
    assert img[..., 3].sum() > 0, "the rewrite drew nothing"
    assert len(calls) <= 2, "%d to_widget calls for a 40-point path: per-point wrappers are back" % len(calls)


def _render(draw, view, path, size=(200, 160)):
    canvas = QImage(size[0], size[1], QImage.Format.Format_ARGB32)
    canvas.fill(QColor(0, 0, 0, 0))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    draw(painter, view, path, QColor(255, 80, 20), 255, 1.6)
    painter.end()
    raw = bytes(canvas.constBits())
    return np.frombuffer(raw, dtype=np.uint8).reshape(size[1], size[0], 4).astype(np.int16)


def _reference_polyline(painter, view, path, colour, alpha, width):
    """What `_draw` used to do, kept here as the oracle the rewrite is measured against."""
    polygon = QPolygonF()
    for x, y in path:
        p = view.to_widget(x, y)
        polygon.append(QPointF(p.x(), p.y()))
    painter.setPen(QPen(QColor(colour.red(), colour.green(), colour.blue(), int(alpha)), width))
    painter.drawPolyline(polygon)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_rewrite_draws_the_same_pixels_as_the_old_polyline(seed):
    """MEASURED, not eyeballed: on a scaled and offset letterbox like the real preview, the new
    path must be what the polyline was. A tiny tolerance covers antialiasing seams only."""
    rng = np.random.default_rng(seed)
    view = _View(1280, 1024, QRect(15, 10, 170, 136))          # scaled ~0.13x, offset
    path = [(float(x), float(y)) for x, y in rng.uniform([0, 0], [1280, 1024], size=(40, 2))]

    new = _render(TrackOverlay._draw, view, path)
    old = _render(_reference_polyline, view, path)

    diff = np.abs(new - old)
    assert new[..., 3].sum() > 0, "the rewrite drew nothing"
    assert diff.max() <= 2, "max channel difference %d: not the same line" % diff.max()
    assert (diff > 0).mean() < 0.002, "%.4f of pixels differ" % (diff > 0).mean()


def test_an_empty_or_single_point_path_is_harmless():
    """A path with nothing to join must not raise, and must not draw a stray dot at the origin."""
    view = _View(100, 100, QRect(0, 0, 100, 100))
    empty = _render(TrackOverlay._draw, view, [])
    assert empty[..., 3].sum() == 0
    _render(TrackOverlay._draw, view, [(10.0, 10.0)])     # must simply not raise


# =============================================================================================
# The collector is paused while painting, and always restored
# =============================================================================================
import gc                                                                    # noqa: E402


def _live_overlay():
    overlay = TrackOverlay()
    overlay.update({"dwell": 0, "face": "A", "tracks": {1: [[(5.0, 5.0), (50.0, 50.0)]]}})
    return overlay


def test_cyclic_gc_is_off_while_the_overlay_paints(qapp):
    """THE MECHANISM: the crash is a cyclic collection on the run thread colliding with a paint on
    the GUI thread. While `paint` runs, collection must be disabled process-wide."""
    overlay = _live_overlay()
    seen = []
    overlay._paint = lambda painter, view: seen.append(gc.isenabled())
    assert gc.isenabled(), "test precondition: the collector starts enabled"
    overlay.paint(object(), object())
    assert seen == [False], "the collector was still enabled during the paint"
    assert gc.isenabled(), "the collector was not restored after the paint"


def test_the_collector_is_restored_even_when_drawing_raises(qapp):
    """`finally`, not a happy-path re-enable: a paint that raises must not leave GC off for the
    rest of the process -- that would be a memory leak wearing the crash fix as a disguise."""
    overlay = _live_overlay()

    def boom(painter, view):
        raise RuntimeError("boom")

    overlay._paint = boom
    with pytest.raises(RuntimeError):
        overlay.paint(object(), object())
    assert gc.isenabled()


def test_a_disabled_collector_is_left_disabled(qapp):
    """If someone above us already paused it, we must not re-enable it behind their back."""
    overlay = _live_overlay()
    overlay._paint = lambda painter, view: None
    gc.disable()
    try:
        overlay.paint(object(), object())
        assert not gc.isenabled()
    finally:
        gc.enable()

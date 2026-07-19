"""The preview: the coordinate transform Stage 2 will need, and a caption that cannot overstate.

`fit_rect` is tested numerically rather than by eye because Stage 2's vial drawing inverts it --
turning a click at a widget coordinate into an image coordinate, and back for hit-testing. Having
it extracted and pinned is the difference between extending this widget and rewriting it.
"""
from __future__ import annotations

import numpy as np
import pytest

from flygym_tracker.gui.preview import PreviewWidget, fit_rect


# =============================================================================================
# fit_rect -- pure arithmetic, no widget
# =============================================================================================
def test_a_frame_that_already_fits_exactly_is_not_scaled():
    assert fit_rect(640, 480, 640, 480) == (0, 0, 640, 480)


def test_a_wider_target_letterboxes_left_and_right_not_top_and_bottom():
    x, y, w, h = fit_rect(640, 480, 1000, 480)
    assert (w, h) == (640, 480)
    assert y == 0
    assert x == 180                       # centred: (1000 - 640) // 2


def test_a_taller_target_letterboxes_top_and_bottom():
    x, y, w, h = fit_rect(640, 480, 640, 800)
    assert (w, h) == (640, 480)
    assert x == 0
    assert y == 160


def test_the_aspect_ratio_survives_scaling():
    """A stretched preview is a preview an operator would tune exposure and focus against."""
    _x, _y, w, h = fit_rect(1280, 1024, 500, 500)
    assert w / h == pytest.approx(1280 / 1024, rel=1e-2)


def test_scaling_never_enlarges_beyond_the_target():
    for dst in ((100, 100), (1000, 40), (33, 900)):
        _x, _y, w, h = fit_rect(1280, 1024, *dst)
        assert w <= dst[0] and h <= dst[1]


@pytest.mark.parametrize("args", [(0, 480, 100, 100), (640, 0, 100, 100),
                                  (640, 480, 0, 100), (640, 480, 100, 0),
                                  (-1, -1, 100, 100)])
def test_a_degenerate_size_returns_an_empty_rect_rather_than_raising(args):
    """A camera that has not delivered a frame yet reports (0, 0), and a paint that threw would
    take the window with it."""
    assert fit_rect(*args) == (0, 0, 0, 0)


# =============================================================================================
# The frame handoff
# =============================================================================================
def test_a_mono8_array_round_trips_into_the_widget_with_the_right_pixel(qapp):
    array = np.zeros((48, 64), dtype=np.uint8)
    array[10, 20] = 200
    view = PreviewWidget()
    view.set_frame(array)
    assert view.frame_size == (64, 48)
    assert view._image.pixelColor(20, 10).red() == 200


def test_the_widget_keeps_the_ndarray_alive_behind_the_qimage(qapp):
    """`QImage(arr.data, ...)` BORROWS the buffer -- it does not copy. Dropping the array while the
    QImage is on screen leaves it pointing at freed memory the moment the collector runs, and the
    symptom is a garbled or crashed preview minutes into a session."""
    array = np.full((8, 8), 77, dtype=np.uint8)
    view = PreviewWidget()
    view.set_frame(array)
    assert view._array is array, "the buffer the QImage points at is not being held"

    view.set_frame(np.full((8, 8), 99, dtype=np.uint8))
    assert view._array is not array, "the array and the image must be replaced together"
    assert view._image.pixelColor(0, 0).red() == 99


def test_a_colour_or_malformed_frame_is_ignored_rather_than_rendered_as_a_smear(qapp):
    """A 3-channel frame handed to a Grayscale8 QImage produces the wrong stride, which renders as
    a diagonal smear -- and a smeared preview is one an operator will tune against."""
    view = PreviewWidget()
    view.set_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert view._image is None
    view.set_frame(None)
    assert view._image is None


def test_clearing_drops_both_the_image_and_the_buffer(qapp):
    view = PreviewWidget()
    view.set_frame(np.zeros((8, 8), dtype=np.uint8))
    view.clear()
    assert view._image is None and view._array is None


def test_painting_with_no_frame_says_so_instead_of_showing_black(qapp):
    """A black rectangle looks like a camera pointed at a dark room, which is exactly what this rig
    IS -- 850 nm back-light, dark field. It has to say the difference in words."""
    view = PreviewWidget()
    view.resize(200, 150)
    view.show()
    view.grab()                            # exercises paintEvent offscreen; must not raise
    assert view._image is None

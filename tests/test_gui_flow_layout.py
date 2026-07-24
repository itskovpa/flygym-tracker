"""A strip of controls must never be the reason a window does not fit the rig laptop's screen.

FOURTH OCCURRENCE OF THIS BUG CLASS on this project, which is why it now has a layout of its own
rather than another hand-tuned set of button labels:

    1. the cv2 vial selector drew the lower vial row ~130 px below the screen bottom -- not visible
       and impossible to click, i.e. exactly the part of the tube being outlined;
    2. the cv2 settings panel was then placed 66 px past the right edge even though it fitted;
    3. the Qt window came to 873 px on an 852 px work area, putting the filter box under the
       taskbar;
    4. and the video-stage strip (7 buttons) plus the run band's tool strip forced the window's
       minimum width to 1646 px on a 1440 px desktop.

Each of the first three was fixed where it happened. This one is fixed in the layout, so the next
strip of buttons somebody adds cannot reintroduce it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, QSize                                     # noqa: E402
from PySide6.QtWidgets import QPushButton, QWidget                          # noqa: E402

from flygym_tracker.gui.flow_layout import FlowLayout, flow_strip           # noqa: E402

#: The rig laptop: a 2880x1800 panel at 200% scaling, i.e. a 1440x900 desktop.
RIG_DESKTOP = (1440, 900)


def _strip(qapp, labels, width=120):
    widget, layout = flow_strip()
    for label in labels:
        button = QPushButton(label)
        button.setFixedSize(width, 24)
        layout.addWidget(button)
    return widget, layout


def test_the_minimum_width_is_the_widest_item_not_the_sum_of_them(qapp):
    """THE ONE LINE THAT MATTERS. A QHBoxLayout's minimum width is the sum, and its only answer to
    "there is less room than that" is to make the window wider -- past the screen edge if need be,
    with no warning and no way to reach the controls that fall off."""
    _widget, layout = _strip(qapp, ["a", "b", "c", "d", "e", "f", "g"], width=200)
    assert layout.minimumSize().width() <= 210, layout.minimumSize().width()
    assert layout.sizeHint().width() > 1200, "the test strip is not wide enough to prove anything"


def test_it_wraps_downwards_when_there_is_not_enough_room_across(qapp):
    _widget, layout = _strip(qapp, ["a", "b", "c", "d"], width=100)
    one_row = layout.heightForWidth(1000)
    three_rows = layout.heightForWidth(220)          # fits two per row
    assert three_rows > one_row, "the strip did not wrap; it would be clipped instead"


def test_every_control_is_actually_placed_when_it_wraps(qapp):
    """Wrapping that DROPPED a control would be worse than the bug it fixes: a button that is not
    on screen is a job the operator cannot start, and nothing would say so."""
    widget, layout = _strip(qapp, ["a", "b", "c", "d", "e"], width=100)
    widget.show()
    layout.setGeometry(QRect(0, 0, 240, layout.heightForWidth(240)))
    placed = [layout.itemAt(i).geometry() for i in range(layout.count())]
    assert len(placed) == 5
    assert all(g.width() > 0 and g.height() > 0 for g in placed), "an item was placed with no size"
    # No two items may overlap -- a wrap that stacked buttons on top of each other would look
    # like a rendering glitch and make the covered one unclickable.
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            assert not a.intersects(b), "two controls were placed on top of each other"


def test_an_empty_strip_is_harmless(qapp):
    """A layout that raised on an empty strip would take the window with it during construction."""
    holder = QWidget()                      # held: a temporary parent takes the layout with it
    layout = FlowLayout(holder)
    assert layout.count() == 0
    assert layout.sizeHint().isValid()
    assert layout.heightForWidth(100) >= 0


def test_a_flow_container_grows_its_own_minimum_for_the_wrap(qapp):
    """THE RECORDING-ROW BUG. `FlowLayout` reports heightForWidth and a QVBoxLayout honours it -- so
    the run band's tool strip wraps and grows. But a QGridLayout cell never ASKS for heightForWidth:
    it gives the cell the one-line sizeHint height and clips the wrapped second line -- which is
    where the recording row's "size x" spin vanished. `FlowContainer` closes it from the other side,
    setting its OWN minimum height to the wrapped height, which a grid cell DOES respect."""
    from PySide6.QtWidgets import QPushButton
    from flygym_tracker.gui.flow_layout import FlowContainer

    row = FlowContainer(spacing=8)
    for label in ("save a video of the run", "every 2 frame(s)", "size 0.5x"):
        button = QPushButton(label)
        button.setFixedSize(160, 24)
        row.flow().addWidget(button)
    row.show()
    row.resize(200, 500)              # narrower than 3*160 -> the strip must wrap to more than one row
    qapp.processEvents()
    assert row.minimumHeight() == row.flow().heightForWidth(row.width()), \
        "the container's minimum height is not tracking its flow's wrapped height"
    assert row.minimumHeight() > 24, "no room was reserved for the wrapped rows -- they would clip"
    row.hide()


# =============================================================================================
# The claim this exists for, measured on the real window
# =============================================================================================
def test_the_whole_window_fits_the_rig_laptop_in_both_directions(qapp, tmp_path):
    """The end-to-end version. Width was 1646 before the strips wrapped; height was 864 on an
    852 px work area after the run band was added. Both are checked, because fixing one of them
    and re-breaking the other is exactly what happened last time."""
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    window = MainWindow(config=load_config(path="config/flygym_rig.yaml"),
                        config_path="config/flygym_rig.yaml",
                        state=gui_state.default_state(), root=str(tmp_path),
                        camera_factory=lambda: None, confirm=lambda text: False)
    window.show()
    hint = window.minimumSizeHint()
    window.session.shutdown()
    assert hint.width() <= RIG_DESKTOP[0] - 40, "minimum width %d" % hint.width()
    # The work area is the desktop minus the taskbar (~48 px), and the frame adds a title bar.
    assert hint.height() <= RIG_DESKTOP[1] - 100, "minimum height %d" % hint.height()


# =============================================================================================
# The window must sit INSIDE the work area, so its bottom row cannot hide under the taskbar
# ---------------------------------------------------------------------------------------------
# Reported from the rig: the run band's plot buttons vanished under the Windows taskbar. The window
# fits the screen at its fitted size, so the cause is placement -- it ended up low. `_fit_frame_
# within` is the after-show clamp that pulls it back on-screen; tested here as pure arithmetic.
# =============================================================================================
def test_a_window_that_already_sits_inside_is_left_where_it_is():
    from flygym_tracker.gui.main_window import _fit_frame_within
    # 1080p minus a 48 px taskbar = a 1920x1032 work area; a 1316x851 window near the top-left fits.
    assert _fit_frame_within(1316, 851, 100, 100, 0, 0, 1920, 1032) == (1316, 851, 100, 100)


def test_a_window_whose_bottom_is_under_the_taskbar_is_lifted_clear():
    from flygym_tracker.gui.main_window import _fit_frame_within
    # Same window placed low: bottom would be 300 + 851 = 1151, past the 1032 work-area edge.
    w, h, x, y = _fit_frame_within(1316, 851, 100, 300, 0, 0, 1920, 1032)
    assert (w, h, x) == (1316, 851, 100), "only the vertical position should have changed"
    assert y + h == 1032, "the window's bottom was not brought to rest on the work-area edge"


def test_a_window_taller_than_the_work_area_is_shrunk_and_pinned_to_the_top():
    from flygym_tracker.gui.main_window import _fit_frame_within
    w, h, _x, y = _fit_frame_within(1316, 1200, 100, 60, 0, 0, 1920, 1032)
    assert h == 1032, "a window taller than the work area must be shrunk to it"
    assert y == 0, "a too-tall window pins to the top so its TOP controls stay reachable"


def test_the_work_area_origin_is_respected_not_assumed_to_be_zero():
    """A second monitor's work area does not start at (0, 0); a window just off its left edge must
    be pulled onto the monitor, not onto the primary screen."""
    from flygym_tracker.gui.main_window import _fit_frame_within
    _w, _h, x, _y = _fit_frame_within(800, 600, 1900, 10, 1920, 0, 1920, 1032)
    assert x == 1920


def test_the_clamp_never_shrinks_below_the_apps_own_minimum():
    from flygym_tracker.gui.main_window import _fit_frame_within
    w, h, _x, _y = _fit_frame_within(1316, 851, 0, 0, 0, 0, 500, 400)
    assert (w, h) == (640, 480), "a tiny work area must not force the window below its minimum"

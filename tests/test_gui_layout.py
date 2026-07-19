"""Geometry, measured -- because a control nobody can reach passes every other test in this suite.

THE DEFECT THIS FILE EXISTS FOR, AND IT WAS A BAD ONE. The settings pane's content had a minimum
width of 1222 px inside a 584 px viewport, and the horizontal scrollbar was deliberately disabled
(so nothing could hide sideways). The result: EVERY value control on EVERY row sat off the right
edge, unreachable, in a window whose entire purpose is reaching them. The app was unusable.

Every other test in this suite passed, and would have gone on passing forever, because they drive
the widgets by calling their methods -- `row.value_widget.setValue(20.0)`, `row.arm_button.click()`
-- and a control works perfectly well when it is off-screen. It was found by rendering the window
with `widget.grab()` and looking at the picture, then measuring what the picture suggested.

So the lesson is narrower than "offscreen tests cannot check appearance". LAYOUT IS NOT
APPEARANCE. Whether a control is inside its viewport is a NUMBER, it is available offscreen, and
not asserting it was the gap. Fonts, colour and readability genuinely do need a human on the rig's
own monitor; "can the operator click the box" does not.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QAbstractSpinBox

from flygym_tracker.config import load_config
from flygym_tracker.gui import gui_state
from flygym_tracker.gui.main_window import MainWindow

from test_gui_camera_session import FakeSource

#: Sizes to check. The narrow one is a laptop half-screen; the wide one is the rig's monitor. A
#: layout that only works at one size is a layout that works by accident.
WINDOW_SIZES = [(1000, 700), (1240, 860), (1600, 1000)]


@pytest.fixture
def window(qapp, tmp_path):
    state = gui_state.default_state()
    w = MainWindow(config=load_config(path="config/flygym_rig.yaml"),
                   config_path="config/flygym_rig.yaml", state=state, root=str(tmp_path),
                   camera_factory=lambda: FakeSource(), confirm=lambda text: False)
    w.show()
    yield w
    w.session.shutdown()


def _controls(view):
    """``(key, widget)`` for the thing an operator has to reach on each row."""
    for key, row in view.rows.items():
        widget = row.value_widget if row.value_widget is not None else getattr(row, "arm_button",
                                                                              None)
        if widget is not None:
            yield key, widget


@pytest.mark.parametrize("size", WINDOW_SIZES)
def test_every_value_control_is_inside_the_visible_pane(qapp, window, size):
    """The one that was broken: at 1240x860, all eleven controls were off the right edge."""
    window.resize(*size)
    qapp.processEvents()
    view = window.settings_view
    viewport = view.scroll.viewport()

    off_screen = []
    for key, widget in _controls(view):
        right = widget.mapTo(view._body, widget.rect().bottomRight()).x()
        left = widget.mapTo(view._body, widget.rect().topLeft()).x()
        if right > viewport.width() or left < 0:
            off_screen.append((key, left, right, viewport.width()))
    assert off_screen == [], \
        "controls outside the %dx%d viewport: %r" % (size[0], size[1], off_screen)


@pytest.mark.parametrize("size", WINDOW_SIZES)
def test_the_settings_pane_never_needs_sideways_scrolling(qapp, window, size):
    """Its horizontal scrollbar is off by design, so content wider than the viewport is not
    scrolled to -- it is simply lost."""
    window.resize(*size)
    qapp.processEvents()
    view = window.settings_view
    assert view.scroll.horizontalScrollBar().isVisible() is False
    assert view._body.width() <= view.scroll.viewport().width() + 1, \
        "content is %d px wide in a %d px viewport, and there is no way to scroll to the rest" % (
            view._body.width(), view.scroll.viewport().width())


@pytest.mark.parametrize("size", WINDOW_SIZES)
def test_every_control_is_big_enough_to_hit_with_a_mouse(qapp, window, size):
    """A control squeezed to a few pixels is off-screen by another name. Rows shrink by giving up
    LABEL width (the full text stays in a tooltip), never control width."""
    window.resize(*size)
    qapp.processEvents()
    for key, widget in _controls(window.settings_view):
        assert widget.width() >= 40, "%s control is %d px wide" % (key, widget.width())
        assert widget.height() >= 16, "%s control is %d px tall" % (key, widget.height())


def test_arming_a_row_does_not_push_its_editor_off_the_edge(qapp, window):
    """The row swaps a label for a spinbox plus a "back to camera default" button -- more content
    than the default state holds, so this is where a width regression would appear first."""
    window.resize(1240, 860)
    view = window.settings_view
    for key in [k for k in view.rows if k.startswith("source.camera.")]:
        view.rows[key].arm_button.click()
    qapp.processEvents()

    viewport = view.scroll.viewport()
    for key, row in view.rows.items():
        if not key.startswith("source.camera."):
            continue
        assert row.value_widget is not None, "%s did not arm" % key
        right = row.value_widget.mapTo(view._body, row.value_widget.rect().bottomRight()).x()
        assert right <= viewport.width(), "%s editor ends at %d in a %d px viewport" % (
            key, right, viewport.width())
        back = row.default_button
        assert back.isVisible()
        assert back.mapTo(view._body, back.rect().bottomRight()).x() <= viewport.width()


def test_the_preview_pane_keeps_a_usable_share_of_the_window(qapp, window):
    """Exposure and gain are tuned by looking at the picture. A settings pane that grew until the
    preview was a sliver would defeat the reason the two are side by side."""
    window.resize(1240, 860)
    qapp.processEvents()
    assert window.preview.width() >= 320, window.preview.width()
    assert window.settings_view.width() >= 400, window.settings_view.width()


def test_the_window_body_never_forces_a_wider_window_than_the_rig_monitor(qapp, window):
    """The rig laptop is a 2880x1800 panel at 200% scaling, i.e. a 1440x900 desktop. A window whose
    minimum width exceeds that cannot be fully seen at all -- which is the same class of bug
    `live_vial_selector.screen_view_limit` exists to fix for the cv2 tools."""
    assert window.minimumSizeHint().width() <= 1400, window.minimumSizeHint().width()


def test_no_row_hides_its_state_when_the_pane_is_narrow(qapp, window):
    """At the narrow end the tri-state must still be readable: the words "camera default" and the
    arm button are the two channels that do not depend on colour."""
    window.resize(1000, 700)
    qapp.processEvents()
    row = window.settings_view.rows["source.camera.frame_rate"]
    assert row.default_label.isVisible()
    assert row.arm_button.isVisible()
    assert row.findChild(QAbstractSpinBox) is None


# =========================================================================================
# The window must fit the desktop it opens on
# =========================================================================================
def test_the_window_shrinks_to_fit_a_short_desktop(qapp, window):
    """Regression, and the THIRD of this class on this project.

    A hard-coded size is a guess about a screen the developer cannot see. The cv2 selector once
    drew 130 px of frame below the screen edge -- the lower vial row, unclickable. The cv2
    settings panel was then placed 66 px off the right edge even though it fitted. Here
    `resize(1180, 820)` plus the title bar came to 873 px on an 852 px work area, putting the
    filter box under the taskbar.
    """
    win = window
    w, h = win._fitted_size(1180, 820)
    screen = win.screen()
    if screen is not None:
        area = screen.availableGeometry()
        assert w <= area.width() and h <= area.height()
    assert w >= 640 and h >= 480          # never shrinks into uselessness


def test_a_roomy_desktop_gets_the_size_that_was_asked_for(qapp, window):
    """The clamp must not cost anything on a big monitor -- it only ever shrinks."""
    w, h = window._fitted_size(400, 300)
    assert (w, h) == (640, 480) or (w <= 400 and h <= 300)


def test_sizing_never_stops_the_window_opening(qapp, window, monkeypatch):
    """A screen query that raises must not be the reason an experiment cannot be set up."""
    monkeypatch.setattr(type(window), "screen",
                        lambda self: (_ for _ in ()).throw(RuntimeError("no screen")))
    assert window._fitted_size(1180, 820) == (1180, 820)

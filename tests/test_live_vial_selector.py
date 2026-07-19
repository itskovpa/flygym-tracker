"""Tests for the LIVE-VIDEO polygon vial selector and the N-vertex `polygon` ROI it produces.

HEADLESS BY CONSTRUCTION. Nothing here opens a cv2 window. `live_vial_selector` is split into
pure state/keyboard/rendering (`SelectorState`, `decode_key`, `handle_key`, `render_frame`) and a
thin `select_vials_live` driver, so even the driver loop is exercised -- against a stubbed highgui
and a synthetic `FrameSource` -- without a display or a camera.

Five things are being protected here:
  1. the state machine the operator actually feels (click, ENTER, BACKSPACE, u, c, SPACE, q),
  2. the driver WIRING (a click really does become a vertex; ENTER really does store a vial),
  3. that the feed stays LIVE while clicks are collected, and that SPACE really holds a frame,
  4. `build_calibration_from_polygons`: what was drawn is what gets saved,
  5. the measurement path: the pipeline honours `polygon`, and a bundle with NEITHER polygon nor
     quad measures byte-identically to before any of this existed.
"""
from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np
import pytest

from flygym_tracker import calibration as C
from flygym_tracker import live_vial_selector as LVS
from flygym_tracker.live_vial_selector import (
    SelectorState,
    decode_key,
    handle_key,
    render_frame,
    select_vials_live,
    view_scale,
)
from flygym_tracker.frame_source import FrameSource
from flygym_tracker.types import Calibration, FaceCalibration, Frame, VialROI

W, H = 200, 200

#: A triangle, a square and a pentagon -- the point being that vials are NOT all 4-gons.
TRI = [[10, 10], [50, 10], [30, 50]]
SQUARE = [[60, 10], [100, 10], [100, 50], [60, 50]]
PENTAGON = [[10, 120], [50, 120], [60, 150], [30, 180], [5, 150]]


def _state(n_vials=16, face="A", polygons=()):
    """A state with `polygons` already finished (drawn through the public API, not injected)."""
    s = SelectorState(n_vials=n_vials, face=face)
    for poly in polygons:
        for x, y in poly:
            s.add_vertex(x, y)
        assert s.finish_vial()
    return s


# =========================================================================================
# 1. SelectorState -- vertices
# =========================================================================================
def test_add_vertex_appends_in_click_order_and_rounds_to_pixels():
    s = SelectorState(n_vials=4)
    assert s.add_vertex(10.4, 20.6) == [10, 21]
    s.add_vertex(30, 40)
    assert s.current == [[10, 21], [30, 40]]
    assert s.polygons == []          # nothing is finished until ENTER


def test_current_and_polygons_are_copies_that_cannot_corrupt_the_state():
    s = _state(polygons=[TRI])
    s.add_vertex(1, 2)
    s.current.append([999, 999])
    s.polygons[0][0][0] = -1
    assert s.current == [[1, 2]]
    assert s.polygons == [TRI]


def test_vial_number_is_one_based_and_follows_the_finished_count():
    s = SelectorState(n_vials=16)
    assert s.vial_number == 1
    _state_finish(s, TRI)
    assert s.vial_number == 2


def _state_finish(s, poly):
    for x, y in poly:
        s.add_vertex(x, y)
    return s.finish_vial()


# =========================================================================================
# 2. SelectorState -- ENTER needs three points
# =========================================================================================
@pytest.mark.parametrize("n_points", [0, 1, 2])
def test_finish_vial_refuses_fewer_than_three_points_and_does_not_advance(n_points):
    s = SelectorState(n_vials=16)
    for i in range(n_points):
        s.add_vertex(i, i)
    assert s.finish_vial() is False
    assert s.polygons == []                        # nothing stored
    assert len(s.current) == n_points              # and nothing thrown away either
    assert s.vial_number == 1                      # still on the same vial
    assert "at least 3 points" in s.message        # the on-screen nag


def test_finish_vial_accepts_exactly_three_points():
    s = SelectorState(n_vials=16)
    assert _state_finish(s, TRI) is True
    assert s.polygons == [TRI]
    assert s.current == []
    assert s.vial_number == 2


def test_vials_keep_their_draw_order_and_their_own_vertex_counts():
    s = _state(polygons=[TRI, SQUARE, PENTAGON])
    assert s.polygons == [TRI, SQUARE, PENTAGON]
    assert [len(p) for p in s.polygons] == [3, 4, 5]


def test_a_polygon_may_have_many_vertices():
    """N is arbitrary -- an edge tube on a cylindrical drum is not a quadrilateral."""
    circle = [[int(100 + 40 * np.cos(t)), int(100 + 40 * np.sin(t))]
              for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    s = _state(polygons=[circle])
    assert len(s.polygons[0]) == 24


# =========================================================================================
# 3. SelectorState -- undo / clear
# =========================================================================================
def test_backspace_removes_the_last_vertex_only():
    s = SelectorState(n_vials=4)
    for x, y in TRI:
        s.add_vertex(x, y)
    assert s.undo_vertex() is True
    assert s.current == TRI[:2]


def test_backspace_on_an_empty_polygon_is_a_no_op_with_a_message():
    s = SelectorState(n_vials=4)
    assert s.undo_vertex() is False
    assert s.current == []
    assert "no point to undo" in s.message


def test_undo_vial_reopens_the_previous_polygon_for_editing():
    s = _state(polygons=[TRI, SQUARE])
    assert s.undo_vial() is True
    assert s.polygons == [TRI]          # the square is no longer finished...
    assert s.current == SQUARE          # ...it is back under the cursor
    assert s.vial_number == 2           # and we are drawing vial 2 again
    # and it can be edited and re-finished
    s.undo_vertex()
    s.add_vertex(120, 60)
    assert s.finish_vial() is True
    assert s.polygons == [TRI, SQUARE[:3] + [[120, 60]]]


def test_undo_vial_discards_the_in_progress_points_and_says_so():
    s = _state(polygons=[TRI])
    s.add_vertex(70, 70)
    s.add_vertex(80, 80)
    assert s.undo_vial() is True
    assert s.current == TRI
    assert "discarded 2 in-progress point(s)" in s.message


def test_undo_vial_with_nothing_finished_is_a_no_op():
    s = SelectorState(n_vials=4)
    s.add_vertex(1, 1)
    assert s.undo_vial() is False
    assert s.current == [[1, 1]]
    assert "no completed vial" in s.message


def test_clear_empties_only_the_polygon_in_progress():
    s = _state(polygons=[TRI])
    s.add_vertex(70, 70)
    assert s.clear() is True
    assert s.current == []
    assert s.polygons == [TRI]


def test_clear_on_an_empty_polygon_is_a_no_op():
    s = SelectorState(n_vials=4)
    assert s.clear() is False
    assert "already empty" in s.message


# =========================================================================================
# 4. SelectorState -- freeze, completion, early finish, status line
# =========================================================================================
def test_freeze_toggles_and_defaults_to_live():
    s = SelectorState(n_vials=4)
    assert s.frozen is False
    assert s.toggle_freeze() is True
    assert "FROZEN" in s.message
    assert s.toggle_freeze() is False


def test_is_complete_only_when_every_vial_is_drawn():
    s = _state(n_vials=2, polygons=[TRI])
    assert s.is_complete is False and s.done is False
    _state_finish(s, SQUARE)
    assert s.is_complete is True and s.done is True


def test_undo_vial_after_the_last_one_makes_the_session_incomplete_again():
    s = _state(n_vials=2, polygons=[TRI, SQUARE])
    assert s.is_complete
    s.undo_vial()
    assert s.is_complete is False


def test_finish_early_keeps_what_was_drawn():
    s = _state(n_vials=16, polygons=[TRI, SQUARE])
    s.finish_early()
    assert s.done is True and s.is_complete is False
    assert s.polygons == [TRI, SQUARE]


def test_n_vials_must_be_positive():
    with pytest.raises(ValueError):
        SelectorState(n_vials=0)


def test_the_status_line_expires_after_its_ttl():
    s = SelectorState(n_vials=4)
    s.note("hello", ttl=2)
    s.tick()
    assert s.message == "hello"
    s.tick()
    assert s.message == ""


# =========================================================================================
# 5. Keyboard (pure)
# =========================================================================================
@pytest.mark.parametrize("code,name", [
    (13, "enter"), (10, "enter"), (1048586, "enter"),      # main / keypad / Qt-with-modifiers
    (8, "backspace"), (65288, "backspace"), (127, "backspace"),   # Windows / GTK / macOS
    (32, "space"), (27, "esc"), (ord("q"), "q"), (ord("u"), "u"), (ord("c"), "c"),
    (ord("U"), "u"),                                        # caps lock must not break undo
])
def test_decode_key_maps_every_control_this_selector_uses(code, name):
    assert decode_key(code) == name


@pytest.mark.parametrize("code", [-1, None])
def test_decode_key_reports_no_keypress(code):
    assert decode_key(code) is None


def test_handle_key_enter_finishes_a_vial():
    s = SelectorState(n_vials=4)
    for x, y in TRI:
        s.add_vertex(x, y)
    assert handle_key(s, "enter") is None
    assert s.polygons == [TRI]


def test_handle_key_routes_every_control_to_the_state():
    s = _state(n_vials=4, polygons=[TRI])
    s.add_vertex(70, 70)
    s.add_vertex(80, 80)
    handle_key(s, "backspace")
    assert s.current == [[70, 70]]
    handle_key(s, "c")
    assert s.current == []
    handle_key(s, "u")
    assert s.current == TRI and s.polygons == []
    handle_key(s, "space")
    assert s.frozen is True


@pytest.mark.parametrize("key", ["q", "esc"])
def test_handle_key_quit_ends_the_session(key):
    s = _state(n_vials=4, polygons=[TRI])
    assert handle_key(s, key) == "done"
    assert s.polygons == [TRI]


def test_handle_key_returns_done_when_the_last_vial_is_finished():
    s = _state(n_vials=2, polygons=[TRI])
    for x, y in SQUARE:
        s.add_vertex(x, y)
    assert handle_key(s, "enter") == "done"


def test_handle_key_ignores_idle_frames_and_unknown_keys():
    s = _state(n_vials=4, polygons=[TRI])
    assert handle_key(s, None) is None
    assert handle_key(s, "z") is None
    assert s.polygons == [TRI] and s.current == []


# =========================================================================================
# 6. Rendering (no window)
# =========================================================================================
def _image_region(canvas):
    """The part of the canvas that is the CAMERA PICTURE -- everything right of the panel."""
    return canvas[:, LVS.PANEL_WIDTH:]


def test_render_frame_puts_the_panel_beside_the_frame_not_over_it():
    canvas = render_frame(np.full((H, W), 90, np.uint8), _state(polygons=[TRI]))
    assert canvas.shape == (H, LVS.PANEL_WIDTH + W, 3) and canvas.dtype == np.uint8


def test_no_ui_text_is_ever_drawn_on_the_camera_image():
    """The requirement, asserted directly: nothing occludes the picture except the operator's
    own polygons. Text used to sit in a band across the top of the frame, hiding the upper tube
    row -- the very thing being outlined."""
    frame = np.full((H, W), 90, np.uint8)
    state = SelectorState(n_vials=16, face="A", source_label="CAMERA DA4282883 (live)")
    state.note("a status message that must not land on the picture")

    picture = _image_region(render_frame(frame, state))       # nothing drawn yet
    assert np.array_equal(picture, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    # ...and with vials drawn, the ONLY changed pixels are the polygon's own ink.
    drawn = _image_region(render_frame(frame, _state(polygons=[PENTAGON])))
    changed = np.argwhere(cv2.absdiff(drawn, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)).max(axis=2) > 8)
    xs, ys = changed[:, 1], changed[:, 0]
    poly = np.array(PENTAGON)
    assert xs.min() >= poly[:, 0].min() - 12 and xs.max() <= poly[:, 0].max() + 12
    assert ys.min() >= poly[:, 1].min() - 12 and ys.max() <= poly[:, 1].max() + 12


def test_the_panel_carries_the_things_the_operator_needs():
    """Ink in the panel column must actually change when the state does."""
    frame = np.full((H, W), 90, np.uint8)
    bare = SelectorState(n_vials=16, face="A")
    busy = _state(n_vials=16, polygons=[TRI, SQUARE])
    panel = lambda st: render_frame(frame, st)[:, :LVS.PANEL_WIDTH]
    assert not np.array_equal(panel(bare), panel(busy))       # progress is shown
    frozen = SelectorState(n_vials=16, face="A")
    frozen.toggle_freeze()
    assert not np.array_equal(panel(bare), panel(frozen))     # PLAYING/FROZEN is shown


def test_render_frame_draws_something_for_finished_and_in_progress_vials():
    blank = np.zeros((H, W), np.uint8)
    ink = lambda img: int(np.count_nonzero(_image_region(img)))

    empty = render_frame(blank, SelectorState(n_vials=4))
    done = render_frame(blank, _state(n_vials=4, polygons=[PENTAGON]))
    s = SelectorState(n_vials=4)
    s.add_vertex(*PENTAGON[0])
    dot = render_frame(blank, s)

    assert ink(empty) == 0
    assert ink(done) > 0            # outline + the vial number
    assert ink(dot) > 0             # the first vertex marker
    assert ink(done) > ink(dot)


def test_render_frame_marks_the_frozen_state_visibly():
    blank = np.zeros((H, W), np.uint8)
    live = render_frame(blank, SelectorState(n_vials=4))
    s = SelectorState(n_vials=4)
    s.toggle_freeze()
    frozen = render_frame(blank, s)
    assert not np.array_equal(live, frozen)
    # the border sits just inside the picture's own edge, not over the panel
    assert _image_region(frozen)[H - 2, W // 2].tolist() == list(LVS.COLOR_FROZEN)


def test_render_frame_accepts_a_colour_frame_too():
    canvas = render_frame(np.zeros((H, W, 3), np.uint8), _state(polygons=[TRI]))
    assert canvas.shape == (H, LVS.PANEL_WIDTH + W, 3)


def test_view_scale_never_enlarges_and_shrinks_to_fit():
    assert view_scale((640, 480), (1280 + LVS.PANEL_WIDTH, 960)) == 1.0
    assert view_scale((2560, 1920), (1280 + LVS.PANEL_WIDTH, 960)) == pytest.approx(0.5)


def test_render_frame_scales_the_canvas_and_the_polygons_together():
    canvas = render_frame(np.zeros((400, 400), np.uint8), _state(polygons=[TRI]), scale=0.5)
    assert canvas.shape == (200, LVS.PANEL_WIDTH + 200, 3)


# =========================================================================================
# 7. THE DRIVER LOOP, with the cv2 window stubbed out (still no display needed)
# =========================================================================================
class _SyntheticSource(FrameSource):
    """A `FrameSource` that yields a DIFFERENT frame every read, so "is the feed live?" is testable."""

    def __init__(self, n_frames=10 ** 6, size=(W, H)):
        self.n_frames = n_frames
        self.size = size
        self.n_read = 0
        self.opens = 0
        self.closes = 0

    def open(self):
        self.opens += 1

    def read(self):
        if self.n_read >= self.n_frames:
            return None
        img = np.full((self.size[1], self.size[0]), (self.n_read * 9) % 250, np.uint8)
        frame = Frame(image=img, index=self.n_read, t_monotonic=float(self.n_read), t_wall_iso="t")
        self.n_read += 1
        return frame

    def close(self):
        self.closes += 1

    @property
    def fps(self):
        return 30.0

    @property
    def frame_size(self):
        return self.size


class _FakeWindow:
    """Stands in for the whole cv2 highgui surface `select_vials_live` touches.

    `script` is a list of ("click", x, y) / ("key", code) events. Clicks are fired at the driver's
    own mouse callback from inside the `waitKeyEx` stub -- exactly where a real event arrives --
    so the test proves the WIRING (callback -> state -> returned polygons), not just the state.
    """

    def __init__(self, monkeypatch, script):
        self.script = list(script)
        self.shown = []
        self.callback = None
        self.destroyed = []
        monkeypatch.setattr(LVS, "require_gui", lambda *_a, **_k: None)
        for name, fn in [
            ("namedWindow", lambda *a, **k: None),
            ("setMouseCallback", lambda _w, cb, *a: setattr(self, "callback", cb)),
            ("imshow", lambda _w, img: self.shown.append(img.copy())),
            ("waitKeyEx", self._next),
            ("waitKey", lambda *_a: -1),
            ("getWindowProperty", lambda *_a: 1.0),
            ("destroyWindow", lambda w: self.destroyed.append(w)),
        ]:
            monkeypatch.setattr(cv2, name, fn)

    def _next(self, *_a):
        if not self.script:
            return ord("q")                       # never hang: an unscripted loop finishes early
        event = self.script.pop(0)
        if event[0] == "click":
            self.callback(cv2.EVENT_LBUTTONDOWN, event[1], event[2], 0, None)
            return -1                             # a click is an idle frame as far as the keys go
        if event[0] == "idle":
            return -1
        return event[1]


def _clicks(polygon, scale=1.0):
    """Image coords -> the CANVAS coords a real click carries.

    A real mouse event arrives in canvas space: the instruction panel occupies the left
    `PANEL_WIDTH` columns and the frame is drawn at `scale` to its right. Modelling that here is
    what makes these tests prove the driver undoes BOTH transforms.
    """
    return [("click", LVS.PANEL_WIDTH + x * scale, y * scale) for x, y in polygon]


def _draw(polygon):
    return _clicks(polygon) + [("key", 13)]


def test_driver_returns_exactly_the_polygons_that_were_clicked(monkeypatch):
    """END TO END through the real loop: synthetic clicks + ENTER -> the returned polygons."""
    win = _FakeWindow(monkeypatch, _draw(TRI) + _draw(SQUARE) + _draw(PENTAGON))
    source = _SyntheticSource()

    out = select_vials_live(source, n_vials=3, face="B")

    assert out == [TRI, SQUARE, PENTAGON]
    assert source.opens == 1                       # the driver opens it...
    assert source.closes == 0                      # ...and leaves closing to the caller
    assert win.destroyed == [LVS.DEFAULT_WINDOW]


def test_driver_keeps_the_feed_live_while_clicks_are_collected(monkeypatch):
    """The whole point: frames keep arriving and being re-shown DURING the clicking."""
    win = _FakeWindow(monkeypatch, _draw(TRI) + _draw(SQUARE))
    source = _SyntheticSource()

    select_vials_live(source, n_vials=2)

    assert source.n_read >= len(TRI) + len(SQUARE)      # a fresh frame per iteration
    assert len(win.shown) >= source.n_read             # every one of them re-shown
    # The frames really do differ, i.e. this is a video and not one frozen still.
    backgrounds = {int(img[H - 1, LVS.PANEL_WIDTH, 0]) for img in win.shown}
    assert len(backgrounds) > 1


def test_space_freezes_the_feed_and_space_again_resumes_it(monkeypatch):
    frozen_script = ([("key", 32)] + [("idle",)] * 5 + _clicks(TRI)
                     + [("key", 32), ("idle",), ("key", 13)])
    _FakeWindow(monkeypatch, frozen_script)
    source = _SyntheticSource()

    out = select_vials_live(source, n_vials=1)

    assert out == [TRI]
    # 1 frame before SPACE + 1 after unfreezing (+/- the loop's own bookkeeping): the point is
    # that the ~10 iterations spent frozen did NOT consume frames.
    assert source.n_read <= 4


def test_a_click_while_frozen_still_lands_on_the_frozen_image(monkeypatch):
    _FakeWindow(monkeypatch, [("key", 32)] + _draw(TRI))
    source = _SyntheticSource()
    assert select_vials_live(source, n_vials=1) == [TRI]


def test_driver_quits_early_and_returns_the_finished_vials_only(monkeypatch):
    _FakeWindow(monkeypatch, _draw(TRI) + _clicks(SQUARE) + [("key", ord("q"))])
    out = select_vials_live(_SyntheticSource(), n_vials=16)
    assert out == [TRI]                    # the half-drawn square is not a vial


def test_driver_undo_vial_and_backspace_reach_the_state_through_the_loop(monkeypatch):
    script = (_draw(TRI) + _draw(SQUARE)
              + [("key", ord("u"))]        # re-open the square
              + [("key", 8)]               # drop its last vertex
              + _clicks([[120, 60]]) + [("key", 13)]
              + [("key", ord("q"))])       # 3 vials asked for, 2 drawn -> finish early
    _FakeWindow(monkeypatch, script)
    out = select_vials_live(_SyntheticSource(), n_vials=3)
    assert out == [TRI, SQUARE[:3] + [[120, 60]]]


def test_driver_maps_clicks_back_through_the_display_scale(monkeypatch):
    """On a frame too big for the screen the canvas is shrunk; the CLICKS must not be.

    Both transforms are exercised at once: the panel offset AND the display scale.
    """
    P = LVS.PANEL_WIDTH
    _FakeWindow(monkeypatch, [("click", P + 100, 50), ("click", P + 300, 50),
                              ("click", P + 200, 250), ("key", 13)])
    # Pinned, because the real limit is now measured from whatever desktop runs the suite -- the
    # expected coordinates below must not depend on the test machine's screen.
    monkeypatch.setattr(LVS, "screen_view_limit", lambda *a, **k: (1280 + P, 960))
    source = _SyntheticSource(size=(2560, 1920))          # -> scale 0.5

    out = select_vials_live(source, n_vials=1)

    assert out == [[[200, 100], [600, 100], [400, 500]]]  # image px, not canvas px


def test_a_click_on_the_instruction_panel_is_not_a_vertex(monkeypatch):
    """The panel is not the picture: clicking it must not silently drop a corner on the frame."""
    _FakeWindow(monkeypatch, [("click", 10, 40)] + _clicks(TRI) + [("key", 13)])
    out = select_vials_live(_SyntheticSource(), n_vials=1)
    assert out == [TRI]                                   # the panel click contributed nothing


def test_driver_holds_the_last_frame_at_end_of_video(monkeypatch):
    """A short clip must not end the session -- the operator is still drawing on it."""
    _FakeWindow(monkeypatch, [("idle",), ("idle",)] + _draw(TRI))
    source = _SyntheticSource(n_frames=2)
    assert select_vials_live(source, n_vials=1) == [TRI]


def test_driver_raises_when_the_source_yields_nothing(monkeypatch):
    _FakeWindow(monkeypatch, [("idle",)])
    with pytest.raises(RuntimeError, match="no frames"):
        select_vials_live(_SyntheticSource(n_frames=0), n_vials=1)


def test_driver_stops_when_the_window_is_closed(monkeypatch):
    """Closing the window with its X counts as finishing early, not as losing the work."""
    win = _FakeWindow(monkeypatch, _draw(TRI))
    # The window "disappears" once the scripted events run out, mid-session (16 vials asked for).
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_a: 1.0 if win.script else 0.0)

    out = select_vials_live(_SyntheticSource(), n_vials=16)

    assert out == [TRI]
    assert win.destroyed == [LVS.DEFAULT_WINDOW]


def test_driver_passes_every_new_frame_to_on_frame(monkeypatch):
    _FakeWindow(monkeypatch, _draw(TRI))
    seen = []
    select_vials_live(_SyntheticSource(), n_vials=1, on_frame=seen.append)
    assert seen and all(img.shape == (H, W) for img in seen)


def test_driver_refuses_to_run_without_gui_support(monkeypatch):
    monkeypatch.setattr(LVS, "require_gui",
                        lambda *_a, **_k: (_ for _ in ()).throw(SystemExit(2)))
    with pytest.raises(SystemExit):
        select_vials_live(_SyntheticSource(), n_vials=1)


# =========================================================================================
# 8. types.VialROI.polygon
# =========================================================================================
def _vial(vid=1, row=0, col=0, x=10, y=10, w=40, h=60, present=True, quad=None, polygon=None):
    return VialROI(id=vid, row=row, col=col, x=x, y=y, w=w, h=h, present=present,
                   quad=quad, polygon=polygon)


def _face(vials, name="A", mask_path="illum_mask_A.png"):
    return FaceCalibration(name=name, vials=list(vials), illum_mask_path=mask_path)


def test_polygon_normalises_any_point_sequence_to_int_lists():
    v = _vial(polygon=[(1.4, 2.6), np.array([3, 4]), [5.5, 6], (7, 8), (9, 10)])
    assert v.polygon == [[1, 3], [3, 4], [6, 6], [7, 8], [9, 10]]


@pytest.mark.parametrize("n", [0, 1, 2])
def test_polygon_rejects_fewer_than_three_points(n):
    with pytest.raises(ValueError, match="at least 3"):
        _vial(polygon=[[i, i] for i in range(n)])


def test_polygon_accepts_any_count_from_three_up():
    for n in (3, 4, 5, 12, 40):
        poly = [[i, i * 2] for i in range(n)]
        assert len(_vial(polygon=poly).polygon) == n


def test_the_quad_field_still_works_and_still_demands_four_corners():
    assert _vial(quad=[[0, 0], [1, 0], [1, 1], [0, 1]]).quad == [[0, 0], [1, 0], [1, 1], [0, 1]]
    with pytest.raises(ValueError, match="4 corners"):
        _vial(quad=[[0, 0], [1, 1], [2, 2]])


def test_a_vial_may_carry_both_a_polygon_and_a_quad():
    v = _vial(quad=[[0, 0], [1, 0], [1, 1], [0, 1]], polygon=TRI)
    assert v.quad is not None and v.polygon == TRI
    assert C.vial_shape(v) == TRI          # polygon wins
    assert C.vial_quad(v) == [[0, 0], [1, 0], [1, 1], [0, 1]]   # quad accessor is unaffected


def test_vial_shape_falls_back_quad_then_bbox():
    assert C.vial_shape(_vial(quad=[[0, 0], [1, 0], [1, 1], [0, 1]])) == [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert C.vial_shape(_vial(x=10, y=20, w=30, h=40)) == C.quad_from_bbox((10, 20, 30, 40))


def test_calibration_json_round_trips_polygons(tmp_path):
    calib = Calibration(image_width=W, image_height=H,
                        faces={"A": _face([_vial(1, polygon=PENTAGON), _vial(2)])})
    path = str(tmp_path / "calibration.json")
    calib.to_json(path)
    back = Calibration.from_json(path)
    assert back.faces["A"].vials[0].polygon == PENTAGON
    assert back.faces["A"].vials[1].polygon is None


def test_a_bundle_written_before_polygons_existed_still_loads(tmp_path):
    """calibration.json from an older version has no `polygon` key at all."""
    payload = {
        "image_width": W, "image_height": H, "created": "", "notes": "",
        "faces": {"A": {"name": "A", "illum_mask_path": "illum_mask_A.png", "marker": None,
                        "vials": [{"id": 1, "row": 0, "col": 0, "x": 10, "y": 10,
                                   "w": 40, "h": 60, "present": True}]}},
    }
    path = str(tmp_path / "calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    v = Calibration.from_json(path).faces["A"].vials[0]
    assert v.polygon is None and v.quad is None
    assert (v.x, v.y, v.w, v.h) == (10, 10, 40, 60)


# =========================================================================================
# 9. calibration.build_calibration_from_polygons
# =========================================================================================
def _frame():
    """A frame with two bright rows of vials, so a derived illum mask is not degenerate."""
    gray = np.full((H, W), 15, np.uint8)
    gray[10:60, 5:195] = 210
    gray[120:180, 5:195] = 210
    return gray


def _build(polygons, **kw):
    return C.build_calibration_from_polygons(polygons, "A", _frame(), (W, H), **kw)


def test_build_from_polygons_numbers_vials_in_draw_order():
    face_cal, _mask, _overlay = _build([TRI, SQUARE, PENTAGON])
    assert [v.id for v in face_cal.vials] == [1, 2, 3]
    assert [v.polygon for v in face_cal.vials] == [TRI, SQUARE, PENTAGON]
    assert face_cal.name == "A"


def test_build_from_polygons_sets_the_bbox_to_the_polygon_bounds():
    face_cal, _m, _o = _build([TRI, PENTAGON])
    for v, poly in zip(face_cal.vials, [TRI, PENTAGON]):
        assert (v.x, v.y, v.w, v.h) == C.bbox_from_quad(poly)


def test_build_from_polygons_marks_every_vial_present():
    """A slot the operator did not want measured is a slot they did not draw."""
    face_cal, _m, _o = _build([TRI, SQUARE, PENTAGON])
    assert all(v.present for v in face_cal.vials)


def test_build_from_polygons_labels_rows_by_the_frame_midline():
    face_cal, _m, _o = _build([TRI, SQUARE, PENTAGON])   # TRI/SQUARE up top, PENTAGON below
    assert [v.row for v in face_cal.vials] == [0, 0, 1]


def test_build_from_polygons_labels_columns_left_to_right_within_each_row():
    """Draw order is deliberately NOT left-to-right: the labels must come from position."""
    right = [[150, 20], [190, 20], [190, 50]]
    middle = [[80, 20], [120, 20], [120, 50]]
    left = [[10, 20], [50, 20], [50, 50]]
    lower = [[10, 150], [50, 150], [50, 175]]
    face_cal, _m, _o = _build([right, middle, left, lower])
    assert [v.id for v in face_cal.vials] == [1, 2, 3, 4]      # ids still follow draw order
    assert [(v.row, v.col) for v in face_cal.vials] == [(0, 2), (0, 1), (0, 0), (1, 0)]


def test_build_from_polygons_derives_a_lit_mask_when_none_is_given():
    _face_cal, mask, _o = _build([TRI])
    assert mask.shape == (H, W) and mask.dtype == np.uint8
    assert mask[30, 100] == 255          # inside the bright band
    assert mask[100, 100] == 0           # the dark gap between the rows
    assert 0 < int((mask > 0).sum()) < mask.size


def test_the_derived_mask_treats_everything_inside_a_drawn_vial_as_trackable():
    """A fly resting in a tube at selection time must not punch a permanent hole in that vial.

    The operator said "this region is a vial"; the derived mask must not then veto part of it
    because a dark blob happened to be sitting there when they clicked.
    """
    gray = _frame()
    dark_vial = [[70, 130], [130, 130], [130, 170], [70, 170]]
    gray[135:165, 75:125] = 5                      # the whole slot is dark in this frame
    face_cal, mask, _o = C.build_calibration_from_polygons([dark_vial], "A", gray, (W, H))
    assert mask[150, 100] == 255                   # ...and still fully measurable
    _b, sub = _pipeline_submask_from_face(face_cal, mask)
    assert int(sub.sum()) == pytest.approx(C.polygon_area(dark_vial), rel=0.05)


def _pipeline_submask_from_face(face_cal, mask, tmp_dir=None):
    """Run one built face through the pipeline and return its first vial's (bbox, submask)."""
    import tempfile
    out = tmp_dir or tempfile.mkdtemp()
    calib = Calibration(image_width=W, image_height=H, faces={"A": face_cal})
    C.save_calibration(calib, mask, out)
    return _null_pipeline(C.load_calibration(out))._face_active["A"][1]


def test_build_from_polygons_uses_a_supplied_mask_unchanged():
    supplied = np.zeros((H, W), np.uint8)
    supplied[0:100, :] = 255
    _face_cal, mask, _o = _build([TRI], illum_mask=supplied)
    assert mask is supplied


def test_build_from_polygons_returns_an_overlay_image():
    _face_cal, _mask, overlay = _build([TRI, PENTAGON])
    assert overlay.shape == (H, W, 3) and overlay.dtype == np.uint8


def test_build_from_polygons_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="at least one vial"):
        _build([])


def test_build_from_polygons_survives_a_save_load_round_trip(tmp_path):
    face_cal, mask, overlay = _build([TRI, SQUARE, PENTAGON])
    calib = Calibration(image_width=W, image_height=H, faces={"A": face_cal})
    out = str(tmp_path / "bundle")
    C.save_calibration(calib, mask, out, overlay=overlay)

    back = C.load_calibration(out)
    vials = back.faces["A"].vials
    assert [v.polygon for v in vials] == [TRI, SQUARE, PENTAGON]
    assert [(v.x, v.y, v.w, v.h) for v in vials] == [C.bbox_from_quad(p)
                                                     for p in (TRI, SQUARE, PENTAGON)]
    assert all(v.present for v in vials)
    assert os.path.exists(os.path.join(out, "illum_mask_A.png"))
    assert os.path.exists(os.path.join(out, "overlay_A.png"))
    assert cv2.imread(back.faces["A"].illum_mask_path, cv2.IMREAD_GRAYSCALE) is not None


# =========================================================================================
# 10. THE MEASUREMENT PATH: the pipeline honours `polygon`
# =========================================================================================
class _NullSource:
    fps = 10.0

    def open(self): pass

    def read(self): return None

    def close(self): pass


class _NullLogger:
    run_id = "t"

    def log_activity(self, records): pass

    def log_event(self, record): pass

    def close(self): pass


def _null_pipeline(calib):
    """A `TrackerPipeline` built over `calib` that never reads a frame -- just its precomputed ROIs."""
    from flygym_tracker.config import load_config
    from flygym_tracker.pipeline import TrackerPipeline

    config = load_config(overrides={
        "activity": {"pixel_threshold": 10.0},
        "rotation": {"enter_threshold": 40.0, "exit_threshold": 15.0},
    })
    return TrackerPipeline(config, calib, _NullSource(), _NullLogger())


def _pipeline_submask(tmp_path, vial, mask):
    """Build a one-vial pipeline over `mask` and return (bbox, effective bool submask)."""
    mask_path = str(tmp_path / ("illum_%s.png" % vial.id))
    cv2.imwrite(mask_path, mask)
    calib = Calibration(image_width=mask.shape[1], image_height=mask.shape[0],
                        faces={"A": _face([vial], mask_path=mask_path)})
    return _null_pipeline(calib)._face_active["A"][int(vial.id)]


def _lit_block_mask():
    mask = np.zeros((H, W), np.uint8)
    mask[10:70, 10:90] = 255            # an 80 x 60 lit block
    return mask


def test_pipeline_measures_the_polygon_not_the_bounding_box(tmp_path):
    """A polygon that cuts a corner off the bbox must measure FEWER lit pixels."""
    mask = _lit_block_mask()
    corner_cut = [[10, 10], [90, 10], [90, 50], [50, 70], [10, 70]]   # 5 points, not 4

    _b1, plain = _pipeline_submask(tmp_path, _vial(1, x=10, y=10, w=80, h=60), mask)
    _b2, cut = _pipeline_submask(
        tmp_path, _vial(2, x=10, y=10, w=80, h=60, polygon=corner_cut), mask)

    assert plain.shape == cut.shape == (60, 80)
    assert int(plain.sum()) == 80 * 60 == 4800
    assert int(cut.sum()) < int(plain.sum())
    # The removed pixels are exactly the clipped corner: bottom-RIGHT is out, bottom-LEFT is in.
    assert not cut[58, 78]
    assert cut[58, 2]
    # Analytic area of the cut polygon, for a number rather than an impression.
    assert C.polygon_area(corner_cut) == pytest.approx(4400.0)
    assert int(cut.sum()) == pytest.approx(4400, rel=0.03)


def test_pipeline_polygon_beats_quad(tmp_path):
    """PRECEDENCE: polygon > quad > bbox. Both set -> the polygon is what gets measured."""
    mask = np.full((H, W), 255, np.uint8)
    left_half = [[10, 10], [50, 10], [50, 70], [10, 70]]
    full_quad = [[10, 10], [90, 10], [90, 70], [10, 70]]

    _b, both = _pipeline_submask(
        tmp_path, _vial(1, x=10, y=10, w=80, h=60, quad=full_quad, polygon=left_half), mask)
    _b2, quad_only = _pipeline_submask(
        tmp_path, _vial(2, x=10, y=10, w=80, h=60, quad=full_quad), mask)

    assert int(both.sum()) < int(quad_only.sum())
    assert not both[:, 45:].any()          # nothing beyond the polygon's right edge


def test_pipeline_polygon_is_intersected_with_the_illum_mask_not_substituted(tmp_path):
    mask = np.zeros((H, W), np.uint8)
    mask[10:70, 10:50] = 255                        # lit on the LEFT half of the bbox only
    whole_box = [[10, 10], [90, 10], [90, 70], [10, 70]]
    _b, sub = _pipeline_submask(tmp_path, _vial(1, x=10, y=10, w=80, h=60, polygon=whole_box), mask)
    assert int(sub.sum()) == 40 * 60
    assert not sub[:, 40:].any()


def test_pipeline_crop_follows_the_polygon_when_the_stored_bbox_is_stale(tmp_path):
    mask = np.full((H, W), 255, np.uint8)
    bbox, sub = _pipeline_submask(
        tmp_path, _vial(1, x=0, y=0, w=200, h=200, polygon=TRI), mask)
    assert bbox == C.bbox_from_quad(TRI)
    assert sub.shape == (bbox[3], bbox[2])


def test_pipeline_mask_without_polygon_or_quad_is_byte_identical_to_before(tmp_path):
    """THE backward-compatibility invariant: an old bundle must measure exactly as it always did."""
    mask = _lit_block_mask()
    mask[30:40, 40:50] = 0                     # a hole, to prove the illum mask still governs
    bbox, sub = _pipeline_submask(tmp_path, _vial(1, x=10, y=10, w=80, h=60), mask)
    assert bbox == (10, 10, 80, 60)
    assert np.array_equal(sub, mask[10:70, 10:90] == 255)


def test_a_polygon_that_is_just_the_bbox_measures_exactly_the_bbox(tmp_path):
    """Drawing a rectangle by hand must not change a single pixel versus no polygon at all."""
    mask = _lit_block_mask()
    mask[30:40, 40:50] = 0
    box = (10, 10, 80, 60)
    _b1, plain = _pipeline_submask(tmp_path, _vial(1, x=10, y=10, w=80, h=60), mask)
    _b2, drawn = _pipeline_submask(
        tmp_path, _vial(2, x=10, y=10, w=80, h=60, polygon=C.quad_from_bbox(box)), mask)
    assert np.array_equal(plain, drawn)


def test_registration_shifts_an_n_vertex_polygon_with_its_bbox(tmp_path):
    """The polygon must track a registration shift, or it would drift off its own crop."""
    mask = np.full((H, W), 255, np.uint8)
    mask_path = str(tmp_path / "illum_shift.png")
    cv2.imwrite(mask_path, mask)
    vial = _vial(1, x=0, y=0, w=200, h=200, polygon=PENTAGON)
    calib = Calibration(image_width=W, image_height=H,
                        faces={"A": _face([vial], mask_path=mask_path)})

    pipe = _null_pipeline(calib)
    before_bbox, before_sub = pipe._face_active["A"][1]
    pipe._apply_registration("A", 3.0, -2.0)
    after_bbox, after_sub = pipe._face_active["A"][1]

    assert after_bbox == (before_bbox[0] + 3, before_bbox[1] - 2, before_bbox[2], before_bbox[3])
    assert int(after_sub.sum()) == int(before_sub.sum())     # same shape, just moved


# =========================================================================================
# 11. ONE drawing -> BOTH faces, identical coordinates
# =========================================================================================
def _build_two(polygons, faces=("A", "B"), **kw):
    return C.build_two_face_calibration_from_polygons(polygons, _frame(), (W, H),
                                                      faces=faces, **kw)


def test_two_face_build_gives_both_faces_the_very_same_coordinates():
    """The owner's constraint: one drawing, and the other side's vials are in the same places."""
    calib, _masks, _overlays = _build_two([TRI, SQUARE, PENTAGON])
    assert sorted(calib.faces) == ["A", "B"]
    a = [v.polygon for v in calib.faces["A"].vials]
    b = [v.polygon for v in calib.faces["B"].vials]
    assert a == b == [TRI, SQUARE, PENTAGON]        # verbatim: no mirroring, no rescaling
    assert [(v.x, v.y, v.w, v.h) for v in calib.faces["A"].vials] == \
           [(v.x, v.y, v.w, v.h) for v in calib.faces["B"].vials]


def test_two_face_build_numbers_vials_1_to_16_on_each_face():
    polygons = [[[10 * i, 10], [10 * i + 8, 10], [10 * i + 8, 40]] for i in range(1, 17)]
    calib, _m, _o = _build_two(polygons)
    for name in ("A", "B"):
        assert [v.id for v in calib.faces[name].vials] == list(range(1, 17))


def test_two_face_build_yields_global_ids_1_to_32_through_the_pipeline(tmp_path):
    """`face_index * 16 + local_id` must come out 1..16 and 17..32, with no collision."""
    from flygym_tracker.config import load_config
    from flygym_tracker.pipeline import TrackerPipeline

    polygons = [[[10 * i, 10], [10 * i + 8, 10], [10 * i + 8, 40]] for i in range(1, 17)]
    calib, masks, overlays = _build_two(polygons)
    out = str(tmp_path / "bundle")
    C.save_calibration(calib, masks, out, overlay=overlays)

    pipe = _null_pipeline(C.load_calibration(out))
    assert sorted(pipe._face_active["A"]) == list(range(1, 17))
    assert sorted(pipe._face_active["B"]) == list(range(17, 33))


def test_two_face_build_refuses_more_vials_than_the_global_id_scheme_allows():
    polygons = [[[i, 1], [i + 1, 1], [i + 1, 5]] for i in range(17)]
    with pytest.raises(ValueError, match="global-id"):
        _build_two(polygons)
    # ...but a SINGLE-face bundle has no such collision to worry about.
    calib, _m, _o = _build_two(polygons, faces=("A",))
    assert len(calib.faces["A"].vials) == 17


def test_two_face_build_writes_a_mask_and_overlay_per_face(tmp_path):
    calib, masks, overlays = _build_two([TRI, SQUARE])
    out = str(tmp_path / "bundle")
    C.save_calibration(calib, masks, out, overlay=overlays)
    for face in ("A", "B"):
        assert os.path.exists(os.path.join(out, "illum_mask_%s.png" % face))
        assert os.path.exists(os.path.join(out, "overlay_%s.png" % face))
    with open(os.path.join(out, "calibration.json"), encoding="utf-8") as f:
        raw = json.load(f)
    # Mask paths stay bare filenames or the bundle stops being movable between machines.
    assert raw["faces"]["B"]["illum_mask_path"] == "illum_mask_B.png"


def test_two_face_build_records_when_it_was_saved():
    calib, _m, _o = _build_two([TRI])
    assert calib.created                       # needed for the "saved <when>" prompt
    assert "identical coordinates" in calib.notes


# =========================================================================================
# 12. Reuse: saved_selection / prompt / load_or_select_vials
# =========================================================================================
def _save_bundle(tmp_path, polygons=(TRI, SQUARE), faces=("A", "B"), name="bundle"):
    calib, masks, overlays = _build_two(list(polygons), faces=faces)
    out = str(tmp_path / name)
    C.save_calibration(calib, masks, out, overlay=overlays)
    return out


def test_saved_selection_reads_back_what_was_drawn(tmp_path):
    out = _save_bundle(tmp_path, [TRI, SQUARE, PENTAGON])
    saved = C.saved_selection(out)
    assert saved is not None
    assert saved.polygons == [TRI, SQUARE, PENTAGON]
    assert saved.n_vials == 3
    assert saved.faces == ["A", "B"]
    assert saved.image_size == (W, H)
    assert saved.created


@pytest.mark.parametrize("prepare", ["missing", "corrupt", "no_vials"])
def test_saved_selection_offers_nothing_when_there_is_nothing_to_offer(tmp_path, prepare):
    out = str(tmp_path / "bundle")
    os.makedirs(out, exist_ok=True)
    if prepare == "corrupt":
        with open(os.path.join(out, "calibration.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
    elif prepare == "no_vials":
        Calibration(image_width=W, image_height=H,
                    faces={"A": _face([])}).to_json(os.path.join(out, "calibration.json"))
    assert C.saved_selection(out) is None


def test_an_older_box_bundle_is_still_offered_rather_than_silently_redrawn(tmp_path):
    """A pre-existing calibration must not be invisible just because it predates the selector.

    Regression: `saved_selection` keyed on "has polygons", so a rig that was already calibrated
    the old way looked like an empty folder and the operator was sent to redraw all 16 vials
    without ever being told a bundle was sitting there.
    """
    out = str(tmp_path / "bundle")
    os.makedirs(out, exist_ok=True)
    quad = [[0, 0], [9, 0], [9, 9], [0, 9]]
    Calibration(image_width=W, image_height=H,
                faces={"A": _face([_vial(1, quad=quad), _vial(2, quad=quad)])}
                ).to_json(os.path.join(out, "calibration.json"))

    saved = C.saved_selection(out)
    assert saved is not None
    assert saved.kind == "boxes" and not saved.hand_drawn
    assert saved.n_vials == 2
    assert saved.polygons[0] == quad          # usable shapes, not an empty shell


def test_a_bbox_only_bundle_falls_back_to_its_rectangle(tmp_path):
    out = str(tmp_path / "bundle")
    os.makedirs(out, exist_ok=True)
    Calibration(image_width=W, image_height=H, faces={"A": _face([_vial(1)])}
                ).to_json(os.path.join(out, "calibration.json"))
    saved = C.saved_selection(out)
    assert saved is not None and saved.kind == "boxes"
    assert len(saved.polygons[0]) == 4


def test_the_reuse_question_names_the_count_and_the_time(tmp_path):
    out = _save_bundle(tmp_path, [TRI, SQUARE])
    question = LVS.reuse_question(C.saved_selection(out))
    assert question.startswith("Found saved vial positions (2 vials, drawn by hand, saved ")
    assert question.endswith("Load them? [Y/n]: ")
    assert re.search(r"saved \d{4}-\d{2}-\d{2} \d{2}:\d{2}\)", question)


def test_the_question_for_auto_detected_boxes_says_so_and_defaults_to_no(tmp_path):
    """The detector was retired for misaligning ROIs; reusing its output must be a deliberate act."""
    out = str(tmp_path / "bundle")
    os.makedirs(out, exist_ok=True)
    Calibration(image_width=W, image_height=H, faces={"A": _face([_vial(1)])}
                ).to_json(os.path.join(out, "calibration.json"))
    saved = C.saved_selection(out)

    question = LVS.reuse_question(saved)
    assert "AUTO-DETECTED" in question
    assert question.endswith("[y/N]: ")           # capital N: declining is the default
    assert LVS.prompt_reuse(saved, input_fn=lambda _p: "") is False      # ENTER = do not reuse
    assert LVS.prompt_reuse(saved, input_fn=lambda _p: "y") is True      # only an explicit yes


def test_without_a_terminal_the_saved_positions_are_used_instead_of_asking(tmp_path, monkeypatch,
                                                                          capsys):
    """A scripted/unattended start has nobody to answer and nobody to draw -- reuse and say so."""
    saved = C.saved_selection(_save_bundle(tmp_path, [TRI]))
    monkeypatch.setattr(LVS, "_stdin_is_interactive", lambda: False)
    monkeypatch.setattr("builtins.input",
                        lambda _p: pytest.fail("must not prompt without a terminal"))

    assert LVS.prompt_reuse(saved) is True
    assert "no terminal to ask on" in capsys.readouterr().out


@pytest.mark.parametrize("answer,expected", [
    ("", True), ("y", True), ("Y", True), ("yes", True), ("  ", True),
    ("n", False), ("N", False), ("no", False), ("No", False),
])
def test_prompt_reuse_defaults_to_yes_on_enter(tmp_path, answer, expected):
    saved = C.saved_selection(_save_bundle(tmp_path, [TRI]))
    assert LVS.prompt_reuse(saved, input_fn=lambda _p: answer) is expected


def test_prompt_reuse_keeps_the_saved_positions_when_there_is_no_terminal(tmp_path):
    saved = C.saved_selection(_save_bundle(tmp_path, [TRI]))

    def no_stdin(_prompt):
        raise EOFError

    assert LVS.prompt_reuse(saved, input_fn=no_stdin) is True


def _stub_selector(monkeypatch, polygons, frame=None, raises=None):
    """Replace the interactive selector with a scripted one; record how it was called."""
    calls = {"n_calls": 0}
    img = _frame() if frame is None else frame

    def fake(source, n_vials=16, face="A", window="", on_frame=None, **kw):
        calls.update(source=source, n_vials=n_vials, face=face)
        calls["n_calls"] += 1
        if raises is not None:
            raise raises
        if on_frame is not None:
            on_frame(img)
        return [list(map(list, p)) for p in polygons]

    monkeypatch.setattr(LVS, "select_vials_live", fake)
    return calls


def test_load_or_select_draws_and_saves_when_nothing_is_stored(tmp_path, monkeypatch):
    calls = _stub_selector(monkeypatch, [TRI, SQUARE])
    out = str(tmp_path / "bundle")

    result = LVS.load_or_select_vials(_SyntheticSource(), out, n_vials=2)

    assert calls["n_calls"] == 1                      # it drew...
    assert result.reused is False
    assert result.polygons == [TRI, SQUARE]
    # ...and saved, so the very next round can offer it back.
    assert C.saved_selection(out).polygons == [TRI, SQUARE]
    assert sorted(result.calibration.faces) == ["A", "B"]


def test_load_or_select_does_not_prompt_when_nothing_is_stored(tmp_path, monkeypatch):
    _stub_selector(monkeypatch, [TRI])

    def no_prompting(_p):
        raise AssertionError("the operator must not be asked when there is nothing saved")

    LVS.load_or_select_vials(_SyntheticSource(), str(tmp_path / "b"), n_vials=1,
                             input_fn=no_prompting)


def test_load_or_select_reuses_saved_positions_and_skips_drawing_entirely(tmp_path, monkeypatch,
                                                                         capsys):
    out = _save_bundle(tmp_path, [TRI, SQUARE, PENTAGON])
    calls = _stub_selector(monkeypatch, [[[1, 1], [2, 2], [3, 3]]])
    asked = []

    result = LVS.load_or_select_vials(_SyntheticSource(), out, n_vials=3,
                                      input_fn=lambda p: asked.append(p) or "")

    assert calls["n_calls"] == 0                      # NO drawing happened
    assert result.reused is True
    assert result.polygons == [TRI, SQUARE, PENTAGON]
    assert sorted(result.calibration.faces) == ["A", "B"]
    assert asked and "Load them? [Y/n]" in asked[0]
    assert "using the saved vial positions" in capsys.readouterr().out


def test_declining_the_prompt_goes_straight_to_drawing(tmp_path, monkeypatch):
    out = _save_bundle(tmp_path, [TRI, SQUARE, PENTAGON])
    calls = _stub_selector(monkeypatch, [SQUARE])

    result = LVS.load_or_select_vials(_SyntheticSource(), out, n_vials=1, input_fn=lambda _p: "n")

    assert calls["n_calls"] == 1
    assert result.reused is False
    assert result.polygons == [SQUARE]
    assert C.saved_selection(out).polygons == [SQUARE]      # the redraw replaced it


def test_reuse_true_and_false_skip_the_question(tmp_path, monkeypatch):
    out = _save_bundle(tmp_path, [TRI, SQUARE])

    def no_prompting(_p):
        raise AssertionError("--reuse/--redraw must not ask")

    _stub_selector(monkeypatch, [PENTAGON])
    assert LVS.load_or_select_vials(_SyntheticSource(), out, reuse=True,
                                    input_fn=no_prompting).reused is True
    assert LVS.load_or_select_vials(_SyntheticSource(), out, n_vials=1, reuse=False,
                                    input_fn=no_prompting).polygons == [PENTAGON]


def test_load_or_select_saves_nothing_when_the_operator_drew_nothing(tmp_path, monkeypatch):
    _stub_selector(monkeypatch, [])
    out = str(tmp_path / "bundle")
    result = LVS.load_or_select_vials(_SyntheticSource(), out, n_vials=16)
    assert result.polygons == [] and result.calibration is None
    assert C.saved_selection(out) is None


def test_a_saved_selection_round_trips_through_a_new_session(tmp_path, monkeypatch):
    """Session 1 draws; session 2 (a fresh process would do the same) needs ONLY the folder."""
    out = str(tmp_path / "bundle")
    _stub_selector(monkeypatch, [TRI, SQUARE, PENTAGON])
    first = LVS.load_or_select_vials(_SyntheticSource(), out, n_vials=3)

    _stub_selector(monkeypatch, [], raises=AssertionError("must not draw again"))
    second = LVS.load_or_select_vials(_SyntheticSource(), out, n_vials=3, input_fn=lambda _p: "")

    assert second.reused is True
    assert second.polygons == first.polygons
    calib = second.calibration
    assert [v.polygon for v in calib.faces["A"].vials] == \
           [v.polygon for v in calib.faces["B"].vials] == [TRI, SQUARE, PENTAGON]
    assert (calib.image_width, calib.image_height) == (W, H)


# =========================================================================================
# 13. CLI `select-vials`
# =========================================================================================
def test_cli_select_vials_saves_both_faces_from_one_drawing(tmp_path, monkeypatch, capsys):
    from flygym_tracker import cli

    calls = _stub_selector(monkeypatch, [TRI, SQUARE, PENTAGON])
    out = str(tmp_path / "bundle")
    rc = cli.main(["select-vials", "--out", out, "--n-vials", "3", "--video", "clip.avi"])

    assert rc == 0
    assert calls["n_vials"] == 3 and calls["face"] == "A"
    calib = C.load_calibration(out)
    assert sorted(calib.faces) == ["A", "B"]
    assert [v.polygon for v in calib.faces["A"].vials] == \
           [v.polygon for v in calib.faces["B"].vials] == [TRI, SQUARE, PENTAGON]
    assert (calib.image_width, calib.image_height) == (W, H)
    assert os.path.exists(os.path.join(out, "overlay_A.png"))
    text = capsys.readouterr().out
    assert "saved vial positions" in text and "lit fraction" in text


def test_cli_select_vials_can_write_a_single_face_bundle(tmp_path, monkeypatch):
    from flygym_tracker import cli

    out = str(tmp_path / "bundle")
    _stub_selector(monkeypatch, [TRI])
    assert cli.main(["select-vials", "--out", out, "--face", "B", "--n-vials", "1",
                     "--video", "clip.avi"]) == 0
    calib = C.load_calibration(out)
    assert sorted(calib.faces) == ["B"]
    assert [v.polygon for v in calib.faces["B"].vials] == [TRI]


def test_cli_select_vials_offers_the_saved_positions_and_skips_drawing(tmp_path, monkeypatch,
                                                                      capsys):
    from flygym_tracker import cli

    out = _save_bundle(tmp_path, [TRI, SQUARE])
    calls = _stub_selector(monkeypatch, [PENTAGON])
    monkeypatch.setattr("builtins.input", lambda _p: "")        # ENTER = yes

    rc = cli.main(["select-vials", "--out", out, "--n-vials", "2", "--video", "clip.avi"])

    assert rc == 0 and calls["n_calls"] == 0
    assert C.saved_selection(out).polygons == [TRI, SQUARE]     # untouched
    assert "reusing the vial positions" in capsys.readouterr().out


def test_cli_select_vials_redraws_when_the_operator_declines(tmp_path, monkeypatch):
    from flygym_tracker import cli

    out = _save_bundle(tmp_path, [TRI, SQUARE])
    calls = _stub_selector(monkeypatch, [PENTAGON])
    # An operator AT A TERMINAL who answers "n": the prompt is only reached when one exists.
    monkeypatch.setattr(LVS, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _p: "n")

    assert cli.main(["select-vials", "--out", out, "--n-vials", "1", "--video", "c.avi"]) == 0
    assert calls["n_calls"] == 1
    assert C.saved_selection(out).polygons == [PENTAGON]


def test_cli_redraw_flag_never_asks(tmp_path, monkeypatch):
    from flygym_tracker import cli

    out = _save_bundle(tmp_path, [TRI, SQUARE])
    _stub_selector(monkeypatch, [PENTAGON])
    monkeypatch.setattr("builtins.input",
                        lambda _p: pytest.fail("--redraw must not prompt"))

    assert cli.main(["select-vials", "--out", out, "--n-vials", "1", "--video", "c.avi",
                     "--redraw"]) == 0
    assert C.saved_selection(out).polygons == [PENTAGON]


def test_cli_reuse_flag_never_asks_and_never_draws(tmp_path, monkeypatch):
    from flygym_tracker import cli

    out = _save_bundle(tmp_path, [TRI, SQUARE])
    calls = _stub_selector(monkeypatch, [PENTAGON])
    monkeypatch.setattr("builtins.input", lambda _p: pytest.fail("--reuse must not prompt"))

    assert cli.main(["select-vials", "--out", out, "--n-vials", "2", "--video", "c.avi",
                     "--reuse"]) == 0
    assert calls["n_calls"] == 0
    assert C.saved_selection(out).polygons == [TRI, SQUARE]


def test_cli_select_vials_writes_nothing_when_nothing_was_selected(tmp_path, monkeypatch, capsys):
    from flygym_tracker import cli

    out = str(tmp_path / "bundle")
    _stub_selector(monkeypatch, [])
    rc = cli.main(["select-vials", "--out", out, "--n-vials", "16", "--video", "clip.avi"])

    assert rc == 0
    assert not os.path.exists(os.path.join(out, "calibration.json"))
    assert "nothing saved" in capsys.readouterr().out


def test_cli_select_vials_warns_when_the_operator_finished_early(tmp_path, monkeypatch, capsys):
    from flygym_tracker import cli

    _stub_selector(monkeypatch, [TRI, SQUARE])
    cli.main(["select-vials", "--out", str(tmp_path / "b"), "--n-vials", "16", "--video", "c.avi"])
    assert "finished early" in capsys.readouterr().out


def test_cli_select_vials_explains_a_busy_camera_instead_of_a_traceback(tmp_path, monkeypatch,
                                                                       capsys):
    from flygym_tracker import cli

    _stub_selector(monkeypatch, [], raises=RuntimeError(
        "MV_CC_OpenDevice failed (ret=0x80000007) - camera may already be in use by another "
        "application (e.g. the MVS Viewer - USB3 Vision access is exclusive)"))
    rc = cli.main(["select-vials", "--out", str(tmp_path / "b"), "--n-vials", "16"])

    err = capsys.readouterr().err
    assert rc == 1
    assert "MVS Viewer" in err and "--video" in err
    assert "Traceback" not in err


def test_cli_select_vials_uses_the_live_camera_when_no_video_is_given(tmp_path, monkeypatch):
    from flygym_tracker import cli
    from flygym_tracker.frame_source import HikCameraSource

    calls = _stub_selector(monkeypatch, [TRI])
    rc = cli.main(["select-vials", "--out", str(tmp_path / "b"), "--n-vials", "1"])
    assert rc == 0
    assert isinstance(calls["source"], HikCameraSource)     # constructed, never opened here


def test_cli_select_vials_uses_a_video_source_with_video(tmp_path, monkeypatch):
    from flygym_tracker import cli
    from flygym_tracker.frame_source import VideoFileSource

    calls = _stub_selector(monkeypatch, [TRI])
    cli.main(["select-vials", "--out", str(tmp_path / "b"), "--n-vials", "1",
              "--video", "clip.avi"])
    assert isinstance(calls["source"], VideoFileSource)
    assert calls["source"].path == "clip.avi"


def test_the_saved_bundle_runs_through_the_pipeline(tmp_path, monkeypatch):
    """END TO END: draw -> save -> the pipeline builds per-vial masks from those polygons."""
    from flygym_tracker import cli

    out = str(tmp_path / "bundle")
    _stub_selector(monkeypatch, [TRI, SQUARE, PENTAGON])
    assert cli.main(["select-vials", "--out", out, "--n-vials", "3", "--video", "c.avi"]) == 0

    pipe = _null_pipeline(C.load_calibration(out))
    assert sorted(pipe._face_active["A"]) == [1, 2, 3]
    assert sorted(pipe._face_active["B"]) == [17, 18, 19]
    for gvid, poly in zip((1, 2, 3), (TRI, SQUARE, PENTAGON)):
        bbox, sub = pipe._face_active["A"][gvid]
        assert bbox == C.bbox_from_quad(poly)
        assert int(sub.sum()) > 0            # the drawn polygon covers lit pixels
        # ...and the same vial on the other face measures exactly the same region.
        assert pipe._face_active["B"][gvid + 16][0] == bbox


# =========================================================================================
# 14. The window has to FIT the screen (regression: bottom rows were unclickable)
# =========================================================================================
def test_the_view_limit_comes_from_the_real_desktop_not_a_constant():
    limit = LVS.screen_view_limit()
    assert isinstance(limit, tuple) and len(limit) == 2
    assert limit[0] >= 320 and limit[1] >= 240


def test_a_tall_frame_is_scaled_to_fit_a_short_desktop(monkeypatch):
    """Regression: a 1280x1024 frame became a 1200x960 canvas on a desktop only 829 px tall,
    so ~130 image rows sat below the screen edge - invisible and impossible to click."""
    monkeypatch.setattr(LVS, "screen_view_limit", lambda *a, **k: (1416, 813))

    scale = LVS.view_scale((1280, 1024))
    assert round(1024 * scale) <= 813        # the WHOLE frame is on screen
    assert round(1280 * scale) <= 1416
    assert scale == pytest.approx(813 / 1024, abs=1e-6)   # height is the binding constraint


def test_a_frame_that_already_fits_is_left_at_exactly_1_to_1(monkeypatch):
    monkeypatch.setattr(LVS, "screen_view_limit", lambda *a, **k: (1416, 813))
    assert LVS.view_scale((640, 480)) == 1.0


def test_the_view_limit_falls_back_when_the_desktop_cannot_be_measured(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_ctypes(name, *args, **kwargs):
        if name == "ctypes":
            raise ImportError("no ctypes here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ctypes)
    assert LVS.screen_view_limit(fallback=(800, 600)) == (800, 600)


def test_every_clicked_point_of_a_scaled_view_maps_back_inside_the_frame(monkeypatch):
    """The scale that keeps the window on screen must also map clicks back correctly."""
    monkeypatch.setattr(LVS, "screen_view_limit", lambda *a, **k: (1416, 813))
    w, h = 1280, 1024
    scale = LVS.view_scale((w, h))
    state = LVS.SelectorState(n_vials=1)
    # the extreme corners of the displayed canvas
    for sx, sy in [(0, 0), (round(w * scale) - 1, 0), (0, round(h * scale) - 1),
                   (round(w * scale) - 1, round(h * scale) - 1)]:
        x, y = state.add_vertex(sx / scale, sy / scale)
        assert 0 <= x < w and 0 <= y < h


def test_the_window_is_moved_fully_onto_the_screen(monkeypatch):
    """Regression: the canvas fitted (1342 px wide on a 1440 px desktop) but OpenCV placed the
    window at x=164, so 66 px hung off the right edge - unclickable, same as the truncated rows."""
    moved = {}
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda _w: (164, 25, 1342, 796))
    monkeypatch.setattr(cv2, "moveWindow", lambda w, x, y: moved.update(w=w, x=x, y=y))

    LVS.place_window_on_screen("Select vials", (1416, 813))

    assert moved["x"] + 1342 <= 1416 and moved["x"] >= 0
    assert moved["y"] + 796 <= 813 and moved["y"] >= 0


def test_window_placement_never_breaks_the_session(monkeypatch):
    """Placement is cosmetic; a highgui that cannot report its rect must not stop the drawing."""
    monkeypatch.setattr(cv2, "getWindowImageRect",
                        lambda _w: (_ for _ in ()).throw(cv2.error("no such window")))
    LVS.place_window_on_screen("Select vials", (1416, 813))     # must not raise

"""Tests for the live tracking-settings panel and everything it is wired into.

HEADLESS BY CONSTRUCTION. Nothing here opens a cv2 window. `settings_panel` splits into pure
state (`Setting`, `SettingsModel`), pure geometry (`layout`, `value_at`, `hit`), pure rendering
(`render`) and a thin `SettingsWindow` driver -- so even the DRAG is exercised, against a stubbed
highgui, with no display (same harness idea as `tests/test_live_vial_selector.py`).

Six things are being protected:
  1. the arithmetic an operator feels through the handle -- clamping, step snapping, and a drag
     from one end of the track to the other sweeping exactly lo..hi and nothing outside it,
  2. the driver WIRING (a press really does become a value; an arrow key really does nudge),
  3. that a value change reaches `on_change` ONCE per distinct value, not once per mouse event,
  4. that `apply_setting` changes what the pipeline MEASURES on the next frame -- asserted on the
     motion numbers coming out of the pipeline, not on the attribute that was written,
  5. that every applied change leaves exactly one `setting_change` row in events.csv, naming the
     key and both values (a run whose threshold moved at hour 40 must not look like one regime),
  6. that saving keeps the config file's comments, which are the measurement notes justifying the
     numbers being saved over.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import pytest

from flygym_tracker import monitor as MON
from flygym_tracker import settings_panel as SP
from flygym_tracker.config import load_config
from flygym_tracker.frame_source import FrameSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.pipeline import TrackerPipeline
from flygym_tracker.settings_panel import (
    Setting,
    SettingsModel,
    SettingsWindow,
    apply_overrides_to_yaml_text,
    build_settings,
    decode_key,
    format_value,
    hit,
    layout,
    panel_size,
    render,
    save_settings_to_yaml,
    value_at,
)
from flygym_tracker.types import Calibration, FaceCalibration, Frame, VialROI


def _float_setting(**kw):
    base = dict(key="a.f", label="f", value=5.0, lo=0.0, hi=10.0, step=0.5,
                kind="float", group="G", help="h")
    base.update(kw)
    return Setting(**base)


def _int_setting(**kw):
    base = dict(key="a.i", label="i", value=4, lo=1, hi=30, step=1,
                kind="int", group="G", help="h")
    base.update(kw)
    return Setting(**base)


def _bool_setting(**kw):
    base = dict(key="a.b", label="b", value=True, lo=0, hi=1, step=1,
                kind="bool", group="G", help="h")
    base.update(kw)
    return Setting(**base)


def _model(*settings) -> SettingsModel:
    return SettingsModel(settings or (_float_setting(), _int_setting()))


# =========================================================================================
# 1. Setting -- clamping, snapping, casting
# =========================================================================================
@pytest.mark.parametrize("raw,expected", [(-5.0, 0.0), (0.0, 0.0), (10.0, 10.0), (99.0, 10.0)])
def test_a_value_outside_the_bounds_is_clamped_to_them(raw, expected):
    m = _model(_float_setting())
    assert m.set("a.f", raw) == expected


def test_a_value_between_two_steps_snaps_to_the_nearer_one():
    m = _model(_float_setting())
    assert m.set("a.f", 5.24) == 5.0
    assert m.set("a.f", 5.26) == 5.5


def test_snapping_is_measured_from_lo_not_from_zero():
    """A track starting at 0.2 in steps of 0.1 offers 0.2, 0.3, ... -- never 0.25."""
    m = _model(_float_setting(lo=0.2, hi=5.0, step=0.1, value=1.0))
    assert m.set("a.f", 0.34) == pytest.approx(0.3)
    assert m.set("a.f", 0.26) == pytest.approx(0.3)


def test_both_endpoints_stay_reachable_when_the_span_is_not_a_whole_number_of_steps():
    """REGRESSION GUARD. lo=0, hi=1, step=0.3 snaps to 0.9 at best, so a plain snap would make the
    number printed under the right-hand end of the track unreachable BY dragging to that end."""
    m = _model(_float_setting(lo=0.0, hi=1.0, step=0.3, value=0.0))
    assert m.set("a.f", 1.0) == 1.0
    assert m.set("a.f", 0.999) == 0.9      # not at the end -> ordinary snapping still applies
    assert m.set("a.f", 0.0) == 0.0


def test_an_int_setting_stores_an_int_not_a_float():
    m = _model(_int_setting())
    value = m.set("a.i", 7.6)
    assert value == 8 and isinstance(value, int)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_falls_back_to_lo_instead_of_poisoning_the_run(bad):
    """A NaN reaching `pixel_threshold` would make every comparison False, i.e. silently measure
    zero activity for the rest of the run. `lo` is the least destructive landing place."""
    m = _model(_float_setting())
    assert m.set("a.f", bad) == 0.0


# -- oddness (an OpenCV kernel size must be odd) ------------------------------------------
def _kernel_model(lo=3, hi=51):
    """A `bg_kernel`-shaped setting. Declared with step 1 on purpose: `odd=True` forces the step
    to 2, which is what keeps the arrow keys and the drag agreeing about the legal values."""
    return _model(_int_setting(key="a.k", lo=lo, hi=hi, step=1, value=31, odd=True))


def test_an_odd_only_setting_takes_a_step_of_two_whatever_was_declared():
    assert _kernel_model().get("a.k").step == 2


def test_odd_is_refused_on_anything_but_an_int():
    with pytest.raises(ValueError, match="odd=True"):
        _float_setting(odd=True)


def test_an_odd_only_setting_never_stores_an_even_number():
    m = _kernel_model()
    for raw in range(3, 52):
        assert m.set("a.k", raw) % 2 == 1, "%d snapped to an even kernel size" % raw
        assert m.set("a.k", raw + 0.5) % 2 == 1


def test_an_odd_only_setting_rounds_to_the_nearer_odd_neighbour():
    """REGRESSION GUARD: snapping to the step grid FIRST and then hopping to an odd neighbour
    loses which way the value was leaning, and sends both 9.6 and 10.4 to 11."""
    m = _kernel_model()
    assert m.set("a.k", 9.6) == 9
    assert m.set("a.k", 10.4) == 11
    assert m.set("a.k", 11.0) == 11


def test_an_odd_only_setting_stays_inside_its_bounds_at_the_ends():
    m = _kernel_model()
    assert m.set("a.k", 0) == 3
    assert m.set("a.k", 999) == 51


def test_an_odd_only_setting_with_even_bounds_still_only_yields_odd_values():
    m = _kernel_model(lo=2, hi=50)
    for raw in range(2, 51):
        assert m.set("a.k", raw) % 2 == 1
    assert 2 <= m.set("a.k", 0) <= 50
    assert 2 <= m.set("a.k", 999) <= 50


def test_nudging_an_odd_only_setting_lands_on_the_next_odd_value():
    """REGRESSION GUARD: with a declared step of 1, "down" from 13 landed on 12, which snapped
    straight back to 13 -- an arrow key the operator could press forever with nothing happening."""
    m = _kernel_model()
    assert m.set("a.k", 11) == 11
    assert m.nudge("a.k", +1) == 13
    assert m.nudge("a.k", -1) == 11
    assert m.nudge("a.k", -1) == 9


# -- bools ---------------------------------------------------------------------------------
def test_a_bool_setting_stores_a_bool():
    m = _model(_bool_setting())
    assert m.set("a.b", 0) is False
    assert m.set("a.b", 1) is True


def test_a_bool_is_not_nudged_around_a_cycle():
    """Right is ON and left is OFF, always -- so the same key gives the same state every time."""
    m = _model(_bool_setting())
    assert m.nudge("a.b", +1) is True
    assert m.nudge("a.b", +1) is True
    assert m.nudge("a.b", -1) is False
    assert m.nudge("a.b", -1) is False


# -- construction guards -------------------------------------------------------------------
def test_a_setting_declaring_an_impossible_range_or_kind_is_rejected_at_construction():
    with pytest.raises(ValueError, match="kind"):
        _float_setting(kind="colour")
    with pytest.raises(ValueError, match="below lo"):
        _float_setting(lo=10.0, hi=0.0)
    with pytest.raises(ValueError, match="step"):
        _float_setting(step=0.0)


def test_a_setting_is_coerced_at_construction_so_a_model_can_never_hold_a_bad_value():
    assert _float_setting(value=999.0).value == 10.0
    assert _float_setting(value=5.26).value == 5.5


# =========================================================================================
# 2. SettingsModel
# =========================================================================================
def test_nudge_moves_by_exactly_one_step_in_each_direction():
    m = _model(_float_setting(value=5.0))
    assert m.nudge("a.f", +1) == 5.5
    assert m.nudge("a.f", -1) == 5.0
    assert m.nudge("a.f", -1) == 4.5


def test_nudging_past_a_bound_stops_at_the_bound():
    m = _model(_float_setting(value=10.0))
    assert m.nudge("a.f", +1) == 10.0
    m.set("a.f", 0.0)
    assert m.nudge("a.f", -1) == 0.0


def test_changed_lists_only_what_moved_away_from_the_file():
    m = _model(_float_setting(), _int_setting())
    assert m.changed() == []
    m.set("a.f", 7.0)
    assert [s.key for s in m.changed()] == ["a.f"]
    m.set("a.f", 5.0)                      # back to the file's value: no longer a change
    assert m.changed() == []


def test_reset_puts_everything_back_and_names_what_moved():
    m = _model(_float_setting(), _int_setting())
    m.set("a.f", 7.0)
    m.set("a.i", 9)
    assert sorted(m.reset()) == ["a.f", "a.i"]
    assert m.value("a.f") == 5.0 and m.value("a.i") == 4
    assert m.changed() == []
    assert m.reset() == []                 # nothing left to reset


def test_mark_saved_adopts_the_current_values_as_the_new_baseline():
    m = _model(_float_setting())
    m.set("a.f", 7.0)
    m.mark_saved()
    assert m.changed() == []
    assert m.baseline("a.f") == 7.0
    m.set("a.f", 5.0)
    assert [s.key for s in m.changed()] == ["a.f"]


def test_to_overrides_nests_dotted_keys_into_the_config_tree():
    m = SettingsModel([
        _float_setting(key="activity.pixel_threshold", value=5.0),
        _float_setting(key="rotation.sensitivity", value=5.0),
        _int_setting(key="rotation.debounce_frames", value=4),
    ])
    assert m.to_overrides() == {
        "activity": {"pixel_threshold": 5.0},
        "rotation": {"sensitivity": 5.0, "debounce_frames": 4},
    }


def test_to_overrides_changed_only_narrows_to_what_this_session_touched():
    """What saving writes: an untouched default must not be stamped over a hand-tuned file."""
    m = SettingsModel([
        _float_setting(key="activity.pixel_threshold", value=5.0),
        _float_setting(key="rotation.sensitivity", value=5.0),
    ])
    m.set("rotation.sensitivity", 2.0)
    assert m.to_overrides(changed_only=True) == {"rotation": {"sensitivity": 2.0}}


def test_a_deeply_nested_key_becomes_a_deeply_nested_override():
    m = SettingsModel([_int_setting(key="source.camera.width", lo=1, hi=4096, value=1280)])
    assert m.to_overrides() == {"source": {"camera": {"width": 1280}}}


def test_duplicate_keys_are_refused_rather_than_silently_shadowing_each_other():
    with pytest.raises(ValueError, match="duplicate"):
        SettingsModel([_float_setting(key="a.f"), _int_setting(key="a.f")])


def test_an_unknown_key_says_what_the_known_ones_are():
    m = _model(_float_setting())
    with pytest.raises(KeyError, match="a.f"):
        m.get("nope.nothing")


def test_groups_are_listed_once_each_in_first_appearance_order():
    m = SettingsModel([
        _float_setting(key="x.1", group="Activity"),
        _float_setting(key="x.2", group="Rotation"),
        _float_setting(key="x.3", group="Activity"),
    ])
    assert m.groups() == ["Activity", "Rotation"]


# =========================================================================================
# 3. Layout, hit testing and the drag arithmetic
# =========================================================================================
def _laid_out(model=None):
    model = model or _model()
    w, h = panel_size(model)
    return model, layout(model, w, h), w, h


def test_layout_gives_one_rect_per_setting_in_model_order():
    model, rects, _w, _h = _laid_out()
    assert [r.key for r in rects] == model.keys()
    assert [r.index for r in rects] == list(range(len(model)))


def test_rows_do_not_overlap_and_run_down_the_panel():
    _model_, rects, _w, _h = _laid_out(SettingsModel([
        _float_setting(key="x.1", group="A"), _float_setting(key="x.2", group="A"),
        _float_setting(key="x.3", group="B"),
    ]))
    for a, b in zip(rects, rects[1:]):
        assert a.y + a.h <= b.y


def test_only_the_first_row_of_each_group_carries_a_group_heading():
    _m, rects, _w, _h = _laid_out(SettingsModel([
        _float_setting(key="x.1", group="A"), _float_setting(key="x.2", group="A"),
        _float_setting(key="x.3", group="B"),
    ]))
    assert [r.group_header_y is not None for r in rects] == [True, False, True]


def test_the_panel_is_tall_enough_for_every_row_it_lays_out():
    model = build_settings(load_config())
    _w, h = panel_size(model)
    rects = layout(model, _w, h)
    assert rects[-1].y + rects[-1].h <= h - SP.FOOTER_H


# -- the drag ------------------------------------------------------------------------------
def test_the_two_ends_of_the_track_are_exactly_lo_and_hi():
    _m, rects, _w, _h = _laid_out(_model(_float_setting(lo=0.0, hi=60.0, step=0.5)))
    r = rects[0]
    assert value_at(r, r.track_x0) == 0.0
    assert value_at(r, r.track_x1) == 60.0


def test_a_drag_from_one_end_to_the_other_sweeps_exactly_lo_to_hi():
    """THE contract of a slider: every pixel of the track maps inside the declared range, the
    sweep never goes backwards, and both ends of the range are actually produced."""
    _m, rects, _w, _h = _laid_out(_model(_float_setting(lo=0.0, hi=60.0, step=0.5)))
    r = rects[0]
    swept = [value_at(r, x) for x in range(r.track_x0, r.track_x1 + 1)]
    assert swept[0] == 0.0
    assert swept[-1] == 60.0
    assert min(swept) == 0.0 and max(swept) == 60.0
    assert swept == sorted(swept)                                  # monotonic, never jumps back
    assert all(v == pytest.approx(round(v * 2) / 2) for v in swept)  # every value is on a step


def test_a_drag_covers_every_step_when_the_track_is_wider_than_the_range():
    """No step is skipped: an operator dragging slowly can land on any value the scale offers."""
    _m, rects, _w, _h = _laid_out(_model(_int_setting(lo=1, hi=30, step=1)))
    r = rects[0]
    swept = {value_at(r, x) for x in range(r.track_x0, r.track_x1 + 1)}
    assert swept == set(range(1, 31))


def test_a_drag_past_either_end_of_the_track_clamps_instead_of_running_away():
    _m, rects, _w, _h = _laid_out(_model(_float_setting(lo=0.0, hi=60.0, step=0.5)))
    r = rects[0]
    assert value_at(r, r.track_x0 - 500) == 0.0
    assert value_at(r, r.track_x1 + 500) == 60.0


def test_a_bool_row_has_two_halves_not_a_continuum():
    _m, rects, _w, _h = _laid_out(_model(_bool_setting()))
    r = rects[0]
    assert value_at(r, r.track_x0 + 1) is False
    assert value_at(r, r.track_x1 - 1) is True
    assert {value_at(r, x) for x in range(r.track_x0, r.track_x1 + 1)} == {False, True}


def test_value_fraction_is_the_inverse_of_value_at():
    _m, rects, _w, _h = _laid_out(_model(_float_setting(lo=0.0, hi=60.0, step=0.5)))
    r = rects[0]
    for x in range(r.track_x0, r.track_x1 + 1, 7):
        value = value_at(r, x)
        back = r.track_x0 + SP.value_fraction(r, value) * r.track_w
        assert abs(back - x) <= 3        # within the width of one step on this track


# -- hit testing ---------------------------------------------------------------------------
def test_hit_returns_the_row_under_the_cursor():
    _m, rects, _w, _h = _laid_out()
    for r in rects:
        assert hit(rects, r.x + 5, r.y + 5) is r
        assert hit(rects, r.track_x0, r.track_y) is r


def test_hit_returns_nothing_for_the_header_and_the_footer():
    _m, rects, w, h = _laid_out()
    assert hit(rects, w // 2, 10) is None
    assert hit(rects, w // 2, h - 5) is None


def test_the_grab_band_around_a_track_is_wide_enough_to_hit_with_a_mouse():
    """A 6 px line is not a mouse target; a press a few px off must still grab the handle."""
    _m, rects, _w, _h = _laid_out()
    r = rects[0]
    assert r.on_track(r.track_x0 + 10, r.track_y)
    assert r.on_track(r.track_x0 + 10, r.track_y - 10)
    assert not r.on_track(r.track_x0 + 10, r.track_y - 40)


# =========================================================================================
# 4. Rendering
# =========================================================================================
def test_render_returns_a_bgr_canvas_of_the_requested_size():
    model, rects, w, h = _laid_out()
    img = render(model, rects, w, h)
    assert img.shape == (h, w, 3) and img.dtype == np.uint8


def test_changing_a_value_changes_the_pixels():
    model, rects, w, h = _laid_out()
    before = render(model, rects, w, h)
    model.set("a.f", 9.0)
    after = render(model, rects, w, h)
    assert np.any(before != after)


def test_the_value_is_shown_in_real_units_not_as_a_scaled_integer():
    """The whole reason these sliders are hand-drawn instead of `cv2.createTrackbar`."""
    s = _float_setting(hi=60.0, value=12.0, step=0.5, unit="grey levels")
    assert format_value(s) == "12.0 grey levels"
    assert format_value(_int_setting(value=4, unit="frames")) == "4 frames"
    assert format_value(_float_setting(lo=0.0, hi=1.0, step=0.05, value=0.6)) == "0.60"


def test_a_bool_shows_its_state_as_a_word_and_draws_differently_when_toggled():
    model = _model(_bool_setting(value=True))
    rects = layout(model, *panel_size(model))
    w, h = panel_size(model)
    assert format_value(model.get("a.b")) == "ON"
    on = render(model, rects, w, h)
    model.set("a.b", False)
    assert format_value(model.get("a.b")) == "OFF"
    assert np.any(on != render(model, rects, w, h))


def test_a_next_run_row_is_drawn_differently_from_a_live_one():
    """A setting that cannot take effect now must SAY so, not look identical to one that can."""
    live = _model(_float_setting(live=True))
    later = _model(_float_setting(live=False))
    w, h = panel_size(live)
    assert np.any(render(live, layout(live, w, h), w, h)
                  != render(later, layout(later, w, h), w, h))


def test_a_changed_row_is_marked_so_the_operator_can_see_what_they_moved():
    model, rects, w, h = _laid_out()
    plain = render(model, rects, w, h)
    model.set("a.f", 5.0)                     # same value -> still unchanged, still plain
    assert np.array_equal(plain, render(model, rects, w, h))
    model.set("a.f", 6.0)
    assert np.any(plain != render(model, rects, w, h))


def test_the_footer_lists_the_keys():
    assert "r = reset all" in SP.KEY_HINT and "s = save to config" in SP.KEY_HINT


def test_the_startup_banner_names_every_setting_and_flags_the_non_live_ones():
    model = SettingsModel([_float_setting(key="a.now", live=True),
                           _float_setting(key="a.later", live=False)])
    banner = SP.startup_banner(model)
    assert "a.now" in banner and "a.later" in banner
    assert banner.count("NEXT RUN") == 1


# =========================================================================================
# 5. Keyboard decoding
# =========================================================================================
@pytest.mark.parametrize("code,name", [
    (2490368, "up"), (2621440, "down"), (2424832, "left"), (2555904, "right"),   # Windows
    (65362, "up"), (65364, "down"), (65361, "left"), (65363, "right"),           # GTK/Qt
    (63232, "up"), (63233, "down"), (63234, "left"), (63235, "right"),           # macOS/Qt
])
def test_arrow_keys_decode_on_every_platform_opencv_runs_on(code, name):
    """REGRESSION GUARD. The low-byte trick the vial selector uses cannot decode these: Windows
    sends 2490368 for Up, and 2490368 & 0xFF == 0, i.e. it looks like "no key pressed"."""
    assert decode_key(code) == name


@pytest.mark.parametrize("code,name", [(ord("r"), "r"), (ord("S"), "s"), (27, "esc"), (13, "enter"),
                                       (32, "space")])
def test_ordinary_keys_still_decode(code, name):
    assert decode_key(code) == name


@pytest.mark.parametrize("code", [None, -1])
def test_no_keypress_decodes_to_nothing(code):
    assert decode_key(code) is None


# =========================================================================================
# 6. The driver, against a stubbed highgui (NO window is opened anywhere here)
# =========================================================================================
class _FakePanel:
    """Stands in for the whole cv2 highgui surface `SettingsWindow` touches.

    `script` is a list of ("press", x, y) / ("move", x, y) / ("release",) / ("key", code) /
    ("idle",) events, fired from inside the `waitKeyEx` stub -- exactly where a real event
    arrives -- so these tests prove the WIRING (callback -> model -> on_change), not just the
    pure functions underneath.
    """

    def __init__(self, monkeypatch, script):
        self.script = list(script)
        self.shown = []
        self.callback = None
        self.destroyed = []
        self.visible = 1.0
        self.banners = []
        monkeypatch.setattr(SP, "require_gui", lambda *_a, **_k: None)
        monkeypatch.setattr(SP, "place_window_on_screen", lambda *_a, **_k: None)
        monkeypatch.setattr(SP, "screen_view_limit", lambda *_a, **_k: (1920, 1080))
        monkeypatch.setattr(SP, "PUMP_FPS", 1e9)          # never skip an iteration in a test
        monkeypatch.setattr("builtins.print", lambda *a, **k: self.banners.append(a))
        for name, fn in [
            ("namedWindow", lambda *a, **k: None),
            ("setMouseCallback", lambda _w, cb, *a: setattr(self, "callback", cb)),
            ("imshow", lambda _w, img: self.shown.append(img.copy())),
            ("waitKeyEx", self._next),
            ("waitKey", lambda *_a: -1),
            ("getWindowProperty", lambda *_a: self.visible),
            ("destroyWindow", lambda w: self.destroyed.append(w)),
        ]:
            monkeypatch.setattr(cv2, name, fn)

    def _next(self, *_a):
        if not self.script:
            return ord("q")                     # never hang: an unscripted loop closes
        event = self.script.pop(0)
        kind = event[0]
        if kind == "press":
            self.callback(cv2.EVENT_LBUTTONDOWN, event[1], event[2], 0, None)
            return -1
        if kind == "move":
            self.callback(cv2.EVENT_MOUSEMOVE, event[1], event[2], 0, None)
            return -1
        if kind == "release":
            self.callback(cv2.EVENT_LBUTTONUP, 0, 0, 0, None)
            return -1
        if kind == "idle":
            return -1
        return event[1]


def _window(monkeypatch, script, model=None, on_change=None, on_save=None):
    model = model or _model(_float_setting(lo=0.0, hi=60.0, step=0.5, value=12.0),
                            _int_setting(value=4))
    fake = _FakePanel(monkeypatch, script)
    win = SettingsWindow(model, on_change=on_change, on_save=on_save)
    return win, fake, model


def _track_x(win, index, frac):
    r = win.rects[index]
    return int(round(r.track_x0 + frac * r.track_w))


def test_a_press_on_a_track_sets_that_row_to_the_pressed_value(monkeypatch):
    win, _fake, model = _window(monkeypatch, [])
    win.on_mouse(cv2.EVENT_LBUTTONDOWN, _track_x(win, 0, 0.5), win.rects[0].track_y)
    assert model.value("a.f") == 30.0


def test_a_drag_end_to_end_through_the_real_loop_sweeps_the_row(monkeypatch):
    """END TO END: press, three moves, release -- driven by the loop, not by calling the model."""
    seen: List[tuple] = []
    win, _fake, model = _window(
        monkeypatch,
        [("press", 0, 0), ("move", 0, 0), ("move", 0, 0), ("release",), ("key", ord("q"))],
        on_change=lambda k, v: seen.append((k, v)),
    )
    y = win.rects[0].track_y
    xs = [_track_x(win, 0, f) for f in (0.0, 0.25, 0.5, 1.0)]
    _fake.script = [("press", xs[0], y), ("move", xs[1], y), ("move", xs[2], y),
                    ("move", xs[3], y), ("release",), ("key", ord("q"))]

    win.run(poll_ms=1)

    assert [k for k, _v in seen] == ["a.f"] * 4
    assert [v for _k, v in seen] == [0.0, 15.0, 30.0, 60.0]
    assert model.value("a.f") == 60.0


def test_a_drag_reports_one_change_per_distinct_value_not_one_per_mouse_event(monkeypatch):
    """The pipeline logs an event per applied change; a slider that fired per PIXEL would bury
    the real transitions in its own noise."""
    seen: List[tuple] = []
    win, fake, _model = _window(monkeypatch, [], on_change=lambda k, v: seen.append((k, v)))
    y = win.rects[0].track_y
    x = _track_x(win, 0, 0.5)
    fake.script = ([("press", x, y)] + [("move", x, y)] * 20
                   + [("release",), ("key", ord("q"))])

    win.run(poll_ms=1)

    assert len(seen) == 1                # 21 mouse events, one value, one report


def test_a_press_off_the_track_selects_the_row_without_changing_it(monkeypatch):
    win, _fake, model = _window(monkeypatch, [])
    before = model.value("a.i")
    r = win.rects[1]
    win.on_mouse(cv2.EVENT_LBUTTONDOWN, r.x + 5, r.y + 5)
    assert win.selected == 1
    assert model.value("a.i") == before


def test_a_press_outside_every_row_changes_nothing(monkeypatch):
    win, _fake, model = _window(monkeypatch, [])
    win.on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5)
    assert win.selected == 0 and model.value("a.f") == 12.0


def test_a_drag_that_strays_off_the_row_keeps_following_the_cursor(monkeypatch):
    """Once the handle is grabbed the drag follows x, like every slider the operator has used."""
    win, _fake, model = _window(monkeypatch, [])
    r = win.rects[0]
    win.on_mouse(cv2.EVENT_LBUTTONDOWN, _track_x(win, 0, 0.1), r.track_y)
    win.on_mouse(cv2.EVENT_MOUSEMOVE, _track_x(win, 0, 0.9), r.track_y + 400)
    assert model.value("a.f") == 54.0


def test_a_drag_stops_following_after_the_button_is_released(monkeypatch):
    win, _fake, model = _window(monkeypatch, [])
    r = win.rects[0]
    win.on_mouse(cv2.EVENT_LBUTTONDOWN, _track_x(win, 0, 0.5), r.track_y)
    win.on_mouse(cv2.EVENT_LBUTTONUP, 0, 0)
    win.on_mouse(cv2.EVENT_MOUSEMOVE, _track_x(win, 0, 1.0), r.track_y)
    assert model.value("a.f") == 30.0


# -- keys ----------------------------------------------------------------------------------
def test_arrow_keys_nudge_the_selected_row_through_the_real_loop(monkeypatch):
    seen: List[tuple] = []
    win, _fake, model = _window(
        monkeypatch,
        [("key", 2555904), ("key", 2555904), ("key", 2424832), ("key", ord("q"))],
        on_change=lambda k, v: seen.append((k, v)),
    )
    win.run(poll_ms=1)
    assert model.value("a.f") == 12.5
    assert [v for _k, v in seen] == [12.5, 13.0, 12.5]


def test_up_and_down_move_the_selection_and_stop_at_the_ends(monkeypatch):
    win, _fake, _model = _window(monkeypatch, [])
    win.handle_key("down")
    assert win.selected == 1
    win.handle_key("down")
    assert win.selected == 1               # clamped, NOT wrapped back to the top
    win.handle_key("up")
    win.handle_key("up")
    assert win.selected == 0


def test_the_arrow_keys_nudge_whichever_row_is_selected(monkeypatch):
    win, _fake, model = _window(monkeypatch, [])
    win.handle_key("down")
    win.handle_key("right")
    assert model.value("a.i") == 5 and model.value("a.f") == 12.0


def test_plus_and_minus_nudge_too_because_the_monitor_taught_those_keys(monkeypatch):
    win, _fake, model = _window(monkeypatch, [])
    win.handle_key("+")
    assert model.value("a.f") == 12.5
    win.handle_key("-")
    assert model.value("a.f") == 12.0


def test_enter_toggles_a_bool_row(monkeypatch):
    win, _fake, model = _window(monkeypatch, [], model=_model(_bool_setting(value=False)))
    win.handle_key("enter")
    assert model.value("a.b") is True
    win.handle_key("space")
    assert model.value("a.b") is False


def test_r_resets_every_row_and_tells_the_pipeline_about_each_one(monkeypatch):
    seen: List[tuple] = []
    win, _fake, model = _window(monkeypatch, [], on_change=lambda k, v: seen.append((k, v)))
    win.apply("a.f", 40.0)
    win.apply("a.i", 9)
    seen.clear()
    win.handle_key("r")
    assert model.value("a.f") == 12.0 and model.value("a.i") == 4
    assert sorted(k for k, _v in seen) == ["a.f", "a.i"]


@pytest.mark.parametrize("key", ["q", "esc"])
def test_q_and_esc_close_the_panel(monkeypatch, key):
    win, _fake, _model = _window(monkeypatch, [])
    assert win.handle_key(key) == "done"
    assert win.closed


def test_closing_the_window_with_its_x_ends_the_loop(monkeypatch):
    win, fake, _model = _window(monkeypatch, [("idle",), ("idle",), ("idle",)])
    fake.visible = 0.0
    win.run(poll_ms=1)
    assert win.closed
    assert fake.destroyed == [SP.DEFAULT_WINDOW]


def test_the_modal_loop_draws_the_panel_and_tears_the_window_down(monkeypatch):
    win, fake, _model = _window(monkeypatch, [("idle",), ("key", ord("q"))])
    win.run(poll_ms=1)
    assert fake.shown, "the panel was never drawn"
    assert fake.destroyed == [SP.DEFAULT_WINDOW]


def test_the_panel_refuses_to_open_without_gui_support(monkeypatch):
    _FakePanel(monkeypatch, [])
    monkeypatch.setattr(SP, "require_gui",
                        lambda *_a, **_k: (_ for _ in ()).throw(SystemExit(2)))
    with pytest.raises(SystemExit):
        SettingsWindow(_model()).open()


def test_the_panel_is_scaled_down_to_fit_a_short_desktop(monkeypatch):
    """REGRESSION GUARD (the one `screen_view_limit` exists for): the bottom rows of an oversized
    window land under the taskbar, where they cannot be clicked."""
    fake = _FakePanel(monkeypatch, [])
    monkeypatch.setattr(SP, "screen_view_limit", lambda *_a, **_k: (400, 300))
    win = SettingsWindow(_model())
    win.open()
    assert win._scale < 1.0
    assert fake.shown[0].shape[0] <= 300 and fake.shown[0].shape[1] <= 400


def test_a_click_is_mapped_back_through_the_display_scale(monkeypatch):
    """A shrunk panel must still put the value the operator aimed at under the cursor."""
    _FakePanel(monkeypatch, [])
    monkeypatch.setattr(SP, "screen_view_limit", lambda *_a, **_k: (400, 300))
    model = _model(_float_setting(lo=0.0, hi=60.0, step=0.5, value=12.0))
    win = SettingsWindow(model)
    win.open()
    r = win.rects[0]
    scaled_x = (r.track_x0 + 0.5 * r.track_w) * win._scale
    win.on_mouse(cv2.EVENT_LBUTTONDOWN, scaled_x, r.track_y * win._scale)
    assert model.value("a.f") == 30.0


def test_a_change_the_pipeline_refuses_is_said_out_loud_rather_than_hidden(monkeypatch):
    win, _fake, _model = _window(monkeypatch, [], on_change=lambda _k, _v: False)
    win.apply("a.f", 40.0)
    assert "not applied" in win.message


def test_a_callback_that_raises_cannot_take_the_panel_down(monkeypatch):
    def boom(_k, _v):
        raise RuntimeError("detector exploded")

    win, _fake, model = _window(monkeypatch, [], on_change=boom)
    win.apply("a.f", 40.0)
    assert model.value("a.f") == 40.0
    assert "detector exploded" in win.message


def test_s_saves_and_says_nothing_to_save_when_nothing_changed(monkeypatch):
    saved: List[SettingsModel] = []
    win, _fake, _model = _window(monkeypatch, [], on_save=lambda m: saved.append(m) or ["note"])
    win.handle_key("s")
    assert saved == [] and "nothing" in win.message
    win.apply("a.f", 40.0)
    win.handle_key("s")
    assert len(saved) == 1 and "saved" in win.message


def test_pump_returns_false_once_the_panel_is_closed(monkeypatch):
    win, _fake, _model = _window(monkeypatch, [("key", ord("q"))])
    win.open()
    assert win.pump(timeout_ms=1) is False
    assert win.pump(timeout_ms=1) is False       # and stays closed


def test_close_is_idempotent_and_safe_before_open(monkeypatch):
    _FakePanel(monkeypatch, [])
    win = SettingsWindow(_model())
    win.close()
    win.close()
    assert win.closed


# =========================================================================================
# 7. build_settings -- only what can actually be routed
# =========================================================================================
def test_build_settings_seeds_every_value_from_the_config_file():
    config = load_config("config/flygym_rig.yaml")
    model = build_settings(config)
    assert model.value("activity.pixel_threshold") == pytest.approx(12.0)
    assert model.value("rotation.debounce_frames") == 4
    assert model.value("rotation.min_stationary_frames") == 3
    assert model.value("rotation.sensitivity") == pytest.approx(1.0)


def test_every_built_setting_has_a_help_line_a_group_and_a_unit_scale_that_makes_sense():
    model = build_settings(load_config())
    for s in model.settings:
        assert s.help and not s.help.startswith(s.label), "%s: help restates the label" % s.key
        assert s.group
        assert s.lo <= s.value <= s.hi
        assert "." in s.key, "%s is not a config path" % s.key


def test_build_settings_shows_what_the_pipeline_is_actually_using_not_what_the_file_says(tmp_path):
    """A threshold passed on the command line, or already nudged with +, differs from the file;
    a panel opening on the file's number would be describing a run that is not happening."""
    pipe = _mini_pipeline(tmp_path, [_bg(), _bg()], pixel_threshold=33.0)
    model = build_settings(pipe.config, pipeline=pipe)
    assert model.value("activity.pixel_threshold") == pytest.approx(33.0)


def test_every_built_setting_is_routable_by_the_pipeline_it_was_built_from(tmp_path):
    """The panel must not offer a slider the run cannot honour."""
    pipe = _mini_pipeline(tmp_path, [_bg(), _bg()], adaptive=True)
    model = build_settings(pipe.config, pipeline=pipe)
    assert set(model.keys()) <= set(pipe.settable_keys())


# =========================================================================================
# 8. Saving to YAML -- the comments are the measurement notes, they must survive
# =========================================================================================
CONFIG_TEXT = """\
# Tuned config, validated on real flies.

rotation:
  detector: adaptive          # speed-independent
  sensitivity: 1.0            # do not preset a magnitude
  debounce_frames: 4          # dwells are short (~2 s)
  min_stationary_frames: 3

activity:
  pixel_threshold: 12.0       # above the sensor-noise floor; catches fly shadows
  k: 5.0

binning:
  bin_seconds: 10
"""


def test_saving_rewrites_the_value_and_keeps_the_comment_that_justifies_it():
    out, notes = apply_overrides_to_yaml_text(CONFIG_TEXT, {"activity": {"pixel_threshold": 14.5}})
    assert "pixel_threshold: 14.5" in out
    assert "above the sensor-noise floor; catches fly shadows" in out
    assert notes == ["activity.pixel_threshold: 12.0 -> 14.5"]


def test_saving_leaves_every_untouched_line_byte_identical():
    out, _notes = apply_overrides_to_yaml_text(CONFIG_TEXT, {"activity": {"pixel_threshold": 14.5}})
    before = [ln for ln in CONFIG_TEXT.splitlines() if "pixel_threshold" not in ln]
    after = [ln for ln in out.splitlines() if "pixel_threshold" not in ln]
    assert before == after


def test_saving_writes_several_keys_across_several_sections_at_once():
    out, notes = apply_overrides_to_yaml_text(CONFIG_TEXT, {
        "activity": {"pixel_threshold": 20.0},
        "rotation": {"sensitivity": 2.5, "debounce_frames": 9},
    })
    assert "pixel_threshold: 20.0" in out
    assert "sensitivity: 2.5" in out
    assert "debounce_frames: 9" in out
    assert len(notes) == 3


def test_a_key_the_file_never_had_is_added_inside_its_own_section():
    out, notes = apply_overrides_to_yaml_text(CONFIG_TEXT, {"rotation": {"min_consistency": 0.55}})
    lines = out.splitlines()
    i = lines.index("  min_consistency: 0.55")
    assert lines.index("rotation:") < i < lines.index("activity:")
    assert notes == ["rotation.min_consistency: added = 0.55"]


def test_a_key_whose_whole_section_is_missing_gets_a_new_section():
    out, notes = apply_overrides_to_yaml_text(CONFIG_TEXT, {"detection": {"k_sigma": 10.0}})
    assert "detection:" in out and "  k_sigma: 10.0" in out
    assert "new 'detection' block" in notes[0]


def test_two_new_keys_in_one_missing_section_produce_one_section():
    out, _notes = apply_overrides_to_yaml_text(
        CONFIG_TEXT, {"detection": {"k_sigma": 10.0, "min_area": 8}})
    assert out.count("detection:") == 1


def test_a_hash_inside_a_quoted_string_is_not_mistaken_for_a_comment():
    text = 'source:\n  serial: "DA#4282883"\n  width: 1280\n'
    out, _notes = apply_overrides_to_yaml_text(text, {"source": {"width": 640}})
    assert '"DA#4282883"' in out and "width: 640" in out


def test_a_bool_is_written_as_yaml_true_false_not_python_True_False():
    text = "activity:\n  normalize: true\n"
    out, _notes = apply_overrides_to_yaml_text(text, {"activity": {"normalize": False}})
    assert "normalize: false" in out


def test_a_float_keeps_looking_like_a_float():
    """`sensitivity: 2` would reload as an int; the file should keep saying what it means."""
    out, _notes = apply_overrides_to_yaml_text(CONFIG_TEXT, {"rotation": {"sensitivity": 2.0}})
    assert "sensitivity: 2.0" in out


def test_line_endings_are_not_rewritten():
    """A slider drag must not turn into a whole-file diff in git: a CRLF file stays CRLF, and an
    LF file (which is what this repo ships) does NOT gain carriage returns on Windows."""
    crlf = CONFIG_TEXT.replace("\n", "\r\n")
    out, _notes = apply_overrides_to_yaml_text(crlf, {"activity": {"pixel_threshold": 14.5}})
    assert out.count("\r\n") == crlf.count("\r\n")
    assert "\n" not in out.replace("\r\n", "")           # no bare LF crept in

    lf, _notes = apply_overrides_to_yaml_text(CONFIG_TEXT, {"activity": {"pixel_threshold": 14.5}})
    assert "\r" not in lf
    assert lf.count("\n") == CONFIG_TEXT.count("\n")


def test_nothing_to_save_leaves_the_text_alone():
    out, notes = apply_overrides_to_yaml_text(CONFIG_TEXT, {})
    assert out == CONFIG_TEXT and notes == []


def test_save_writes_only_the_changed_keys_and_advances_the_baseline(tmp_path):
    path = tmp_path / "rig.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    model = build_settings(load_config(str(path)))
    model.set("activity.pixel_threshold", 18.0)

    notes = save_settings_to_yaml(str(path), model)

    text = path.read_text(encoding="utf-8")
    assert "pixel_threshold: 18.0" in text
    assert "sensitivity: 1.0" in text                 # untouched value not restamped
    assert len(notes) == 1
    assert model.changed() == []                      # baseline advanced


def test_what_was_saved_is_what_the_next_run_loads(tmp_path):
    """The whole point of `s`: re-running the same clip must pick the tuned values up."""
    path = tmp_path / "rig.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    model = build_settings(load_config(str(path)))
    model.set("activity.pixel_threshold", 18.5)
    model.set("rotation.sensitivity", 2.4)
    model.set("rotation.min_consistency", 0.75)
    save_settings_to_yaml(str(path), model)

    reloaded = load_config(str(path))
    assert reloaded.activity.pixel_threshold == pytest.approx(18.5)
    assert reloaded.rotation.sensitivity == pytest.approx(2.4)
    assert reloaded.rotation.min_consistency == pytest.approx(0.75)


def test_saving_nothing_does_not_touch_the_file(tmp_path):
    path = tmp_path / "rig.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    model = build_settings(load_config(str(path)))
    assert save_settings_to_yaml(str(path), model) == []
    assert path.read_text(encoding="utf-8") == CONFIG_TEXT


def test_the_real_rig_config_survives_a_save_with_its_notes_intact(tmp_path):
    """Against the ACTUAL shipped file, not a toy: those comments are the validation record."""
    original = open("config/flygym_rig.yaml", encoding="utf-8").read()
    path = tmp_path / "flygym_rig.yaml"
    path.write_text(original, encoding="utf-8")
    model = build_settings(load_config(str(path)))
    model.set("activity.pixel_threshold", 15.0)
    save_settings_to_yaml(str(path), model)

    text = path.read_text(encoding="utf-8")
    assert "pixel_threshold: 15.0" in text
    assert "above the uncompressed sensor-noise floor" in text
    assert "42:1 activity" in text                    # the header block survived too
    assert text.count("\n") == original.count("\n")   # no lines gained or lost


# =========================================================================================
# 9. WIRING -- pipeline.apply_setting really changes what gets MEASURED
# =========================================================================================
FH = FW = 40
BG_LEVEL, BLOCK_LEVEL = 100, 120        # delta of exactly 20 grey levels
BLOCK = (slice(4, 7), slice(4, 7))      # 3x3 = 9 px, inside vial 1
BLOCK_PX = 9


class _ListSource(FrameSource):
    """The scripted frames, served as `Frame`s. Nothing rig-specific."""

    def __init__(self, frames: List[np.ndarray], fps: float = 10.0):
        self._frames, self._fps, self._i = frames, float(fps), 0

    def open(self) -> None:
        pass

    def read(self) -> Optional[Frame]:
        if self._i >= len(self._frames):
            return None
        img, idx = self._frames[self._i], self._i
        self._i += 1
        return Frame(image=img, index=idx, t_monotonic=float(idx),
                     t_wall_iso="2026-07-19T00:00:00")

    def close(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self):
        return (FW, FH)


def _bg() -> np.ndarray:
    f = np.zeros((FH, FW), np.uint8)
    f[2:12, 2:12] = BG_LEVEL
    return f


def _bg_with_block() -> np.ndarray:
    f = _bg()
    f[BLOCK] = BLOCK_LEVEL
    return f


def _mini_calibration(tmp_path) -> Calibration:
    mask = np.zeros((FH, FW), np.uint8)
    mask[2:12, 2:12] = 255
    png = tmp_path / "illum_mask_A.png"
    cv2.imwrite(str(png), mask)
    vial = VialROI(id=1, row=0, col=0, x=2, y=2, w=10, h=10, present=True)
    fc = FaceCalibration(name="A", vials=[vial], illum_mask_path=str(png), marker=None)
    return Calibration(image_width=FW, image_height=FH, faces={"A": fc}, created="", notes="")


def _mini_pipeline(tmp_path, frames, *, pixel_threshold=30.0, adaptive=False, **kw):
    overrides = {
        "rotation": {
            "detector": "adaptive" if adaptive else "threshold",
            "enter_threshold": 40.0, "exit_threshold": 15.0,
            "debounce_frames": 1, "min_stationary_frames": 1,
        },
        "activity": {"pixel_threshold": pixel_threshold},
        "binning": {"bin_seconds": 1.0},
    }
    overrides.update(kw)
    return TrackerPipeline(
        load_config(overrides=overrides), _mini_calibration(tmp_path),
        _ListSource(frames), ActivityLogger(output_dir=tmp_path, run_id="t", fmt="csv"),
        reference_frames={"A": _bg()}, clock="index",
    )


def _scene(n_pairs: int = 12) -> List[np.ndarray]:
    """Quiet frames alternating with/without the 9-px block: a known, constant 20-level delta."""
    return [_bg() if i % 2 == 0 else _bg_with_block() for i in range(n_pairs)]


def test_apply_setting_changes_what_the_pipeline_measures_on_the_next_frame(tmp_path):
    """ASSERTED ON THE MEASUREMENT, not on the attribute. The block differs from the background by
    exactly 20 grey levels over 9 px, so a threshold of 30 must see NOTHING and a threshold of 10
    must see all 9 -- and the switch must land on the very next frame."""
    pipe = _mini_pipeline(tmp_path, _scene(), pixel_threshold=30.0)
    measured: List[tuple] = []

    def watch(payload):
        results = payload.get("vial_results") or {}
        if 1 in results:
            measured.append((payload["index"], results[1][0]))     # (frame index, motion_px)
        if payload["index"] == 6:
            assert pipe.apply_setting("activity.pixel_threshold", 10.0) is True

    pipe.add_observer(watch)
    pipe.run()

    before = [px for idx, px in measured if idx < 6]
    after = [px for idx, px in measured if idx > 6]
    assert before and after
    assert set(before) == {0}, "a 20-level delta must be invisible at threshold 30"
    assert set(after) == {BLOCK_PX}, "the same delta must be fully visible at threshold 10"


def test_a_setting_change_leaves_exactly_one_event_naming_the_key_and_both_values(tmp_path):
    """Without this row, a run whose threshold moved at hour 40 produces one activity.csv holding
    two measurement regimes, with nothing anywhere saying that it does."""
    pipe = _mini_pipeline(tmp_path, _scene(), pixel_threshold=30.0)
    pipe.add_observer(lambda p: p["index"] == 6
                      and pipe.apply_setting("activity.pixel_threshold", 10.0))
    pipe.run()

    events = pd.read_csv(tmp_path / "events.csv", keep_default_na=False)
    rows = events[events["event"] == "setting_change"]
    assert len(rows) == 1
    detail = rows.iloc[0]["detail"]
    assert "activity.pixel_threshold" in detail
    assert "30.0" in detail and "10.0" in detail
    assert list(events.columns) == ["run_id", "iso_time", "elapsed_s", "event", "detail"]


def test_setting_the_same_value_again_is_accepted_but_logged_once_only(tmp_path):
    """A drag lands on the same step repeatedly; the log must record transitions, not mouse work."""
    pipe = _mini_pipeline(tmp_path, _scene(4), pixel_threshold=30.0)
    assert pipe.apply_setting("activity.pixel_threshold", 12.0) is True
    for _ in range(5):
        assert pipe.apply_setting("activity.pixel_threshold", 12.0) is True
    pipe.run()

    events = pd.read_csv(tmp_path / "events.csv", keep_default_na=False)
    assert (events["event"] == "setting_change").sum() == 1


def test_a_multi_step_drag_logs_an_unbroken_chain_of_transitions(tmp_path):
    """The panel applies CONTINUOUSLY (that is the point -- the operator watches the effect while
    turning the knob), so frames really were measured at each value the drag stopped on. The log
    therefore records each one, and each row's "old" must be the previous row's "new" so any
    frame's threshold can be reconstructed."""
    pipe = _mini_pipeline(tmp_path, _scene(), pixel_threshold=30.0)
    dragged = [25.0, 20.0, 15.0, 10.0]
    pipe.add_observer(lambda p: p["index"] == 4 and [
        pipe.apply_setting("activity.pixel_threshold", v) for v in dragged])
    pipe.run()

    events = pd.read_csv(tmp_path / "events.csv", keep_default_na=False)
    details = events[events["event"] == "setting_change"]["detail"].tolist()
    assert len(details) == len(dragged)
    assert details[0].startswith("activity.pixel_threshold: 30.0 ->")
    assert details[-1].endswith("-> 10.0")
    for earlier, later in zip(details, details[1:]):
        assert earlier.split("-> ")[1] == later.split(": ")[1].split(" ->")[0]


def test_the_panel_opened_mid_run_changes_the_measurement_and_leaves_an_event(tmp_path,
                                                                              monkeypatch):
    """The `t` path, end to end: monitor -> panel -> apply_setting -> measurement AND events.csv."""
    _FakePanel(monkeypatch, [])
    pipe = _mini_pipeline(tmp_path, _scene(), pixel_threshold=30.0)
    model = build_settings(pipe.config, pipeline=pipe)
    mon = MON.LiveMonitor(_mini_calibration(tmp_path), pipe.config, auto_render=False,
                          on_setting_change=pipe.apply_setting, settings_model=model)
    measured: List[tuple] = []

    def watch(payload):
        results = payload.get("vial_results") or {}
        if 1 in results:
            measured.append((payload["index"], results[1][0]))
        if payload["index"] == 6:
            mon.handle_key(ord("t"))                       # open the panel...
            mon._settings_window.apply("activity.pixel_threshold", 10.0)   # ...and drag it

    pipe.add_observer(watch)
    pipe.run()

    assert set(px for idx, px in measured if idx < 6) == {0}
    assert set(px for idx, px in measured if idx > 6) == {BLOCK_PX}
    events = pd.read_csv(tmp_path / "events.csv", keep_default_na=False)
    rows = events[events["event"] == "setting_change"]
    assert len(rows) == 1 and "30.0 -> 10.0" in rows.iloc[0]["detail"]


def test_an_unknown_key_is_refused_and_writes_nothing(tmp_path):
    pipe = _mini_pipeline(tmp_path, _scene(4))
    assert pipe.apply_setting("activity.made_up", 1.0) is False
    assert pipe.apply_setting("logger", "nonsense") is False
    pipe.run()
    events = pd.read_csv(tmp_path / "events.csv", keep_default_na=False)
    assert (events["event"] == "setting_change").sum() == 0


def test_a_dotted_key_can_never_reach_an_arbitrary_attribute_of_the_pipeline(tmp_path):
    """The routing table is a literal, so a GUI string is a dict lookup and nothing more."""
    pipe = _mini_pipeline(tmp_path, _scene(4))
    before = pipe.max_shift
    assert pipe.apply_setting("x.max_shift", 999.0) is False
    assert pipe.apply_setting("max_shift", 999.0) is False
    assert pipe.max_shift == before


# -- the rotation knobs --------------------------------------------------------------------
@pytest.mark.parametrize("key,attr,value", [
    ("rotation.sensitivity", "sensitivity", 2.5),
    ("rotation.debounce_frames", "debounce_frames", 7),
    ("rotation.min_stationary_frames", "min_stationary_frames", 6),
    ("rotation.min_consistency", "min_consistency", 0.8),
])
def test_every_rotation_knob_reaches_the_live_adaptive_detector(tmp_path, key, attr, value):
    pipe = _mini_pipeline(tmp_path, _scene(4), adaptive=True)
    assert pipe.apply_setting(key, value) is True
    assert getattr(pipe.rotation, attr) == value


def test_the_adaptive_detector_rereads_every_knob_on_the_next_frame(tmp_path):
    """These are plain mutable attributes read inside `update()`; this pins that down so a future
    refactor that caches them at construction fails here instead of in a 3-day experiment."""
    pipe = _mini_pipeline(tmp_path, _scene(4), adaptive=True)
    detector = pipe.rotation
    pipe.apply_setting("rotation.debounce_frames", 9)
    pipe.apply_setting("rotation.min_consistency", 0.9)
    pipe.apply_setting("rotation.sensitivity", 3.0)
    detector.update(_bg())
    detector.update(_bg_with_block())
    assert detector.debounce_frames == 9
    assert detector.min_consistency == 0.9
    assert detector.sensitivity == 3.0


def test_a_knob_this_runs_detector_does_not_have_is_refused_not_faked(tmp_path):
    """The fixed-threshold detector has no `sensitivity`/`min_consistency` at all, so the panel
    must be told 'not applied' rather than moving a handle that does nothing."""
    pipe = _mini_pipeline(tmp_path, _scene(4), adaptive=False)
    assert pipe.apply_setting("rotation.sensitivity", 2.0) is False
    assert pipe.apply_setting("rotation.min_consistency", 0.9) is False
    # ...but the two knobs that detector DOES have are still routed.
    assert pipe.apply_setting("rotation.debounce_frames", 5) is True
    assert pipe.rotation.debounce_frames == 5


def test_settable_keys_reports_what_this_run_will_accept(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    adaptive = _mini_pipeline(tmp_path / "a", _scene(4), adaptive=True)
    fixed = _mini_pipeline(tmp_path / "b", _scene(4), adaptive=False)
    assert "rotation.min_consistency" in adaptive.settable_keys()
    assert "rotation.min_consistency" not in fixed.settable_keys()
    assert "activity.pixel_threshold" in fixed.settable_keys()


@pytest.mark.parametrize("key,sent,expected", [
    ("activity.pixel_threshold", -5.0, 0.0),
    ("rotation.debounce_frames", 0, 1),
    ("rotation.min_stationary_frames", -3, 1),
    ("rotation.min_consistency", 4.0, 1.0),
    ("rotation.sensitivity", 0.0, 1e-3),
])
def test_a_value_the_detectors_constructor_would_reject_is_clamped_not_stored(
        tmp_path, key, sent, expected):
    """The detectors validate in their CONSTRUCTORS but not on assignment, so a live write is the
    one path that could put the state machine somewhere it refuses to start in."""
    pipe = _mini_pipeline(tmp_path, _scene(4), adaptive=True)
    assert pipe.apply_setting(key, sent) is True
    target = pipe if key.startswith("activity.") else pipe.rotation
    assert getattr(target, key.split(".")[-1]) == pytest.approx(expected)


def test_min_consistency_saved_to_the_config_is_honoured_on_the_next_run(tmp_path):
    """It has no entry in either shipped YAML, so this is the path that makes `s` mean anything
    for that knob."""
    pipe = _mini_pipeline(tmp_path, _scene(4), adaptive=True,
                          rotation={"detector": "adaptive", "min_consistency": 0.42,
                                    "debounce_frames": 1, "min_stationary_frames": 1,
                                    "enter_threshold": 40.0, "exit_threshold": 15.0})
    assert pipe.rotation.min_consistency == pytest.approx(0.42)


def test_a_config_without_min_consistency_keeps_the_detectors_own_default(tmp_path):
    pipe = _mini_pipeline(tmp_path, _scene(4), adaptive=True)
    assert pipe.rotation.min_consistency == pytest.approx(0.6)


# =========================================================================================
# 10. WIRING -- the monitor's keys and the `t` panel
# =========================================================================================
def _monitor(tmp_path, **kw):
    return MON.LiveMonitor(_mini_calibration(tmp_path), load_config(
        overrides={"activity": {"pixel_threshold": 12.0}}), auto_render=False, **kw)


def test_the_original_threshold_callback_still_fires_for_callers_that_wired_it(tmp_path):
    """BACK-COMPAT: `on_threshold_change` predates the panel and must keep working untouched."""
    seen: List[float] = []
    mon = _monitor(tmp_path, threshold_step=2.0, on_threshold_change=seen.append)
    mon.handle_key(ord("+"))
    mon.handle_key(ord("-"))
    mon.handle_key(ord("-"))
    assert seen == [14.0, 12.0, 10.0]
    assert mon.pixel_threshold == pytest.approx(10.0)


def test_the_plus_minus_keys_go_out_as_a_named_setting_when_that_callback_is_wired(tmp_path):
    seen: List[tuple] = []
    mon = _monitor(tmp_path, threshold_step=2.0,
                   on_setting_change=lambda k, v: seen.append((k, v)))
    mon.handle_key(ord("+"))
    assert seen == [("activity.pixel_threshold", 14.0)]


def test_only_one_callback_fires_when_both_are_wired(tmp_path):
    """Firing both would run the CLI's `apply_setting` twice for one keypress and write TWO
    setting_change rows for a single move -- the exact double-bookkeeping the event prevents."""
    general: List[tuple] = []
    threshold: List[float] = []
    mon = _monitor(tmp_path, on_setting_change=lambda k, v: general.append((k, v)),
                   on_threshold_change=threshold.append)
    mon.handle_key(ord("+"))
    assert len(general) == 1 and threshold == []


def test_a_raising_callback_cannot_abort_the_experiment(tmp_path):
    def boom(_k, _v):
        raise RuntimeError("pipeline gone")

    mon = _monitor(tmp_path, on_setting_change=boom)
    mon.handle_key(ord("+"))              # must not raise
    assert mon.pixel_threshold == pytest.approx(13.0)


def test_t_opens_the_settings_panel_and_t_again_closes_it(tmp_path, monkeypatch):
    _FakePanel(monkeypatch, [])
    model = build_settings(load_config())
    mon = _monitor(tmp_path, settings_model=model)

    mon.handle_key(ord("t"))
    assert mon._settings_window is not None
    mon.handle_key(ord("t"))
    assert mon._settings_window is None


def test_t_without_a_settings_model_does_nothing_rather_than_opening_an_empty_window(tmp_path):
    mon = _monitor(tmp_path)
    mon.handle_key(ord("t"))
    assert mon._settings_window is None


def test_the_panel_opened_with_t_routes_its_changes_the_same_way_the_keys_do(tmp_path, monkeypatch):
    _FakePanel(monkeypatch, [])
    seen: List[tuple] = []
    model = build_settings(load_config(overrides={"activity": {"pixel_threshold": 12.0}}))
    mon = _monitor(tmp_path, settings_model=model,
                   on_setting_change=lambda k, v: seen.append((k, v)))
    mon.handle_key(ord("t"))
    mon._settings_window.apply("activity.pixel_threshold", 20.0)
    assert seen == [("activity.pixel_threshold", 20.0)]


def test_the_plus_key_keeps_an_open_panel_showing_the_same_number_as_the_banner(tmp_path,
                                                                               monkeypatch):
    """Two widgets disagreeing about the live threshold is worse than having only one of them."""
    _FakePanel(monkeypatch, [])
    model = build_settings(load_config(overrides={"activity": {"pixel_threshold": 12.0}}))
    mon = _monitor(tmp_path, settings_model=model, threshold_step=2.0)
    mon.handle_key(ord("t"))
    mon.handle_key(ord("+"))
    assert mon.pixel_threshold == pytest.approx(14.0)
    assert model.value("activity.pixel_threshold") == pytest.approx(14.0)


def test_the_monitor_pumps_an_open_panel_from_its_render_tick(tmp_path, monkeypatch):
    fake = _FakePanel(monkeypatch, [("idle",)] * 5)
    mon = _monitor(tmp_path, settings_model=build_settings(load_config()))
    mon.handle_key(ord("t"))
    drawn = len(fake.shown)
    mon.maybe_render()
    assert len(fake.shown) > drawn


def test_a_panel_that_the_operator_closed_is_dropped_by_the_next_pump(tmp_path, monkeypatch):
    fake = _FakePanel(monkeypatch, [("key", ord("q"))])
    mon = _monitor(tmp_path, settings_model=build_settings(load_config()))
    mon.handle_key(ord("t"))
    fake.script = [("key", ord("q"))]
    mon.maybe_render()
    assert mon._settings_window is None


def test_closing_the_monitor_closes_the_panel_with_it(tmp_path, monkeypatch):
    _FakePanel(monkeypatch, [])
    mon = _monitor(tmp_path, settings_model=build_settings(load_config()))
    mon.handle_key(ord("t"))
    mon.close()
    assert mon._settings_window is None


def test_the_banner_advertises_the_settings_key(tmp_path):
    mon = _monitor(tmp_path)
    mon.latest_payload = {"state": None, "face": "A", "elapsed_s": 0.0, "fps_est": 0.0,
                          "n_rotations": 0, "pixel_threshold": 12.0}
    banner = mon._render_banner(mon.canvas_w, mon.banner_h)
    assert banner.shape == (mon.banner_h, mon.canvas_w, 3)
    # The hint text is drawn, not returned, so assert on the source of truth for it instead.
    import inspect
    assert "t settings" in inspect.getsource(MON.LiveMonitor._render_banner)


# =========================================================================================
# 11. WIRING -- the CLI flag
# =========================================================================================
def test_run_and_replay_both_accept_the_settings_flag():
    from flygym_tracker.cli import build_parser

    parser = build_parser()
    for argv in (["run", "--config", "c.yaml", "--calib", "d"],
                 ["replay", "--video", "v.avi", "--config", "c.yaml", "--calib", "d"]):
        assert parser.parse_args(argv).settings is False
        assert parser.parse_args(argv + ["--settings"]).settings is True

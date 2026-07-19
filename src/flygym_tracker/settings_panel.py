"""Drag the tracking parameters while the rig runs, and watch the numbers move.

WHY THIS EXISTS. Tuning `pixel_threshold` (and the rotation state machine around it) by editing a
YAML file, restarting, and squinting at a CSV is a minutes-long loop for a one-keystroke decision.
The scientist needs to see the effect of a knob on the live data *while turning it*, which means
the knob has to be on screen next to the data.

WHY THE SLIDERS ARE DRAWN BY HAND AND NOT `cv2.createTrackbar`. OpenCV trackbars are INTEGER-ONLY
and their label is fixed at creation -- it cannot be updated afterwards. A float parameter must
therefore be smuggled through as a scaled integer, and the operator reads

    pixel_threshold x10 = 120

instead of

    pixel_threshold  12.0 grey levels

That is precisely backwards for this panel, whose entire purpose is that a scientist reads REAL
VALUES IN REAL UNITS and decides whether 12.0 grey levels is above the sensor noise floor. A
trackbar also cannot show a help line, a group heading, or a "this one only takes effect next
run" marker. Roughly sixty lines of `cv2.line`/`cv2.putText` buy all of that, so they are drawn.

THE CAMERA GROUP IS TRI-STATE, AND THAT IS NOT DECORATION. Frame rate, exposure, gain, width and
height are each either an EXPLICIT VALUE this software sends to the sensor, or the DEVICE DEFAULT,
in which case nothing is sent at all and the camera keeps whatever MVS left it at. A slider alone
cannot say "unset" -- every position on a track is a number -- so those rows draw a distinct
default state (muted green, empty track, no handle, a `[d]` badge) and returning to it writes
`null` to the config rather than the number the camera happens to be sitting at. Before this, the
config FORCED 1280x1024 at 20 fps on every run, so "start from the MVS settings" was not
expressible at all; with it, the operator can see at a glance which settings the software is
imposing and which it is leaving alone.

Two further camera-specific rules, both learned from the rig rather than from taste:

  * LIMITS ARE READ FROM THE SENSOR (`frame_source.HikCameraSource.ranges`) when a camera is open.
    A width off the node's increment grid is REJECTED by the SDK, not clamped, so a guessed
    constant is a failed run. With no camera attached -- every test machine, and the tuning-
    between-runs case -- documented ranges are used and the panel SAYS SO next to the group
    heading, because a datasheet number displayed like a measured one will be believed.
  * WIDTH AND HEIGHT ARE START-ONLY. They are fixed at StartGrabbing time, so they are editable
    before a run and greyed out with the reason during one (`SettingsWindow.blocked`, wired to
    `pipeline.setting_block_reason`). Restarting the stream to apply them would cost a gap in a
    days-long recording plus a frame-diff baseline reset -- two measurement regimes in one file.

STRUCTURE (identical split to `live_vial_selector`, for the identical reason: no test may need a
display). Everything that can be WRONG is pure:

    Setting / SettingsModel     the values, clamping, step snapping, reset, changed(), overrides
    layout / value_at / hit     geometry + drag arithmetic, as plain functions of numbers
    render                      takes a model + a layout, returns a BGR image
    SettingsWindow              a thin driver that only pumps cv2 events into the above

THE FIRST OF THOSE FOUR NOW LIVES IN `settings_model`, and is re-exported here unchanged. The
split is by DEPENDENCY, not by design: this module imports cv2, `gui_support.require_gui` and
`live_vial_selector` (which drags `calibration` behind it), and the Qt app in `flygym_tracker.gui`
must import none of them -- a settings dialog that refuses to open because OpenCV was installed as
the headless build, for a window OpenCV is not drawing, is a support call caused by nothing.
Importing `settings_panel` still gets every name it always did.

THIS PANEL IS NO LONGER THE CANONICAL SETTINGS SURFACE. `flygym_tracker.gui` is: it edits the same
`SettingsModel` through the same `save_settings_to_yaml`, with the same green/imposed colour
meaning, and it needs no OpenCV window. This one stays reachable from `run --settings` and the
monitor's ``t`` key, because those are mid-run paths the app does not cover until the run view
lands. TWO FRONT ENDS OVER ONE MODEL IS A TEMPORARY STATE, NOT THE DESIGN: nothing asserts that
they behave identically, and the moment the app grows a run view this module should be deleted
along with its ``t`` binding rather than maintained in parallel. The dated decision and the removal
checklist are in `docs/settings_surfaces.md`.

`SettingsWindow.__init__` deliberately opens NOTHING; `open()` does. So the whole keymap can be
exercised by constructing a window and calling `handle_key` with no highgui at all.

WHAT IS EXPOSED, AND WHY IT IS SO SHORT. Only parameters that can actually be routed into the
running code are here -- a slider that moves but changes nothing is worse than no slider, because
it invites the operator to "tune" against noise. Two families named in the original brief were
left out after checking the code:

  * ``activity.normalize_by_lit_area`` -- present in both YAML files, read by NO code.
    `activity.per_frame_activity` divides by `lit_area_px` unconditionally. Honouring the flag
    would silently change what the `active_fraction_mean` COLUMN means, which is a DESIGN.md 5.3
    spec change, not a knob.
  * ``fly_tracking.DetectParams`` (`k_sigma`, `min_area`, `max_area`, `bg_kernel`) -- `detect_flies`
    is never called by `TrackerPipeline`; the whole module is standalone analysis code. There is no
    per-frame OR per-run path from these to a `run`/`replay`, so they are not "live=False", they
    are unrouted.

The machinery both of them would need is implemented and tested anyway (`kind="bool"` renders as a
two-state toggle, `odd=True` snaps to odd values for a kernel size, `live=False` labels a row
"next run" and says so in its help line), so wiring either one later is a one-line addition to
`build_settings` -- see `tests/test_settings_panel.py`.

SAVING (`s`) PRESERVES COMMENTS. `apply_overrides_to_yaml_text` rewrites only the VALUE on the
line that already defines each key, leaving every comment, blank line and key order untouched --
no round-trip through PyYAML (which drops comments) and no new dependency such as ruamel.yaml.
The rig's config is a wall of hard-won measurement notes ("above the uncompressed sensor-noise
floor; catches fly shadows"); losing them to a slider drag would be a bad trade.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# The value layer lives in `settings_model` (toolkit-free) and is RE-EXPORTED here, so every
# existing importer of `settings_panel` -- the CLI, the monitor, the 216 tests -- keeps working
# unchanged. What moved is the dependency direction: `settings_model` can now be imported by a Qt
# app that must not drag `cv2`, `gui_support.require_gui` or `live_vial_selector` behind it. See
# that module's docstring for why a settings dialog must not require an OpenCV GUI build.
from flygym_tracker.settings_model import (  # noqa: F401  (re-exported for backwards compatibility)
    BIN_SECONDS_KEY,
    CAMERA_PANEL_CAPS,
    CAMERA_ROWS,
    CAMERA_START_ONLY_ATTRS,
    CAMERA_START_ONLY_KEYS,
    COLOR_ACCENT,
    COLOR_BLOCKED,
    COLOR_DEFAULT,
    COLOR_VALUE,
    DEFAULT_TEXT,
    Setting,
    SettingsModel,
    _camera_setting,
    _cast,
    _cfg,
    _cfg_optional,
    _decimals_for,
    _find_block_end,
    _flatten,
    _split_comment,
    apply_overrides_to_yaml_text,
    build_app_settings,
    build_camera_settings,
    build_settings,
    coerce,
    format_bound,
    format_hint,
    format_value,
    format_yaml_value,
    save_settings_to_yaml,
    start_only_block_reason,
)
from flygym_tracker.gui_support import require_gui
from flygym_tracker.live_vial_selector import place_window_on_screen, screen_view_limit

DEFAULT_WINDOW = "Tracking settings"

#: Natural canvas width, in display px. Wide enough that a full-length track still has room for
#: the lo/hi endpoint labels beneath its ends without them colliding.
PANEL_WIDTH = 560
PAD = 18

#: Row metrics. One "row" is one setting: label + value, help line, track, endpoints.
ROW_H = 76
GROUP_HEADER_H = 34
HEADER_H = 88
#: Tall enough for the status line plus TWO wrapped lines of key hints. One line was not: the key
#: list measured 626 px at the size it is drawn, on a 560 px panel, so `q / ESC = close` was
#: rendered past the right edge and the operator never saw how to close the window. `key_hint_lines`
#: now wraps it to the real canvas width instead of assuming it fits.
FOOTER_H = 76
#: Baseline pitch between the wrapped key-hint lines.
HINT_LINE_H = 15
HINT_SCALE = 0.4
#: Vertical half-height of the band around a track that counts as "on the track" for a mouse
#: press. Generous on purpose: a 6 px line is not a mouse target a human can hit reliably.
TRACK_GRAB_PX = 16
TRACK_THICKNESS = 6
HANDLE_R = 8
#: The ``[d]`` back-to-device-default badge, at the right end of a nullable row's track line. The
#: track is SHORTENED by this much on those rows (see `layout`) rather than the badge being laid
#: over it, so a drag can never end underneath the button that undoes it.
DEFAULT_BADGE_W = 26
DEFAULT_BADGE_H = 20
DEFAULT_BADGE_GAP = 8

#: `cv2.waitKey` timeout for the modal `run()` loop.
POLL_MS = 30
#: `pump()` self-throttle. The monitor calls it once per processed frame (up to camera rate); the
#: panel does not need redrawing 20+ times a second, and a bounded redraw keeps a `t`-opened panel
#: harmless if it is left open for the rest of a multi-day run.
PUMP_FPS = 30.0
#: How many `pump()`/`run()` iterations a status line stays on screen (~1.5 s at POLL_MS).
MESSAGE_TTL = 50

COLOR_BG = (26, 22, 20)
COLOR_ROW = (36, 31, 28)
COLOR_ROW_SEL = (52, 45, 40)
COLOR_TEXT = (255, 255, 255)
COLOR_LABEL = (150, 150, 150)
COLOR_RULE = (70, 64, 60)
COLOR_TRACK = (70, 64, 60)
COLOR_FILL = (80, 220, 80)
COLOR_HANDLE = (235, 250, 235)
COLOR_CHANGED = (0, 200, 255)
COLOR_NEXTRUN = (0, 170, 255)
COLOR_GROUP = (200, 200, 200)
# COLOR_DEFAULT / COLOR_VALUE / COLOR_ACCENT / COLOR_BLOCKED are imported from `settings_model`:
# they carry MEANING (green = left alone, bright = imposed, grey = cannot be edited now) and the Qt
# app has to teach the same one, so there is exactly one definition and the app converts it rather
# than restating it. The rest of this list is panel chrome and stays here.
#
# CORRECTION, 2026-07-19: the comment that used to sit on COLOR_VALUE called it "cyan". These are
# BGR triples, because that is what cv2 takes, so (0, 235, 255) renders AMBER (#FFEB00) -- cyan is
# what the same numbers would be if they were RGB. The panel has always drawn amber; only the
# comment was wrong, and `gui/theme.py` converts the tuple in code so the two surfaces cannot
# drift apart again.

#: ``(key, what it does)`` -- the footer, and the terminal banner.
KEY_ROWS = [
    ("drag", "set a value"),
    ("up/down", "pick a row"),
    ("left/right", "nudge by one step"),
    ("d", "back to the camera default"),
    ("r", "reset all"),
    ("s", "save to config"),
    ("q / ESC", "close"),
]
KEY_HINT = "   ".join("%s = %s" % (k, v) for k, v in KEY_ROWS)


def key_hint_lines(width: int, scale: float = HINT_SCALE) -> List[str]:
    """`KEY_ROWS` packed greedily into lines that FIT `width`, measured with the real font.

    Written as a measurement rather than a fixed split so the list stays readable when a key is
    added: the previous single-string hint was 626 px wide on this 560 px panel, which silently
    cut off the last two entries -- including how to close the window.
    """
    budget = max(1, int(width) - 2 * PAD)
    lines: List[str] = []
    current = ""
    for key, what in KEY_ROWS:
        item = "%s = %s" % (key, what)
        candidate = ("%s   %s" % (current, item)) if current else item
        (tw, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if current and tw > budget:
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# ==========================================================================================
# Layout + hit testing (pure -- a drag can be tested with no window)
# ==========================================================================================
@dataclass
class SliderRect:
    """Where one setting's row lives on the canvas, and everything `value_at` needs.

    It carries `lo`/`hi`/`step`/`kind`/`odd` so drag arithmetic is a function of the RECT alone
    (`value_at(rect, x)`), with no lookup back into the model -- which is what makes a drag
    testable as pure arithmetic.
    """

    key: str
    index: int
    group: str
    kind: str
    lo: float
    hi: float
    step: float
    odd: bool
    x: int
    y: int
    w: int
    h: int
    track_x0: int
    track_x1: int
    track_y: int
    #: y of this row's group heading, or None when the row is not the first of its group.
    group_header_y: Optional[int] = None
    #: True if this row can be returned to its device default, i.e. it carries a ``[d]`` badge and
    #: its track is shortened to make room for one.
    nullable: bool = False
    #: True when the row currently holds NO value -- a camera setting left at the device default.
    #: `render` draws such a row with no fill and no handle, and `on_track` refuses it, so the two
    #: sides agree that there is nothing on the track to grab.
    empty: bool = False

    @property
    def track_w(self) -> int:
        return max(1, self.track_x1 - self.track_x0)

    @property
    def default_x1(self) -> int:
        """Right edge of the ``[d]`` badge -- the row's right margin, where the track would end
        on a non-nullable row."""
        return self.x + self.w - 6

    @property
    def default_x0(self) -> int:
        return self.default_x1 - DEFAULT_BADGE_W

    @property
    def default_y(self) -> int:
        return self.track_y - DEFAULT_BADGE_H // 2

    def contains(self, x: float, y: float) -> bool:
        return (self.x <= x < self.x + self.w) and (self.y <= y < self.y + self.h)

    def on_track(self, x: float, y: float) -> bool:
        """True for a press close enough to the track to count as grabbing the handle.

        NEVER true on an `empty` row. `render` draws a row that is at its device default with no
        fill and no handle -- deliberately, because there is no value to point at -- but this hit
        test used to accept a press anywhere in the 32 px band around the track, which INCLUDES
        the row's help-text line. Clicking that line to select the row therefore imposed a value
        near the slider's minimum on a setting the operator had left alone, and on a live camera
        `on_change` routes straight to the sensor: one stray click took an 88 fps multi-day
        recording down to 2.6 fps, and armed a drag that kept rewriting it. Leaving the device
        default has to be a deliberate act, so it is done with the arrow keys (which land on the
        setting's own `default_hint`, not near `lo`).
        """
        if self.empty:
            return False
        return (self.track_x0 - HANDLE_R <= x <= self.track_x1 + HANDLE_R
                and abs(y - self.track_y) <= TRACK_GRAB_PX)

    def on_default_badge(self, x: float, y: float) -> bool:
        """True for a press on the ``[d]`` badge. Always False on a row that has no default."""
        return (self.nullable
                and self.default_x0 <= x <= self.default_x1
                and self.default_y <= y <= self.default_y + DEFAULT_BADGE_H)


def panel_size(model: SettingsModel, width: int = PANEL_WIDTH) -> Tuple[int, int]:
    """Natural ``(width, height)`` of the panel for `model` -- what `open()` sizes the window to."""
    height = HEADER_H + FOOTER_H
    height += GROUP_HEADER_H * len(model.groups())
    height += ROW_H * len(model)
    return int(width), int(height)


def layout(model: SettingsModel, width: int, height: int) -> List[SliderRect]:
    """Place every row. Pure geometry: same inputs, same rectangles, no window required.

    `height` is only consulted by `render` (for the footer, which is pinned to the bottom); rows
    are stacked from the header down at fixed metrics, so a canvas taller than the content simply
    has space under the last row rather than stretching the rows apart.
    """
    rects: List[SliderRect] = []
    y = HEADER_H
    seen_groups: List[str] = []
    for i, s in enumerate(model.settings):
        header_y: Optional[int] = None
        if s.group not in seen_groups:
            seen_groups.append(s.group)
            header_y = y
            y += GROUP_HEADER_H
        track_x0 = PAD + 6
        track_x1 = max(track_x0 + 1, int(width) - PAD - 6)
        if s.nullable:
            # Give the badge its own space instead of overlapping the track: a click there must
            # mean "back to default" unambiguously, never "drag the handle to the far right".
            track_x1 = max(track_x0 + 1, track_x1 - DEFAULT_BADGE_W - DEFAULT_BADGE_GAP)
        rects.append(SliderRect(
            key=s.key, index=i, group=s.group, kind=s.kind,
            lo=float(s.lo), hi=float(s.hi), step=float(s.step), odd=bool(s.odd),
            x=PAD, y=y, w=max(1, int(width) - 2 * PAD), h=ROW_H,
            track_x0=track_x0, track_x1=track_x1, track_y=y + 48,
            group_header_y=header_y, nullable=bool(s.nullable),
            # Same condition `render` uses to draw an empty track, read from the SAME setting, so
            # the drawn row and the clickable row can never disagree about whether it has a value.
            empty=bool(s.nullable and s.value is None),
        ))
        y += ROW_H
    return rects


def value_at(rect: SliderRect, x: float) -> Any:
    """The value a press/drag at canvas `x` selects on `rect`'s track.

    Guaranteed at the ends: ``value_at(rect, rect.track_x0) == lo`` and
    ``value_at(rect, rect.track_x1) == hi``, exactly, for every kind -- a drag from one end of the
    track to the other sweeps the full declared range and nothing outside it.

    A bool has no track to sweep, so its two halves are its two states: left half OFF, right ON.
    """
    frac = (float(x) - rect.track_x0) / float(rect.track_w)
    frac = min(1.0, max(0.0, frac))
    if rect.kind == "bool":
        return frac >= 0.5
    raw = rect.lo + frac * (rect.hi - rect.lo)
    # Reuse the model's own coercion so a dragged value and a typed value snap identically.
    probe = Setting(key=rect.key, label="", value=raw, lo=rect.lo, hi=rect.hi, step=rect.step,
                    kind=rect.kind, group=rect.group, help="", odd=rect.odd)
    return probe.value


def value_fraction(rect: SliderRect, value: Any) -> float:
    """Where `value` sits along the track, in ``[0, 1]`` -- the inverse of `value_at`.

    A row at its device default has no position (`render` draws no handle for it); 0.0 is returned
    so callers that ask anyway get a number rather than a TypeError from ``float(None)``.
    """
    if value is None:
        return 0.0
    if rect.kind == "bool":
        return 1.0 if value else 0.0
    span = float(rect.hi) - float(rect.lo)
    if span <= 0:
        return 0.0
    return min(1.0, max(0.0, (float(value) - float(rect.lo)) / span))


def hit(rects: Sequence[SliderRect], x: float, y: float) -> Optional[SliderRect]:
    """The row under ``(x, y)``, or None."""
    for rect in rects:
        if rect.contains(x, y):
            return rect
    return None


# ==========================================================================================
# Rendering (no window -- returns the canvas)
# ==========================================================================================
def _text(vis: np.ndarray, text: str, org: Tuple[int, int], color=COLOR_TEXT,
          scale: float = 0.5, thickness: int = 1) -> None:
    cv2.putText(vis, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _right_text(vis: np.ndarray, text: str, right_x: int, y: int, color=COLOR_TEXT,
                scale: float = 0.5, thickness: int = 1) -> None:
    (tw, _), _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    _text(vis, text, (int(right_x - tw), y), color, scale, thickness)


def fit_text(text: str, budget_px: int, scale: float = 0.38) -> str:
    """`text`, ellipsized to fit `budget_px` at the font size it will be drawn at.

    REGRESSION THIS CLOSES. `cv2.putText` does not wrap or clip to anything -- it draws until it
    runs out of image and the rest is simply gone. The exposure row's help line, once its
    "(nothing sent - camera: 4990 us)" suffix was appended, measured 610 px on a 560 px panel: the
    end of the sentence was painted off the edge of the canvas, and on a nullable row it would
    have run under the `[d]` badge first. An ellipsis says "there is more"; silent truncation at
    the canvas edge looks like a sentence that just stops.
    """
    text = str(text)
    budget = int(budget_px)
    if budget <= 0:
        return ""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    if tw <= budget:
        return text
    for cut in range(len(text) - 1, 0, -1):
        candidate = text[:cut].rstrip() + "..."
        (tw, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw <= budget:
            return candidate
    return ""


def render(model: SettingsModel, rects: Sequence[SliderRect], width: int, height: int,
           selected: int = 0, message: str = "", subtitle: str = "",
           blocked: Optional[Dict[str, str]] = None) -> np.ndarray:
    """Draw the panel. Pure: a model plus a layout in, a BGR image out; no window is touched.

    Each row shows, top to bottom: the label and the CURRENT VALUE IN REAL UNITS, one line of help
    in rig terms, the track with its filled portion and handle, and the two endpoint values under
    the ends of the track. A row whose value differs from the file's is marked; a row that cannot
    take effect until the next run says "next run" beside its label rather than pretending.

    THREE STATES ARE VISUALLY DISTINCT, which is the whole point of the Camera group:

      * IMPOSED -- a cyan number and a filled track: this software is sending this value.
      * DEFAULT -- muted green "camera default", an empty greyed track, no handle, and a ``[d]``
        badge: nothing is sent, the camera keeps what MVS left it at. Drawing a handle here (say,
        parked at the camera's current reading) would make an unset row indistinguishable from a
        row deliberately set to that same number, which is exactly the confusion this state exists
        to remove.
      * BLOCKED -- everything dimmed and the reason printed where the value goes, for a start-only
        row while acquisition is running (`blocked[key]`, from `pipeline.setting_block_reason`).

    `blocked` is passed per draw rather than baked into the settings, because the SAME model is
    shared by the pre-run panel (where geometry is editable) and the mid-run ``t`` panel (where it
    is not).
    """
    canvas = np.full((int(height), int(width), 3), COLOR_BG, np.uint8)
    settings = model.settings
    n_changed = len(model.changed())
    blocked = blocked or {}

    # -- header ---------------------------------------------------------------------------
    _text(canvas, "TRACKING SETTINGS", (PAD, 32), COLOR_TEXT, 0.72, 2)
    if subtitle:
        _text(canvas, subtitle, (PAD, 54), COLOR_LABEL, 0.45)
    status = ("%d changed - s saves them to the config file" % n_changed) if n_changed else \
        "unchanged from the config file"
    _text(canvas, status, (PAD, 72), COLOR_CHANGED if n_changed else COLOR_LABEL, 0.45)
    cv2.line(canvas, (PAD, HEADER_H - 8), (int(width) - PAD, HEADER_H - 8), COLOR_RULE, 1)

    # -- rows -----------------------------------------------------------------------------
    for rect in rects:
        s = settings[rect.index]
        if rect.group_header_y is not None:
            _text(canvas, s.group.upper(), (PAD, rect.group_header_y + 24), COLOR_GROUP, 0.5)
            note = model.group_notes.get(s.group)
            if note:
                (gw, _), _ = cv2.getTextSize(s.group.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                _text(canvas, note, (PAD + gw + 14, rect.group_header_y + 24), COLOR_LABEL, 0.38)

        block_reason = blocked.get(s.key)
        is_blocked = block_reason is not None
        is_default = s.value is None
        is_sel = rect.index == selected
        cv2.rectangle(canvas, (rect.x, rect.y + 2), (rect.x + rect.w, rect.y + rect.h - 4),
                      COLOR_ROW_SEL if is_sel else COLOR_ROW, -1)
        if is_sel:
            cv2.rectangle(canvas, (rect.x, rect.y + 2), (rect.x + 3, rect.y + rect.h - 4),
                          COLOR_ACCENT, -1)

        label_x = rect.x + 12
        changed = s.value != model.baseline(s.key)
        if changed:
            cv2.circle(canvas, (label_x - 5, rect.y + 17), 3, COLOR_CHANGED, -1, cv2.LINE_AA)
        _text(canvas, s.label, (label_x + 4, rect.y + 22),
              COLOR_BLOCKED if is_blocked else COLOR_TEXT, 0.5)
        badge = "next run" if not s.live else ("at next start" if s.start_only else "")
        if badge:
            (lw, _), _ = cv2.getTextSize(s.label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            _text(canvas, badge, (label_x + 12 + lw, rect.y + 22),
                  COLOR_BLOCKED if is_blocked else COLOR_NEXTRUN, 0.4)

        if is_blocked:
            value_text, value_color = block_reason, COLOR_BLOCKED
        elif is_default:
            value_text, value_color = format_value(s), COLOR_DEFAULT
        else:
            value_text, value_color = format_value(s), COLOR_VALUE
        _right_text(canvas, value_text, rect.x + rect.w - 12, rect.y + 22, value_color, 0.6,
                    1 if is_blocked else 2)

        help_text = s.help
        if not s.live:
            help_text = "%s  (applies to the NEXT run, not this one)" % help_text
        elif is_default:
            # STATE FIRST, DESCRIPTION SECOND, on a defaulted row. The line is one line and the
            # camera rows are the longest on the panel, so something has to give when it is
            # ellipsized -- and what the operator needs from a row they are LEAVING ALONE is what
            # the camera is doing with it, not a definition of exposure.
            hint = format_hint(s)
            help_text = "nothing sent - %s  |  %s" % (
                hint if hint else "the camera keeps what MVS left it at", help_text)
        help_x = label_x + 4
        help_limit = (rect.default_x0 - 6) if rect.nullable else (rect.x + rect.w - 6)
        _text(canvas, fit_text(help_text, help_limit - help_x), (help_x, rect.y + 38),
              COLOR_BLOCKED if is_blocked else COLOR_LABEL, 0.38)

        if s.kind == "bool":
            _draw_toggle(canvas, rect, bool(s.value))
        else:
            _draw_track(canvas, rect, value_fraction(rect, s.value),
                        empty=is_default, dimmed=is_blocked)
            bound_color = COLOR_BLOCKED if is_blocked else COLOR_LABEL
            _text(canvas, format_bound(s, s.lo), (rect.track_x0, rect.track_y + 22),
                  bound_color, 0.38)
            _right_text(canvas, format_bound(s, s.hi), rect.track_x1, rect.track_y + 22,
                        bound_color, 0.38)
        if s.nullable and not is_blocked:
            _draw_default_badge(canvas, rect, active=is_default)

    # -- footer ---------------------------------------------------------------------------
    foot_y = int(height) - FOOTER_H
    cv2.line(canvas, (PAD, foot_y), (int(width) - PAD, foot_y), COLOR_RULE, 1)
    if message:
        _text(canvas, message, (PAD, foot_y + 22), COLOR_VALUE, 0.44)
    lines = key_hint_lines(int(width))
    base_y = int(height) - 10 - HINT_LINE_H * (len(lines) - 1)
    for i, line in enumerate(lines):
        _text(canvas, line, (PAD, base_y + i * HINT_LINE_H), COLOR_LABEL, HINT_SCALE)
    return canvas


def _draw_track(canvas: np.ndarray, rect: SliderRect, frac: float, *, empty: bool = False,
                dimmed: bool = False) -> None:
    """The track. `empty` (a row at its device default) draws NO handle and NO fill -- there is no
    value to point at, and a handle parked anywhere would read as one."""
    y = rect.track_y
    base = COLOR_BLOCKED if dimmed else COLOR_TRACK
    cv2.line(canvas, (rect.track_x0, y), (rect.track_x1, y), base, TRACK_THICKNESS, cv2.LINE_AA)
    if empty:
        return
    hx = int(round(rect.track_x0 + frac * rect.track_w))
    if hx > rect.track_x0:
        cv2.line(canvas, (rect.track_x0, y), (hx, y), COLOR_BLOCKED if dimmed else COLOR_FILL,
                 TRACK_THICKNESS, cv2.LINE_AA)
    cv2.circle(canvas, (hx, y), HANDLE_R, COLOR_BLOCKED if dimmed else COLOR_HANDLE, -1,
               cv2.LINE_AA)
    cv2.circle(canvas, (hx, y), HANDLE_R, (40, 40, 40), 1, cv2.LINE_AA)


def _draw_default_badge(canvas: np.ndarray, rect: SliderRect, active: bool) -> None:
    """The ``[d]`` affordance on a nullable row: filled while the row IS at its default.

    On the row rather than only in the footer, because "there is a way back to the camera's own
    setting" is not something an operator will go looking for in a key list -- and clickable,
    because every other control on this panel is worked with the mouse.
    """
    x0, x1 = rect.default_x0, rect.default_x1
    y0, y1 = rect.default_y, rect.default_y + DEFAULT_BADGE_H
    color = COLOR_DEFAULT if active else COLOR_RULE
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1 if active else 1)
    (tw, th), _ = cv2.getTextSize("d", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    _text(canvas, "d", ((x0 + x1) // 2 - tw // 2, (y0 + y1) // 2 + th // 2),
          (20, 20, 20) if active else COLOR_DEFAULT, 0.4)


def _draw_toggle(canvas: np.ndarray, rect: SliderRect, value: bool) -> None:
    """A boolean is two states, so it draws as two boxes -- never as a track with a handle that
    could sit anywhere between them."""
    y = rect.track_y
    mid = rect.track_x0 + rect.track_w // 2
    boxes = [("OFF", rect.track_x0, mid - 4, not value), ("ON", mid + 4, rect.track_x1, value)]
    for text, x0, x1, active in boxes:
        cv2.rectangle(canvas, (int(x0), y - 12), (int(x1), y + 12),
                      COLOR_FILL if active else COLOR_TRACK, -1)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        _text(canvas, text, (int((x0 + x1) // 2 - tw // 2), y + th // 2),
              (20, 20, 20) if active else COLOR_LABEL, 0.45)


def startup_banner(model: SettingsModel) -> str:
    """The same panel, on stdout, for an operator looking at the terminal."""
    lines = ["", "Tracking settings (drag the sliders; the run keeps going):"]
    group = None
    for s in model.settings:
        if s.group != group:
            group = s.group
            note = model.group_notes.get(group)
            lines.append("  -- %s --%s" % (group, ("   " + note) if note else ""))
        if not s.live:
            flag = "   [takes effect NEXT RUN]"
        elif s.start_only:
            flag = "   [takes effect at the NEXT START]"
        else:
            flag = ""
        lines.append("    %-34s %s%s" % (s.key, format_value(s), flag))
    lines += [""] + ["  " + line for line in key_hint_lines(PANEL_WIDTH)] + [""]
    return "\n".join(lines)


# ==========================================================================================
# Keyboard (pure)
# ==========================================================================================
#: Arrow keys do NOT survive the "low byte only" trick `live_vial_selector.decode_key` uses --
#: their codes have nothing in the low byte in common across platforms (Windows `waitKeyEx` sends
#: 2490368 for Up, GTK sends 65362, macOS/Qt 63232), and 2490368 & 0xFF is 0, i.e. indistinguishable
#: from "no key". So the full code is matched first, and only then does the low byte get a look.
_ARROW_KEYS = {
    2490368: "up", 2621440: "down", 2424832: "left", 2555904: "right",   # Windows waitKeyEx
    65362: "up", 65364: "down", 65361: "left", 65363: "right",           # GTK / Qt
    63232: "up", 63233: "down", 63234: "left", 63235: "right",           # macOS / Qt
}
_NAMED_KEYS = {13: "enter", 10: "enter", 32: "space", 27: "esc", 8: "backspace", 127: "backspace"}


def decode_key(code: Optional[int]) -> Optional[str]:
    """Map a raw `cv2.waitKeyEx` code to a key NAME, or None for "nothing was pressed"."""
    if code is None or code < 0:
        return None
    code = int(code)
    if code in _ARROW_KEYS:
        return _ARROW_KEYS[code]
    low = code & 0xFF
    if low in _NAMED_KEYS:
        return _NAMED_KEYS[low]
    if 32 < low < 127:
        return chr(low).lower()
    return None






# ==========================================================================================
# Driver (thin)
# ==========================================================================================
class SettingsWindow:
    """The panel's cv2 window: mouse in, `on_change(key, value)` out.

    Constructing one opens NOTHING -- `open()` does. That is what lets the entire keymap and the
    whole drag path be tested by building a window and calling `handle_key`/`on_mouse` directly,
    with no highgui at all (`tests/test_settings_panel.py`).

    Two ways to drive it, both of which end at the same one-iteration `pump()`:

      * `run()` -- a modal loop, for ``--settings`` at the start of a run: adjust, then close and
        let the clip/experiment play.
      * `pump()` -- one non-blocking iteration, for the panel opened with ``t`` DURING a run; the
        `LiveMonitor` calls it from its own render tick, so the panel and the data move together.

    Args:
        model: the `SettingsModel` to edit.
        on_change: called with ``(key, value)`` on every value that actually MOVES -- not on every
            mouse event, so a drag across the track produces one call per distinct step rather
            than one per pixel. May return False to mean "not applied", which the panel shows.
        on_save: called with the model on ``s``; returns operator-readable notes to print/show.
        subtitle: one line under the title, e.g. which config file this panel will save to.
        blocked: optional ``key -> reason | None``, asked AT EVERY DRAW. A key with a reason is
            drawn greyed with the reason in place of its value, and refuses edits. Wired to
            `pipeline.TrackerPipeline.setting_block_reason`, which blocks the start-only camera
            geometry while acquisition is running -- the same model object is also used by the
            pre-run panel, where those rows must stay editable, so this cannot be a property of
            the settings themselves.
    """

    def __init__(
        self,
        model: SettingsModel,
        *,
        on_change: Optional[Callable[[str, Any], Any]] = None,
        on_save: Optional[Callable[[SettingsModel], Any]] = None,
        window: str = DEFAULT_WINDOW,
        width: int = PANEL_WIDTH,
        max_view: Optional[Tuple[int, int]] = None,
        subtitle: str = "",
        blocked: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.model = model
        self.on_change = on_change
        self.on_save = on_save
        self.window = window
        self.subtitle = subtitle
        self._blocked = blocked
        self._max_view = max_view

        self.width, self.height = panel_size(model, width)
        self.rects = layout(model, self.width, self.height)
        self.selected = 0
        self.message = ""
        self.closed = False
        self.opened = False

        self._message_ttl = 0
        self._dragging: Optional[SliderRect] = None
        #: display scale, in case the natural panel is taller than this desktop (see `open`).
        self._scale = 1.0
        self._last_pump = -1e9

    # -- status line -----------------------------------------------------------------------
    def note(self, text: str, ttl: int = MESSAGE_TTL) -> None:
        self.message = str(text)
        self._message_ttl = int(ttl)

    def tick(self) -> None:
        if self._message_ttl > 0:
            self._message_ttl -= 1
            if self._message_ttl == 0:
                self.message = ""

    # -- editing ---------------------------------------------------------------------------
    def select(self, index: int) -> int:
        """Move the selection, clamped (NOT wrapped -- an operator holding Down should stop at the
        end rather than silently jump back to the top of a list they were reading)."""
        self.selected = min(len(self.rects) - 1, max(0, int(index)))
        return self.selected

    def blocked_reason(self, key: str) -> Optional[str]:
        """Why `key` cannot be edited right now, or None. Never raises -- a diagnostic that throws
        would take down the panel it is meant to annotate."""
        if self._blocked is None:
            return None
        try:
            return self._blocked(key)
        except Exception:
            return None

    def blocked_map(self) -> Dict[str, str]:
        """``{key: reason}`` for every currently un-editable row, rebuilt per draw."""
        if self._blocked is None:
            return {}
        out = {}
        for s in self.model.settings:
            reason = self.blocked_reason(s.key)
            if reason:
                out[s.key] = reason
        return out

    def _refuse_if_blocked(self, key: str) -> bool:
        """True (and a status line) if `key` is blocked. The row is left exactly as it was."""
        reason = self.blocked_reason(key)
        if reason is None:
            return False
        self.note("%s: %s" % (self.model.get(key).label, reason))
        return True

    def apply(self, key: str, value: Any) -> bool:
        """Set `key` and, only if the stored value actually moved, tell `on_change`."""
        if self._refuse_if_blocked(key):
            return False
        before = self.model.value(key)
        after = self.model.set(key, value)
        return self._changed(key, before, after)

    def nudge_selected(self, direction: int) -> bool:
        """One step on the selected row, through the same notification funnel as a drag."""
        key = self.rects[self.selected].key
        if self._refuse_if_blocked(key):
            return False
        before = self.model.value(key)
        after = self.model.nudge(key, direction)
        return self._changed(key, before, after)

    def default_selected(self) -> bool:
        """``d``: return the selected row to its device default (impose nothing).

        Goes out through the SAME `_changed` funnel as a drag, so the pipeline is told and the
        change is logged like any other -- clearing a value mid-run is a regime change too.
        """
        key = self.rects[self.selected].key
        return self.set_default(key)

    def set_default(self, key: str) -> bool:
        """Return one row to its device default. False (with a status line) if it has none."""
        if self._refuse_if_blocked(key):
            return False
        setting = self.model.get(key)
        if not setting.nullable:
            self.note("%s has no camera default - r resets it to the config file" % setting.label)
            return False
        before = self.model.value(key)
        after = self.model.to_default(key)
        return self._changed(key, before, after)

    def _changed(self, key: str, before: Any, after: Any) -> bool:
        """The ONE place a value change becomes an `on_change` call. Returns True if it moved.

        Nothing is reported when the value did not actually move: a drag fires a mouse event per
        pixel, and a pipeline that logged a `setting_change` event for each of them would bury the
        real transitions in its own noise.

        A rejected change (`on_change` returning False -- e.g. a rotation knob this run's detector
        does not have) is kept in the panel but SAID so. Silently snapping the handle back looks
        like a broken UI; silently keeping it would misrepresent the run.
        """
        if after == before:
            return False
        setting = self.model.get(key)
        if self.on_change is not None:
            try:
                result = self.on_change(key, after)
            except Exception as exc:
                self.note("%s: change failed (%s)" % (key, exc))
                return True
            if result is False:
                self.note("%s = %s not applied to this run" % (key, format_value(setting)))
                return True
        self.note("%s = %s" % (setting.label, format_value(setting)))
        return True

    def reset(self) -> None:
        """``r``: every row back to the config file's value -- except the ones that are blocked.

        A blocked row is greyed out and refuses a drag; letting ``r`` move it anyway would be the
        one way round the guard, and would leave the panel showing a value the run is not using.
        Blocked rows are restored to what they were and named, so nothing changes silently.
        """
        held = {s.key: s.value for s in self.model.settings if self.blocked_reason(s.key)}
        moved = self.model.reset()
        for key, value in held.items():
            self.model.set(key, value)
        moved = [key for key in moved if key not in held]
        for key in moved:
            if self.on_change is not None:
                try:
                    self.on_change(key, self.model.value(key))
                except Exception:
                    pass
        if held and not moved:
            self.note("nothing to reset (%d row(s) cannot change right now)" % len(held))
        elif held:
            self.note("reset %d setting(s); %d left alone (cannot change right now)"
                      % (len(moved), len(held)))
        else:
            self.note(("reset %d setting(s) to the config file" % len(moved)) if moved
                      else "nothing to reset")

    def save(self) -> None:
        if self.on_save is None:
            self.note("no config file to save to")
            return
        changed = len(self.model.changed())
        if not changed:
            self.note("nothing changed - nothing to save")
            return
        try:
            notes = self.on_save(self.model)
        except Exception as exc:
            self.note("save FAILED: %s" % exc)
            print("settings: save failed: %s" % exc)
            return
        self.note("saved %d setting(s)" % changed)
        for line in (notes or []):
            print("settings: %s" % line)

    # -- input -----------------------------------------------------------------------------
    def handle_key(self, key: Optional[str]) -> Optional[str]:
        """Apply one decoded keystroke. Returns ``"done"`` when the window should close."""
        if key is None:
            return None
        if key in ("q", "esc"):
            self.closed = True
            return "done"
        if key == "up":
            self.select(self.selected - 1)
        elif key == "down":
            self.select(self.selected + 1)
        elif key in ("left", "-", "_"):
            self.nudge_selected(-1)
        elif key in ("right", "+", "="):
            self.nudge_selected(+1)
        elif key in ("enter", "space"):
            rect = self.rects[self.selected]
            if rect.kind == "bool":
                self.apply(rect.key, not self.model.value(rect.key))
        elif key == "d":
            self.default_selected()
        elif key == "r":
            self.reset()
        elif key == "s":
            self.save()
        return None

    def on_mouse(self, event: int, x: int, y: int, _flags: int = 0, _param=None) -> None:
        """cv2 mouse callback. Canvas coords are undone by `self._scale` before hit-testing, so a
        panel shrunk to fit a small desktop still maps a click to the row under the cursor."""
        scale = self._scale or 1.0
        px, py = x / scale, y / scale
        if event == cv2.EVENT_LBUTTONDOWN:
            rect = hit(self.rects, px, py)
            if rect is None:
                return
            self.select(rect.index)
            if rect.on_default_badge(px, py):
                self.set_default(rect.key)
            elif rect.on_track(px, py):
                self._dragging = rect
                self.apply(rect.key, value_at(rect, px))
            elif rect.empty:
                # The row IS selected now, but nothing was imposed. Say how to leave the default,
                # rather than let a click that visibly did nothing read as an unresponsive UI --
                # which is what would push an operator into clicking harder, repeatedly.
                self.note("%s is at the camera default - press the LEFT/RIGHT arrow keys to set "
                          "a value" % self.model.get(rect.key).label)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging is not None:
                # Deliberately NOT re-hit-tested: once the handle is grabbed the drag follows the
                # cursor's x even if it strays off the row vertically, which is how every slider
                # the operator has ever used behaves.
                self.apply(self._dragging.key, value_at(self._dragging, px))
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = None

    # -- window ----------------------------------------------------------------------------
    def canvas(self) -> np.ndarray:
        """The panel as it would appear right now (scaled to fit the desktop if need be)."""
        img = render(self.model, self.rects, self.width, self.height,
                     selected=self.selected, message=self.message, subtitle=self.subtitle,
                     blocked=self.blocked_map())
        if self._scale != 1.0:
            img = cv2.resize(img, (max(1, int(round(self.width * self._scale))),
                                   max(1, int(round(self.height * self._scale)))),
                             interpolation=cv2.INTER_AREA)
        return img

    def open(self) -> None:
        """Create the window, size it to fit this desktop, and place it fully on screen.

        Both of those were real bugs in this project's other cv2 windows: a canvas taller than the
        desktop had its bottom rows below the screen edge (unclickable -- see
        `live_vial_selector.screen_view_limit`), and OpenCV's own window placement pushed a
        correctly-sized window off the right edge (`place_window_on_screen`). A settings panel
        whose last slider is under the taskbar is exactly as useless as a vial selector's was.
        """
        require_gui("The settings panel")
        limit = self._max_view or screen_view_limit()
        self._scale = float(min(1.0, limit[0] / float(self.width), limit[1] / float(self.height)))
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.on_mouse)
        cv2.imshow(self.window, self.canvas())
        place_window_on_screen(self.window, limit)
        self.opened = True
        self.closed = False
        print(startup_banner(self.model))

    def pump(self, timeout_ms: int = 1) -> bool:
        """One non-blocking iteration: redraw, poll one key, dispatch. False once closed.

        Self-throttled to `PUMP_FPS`, because the caller is the monitor's render tick and may call
        this once per acquired frame; an idle panel must not cost the run a redraw per frame.
        Never raises -- a display failure closes the panel rather than taking the run down with it.
        """
        if self.closed:
            return False
        if not self.opened:
            self.open()
        now = time.monotonic()
        if now - self._last_pump < 1.0 / PUMP_FPS:
            return True
        self._last_pump = now
        try:
            cv2.imshow(self.window, self.canvas())
            code = cv2.waitKeyEx(max(1, int(timeout_ms)))
            if self._window_is_gone():
                self.closed = True
                return False
            self.handle_key(decode_key(code))
            self.tick()
        except Exception as exc:
            self.note("display error: %s" % exc)
            self.closed = True
        if self.closed:
            self.close()
            return False
        return True

    def run(self, poll_ms: int = POLL_MS) -> SettingsModel:
        """Modal loop: hold the panel open until the operator closes it. Returns the model."""
        if not self.opened:
            self.open()
        try:
            while not self.closed:
                self._last_pump = -1e9          # a modal loop is not competing with anything
                if not self.pump(timeout_ms=poll_ms):
                    break
        finally:
            self.close()
        return self.model

    def _window_is_gone(self) -> bool:
        """True if the operator closed the window with its X (which counts as ``q``)."""
        try:
            return cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1
        except Exception:
            return True

    def close(self) -> None:
        """Best-effort teardown. Idempotent; safe even if `open()` was never called."""
        self.closed = True
        if not self.opened:
            return
        self.opened = False
        try:
            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
        except Exception:
            pass

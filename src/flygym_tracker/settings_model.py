"""The settings VALUE LAYER: what a tunable is, what it may hold, and how it reaches the file.

WHY THIS IS A SEPARATE MODULE FROM `settings_panel`. Everything here was `settings_panel`'s, and
still is -- that module re-exports every name below, so its cv2 panel is unchanged. What moved is
the DEPENDENCY. `settings_panel` imports `cv2`, `gui_support.require_gui` and
`live_vial_selector`, the last of which pulls in `calibration` behind it; a Qt settings dialog that
had to import all of that would refuse to open on a machine whose OpenCV is the headless build,
for a window OpenCV is not drawing. That is a support call on a rig the customer paid for, caused
by nothing.

So the rule this module keeps is narrow and checkable: NOTHING HERE MAY IMPORT A TOOLKIT. No cv2,
no PySide6, no `gui_support`, no `live_vial_selector`. `tests/test_settings_model.py` asserts it by
importing this module with `cv2` and `PySide6` blocked out of `sys.modules` -- if a future edit
reaches for a drawing helper, that test fails rather than the customer's machine.

WHAT IS HERE, and what each piece is the single source of truth for:

    Setting / coerce / SettingsModel    one row's value, its clamping/snapping, and the baseline
                                        that makes `changed()` mean "what this session altered"
    format_value / format_hint / ...    real-units display strings, shared by BOTH front ends so
                                        the cv2 panel and the Qt app cannot word a value differently
    build_settings / build_camera_...   the rows themselves, from a Config (+ optional live camera)
    apply_overrides_to_yaml_text        the comment-preserving writer -- the save button, verbatim
    start_only_block_reason             THE start-only rule, asked by the pipeline AND by the app

THE START-ONLY RULE LIVES HERE FOR ONE REASON: SO IT CANNOT BE ASKED TWICE. It used to live in
`pipeline.py`, which the Qt app cannot import (it pulls cv2, calibration and the rotation
detectors), so the app would have needed its own copy of "is Width blocked right now". Two copies
of a safety rule is one copy plus a future disagreement, and `frame_source.HikCameraSource.
is_acquiring` says so in its own docstring -- it is public precisely so that everyone asks the
same object rather than keeping private ideas about whether the stream is live.
`TrackerPipeline.setting_block_reason` is now a call to the function below, so the pipeline and the
app answer identically by construction rather than by test.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from flygym_tracker.frame_source import (
    START_ONLY_ATTRS as CAMERA_START_ONLY_ATTRS,
    camera_ranges,
    fallback_camera_ranges,
)

#: What a nullable row shows instead of a number when nothing is being imposed.
DEFAULT_TEXT = "camera default"

# ------------------------------------------------------------------------------------------
# The two colours that carry MEANING, as BGR triples (which is what cv2 takes).
#
# They live here rather than in the cv2 panel because the Qt app must teach the SAME colour: a
# green row is one this software is leaving alone, a bright row is one it is imposing. That is the
# reading the rig owner asked for, and an operator who learns it at the panel must not have to
# relearn it in the app. `gui/theme.py` CONVERTS these rather than restating them -- see
# `bgr_to_hex` there -- because a hex literal copied by eye from a BGR tuple is how the app ends up
# cyan where the panel is amber, with a comment in both places claiming they match.
# ------------------------------------------------------------------------------------------
#: The "device default, nothing sent" state. Deliberately a MUTED GREEN rather than another shade
#: of the value colour: at a glance across the panel, green rows are ones the software is leaving
#: alone and bright rows are ones it is imposing.
COLOR_DEFAULT = (120, 200, 120)
#: An explicit value THIS SOFTWARE IS IMPOSING. BGR, so this renders AMBER (#FFEB00), not cyan --
#: see `gui/theme.py` for the note on the comment that used to claim otherwise.
COLOR_VALUE = (0, 235, 255)
COLOR_ACCENT = (0, 235, 255)
#: A row that cannot be edited right now (start-only, mid-run). Dimmer than everything else.
COLOR_BLOCKED = (95, 95, 95)

#: The config keys whose value cannot take effect until acquisition (re)starts, spelled as the
#: dotted paths the panel and the router use. Derived from `frame_source.START_ONLY_ATTRS` so the
#: sensor-level fact and the config-level key list cannot disagree.
CAMERA_START_ONLY_KEYS = tuple("source.camera.%s" % a for a in CAMERA_START_ONLY_ATTRS)


def start_only_block_reason(key: str, source, *, recording: bool = False,
                            closable: bool = False) -> Optional[str]:
    """Why `key` cannot be changed right now, or None if it can be. Operator-readable.

    THE ONE PLACE THIS QUESTION IS ANSWERED. `TrackerPipeline.setting_block_reason` calls it, the
    Qt app calls it, and a future caller should call it too -- see the module docstring for why a
    second copy is a future disagreement rather than a duplication.

    Only the start-only camera geometry (`CAMERA_START_ONLY_KEYS`) is ever blocked, and only while
    the camera is actually open. Width/Height are fixed at StartGrabbing time, so applying one
    would mean stopping and restarting acquisition -- under an experiment that may have been
    recording for days, that is a gap in the series PLUS a frame-diff baseline reset (DESIGN.md
    section 5.3), i.e. two incomparable regimes in one file with nothing marking the seam. Not
    worth it for a geometry tweak, ever.

    THREE SITUATIONS, THREE MESSAGES, because a greyed control whose reason does not fit what the
    operator is looking at is worse than no reason at all:

      * `recording` -- frames have been measured. "applies at next start - this run is recording".
      * `closable` -- the caller owns the camera and has a button that closes it (the Qt app).
        Telling that operator to go and use a command-line tool would be absurd, so the message
        names the button they can actually press.
      * neither -- `run` opens the camera during vial selection and hands it to the pipeline still
        open, so a `--settings` panel can meet an already-open camera with zero frames processed.
        Geometry then belongs in the standalone `settings` command, and the message says so
        instead of leaving the operator to guess.

    Args:
        key: dotted config key, e.g. ``"source.camera.width"``.
        source: anything with an `is_acquiring` attribute -- a `HikCameraSource`, a video file
            source (which has none, so nothing is ever blocked), or None.
        recording: True once frames have actually been measured under this source.
        closable: True when the caller can close the camera itself.
    """
    if key not in CAMERA_START_ONLY_KEYS:
        return None
    if not bool(getattr(source, "is_acquiring", False)):
        return None
    if recording:
        return "applies at next start - this run is recording"
    if closable:
        return "the camera is open - close it to change this (it applies at the next start)"
    return "the camera is already open - set this with the `settings` command"


# ==========================================================================================
# Setting + model (pure)
# ==========================================================================================
@dataclass
class Setting:
    """One tunable parameter, in the units the operator thinks in.

    Attributes:
        key: dotted path into the config tree, e.g. ``"activity.pixel_threshold"``. This is also
            the routing key handed to `pipeline.TrackerPipeline.apply_setting`, so it must match
            a key that pipeline explicitly knows -- there is no reflection anywhere in the chain.
        label: what the row says on screen. Short; the `help` line carries the meaning.
        value: current value. Coerced through `SettingsModel.set` on construction of the model, so
            a model can never hold an out-of-range or off-step value.
        lo, hi: inclusive bounds. A drag to either end of the track lands EXACTLY here (see
            `value_at`) even when the span is not a whole number of steps.
        step: snap granularity, and the size of one arrow-key nudge.
        kind: ``"float"`` | ``"int"`` | ``"bool"``. A bool draws as a two-state toggle, not a
            slider -- a 0..1 track with a handle in the middle would imply a half-on state.
        group: heading this row sits under ("Activity", "Rotation").
        help: ONE short line, in rig terms, saying what moving this does. Not a restatement of the
            variable name: "how different a pixel must be to count as fly motion", not
            "the pixel threshold".
        live: True if writing it mid-run takes effect on the NEXT FRAME. False rows are labelled
            "next run" on screen and say so in their help line -- the panel must never look like
            it did something it did not do.
        odd: int only -- snap to odd values (an OpenCV kernel size must be odd).
        unit: appended to the displayed value, e.g. "grey levels", "frames".
        nullable: True if ``None`` is a legal value meaning "the software imposes nothing here,
            so the device keeps its own default". A slider alone cannot express that -- every
            position on a track is a number -- so a nullable row draws a DEFAULT state instead of
            a handle, and returning to it (the ``d`` key, or the ``[d]`` badge) writes ``null``
            rather than the number the device happens to be sitting at. Without this the operator
            could not tell "I chose 20 fps" from "the camera came up at 20 fps", which is exactly
            the distinction the rig owner asked to see at a glance.
        default_hint: what the device reports for this row right now, if known. Shown beside the
            DEFAULT state ("camera: 88.5") so the operator can see what they are leaving alone,
            and used as the landing value when a nudge takes the row OUT of the default state --
            arriving at `lo` would be a wild jump on a range like 20..50000 microseconds.
        start_only: True if the value cannot take effect until acquisition (re)starts. This is a
            property of the SETTING; whether it is currently blocked is a property of the RUN, and
            is asked of the pipeline per draw (`SettingsWindow.blocked`) rather than frozen here --
            the same model object is used by the pre-run panel, where these rows are editable, and
            by the mid-run ``t`` panel, where they must not be.
    """

    key: str
    label: str
    value: Any
    lo: float
    hi: float
    step: float
    kind: str
    group: str
    help: str
    live: bool = True
    odd: bool = False
    unit: str = ""
    nullable: bool = False
    default_hint: Optional[float] = None
    start_only: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("float", "int", "bool"):
            raise ValueError("Setting.kind must be 'float'|'int'|'bool', got %r" % (self.kind,))
        if self.value is None and not self.nullable:
            raise ValueError("%s: value is None but the setting is not nullable" % (self.key,))
        if self.kind != "bool":
            if float(self.hi) < float(self.lo):
                raise ValueError("%s: hi (%r) is below lo (%r)" % (self.key, self.hi, self.lo))
            if float(self.step) <= 0:
                raise ValueError("%s: step must be > 0, got %r" % (self.key, self.step))
        if self.odd:
            if self.kind != "int":
                raise ValueError("%s: odd=True only makes sense for an int" % (self.key,))
            # Adjacent odd values are two apart, so 2 IS this setting's step -- forced here rather
            # than left to the declaration, because any other value makes `nudge` inconsistent
            # with `coerce`. (A step of 1 sent one arrow-key press to 12, which snapped straight
            # back to 13: an arrow key that visibly did nothing.)
            self.step = 2
        self.value = coerce(self, self.value)


def coerce(setting: Setting, value: Any) -> Any:
    """Clamp `value` into ``[lo, hi]``, snap it to `step`, and cast it to `setting.kind`.

    ENDPOINTS ARE EXACT. Anything at or past a bound returns that bound verbatim, before snapping.
    Otherwise a span that is not a whole number of steps could never reach its own maximum -- with
    ``lo=0, hi=1, step=0.3`` the largest snapped value is 0.9, so dragging the handle to the far
    right of the track would leave it short of the number printed under that end of the track. The
    operator would be looking at a slider that visibly disagrees with its own scale.

    AN `odd` SETTING SNAPS TO THE ODD GRID DIRECTLY, not to `step` and then to an odd neighbour:
    snapping twice loses which way the raw value was leaning, so 9.6 and 10.4 would both land on
    11. Odd values are two apart by definition, so the odd grid IS the step grid for these.

    A non-finite value (a NaN out of some upstream division) falls back to `lo` rather than
    propagating: `lo` is the least destructive value a measurement knob can take, and NaN has no
    ordering to clamp with anyway.

    `None` SURVIVES UNTOUCHED on a nullable setting -- it is the "impose nothing" state, not a
    missing number, so it must not be clamped into the range (which would silently turn "the
    camera's own frame rate" into "20.0 fps, chosen by this software").
    """
    if value is None and setting.nullable:
        return None
    if setting.kind == "bool":
        return bool(value)

    lo, hi, step = float(setting.lo), float(setting.hi), float(setting.step)
    v = float(value)
    if not math.isfinite(v):
        v = lo
    if v <= lo:
        return _cast(lo, setting)
    if v >= hi:
        return _cast(hi, setting)
    v = 2.0 * round((v - 1.0) / 2.0) + 1.0 if setting.odd else lo + round((v - lo) / step) * step
    v = min(hi, max(lo, v))
    return _cast(v, setting)


def _cast(value: float, setting: Setting) -> Any:
    """Final cast to the setting's kind, enforcing oddness for `odd` ints."""
    if setting.kind == "int":
        n = int(round(float(value)))
        if setting.odd and n % 2 == 0:
            # Step to the nearer odd neighbour, then back inside the bounds if that overshot.
            n = n + 1 if float(value) >= n else n - 1
            n = min(int(setting.hi), max(int(setting.lo), n))
            if n % 2 == 0:
                n = n - 1 if n > int(setting.lo) else n + 1
        return n
    # Kill float dust from the snap arithmetic so 12.500000000000002 never reaches the display or
    # the YAML file.
    return round(float(value), 6)


class SettingsModel:
    """An ordered list of `Setting`s plus the values they had when the run started.

    Pure: no cv2, no window, no file I/O. `reset()`/`changed()` are both defined against the
    BASELINE captured at construction, which is by contract "what the config file said", so
    `changed()` answers the only question that matters at save time -- what has this session
    actually altered.
    """

    def __init__(self, settings: Sequence[Setting], group_notes: Optional[Dict[str, str]] = None):
        self._settings: List[Setting] = list(settings)
        self._index: Dict[str, Setting] = {}
        for s in self._settings:
            if s.key in self._index:
                raise ValueError("duplicate setting key %r" % (s.key,))
            self._index[s.key] = s
        #: values as loaded from the config file -- the target of `reset()` and the reference
        #: `changed()` compares against.
        self._baseline: Dict[str, Any] = {s.key: s.value for s in self._settings}
        #: ``group -> one line drawn beside that group's heading``. Currently carries the one thing
        #: a Camera row cannot say for itself: whether its limits were read from the camera or
        #: taken from the datasheet. A range that is merely documented must not be presented with
        #: the same authority as one the sensor reported.
        self.group_notes: Dict[str, str] = dict(group_notes or {})

    # -- queries ---------------------------------------------------------------------------
    @property
    def settings(self) -> List[Setting]:
        """The rows, in display order (the live objects -- `set`/`nudge` mutate them)."""
        return list(self._settings)

    def __len__(self) -> int:
        return len(self._settings)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def keys(self) -> List[str]:
        return [s.key for s in self._settings]

    def get(self, key: str) -> Setting:
        try:
            return self._index[key]
        except KeyError:
            raise KeyError("no such setting %r (have: %s)" % (key, ", ".join(self.keys()))) from None

    def value(self, key: str) -> Any:
        return self.get(key).value

    def baseline(self, key: str) -> Any:
        """What `key` was when the model was built, i.e. what the config file holds."""
        return self._baseline[key]

    def groups(self) -> List[str]:
        """Group names in first-appearance order."""
        out: List[str] = []
        for s in self._settings:
            if s.group not in out:
                out.append(s.group)
        return out

    def changed(self) -> List[Setting]:
        """The settings whose value differs from the file's."""
        return [s for s in self._settings if s.value != self._baseline[s.key]]

    # -- mutations -------------------------------------------------------------------------
    def set(self, key: str, value: Any) -> Any:
        """Clamp/snap/cast `value` and store it. Returns the value actually stored.

        Callers compare the return against the previous value to decide whether anything really
        moved -- a drag fires a mouse event per pixel, and only the steps that change the value
        should reach the pipeline (and therefore the events log).
        """
        setting = self.get(key)
        setting.value = coerce(setting, value)
        return setting.value

    def nudge(self, key: str, direction: int) -> Any:
        """One step up (`direction > 0`) or down. Returns the value actually stored.

        A bool is not nudged around a cycle: right/up is ON, left/down is OFF, so the same key
        always produces the same state regardless of where the toggle happens to be.

        NUDGING A ROW THAT IS AT ITS DEFAULT lands on `default_hint` (what the device is actually
        doing) rather than one step away from it, because "one step from nothing" has no meaning.
        Falling back to `lo` instead would send an operator who tapped an arrow key on the exposure
        row from the camera's 5000 microseconds straight down to 20 -- a 250x change from a
        keystroke that was meant to nudge.
        """
        setting = self.get(key)
        if setting.kind == "bool":
            return self.set(key, direction > 0)
        if setting.value is None:
            hint = setting.default_hint
            return self.set(key, float(hint) if hint is not None else float(setting.lo))
        delta = float(setting.step) * (1 if direction > 0 else -1)
        return self.set(key, float(setting.value) + delta)

    def to_default(self, key: str) -> Any:
        """Put a nullable setting back to "impose nothing". Returns the stored value.

        Refuses on a non-nullable setting rather than inventing a default for it: `pixel_threshold`
        has no "device default" to fall back to, and `reset()` (back to the config file) is the
        operation that row actually wants.
        """
        setting = self.get(key)
        if not setting.nullable:
            raise ValueError("%s has no device default to return to" % (key,))
        setting.value = None
        return None

    def reset(self) -> List[str]:
        """Put every setting back to the file's value. Returns the keys that actually moved."""
        moved = [s.key for s in self.changed()]
        for s in self._settings:
            s.value = self._baseline[s.key]
        return moved

    def mark_saved(self) -> None:
        """Adopt the current values as the new baseline -- called after a successful save, so the
        panel stops reporting changes that are now IN the file."""
        self._baseline = {s.key: s.value for s in self._settings}

    def to_overrides(self, changed_only: bool = False) -> dict:
        """The settings as a nested dict shaped like the config tree.

        ``{"activity": {"pixel_threshold": 12.0}, "rotation": {...}}`` -- i.e. exactly what
        `config.load_config(overrides=...)` deep-merges, and exactly the shape the YAML writer
        walks. `changed_only=True` narrows it to `changed()`, which is what saving writes: a
        save must not stamp untouched defaults over a file the operator hand-tuned.
        """
        source = self.changed() if changed_only else self._settings
        out: dict = {}
        for s in source:
            parts = s.key.split(".")
            node = out
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = s.value
        return out


# ==========================================================================================
# Formatting (pure)
# ==========================================================================================
def _decimals_for(step: float) -> int:
    """Decimal places that can actually distinguish two adjacent steps (0..3)."""
    step = abs(float(step))
    if step <= 0:
        return 2
    return int(min(3, max(0, math.ceil(-math.log10(step)))))


def format_value(setting: Setting) -> str:
    """The value as the operator should read it -- real units, no scaling tricks."""
    if setting.value is None:
        return DEFAULT_TEXT
    if setting.kind == "bool":
        return "ON" if setting.value else "OFF"
    if setting.kind == "int":
        text = "%d" % int(setting.value)
    else:
        text = "%.*f" % (_decimals_for(setting.step), float(setting.value))
    return ("%s %s" % (text, setting.unit)).strip()


def format_bound(setting: Setting, bound: float) -> str:
    """One end of the track's scale, formatted like the value it will become when dragged there."""
    if setting.kind == "int":
        return "%d" % int(round(float(bound)))
    return "%.*f" % (_decimals_for(setting.step), float(bound))


def format_hint(setting: Setting) -> str:
    """"what the camera is doing", for a row that is imposing nothing. Empty when unknown.

    The bounds are widened around the hint before formatting so a read-back is NEVER clamped into
    the slider's range. This is the camera reporting a fact; showing "camera: 120.0 fps" because
    the track happens to stop at 120 would be the display inventing a measurement.
    """
    hint = setting.default_hint
    if hint is None:
        return ""
    probe = Setting(key=setting.key, label="", value=hint,
                    lo=min(float(setting.lo), float(hint)),
                    hi=max(float(setting.hi), float(hint)),
                    step=setting.step, kind=setting.kind, group=setting.group,
                    help="", unit=setting.unit)
    return "camera: %s" % format_value(probe)

# ==========================================================================================
# Building the model from the real config / the running pipeline
# ==========================================================================================
def _cfg(config, path: str, default=None):
    """``config.a.b`` by dotted path, tolerating a missing branch (returns `default`)."""
    node = config
    for part in path.split("."):
        try:
            node = getattr(node, part)
        except Exception:
            return default
    return default if node is None else node


def _cfg_optional(config, path: str):
    """``config.a.b``, where a NULL value stays None instead of becoming a default.

    `_cfg` cannot be reused for the camera rows: it folds "the key is null" into "the key is
    missing" and hands back the caller's default, which is precisely the distinction the camera
    tri-state is built on. A null `frame_rate` means "impose nothing"; silently substituting 20.0
    would put the old forced value back on screen and, on save, back into the file.
    """
    node = config
    for part in path.split("."):
        try:
            node = getattr(node, part)
        except Exception:
            return None
        if node is None:
            return None
    return node


#: Upper end of each camera slider, where the sensor's own maximum is uselessly large. The
#: MV-CA013 advertises exposure up to ~10 s; a 500 px track across 15 us..10 s is ~20 ms per pixel,
#: so the ~5 ms this rig runs at would be literally unselectable by dragging. The camera's real
#: limit still wins when it is LOWER than the cap, and the cap is never allowed to clip a value the
#: config already holds (see `_camera_setting`) -- a slider must always be able to show the number
#: the run is actually using.
CAMERA_PANEL_CAPS = {
    "ExposureTime": 50000.0,
    "AcquisitionFrameRate": 120.0,
    "Gain": 20.0,
}

#: The camera rows, in display order: ``(config key, node, label, unit, kind, help)``.
#: Help lines are kept SHORT here, unlike the tracking rows: a camera row at its default prefixes
#: its state to this text on the same single line, and anything longer gets ellipsized away.
CAMERA_ROWS = [
    ("frame_rate", "AcquisitionFrameRate", "frame rate", "fps", "float",
     "frames per second delivered"),
    ("exposure_us", "ExposureTime", "exposure", "us", "float",
     "light collected per frame; more = brighter, blurrier"),
    ("gain_db", "Gain", "gain", "dB", "float",
     "electronic brightening; lifts noise too"),
    ("width", "Width", "width", "px", "int",
     "sensor width read out"),
    ("height", "Height", "height", "px", "int",
     "sensor height read out"),
]


def _camera_setting(key_attr, node, label, unit, kind, help_text, *, value, rng, hint):
    """One camera row, with the panel's range derived from `rng` (the camera's or the datasheet's).

    A value already in the config is never clipped away by `CAMERA_PANEL_CAPS`: the bounds are
    widened to include it instead. A panel that opened showing 30 fps because its slider stopped
    at 30, on a run configured for 88, would be describing a rig that does not exist.
    """
    lo, hi = float(rng.lo), float(rng.hi)
    cap = CAMERA_PANEL_CAPS.get(node)
    if cap is not None:
        hi = min(hi, cap)
    if value is not None:
        lo, hi = min(lo, float(value)), max(hi, float(value))
    if hint is not None:
        hi = max(hi, float(hint))
    step = float(rng.inc) if rng.inc > 0 else 1.0
    if kind == "int":
        step = max(1.0, round(step))
    return Setting(
        key="source.camera.%s" % key_attr, label=label, value=value,
        lo=lo, hi=max(hi, lo + step), step=step, kind=kind, group="Camera", unit=unit,
        help=help_text, nullable=True, default_hint=hint,
        start_only=key_attr in CAMERA_START_ONLY_ATTRS,
    )


def build_camera_settings(config, camera=None, *, for_run: bool = False):
    """The Camera rows plus the group note, or ``([], {})`` when there is nothing to offer.

    Values come from the live `camera` when there is one (it may have been adjusted mid-run) and
    from the config otherwise, with `None` preserved end to end in both cases. Limits come from
    `frame_source.camera_ranges`, which reads the sensor when a camera is open and falls back to
    documented values when it is not -- and the group note says WHICH, because a datasheet range
    presented like a measured one is the kind of thing that gets believed.

    `for_run=True` means "this panel belongs to a pipeline that is about to measure", and then the
    rows are built ONLY if that pipeline's source is really a camera. A replay reads from a video
    file: its config still carries a `source.camera` block (every config does), but there is no
    sensor to send anything to, and the panel's standing rule is that it must never show a slider
    the run cannot honour -- a knob that moves and changes nothing invites tuning against noise.
    The standalone `settings` command has no run at all, so it always builds them: editing the
    file for NEXT time is the entire job there.
    """
    if _cfg(config, "source.camera") is None:
        return [], {}

    ranges = camera_ranges(camera)
    live_camera = camera if callable(getattr(camera, "set_frame_rate", None)) else None
    if for_run and live_camera is None:
        return [], {}
    hints = {}
    if live_camera is not None:
        try:
            hints = live_camera.current_values() or {}
        except Exception:
            hints = {}

    settings = []
    for key_attr, node, label, unit, kind, help_text in CAMERA_ROWS:
        if live_camera is not None:
            value = getattr(live_camera, key_attr, None)
        else:
            value = _cfg_optional(config, "source.camera.%s" % key_attr)
        rng = ranges.get(node) or fallback_camera_ranges()[node]
        settings.append(_camera_setting(
            key_attr, node, label, unit, kind, help_text,
            value=None if value is None else (int(value) if kind == "int" else float(value)),
            rng=rng, hint=hints.get(key_attr),
        ))

    note = ("limits read from the camera" if any(r.live for r in ranges.values())
            else "camera not open - limits are the rig camera's, not live")
    return settings, {"Camera": note}


def build_settings(config, pipeline=None, camera=None) -> SettingsModel:
    """The settings this build can actually route, seeded from `config` (and the live `pipeline`).

    Current values come from the PIPELINE when one is given, because that is what is measuring:
    a `pixel_threshold` supplied on the command line, or already nudged with the monitor's ``+``
    key, differs from the file, and a panel that opened showing the file's number would be lying
    about the run in progress. `config` is the fallback, and always supplies the baseline that
    `reset()` returns to and `changed()` compares against... with one deliberate exception noted
    below.

    Only routable parameters are built -- see the module docstring for the two families that were
    checked and left out.

    `camera` is the `frame_source.HikCameraSource` to read limits and current values from; it
    defaults to the pipeline's own source, so a `run` gets live sensor limits for free and a
    `replay` (whose source is a video file) simply gets the documented ones.
    """
    if camera is None:
        camera = getattr(pipeline, "source", None)

    live_thr = getattr(pipeline, "pixel_threshold", None)
    thr = float(live_thr) if live_thr is not None else float(_cfg(config, "activity.pixel_threshold", 12.0))

    detector = getattr(pipeline, "rotation", None)

    def det(attr: str, cfg_path: str, default):
        """Live attribute if the detector has one, else the config, else the code's default."""
        value = getattr(detector, attr, None) if detector is not None else None
        return value if value is not None else _cfg(config, cfg_path, default)

    settings = [
        Setting(
            key="activity.pixel_threshold", label="pixel threshold", value=thr,
            lo=0.0, hi=60.0, step=0.5, kind="float", group="Activity", unit="grey levels",
            help="how different a pixel must be, frame to frame, to count as fly motion",
        ),
        Setting(
            key="rotation.sensitivity", label="sensitivity",
            value=float(det("sensitivity", "rotation.sensitivity", 1.0)),
            lo=0.2, hi=5.0, step=0.1, kind="float", group="Rotation",
            help="higher = the drum is called moving on smaller shifts (adaptive detector only)",
        ),
        Setting(
            key="rotation.debounce_frames", label="debounce frames",
            value=int(det("debounce_frames", "rotation.debounce_frames", 4)),
            lo=1, hi=30, step=1, kind="int", group="Rotation", unit="frames",
            help="quiet frames in a row before the drum counts as stopped",
        ),
        Setting(
            key="rotation.min_stationary_frames", label="settling frames",
            value=int(det("min_stationary_frames", "rotation.min_stationary_frames", 3)),
            lo=1, hi=30, step=1, kind="int", group="Rotation", unit="frames",
            help="frames skipped after the drum stops, before activity is measured again",
        ),
        Setting(
            key="rotation.min_consistency", label="direction consistency",
            value=float(det("min_consistency", "rotation.min_consistency", 0.6)),
            lo=0.0, hi=1.0, step=0.05, kind="float", group="Rotation",
            help="how straight motion must be to be the drum turning, not flies milling about",
        ),
    ]
    camera_settings, group_notes = build_camera_settings(
        config, camera, for_run=pipeline is not None)
    return SettingsModel(settings + camera_settings, group_notes=group_notes)

# ==========================================================================================
# Saving back to the YAML the run started from (comments preserved)
# ==========================================================================================
_KEY_LINE = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)[ \t]*:(?P<rest>.*)$")


def format_yaml_value(value: Any) -> str:
    """A Python value as the scalar this config file would have written by hand.

    `None` BECOMES `null`, AND THE LINE STAYS. Deleting the key would have been the other way to
    say "impose nothing", and it is wrong twice over here. First, `config.load_config` deep-merges
    the packaged `default_config.yaml` UNDER the run config, so a key that is absent is not an
    absent value -- it is whatever the packaged default says, which is the opposite of what the
    operator asked for. Second, this module's entire reason for hand-editing YAML instead of
    round-tripping it through PyYAML is to preserve the comments that justify the numbers; a
    deleted line takes its comment with it. `null` is what both shipped configs already use for an
    unset camera field, so it is also what the file's own conventions expect.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return "%d" % value
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return "%.1f" % value          # 12 -> "12.0": keep a float looking like a float
        return repr(round(value, 6))
    return str(value)


def _split_comment(rest: str) -> Tuple[str, str]:
    """Split a value from its trailing ``# comment``, ignoring ``#`` inside quotes."""
    quote = None
    for i, ch in enumerate(rest):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            return rest[:i], rest[i:]
    return rest, ""


def _flatten(overrides: dict, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in overrides.items():
        path = "%s.%s" % (prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def apply_overrides_to_yaml_text(text: str, overrides: dict) -> Tuple[str, List[str]]:
    """Rewrite just the VALUES named by `overrides` in `text`. Returns ``(new_text, notes)``.

    Comments, blank lines, key order and indentation all survive, because nothing is re-serialised:
    the file is walked line by line, an indentation stack turns each mapping line into its dotted
    path, and a line whose path is being overridden has only the span between its ``:`` and its
    ``#`` replaced. That is what lets the rig config keep the measurement notes that justify its
    numbers ("above the uncompressed sensor-noise floor; catches fly shadows") through a slider
    drag -- a PyYAML round-trip would delete every one of them, and ruamel.yaml would be a new
    dependency for one keystroke.

    A key not already in the file is APPENDED inside its parent block if that block exists, else a
    new block is appended at the end. `notes` describes every edit in operator-readable form.

    Scope, stated plainly: this handles the block-style, plain-scalar YAML this project's configs
    are written in. It does not attempt flow mappings (``{a: 1}``), anchors, multi-line scalars or
    lists of mappings; a key living inside one of those is simply not found, and is appended
    instead -- which is visible in `notes` rather than silent.
    """
    flat = _flatten(overrides or {})
    if not flat:
        return text, []

    # Line endings are preserved rather than normalised. These files live in a git repo; rewriting
    # every line of a LF file to CRLF (which is what a plain text-mode write does on Windows) would
    # turn a one-value slider change into a whole-file diff.
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    stack: List[Tuple[int, str]] = []          # (indent, key) of the enclosing mappings
    notes: List[str] = []
    remaining = dict(flat)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue
        m = _KEY_LINE.match(line)
        if not m:
            continue
        indent = len(m.group("indent").expandtabs(4))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([k for _ind, k in stack] + [m.group("key")])
        rest = m.group("rest")
        if path in remaining:
            value, comment = _split_comment(rest)
            old = value.strip()
            new = format_yaml_value(remaining.pop(path))
            head = "%s%s: %s" % (m.group("indent"), m.group("key"), new)
            comment = comment.strip()
            if comment:
                # Keep the comment in the column it was hand-aligned to, when the new value still
                # leaves room for it; fall back to two spaces when the number got longer.
                column = len(line) - len(comment)
                head += " " * max(2, column - len(head))
                head += comment
            lines[i] = head
            notes.append("%s: %s -> %s" % (path, old if old else "(empty)", new))
        stack.append((indent, m.group("key")))

    # -- keys the file never had: put them inside their parent block if it exists ------------
    # `lines` is re-scanned per key, so two new keys sharing a missing parent get ONE new block:
    # the first iteration creates it, the second finds it.
    for path in sorted(remaining):
        value = remaining[path]
        parts = path.split(".")
        parent = ".".join(parts[:-1])
        insert_at, child_indent = _find_block_end(lines, parent)
        if insert_at is None:
            if lines and lines[-1].strip():
                lines.append("")
            for depth, part in enumerate(parts[:-1]):
                lines.append("%s%s:" % ("  " * depth, part))
            lines.append("%s%s: %s" % ("  " * (len(parts) - 1), parts[-1],
                                       format_yaml_value(value)))
            notes.append("%s: added (new '%s' block)" % (path, parent))
        else:
            lines.insert(insert_at, "%s%s: %s" % (" " * child_indent, parts[-1],
                                                  format_yaml_value(value)))
            notes.append("%s: added = %s" % (path, format_yaml_value(value)))

    out = newline.join(lines)
    if text.endswith(("\n", "\r")) or not text:
        out += newline
    return out, notes


def _find_block_end(lines: List[str], parent: str) -> Tuple[Optional[int], int]:
    """Where a new child of `parent` should be inserted, plus the indent it should carry.

    The index is just past the block's last line WITH CONTENT, not past its trailing blank lines
    and comments -- appending after those would drop the new key on the far side of the blank line
    that visually separates this block from the next one, i.e. it would look like it belonged to
    the following section.

    ``(None, 0)`` when `parent` is not a block in this file; ``(len(lines), 0)`` when `parent` is
    empty (a top-level key, appended at the end).
    """
    if not parent:
        return (len(lines), 0)
    want = parent.split(".")
    stack: List[Tuple[int, str]] = []
    start: Optional[int] = None
    start_indent = 0
    last_content: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _KEY_LINE.match(line)
        if m:
            indent = len(m.group("indent").expandtabs(4))
            while stack and stack[-1][0] >= indent:
                stack.pop()
            path = [k for _ind, k in stack] + [m.group("key")]
            stack.append((indent, m.group("key")))
            if start is None:
                if path == want:
                    start, start_indent, last_content = i, indent, i
                continue
            if indent <= start_indent:
                return ((last_content or start) + 1, start_indent + 2)
        if start is not None:
            last_content = i
    if start is None:
        return (None, 0)
    return ((last_content or start) + 1, start_indent + 2)


def save_settings_to_yaml(path: str, model: SettingsModel) -> List[str]:
    """Write `model`'s CHANGED values into the YAML at `path`. Returns operator-readable notes.

    Only what changed is written, so the file keeps every value the operator never touched. On
    success the model's baseline advances (`mark_saved`), which is what makes the panel stop
    showing those rows as pending. An empty return means there was nothing to save.
    """
    overrides = model.to_overrides(changed_only=True)
    if not overrides:
        return []
    text = ""
    if path and os.path.exists(path):
        # newline="" on BOTH ends: read without translating CRLF away, write without translating
        # LF back into CRLF. See `apply_overrides_to_yaml_text` on why that matters here.
        with open(path, "r", encoding="utf-8", newline="") as f:
            text = f.read()
    new_text, notes = apply_overrides_to_yaml_text(text, overrides)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    model.mark_saved()
    return notes


# ==========================================================================================
# The app's model: the tracking rows, what run.bat used to put in a batch file, and the camera
# ==========================================================================================
#: How long one output row covers. It is NOT an app path -- it is a measurement parameter, and it
#: belongs in the experiment's YAML next to the thresholds it will be compared against. It is here
#: rather than in `build_settings` because `build_settings` serves the cv2 panel too, whose
#: standing rule is that every row it shows can be routed into the RUNNING pipeline
#: (`TrackerPipeline.apply_setting`); bin size cannot be, since `ActivityLogger` fixes its binning
#: at construction. So the app -- which edits the file for next time -- offers it, and the mid-run
#: panel does not. `live=False` is what makes the row say "next run" on its own face.
BIN_SECONDS_KEY = "binning.bin_seconds"


def build_app_settings(config) -> SettingsModel:
    """`build_settings` plus the one row `run.bat` kept in a batch file: how long a bin is.

    THE REGRESSION THIS CLOSES. `run.bat` carried four values in a header block the operator was
    expected to edit in Notepad -- config path, calibration folder, output folder and BIN_SECONDS.
    Three of those are app paths and belong in the app's own state; the fourth is a measurement
    parameter and belongs in the config file, which is where every other measurement parameter
    already is and where `run_meta.json` will record it. Leaving it in the batch file meant the
    number that decides what a row of the results MEANS was the one setting with no UI at all.

    No camera is passed: the app opens the camera only when the operator asks, and rebuilds this
    model when it does (see `gui.settings_view.SettingsView.rebuild_camera_rows`).
    """
    base = build_settings(config)
    bin_seconds = Setting(
        key=BIN_SECONDS_KEY, label="bin size",
        value=float(_cfg(config, "binning.bin_seconds", 60.0)),
        lo=1.0, hi=3600.0, step=1.0, kind="float", group="Recording", unit="s",
        live=False,
        help="how long one row of the results covers; changing it changes what a number means",
    )
    tracking = [s for s in base.settings if s.group != "Camera"]
    camera = [s for s in base.settings if s.group == "Camera"]
    return SettingsModel(tracking + [bin_seconds] + camera, group_notes=base.group_notes)

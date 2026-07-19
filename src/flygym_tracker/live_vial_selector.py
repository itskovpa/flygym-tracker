"""Draw every vial by hand, as a polygon, on the LIVE camera feed. One click per vertex.

WHY THIS EXISTS (read before changing anything here). Automatic vial detection is ABANDONED for
this rig: the drum is cylindrical, the tubes are foreshortened at the edges, and every attempt to
find them from brightness produced ROIs the rig owner had to fix by hand anyway. The verdict was
blunt -- the detector is not to be used, and every session starts with the user selecting the
vials themselves.

So this module is deliberately, aggressively SIMPLE, and must stay that way:

    left click            add one vertex to the vial being drawn
    ENTER                 that vial is done -> store it, start the next one (needs >= 3 points)
    BACKSPACE             remove the last vertex
    u                     undo the whole PREVIOUS vial and re-open it for editing
    c                     clear the vial currently being drawn
    SPACE                 freeze / unfreeze the feed (the drum moves; hold a frame to click)
    q  or  ESC            finish early and keep what has been drawn so far

There are NO drag handles, NO snapping, NO auto-fit, NO seeding from a detector. A vertex lands
where it was clicked. If that sounds primitive, that is the requirement.

The feed stays LIVE while the clicks are collected: the loop reads and re-shows a frame on every
iteration (~30 fps) and the polygon-in-progress is drawn over whatever frame just arrived, so the
operator is looking at the real rig rather than a stale still. SPACE freezes it for precision
work, and the frozen state is impossible to miss on screen.

DRAW ONCE PER RIG, NOT ONCE PER ROUND. `load_or_select_vials` is the entry point a session
should use: if the target folder already holds a hand-drawn bundle it offers it back
(``Found saved vial positions (16 vials, saved 2026-07-19 02:14). Load them? [Y/n]:``) and
skips drawing entirely; otherwise the operator draws, and what they drew is saved immediately so
the next round can offer it. The saved bundle covers BOTH drum faces from the one drawing --
the faces present in the same orientation, so face B carries face A's coordinates verbatim (see
`calibration.build_two_face_calibration_from_polygons`).

Everything that can be wrong is in `SelectorState` / `decode_key` / `handle_key` / `render_frame`
-- pure, headlessly testable code. `select_vials_live` is a thin driver that only pumps cv2
events into them (same split as `roi_editor`, for the same reason: no test may need a display).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from flygym_tracker.calibration import (
    SavedSelection,
    build_two_face_calibration_from_polygons,
    load_calibration,
    save_calibration,
    saved_selection,
)
from flygym_tracker.frame_source import FrameSource
from flygym_tracker.gui_support import require_gui
from flygym_tracker.types import Calibration

#: One polygon vertex, ``[x, y]`` in FULL-FRAME image pixels (never view pixels).
Point = List[int]
#: One vial: ``[[x, y], ...]``, >= 3 points, in click order.
Polygon = List[Point]

DEFAULT_WINDOW = "Select vials"
#: Fallback cap on the displayed canvas, used only when the desktop cannot be measured. The real
#: limit comes from `screen_view_limit()` -- see the regression note there.
DEFAULT_MAX_VIEW = (1280, 960)
#: Deducted from the usable desktop before scaling the frame. The vertical figure is the part
#: that matters: `SM_CYFULLSCREEN` is the CLIENT height of a maximised window, but the selector's
#: window is not maximised -- it also carries a title bar and is placed a little below the top of
#: the screen by the window manager. Measured on the rig laptop: ~39 px of chrome and a ~25 px
#: drop, so the canvas must give back more than the chrome alone or the last rows land under the
#: taskbar. Horizontal slack is cosmetic; there is far more width to spare than height.
VIEW_MARGIN = (24, 72)
#: `cv2.waitKeyEx` timeout per iteration -- the frame rate of the preview (~30 fps).
POLL_MS = 33
#: How many iterations a status/nag line stays on screen (~1.5 s at POLL_MS).
MESSAGE_TTL = 45

COLOR_DONE = (80, 220, 80)        # completed vials
COLOR_CURRENT = (0, 235, 255)     # the vial being drawn
COLOR_FIRST = (0, 140, 255)       # its first vertex (the one ENTER closes back to)
COLOR_TEXT = (255, 255, 255)
COLOR_LIVE = (120, 255, 120)
COLOR_FROZEN = (60, 200, 255)
COLOR_SOURCE = (200, 200, 200)

KEY_HINT = ("click=point   ENTER=next vial   BACKSPACE=undo point   "
            "u=undo vial   c=clear   SPACE=freeze   q=finish")


# ==========================================================================================
# State (pure)
# ==========================================================================================
class SelectorState:
    """All the geometry and bookkeeping of a selection session. No cv2, no window, no I/O.

    Vials are collected in DRAW ORDER: `polygons` is what has been finished, `current` is the one
    being clicked right now. That ordering is the contract -- `calibration.
    build_calibration_from_polygons` numbers vials 1..N by it, which is what the operator saw
    labelled on screen.
    """

    #: A polygon needs three points to enclose any area at all; two is a line and measures nothing.
    MIN_VERTICES = 3

    def __init__(self, n_vials: int = 16, face: str = "A", source_label: str = ""):
        if int(n_vials) < 1:
            raise ValueError("n_vials must be >= 1, got %r" % (n_vials,))
        self.n_vials = int(n_vials)
        self.face = str(face)
        #: What is actually being shown, e.g. "CAMERA" or "FILE Good Markers.avi". Named on the
        #: HUD because a recorded clip and the rig camera look IDENTICAL on screen -- both are
        #: grey IR frames of the same drum -- and drawing vial positions against yesterday's video
        #: while believing it is the camera would silently miscalibrate the whole experiment.
        self.source_label = str(source_label)
        self.frozen = False
        self.finished = False          # set by q/ESC: "stop now, keep what I have"
        self.message = ""
        self._message_ttl = 0
        self._polygons: List[Polygon] = []
        self._current: Polygon = []

    # -- queries ---------------------------------------------------------------------------
    @property
    def polygons(self) -> List[Polygon]:
        """Completed vials, in draw order (a copy -- callers cannot mutate the state)."""
        return [[list(p) for p in poly] for poly in self._polygons]

    @property
    def current(self) -> Polygon:
        """The vial being drawn, in click order (a copy)."""
        return [list(p) for p in self._current]

    @property
    def is_complete(self) -> bool:
        """True once `n_vials` vials have been drawn."""
        return len(self._polygons) >= self.n_vials

    @property
    def done(self) -> bool:
        """True when the driver loop should stop: all vials drawn, or finished early."""
        return self.finished or self.is_complete

    @property
    def vial_number(self) -> int:
        """1-based number of the vial being drawn right now."""
        return len(self._polygons) + 1

    # -- mutations -------------------------------------------------------------------------
    def add_vertex(self, x: float, y: float) -> Point:
        """Add one clicked vertex (rounded to whole pixels) and return it."""
        point = [int(round(float(x))), int(round(float(y)))]
        self._current.append(point)
        self.note("vial %d: %d point(s)" % (self.vial_number, len(self._current)))
        return point

    def finish_vial(self) -> bool:
        """ENTER: store the current polygon and start a fresh one. False (+ a nag) if < 3 points."""
        if len(self._current) < self.MIN_VERTICES:
            self.note("vial %d needs at least %d points - only %d clicked"
                      % (self.vial_number, self.MIN_VERTICES, len(self._current)))
            return False
        self._polygons.append(list(self._current))
        self._current = []
        self.note("vial %d saved (%d points)" % (len(self._polygons), len(self._polygons[-1])))
        return True

    def undo_vertex(self) -> bool:
        """BACKSPACE: drop the last clicked vertex of the vial in progress."""
        if not self._current:
            self.note("no point to undo (vial %d is empty)" % self.vial_number)
            return False
        self._current.pop()
        self.note("removed a point (%d left)" % len(self._current))
        return True

    def undo_vial(self) -> bool:
        """``u``: re-open the previous vial for editing.

        The previous polygon becomes the one being drawn, so the operator can BACKSPACE into it
        and re-click. Anything already clicked for the current vial is dropped (and said so) --
        one key, one obvious outcome, no hidden buffer to reason about.
        """
        if not self._polygons:
            self.note("no completed vial to undo")
            return False
        dropped = len(self._current)
        self._current = self._polygons.pop()
        extra = " (discarded %d in-progress point(s))" % dropped if dropped else ""
        self.note("re-opened vial %d for editing%s" % (len(self._polygons) + 1, extra))
        return True

    def clear(self) -> bool:
        """``c``: throw away the vial in progress and start it over."""
        if not self._current:
            self.note("vial %d is already empty" % self.vial_number)
            return False
        n = len(self._current)
        self._current = []
        self.note("cleared %d point(s) from vial %d" % (n, self.vial_number))
        return True

    def toggle_freeze(self) -> bool:
        """SPACE: hold the last frame (the drum rotates; clicking a moving tube is hopeless)."""
        self.frozen = not self.frozen
        self.note("FROZEN - press SPACE to go live again" if self.frozen else "live again")
        return self.frozen

    def finish_early(self) -> None:
        """``q``/ESC: stop now and keep every vial finished so far."""
        self.finished = True
        self.note("finished with %d vial(s)" % len(self._polygons))

    # -- transient status line --------------------------------------------------------------
    def note(self, text: str, ttl: int = MESSAGE_TTL) -> None:
        """Show `text` on the HUD for `ttl` iterations."""
        self.message = text
        self._message_ttl = int(ttl)

    def tick(self) -> None:
        """Age the status line by one iteration (called once per frame by the driver)."""
        if self._message_ttl > 0:
            self._message_ttl -= 1
            if self._message_ttl == 0:
                self.message = ""


# ==========================================================================================
# Keyboard (pure)
# ==========================================================================================
#: `cv2.waitKeyEx` low bytes that are not printable characters. 10/13 are both Enter (keypad and
#: main), 127 is Backspace on some macOS/Qt builds.
_NAMED_KEYS = {13: "enter", 10: "enter", 8: "backspace", 127: "backspace",
               32: "space", 27: "esc"}


def decode_key(code: Optional[int]) -> Optional[str]:
    """Map a raw `cv2.waitKeyEx` code to a key NAME, or None for "nothing was pressed".

    Only the low byte matters here: every key this selector uses is either ASCII or one of the
    named control keys above, and high-bit noise differs per platform/build (GTK sends 65288 for
    Backspace, Windows sends 8) for no useful reason.
    """
    if code is None or code < 0:
        return None
    low = int(code) & 0xFF
    if low in _NAMED_KEYS:
        return _NAMED_KEYS[low]
    if 32 < low < 127:
        return chr(low).lower()
    return None


def handle_key(state: SelectorState, key: Optional[str]) -> Optional[str]:
    """Apply one keystroke to `state`. Returns ``"done"`` when the loop should stop, else None.

    This is the whole keymap; the driver contains no key handling of its own.
    """
    if key is None:
        return "done" if state.done else None
    if key == "enter":
        state.finish_vial()
    elif key == "backspace":
        state.undo_vertex()
    elif key == "u":
        state.undo_vial()
    elif key == "c":
        state.clear()
    elif key == "space":
        state.toggle_freeze()
    elif key in ("q", "esc"):
        state.finish_early()
    return "done" if state.done else None


# ==========================================================================================
# Rendering (no window -- returns the canvas)
# ==========================================================================================
def screen_view_limit(fallback: Tuple[int, int] = DEFAULT_MAX_VIEW) -> Tuple[int, int]:
    """Largest canvas that actually fits on this operator's desktop, in display pixels.

    REGRESSION THIS EXISTS FOR. The cap used to be the hard-coded ``(1280, 960)``. On the rig
    laptop (2880x1800 panel at 200% scaling, so a 1440x900 desktop) a 1280x1024 camera frame
    became a 1200x960 canvas -- but only 829 px of window height exist there, so the bottom ~130
    rows of every frame sat BELOW the screen edge: not visible, and impossible to click. That is
    the bottom of the lower vial row, i.e. exactly the part of the tube the operator most needs
    to enclose. Nothing warned about it; the window simply looked fine and was quietly truncated.

    `SM_CXFULLSCREEN`/`SM_CYFULLSCREEN` give the client area a maximised window would get -- the
    taskbar and title bar already deducted -- expressed in the SAME coordinate space the OpenCV
    window is laid out in, provided this process stays DPI-unaware (which is why nothing here
    calls `SetProcessDPIAware`: doing so would change how the window itself is rendered).

    Falls back to `fallback` wherever the desktop cannot be measured (non-Windows, no display).
    """
    try:
        import ctypes
        user32 = ctypes.windll.user32                      # type: ignore[attr-defined]
        width, height = user32.GetSystemMetrics(16), user32.GetSystemMetrics(17)
    except Exception:
        return fallback
    if width <= 0 or height <= 0:
        return fallback
    return (max(320, int(width) - VIEW_MARGIN[0]), max(240, int(height) - VIEW_MARGIN[1]))


def view_scale(image_size: Tuple[int, int], max_view: Optional[Tuple[int, int]] = None) -> float:
    """Display scale for a frame of `image_size`: shrink to fit `max_view`, NEVER enlarge.

    `max_view` defaults to whatever actually fits this desktop (`screen_view_limit`), so the
    whole frame is always reachable by the mouse. Capping at 1.0 keeps a frame that already fits
    at an exact 1:1 mapping between screen pixels and image pixels.
    """
    if max_view is None:
        max_view = screen_view_limit()
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        return 1.0
    return float(min(max_view[0] / float(w), max_view[1] / float(h), 1.0))


def _text(vis: np.ndarray, text: str, org: Tuple[int, int], color=COLOR_TEXT,
          scale: float = 0.6) -> None:
    """Light text on a dark outline -- readable over both a bright vial and a black gap."""
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _scaled(points: Sequence[Sequence[float]], scale: float) -> np.ndarray:
    return np.round(np.asarray(points, dtype=np.float64).reshape(-1, 2) * scale).astype(np.int32)


def render_frame(image: np.ndarray, state: SelectorState, scale: float = 1.0) -> np.ndarray:
    """Build the BGR canvas: live frame + completed vials + the polygon in progress + HUD.

    Pure: takes a frame, returns an image. Nothing here touches a window, which is what lets the
    driver loop be tested with a stubbed highgui.
    """
    img = np.asarray(image)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if scale != 1.0:
        vis = cv2.resize(vis, (max(1, int(round(vis.shape[1] * scale))),
                               max(1, int(round(vis.shape[0] * scale)))),
                         interpolation=cv2.INTER_AREA)

    # --- finished vials: closed outline + their number at the centroid --------------------
    for i, poly in enumerate(state.polygons):
        pts = _scaled(poly, scale)
        cv2.polylines(vis, [pts], True, COLOR_DONE, 2, cv2.LINE_AA)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        _text(vis, str(i + 1), (cx - 8, cy + 6), COLOR_DONE, 0.8)

    # --- the vial being drawn: dots, the chain so far, and a preview of the closing edge ---
    current = state.current
    if current:
        pts = _scaled(current, scale)
        if len(pts) >= 2:
            cv2.polylines(vis, [pts], False, COLOR_CURRENT, 2, cv2.LINE_AA)
            # Thin closing edge back to the first point: shows the shape ENTER would store,
            # without pretending the polygon is finished.
            cv2.line(vis, (int(pts[-1][0]), int(pts[-1][1])), (int(pts[0][0]), int(pts[0][1])),
                     COLOR_CURRENT, 1, cv2.LINE_AA)
        for j, (px, py) in enumerate(pts):
            cv2.circle(vis, (int(px), int(py)), 5 if j == 0 else 4,
                       COLOR_FIRST if j == 0 else COLOR_CURRENT, -1, cv2.LINE_AA)

    # --- frozen: a border the operator cannot fail to notice --------------------------------
    if state.frozen:
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), COLOR_FROZEN, 6)

    _draw_hud(vis, state)
    return vis


def _draw_hud(vis: np.ndarray, state: SelectorState) -> None:
    """Status band: which vial, how many points, where the picture comes from, the keys, status."""
    band_h = 114
    band = vis.copy()
    cv2.rectangle(band, (0, 0), (vis.shape[1], band_h), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.55, vis, 0.45, 0, dst=vis)

    n_done = len(state.polygons)
    head = "face %s   vial %d of %d   points: %d   done: %d" % (
        state.face, min(state.vial_number, state.n_vials), state.n_vials,
        len(state.current), n_done)
    _text(vis, head, (12, 26), COLOR_TEXT, 0.7)

    # "PLAYING"/"FROZEN" is only about the picture being updated. It deliberately does NOT say
    # "LIVE": that word was read as "this is the camera" when the window was in fact showing a
    # recorded file, which is the one confusion this HUD must never cause. What the frames come
    # from is stated separately, in full, on its own line.
    live = "FROZEN" if state.frozen else "PLAYING"
    _text(vis, live, (vis.shape[1] - 150, 26), COLOR_FROZEN if state.frozen else COLOR_LIVE, 0.8)

    if state.source_label:
        _text(vis, state.source_label, (12, 52), COLOR_SOURCE, 0.55)
    _text(vis, KEY_HINT, (12, 76), COLOR_TEXT, 0.5)
    if state.message:
        _text(vis, state.message, (12, 100), COLOR_CURRENT, 0.55)


def startup_banner(state: SelectorState) -> str:
    """The same instructions, on stdout, for an operator who is looking at the terminal."""
    return "\n".join([
        "",
        "Showing: %s" % (state.source_label or "unknown source"),
        "Select %d vial(s) on face %s by drawing a polygon around each one." % (
            state.n_vials, state.face),
        "  left click   add a vertex",
        "  ENTER        finish this vial and move to the next (>= 3 vertices)",
        "  BACKSPACE    remove the last vertex",
        "  u            undo the previous vial and re-open it",
        "  c            clear the vial being drawn",
        "  SPACE        freeze / unfreeze the picture",
        "  q / ESC      finish early, keeping the vials drawn so far",
        "",
    ])


# ==========================================================================================
# Driver (thin)
# ==========================================================================================
def source_label(source: FrameSource) -> str:
    """Describe where the frames come from, for the HUD and the terminal.

    Derived from the source OBJECT rather than passed in by the caller: a label that can disagree
    with reality is worse than none, and this is the fact the operator most needs to be sure of
    before drawing vial positions -- a recorded clip of this rig is visually indistinguishable
    from the camera pointed at it.
    """
    path = getattr(source, "path", None)
    if path:
        return "FILE  %s  (recorded - not the camera)" % os.path.basename(str(path))
    serial = getattr(source, "serial", None)
    return "CAMERA  %s (live)" % serial if serial else "CAMERA (live)"


def _window_is_gone(window: str) -> bool:
    """True if the operator closed the window (its X button counts as "finish early")."""
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return True


def select_vials_live(
    source: FrameSource,
    n_vials: int = 16,
    face: str = "A",
    window: str = DEFAULT_WINDOW,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    max_view: Optional[Tuple[int, int]] = None,
    poll_ms: int = POLL_MS,
) -> List[Polygon]:
    """Draw `n_vials` polygons on a LIVE feed. Returns them as ``[[[x, y], ...], ...]``.

    The window shows the frames as they arrive and redraws the polygon-in-progress over each one,
    so the operator is clicking on the rig as it is now, not on a still. SPACE holds the last
    frame when precision matters.

    Args:
        source: any `frame_source.FrameSource` -- `HikCameraSource` on the rig,
            `VideoFileSource` for a dry run (at EOF the last frame is held so the drawing in
            progress is never lost). It is opened here if it is not open already; CLOSING IT IS
            THE CALLER'S JOB (use ``with source:``), because the caller may still want frames.
        n_vials: how many vials to collect. The loop ends when they are all drawn.
        face: face name, shown on the HUD ("A"/"B").
        window: OpenCV window title.
        on_frame: optional hook called with every newly READ frame (HxW grayscale). This is how a
            caller gets hold of the image the polygons were drawn on -- the CLI keeps the last one
            for the illumination mask and the overlay.
        max_view: cap on the displayed size; larger frames are scaled down and clicks scaled back
            up (see `view_scale`). None (the default) measures what fits this desktop, so no part
            of the frame can end up below the screen edge where it cannot be clicked.
        poll_ms: `cv2.waitKeyEx` timeout, i.e. the preview's frame interval.

    Returns:
        One polygon per vial, in draw order. Fewer than `n_vials` if the operator finished early
        (q/ESC/closing the window); possibly empty, which means "nothing was selected".

    Raises:
        SystemExit: if this OpenCV build cannot open a window (`gui_support.require_gui`).
        RuntimeError: if the source yields no frame at all -- there is nothing to draw on.
    """
    require_gui("The live vial selector")
    state = SelectorState(n_vials=n_vials, face=face, source_label=source_label(source))
    source.open()   # idempotent on both FrameSource implementations

    scale = [1.0]   # boxed: the mouse callback needs the value the driver computes

    def on_mouse(event, sx, sy, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            s = scale[0] or 1.0
            state.add_vertex(sx / s, sy / s)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    print(startup_banner(state))

    image: Optional[np.ndarray] = None
    at_eof = False
    try:
        while not state.done:
            if image is None or (not state.frozen and not at_eof):
                frame = source.read()
                if frame is None:                      # video ran out; the camera never does
                    at_eof = True
                    if image is None:
                        raise RuntimeError(
                            "the frame source returned no frames - nothing to select vials on "
                            "(is the video empty, or the camera not delivering?)")
                    state.note("end of video - holding the last frame", ttl=10 ** 9)
                else:
                    image = frame.image
                    scale[0] = view_scale((image.shape[1], image.shape[0]), max_view)
                    if on_frame is not None:
                        on_frame(image)

            cv2.imshow(window, render_frame(image, state, scale[0]))
            command = handle_key(state, decode_key(cv2.waitKeyEx(poll_ms)))
            state.tick()
            if command == "done":
                break
            if _window_is_gone(window):
                break                                   # closed with the X == finish early
    finally:
        try:
            cv2.destroyWindow(window)
            cv2.waitKey(1)
        except Exception:
            pass
    return state.polygons


# ==========================================================================================
# Reuse-or-draw: the entry point every round of every session goes through
# ==========================================================================================
def _format_saved_time(created: str) -> str:
    """`Calibration.created` as ``YYYY-MM-DD HH:MM``, or whatever it was if it is not a timestamp."""
    try:
        return datetime.fromisoformat(created).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return created or "unknown time"


def _stdin_is_interactive() -> bool:
    """True if there is a terminal on stdin that a question could actually be answered on."""
    try:
        return bool(sys.stdin is not None and sys.stdin.isatty())
    except Exception:
        return False


def reuse_question(saved: SavedSelection) -> str:
    """The one-line prompt offering the saved vial positions back.

    It names what was actually found, because the two kinds do not deserve the same trust and
    the operator cannot see the difference from the folder name.
    """
    if saved.hand_drawn:
        return ("Found saved vial positions (%d vials, drawn by hand, saved %s). "
                "Load them? [Y/n]: " % (saved.n_vials, _format_saved_time(saved.created)))
    return ("Found %d AUTO-DETECTED vial boxes (not hand-drawn, saved %s) - these are the ones "
            "known to sit crookedly on the tubes. Load them anyway? [y/N]: "
            % (saved.n_vials, _format_saved_time(saved.created)))


def prompt_reuse(saved: SavedSelection,
                 input_fn: Optional[Callable[[str], str]] = None) -> bool:
    """Ask whether to reuse `saved`. The DEFAULT depends on what was found.

    A hand-drawn selection defaults to YES: it is the non-destructive answer, since saying no
    costs one keystroke while a mistaken yes would throw away a whole clicking session. Older
    auto-detected boxes default to NO, because the detector was retired for producing ROIs that
    had to be corrected by hand every time -- silently reusing them would quietly reintroduce
    exactly the misalignment the drawing flow exists to avoid.

    An unanswerable stdin (no terminal) takes the same default, so an unattended start never
    hangs and never silently accepts the weaker option.

    `input_fn` defaults to the builtin `input` LOOKED UP AT CALL TIME -- binding it as a default
    argument value would freeze the builtin at import and silently ignore any later redirection.
    """
    default = saved.hand_drawn
    if input_fn is None and not _stdin_is_interactive():
        # Nobody is there to answer, and drawing needs a person with a mouse. Using what is
        # already saved is the only action that can actually succeed, so take it and SAY SO
        # rather than prompting into the void or failing a scripted run.
        print("no terminal to ask on - using the %d saved vial position(s) (%s)"
              % (saved.n_vials, "hand-drawn" if saved.hand_drawn else "auto-detected boxes"))
        return True

    ask = input_fn if input_fn is not None else input
    try:
        answer = ask(reuse_question(saved))
    except (EOFError, KeyboardInterrupt):
        print("(no answer - %s)" % ("keeping the saved positions" if default
                                    else "ignoring the auto-detected boxes"))
        return default
    text = str(answer).strip().lower()
    if not text:
        return default
    return text.startswith("y") if not default else not text.startswith("n")


@dataclass
class SelectionResult:
    """What a round's vial selection produced, however it was obtained."""
    polygons: List[Polygon]
    calibration: Optional[Calibration]
    reused: bool                       # True = loaded from disk, no drawing happened
    out_dir: str

    @property
    def n_vials(self) -> int:
        return len(self.polygons)


def load_or_select_vials(
    source: FrameSource,
    out_dir: str,
    n_vials: int = 16,
    faces: Sequence[str] = ("A", "B"),
    window: str = DEFAULT_WINDOW,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    input_fn: Optional[Callable[[str], str]] = None,
    reuse: Optional[bool] = None,
) -> SelectionResult:
    """Start a round: offer the saved vial positions, else draw them live. Then SAVE them.

    This is the single path both ``select-vials`` and the run flow are meant to call, so a
    session behaves identically however it was launched:

      1. if `out_dir` already holds a hand-drawn bundle, ask ``Load them? [Y/n]`` -- yes (the
         default) loads it and NO drawing happens at all;
      2. otherwise (or on "no") the operator draws every vial on the live feed;
      3. what was drawn is saved to `out_dir` immediately, so the next round can offer it back.

    The saved bundle is self-contained -- polygons, image size, face assignment, illumination
    masks and overlays -- so a later session needs nothing but this directory.

    Args:
        source: the live camera (or a video, for a dry run). Opened by the selector; CLOSING IT
            IS THE CALLER'S JOB.
        out_dir: the bundle directory to offer, and to save into.
        n_vials: vials to draw when drawing happens (default 16).
        faces: faces the drawn polygons apply to. The default ("A", "B") writes the 32-vial
            two-face bundle this rig wants: the drum's faces present in the same orientation, so
            both carry the SAME coordinates (`build_two_face_calibration_from_polygons`).
        window: OpenCV window title.
        on_frame: forwarded to `select_vials_live`.
        input_fn: injected for testing; defaults to the builtin `input`, looked up when asked.
        reuse: None = ask when a saved bundle exists (the normal case); True = reuse it without
            asking; False = always draw, ignoring anything saved.

    Returns:
        A `SelectionResult`. ``polygons`` is empty only if the operator drew nothing, in which
        case NOTHING is written and `calibration` is None. The returned calibration always has
        its illumination-mask paths RESOLVED for this machine, so it can be handed straight to
        `pipeline.TrackerPipeline` whether it was just drawn or loaded from disk.
    """
    saved = None if reuse is False else saved_selection(out_dir)
    if saved is not None and (reuse is True or prompt_reuse(saved, input_fn)):
        print("using the saved vial positions (%d vials, face(s) %s) - no drawing needed"
              % (saved.n_vials, ", ".join(saved.faces)))
        return SelectionResult(polygons=saved.polygons, calibration=load_calibration(out_dir),
                               reused=True, out_dir=out_dir)

    last: dict = {"image": None}

    def capture(image: np.ndarray) -> None:
        last["image"] = image
        if on_frame is not None:
            on_frame(image)

    polygons = select_vials_live(source, n_vials=n_vials, face=faces[0], window=window,
                                 on_frame=capture)
    if not polygons:
        return SelectionResult(polygons=[], calibration=None, reused=False, out_dir=out_dir)

    frame = last["image"]
    height, width = frame.shape[:2]
    calib, masks, overlays = build_two_face_calibration_from_polygons(
        polygons, frame, (width, height), faces=faces)
    save_calibration(calib, masks, out_dir, overlay=overlays)
    # Saved with RELATIVE mask paths (so the bundle stays movable), but handed back with them
    # resolved -- exactly what `load_calibration` returns in the reuse branch above, so a caller
    # can feed either straight to the pipeline. Re-saving this object needs
    # `calibration.relativize_mask_paths` first, same as any loaded bundle.
    calib.resolve_mask_paths(os.path.abspath(out_dir))
    return SelectionResult(polygons=polygons, calibration=calib, reused=False, out_dir=out_dir)

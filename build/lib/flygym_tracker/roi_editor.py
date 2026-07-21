"""Interactive 4-vertex (QUAD) vial-ROI editor for one drum face.

WHY THIS EXISTS
---------------
The drum is a CYLINDER. Tubes near the left/right edge of the frame curve away from the camera
and are foreshortened, so an axis-aligned rectangle either spills onto the dark surround or
clips the tube. Measured on the real 2-face bundle (`calib_faces`), face A's edge vials sit at
lit fractions of 0.28 (vial 1) and 0.50 (vial 8) against ~0.95 for the central ones -- i.e. half
to three quarters of those ROIs is measuring nothing. `types.VialROI.quad` lets the ROI follow
the taper; this module is how a human sets it.

The workflow the rig owner asked for: edit ONE face before an experiment, save, and have the
shapes used for the whole run and for BOTH faces. The second half is
`calibration.transfer_quads` (the faces present in the SAME orientation -- verified, identity
correlation +0.60, flips strongly negative -- so shapes transfer directly and are then re-snapped
to the destination face's own marker-derived columns). The CLI stitches the two together:
``flygym-tracker edit-rois --calib DIR [--face A]``.

STRUCTURE: PURE LOGIC + A THIN cv2 DRIVER
-----------------------------------------
Everything that decides anything is a pure function or a plain object with no cv2 window
attached, and is unit-tested headlessly in `tests/test_roi_editor.py`:

  * `EditorState`   -- the quads, selection, undo/redo stack, hit-testing, lit-fraction readout.
  * `ViewTransform` -- zoom/pan, screen <-> image coordinate mapping.
  * `DragController`-- press/move/release/wheel semantics, in IMAGE coordinates.
  * `decode_key`    -- raw `cv2.waitKeyEx` codes (which differ per platform) -> key names.
  * `handle_key`    -- one keystroke -> a state mutation plus an optional command for the driver.
  * `render_view`   -- the whole frame composite, returned as an ndarray (draws, never displays).

`run_roi_editor` is then only: create window, pump events, hand them to the above, `imshow` the
returned image, and translate the returned command into "save and exit" / "quit". No geometry,
no measurement and no selection logic lives inside the loop.

CONTROLS
--------
  mouse
    drag a VERTEX (grab within `grab_radius` px)   -- reshape the ROI; a magnifier pops up
    drag INSIDE an ROI                             -- move the whole ROI
    left-click an ROI                              -- select it
    right / middle drag                            -- pan
    wheel                                          -- zoom about the cursor
  keys
    Tab, n / p        next / previous vial            v      cycle the keyboard-selected vertex
    arrows            nudge selected vertex, else the whole ROI, by 1 px
    c                 copy this ROI's SHAPE to all vials (each keeps its own centre/column)
    r / R             reset this vial / ALL vials to the auto-detected quad
    z / y             undo / redo   (Ctrl+Z / Ctrl+Y also work)
    + / - / 0         zoom in / out / reset view          m   pin the magnifier on
    l                 toggle the illuminated-mask tint    h   toggle this help
    s                 SAVE and exit
    q / Esc           quit WITHOUT saving -- press twice when there are unsaved edits
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from flygym_tracker.calibration import (
    Quad,
    apply_quads_to_face,
    bbox_from_quad,
    face_quads,
    quad_lit_fraction,
)
from flygym_tracker.types import FaceCalibration

#: Grab radius (image px) for picking up a vertex with the mouse.
DEFAULT_GRAB_RADIUS = 14
#: Zoom limits for `ViewTransform`.
MIN_ZOOM, MAX_ZOOM = 0.15, 12.0
#: Default window budget; the initial view is fitted inside it.
DEFAULT_VIEW_SIZE = (1500, 860)
#: Magnifier: source crop half-width (image px) and the on-screen panel size (px).
MAG_SRC_HALF = 34
MAG_PANEL = 240

_VERT_NAMES = ("TL", "TR", "BR", "BL")
#: Shown in the status bar when a `q` would discard unsaved edits (press it again to confirm).
QUIT_WARNING = "UNSAVED EDITS -- press q again to DISCARD them, or s to save"


# ==========================================================================================
# View transform (pure)
# ==========================================================================================
@dataclass
class ViewTransform:
    """Zoom + pan of the image inside the window.

    ``screen = (image - offset) * zoom``. `ox`/`oy` are the image coordinates sitting at the
    view's top-left corner, which makes both directions one line and keeps `matrix()` (fed
    straight to `cv2.warpAffine`) trivially consistent with `to_screen`.
    """

    zoom: float = 1.0
    ox: float = 0.0
    oy: float = 0.0

    @classmethod
    def fit(cls, image_size: Tuple[int, int], view_size: Tuple[int, int]) -> "ViewTransform":
        """A view showing the whole image, centred, never magnified beyond 1:1."""
        iw, ih = float(image_size[0]), float(image_size[1])
        vw, vh = float(view_size[0]), float(view_size[1])
        z = min(vw / max(iw, 1.0), vh / max(ih, 1.0), 1.0)
        z = min(max(z, MIN_ZOOM), MAX_ZOOM)
        return cls(zoom=z, ox=(iw - vw / z) / 2.0, oy=(ih - vh / z) / 2.0)

    def to_screen(self, pt: Sequence[float]) -> Tuple[float, float]:
        return ((float(pt[0]) - self.ox) * self.zoom, (float(pt[1]) - self.oy) * self.zoom)

    def to_image(self, pt: Sequence[float]) -> Tuple[float, float]:
        return (float(pt[0]) / self.zoom + self.ox, float(pt[1]) / self.zoom + self.oy)

    def matrix(self) -> np.ndarray:
        """2x3 affine for `cv2.warpAffine` producing exactly `to_screen`'s mapping."""
        z = float(self.zoom)
        return np.array([[z, 0.0, -z * self.ox], [0.0, z, -z * self.oy]], dtype=np.float32)

    def zoom_by(self, factor: float, anchor_screen: Sequence[float]) -> "ViewTransform":
        """Zoom by `factor`, keeping the image point under `anchor_screen` where it is."""
        ax, ay = float(anchor_screen[0]), float(anchor_screen[1])
        ix, iy = self.to_image((ax, ay))
        self.zoom = min(max(self.zoom * float(factor), MIN_ZOOM), MAX_ZOOM)
        self.ox = ix - ax / self.zoom
        self.oy = iy - ay / self.zoom
        return self

    def pan_by(self, dx_screen: float, dy_screen: float) -> "ViewTransform":
        """Pan by a SCREEN-pixel delta (drag the image with the cursor)."""
        self.ox -= float(dx_screen) / self.zoom
        self.oy -= float(dy_screen) / self.zoom
        return self

    def clamp(self, image_size: Tuple[int, int], view_size: Tuple[int, int],
              margin: float = 0.25) -> "ViewTransform":
        """Keep the image from being panned entirely out of the window.

        `margin` is the fraction of the view that may show empty space beyond an image edge, so
        there is always something to grab hold of and pan back.
        """
        iw, ih = float(image_size[0]), float(image_size[1])
        vw, vh = float(view_size[0]) / self.zoom, float(view_size[1]) / self.zoom
        self.ox = min(max(self.ox, -margin * vw), iw - (1.0 - margin) * vw)
        self.oy = min(max(self.oy, -margin * vh), ih - (1.0 - margin) * vh)
        return self


# ==========================================================================================
# Editor state (pure)
# ==========================================================================================
@dataclass(frozen=True)
class Hit:
    """What the cursor is over: a vertex of a vial, or the interior of one."""

    kind: str                  # "vertex" | "roi"
    vial: int                  # index into EditorState.quads
    vertex: Optional[int] = None


class EditorState:
    """The editable quads for one face, plus selection and an undo/redo history.

    No cv2 window, no rendering, no I/O -- every method here is a deterministic function of the
    state, which is what makes the editor testable headlessly (`tests/test_roi_editor.py`).

    Args:
        face_cal: the face being edited. Vials without a `quad` get their bbox as a rectangle
            (`calibration.vial_quad`), so an OLD, quad-less bundle opens with sensible shapes.
        illum_mask: full-frame illuminated mask (255 = trackable). Optional; without it
            `lit_fraction` returns NaN and the editor is a pure geometry tool.
        grab_radius: vertex pick-up radius in IMAGE pixels.
        image_size: ``(width, height)``; defaults to the mask's shape when one is given.
    """

    def __init__(
        self,
        face_cal: FaceCalibration,
        illum_mask: Optional[np.ndarray] = None,
        grab_radius: int = DEFAULT_GRAB_RADIUS,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.face_cal = face_cal
        self.illum_mask = illum_mask
        self.grab_radius = int(grab_radius)
        if image_size is not None:
            self.image_size = (int(image_size[0]), int(image_size[1]))
        elif illum_mask is not None:
            self.image_size = (int(illum_mask.shape[1]), int(illum_mask.shape[0]))
        else:
            self.image_size = None

        #: The auto/loaded shapes, used by `reset_vial` and as the reference boxes for
        #: `copy_shape_to_all` (so a copy is idempotent and column widths survive it).
        self.base_quads: List[Quad] = face_quads(face_cal)
        self.quads: List[Quad] = copy.deepcopy(self.base_quads)

        self.selected: int = 0
        self.selected_vertex: Optional[int] = None
        self.dirty: bool = False
        self._undo: List[List[Quad]] = []
        self._redo: List[List[Quad]] = []
        self._lit_cache: Dict[int, float] = {}

    # -- basics ---------------------------------------------------------------------------
    @property
    def n_vials(self) -> int:
        return len(self.quads)

    def vial(self, index: Optional[int] = None):
        return self.face_cal.vials[self.selected if index is None else index]

    def quad(self, index: Optional[int] = None) -> Quad:
        return self.quads[self.selected if index is None else index]

    def select(self, index: int, vertex: Optional[int] = None) -> int:
        """Select a vial by index (wraps around) and optionally one of its vertices."""
        if self.n_vials:
            self.selected = int(index) % self.n_vials
        self.selected_vertex = None if vertex is None else int(vertex) % 4
        return self.selected

    def next_vial(self) -> int:
        return self.select(self.selected + 1)

    def prev_vial(self) -> int:
        return self.select(self.selected - 1)

    def cycle_vertex(self) -> Optional[int]:
        """Step the keyboard vertex selection TL -> TR -> BR -> BL -> whole-ROI -> TL ..."""
        self.selected_vertex = 0 if self.selected_vertex is None else self.selected_vertex + 1
        if self.selected_vertex > 3:
            self.selected_vertex = None
        return self.selected_vertex

    # -- undo / redo ----------------------------------------------------------------------
    def push_undo(self) -> None:
        """Snapshot the quads. Call ONCE per user gesture -- e.g. at mouse-down, not per
        mouse-move -- so one drag is one undo step."""
        self._undo.append(copy.deepcopy(self.quads))
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self.quads))
        self.quads = self._undo.pop()
        self._invalidate()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self.quads))
        self.quads = self._redo.pop()
        self._invalidate()
        self.dirty = True
        return True

    def _invalidate(self, index: Optional[int] = None) -> None:
        if index is None:
            self._lit_cache.clear()
        else:
            self._lit_cache.pop(int(index), None)

    def _touch(self, index: Optional[int] = None) -> None:
        self.dirty = True
        self._invalidate(index)

    # -- edits ----------------------------------------------------------------------------
    def set_vertex(self, vial: int, vertex: int, x: float, y: float, push: bool = False) -> None:
        """Place one vertex at an absolute image position (the drag path)."""
        if push:
            self.push_undo()
        pt = self.quads[int(vial)][int(vertex)]
        pt[0], pt[1] = self._clip_point(x, y)
        self._touch(vial)

    def move_vertex(self, vial: int, vertex: int, dx: float, dy: float, push: bool = True) -> None:
        """Nudge one vertex by a delta (the arrow-key path)."""
        if push:
            self.push_undo()
        pt = self.quads[int(vial)][int(vertex)]
        pt[0], pt[1] = self._clip_point(pt[0] + dx, pt[1] + dy)
        self._touch(vial)

    def move_roi(self, vial: int, dx: float, dy: float, push: bool = True) -> None:
        """Translate a whole quad by a delta."""
        if push:
            self.push_undo()
        for pt in self.quads[int(vial)]:
            pt[0], pt[1] = self._clip_point(pt[0] + dx, pt[1] + dy)
        self._touch(vial)

    def reset_vial(self, vial: int, push: bool = True) -> None:
        """Restore one vial's auto-detected quad."""
        if push:
            self.push_undo()
        self.quads[int(vial)] = copy.deepcopy(self.base_quads[int(vial)])
        self._touch(vial)

    def reset_all(self, push: bool = True) -> None:
        if push:
            self.push_undo()
        self.quads = copy.deepcopy(self.base_quads)
        self._touch()

    def copy_shape_to_all(self, src: Optional[int] = None, normalize: bool = True,
                          push: bool = True) -> int:
        """Stamp the source vial's SHAPE onto every other vial. Returns how many changed.

        The point of the feature is that all 16 tubes on a face are the same object seen at
        different angles, so the operator should only have to get ONE right. What must NOT be
        copied is the source's position or its column width -- the columns genuinely differ
        (measured on face A: 118..153 px wide).

        So with ``normalize=True`` (default) the source quad is expressed in the normalised
        coordinates of the source's ORIGINAL (auto-detected) bounding box and re-stamped into
        each target's ORIGINAL bounding box: the taper/skew transfers, while each vial keeps its
        own centre, width and height.

        Normalising by the ORIGINAL box rather than the edited quad's own box is essential and
        not a detail -- an edited quad's bounding box has already shrunk to fit the edit, so
        normalising by it would map every corner back to 0 or 1 and copy a plain RECTANGLE,
        silently discarding the very taper the operator drew. Using the original box also makes
        repeated copies idempotent, since it never changes.

        ``normalize=False`` copies the raw shape (source quad minus its centroid, re-centred on
        each target's original centroid), i.e. every ROI ends up literally the same size.
        """
        if push:
            self.push_undo()
        s = self.selected if src is None else int(src)
        shape = self.quads[s]
        changed = 0
        if normalize:
            sx, sy, sw, sh = bbox_from_quad(self.base_quads[s])
            norm = [((px - sx) / float(sw), (py - sy) / float(sh)) for px, py in shape]
        else:
            cx, cy = _centroid(shape)
            offs = [(px - cx, py - cy) for px, py in shape]
        for i in range(self.n_vials):
            if i == s:
                continue
            if normalize:
                bx, by, bw, bh = bbox_from_quad(self.base_quads[i])
                new = [self._clip_point(bx + u * bw, by + v * bh) for u, v in norm]
            else:
                tcx, tcy = _centroid(self.base_quads[i])
                new = [self._clip_point(tcx + ox, tcy + oy) for ox, oy in offs]
            if new != self.quads[i]:
                changed += 1
            self.quads[i] = [list(p) for p in new]
        self._touch()
        return changed

    def _clip_point(self, x: float, y: float) -> List[int]:
        """Round to the pixel grid and keep the point inside the frame (when the size is known)."""
        ix, iy = int(round(float(x))), int(round(float(y)))
        if self.image_size is not None:
            ix = min(max(ix, 0), int(self.image_size[0]))
            iy = min(max(iy, 0), int(self.image_size[1]))
        return [ix, iy]

    # -- queries --------------------------------------------------------------------------
    def hit_test(self, x: float, y: float, radius: Optional[float] = None) -> Optional[Hit]:
        """What is under the cursor at image position (x, y)?

        Priority is what makes the editor feel predictable: a vertex of the SELECTED vial beats
        any other vertex (so a shared edge between two neighbouring vials still lets you grab the
        one you are working on), a vertex beats an interior, and the selected vial's interior
        beats a neighbour's.
        """
        r = float(self.grab_radius if radius is None else radius)
        best = self._nearest_vertex(x, y, r, only=self.selected)
        if best is None:
            best = self._nearest_vertex(x, y, r)
        if best is not None:
            return best
        if self.n_vials and _point_in_quad(self.quads[self.selected], x, y):
            return Hit("roi", self.selected)
        for i in range(self.n_vials):
            if _point_in_quad(self.quads[i], x, y):
                return Hit("roi", i)
        return None

    def _nearest_vertex(self, x: float, y: float, radius: float,
                        only: Optional[int] = None) -> Optional[Hit]:
        best: Optional[Hit] = None
        best_d = radius * radius
        idxs = range(self.n_vials) if only is None else [only]
        for i in idxs:
            for j, (px, py) in enumerate(self.quads[i]):
                d = (px - x) ** 2 + (py - y) ** 2
                if d <= best_d:
                    best_d = d
                    best = Hit("vertex", i, j)
        return best

    def lit_fraction(self, index: Optional[int] = None) -> float:
        """``(quad ∩ illuminated) / quad`` for one vial -- the live coverage readout.

        This is exactly the ratio the pipeline will measure once the bundle is saved
        (`calibration.quad_lit_fraction`), so watching it go up IS the proof that an edit
        helped. NaN when the editor was opened without an illumination mask.
        """
        i = self.selected if index is None else int(index)
        if self.illum_mask is None:
            return float("nan")
        if i not in self._lit_cache:
            self._lit_cache[i] = quad_lit_fraction(self.quads[i], self.illum_mask)
        return self._lit_cache[i]

    def lit_fractions(self) -> List[float]:
        return [self.lit_fraction(i) for i in range(self.n_vials)]

    def to_face_calibration(self) -> FaceCalibration:
        """The edited face: quads written onto every vial, bboxes re-synced to them."""
        return apply_quads_to_face(self.face_cal, self.quads)


def _centroid(quad: Sequence[Sequence[float]]) -> Tuple[float, float]:
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _point_in_quad(quad: Sequence[Sequence[float]], x: float, y: float) -> bool:
    contour = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


# ==========================================================================================
# Mouse semantics (pure -- takes IMAGE coordinates, never touches a window)
# ==========================================================================================
class DragController:
    """Turns press/move/release/wheel into edits, in image coordinates.

    Holds the only mutable interaction state there is (what is being dragged, and where the
    previous pan sample was), so the cv2 callback in `run_roi_editor` is a pure forwarder.
    """

    def __init__(self, state: EditorState, view: ViewTransform) -> None:
        self.state = state
        self.view = view
        self.mode: Optional[str] = None       # "vertex" | "roi" | "pan"
        self.drag_vial: Optional[int] = None
        self.drag_vertex: Optional[int] = None
        self.hover: Optional[Hit] = None
        self.cursor_image: Tuple[float, float] = (0.0, 0.0)
        self._last: Tuple[float, float] = (0.0, 0.0)
        self._grab_offset: Tuple[float, float] = (0.0, 0.0)

    @property
    def dragging_vertex(self) -> bool:
        return self.mode == "vertex"

    def on_press(self, ix: float, iy: float) -> Optional[Hit]:
        """Left button down at image position (ix, iy). Starts a vertex or ROI drag."""
        self.cursor_image = (ix, iy)
        hit = self.state.hit_test(ix, iy)
        self._last = (ix, iy)
        if hit is None:
            self.mode = None
            return None
        self.state.select(hit.vial, hit.vertex)
        self.state.push_undo()                     # one undo entry per gesture
        self.drag_vial = hit.vial
        if hit.kind == "vertex":
            self.mode = "vertex"
            self.drag_vertex = hit.vertex
            px, py = self.state.quads[hit.vial][hit.vertex]
            self._grab_offset = (px - ix, py - iy)  # no jump when the grab is slightly off-centre
        else:
            self.mode = "roi"
            self.drag_vertex = None
        return hit

    def on_move(self, ix: float, iy: float) -> None:
        """Cursor moved to image position (ix, iy) -- drags, or just updates the hover highlight."""
        self.cursor_image = (ix, iy)
        if self.mode == "vertex" and self.drag_vial is not None and self.drag_vertex is not None:
            self.state.set_vertex(self.drag_vial, self.drag_vertex,
                                  ix + self._grab_offset[0], iy + self._grab_offset[1])
        elif self.mode == "roi" and self.drag_vial is not None:
            self.state.move_roi(self.drag_vial, ix - self._last[0], iy - self._last[1], push=False)
        else:
            self.hover = self.state.hit_test(ix, iy)
        self._last = (ix, iy)

    def on_release(self) -> None:
        self.mode = None
        self.drag_vial = None
        self.drag_vertex = None

    def on_pan_start(self, sx: float, sy: float) -> None:
        self.mode = "pan"
        self._last = (sx, sy)

    def on_pan_move(self, sx: float, sy: float) -> None:
        """Pan sample in SCREEN coordinates (panning is a view operation, not an edit)."""
        if self.mode != "pan":
            return
        self.view.pan_by(sx - self._last[0], sy - self._last[1])
        self._last = (sx, sy)

    def on_wheel(self, delta: float, sx: float, sy: float) -> None:
        self.view.zoom_by(1.25 if delta > 0 else 1.0 / 1.25, (sx, sy))


# ==========================================================================================
# Keyboard (pure)
# ==========================================================================================
#: Raw `cv2.waitKeyEx` codes for the arrow keys, which are NOT ascii and differ per platform.
_ARROWS = {
    2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",   # Windows
    65361: "left", 65362: "up", 65363: "right", 65364: "down",           # GTK / Qt (Linux)
    63234: "left", 63232: "up", 63235: "right", 63233: "down",           # macOS
    81: "left", 82: "up", 83: "right", 84: "down",                       # some Qt builds
}


def decode_key(code: int) -> Optional[str]:
    """Map a raw `cv2.waitKeyEx` code to a key NAME (or None for "nothing pressed").

    Isolated and tested because the arrow keys are the one input whose encoding varies by
    platform, and a silent mismatch there would break the 1-px nudge -- the whole point of
    keyboard editing -- with no visible error.
    """
    if code is None or code < 0:
        return None
    if code in _ARROWS:
        return _ARROWS[code]
    low = code & 0xFF
    if low in _ARROWS and code > 255:
        return _ARROWS[low]
    if low == 9:
        return "tab"
    if low == 27:
        return "esc"
    if low in (13, 10):
        return "enter"
    if low == 26:
        return "ctrl+z"
    if low == 25:
        return "ctrl+y"
    if 32 <= low < 127:
        return chr(low)
    return None


_NUDGE = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}


def handle_key(state: EditorState, view: ViewTransform, key: Optional[str],
               view_size: Tuple[int, int] = DEFAULT_VIEW_SIZE) -> Optional[str]:
    """Apply one keystroke to `state`/`view`. Returns a command for the driver, or None.

    Commands: ``"save"``, ``"quit"``, ``"toggle_help"``, ``"toggle_magnifier"``,
    ``"toggle_mask"``. Everything else is handled here, which is what keeps the cv2 loop free of
    editing logic.
    """
    if key is None:
        return None
    if key in _NUDGE:
        dx, dy = _NUDGE[key]
        if state.selected_vertex is None:
            state.move_roi(state.selected, dx, dy)
        else:
            state.move_vertex(state.selected, state.selected_vertex, dx, dy)
        return None
    if key in ("tab", "n"):
        state.next_vial()
    elif key == "p":
        state.prev_vial()
    elif key == "v":
        state.cycle_vertex()
    elif key == "c":
        state.copy_shape_to_all()
    elif key == "r":
        state.reset_vial(state.selected)
    elif key == "R":
        state.reset_all()
    elif key in ("z", "ctrl+z"):
        state.undo()
    elif key in ("y", "ctrl+y"):
        state.redo()
    elif key in ("+", "="):
        view.zoom_by(1.25, (view_size[0] / 2.0, view_size[1] / 2.0))
    elif key in ("-", "_"):
        view.zoom_by(1.0 / 1.25, (view_size[0] / 2.0, view_size[1] / 2.0))
    elif key == "0":
        fit = ViewTransform.fit(state.image_size or (view_size[0], view_size[1]), view_size)
        view.zoom, view.ox, view.oy = fit.zoom, fit.ox, fit.oy
    elif key == "s":
        return "save"
    elif key in ("q", "esc"):
        return "quit"
    elif key == "h":
        return "toggle_help"
    elif key == "m":
        return "toggle_magnifier"
    elif key == "l":
        return "toggle_mask"
    return None


# ==========================================================================================
# Rendering (draws into ndarrays; never opens a window, so it is headless-safe)
# ==========================================================================================
_COL_IDLE = (90, 170, 90)
_COL_SEL = (60, 230, 255)
_COL_HOVER = (255, 200, 80)
_COL_VERTEX = (40, 40, 245)
_COL_TEXT = (245, 245, 245)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_HELP_LINES = [
    "FlyGym quad-ROI editor",
    "",
    "mouse   drag a VERTEX (grab within a few px) to reshape",
    "        drag INSIDE an ROI to move the whole thing",
    "        right/middle drag = pan     wheel = zoom at cursor",
    "keys    Tab / n / p   next / previous vial",
    "        v             cycle vertex (TL/TR/BR/BL/whole ROI)",
    "        arrows        nudge selected vertex (or whole ROI) 1 px",
    "        c             copy this SHAPE to all vials",
    "        r / R         reset this vial / all vials",
    "        z / y         undo / redo",
    "        + / - / 0     zoom in / out / fit",
    "        m             pin magnifier      l  toggle mask tint",
    "        h             this help",
    "        s             SAVE and exit",
    "        q / Esc       quit WITHOUT saving (press twice if you have edits)",
    "",
    "'lit' = quad n illuminated / quad. Higher is more measured vial.",
]


def tinted_frame(frame_gray: np.ndarray, illum_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Grayscale frame as BGR with the trackable (lit) pixels tinted green.

    The tint is the whole reason the operator can place a vertex meaningfully: it shows where
    the illuminated region the pipeline will actually measure ends.
    """
    vis = cv2.cvtColor(np.ascontiguousarray(frame_gray), cv2.COLOR_GRAY2BGR)
    if illum_mask is not None:
        tint = np.zeros_like(vis)
        tint[illum_mask > 0] = (0, 55, 0)
        vis = cv2.add(vis, tint)
    return vis


def render_view(
    base_bgr: np.ndarray,
    state: EditorState,
    view: ViewTransform,
    view_size: Tuple[int, int] = DEFAULT_VIEW_SIZE,
    hover: Optional[Hit] = None,
    cursor_image: Optional[Tuple[float, float]] = None,
    show_help: bool = False,
    show_magnifier: bool = False,
    status: str = "",
) -> np.ndarray:
    """Compose one editor frame and RETURN it (the driver is what calls `imshow`)."""
    vw, vh = int(view_size[0]), int(view_size[1])
    canvas = cv2.warpAffine(base_bgr, view.matrix(), (vw, vh), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(25, 25, 25))

    for i in range(state.n_vials):
        _draw_quad(canvas, state, view, i, hover)
    _draw_hud(canvas, state, status)
    if show_magnifier and cursor_image is not None:
        _draw_magnifier(canvas, base_bgr, state, cursor_image)
    if show_help:
        _draw_help(canvas)
    return canvas


def _screen_pts(quad: Sequence[Sequence[float]], view: ViewTransform) -> np.ndarray:
    return np.asarray([view.to_screen(p) for p in quad], dtype=np.int32)


def _draw_quad(canvas: np.ndarray, state: EditorState, view: ViewTransform, i: int,
               hover: Optional[Hit]) -> None:
    selected = i == state.selected
    hovered = hover is not None and hover.vial == i
    pts = _screen_pts(state.quads[i], view)
    color = _COL_SEL if selected else (_COL_HOVER if hovered else _COL_IDLE)
    thickness = 2 if selected else 1

    if selected:  # translucent fill so the selection is unmistakable at a glance
        fill = canvas.copy()
        cv2.fillPoly(fill, [pts.reshape(-1, 1, 2)], (70, 90, 40))
        cv2.addWeighted(canvas, 0.82, fill, 0.18, 0, dst=canvas)
    cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)

    if selected or hovered:
        for j, (sx, sy) in enumerate(pts):
            grabbed = hovered and hover is not None and hover.kind == "vertex" and hover.vertex == j
            key_sel = selected and state.selected_vertex == j
            r = 7 if (grabbed or key_sel) else 5
            cv2.circle(canvas, (int(sx), int(sy)), r,
                       _COL_VERTEX if (grabbed or key_sel) else color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(sx), int(sy)), r, (20, 20, 20), 1, cv2.LINE_AA)

    vial = state.face_cal.vials[i]
    lit = state.lit_fraction(i)
    label = "%d" % vial.id if vial.present else "%d X" % vial.id
    if lit == lit:  # not NaN
        label += "  %.2f" % lit
    cx, cy = view.to_screen(_centroid(state.quads[i]))
    _text(canvas, label, (int(cx) - 26, int(cy)), color, 0.5, 1)


def _draw_hud(canvas: np.ndarray, state: EditorState, status: str) -> None:
    vh, vw = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (vw, 34), (28, 28, 28), -1)
    cv2.rectangle(canvas, (0, vh - 26), (vw, vh), (28, 28, 28), -1)

    vial = state.face_cal.vials[state.selected] if state.n_vials else None
    lit = state.lit_fraction()
    known = [f for f in state.lit_fractions() if f == f]     # drop NaN (no illum mask)
    mean = float(np.mean(known)) if known else float("nan")
    vsel = "whole ROI" if state.selected_vertex is None else _VERT_NAMES[state.selected_vertex]
    head = "face %s | vial %d/%d (id %s) | %s | lit %s | face mean %s%s" % (
        state.face_cal.name, state.selected + 1, state.n_vials,
        getattr(vial, "id", "-"), vsel,
        "--" if lit != lit else "%.3f" % lit,
        "--" if mean != mean else "%.3f" % mean,
        "  *UNSAVED*" if state.dirty else "",
    )
    _text(canvas, head, (10, 23), _COL_TEXT, 0.6, 1)
    _text(canvas, status or "h help | Tab next | c copy shape | z undo | s save | q quit",
          (10, vh - 8), (170, 170, 170), 0.5, 1)


def _draw_magnifier(canvas: np.ndarray, base_bgr: np.ndarray, state: EditorState,
                    cursor_image: Tuple[float, float]) -> None:
    """Zoomed inset of the image under the cursor -- the precision-placement aid.

    Nearest-neighbour so single pixels stay square and countable, drawn from the SAME tinted
    source as the main view so the lit/unlit boundary the operator is aiming at is visible, with
    the selected quad's edges overlaid in magnifier coordinates and a crosshair on the cursor.
    """
    H, W = base_bgr.shape[:2]
    vh, vw = canvas.shape[:2]
    cx, cy = int(round(cursor_image[0])), int(round(cursor_image[1]))
    x0, y0 = cx - MAG_SRC_HALF, cy - MAG_SRC_HALF
    src = np.full((2 * MAG_SRC_HALF, 2 * MAG_SRC_HALF, 3), 25, dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + 2 * MAG_SRC_HALF), min(H, y0 + 2 * MAG_SRC_HALF)
    if sx1 > sx0 and sy1 > sy0:
        src[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = base_bgr[sy0:sy1, sx0:sx1]

    panel = cv2.resize(src, (MAG_PANEL, MAG_PANEL), interpolation=cv2.INTER_NEAREST)
    scale = MAG_PANEL / float(2 * MAG_SRC_HALF)
    quad = state.quads[state.selected]
    pts = np.asarray([[(px - x0) * scale, (py - y0) * scale] for px, py in quad], dtype=np.int32)
    cv2.polylines(panel, [pts.reshape(-1, 1, 2)], True, _COL_SEL, 2, cv2.LINE_AA)
    for j, (px, py) in enumerate(pts):
        cv2.circle(panel, (int(px), int(py)), 6,
                   _COL_VERTEX if state.selected_vertex == j else _COL_SEL, -1, cv2.LINE_AA)
    mid = MAG_PANEL // 2
    cv2.line(panel, (mid, mid - 12), (mid, mid + 12), (255, 255, 255), 1)
    cv2.line(panel, (mid - 12, mid), (mid + 12, mid), (255, 255, 255), 1)
    cv2.rectangle(panel, (0, 0), (MAG_PANEL - 1, MAG_PANEL - 1), _COL_SEL, 2)
    _text(panel, "%d,%d  x%.0f" % (cx, cy, scale), (6, MAG_PANEL - 8), (230, 230, 230), 0.45, 1)

    # Park it in the bottom corner AWAY from the cursor so it never covers what is being edited.
    width = state.image_size[0] if state.image_size else max(1, W)
    px = 10 if cursor_image[0] > width / 2.0 else vw - MAG_PANEL - 10
    py = vh - MAG_PANEL - 36
    px = min(max(px, 0), max(0, vw - MAG_PANEL))
    py = min(max(py, 0), max(0, vh - MAG_PANEL))
    canvas[py:py + MAG_PANEL, px:px + MAG_PANEL] = panel


def _draw_help(canvas: np.ndarray) -> None:
    pad, lh = 18, 24
    w = 640
    h = pad * 2 + lh * len(_HELP_LINES)
    x0 = max(0, (canvas.shape[1] - w) // 2)
    y0 = max(0, (canvas.shape[0] - h) // 2)
    box = canvas[y0:y0 + h, x0:x0 + w]
    if box.size == 0:
        return
    cv2.addWeighted(box, 0.15, np.full_like(box, 20), 0.85, 0, dst=box)
    cv2.rectangle(canvas, (x0, y0), (x0 + w - 1, y0 + h - 1), _COL_SEL, 1)
    for i, line in enumerate(_HELP_LINES):
        _text(canvas, line, (x0 + pad, y0 + pad + lh * (i + 1) - 8),
              _COL_SEL if i == 0 else _COL_TEXT, 0.52, 1)


def _text(img: np.ndarray, s: str, org: Tuple[int, int], color, scale: float, th: int) -> None:
    """Text with a dark halo so it stays readable over both the bright tubes and the dark rig."""
    cv2.putText(img, s, org, _FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, _FONT, scale, color, th, cv2.LINE_AA)


# ==========================================================================================
# The driver -- window + event pump ONLY
# ==========================================================================================
def run_roi_editor(
    frame_gray: np.ndarray,
    face_cal: FaceCalibration,
    illum_mask: Optional[np.ndarray] = None,
    window: str = "FlyGym ROI editor",
    view_size: Tuple[int, int] = DEFAULT_VIEW_SIZE,
    grab_radius: int = DEFAULT_GRAB_RADIUS,
    start_vial: int = 0,
    on_close: Optional[Callable[[], None]] = None,
) -> Optional[FaceCalibration]:
    """Open the editor on one face. Returns the edited `FaceCalibration`, or None if cancelled.

    Args:
        frame_gray: HxW grayscale still of this face (the calibration frame).
        face_cal: the face to edit; not mutated.
        illum_mask: full-frame illuminated mask (255 = trackable). Enables the live lit-fraction
            readout and the green tint that makes vertex placement meaningful.
        window: OpenCV window title.
        view_size: window size in px; the frame is fitted into it initially.
        grab_radius: vertex pick-up radius, image px.
        start_vial: index of the vial selected on open.
        on_close: optional callback fired just before the window is destroyed.

    Returns:
        The edited face on ``s`` (save), or ``None`` on ``q``/Esc/window-closed.
    """
    gray = np.ascontiguousarray(frame_gray)
    H, W = gray.shape[:2]
    state = EditorState(face_cal, illum_mask, grab_radius=grab_radius, image_size=(W, H))
    state.select(start_vial)
    view = ViewTransform.fit((W, H), view_size)
    drag = DragController(state, view)

    with_tint = tinted_frame(gray, illum_mask)
    plain = tinted_frame(gray, None)
    flags = {"help": False, "magnifier": False, "mask": illum_mask is not None}

    def on_mouse(event, sx, sy, mouse_flags, _param):
        ix, iy = view.to_image((sx, sy))
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.on_press(ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drag.mode == "pan":
                drag.on_pan_move(sx, sy)
            else:
                drag.on_move(ix, iy)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            drag.on_release()
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            drag.on_pan_start(sx, sy)
        elif event == cv2.EVENT_MOUSEWHEEL:
            drag.on_wheel(cv2.getMouseWheelDelta(mouse_flags), sx, sy)
            view.clamp((W, H), view_size)

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    print(_startup_banner(state))

    result: Optional[FaceCalibration] = None
    confirm_quit = False
    try:
        while True:
            base = with_tint if flags["mask"] else plain
            canvas = render_view(
                base, state, view, view_size,
                hover=drag.hover, cursor_image=drag.cursor_image,
                show_help=flags["help"],
                show_magnifier=flags["magnifier"] or drag.dragging_vertex,
                status=(QUIT_WARNING if confirm_quit else ""),
            )
            cv2.imshow(window, canvas)
            key = decode_key(cv2.waitKeyEx(16))
            command = handle_key(state, view, key, view_size)
            if command == "save":
                result = state.to_face_calibration()
                break
            if command == "quit":
                # Never throw away hand-placed vertices on one keystroke: the first `q` after an
                # edit only arms the quit, and any other keypress disarms it again.
                if state.dirty and not confirm_quit:
                    confirm_quit = True
                    continue
                break
            if key is not None:
                # Only a real keypress disarms -- `waitKeyEx` returns -1 on every idle frame, and
                # disarming on those would spin the warning away before it could be read.
                confirm_quit = False
            if command == "toggle_help":
                flags["help"] = not flags["help"]
            elif command == "toggle_magnifier":
                flags["magnifier"] = not flags["magnifier"]
            elif command == "toggle_mask":
                flags["mask"] = not flags["mask"]
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break  # user closed the window == cancel
    finally:
        if on_close is not None:
            on_close()
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    return result


def _startup_banner(state: EditorState) -> str:
    lits = [f for f in state.lit_fractions() if f == f]
    head = "\n".join(_HELP_LINES)
    if not lits:
        return head + "\n"
    worst = int(np.argmin(state.lit_fractions()))
    return head + ("\n\nface %s: %d vials, lit fraction mean %.3f, worst vial id %s at %.3f\n"
                   % (state.face_cal.name, state.n_vials, float(np.mean(lits)),
                      state.face_cal.vials[worst].id, min(lits)))

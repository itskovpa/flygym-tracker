"""Calibration for a FlyGym v2 drum face.

Produces the `Calibration` bundle described in DESIGN.md §5.4/§5.5:
  * an illuminated mask (255 = trackable lit pixel, 0 = excluded), with the central
    LED/hardware band zeroed out,
  * a 2x8 vial lattice (16 `VialROI`, ids 1..16 row-major), and
  * per-slot tube-presence flags.

Three entry points, all emitting an IDENTICAL bundle:
  * `detect_calibration(frame, face)` -- auto-detect accelerator (DESIGN §5.4 B). It finds
    the lattice geometry and hands the boxes to `build_calibration_from_boxes`, which does
    all the bundle building.  Its boxes also serve as *seed boxes* for the manual wizard.
  * `build_calibration_from_boxes(frame, face, boxes, present_flags)` -- the pure,
    unit-testable core used by the manual ROI wizard (DESIGN §5.5): given vial boxes and
    optional present/absent flags it derives each box's illuminated sub-mask (bright pixels
    inside the box, minus the central band) and assembles the same bundle.
  * `build_calibration_from_markers(frame, face, marker_detector)` -- MARKER-DRIVEN path
    (see below), the preferred automatic path on a rig that carries the physical sticker
    band, plus `build_two_face_calibration(video, detector, out_dir)` which drives it over a
    whole flip video to calibrate BOTH drum faces at once.

The interactive wizard driver lives in `calibrate_wizard.py` and only wraps
`build_calibration_from_boxes`; no CV logic lives there.

Nothing here is hard-coded to the pixel coordinates of any one frame: geometry comes from
row/column intensity profiles and presence from illumination-relative thresholds, all of
which are exposed as `CalibParams` fields.

MARKER-DRIVEN CALIBRATION (`build_calibration_from_markers`, `build_two_face_calibration`)
-------------------------------------------------------------------------------------------
`detect_calibration` infers the 8 vial columns from a *brightness* profile. That works but it
is guessing at a lattice from illumination alone, and it is fragile exactly where the rig is
dimmest. `marker_band.MarkerBandDetector` reads the rig's physical IR-sticker band instead and
returns the vial column spans as a *measurement* (validated 8/8 against a hand calibration),
plus the face identity (validated 43/43 on real dwells). So when the rig carries the marker
band, the marker spans -- not a brightness guess -- are the right source of vial x-positions.

What this path takes from where:
  * **x** (8 vial columns)  <- `marker_detector.vial_boundaries(frame)`.
  * **y** (2 tube rows)     <- the two glowing tube row-bands found in the frame, selected as
    the bright row-runs immediately ABOVE and BELOW the marker band (the band physically sits
    between the two rows, so "nearest run on each side" identifies them unambiguously). This
    is what keeps the blindingly-lit stage along the bottom edge of the frame out of the
    lattice: it is a *further* run below the band, never the nearest one.
  * **ids**                 <- row-major: upper band = ids 1..8, lower band = 9..16, both
    left->right, matching the canonical numbering in DESIGN.md §2.

PRESENCE IS ALWAYS `True` ON THIS PATH -- AND THAT IS DELIBERATE
----------------------------------------------------------------
`detect_calibration` judges tube presence from brightness/rim cues and gets it WRONG on the
dim end columns: it flagged a vial that was in fact full of flies as empty. A false "absent"
is silent and unrecoverable -- `pipeline.TrackerPipeline._precompute_faces` skips non-present
vials outright, so that vial's flies are simply never measured and nothing in the output says
so. A false "present" costs, at worst, one boring row of near-zero activity that a human can
see and ignore.

The two error costs are therefore wildly asymmetric, so this path does not make the call at
all: **every slot is emitted with `present=True`**. Slots that *look* empty (very little of
their box is lit relative to the rest of the face) are recorded as SUSPICIOUS instead -- drawn
in orange with a "?" on the overlay, listed in the bundle's `notes`, and stored in
`FaceCalibration.marker["suspicious"]` (read it back with `suspicious_vials`). A human then
decides, and marks a slot absent by hand (edit `calibration.json`, or re-run the wizard).
"""
from __future__ import annotations

import collections.abc
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from flygym_tracker.types import Calibration, FaceCalibration, VialROI

# (x, y, w, h) pixel box in full-frame coords.
Box = Tuple[int, int, int, int]
#: (y0, y1) HALF-OPEN row range, as used for `gray[y0:y1]`.
RowBand = Tuple[int, int]
#: Vial columns per drum face (DESIGN.md §2: 16 slots = 2 rows x 8). `pipeline.TrackerPipeline`
#: hard-codes the resulting 16 slots per face in its global-vial-id arithmetic, so this is a
#: contract, not a tunable.
N_VIAL_COLUMNS = 8


# --------------------------------------------------------------------------------------
# Tunable parameters (defaults chosen from the real reference frame; see report).
# --------------------------------------------------------------------------------------
@dataclass
class CalibParams:
    # --- illuminated mask ---
    illum_method: str = "otsu"          # "otsu" | "percentile"
    illum_percentile: float = 55.0      # brightness percentile used when method == "percentile"
    morph_open: int = 3                 # opening kernel (px) -- remove specks; 0 disables
    morph_close: int = 7                # closing kernel (px) -- fill pinholes; 0 disables
    min_component_frac: float = 0.0004  # drop mask blobs smaller than frac * image area

    # --- horizontal (row) bands: two tube rows + central hardware band ---
    row_body_frac: float = 0.45         # a row is "bright" if its mean > frac * (robust bright level)
    row_min_run_frac: float = 0.10      # min tube-band height as a fraction of image height
    row_smooth: int = 9                 # smoothing window (px) on the row-mean profile

    # --- vertical columns within a band (4 + 4 split by the central gap) ---
    n_cols_per_group: int = 4
    col_lit_frac: float = 0.5           # a column is "lit" if its mean > frac * p90(col means)
    col_smooth: int = 9                 # smoothing window (px) on the column-mean profile
    gap_search_frac: Tuple[float, float] = (0.40, 0.60)  # search central gap in this x-fraction

    # --- vial body boxes ---
    box_w_frac: float = 0.86            # box width as a fraction of the column pitch
    box_inset_frac: float = 0.04        # inset each band top/bottom by frac of band height

    # --- tube presence (all thresholds relative to the face's own lit level) ---
    presence_dark_frac: float = 0.35    # body dimmer than frac*lit_ref  -> empty (dark/blanked slot)
    presence_bright_frac: float = 0.85  # body brighter than frac*lit_ref -> a lit column; check for a rim
    presence_rim_frac: float = 0.40     # in a lit column, mouth rim below frac*lit_ref -> no tube (empty)
    mouth_h_frac: float = 0.30          # mouth band height as a fraction of the vial band height
    mouth_gap_frac: float = 0.04        # gap between box and mouth band (frac of band height)


# --------------------------------------------------------------------------------------
# Low-level geometry / mask helpers
# --------------------------------------------------------------------------------------
def _as_gray(frame: np.ndarray) -> np.ndarray:
    """Return a contiguous 2-D uint8 view of `frame`."""
    if frame is None:
        raise ValueError("frame is None")
    img = np.asarray(frame)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(img)


def _smooth1d(profile: np.ndarray, win: int) -> np.ndarray:
    if win and win > 1:
        k = np.ones(int(win), dtype=np.float64) / float(win)
        return np.convolve(profile, k, mode="same")
    return profile


def _illuminated_mask(gray: np.ndarray, p: CalibParams) -> np.ndarray:
    """Threshold the bright back-lit region and clean it up. 255 = bright, 0 = dark."""
    if p.illum_method == "percentile":
        thr = float(np.percentile(gray, p.illum_percentile))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    else:  # otsu (default): robust to the exact backlight level
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if p.morph_open and p.morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_open, p.morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if p.morph_close and p.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_close, p.morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _bright_runs(bright: np.ndarray, min_run: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, b in enumerate(bright):
        if b and start is None:
            start = i
        elif not b and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(bright)))
    return [r for r in runs if (r[1] - r[0]) >= min_run]


def _row_bands(gray: np.ndarray, p: CalibParams
               ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Return (upper_band, lower_band, central_band) as (y0, y1) row ranges.

    The two tube rows are the two tall bright horizontal runs (filtered by height so the
    thin, sparse mouth rims and the fragmented central LED runs are ignored even though they
    are bright); the central hardware/LED band is everything between them. The bright level
    is taken as the median of the upper-half of the row-mean profile, so a saturated central
    band does not inflate the threshold.
    """
    H = gray.shape[0]
    prof = _smooth1d(gray.mean(axis=1), p.row_smooth)
    bright_level = float(np.median(prof[prof >= np.median(prof)]))
    thr = p.row_body_frac * bright_level
    runs = _bright_runs(prof > thr, int(p.row_min_run_frac * H))
    if len(runs) < 2:
        raise ValueError(
            "row-band detection found %d tube band(s); cannot auto-detect the lattice. "
            "Use the manual wizard (calibrate_wizard.run_wizard)." % len(runs)
        )
    runs.sort(key=lambda r: r[0])
    upper, lower = runs[0], runs[-1]
    central = (upper[1], lower[0])
    return upper, lower, central


def _columns(gray: np.ndarray, band: Tuple[int, int], p: CalibParams
             ) -> Tuple[List[float], float]:
    """Locate 2*n_cols_per_group column centers in a band via its column-mean profile.

    Strategy (robust to the rightmost column falling into darkness): find the left lit
    edge and the central vertical gap, giving the left group's extent -> pitch. Mirror the
    pitch across the gap to place the right group, so a dim/dark rightmost tube still gets a
    box by extrapolation instead of being dropped.
    """
    y0, y1 = band
    W = gray.shape[1]
    cp = _smooth1d(gray[y0:y1].mean(axis=0), p.col_smooth)
    thr = p.col_lit_frac * float(np.percentile(cp, 90))
    lit = cp > thr
    xs = np.where(lit)[0]
    if xs.size == 0:
        raise ValueError("no lit columns found in band %r" % (band,))
    left_edge = int(xs.min())

    g0 = int(p.gap_search_frac[0] * W)
    g1 = int(p.gap_search_frac[1] * W)
    gap_center = g0 + int(np.argmin(cp[g0:g1]))

    lg = gap_center
    while lg > left_edge and cp[lg] < thr:
        lg -= 1
    left_gap = lg + 1
    rg = gap_center
    while rg < W - 1 and cp[rg] < thr:
        rg += 1
    right_gap = rg - 1

    n = p.n_cols_per_group
    pitch = (left_gap - left_edge) / float(n)
    if pitch <= 1:
        raise ValueError("degenerate column pitch (%.1f) in band %r" % (pitch, band))
    left = [left_edge + pitch * (i + 0.5) for i in range(n)]
    right = [right_gap + pitch * (i + 0.5) for i in range(n)]
    return left + right, pitch


def _make_boxes(upper: Tuple[int, int], lower: Tuple[int, int],
                ucols: Sequence[float], up_pitch: float,
                lcols: Sequence[float], lo_pitch: float,
                p: CalibParams, W: int) -> List[Box]:
    """Build 16 body boxes (row-major: upper row first, then lower)."""
    boxes: List[Box] = []
    for band, cols, pitch in ((upper, ucols, up_pitch), (lower, lcols, lo_pitch)):
        y0, y1 = band
        inset = int(round(p.box_inset_frac * (y1 - y0)))
        by = y0 + inset
        bh = (y1 - y0) - 2 * inset
        bw = int(round(pitch * p.box_w_frac))
        for c in cols:
            bx = int(round(c - bw / 2.0))
            bx = max(0, min(bx, W - 1))
            w = min(bw, W - bx)
            boxes.append((bx, by, w, bh))
    return boxes


# --------------------------------------------------------------------------------------
# Illuminated sub-mask + presence, per box
# --------------------------------------------------------------------------------------
def _box_slice(gray: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    H, W = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return gray[y0:y1, x0:x1]


def _mouth_band(gray: np.ndarray, box: Box, row: int, band_h: int, p: CalibParams) -> np.ndarray:
    """Grab the tube-mouth strip just outside the body box on its outer edge.

    Upper row (row 0): mouth is above the box; lower row (row 1): below it.
    """
    x, y, w, h = box
    H, W = gray.shape[:2]
    mh = max(1, int(round(p.mouth_h_frac * band_h)))
    gap = int(round(p.mouth_gap_frac * band_h))
    if row == 0:
        my1 = max(0, y - gap)
        my0 = max(0, my1 - mh)
    else:
        my0 = min(H, y + h + gap)
        my1 = min(H, my0 + mh)
    x0, x1 = max(0, x), min(W, x + w)
    return gray[my0:my1, x0:x1]


def _presence(gray: np.ndarray, box: Box, row: int, lit_ref: float,
              band_h: int, p: CalibParams) -> bool:
    """Decide present vs empty for one slot (True = tube present).

    A tube body always glows (the diffuser shows through an empty slot too), so body
    brightness alone cannot separate a missing tube from a present one. The reliable cue is
    the bright tube *mouth rim* at the outer edge:
      * dark body                       -> empty (blanked/dark slot),
      * bright body but *no* mouth rim  -> empty (missing tube in a lit column),
      * otherwise                       -> present (incl. dim rightmost tubes, which still
                                           have a faint rim and a below-`bright` body).
    """
    body = float(np.median(_box_slice(gray, box)))
    if body < p.presence_dark_frac * lit_ref:
        return False  # dark / blanked slot
    if body > p.presence_bright_frac * lit_ref:
        mouth = _mouth_band(gray, box, row, band_h, p)
        if mouth.size >= 16:
            rim = float(np.percentile(mouth, 90))
            if rim < p.presence_rim_frac * lit_ref:
                return False  # lit column, no rim -> missing tube
    return True


def _row_of_box(box: Box, central_band: Optional[Tuple[int, int]], idx: int, nboxes: int) -> int:
    """0 = upper, 1 = lower. Prefer id order; fall back to position vs the central band."""
    if nboxes == 16:
        return 0 if idx < 8 else 1
    if central_band is not None:
        cy = box[1] + box[3] / 2.0
        return 1 if cy > (central_band[0] + central_band[1]) / 2.0 else 0
    return 0


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def build_calibration_from_boxes(
    frame_gray: np.ndarray,
    face: str,
    boxes: Sequence[Box],
    present_flags: Optional[Sequence[Optional[bool]]] = None,
    params: Optional[CalibParams] = None,
    central_band: Optional[Tuple[int, int]] = None,
    illum_mask_path: Optional[str] = None,
    notes: str = "",
) -> Tuple[Calibration, np.ndarray, np.ndarray]:
    """Assemble the calibration bundle from a list of vial body boxes (PURE / testable).

    This is the shared core for BOTH the auto-detect path and the manual ROI wizard, so the
    two always emit an identical `Calibration` bundle.

    Args:
        frame_gray: HxW grayscale face frame.
        face: face name ("A"/"B").
        boxes: vial body boxes (x, y, w, h), row-major (upper row then lower row).
        present_flags: optional per-box override. Entry True/False forces present/absent;
            None (or a shorter/omitted list) means "derive it" from the frame.
        params: `CalibParams` (defaults if None).
        central_band: (y0, y1) rows of the excluded central hardware band. Auto-detected
            (best effort) if None.
        illum_mask_path: stored path for the mask PNG (default "illum_mask_<face>.png").
        notes: free-text stored in the bundle.

    Returns:
        (calibration, illum_mask_uint8, overlay_bgr)
    """
    p = params or CalibParams()
    gray = _as_gray(frame_gray)
    H, W = gray.shape[:2]
    boxes = [tuple(int(v) for v in b) for b in boxes]
    if illum_mask_path is None:
        illum_mask_path = "illum_mask_%s.png" % face

    if central_band is None:
        try:
            _, _, central_band = _row_bands(gray, p)
        except ValueError:
            central_band = None

    # Illuminated mask: bright pixels, central band zeroed, fixtures outside the lattice clipped.
    mask = _illuminated_mask(gray, p)
    if central_band is not None:
        cy0 = max(0, min(H, central_band[0]))
        cy1 = max(0, min(H, central_band[1]))
        mask[cy0:cy1, :] = 0
    if boxes:
        xs0 = min(b[0] for b in boxes)
        xs1 = max(b[0] + b[2] for b in boxes)
        pitch = max((b[2] for b in boxes), default=0)
        m = int(0.4 * pitch)
        mask[:, :max(0, xs0 - m)] = 0
        mask[:, min(W, xs1 + m):] = 0
    mask = _drop_small_components(mask, int(p.min_component_frac * H * W))

    # lit reference = median body level across boxes (the face's own lit level).
    body_meds = [float(np.median(_box_slice(gray, b))) for b in boxes]
    lit_ref = float(np.median(body_meds)) if body_meds else 0.0

    flags = list(present_flags) if present_flags is not None else []
    vials: List[VialROI] = []
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        row = _row_of_box(box, central_band, idx, len(boxes))
        col = idx % 8 if len(boxes) == 16 else idx
        given = flags[idx] if idx < len(flags) else None
        if given is None:
            present = _presence(gray, box, row, lit_ref, h, p)
        else:
            present = bool(given)
        vials.append(VialROI(id=idx + 1, row=row, col=col, x=x, y=y, w=w, h=h, present=present))

    face_cal = FaceCalibration(name=face, vials=vials, illum_mask_path=illum_mask_path, marker=None)
    calib = Calibration(
        image_width=W, image_height=H,
        faces={face: face_cal},
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
    overlay = _draw_overlay(gray, vials, mask, central_band)
    return calib, mask, overlay


def detect_calibration(
    frame_gray: np.ndarray,
    face: str = "A",
    params: Optional[CalibParams] = None,
) -> Tuple[Calibration, np.ndarray, np.ndarray]:
    """Auto-detect the lattice + illuminated mask + presence on a still (DESIGN §5.4 B).

    Best-effort accelerator: the boxes it returns also seed the manual wizard
    (`calibrate_wizard.run_wizard`). Raises `ValueError` if the lattice can't be found (the
    caller should then fall back to the manual wizard).

    Returns (calibration, illum_mask_uint8, overlay_bgr).
    """
    p = params or CalibParams()
    gray = _as_gray(frame_gray)
    W = gray.shape[1]

    upper, lower, central = _row_bands(gray, p)
    ucols, up_pitch = _columns(gray, upper, p)
    lcols, lo_pitch = _columns(gray, lower, p)
    boxes = _make_boxes(upper, lower, ucols, up_pitch, lcols, lo_pitch, p, W)

    # Presence derived inside build_calibration_from_boxes (present_flags=None) so both
    # paths share one implementation.
    return build_calibration_from_boxes(
        gray, face, boxes, present_flags=None, params=p, central_band=central,
        notes="auto-detected lattice (seed for manual wizard)",
    )


def detect_seed_boxes(
    frame_gray: np.ndarray,
    face: str = "A",
    params: Optional[CalibParams] = None,
) -> Tuple[List[Box], List[bool]]:
    """Convenience for the wizard: auto-detected (boxes, present_flags) to pre-fill slots."""
    calib, _, _ = detect_calibration(frame_gray, face, params)
    return boxes_from_calibration(calib, face)


def boxes_from_calibration(calib: Calibration, face: str) -> Tuple[List[Box], List[bool]]:
    """Extract (boxes, present_flags) from an existing bundle (e.g. to re-seed the wizard)."""
    fc = calib.faces[face]
    boxes = [(v.x, v.y, v.w, v.h) for v in fc.vials]
    present = [bool(v.present) for v in fc.vials]
    return boxes, present


# ======================================================================================
# QUAD (4-vertex) VIAL ROIs
# ======================================================================================
# The drum is a CYLINDER. Tubes near the left/right edge of the frame curve away from the
# camera and are foreshortened, so an axis-aligned rectangle either spills onto the dark
# surround or clips the tube -- measured on the real 2-face bundle, face A's edge vials 1 and
# 8 have lit fractions of only 0.28 and 0.50 against ~0.95 for the central ones. A 4-corner
# quad can follow that taper; a rectangle cannot.
#
# Everything below is PURE geometry over `VialROI.quad` (``[[x, y]] * 4``, clockwise from the
# top-left) so it is testable headlessly. `roi_editor` drives it interactively, `pipeline`
# consumes it (`quad_polygon_mask`), and `transfer_quads` moves an edited face's shapes onto
# the other face.
#
# INVARIANT: a vial's `x, y, w, h` is the bounding box of its quad. `sync_bbox_to_quad`
# restores it after any edit; the pipeline additionally re-derives the crop rectangle from the
# quad, so a hand-edited bundle whose bbox went stale still measures the full polygon rather
# than silently truncating it.

#: A 4-corner polygon: ``[[x, y], [x, y], [x, y], [x, y]]``, clockwise from the top-left.
Quad = List[List[int]]
#: Inclusive ``(x0, x1)`` column span, as `marker_band.MarkerBandDetector.vial_boundaries` returns.
Span = Tuple[int, int]


def quad_from_bbox(box: Box) -> Quad:
    """The rectangle `box` as a quad -- clockwise from the top-left: TL, TR, BR, BL.

    This is the default shape for a vial that has never been hand-edited, so a bundle with no
    quads at all behaves identically to one whose quads are all straight rectangles.
    """
    x, y, w, h = (int(v) for v in box)
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def bbox_from_quad(quad: Sequence[Sequence[float]]) -> Box:
    """The integer bounding box ``(x, y, w, h)`` of a quad (w/h always >= 1)."""
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    x0 = int(np.floor(pts[:, 0].min()))
    y0 = int(np.floor(pts[:, 1].min()))
    x1 = int(np.ceil(pts[:, 0].max()))
    y1 = int(np.ceil(pts[:, 1].max()))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def vial_quad(vial: VialROI) -> Quad:
    """This vial's EFFECTIVE 4-CORNER quad: its own if it has one, else its bbox as a rectangle.

    Use this anywhere a strictly 4-vertex shape is needed regardless of whether the bundle
    predates quads -- notably `roi_editor`, whose whole model is four draggable corners. It
    deliberately IGNORES `VialROI.polygon` (arbitrary vertex count); use `vial_shape` for
    anything that just needs the shape the pipeline will actually measure.
    """
    if vial.quad is not None:
        return [[int(px), int(py)] for px, py in vial.quad]
    return quad_from_bbox((vial.x, vial.y, vial.w, vial.h))


def vial_shape(vial: VialROI) -> Quad:
    """This vial's EFFECTIVE MEASURED shape, applying the `types.VialROI` precedence.

    ``polygon`` (N >= 3, hand-drawn on the live feed) > ``quad`` (4 corners) > bbox rectangle.
    This is the one place that precedence is spelled out for non-pipeline callers (overlays,
    reports), so a polygon bundle and a quad bundle can be drawn and scored by the same code.
    """
    polygon = getattr(vial, "polygon", None)
    if polygon is not None:
        return [[int(px), int(py)] for px, py in polygon]
    return vial_quad(vial)


def face_quads(face_cal: FaceCalibration) -> List[Quad]:
    """Effective quads for every vial of a face, in `face_cal.vials` order."""
    return [vial_quad(v) for v in face_cal.vials]


def sync_bbox_to_quad(vial: VialROI) -> VialROI:
    """Reset `vial`'s bbox to the bounding box of its quad, IN PLACE. No-op without a quad."""
    if vial.quad is not None:
        vial.x, vial.y, vial.w, vial.h = bbox_from_quad(vial.quad)
    return vial


def shift_quad(quad: Sequence[Sequence[float]], dx: float, dy: float) -> Quad:
    """Translate a quad by ``(dx, dy)``, rounding exactly as `registration.apply_shift` does.

    Matching that rounding is what keeps a registered quad aligned with its registered bbox
    (they must land on the same integer grid or the polygon would drift inside the crop).
    """
    ix, iy = int(round(float(dx))), int(round(float(dy)))
    return [[int(px) + ix, int(py) + iy] for px, py in quad]


def polygon_area(quad: Sequence[Sequence[float]]) -> float:
    """Shoelace area of a quad (always >= 0). Analytic; see `quad_polygon_mask` for the raster."""
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def quad_polygon_mask(quad: Sequence[Sequence[float]], bbox: Box) -> np.ndarray:
    """Filled polygon as a BBOX-LOCAL bool mask of shape ``(h, w)``.

    `bbox` is the crop rectangle the mask must line up with (normally the quad's own bounding
    box, already clipped to the frame). Vertices outside it are clipped by `cv2.fillPoly`, so a
    quad hanging off the frame edge simply contributes the part that is inside.
    """
    x, y, w, h = (int(v) for v in bbox)
    if w <= 0 or h <= 0:
        return np.zeros((0, 0), dtype=bool)
    pts = np.asarray(quad, dtype=np.float64).reshape(-1, 2) - np.array([x, y], dtype=np.float64)
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 255)
    return canvas > 0


def quad_lit_fraction(quad: Sequence[Sequence[float]], illum_mask: np.ndarray) -> float:
    """``(quad ∩ illuminated) / quad`` in PIXELS -- the editor's live coverage readout.

    Both terms are rasterised over the quad's frame-clipped bounding box, so this is exactly
    the ratio the pipeline will measure: the numerator is the vial's effective measurement mask
    (`pipeline.TrackerPipeline._bbox_submask`) and the denominator is the polygon's own pixel
    count. Returns 0.0 for a degenerate (zero-area or fully off-frame) quad.
    """
    H, W = illum_mask.shape[:2]
    x, y, w, h = bbox_from_quad(quad)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    cb = (x0, y0, x1 - x0, y1 - y0)
    poly = quad_polygon_mask(quad, cb)
    area = int(np.count_nonzero(poly))
    if area == 0:
        return 0.0
    lit = int(np.count_nonzero(poly & (illum_mask[y0:y1, x0:x1] == 255)))
    return lit / float(area)


def face_column_spans(face_cal: FaceCalibration) -> List[Span]:
    """Per-column inclusive ``(x0, x1)`` spans for a face, indexed by `VialROI.col`.

    Prefers the marker band's measured spans (``marker["vial_spans"]`` -- see
    `build_calibration_from_markers`, where the columns come from the physical sticker band
    rather than a brightness guess). Falls back to the union of each column's own vial bboxes,
    so a face built by any other path still has usable spans.
    """
    marker = face_cal.marker
    if isinstance(marker, dict) and marker.get("vial_spans"):
        return [(int(a), int(b)) for a, b in marker["vial_spans"]]
    by_col: Dict[int, List[VialROI]] = {}
    for v in face_cal.vials:
        by_col.setdefault(int(v.col), []).append(v)
    return [
        (min(v.x for v in by_col[c]), max(v.x + v.w - 1 for v in by_col[c]))
        for c in sorted(by_col)
    ]


def face_row_bands(face_cal: FaceCalibration) -> Dict[int, RowBand]:
    """Per-row half-open ``(y0, y1)`` tube bands for a face, keyed by `VialROI.row`.

    Prefers ``marker["row_bands"]``; falls back to the union of each row's own vial bboxes.
    """
    marker = face_cal.marker
    if isinstance(marker, dict) and isinstance(marker.get("row_bands"), dict):
        rb = marker["row_bands"]
        out = {}
        for row, key in ((0, "upper"), (1, "lower")):
            if rb.get(key):
                out[row] = (int(rb[key][0]), int(rb[key][1]))
        if out:
            return out
    bands: Dict[int, RowBand] = {}
    for v in face_cal.vials:
        r = int(v.row)
        y0, y1 = (v.y, v.y + v.h)
        if r in bands:
            y0 = min(y0, bands[r][0])
            y1 = max(y1, bands[r][1])
        bands[r] = (y0, y1)
    return bands


def transfer_quads(
    src_face_cal: FaceCalibration,
    dst_face_cal: FaceCalibration,
    src_spans: Optional[Sequence[Span]] = None,
    dst_spans: Optional[Sequence[Span]] = None,
    *,
    src_bands: Optional[Mapping[int, RowBand]] = None,
    dst_bands: Optional[Mapping[int, RowBand]] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> FaceCalibration:
    """Carry hand-edited quads from one drum face onto the other. Returns a NEW FaceCalibration.

    WHY THIS IS AN IDENTITY MAP AND NOT A MIRROR. The rig owner edits ONE face before an
    experiment and expects both to be covered ("the system is symmetric"). It is -- and,
    measured, it is symmetric in the *simplest* way: correlating face A's frame against face B's
    gives +0.60 under the IDENTITY transform, while a vertical flip and a 180 deg rotation both
    score strongly NEGATIVE. The drum presents its two faces in the SAME orientation, so vial 1
    is top-left on both and a shape transfers directly. Mirroring would be wrong.

    What is left over is the small pose difference between the two dwells (the drum does not
    stop at exactly the same angle, and the two faces are not machined identically): face A's
    column 0 spans x=57..209 while face B's spans x=34..195. So each quad is rescaled from its
    SOURCE column span onto the DESTINATION column span, and from the source row band onto the
    DESTINATION row band -- i.e. the shape is expressed in the source column's own normalised
    frame and stamped into the destination column's. The destination face keeps its own marker-
    derived geometry; only the SHAPE comes from the source.

    Args:
        src_face_cal: the face that was edited (quads are read from here).
        dst_face_cal: the face to write onto; untouched (a copy is returned).
        src_spans / dst_spans: inclusive per-column ``(x0, x1)`` spans. Default: each face's own
            (`face_column_spans`) -- normally the marker band's measured columns.
        src_bands / dst_bands: per-row half-open ``(y0, y1)`` tube bands. Default:
            `face_row_bands` of each face, i.e. the DESTINATION's own vertical extent is
            preserved.
        image_size: optional ``(width, height)`` to clip transferred quads to the frame.

    Returns:
        A new `FaceCalibration` for `dst_face_cal.name` with quads set and bboxes re-synced.
        Vials the source has no quad for (or no matching id) are copied through unchanged.
    """
    src_spans = list(src_spans) if src_spans is not None else face_column_spans(src_face_cal)
    dst_spans = list(dst_spans) if dst_spans is not None else face_column_spans(dst_face_cal)
    sb = dict(src_bands) if src_bands is not None else face_row_bands(src_face_cal)
    db = dict(dst_bands) if dst_bands is not None else face_row_bands(dst_face_cal)
    src_by_id = {int(v.id): v for v in src_face_cal.vials}

    vials: List[VialROI] = []
    n_transferred = 0
    for dv in dst_face_cal.vials:
        sv = src_by_id.get(int(dv.id))
        new = VialROI(id=dv.id, row=dv.row, col=dv.col, x=dv.x, y=dv.y, w=dv.w, h=dv.h,
                      present=dv.present, quad=dv.quad)
        if (sv is not None and sv.quad is not None
                and int(sv.col) < len(src_spans) and int(dv.col) < len(dst_spans)
                and int(sv.row) in sb and int(dv.row) in db):
            new.quad = _map_quad(sv.quad, src_spans[int(sv.col)], dst_spans[int(dv.col)],
                                 sb[int(sv.row)], db[int(dv.row)], image_size)
            sync_bbox_to_quad(new)
            n_transferred += 1
        vials.append(new)

    marker = dict(dst_face_cal.marker) if isinstance(dst_face_cal.marker, dict) else dst_face_cal.marker
    if isinstance(marker, dict):
        marker["quad_source"] = {"face": src_face_cal.name, "n_transferred": n_transferred,
                                 "orientation": "identity (faces present in the same orientation)"}
    return FaceCalibration(name=dst_face_cal.name, vials=vials,
                           illum_mask_path=dst_face_cal.illum_mask_path, marker=marker)


def _map_quad(quad: Sequence[Sequence[float]], src_span: Span, dst_span: Span,
              src_band: RowBand, dst_band: RowBand,
              image_size: Optional[Tuple[int, int]]) -> Quad:
    """Affine-map one quad from (src column span x src row band) into the destination's."""
    sx0, sx1 = int(src_span[0]), int(src_span[1])
    dx0, dx1 = int(dst_span[0]), int(dst_span[1])
    sy0, sy1 = int(src_band[0]), int(src_band[1])
    dy0, dy1 = int(dst_band[0]), int(dst_band[1])
    kx = (dx1 - dx0) / float(sx1 - sx0) if sx1 != sx0 else 1.0
    ky = (dy1 - dy0) / float(sy1 - sy0) if sy1 != sy0 else 1.0
    out: Quad = []
    for px, py in quad:
        nx = dx0 + (float(px) - sx0) * kx
        ny = dy0 + (float(py) - sy0) * ky
        if image_size is not None:
            nx = min(max(nx, 0.0), float(image_size[0]))
            ny = min(max(ny, 0.0), float(image_size[1]))
        out.append([int(round(nx)), int(round(ny))])
    return out


def apply_quads_to_face(face_cal: FaceCalibration, quads: Sequence[Sequence[Sequence[float]]],
                        ) -> FaceCalibration:
    """A copy of `face_cal` carrying `quads` (one per vial, same order) with bboxes re-synced."""
    if len(quads) != len(face_cal.vials):
        raise ValueError("expected %d quad(s) for face %r, got %d"
                         % (len(face_cal.vials), face_cal.name, len(quads)))
    vials: List[VialROI] = []
    for v, q in zip(face_cal.vials, quads):
        nv = VialROI(id=v.id, row=v.row, col=v.col, x=v.x, y=v.y, w=v.w, h=v.h,
                     present=v.present, quad=[[int(round(float(p[0]))), int(round(float(p[1])))]
                                              for p in q])
        sync_bbox_to_quad(nv)
        vials.append(nv)
    return FaceCalibration(name=face_cal.name, vials=vials,
                           illum_mask_path=face_cal.illum_mask_path, marker=face_cal.marker)


def draw_quad_overlay(gray: np.ndarray, face_cal: FaceCalibration,
                      illum_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Human-check overlay for a quad-edited face: polygons + per-vial lit fraction.

    Same idea as `_draw_marker_overlay` (green tint over trackable pixels, ids drawn on each
    slot) but the ROI is drawn as its POLYGON with its corners marked, and each label carries
    the coverage number the editor was showing -- so the saved overlay is a faithful record of
    what the operator approved.
    """
    vis = cv2.cvtColor(_as_gray(gray), cv2.COLOR_GRAY2BGR)
    if illum_mask is not None:
        tint = np.zeros_like(vis)
        tint[illum_mask > 0] = (0, 60, 0)
        vis = cv2.add(vis, tint)
    for v in face_cal.vials:
        # `vial_shape`, not `vial_quad`: a face drawn with `live_vial_selector` carries N-vertex
        # polygons, and an overlay that quietly drew their bounding boxes instead would be a
        # record of something the operator never approved.
        quad = vial_shape(v)
        edited = v.quad is not None or getattr(v, "polygon", None) is not None
        color = (0, 200, 0) if v.present else (0, 0, 255)
        pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, 2)
        for px, py in quad:
            cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 255) if edited else color, -1)
        label = str(v.id) if v.present else "%d X" % v.id
        if illum_mask is not None:
            label += "  lit %.2f" % quad_lit_fraction(quad, illum_mask)
        # Stagger alternate columns' labels: edge vials are narrow, so same-height labels on
        # neighbouring slots overprint each other and the overlay stops being checkable.
        dy = 24 + 28 * (int(v.col) % 2)
        cv2.putText(vis, label, (int(quad[0][0]) + 4, int(quad[0][1]) + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


# --------------------------------------------------------------------------------------
# HAND-DRAWN POLYGON CALIBRATION  (the `live_vial_selector` path)
# --------------------------------------------------------------------------------------
def polygon_centroid(polygon: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """Mean of a polygon's vertices -- good enough for labelling and row/col ordering."""
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def build_calibration_from_polygons(
    polygons: Sequence[Sequence[Sequence[float]]],
    face: str,
    frame_gray: np.ndarray,
    image_size: Tuple[int, int],
    illum_mask: Optional[np.ndarray] = None,
) -> Tuple[FaceCalibration, np.ndarray, np.ndarray]:
    """Turn hand-drawn polygons into a `FaceCalibration`. Returns ``(face_cal, mask, overlay)``.

    This is the whole calibration path for a session that starts with
    `live_vial_selector.select_vials_live`: the operator draws every vial, and THE POLYGON IS THE
    VIAL. Nothing here detects, fits, snaps or second-guesses the drawn shape -- vial `i` is
    exactly the points clicked for it, in the order they were clicked.

    * ids are sequential 1..N in DRAW ORDER (so "vial 3" is the third one drawn, which is what
      the operator watched being numbered on screen).
    * `polygon` carries the shape; `x, y, w, h` are its bounding box (`bbox_from_quad`), i.e. the
      pipeline's crop rectangle, kept in sync by construction.
    * every vial is ``present=True``. A slot the operator did not want measured is a slot they
      did not draw -- there is no auto-exclusion here, exactly as in the marker-band path.
    * `row`/`col` are LABELS ONLY, derived from position (centroid above the frame's mid-line =>
      row 0, else row 1; `col` = left-to-right rank within that row). Nothing measures with them;
      they exist so the activity table's row/col columns stay meaningful.

    Args:
        polygons: one ``[[x, y], ...]`` (>= 3 points) per vial, in draw order.
        face: face name ("A"/"B").
        frame_gray: the frame the polygons were drawn on (HxW grayscale) -- used for the derived
            illumination mask and the overlay.
        image_size: ``(width, height)`` of the full frame.
        illum_mask: full-frame lit mask (255 = trackable), used EXACTLY as given when supplied.
            When None it is derived: a simple Otsu lit mask of `frame_gray`, PLUS the interior of
            every drawn polygon. That second part matters -- the operator said this region is a
            vial, so a fly sitting in a tube (a dark blob) at selection time must not carve a
            permanent hole out of that vial's measurement mask. Outside the drawn vials the mask
            is still just "what is lit", which is all the rest of the pipeline wants from it.

    Raises:
        ValueError: if `polygons` is empty (a face with no vials measures nothing).
    """
    if not len(polygons):
        raise ValueError("no polygons were drawn - a face needs at least one vial")

    gray = _as_gray(frame_gray)
    width, height = int(image_size[0]), int(image_size[1])
    if illum_mask is None:
        mask = _illuminated_mask(gray, CalibParams())
        cv2.fillPoly(mask, [np.round(np.asarray(p, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
                            for p in polygons], 255)
    else:
        mask = illum_mask

    # Row/col are pure labels, so they are assigned from geometry AFTER ids are fixed by draw
    # order: the operator may well draw the bottom row first.
    centroids = [polygon_centroid(p) for p in polygons]
    rows = [0 if cy < height / 2.0 else 1 for _cx, cy in centroids]
    cols = [0] * len(polygons)
    for row in (0, 1):
        members = [i for i, r in enumerate(rows) if r == row]
        for col, i in enumerate(sorted(members, key=lambda j: centroids[j][0])):
            cols[i] = col

    vials = []
    for i, poly in enumerate(polygons):
        x, y, w, h = bbox_from_quad(poly)
        vials.append(VialROI(id=i + 1, row=rows[i], col=cols[i], x=x, y=y, w=w, h=h,
                             present=True, polygon=[[int(px), int(py)] for px, py in poly]))

    face_cal = FaceCalibration(
        name=face,
        vials=vials,
        illum_mask_path="illum_mask_%s.png" % face,
        marker={"source": "live_vial_selector", "n_vials": len(vials),
                "image_size": [width, height]},
    )
    overlay = draw_quad_overlay(gray, face_cal, mask)
    return face_cal, mask, overlay


#: Vials per drum face. `pipeline.TrackerPipeline` computes global vial ids as
#: ``face_index * 16 + local_id``, so more than this per face would alias face B's vial 1 onto
#: face A's vial 17. It is a contract, not a tunable (DESIGN.md section 2).
VIALS_PER_FACE = 16


def build_two_face_calibration_from_polygons(
    polygons: Sequence[Sequence[Sequence[float]]],
    frame_gray: np.ndarray,
    image_size: Tuple[int, int],
    faces: Sequence[str] = ("A", "B"),
    illum_mask: Optional[np.ndarray] = None,
) -> Tuple[Calibration, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """One hand-drawn set of polygons -> a 32-vial, TWO-FACE bundle with IDENTICAL coordinates.

    The rig owner's constraint, verbatim: *"I will draw 16 vials on 1 face of the machine, and
    they will be exactly the same as the positions of the vials on the other side."* The drum's
    two faces present in the SAME orientation (measured: identity correlates +0.60, while a
    vertical flip and a 180 degree rotation are both strongly negative), so face B gets face A's
    polygons COPIED VERBATIM -- no mirroring, no rescaling, no snapping to marker spans, no
    `transfer_quads`. The operator draws once.

    Local ids run 1..16 on each face, so the pipeline's ``face_index * 16 + local_id`` yields
    global ids 1..16 (face A) and 17..32 (face B).

    Returns:
        ``(calibration, {face: illum_mask}, {face: overlay})`` -- the mapping form
        `save_calibration` wants, so every face gets its own PNGs.

    Raises:
        ValueError: if `polygons` is empty, `faces` is empty, or more than `VIALS_PER_FACE`
            polygons are given for a multi-face bundle (which would alias global vial ids).
    """
    if not len(faces):
        raise ValueError("at least one face name is required")
    if len(faces) > 1 and len(polygons) > VIALS_PER_FACE:
        raise ValueError(
            "%d polygons is more than the %d vials per face the pipeline's global-id arithmetic "
            "allows; face %r's ids would collide with face %r's"
            % (len(polygons), VIALS_PER_FACE, faces[0], faces[1])
        )

    calib_faces: Dict[str, FaceCalibration] = {}
    masks: Dict[str, np.ndarray] = {}
    overlays: Dict[str, np.ndarray] = {}
    for name in faces:
        face_cal, mask, overlay = build_calibration_from_polygons(
            polygons, name, frame_gray, image_size, illum_mask=illum_mask)
        calib_faces[name] = face_cal
        masks[name] = mask
        overlays[name] = overlay

    calib = Calibration(
        image_width=int(image_size[0]), image_height=int(image_size[1]), faces=calib_faces,
        created=datetime.now().isoformat(timespec="seconds"),
        notes="%d vial(s) drawn by hand on the live feed; face(s) %s share identical coordinates"
              % (len(polygons), ", ".join(faces)),
    )
    return calib, masks, overlays


def polygons_from_calibration(calib: Calibration, face: Optional[str] = None
                              ) -> Optional[List[List[List[int]]]]:
    """The hand-drawn polygons stored in a bundle, in vial-id order. None if it has none.

    Returning None (rather than falling back to bboxes) is deliberate: this answers "was this
    bundle DRAWN BY HAND?", which is a different question from "can this bundle be reused?".
    Reuse is `saved_selection`'s job, and it accepts older box/quad bundles too -- but it has to
    be able to tell the operator WHICH KIND it found, and this is how.
    """
    if not calib.faces:
        return None
    name = face if face is not None else sorted(calib.faces)[0]
    face_cal = calib.faces.get(name)
    if face_cal is None or not face_cal.vials:
        return None
    vials = sorted(face_cal.vials, key=lambda v: int(v.id))
    if any(getattr(v, "polygon", None) is None for v in vials):
        return None
    return [[[int(px), int(py)] for px, py in v.polygon] for v in vials]


@dataclass
class SavedSelection:
    """A previously saved vial selection, as offered back to the operator at the start of a round."""
    polygons: List[List[List[int]]]
    faces: List[str]
    image_size: Tuple[int, int]
    created: str
    path: str
    #: How the shapes got into the bundle. ``"drawn"`` = polygons clicked on the feed by hand.
    #: ``"boxes"`` = rectangles/quads from the retired auto-detector, which the rig owner judged
    #: too misaligned to trust. Both are REUSABLE -- an existing bundle must never be thrown away
    #: just because it predates the selector -- but callers are expected to treat "boxes" as the
    #: weaker offer and not default to accepting it.
    kind: str = "drawn"

    @property
    def n_vials(self) -> int:
        return len(self.polygons)

    @property
    def hand_drawn(self) -> bool:
        return self.kind == "drawn"


def saved_selection(out_dir: str) -> Optional[SavedSelection]:
    """Read a reusable vial selection back from the bundle in `out_dir`, or None.

    None means there is genuinely nothing to offer: no bundle, an unreadable one, or one with no
    vials in it. Never raises -- a corrupt leftover bundle must send the operator to the drawing
    flow, not to a traceback.

    ANY bundle holding vials is offered, not just one the live selector wrote. A bundle full of
    hand-drawn polygons and one full of older auto-detected boxes are both real work that the
    operator may not want to redo, and refusing to see the second kind would silently force a
    redraw of a rig that was already calibrated. `kind` says which was found so the caller can
    ask about it honestly.
    """
    path = os.path.join(out_dir, "calibration.json")
    if not os.path.isfile(path):
        return None
    try:
        calib = Calibration.from_json(path)
        polygons = polygons_from_calibration(calib)
        kind = "drawn"
        if not polygons:
            # No polygons: fall back to each vial's effective shape (quad, else bbox corners),
            # which is exactly what the pipeline would have measured from this bundle anyway.
            kind = "boxes"
            name = sorted(calib.faces)[0]
            vials = sorted(calib.faces[name].vials, key=lambda v: int(v.id))
            polygons = [[[int(px), int(py)] for px, py in vial_shape(v)] for v in vials]
    except Exception:
        return None
    if not polygons:
        return None
    return SavedSelection(
        polygons=polygons, faces=sorted(calib.faces),
        image_size=(int(calib.image_width), int(calib.image_height)),
        created=str(calib.created or ""), path=path, kind=kind,
    )


# --------------------------------------------------------------------------------------
# Overlay + persistence
# --------------------------------------------------------------------------------------
def _draw_overlay(gray: np.ndarray, vials: Sequence[VialROI], mask: np.ndarray,
                  central_band: Optional[Tuple[int, int]]) -> np.ndarray:
    """Human-check overlay: illum-mask region outlined, central band marked, boxes drawn
    (green = present, red = empty) with ids."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # faint green tint over trackable (lit) pixels
    tint = np.zeros_like(vis)
    tint[mask > 0] = (0, 60, 0)
    vis = cv2.add(vis, tint)
    # yellow outline of the illuminated region
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

    # central hardware band: cyan hatched rectangle + label
    if central_band is not None:
        y0, y1 = central_band
        band = vis.copy()
        cv2.rectangle(band, (0, y0), (vis.shape[1] - 1, y1), (255, 255, 0), -1)
        vis = cv2.addWeighted(vis, 0.75, band, 0.25, 0)
        cv2.rectangle(vis, (0, y0), (vis.shape[1] - 1, y1), (255, 255, 0), 2)
        cv2.putText(vis, "central band (LED/hardware) - EXCLUDED", (10, max(0, y0) + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    for v in vials:
        color = (0, 200, 0) if v.present else (0, 0, 255)
        cv2.rectangle(vis, (v.x, v.y), (v.x + v.w, v.y + v.h), color, 2)
        label = str(v.id) if v.present else "%d X" % v.id
        cv2.putText(vis, label, (v.x + 4, v.y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def _for_face(value, face: str):
    """Resolve a per-face image argument: a plain array applies to every face, a mapping is
    looked up by face name (missing -> None, i.e. "nothing to write for this face")."""
    if isinstance(value, collections.abc.Mapping):
        return value.get(face)
    return value


def save_calibration(calib: Calibration, illum_mask, out_dir: str, overlay=None) -> None:
    """Write the bundle to `out_dir`: calibration.json + illum_mask_<face>.png (+ overlay).

    `illum_mask` and `overlay` are each either ONE image (applied to every face in `calib` --
    the single-face behaviour this has always had) or a ``{face_name: image}`` mapping, which
    is what a multi-face bundle needs so face A and face B get their own mask/overlay PNGs
    (`build_two_face_calibration`).
    """
    os.makedirs(out_dir, exist_ok=True)
    for fc in calib.faces.values():
        mask_name = os.path.basename(fc.illum_mask_path) or ("illum_mask_%s.png" % fc.name)
        mask_img = _for_face(illum_mask, fc.name)
        if mask_img is not None:
            cv2.imwrite(os.path.join(out_dir, mask_name), mask_img)
        overlay_img = _for_face(overlay, fc.name)
        if overlay_img is not None:
            cv2.imwrite(os.path.join(out_dir, "overlay_%s.png" % fc.name), overlay_img)
    calib.to_json(os.path.join(out_dir, "calibration.json"))


def load_calibration(out_dir: str) -> Calibration:
    """Load a bundle from `out_dir`, resolving mask paths to absolute locations."""
    calib = Calibration.from_json(os.path.join(out_dir, "calibration.json"))
    calib.resolve_mask_paths(os.path.abspath(out_dir))
    return calib


def relativize_mask_paths(calib: Calibration) -> Calibration:
    """Strip mask paths back to bare filenames, IN PLACE (inverse of `resolve_mask_paths`).

    `load_calibration` absolutizes them for the current machine; re-saving a loaded bundle
    without undoing that would bake this machine's directory into `calibration.json` and make
    the bundle unmovable. Any code path that loads a bundle, edits it and saves it again (see
    `cli` ``edit-rois``) must call this first.
    """
    for fc in calib.faces.values():
        fc.illum_mask_path = os.path.basename(fc.illum_mask_path)
    return calib


# ======================================================================================
# MARKER-DRIVEN CALIBRATION  (see the module docstring for the rationale)
# ======================================================================================
@dataclass
class MarkerCalibParams:
    """Knobs for the marker-driven path. Everything intensity-related is RELATIVE to the
    frame's own levels, so nothing here depends on this rig's absolute grey values."""

    # --- tube row bands (the two glowing rows either side of the marker band) ---
    band_body_frac: float = 0.45     # a row is "tube" if its mean > frac * (robust bright level)
    band_smooth: int = 9             # smoothing window (px) on the row-mean profile
    band_min_h_frac: float = 0.06    # reject row-runs shorter than frac * image height
    band_inset_frac: float = 0.0     # shrink each band top+bottom by frac of its height

    # --- illuminated mask (computed over IN-BAND pixels only; see `_band_threshold`) ---
    illum_method: str = "otsu"       # "otsu" | "relative"
    illum_dark_pct: float = 5.0      # "unlit" reference percentile (method == "relative")
    illum_bright_pct: float = 95.0   # "fully lit" reference percentile (method == "relative")
    illum_rel: float = 0.40          # threshold = dark + rel*(bright - dark)
    morph_open: int = 3              # opening kernel (px); 0 disables
    morph_close: int = 7             # closing kernel (px); 0 disables
    min_component_frac: float = 0.0004   # drop mask blobs smaller than frac * image area

    # --- vial boxes ---
    box_w_frac: float = 1.0          # box width as a fraction of the marker span (1.0 = as measured)

    # --- suspicion flags (ADVISORY ONLY -- these never set present=False) ---
    # A slot is suspicious when its lit fraction is under BOTH of these (i.e. under their
    # minimum) -- see `_flag_suspicious` for why it is an AND and not an OR.
    suspect_lit_frac_abs: float = 0.60   # ceiling: never flag a slot this well lit
    suspect_lit_frac_rel: float = 0.60   # ... and it must also be this far under the face median

    # --- dwell finding (`find_dwells`, used by `build_two_face_calibration`) ---
    dwell_quiet_ratio: float = 10.0  # frames quieter than ratio * median|diff| are "dwelling"
    dwell_min_frames: int = 5        # ignore quiet runs shorter than this
    face_match_score: float = 0.50   # marker score below which a dwell is a NEW face


@dataclass
class VialQC:
    """Per-slot quality numbers behind a suspicion flag (advisory; see module docstring)."""

    id: int
    lit_frac: float        # fraction of the box's pixels that are in the illuminated mask
    median_gray: float     # median grey level inside the box
    suspicious: bool

    def as_dict(self) -> dict:
        return {"id": int(self.id), "lit_frac": round(float(self.lit_frac), 4),
                "median_gray": round(float(self.median_gray), 2),
                "suspicious": bool(self.suspicious)}


# --------------------------------------------------------------------------------------
# Tube row bands
# --------------------------------------------------------------------------------------
def _marker_band_rows(strips: Sequence[Sequence[int]]) -> Tuple[int, int]:
    """(first_row, last_row) INCLUSIVE covering every marker strip."""
    return min(int(s[0]) for s in strips), max(int(s[1]) for s in strips)


def tube_row_bands(
    gray: np.ndarray,
    strips: Sequence[Sequence[int]],
    x_extent: Tuple[int, int],
    params: Optional[MarkerCalibParams] = None,
) -> Tuple[RowBand, RowBand, RowBand]:
    """Locate the two glowing tube rows around the marker band.

    Returns ``(upper, lower, central)`` as HALF-OPEN ``(y0, y1)`` row ranges, where `central`
    is everything between the two tube rows -- the marker band plus its surrounding mounting
    hardware -- which is excluded from the illuminated mask.

    Method. Take the row-mean profile of the frame **restricted to the vial columns' x extent**
    (so the drum's side hardware cannot contribute), smooth it, and threshold it at
    ``band_body_frac`` of a robust bright level (the median of the profile's own upper half, so
    a saturated stage or LED slot cannot inflate it). That yields a handful of bright row-runs.
    The tube rows are then simply **the run nearest the marker band on each side**: the band
    physically sits between the two vial rows (DESIGN.md §2), so nearest-above is the upper row
    and nearest-below is the lower row.

    That last step is the whole trick, and it is what makes this robust to the brightest object
    in the frame. On the real rig the illuminated stage along the bottom edge is *brighter and
    taller* than either tube row (measured: rows ~893-1023 at mean ~228, versus the lower tube
    row at ~127), so any "pick the brightest/biggest runs" rule picks the stage. Picking by
    ADJACENCY TO THE BAND instead ignores brightness entirely and never can.

    Raises:
        ValueError: no qualifying bright run above and/or below the marker band.
    """
    p = params or MarkerCalibParams()
    H, W = gray.shape[:2]
    x0, x1 = int(x_extent[0]), int(x_extent[1])
    x0 = max(0, min(x0, W - 1))
    x1 = max(x0, min(x1, W - 1))

    prof = _smooth1d(gray[:, x0:x1 + 1].mean(axis=1).astype(np.float64), p.band_smooth)
    bright_level = float(np.median(prof[prof >= np.median(prof)]))
    runs = _bright_runs(prof > p.band_body_frac * bright_level,
                        max(1, int(p.band_min_h_frac * H)))
    if not runs:
        raise ValueError("no bright row-runs found; cannot locate the tube rows")

    b0, b1 = _marker_band_rows(strips)
    above = [r for r in runs if r[1] <= b0]      # entirely above the band (half-open end)
    below = [r for r in runs if r[0] > b1]       # entirely below it
    if not above or not below:
        raise ValueError(
            "expected a tube row on each side of the marker band (rows %d..%d); found %d above "
            "and %d below among runs %r" % (b0, b1, len(above), len(below), runs)
        )
    upper = max(above, key=lambda r: r[1])       # nearest above
    lower = min(below, key=lambda r: r[0])       # nearest below

    if p.band_inset_frac > 0:
        upper = _inset_band(upper, p.band_inset_frac)
        lower = _inset_band(lower, p.band_inset_frac)
    return upper, lower, (upper[1], lower[0])


def _inset_band(band: RowBand, frac: float) -> RowBand:
    y0, y1 = band
    d = int(round(frac * (y1 - y0)))
    return (y0 + d, max(y0 + d + 1, y1 - d))


# --------------------------------------------------------------------------------------
# Illuminated mask over the tube bands
# --------------------------------------------------------------------------------------
def _band_threshold(gray: np.ndarray, bands: Sequence[RowBand], x_extent: Tuple[int, int],
                    p: MarkerCalibParams) -> float:
    """Grey level separating "lit" from "unlit", computed over IN-BAND pixels only.

    Restricting the statistic to the tube rows (and to the vial columns' x extent) is what
    makes it meaningful: an Otsu over the FULL frame is dominated by the saturated stage and
    the large dark surround, neither of which is a tube.
    """
    x0, x1 = x_extent
    pool = np.concatenate([gray[y0:y1, x0:x1 + 1].ravel() for y0, y1 in bands])
    if pool.size == 0:
        return 0.0
    if p.illum_method == "relative":
        dark = float(np.percentile(pool, p.illum_dark_pct))
        bright = float(np.percentile(pool, p.illum_bright_pct))
        return dark + p.illum_rel * (bright - dark)
    thr, _ = cv2.threshold(pool.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def _marker_illum_mask(gray: np.ndarray, bands: Sequence[RowBand], x_extent: Tuple[int, int],
                       p: MarkerCalibParams) -> np.ndarray:
    """Lit pixels inside the tube rows; everything else (incl. the central band) zeroed."""
    H, W = gray.shape[:2]
    thr = _band_threshold(gray, bands, x_extent, p)
    mask = np.where(gray > thr, np.uint8(255), np.uint8(0))

    if p.morph_open and p.morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_open, p.morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if p.morph_close and p.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_close, p.morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Keep ONLY the two tube rows: this is what excludes the central marker/hardware band
    # (and, for free, the saturated stage along the bottom edge and the dark surround).
    keep = np.zeros((H, W), dtype=bool)
    x0, x1 = x_extent
    for y0, y1 in bands:
        keep[max(0, y0):min(H, y1), max(0, x0):min(W, x1 + 1)] = True
    mask[~keep] = 0
    return _drop_small_components(mask, int(p.min_component_frac * H * W))


# --------------------------------------------------------------------------------------
# Public API -- one face
# --------------------------------------------------------------------------------------
def build_calibration_from_markers(
    frame_gray: np.ndarray,
    face: str,
    marker_detector,
    params: Optional[MarkerCalibParams] = None,
    illum_mask_path: Optional[str] = None,
) -> Tuple[FaceCalibration, np.ndarray, np.ndarray]:
    """Build ONE face's calibration from the physical marker band (see module docstring).

    Args:
        frame_gray: HxW grayscale still of this face, taken during a dwell.
        face: face name ("A"/"B").
        marker_detector: duck-typed; needs ``vial_boundaries(frame) -> [(x0, x1), ...]`` and
            ``find_strips(frame) -> [(r0, r1), ...]`` -- i.e. a
            `marker_band.MarkerBandDetector`.
        params: `MarkerCalibParams` (defaults if None).
        illum_mask_path: stored path for the mask PNG (default ``illum_mask_<face>.png``).

    Returns:
        ``(face_calibration, illum_mask_uint8, overlay_bgr)``. **Every vial is
        `present=True`** -- see the module docstring for why this path refuses to auto-exclude.
        Slots that look empty are reported instead: on the overlay (orange box, "?" suffix) and
        in ``face_calibration.marker["suspicious"]`` (read back via `suspicious_vials`).

    Raises:
        ValueError: the detector found no marker band / not enough vial spans, or no tube row
            could be located on one side of the band.
    """
    p = params or MarkerCalibParams()
    gray = _as_gray(frame_gray)
    H, W = gray.shape[:2]

    find_strips = getattr(marker_detector, "find_strips", None)
    if not callable(find_strips):
        raise ValueError(
            "marker_detector has no find_strips(); the marker path needs the marker band's row "
            "position to tell the two tube rows apart (use marker_band.MarkerBandDetector)"
        )
    strips = list(find_strips(gray))
    spans = [(int(a), int(b)) for a, b in marker_detector.vial_boundaries(gray)]
    if not strips:
        raise ValueError("marker band not visible in this frame; cannot calibrate face %r" % face)
    if len(spans) != N_VIAL_COLUMNS:
        # Hard failure, not a best-effort fallback. Vial ids run 1..2*len(spans) here, while
        # `pipeline.TrackerPipeline` computes global ids as `face_index * 16 + local_id` with 16
        # hard-coded (DESIGN.md §2). A face that produced 9 columns would emit local ids up to
        # 18, and id 17 on face A would silently collide with id 1 on face B -- two different
        # physical vials logged under one global id, with nothing anywhere reporting it. Better
        # to stop now, while a human is watching the calibration.
        raise ValueError(
            "marker detector returned %d vial span(s) for face %r; expected exactly %d (one per "
            "vial column). The 16-slots-per-face id scheme depends on it -- check the marker "
            "band is unobstructed in this frame, or calibrate by hand with `calibrate --wizard`."
            % (len(spans), face, N_VIAL_COLUMNS)
        )

    x_extent = (min(a for a, _ in spans), max(b for _, b in spans))
    upper, lower, central = tube_row_bands(gray, strips, x_extent, p)
    mask = _marker_illum_mask(gray, (upper, lower), x_extent, p)

    # 16 boxes, row-major: upper band -> ids 1..8, lower band -> ids 9..16, left->right.
    vials: List[VialROI] = []
    qc: List[VialQC] = []
    for row, (y0, y1) in enumerate((upper, lower)):
        for col, (sx0, sx1) in enumerate(spans):
            span_w = sx1 - sx0 + 1
            bw = max(1, int(round(span_w * p.box_w_frac)))
            bx = int(round(sx0 + (span_w - bw) / 2.0))
            bx = max(0, min(bx, W - 1))
            bw = min(bw, W - bx)
            box = (bx, int(y0), int(bw), int(y1 - y0))
            # PRESENCE IS ALWAYS TRUE HERE -- see the module docstring.
            vials.append(VialROI(id=row * len(spans) + col + 1, row=row, col=col,
                                 x=box[0], y=box[1], w=box[2], h=box[3], present=True))
            qc.append(_vial_qc(vials[-1].id, gray, mask, box))

    _flag_suspicious(qc, p)
    face_cal = FaceCalibration(
        name=face,
        vials=vials,
        illum_mask_path=illum_mask_path or ("illum_mask_%s.png" % face),
        marker={
            "source": "marker_band",
            "strips": [[int(a), int(b)] for a, b in strips],
            "vial_spans": [[int(a), int(b)] for a, b in spans],
            "row_bands": {"upper": [int(upper[0]), int(upper[1])],
                          "lower": [int(lower[0]), int(lower[1])]},
            "central_band": [int(central[0]), int(central[1])],
            "presence_policy": "all present; empties are reported, never auto-excluded",
            "suspicious": [q.id for q in qc if q.suspicious],
            "vial_qc": [q.as_dict() for q in qc],
        },
    )
    overlay = _draw_marker_overlay(gray, vials, mask, spans, (upper, lower), central, qc)
    return face_cal, mask, overlay


def _vial_qc(vial_id: int, gray: np.ndarray, mask: np.ndarray, box: Box) -> VialQC:
    x, y, w, h = box
    sub_mask = mask[y:y + h, x:x + w]
    sub_gray = gray[y:y + h, x:x + w]
    lit_frac = float((sub_mask > 0).mean()) if sub_mask.size else 0.0
    median_gray = float(np.median(sub_gray)) if sub_gray.size else 0.0
    return VialQC(id=vial_id, lit_frac=lit_frac, median_gray=median_gray, suspicious=False)


def _flag_suspicious(qc: Sequence[VialQC], p: MarkerCalibParams) -> None:
    """Mark slots whose lit area is far below the rest of the face's. ADVISORY ONLY.

    A slot is suspicious when its lit fraction is below BOTH thresholds, i.e. below their
    MINIMUM:

      * ``suspect_lit_frac_rel * median(lit_frac)`` -- the substantive test. It has to be
        relative to the face's own median because the back-lighting is non-uniform and
        left-biased (DESIGN.md §2): "dim" only means anything compared with this face's norm.
      * ``suspect_lit_frac_abs`` -- a ceiling, not a floor. However far under the median a slot
        sits, a slot that is still 60% lit is plainly a real, measurable vial and saying
        otherwise would be noise.

    Taking the MINIMUM (an AND) is what keeps the flag informative on a globally dim rig: if
    every slot images at ~0.5 lit, the maximum would put the cut at 0.60 and flag all 16, which
    tells a human nothing. The minimum puts it at 0.30 and flags only what genuinely stands out.

    Nothing here touches `VialROI.present` -- see the module docstring.
    """
    if not qc:
        return
    med = float(np.median([q.lit_frac for q in qc]))
    thr = min(p.suspect_lit_frac_abs, p.suspect_lit_frac_rel * med)
    for q in qc:
        q.suspicious = bool(q.lit_frac < thr)


def suspicious_vials(face_cal: FaceCalibration) -> List[int]:
    """Ids a marker-built face flagged as looking empty (they are still `present=True`).

    Returns ``[]`` for a face built by any other path, or one with nothing flagged.
    """
    marker = face_cal.marker
    if not isinstance(marker, dict):
        return []
    return [int(i) for i in marker.get("suspicious", [])]


def _draw_marker_overlay(gray: np.ndarray, vials: Sequence[VialROI], mask: np.ndarray,
                         spans: Sequence[Tuple[int, int]], bands: Sequence[RowBand],
                         central: RowBand, qc: Sequence[VialQC]) -> np.ndarray:
    """Human-check overlay for the marker path.

    Draws: the trackable (lit) pixels tinted green; the two tube row-bands as blue rules; the
    excluded central marker/hardware band as a cyan wash; the 8 marker-derived vial spans as
    magenta ticks along the top; and the 16 boxes with their ids -- GREEN for a normal slot,
    ORANGE with a "?" for a slot flagged suspicious (which is still measured; see the module
    docstring).
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = gray.shape[:2]

    tint = np.zeros_like(vis)
    tint[mask > 0] = (0, 60, 0)
    vis = cv2.add(vis, tint)

    # excluded central band (marker strips + mounting hardware)
    cy0, cy1 = max(0, central[0]), min(H, central[1])
    if cy1 > cy0:
        band = vis.copy()
        cv2.rectangle(band, (0, cy0), (W - 1, cy1), (255, 255, 0), -1)
        vis = cv2.addWeighted(vis, 0.75, band, 0.25, 0)
        cv2.rectangle(vis, (0, cy0), (W - 1, cy1), (255, 255, 0), 2)
        cv2.putText(vis, "marker/hardware band - EXCLUDED", (10, cy0 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # tube row bands
    for y0, y1 in bands:
        for y in (y0, y1 - 1):
            cv2.line(vis, (0, int(y)), (W - 1, int(y)), (255, 120, 0), 1)

    # marker-derived vial spans (the x source of truth)
    for i, (sx0, sx1) in enumerate(spans):
        cv2.line(vis, (int(sx0), 2), (int(sx1), 2), (255, 0, 255), 4)
        cv2.line(vis, (int(sx0), 0), (int(sx0), 14), (255, 0, 255), 2)
        cv2.line(vis, (int(sx1), 0), (int(sx1), 14), (255, 0, 255), 2)
        cv2.putText(vis, "c%d" % (i + 1), (int(sx0) + 4, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    flagged = {q.id: q for q in qc}
    for v in vials:
        q = flagged.get(v.id)
        suspect = bool(q is not None and q.suspicious)
        color = (0, 165, 255) if suspect else (0, 200, 0)
        cv2.rectangle(vis, (v.x, v.y), (v.x + v.w, v.y + v.h), color, 2)
        label = "%d?" % v.id if suspect else str(v.id)
        cv2.putText(vis, label, (v.x + 4, v.y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if q is not None:
            cv2.putText(vis, "lit %.2f" % q.lit_frac, (v.x + 4, v.y + v.h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if any(q.suspicious for q in qc):
        ids = ",".join(str(q.id) for q in qc if q.suspicious)
        cv2.putText(vis, "suspicious (still measured): %s" % ids, (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    return vis


# --------------------------------------------------------------------------------------
# Public API -- both faces, from a flip video
# --------------------------------------------------------------------------------------
@dataclass
class Dwell:
    """One stationary period of the drum, as found by `find_dwells`."""

    start: int            # first frame index of the dwell
    end: int              # last frame index (inclusive)
    rep_index: int        # the representative frame index picked from it (its midpoint)
    face: Optional[str] = None        # filled in by `register_faces_from_dwells`
    face_score: float = float("nan")  # marker similarity to `face` (1.0 = the registration frame)

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def find_dwells(video_path: str, params: Optional[MarkerCalibParams] = None
                ) -> Tuple[List[Dwell], Dict[int, np.ndarray], Tuple[int, int]]:
    """Find the drum's stationary periods in a flip video and grab one frame from each.

    The drum alternates rotate / dwell / rotate / dwell..., and only a dwell is calibratable.
    Dwells are the runs of low global ``mean|frame - prev_frame|`` (DESIGN.md §5.1's metric,
    used here purely as an offline segmenter, not as the online detector).

    The threshold is ``dwell_quiet_ratio`` x the clip's own MEDIAN metric, so it scales itself
    to the rig's noise floor instead of assuming a magnitude. The median is a dwell-population
    value whenever the drum spends more of the clip parked than turning, which is true by
    construction for a calibration recording. Measured on `Good Markers.avi`: dwell frames sit
    at 0.32-0.45 (median 0.352, occasional micro-event excursions to ~1.5) and rotation frames
    at 8.4-72.7, so the default 10x (= 3.5) lands in an empty gap that is 5x wide on either
    side. Anything from ~4x to ~20x segments this clip identically; dropping to 2-3x additionally
    splits each dwell at its internal micro-events (43 fragments instead of 14 true dwells),
    which costs nothing but noise since only a midpoint frame per dwell is used.

    Two passes over the file: the first measures the metric (a float per frame), the second
    re-reads and keeps ONLY the chosen representative frames, so memory stays at ~one frame per
    dwell rather than the whole clip.

    Returns:
        ``(dwells, {frame_index: gray_frame}, (width, height))``.

    Raises:
        ValueError: the video cannot be opened, or holds fewer than 2 frames.
    """
    p = params or MarkerCalibParams()
    metric: List[float] = []
    prev: Optional[np.ndarray] = None
    size = (0, 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("could not open video %r" % video_path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = _as_gray(frame)
            size = (gray.shape[1], gray.shape[0])
            if prev is not None:
                metric.append(float(cv2.absdiff(gray, prev).mean()))
            prev = gray
    finally:
        cap.release()

    if len(metric) < 1:
        raise ValueError("video %r holds fewer than 2 frames" % video_path)

    # `<=` (not `<`) so a perfectly static clip -- median 0, threshold 0 -- still reads as one
    # long dwell instead of raising "no dwell found". On real footage sensor noise makes the
    # metric strictly positive and the distinction is immaterial.
    quiet = np.asarray(metric) <= p.dwell_quiet_ratio * float(np.median(metric))
    dwells: List[Dwell] = []
    for a, b in _bright_runs(quiet, max(1, int(p.dwell_min_frames))):
        # metric[i] compares frame i+1 with frame i, so a quiet metric run [a, b) means
        # frames a..b are mutually still.
        start, end = a, b
        dwells.append(Dwell(start=start, end=end, rep_index=(start + end) // 2))
    if not dwells:
        raise ValueError(
            "no dwell (stationary period) found in %r: the drum never held still for "
            "%d frames" % (video_path, p.dwell_min_frames)
        )

    wanted = {d.rep_index for d in dwells}
    reps: Dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(video_path)
    try:
        idx = 0
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                reps[idx] = _as_gray(frame)
                wanted.discard(idx)
            idx += 1
    finally:
        cap.release()
    return dwells, reps, size


#: Face names assigned to distinct faces in the order they are first seen.
FACE_NAMES = ("A", "B")


def register_faces_from_dwells(
    dwells: Sequence[Dwell],
    reps: Mapping[int, np.ndarray],
    marker_detector,
    params: Optional[MarkerCalibParams] = None,
) -> Dict[str, List[Dwell]]:
    """Register one marker template per DISTINCT face and label every dwell with its face.

    This is the bootstrap: `identify_face` needs a template per face, but a template can only be
    registered from a frame already known to show that face. Walking the dwells in time order
    solves it -- the first dwell defines face "A"; the first dwell that does not match "A"
    defines face "B"; everything after that is scored against both.

    WHY THE MARKER SIGNATURE AND NOT RAW FRAME SIMILARITY. Comparing raw frames
    (``mean|rep_i - rep_0|``) looks like it should work -- on adjacent dwells of `Good Markers.avi`
    same-face pairs differ by ~1.2 grey levels and flips by ~19.4. But that statistic DRIFTS
    over a clip: face A's own dwells measure 1.28 against the first dwell early on and 9.28-9.71
    by the end, which overlaps the flipped population and mislabels 5 of 14 dwells (measured).
    The marker signature has no such drift -- it is percentile-relative, cropped to the strips'
    lit extent and resampled (see `marker_band`), so it measures the sticker PATTERN and not the
    scene. Measured over the same dwells: same-face score 0.9876..1.0000, other-face
    -0.2494..-0.2327, i.e. a **gap of 1.22** with the `face_match_score` cut (0.50) sitting in
    the middle of it. It labels 14/14 (and 43/43 under the finer segmentation) correctly.

    Mutates `marker_detector` (registers templates) and the `Dwell` objects (sets `face` and
    `face_score`), so the same detector instance can be handed straight to `TrackerPipeline`.

    Returns:
        ``{face_name: [Dwell, ...]}`` in time order. Dwells whose marker band is not readable
        are skipped entirely (they appear in no group).

    Raises:
        ValueError: not one dwell had a readable marker band.
    """
    p = params or MarkerCalibParams()
    groups: Dict[str, List[Dwell]] = {}
    registered: List[str] = []

    for d in dwells:
        frame = reps.get(d.rep_index)
        if frame is None:
            continue
        if not registered:
            try:
                marker_detector.register_face(frame, FACE_NAMES[0])
            except ValueError:
                continue                       # no band in this frame; try the next dwell
            registered.append(FACE_NAMES[0])
            d.face, d.face_score = FACE_NAMES[0], 1.0
            groups.setdefault(FACE_NAMES[0], []).append(d)
            continue

        scores = marker_detector.score_faces(frame)
        if not scores:
            continue                           # band not readable in this dwell
        best = max(scores, key=lambda f: scores[f])
        if scores[best] < p.face_match_score and len(registered) < len(FACE_NAMES):
            best = FACE_NAMES[len(registered)]
            marker_detector.register_face(frame, best)
            registered.append(best)
            d.face_score = 1.0
        else:
            d.face_score = float(scores[best])
        d.face = best
        groups.setdefault(best, []).append(d)

    if not groups:
        raise ValueError(
            "the marker band was not readable in any of the %d dwell frame(s); cannot identify "
            "faces (is this a marker-band rig? see marker_band.MarkerBandDetector)" % len(dwells)
        )
    return groups


def build_two_face_calibration(
    video_path: str,
    marker_detector,
    out_dir: str,
    params: Optional[MarkerCalibParams] = None,
) -> Calibration:
    """Calibrate BOTH drum faces from one flip video and save the bundle to `out_dir`.

    Pipeline:
      1. `find_dwells` -- segment the clip into stationary periods, one representative frame each.
      2. `register_faces_from_dwells` -- walk the dwells in time order, registering a
         `marker_detector` template from the FIRST dwell of each distinct face ("A", then "B")
         and labelling every dwell with the face it shows.
      3. Pick each face's representative frame from its LONGEST dwell -- the most settled one.
      4. `build_calibration_from_markers` per face -> a 16-slot `FaceCalibration` (all
         `present=True`), an illuminated mask and an overlay.
      5. Save via `save_calibration`: ``calibration.json`` + ``illum_mask_<face>.png`` +
         ``overlay_<face>.png`` per face.

    The detector is mutated in place (its templates are registered in step 2), so the SAME
    instance can be handed straight to `pipeline.TrackerPipeline(marker_detector=...)`. The
    templates are also stored in each face's ``marker["band_templates"]``, so a later run can
    rebuild an equivalent detector from the bundle alone -- see
    `marker_detector_from_calibration`.

    Returns:
        The saved `Calibration` (1 face if the clip only ever showed one, else 2).

    Raises:
        ValueError: no dwell found, or the marker band is unusable in the chosen frames.
    """
    p = params or MarkerCalibParams()
    dwells, reps, (width, height) = find_dwells(video_path, p)
    groups = register_faces_from_dwells(dwells, reps, marker_detector, p)
    n_labelled = sum(len(g) for g in groups.values())

    faces: Dict[str, FaceCalibration] = {}
    masks: Dict[str, np.ndarray] = {}
    overlays: Dict[str, np.ndarray] = {}
    notes: List[str] = []
    for name in sorted(groups):
        group = groups[name]
        best = max(group, key=lambda d: d.length)
        face_cal, mask, overlay = build_calibration_from_markers(
            reps[best.rep_index], name, marker_detector, p)
        sig = getattr(marker_detector, "templates", {}).get(name)
        if sig is not None:
            face_cal.marker["band_templates"] = [np.asarray(s).tolist() for s in sig]
        face_cal.marker["source_frame"] = int(best.rep_index)
        face_cal.marker["n_dwells"] = len(group)
        faces[name] = face_cal
        masks[name] = mask
        overlays[name] = overlay
        scored = [d.face_score for d in group if d.face_score == d.face_score]  # drop NaN
        flagged = suspicious_vials(face_cal)
        notes.append("face %s: frame %d (longest of %d dwells, %d frames), %d vials, "
                     "min marker score %.3f, suspicious=%s"
                     % (name, best.rep_index, len(group), best.length, len(face_cal.vials),
                        min(scored) if scored else float("nan"),
                        flagged if flagged else "none"))

    calib = Calibration(
        image_width=int(width), image_height=int(height), faces=faces,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=("marker-band calibration from %s: %d dwells, %d labelled, %d face(s). "
               "ALL slots present=True by policy -- suspicious slots are reported, not "
               "excluded. " % (os.path.basename(video_path), len(dwells), n_labelled,
                               len(faces))) + "; ".join(notes),
    )
    save_calibration(calib, masks, out_dir, overlay=overlays)
    return calib


def marker_detector_from_calibration(calib: Calibration):
    """Rebuild a `marker_band.MarkerBandDetector` from templates stored in a saved bundle.

    Lets a restarted run identify faces with exactly the templates it was calibrated with,
    instead of re-deriving them. Returns ``None`` if the bundle carries no band templates
    (i.e. it was not built by the marker path).
    """
    from flygym_tracker.marker_band import MarkerBandDetector

    templates = {}
    for name, fc in calib.faces.items():
        marker = fc.marker
        if isinstance(marker, dict) and marker.get("band_templates"):
            templates[name] = marker["band_templates"]
    if not templates:
        return None
    return MarkerBandDetector(templates=templates)

"""Calibration for a FlyGym v2 drum face.

Produces the `Calibration` bundle described in DESIGN.md §5.4/§5.5:
  * an illuminated mask (255 = trackable lit pixel, 0 = excluded), with the central
    LED/hardware band zeroed out,
  * a 2x8 vial lattice (16 `VialROI`, ids 1..16 row-major), and
  * per-slot tube-presence flags.

Two entry points, emitting an IDENTICAL bundle:
  * `detect_calibration(frame, face)` -- auto-detect accelerator (DESIGN §5.4 B). It finds
    the lattice geometry and hands the boxes to `build_calibration_from_boxes`, which does
    all the bundle building.  Its boxes also serve as *seed boxes* for the manual wizard.
  * `build_calibration_from_boxes(frame, face, boxes, present_flags)` -- the pure,
    unit-testable core used by the manual ROI wizard (DESIGN §5.5): given vial boxes and
    optional present/absent flags it derives each box's illuminated sub-mask (bright pixels
    inside the box, minus the central band) and assembles the same bundle.

The interactive wizard driver lives in `calibrate_wizard.py` and only wraps
`build_calibration_from_boxes`; no CV logic lives there.

Nothing here is hard-coded to the pixel coordinates of any one frame: geometry comes from
row/column intensity profiles and presence from illumination-relative thresholds, all of
which are exposed as `CalibParams` fields.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from flygym_tracker.types import Calibration, FaceCalibration, VialROI

# (x, y, w, h) pixel box in full-frame coords.
Box = Tuple[int, int, int, int]


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


def save_calibration(calib: Calibration, illum_mask: np.ndarray, out_dir: str,
                     overlay: Optional[np.ndarray] = None) -> None:
    """Write the bundle to `out_dir`: calibration.json + illum_mask_<face>.png (+ overlay)."""
    os.makedirs(out_dir, exist_ok=True)
    for fc in calib.faces.values():
        mask_name = os.path.basename(fc.illum_mask_path) or ("illum_mask_%s.png" % fc.name)
        cv2.imwrite(os.path.join(out_dir, mask_name), illum_mask)
        if overlay is not None:
            cv2.imwrite(os.path.join(out_dir, "overlay_%s.png" % fc.name), overlay)
    calib.to_json(os.path.join(out_dir, "calibration.json"))


def load_calibration(out_dir: str) -> Calibration:
    """Load a bundle from `out_dir`, resolving mask paths to absolute locations."""
    calib = Calibration.from_json(os.path.join(out_dir, "calibration.json"))
    calib.resolve_mask_paths(os.path.abspath(out_dir))
    return calib

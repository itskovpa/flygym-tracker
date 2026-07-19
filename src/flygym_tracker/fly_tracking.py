"""Individual fly detection + behavioural metrics for the climbing (negative-geotaxis) assay.

This is a COMPLEMENT to `activity.py`, not a replacement. The two answer different questions:

    activity.py   "how much movement happened in this vial?"   (frame differencing)
    fly_tracking  "where are the flies, and how high did they climb?" (silhouette detection)

`activity.py` is the robust workhorse for long runs: it needs no fly model, it is one absdiff
per frame, and it degrades gracefully. What it cannot give you is POSITION — and in a climbing
assay, position along the tube axis over time IS the biology. That is what this module adds.

The physics it exploits
-----------------------
Flies are back-lit at 850 nm (DESIGN.md §2) and imaged as **dark silhouettes on bright glowing
tubes**. So a fly is simply a compact patch of pixels that is darker than the tube glow around
it. Detection is therefore: threshold dark pixels -> connected components -> centroids. Light,
single-pass, no background model to maintain across frames.

Why the threshold is RELATIVE, and why that matters here
--------------------------------------------------------
The rig's back-light has a strong left-right gradient (DESIGN.md §2: "columns 1-7 well lit; the
rightmost column falls into darkness"), so a FIXED grey level is useless: on the real frame the
per-vial median glow runs from 116 to 184 counts. A global cut at, say, 130 would flag the whole
of the dim vials as "fly" and miss every fly in the bright ones.

Instead the threshold is built per-ROI from the image itself, in two stages:

1.  **Local background estimate** — a large-kernel median blur (`bg_kernel`, default 31 px, i.e.
    ~3x a fly's long axis). A median over a window that big is dominated by tube glow, so the
    fly's own pixels cannot pull it down; the result is "what this patch of tube would look like
    with no fly on it". Because the window is local, the left-right gradient is *inside* the
    background estimate and cancels exactly. `residual = background - image` is then a flat,
    zero-centred field with flies as positive bumps, whatever the local glow level.

2.  **Robust noise scale** — `sigma` from the residual's INTERQUARTILE RANGE (`IQR/1.349`),
    which ignores both tails: the positive tail (the flies themselves, which would otherwise
    inflate the threshold in exactly the vials that have flies) and the negative tail (specular
    streaks on the tube wall, which are brighter than the background estimate). Measured on the
    real rig this matters a lot: the lower-tail estimator that a naive MAD would use reports
    sigma = 13.0 on vial 12 (a vial with bright reflections) where the IQR estimator reports
    6.67, and the inflated value silently desensitises that vial.

    The final cut is `delta = max(min_delta, k_sigma * sigma)` — relative in the normal case,
    with an absolute floor so a freakishly quiet ROI cannot produce a hair-trigger threshold.

Rejecting the tube walls
------------------------
Thresholding alone also finds the dark tube-wall edges and moulding seams, which are *long thin*
dark structures. Two shape filters remove them, both stated as fly physics rather than as tuned
constants:
  * `min_thickness` — a fly silhouette is at least a few px across in its narrow direction; a
    wall seam is 2-4 px wide but 250 px long, so its fragments die here.
  * `max_aspect` — a fly (or a small clump) is not 30x longer than it is wide; a full wall edge is.

Both are single-frame filters. Static structure is fundamentally better handled by a temporal
background, which this module does not build — see "Robust vs approximate" below.

Merging is real and is reported, not hidden
--------------------------------------------
There are 15-25 flies per vial (DESIGN.md §10). They WILL touch, and touching flies threshold
into ONE connected component. Rather than pretend each blob is one fly, every blob carries
`is_merged` and `n_flies`, both derived from a single-fly area that `estimate_single_fly_area`
learns from the small-blob population of the data itself — so it self-calibrates per vial and
per experiment instead of assuming this rig's pixel scale.

Robust vs approximate — please read before trusting a number
-------------------------------------------------------------
ROBUST (population-level; these are what to put in a figure):
  * blob counts and `est_n_flies` per frame,
  * the HEIGHT DISTRIBUTION along the vial axis (`mean_height`, `median_height`, `frac_above`,
    `max_height`) — this is the negative-geotaxis readout and it needs no identity at all,
  * aggregate speed statistics pooled over many short fragments,
  * `total_blob_area` as a crude occupancy/biomass proxy.

APPROXIMATE (use with the stated caveat, never as per-animal data):
  * INDIVIDUAL IDENTITY. `link_blobs` is nearest-neighbour with a distance gate. With 15-25
    flies in one tube, identity swaps when two flies pass each other are not a bug, they are
    expected and frequent. No fly can be followed for the length of a dwell.
  * PER-FLY TRACK LENGTH under crowding. A track ends whenever its fly merges into a clump and
    a NEW track id begins when the clump splits, so tracks are FRAGMENTS, not trajectories.
    `mean_fragment_frames` in `summarize()` is exactly the diagnostic for this: high (tens of
    frames) means the tracks are worth something; ~1-3 means they are shredded and only the
    pooled statistics are meaningful.
  * ABSOLUTE FLY COUNT. `est_n_flies` divides area by a single-fly area, which under-counts a
    tight ball of flies (they occlude one another) and cannot see a fly sitting on a wall seam
    (its silhouette is inside the background estimate there).

    It can also OVER-count in a SPARSE vial, and that was measured, not hypothesised. On the
    real 90 s capture (vial 7, only ~3 flies in the tube, so essentially no touching) the pooled
    blob areas were a smooth unimodal continuum — 16, 17, 21, ... 48 (median), ... 102, 109, 154
    — with no clump mode at all. That ~7x spread is not clumping; it is how much of each fly's
    silhouette clears the threshold, which depends on the fly's pose and depth. Fed a continuum,
    `estimate_single_fly_area` anchors on the small end (39.5 px) and the ordinary 80-110 px
    singles then read as 2-3 flies: 3.27 blobs/frame became 4.5 est_flies/frame, ~1.4x too many.
    The estimator is not misbehaving — it is answering the question it was asked in a vial that
    violates its premise. Under the DESIGN §10 condition it was built for (15-25 flies, where
    touching is constant and a genuine clump mode exists) that premise holds.

    PRACTICAL RULE: `summarize()` deliberately reports `n_blobs_mean` AND `est_n_flies_mean`
    side by side. Read them as a bracket — `n_blobs_mean` is a FLOOR on the fly count (merged
    flies counted once), `est_n_flies_mean` is a CEILING (area scatter counted as merging). In
    a sparse vial trust the floor; in a crowded one the truth is nearer the ceiling.
  * ANY vial in the rightmost, badly-lit column, where the SNR premise fails outright.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: (x, y) in full-frame pixel coordinates.
Point = Tuple[float, float]
#: (x, y, w, h) in full-frame pixel coordinates.
BBox = Tuple[int, int, int, int]


# ======================================================================================
# Blob detection
# ======================================================================================
@dataclass
class DetectParams:
    """Knobs for `detect_flies`. Every intensity knob is RELATIVE to the ROI's own statistics.

    The defaults were measured on the real rig (1280x1024, HikRobot, 850 nm back-light) with a
    fly silhouette of roughly 10x6 px / 40-130 px area. `bg_kernel`, `min_area`, `max_area`,
    `min_thickness` are the four that scale with pixel size — change them together if the optics
    change; the intensity knobs should carry over unmodified.
    """

    # --- background estimate ---
    bg_kernel: int = 31          # median-blur window (odd, px). Must comfortably exceed a fly.
    roi_erode: int = 3           # shrink the ROI by this many px, to drop mask-boundary edges.
    bg_margin: int = 0           # extra context px fed to the blur (0 -> bg_kernel // 2).

    # --- threshold ---
    # k_sigma=10 is MEASURED, not guessed, and the optimum is a broad plateau, not a knife edge.
    # Swept on the real 1802-frame capture over the 15 Face-A dwells (vial 7 = the fly vial, the
    # other 15 vials empty), counting mean blobs/frame:
    #     k_sigma      3.5   4.0   5.0   6.0   8.0   9.0  10.0  12.0  14.0  16.0  18.0
    #     vial 7      3.00  2.87  2.40  2.53  2.67  3.20  3.27  3.13  2.87  2.33  1.40
    #     worst other 2.07  4.07  3.00  0.47  0.00  0.00  0.00  0.00  0.00  0.00  0.00
    # Two things to read off it. (a) False positives hit exactly ZERO at k>=8 and stay there, so
    # 10 sits two sigma inside the clean region. (b) vial 7's count RISES from 6 to 10 — lowering
    # the threshold does not find more flies, it finds the tube-wall edges, which then fuse with
    # nearby flies into one component that `max_aspect` throws away. So a too-low threshold loses
    # real flies. Anywhere in 9..12 behaves the same; the roll-off above 14 is genuine flies
    # dropping below the cut.
    # Physically: sensor noise sigma is ~3 counts and a real fly silhouette measures 52-72 counts
    # below local background, i.e. ~20 sigma, so a 10-sigma cut sits midway between noise and
    # signal with ~1.7x headroom on the faintest real fly.
    k_sigma: float = 10.0        # cut at k_sigma * robust sigma of the residual ...
    min_delta: float = 10.0      # ... but never below this many grey levels.

    # --- shape / size filters ---
    min_area: int = 8            # px; below this it is sensor noise.
    max_area: int = 2000         # px; above this it is structure, not a fly clump.
    min_thickness: int = 4       # px; narrow bbox side. Kills wall seams and scratches.
    max_aspect: float = 6.0      # long bbox side / short side. Kills wall edges.

    # --- merge heuristic ---
    single_fly_area: Optional[float] = None  # None -> self-calibrate from this frame's blobs.
    merge_area_mult: float = 1.8             # area > mult * single-fly area -> is_merged.


@dataclass
class Blob:
    """One thresholded dark silhouette: a fly, or a clump of touching flies.

    Attributes:
        centroid: ``(x, y)`` sub-pixel centre of mass, in FULL-FRAME pixel coordinates.
        area: pixel count of the connected component.
        bbox: ``(x, y, w, h)`` full-frame bounding box.
        mean_intensity: mean of the ORIGINAL grey levels under the blob (low = deep silhouette).
        is_merged: `area` sits far above the single-fly mode, so this is probably several flies
            touching. A heuristic, not a measurement — see the module docstring.
        n_flies: ``round(area / single_fly_area)``, floored at 1. With `is_merged` False this is
            1 by construction.
    """

    centroid: Point
    area: int
    bbox: BBox
    mean_intensity: float
    is_merged: bool = False
    n_flies: int = 1

    @property
    def x(self) -> float:
        return self.centroid[0]

    @property
    def y(self) -> float:
        return self.centroid[1]


def estimate_single_fly_area(blobs: Sequence[Blob], window: float = 1.5) -> float:
    """Robust area of ONE fly, learned from the small-blob population.

    A vial's blob areas are a mixture: a mode of single flies plus a right tail of merged clumps
    at ~2x, ~3x, ... that size. The mean is obviously dragged up by the tail, but so is the
    plain median as soon as MOST flies are in clumps — which is exactly the crowded regime this
    rig runs in (15-25 flies per vial, DESIGN.md §10). Iteratively trimming from the median does
    not fix that either: if the median already sits on the 2-fly mode, everything within 1.5x of
    it is also a clump and the estimate never comes down.

    So the anchor comes from the BOTTOM of the distribution instead, where clumps cannot be:
      1. ``anchor = median(areas <= p25)`` — the lowest quartile. A merged blob is >= 2 flies by
         definition, so the smallest quarter of blobs is overwhelmingly singles. Taking the
         *median of* that quartile (not its minimum, and not a raw low percentile) is what keeps
         one undersized blob from dragging the anchor down.
      2. ``est = median(areas <= window * anchor)`` — widen back out to the full single-fly mode
         around that anchor and take its median.

    This holds up when clumps are the majority: for areas ``[80, 80, 81] + [160] * 8`` it
    returns 80, where a median-seeded trim returns 160.

    This is what makes the merge heuristic self-calibrating: it is measured per vial per
    experiment from the data, so it carries no assumption about this rig's magnification, and it
    tracks fly size (a vial of young small flies calibrates itself).

    Fails, honestly and unavoidably, if EVERY fly in a vial is in a clump for the whole sample —
    there is then no single-fly evidence in the data and the estimate lands on the smallest clump
    size. Pool over a whole dwell (flies separate at some point) rather than trusting one frame.

    Args:
        blobs: the blobs to learn from — typically one frame's, or pooled over a dwell (pooling
            is better: more of the population, less frame-to-frame jitter).
        window: how far above the anchor still counts as "one fly".

    Returns:
        The estimated single-fly area in px, or ``0.0`` if `blobs` is empty.
    """
    areas = np.asarray([b.area for b in blobs], dtype=np.float64)
    if areas.size == 0:
        return 0.0

    low = areas[areas <= np.percentile(areas, 25.0)]
    anchor = float(np.median(low)) if low.size else float(np.median(areas))
    if anchor <= 0:
        return float(np.median(areas))

    group = areas[areas <= window * anchor]
    return float(np.median(group)) if group.size else anchor


def _as_gray(frame: np.ndarray) -> np.ndarray:
    """Contiguous 2-D uint8 view of `frame` (accepts BGR / non-uint8 input)."""
    img = np.asarray(frame)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(img)


def _odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _robust_sigma(values: np.ndarray) -> float:
    """Noise scale of the residual, immune to BOTH tails (see the module docstring).

    ``IQR / 1.349`` equals the standard deviation for a Gaussian, but is computed from the p25
    and p75 points only — so neither the flies (positive tail) nor specular tube-wall streaks
    (negative tail) can move it.
    """
    if values.size == 0:
        return 0.0
    q1, q3 = np.percentile(values, (25.0, 75.0))
    return float(q3 - q1) / 1.349


def detect_flies(
    frame_gray: np.ndarray,
    roi_mask_bool: np.ndarray,
    params: Optional[DetectParams] = None,
) -> List[Blob]:
    """Find fly silhouettes inside one ROI. Single frame, no state.

    Method (all four steps justified in the module docstring):
      1. crop to the ROI's bounding box, plus `bg_margin` px of context so the background
         estimate at the ROI border sees real neighbouring tube instead of a replicated edge;
      2. ``residual = medianBlur(crop, bg_kernel) - crop`` — flies become positive bumps on a
         flat field, with the back-light's left-right gradient cancelled by construction;
      3. threshold at ``median(residual) + max(min_delta, k_sigma * robust_sigma(residual))``,
         with both statistics taken over ROI pixels ONLY;
      4. connected components, filtered by area and by two shape gates that reject the tube
         walls and seams (`min_thickness`, `max_aspect`).

    Finally every surviving blob is scored for merging against `estimate_single_fly_area` (or
    `params.single_fly_area`, if the caller has a better, pooled estimate — recommended, since a
    single frame's estimate is noisy).

    Args:
        frame_gray: HxW grayscale frame (full frame; BGR/non-uint8 is converted).
        roi_mask_bool: HxW boolean mask, True = inside this vial's trackable region. Typically
            ``(illum_mask == 255) & vial_bbox``, the same effective mask `activity.py` uses.
        params: `DetectParams`; defaults if None.

    Returns:
        Blobs with centroids/bboxes in FULL-FRAME coordinates, in no particular order. An empty
        ROI, an all-False mask, or a mask smaller than the filters allow returns ``[]``.

    Raises:
        ValueError: `frame_gray` and `roi_mask_bool` have different shapes.
    """
    p = params or DetectParams()
    gray = _as_gray(frame_gray)
    mask = np.asarray(roi_mask_bool)
    if mask.dtype != bool:
        mask = mask > 0
    if mask.shape != gray.shape[:2]:
        raise ValueError(
            "roi_mask_bool shape %r does not match frame shape %r" % (mask.shape, gray.shape[:2])
        )
    if not mask.any():
        return []

    # --- 1. crop to the ROI bbox (+ context margin for the background filter) ---------------
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    margin = int(p.bg_margin) if p.bg_margin else max(1, _odd(p.bg_kernel) // 2)
    H, W = gray.shape[:2]
    cy0, cy1 = max(0, y0 - margin), min(H, y1 + margin)
    cx0, cx1 = max(0, x0 - margin), min(W, x1 + margin)

    crop = np.ascontiguousarray(gray[cy0:cy1, cx0:cx1])
    roi = np.ascontiguousarray(mask[cy0:cy1, cx0:cx1])

    # Drop a rim of the ROI: the mask boundary follows the dark tube walls, and a background
    # estimate straddling that step produces a spurious ridge of "fly" along the edge.
    if p.roi_erode and p.roi_erode > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(p.roi_erode) + 1, 2 * int(p.roi_erode) + 1)
        )
        roi = cv2.erode(roi.astype(np.uint8), k).astype(bool)
    if not roi.any():
        return []

    # --- 2. local background -> residual ------------------------------------------------------
    ksz = _odd(max(3, p.bg_kernel))
    # medianBlur's window must fit the crop, else OpenCV reads outside it.
    ksz = _odd(min(ksz, min(crop.shape[0], crop.shape[1]) - 1)) if min(crop.shape) > 3 else 3
    background = cv2.medianBlur(crop, ksz)
    residual = background.astype(np.float32) - crop.astype(np.float32)

    # --- 3. relative threshold ----------------------------------------------------------------
    vals = residual[roi]
    med = float(np.median(vals))
    sigma = _robust_sigma(vals)
    delta = max(float(p.min_delta), float(p.k_sigma) * sigma)
    fly_mask = ((residual > med + delta) & roi).astype(np.uint8)

    # --- 4. connected components + size/shape filters ------------------------------------------
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fly_mask, 8)
    blobs: List[Blob] = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p.min_area or area > p.max_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        short, long = min(bw, bh), max(bw, bh)
        if short < p.min_thickness:
            continue                                    # wall seam / scratch, not a fly
        if long / float(max(1, short)) > p.max_aspect:
            continue                                    # tube-wall edge, not a fly
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        mean_intensity = float(crop[labels == i].mean())
        blobs.append(
            Blob(
                centroid=(float(centroids[i][0]) + cx0, float(centroids[i][1]) + cy0),
                area=area,
                bbox=(bx + cx0, by + cy0, bw, bh),
                mean_intensity=mean_intensity,
            )
        )

    _score_merges(blobs, p)
    return blobs


def _score_merges(blobs: List[Blob], p: DetectParams) -> None:
    """Fill in `is_merged` / `n_flies` in place (see `estimate_single_fly_area`).

    `n_flies` is GATED on `is_merged` so the two can never contradict each other: a blob we do
    not believe is a merge is one fly, full stop. (Ungated, area/single rounds to 2 from 1.5x
    upward while `is_merged` only trips at `merge_area_mult`, leaving a band where a blob claimed
    to be 2 flies and not-merged at the same time.)
    """
    if not blobs:
        return
    single = p.single_fly_area if p.single_fly_area else estimate_single_fly_area(blobs)
    if not single or single <= 0:
        return
    for b in blobs:
        b.is_merged = bool(b.area > p.merge_area_mult * single)
        b.n_flies = max(1, int(round(b.area / single))) if b.is_merged else 1


# ======================================================================================
# Frame-to-frame linking
# ======================================================================================
def link_blobs(
    prev: Sequence[Blob],
    cur: Sequence[Blob],
    max_dist: float,
) -> List[Tuple[int, int]]:
    """Match blobs between consecutive frames: greedy nearest-neighbour with a distance gate.

    Every ``prev x cur`` pair within `max_dist` is scored by centroid distance, sorted, and taken
    greedily — each blob is used at most once. Unmatched `cur` blobs begin new tracks; unmatched
    `prev` blobs end theirs.

    IDENTITY SWAPS ARE EXPECTED HERE. With 15-25 flies in one tube (DESIGN.md §10) two flies
    routinely pass within `max_dist` of each other, and nearest-neighbour will then happily swap
    them; a fly that walks into a clump disappears into one merged blob and re-emerges as a NEW
    id. Nothing downstream should treat a track as one animal's biography. What linking IS good
    for is a short-horizon DISPLACEMENT estimate — over one frame interval the assignment is
    usually right, so speeds pooled over many fragments are meaningful even though no single
    fragment is trustworthy. `VialTracker` reports `mean_fragment_frames` so the caller can see
    how badly fragmented a given run was.

    Greedy is used over the optimal (Hungarian) assignment deliberately: it is O(n^2 log n) with
    no SciPy dependency, and at this crowding the assignment error is dominated by genuinely
    ambiguous crossings that an optimal matcher gets wrong too.

    Args:
        prev: blobs from the previous frame.
        cur: blobs from the current frame.
        max_dist: gate in px. Pairs farther apart are never matched — set it to a bit more than
            the farthest a fly can travel in one frame interval.

    Returns:
        ``[(i, j), ...]`` with `i` indexing `prev` and `j` indexing `cur`, sorted by `i`.
    """
    if not prev or not cur or max_dist <= 0:
        return []

    pairs: List[Tuple[float, int, int]] = []
    for i, a in enumerate(prev):
        ax, ay = a.centroid
        for j, b in enumerate(cur):
            bx, by = b.centroid
            d = math.hypot(ax - bx, ay - by)
            if d <= max_dist:
                pairs.append((d, i, j))
    pairs.sort()

    used_prev: set = set()
    used_cur: set = set()
    matches: List[Tuple[int, int]] = []
    for _d, i, j in pairs:
        if i in used_prev or j in used_cur:
            continue
        used_prev.add(i)
        used_cur.add(j)
        matches.append((i, j))
    matches.sort()
    return matches


# ======================================================================================
# Vial axis ("climbing" direction)
# ======================================================================================
def _resolve_axis(axis, roi_mask_bool: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalise the `axis` argument into ``(origin, unit_direction, length)``.

    Two accepted forms, both meaning "0 = bottom / food end, 1 = top":
      * TWO ENDPOINTS ``((x0, y0), (x1, y1))`` — explicit and preferred. `origin` is the bottom
        point, `length` the distance between them.
      * A DIRECTION VECTOR ``(dx, dy)`` — the extent is then measured from the ROI itself: every
        True pixel is projected onto the direction and the min/max define 0 and 1. Convenient
        (the vial box already delimits the tube), and it makes the height self-normalising per
        vial, but it means the scale depends on the mask, so two vials with differently sized
        boxes are only comparable in normalized units, not px.

    Raises:
        ValueError: the axis has zero length, or is not one of the two accepted shapes.
    """
    arr = np.asarray(axis, dtype=np.float64)

    if arr.shape == (2, 2):                       # two endpoints
        origin = arr[0]
        vec = arr[1] - arr[0]
        length = float(np.hypot(vec[0], vec[1]))
        if length <= 0:
            raise ValueError("axis endpoints are identical; cannot define a climbing direction")
        return origin, vec / length, length

    if arr.shape == (2,):                         # direction vector, extent from the ROI
        length_v = float(np.hypot(arr[0], arr[1]))
        if length_v <= 0:
            raise ValueError("axis direction vector is zero-length")
        direction = arr / length_v
        ys, xs = np.nonzero(np.asarray(roi_mask_bool))
        if xs.size == 0:
            raise ValueError("cannot derive the axis extent from an empty ROI mask")
        proj = xs * direction[0] + ys * direction[1]
        lo, hi = float(proj.min()), float(proj.max())
        length = hi - lo
        if length <= 0:
            raise ValueError("ROI has zero extent along the given axis direction")
        origin = direction * lo
        return origin, direction, length

    raise ValueError(
        "axis must be ((x0, y0), (x1, y1)) endpoints or (dx, dy) a direction vector; got shape %r"
        % (arr.shape,)
    )


def project_heights(
    blobs: Sequence[Blob],
    axis,
    roi_mask_bool: np.ndarray,
) -> List[float]:
    """Normalized height in [0, 1] for each blob centroid (0 = bottom/food end, 1 = top).

    Values are clipped into [0, 1]: a centroid can land marginally outside the axis extent (the
    axis is a straight line, the tube is a 3-D object seen in projection), and a height of 1.03
    is noise, not a fly above the tube.
    """
    origin, direction, length = _resolve_axis(axis, roi_mask_bool)
    out: List[float] = []
    for b in blobs:
        cx, cy = b.centroid
        h = ((cx - origin[0]) * direction[0] + (cy - origin[1]) * direction[1]) / length
        out.append(float(min(1.0, max(0.0, h))))
    return out


# ======================================================================================
# Per-frame results + per-dwell tracker
# ======================================================================================
@dataclass
class FrameStats:
    """One frame's population readout for one vial.

    `heights` is the raw distribution — keep it if you want to plot or re-bin; everything else
    here is a summary of it.
    """

    t: float                                   # seconds (the tracker's clock)
    index: int                                 # 0-based frame counter within this dwell
    n_blobs: int
    est_n_flies: int                           # sum of per-blob `n_flies` (merge-corrected)
    heights: List[float] = field(default_factory=list)
    total_blob_area: int = 0

    @property
    def mean_height(self) -> float:
        return float(np.mean(self.heights)) if self.heights else float("nan")

    @property
    def median_height(self) -> float:
        return float(np.median(self.heights)) if self.heights else float("nan")

    def frac_above(self, h: float) -> float:
        """Fraction of blobs whose normalized height exceeds `h` (the classic climbing score).

        Returns ``nan`` for a frame with no blobs, so an empty frame is distinguishable from a
        frame where every fly is at the bottom (which is ``0.0``).
        """
        if not self.heights:
            return float("nan")
        return float(np.mean(np.asarray(self.heights) > h))


@dataclass
class Track:
    """A track FRAGMENT: one blob followed while nearest-neighbour linking held it.

    Not one fly's trajectory — see `link_blobs`. `positions` are full-frame px centroids,
    `heights` the matching normalized heights, `times` the tracker clock.
    """

    id: int
    positions: List[Point] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.positions)

    @property
    def duration_s(self) -> float:
        return (self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0

    @property
    def path_length(self) -> float:
        """Summed frame-to-frame displacement, px."""
        total = 0.0
        for (x0, y0), (x1, y1) in zip(self.positions, self.positions[1:]):
            total += math.hypot(x1 - x0, y1 - y0)
        return total

    @property
    def path_length_norm(self) -> float:
        """Summed |change in normalized height| — climbing distance, axis units."""
        return float(sum(abs(b - a) for a, b in zip(self.heights, self.heights[1:])))

    @property
    def speed(self) -> float:
        """Mean speed over the fragment, px/s (``nan`` for a 1-frame fragment)."""
        d = self.duration_s
        return self.path_length / d if d > 0 else float("nan")

    @property
    def speed_norm(self) -> float:
        """Mean speed along the vial axis, normalized-height per second."""
        d = self.duration_s
        return self.path_length_norm / d if d > 0 else float("nan")


class VialTracker:
    """Per-vial state across the frames of ONE dwell.

    Scope is deliberately one dwell (one stationary period of the drum, DESIGN.md §5.1). Across
    a rotation the flies are shaken, the pose changes and every identity is lost, so linking
    across that boundary would be fiction — build a new `VialTracker` per vial per dwell.

    Typical use::

        tr = VialTracker(fps=20.0)
        for frame in dwell_frames:
            tr.update(frame, vial_mask, axis=((x_bottom, y_bottom), (x_top, y_top)))
        row = tr.summarize()          # plain dict, joins onto an ActivityRecord row

    Args:
        params: `DetectParams` for `detect_flies`.
        fps: frame rate, used to derive timestamps when `update` is not given one. Also sets the
            default link gate.
        max_link_dist: linking gate in px (see `link_blobs`). Defaults to `default_link_dist`,
            a fly-speed-based estimate.
        single_fly_area: pin the single-fly area instead of re-estimating it per frame.
            Recommended once known — see `estimate_single_fly_area`.
    """

    #: Fastest a walking/climbing fly is assumed to travel, px/s, for the default link gate.
    #: ~15 mm/s at the rig's ~7 px/mm; generous, since over-gating only costs a swap that
    #: crowding would have caused anyway.
    DEFAULT_MAX_SPEED_PX_S = 120.0

    def __init__(
        self,
        params: Optional[DetectParams] = None,
        fps: float = 20.0,
        max_link_dist: Optional[float] = None,
        single_fly_area: Optional[float] = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.params = params or DetectParams()
        self.fps = float(fps)
        self.max_link_dist = (
            float(max_link_dist) if max_link_dist is not None else self.default_link_dist(fps)
        )
        if single_fly_area is not None:
            self.params = _replace_single_fly_area(self.params, single_fly_area)

        self.frames: List[FrameStats] = []
        self.tracks: List[Track] = []
        self._prev_blobs: List[Blob] = []
        self._prev_track_ids: List[int] = []
        self._next_track_id = 0
        self._n_updates = 0

    @classmethod
    def default_link_dist(cls, fps: float) -> float:
        """Link gate in px for a given frame rate: how far a fly can move between frames."""
        return cls.DEFAULT_MAX_SPEED_PX_S / float(fps)

    # -- main entry point ------------------------------------------------------------------
    def update(
        self,
        frame_gray: np.ndarray,
        roi_mask,
        axis,
        t: Optional[float] = None,
    ) -> FrameStats:
        """Process one frame: detect -> project onto the vial axis -> link to the previous frame.

        Args:
            frame_gray: HxW grayscale frame.
            roi_mask: HxW boolean (or 0/255) mask of this vial's trackable region.
            axis: the vial's long ("climbing") direction, as either ``((x0, y0), (x1, y1))``
                endpoints with the FIRST point at the bottom/food end, or a ``(dx, dy)``
                direction vector (extent then taken from `roi_mask`). See `_resolve_axis`.
            t: timestamp in seconds. Defaults to ``frame_index / fps``.

        Returns:
            This frame's `FrameStats` (also appended to `self.frames`).
        """
        mask = np.asarray(roi_mask)
        if mask.dtype != bool:
            mask = mask > 0
        if t is None:
            t = self._n_updates / self.fps

        blobs = detect_flies(frame_gray, mask, self.params)
        heights = project_heights(blobs, axis, mask) if blobs else []

        stats = FrameStats(
            t=float(t),
            index=self._n_updates,
            n_blobs=len(blobs),
            est_n_flies=int(sum(b.n_flies for b in blobs)),
            heights=heights,
            total_blob_area=int(sum(b.area for b in blobs)),
        )
        self.frames.append(stats)

        self._link(blobs, heights, float(t))
        self._n_updates += 1
        return stats

    def _link(self, blobs: List[Blob], heights: List[float], t: float) -> None:
        """Extend / start / end track fragments from this frame's blobs."""
        matches = link_blobs(self._prev_blobs, blobs, self.max_link_dist)
        prev_to_cur = dict(matches)

        cur_track_ids: List[Optional[int]] = [None] * len(blobs)
        for prev_i, cur_j in prev_to_cur.items():
            cur_track_ids[cur_j] = self._prev_track_ids[prev_i]

        for j, blob in enumerate(blobs):
            tid = cur_track_ids[j]
            if tid is None:                       # unmatched -> a new fragment starts here
                tid = self._next_track_id
                self._next_track_id += 1
                self.tracks.append(Track(id=tid))
                cur_track_ids[j] = tid
            track = self._track_by_id(tid)
            track.positions.append(blob.centroid)
            track.heights.append(heights[j] if j < len(heights) else float("nan"))
            track.times.append(t)

        self._prev_blobs = blobs
        self._prev_track_ids = [int(tid) for tid in cur_track_ids]

    def _track_by_id(self, tid: int) -> Track:
        # Fragments are short and appended in id order, so scanning from the end is O(1) in
        # practice; a dict would be premature for the tens-of-tracks scale here.
        for track in reversed(self.tracks):
            if track.id == tid:
                return track
        raise KeyError("unknown track id %r" % tid)

    # -- aggregate readouts ------------------------------------------------------------------
    @property
    def mean_fragment_frames(self) -> float:
        """Mean track-fragment length in frames — the trustworthiness diagnostic.

        Tens of frames: linking is holding, per-track numbers mean something. Near 1-3: tracks
        are shredded by crowding/merging and ONLY the pooled statistics should be quoted.
        """
        if not self.tracks:
            return float("nan")
        return float(np.mean([t.n_frames for t in self.tracks]))

    def summarize(self, mid_height: float = 0.5) -> dict:
        """This vial's behavioural summary for the frames seen so far. See module `summarize`."""
        return summarize(self.frames, self.tracks, mid_height=mid_height)


def _replace_single_fly_area(p: DetectParams, area: float) -> DetectParams:
    q = DetectParams(**{k: getattr(p, k) for k in p.__dataclass_fields__})
    q.single_fly_area = float(area)
    return q


# ======================================================================================
# Summary
# ======================================================================================
def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)  # drop NaN
    return float(arr.mean()) if arr.size else float("nan")


def summarize(
    frames: Sequence[FrameStats],
    tracks: Sequence[Track],
    mid_height: float = 0.5,
) -> dict:
    """Per-vial-per-bin behavioural summary, as a plain dict.

    Deliberately a flat ``dict`` of scalars and not a dataclass, so it can be dropped straight
    into a DataFrame and joined onto the existing `types.ActivityRecord` table (by `run_id`,
    bin and `vial_id`) without either module having to know about the other.

    Keys, and how far to trust each (module docstring has the long version):

    Height / climbing — THE ROBUST PART, no identity needed:
        ``mean_height``      mean normalized height over all blobs in all frames (0 bottom, 1 top)
        ``median_height``    median of the same pooled distribution
        ``frac_above_mid``   fraction of blob observations above `mid_height` — the classic
                             negative-geotaxis climbing score
        ``max_height``       highest single blob observed in the bin

    Counts — robust as a RELATIVE measure between vials, approximate in absolute terms:
        ``n_blobs_mean``     mean connected components per frame (under-counts when flies touch)
        ``est_n_flies_mean`` mean merge-corrected fly count per frame

    Speed / path — pooled over fragments, so meaningful in aggregate only:
        ``mean_speed``       mean fragment speed, px/s
        ``median_speed``     median fragment speed, px/s (prefer this; the mean has a tail)
        ``p90_speed``        90th percentile fragment speed, px/s
        ``mean_speed_norm``  mean fragment speed in normalized-height units/s
        ``total_path_length`` summed displacement over all fragments, px

    Diagnostics — read these BEFORE quoting anything above:
        ``mean_fragment_frames`` mean fragment length in frames. Low = tracks are shredded.
        ``n_tracks``         number of fragments started (>> the number of flies when crowded)
        ``n_frames``         frames folded into this summary

    Fields are ``nan`` (not 0) when undefined — an empty vial has no height, which is not the
    same claim as "its flies are at the bottom".
    """
    all_heights: List[float] = []
    for f in frames:
        all_heights.extend(f.heights)
    heights_arr = np.asarray(all_heights, dtype=np.float64)

    # Fragments of a single frame have no displacement and no duration, so they carry no speed;
    # including them as 0.0 would bias every speed statistic toward zero exactly when crowding
    # is worst. They still count in `n_tracks` / `mean_fragment_frames`, which is where the
    # caller is meant to see that the tracking fell apart.
    speeds = [t.speed for t in tracks if t.n_frames > 1 and t.duration_s > 0]
    speeds_norm = [t.speed_norm for t in tracks if t.n_frames > 1 and t.duration_s > 0]
    speeds_arr = np.asarray([s for s in speeds if s == s], dtype=np.float64)

    return {
        # height / climbing
        "mean_height": float(heights_arr.mean()) if heights_arr.size else float("nan"),
        "median_height": float(np.median(heights_arr)) if heights_arr.size else float("nan"),
        "frac_above_mid": (
            float(np.mean(heights_arr > mid_height)) if heights_arr.size else float("nan")
        ),
        "max_height": float(heights_arr.max()) if heights_arr.size else float("nan"),
        # counts
        "n_blobs_mean": _nanmean([f.n_blobs for f in frames]),
        "est_n_flies_mean": _nanmean([f.est_n_flies for f in frames]),
        # speed / path
        "mean_speed": float(speeds_arr.mean()) if speeds_arr.size else float("nan"),
        "median_speed": float(np.median(speeds_arr)) if speeds_arr.size else float("nan"),
        "p90_speed": float(np.percentile(speeds_arr, 90)) if speeds_arr.size else float("nan"),
        "mean_speed_norm": _nanmean(speeds_norm),
        "total_path_length": float(sum(t.path_length for t in tracks)),
        # diagnostics
        "mean_fragment_frames": (
            float(np.mean([t.n_frames for t in tracks])) if tracks else float("nan")
        ),
        "n_tracks": int(len(tracks)),
        "n_frames": int(len(frames)),
    }

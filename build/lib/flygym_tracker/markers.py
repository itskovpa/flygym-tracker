"""Marker-based face identification for the FlyGym v2 drum rig (DESIGN.md §5.2, `markers` row §4).

FRAMEWORK, NOT A FINISHED DECODER
----------------------------------
Physical markers do not exist yet (DESIGN.md §9 explicitly defers "finalize marker decode once
physical markers are added"). This module implements only the parts that do NOT depend on knowing
what the real marker looks like:

  - geometric detection of a dark silhouette blob within a `search_region` (real, exercised by
    `tests/test_markers.py` against synthetic shapes), and
  - a shape-signature + nearest-registered-signature matching scheme that is provably sensitive to
    the ~180 degree pose change between Face A and Face B (see `_signature`'s docstring for the
    moment-theory argument, and the module-level note below on why BOTH a true rotation and a
    mirror flip are covered) -- this is the one property DESIGN.md actually requires, since the
    two faces are the drum's two rotational positions (§2: "flipping ~180 degrees"; also noted as
    appearing "mirror/flip" in image space, since a 3-D flip can project to either a 2-D rotation
    or a 2-D mirror depending on the (not-yet-fixed) drum axis -- see below).

What is NOT yet real: `MarkerDetector`'s `min_area`/`max_area`, and `MarkerParams`'s
`dark_frac`/`bright_percentile`/`max_aspect_ratio`/`max_match_distance`, are placeholder numbers
sized for the synthetic shapes in this module's own test suite -- NOT for whatever physical marker
eventually gets built and photographed through the real back-lit optics. In particular
`max_match_distance` defaults to `None` (no rejection-by-distance; nearest registered signature
always wins once a candidate is found) because a meaningful distance can only be chosen from real
signature separation, which does not exist yet -- see "To finalize" below.

Why the signature uses BOTH Hu moments AND raw 3rd-order skew terms
---------------------------------------------------------------------
DESIGN.md §2 says Face B "appears geometrically transformed (mirror/flip) vs Face A" without
committing to which: whether the on-screen transform between the two faces is a true 180 degree
rotation (point reflection) or a mirror flip (reflection across one axis) depends on which physical
axis the drum rotates about relative to the camera -- a rig detail not yet pinned down. The two
cases stress different parts of a shape descriptor:

  - A pure 180 degree rotation leaves all 7 Hu moments UNCHANGED (Hu moments are constructed to be
    invariant to rotation by any angle -- that is their whole point) but flips the SIGN of every
    3rd-order (odd total degree) normalized central moment.
  - A pure mirror flip leaves Hu moments I1-I6 unchanged but flips the sign of I7 (the classical
    "skew/reflection" invariant), and flips only TWO of the four 3rd-order terms (the two
    "aligned" with the mirrored axis), not all four.

(Verified numerically against a synthetic asymmetric triangle while building this module: under an
exact 180 degree point reflection, Hu-moment difference was ~1e-13 [float noise] while all four
3rd-order terms flipped sign exactly; under a single-axis mirror, Hu's 7th component flipped by
~11.9 [log-scaled] while only 2 of 4 skew terms flipped.)

So relying on Hu moments alone would silently fail for a pure-rotation rig, and relying on the
3rd-order terms alone would under-use the (much stronger) signal a mirror-flip rig actually
provides. `_signature` concatenates both, which is correct for either physical case without having
to know which one applies.

To finalize once a physical marker exists
-------------------------------------------
1. Build the asymmetric marker (DESIGN.md §2: ONE IR-opaque fiducial per face, mounted on the rigid
   frame in view) and capture a still of each face through the real back-lit optics.
2. Re-tune `min_area`/`max_area`/`MarkerParams.max_aspect_ratio` to the marker's real pixel
   footprint at the camera's working distance (see `docs/frame_full.png`-scale reference,
   DESIGN.md §2), and re-check `MarkerParams.dark_frac`/`bright_percentile` against the real
   back-lit contrast.
3. Call `register_marker(frame_gray, "A")` / `register_marker(frame_gray, "B")` on those two stills
   to seed the registry from real signatures.
4. Measure the real signature distance between the two registered faces (and, ideally, repeat
   photos of the *same* face to measure same-class noise) and set `MarkerParams.max_match_distance`
   between those two numbers -- see `identify_face`/`match_signature`.
5. Persist the tuned registry (`to_dict()`) into each face's calibration record
   (`types.FaceCalibration.marker`) alongside the rest of the calibration bundle (wiring this into
   `calibration.py` is future work, out of this module's scope).

Everything else here (connected-component search within `search_region`, area/aspect filtering,
the Hu+skew signature, nearest-neighbour matching, the `enabled=False` bypass) is intended to be
final as implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# (x, y, w, h) pixel box -- same convention as calibration.py's Box.
Box = Tuple[int, int, int, int]


# --------------------------------------------------------------------------------------
# Tunable parameters (placeholders sized for this module's synthetic tests -- see module
# docstring's "To finalize" section).
# --------------------------------------------------------------------------------------
@dataclass
class MarkerParams:
    """Knobs for dark-silhouette segmentation and signature matching."""

    # --- dark-silhouette segmentation (opaque marker on a bright back-lit ground) ---
    bright_percentile: float = 90.0   # percentile of search_region used as the "bright ground" ref
    dark_frac: float = 0.5            # a pixel is "marker" if it is < dark_frac * bright_ref
    morph_open: int = 3               # opening kernel (px); removes speckle noise. 0 disables.
    morph_close: int = 3              # closing kernel (px); fills pinholes in the silhouette. 0 disables.

    # --- candidate shape filter (besides the detector's own min_area/max_area) ---
    max_aspect_ratio: float = 6.0     # reject a candidate whose bbox long/short side ratio exceeds this
    #                                   (filters thin slivers/noise streaks; a real marker's own aspect
    #                                   ratio must stay under this once known).

    # --- signature matching ---
    #: reject a match whose distance to the nearest registered signature exceeds this. `None`
    #: (default) never rejects by distance -- see module docstring, "To finalize" step 4.
    max_match_distance: Optional[float] = None


# --------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------
def _as_gray2d(frame: np.ndarray) -> np.ndarray:
    """Return a 2-D uint8 view of `frame` (defensive; frames are nominally already this)."""
    if frame is None:
        raise ValueError("frame is None")
    img = np.asarray(frame)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _clip_region(region: Box, width: int, height: int) -> Box:
    """Clip an (x, y, w, h) region to the [0, width) x [0, height) frame bounds."""
    x, y, w, h = region
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _dark_mask(roi_gray: np.ndarray, p: MarkerParams) -> np.ndarray:
    """Threshold DARK pixels (opaque marker silhouette on a bright back-lit ground). 255 = dark.

    The bright "ground" reference is a high percentile of the ROI (default: 90th), not a fixed
    constant, so this tracks whatever back-light level is actually present -- consistent with how
    `calibration.py` derives its own lit-level references from the frame rather than hard-coding
    them.
    """
    bright_ref = float(np.percentile(roi_gray, p.bright_percentile))
    thr = p.dark_frac * bright_ref
    mask = (roi_gray < thr).astype(np.uint8) * 255
    if p.morph_open and p.morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_open, p.morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if p.morph_close and p.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_close, p.morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


@dataclass
class MarkerCandidate:
    """One detected dark blob within the search region, in search-region-local pixel coords."""

    contour: np.ndarray
    bbox: Box          # (x, y, w, h), local to the search region (NOT full-frame coords)
    area: float


def _find_candidate(
    roi_gray: np.ndarray, min_area: float, max_area: float, p: MarkerParams
) -> Optional[MarkerCandidate]:
    """Largest dark connected component in `roi_gray` passing the area/aspect filters, else None."""
    mask = _dark_mask(roi_gray, p)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best: Optional[MarkerCandidate] = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        long_side, short_side = max(w, h), min(w, h)
        if short_side == 0 or (long_side / short_side) > p.max_aspect_ratio:
            continue
        if best is None or area > best.area:
            best = MarkerCandidate(contour=c, bbox=(x, y, w, h), area=area)
    return best


# --------------------------------------------------------------------------------------
# Signature
# --------------------------------------------------------------------------------------
#: signature layout: 7 log-scaled Hu moments, then 4 raw 3rd-order normalized central moments.
SIGNATURE_LENGTH = 11


def _signature(contour: np.ndarray) -> np.ndarray:
    """Shape signature for one marker silhouette contour -- an 11-element float vector.

    `sig[0:7]`: log-scaled Hu invariant moments (`cv2.HuMoments`). Hu moments are built to be
    invariant to translation, scale, AND rotation, so they capture "what general shape is this"
    independent of exactly how the marker is mounted/oriented -- e.g. robust to whatever small
    angular jitter the drum's mechanical stop leaves face to face. Raw Hu moments span many orders
    of magnitude, so each is log-scaled (`-sign(h) * log10(|h|)`, the standard transform) to keep
    them comparable/summable in a Euclidean distance.

    `sig[7:11]`: the four 3rd-order normalized central moments (`nu30, nu21, nu12, nu03` from
    `cv2.moments`), unscaled. These are ALSO translation- and scale-invariant, but -- unlike Hu
    moments -- are not invariant to rotation. A 180 degree rotation is a point reflection through
    the shape's centroid: in centroid-relative coordinates every point (x, y) maps to (-x, -y), and
    every 3rd-order (odd total degree) central moment is an odd function of (x, y), so ALL FOUR
    flip sign under that reflection, regardless of the shape's absolute orientation. A single-axis
    mirror flip (x, y) -> (-x, y) flips only the two terms with odd x-degree (nu30, nu12) and
    leaves the other two unchanged -- still a nonzero difference for any 2-D-asymmetric shape. Hu
    moments, by contrast, are IDENTICAL before and after a 180 degree rotation (that is exactly
    what "rotation invariant" means) and unchanged in 6 of their 7 components under a mirror -- so
    Hu moments alone cannot reliably tell Face A's marker from Face B's marker when B is A rotated
    ~180 degrees, which is exactly the situation DESIGN.md §2 describes. See the module docstring
    for why both halves of this vector are kept (the physical rig could present either a rotation
    or a mirror between faces) and the numeric check that motivated this design.

    A shape with zero skew in all four 3rd-order terms (i.e. exactly point-symmetric) would be
    genuinely ambiguous under a pure 180 degree rotation -- which is exactly why DESIGN.md calls
    for an ASYMMETRIC marker (§2: "ONE IR-opaque asymmetric fiducial per face"). This function
    trusts the caller to supply an asymmetric marker; it does not itself detect/reject symmetric
    ones.

    `contour` must have nonzero area (callers filter for that via `min_area` before reaching here).
    """
    m = cv2.moments(contour)
    hu = cv2.HuMoments(m).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    skew = np.array([m["nu30"], m["nu21"], m["nu12"], m["nu03"]], dtype=np.float64)
    return np.concatenate([hu_log, skew]).astype(np.float64)


#: signature -> face-name matcher. Swappable via `MarkerDetector.match_fn` (see class docstring).
MatchFn = Callable[[np.ndarray, Dict[str, np.ndarray], Optional[float]], Optional[str]]


def match_signature(
    sig: np.ndarray, registry: Dict[str, np.ndarray], max_distance: Optional[float]
) -> Optional[str]:
    """Nearest-registered-signature match: the default (and simplest possible) `MatchFn`.

    Returns the face name whose registered signature is closest to `sig` (Euclidean distance), or
    None if the registry is empty or (when `max_distance` is given) the best match is farther than
    that.
    """
    if not registry:
        return None
    best_face: Optional[str] = None
    best_dist: Optional[float] = None
    for face, ref in registry.items():
        d = float(np.linalg.norm(sig - ref))
        if best_dist is None or d < best_dist:
            best_dist = d
            best_face = face
    if max_distance is not None and best_dist is not None and best_dist > max_distance:
        return None
    return best_face


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
class MarkerDetector:
    """Detects an IR-opaque fiducial marker and maps it to a face name (DESIGN.md §5.2).

    Call `identify_face(frame_gray)` once per stationary onset (per DESIGN.md §5.2). While
    `enabled` is False (the current state -- physical markers don't exist yet, DESIGN.md §9) it
    always returns None immediately, and the caller (the future `pipeline.py`) defaults to face
    "A" and logs `marker_absent`, per DESIGN.md §5.2. This makes turning marker support on purely a
    config flip (`config/default_config.yaml`'s `markers.enabled`) once real markers exist, with no
    caller-side code change.

    `register_marker(frame_gray, face_name)` is calibration-time teaching, independent of
    `enabled` -- learning a marker's signature from a labelled still is exactly how a registry gets
    built in the first place (DESIGN.md §5.5's calibration flow), before `enabled` would ever be
    flipped on for a run.

    Detection (segmentation + candidate filtering) is scoped to `search_region` (x, y, w, h) if
    given, else the whole frame -- DESIGN.md's example is "the rigid frame area" a marker would be
    mounted on, so callers should pass that ROI to avoid false positives elsewhere in the scene
    (e.g. vial mouths, which are also dark-ish silhouette features).

    Matching is pluggable: `self.match_fn` defaults to `match_signature` (nearest registered
    signature by Euclidean distance) but can be swapped per instance (`detector.match_fn = ...`)
    without subclassing, e.g. to try a different distance metric once real marker signatures are
    available.
    """

    def __init__(
        self,
        enabled: bool = False,
        search_region: Optional[Box] = None,
        min_area: float = 200.0,
        max_area: float = 50000.0,
        registry: Optional[Dict[str, Sequence[float]]] = None,
        params: Optional[MarkerParams] = None,
    ) -> None:
        self.enabled = enabled
        self.search_region = search_region
        self.min_area = min_area
        self.max_area = max_area
        self.params = params or MarkerParams()
        #: face_name -> signature vector (np.ndarray, shape (SIGNATURE_LENGTH,)).
        self.registry: Dict[str, np.ndarray] = {
            face: np.asarray(sig, dtype=np.float64) for face, sig in (registry or {}).items()
        }
        #: pluggable signature -> face-name matcher (see class docstring).
        self.match_fn: MatchFn = match_signature

    def _roi(self, frame_gray: np.ndarray) -> np.ndarray:
        """Crop to `search_region`, clipped to the frame bounds (whole frame if None)."""
        gray = _as_gray2d(frame_gray)
        h_img, w_img = gray.shape[:2]
        if self.search_region is None:
            return gray
        x, y, w, h = _clip_region(self.search_region, w_img, h_img)
        return gray[y:y + h, x:x + w]

    def _detect_candidate(self, frame_gray: np.ndarray) -> Optional[MarkerCandidate]:
        roi = self._roi(frame_gray)
        if roi.size == 0:
            return None
        return _find_candidate(roi, self.min_area, self.max_area, self.params)

    def identify_face(self, frame_gray: np.ndarray) -> Optional[str]:
        """Detect the marker and return the matching face name, or None.

        Returns None when:
          - `enabled` is False (no physical marker to look for yet -- caller defaults to "A"), or
          - no dark-silhouette candidate passes the area/aspect filters in `search_region`, or
          - a candidate was found but nothing in `registry` matches it closely enough (see
            `match_fn` / `MarkerParams.max_match_distance`).
        """
        if not self.enabled:
            return None
        candidate = self._detect_candidate(frame_gray)
        if candidate is None:
            return None
        sig = _signature(candidate.contour)
        return self.match_fn(sig, self.registry, self.params.max_match_distance)

    def can_identify(self) -> bool:
        """True if `identify_face` could ever return a face: enabled, with two faces to tell apart.

        `pipeline.TrackerPipeline` asks this at startup to decide whether face identification is
        something it should WAIT for. A disabled or empty detector must not put the run into
        "face unknown" forever; it has to be reported instead (`cli.face_id_readiness`).
        """
        return bool(self.enabled) and len(self.registry) >= 2

    def register_marker(self, frame_gray: np.ndarray, face_name: str) -> None:
        """Learn/store the current frame's marker signature under `face_name`.

        Re-registering an already-known `face_name` overwrites its stored signature. Runs
        regardless of `enabled` (see class docstring). Raises ValueError if no marker candidate is
        found in `search_region` -- there is nothing to learn from that frame.
        """
        candidate = self._detect_candidate(frame_gray)
        if candidate is None:
            raise ValueError(
                "no marker candidate found in search_region; cannot register a signature for %r"
                % (face_name,)
            )
        self.registry[face_name] = _signature(candidate.contour)

    def to_dict(self) -> Dict[str, List[float]]:
        """Serialize the registry to plain JSON-safe types: `{face_name: [float, ...]}`.

        This is the shape to nest into `types.FaceCalibration.marker` once a real calibration flow
        exists (that wiring is out of this module's scope) -- e.g. `face_calib.marker =
        {"signature": detector.to_dict()[face_name]}` for that specific face; this method
        intentionally returns the whole registry (every learned face at once), since one
        `MarkerDetector` / registry is shared across both faces, not owned per-face.
        """
        return {face: sig.tolist() for face, sig in self.registry.items()}

    @staticmethod
    def from_dict(registry: Dict[str, Sequence[float]], **kwargs) -> "MarkerDetector":
        """Build a detector whose registry is restored from `to_dict()` output.

        Any other constructor argument (`enabled`, `search_region`, `min_area`, `max_area`,
        `params`) may be passed through via `kwargs`.
        """
        return MarkerDetector(registry=dict(registry), **kwargs)

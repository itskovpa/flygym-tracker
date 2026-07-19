"""Marker-band face identification + vial-boundary decoding for the FlyGym v2 drum rig.

WHAT THIS IS (and how it differs from `markers.py`)
---------------------------------------------------
`markers.py` holds the GENERIC `MarkerDetector`: a blob-shape/Hu-moment scheme written before any
physical marker existed, deliberately agnostic about what the fiducial would look like. This module
is the RIG-SPECIFIC decoder for the markers the rig owner actually built. Both are kept: the generic
one stays the fallback for any future single-fiducial scheme; this one is what the FlyGym v2 drum
uses today. `MarkerBandDetector` is duck-type compatible with `MarkerDetector` where it matters --
it exposes `identify_face(frame_gray) -> str | None`, which is the whole of the contract
`pipeline.TrackerPipeline` relies on (see `TrackerPipeline._handle_onset`) -- so it drops straight
into the `marker_detector=` slot.

THE PHYSICAL MARKER SCHEME
--------------------------
The drum's central rotation axis carries two horizontal LED slots (DESIGN.md §2: "the two
blinding-bright horizontal slots in the middle are LED shining through frame hardware"). The rig
owner stuck **opaque IR stickers** along that axis. Each sticker spans exactly ONE vial's width, and
they alternate **up, down, up, down, ...** across the 8 vial columns. Because the stickers block the
850 nm back-light, each one punches a **dark block** into one of the two bright strips:

      col:      1     2     3     4     5     6     7     8
      upper:  [###]  ....  [###]  ....  [###]  ....  [###]  ....      [###] = sticker (dark)
      lower:   ....  [###]  ....  [###]  ....  [###]  ....  [###]     ....  = lit (bright)

so the two strips carry **complementary, interleaved** bright runs -- a bright run in one strip is
bounded left and right by the stickers of its own strip, and therefore spans exactly one vial.

WHY THE FACES ARE DISTINGUISHABLE (the key insight, verified on real data)
--------------------------------------------------------------------------
The drum flips ~180 degrees to swap Face A / Face B (DESIGN.md §2). A 180 degree flip about the
horizontal rotation axis maps up -> down, so **the same physical sticker row images as the SAME
pattern with the UPPER and LOWER strip profiles SWAPPED**. Measured on real dwell frames from
`Good Markers.avi` (bright-run transition x-positions):

    FaceX upper: [191, 362, 471, 607, 762, 934, 1031, 1086]  ~=  FaceY lower: [179, 354, 466, 602, 766, 904, 1044, 1076]
    FaceX lower: [147, 222, 322, 507, 664, 795,  900, 1061]  ~=  FaceY upper: [151, 213, 312, 507, 664, 804,  914, 1080]

So the face is identified by **which profile is on top**, not by any left/right or shape difference.
Two consequences this module exploits. All figures below are lag-searched NCC over the 903 dwell
pairs of `Good Markers.avi` (43 dwells, 20 Face A / 23 Face B):

1. Comparing the current upper profile to each registered face's upper template already separates
   the faces cleanly: same-face upper-to-upper correlation is 0.988..1.000, different-face is
   -0.258..-0.167. It separates so hard because a face's own upper and lower profiles are the two
   INTERLEAVED HALVES of one alternating pattern, i.e. actively anti-correlated (own upper vs own
   lower, same frame: -0.272..-0.224), not merely unrelated.
2. The swap gives a free second opinion: face F's upper should also match the OTHER face's LOWER
   template -- measured at 0.819..0.895, versus -0.272..-0.229 for F's upper against F's OWN lower.
   `identify_face` averages that cross-check into the score (`use_swap_check`, default on). On
   clean templates this costs a little margin (worst-case margin 1.143 with the cross-check vs
   1.212 without; both classify 43/43). It earns that back when a template is imperfect, which is
   the real risk since registration happens once from a single frame: with 60% of Face A's upper
   strip occluded in its registration frame, the worst-case best-score holds at 0.753 with the
   cross-check but falls to 0.662 without it. Set `params.use_swap_check=False` for the last drop
   of margin if you trust your registration frames.

ROBUSTNESS DESIGN (all thresholds are relative, nothing is hard-coded to this rig's pixel values)
--------------------------------------------------------------------------------------------------
* **Band + strips are found per frame**, never assumed. `band_rows=None` auto-locates from the
  row-wise bright-pixel profile inside a central vertical search window (DESIGN.md §2 puts the LED
  slots "in the middle"; the window is needed because the brightest thing in the FULL frame is
  actually the illuminated stage along the bottom edge -- measured rows ~937-1023, brighter and
  taller than either strip).
* **Intensity thresholds are per-frame relative**, derived from a low/high percentile pair of the
  search window (`dark_percentile` / `bright_percentile`), not absolute grey levels. An absolute
  threshold of 250 worked on the as-recorded video but collapsed completely (0/43 frames located)
  under a 0.75x exposure/gain change; the relative version holds 43/43 from 0.6x to 1.4x gain.
* **The column profile is a per-column MAX over the strip's rows** (after a small 2-D median blur to
  kill hot pixels), not a mean or a bright-pixel fraction. Physically the question is "does ANY
  light get through this column?", since a sticker is opaque -- and the LED slot is partly occluded
  by frame hardware, so in places only a thin sliver of the strip is lit. A row-mean/row-fraction
  profile washes those slivers out (it reported only 2-3 of the 4 blocks per strip on real Face A
  frames); the max profile recovers all 4.
* **Tilt.** The drum sits tilted: the strips run at a measured 1.05-1.69 degrees across the real
  recording (mean 1.35, and only +-0.18 within a face). A per-column max over the strip's full row
  extent is inherently tilt-tolerant -- a tilted strip still passes through every column of its own
  bounding band, so the max still finds it, whereas a row-mean gets diluted by the empty rows the
  tilt adds to the bounding box. Vertical sticker edges stay vertical under such a small rotation
  (a 1.35 degree tilt displaces an edge by ~1 px across a 45-row strip), so the block structure is
  unaffected. The limit is geometric, not algorithmic: past ~2.4 degrees total tilt the two strips
  overlap in row space and no row-profile method can separate them (measured: OK at +-1.0 degrees of
  ADDED tilt on top of the rig's own 1.35; fails at -1.5 added, i.e. 2.85 total).
* **Shifts.** Profiles are cropped to the strips' lit x-extent and resampled to a fixed length, which
  removes global translation and scale; residual misalignment is absorbed by a small lag search in
  the normalized cross-correlation (`max_lag` samples each way). Both together cover the ~30 px
  face-to-face x-offsets seen in the transition tables above.

Tuned and validated against `Good Markers.avi` (1280 frames, 1280x1024, 30.27 fps, 43 dwells);
see this module's report in `tests/test_marker_band.py` for the synthetic contract tests.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: (row_start, row_end) INCLUSIVE, in full-frame row coordinates.
Strip = Tuple[int, int]
#: (x0, x1) INCLUSIVE, in full-frame column coordinates.
Span = Tuple[int, int]


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
@dataclass
class MarkerBandParams:
    """Knobs for band/strip location, profiling, matching and block decoding.

    Everything intensity-related is expressed RELATIVE to a per-frame low/high percentile pair, so
    no value here depends on this rig's absolute grey levels or exposure (see module docstring).
    """

    # --- vertical search window for auto band location (fractions of frame height) ---
    search_frac: Tuple[float, float] = (0.20, 0.80)

    # --- per-frame intensity references ---
    dark_percentile: float = 5.0      # "unlit" reference level of the search window
    #: a pixel counts as "lit strip" if it is above dark + bright_rel*(bright - dark).
    bright_rel: float = 0.90

    # --- strip location from the row-wise lit-pixel profile ---
    strip_run_frac: float = 0.25      # keep rows whose lit-count exceeds this fraction of the peak
    row_join_gap: int = 3             # rows this close are one run
    min_strip_h: int = 6              # reject runs shorter than this (px)
    max_strip_h: int = 110            # reject runs taller than this (px): not a thin LED slot
    max_strip_gap: int = 160          # the two strips must be within this many rows of each other
    min_peak_frac: float = 0.02       # need a row with >= this fraction of the width lit, else no band
    min_dynamic_range: float = 20.0   # bright-dark below this => featureless frame, no band

    # --- column profile ---
    median_blur: int = 3              # 2-D median blur (px, odd) applied to the strip before max
    profile_length: int = 256         # resample length; makes profiles comparable across frames
    lit_frac_level: float = 0.35      # level on the lit-fraction profile that defines the x-window

    # --- matching ---
    max_lag: int = 8                  # +-samples of lag searched in the NCC (1 sample ~ 3.7 px here)
    min_score: float = 0.45           # reject if even the best face correlates worse than this
    min_margin: float = 0.15          # reject if best and runner-up are closer than this
    use_swap_check: bool = True       # fold the upper<->lower swap cross-check into the score

    # --- vial-block decoding ---
    block_level: float = 0.50         # binarization level on the [0,1]-normalized column profile
    close_px: int = 21                # bridge binary gaps shorter than this (hardware inside the slot)
    edge_tol_px: int = 12             # a run within this of the lit extent is edge-truncated


# --------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------
def _as_gray2d(frame) -> np.ndarray:
    """Return a 2-D uint8 view of `frame` (defensive; frames are nominally already this)."""
    if frame is None:
        raise ValueError("frame is None")
    img = np.asarray(frame)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _runs(mask_1d: np.ndarray, join_gap: int = 1) -> List[Tuple[int, int]]:
    """Contiguous True runs of a 1-D boolean array as inclusive (start, end), joining small gaps."""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    out: List[Tuple[int, int]] = []
    start, prev = int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= join_gap:
            prev = i
        else:
            out.append((start, prev))
            start = prev = i
    out.append((start, prev))
    return out


def _ncc(a: np.ndarray, b: np.ndarray, max_lag: int = 0) -> float:
    """Best zero-mean normalized cross-correlation of `a` and `b` over integer lags in +-max_lag.

    Returns a value in [-1, 1] (or -1.0 if either overlap is constant/degenerate). The lag search is
    what makes matching tolerant of the residual x-misalignment left after the profiles have been
    cropped to their lit extent and resampled -- see module docstring, "Shifts".
    """
    best = -1.0
    n = min(len(a), len(b))
    for lag in range(-int(max_lag), int(max_lag) + 1):
        if lag < 0:
            x, y = a[-lag:n], b[:n + lag]
        elif lag > 0:
            x, y = a[:n - lag], b[lag:n]
        else:
            x, y = a[:n], b[:n]
        if x.size < 8:
            continue
        x = x - x.mean()
        y = y - y.mean()
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom <= 1e-12:
            continue
        best = max(best, float(x @ y) / denom)
    return best


def _resample(seg: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a 1-D profile to exactly `length` samples."""
    if seg.size == 0:
        return np.zeros(length, dtype=np.float64)
    if seg.size == 1:
        return np.full(length, float(seg[0]))
    return np.interp(np.linspace(0.0, seg.size - 1.0, length), np.arange(seg.size), seg)


@dataclass
class _BandContext:
    """Everything derived once per frame: the strips and the per-frame intensity references."""

    strips: List[Strip]
    dark_ref: float
    bright_ref: float
    lit_thr: float


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
class MarkerBandDetector:
    """Identify the drum face, and decode vial column boundaries, from the central marker band.

    Typical use (calibration then run):

        det = MarkerBandDetector()
        det.register_face(face_a_still, "A")
        det.register_face(face_b_still, "B")
        ...
        pipeline = TrackerPipeline(..., marker_detector=det)   # duck-typed on identify_face

    `identify_face` returns None -- and `TrackerPipeline` then falls back to the default face and
    logs `marker_absent`, per DESIGN.md §5.2 -- whenever the band cannot be found, fewer than two
    faces are registered, the best correlation is below `params.min_score`, or the best/runner-up
    margin is below `params.min_margin`.
    """

    def __init__(
        self,
        band_rows: Optional[Tuple[int, int]] = None,
        bright_percentile: float = 99.5,
        min_run_px: int = 25,
        params: Optional[MarkerBandParams] = None,
        templates: Optional[Dict[str, Sequence[Sequence[float]]]] = None,
    ) -> None:
        """
        Args:
            band_rows: ``(row0, row1)`` inclusive full-frame rows to search for the two strips.
                ``None`` (default) auto-locates the band per frame from the row-wise lit-pixel
                profile inside ``params.search_frac`` of the frame height.
            bright_percentile: percentile of the search window taken as the "fully lit" reference.
                Used both to threshold strip rows and to normalize column profiles to [0, 1]. A
                percentile rather than an absolute grey level so exposure/gain changes do not break
                detection (see module docstring).
            min_run_px: minimum width in pixels of a bright run for it to count as one vial's lit
                gap in `vial_boundaries`. Narrower runs are hardware glints, not a vial.
            params: fine-tuning knobs; defaults are the values validated on `Good Markers.avi`.
            templates: optional pre-registered ``{face: [upper_profile, lower_profile]}``, e.g. from
                `to_dict()`.
        """
        self.band_rows = tuple(band_rows) if band_rows is not None else None
        self.bright_percentile = float(bright_percentile)
        self.min_run_px = int(min_run_px)
        self.params = params or MarkerBandParams()
        #: face name -> (upper_profile, lower_profile), each `params.profile_length` floats in [0,1].
        self.templates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for face, pair in (templates or {}).items():
            up, lo = pair
            self.templates[face] = (
                np.asarray(up, dtype=np.float64), np.asarray(lo, dtype=np.float64),
            )

    # -- band / strips ------------------------------------------------------------------
    def _search_window(self, gray: np.ndarray) -> Tuple[int, int]:
        """Inclusive (row0, row1) of the region searched for the two strips."""
        h = gray.shape[0]
        if self.band_rows is not None:
            r0, r1 = self.band_rows
            return max(0, int(r0)), min(h - 1, int(r1))
        f0, f1 = self.params.search_frac
        return int(h * f0), min(h - 1, int(h * f1) - 1)

    def _context(self, gray: np.ndarray) -> Optional[_BandContext]:
        """Locate the two strips and the per-frame intensity references. None if there is no band."""
        p = self.params
        r0, r1 = self._search_window(gray)
        if r1 <= r0:
            return None
        sub = gray[r0:r1 + 1]
        dark_ref = float(np.percentile(sub, p.dark_percentile))
        bright_ref = float(np.percentile(sub, self.bright_percentile))
        if bright_ref - dark_ref < p.min_dynamic_range:
            return None                                   # featureless: no strips to find
        lit_thr = dark_ref + p.bright_rel * (bright_ref - dark_ref)

        prof = (sub >= lit_thr).sum(axis=1).astype(np.float64)
        if prof.max() < gray.shape[1] * p.min_peak_frac:
            return None                                   # nothing strip-like is lit
        cand = [
            (a, b) for a, b in _runs(prof > prof.max() * p.strip_run_frac, p.row_join_gap)
            if p.min_strip_h <= (b - a + 1) <= p.max_strip_h
        ]
        if len(cand) < 2:
            return None
        # Best pair of runs (by total lit mass) that sit within `max_strip_gap` rows of each other.
        # "Mass" rather than height so a tall dim run cannot outrank the real, blindingly-lit slot.
        mass = [float(prof[a:b + 1].sum()) for a, b in cand]
        best: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        best_mass = -1.0
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                gap = cand[j][0] - cand[i][1]
                if gap <= 0 or gap > p.max_strip_gap:
                    continue
                if mass[i] + mass[j] > best_mass:
                    best_mass = mass[i] + mass[j]
                    best = (cand[i], cand[j])
        if best is None:
            return None
        strips = [(int(a + r0), int(b + r0)) for a, b in best]     # -> full-frame rows
        return _BandContext(strips=strips, dark_ref=dark_ref, bright_ref=bright_ref, lit_thr=lit_thr)

    def find_strips(self, frame_gray) -> List[Strip]:
        """Locate the two bright horizontal strips of the marker band.

        Returns ``[(r0, r1), (r0, r1)]`` -- INCLUSIVE full-frame row bounds, upper strip first --
        or ``[]`` if no plausible strip pair is present (blank frame, lights off, band out of view).
        """
        ctx = self._context(_as_gray2d(frame_gray))
        return list(ctx.strips) if ctx is not None else []

    # -- column profiles ----------------------------------------------------------------
    def _raw_profile(self, gray: np.ndarray, strip: Strip, ctx: _BandContext) -> np.ndarray:
        """Per-column [0, 1] "how lit is this column of this strip", in FULL-FRAME x coordinates.

        Per-column MAX over the strip's rows, after a small 2-D median blur. Max because a sticker
        is opaque, so the physical question is "does any light get through here?" -- and parts of
        the LED slot are occluded down to a thin lit sliver, which a row-mean would erase. The
        median blur first makes that max hot-pixel-proof. See module docstring, "Robustness design".
        """
        r0, r1 = strip
        s = gray[r0:r1 + 1, :]
        k = self.params.median_blur
        if k and k >= 3:
            s = cv2.medianBlur(s, k if k % 2 else k + 1)
        v = s.max(axis=0).astype(np.float64)
        span = max(1.0, ctx.bright_ref - ctx.dark_ref)
        return np.clip((v - ctx.dark_ref) / span, 0.0, 1.0)

    def _x_window(self, gray: np.ndarray, ctx: _BandContext) -> Span:
        """Inclusive (x0, x1) covering the lit extent of BOTH strips.

        Derived from the LIT-FRACTION profile (fraction of the strip's rows above `lit_thr`), not
        from the max profile used for the values: the fraction profile ignores the isolated bright
        hardware glints out at the drum's edges that would otherwise stretch the window to the frame
        border. Measured stability across the 43 real dwells: x0 in 148..152, x1 in 1082..1089.

        A window shared by both strips (rather than one per strip) is what keeps the upper<->lower
        swap comparison meaningful -- both profiles must live on the same x axis.
        """
        lo: List[int] = []
        hi: List[int] = []
        for r0, r1 in ctx.strips:
            frac = (gray[r0:r1 + 1, :] >= ctx.lit_thr).mean(axis=0)
            xs = np.flatnonzero(frac > self.params.lit_frac_level)
            if xs.size:
                lo.append(int(xs[0]))
                hi.append(int(xs[-1]))
        if not lo:
            return 0, gray.shape[1] - 1
        return min(lo), max(hi)

    def strip_profile(self, frame_gray, strip: Strip) -> np.ndarray:
        """Normalized column profile of one strip, resampled to `params.profile_length`.

        Float array in [0, 1]: ~1 where the LED strip shines through, ~0 where an opaque sticker
        (or unlit hardware) blocks it. Cropping to the shared lit x-window and resampling to a fixed
        length is what makes profiles from different frames directly comparable despite small
        translations and scale changes.
        """
        gray = _as_gray2d(frame_gray)
        ctx = self._context(gray)
        if ctx is None:
            return np.zeros(self.params.profile_length, dtype=np.float64)
        x0, x1 = self._x_window(gray, ctx)
        return _resample(self._raw_profile(gray, strip, ctx)[x0:x1 + 1], self.params.profile_length)

    def signature(self, frame_gray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``(upper_profile, lower_profile)`` for this frame, or ``None`` if no band is visible.

        This pair IS the face code: a 180 degree drum flip presents the same physical sticker row
        with these two swapped (see module docstring).
        """
        gray = _as_gray2d(frame_gray)
        ctx = self._context(gray)
        if ctx is None:
            return None
        x0, x1 = self._x_window(gray, ctx)
        L = self.params.profile_length
        upper, lower = (
            _resample(self._raw_profile(gray, s, ctx)[x0:x1 + 1], L) for s in ctx.strips
        )
        return upper, lower

    # -- registration / identification ---------------------------------------------------
    def register_face(self, frame_gray, face_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Learn this frame's (upper, lower) profiles as the template for `face_name`.

        Re-registering an existing face overwrites it. Raises ValueError if the marker band is not
        visible in `frame_gray` -- there is nothing to learn from such a frame.
        """
        sig = self.signature(frame_gray)
        if sig is None:
            raise ValueError(
                "marker band not found; cannot register a template for face %r" % (face_name,)
            )
        self.templates[face_name] = sig
        return sig

    def can_identify(self) -> bool:
        """True if `identify_face` could ever return a face -- i.e. two templates are registered.

        `pipeline.TrackerPipeline` asks this at startup to decide whether face identification is
        something it should WAIT for. A detector that cannot discriminate must not put the run
        into "face unknown" forever; it has to be reported instead (`cli.face_id_readiness`).
        """
        return len(self.templates) >= 2

    def score_faces(self, frame_gray) -> Dict[str, float]:
        """Similarity in [-1, 1] of this frame to every registered face. ``{}`` if no band.

        Diagnostic counterpart of `identify_face` -- use it to inspect margins when tuning
        `params.min_score` / `params.min_margin` on a new rig.
        """
        sig = self.signature(frame_gray)
        if sig is None or not self.templates:
            return {}
        return self._score(sig)

    def _score(self, sig: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        up, lo = sig
        p = self.params
        names = list(self.templates)
        out: Dict[str, float] = {}
        for face in names:
            t_up, t_lo = self.templates[face]
            direct = 0.5 * (_ncc(up, t_up, p.max_lag) + _ncc(lo, t_lo, p.max_lag))
            others = [n for n in names if n != face]
            if p.use_swap_check and len(names) == 2:
                # The drum has exactly two faces and a flip swaps up<->down, so the OTHER face's
                # lower template is a second, independently captured picture of THIS face's upper
                # pattern. Averaging the two halves cancels per-capture template noise -- see the
                # module docstring for the measured cost (clean templates) and benefit (a template
                # registered from a partly occluded frame). Only valid for exactly two faces.
                o_up, o_lo = self.templates[others[0]]
                swapped = 0.5 * (_ncc(up, o_lo, p.max_lag) + _ncc(lo, o_up, p.max_lag))
                out[face] = 0.5 * (direct + swapped)
            else:
                out[face] = direct
        return out

    def identify_face(self, frame_gray) -> Optional[str]:
        """Best-matching registered face name, or ``None`` if the call is not confident.

        Duck-typed to `markers.MarkerDetector.identify_face`, which is the whole contract
        `pipeline.TrackerPipeline` depends on. ``None`` (-> pipeline defaults to face "A" and logs
        `marker_absent`, DESIGN.md §5.2) is returned when:

          * the marker band is not visible in the frame,
          * fewer than two faces are registered (nothing to discriminate against),
          * the best correlation is below `params.min_score`, or
          * best minus runner-up is below `params.min_margin`.

        The discrimination works because one face's own upper and lower profiles are ANTI-correlated
        (they are the two interleaved halves of the alternating sticker pattern), while the same
        face's upper matches its own upper template almost perfectly. Measured over the 43 real
        dwells of `Good Markers.avi`: same-face upper-to-upper correlation 0.988..1.000,
        different-face -0.258..-0.167.
        """
        sig = self.signature(frame_gray)
        if sig is None or len(self.templates) < 2:
            return None
        scores = self._score(sig)
        order = sorted(scores, key=lambda f: -scores[f])
        best = order[0]
        if scores[best] < self.params.min_score:
            return None
        if scores[best] - scores[order[1]] < self.params.min_margin:
            return None
        return best

    # -- vial boundary decoding ----------------------------------------------------------
    def _bright_runs(self, prof: np.ndarray, x0: int, x1: int) -> List[Span]:
        """Binarize a [0,1] column profile inside [x0, x1] and return its bright runs.

        Small dark gaps are bridged (`params.close_px`): the LED slot has mounting hardware inside
        it that notches an otherwise continuous lit run, and those notches are far narrower than a
        vial.
        """
        p = self.params
        binr = np.zeros(prof.shape, dtype=bool)
        binr[x0:x1 + 1] = prof[x0:x1 + 1] >= p.block_level
        runs = _runs(binr, join_gap=max(1, p.close_px))
        return [(a, b) for a, b in runs if (b - a + 1) >= self.min_run_px]

    def vial_boundaries(self, frame_gray) -> List[Span]:
        """Decode per-vial column extents from the sticker pattern. ``[]`` if no band is visible.

        Returns inclusive ``(x0, x1)`` full-frame column spans, sorted left -> right; on this rig
        that is 8 spans, one per vial column, applying to BOTH the upper and lower vial rows
        (DESIGN.md §2: 16 slots = 2 rows x 8 columns, and the sticker row runs between them).

        How it works: the stickers alternate up/down, so each strip's bright runs are bounded left
        and right by that strip's own stickers -- one bright run == one vial. Collecting the bright
        runs of BOTH strips and sorting by x therefore yields the vial columns in order, alternating
        which strip each came from.

        **Partially-lit end columns.** The outermost run on each side is bounded by a sticker on its
        inner edge only; its outer edge is where the LED slot simply stops (or falls into the
        rig's dark right-hand margin, DESIGN.md §2). Such a run -- one whose outer edge sits within
        `params.edge_tol_px` of the strips' lit extent -- is flagged as edge-truncated and its outer
        edge is EXTRAPOLATED to the median width of the interior (fully bounded) runs, then clipped
        to the frame. That yields a plausible full-width column instead of a short stub. If there
        are no interior runs to take a median from, the truncated run is returned as measured.

        Overlapping neighbours (the two strips' runs can overlap by a few px, since a sticker's own
        width is what separates them) are split at their midpoint so the returned spans are disjoint
        and strictly increasing.
        """
        gray = _as_gray2d(frame_gray)
        ctx = self._context(gray)
        if ctx is None:
            return []
        x0, x1 = self._x_window(gray, ctx)
        runs: List[Span] = []
        for strip in ctx.strips:
            runs.extend(self._bright_runs(self._raw_profile(gray, strip, ctx), x0, x1))
        if not runs:
            return []
        runs.sort()

        tol = self.params.edge_tol_px
        interior = [
            (a, b) for a, b in runs
            if not (a - x0 <= tol or x1 - b <= tol)
        ]
        med_w = int(np.median([b - a + 1 for a, b in interior])) if interior else 0

        out: List[Span] = []
        w_img = gray.shape[1]
        for i, (a, b) in enumerate(runs):
            if med_w > 0 and i == 0 and a - x0 <= tol and (b - a + 1) < med_w:
                a = max(0, b - med_w + 1)                      # left column truncated by slot end
            if med_w > 0 and i == len(runs) - 1 and x1 - b <= tol and (b - a + 1) < med_w:
                b = min(w_img - 1, a + med_w - 1)              # right column truncated by slot end
            out.append((int(a), int(b)))

        # Make the spans disjoint: split any overlap at its midpoint.
        for i in range(1, len(out)):
            pa, pb = out[i - 1]
            a, b = out[i]
            if a <= pb:
                mid = (a + pb) // 2
                out[i - 1] = (pa, max(pa, mid))
                out[i] = (min(b, mid + 1), b)
        return out

    # -- persistence ---------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        """JSON-safe snapshot: templates + the settings needed to reproduce them.

        Store this next to the calibration bundle (e.g. inside `types.FaceCalibration.marker`) so a
        restarted run identifies faces with exactly the templates it was calibrated with.
        """
        return {
            "band_rows": list(self.band_rows) if self.band_rows is not None else None,
            "bright_percentile": self.bright_percentile,
            "min_run_px": self.min_run_px,
            "params": asdict(self.params),
            "templates": {
                face: [up.tolist(), lo.tolist()] for face, (up, lo) in self.templates.items()
            },
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "MarkerBandDetector":
        """Rebuild a detector from `to_dict()` output (round-trips templates and settings)."""
        raw = dict(d.get("params") or {})
        if "search_frac" in raw:
            raw["search_frac"] = tuple(raw["search_frac"])
        params = MarkerBandParams(**raw) if raw else MarkerBandParams()
        band_rows = d.get("band_rows")
        return MarkerBandDetector(
            band_rows=tuple(band_rows) if band_rows else None,
            bright_percentile=float(d.get("bright_percentile", 99.5)),
            min_run_px=int(d.get("min_run_px", 25)),
            params=params,
            templates=d.get("templates") or {},
        )

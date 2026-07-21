"""Frame-to-calibration registration (DESIGN.md §5.2).

On each stationary onset the pipeline needs to know how far the rig's rigid structure has drifted
since calibration so per-vial ROIs (defined once, on the calibration frame) can be re-anchored
onto the live frame. This module estimates that translation via phase correlation and applies it
to a bbox. (Rotation-angle (dθ) correction is out of scope here — DESIGN.md §5.2 lists it as an
option alongside translation; this implementation covers the translation term, which is what
`apply_shift` consumes.)
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def estimate_shift(
    cur_gray: np.ndarray,
    ref_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Estimate the translation of `cur_gray`'s content relative to `ref_gray` via phase correlation.

    Returns `(dx, dy, residual)`:

    - `(dx, dy)`: content found at pixel `(x, y)` in `ref_gray` is found at approximately
      `(x + dx, y + dy)` in `cur_gray`. This is the offset to ADD to a bbox/ROI that was defined in
      `ref_gray` (calibration-frame) coordinates to relocate it onto `cur_gray` — see
      `apply_shift`. Verified empirically against `cv2.phaseCorrelate`'s own convention: calling
      `cv2.phaseCorrelate(ref_f, cur_f)` returns exactly this `(dx, dy)` (confirmed against known
      `np.roll` shifts to << 0.5 px).

    - `residual = max(0.0, 1.0 - response)`, where `response` is `cv2.phaseCorrelate`'s normalized
      cross-power-spectrum peak magnitude (empirically in ~[0, 1]: ~1.0 for a sharp, confident
      peak — e.g. identical or purely-translated frames — down to ~0.0-0.05 for two frames with no
      coherent relationship). So `residual` is LOW when the shift estimate is trustworthy and HIGH
      (approaching 1.0) when the two frames don't correlate well (occlusion, big illumination
      change, wrong face, etc.) and `(dx, dy)` should be treated as unreliable. This is the
      "confidence" signal DESIGN.md §5.2 calls for in its mis-registration guard: reject/flag a
      registration when `residual` exceeds a configured threshold.

    `mask`, if given, is a bool ndarray the same shape as the frames; it is passed to
    `cv2.phaseCorrelate` as a 0/1 weighting window, so only True pixels contribute to the
    correlation. A hard binary mask tapers less smoothly at its edges than e.g. a Hanning window,
    so expect a slightly lower `response` (higher `residual`) than the unmasked case — the shift
    estimate itself remains accurate to sub-pixel level in practice.

    Both inputs are cast to float32 (the dtype `cv2.phaseCorrelate` requires); they are not
    modified in place.
    """
    ref_f = ref_gray.astype(np.float32)
    cur_f = cur_gray.astype(np.float32)
    window = mask.astype(np.float32) if mask is not None else None

    (dx, dy), response = cv2.phaseCorrelate(ref_f, cur_f, window)
    # `response` can land a hair above 1.0 on a near-perfect match (float noise in the FFT-based
    # peak estimate) — clip so `residual` never reports a nonsensical tiny-negative "confidence".
    residual = max(0.0, 1.0 - float(response))
    return float(dx), float(dy), residual


def apply_shift(
    bbox_xywh: Tuple[int, int, int, int],
    dx: float,
    dy: float,
) -> Tuple[int, int, int, int]:
    """Translate a `(x, y, w, h)` bbox by `(dx, dy)`, integer-rounding the new origin.

    Width/height are left unchanged — registration here corrects translation only (see module
    docstring), matching `VialROI`'s fixed `w`/`h` from calibration.
    """
    x, y, w, h = bbox_xywh
    new_x = int(round(x + dx))
    new_y = int(round(y + dy))
    return (new_x, new_y, w, h)

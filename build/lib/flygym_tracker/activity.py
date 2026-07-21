"""Per-vial activity metric + bin aggregation (DESIGN.md §5.3).

Two deliberately separate pieces:
  - `per_frame_activity`: a pure function, no state — one frame's motion for one vial.
  - `ActivityAccumulator`: a stateful bucket that sums/means `per_frame_activity` results (plus the
    rotation `TrackState`) into per-bin summaries, ready to become `ActivityRecord` rows.

Reference-frame ("prev") reset semantics
-----------------------------------------
Neither piece here owns a "reference frame". DESIGN.md §5.3 requires the diff baseline
(`prev_stationary`) to be reset after any rotation or face change, so a stationary frame is never
diffed against a frame from before the rotation/change. That reset is entirely the PIPELINE's
responsibility: the pipeline decides which frame is `prev_gray` on each call to
`per_frame_activity` (per DESIGN.md: "reference = previous frame classified stationary on the same
face; reset the reference after any rotation / face change"). `ActivityAccumulator` never sees raw
frames — only the already-computed per-frame `(motion_px, lit_area_px, active_fraction)` results
plus the `TrackState` for that frame. It has no `prev` of its own, so there is nothing here to
reset; whatever `prev_gray` the pipeline's reference policy selects, just call
`per_frame_activity(cur, that_prev, ...)` and feed the result into `ActivityAccumulator.add()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from flygym_tracker.types import TrackState

#: vial_id -> (motion_px, lit_area_px, active_fraction), as returned by `per_frame_activity`.
VialResults = Dict[int, Tuple[int, int, float]]


def per_frame_activity(
    cur_gray: np.ndarray,
    prev_gray: np.ndarray,
    vial_mask_bool: np.ndarray,
    pixel_threshold: float,
) -> Tuple[int, int, float]:
    """One frame's motion for one vial (DESIGN.md §5.3). Pure function, no state.

    - `motion_px`: count of pixels within `vial_mask_bool` where `|cur - prev| > pixel_threshold`.
    - `lit_area_px`: count of True pixels in `vial_mask_bool` (the vial's trackable area).
    - `active_fraction`: `motion_px / lit_area_px`, or `0.0` if `lit_area_px == 0`
      (guards divide-by-zero for an empty/absent vial's mask).

    `vial_mask_bool` is expected to already be the effective per-vial mask
    `M = (illum_mask == 255) ∩ vial.bbox` (DESIGN.md §5.3) — this function has no knowledge of
    calibration or illumination masks; it only ever sees the final boolean mask.

    Uses `cv2.absdiff` (not plain numpy subtraction) so unsigned frames don't wrap on underflow.
    """
    diff = cv2.absdiff(cur_gray, prev_gray)
    motion = diff > pixel_threshold

    lit_area_px = int(np.count_nonzero(vial_mask_bool))
    motion_px = int(np.count_nonzero(motion & vial_mask_bool))
    active_fraction = (motion_px / lit_area_px) if lit_area_px > 0 else 0.0
    return motion_px, lit_area_px, active_fraction


@dataclass
class _VialBinAccum:
    """Internal running totals for one vial within the bin currently being accumulated."""

    motion_px_sum: int = 0
    #: summed active_fraction over STATIONARY frames only; divide by n_stationary_frames for the mean.
    active_fraction_sum: float = 0.0
    n_stationary_frames: int = 0
    n_rotating_frames: int = 0
    #: latest value seen (constant per vial unless calibration/registration changes it).
    lit_area_px: int = 0


@dataclass
class ActivityBin:
    """One completed bin's worth of per-vial results, ready to feed `ActivityRecord` rows.

    `vials[vial_id]` holds exactly the fields DESIGN.md §5.3 asks for per (vial, bin):
    `motion_px_sum`, `active_fraction_mean`, `n_stationary_frames`, `n_rotating_frames`,
    `lit_area_px`. Everything else an `ActivityRecord` needs (`run_id`, `face`, `row`, `col`,
    `present`, `bin_start_iso`/`bin_end_iso`, ...) comes from run metadata + `Calibration`, which
    this module deliberately knows nothing about. `bin_start_s`/`bin_end_s` (elapsed seconds, same
    clock as the `elapsed_s` passed into `add()`) are included so the pipeline can derive
    `elapsed_s`/`bin_start_iso`/`bin_end_iso` without having to track bin boundaries itself.
    """

    bin_start_s: float
    bin_end_s: float
    vials: Dict[int, dict] = field(default_factory=dict)


class ActivityAccumulator:
    """Buckets per-frame, per-vial `per_frame_activity` results into per-bin summaries.

    Call `add()` once per processed frame with that frame's `elapsed_s` (elapsed seconds since run
    start — same clock `ActivityRecord.elapsed_s` / `Frame.t_monotonic` use), its `TrackState`, and
    every vial's per-frame result for that frame. `add()` returns an `ActivityBin` when the just-
    added sample rolled the accumulator into a new bin — the just-completed *previous* bin is
    returned at that point; otherwise it returns `None`. Call `flush()` once at shutdown (or
    whenever a partial bin must be emitted early) to get the final, possibly-partial bin.

    Only STATIONARY frames feed `motion_px_sum` / `active_fraction_mean` / `n_stationary_frames`
    (DESIGN.md §5.3: "for each stationary, non-settling frame"). ROTATING frames feed only
    `n_rotating_frames` — their `motion_px`/`active_fraction` numbers (if the caller even computed
    any — a rotating frame is global blur, not per-vial activity) are ignored. SETTLING and UNKNOWN
    frames are not counted anywhere, matching `TrackState.SETTLING`'s doc comment ("excluded from
    activity"); pass `vial_results=None`/`{}` for these if convenient. `lit_area_px` is simply the
    latest value seen per vial from whichever frames DO carry vial results.
    """

    def __init__(self, bin_seconds: float) -> None:
        if bin_seconds <= 0:
            raise ValueError("bin_seconds must be > 0")
        self.bin_seconds = bin_seconds
        self._bin_index: Optional[int] = None
        self._accum: Dict[int, _VialBinAccum] = {}

    def add(
        self,
        elapsed_s: float,
        state: TrackState,
        vial_results: Optional[VialResults] = None,
    ) -> Optional[ActivityBin]:
        """Fold one frame's results in; returns the completed previous bin on rollover, else None."""
        vial_results = vial_results or {}
        bin_index = int(elapsed_s // self.bin_seconds)

        rollover: Optional[ActivityBin] = None
        if self._bin_index is None:
            self._bin_index = bin_index
        elif bin_index != self._bin_index:
            rollover = self._flush(self._bin_index)
            self._bin_index = bin_index

        if state == TrackState.STATIONARY:
            for vial_id, (motion_px, lit_area_px, active_fraction) in vial_results.items():
                a = self._accum.setdefault(vial_id, _VialBinAccum())
                a.motion_px_sum += motion_px
                a.active_fraction_sum += active_fraction
                a.n_stationary_frames += 1
                a.lit_area_px = lit_area_px
        elif state == TrackState.ROTATING:
            for vial_id, (_motion_px, lit_area_px, _active_fraction) in vial_results.items():
                a = self._accum.setdefault(vial_id, _VialBinAccum())
                a.n_rotating_frames += 1
                a.lit_area_px = lit_area_px
        # SETTLING / UNKNOWN: intentionally not accumulated (see class docstring).

        return rollover

    def flush(self) -> Optional[ActivityBin]:
        """Force-emit the current (possibly partial) bin, e.g. when a run stops mid-bin.

        Returns None if nothing has been added since the last flush/rollover.
        """
        if self._bin_index is None:
            return None
        result = self._flush(self._bin_index)
        self._bin_index = None
        return result

    def _flush(self, bin_index: int) -> ActivityBin:
        vials: Dict[int, dict] = {}
        for vial_id, a in self._accum.items():
            mean_af = (
                a.active_fraction_sum / a.n_stationary_frames if a.n_stationary_frames > 0 else 0.0
            )
            vials[vial_id] = {
                "motion_px_sum": a.motion_px_sum,
                "active_fraction_mean": mean_af,
                "n_stationary_frames": a.n_stationary_frames,
                "n_rotating_frames": a.n_rotating_frames,
                "lit_area_px": a.lit_area_px,
            }
        bin_result = ActivityBin(
            bin_start_s=bin_index * self.bin_seconds,
            bin_end_s=(bin_index + 1) * self.bin_seconds,
            vials=vials,
        )
        self._accum = {}
        return bin_result

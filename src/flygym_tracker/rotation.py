"""Rotation vs. stationary detection (DESIGN.md §5.1).

`RotationDetector` classifies each incoming frame against the previous frame using a global
motion metric — mean absolute pixel difference, optionally restricted to an ROI/illuminated mask
— with hysteresis (separate enter/exit thresholds) and a debounce counter so a single noisy frame
can't flip the state. On confirmed stationary onset the detector reports SETTLING for
`min_stationary_frames` frames (time for downstream face-id + re-registration, DESIGN.md §5.2)
before reporting STATIONARY.

This module only classifies motion; it does not know about `EventRecord` (that dataclass lives in
`types.py` and is populated by the pipeline). `RotationDetector.events` is a raw transition log —
`(frame_index, from_state, to_state)` — that the pipeline maps into whichever `EventRecord.event`
strings it cares about (DESIGN.md §5.1: entering ROTATING -> "rotation_start", leaving ROTATING
-> "rotation_end"); transitions it doesn't care about (e.g. SETTLING -> STATIONARY) it can ignore.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from flygym_tracker.types import TrackState


class RotationDetector:
    """Stateful STATIONARY/ROTATING/SETTLING classifier driven by global frame-diff motion.

    Call `update(frame_gray)` once per acquired frame, in acquisition order. The detector owns the
    previous frame and current state internally — it is not re-entrant across independent frame
    streams; use one instance per stream.

    State machine (per DESIGN.md §5.1):
      - Any frame with `m > enter_threshold` is ROTATING immediately (single-frame trigger),
        regardless of the prior state. This also resets the debounce/settling counters.
      - From ROTATING (or the initial UNKNOWN state), `debounce_frames` *consecutive* frames with
        `m < exit_threshold` are required before declaring a stationary onset. A frame that is
        neither `> enter_threshold` nor `< exit_threshold` (the hysteresis mid-band) breaks that
        streak without itself causing a ROTATING transition.
      - A stationary onset moves the state to SETTLING, not directly to STATIONARY. The next
        `min_stationary_frames` frames (counting the onset frame itself) are reported SETTLING;
        after that the state becomes STATIONARY and stays there for as long as
        `m <= enter_threshold` (mid-band frames while SETTLING/STATIONARY do not reset anything —
        only exceeding `enter_threshold` does).
      - The very first frame ever seen has no previous frame to diff against: it is reported
        UNKNOWN and only seeds `_prev_frame`; no metric is computed and no transition is logged.
    """

    def __init__(
        self,
        enter_threshold: float,
        exit_threshold: float,
        debounce_frames: int,
        min_stationary_frames: int,
        roi_mask: Optional[np.ndarray] = None,
    ) -> None:
        if debounce_frames < 1:
            raise ValueError("debounce_frames must be >= 1")
        if min_stationary_frames < 1:
            raise ValueError("min_stationary_frames must be >= 1")

        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.debounce_frames = debounce_frames
        self.min_stationary_frames = min_stationary_frames
        self.roi_mask = roi_mask

        #: current classification; starts UNKNOWN until the first real diff is computed.
        self.state: TrackState = TrackState.UNKNOWN
        #: mean(|cur - prev|) from the most recent `update()` call; None before any diff exists.
        self.last_metric: Optional[float] = None
        #: raw transition log: (frame_index, from_state, to_state), one entry per state change.
        self.events: List[Tuple[int, TrackState, TrackState]] = []

        self._prev_frame: Optional[np.ndarray] = None
        self._frame_index: int = 0
        self._quiet_streak: int = 0
        self._settling_count: int = 0

    @staticmethod
    def metric(cur: np.ndarray, prev: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Global motion metric: mean(|cur - prev|) over `mask` (bool ndarray) or the whole frame.

        Uses `cv2.absdiff` rather than plain numpy subtraction so unsigned (e.g. uint8) frames
        don't wrap around on underflow.
        """
        diff = cv2.absdiff(cur, prev)
        if mask is not None:
            if not np.any(mask):
                return 0.0
            return float(diff[mask].mean())
        return float(diff.mean())

    def update(self, frame_gray: np.ndarray) -> TrackState:
        """Classify one new frame; returns the (possibly updated) current state."""
        idx = self._frame_index
        self._frame_index += 1

        if self._prev_frame is None:
            # First frame ever: nothing to diff against yet. Seed and stay UNKNOWN.
            self._prev_frame = frame_gray
            return self.state

        m = self.metric(frame_gray, self._prev_frame, self.roi_mask)
        self.last_metric = m
        self._prev_frame = frame_gray

        if m > self.enter_threshold:
            self._quiet_streak = 0
            self._settling_count = 0
            self._set_state(idx, TrackState.ROTATING)
            return self.state

        # m <= enter_threshold: this frame alone does not indicate rotation.
        if self.state in (TrackState.ROTATING, TrackState.UNKNOWN):
            if m < self.exit_threshold:
                self._quiet_streak += 1
            else:
                self._quiet_streak = 0  # hysteresis mid-band frame breaks the quiet streak

            if self._quiet_streak >= self.debounce_frames:
                self._settling_count = 0
                self._set_state(idx, TrackState.SETTLING)
        elif self.state == TrackState.SETTLING:
            self._settling_count += 1
            if self._settling_count >= self.min_stationary_frames:
                self._set_state(idx, TrackState.STATIONARY)
        # else state == STATIONARY: m <= enter_threshold keeps it STATIONARY; nothing to do.

        return self.state

    def _set_state(self, frame_index: int, new_state: TrackState) -> None:
        if new_state != self.state:
            self.events.append((frame_index, self.state, new_state))
            self.state = new_state

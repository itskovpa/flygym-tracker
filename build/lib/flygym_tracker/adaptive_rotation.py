"""Speed-adaptive rotation vs. stationary detection (DESIGN.md §5.1, adaptive variant).

`AdaptiveRotationDetector` is a drop-in replacement for `rotation.RotationDetector` that removes the
two preset magnitude thresholds (`enter_threshold` / `exit_threshold` on mean|frame-diff|). Those
presets are fragile because the rig's rotation SPEED varies between experiments: a slow rotation
produces a tiny per-frame intensity change, close to the static-sensor floor, so any fixed
magnitude either misses slow rotations or false-triggers on noise.

WHY THIS IS SPEED-INDEPENDENT
-----------------------------
The rig's rigid structure (mounting frame, tube walls, edges) is displaced across the image if and
only if the whole setup rotates — at ANY speed. Flies only move *locally* inside vials and the
850 nm LED slots merely flicker *in place*; neither shifts the global scene. So the discriminating
signal is the **global inter-frame DISPLACEMENT** of the rigid structure, estimated with
`cv2.phaseCorrelate` — not the *magnitude* of pixel change (mean|diff|), which scales with speed.

- A static dwell → consecutive frames are (near-)identical → displacement ≈ 0 px, high response.
- A localized fly patch or an in-place LED flicker → intensity changes but the scene does NOT shift
  → phase-correlation peak stays at (0, 0) → displacement ≈ 0 px. (Empirically both give < 0.05 px.)
- A rotation, fast OR slow → the rigid structure shifts → sustained non-zero displacement. An 8 px/
  frame rotation gives ~8 px; a 1 px/frame drift gives ~1 px — both are 10-100x above the static
  floor. The *magnitude* of displacement changes with speed, but its *presence* (a non-zero,
  direction-consistent shift) does not. That presence is what we threshold.

The static displacement floor (~0.01-0.06 px in practice) is a property of the camera/sensor noise
and image size, NOT of rotation speed, so thresholding displacement a robust factor above that
auto-estimated floor catches rotations at every speed with the same settings.

ADAPTIVE THRESHOLD
------------------
The static noise floor of the displacement metric is estimated ONLINE from quiet periods (an initial
`calibration_frames` window seeds it; thereafter a rolling buffer of displacements observed while the
detector is STATIONARY/SETTLING keeps it current). Robust statistics (median + MAD) drive
`enter`/`exit` displacement thresholds a `sensitivity`-controlled margin above the floor, with a hard
sub-pixel minimum as a safety net.

SLOW-DRIFT (ACCUMULATION) PATH
------------------------------
To catch rotations slower than the instantaneous displacement noise band, a second, speed-independent
cue is the CONSISTENCY of displacement DIRECTION over a short window: a slow steady drift points the
same way every frame, so its per-frame vectors ACCUMULATE (net displacement grows ~linearly with the
window), whereas static jitter is random and cancels (net stays ~0). Rotation is flagged when the
accumulated directional displacement over `window_frames` clears an adaptive threshold AND the
directional consistency is high. This is what a fixed-magnitude detector fundamentally cannot do.

Everything downstream of the two cues — hysteresis, `debounce_frames`, `min_stationary_frames`
settling, and the `(frame_index, from_state, to_state)` transition log in `events` — matches
`RotationDetector` exactly, so the pipeline maps transitions to the same rotation_start/rotation_end
`EventRecord`s.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from flygym_tracker.types import TrackState


class AdaptiveRotationDetector:
    """Stateful STATIONARY/ROTATING/SETTLING classifier driven by adaptive global-displacement.

    Drop-in for `RotationDetector`: same `update()/state/last_metric/events` surface and identical
    hysteresis + debounce + settling state machine and transition log. The constructor differs only
    by dropping the two required magnitude thresholds (auto-estimated instead) and adding
    `sensitivity` / `calibration_frames` (plus optional `window_frames` / `min_consistency`, both
    defaulted, so it remains a positional drop-in for the shared `roi_mask/debounce_frames/
    min_stationary_frames` arguments).

    Call `update(frame_gray)` once per acquired frame, in acquisition order. One instance per stream
    (it owns the previous frame + running floor estimate).

    State machine (mirrors DESIGN.md §5.1 / `RotationDetector`), driven by two booleans per frame:
      - ``is_moving`` (displacement above the adaptive enter threshold, OR a consistent accumulated
        drift) → ROTATING immediately (single-frame trigger), resetting debounce/settling counters.
      - ``is_quiet`` (displacement below the adaptive exit threshold AND no residual accumulated
        drift). From ROTATING/UNKNOWN, `debounce_frames` *consecutive* quiet frames are required
        before a stationary onset. A frame that is neither moving nor quiet (the hysteresis mid-band)
        breaks the quiet streak without triggering ROTATING.
      - A stationary onset goes to SETTLING; the next `min_stationary_frames` frames (incl. the onset)
        report SETTLING, then STATIONARY. Mid-band frames while SETTLING/STATIONARY don't demote —
        only a moving frame does.
      - The first frame ever seen has no previous frame to diff against: reported UNKNOWN, seeds the
        previous frame, no metric, no transition.
    """

    # --- Displacement-domain constants (px). These are SENSOR-noise / sub-pixel guards, NOT
    #     rotation-magnitude presets: they do not encode how fast/large a rotation is, and any real
    #     rig rotation (>= ~0.3 px/frame) sits far above them. ---
    HARD_FLOOR_PX: float = 0.08        # absolute lower bound on the enter threshold (FFT sub-pixel jitter)
    DEFAULT_MIN_DISP_PX: float = 0.15  # default enter minimum at sensitivity=1.0 (scaled by sensitivity)
    BASE_MAD: float = 4.0              # robust MADs above the floor center to enter (scaled by sensitivity)
    EXIT_RATIO: float = 0.5            # exit threshold = this fraction of the enter margin (hysteresis)
    ABS_ACCUM_PX: float = 0.8          # absolute lower bound on the accumulated-drift enter threshold
    ACCUM_ENTER_FACTOR: float = 2.5    # accum enter = max(ABS_ACCUM_PX, enter * this)
    ACCUM_EXIT_RATIO: float = 0.5      # accum exit = accum_enter * this
    FLOOR_BUFFER: int = 200            # rolling window of quiet displacements for the floor estimate
    MIN_FLOOR_SAMPLES: int = 8         # until this many quiet samples exist, use the absolute minimum

    def __init__(
        self,
        roi_mask: Optional[np.ndarray] = None,
        debounce_frames: int = 4,
        min_stationary_frames: int = 3,
        sensitivity: float = 1.0,
        calibration_frames: int = 30,
        window_frames: int = 6,
        min_consistency: float = 0.6,
    ) -> None:
        if debounce_frames < 1:
            raise ValueError("debounce_frames must be >= 1")
        if min_stationary_frames < 1:
            raise ValueError("min_stationary_frames must be >= 1")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        if calibration_frames < 0:
            raise ValueError("calibration_frames must be >= 0")
        if window_frames < 1:
            raise ValueError("window_frames must be >= 1")
        if not (0.0 <= min_consistency <= 1.0):
            raise ValueError("min_consistency must be in [0, 1]")

        self.roi_mask = roi_mask
        self.debounce_frames = debounce_frames
        self.min_stationary_frames = min_stationary_frames
        self.sensitivity = float(sensitivity)
        self.calibration_frames = calibration_frames
        self.window_frames = window_frames
        self.min_consistency = float(min_consistency)

        #: current classification; starts UNKNOWN until the first real displacement is computed.
        self.state: TrackState = TrackState.UNKNOWN
        #: PRIMARY adaptive signal from the most recent update(): global inter-frame displacement
        #: magnitude sqrt(dx^2+dy^2) in px (speed-independent). None before any diff exists.
        self.last_metric: Optional[float] = None
        #: raw transition log: (frame_index, from_state, to_state), one entry per state change.
        self.events: List[Tuple[int, TrackState, TrackState]] = []

        # --- diagnostics from the most recent update() (all None/0 before the first diff) ---
        self.last_disp: Optional[float] = None          # alias of last_metric (displacement px)
        self.last_response: Optional[float] = None       # phaseCorrelate peak (confidence, ~[0,1])
        self.last_accum: Optional[float] = None          # accumulated directional displacement (px)
        self.last_consistency: Optional[float] = None    # net/path over the window, [0,1]
        self.last_mean_diff: Optional[float] = None       # secondary cue: mean|cur-prev| (magnitude)
        self.enter_threshold: Optional[float] = None      # current adaptive displacement enter (px)
        self.exit_threshold: Optional[float] = None       # current adaptive displacement exit (px)
        self.floor_center: Optional[float] = None         # current estimated static displacement floor
        self.floor_spread: Optional[float] = None         # current robust spread of the floor

        self._prev_frame: Optional[np.ndarray] = None
        self._frame_index: int = 0
        self._quiet_streak: int = 0
        self._settling_count: int = 0
        self._floor_buf: Deque[float] = deque(maxlen=self.FLOOR_BUFFER)
        self._vec_buf: Deque[Tuple[float, float]] = deque(maxlen=self.window_frames)
        self._window_cache: dict = {}  # shape -> Hanning window (float32), when no roi_mask

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def displacement(
        cur: np.ndarray, prev: np.ndarray, window: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float]:
        """Global inter-frame motion via `cv2.phaseCorrelate` on float32.

        Returns `(disp, dx, dy, response)` where `disp = sqrt(dx^2 + dy^2)` is the global
        displacement magnitude in px, `(dx, dy)` the sub-pixel shift of `cur` relative to `prev`,
        and `response` the normalized cross-power-spectrum peak (~1.0 for a confident/coherent
        relationship — identical, noisy-static, or purely-translated frames — down toward 0 for no
        coherent relationship). `window`, if given, is a float32 weighting the same shape as the
        frames (an ROI mask, or a Hanning window to suppress spectral leakage from the borders).
        """
        prev_f = prev.astype(np.float32)
        cur_f = cur.astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(prev_f, cur_f, window)
        disp = float((dx * dx + dy * dy) ** 0.5)
        return disp, float(dx), float(dy), float(response)

    @staticmethod
    def metric(cur: np.ndarray, prev: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Secondary cue (magnitude): mean(|cur - prev|) over `mask` or the whole frame.

        Kept for parity with `RotationDetector.metric` (drop-in) and logged as `last_mean_diff`;
        it is NOT the primary discriminator here precisely because it scales with rotation speed.
        Uses `cv2.absdiff` so unsigned frames don't wrap on underflow.
        """
        diff = cv2.absdiff(cur, prev)
        if mask is not None:
            m = mask.astype(bool) if mask.dtype != bool else mask
            if not np.any(m):
                return 0.0
            return float(diff[m].mean())
        return float(diff.mean())

    def _window_for(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Correlation weighting window matching `frame`: the roi_mask if given, else a cached
        Hanning window (reduces border spectral leakage → cleaner/steadier displacement estimate)."""
        if self.roi_mask is not None:
            return self.roi_mask.astype(np.float32)
        key = frame.shape
        win = self._window_cache.get(key)
        if win is None:
            h, w = frame.shape[:2]
            win = cv2.createHanningWindow((w, h), cv2.CV_32F)
            self._window_cache[key] = win
        return win

    def _thresholds(self) -> Tuple[float, float]:
        """Current adaptive (enter, exit) DISPLACEMENT thresholds in px from the online floor."""
        s = self.sensitivity
        min_disp = max(self.HARD_FLOOR_PX, self.DEFAULT_MIN_DISP_PX / s)
        if len(self._floor_buf) >= self.MIN_FLOOR_SAMPLES:
            buf = np.fromiter(self._floor_buf, dtype=np.float64)
            center = float(np.median(buf))
            spread = 1.4826 * float(np.median(np.abs(buf - center)))
        else:
            center, spread = 0.0, 0.0
        self.floor_center, self.floor_spread = center, spread
        enter = max(min_disp, center + (self.BASE_MAD / s) * spread)
        exit_ = max(min_disp * self.EXIT_RATIO, center + (self.BASE_MAD * self.EXIT_RATIO / s) * spread)
        return enter, exit_

    def _accumulation(self) -> Tuple[float, float]:
        """Accumulated directional displacement (net vector magnitude) and directional consistency
        (net / path, in [0,1]) over the current `window_frames` displacement vectors.

        Consistency ~1 when every step points the same way (a steady drift = rotation at any speed);
        ~0 when steps are random (static jitter cancels). This is the slow-rotation discriminator.
        """
        if not self._vec_buf:
            return 0.0, 0.0
        sx = sum(v[0] for v in self._vec_buf)
        sy = sum(v[1] for v in self._vec_buf)
        net = float((sx * sx + sy * sy) ** 0.5)
        path = float(sum((v[0] * v[0] + v[1] * v[1]) ** 0.5 for v in self._vec_buf))
        consistency = net / (path + 1e-6)
        return net, consistency

    # ------------------------------------------------------------------ main

    def update(self, frame_gray: np.ndarray) -> TrackState:
        """Classify one new frame; returns the (possibly updated) current state."""
        idx = self._frame_index
        self._frame_index += 1

        if self._prev_frame is None:
            # First frame ever: nothing to diff against yet. Seed and stay UNKNOWN.
            self._prev_frame = frame_gray
            return self.state

        prev = self._prev_frame
        window = self._window_for(frame_gray)
        disp, dx, dy, response = self.displacement(frame_gray, prev, window)
        mean_diff = self.metric(frame_gray, prev, self.roi_mask)  # secondary cue (magnitude)
        self._prev_frame = frame_gray
        self._vec_buf.append((dx, dy))

        accum, consistency = self._accumulation()
        enter, exit_ = self._thresholds()
        accum_enter = max(self.ABS_ACCUM_PX, enter * self.ACCUM_ENTER_FACTOR)
        accum_exit = accum_enter * self.ACCUM_EXIT_RATIO

        # record diagnostics / primary signal
        self.last_metric = disp
        self.last_disp = disp
        self.last_response = response
        self.last_accum = accum
        self.last_consistency = consistency
        self.last_mean_diff = mean_diff
        self.enter_threshold = enter
        self.exit_threshold = exit_

        # Rotation is flagged from two speed-independent cues, BOTH gated on DIRECTIONAL CONSISTENCY
        # (the rig shifts coherently one way; flies churn incoherently and the LED flickers in place):
        #  - instantaneous displacement above the adaptive enter threshold (fast/moderate rotation), or
        #  - accumulated directional drift over the window (slow rotation below the instantaneous band).
        # The consistency gate is what rejects localized fly motion, whose apparent inter-frame shift
        # is small AND random in direction (consistency ~0.3), while a rotation at ANY speed shifts the
        # rigid structure the same way every frame (consistency ~1.0). Neither cue looks at the
        # magnitude of pixel change, so detection is speed-independent.
        directional = consistency > self.min_consistency
        moving_fast = directional and (disp > enter)
        moving_slow = directional and (accum > accum_enter)
        is_moving = moving_fast or moving_slow
        # Quiet requires BOTH the instantaneous displacement AND the residual accumulated drift to be
        # low, so the accumulation window drains before a stationary onset (prevents flip-flop and
        # premature settling right after a rotation).
        is_quiet = (disp < exit_) and (accum < accum_exit)

        if is_moving:
            self._quiet_streak = 0
            self._settling_count = 0
            self._set_state(idx, TrackState.ROTATING)
        else:
            if self.state in (TrackState.ROTATING, TrackState.UNKNOWN):
                if is_quiet:
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
            # else STATIONARY: a non-moving frame keeps it STATIONARY; nothing to do.

        # Update the static-floor estimate from confirmed-quiet frames (and seed it during the
        # initial calibration window). Rotation displacements never enter the floor buffer, so the
        # floor tracks true sensor noise even across long runs.
        in_calibration = idx < self.calibration_frames
        if in_calibration or self.state in (TrackState.STATIONARY, TrackState.SETTLING):
            self._floor_buf.append(disp)

        return self.state

    def _set_state(self, frame_index: int, new_state: TrackState) -> None:
        if new_state != self.state:
            self.events.append((frame_index, self.state, new_state))
            self.state = new_state

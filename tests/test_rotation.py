"""Tests for flygym_tracker.rotation.RotationDetector (DESIGN.md §5.1)."""
import numpy as np
import pytest

from flygym_tracker.rotation import RotationDetector
from flygym_tracker.types import TrackState


def make_frame(value, shape=(20, 20)):
    return np.full(shape, value, dtype=np.uint8)


# ---- static metric() -------------------------------------------------------


def test_metric_static_mean_abs_diff_whole_frame():
    cur = np.full((5, 5), 10, dtype=np.uint8)
    prev = np.full((5, 5), 4, dtype=np.uint8)
    assert RotationDetector.metric(cur, prev) == pytest.approx(6.0)


def test_metric_static_uses_absdiff_no_uint8_wraparound():
    # prev > cur: plain numpy subtraction would wrap (uint8), absdiff must not.
    cur = np.full((3, 3), 4, dtype=np.uint8)
    prev = np.full((3, 3), 10, dtype=np.uint8)
    assert RotationDetector.metric(cur, prev) == pytest.approx(6.0)


def test_metric_static_respects_mask():
    cur = np.zeros((4, 4), dtype=np.uint8)
    prev = np.zeros((4, 4), dtype=np.uint8)
    cur[0, 0] = 100  # outside mask - must be ignored
    cur[2, 2] = 40   # inside mask
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 2] = True
    mask[2, 3] = True  # this pixel is unchanged (0 diff)

    # mean over masked pixels only: (40 + 0) / 2 = 20
    assert RotationDetector.metric(cur, prev, mask) == pytest.approx(20.0)


def test_metric_static_all_false_mask_returns_zero():
    cur = np.full((4, 4), 200, dtype=np.uint8)
    prev = np.zeros((4, 4), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    assert RotationDetector.metric(cur, prev, mask) == 0.0


# ---- full state machine -----------------------------------------------------


ENTER = 30.0
EXIT = 10.0
DEBOUNCE = 3
MIN_STATIONARY = 4


def _build_sequence():
    """21 constant-value frames engineered against ENTER/EXIT/DEBOUNCE/MIN_STATIONARY so every
    frame's expected TrackState is known exactly (see inline table below)."""
    quiet_a = [50] * 10          # indices 0..9
    loud = [250, 10, 230]        # indices 10..12 (each jump vs. previous frame > ENTER)
    quiet_b = [230] * 8          # indices 13..20 (equal to loud[-1] -> zero diff resumes cleanly)
    values = quiet_a + loud + quiet_b
    return [make_frame(v) for v in values]


EXPECTED_STATES = (
    [TrackState.UNKNOWN] * 3          # idx 0,1,2: no prev / quiet_streak 1,2 (<3)
    + [TrackState.SETTLING] * 4       # idx 3 (onset, streak==3) .. idx 6 (settling_count 1,2,3 <4)
    + [TrackState.STATIONARY] * 3     # idx 7 (settling_count==4) .. idx 9
    + [TrackState.ROTATING] * 5       # idx 10,11,12 (loud) + idx 13,14 (quiet but streak 1,2 <3)
    + [TrackState.SETTLING] * 4       # idx 15 (onset, streak==3) .. idx 18 (settling_count 1,2,3 <4)
    + [TrackState.STATIONARY] * 2     # idx 19 (settling_count==4) .. idx 20
)


def test_state_sequence_matches_expected_exactly():
    frames = _build_sequence()
    assert len(frames) == len(EXPECTED_STATES) == 21

    det = RotationDetector(
        enter_threshold=ENTER,
        exit_threshold=EXIT,
        debounce_frames=DEBOUNCE,
        min_stationary_frames=MIN_STATIONARY,
    )

    actual_states = [det.update(f) for f in frames]
    assert actual_states == EXPECTED_STATES


def test_state_sequence_hits_every_state_including_rotating_spike():
    frames = _build_sequence()
    det = RotationDetector(ENTER, EXIT, DEBOUNCE, MIN_STATIONARY)
    states = [det.update(f) for f in frames]

    assert states[0] == TrackState.UNKNOWN
    assert TrackState.SETTLING in states
    assert TrackState.STATIONARY in states
    # loud frames (indices 10-12) must register as ROTATING
    assert states[10] == TrackState.ROTATING
    assert states[11] == TrackState.ROTATING
    assert states[12] == TrackState.ROTATING
    # and it must recover to STATIONARY afterwards
    assert states[-1] == TrackState.STATIONARY


def test_events_capture_exact_transitions():
    frames = _build_sequence()
    det = RotationDetector(ENTER, EXIT, DEBOUNCE, MIN_STATIONARY)
    for f in frames:
        det.update(f)

    expected_events = [
        (3, TrackState.UNKNOWN, TrackState.SETTLING),
        (7, TrackState.SETTLING, TrackState.STATIONARY),
        (10, TrackState.STATIONARY, TrackState.ROTATING),
        (15, TrackState.ROTATING, TrackState.SETTLING),
        (19, TrackState.SETTLING, TrackState.STATIONARY),
    ]
    assert det.events == expected_events


def test_last_metric_tracks_most_recent_diff():
    det = RotationDetector(ENTER, EXIT, DEBOUNCE, MIN_STATIONARY)
    assert det.last_metric is None  # nothing computed yet

    det.update(make_frame(50))
    assert det.last_metric is None  # first frame: no prev to diff against

    det.update(make_frame(50))
    assert det.last_metric == pytest.approx(0.0)

    det.update(make_frame(250))
    assert det.last_metric == pytest.approx(200.0)


def test_first_frame_is_unknown_and_logs_no_event():
    det = RotationDetector(ENTER, EXIT, DEBOUNCE, MIN_STATIONARY)
    state = det.update(make_frame(123))
    assert state == TrackState.UNKNOWN
    assert det.events == []


def test_roi_mask_is_honored_by_stateful_detector():
    shape = (10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[0:3, 0:3] = True  # only a small corner is "illuminated"

    det = RotationDetector(ENTER, EXIT, DEBOUNCE, MIN_STATIONARY, roi_mask=mask)

    base = np.full(shape, 50, dtype=np.uint8)
    det.update(base)  # seed

    # Large change OUTSIDE the mask only -> should NOT trigger ROTATING.
    outside_change = base.copy()
    outside_change[5:9, 5:9] = 255
    state = det.update(outside_change)
    assert state != TrackState.ROTATING
    assert det.last_metric == pytest.approx(0.0)

    # Large change INSIDE the mask -> SHOULD trigger ROTATING.
    inside_change = outside_change.copy()
    inside_change[0:3, 0:3] = 255
    state = det.update(inside_change)
    assert state == TrackState.ROTATING


# ---- hysteresis mid-band edge cases -----------------------------------------


def test_mid_band_frame_does_not_rotate_and_resets_debounce_streak():
    """A frame with EXIT <= m <= ENTER is ambiguous: it must not trigger ROTATING, but it must
    also break an in-progress quiet streak (so debounce truly requires *consecutive* quiet
    frames)."""
    det = RotationDetector(ENTER, EXIT, DEBOUNCE, MIN_STATIONARY)
    det.update(make_frame(0))          # seed (UNKNOWN)
    det.update(make_frame(0))          # quiet_streak=1
    det.update(make_frame(0))          # quiet_streak=2

    mid = make_frame(0 + int((ENTER + EXIT) / 2))  # inside (EXIT, ENTER) band
    state = det.update(mid)
    assert state == TrackState.UNKNOWN  # not enough to rotate, not yet settled
    assert det._quiet_streak == 0  # streak was reset by the mid-band frame

    # Confirm the streak really was reset: two more quiet frames are NOT enough (need DEBOUNCE=3).
    det.update(mid)  # quiet vs previous(mid) -> diff 0, quiet_streak=1
    state = det.update(mid)  # quiet_streak=2
    assert state == TrackState.UNKNOWN


def test_mid_band_frame_during_stationary_does_not_demote():
    frames = _build_sequence()
    det = RotationDetector(ENTER, EXIT, DEBOUNCE, MIN_STATIONARY)
    for f in frames[:10]:  # drive it to STATIONARY (reached at index 7, holds through 9)
        state = det.update(f)
    assert state == TrackState.STATIONARY

    # A mid-band jump from the STATIONARY baseline (230->? no, quiet_a ended at 50) — use current
    # last value (50) as baseline and move into the mid-band, well below ENTER.
    mid = make_frame(50 + int((ENTER + EXIT) / 2))
    state = det.update(mid)
    assert state == TrackState.STATIONARY  # must not drop out of STATIONARY

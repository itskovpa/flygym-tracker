"""Tests for flygym_tracker.adaptive_rotation.AdaptiveRotationDetector (DESIGN.md §5.1, adaptive).

The point of this detector is SPEED-INDEPENDENCE: it keys on the global inter-frame DISPLACEMENT of
the rig's rigid structure (via cv2.phaseCorrelate) plus the DIRECTIONAL CONSISTENCY of that shift,
NOT on the magnitude of pixel change (mean|diff|) that a preset-threshold detector uses. So it must:
  (a) detect a FAST global shift (8 px/frame),
  (b) STILL detect a SLOW steady drift (1 px/frame) that a fixed-magnitude detector would miss,
  (c) NOT fire on localized fly motion (a small patch changing incoherently -> no global shift),
  (d) NOT fire on an in-place LED flicker (intensity pulses, no shift).
All scenarios are deterministic (seeded RNG + cv2 warps).
"""
import numpy as np
import cv2
import pytest

from flygym_tracker.adaptive_rotation import AdaptiveRotationDetector
from flygym_tracker.rotation import RotationDetector
from flygym_tracker.types import TrackState


H, W = 120, 160


# --------------------------------------------------------------------------- builders

def rig_structure(seed=0):
    """A rig-like frame with STRONG high-contrast rigid structure (bright back-light, dark tube
    walls = sharp vertical edges, dark rails, a blinding central LED band). This structure is what
    phase-correlation locks onto; it dominates over any localized fly motion, exactly as on the real
    rig (where dwell frames stay at displacement ~0.01 px even with flies moving)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 200.0, np.float32)
    for cx in range(12, W, 20):
        img[:, cx:cx + 3] = 30.0            # dark tube walls -> strong vertical edges
    img[0:6, :] = 40.0
    img[H - 6:H, :] = 40.0                    # dark top/bottom rails
    img[56:64, :] = 255.0                     # central LED-frame band (blinding)
    img += rng.normal(0, 6, (H, W)).astype(np.float32)  # fixed-pattern texture
    return np.clip(img, 0, 255).astype(np.uint8)


def smooth_structure(seed=0):
    """A SMOOTH, low-contrast structure. A slow (1 px) shift of it produces a mean|diff| barely above
    the static floor -- the regime where a fixed-magnitude threshold is blind but displacement is
    not. Used to demonstrate speed-independence vs. RotationDetector."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 256, (H, W)).astype(np.float32), (0, 0), 3.0)
    base = 128 + (base - base.mean()) * 0.5   # low contrast
    return np.clip(base, 0, 255).astype(np.uint8)


def noisy(img, rng, sigma=1.0):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def rotation_scenario(bg, rate, n_static=14, n_move=12, seed=100, sigma=1.0):
    """static hold -> `n_move` frames each shifted a further `rate` px (a rotation at that speed) ->
    static hold at the final pose. Returns (frames, move_start_idx, move_end_idx)."""
    rng = np.random.default_rng(seed)
    frames = [noisy(bg, rng, sigma) for _ in range(n_static)]
    move_start = len(frames)
    for k in range(1, n_move + 1):
        frames.append(noisy(shift(bg, rate * k, 0), rng, sigma))
    move_end = len(frames) - 1
    final = shift(bg, rate * n_move, 0)
    frames += [noisy(final, rng, sigma) for _ in range(n_static)]
    return frames, move_start, move_end


def run(det, frames):
    return [det.update(f) for f in frames]


def has_rotating(states):
    return TrackState.ROTATING in states


# --------------------------------------------------------------------------- constructor / interface

def test_constructor_requires_no_magnitude_thresholds():
    # Must be constructible with NO thresholds at all (the whole premise) and as a positional
    # drop-in for RotationDetector's shared (roi_mask/debounce/min_stationary) trailing args.
    det = AdaptiveRotationDetector()
    assert det.state == TrackState.UNKNOWN
    assert det.last_metric is None
    assert det.events == []
    # signature has no required enter/exit; defaults are sane
    assert det.debounce_frames == 4
    assert det.min_stationary_frames == 3


def test_constructor_validates_params():
    with pytest.raises(ValueError):
        AdaptiveRotationDetector(debounce_frames=0)
    with pytest.raises(ValueError):
        AdaptiveRotationDetector(min_stationary_frames=0)
    with pytest.raises(ValueError):
        AdaptiveRotationDetector(sensitivity=0)
    with pytest.raises(ValueError):
        AdaptiveRotationDetector(window_frames=0)
    with pytest.raises(ValueError):
        AdaptiveRotationDetector(min_consistency=1.5)


def test_first_frame_is_unknown_and_logs_no_event():
    det = AdaptiveRotationDetector()
    state = det.update(rig_structure(0))
    assert state == TrackState.UNKNOWN
    assert det.events == []
    assert det.last_metric is None  # no prev frame to diff against yet


def test_update_returns_trackstate_and_exposes_diagnostics():
    det = AdaptiveRotationDetector()
    bg = rig_structure(0)
    rng = np.random.default_rng(1)
    det.update(noisy(bg, rng))
    st = det.update(noisy(bg, rng))
    assert isinstance(st, TrackState)
    # primary signal recorded, plus diagnostics
    assert det.last_metric is not None and det.last_metric == det.last_disp
    assert det.last_response is not None
    assert det.last_accum is not None
    assert det.enter_threshold is not None and det.exit_threshold is not None
    assert det.last_metric >= 0.0


# --------------------------------------------------------------------------- static helpers

def test_displacement_helper_identical_and_shifted():
    bg = rig_structure(2)
    disp0, dx0, dy0, resp0 = AdaptiveRotationDetector.displacement(bg, bg)
    assert disp0 < 0.02          # identical frames -> ~0 displacement
    assert resp0 > 0.9           # ...and a confident peak

    disp5, dx5, dy5, resp5 = AdaptiveRotationDetector.displacement(shift(bg, 5, 0), bg)
    assert disp5 == pytest.approx(5.0, abs=0.5)   # recovers the 5 px shift magnitude
    assert abs(dx5) == pytest.approx(5.0, abs=0.5)


def test_metric_matches_rotationdetector_meandiff():
    # Secondary-cue parity with RotationDetector.metric (drop-in): identical numeric mean|diff|.
    cur = np.full((5, 5), 10, dtype=np.uint8)
    prev = np.full((5, 5), 4, dtype=np.uint8)
    assert AdaptiveRotationDetector.metric(cur, prev) == pytest.approx(6.0)
    assert AdaptiveRotationDetector.metric(cur, prev) == RotationDetector.metric(cur, prev)
    # with a mask
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    assert AdaptiveRotationDetector.metric(cur, prev, mask) == RotationDetector.metric(cur, prev, mask)


# --------------------------------------------------------------------------- (a) FAST rotation

def test_fast_rotation_detected_and_recovers():
    bg = rig_structure(0)
    frames, ms, me = rotation_scenario(bg, rate=8.0)
    det = AdaptiveRotationDetector()
    states = run(det, frames)

    assert has_rotating(states)
    # ROTATING somewhere within the moving block
    assert any(states[i] == TrackState.ROTATING for i in range(ms, me + 1))
    # recovers to STATIONARY after the drum stops
    assert states[-1] == TrackState.STATIONARY
    # emits an enter-rotation and a leave-rotation transition (pipeline -> rotation_start/end)
    to_rot = [e for e in det.events if e[2] == TrackState.ROTATING]
    from_rot = [e for e in det.events if e[1] == TrackState.ROTATING]
    assert len(to_rot) >= 1 and len(from_rot) >= 1


# --------------------------------------------------------------------------- (b) SLOW rotation

def test_slow_rotation_still_detected():
    """1 px/frame steady drift -- a fixed-magnitude detector's whole failure mode -- must still be
    flagged as rotation."""
    bg = rig_structure(0)
    frames, ms, me = rotation_scenario(bg, rate=1.0)
    det = AdaptiveRotationDetector()
    states = run(det, frames)

    assert has_rotating(states), "slow 1px/frame drift must be detected as rotation"
    assert any(states[i] == TrackState.ROTATING for i in range(ms, me + 1))
    assert states[-1] == TrackState.STATIONARY


def test_speed_independence_preset_detector_misses_slow_but_adaptive_catches():
    """Direct proof of the point. On a smooth structure, a SLOW drift's mean|diff| sits just above the
    static floor, so a RotationDetector tuned (generously) for the rig's FAST rotations is blind to it
    -- while AdaptiveRotationDetector, keying on displacement+consistency, catches BOTH."""
    bg = smooth_structure(0)
    fast_frames, _, _ = rotation_scenario(bg, rate=8.0, sigma=0.6)
    slow_frames, sms, sme = rotation_scenario(bg, rate=1.0, sigma=0.6)

    # Characterise mean|diff| on this structure.
    def meandiff(frames, a, b):
        return float(np.median([RotationDetector.metric(frames[i], frames[i - 1]) for i in range(a, b)]))

    static_floor = meandiff(fast_frames, 1, 13)          # quiet frames
    fast_move = meandiff(fast_frames, 15, 26)            # fast rotation frames
    slow_move = meandiff(slow_frames, sms + 2, sme)      # slow rotation frames

    # A preset detector must sit well above the static floor to avoid false triggers; tune it
    # generously for the rig's fast rotations (enter halfway between floor and the fast level).
    enter = 0.5 * (static_floor + fast_move)
    exit_ = 0.5 * (static_floor + enter)
    assert slow_move < enter, "test premise: slow drift's mean|diff| is below a fast-tuned threshold"

    preset = RotationDetector(enter, exit_, debounce_frames=4, min_stationary_frames=3)
    preset_states = run(preset, slow_frames)
    adaptive = AdaptiveRotationDetector()
    adaptive_states = run(adaptive, slow_frames)

    assert not has_rotating(preset_states), "fixed-magnitude detector misses the slow rotation"
    assert has_rotating(adaptive_states), "adaptive detector catches the slow rotation"


def test_very_slow_subpixel_drift_detected_via_accumulation():
    """A 0.4 px/frame drift can sit inside the instantaneous displacement noise band; the directional
    ACCUMULATION over the window is what catches it (steady drift accumulates; jitter cancels)."""
    bg = rig_structure(0)
    frames, ms, me = rotation_scenario(bg, rate=0.4, n_move=16, sigma=1.0)
    det = AdaptiveRotationDetector()
    states = run(det, frames)
    assert has_rotating(states)


# --------------------------------------------------------------------------- (c) flies, (d) LED

def test_localized_fly_motion_does_not_trigger():
    """Dark fly silhouettes churning inside one vial column: a small localized patch whose pixels
    change incoherently each frame (flies come, go, and shuffle). This changes many pixels
    (non-trivial mean|diff|) but produces NO coherent global shift and NO consistent direction ->
    must stay STATIONARY. This is exactly where a magnitude detector would false-trigger."""
    bg = rig_structure(0)
    rng = np.random.default_rng(7)
    py0, py1, px0, px1 = 16, 40, 70, 84       # one vial column
    frames = []
    for i in range(34):
        f = noisy(bg, rng, 1.0)
        if 10 <= i < 26:  # flies active in the middle stretch
            patch = f[py0:py1, px0:px1].astype(np.int32)
            patch += rng.integers(-50, 1, patch.shape)   # incoherent darkening (flies), no shift
            f[py0:py1, px0:px1] = np.clip(patch, 0, 255)
        frames.append(f)

    det = AdaptiveRotationDetector()
    states = run(det, frames)

    assert not has_rotating(states), "localized incoherent fly motion must not read as rotation"
    # sanity: the fly frames really do change pixels (a magnitude detector would be tempted)
    md = np.median([RotationDetector.metric(frames[i], frames[i - 1]) for i in range(11, 26)])
    assert md > 0.8


def test_led_flicker_band_does_not_trigger():
    """A central band pulses bright/dark in place (LED through frame hardware). Large intensity
    change, ZERO spatial shift -> must stay STATIONARY."""
    bg = rig_structure(1)
    rng = np.random.default_rng(9)
    frames = []
    for i in range(34):
        f = noisy(bg, rng, 1.0).astype(np.int32)
        f[56:64, :] = np.clip(f[56:64, :] + (60 if i % 2 == 0 else -60), 0, 255)
        frames.append(f.astype(np.uint8))

    det = AdaptiveRotationDetector()
    states = run(det, frames)

    assert not has_rotating(states), "in-place LED flicker must not read as rotation"
    md = np.median([RotationDetector.metric(frames[i], frames[i - 1]) for i in range(1, 34)])
    assert md > 2.0  # big magnitude change, yet no rotation flagged


# --------------------------------------------------------------------------- events / state machine

def test_events_are_frameindex_from_to_transitions():
    bg = rig_structure(0)
    frames, _, _ = rotation_scenario(bg, rate=8.0)
    det = AdaptiveRotationDetector()
    run(det, frames)

    assert len(det.events) >= 2
    prev_idx = -1
    for idx, frm, to in det.events:
        assert isinstance(idx, int)
        assert isinstance(frm, TrackState) and isinstance(to, TrackState)
        assert frm != to                       # only real transitions logged
        assert idx > prev_idx                  # monotonic frame indices
        prev_idx = idx
    # the log must contain the rotation entry and exit, in that order
    kinds = [(e[1], e[2]) for e in det.events]
    i_enter = next(k for k, (a, b) in enumerate(kinds) if b == TrackState.ROTATING)
    i_leave = next(k for k, (a, b) in enumerate(kinds) if a == TrackState.ROTATING)
    assert i_enter < i_leave


def test_settling_reported_for_min_stationary_frames_before_stationary():
    bg = rig_structure(0)
    frames, _, _ = rotation_scenario(bg, rate=8.0, n_static=20)
    det = AdaptiveRotationDetector(min_stationary_frames=5)
    states = run(det, frames)
    # after the rotation there must be exactly one SETTLING run of length 5, then STATIONARY
    # find the LAST settling run
    runs = []
    i = 0
    while i < len(states):
        if states[i] == TrackState.SETTLING:
            j = i
            while j + 1 < len(states) and states[j + 1] == TrackState.SETTLING:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    assert runs, "expected a SETTLING phase"
    s, e = runs[-1]
    assert e - s + 1 == 5                       # min_stationary_frames settling frames
    assert states[e + 1] == TrackState.STATIONARY


# --------------------------------------------------------------------------- roi_mask / adaptivity / determinism

def test_roi_mask_accepted_and_rotation_detected_through_it():
    mask = np.zeros((H, W), dtype=bool)
    mask[10:110, 20:140] = True               # central ROI over the structure
    bg = rig_structure(0)
    frames, ms, me = rotation_scenario(bg, rate=8.0)
    det = AdaptiveRotationDetector(roi_mask=mask)
    states = run(det, frames)
    assert det.roi_mask is mask
    assert has_rotating(states)


def test_adaptive_floor_estimated_from_quiet_frames():
    bg = rig_structure(0)
    rng = np.random.default_rng(3)
    det = AdaptiveRotationDetector(calibration_frames=20)
    for _ in range(25):
        det.update(noisy(bg, rng, 1.0))
    # after a run of quiet frames the online floor is small and the enter threshold is finite/small
    assert det.floor_center is not None
    assert det.floor_center < 0.15
    assert 0.0 < det.enter_threshold < 1.0
    # never falsely rotated on pure static noise
    assert det.state in (TrackState.SETTLING, TrackState.STATIONARY)


def test_higher_sensitivity_lowers_enter_threshold():
    bg = rig_structure(0)
    rng1 = np.random.default_rng(4)
    rng2 = np.random.default_rng(4)
    low = AdaptiveRotationDetector(sensitivity=0.5)
    high = AdaptiveRotationDetector(sensitivity=2.0)
    for _ in range(15):
        low.update(noisy(bg, rng1, 1.0))
        high.update(noisy(bg, rng2, 1.0))
    assert high.enter_threshold < low.enter_threshold


def test_deterministic_same_input_same_events():
    bg = rig_structure(0)
    frames, _, _ = rotation_scenario(bg, rate=6.0)
    d1 = AdaptiveRotationDetector()
    d2 = AdaptiveRotationDetector()
    run(d1, frames)
    run(d2, frames)
    assert d1.events == d2.events
    assert d1.state == d2.state

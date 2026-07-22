"""Integration tests for flygym_tracker.pipeline (DESIGN.md §3, §5.1-§5.3, §7).

Everything runs against a FAKE in-memory FrameSource (defined below) and a hand-built synthetic
scene: two vials, a small illum mask, a textured background, a moving block in vial 1, a burst of
loud global-motion "rotation" frames, then a different quiet background. Thresholds are passed
explicitly so every state transition is deterministic.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import pytest

from flygym_tracker.config import load_config
from flygym_tracker.frame_source import FrameSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.pipeline import TrackerPipeline, measure_noise
from flygym_tracker.types import (


    ACTIVITY_COLUMNS,
    Calibration,
    FaceCalibration,
    Frame,
    VialROI,
)


def _one(directory, pattern):
    """The single file matching `pattern` in `directory`.

    OUTPUT FILES CARRY THE RUN'S START STAMP now (`events_20260720-142233.csv`), so a test cannot
    spell the name. Globbing keeps the test about the CONTENT, which is what it was ever checking.
    """
    import pathlib

    matches = sorted(pathlib.Path(directory).glob(pattern))
    assert matches, "no file matching %r in %s" % (pattern, directory)
    return matches[0]



# =============================================================================================
# Fake in-memory frame source
# =============================================================================================
class FakeSource(FrameSource):
    """Scripted list of grayscale ndarrays served as `Frame`s.

    `raise_on_calls` is a set of 1-based `read()` call ordinals that raise instead of returning a
    frame (to exercise the pipeline's transient-read tolerance). A raise does NOT consume a frame,
    mirroring a camera hiccup where the same next frame is retried.
    """

    def __init__(self, frames: List[np.ndarray], fps: float = 10.0,
                 raise_on_calls: Optional[set] = None):
        self._frames = frames
        self._fps = float(fps)
        self._i = 0
        self._read_calls = 0
        self._raise_on = set(raise_on_calls or ())
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read(self) -> Optional[Frame]:
        self._read_calls += 1
        if self._read_calls in self._raise_on:
            raise IOError(f"simulated transient read error on call {self._read_calls}")
        if self._i >= len(self._frames):
            return None
        img = self._frames[self._i]
        idx = self._i
        self._i += 1
        return Frame(image=img, index=idx, t_monotonic=float(idx),
                     t_wall_iso="2026-07-18T00:00:00")

    def close(self) -> None:
        self.closed = True

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self):
        h, w = self._frames[0].shape[:2]
        return (w, h)


# =============================================================================================
# Synthetic scene
# =============================================================================================
H = W = 40
# Vial 1: x2..12, y2..12 (10x10). Vial 2: x20..30, y2..12 (10x10).
V1 = dict(id=1, row=0, col=0, x=2, y=2, w=10, h=10)
V2 = dict(id=2, row=0, col=1, x=20, y=2, w=10, h=10)
BLOCK = (slice(4, 6), slice(4, 6))   # 2x2 moving block inside vial 1
BLOCK_LO, BLOCK_HI = 40, 220         # toggled block values (delta 180 >> pixel_threshold)

ENTER, EXIT, PIXEL_THR = 40.0, 15.0, 30.0
DEBOUNCE, MIN_STATIONARY = 2, 2
FPS, BIN_SECONDS = 10.0, 1.0


def _illum_mask() -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    m[2:12, 2:12] = 255
    m[2:12, 20:30] = 255
    return m


def _background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.zeros((H, W), np.uint8)
    f[2:12, 2:12] = rng.integers(60, 160, size=(10, 10), dtype=np.uint8)
    f[2:12, 20:30] = rng.integers(60, 160, size=(10, 10), dtype=np.uint8)
    return f


P = _background(1)          # pre-rotation background
Q = _background(2)          # post-rotation background (deliberately different from P)


def _pre_frame(block_val: int) -> np.ndarray:
    f = P.copy()
    f[BLOCK] = block_val
    return f


def _loud_frame(val: int) -> np.ndarray:
    f = np.zeros((H, W), np.uint8)
    f[2:12, 2:12] = val
    f[2:12, 20:30] = val
    return f


def _full_scene() -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    # Phase 1 (idx 0..9, bin 0): quiet, block toggles each frame -> vial 1 motion, vial 2 none.
    for i in range(10):
        frames.append(_pre_frame(BLOCK_LO if i % 2 == 0 else BLOCK_HI))
    # Phase 2 (idx 10..19, bin 1): loud global motion -> ROTATING.
    for i in range(10):
        frames.append(_loud_frame(0 if i % 2 == 0 else 255))
    # Phase 3 (idx 20..29, bin 2): different quiet background, no moving block.
    for _ in range(10):
        frames.append(Q.copy())
    return frames


def _calibration(tmp_path) -> Calibration:
    mask_png = tmp_path / "illum_mask_A.png"
    cv2.imwrite(str(mask_png), _illum_mask())
    vials = [VialROI(present=True, **V1), VialROI(present=True, **V2)]
    fc = FaceCalibration(name="A", vials=vials, illum_mask_path=str(mask_png), marker=None)
    return Calibration(image_width=W, image_height=H, faces={"A": fc}, created="", notes="")


def _config():
    return load_config(overrides={
        "rotation": {
            "enter_threshold": ENTER, "exit_threshold": EXIT,
            "debounce_frames": DEBOUNCE, "min_stationary_frames": MIN_STATIONARY,
        },
        "activity": {"pixel_threshold": PIXEL_THR},
        "binning": {"bin_seconds": BIN_SECONDS},
    })


def _pipeline(tmp_path, frames, *, reference=True, raise_on_calls=None, marker_detector=None):
    calib = _calibration(tmp_path)
    config = _config()
    logger = ActivityLogger(output_dir=tmp_path, run_id="test_run", fmt="csv")
    source = FakeSource(frames, fps=FPS, raise_on_calls=raise_on_calls)
    ref_frames = {"A": P.copy()} if reference else None
    pipe = TrackerPipeline(
        config, calib, source, logger,
        marker_detector=marker_detector, reference_frames=ref_frames,
        clock="index", read_retry_sleep=0.0,
    )
    return pipe


# ISO helpers: wall-start is the fake frame's t_wall_iso; bins offset by bin_start_s.
BIN0_ISO = "2026-07-18T00:00:00"
BIN1_ISO = "2026-07-18T00:00:01"
BIN2_ISO = "2026-07-18T00:00:02"


def _events(tmp_path) -> pd.DataFrame:
    return pd.read_csv(_one(tmp_path, "events_*.csv"), keep_default_na=False)


def _activity(tmp_path) -> pd.DataFrame:
    return pd.read_csv(_one(tmp_path, "activity_*_20260718.csv"))


def _row(df, vial_id, iso):
    sub = df[(df["vial_id"] == vial_id) & (df["bin_start_iso"] == iso)]
    assert len(sub) == 1, f"expected exactly one row for vial {vial_id} @ {iso}, got {len(sub)}"
    return sub.iloc[0]


# =============================================================================================
# End-to-end
# =============================================================================================
def test_end_to_end_events_activity_and_baseline_reset(tmp_path):
    pipe = _pipeline(tmp_path, _full_scene())
    summary = pipe.run()

    # -- summary ------------------------------------------------------------------------------
    assert summary["frames_processed"] == 30
    assert summary["n_rotations"] == 1
    assert summary["n_bins"] == 3
    assert summary["faces_seen"] == ["A"]

    # -- events: rotation_start + rotation_end emitted, exactly one each; face defaulted twice --
    ev = _events(tmp_path)
    kinds = ev["event"].tolist()
    assert kinds.count("rotation_start") == 1
    assert kinds.count("rotation_end") == 1
    assert kinds.count("marker_absent") == 2          # one per stationary onset (initial + post-rot)
    assert kinds.count("face_change") == 0            # single face, never changes
    # rotation_start precedes rotation_end, and start is at the first loud frame (elapsed 1.0s).
    start = ev[ev["event"] == "rotation_start"].iloc[0]
    end = ev[ev["event"] == "rotation_end"].iloc[0]
    assert start["elapsed_s"] == pytest.approx(10 / FPS)   # idx 10
    assert end["elapsed_s"] > start["elapsed_s"]

    # -- activity schema ----------------------------------------------------------------------
    df = _activity(tmp_path)
    assert list(df.columns) == ACTIVITY_COLUMNS
    assert set(df["face"]) == {"A"}
    assert set(df["vial_id"]) == {1, 2}               # face A -> global id == local id
    assert bool(df["present"].all())

    # -- vial 1 accumulates motion in bin 0; vial 2 ~zero -------------------------------------
    v1_b0 = _row(df, 1, BIN0_ISO)
    v2_b0 = _row(df, 2, BIN0_ISO)
    assert v1_b0["motion_px_sum"] == 20               # 5 stationary compute frames x 4 px
    assert v1_b0["n_stationary_frames"] == 5
    assert v1_b0["lit_area_px"] == 100
    assert v1_b0["row"] == 0 and v1_b0["col"] == 0
    assert v2_b0["motion_px_sum"] == 0                # vial 2 background static
    assert v2_b0["col"] == 1

    # -- baseline reset across the rotation: first post-rotation stationary frame does NOT diff
    #    against a pre-rotation frame, so the post-rotation bin shows ~no spurious motion even
    #    though the background changed completely (P -> Q). Without the reset this would be huge.
    v1_b2 = _row(df, 1, BIN2_ISO)
    v2_b2 = _row(df, 2, BIN2_ISO)
    assert v1_b2["motion_px_sum"] == 0
    assert v2_b2["motion_px_sum"] == 0
    assert v1_b2["n_stationary_frames"] == 5
    assert v1_b2["n_rotating_frames"] == 2            # idx 20,21 counted as rotating in bin 2

    # -- the all-rotation bin 1 is captured as rotating-only, no stationary frames -------------
    # Motion/activity here are a GAP, not a measured zero: with zero stationary frames the vial was
    # not observed this bin (the drum was rotating, or showing the other face). Emitting 0 made each
    # face's curve dip to the floor every time the other face was up; None -> blank -> NaN is the
    # honest "no measurement", and n_rotating_frames still records that the drum passed through.
    v1_b1 = _row(df, 1, BIN1_ISO)
    assert v1_b1["n_stationary_frames"] == 0
    assert v1_b1["n_rotating_frames"] == 10
    assert pd.isna(v1_b1["motion_px_sum"]), "an unobserved bin must be a gap, not a measured zero"
    assert pd.isna(v1_b1["active_fraction_mean"])


def test_baseline_reset_is_load_bearing(tmp_path):
    """Sanity that the P->Q background swap really would light up without a reset: a direct diff of
    the last pre-rotation stationary frame vs the first post-rotation frame is massive. This is the
    spurious motion the reset suppresses (asserted ==0 above)."""
    last_pre = _pre_frame(BLOCK_HI)        # ~ idx 9
    first_post = Q.copy()                  # ~ idx 24
    mask = _illum_mask() == 255
    diff = cv2.absdiff(first_post, last_pre)
    spurious = int(np.count_nonzero((diff > PIXEL_THR) & mask))
    assert spurious > 100                  # would swamp the true signal if diffed across rotation


# =============================================================================================
# Registration reference: adopt first stationary frame when none supplied
# =============================================================================================
def test_registration_reference_adopted_when_absent(tmp_path):
    pipe = _pipeline(tmp_path, _full_scene(), reference=False)
    summary = pipe.run()
    # Still produces a full, well-formed run; vial 1 motion still detected in bin 0.
    assert summary["frames_processed"] == 30
    df = _activity(tmp_path)
    assert _row(df, 1, BIN0_ISO)["motion_px_sum"] == 20


# =============================================================================================
# Read-error tolerance
# =============================================================================================
def _quiet_scene(n=12):
    # Enough constant quiet frames to reach STATIONARY; no motion needed for control tests.
    return [_pre_frame(BLOCK_LO) for _ in range(n)]


def test_single_read_error_is_retried_no_frame_lost(tmp_path):
    # Call #3 raises once; the retry immediately succeeds -> no frame lost, no read_error logged.
    pipe = _pipeline(tmp_path, _quiet_scene(12), raise_on_calls={3})
    summary = pipe.run()
    assert summary["frames_processed"] == 12
    assert summary["frames_read_errors"] == 0
    ev = _events(tmp_path)
    assert (ev["event"] == "read_error").sum() == 0


def test_exhausted_read_retries_logs_read_error_and_continues(tmp_path):
    # read_retries defaults to 3; make all 3 attempts of one logical frame raise (calls 3,4,5).
    pipe = _pipeline(tmp_path, _quiet_scene(12), raise_on_calls={3, 4, 5})
    summary = pipe.run()
    assert summary["frames_read_errors"] == 1
    assert summary["frames_processed"] == 12          # fake retries same frame, so none truly lost
    ev = _events(tmp_path)
    assert (ev["event"] == "read_error").sum() == 1


# =============================================================================================
# Loop control: max_frames + stop_flag
# =============================================================================================
def test_max_frames_limits_processing(tmp_path):
    pipe = _pipeline(tmp_path, _full_scene())
    summary = pipe.run(max_frames=5)
    assert summary["frames_processed"] == 5
    assert summary["stopped_reason"] == "max_frames"


def test_stop_flag_callable_stops_gracefully(tmp_path):
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 6      # allow ~6 iterations then stop

    pipe = _pipeline(tmp_path, _full_scene())
    summary = pipe.run(stop_flag=stop)
    assert summary["stopped_reason"] == "stop_flag"
    assert summary["frames_processed"] <= 6


def test_stop_flag_event_like_object(tmp_path):
    class Ev:
        def __init__(self):
            self._set = False

        def is_set(self):
            return self._set

    ev = Ev()
    ev._set = True                  # already set -> stop before processing anything
    pipe = _pipeline(tmp_path, _full_scene())
    summary = pipe.run(stop_flag=ev)
    assert summary["frames_processed"] == 0
    assert summary["stopped_reason"] == "stop_flag"


# =============================================================================================
# Marker detector duck-typing
# =============================================================================================
def test_marker_detector_returning_face_a_logs_no_marker_absent(tmp_path):
    class MarkerA:
        def identify_face(self, gray):
            return "A"

    pipe = _pipeline(tmp_path, _full_scene(), marker_detector=MarkerA())
    pipe.run()
    ev = _events(tmp_path)
    assert (ev["event"] == "marker_absent").sum() == 0     # detector supplied a face
    assert (ev["event"] == "face_change").sum() == 0       # it's the same (default) face


# =============================================================================================
# measure_noise
# =============================================================================================
def test_measure_noise_returns_sensible_thresholds():
    rng = np.random.default_rng(7)
    base = np.zeros((H, W), np.uint8)
    base[2:12, 2:12] = 120
    base[2:12, 20:30] = 120
    mask = _illum_mask()

    # 20 static frames with small independent per-pixel noise (amplitude ~3) inside the mask.
    frames = []
    for _ in range(20):
        f = base.copy()
        noise = rng.integers(-3, 4, size=(H, W)).astype(np.int16)
        noisy = np.clip(f.astype(np.int16) + noise * (mask > 0), 0, 255).astype(np.uint8)
        frames.append(noisy)

    src = FakeSource(frames, fps=FPS)
    with src:
        res = measure_noise(src, mask, n_frames=100, k=5.0)

    assert res["n_frames"] == 20
    assert res["n_pairs"] == 19
    assert res["noise_std"] > 0
    # pixel threshold is exactly mean + k*std of the per-pixel distribution.
    assert res["suggested_pixel_threshold"] == pytest.approx(res["noise_mean"] + 5.0 * res["noise_std"])
    # enter/exit come from the per-frame metric distribution, with hysteresis enter > exit >= mean.
    assert res["suggested_enter_threshold"] > res["suggested_exit_threshold"] > res["metric_mean"]
    # thresholds clear the noise floor.
    assert res["suggested_pixel_threshold"] > res["noise_mean"]


def test_measure_noise_needs_two_frames():
    one = [np.full((H, W), 100, np.uint8)]
    src = FakeSource(one)
    with pytest.raises(ValueError):
        measure_noise(src, _illum_mask(), n_frames=100)


# =============================================================================================
# Threshold resolution: null thresholds must be provided, not invented
# =============================================================================================
def test_null_pixel_threshold_raises_without_override(tmp_path):
    calib = _calibration(tmp_path)
    config = load_config()  # defaults: pixel_threshold / enter / exit are all null
    logger = ActivityLogger(output_dir=tmp_path, run_id="r", fmt="csv")
    with pytest.raises(ValueError):
        TrackerPipeline(config, calib, FakeSource(_quiet_scene()), logger)
    logger.close()

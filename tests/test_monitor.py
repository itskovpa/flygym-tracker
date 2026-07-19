"""Tests for flygym_tracker.monitor (LiveMonitor) + the pipeline.py observer hooks it's built on.

The pipeline-integration tests reuse the same proven synthetic scene design as
tests/test_pipeline.py (quiet frames with a moving block in vial 1 only, then loud global-motion
"rotation" frames, then a different quiet background) so the exact per-vial/per-bin numbers are
known -- this file duplicates a trimmed copy of that scaffolding locally, matching this repo's
convention that each test file is self-contained (see test_pipeline.py, test_pipeline_adaptive_mode.py,
test_cli.py -- none of them import fixtures from one another).

LiveMonitor tests never trigger a real cv2 window: `auto_render=False` disables the automatic
`maybe_render()` call `on_frame` would otherwise make, `render_composite()` (and the `_render_*`
panel helpers) never touch a cv2 window/display API at all, and the one test that exercises the
headless-degradation path monkeypatches `cv2.imshow` to raise *before* any real display call can
happen.
"""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import pytest

from flygym_tracker.config import load_config
from flygym_tracker.frame_source import FrameSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.monitor import LiveMonitor
from flygym_tracker.pipeline import TrackerPipeline
from flygym_tracker.types import Calibration, FaceCalibration, Frame, TrackState, VialROI


# =============================================================================================
# Fake in-memory frame source + synthetic scene (trimmed copy of tests/test_pipeline.py's design)
# =============================================================================================
class FakeSource(FrameSource):
    def __init__(self, frames: List[np.ndarray], fps: float = 10.0):
        self._frames = frames
        self._fps = float(fps)
        self._i = 0

    def open(self) -> None:
        pass

    def read(self) -> Optional[Frame]:
        if self._i >= len(self._frames):
            return None
        img = self._frames[self._i]
        idx = self._i
        self._i += 1
        return Frame(image=img, index=idx, t_monotonic=float(idx), t_wall_iso="2026-07-19T00:00:00")

    def close(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self):
        h, w = self._frames[0].shape[:2]
        return (w, h)


H = W = 40
V1 = dict(id=1, row=0, col=0, x=2, y=2, w=10, h=10)
V2 = dict(id=2, row=0, col=1, x=20, y=2, w=10, h=10)
BLOCK = (slice(4, 6), slice(4, 6))
BLOCK_LO, BLOCK_HI = 40, 220

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


P = _background(1)   # pre-rotation background
Q = _background(2)   # post-rotation background (deliberately different from P)


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
    for i in range(10):                                     # bin 0: quiet, block toggles -> vial 1 motion
        frames.append(_pre_frame(BLOCK_LO if i % 2 == 0 else BLOCK_HI))
    for i in range(10):                                      # bin 1: loud global motion -> ROTATING
        frames.append(_loud_frame(0 if i % 2 == 0 else 255))
    for _ in range(10):                                      # bin 2: different quiet background
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


def _pipeline(tmp_path, out_subdir="out"):
    calib = _calibration(tmp_path)
    config = _config()
    logger = ActivityLogger(output_dir=tmp_path / out_subdir, run_id="test_run", fmt="csv")
    source = FakeSource(_full_scene(), fps=FPS)
    pipe = TrackerPipeline(
        config, calib, source, logger,
        reference_frames={"A": P.copy()}, clock="index", read_retry_sleep=0.0,
    )
    return pipe, calib, config


# =============================================================================================
# pipeline.py: observer hooks
# =============================================================================================
def test_frame_observer_receives_documented_keys_every_frame(tmp_path):
    pipe, _calib, _config = _pipeline(tmp_path)
    payloads = []
    pipe.add_observer(payloads.append)
    summary = pipe.run()

    assert summary["frames_processed"] == 30
    assert len(payloads) == 30

    required_keys = {"frame", "index", "elapsed_s", "state", "face", "vial_results",
                      "n_rotations", "fps_est", "pixel_threshold"}
    seen_states = set()
    for i, p in enumerate(payloads):
        assert required_keys <= set(p.keys())
        assert isinstance(p["frame"], np.ndarray) and p["frame"].shape == (H, W)
        assert p["index"] == i
        assert isinstance(p["elapsed_s"], float)
        assert isinstance(p["state"], TrackState)
        assert p["face"] == "A"
        assert p["vial_results"] is None or isinstance(p["vial_results"], dict)
        assert isinstance(p["n_rotations"], int)
        assert isinstance(p["fps_est"], float) and p["fps_est"] >= 0.0
        assert p["pixel_threshold"] == pytest.approx(PIXEL_THR)
        seen_states.add(p["state"])

    # the scripted scene visits both activity-relevant states.
    assert TrackState.STATIONARY in seen_states
    assert TrackState.ROTATING in seen_states

    # vial 1 (moving block) shows nonzero motion at some point; vial 2 (static) never does.
    v1_motion = [p["vial_results"][1][0] for p in payloads if p["vial_results"] and 1 in p["vial_results"]]
    v2_motion = [p["vial_results"][2][0] for p in payloads if p["vial_results"] and 2 in p["vial_results"]]
    assert any(m > 0 for m in v1_motion)
    assert all(m == 0 for m in v2_motion)


def test_raising_frame_observer_does_not_break_run(tmp_path):
    pipe, _calib, _config = _pipeline(tmp_path)

    def bad_observer(_payload):
        raise RuntimeError("boom")

    pipe.add_observer(bad_observer)
    summary = pipe.run()

    assert summary["frames_processed"] == 30      # run completed in full despite every call raising
    assert summary["observer_failures"] == 30
    assert pipe.observer_failures == 30


def test_raising_bin_observer_does_not_break_run(tmp_path):
    pipe, _calib, _config = _pipeline(tmp_path)

    def bad_bin_observer(_payload):
        raise RuntimeError("boom")

    pipe.add_bin_observer(bad_bin_observer)
    summary = pipe.run()

    assert summary["frames_processed"] == 30
    assert summary["n_bins"] == 3
    assert summary["observer_failures"] == 3       # one failure per completed bin


def test_bin_observer_receives_bin_and_records(tmp_path):
    pipe, _calib, _config = _pipeline(tmp_path)
    bin_payloads = []
    pipe.add_bin_observer(bin_payloads.append)
    summary = pipe.run()

    assert summary["n_bins"] == 3
    assert len(bin_payloads) == 3
    for bp in bin_payloads:
        assert set(bp.keys()) >= {"bin", "records"}
        assert hasattr(bp["bin"], "vials")
        assert hasattr(bp["bin"], "bin_start_s") and hasattr(bp["bin"], "bin_end_s")
        assert isinstance(bp["records"], list)

    # bin 0's records carry vial 1's known motion total (mirrors test_pipeline.py's proven numbers).
    bin0_records = {r.vial_id: r for r in bin_payloads[0]["records"]}
    assert bin0_records[1].motion_px_sum == 20
    assert bin0_records[2].motion_px_sum == 0


def test_no_observer_registered_leaves_activity_output_unchanged(tmp_path):
    pipe_plain, _c1, _cfg1 = _pipeline(tmp_path, out_subdir="plain")
    summary_plain = pipe_plain.run()

    pipe_observed, _c2, _cfg2 = _pipeline(tmp_path, out_subdir="observed")
    pipe_observed.add_observer(lambda _p: None)
    pipe_observed.add_bin_observer(lambda _p: None)
    summary_observed = pipe_observed.run()

    for key in ("frames_processed", "frames_read_errors", "n_rotations", "n_bins",
                "n_activity_records", "faces_seen", "per_face_frames", "stopped_reason"):
        assert summary_plain[key] == summary_observed[key], key

    plain_csv = (tmp_path / "plain" / "activity_20260719.csv").read_text(encoding="utf-8")
    observed_csv = (tmp_path / "observed" / "activity_20260719.csv").read_text(encoding="utf-8")
    assert plain_csv == observed_csv


def test_live_monitor_wired_to_real_pipeline_headless(tmp_path):
    """End-to-end: LiveMonitor registered as both observers on a real pipeline run, with
    auto_render=False so this never touches cv2's window/display APIs."""
    pipe, calib, config = _pipeline(tmp_path)
    mon = LiveMonitor(calib, config, auto_render=False)
    pipe.add_observer(mon.on_frame)
    pipe.add_bin_observer(mon.on_bin)

    summary = pipe.run()

    assert summary["frames_processed"] == 30
    assert summary["observer_failures"] == 0
    assert mon.frame_count == 30
    assert len(mon.heatmap_buffer) == 3            # one row per completed bin
    img = mon.render_composite()
    assert img.shape == (mon.canvas_h, mon.canvas_w, 3)
    assert img.dtype == np.uint8


# =============================================================================================
# LiveMonitor fixtures (headless unit tests -- no pipeline required)
# =============================================================================================
def _live_monitor_calibration() -> Calibration:
    v_present = VialROI(id=1, row=0, col=0, x=10, y=10, w=20, h=20, present=True)
    v_absent = VialROI(id=2, row=0, col=1, x=50, y=10, w=20, h=20, present=False)
    fc = FaceCalibration(name="A", vials=[v_present, v_absent], illum_mask_path="unused.png", marker=None)
    return Calibration(image_width=100, image_height=100, faces={"A": fc}, created="", notes="")


class _FakeActivityConfig:
    class activity:
        pixel_threshold = 12.0


def _frame_payload(gray, *, state=TrackState.STATIONARY, face="A", vial_results=None,
                    index=0, elapsed_s=0.0, n_rotations=0, fps_est=10.0, pixel_threshold=12.0):
    return {
        "frame": gray, "index": index, "elapsed_s": elapsed_s, "state": state, "face": face,
        "vial_results": vial_results, "n_rotations": n_rotations, "fps_est": fps_est,
        "pixel_threshold": pixel_threshold,
    }


# =============================================================================================
# LiveMonitor: on_frame / on_bin bookkeeping
# =============================================================================================
def test_on_frame_updates_state_without_window():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)
    assert mon.frame_count == 0
    assert mon.latest_payload is None

    gray = np.full((100, 100), 50, np.uint8)
    mon.on_frame(_frame_payload(gray, vial_results={1: (0, 400, 0.0)}))
    assert mon.frame_count == 1
    assert mon.latest_payload["index"] == 0
    assert mon.pixel_threshold == pytest.approx(12.0)
    assert mon.cur_bin_totals[1] == [0.0, 1]

    gray2 = gray.copy()
    mon.on_frame(_frame_payload(gray2, index=1, vial_results={1: (40, 400, 0.1)}, pixel_threshold=9.0))
    assert mon.frame_count == 2
    assert mon.pixel_threshold == pytest.approx(9.0)           # resynced from the payload
    assert mon.cur_bin_totals[1] == [pytest.approx(0.1), 2]    # accumulated


def test_on_frame_never_touches_cv2_when_auto_render_disabled(monkeypatch):
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)

    def boom(*a, **kw):
        raise AssertionError("cv2.imshow must not be called when auto_render=False")

    monkeypatch.setattr(cv2, "imshow", boom)

    gray = np.zeros((100, 100), np.uint8)
    for i in range(5):
        mon.on_frame(_frame_payload(gray, index=i))
    assert mon.frame_count == 5


def test_non_stationary_frame_does_not_feed_bin_accumulator():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)
    gray = np.zeros((100, 100), np.uint8)
    mon.on_frame(_frame_payload(gray, state=TrackState.STATIONARY, vial_results={1: (5, 400, 0.01)}))
    assert mon.cur_bin_totals[1] == [pytest.approx(0.01), 1]

    mon.on_frame(_frame_payload(gray, state=TrackState.ROTATING, vial_results={1: (0, 400, 0.0)}))
    # ROTATING frames do not add to the bar-chart accumulator (mirrors ActivityAccumulator).
    assert mon.cur_bin_totals[1] == [pytest.approx(0.01), 1]


def test_on_bin_pushes_heatmap_row_and_resets_current_bin():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)
    gray = np.zeros((100, 100), np.uint8)
    mon.on_frame(_frame_payload(gray, vial_results={1: (5, 400, 0.25)}))
    assert mon.cur_bin_totals

    class FakeBin:
        vials = {1: {"active_fraction_mean": 0.25}}

    mon.on_bin({"bin": FakeBin(), "records": []})

    assert mon.cur_bin_totals == {}
    assert len(mon.heatmap_buffer) == 1
    assert mon.heatmap_buffer[0] == {1: 0.25}


def test_heatmap_buffer_rolls_and_stays_bounded():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False, heatmap_bins=5)

    class Bin:
        def __init__(self, val):
            self.vials = {1: {"active_fraction_mean": val}}

    for i in range(12):
        mon.on_bin({"bin": Bin(float(i)), "records": []})

    assert len(mon.heatmap_buffer) == 5
    assert mon.heatmap_buffer.maxlen == 5
    # FIFO: pushes 0..6 fell off; 7..11 remain, oldest -> newest.
    assert [row[1] for row in mon.heatmap_buffer] == [7.0, 8.0, 9.0, 10.0, 11.0]


# =============================================================================================
# LiveMonitor: render_composite (pure -- no cv2 window/display call)
# =============================================================================================
def test_render_composite_shape_and_dtype_before_and_after_frames():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)

    img_empty = mon.render_composite()
    assert img_empty.shape == (mon.canvas_h, mon.canvas_w, 3)
    assert img_empty.dtype == np.uint8

    gray = np.random.default_rng(0).integers(0, 255, size=(100, 100), dtype=np.uint8)
    mon.on_frame(_frame_payload(gray, vial_results={1: (10, 400, 0.025)}))
    img = mon.render_composite()
    assert img.shape == (mon.canvas_h, mon.canvas_w, 3)
    assert img.dtype == np.uint8
    assert mon.last_composite is img


def test_bar_chart_and_heatmap_panels_handle_present_and_absent_vials():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)
    mon.on_frame(_frame_payload(np.zeros((100, 100), np.uint8), vial_results={1: (20, 400, 0.05)}))

    bar = mon._render_bar_chart(mon.right_w, mon.bar_h)
    assert bar.shape == (mon.bar_h, mon.right_w, 3)
    assert bar.dtype == np.uint8

    heat = mon._render_heatmap(mon.right_w, mon.heatmap_h)
    assert heat.shape == (mon.heatmap_h, mon.right_w, 3)
    assert heat.dtype == np.uint8


def test_live_view_draws_roi_colors_and_motion_tint():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)
    # vial 1 present @ (10,10,20,20); vial 2 absent @ (50,10,20,20) -- see _live_monitor_calibration.

    baseline = np.full((100, 100), 50, np.uint8)
    moved = baseline.copy()
    moved[15:18, 15:18] = 200          # inside vial 1's box, well over threshold=12

    mon.on_frame(_frame_payload(baseline, vial_results={1: (0, 400, 0.0)}))
    mon.on_frame(_frame_payload(moved, index=1, vial_results={1: (9, 400, 0.0225)}))

    img = mon._render_live_view(100, 100)          # same size as the source frame -> 1:1 pixels
    assert img.shape == (100, 100, 3)

    # motion pixel: red-ish tint (R pulled up, B pulled down vs. plain gray).
    b, g, r = (int(c) for c in img[16, 16])
    assert r > b, f"expected a red-ish tint at a motion pixel, got BGR=({b},{g},{r})"

    # a non-motion pixel elsewhere inside the same vial box stays exactly plain gray (value 50).
    assert tuple(int(c) for c in img[11, 11]) == (50, 50, 50)

    # ROI border colours: present = solid green, absent = solid red (top edge of each box).
    assert tuple(int(c) for c in img[10, 15]) == (0, 200, 0)
    assert tuple(int(c) for c in img[10, 55]) == (0, 0, 220)


def test_roi_overlay_toggle_removes_boxes_from_render():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)
    gray = np.full((100, 100), 50, np.uint8)
    mon.on_frame(_frame_payload(gray))

    with_roi = mon._render_live_view(100, 100)
    assert tuple(int(c) for c in with_roi[10, 15]) == (0, 200, 0)

    mon.show_roi = False
    without_roi = mon._render_live_view(100, 100)
    assert tuple(int(c) for c in without_roi[10, 15]) == (50, 50, 50)


# =============================================================================================
# LiveMonitor: keyboard handling (pure dispatch on a keycode -- no window needed)
# =============================================================================================
def test_threshold_adjust_callback_fires_with_right_value():
    calls = []
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False,
                       threshold_step=2.0, on_threshold_change=calls.append)
    assert mon.pixel_threshold == pytest.approx(12.0)

    mon.handle_key(ord('+'))
    assert mon.pixel_threshold == pytest.approx(14.0)
    assert calls[-1] == pytest.approx(14.0)

    mon.handle_key(ord('-'))
    mon.handle_key(ord('-'))
    assert mon.pixel_threshold == pytest.approx(10.0)
    assert calls == pytest.approx([14.0, 12.0, 10.0])


def test_threshold_never_goes_negative():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False,
                       threshold_step=100.0)
    mon.handle_key(ord('-'))
    assert mon.pixel_threshold == 0.0


def test_roi_and_pause_and_quit_key_toggles():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)

    assert mon.show_roi is True
    mon.handle_key(ord('o'))
    assert mon.show_roi is False
    mon.handle_key(ord('O'))            # case-insensitive
    assert mon.show_roi is True

    assert mon.paused is False
    mon.handle_key(ord('p'))
    assert mon.paused is True

    assert mon.quit_requested is False
    assert mon.render_enabled is True
    mon.handle_key(ord('q'))
    assert mon.quit_requested is True
    assert mon.render_enabled is False       # quitting disables rendering; run itself is untouched


def test_handle_key_ignores_no_key_sentinel():
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)
    mon.handle_key(-1)                       # cv2.waitKey()'s "no key pressed" sentinel
    assert mon.paused is False
    assert mon.show_roi is True


# =============================================================================================
# LiveMonitor: snapshot saving
# =============================================================================================
def test_save_snapshot_writes_png(tmp_path):
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False,
                       snapshot_dir=str(tmp_path / "snaps"))

    assert mon.save_snapshot() is None       # nothing to save yet -- no frame has ever arrived

    mon.on_frame(_frame_payload(np.zeros((100, 100), np.uint8)))
    path = mon.save_snapshot()
    assert path is not None and os.path.isfile(path)
    saved = cv2.imread(path)
    assert saved.shape == (mon.canvas_h, mon.canvas_w, 3)


def test_handle_key_s_triggers_snapshot(tmp_path):
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False,
                       snapshot_dir=str(tmp_path / "snaps"))
    mon.on_frame(_frame_payload(np.zeros((100, 100), np.uint8)))
    mon.handle_key(ord('s'))
    files = list((tmp_path / "snaps").glob("*.png"))
    assert len(files) == 1


# =============================================================================================
# LiveMonitor: headless degradation (cv2 display errors never crash the run)
# =============================================================================================
def test_maybe_render_disables_gracefully_on_cv2_display_error(monkeypatch):
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False, max_fps=1000)
    mon.on_frame(_frame_payload(np.zeros((100, 100), np.uint8)))

    def raiser(*a, **kw):
        raise cv2.error("simulated: no display available")

    monkeypatch.setattr(cv2, "imshow", raiser)

    assert mon.maybe_render() is False
    assert mon.render_enabled is False
    assert mon.disabled_reason is not None

    # subsequent calls are instant no-ops (never touch cv2 again) -- and on_frame keeps working.
    assert mon.maybe_render() is False
    mon.on_frame(_frame_payload(np.zeros((100, 100), np.uint8), index=1))
    assert mon.frame_count == 2


def test_maybe_render_throttles_to_max_fps(monkeypatch):
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False, max_fps=1.0)
    mon.on_frame(_frame_payload(np.zeros((100, 100), np.uint8)))

    calls = {"n": 0}

    def fake_imshow(*a, **kw):
        calls["n"] += 1

    monkeypatch.setattr(cv2, "imshow", fake_imshow)
    monkeypatch.setattr(cv2, "waitKey", lambda *_a, **_kw: -1)

    # two calls in immediate succession at max_fps=1 -> only the first is due.
    assert mon.maybe_render() is True
    assert mon.maybe_render() is False
    assert calls["n"] == 1


def test_close_is_idempotent(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: calls.append(name))
    mon = LiveMonitor(_live_monitor_calibration(), _FakeActivityConfig(), auto_render=False)
    mon.close()
    mon.close()
    assert mon.render_enabled is False
    assert calls == [mon.window_name, mon.window_name]

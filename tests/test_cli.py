"""Tests for flygym_tracker.cli (DESIGN.md section 4 `cli.py` row, section 7, section 9, README quick-start).

Everything here is synthetic/offline: a small MJPG clip written with `cv2.VideoWriter` (no camera),
and a hand-built calibration bundle written via `calibration.save_calibration`. The scene mirrors
tests/test_pipeline.py's proven synthetic design (quiet frames with a moving block in one vial,
then loud global-motion "rotation" frames, then a different quiet background) so the same
enter/exit/pixel thresholds are known-good, just re-encoded through a real video file instead of an
in-memory FakeSource -- this is what actually exercises the CLI's argument parsing and module
wiring (`config -> calibration -> frame_source -> logger -> markers -> pipeline`), which
tests/test_pipeline.py does not touch at all.

Two `--from-camera`/live-camera tests use a serial that cannot possibly match a real attached
device (same trick as tests/test_frame_source.py's
`test_hik_camera_source_open_raises_when_sdk_or_camera_absent`), so they fail deterministically
without hardware instead of skipping.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from flygym_tracker.calibration import load_calibration, save_calibration
from flygym_tracker.cli import main
from flygym_tracker.config import load_config
from flygym_tracker.types import ACTIVITY_COLUMNS, Calibration, FaceCalibration, VialROI

# =============================================================================================
# Synthetic scene (mirrors tests/test_pipeline.py's design; re-encoded through a real video file)
# =============================================================================================
H, W = 60, 80
FPS = 10.0
V1 = dict(id=1, row=0, col=0, x=4, y=4, w=16, h=16)
V2 = dict(id=2, row=0, col=1, x=48, y=4, w=16, h=16)
BLOCK = (slice(8, 12), slice(8, 12))          # 4x4 moving block inside vial 1 only
BLOCK_LO, BLOCK_HI = 20, 235                  # delta 215 >> pixel_threshold, survives MJPG easily

ENTER, EXIT, PIXEL_THR = 40.0, 15.0, 30.0
DEBOUNCE, MIN_STATIONARY = 2, 2
N_PER_PHASE = 20                              # 60 frames total, 3 phases


def _illum_mask() -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    m[4:20, 4:20] = 255
    m[4:20, 48:64] = 255
    return m


def _background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.zeros((H, W), np.uint8)
    f[4:20, 4:20] = rng.integers(90, 110, size=(16, 16), dtype=np.uint8)
    f[4:20, 48:64] = rng.integers(90, 110, size=(16, 16), dtype=np.uint8)
    return f


P = _background(1)   # pre-rotation background
Q = _background(2)   # post-rotation background (deliberately different from P)


def _pre_frame(block_val: int) -> np.ndarray:
    f = P.copy()
    f[BLOCK] = block_val
    return f


def _loud_frame(val: int) -> np.ndarray:
    f = np.zeros((H, W), np.uint8)
    f[4:20, 4:20] = val
    f[4:20, 48:64] = val
    return f


def _scene_frames() -> list:
    frames = []
    for i in range(N_PER_PHASE):                      # quiet + toggling block -> vial 1 motion only
        frames.append(_pre_frame(BLOCK_LO if i % 2 == 0 else BLOCK_HI))
    for i in range(N_PER_PHASE):                       # loud global motion -> ROTATING
        frames.append(_loud_frame(0 if i % 2 == 0 else 255))
    for _ in range(N_PER_PHASE):                        # different quiet background, no block
        frames.append(Q.copy())
    return frames


def _write_video(path, frames, fps: float = FPS) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H), isColor=True)
    assert writer.isOpened(), "cv2.VideoWriter failed to open (MJPG codec unavailable?)"
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    writer.release()


def _write_noise_video(path, n: int = 15, fps: float = FPS) -> None:
    """A short static clip with small per-pixel jitter, for the `noise` command."""
    rng = np.random.default_rng(42)
    base = np.zeros((H, W), np.uint8)
    base[4:20, 4:20] = 120
    base[4:20, 48:64] = 120
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H), isColor=True)
    assert writer.isOpened()
    for _ in range(n):
        noise = rng.integers(-3, 4, size=(H, W)).astype(np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def _write_calib_bundle(out_dir) -> str:
    mask = _illum_mask()
    vials = [VialROI(present=True, **V1), VialROI(present=True, **V2)]
    fc = FaceCalibration(name="A", vials=vials, illum_mask_path="illum_mask_A.png", marker=None)
    calib = Calibration(image_width=W, image_height=H, faces={"A": fc}, created="", notes="")
    save_calibration(calib, mask, str(out_dir))
    return str(out_dir)


def _write_config_yaml(path, *, thresholds: bool, output_format: str = "csv") -> str:
    data = {"markers": {"enabled": False}, "output": {"format": output_format}}
    if thresholds:
        data["rotation"] = {
            "enter_threshold": ENTER, "exit_threshold": EXIT,
            "debounce_frames": DEBOUNCE, "min_stationary_frames": MIN_STATIONARY,
        }
        data["activity"] = {"pixel_threshold": PIXEL_THR}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return str(path)


# =============================================================================================
# replay: end-to-end
# =============================================================================================
def test_replay_end_to_end_writes_activity_csv(tmp_path):
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, _scene_frames())
    calib_dir = _write_calib_bundle(tmp_path / "calib")
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    out_dir = tmp_path / "out"

    rc = main([
        "replay", "--video", str(video_path), "--config", config_path,
        "--calib", calib_dir, "--bin-seconds", "1", "--out", str(out_dir),
    ])
    assert rc == 0

    activity_files = sorted(out_dir.glob("activity_*.csv"))
    assert len(activity_files) == 1, f"expected exactly one activity_*.csv, found {activity_files}"
    df = pd.read_csv(activity_files[0])

    # schema + basic shape
    assert list(df.columns) == ACTIVITY_COLUMNS
    assert len(df) > 0
    assert set(df["vial_id"].unique()) <= {1, 2}
    assert bool(df["present"].all())
    assert set(df["face"]) == {"A"}

    # --bin-seconds 1 actually took effect: ~6s of 10fps content -> several distinct bins
    # (config.yaml itself leaves binning.bin_seconds at the packaged default of 60, which alone
    # would produce a single bin -- so this also proves the CLI override wins over the file).
    assert df["bin_start_iso"].nunique() >= 2

    # vial 1 (moving block) accumulates strictly more motion than vial 2 (static throughout).
    totals = df.groupby("vial_id")["motion_px_sum"].sum()
    assert totals.get(1, 0) > totals.get(2, 0)

    assert (out_dir / "events.csv").exists()
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"].startswith("run_")
    assert meta["stop_iso"] is not None


def test_replay_null_thresholds_gives_friendly_error(tmp_path, capsys):
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, _scene_frames()[:10])
    calib_dir = _write_calib_bundle(tmp_path / "calib")
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=False)
    out_dir = tmp_path / "out"

    rc = main([
        "replay", "--video", str(video_path), "--config", config_path,
        "--calib", calib_dir, "--out", str(out_dir),
    ])
    assert rc == 2
    captured = capsys.readouterr()
    assert "noise" in captured.err.lower()


def test_run_null_thresholds_gives_friendly_error_without_touching_camera(tmp_path, capsys):
    # TrackerPipeline resolves thresholds before ever calling source.open(), so this exercises the
    # full `run` wiring (config, calibration, camera-source construction, logger, markers) without
    # any real hardware.
    calib_dir = _write_calib_bundle(tmp_path / "calib")
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=False)
    out_dir = tmp_path / "out"

    rc = main(["run", "--config", config_path, "--calib", calib_dir, "--out", str(out_dir)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "noise" in captured.err.lower()


# =============================================================================================
# noise
# =============================================================================================
def test_noise_command_prints_and_writes_thresholds(tmp_path, capsys):
    video_path = tmp_path / "noise.avi"
    _write_noise_video(video_path)
    calib_dir = _write_calib_bundle(tmp_path / "calib")
    out_yaml = tmp_path / "suggested.yaml"

    rc = main([
        "noise", "--video", str(video_path), "--calib", calib_dir,
        "--frames", "50", "--out", str(out_yaml),
    ])
    assert rc == 0

    captured = capsys.readouterr()
    assert "suggested_pixel_threshold" in captured.out
    assert "suggested_enter_threshold" in captured.out
    assert "suggested_exit_threshold" in captured.out

    assert out_yaml.exists()
    data = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert data["activity"]["pixel_threshold"] > 0
    assert data["rotation"]["enter_threshold"] > data["rotation"]["exit_threshold"]

    # "ready to pass to run --config": load_config must accept it directly as an override layer.
    cfg = load_config(path=str(out_yaml))
    assert cfg.activity.pixel_threshold == pytest.approx(data["activity"]["pixel_threshold"])
    assert cfg.rotation.enter_threshold == pytest.approx(data["rotation"]["enter_threshold"])
    assert cfg.rotation.exit_threshold == pytest.approx(data["rotation"]["exit_threshold"])


def test_noise_without_out_flag_skips_writing_file(tmp_path, capsys):
    video_path = tmp_path / "noise.avi"
    _write_noise_video(video_path)
    calib_dir = _write_calib_bundle(tmp_path / "calib")

    rc = main(["noise", "--video", str(video_path), "--calib", calib_dir])
    assert rc == 0
    captured = capsys.readouterr()
    assert "suggested_pixel_threshold" in captured.out
    assert not list(tmp_path.glob("*.yaml"))


# =============================================================================================
# calibrate
# =============================================================================================
REAL_FRAME = os.path.join(os.path.dirname(__file__), "..", "docs", "frame_full.png")


@pytest.mark.skipif(not os.path.isfile(REAL_FRAME), reason="real reference frame not available")
def test_calibrate_frame_writes_bundle_and_round_trips(tmp_path, capsys):
    out_dir = tmp_path / "calib_out"
    rc = main(["calibrate", "--frame", REAL_FRAME, "--face", "A", "--out", str(out_dir)])
    assert rc == 0

    assert (out_dir / "calibration.json").exists()
    assert (out_dir / "illum_mask_A.png").exists()
    assert (out_dir / "overlay_A.png").exists()

    captured = capsys.readouterr()
    assert "present" in captured.out and "empty" in captured.out

    loaded = load_calibration(str(out_dir))
    assert "A" in loaded.faces
    vials = loaded.faces["A"].vials
    assert len(vials) == 16
    # ground truth for this exact reference frame (see tests/test_calibration.py): ids 7 and 10.
    assert sorted(v.id for v in vials if not v.present) == [7, 10]


def test_calibrate_requires_frame_or_camera():
    with pytest.raises(SystemExit) as exc_info:
        main(["calibrate"])
    assert exc_info.value.code != 0


def test_calibrate_missing_frame_file_reports_error_not_traceback(tmp_path, capsys):
    rc = main(["calibrate", "--frame", str(tmp_path / "does_not_exist.png"), "--out", str(tmp_path / "out")])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err


def test_calibrate_from_camera_reports_error_without_matching_camera(tmp_path, capsys):
    # Same trick as test_frame_source.py's HikCameraSource test: a serial that cannot possibly
    # match a real attached device, so open() fails deterministically with no hardware present.
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"source": {"camera": {"serial": "NO-SUCH-CAMERA-SERIAL-0000000"}}}, f)
    out_dir = tmp_path / "calib_out"

    rc = main(["calibrate", "--from-camera", "--config", str(config_path), "--out", str(out_dir)])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err


# =============================================================================================
# argument parsing
# =============================================================================================
def test_main_no_args_prints_help_and_returns_nonzero(capsys):
    rc = main([])
    assert rc != 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_main_help_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_main_unknown_subcommand_errors():
    with pytest.raises(SystemExit) as exc_info:
        main(["frobnicate"])
    assert exc_info.value.code != 0


def test_run_requires_config_and_calib():
    with pytest.raises(SystemExit) as exc_info:
        main(["run"])
    assert exc_info.value.code != 0


def test_replay_requires_video_config_and_calib():
    with pytest.raises(SystemExit) as exc_info:
        main(["replay"])
    assert exc_info.value.code != 0


def test_noise_requires_calib():
    with pytest.raises(SystemExit) as exc_info:
        main(["noise"])
    assert exc_info.value.code != 0

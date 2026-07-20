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
import re

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from flygym_tracker.calibration import load_calibration, save_calibration
from flygym_tracker.cli import main
from flygym_tracker.config import load_config
from flygym_tracker.types import ACTIVITY_COLUMNS, Calibration, FaceCalibration, VialROI


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

    assert (_one(out_dir, "events_*.csv")).exists()
    meta = json.loads((_one(out_dir, "run_meta_*.json")).read_text(encoding="utf-8"))
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


# =============================================================================================
# settings -- the panel had to become findable
# =============================================================================================
#
# The panel existed long before this command did, and the operator still could not find it: it was
# reachable only from `run`/`replay --settings` or the monitor's `t` key -- i.e. only while
# something was already running -- and `run.bat` never passed the flag. Between-runs tuning had no
# entry point at all. So this command must need NO camera and NO calibration bundle: demanding a
# rig be present to edit a YAML file is what made the feature invisible.


class _StubPanel:
    """Stands in for `SettingsWindow`. Records what it was given; opens nothing."""

    last = None

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.ran = False
        _StubPanel.last = self

    def run(self, *_a, **_kw):
        self.ran = True
        return self.model


@pytest.fixture
def stub_panel(monkeypatch):
    from flygym_tracker import cli as CLI

    monkeypatch.setattr(CLI, "SettingsWindow", _StubPanel)
    monkeypatch.setattr(CLI, "has_gui_support", lambda: True)
    _StubPanel.last = None
    return _StubPanel


@pytest.fixture
def no_camera_allowed(monkeypatch):
    """Fail loudly if anything in this test opens the camera. USB3 Vision access is EXCLUSIVE, so a
    tuning command that grabbed the camera could block the experiment it is being tuned for."""
    from flygym_tracker.frame_source import HikCameraSource

    def forbidden(_self):
        raise AssertionError("the settings command opened the camera")

    monkeypatch.setattr(HikCameraSource, "open", forbidden)


def test_settings_runs_with_no_camera_and_no_calibration_bundle(tmp_path, stub_panel,
                                                                no_camera_allowed, capsys):
    """The whole point of the command, in one test: a config file is all it needs."""
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    rc = main(["settings", "--config", config_path])
    assert rc == 0
    assert _StubPanel.last is not None and _StubPanel.last.ran
    assert "unchanged" in capsys.readouterr().out


def test_settings_needs_no_config_file_either(stub_panel, no_camera_allowed):
    """With no --config it edits the packaged defaults, which is still better than a traceback."""
    assert main(["settings"]) == 0


def test_settings_list_prints_the_values_without_opening_a_window(tmp_path, no_camera_allowed,
                                                                  capsys):
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    rc = main(["settings", "--config", config_path, "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "activity.pixel_threshold" in out
    assert "source.camera.frame_rate" in out
    assert "camera default" in out, "an unset camera row must not print as a number"


def test_settings_offers_every_camera_control_the_rig_owner_asked_for(tmp_path, no_camera_allowed,
                                                                      capsys):
    """Verbatim: "the framerate, exposure settings and image size can be adjusted from inside the
    software"."""
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    main(["settings", "--config", config_path, "--list"])
    out = capsys.readouterr().out
    for key in ("frame_rate", "exposure_us", "gain_db", "width", "height"):
        assert "source.camera.%s" % key in out


def test_settings_says_the_image_size_only_takes_effect_at_the_next_start(tmp_path,
                                                                         no_camera_allowed, capsys):
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    main(["settings", "--config", config_path, "--list"])
    out = capsys.readouterr().out
    for line in out.splitlines():
        if "source.camera.width" in line or "source.camera.height" in line:
            assert "NEXT START" in line
        if "source.camera.frame_rate" in line:
            assert "NEXT START" not in line, "the frame rate IS live-adjustable on this camera"


def test_settings_does_not_touch_the_camera_unless_asked(tmp_path, stub_panel, no_camera_allowed):
    """USB3 Vision is exclusive: grabbing the camera to draw a slider could block a run."""
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    assert main(["settings", "--config", config_path]) == 0


def test_settings_says_on_screen_that_the_limits_are_not_live_without_a_camera(tmp_path,
                                                                              stub_panel,
                                                                              no_camera_allowed):
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    main(["settings", "--config", config_path])
    note = _StubPanel.last.model.group_notes.get("Camera", "")
    assert "not live" in note or "documented" in note


def test_probe_camera_degrades_to_fallback_limits_when_the_camera_is_busy(tmp_path, stub_panel,
                                                                         monkeypatch, capsys):
    """Explicitly opt-in, and never fatal. A tuning command must not fail because an experiment is
    already using the camera -- it degrades and says why."""
    from flygym_tracker.frame_source import HikCameraSource

    def busy(_self):
        raise RuntimeError("MV_CC_OpenDevice failed (ret=0x80000203) - camera may already be in use")

    monkeypatch.setattr(HikCameraSource, "open", busy)
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    rc = main(["settings", "--config", config_path, "--probe-camera"])
    assert rc == 0
    out = capsys.readouterr().out
    # The operator must be told the numbers are not this camera's. They are the rig sensor's own,
    # measured 2026-07-19, but a measurement of a camera you cannot currently reach is still not
    # a live read -- so "not live" is the part that has to survive any rewording of this line.
    assert "not live" in out
    assert _StubPanel.last.ran, "a busy camera must not stop the operator editing the file"


def test_probe_camera_names_what_holds_the_camera_instead_of_a_bare_error_code(tmp_path,
                                                                              stub_panel,
                                                                              monkeypatch, capsys):
    """0x80000203 names no culprit, which is the whole problem -- the usual holder is a headless
    Bonsai with no window to close. `camera_lock` is reused rather than re-implemented."""
    from flygym_tracker import camera_lock
    from flygym_tracker.frame_source import HikCameraSource

    monkeypatch.setattr(HikCameraSource, "open", lambda _self: (_ for _ in ()).throw(
        RuntimeError("MV_CC_OpenDevice failed - camera may already be in use")))
    monkeypatch.setattr(camera_lock, "find_camera_holders",
                        lambda *a, **k: [camera_lock.CameraHolder(
                            pid=4242, name="Bonsai.exe", what="a Bonsai workflow", headless=True)])
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    assert main(["settings", "--config", config_path, "--probe-camera"]) == 0
    out = capsys.readouterr().out
    assert "4242" in out and "Bonsai" in out


def test_probe_camera_never_stops_anything_it_only_reports(tmp_path, stub_panel, monkeypatch):
    """Ending someone's acquisition to draw a slider would be absurd. `settings` reports; only
    `free-camera` stops."""
    from flygym_tracker import camera_lock
    from flygym_tracker.frame_source import HikCameraSource

    monkeypatch.setattr(HikCameraSource, "open", lambda _self: (_ for _ in ()).throw(
        RuntimeError("camera may already be in use")))
    monkeypatch.setattr(camera_lock, "find_camera_holders", lambda *a, **k: [])
    monkeypatch.setattr(camera_lock, "release_camera",
                        lambda *a, **k: pytest.fail("settings tried to kill a process"))
    monkeypatch.setattr(camera_lock, "stop_process",
                        lambda *a, **k: pytest.fail("settings tried to kill a process"))
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    assert main(["settings", "--config", config_path, "--probe-camera"]) == 0


def test_settings_saves_the_operators_change_back_to_the_config_file(tmp_path, stub_panel,
                                                                     no_camera_allowed):
    """`s` in the panel is the whole delivery mechanism; the CLI wires the save hook.

    Started from a config that FORCES a width, which is the state every rig was in before this
    work: clearing that row has to leave `null` in the file, not the 1280 the camera is running at.
    """
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("source:\n  camera:\n    width: 1280       # forced on every run\n"
                "activity:\n  pixel_threshold: 12.0\n")
    main(["settings", "--config", config_path])
    model = _StubPanel.last.model
    assert model.value("source.camera.width") == 1280
    model.set("activity.pixel_threshold", 18.5)
    model.to_default("source.camera.width")
    _StubPanel.last.kwargs["on_save"](model)

    text = open(config_path, encoding="utf-8").read()
    saved = yaml.safe_load(text)
    assert saved["activity"]["pixel_threshold"] == pytest.approx(18.5)
    assert saved["source"]["camera"]["width"] is None
    assert "width: 1280" not in text
    assert "forced on every run" in text, "the note explaining the value must survive"
    # The property that actually matters: the next run sends nothing for Width.
    assert load_config(path=config_path).source.camera.width is None


def test_settings_warns_when_the_operator_closes_without_saving(tmp_path, stub_panel,
                                                                no_camera_allowed, capsys):
    """Silently discarding a tuning session would lose work with no trace."""
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)

    class _ChangingPanel(_StubPanel):
        def run(self, *_a, **_kw):
            self.model.set("activity.pixel_threshold", 22.0)
            return super().run()

    from flygym_tracker import cli as CLI
    CLI.SettingsWindow = _ChangingPanel
    rc = main(["settings", "--config", config_path])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NOT saved" in out and "activity.pixel_threshold" in out


def test_settings_reports_a_bad_config_path_as_a_message_not_a_traceback(tmp_path, capsys):
    rc = main(["settings", "--config", str(tmp_path / "nope.yaml")])
    assert rc == 1
    assert capsys.readouterr().err


def test_settings_without_a_gui_says_how_to_see_the_values_anyway(tmp_path, monkeypatch, capsys):
    from flygym_tracker import cli as CLI

    monkeypatch.setattr(CLI, "has_gui_support", lambda: False)
    config_path = _write_config_yaml(tmp_path / "config.yaml", thresholds=True)
    rc = main(["settings", "--config", config_path])
    assert rc == 2
    assert "--list" in capsys.readouterr().err or "settings --list" in capsys.readouterr().err


def test_the_settings_subcommand_is_registered_with_its_flags():
    from flygym_tracker.cli import build_parser

    args = build_parser().parse_args(["settings"])
    assert args.probe_camera is False and args.list is False
    assert build_parser().parse_args(["settings", "--probe-camera"]).probe_camera is True


# =============================================================================================
# run.bat -- now a LAUNCHER, verified by PARSING the file
#
# THIS WHOLE SECTION WAS INVERTED, DELIBERATELY, AND THAT IS WORTH READING BEFORE CHANGING IT BACK.
#
# It used to assert that `run.bat` carried a numbered menu: [S] settings near the top, [1]..[5]
# unrenumbered because those digits are what is written on the note stuck to the rig and what is in
# the operator's fingers. Every one of those assertions was RIGHT for a file that was the way in.
#
# It is not the way in any more. Starting a run, drawing vial positions, replaying a recording,
# measuring the noise floor and freeing the camera are all buttons in the window, so a menu here
# would be a SECOND way to do the same five jobs -- and a second place for the paths, the defaults
# and the wording to drift out of step with the app. The old tests would now be pinning a duplicate
# surface in place.
#
# What is still true, and is what these tests guard, is the part a window cannot do for itself:
# find a Python, put `src` on the path, and check the imports the app needs BEFORE the app tries to
# open. A missing PySide6 has to be reported by something that is not PySide6.
# =============================================================================================
RUN_BAT = os.path.join(os.path.dirname(__file__), "..", "run.bat")


def _run_bat_text() -> str:
    with open(RUN_BAT, encoding="utf-8", errors="replace") as f:
        return f.read()


def _menu_choices(text: str):
    """``{choice: label}`` from any menu block. Expected to be EMPTY now."""
    return dict(re.findall(r"^echo\s+\[([^\]]+)\]\s+(.*)$", text, re.MULTILINE))


def test_run_bat_no_longer_offers_a_numbered_menu():
    """"No terminal prompt anywhere in the app", and the menu was the last one. A digit typed at a
    console is the interface the window replaced."""
    choices = _menu_choices(_run_bat_text())
    assert choices == {}, "run.bat still has a menu: %s" % sorted(choices)


def test_run_bat_launches_the_app_and_nothing_else():
    """One command. If a second `cli` subcommand appeared here it would be a job the window also
    offers, reachable two ways, with two sets of defaults."""
    commands = re.findall(r"flygym_tracker\.cli\s+([a-z-]+)", _run_bat_text())
    assert "gui" in commands, "run.bat does not launch the app"
    assert set(commands) == {"gui"}, "run.bat still runs other subcommands: %s" % sorted(set(commands))


def test_run_bat_does_not_ask_the_operator_to_choose_anything_about_the_experiment():
    """`set /p` survives ONLY for the two install questions, which are about this computer rather
    than about the experiment -- and which cannot move into an app that will not import."""
    prompts = re.findall(r'set\s+/p\s+(\w+)=', _run_bat_text())
    assert set(prompts) <= {"INSTALL", "FIXCV"},         "run.bat still prompts for experiment choices: %s" % sorted(set(prompts))


def test_run_bat_still_checks_the_imports_the_window_cannot_report_on_itself():
    """A missing PySide6 has to be reported by something that is not PySide6, or the operator gets
    a traceback with no window behind it -- which is where a support call starts."""
    text = _run_bat_text()
    assert "PySide6" in text
    assert "requirements.txt" in text


def test_run_bat_no_longer_warns_about_a_headless_opencv_build():
    """THIS TEST USED TO ASSERT THE OPPOSITE, and the behaviour it pinned is genuinely gone.

    It was right at the time: the app drew with Qt and did not care, but the tools it LAUNCHED --
    vial drawing, replay with the monitor -- opened OpenCV windows in child processes, and finding
    that out from a child that silently fails to open a window is worse than being told up front.

    Those tools are not launched any more. Every video job happens inside the app's own window,
    drawn with Qt (`gui/video_stage.py`), so nothing the launcher starts opens an OpenCV window.
    OpenCV is still used for the maths -- masks, contours, frame differences -- and the headless
    build does all of that identically. Warning about it now would send an operator to uninstall
    and reinstall a package to fix a window that no longer exists.
    """
    code = "\n".join(line for line in _run_bat_text().splitlines()
                     if not line.strip().upper().startswith("REM"))
    assert "has_gui_support" not in code
    assert "opencv-python-headless" not in code


def test_run_bat_no_longer_hard_codes_the_experiment_paths():
    """The app owns them: config file, vial positions and output folder are chosen in the window
    and saved. A second copy here meant choosing an output folder in the app and still getting
    results somewhere else."""
    text = _run_bat_text()
    for stale in ('set "CONFIG=', 'set "CALIB=', 'set "OUTDIR='):
        assert stale not in text, "run.bat still sets a path: %s" % stale


def test_run_bat_puts_src_on_the_path_so_the_package_imports_uninstalled():
    """The one piece of environment setup that has to happen before Python is asked to import
    anything from this repo."""
    assert "PYTHONPATH" in _run_bat_text()

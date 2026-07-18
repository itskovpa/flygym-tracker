"""Command-line entry points for flygym_tracker (DESIGN.md section 4 `cli.py` row, section 7, section 9).

Four subcommands, matching the build order in DESIGN.md section 9 ("validate on the empty rig:
live capture, noise floor, calibration on the real face, rotation detection"):

  * ``calibrate`` -- build a per-face `Calibration` bundle (DESIGN.md section 5.4/5.5) from a still
    image or a single live-camera grab, optionally refined with the interactive manual wizard.
  * ``noise``     -- measure the static-rig noise floor (`pipeline.measure_noise`) and suggest the
    ``activity.pixel_threshold`` / ``rotation.enter_threshold`` / ``rotation.exit_threshold`` values
    a calibration alone cannot provide (DESIGN.md section 5.1/5.3: thresholds are seeded from real
    noise, never hard-coded).
  * ``run``       -- live tracking against the HikRobot camera (`frame_source.HikCameraSource`).
  * ``replay``    -- the offline dev path: identical pipeline against a recorded video
    (`frame_source.VideoFileSource`) instead of the camera.

None of the actual CV/IO logic lives here -- this module only parses arguments, wires the modules
documented in DESIGN.md section 4 together in the order `config -> calibration -> frame_source ->
logger -> markers -> pipeline`, and turns the handful of user-actionable failure modes (most
notably "thresholds are still null" -- DESIGN.md section 5.1/5.3 -- since `TrackerPipeline` refuses
to invent them) into a short message instead of a traceback.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import yaml

from flygym_tracker.calibrate_wizard import run_wizard
from flygym_tracker.calibration import (
    boxes_from_calibration,
    detect_calibration,
    load_calibration,
    save_calibration,
)
from flygym_tracker.config import load_config
from flygym_tracker.frame_source import HikCameraSource, VideoFileSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.markers import MarkerDetector
from flygym_tracker.pipeline import TrackerPipeline, measure_noise

#: Shared "thresholds are missing" remediation, appended to the pipeline's own ValueError text.
_THRESHOLD_HINT = (
    "Thresholds are not configured. Run `flygym-tracker noise --calib <dir> --video <clip> "
    "--out <file.yaml>` (or point --video at a live-camera-free static clip) to measure the "
    "noise floor and get suggested values, then pass that file as --config (or merge its values "
    "into your own config)."
)


# =============================================================================================
# Shared helpers
# =============================================================================================
def _camera_source_from_config(config) -> HikCameraSource:
    """Build a `HikCameraSource` from `config.source.camera` (DESIGN.md section 4 `frame_source.py`).

    Construction never touches hardware (see frame_source.py) -- safe to call even when no camera
    is attached; only `.open()`/`.read()` can fail.
    """
    cam = config.source.camera
    return HikCameraSource(
        serial=cam.get("serial"),
        index=cam.get("index", 0),
        width=cam.get("width"),
        height=cam.get("height"),
        exposure_us=cam.get("exposure_us"),
        gain_db=cam.get("gain_db"),
        frame_rate=cam.get("frame_rate"),
        pixel_format=cam.get("pixel_format", "Mono8"),
    )


def _make_run_id() -> str:
    """A sortable, filesystem-safe run id from the current timestamp."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _build_marker_detector(config, calib) -> MarkerDetector:
    """Build a `MarkerDetector` from `config.markers`, seeded with any registry already saved in
    the calibration bundle (DESIGN.md section 5.2; see markers.py's `to_dict`/`from_dict` for the
    `FaceCalibration.marker = {"signature": [...]}` convention this reads back)."""
    registry = {}
    for name, fc in calib.faces.items():
        marker = fc.marker
        if isinstance(marker, dict) and marker.get("signature") is not None:
            registry[name] = marker["signature"]

    markers_cfg = config.markers
    kwargs = {}
    if markers_cfg.get("search_region") is not None:
        kwargs["search_region"] = tuple(markers_cfg.search_region)
    if markers_cfg.get("min_area") is not None:
        kwargs["min_area"] = markers_cfg.min_area
    if markers_cfg.get("max_area") is not None:
        kwargs["max_area"] = markers_cfg.max_area

    return MarkerDetector.from_dict(registry, enabled=bool(markers_cfg.enabled), **kwargs)


def _build_logger(config, args, run_id: str) -> ActivityLogger:
    out_dir = args.out or config.output.dir
    meta = {"config": config.to_dict(), "calibration_dir": args.calib}
    return ActivityLogger(
        output_dir=out_dir,
        run_id=run_id,
        fmt=config.output.format,
        rolling=config.output.rolling,
        meta=meta,
    )


def _load_run_config(args):
    """`load_config(path=args.config, overrides=...)`, folding --bin-seconds in as an override
    (DESIGN.md section 10: bin size must be easy to change per experiment, via config AND a CLI flag)."""
    overrides = {"binning": {"bin_seconds": args.bin_seconds}} if args.bin_seconds is not None else None
    return load_config(path=args.config, overrides=overrides)


def _run_pipeline_or_report(config, calib, source, logger, marker_detector, *, clock, max_frames, stop_flag) -> int:
    """Construct + run a `TrackerPipeline`, turning its two documented construction-time failure
    modes into a short message instead of a traceback: null thresholds (ValueError -- DESIGN.md
    section 5.1/5.3, the "last bit" that needs `noise`) and an unreadable calibration mask
    (RuntimeError -- e.g. a hand-edited or half-written calibration bundle)."""
    try:
        pipe = TrackerPipeline(config, calib, source, logger, marker_detector=marker_detector, clock=clock)
    except ValueError as e:
        logger.close()
        print(f"error: {e}\n{_THRESHOLD_HINT}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        logger.close()
        print(f"error: {e}", file=sys.stderr)
        return 1

    summary = pipe.run(max_frames=max_frames, stop_flag=stop_flag)
    print(json.dumps(summary, indent=2))
    return 0


def _load_calibration_or_report(calib_dir: str):
    """`load_calibration`, wrapped so a bad --calib path is a message, not a traceback.

    Returns the `Calibration`, or `None` (with an error already printed) on failure.
    """
    try:
        return load_calibration(calib_dir)
    except Exception as e:  # bad path / corrupt json / etc. -- not worth enumerating exception types
        print(f"error: could not load calibration from {calib_dir!r}: {e}", file=sys.stderr)
        return None


# =============================================================================================
# calibrate
# =============================================================================================
def _load_calibration_frame(args):
    """Return an HxW grayscale still: from --frame, or a single grab from the live camera."""
    if args.from_camera:
        config = load_config(path=args.config)
        source = _camera_source_from_config(config)
        with source:
            frame = source.read()
        if frame is None:
            raise RuntimeError("camera returned no frame")
        return frame.image
    gray = cv2.imread(args.frame, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"could not read frame image {args.frame!r}")
    return gray


def _cmd_calibrate(args) -> int:
    try:
        gray = _load_calibration_frame(args)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # Always run auto-detect first (DESIGN.md section 5.4 B): its boxes seed the wizard even when
    # the caller wants to nudge them, and are the whole calibration when the caller does not.
    try:
        calib, mask, overlay = detect_calibration(gray, face=args.face)
        seed_boxes, seed_present = boxes_from_calibration(calib, args.face)
    except ValueError as e:
        if not args.wizard:
            print(
                f"error: auto-detect calibration failed: {e}\n"
                "Re-run with --wizard to calibrate this face manually.",
                file=sys.stderr,
            )
            return 2
        seed_boxes, seed_present = None, None
        calib, mask, overlay = None, None, None

    if args.wizard:
        calib, mask, overlay = run_wizard(gray, face=args.face, seed_boxes=seed_boxes, seed_present=seed_present)

    save_calibration(calib, mask, args.out, overlay=overlay)

    vials = calib.faces[args.face].vials
    n_present = sum(1 for v in vials if v.present)
    n_empty = len(vials) - n_present
    print(
        f"calibration saved to {args.out!r}: face {args.face!r}, "
        f"{n_present} present / {n_empty} empty (of {len(vials)} slots)"
    )
    return 0


# =============================================================================================
# noise
# =============================================================================================
def _cmd_noise(args) -> int:
    try:
        config = load_config(path=args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    calib = _load_calibration_or_report(args.calib)
    if calib is None:
        return 1

    face = args.face or ("A" if "A" in calib.faces else sorted(calib.faces)[0])
    if face not in calib.faces:
        print(f"error: face {face!r} not in calibration (have: {sorted(calib.faces)})", file=sys.stderr)
        return 1

    mask_path = calib.faces[face].illum_mask_path
    illum_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if illum_mask is None:
        print(f"error: could not read illum mask at {mask_path!r}", file=sys.stderr)
        return 1

    source = VideoFileSource(args.video) if args.video else _camera_source_from_config(config)
    k = float(config.activity.k)

    try:
        with source:
            result = measure_noise(source, illum_mask, n_frames=args.frames, k=k)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"measured over {result['n_frames']} frame(s), {result['n_pairs']} consecutive pair(s), face {face!r}:")
    print(f"  noise_mean                = {result['noise_mean']:.4f}")
    print(f"  noise_std                 = {result['noise_std']:.4f}")
    print(f"  suggested_pixel_threshold = {result['suggested_pixel_threshold']:.4f}   (k={result['k']})")
    print(f"  metric_mean               = {result['metric_mean']:.4f}")
    print(f"  metric_std                = {result['metric_std']:.4f}")
    print(f"  suggested_enter_threshold = {result['suggested_enter_threshold']:.4f}   (enter_k={result['enter_k']})")
    print(f"  suggested_exit_threshold  = {result['suggested_exit_threshold']:.4f}   (exit_k={result['exit_k']})")

    if args.out:
        overrides = {
            "activity": {"pixel_threshold": result["suggested_pixel_threshold"]},
            "rotation": {
                "enter_threshold": result["suggested_enter_threshold"],
                "exit_threshold": result["suggested_exit_threshold"],
            },
        }
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.safe_dump(overrides, f, default_flow_style=False, sort_keys=False)
        print(f"\nwrote suggested thresholds to {args.out!r} - pass via `run`/`replay --config {args.out}`")

    return 0


# =============================================================================================
# run / replay
# =============================================================================================
def _cmd_run(args) -> int:
    try:
        config = _load_run_config(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    calib = _load_calibration_or_report(args.calib)
    if calib is None:
        return 1

    source = _camera_source_from_config(config)
    logger = _build_logger(config, args, _make_run_id())
    marker_detector = _build_marker_detector(config, calib)

    stop_flag = None
    if args.duration is not None:
        deadline = time.monotonic() + float(args.duration)
        stop_flag = lambda: time.monotonic() >= deadline  # noqa: E731

    return _run_pipeline_or_report(
        config, calib, source, logger, marker_detector,
        clock="auto", max_frames=args.max_frames, stop_flag=stop_flag,
    )


def _cmd_replay(args) -> int:
    try:
        config = _load_run_config(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    calib = _load_calibration_or_report(args.calib)
    if calib is None:
        return 1

    source = VideoFileSource(args.video)
    logger = _build_logger(config, args, _make_run_id())
    marker_detector = _build_marker_detector(config, calib)

    # clock="auto" already resolves to the video's own index/fps clock for a VideoFileSource
    # (pipeline.py), which is exactly the "offline dev path bins by content time" behaviour wanted.
    return _run_pipeline_or_report(
        config, calib, source, logger, marker_detector,
        clock="auto", max_frames=args.max_frames, stop_flag=None,
    )


# =============================================================================================
# argument parser
# =============================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flygym-tracker",
        description="Per-vial Drosophila locomotor-activity tracker for the FlyGym v2 rig.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- calibrate --------------------------------------------------------------------------
    p_cal = subparsers.add_parser(
        "calibrate", help="Build a vial-ROI calibration bundle for one drum face (DESIGN.md section 5.4/5.5)."
    )
    frame_src = p_cal.add_mutually_exclusive_group(required=True)
    frame_src.add_argument("--frame", help="Path to a still image of the face.")
    frame_src.add_argument(
        "--from-camera", action="store_true", help="Grab a single still frame from the live camera instead."
    )
    p_cal.add_argument("--face", default="A", help="Face name, e.g. A or B (default: A).")
    p_cal.add_argument("--out", default="calib", help="Output calibration bundle directory (default: calib).")
    p_cal.add_argument(
        "--wizard", action="store_true",
        help="Launch the interactive manual ROI wizard, pre-seeded from auto-detect (primary path per DESIGN.md).",
    )
    p_cal.add_argument("--config", default=None, help="Config YAML (only consulted with --from-camera).")
    p_cal.set_defaults(handler=_cmd_calibrate)

    # -- noise --------------------------------------------------------------------------------
    p_noise = subparsers.add_parser(
        "noise", help="Measure the static-rig noise floor and suggest activity/rotation thresholds."
    )
    p_noise.add_argument("--config", default=None, help="Config YAML (base layer is always the packaged default).")
    p_noise.add_argument("--video", default=None, help="Measure on a recorded (stationary) video instead of the camera.")
    p_noise.add_argument("--calib", required=True, help="Calibration bundle directory (from `calibrate`).")
    p_noise.add_argument("--face", default=None, help="Face to measure (default: the calibration's default face).")
    p_noise.add_argument("--frames", type=int, default=100, help="Number of frames to sample (default: 100).")
    p_noise.add_argument("--out", default=None, help="Write suggested thresholds as a config-override YAML.")
    p_noise.set_defaults(handler=_cmd_noise)

    # -- run ----------------------------------------------------------------------------------
    p_run = subparsers.add_parser("run", help="Live tracking against the HikRobot camera.")
    p_run.add_argument("--config", required=True, help="Config YAML (thresholds normally come from `noise --out`).")
    p_run.add_argument("--calib", required=True, help="Calibration bundle directory (from `calibrate`).")
    p_run.add_argument("--bin-seconds", type=float, default=None, help="Override binning.bin_seconds.")
    p_run.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    p_run.add_argument("--duration", type=float, default=None, help="Stop after this many wall-clock seconds.")
    p_run.add_argument("--out", default=None, help="Output directory (default: config output.dir).")
    p_run.set_defaults(handler=_cmd_run)

    # -- replay -------------------------------------------------------------------------------
    p_replay = subparsers.add_parser(
        "replay", help="Offline dev path: run the same pipeline against a recorded video."
    )
    p_replay.add_argument("--video", required=True, help="Recorded video file (e.g. an .avi clip).")
    p_replay.add_argument("--config", required=True, help="Config YAML (thresholds normally come from `noise --out`).")
    p_replay.add_argument("--calib", required=True, help="Calibration bundle directory (from `calibrate`).")
    p_replay.add_argument("--bin-seconds", type=float, default=None, help="Override binning.bin_seconds.")
    p_replay.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    p_replay.add_argument("--out", default=None, help="Output directory (default: config output.dir).")
    p_replay.set_defaults(handler=_cmd_replay)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())

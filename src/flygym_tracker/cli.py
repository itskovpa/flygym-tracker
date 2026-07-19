"""Command-line entry points for flygym_tracker (DESIGN.md section 4 `cli.py` row, section 7, section 9).

Four subcommands, matching the build order in DESIGN.md section 9 ("validate on the empty rig:
live capture, noise floor, calibration on the real face, rotation detection"):

  * ``calibrate`` -- build a per-face `Calibration` bundle (DESIGN.md section 5.4/5.5) from a still
    image or a single live-camera grab, optionally refined with the interactive manual wizard.
  * ``edit-rois`` -- hand-edit a face's 4-vertex vial ROIs (`roi_editor`) so they follow the
    cylindrical drum's foreshortened edge tubes, then transfer the shapes to the other face
    (`calibration.transfer_quads`) and re-write the bundle.
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
import os
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import yaml

from flygym_tracker.calibrate_wizard import run_wizard
from flygym_tracker.calibration import (
    MarkerCalibParams,
    boxes_from_calibration,
    build_two_face_calibration,
    detect_calibration,
    draw_quad_overlay,
    load_calibration,
    quad_lit_fraction,
    relativize_mask_paths,
    save_calibration,
    suspicious_vials,
    transfer_quads,
    vial_quad,
)
from flygym_tracker.config import load_config
from flygym_tracker.gui_support import gui_diagnosis, has_gui_support, require_gui
from flygym_tracker.frame_source import HikCameraSource, VideoFileSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.marker_band import MarkerBandDetector
from flygym_tracker.markers import MarkerDetector
from flygym_tracker.monitor import LiveMonitor
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


def _run_pipeline_or_report(
    config, calib, source, logger, marker_detector, *, clock, max_frames, stop_flag, monitor=False,
) -> int:
    """Construct + run a `TrackerPipeline`, turning its two documented construction-time failure
    modes into a short message instead of a traceback: null thresholds (ValueError -- DESIGN.md
    section 5.1/5.3, the "last bit" that needs `noise`) and an unreadable calibration mask
    (RuntimeError -- e.g. a hand-edited or half-written calibration bundle).

    `monitor=True` (the `--monitor` flag) wires a `LiveMonitor` (monitor.py) as a pipeline
    observer -- a live tracking/activity window the scientist can watch (and nudge
    `pixel_threshold` from) while the run is in progress, without changing anything about the run
    itself (DESIGN.md's `noise`/output-file wiring is unaffected either way)."""
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

    live_monitor = None
    if monitor:
        if not has_gui_support():
            # Not fatal: the run itself is headless-safe, so warn and keep acquiring.
            print("\nWARNING: --monitor requested but this OpenCV build cannot open a window.\n"
                  + gui_diagnosis("The live monitor")
                  + "\nContinuing WITHOUT the monitor; measurement and logging are unaffected.\n",
                  file=sys.stderr)
            monitor = False
    if monitor:
        live_monitor = LiveMonitor(
            calib, config, on_threshold_change=lambda v: setattr(pipe, "pixel_threshold", v),
        )
        pipe.add_observer(live_monitor.on_frame)
        pipe.add_bin_observer(live_monitor.on_bin)

    try:
        summary = pipe.run(max_frames=max_frames, stop_flag=stop_flag)
    finally:
        if live_monitor is not None:
            live_monitor.close()
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
        require_gui("The calibration wizard")
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
# calibrate-faces (marker-band driven, both drum faces from one flip video)
# =============================================================================================
def _marker_params_from_config(config) -> Optional[MarkerCalibParams]:
    """Build a `MarkerCalibParams` from an optional ``calibration.marker_params`` config block.

    The packaged config has no such block, so this normally returns None (= library defaults).
    It exists so an odd rig can be re-tuned from a YAML file instead of a code change; unknown
    keys are rejected loudly rather than silently ignored.
    """
    calibration_cfg = config.get("calibration") if config is not None else None
    raw = calibration_cfg.get("marker_params") if calibration_cfg is not None else None
    if raw is None:
        return None
    raw = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
    known = set(MarkerCalibParams.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            "unknown calibration.marker_params key(s): %s (valid: %s)"
            % (", ".join(unknown), ", ".join(sorted(known)))
        )
    return MarkerCalibParams(**raw)


def _cmd_calibrate_faces(args) -> int:
    """Calibrate BOTH drum faces from one flip video, using the physical marker band.

    This is the 32-vial path: the rig shows Face A then Face B (16 vials each), so a clip that
    contains at least one flip carries everything needed to calibrate both at once. Vial x
    positions come from the marker band (a measurement) rather than a brightness guess, and NO
    slot is auto-excluded -- see `calibration`'s module docstring for why that asymmetry matters.
    """
    config = None
    if args.config is not None:
        try:
            config = load_config(path=args.config)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    try:
        params = _marker_params_from_config(config)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    detector = MarkerBandDetector()
    try:
        calib = build_two_face_calibration(args.video, detector, args.out, params=params)
    except ValueError as e:
        print(
            f"error: {e}\n"
            "Check that the clip contains at least one full dwell and that the marker band is "
            "visible; otherwise calibrate each face by hand with `calibrate --wizard`.",
            file=sys.stderr,
        )
        return 2

    total = sum(len(fc.vials) for fc in calib.faces.values())
    print(f"calibration saved to {args.out!r}")
    print(f"  faces found : {', '.join(sorted(calib.faces))} ({len(calib.faces)})")
    print(f"  vials       : {total} total")
    for name in sorted(calib.faces):
        fc = calib.faces[name]
        flagged = suspicious_vials(fc)
        gvids = [_face_index(calib, name) * 16 + v.id for v in fc.vials]
        print(f"  face {name}: {len(fc.vials)} vials (global ids {min(gvids)}-{max(gvids)}), "
              f"all present=True")
        if flagged:
            print(f"    SUSPICIOUS (little of the slot is lit -- still measured, review "
                  f"overlay_{name}.png): local ids {flagged}")
        else:
            print("    no suspicious slots")
    print(f"  notes       : {calib.notes}")
    return 0


def _face_index(calib, name: str) -> int:
    """Face ordinal used for global vial ids (matches `pipeline.TrackerPipeline`)."""
    return sorted(calib.faces).index(name)


# =============================================================================================
# edit-rois (manual QUAD ROI editor + transfer to the other face)
# =============================================================================================
def _grab_video_frame(video_path: str, index: int):
    """Grab frame `index` from a video as grayscale, or None.

    Tries a seek first and falls back to reading sequentially -- seeking is unreliable on some
    AVI/codec combinations and silently lands on the wrong frame, which would put the operator in
    front of the WRONG drum face without saying so.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        if index > 0 and cap.set(cv2.CAP_PROP_POS_FRAMES, float(index)):
            ok, frame = cap.read()
            if ok and abs(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - (index + 1)) <= 1:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            cap.release()
            cap = cv2.VideoCapture(video_path)
        for i in range(index + 1):
            ok, frame = cap.read()
            if not ok:
                return None
            if i == index:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    finally:
        cap.release()
    return None


def _face_source_frame(calib, face: str, args):
    """The still to edit/draw a face on: --frame, --video (the face's own calibration frame), or
    the saved overlay as a last resort. Returns ``(gray, description)`` or ``(None, reason)``."""
    if getattr(args, "frame", None):
        gray = cv2.imread(args.frame, cv2.IMREAD_GRAYSCALE)
        return (gray, args.frame) if gray is not None else (None, f"could not read {args.frame!r}")
    if getattr(args, "video", None):
        marker = calib.faces[face].marker
        idx = int(marker.get("source_frame", 0)) if isinstance(marker, dict) else 0
        gray = _grab_video_frame(args.video, idx)
        if gray is None:
            return None, f"could not read frame {idx} of {args.video!r}"
        return gray, f"{args.video} frame {idx} (face {face}'s own calibration frame)"
    path = os.path.join(args.calib, "overlay_%s.png" % face)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, ("no --frame/--video given and no %s to fall back on" % path)
    return gray, path + " (saved overlay -- pass --frame/--video for a clean image)"


def _lit_report(face_cal, illum_mask):
    """``{vial_id: lit_fraction}`` for a face -- the before/after evidence an edit actually helped."""
    return {int(v.id): quad_lit_fraction(vial_quad(v), illum_mask) for v in face_cal.vials}


def _cmd_edit_rois(args) -> int:
    """Hand-edit one face's 4-vertex vial ROIs, then mirror the shapes onto the other face.

    The drum is cylindrical, so edge tubes are foreshortened and no axis-aligned rectangle can
    follow them (DESIGN.md section 2's non-uniform, curved geometry). This is the escape hatch the
    rig owner asked for: shape ONE face by hand before an experiment, save, and have both faces
    covered for the whole run -- the two present in the SAME orientation, so
    `calibration.transfer_quads` maps shapes across directly and re-snaps them to the destination
    face's own marker-derived columns.
    """
    from flygym_tracker.roi_editor import run_roi_editor  # interactive-only import

    calib = _load_calibration_or_report(args.calib)
    if calib is None:
        return 1
    face = args.face or ("A" if "A" in calib.faces else sorted(calib.faces)[0])
    if face not in calib.faces:
        print(f"error: face {face!r} not in calibration (have: {sorted(calib.faces)})", file=sys.stderr)
        return 1

    illum = cv2.imread(calib.faces[face].illum_mask_path, cv2.IMREAD_GRAYSCALE)
    if illum is None:
        print(f"error: could not read illum mask at {calib.faces[face].illum_mask_path!r}", file=sys.stderr)
        return 1
    gray, source = _face_source_frame(calib, face, args)
    if gray is None:
        print(f"error: {source}", file=sys.stderr)
        return 1

    require_gui("The ROI editor")
    before = _lit_report(calib.faces[face], illum)
    print(f"editing face {face!r} from {source}")
    edited = run_roi_editor(gray, calib.faces[face], illum)
    if edited is None:
        print("cancelled; calibration bundle left unchanged")
        return 0

    calib.faces[face] = edited
    after = _lit_report(edited, illum)
    overlays = {face: draw_quad_overlay(gray, edited, illum)}

    transferred = []
    if not args.no_transfer:
        for other in sorted(n for n in calib.faces if n != face):
            calib.faces[other] = transfer_quads(
                edited, calib.faces[other], image_size=(calib.image_width, calib.image_height))
            transferred.append(other)
            other_gray, _ = _face_source_frame(calib, other, args) if args.video else (None, "")
            other_mask = cv2.imread(calib.faces[other].illum_mask_path, cv2.IMREAD_GRAYSCALE)
            if other_gray is not None:
                overlays[other] = draw_quad_overlay(other_gray, calib.faces[other], other_mask)

    # Mask paths were absolutized by `load_calibration`; undo that or the re-saved bundle bakes in
    # this machine's directory and stops being movable.
    relativize_mask_paths(calib)
    save_calibration(calib, {}, args.calib, overlay=overlays)

    print(f"\nsaved to {args.calib!r}: face {face!r} quads updated"
          + (f", transferred to face(s) {', '.join(transferred)}" if transferred else ""))
    print("  vial   lit before -> after")
    for vid in sorted(after):
        b, a = before.get(vid, float("nan")), after[vid]
        flag = "  <-- worse" if a < b - 0.005 else ("  ++" if a > b + 0.005 else "")
        print(f"  {vid:>4}   {b:9.3f} -> {a:.3f}{flag}")
    mean_b = sum(before.values()) / max(1, len(before))
    mean_a = sum(after.values()) / max(1, len(after))
    print(f"  mean   {mean_b:9.3f} -> {mean_a:.3f}")
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
        clock="auto", max_frames=args.max_frames, stop_flag=stop_flag, monitor=args.monitor,
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
        clock="auto", max_frames=args.max_frames, stop_flag=None, monitor=args.monitor,
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

    # -- calibrate-faces ----------------------------------------------------------------------
    p_faces = subparsers.add_parser(
        "calibrate-faces",
        help="Calibrate BOTH drum faces (32 vials) from one flip video, using the marker band.",
        description=(
            "Marker-band calibration of both drum faces from a single video containing at least "
            "one 180-degree flip. Vial columns are read from the rig's physical IR sticker band "
            "rather than inferred from brightness, and every slot is emitted present=True -- "
            "slots that look empty are flagged for review, never silently excluded."
        ),
    )
    p_faces.add_argument("--video", required=True, help="Flip video (must contain >= 1 dwell per face).")
    p_faces.add_argument("--out", required=True, help="Output calibration bundle directory.")
    p_faces.add_argument(
        "--config", default=None,
        help="Config YAML; an optional `calibration.marker_params` block overrides the tuning defaults.",
    )
    p_faces.set_defaults(handler=_cmd_calibrate_faces)

    # -- edit-rois ----------------------------------------------------------------------------
    p_edit = subparsers.add_parser(
        "edit-rois",
        help="Hand-edit one face's 4-vertex vial ROIs, then transfer the shapes to the other face.",
        description=(
            "Interactive quad-ROI editor. The drum is cylindrical, so tubes near the left/right "
            "edge are foreshortened and an axis-aligned rectangle cannot follow them. Drag the "
            "VERTICES of a vial's ROI (with a magnifier for precision), press 'c' to copy that "
            "shape to every vial, 's' to save. Saving writes the edited quads back to the bundle "
            "and, unless --no-transfer, maps them onto the other drum face as well -- the two "
            "faces present in the same orientation, so shapes carry across directly and are "
            "re-snapped to the destination face's own marker-derived columns."
        ),
    )
    p_edit.add_argument("--calib", required=True, help="Calibration bundle directory to edit IN PLACE.")
    p_edit.add_argument("--face", default=None, help="Face to edit (default: A, or the first face).")
    edit_src = p_edit.add_mutually_exclusive_group()
    edit_src.add_argument("--frame", default=None, help="Still image of the face to edit on.")
    edit_src.add_argument(
        "--video", default=None,
        help="Video the bundle was calibrated from; each face's own calibration frame is pulled "
             "from it (and both overlays are regenerated).",
    )
    p_edit.add_argument(
        "--no-transfer", action="store_true",
        help="Edit this face only; leave the other face's ROIs alone.",
    )
    p_edit.set_defaults(handler=_cmd_edit_rois)

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
    p_run.add_argument(
        "--monitor", action="store_true",
        help="Show a live tracking/activity monitor window while running (monitor.py).",
    )
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
    p_replay.add_argument(
        "--monitor", action="store_true",
        help="Show a live tracking/activity monitor window while replaying (monitor.py).",
    )
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

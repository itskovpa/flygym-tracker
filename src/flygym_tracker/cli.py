"""Command-line entry points for flygym_tracker (DESIGN.md section 4 `cli.py` row, section 7, section 9).

Four subcommands, matching the build order in DESIGN.md section 9 ("validate on the empty rig:
live capture, noise floor, calibration on the real face, rotation detection"):

  * ``select-vials`` -- THE START OF EVERY SESSION: the operator draws each vial as a polygon on
    the LIVE feed (`live_vial_selector`) and that bundle is the calibration. No detection is run.
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
    VIALS_PER_FACE,
    boxes_from_calibration,
    build_two_face_calibration,
    calibration_band_faces,
    calibration_signature_faces,
    detect_calibration,
    draw_quad_overlay,
    load_calibration,
    marker_detector_from_calibration,
    quad_lit_fraction,
    relativize_mask_paths,
    save_calibration,
    suspicious_vials,
    transfer_quads,
    vial_quad,
    vial_shape,
)
from flygym_tracker.config import load_config
from flygym_tracker.gui_support import gui_diagnosis, has_gui_support, require_gui
from flygym_tracker.frame_source import HikCameraSource, VideoFileSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.marker_band import MarkerBandDetector
from flygym_tracker.markers import MarkerDetector
from flygym_tracker.monitor import LiveMonitor
from flygym_tracker.pipeline import TrackerPipeline, measure_noise
from flygym_tracker.settings_panel import (
    SettingsWindow,
    build_settings,
    save_settings_to_yaml,
    startup_banner,
)

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


def face_id_readiness(config, calib) -> list:
    """Lines warning that this bundle cannot identify drum faces. Empty list = it can (or need not).

    THE REGRESSION THIS EXISTS FOR. A two-face bundle with no marker data runs perfectly happily:
    `identify_face` returns None on every frame, the pipeline falls back to one face, and a
    3-day experiment fills its CSV with face-B vials labelled face A. Nothing failed, nothing
    was empty, and the only trace was a repeated `marker_absent` line buried in the event log.
    A run that cannot possibly produce correct face identities has to SAY SO at startup, while
    it still costs 20 seconds to fix instead of three days.

    A SINGLE-face bundle never warns: with one face there is nothing to identify, and inventing
    a marker requirement for it would break the rigs that only ever show one side.
    """
    if len(calib.faces) < 2:
        return []
    if len(calibration_band_faces(calib)) >= 2 or len(calibration_signature_faces(calib)) >= 2:
        return []

    default = sorted(calib.faces)[0]
    lines = [
        "WARNING: this calibration covers %d drum faces but carries no marker templates."
        % len(calib.faces),
        "         It CANNOT identify faces: all activity will be attributed to face %s and the"
        % default,
        "         other face's vials will never appear in the output.",
        "         Fix: run the face-learning step (`select-vials`, and answer yes when it offers",
        "         to learn the drum faces) before starting a real experiment.",
    ]
    if not bool(config.markers.enabled):
        # Both halves of the original failure at once: no templates AND markers switched off.
        # Turning markers on alone would not help, so say what actually has to happen.
        lines.insert(1, "         `markers.enabled` is also false, so face identification is "
                        "switched off entirely.")
    return lines


def _build_marker_detector(config, calib):
    """Build the face-ID detector this bundle can actually use, preferring the validated one.

    ORDER MATTERS, and getting it wrong is what caused the bug this function was rewritten for.
    `marker_band.MarkerBandDetector` is the scheme validated on real rig footage (43/43 dwells,
    and 943/943 stationary frames of `Good Markers.avi`); it is what both calibration flows
    produce, storing its templates in ``FaceCalibration.marker["band_templates"]``.
    `markers.MarkerDetector` is the older generic contour scheme, which reads
    ``marker["signature"]`` -- a key NO current flow writes. This function used to build only the
    generic one, so it was always handed an empty registry, always returned None from
    `identify_face`, and the tested-and-validated detector was never in the run path at all.

    So: band templates win; the generic detector is built only for a bundle that really carries
    contour signatures; and a bundle with neither is reported by `face_id_readiness` rather than
    quietly starting a run that can only ever produce half the data.
    """
    for line in face_id_readiness(config, calib):
        print(line, file=sys.stderr)

    if calibration_band_faces(calib):
        return marker_detector_from_calibration(calib)

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


def _settings_saver(config_path: Optional[str]):
    """A save hook for the settings panel, or None when there is no file to write to.

    Prints what it wrote and where, ALWAYS. The panel's ``s`` key changes a file the operator is
    not looking at; a save that produced no visible output would leave them unable to tell a
    successful write from a silently skipped one, and the next run would then be a mystery.
    """
    if not config_path:
        return None

    def save(model):
        notes = save_settings_to_yaml(config_path, model)
        if notes:
            print("\nsettings: wrote %d change(s) to %s" % (len(notes), config_path))
        else:
            print("\nsettings: nothing changed, %s left alone" % config_path)
        return notes

    return save


def _settings_gui_available(flag_name: str) -> bool:
    """True if a settings window can be opened; warns (without failing the run) if it cannot."""
    if has_gui_support():
        return True
    # Not fatal: the run itself is headless-safe, so warn and keep acquiring.
    print("\nWARNING: %s requested but this OpenCV build cannot open a window.\n" % flag_name
          + gui_diagnosis("The settings panel")
          + "\nContinuing WITHOUT it; measurement and logging are unaffected.\n",
          file=sys.stderr)
    return False


def _run_pipeline_or_report(
    config, calib, source, logger, marker_detector, *, clock, max_frames, stop_flag, monitor=False,
    settings=False, config_path=None,
) -> int:
    """Construct + run a `TrackerPipeline`, turning its two documented construction-time failure
    modes into a short message instead of a traceback: null thresholds (ValueError -- DESIGN.md
    section 5.1/5.3, the "last bit" that needs `noise`) and an unreadable calibration mask
    (RuntimeError -- e.g. a hand-edited or half-written calibration bundle).

    `monitor=True` (the `--monitor` flag) wires a `LiveMonitor` (monitor.py) as a pipeline
    observer -- a live tracking/activity window the scientist can watch (and nudge
    `pixel_threshold` from) while the run is in progress, without changing anything about the run
    itself (DESIGN.md's `noise`/output-file wiring is unaffected either way).

    `settings=True` (the `--settings` flag) opens the settings panel (settings_panel.py) BEFORE
    the first frame is measured, and holds the run until the operator closes it. That ordering is
    the point: values chosen in the panel are in force for frame 1, so a `replay --settings`
    against the same clip is a clean A/B -- adjust, close, watch, re-run -- with no leading
    stretch of data measured at the old settings. The same panel is reachable mid-run with the
    monitor's ``t`` key, and BOTH routes end at `TrackerPipeline.apply_setting`, so either way the
    change is logged as a `setting_change` event.
    """
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

    # One model shared by the modal panel and the monitor's `t` panel, so a value set before the
    # run is still shown (and still marked as changed) by the panel opened during it.
    settings_model = build_settings(config, pipeline=pipe)
    save_hook = _settings_saver(config_path)

    if settings and _settings_gui_available("--settings"):
        SettingsWindow(
            settings_model, on_change=pipe.apply_setting, on_save=save_hook,
            blocked=pipe.setting_block_reason,
            subtitle="applies from the first frame - close this window to start",
        ).run()
        # The snapshot written when the logger was built records the config as LOADED. Anything
        # adjusted just now is in force for frame 1, so without this the folder would claim its
        # data was measured at a threshold it never used. `apply_setting` deliberately logs no
        # event before the first frame -- nothing was re-measured, only chosen -- which makes
        # this the ONLY record of what the run actually started with.
        changed = settings_model.changed()
        if changed:
            logger.update_meta({
                "config": settings_model.to_overrides(),
                "settings_adjusted_before_start": [
                    "%s: %s -> %s" % (s.key, settings_model.baseline(s.key), s.value)
                    for s in changed
                ],
            })
            print("run starts with %d adjusted setting(s); run_meta.json records the values used"
                  % len(changed))

    live_monitor = None
    if monitor:
        if not has_gui_support():
            print("\nWARNING: --monitor requested but this OpenCV build cannot open a window.\n"
                  + gui_diagnosis("The live monitor")
                  + "\nContinuing WITHOUT the monitor; measurement and logging are unaffected.\n",
                  file=sys.stderr)
            monitor = False
    if monitor:
        live_monitor = LiveMonitor(
            calib, config,
            # One route for every change made from the monitor -- the +/- keys and the `t` panel
            # alike -- so each one is applied AND logged exactly once. (`on_threshold_change` is
            # deliberately not wired: it would be the second of two callbacks for the same key.)
            on_setting_change=pipe.apply_setting,
            settings_model=settings_model,
            settings_on_save=save_hook,
            settings_blocked=pipe.setting_block_reason,
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


def _calibration_for_round(args, source):
    """Every round starts here: offer the saved vial positions, else draw them on the live feed.

    This is the requirement in one function -- "before each round prompt to load the vial
    positions from disk; if the user opts out, go on to drawing". `run` and `replay` both come
    through it, so the rig behaves the same whichever one was launched, and neither can start
    measuring against a bundle the operator never confirmed.

    The source is handed over ALREADY OPEN (the selector opens it and deliberately does not
    close it), so the camera is grabbed exactly once for both the drawing and the experiment --
    it allows only one program at a time. A video is rewound instead, so the run still sees the
    whole clip and not just what was left after the drawing.

    Returns the `Calibration`, or None with the reason already printed.
    """
    from flygym_tracker.live_vial_selector import load_or_select_vials  # interactive-only import

    reuse = True if getattr(args, "reuse", False) else (
        False if getattr(args, "redraw", False) else None)
    n_vials = getattr(args, "n_vials", VIALS_PER_FACE)
    try:
        result = load_or_select_vials(source, args.calib, n_vials=n_vials, reuse=reuse)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        if not _looks_like_camera_busy(e) or not _offer_to_free_the_camera():
            if _looks_like_camera_busy(e):
                print(_CAMERA_BUSY_HINT, file=sys.stderr)
            return None
        # Something WAS holding it and has been stopped, so the thing that just failed is now
        # possible. Retrying here saves re-running the whole menu with all its answers again.
        try:
            result = load_or_select_vials(source, args.calib, n_vials=n_vials, reuse=reuse)
        except (RuntimeError, ValueError) as e2:
            print(f"error: {e2}", file=sys.stderr)
            print(_CAMERA_BUSY_HINT, file=sys.stderr)
            return None

    if not result.polygons:
        print("no vials were selected; nothing to track", file=sys.stderr)
        return None
    if isinstance(source, VideoFileSource):
        source.close()   # reopened (and so rewound to frame 0) by the pipeline

    n_faces = len(result.calibration.faces)
    print(f"tracking {result.n_vials} vial(s) per face on {n_faces} face(s) "
          f"= {result.n_vials * n_faces} vials"
          + (" (positions loaded from disk)" if result.reused else " (just drawn, and saved)"))
    return result.calibration


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
# select-vials (hand-drawn polygons on the LIVE feed -- how every session starts)
# =============================================================================================
#: USB3 Vision is exclusive, and "camera busy" is BY FAR the most common failure on this rig.
_CAMERA_BUSY_HINT = (
    "The camera is already open somewhere else. USB3 Vision access is EXCLUSIVE:\n"
    "  1. close the MVS Viewer (its exit dialog asks 'Exit the client?' -- click OK),\n"
    "  2. close any other Bonsai/Python session holding the camera,\n"
    "  3. run this command again.\n"
    "Or pass --video <clip> to draw the vials on a recorded clip instead."
)


def _looks_like_camera_busy(exc: Exception) -> bool:
    text = str(exc).lower()
    return "already in use" in text or "may already be in use" in text or "openDevice".lower() in text


def _offer_to_free_the_camera() -> bool:
    """On a busy camera, name what is holding it and offer to stop it. True if something was.

    Telling the operator to "close any other session" is useless when the holder is a headless
    `Bonsai.exe --start --no-editor`: it has no window, no taskbar entry, and nothing on screen
    to close, so the rig looks idle while staying locked. This finds it by name and PID.
    """
    from flygym_tracker import camera_lock          # imported lazily: shells out to PowerShell

    print("\nlooking for what is holding the camera...", file=sys.stderr)
    try:
        return camera_lock.prompt_and_release() > 0
    except Exception as e:                          # a diagnostic must never mask the real error
        print(f"(could not check which program holds the camera: {e})", file=sys.stderr)
        return False


def _requested_faces(args) -> list:
    """Faces the drawn polygons apply to: ``--faces A,B`` (default), or ``--face X`` for one."""
    if getattr(args, "face", None):
        return [args.face]
    return [f.strip() for f in str(args.faces).split(",") if f.strip()]


def _cmd_select_vials(args) -> int:
    """Offer the saved vial positions, else draw them live; then save. No detection anywhere.

    Automatic vial detection is not used on this rig (see `live_vial_selector`'s module docstring
    -- it produced ROIs the operator had to fix by hand every time). A round starts here: reuse
    what was drawn before, or draw all 16 vials as polygons while watching the real feed. Either
    way THAT is the calibration -- what was drawn is what gets measured, on both drum faces.
    """
    from flygym_tracker.live_vial_selector import load_or_select_vials  # interactive-only import

    if args.video:
        source, where = VideoFileSource(args.video), args.video
    else:
        try:
            config = load_config(path=args.config)
        except (FileNotFoundError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        source, where = _camera_source_from_config(config), "the live camera"

    faces = _requested_faces(args)
    reuse = True if args.reuse else (False if args.redraw else None)
    try:
        result = load_or_select_vials(
            source, args.out, n_vials=args.n_vials, faces=faces, reuse=reuse)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        if _looks_like_camera_busy(e):
            if not _offer_to_free_the_camera():
                print(_CAMERA_BUSY_HINT, file=sys.stderr)
            else:
                print("\nthe camera is free now - run this again", file=sys.stderr)
        return 1
    finally:
        # Closed HERE, not with `with source:`: a close() failure must never throw away polygons
        # the operator just spent minutes clicking.
        try:
            source.close()
        except Exception as e:
            print(f"warning: closing the frame source failed: {e}", file=sys.stderr)

    if not result.polygons:
        print("no vials were selected; nothing saved")
        return 0

    calib = result.calibration
    if result.reused:
        print(f"\nreusing the vial positions already saved in {args.out!r}")
    else:
        print(f"\nsaved vial positions to {args.out!r} "
              f"(reusable next round -- you will be asked before any redraw)")
    print(f"  faces       : {', '.join(sorted(calib.faces))}   "
          f"(identical polygon coordinates on each)")
    print(f"  vials       : {result.n_vials} per face, {result.n_vials * len(calib.faces)} total")
    print(f"  frame       : {calib.image_width}x{calib.image_height}"
          + ("" if result.reused else f" from {where}"))
    if result.n_vials < args.n_vials and not result.reused:
        print(f"  NOTE: finished early -- only {result.n_vials} of {args.n_vials} vial(s) were "
              f"drawn. Re-run with --redraw to do this over.")

    face = sorted(calib.faces)[0]
    mask = cv2.imread(calib.faces[face].illum_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        print("  vial   points   lit fraction")
        for v in calib.faces[face].vials:
            lit = quad_lit_fraction(vial_shape(v), mask)
            flag = "   <-- little of this polygon is lit; check the overlay" if lit < 0.5 else ""
            n_points = len(v.polygon) if v.polygon is not None else 4
            print(f"  {v.id:>4}   {n_points:>6}   {lit:.3f}{flag}")
    print(f"  overlay     : {os.path.join(args.out, 'overlay_%s.png' % face)}")
    return 0


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

    source = _camera_source_from_config(config)
    # Vial positions FIRST, on the same camera handle the run will use: the round either reuses
    # what is saved or the operator draws it, and only then does anything get measured.
    calib = _calibration_for_round(args, source)
    if calib is None:
        try:
            source.close()
        except Exception:
            pass
        return 1

    logger = _build_logger(config, args, _make_run_id())
    marker_detector = _build_marker_detector(config, calib)

    stop_flag = None
    if args.duration is not None:
        deadline = time.monotonic() + float(args.duration)
        stop_flag = lambda: time.monotonic() >= deadline  # noqa: E731

    return _run_pipeline_or_report(
        config, calib, source, logger, marker_detector,
        clock="auto", max_frames=args.max_frames, stop_flag=stop_flag, monitor=args.monitor,
        settings=getattr(args, "settings", False), config_path=args.config,
    )


def _cmd_replay(args) -> int:
    try:
        config = _load_run_config(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    source = VideoFileSource(args.video)
    calib = _calibration_for_round(args, source)
    if calib is None:
        try:
            source.close()
        except Exception:
            pass
        return 1

    logger = _build_logger(config, args, _make_run_id())
    marker_detector = _build_marker_detector(config, calib)

    # clock="auto" already resolves to the video's own index/fps clock for a VideoFileSource
    # (pipeline.py), which is exactly the "offline dev path bins by content time" behaviour wanted.
    return _run_pipeline_or_report(
        config, calib, source, logger, marker_detector,
        clock="auto", max_frames=args.max_frames, stop_flag=None, monitor=args.monitor,
        settings=getattr(args, "settings", False), config_path=args.config,
    )


# =============================================================================================
# settings (edit + save the tuning values, between runs)
# =============================================================================================
def _probe_camera_for_limits(config):
    """Open the camera JUST to read its real min/max/increment, or explain why we did not.

    Returns ``(camera_or_None, note_lines)``. Opt-in (``--probe-camera``) and never fatal.

    WHY OPT-IN. USB3 Vision access is EXCLUSIVE -- one process at a time. If this command grabbed
    the camera by default, then editing a value while an experiment was running would either fail,
    or (worse, on a rig where the run had not started yet) hold the camera so the RUN could not
    have it. Reading limits is a convenience; blocking an experiment for it is not a trade anyone
    would take, so the default is to use the documented ranges and say so on screen.

    When it IS asked for and the camera is busy, `camera_lock` names the holder rather than
    printing the SDK's culprit-free "0x80000203" -- but nothing is stopped. This is a read-only
    tuning command; ending someone's acquisition to draw a slider would be absurd.
    """
    source = _camera_source_from_config(config)
    try:
        source.open()
    except Exception as exc:
        lines = ["could not open the camera, so the limits shown are the rig camera's, not live:",
                 "  %s" % exc]
        if _looks_like_camera_busy(exc):
            try:
                from flygym_tracker import camera_lock
                lines.append(camera_lock.report(camera_lock.find_camera_holders()))
            except Exception:
                pass
        return None, lines
    return source, ["camera open: the limits shown are this sensor's own"]


def _cmd_settings(args) -> int:
    """Open the settings panel against the config, adjust, save with ``s``, close.

    THE POINT OF THIS COMMAND is that the panel existed but could not be found: it was reachable
    only from `run`/`replay --settings` or the monitor's ``t`` key, i.e. only while something was
    already running, and `run.bat` never passed the flag. Between-runs tuning had no entry point at
    all. So this one needs NO camera and NO calibration bundle -- it edits a YAML file, and
    demanding a rig be present to do that is what made the feature invisible in the first place.
    """
    try:
        config = load_config(path=args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    camera, notes = (None, [])
    if args.probe_camera:
        camera, notes = _probe_camera_for_limits(config)
    for line in notes:
        print(line)

    try:
        model = build_settings(config, camera=camera)
        if args.list:
            print(startup_banner(model))
            return 0
        if not has_gui_support():
            print("\nERROR: " + gui_diagnosis("The settings panel")
                  + "\n(`settings --list` prints the same values without a window.)",
                  file=sys.stderr)
            return 2
        SettingsWindow(
            model, on_save=_settings_saver(args.config),
            subtitle="editing %s - press s to save, q to close" % (args.config or "the defaults"),
        ).run()
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception as e:
                print(f"warning: closing the camera failed: {e}", file=sys.stderr)

    changed = model.changed()
    if changed:
        # `s` already saved anything the operator meant to keep; this is the "you closed without
        # saving" case, and staying silent about it would lose work with no trace.
        print("\n%d setting(s) were changed but NOT saved (press s in the panel to save):"
              % len(changed))
        for s in changed:
            print("  %s: %s -> %s" % (s.key, model.baseline(s.key), s.value))
    else:
        print("\nsettings closed; %s is unchanged" % (args.config or "the config"))
    return 0


# =============================================================================================
# argument parser
# =============================================================================================
def _cmd_free_camera(args) -> int:
    """Name whatever is holding the camera and offer to stop it (`camera_lock`)."""
    from flygym_tracker import camera_lock

    holders = camera_lock.find_camera_holders()
    if args.list:
        print(camera_lock.report(holders))
        return 0
    if args.yes:
        # Unattended: still print everything first, so the log says exactly what was ended.
        print(camera_lock.report(holders))
        stopped = camera_lock.release_camera(holders, confirm=lambda _h: True)
        for holder in stopped:
            print("stopped PID %d  %s" % (holder.pid, holder.what or holder.name))
        return 0 if stopped or not holders else 1
    return 0 if camera_lock.prompt_and_release() or not holders else 1


def _add_selection_flags(parser) -> None:
    """The vial-position flags shared by `run` and `replay`.

    With neither flag the round ASKS (the documented behaviour); the flags exist so an unattended
    or scripted start can answer in advance instead of blocking on a prompt nobody will see.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reuse", action="store_true",
                       help="Use the saved vial positions without asking.")
    group.add_argument("--redraw", action="store_true",
                       help="Always draw the vials, ignoring anything saved.")
    parser.add_argument("--n-vials", type=int, default=VIALS_PER_FACE,
                        help=f"Vials to draw per face when drawing (default: {VIALS_PER_FACE}).")


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

    # -- select-vials -------------------------------------------------------------------------
    p_select = subparsers.add_parser(
        "select-vials",
        help="Draw every vial by hand as a polygon on the LIVE feed (how a session starts).",
        description=(
            "Live-video vial selection. Automatic vial detection is NOT used on this rig, so "
            "every round begins here. If the output folder already holds vial positions you are "
            "asked whether to load them (ENTER = yes) and no drawing happens at all. Otherwise: "
            "watch the live camera, click a vertex at a time around each vial, press ENTER to "
            "store it and move to the next. BACKSPACE removes the last vertex, 'u' re-opens the "
            "previous vial, 'c' clears the one in progress, SPACE freezes the feed for precise "
            "clicking, and q/ESC finishes early with whatever has been drawn. What you draw IS "
            "the measured region -- nothing is fitted or snapped. You draw ONE face; both drum "
            "faces are written with the same coordinates, since they present in the same "
            "orientation. The result is saved immediately and offered back next round."
        ),
    )
    p_select.add_argument("--out", required=True, help="Calibration bundle directory to write.")
    face_sel = p_select.add_mutually_exclusive_group()
    face_sel.add_argument(
        "--faces", default="A,B",
        help="Faces these polygons apply to, comma separated (default: A,B -- one drawing, "
             "32 vials, identical coordinates on both drum faces).",
    )
    face_sel.add_argument(
        "--face", default=None,
        help="Shorthand for a SINGLE-face bundle, e.g. --face A.",
    )
    p_select.add_argument(
        "--video", default=None,
        help="Select on a recorded clip instead of the live camera (dry runs, or a rig you are "
             "not standing at).",
    )
    p_select.add_argument("--n-vials", type=int, default=16,
                          help="Vials to draw per face (default: 16).")
    reuse_sel = p_select.add_mutually_exclusive_group()
    reuse_sel.add_argument("--reuse", action="store_true",
                           help="Load saved vial positions without asking (for scripts).")
    reuse_sel.add_argument("--redraw", action="store_true",
                           help="Always draw, ignoring any saved vial positions.")
    p_select.add_argument(
        "--config", default=None,
        help="Config YAML for the camera settings (only consulted without --video).",
    )
    p_select.set_defaults(handler=_cmd_select_vials)

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

    # -- settings -----------------------------------------------------------------------------
    p_settings = subparsers.add_parser(
        "settings",
        help="Open the tracking + camera settings panel, adjust, save, close (no camera needed).",
        description=(
            "Adjust the tracking and camera settings BETWEEN runs and save them back to the "
            "config file. Drag a slider (or use the arrow keys), press 's' to save and 'q' to "
            "close. Camera rows are tri-state: each is either an explicit value this software "
            "sends, or the camera's own default, in which case NOTHING is sent and the camera "
            "keeps whatever MVS left it at -- press 'd' (or click the [d] badge) to put a row "
            "back to that. Width and height only take effect when acquisition starts, which is "
            "why they belong here rather than in a panel opened during a run. Needs no camera "
            "and no calibration bundle; without --probe-camera the limits shown are documented "
            "values and the panel says so."
        ),
    )
    p_settings.add_argument("--config", default=None,
                            help="Config YAML to edit (default: the packaged defaults only).")
    p_settings.add_argument(
        "--probe-camera", action="store_true",
        help="Briefly open the camera to read its REAL limits. Off by default: USB3 Vision "
             "access is exclusive, so grabbing the camera here could block a run.",
    )
    p_settings.add_argument("--list", action="store_true",
                            help="Print the settings and exit, without opening a window.")
    p_settings.set_defaults(handler=_cmd_settings)

    # -- free-camera --------------------------------------------------------------------------
    p_free = subparsers.add_parser(
        "free-camera",
        help="Find what is holding the camera (often an invisible headless Bonsai) and stop it.",
    )
    p_free.add_argument("--list", action="store_true",
                        help="Only show what holds the camera; stop nothing.")
    p_free.add_argument("--yes", action="store_true",
                        help="Stop them without asking (for unattended restarts).")
    p_free.set_defaults(handler=_cmd_free_camera)

    # -- run ----------------------------------------------------------------------------------
    p_run = subparsers.add_parser("run", help="Live tracking against the HikRobot camera.")
    p_run.add_argument("--config", required=True, help="Config YAML (thresholds normally come from `noise --out`).")
    p_run.add_argument(
        "--calib", required=True,
        help="Vial-position folder. Offered back at the start of the round; drawn live if empty.")
    _add_selection_flags(p_run)
    p_run.add_argument("--bin-seconds", type=float, default=None, help="Override binning.bin_seconds.")
    p_run.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    p_run.add_argument("--duration", type=float, default=None, help="Stop after this many wall-clock seconds.")
    p_run.add_argument("--out", default=None, help="Output directory (default: config output.dir).")
    p_run.add_argument(
        "--monitor", action="store_true",
        help="Show a live tracking/activity monitor window while running (monitor.py).",
    )
    p_run.add_argument(
        "--settings", action="store_true",
        help="Open the settings panel before the first frame; close it to start. "
             "(With --monitor, 't' reopens it at any time during the run.)",
    )
    p_run.set_defaults(handler=_cmd_run)

    # -- replay -------------------------------------------------------------------------------
    p_replay = subparsers.add_parser(
        "replay", help="Offline dev path: run the same pipeline against a recorded video."
    )
    p_replay.add_argument("--video", required=True, help="Recorded video file (e.g. an .avi clip).")
    p_replay.add_argument("--config", required=True, help="Config YAML (thresholds normally come from `noise --out`).")
    p_replay.add_argument(
        "--calib", required=True,
        help="Vial-position folder. Offered back at the start of the round; drawn live if empty.")
    _add_selection_flags(p_replay)
    p_replay.add_argument("--bin-seconds", type=float, default=None, help="Override binning.bin_seconds.")
    p_replay.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    p_replay.add_argument("--out", default=None, help="Output directory (default: config output.dir).")
    p_replay.add_argument(
        "--monitor", action="store_true",
        help="Show a live tracking/activity monitor window while replaying (monitor.py).",
    )
    p_replay.add_argument(
        "--settings", action="store_true",
        help="Open the settings panel before the clip starts; close it to replay. This is the "
             "tuning loop: adjust, watch, 's' to save, re-run the SAME clip and compare.",
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

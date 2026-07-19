"""Run-loop integration layer (DESIGN.md §3 architecture, §5.1-§5.3, §7).

`TrackerPipeline` wires the rig-independent core modules into one loop:

    frame source -> rotation state machine -> (stationary?) -> per-vial activity -> binner -> logger
                          |                                         ^
                          +-- on stationary onset: face id (marker) + ROI re-registration --+

Nothing rig-specific lives here; every CV/IO primitive is delegated to the modules this file
consumes (`rotation`, `activity`, `registration`, `frame_source`, `calibration`, `logger`). This
module only owns the *sequencing*: when to reset the diff baseline, when to re-register ROIs, how
elapsed time maps to bins, and how a completed `ActivityBin` becomes `ActivityRecord` rows.

Elapsed clock (DESIGN.md §5.2/§7)
--------------------------------
`elapsed_s` (used for binning, `ActivityRecord.elapsed_s`, and the ISO bin timestamps) is derived
one of two ways, selected by the `clock` argument:

  * ``"monotonic"`` (live default): ``frame.t_monotonic - t0`` where ``t0`` is the first frame's
    monotonic timestamp. This tracks *real* elapsed wall time, which is what a live multi-day
    experiment wants even if frames are dropped.
  * ``"index"`` (video default): ``frame.index / source.fps``. This tracks *content* time, so an
    offline replay bins by the video's own timeline regardless of how fast it is processed.
  * ``"auto"`` (default): ``"index"`` for a `VideoFileSource`, ``"monotonic"`` for everything else
    (live camera / in-memory sources).

Wall-clock ISO timestamps for the output table are anchored to the first frame's ``t_wall_iso``
(``run wall-start``) and offset by ``bin_start_s``/``bin_end_s``, so `bin_start_iso`/`bin_end_iso`
stay consistent with `elapsed_s` and drive the logger's daily rolling.

Reference-reset + registration reference (DESIGN.md §5.2/§5.3)
-------------------------------------------------------------
The activity diff baseline ``prev_stationary`` is reset (set to ``None``) on *entering* ROTATING
and again at every stationary onset, so the first stationary frame after a rotation/face change is
never diffed against a frame from before it (that first frame just *seeds* the new baseline and
contributes no motion -- it has no valid pair). This is the guard against a giant spurious motion
spike across a rotation.

The registration *reference* per face is either supplied by the caller (`reference_frames`) or, if
absent, adopted from the first stationary frame seen for that face. On each stationary onset the
current frame is phase-correlated against that face's reference (`registration.estimate_shift`) on
the illum-masked frame; if the residual is acceptable the face's vial bboxes are re-derived from
their *calibration* anchors by the estimated shift (so drift never accumulates), otherwise a
`mis_registration` event is logged and the ROIs are left where they were.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flygym_tracker.activity import ActivityAccumulator, per_frame_activity
from flygym_tracker.calibration import bbox_from_quad, quad_polygon_mask, shift_quad, vial_shape
from flygym_tracker.frame_source import FrameSource, VideoFileSource
from flygym_tracker.registration import apply_shift, estimate_shift
from flygym_tracker.adaptive_rotation import AdaptiveRotationDetector
from flygym_tracker.rotation import RotationDetector
from flygym_tracker.types import ActivityRecord, EventRecord, TrackState

#: Default registration residual (0=perfect, ~1=uncorrelated) above which a shift is rejected.
DEFAULT_MAX_RESIDUAL = 0.5
#: A registration shift larger than this fraction of the tightest vial-center pitch is rejected as
#: lattice aliasing. The vial lattice is periodic (8 near-identical vials per row), so phase
#: correlation can lock onto a whole vial-pitch offset with HIGH confidence (low residual) and shift
#: every ROI onto its neighbour. Real drift after the drum returns to pose is far smaller than a
#: pitch, so a magnitude cap is the right guard where the residual check is blind.
DEFAULT_MAX_SHIFT_FRAC = 0.4
#: Retry budget + backoff for a transient `source.read()` exception (multi-day robustness).
DEFAULT_READ_RETRIES = 3
DEFAULT_READ_RETRY_SLEEP = 0.5
#: measure_noise: enter/exit threshold heuristic multipliers on the per-frame metric std.
DEFAULT_ENTER_K = 8.0
DEFAULT_EXIT_K = 4.0
#: Rolling window (frame count) for the observer-facing `fps_est` estimate (see `add_observer`).
DEFAULT_FPS_WINDOW = 30

Bbox = Tuple[int, int, int, int]


def _to_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Coerce a mask to bool: an already-bool mask is returned as-is; otherwise nonzero -> True."""
    m = np.asarray(mask)
    return m if m.dtype == bool else (m > 0)


def _fmt_setting(value) -> str:
    """A setting value for the `setting_change` event detail: readable, and never in exponent form.

    An analyst reading events.csv must be able to compare the number against the config file
    without decoding it, so a float keeps a visible decimal point (``12.0``, not ``12``) and never
    turns into ``1.2e+01``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "%.1f" % value if float(value).is_integer() else ("%s" % round(value, 6))
    return str(value)


def _detector_can_identify(detector) -> bool:
    """Can this marker detector ever name a face? Drives whether a run WAITS for an identification.

    Both shipped detectors answer with `can_identify()` (`marker_band.MarkerBandDetector` needs
    two templates; `markers.MarkerDetector` needs to be enabled and hold two signatures). A
    duck-typed detector that does not implement it is assumed capable -- it implements
    `identify_face`, which is the only contract this pipeline has ever required, and assuming
    otherwise would silently downgrade it to the guess-a-face behaviour that caused the bug.
    """
    if detector is None:
        return False
    probe = getattr(detector, "can_identify", None)
    if not callable(probe):
        return True
    try:
        return bool(probe())
    except Exception:
        return True


def _clip_bbox(bbox: Bbox, width: int, height: int) -> Bbox:
    """Clip an (x, y, w, h) bbox to the image; returns (x, y, w, h) with w/h >= 0."""
    x, y, w, h = bbox
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(int(width), int(x) + int(w))
    y1 = min(int(height), int(y) + int(h))
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


class TrackerPipeline:
    """Wire source -> rotation -> (face/register) -> activity -> binner -> logger; own the run loop.

    Parameters
    ----------
    config:
        A `flygym_tracker.config.Config` (or any object exposing the same nested attributes).
        Thresholds are read from ``config.rotation.{enter_threshold,exit_threshold,
        debounce_frames,min_stationary_frames}``, ``config.activity.pixel_threshold`` and
        ``config.binning.bin_seconds``. A ``null`` (``None``) enter/exit/pixel threshold must be
        supplied by the caller (having run `measure_noise` first) either by merging it into the
        config or via the ``*_threshold`` keyword overrides below -- nothing here is hard-coded.
    calibration:
        A resolved `Calibration` (mask paths absolute -- call `calibration.load_calibration`, which
        does `resolve_mask_paths`, before handing it in). Each face's illum-mask PNG is loaded once
        and the effective per-vial bbox-local boolean masks are precomputed for present vials only.
    source:
        A `FrameSource`; opened by `run()` and closed on exit.
    logger:
        An `ActivityLogger`; ``run_id`` and format come from it. Closed on run exit.
    marker_detector:
        Optional, duck-typed: any object with ``identify_face(gray) -> str | None``. ``None`` (no
        detector, or a detector that returns ``None``) defaults the face to ``"A"`` and logs a
        ``marker_absent`` event (DESIGN.md §5.2).
    reference_frames:
        Optional ``{face_name: gray_ndarray}`` registration references. A face without one adopts
        its first stationary frame as the reference.
    """

    def __init__(
        self,
        config,
        calibration,
        source: FrameSource,
        logger,
        marker_detector=None,
        reference_frames: Optional[Dict[str, np.ndarray]] = None,
        *,
        clock: str = "auto",
        pixel_threshold: Optional[float] = None,
        enter_threshold: Optional[float] = None,
        exit_threshold: Optional[float] = None,
        max_residual: float = DEFAULT_MAX_RESIDUAL,
        max_shift: Optional[float] = None,
        read_retries: int = DEFAULT_READ_RETRIES,
        read_retry_sleep: float = DEFAULT_READ_RETRY_SLEEP,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.source = source
        self.logger = logger
        self.marker_detector = marker_detector
        self.max_residual = float(max_residual)
        self._max_shift_arg = max_shift  # resolved after _precompute_faces (needs vial geometry)
        self.read_retries = max(1, int(read_retries))
        self.read_retry_sleep = float(read_retry_sleep)
        self.run_id = getattr(logger, "run_id", "run")

        # -- rotation detector mode: 'adaptive' (speed-independent, no preset thresholds) or
        #    'threshold' (fixed enter/exit magnitude). Default 'threshold' for back-compat.
        try:
            self.detector_mode = str(config.rotation.detector).lower()
        except Exception:
            self.detector_mode = "threshold"

        # -- resolve thresholds (config, overridable; never hard-coded) -----------------------
        px = pixel_threshold if pixel_threshold is not None else config.activity.pixel_threshold
        if px is None:
            raise ValueError(
                "activity.pixel_threshold is null: run measure_noise() and pass the result via a "
                "config override or the pixel_threshold= argument (DESIGN.md §5.3)."
            )
        self.pixel_threshold = float(px)
        self.bin_seconds = float(config.binning.bin_seconds)
        debounce = int(config.rotation.debounce_frames)
        min_stationary = int(config.rotation.min_stationary_frames)

        # enter/exit are only needed in 'threshold' mode; 'adaptive' derives them online.
        self.enter_threshold = self.exit_threshold = None
        if self.detector_mode != "adaptive":
            enter = enter_threshold if enter_threshold is not None else config.rotation.enter_threshold
            exit_ = exit_threshold if exit_threshold is not None else config.rotation.exit_threshold
            if enter is None or exit_ is None:
                raise ValueError(
                    "rotation.enter_threshold/exit_threshold is null: run measure_noise() and pass "
                    "them via a config override or the enter_threshold=/exit_threshold= arguments, "
                    "or set rotation.detector: adaptive to auto-detect (DESIGN.md §5.1)."
                )
            self.enter_threshold = float(enter)
            self.exit_threshold = float(exit_)

        # -- geometry + per-face masks/bboxes -------------------------------------------------
        self._W = int(calibration.image_width)
        self._H = int(calibration.image_height)
        self._face_index: Dict[str, int] = {
            name: i for i, name in enumerate(sorted(calibration.faces.keys()))
        }
        self._default_face = "A" if "A" in calibration.faces else sorted(calibration.faces)[0]

        self._illum_mask: Dict[str, np.ndarray] = {}     # face -> full-frame uint8 mask
        self._face_lit_mask: Dict[str, np.ndarray] = {}  # face -> full-frame bool (==255)
        self._face_active: Dict[str, Dict[int, Tuple[Bbox, np.ndarray]]] = {}   # face -> gvid -> (bbox, submask)
        self._face_calib_bbox: Dict[str, Dict[int, Bbox]] = {}                  # face -> gvid -> anchor bbox
        self._face_calib_quad: Dict[str, Dict[int, Optional[list]]] = {}        # face -> gvid -> anchor quad|None
        self._vial_meta: Dict[int, Tuple[str, object]] = {}                     # gvid -> (face, VialROI)
        self._precompute_faces()
        self.max_shift = (
            float(self._max_shift_arg) if self._max_shift_arg is not None
            else self._default_max_shift()
        )

        # -- reference frames + working state -------------------------------------------------
        self._face_refs: Dict[str, np.ndarray] = {}
        if reference_frames:
            for name, ref in reference_frames.items():
                self._face_refs[name] = np.asarray(ref).copy()

        # Face identification is only something to WAIT for when it can both matter and work:
        # more than one face in the bundle, and a detector actually able to tell them apart.
        # When it does, the run starts with NO current face and attributes nothing until the
        # first confident identification (see `_handle_onset`). When it does not -- a single-face
        # bundle, or no usable detector -- the default face stands from frame 0, exactly as
        # before, so single-face rigs are untouched and no run is left recording nothing.
        self._face_id_required = len(calibration.faces) > 1 and _detector_can_identify(
            marker_detector)
        self._current_face: Optional[str] = None if self._face_id_required else self._default_face
        self._faces_seen: List[str] = [] if self._face_id_required else [self._default_face]
        self._prev_stationary: Optional[np.ndarray] = None
        self._prev_state: Optional[TrackState] = None

        if self.detector_mode == "adaptive":
            try:
                sensitivity = float(config.rotation.sensitivity)
            except Exception:
                sensitivity = 1.0
            # `rotation.min_consistency` is absent from both shipped YAMLs, so it is passed ONLY
            # when the config actually carries one -- otherwise the detector's own default stands,
            # unduplicated. It is read at all so a value SAVED from the settings panel is honoured
            # on the next run; a knob the panel can write but the pipeline never reads would be a
            # knob that silently forgets itself the moment the run ends.
            extra = {}
            try:
                if config.rotation.min_consistency is not None:
                    extra["min_consistency"] = float(config.rotation.min_consistency)
            except Exception:
                pass
            # whole-frame displacement (roi_mask=None) is the validated configuration; the rigid
            # structure dominates the phase correlation regardless of which face is presented.
            self.rotation = AdaptiveRotationDetector(
                roi_mask=None,
                debounce_frames=debounce,
                min_stationary_frames=min_stationary,
                sensitivity=sensitivity,
                **extra,
            )
        else:
            self.rotation = RotationDetector(
                enter_threshold=self.enter_threshold,
                exit_threshold=self.exit_threshold,
                debounce_frames=debounce,
                min_stationary_frames=min_stationary,
                roi_mask=self._face_lit_mask.get(self._default_face),
            )
        self.accumulator = ActivityAccumulator(bin_seconds=self.bin_seconds)

        # -- clock selection ------------------------------------------------------------------
        if clock == "auto":
            self._use_index_clock = isinstance(source, VideoFileSource)
        elif clock == "index":
            self._use_index_clock = True
        elif clock == "monotonic":
            self._use_index_clock = False
        else:
            raise ValueError(f"clock must be 'auto'|'index'|'monotonic', got {clock!r}")
        self._fps = 1.0
        self._wall_start: Optional[datetime] = None
        self._t0_monotonic: Optional[float] = None

        # -- counters -------------------------------------------------------------------------
        self._last_elapsed = 0.0
        #: Frames actually handled. Distinguishes "the run has not started" from "elapsed is 0.0
        #: on frame 1", which `_last_elapsed` alone cannot -- see `apply_setting`.
        self._frames_seen = 0
        self.n_rotations = 0
        self.n_bins = 0
        self.n_activity_records = 0
        self.frames_read_errors = 0
        self._per_face_frames: Dict[str, int] = {name: 0 for name in calibration.faces}

        # -- observers (opt-in; see "observers" section below) --------------------------------
        self._observers: List[Callable[[dict], None]] = []
        self._bin_observers: List[Callable[[dict], None]] = []
        self.observer_failures = 0
        self._fps_times: List[float] = []

        # -- live settings (see the "live settings" section below) ----------------------------
        self._setting_routes = self._build_setting_routes()

    # ---- observers ----------------------------------------------------------------------------
    #
    # Optional, opt-in hooks for a live-monitoring UI (monitor.py) or any other passive watcher.
    # Registering nothing costs nothing: `_process_frame`/`_emit_bin` only build a payload and walk
    # the observer list when at least one is registered, so an unmonitored run's behaviour and
    # performance are unchanged from before these hooks existed. Every observer call is wrapped in
    # try/except -- a raising observer is counted in `observer_failures` and otherwise ignored; it
    # can never abort the run.

    def add_observer(self, callback: Callable[[dict], None]) -> None:
        """Register a per-frame observer, called after every processed frame with a dict:
        ``{"frame", "index", "elapsed_s", "state", "face", "vial_results", "n_rotations",
        "fps_est", "pixel_threshold"}``. ``vial_results`` is the same
        ``{gvid: (motion_px, lit_area_px, active_fraction)}`` mapping (or ``{}``/``None``) that was
        just fed to the accumulator for this frame -- see `_process_frame`. Purely additive/
        read-only from the pipeline's point of view; nothing an observer does can feed back into
        measurement.
        """
        self._observers.append(callback)

    def add_bin_observer(self, callback: Callable[[dict], None]) -> None:
        """Register a bin-completion observer, called whenever a bin rolls over (including the
        final, possibly-partial bin flushed at run end) with
        ``{"bin": ActivityBin, "records": [ActivityRecord, ...]}``."""
        self._bin_observers.append(callback)

    def _notify(self, observers: List[Callable[[dict], None]], payload: dict) -> None:
        """Call every observer with `payload`, isolating (and counting) failures."""
        for callback in observers:
            try:
                callback(payload)
            except Exception:
                self.observer_failures += 1

    def _notify_frame_observers(self, frame, gray, elapsed_s, state, vial_results) -> None:
        now = float(frame.t_monotonic)
        self._fps_times.append(now)
        if len(self._fps_times) > DEFAULT_FPS_WINDOW:
            del self._fps_times[0]
        if len(self._fps_times) >= 2:
            span = self._fps_times[-1] - self._fps_times[0]
            fps_est = (len(self._fps_times) - 1) / span if span > 0 else 0.0
        else:
            fps_est = 0.0
        payload = {
            "frame": gray,
            "index": int(frame.index),
            "elapsed_s": float(elapsed_s),
            "state": state,
            "face": self._current_face,
            "vial_results": vial_results,
            "n_rotations": self.n_rotations,
            "fps_est": fps_est,
            "pixel_threshold": self.pixel_threshold,
        }
        self._notify(self._observers, payload)

    # ---- live settings ------------------------------------------------------------------------
    #
    # `apply_setting(key, value)` is how a running pipeline is re-tuned from the outside -- the
    # settings panel (settings_panel.py) and the monitor's +/- keys both end here. Two rules:
    #
    #   1. THE ROUTING TABLE IS A LITERAL. A dotted key arriving from a GUI is only ever a dict
    #      lookup; it never becomes an attribute name. `setattr(self, key.rsplit(".")[-1], value)`
    #      would have been three lines shorter and would have let a typo'd or hostile key write
    #      any attribute on this object, including `_prev_stationary` or `logger`.
    #   2. EVERY APPLIED CHANGE IS LOGGED as a `setting_change` event. A 3-day run whose
    #      `pixel_threshold` moved from 12 to 18 at hour 40 produces ONE activity.csv holding two
    #      different measurement regimes. Without a row in events.csv saying when the switch
    #      happened, the analysis would average across both and compare incomparable numbers, with
    #      nothing anywhere in the output hinting that it should not. The event is the only record
    #      that exists -- `run_meta.json` snapshots the config at START, which by then is wrong.

    def _build_setting_routes(self):
        """``key -> (getter, setter)`` for everything this run can actually change.

        Built once, from the live objects: the rotation knobs are included only if THIS run's
        detector really has them (`AdaptiveRotationDetector` does; the fixed-threshold
        `RotationDetector` has no `sensitivity`/`min_consistency` at all). A key that is absent
        here makes `apply_setting` return False, which is how the panel learns to say "not applied
        to this run" instead of moving a slider that does nothing.
        """
        routes = {
            "activity.pixel_threshold": (
                lambda: self.pixel_threshold, self._set_pixel_threshold),
        }
        detector = self.rotation
        if hasattr(detector, "sensitivity"):
            routes["rotation.sensitivity"] = (
                lambda: self.rotation.sensitivity, self._set_rotation_sensitivity)
        if hasattr(detector, "debounce_frames"):
            routes["rotation.debounce_frames"] = (
                lambda: self.rotation.debounce_frames, self._set_rotation_debounce_frames)
        if hasattr(detector, "min_stationary_frames"):
            routes["rotation.min_stationary_frames"] = (
                lambda: self.rotation.min_stationary_frames,
                self._set_rotation_min_stationary_frames)
        if hasattr(detector, "min_consistency"):
            routes["rotation.min_consistency"] = (
                lambda: self.rotation.min_consistency, self._set_rotation_min_consistency)
        return routes

    # Each setter is spelled out rather than generated, so the set of attributes reachable from a
    # GUI is visible in one screenful. They also clamp: the detectors validate these in their
    # CONSTRUCTORS but not on assignment, and a live write must not be able to put the state
    # machine somewhere its own constructor would have rejected.

    def _set_pixel_threshold(self, value) -> None:
        self.pixel_threshold = max(0.0, float(value))

    def _set_rotation_sensitivity(self, value) -> None:
        # `_thresholds()` divides by this every frame; zero or negative would be a ZeroDivisionError
        # (or an inverted threshold) inside the acquisition loop.
        self.rotation.sensitivity = max(1e-3, float(value))

    def _set_rotation_debounce_frames(self, value) -> None:
        # Below 1, every frame satisfies the quiet streak -> instant, spurious stationary onsets.
        self.rotation.debounce_frames = max(1, int(value))

    def _set_rotation_min_stationary_frames(self, value) -> None:
        self.rotation.min_stationary_frames = max(1, int(value))

    def _set_rotation_min_consistency(self, value) -> None:
        self.rotation.min_consistency = min(1.0, max(0.0, float(value)))

    def settable_keys(self) -> List[str]:
        """The setting keys `apply_setting` will accept for THIS run, in table order."""
        return list(self._setting_routes.keys())

    def apply_setting(self, key: str, value) -> bool:
        """Route one setting change into the live objects. True if the key was routed at all.

        The new value is in force from the NEXT frame processed -- `pixel_threshold` is read per
        frame by `_compute_vial_results`, and the rotation knobs are read per frame by
        `AdaptiveRotationDetector.update`, so nothing needs restarting.

        Returns False for a key this run cannot route (unknown, or a rotation knob the configured
        detector does not have), and for a value the setter rejects. Returns True when the value
        is in place -- INCLUDING when it was already that value, in which case nothing moved and
        nothing is logged. That distinction matters: a drag across a slider fires a mouse event per
        pixel, and logging one event per pixel would bury the transitions the log exists to record.

        A drag that PASSES THROUGH several values still logs each one, and that is deliberate: the
        panel applies continuously (which is the point -- the operator watches the effect while
        turning the knob), so frames really were measured at each intermediate value. The log
        therefore reads as an unbroken chain, ``12.0 -> 0.5``, ``0.5 -> 6.0``, ``6.0 -> 15.0``,
        and the analysis can reconstruct which threshold was in force for any frame. Collapsing
        that to the final value would re-create, in miniature, the exact untracked-regime problem
        this event exists to prevent.

        BEFORE THE FIRST FRAME, none of that applies and nothing is logged. A value chosen in the
        ``--settings`` panel that opens ahead of the run was never a regime CHANGE -- no frame was
        ever measured under the value it replaced -- so a chain of eight rows at ``elapsed_s=0``
        from one pre-run drag describes measurements that do not exist. What the run actually
        started with belongs in `run_meta.json`, which the CLI updates once the panel closes.
        """
        route = self._setting_routes.get(str(key))
        if route is None:
            return False
        getter, setter = route
        try:
            old = getter()
            setter(value)
            new = getter()
        except Exception:
            return False
        if new == old:
            return True
        if self._frames_seen > 0:
            self._log_event(
                self._last_elapsed, None, "setting_change",
                detail="%s: %s -> %s" % (key, _fmt_setting(old), _fmt_setting(new)),
            )
        return True

    # ---- precompute -------------------------------------------------------------------------

    def _precompute_faces(self) -> None:
        for name, fc in self.calibration.faces.items():
            mask_img = cv2.imread(fc.illum_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise RuntimeError(
                    f"could not read illum mask for face {name!r} at {fc.illum_mask_path!r} "
                    "(is the calibration bundle resolved? call calibration.load_calibration)"
                )
            self._illum_mask[name] = mask_img
            self._face_lit_mask[name] = mask_img == 255
            fidx = self._face_index[name]
            active: Dict[int, Tuple[Bbox, np.ndarray]] = {}
            calib: Dict[int, Bbox] = {}
            quads: Dict[int, Optional[list]] = {}
            for v in fc.vials:
                if not v.present:
                    continue
                gvid = fidx * 16 + v.id
                # `vial_shape` applies the documented precedence once, here: an N-vertex
                # hand-drawn `polygon` wins, else the 4-corner `quad`, else None (plain bbox).
                # Everything downstream -- crop rectangle, registration shift, submask -- then
                # works on that ONE resolved shape and never has to re-decide.
                shape = vial_shape(v)
                # With a shape, the crop rectangle is the polygon's OWN bounding box rather than
                # the stored one. They are equal for any bundle written by the editor or the live
                # selector (`calibration.sync_bbox_to_quad` / `build_calibration_from_polygons`
                # enforce it), so this changes nothing there -- it only stops a hand-edited bundle
                # whose bbox went stale from silently truncating the polygon it says it wants
                # measured.
                anchor = bbox_from_quad(shape) if shape is not None else (
                    int(v.x), int(v.y), int(v.w), int(v.h))
                calib[gvid] = anchor
                quads[gvid] = shape
                active[gvid] = self._bbox_submask(
                    mask_img, anchor, quad=getattr(v, "quad", None),
                    polygon=getattr(v, "polygon", None))
                self._vial_meta[gvid] = (name, v)
            self._face_active[name] = active
            self._face_calib_bbox[name] = calib
            self._face_calib_quad[name] = quads

    def _bbox_submask(self, illum_mask: np.ndarray, bbox: Bbox,
                      quad: Optional[list] = None,
                      polygon: Optional[list] = None) -> Tuple[Bbox, np.ndarray]:
        """Clip a bbox to the frame; return (clipped_bbox, bbox-local effective bool mask).

        The effective mask is ``illum_mask == 255`` inside the bbox, AND -- when the vial carries
        a shape -- the filled polygon of that shape. PRECEDENCE (`types.VialROI`):

            ``polygon`` (N >= 3 vertices, hand-drawn on the live feed by
            `live_vial_selector`) > ``quad`` (4 corners, `roi_editor`) > plain bbox.

        Both None (every pre-quad calibration bundle) leaves the mask exactly as it has always
        been. The polygon NEVER resurrects pixels the illumination mask excluded -- it is an
        intersection, not a substitution.
        """
        shape = polygon if polygon is not None else quad
        cb = _clip_bbox(bbox, self._W, self._H)
        x, y, w, h = cb
        if w <= 0 or h <= 0:
            return cb, np.zeros((0, 0), dtype=bool)
        sub = illum_mask[y:y + h, x:x + w] == 255
        if shape is not None:
            sub = sub & quad_polygon_mask(shape, cb)
        return cb, sub

    def _default_max_shift(self) -> float:
        """`DEFAULT_MAX_SHIFT_FRAC` x the tightest vial-center pitch, measured WITHIN EACH FACE.

        The tightest pitch is the smallest center-to-center distance between two vials of the same
        face, i.e. the offset at which a registration shift would alias one vial onto its nearest
        neighbour. Capping below it rejects lattice-pitch lock-ons while still allowing realistic
        sub-pitch drift. Falls back to a quarter of the smaller image dimension when no face has
        two vials to measure between.

        REGRESSION THIS GUARDS. The pitch used to be measured over all faces POOLED. That was
        harmless only while the two faces had different coordinates. The hand-drawing flow gives
        face B face A's polygons VERBATIM (one drawing covers the drum), so every vial gained a
        twin at distance exactly 0 -- on the real 32-vial bundle, 16 of the 31 sorted centre gaps
        were 0.0. The tightest pitch collapsed to 0.0, `max_shift` with it, and the guard then
        rejected EVERY shift including (0.0, 0.0): registration silently stopped correcting ROI
        drift for the whole run, reported only as a stream of `mis_registration` events.

        Two vials of DIFFERENT faces are never candidates for aliasing anyway -- only one face is
        in view at a time, and a shift is only ever applied to the face being measured.
        """
        best: Optional[float] = None
        for active in self._face_active.values():
            centers = [(x + w / 2.0, y + h / 2.0) for _gvid, (( x, y, w, h), _sub) in active.items()]
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    d = float(np.hypot(centers[i][0] - centers[j][0],
                                       centers[i][1] - centers[j][1]))
                    if best is None or d < best:
                        best = d
        if best is None or best <= 0.0:
            return 0.25 * float(min(self._W, self._H))
        return DEFAULT_MAX_SHIFT_FRAC * best

    # ---- public run loop --------------------------------------------------------------------

    def run(self, max_frames: Optional[int] = None, stop_flag=None) -> dict:
        """Main loop. Reads frames until EOF, ``max_frames``, or ``stop_flag``; returns a summary.

        `stop_flag` may be a callable ``() -> bool``, a ``threading.Event`` (``.is_set()``), or any
        truthy/falsey object; it is checked once per iteration for a graceful stop.
        """
        self.source.open()
        try:
            fps = float(self.source.fps)
        except Exception:
            fps = 0.0
        self._fps = fps if fps > 0 else 1.0

        frames_processed = 0
        stopped_reason = "eof"
        try:
            while True:
                if self._should_stop(stop_flag):
                    stopped_reason = "stop_flag"
                    break
                if max_frames is not None and frames_processed >= max_frames:
                    stopped_reason = "max_frames"
                    break
                frame, status = self._read_frame()
                if status == "eof":
                    stopped_reason = "eof"
                    break
                if status == "error":
                    continue  # transient read error already logged; skip this frame
                self._process_frame(frame)
                frames_processed += 1

            final = self.accumulator.flush()
            if final is not None:
                self._emit_bin(final)
        finally:
            self.logger.close()
            self.source.close()

        return {
            "run_id": self.run_id,
            "frames_processed": frames_processed,
            "frames_read_errors": self.frames_read_errors,
            "n_rotations": self.n_rotations,
            "n_bins": self.n_bins,
            "n_activity_records": self.n_activity_records,
            "faces_seen": list(self._faces_seen),
            "per_face_frames": dict(self._per_face_frames),
            "stopped_reason": stopped_reason,
            "observer_failures": self.observer_failures,
        }

    # ---- per-frame processing ---------------------------------------------------------------

    def _process_frame(self, frame) -> None:
        gray = frame.image
        if self._wall_start is None:
            self._init_clock(frame)

        state = self.rotation.update(gray)
        elapsed_s = self._elapsed(frame)
        self._last_elapsed = elapsed_s
        self._frames_seen += 1
        prev_state = self._prev_state

        entered_rotating = state == TrackState.ROTATING and prev_state != TrackState.ROTATING
        stationary_onset = (state == TrackState.SETTLING and prev_state != TrackState.SETTLING) or (
            prev_state == TrackState.ROTATING and state == TrackState.STATIONARY
        )

        if entered_rotating:
            self.n_rotations += 1
            self._log_event(elapsed_s, frame, "rotation_start")
            self._prev_stationary = None  # reset diff baseline (DESIGN §5.2)
        if stationary_onset:
            if prev_state == TrackState.ROTATING:
                self._log_event(elapsed_s, frame, "rotation_end")
            self._handle_onset(gray, elapsed_s, frame)
            self._prev_stationary = None  # fresh baseline for the new stationary period

        # Accumulate EVERY frame so bins keep rolling even through long rotations. `obs_vial_results`
        # mirrors exactly whatever vial_results dict (or None) was fed to the accumulator this frame
        # -- captured only so an observer (monitor.py) can see the same numbers; it is never read
        # back into measurement.
        obs_vial_results: Optional[Dict[int, Tuple[int, int, float]]] = None
        if state == TrackState.STATIONARY:
            if self._prev_stationary is None:
                # First stationary frame after a reset: seed the baseline, no pair yet -> no motion.
                self._prev_stationary = gray
                obs_vial_results = {}
                rolled = self.accumulator.add(elapsed_s, TrackState.STATIONARY, obs_vial_results)
            else:
                vial_results = self._compute_vial_results(gray)
                self._prev_stationary = gray
                obs_vial_results = vial_results
                rolled = self.accumulator.add(elapsed_s, TrackState.STATIONARY, vial_results)
        elif state == TrackState.ROTATING:
            # Feed present-vial keys (motion/active ignored by the accumulator for ROTATING) so
            # `n_rotating_frames`/`lit_area_px` stay populated -> a bin straddling a rotation is
            # interpretable (DESIGN §5.3), instead of vanishing from the table entirely.
            obs_vial_results = self._rotating_placeholder()
            rolled = self.accumulator.add(elapsed_s, state, obs_vial_results)
        else:
            # SETTLING / UNKNOWN: excluded from activity; add() only advances the bin clock.
            rolled = self.accumulator.add(elapsed_s, state, None)
        if rolled is not None:
            self._emit_bin(rolled)

        if self._current_face is not None:
            self._per_face_frames[self._current_face] = (
                self._per_face_frames.get(self._current_face, 0) + 1)
        self._prev_state = state

        if self._observers:
            self._notify_frame_observers(frame, gray, elapsed_s, state, obs_vial_results)

    def _handle_onset(self, gray: np.ndarray, elapsed_s: float, frame) -> None:
        """Stationary onset: identify the face, maybe face_change, re-register the ROIs.

        FAILING SAFE IS THE POINT HERE. This used to reset to the default face on every failed
        identification, which is how a 3-day run came to record both drum faces as face A: with
        no marker templates in the bundle `identify_face` returned None every time, and every
        reset looked exactly like a correct answer. Now a failure NEVER invents a face --
        it keeps the last confidently identified one, or, before there has been one, attributes
        nothing at all (see `_face_for_failed_id`).
        """
        face = None
        if self.marker_detector is not None:
            try:
                face = self.marker_detector.identify_face(gray)
            except Exception:
                face = None

        if face is not None and face not in self.calibration.faces:
            # A name the vials know nothing about is not an identification, so it is treated as
            # a FAILURE rather than adopted -- measuring against another face's ROIs would be
            # worse than measuring against none.
            self._log_event(
                elapsed_s, frame, "mis_registration",
                detail=f"identified face {face!r} not in calibration; keeping {self._current_face!r}",
            )
            face = None

        if face is None:
            face = self._face_for_failed_id(elapsed_s, frame)
            if face is None:
                # Still unknown. Do not register, do not adopt a reference frame, and leave
                # `_current_face` None so this stationary period contributes to no vial.
                return

        if face != self._current_face:
            self._log_event(elapsed_s, frame, "face_change", detail=f"{self._current_face} -> {face}")
            self._current_face = face
            if self.detector_mode != "adaptive":
                self.rotation.roi_mask = self._face_lit_mask[face]
        if face not in self._faces_seen:
            self._faces_seen.append(face)

        self._register(gray, face, elapsed_s, frame)

    def _face_for_failed_id(self, elapsed_s: float, frame) -> Optional[str]:
        """Which face to use when identification failed. ``None`` = none; attribute nothing.

        Three cases, and the difference between them is the whole fix:

        * **Face id is not required** (single-face bundle, or no detector able to discriminate):
          the default face stands, as it always has. There is no ambiguity to get wrong.
        * **A face has already been identified**: KEEP IT. The drum was showing that face a
          moment ago and one unreadable onset is not evidence it flipped -- and if it did flip,
          the next readable onset corrects it. Resetting to the default here is precisely what
          mislabelled a whole experiment.
        * **Nothing has been identified yet** (the start of a run): stay unknown. Guessing costs
          mislabelled vial identities for as long as the guess survives; waiting costs a short
          gap at the beginning of a run measured in hours or days. The gap is recoverable, the
          mislabelling is not.
        """
        if not self._face_id_required:
            face = self._current_face or self._default_face
            self._log_event(elapsed_s, frame, "marker_absent", detail=f"defaulted to face {face}")
            return face
        if self._current_face is not None:
            self._log_event(elapsed_s, frame, "marker_absent",
                            detail=f"kept last known face {self._current_face}")
            return self._current_face
        self._log_event(
            elapsed_s, frame, "marker_absent",
            detail="face not identified yet; activity attributed to no face until it is",
        )
        return None

    def _register(self, gray: np.ndarray, face: str, elapsed_s: float, frame) -> None:
        ref = self._face_refs.get(face)
        if ref is None:
            # No supplied reference: adopt this first stationary frame; ROIs stay at calibration.
            self._face_refs[face] = np.asarray(gray).copy()
            return
        dx, dy, residual = estimate_shift(gray, ref, mask=self._face_lit_mask[face])
        too_large = abs(dx) > self.max_shift or abs(dy) > self.max_shift
        if residual <= self.max_residual and not too_large:
            self._apply_registration(face, dx, dy)
        elif too_large:
            # Phase correlation locked onto (near) a vial pitch -> would alias every ROI onto a
            # neighbour. Reject and keep ROIs at their calibration anchors (DESIGN.md §5.2).
            self._log_event(
                elapsed_s, frame, "mis_registration",
                detail=f"shift=({dx:.1f},{dy:.1f}) exceeds max_shift={self.max_shift:.1f}px "
                       "(likely vial-lattice aliasing); ROIs left unshifted",
            )
        else:
            self._log_event(
                elapsed_s, frame, "mis_registration",
                detail=f"residual={residual:.3f} > {self.max_residual:.3f}; ROIs left unshifted",
            )

    def _apply_registration(self, face: str, dx: float, dy: float) -> None:
        """Re-derive each present vial's bbox+submask from its calibration anchor by (dx, dy).

        A vial's shape is translated by the SAME rounded offset as its bbox (`shift_quad` mirrors
        `apply_shift`'s rounding), so the polygon keeps its exact position within the crop.
        """
        illum = self._illum_mask[face]
        active = self._face_active[face]
        quads = self._face_calib_quad[face]
        for gvid, anchor in self._face_calib_bbox[face].items():
            # Already polygon-or-quad resolved by `_precompute_faces`, so it is passed as the
            # WINNING shape rather than re-running the precedence on a shifted copy.
            shape = quads.get(gvid)
            shifted = shift_quad(shape, dx, dy) if shape is not None else None
            active[gvid] = self._bbox_submask(
                illum, apply_shift(anchor, dx, dy), polygon=shifted)

    def _rotating_placeholder(self) -> Dict[int, Tuple[int, int, float]]:
        """Zero-motion per-vial results for the current face (only the keys + lit area matter here).

        The accumulator's ROTATING branch ignores motion/active_fraction and uses each entry only
        to bump `n_rotating_frames` and refresh `lit_area_px`, so we pass the true lit area and
        zeros for the rest.

        Empty while the face is still unknown: which vials these frames belong to is exactly what
        has not been established yet, so crediting them to any face would be a guess.
        """
        out: Dict[int, Tuple[int, int, float]] = {}
        if self._current_face is None:
            return out
        for gvid, (_bbox, submask) in self._face_active[self._current_face].items():
            out[gvid] = (0, int(np.count_nonzero(submask)), 0.0)
        return out

    def _compute_vial_results(self, gray: np.ndarray) -> Dict[int, Tuple[int, int, float]]:
        """Per-frame per-vial activity for the current face vs `self._prev_stationary`.

        Empty while the face is still unknown -- no face, no vials to attribute motion to.
        """
        prev = self._prev_stationary
        results: Dict[int, Tuple[int, int, float]] = {}
        if self._current_face is None:
            return results
        for gvid, (bbox, submask) in self._face_active[self._current_face].items():
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or submask.size == 0:
                results[gvid] = (0, 0, 0.0)
                continue
            cur_crop = gray[y:y + h, x:x + w]
            prev_crop = prev[y:y + h, x:x + w]
            results[gvid] = per_frame_activity(cur_crop, prev_crop, submask, self.pixel_threshold)
        return results

    # ---- bin -> records ---------------------------------------------------------------------

    def _emit_bin(self, bin_obj) -> None:
        self.n_bins += 1
        records = self._bin_to_records(bin_obj)
        if records:
            self.logger.log_activity(records)
            self.n_activity_records += len(records)
        if self._bin_observers:
            self._notify(self._bin_observers, {"bin": bin_obj, "records": records})

    def _bin_to_records(self, bin_obj) -> List[ActivityRecord]:
        bin_start_iso = self._iso_at(bin_obj.bin_start_s)
        bin_end_iso = self._iso_at(bin_obj.bin_end_s)
        records: List[ActivityRecord] = []
        for gvid in sorted(bin_obj.vials):
            vd = bin_obj.vials[gvid]
            face, vial = self._vial_meta[gvid]
            records.append(ActivityRecord(
                run_id=self.run_id,
                bin_start_iso=bin_start_iso,
                bin_end_iso=bin_end_iso,
                elapsed_s=float(bin_obj.bin_start_s),
                face=face,
                vial_id=gvid,
                row=int(vial.row),
                col=int(vial.col),
                present=bool(vial.present),
                n_stationary_frames=int(vd["n_stationary_frames"]),
                n_rotating_frames=int(vd["n_rotating_frames"]),
                motion_px_sum=int(vd["motion_px_sum"]),
                active_fraction_mean=float(vd["active_fraction_mean"]),
                lit_area_px=int(vd["lit_area_px"]),
            ))
        return records

    # ---- clock + events + control -----------------------------------------------------------

    def _init_clock(self, frame) -> None:
        try:
            self._wall_start = datetime.fromisoformat(frame.t_wall_iso)
        except (ValueError, TypeError):
            self._wall_start = datetime.now()
        self._t0_monotonic = float(frame.t_monotonic)

    def _elapsed(self, frame) -> float:
        if self._use_index_clock:
            return float(frame.index) / self._fps
        return float(frame.t_monotonic) - float(self._t0_monotonic)

    def _iso_at(self, elapsed_s: float) -> str:
        base = self._wall_start or datetime.now()
        return (base + timedelta(seconds=float(elapsed_s))).isoformat()

    def _log_event(self, elapsed_s: float, frame, event: str, detail: str = "") -> None:
        iso = getattr(frame, "t_wall_iso", None) or datetime.now().isoformat()
        self.logger.log_event(EventRecord(
            run_id=self.run_id, iso_time=iso, elapsed_s=float(elapsed_s), event=event, detail=detail,
        ))

    def _read_frame(self):
        """Read one frame with retry tolerance. Returns (frame|None, 'ok'|'eof'|'error')."""
        last_exc: Optional[BaseException] = None
        for _ in range(self.read_retries):
            try:
                frame = self.source.read()
            except Exception as exc:  # transient acquisition hiccup
                last_exc = exc
                if self.read_retry_sleep > 0:
                    time.sleep(self.read_retry_sleep)
                continue
            if frame is None:
                return None, "eof"
            return frame, "ok"
        self.frames_read_errors += 1
        self._log_read_error(last_exc)
        return None, "error"

    def _log_read_error(self, exc: Optional[BaseException]) -> None:
        self.logger.log_event(EventRecord(
            run_id=self.run_id,
            iso_time=datetime.now().isoformat(),
            elapsed_s=float(self._last_elapsed),
            event="read_error",
            detail=repr(exc)[:200] if exc is not None else "",
        ))

    @staticmethod
    def _should_stop(stop_flag) -> bool:
        if stop_flag is None:
            return False
        if callable(stop_flag):
            return bool(stop_flag())
        is_set = getattr(stop_flag, "is_set", None)
        if callable(is_set):
            return bool(is_set())
        return bool(stop_flag)


def measure_noise(source: FrameSource, illum_mask, n_frames: int = 100, k: float = 5.0) -> dict:
    """Measure the static-rig noise floor to seed activity + rotation thresholds (DESIGN.md §9).

    Grabs up to ``n_frames`` consecutive frames from a *stationary* source and characterises the
    frame-to-frame difference within ``illum_mask`` at two scales:

    * **per-pixel** ``|cur - prev|`` over the masked pixels, pooled across all consecutive pairs ->
      ``noise_mean`` / ``noise_std``. Because `activity.per_frame_activity` thresholds *individual*
      pixels, the pixel threshold must come from this per-pixel distribution:
      ``suggested_pixel_threshold = noise_mean + k * noise_std`` (DESIGN.md §5.3, default k=5).
    * **per-frame** global metric ``m = mean(|cur - prev|)`` over the mask (exactly
      `RotationDetector.metric`) -> ``metric_mean`` / ``metric_std``. The rotation detector compares
      *this* per-frame scalar against enter/exit, so those thresholds are seeded from the per-frame
      distribution, not the per-pixel one.

    Enter/exit heuristic (DESIGN.md §5.1: "far above static-rig noise and far below rotation
    motion", with hysteresis enter > exit): a static rig's per-frame metric sits at ``metric_mean``
    with tiny spread, so both thresholds are placed several std above it --
    ``suggested_exit_threshold = metric_mean + EXIT_K*metric_std`` and
    ``suggested_enter_threshold = metric_mean + ENTER_K*metric_std`` (EXIT_K=4, ENTER_K=8). These
    clear the noise floor while, given the large separation expected, still landing well below the
    tens-of-gray-levels motion of a real rotation. They are **seeds**: verify/raise ``enter`` on a
    real rotation clip and against loaded-fly motion before trusting them (DESIGN.md §5.1/§9).

    ``source`` is opened if not already (``open()`` is idempotent on the shipped sources); lifecycle
    (closing) stays with the caller -- wrap it in ``with source:`` or close it yourself.
    """
    mask = _to_bool_mask(illum_mask)
    source.open()

    prev: Optional[np.ndarray] = None
    metric_vals: List[float] = []
    sum1 = 0.0            # sum of masked per-pixel abs-diffs
    sum2 = 0.0            # sum of squares
    count = 0             # number of masked pixels pooled
    n_read = 0
    n_pairs = 0

    while n_read < n_frames:
        frame = source.read()
        if frame is None:
            break
        n_read += 1
        cur = frame.image
        if prev is not None:
            diff = cv2.absdiff(cur, prev)
            vals = diff[mask].astype(np.float64)
            if vals.size:
                sum1 += float(vals.sum())
                sum2 += float(np.square(vals).sum())
                count += int(vals.size)
                metric_vals.append(float(vals.mean()))
                n_pairs += 1
        prev = cur

    if n_pairs == 0:
        raise ValueError(
            f"measure_noise needs at least 2 frames within the mask; got {n_read} frame(s), "
            f"{n_pairs} usable pair(s)."
        )

    noise_mean = sum1 / count
    noise_var = max(0.0, sum2 / count - noise_mean * noise_mean)
    noise_std = float(np.sqrt(noise_var))
    metric_arr = np.asarray(metric_vals, dtype=np.float64)
    metric_mean = float(metric_arr.mean())
    metric_std = float(metric_arr.std())

    return {
        "noise_mean": float(noise_mean),
        "noise_std": float(noise_std),
        "suggested_pixel_threshold": float(noise_mean + k * noise_std),
        "metric_mean": metric_mean,
        "metric_std": metric_std,
        "suggested_enter_threshold": float(metric_mean + DEFAULT_ENTER_K * metric_std),
        "suggested_exit_threshold": float(metric_mean + DEFAULT_EXIT_K * metric_std),
        "k": float(k),
        "enter_k": DEFAULT_ENTER_K,
        "exit_k": DEFAULT_EXIT_K,
        "n_frames": n_read,
        "n_pairs": n_pairs,
    }

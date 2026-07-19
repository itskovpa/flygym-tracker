"""Real-time monitoring display for a running `TrackerPipeline` (pipeline.py's observer hooks).

Not a DESIGN.md-specified module: this is an additive live-view feature so a scientist running a
multi-day experiment can *see* the tracking and per-vial activity while it happens -- to catch a
bad ROI, a wrong `pixel_threshold`, or a mis-detected face without waiting for the run to end --
built entirely on top of `TrackerPipeline.add_observer`/`add_bin_observer` (pipeline.py), with no
changes to any measurement logic.

Register with the pipeline::

    monitor = LiveMonitor(calibration, config, max_fps=10)
    pipeline.add_observer(monitor.on_frame)
    pipeline.add_bin_observer(monitor.on_bin)
    ...
    pipeline.run()
    monitor.close()

Two guarantees drive every design choice here:

1. **Acquisition is never slowed.** `on_frame` (called once per processed frame, at full
   acquisition rate, synchronously from inside the pipeline's run loop) does only O(#vials)
   bookkeeping: store the latest payload, resync `pixel_threshold`, roll the diff baseline used
   for the live-view motion tint, and fold this frame's per-vial numbers into the *current*
   (in-progress) bin's running totals. The expensive part -- building the composite image and
   pushing it to the screen -- lives in `maybe_render()`, which self-throttles to `max_fps` (a
   fast no-op, just a monotonic-clock comparison, on any call before the next frame is due) and
   is skipped entirely whenever the caller is behind. `on_frame` calls it last when `auto_render`
   is on (the default -- this is what lets the CLI just register the observer and get a live
   window with no extra wiring). Tests that want `on_frame`'s bookkeeping in isolation, with no
   cv2 GUI call of any kind, construct with `auto_render=False`.
2. **A run survives a headless / display-less machine.** Any `cv2.imshow`/`cv2.waitKey` error is
   caught once, logged, and rendering is permanently disabled for the rest of the run (see
   `_disable_rendering`) -- `on_frame`/`on_bin` keep updating internal state regardless, so
   `render_composite()` and PNG snapshots keep working even with the interactive window off.

The window is a fixed-size composite (`canvas_h` x `canvas_w`, independent of the camera's actual
frame size, so it is deterministic and directly testable):

  * **top banner** -- state (colour-coded STATIONARY/ROTATING/SETTLING/UNKNOWN), face, elapsed
    time, frames processed, estimated fps, rotation count, and the live `pixel_threshold`.
  * **left -- live view** -- the current frame, with each vial's ROI box overlaid (green =
    present, red = absent) and id-labelled, plus a red tint over the pixels currently over
    `pixel_threshold` against the last stationary baseline. That baseline is a local, display-only
    re-derivation of the DESIGN.md 5.2/5.3 reset rule (reset on any non-STATIONARY frame) from the
    `state`/`frame` fields already in the observer payload -- see `on_frame` -- it never reads
    pipeline internals and never feeds back into measurement.
  * **top-right -- bar chart** of the CURRENT (in-progress) bin's mean `active_fraction` per vial,
    for whichever face is presently visible (all 16 slots; empty slots drawn flat/grey so a
    missing tube is as visible as a dead one). Reset every time `on_bin` fires.
  * **bottom-right -- actogram heatmap** -- rows = every vial in the calibration (both faces),
    columns = the last `heatmap_bins` *completed* bins (oldest -> newest, left -> right), colour =
    that bin's mean `active_fraction`. `heatmap_buffer` is a `collections.deque(maxlen=...)`, so
    it is bounded by construction -- old columns simply fall off as new ones are appended.

Keyboard (polled once per `maybe_render()` call via `cv2.waitKey`, dispatched by `handle_key` --
itself a plain function of a keycode + current state, independently testable without a window):
``q`` quit the monitor (the pipeline run itself is untouched -- only the window/rendering stops),
``p`` pause/resume rendering (acquisition-side hooks keep running), ``+``/``-`` nudge
`pixel_threshold` by `threshold_step` and fire `on_threshold_change` (the CLI wires this straight
to the live `pipeline.pixel_threshold`), ``o`` toggle the ROI overlay, ``s`` save a PNG snapshot of
the last-rendered composite to `snapshot_dir`.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

import cv2
import numpy as np

from flygym_tracker.types import TrackState

logger = logging.getLogger(__name__)

#: BGR (OpenCV convention) colours for the state banner + bar chart accents.
_STATE_COLORS = {
    TrackState.STATIONARY: (0, 200, 0),
    TrackState.ROTATING: (0, 60, 255),
    TrackState.SETTLING: (0, 220, 255),
    TrackState.UNKNOWN: (160, 160, 160),
}


def _format_elapsed(seconds: float) -> str:
    """`seconds` -> ``"HH:MM:SS"``, or ``"Dd HH:MM:SS"`` past 24h (multi-day runs, DESIGN.md 10)."""
    total = max(0, int(seconds))
    days, rem = divmod(total, 86400)
    hh, rem = divmod(rem, 3600)
    mm, ss = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hh:02d}:{mm:02d}:{ss:02d}"
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _fit_and_letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize `img` (BGR) to fit within `target_w` x `target_h` preserving aspect ratio, centred
    on a black canvas of exactly that size -- so the return shape is always `(target_h, target_w, 3)`
    regardless of the input's size/aspect ratio."""
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or target_w <= 0 or target_h <= 0:
        return canvas
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


class LiveMonitor:
    """Throttled, exception-safe live view for a running `TrackerPipeline`. See module docstring."""

    def __init__(
        self,
        calibration,
        config,
        max_fps: float = 10.0,
        *,
        auto_render: bool = True,
        window_name: str = "flygym-tracker monitor",
        left_w: int = 640,
        left_h: int = 480,
        right_w: int = 420,
        banner_h: int = 70,
        heatmap_bins: int = 120,
        threshold_step: float = 1.0,
        snapshot_dir: str = "snapshots",
        on_threshold_change: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.calibration = calibration
        self.config = config
        self.max_fps = float(max_fps) if max_fps and max_fps > 0 else 10.0
        self.auto_render = bool(auto_render)
        self.window_name = window_name

        # -- fixed composite layout: banner on top, live view left, bar chart + heatmap stacked
        #    right (same total height as the live view) -- deterministic regardless of camera size.
        self.left_w, self.left_h = int(left_w), int(left_h)
        self.right_w = int(right_w)
        self.banner_h = int(banner_h)
        self.bar_h = max(1, self.left_h // 2)
        self.heatmap_h = max(1, self.left_h - self.bar_h)
        self.canvas_w = self.left_w + self.right_w
        self.canvas_h = self.banner_h + self.left_h

        self.threshold_step = float(threshold_step)
        self.snapshot_dir = snapshot_dir
        self._on_threshold_change = on_threshold_change

        #: face -> {global_vial_id: {"bbox": (x,y,w,h), "local_id", "row", "col", "present"}}.
        self.vial_geom: Dict[str, Dict[int, dict]] = self._precompute_geometry(calibration)

        init_threshold = None
        try:
            init_threshold = config.activity.pixel_threshold
        except Exception:
            init_threshold = None
        #: live, adjustable copy of the pipeline's pixel_threshold (resynced every on_frame call;
        #: nudged by the +/- keys, which is what actually drives the pipeline via the callback).
        self.pixel_threshold = float(init_threshold) if init_threshold is not None else 10.0

        self.show_roi = True
        self.paused = False
        self.quit_requested = False
        self.render_enabled = True
        self.disabled_reason: Optional[str] = None

        self.frame_count = 0
        self.latest_payload: Optional[dict] = None
        self.last_composite: Optional[np.ndarray] = None
        #: gvid -> [active_fraction_sum, n_stationary_frames] for the CURRENT (in-progress) bin.
        self.cur_bin_totals: Dict[int, List[float]] = {}
        #: rolling window of completed bins: each entry is {gvid: active_fraction_mean}. Bounded
        #: by construction (deque maxlen) -- oldest column drops off as a new one is appended.
        self.heatmap_buffer: Deque[Dict[int, float]] = deque(maxlen=max(1, int(heatmap_bins)))

        self._prev_stationary_gray: Optional[np.ndarray] = None
        self._tint_baseline: Optional[np.ndarray] = None
        self._last_render_time: float = -1e9  # -inf-ish: the first maybe_render() is always due
        self._min_render_interval = 1.0 / self.max_fps

    # ---- geometry -----------------------------------------------------------------------------

    @staticmethod
    def _precompute_geometry(calibration) -> Dict[str, Dict[int, dict]]:
        """Static ROI anchors per face, keyed by global vial id (`face_index*16 + local_id`, same
        convention as `pipeline.py`). Deliberately does NOT read the illum-mask PNGs or apply any
        registration shift -- this is a *display* aid, not the measurement path, and a few edge
        pixels of slop against the live per-vial mask is an acceptable trade for zero filesystem/
        registration coupling here.
        """
        geom: Dict[str, Dict[int, dict]] = {}
        face_names = sorted(calibration.faces.keys())
        for fidx, name in enumerate(face_names):
            entries: Dict[int, dict] = {}
            for v in calibration.faces[name].vials:
                gvid = fidx * 16 + int(v.id)
                entries[gvid] = {
                    "bbox": (int(v.x), int(v.y), int(v.w), int(v.h)),
                    "local_id": int(v.id),
                    "row": int(v.row),
                    "col": int(v.col),
                    "present": bool(v.present),
                }
            geom[name] = entries
        return geom

    # ---- observer hooks (cheap; see module docstring guarantee 1) -----------------------------

    def on_frame(self, payload: dict) -> None:
        """Cheap: store the latest payload + O(#vials) running-bin bookkeeping. Safe to call
        directly with a hand-built payload dict -- no cv2 GUI call happens unless `auto_render`
        is on (the default), in which case this ends with a throttled `maybe_render()` call.
        """
        self.latest_payload = payload
        self.frame_count += 1

        state = payload.get("state")
        gray = payload.get("frame")

        # Diff baseline for the live-view motion tint: reset on any non-STATIONARY frame, which
        # reproduces the pipeline's own `_prev_stationary` reset around rotations/face changes
        # (DESIGN.md 5.2/5.3) purely from the `state` sequence -- see module docstring. Display-only;
        # never read by, or fed back into, measurement.
        if state != TrackState.STATIONARY:
            self._prev_stationary_gray = None
            self._tint_baseline = None
        else:
            self._tint_baseline = self._prev_stationary_gray
            self._prev_stationary_gray = gray

        thr = payload.get("pixel_threshold")
        if thr is not None:
            self.pixel_threshold = float(thr)

        vial_results = payload.get("vial_results")
        if state == TrackState.STATIONARY and vial_results:
            for gvid, result in vial_results.items():
                try:
                    _motion_px, _lit_area_px, active_fraction = result
                except (TypeError, ValueError):
                    continue
                acc = self.cur_bin_totals.setdefault(gvid, [0.0, 0])
                acc[0] += float(active_fraction)
                acc[1] += 1

        if self.auto_render:
            self.maybe_render()

    def on_bin(self, payload: dict) -> None:
        """Cheap: push the just-completed bin's per-vial mean active_fraction into the rolling
        heatmap buffer, and reset the current-bin bar-chart accumulator for the bin that just
        started."""
        bin_obj = payload.get("bin")
        row: Dict[int, float] = {}
        if bin_obj is not None:
            for gvid, vd in getattr(bin_obj, "vials", {}).items():
                row[gvid] = float(vd.get("active_fraction_mean", 0.0))
        self.heatmap_buffer.append(row)
        self.cur_bin_totals = {}

    # ---- rendering ----------------------------------------------------------------------------

    def render_composite(self) -> np.ndarray:
        """Pure: build the full composite BGR uint8 image (banner + live view + bar chart +
        heatmap) from current state. Never touches cv2 window/display APIs -- safe to call from
        tests, or to build a snapshot, with no window open. Always returns an ndarray of shape
        `(canvas_h, canvas_w, 3)`, dtype uint8, even before any frame has arrived.
        """
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)
        canvas[0:self.banner_h, :] = self._render_banner(self.canvas_w, self.banner_h)
        canvas[self.banner_h:self.banner_h + self.left_h, 0:self.left_w] = \
            self._render_live_view(self.left_w, self.left_h)
        canvas[self.banner_h:self.banner_h + self.bar_h, self.left_w:self.canvas_w] = \
            self._render_bar_chart(self.right_w, self.bar_h)
        canvas[self.banner_h + self.bar_h:self.canvas_h, self.left_w:self.canvas_w] = \
            self._render_heatmap(self.right_w, self.heatmap_h)
        self.last_composite = canvas
        return canvas

    def _render_banner(self, w: int, h: int) -> np.ndarray:
        img = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
        payload = self.latest_payload
        if payload is None:
            cv2.putText(img, "waiting for frames...", (12, h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (180, 180, 180), 1, cv2.LINE_AA)
            return img

        state = payload.get("state")
        color = _STATE_COLORS.get(state, (200, 200, 200))
        state_label = getattr(state, "value", str(state)).upper()
        face = payload.get("face", "?")
        elapsed_s = float(payload.get("elapsed_s", 0.0))
        fps_est = float(payload.get("fps_est", 0.0))
        n_rot = int(payload.get("n_rotations", 0))
        thr = float(payload.get("pixel_threshold", self.pixel_threshold))

        cv2.putText(img, state_label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        line1 = f"face {face}   elapsed {_format_elapsed(elapsed_s)}   frames {self.frame_count}"
        line2 = f"fps {fps_est:.1f}   rotations {n_rot}   pixel_threshold {thr:.1f}"
        cv2.putText(img, line1, (170, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, line2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        if self.paused:
            cv2.putText(img, "PAUSED", (w - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 255), 2, cv2.LINE_AA)

        hint = "q quit  p pause  +/- threshold  o roi  s snapshot"
        cv2.putText(img, hint, (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        return img

    def _render_live_view(self, w: int, h: int) -> np.ndarray:
        payload = self.latest_payload
        if payload is None or payload.get("frame") is None:
            img = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
            cv2.putText(img, "no frame yet", (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (150, 150, 150), 1, cv2.LINE_AA)
            return img

        gray = np.asarray(payload["frame"])
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else np.ascontiguousarray(gray)
        fh, fw = bgr.shape[:2]

        face = payload.get("face")
        state = payload.get("state")
        geom = self.vial_geom.get(face, {})

        # Motion tint: pixels over threshold vs. the locally-tracked stationary baseline (display
        # only -- see on_frame's docstring).
        if state == TrackState.STATIONARY and self._tint_baseline is not None:
            baseline = np.asarray(self._tint_baseline)
            if baseline.shape == gray.shape:
                diff = cv2.absdiff(gray, baseline)
                motion = diff > self.pixel_threshold
                if np.any(motion):
                    tinted = bgr.copy()
                    tinted[motion] = (0, 60, 255)
                    bgr = cv2.addWeighted(bgr, 0.4, tinted, 0.6, 0)

        if self.show_roi:
            for _gvid, meta in geom.items():
                x, y, bw, bh = meta["bbox"]
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(fw, x + bw), min(fh, y + bh)
                if x1 <= x0 or y1 <= y0:
                    continue
                color = (0, 200, 0) if meta["present"] else (0, 0, 220)
                cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 1)
                cv2.putText(bgr, str(meta["local_id"]), (x0 + 2, max(12, y0 + 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return _fit_and_letterbox(bgr, w, h)

    def _render_bar_chart(self, w: int, h: int) -> np.ndarray:
        img = np.full((h, w, 3), (20, 20, 20), dtype=np.uint8)
        payload = self.latest_payload
        if payload is None:
            cv2.putText(img, "no data yet", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (150, 150, 150), 1, cv2.LINE_AA)
            return img

        face = payload.get("face")
        geom = self.vial_geom.get(face, {})
        cv2.putText(img, f"activity this bin -- face {face}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)
        if not geom:
            cv2.putText(img, "no calibration for this face", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (150, 150, 150), 1, cv2.LINE_AA)
            return img

        ordered = sorted(geom.items(), key=lambda kv: kv[1]["local_id"])
        n = len(ordered)
        top_margin, bottom_margin = 24, 18
        plot_h = max(1, h - top_margin - bottom_margin)
        slot_w = w / n if n else w

        values = []
        for gvid, _meta in ordered:
            acc = self.cur_bin_totals.get(gvid)
            values.append((acc[0] / acc[1]) if acc and acc[1] > 0 else 0.0)
        vmax = max(values) if values and max(values) > 1e-9 else 1.0

        bottom = top_margin + plot_h
        for i, ((gvid, meta), val) in enumerate(zip(ordered, values)):
            cx0 = int(i * slot_w) + 2
            cx1 = max(cx0 + 1, int((i + 1) * slot_w) - 2)
            if not meta["present"]:
                cv2.rectangle(img, (cx0, bottom - 3), (cx1, bottom), (80, 80, 80), -1)
            else:
                bar_h_px = int(plot_h * min(1.0, val / vmax))
                cv2.rectangle(img, (cx0, bottom - bar_h_px), (cx1, bottom), (60, 200, 60), -1)
            label_y = min(h - 4, bottom + 14)
            cv2.putText(img, str(meta["local_id"]), (cx0, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (180, 180, 180), 1, cv2.LINE_AA)
        return img

    def _render_heatmap(self, w: int, h: int) -> np.ndarray:
        img = np.full((h, w, 3), (20, 20, 20), dtype=np.uint8)
        cv2.putText(img, "activity actogram (recent bins)", (8, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 200, 200), 1, cv2.LINE_AA)

        all_vials = sorted(
            ((face, gvid, meta["local_id"])
             for face, fg in self.vial_geom.items() for gvid, meta in fg.items()),
            key=lambda t: (t[0], t[2]),
        )
        n_rows = len(all_vials)
        n_cols = len(self.heatmap_buffer)
        title_h = 16
        if n_rows == 0 or n_cols == 0:
            return img

        plot_h = max(1, h - title_h)
        row_h = max(1, plot_h // n_rows)

        grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        for c, bin_row in enumerate(self.heatmap_buffer):
            for r, (_face, gvid, _local_id) in enumerate(all_vials):
                grid[r, c] = bin_row.get(gvid, 0.0)

        vmax = float(grid.max())
        if vmax > 1e-9:
            norm = np.clip(grid / vmax * 255.0, 0, 255).astype(np.uint8)
        else:
            norm = np.zeros_like(grid, dtype=np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)  # (n_rows, n_cols, 3)

        grid_h = row_h * n_rows
        resized = cv2.resize(colored, (w, grid_h), interpolation=cv2.INTER_NEAREST)
        y0 = title_h
        y1 = min(h, y0 + grid_h)
        img[y0:y1, 0:w] = resized[0:(y1 - y0), :]

        if row_h >= 9:
            for r, (face, _gvid, local_id) in enumerate(all_vials):
                ly = min(h - 2, y0 + r * row_h + min(row_h, 9))
                cv2.putText(img, f"{face}{local_id}", (2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (230, 230, 230), 1, cv2.LINE_AA)
        return img

    # ---- display loop + keyboard ---------------------------------------------------------------

    def maybe_render(self) -> bool:
        """Draw + show the composite if `1/max_fps` seconds have elapsed since the last render;
        otherwise (or if rendering is disabled) a fast no-op -- this is what guarantees the display
        can never slow acquisition (module docstring guarantee 1). Polls one keypress and dispatches
        it via `handle_key`. Never raises: any cv2 display error disables rendering for the rest of
        the run (guarantee 2); acquisition-side state (`on_frame`/`on_bin`) is never affected.
        Returns True iff a frame was actually drawn+shown this call.
        """
        if not self.render_enabled:
            return False
        now = time.monotonic()
        if now - self._last_render_time < self._min_render_interval:
            return False
        self._last_render_time = now

        try:
            if not self.paused:
                composite = self.render_composite()
                cv2.imshow(self.window_name, composite)
            key = cv2.waitKey(1)
        except Exception as exc:
            self._disable_rendering(f"cv2 display error: {exc!r}")
            return False

        if key is not None and key != -1:
            self.handle_key(key)
        return not self.paused

    def handle_key(self, key: int) -> None:
        """Dispatch one `cv2.waitKey()` keycode. A plain function of `key` + current state (plus
        one documented side effect on 's', which writes a PNG) -- independently testable without a
        window: call it directly with an ordinal, e.g. `handle_key(ord('+'))`.
        """
        if key is None or key == -1:
            return
        raw = chr(key & 0xFF) if 0 <= (key & 0xFF) < 128 else ""
        ch = raw.lower() if raw.isalpha() else raw

        if ch == "q":
            self.quit_requested = True
            self._disable_rendering("user quit ('q')", level=logging.INFO)
        elif ch == "p":
            self.paused = not self.paused
        elif ch in ("+", "="):
            self._adjust_threshold(self.threshold_step)
        elif ch == "-":
            self._adjust_threshold(-self.threshold_step)
        elif ch == "o":
            self.show_roi = not self.show_roi
        elif ch == "s":
            self.save_snapshot()

    def _adjust_threshold(self, delta: float) -> None:
        new_value = max(0.0, self.pixel_threshold + delta)
        self.pixel_threshold = new_value
        if self._on_threshold_change is not None:
            try:
                self._on_threshold_change(new_value)
            except Exception:
                logger.warning("LiveMonitor: on_threshold_change callback raised", exc_info=True)

    def save_snapshot(self, path: Optional[str] = None) -> Optional[str]:
        """Write the last-rendered composite to a PNG (rendering fresh first if nothing has been
        rendered yet but at least one frame has arrived). Returns the written path, or None if
        there is nothing to save yet (no frame has ever arrived)."""
        img = self.last_composite
        if img is None and self.latest_payload is not None:
            img = self.render_composite()
        if img is None:
            return None
        if path is None:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(self.snapshot_dir, f"snapshot_{stamp}.png")
        cv2.imwrite(path, img)
        return path

    def _disable_rendering(self, reason: str, *, level: int = logging.WARNING) -> None:
        if self.render_enabled:
            logger.log(level, "LiveMonitor: display rendering disabled (%s)", reason)
        self.render_enabled = False
        self.disabled_reason = reason
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def close(self) -> None:
        """Best-effort window teardown. Idempotent; safe even if a window was never shown."""
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        self.render_enabled = False

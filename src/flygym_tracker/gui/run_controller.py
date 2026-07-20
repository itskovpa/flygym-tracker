"""The experiment, running inside this window, on a worker thread, with the knobs still live.

THE HEADLINE REQUEST. "Camera settings and algorithm controls need to be available when the
experiment is running so I could adjust the setting live." Everything in this file exists to make
that true without any of it reaching the sensor at a moment the pipeline is not expecting.

=================================================================================================
WHY A LIVE CHANGE IS QUEUED AND NOT APPLIED WHERE IT IS TYPED.

A setting is edited on the GUI thread. The frames are read, and the SDK handle is driven, on THIS
worker's thread. Calling `source.set_exposure_us(...)` straight from the GUI thread would be a
second thread writing to a USB3 Vision handle mid-grab -- the one access pattern the whole
`camera_lock` / exclusive-access story exists to keep single-file.

So `queue_setting` appends to a `deque` (thread-safe for append/popleft by the GIL, and the only
shared state here) and the queue is DRAINED ON THE PIPELINE'S OWN THREAD, between frames, through
`TrackerPipeline.add_observer` -- a documented extension point that fires once per processed
frame. The pipeline is not modified, not subclassed and not reached into: it hands us a frame, and
on the way past we apply whatever the operator asked for since the last one.

THE CHANGES GO THROUGH `TrackerPipeline.apply_setting`, WHICH IS THE POINT (invariant 4). That
method is where a change becomes a `setting_change` event in events.csv. A 3-day run whose
`pixel_threshold` moved at hour 40 produces ONE activity.csv holding two different measurement
regimes; without the event nothing anywhere in the output hints that the analysis should not
average across both. Routing live edits anywhere else -- straight onto the source, or onto the
detector -- would apply them correctly and silently, which is the worse failure.

=================================================================================================
WIDTH AND HEIGHT ARE NOT LIVE, AND NOTHING HERE MAKES THEM LIVE (invariant 3).

`pipeline.setting_block_reason` refuses them once frames are flowing, and `apply_setting` asks it
again as a backstop, so a queued geometry change is REFUSED BY THE PIPELINE rather than filtered
out here. That is deliberate: one rule, enforced in one place, asked by every surface. This file
would be the obvious place to add a helpful "restart the stream and reapply" convenience, and that
convenience is exactly the bug -- stopping acquisition under a run that may have been recording for
days is a gap in the series plus a frame-diff baseline reset, i.e. two incomparable regimes in one
file with nothing marking the seam.

=================================================================================================
THE CAMERA IS TAKEN EXCLUSIVELY, ONCE (invariant 5).

USB3 Vision allows one process -- and, in practice, one handle -- at a time. `CameraSession` owns
the preview handle; a run owns its own. They MUST NOT overlap, so `RunController.start` refuses to
begin while the preview session is open and says so, rather than letting the SDK report its
culprit-free 0x80000203 from inside a worker thread where nobody can see it. `main_window` closes
the preview session first; this refusal is the backstop for any caller that forgets.

=================================================================================================
PROGRESS IS THROTTLED, AND THE THROTTLE IS NOT COSMETIC.

At 88 fps a per-frame signal is 88 queued cross-thread emissions per second, each one dragging a
dict of 32 vial results onto the GUI thread. Measured or not, that is a GUI thread doing layout
work instead of a worker doing acquisition, on an app whose entire job is not dropping frames. The
worker emits at most `PROGRESS_HZ` times a second and the numbers it carries -- frames, elapsed,
rotations -- are COUNTED BY THE PIPELINE, not sampled here, so throttling the display cannot
change what is measured or recorded.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot

#: How often the run panel's readouts refresh. See the module docstring -- a per-frame signal at
#: 88 fps is 88 cross-thread emissions a second on the thread that must not fall behind.
PROGRESS_HZ = 5.0

IDLE = "idle"
STARTING = "starting"
RUNNING = "running"
STOPPING = "stopping"
DONE = "done"
FAILED = "failed"


class RunWorker(QObject):
    """Builds and drives one `TrackerPipeline`. Lives on the run thread; owns nothing on the GUI's.

    Every signal carries plain data (str, dict, float) rather than pipeline objects: a queued
    cross-thread signal delivering a live pipeline would put the GUI thread one attribute access
    away from the SDK handle.
    """

    progress = Signal(dict)
    started = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)
    #: `(key, applied)` for each queued change the pipeline actually took. `applied=False` means
    #: the pipeline refused it -- a start-only key, or one this run cannot route -- and the row
    #: says so instead of showing a value that never reached anything.
    setting_applied = Signal(str, bool)

    def __init__(self, plan: Dict[str, Any], latest=None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._plan = dict(plan)
        #: The one-slot frame box the window's picture is pulled from, so a run -- and a replay,
        #: which is the same pipeline over a file -- is WATCHED IN THIS WINDOW rather than in a
        #: separate cv2 monitor window. Same box type and same reasoning as the camera preview's:
        #: Qt does not coalesce queued signals, and a run is watched for days, so a stalled GUI
        #: must show a STALE frame rather than queue every frame of the stall.
        self._latest = latest
        self._stop = threading.Event()
        #: Appended to from the GUI thread, drained on this one. A `deque` because append and
        #: popleft are the two operations, and both are atomic under the GIL -- a lock here would
        #: be held across an SDK write, on the thread that must not stall.
        self._pending: deque = deque()
        self._pipeline = None
        self._last_emit = 0.0
        self._frames = 0

    # -- called from the GUI thread ---------------------------------------------------------
    def queue_setting(self, key: str, value: Any) -> None:
        """Ask for a live change. Applied between frames, on the run thread. Never blocks."""
        self._pending.append((str(key), value))

    def request_stop(self) -> None:
        """Graceful stop. `TrackerPipeline.run` checks this once per iteration and then flushes
        the final bin and closes the logger -- which is why the run is never killed by terminating
        the thread: that would abandon a partial bin and leave the CSV without its last rows."""
        self._stop.set()

    def is_running(self) -> bool:
        return self._pipeline is not None and not self._stop.is_set()

    # -- the run ------------------------------------------------------------------------------
    @Slot()
    def run(self) -> None:
        """Build the pipeline, then hand control to it until it stops.

        Construction failures are turned into a sentence rather than a traceback: the two
        documented ones are null thresholds (the config has never been through `noise`) and an
        unreadable calibration mask (a half-written bundle). Both are things the operator can fix,
        and both arrive here from inside a worker thread where a traceback goes nowhere.
        """
        try:
            pipeline, summary_meta = self._build()
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self._pipeline = pipeline
        pipeline.add_observer(self._on_frame)
        self.started.emit(summary_meta)
        try:
            summary = pipeline.run(max_frames=self._plan.get("max_frames"),
                                   stop_flag=self._stop.is_set)
        except Exception as exc:
            self._pipeline = None
            self.failed.emit(str(exc))
            return
        self._pipeline = None
        self.finished.emit(dict(summary or {}))

    def _build(self):
        """Assemble the pipeline from the SAME helpers the CLI uses.

        Imported inside the function, not at module scope: these drag in cv2, and importing this
        module must stay cheap enough that the settings window opens without an OpenCV load.
        Reusing `cli`'s builders rather than restating them is what keeps a run started from the
        window and a run started from the command line the same run -- the marker-detector choice
        in particular is a bug that was already fixed once, in there.
        """
        from flygym_tracker.calibration import load_calibration
        from flygym_tracker.cli import (_build_marker_detector, _camera_source_from_config,
                                        _make_run_id)
        from flygym_tracker.logger import ActivityLogger
        from flygym_tracker.pipeline import TrackerPipeline

        plan = self._plan
        config = plan["config"]
        calib = load_calibration(plan["calib_dir"])
        source = plan.get("source_factory", lambda: _camera_source_from_config(config))()
        run_id = plan.get("run_id") or _make_run_id()
        logger = ActivityLogger(
            output_dir=plan["output_dir"],
            run_id=run_id,
            fmt=config.output.format,
            rolling=config.output.rolling,
            # `run_meta.json` snapshots the config at START (invariant 4's other half): everything
            # chosen BEFORE the run belongs here, everything changed after belongs in events.csv.
            meta={"config": config.to_dict(), "calibration_dir": plan["calib_dir"],
                  "started_from": "gui"},
        )
        pipeline = TrackerPipeline(
            config=config, calibration=calib, source=source, logger=logger,
            marker_detector=_build_marker_detector(config, calib), clock="auto")
        return pipeline, {"run_id": run_id, "output_dir": plan["output_dir"],
                          "calib_dir": plan["calib_dir"]}

    # -- per frame, on the run thread ----------------------------------------------------------
    def _on_frame(self, payload: dict) -> None:
        """Drain the queued settings, then emit a throttled progress snapshot.

        SETTINGS FIRST. A change queued while this frame was being processed should take effect
        for the NEXT one, and draining before the emit means the progress snapshot the operator
        sees already reflects what they just asked for.

        THIS RUNS INSIDE THE PIPELINE'S FRAME LOOP. It must not raise -- `_notify` counts observer
        failures and carries on, but an exception here would be an exception per frame for the rest
        of a three-day run. Hence the broad guard around each applied setting: a rejected value is
        reported to the row and the acquisition continues.
        """
        self._drain_pending()
        self._frames = int(payload.get("index", self._frames) or 0)

        now = time.monotonic()
        if now - self._last_emit < 1.0 / PROGRESS_HZ:
            return
        # THE PICTURE IS THROTTLED WITH THE READOUT, not per frame. `payload["frame"]` is the
        # pipeline's own grayscale working copy, so it is put in the box only at the moment the
        # GUI is about to be told about it anyway -- at 88 fps, handing over every frame would be
        # a memcpy per frame for a preview the eye samples at about 15.
        #
        # IT IS COPIED. The array in the payload belongs to the pipeline, which keeps the previous
        # frame as the baseline of its next difference; the GUI thread paints from the box whenever
        # it likes. Handing over the live array would make correctness depend on the pipeline never
        # writing into a frame in place -- true today, unowned by anyone, and the failure would be
        # a preview that tears during a measurement rather than an exception anyone could trace.
        # At 5 Hz a 1.3 MB copy is not measurable; the guarantee is.
        if self._latest is not None:
            frame_image = payload.get("frame")
            if frame_image is not None:
                try:
                    self._latest.put(frame_image.copy())
                except Exception:
                    pass                      # a preview must never be able to end a run
        self._last_emit = now
        vial_results = payload.get("vial_results") or {}
        self.progress.emit({
            "frames": self._frames,
            "elapsed_s": float(payload.get("elapsed_s") or 0.0),
            "state": str(payload.get("state") or ""),
            "face": payload.get("face"),
            "n_rotations": int(payload.get("n_rotations") or 0),
            "fps_est": float(payload.get("fps_est") or 0.0),
            "pixel_threshold": payload.get("pixel_threshold"),
            # A shallow copy: the pipeline reuses its own dicts between frames, so handing the
            # live one across a queued signal would let the GUI read a half-written frame.
            "vial_results": dict(vial_results),
        })

    def _drain_pending(self) -> None:
        while True:
            try:
                key, value = self._pending.popleft()
            except IndexError:
                return
            try:
                applied = bool(self._pipeline.apply_setting(key, value))
            except Exception:
                applied = False
            self.setting_applied.emit(key, applied)


class RunController(QObject):
    """The GUI-thread half: owns the thread, forwards the signals, and refuses illegal starts.

    Modelled on `CameraSession` deliberately, including connecting BOUND METHODS rather than
    lambdas -- a lambda capturing `self` keeps this object alive past the window that owns it, and
    the thread it owns with it.
    """

    progress = Signal(dict)
    state_changed = Signal(str, str)          # state, detail
    started = Signal(dict)
    finished = Signal(dict)
    setting_applied = Signal(str, bool)

    def __init__(self, *, camera_is_open: Callable[[], bool],
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        from flygym_tracker.gui.camera_worker import LatestFrame

        #: Frames from the running pipeline, for the window's picture. A run and a replay both fill
        #: this, which is what makes "watch the experiment" and "watch a recording" the same view.
        self.latest = LatestFrame()
        self._camera_is_open = camera_is_open
        self._thread: Optional[QThread] = None
        self._worker: Optional[RunWorker] = None
        self._state = IDLE
        self._detail = ""

    @property
    def state(self) -> str:
        return self._state

    @property
    def detail(self) -> str:
        return self._detail

    @property
    def is_running(self) -> bool:
        return self._state in (STARTING, RUNNING, STOPPING)

    def _set_state(self, state: str, detail: str = "") -> None:
        self._state, self._detail = state, detail
        self.state_changed.emit(state, detail)

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self, plan: Dict[str, Any]) -> bool:
        """Begin a run. False (with a spoken reason) if it must not begin.

        THE CAMERA CHECK IS A BACKSTOP, NOT THE POLICY. `main_window` closes the preview session
        before calling this, because that is where the operator can be told what is about to
        happen. If it did not, the SDK would refuse from inside a worker thread and report
        0x80000203, which names no culprit -- the exact failure `camera_lock` exists to diagnose.
        """
        if self.is_running:
            self._set_state(self._state, "a run is already going")
            return False
        try:
            if self._camera_is_open():
                self._set_state(IDLE, "close the preview camera first - the camera can only be "
                                      "open in one place at a time")
                return False
        except Exception:
            pass
        missing = [name for name in ("config", "calib_dir", "output_dir") if not plan.get(name)]
        if missing:
            self._set_state(IDLE, "cannot start: no %s" % ", ".join(missing))
            return False

        self._thread = QThread()
        self._worker = RunWorker(plan, latest=self.latest)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.started.connect(self._on_started)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.setting_applied.connect(self.setting_applied)
        self._set_state(STARTING, "opening the camera and the output files")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Ask the run to finish. It flushes its final bin and closes its files on the way out."""
        if self._worker is None:
            return
        self._set_state(STOPPING, "finishing the current bin and closing the files")
        self._worker.request_stop()

    def apply_setting(self, key: str, value: Any) -> bool:
        """Route one live change into the running pipeline. False if there is no run to route to.

        Returning False is what puts "takes effect at the next start" on the row, which is TRUE
        when nothing is running -- the value is in the model and will be written to the config.
        """
        if self._worker is None or not self.is_running:
            return False
        self._worker.queue_setting(key, value)
        return True

    def shutdown(self) -> None:
        """Stop and join, synchronously. Called from `closeEvent`.

        LEAVING A RUN THREAD BEHIND leaks the exclusive camera handle with no window on screen to
        explain it, which is precisely how the next session's "camera is busy" is created.
        """
        self.stop()
        thread = self._thread
        if thread is not None:
            thread.quit()
            thread.wait(SHUTDOWN_WAIT_MS)
        self._thread = None
        self._worker = None

    # -- worker signals, on the GUI thread ------------------------------------------------------
    def _on_progress(self, payload: dict) -> None:
        if self._state == STARTING:
            self._set_state(RUNNING, "")
        self.progress.emit(payload)

    def _on_started(self, meta: dict) -> None:
        self._set_state(RUNNING, "run %s" % meta.get("run_id", ""))
        self.started.emit(meta)

    def _on_finished(self, summary: dict) -> None:
        self._set_state(DONE, "%d frames, %d rotations" % (
            summary.get("frames_processed", 0), summary.get("n_rotations", 0)))
        self._join()
        self.finished.emit(summary)

    def _on_failed(self, message: str) -> None:
        self._set_state(FAILED, message)
        self._join()

    def _join(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.quit()
            thread.wait(SHUTDOWN_WAIT_MS)
        self._thread = None
        self._worker = None


#: Matches `camera_session.SHUTDOWN_WAIT_MS`. Long enough for a final bin flush and a logger close;
#: short enough that a wedged thread cannot stop the window closing.
SHUTDOWN_WAIT_MS = 5000

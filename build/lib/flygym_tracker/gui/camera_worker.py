"""The one object in the process that touches the camera SDK, living on its own thread.

THE RULE THIS FILE ENFORCES: every SDK call happens on this object's thread, serialised by that
thread's event loop. No locks around the SDK, because there is never a second thread inside it.
The GUI never holds a `HikCameraSource`; it asks `CameraSession`, which posts a queued slot here.

THREE THINGS ARE BUILT DELIBERATELY AND EACH REPLACES SOMETHING OBVIOUS THAT DOES NOT WORK.

1. THE GRAB LOOP IS SELF-REARMING, NOT `while True`. A loop would starve this thread's event loop,
   so a queued exposure change would sit behind every frame still to be read -- unbounded, on a
   camera delivering 88 fps. `_grab_once` reads ONE frame and re-arms with
   `QTimer.singleShot(0, ...)`; measured, that is drained by a single `processEvents()` pass, so
   setting writes interleave BETWEEN frames. That is also what makes a live exposure change land
   cleanly rather than half-way through a read.

2. FRAMES GO INTO A ONE-SLOT BOX, NOT A SIGNAL. Qt does not coalesce queued signals. Measured on
   this machine: 300 frame-sized payloads emitted at a GUI thread that was not running its loop
   left 300 of 300 undelivered after 0.5 s, holding the memory of all of them. A run is watched for
   DAYS -- a dragged window, a screen lock, a virus scan or a modal dialog is a stall, and a 60 s
   stall at 15 fps decimated would queue 900 frames. Decimating lowers the rate but does not bound
   the queue; only dropping does. So the worker writes the latest frame into `latest_frame`
   under a lock and the GUI pulls it on a timer: a stalled GUI shows a STALE frame instead of
   growing, which is the failure mode that cannot end a run.

3. WRITES ARE TOKENED. `CameraSession` mints a token per authorised write; a call arriving without
   the pending token is refused and says so. `setEnabled(False)` is not a guard -- measured, a
   programmatic `setValue` on a disabled spinbox succeeds -- so the guard has to be somewhere a
   future contributor cannot walk around by accident.

WHAT IS NOT BUILT, AND WHY. There is no periodic `current_values()` poll. At 88 fps there are
~11 ms between reads, and a slow GenICam node read inside that gap drops a frame; a cosmetic
status-bar refresh is not worth a hole in a measurement. Values are read back at the two moments
they are load-bearing: just after a write (so the status bar shows the SDK's clamped value rather
than the requested one) and just after open.
"""
from __future__ import annotations

import threading
import time
from PySide6.QtCore import QObject, QTimer, Signal, Slot

#: Preview cadence, in milliseconds between frames handed to the GUI. ~15 fps: the eye needs about
#: that, the sensor delivers up to 88, and every frame in between is READ (the SDK buffer fills
#: otherwise) and counted as dropped rather than quietly discarded.
PREVIEW_INTERVAL_MS = 66

#: Rolling window for the delivered-frame-rate measurement.
FPS_WINDOW = 30

#: Which `HikCameraSource` setter each config key routes to. A literal table, never `setattr`:
#: the set of things the GUI can push into the SDK is visible in one screenful, and a typo in a
#: key produces a refusal instead of a new attribute on the camera object.
SETTER_FOR_KEY = {
    "source.camera.frame_rate": "set_frame_rate",
    "source.camera.exposure_us": "set_exposure_us",
    "source.camera.gain_db": "set_gain_db",
    "source.camera.width": "set_width",
    "source.camera.height": "set_height",
}

#: The `HikCameraSource` attribute each key reads back from, for the post-write confirmation.
ATTR_FOR_KEY = {
    "source.camera.frame_rate": "frame_rate",
    "source.camera.exposure_us": "exposure_us",
    "source.camera.gain_db": "gain_db",
    "source.camera.width": "width",
    "source.camera.height": "height",
}


class LatestFrame:
    """A one-slot mailbox for frames, written by the grabber and drained by the GUI.

    `put` overwrites; `take` empties. That asymmetry is the whole design: the writer never blocks
    and never accumulates, and the reader gets the newest frame or nothing. `dropped` counts the
    frames that were read from the sensor but never shown, so the preview can SAY how much of the
    stream the operator is not looking at instead of implying they are seeing all of it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None
        self._dropped = 0
        self._shown = 0

    def put(self, image) -> None:
        with self._lock:
            if self._frame is not None:
                self._dropped += 1        # displaced before anyone looked at it
            self._frame = image

    def take(self):
        with self._lock:
            frame, self._frame = self._frame, None
            if frame is not None:
                self._shown += 1
            return frame

    def count_skipped(self, n: int = 1) -> None:
        """Frames read from the sensor but never offered to the preview (the throttle's doing)."""
        with self._lock:
            self._dropped += int(n)

    @property
    def stats(self):
        with self._lock:
            return (self._shown, self._dropped)


class CameraWorker(QObject):
    """Owns a `HikCameraSource` on the thread this object was moved to. Every slot runs there.

    Signals are connected ONLY to bound methods of GUI-thread QObjects. Measured: a lambda has no
    receiver QObject, so the connection's receiver is the SENDER, which lives here -- and the slot
    then runs on THIS thread even with an explicit `QueuedConnection`. A lambda that touched a
    QWidget would be undefined behaviour inside a run that lasts days, with nothing on screen to
    say so.
    """

    opened = Signal(dict)                 # {"serial": str, "size": (w, h), "values": {...}}
    failed = Signal(str, bool)            # (message, looks_like_busy)
    closed = Signal()
    written = Signal(str, bool, object, str)   # (key, ok, confirmed_value, message)
    ranges_changed = Signal(dict)         # {node: CameraRange}
    values_read = Signal(dict)            # current_values() read-back
    read_error = Signal(str)
    #: The attached per-frame job has seen everything it needs; it is already detached.
    tap_finished = Signal()
    #: The attached per-frame job raised. It is detached, and the message says what happened --
    #: a measurement that stopped accumulating must never keep showing progress.
    tap_failed = Signal(str)

    def __init__(self, factory, latest: LatestFrame) -> None:
        """`factory` builds the `HikCameraSource`; it is injected so tests never need an SDK."""
        super().__init__()
        self._factory = factory
        self._latest = latest
        self._source = None
        self._running = False
        self._last_preview = 0.0
        self._stamps: list = []
        self._frames = 0
        #: Authorised writes waiting to be applied, and the tokens that authorise them. A LIST
        #: UNDER A LOCK rather than arguments on the queued call, because PySide6 cannot marshal a
        #: Python `object` through `Q_ARG` at all -- `Q_ARG(object, value)` raises "Unable to find
        #: a QMetaType for 'object'" at the moment of the first write, which is to say the first
        #: time an operator turns a knob on a real camera. The queue is drained by a no-argument
        #: slot, so nothing has to cross Qt's type system.
        self._write_lock = threading.Lock()
        self._queued_writes: list = []
        self._valid_tokens: set = set()
        #: An optional per-frame job (`video_jobs.FrameJob`) fed EVERY frame this worker reads.
        #:
        #: WHY THE TAP IS HERE AND NOT ON THE PREVIEW BOX. The preview is decimated to ~15 fps out
        #: of up to 88, on purpose. A noise floor is measured from |frame - previous frame|, so a
        #: job fed from the preview box would be differencing frames 66 ms apart instead of 11 ms
        #: -- a DIFFERENT measurement, silently, and the number it produces seeds the pixel
        #: threshold every activity reading afterwards is compared against. Face learning has the
        #: same requirement for the same reason: its rotation detector is the tracker's, and it is
        #: tuned against consecutive frames.
        #:
        #: One slot, not a list: two jobs on one camera would be two measurements racing for the
        #: same exclusive stream, which is exactly the confusion this app exists to remove.
        self._tap = None

    # -- lifecycle ---------------------------------------------------------------------------
    @Slot()
    def open(self) -> None:
        """Open the camera and start grabbing. Never raises across the thread boundary."""
        if self._source is not None:
            return
        try:
            source = self._factory()
            source.open()
        except Exception as exc:
            self._source = None
            busy = _looks_busy(exc)
            self.failed.emit(str(exc), busy)
            return
        self._source = source
        self._running = True
        self._frames = 0
        self._stamps = []
        # Read-back at the one moment it is free: nothing is being grabbed yet, so a GenICam round
        # trip cannot cost a frame.
        values = _safe(lambda: source.current_values(), {})
        self.ranges_changed.emit(_safe(lambda: source.ranges(), {}))
        self.opened.emit({
            "serial": getattr(source, "serial", None),
            "size": _safe(lambda: source.frame_size, (0, 0)),
            "values": values,
        })
        QTimer.singleShot(0, self._grab_once)

    @Slot()
    def close(self) -> None:
        """Stop grabbing and release the device. Safe to call when nothing is open.

        Releasing matters more than it looks: an exclusive USB3 handle left open is what creates
        the NEXT session's "camera is busy", and the operator has nothing on screen to blame.
        """
        self._running = False
        source, self._source = self._source, None
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                self.read_error.emit("closing the camera failed: %s" % exc)
        self.closed.emit()

    # -- the grab loop -------------------------------------------------------------------------
    @Slot()
    def _grab_once(self) -> None:
        """Read exactly one frame, offer it to the preview if it is time, and re-arm."""
        if not self._running or self._source is None:
            return
        try:
            frame = self._source.read()
        except Exception as exc:
            self._running = False
            self.read_error.emit(str(exc))
            return
        if frame is not None:
            self._frames += 1
            now = time.monotonic()
            self._stamps.append(now)
            if len(self._stamps) > FPS_WINDOW:
                del self._stamps[:-FPS_WINDOW]
            image = getattr(frame, "image", frame)
            tap = self._tap
            if tap is not None:
                # EVERY frame, before the preview throttle. A job that raises is DETACHED rather
                # than left to raise once per frame for the rest of a session -- and detaching it
                # is what lets the controller notice and say so, instead of a measurement that
                # quietly stopped accumulating while its progress bar kept moving.
                try:
                    tap.observe(image)
                except Exception as exc:
                    self._tap = None
                    self.tap_failed.emit(str(exc))
                else:
                    if tap.done:
                        self._tap = None
                        self.tap_finished.emit()
            if (now - self._last_preview) * 1000.0 >= PREVIEW_INTERVAL_MS:
                self._last_preview = now
                self._latest.put(getattr(frame, "image", frame))
            else:
                self._latest.count_skipped()
        QTimer.singleShot(0, self._grab_once)

    @property
    def measured_fps(self) -> float:
        """Frames per second actually DELIVERED, over the last `FPS_WINDOW` reads. 0 if unknown.

        Named "delivered" everywhere it is shown, and never mixed up with the AcquisitionFrameRate
        SETTING or with the camera's own ResultingFrameRate. On this rig's camera the frame-rate
        limiter is documented to disengage mid-stream while its registers still read back correct,
        so the only number that can be trusted is the one counted here, from frames that arrived.
        """
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    # -- the per-frame tap ------------------------------------------------------------------------
    def set_tap(self, job) -> None:
        """Attach (or, with None, detach) the job fed every frame. See `_tap`.

        A plain attribute write rather than a queued slot: it is one reference, the grab loop reads
        it into a local before using it, and CPython's GIL makes both atomic. A queued slot would
        mean the job starts accumulating some indeterminate number of frames after the operator
        pressed the button, which is a measurement whose length nobody can state.
        """
        self._tap = job

    @property
    def tap(self):
        return self._tap

    # -- settings ------------------------------------------------------------------------------
    def enqueue(self, key: str, value, token) -> None:
        """Authorise one write and queue it. Called from the GUI thread, under the lock."""
        with self._write_lock:
            self._valid_tokens.add(token)
            self._queued_writes.append((key, value, token))

    @Slot()
    def drain_writes(self) -> None:
        """Apply every authorised write that is waiting. Runs on the worker thread."""
        while True:
            with self._write_lock:
                if not self._queued_writes:
                    return
                key, value, token = self._queued_writes.pop(0)
            self.apply_setting(key, value, token)

    def apply_setting(self, key: str, value, token) -> None:
        """Push one setting to the sensor, then READ IT BACK and report what actually took.

        The read-back is the point. The SDK clamps a float and snaps an int, so the number the
        operator asked for and the number the sensor is using are routinely different; a status bar
        showing the request would be showing something that is not happening. `confirmed` is the
        attribute AFTER the call.

        A write with no matching token is REFUSED, and a token is spent when it is used. That is
        what makes `setEnabled(False)` not have to be a guard: the only path to the SDK is through
        `CameraSession.write`, which asks `block_reason` first, and a contributor calling this
        method directly gets a refusal with a message rather than a silent geometry change under a
        recording experiment.
        """
        with self._write_lock:
            authorised = token is not None and token in self._valid_tokens
            self._valid_tokens.discard(token)
        if not authorised:
            self.written.emit(key, False, None,
                              "refused: this write did not come from the settings surface")
            return
        setter_name = SETTER_FOR_KEY.get(key)
        if setter_name is None or self._source is None:
            self.written.emit(key, False, None, "no camera to send %s to" % key)
            return
        try:
            getattr(self._source, setter_name)(value)
        except Exception as exc:
            self.written.emit(key, False, None, str(exc))
            return
        confirmed = getattr(self._source, ATTR_FOR_KEY.get(key, ""), None)
        self.written.emit(key, True, confirmed, "")

    @Slot()
    def refresh_ranges(self) -> None:
        """Re-read the limits from the sensor.

        `ranges()` caches at open, and `refresh_ranges()` existed with nothing calling it. Changing
        Width legitimately changes the legal ExposureTime and AcquisitionFrameRate ranges, so a
        spinbox built from the stale cache can offer a value the SDK will REJECT -- and on a
        start-only node a rejected value is a failed run, not a slightly wrong picture.
        """
        if self._source is None:
            return
        self.ranges_changed.emit(_safe(lambda: self._source.refresh_ranges(), {}))

    @Slot()
    def read_values(self) -> None:
        """Read `current_values()` on demand -- open, after a write, or when the operator asks."""
        if self._source is None:
            self.values_read.emit({})
            return
        self.values_read.emit(_safe(lambda: self._source.current_values(), {}))

    @property
    def is_acquiring(self) -> bool:
        return bool(getattr(self._source, "is_acquiring", False))

    @property
    def source(self):
        """The live source, for `block_reason` to ask `is_acquiring` of. Read-only by convention."""
        return self._source


def _safe(call, default):
    try:
        return call()
    except Exception:
        return default


def _looks_busy(exc: Exception) -> bool:
    """Is this the SDK's exclusive-access complaint? Asked of `camera_lock`, never re-implemented."""
    try:
        from flygym_tracker.camera_lock import looks_like_busy_error

        return bool(looks_like_busy_error(exc))
    except Exception:
        return False

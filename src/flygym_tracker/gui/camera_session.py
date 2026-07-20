"""The GUI-thread face of the camera: a state machine, a thread, and the only door to the SDK.

NOTHING ELSE IN THE APP MAY CONSTRUCT A `HikCameraSource`. Views talk to this object; this object
owns the `QThread` and the `CameraWorker` on it. That is not tidiness -- USB3 Vision access is
exclusive, so a second handle opened anywhere in the process is a self-inflicted "camera is busy",
and the operator would be looking at an app that says the camera is held by something it cannot
name because the something is itself.

THE STATES, and what each one means to the operator (the status bar renders exactly these):

    CLOSED       nothing is being sent; the camera is free for MVS, Bonsai, or a run
    OPENING      enumerating and configuring; every control that could race the open is disabled
    STREAMING    this app has the camera and frames are arriving
    CLOSING      releasing the handle; the window will not finish closing until this is done
    ERROR_BUSY   something else holds it -- `camera_lock` can name what
    ERROR_OTHER  it failed for a reason that is not exclusivity; the message says what

WRITES ARE ORIGINATED HERE OR NOWHERE. `write()` asks `block_reason` first, then mints a ONE-SHOT
TOKEN, queues the write on the worker with it, and posts a no-argument slot to drain the queue.
`CameraWorker.apply_setting` refuses anything whose token it did not issue, and a token is spent
when it is used. So a contributor who calls `camera.set_exposure_us()` -- or the worker's own
`apply_setting` -- directly hits a refusal with a message rather than a comment asking them not to,
which matters because the thing being guarded is a geometry change under an experiment that has
been recording for two days.

(The value travels through a lock-protected list rather than as a queued-call argument because
PySide6 cannot marshal a Python `object` through `Q_ARG` at all -- see `write`.)
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Optional

from PySide6.QtCore import QMetaObject, QObject, Qt, QThread, Signal, Slot

from flygym_tracker.gui.camera_worker import CameraWorker, LatestFrame

CLOSED = "closed"
OPENING = "opening"
STREAMING = "streaming"
CLOSING = "closing"
ERROR_BUSY = "error_busy"
ERROR_OTHER = "error_other"

#: How long `shutdown` waits for the worker thread to finish releasing the device. Generous: the
#: alternative to waiting is leaking an exclusive handle, which is the next session's bug.
SHUTDOWN_WAIT_MS = 5000


class CameraSession(QObject):
    """Owns the camera thread. The views' only camera API.

    Args:
        factory: builds a `HikCameraSource`. Injected, so tests drive a fake and this class is
            testable on a machine with no camera -- which is every machine this was written on.
    """

    state_changed = Signal(str, str)          # (state, one-sentence detail)
    written = Signal(str, bool, object, str)  # (key, ok, confirmed, message)
    ranges_changed = Signal(dict)
    values_read = Signal(dict)
    #: The attached per-frame job finished, or failed with a message. See `attach_tap`.
    tap_finished = Signal()
    tap_failed = Signal(str)

    def __init__(self, factory: Callable[[], Any], parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.latest = LatestFrame()
        self._state = CLOSED
        self._detail = ""
        self._serial: Optional[str] = None
        self._tokens = itertools.count(1)

        self._thread = QThread()
        self._thread.setObjectName("flygym-camera")
        self._worker = CameraWorker(factory, self.latest)
        self._worker.moveToThread(self._thread)

        # EVERY ONE OF THESE IS A BOUND METHOD OF AN OBJECT BUILT ON THE GUI THREAD. Measured: a
        # lambda connected to a signal emitted from the worker thread runs ON THE WORKER THREAD,
        # even with an explicit QueuedConnection, because a lambda has no receiver QObject so the
        # connection's receiver is the sender. A lambda here would touch widgets from the grabber
        # thread: undefined behaviour, silent, in a run that lasts days.
        self._worker.opened.connect(self._on_opened)
        self._worker.failed.connect(self._on_failed)
        self._worker.closed.connect(self._on_closed)
        self._worker.written.connect(self._on_written)
        self._worker.ranges_changed.connect(self._on_ranges)
        self._worker.values_read.connect(self._on_values)
        self._worker.read_error.connect(self._on_read_error)
        self._worker.tap_finished.connect(self.tap_finished)
        self._worker.tap_failed.connect(self.tap_failed)
        self._thread.start()

    # -- state -------------------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def detail(self) -> str:
        return self._detail

    @property
    def serial(self) -> Optional[str]:
        return self._serial

    @property
    def is_open(self) -> bool:
        return self._state == STREAMING

    @property
    def source(self):
        """The live `HikCameraSource`, ONLY for asking `is_acquiring`.

        Exposed because `start_only_block_reason` takes the source object and asks it that one
        question -- the whole reason `is_acquiring` is public is so everyone asks the same object
        instead of keeping a private idea of whether the stream is live. It is not an invitation to
        call setters on it; those go through `write`, which is the only path with a token.
        """
        return self._worker.source

    @property
    def measured_fps(self) -> float:
        """Frames per second actually DELIVERED. Read from the worker without a round trip because
        it is a float written by one thread and read by another -- stale by at most one frame, and
        never used for anything but a label."""
        return self._worker.measured_fps

    def _set_state(self, state: str, detail: str = "") -> None:
        if (state, detail) == (self._state, self._detail):
            return
        self._state, self._detail = state, detail
        self.state_changed.emit(state, detail)

    # -- commands ----------------------------------------------------------------------------
    def open(self) -> None:
        """Ask for the camera. THE APP NEVER DOES THIS BY ITSELF.

        USB3 Vision is exclusive: an app that takes the camera when it launches is an app that
        stops the rig from running. So opening is a button, and until it is pressed the settings
        show the rig camera's documented limits and say so.
        """
        if self._state in (OPENING, STREAMING, CLOSING):
            return
        self._set_state(OPENING, "Opening camera...")
        QMetaObject.invokeMethod(self._worker, "open", Qt.ConnectionType.QueuedConnection)

    def close(self) -> None:
        if self._state not in (STREAMING, ERROR_BUSY, ERROR_OTHER):
            return
        self._set_state(CLOSING, "Releasing the camera...")
        QMetaObject.invokeMethod(self._worker, "close", Qt.ConnectionType.QueuedConnection)

    def write(self, key: str, value, *, block_reason: Optional[str] = None) -> bool:
        """Send one setting to the sensor. False (and nothing sent) if it must not be sent.

        `block_reason` is passed in rather than recomputed: the settings surface has already asked
        the one shared rule, and asking a second time from a second place is how two answers appear.
        """
        if block_reason is not None:
            return False
        if self._state != STREAMING:
            return False
        # The write is QUEUED on the worker and then a no-argument slot is posted to drain it.
        # Passing the value as a `Q_ARG(object, ...)` is the obvious form and it does not work:
        # PySide6 raises "Unable to find a QMetaType for 'object'" -- at the first write, i.e. the
        # first time an operator turns a knob on a real camera. Caught by
        # `tests/test_gui_camera_session.py::test_a_live_setting_reaches_the_camera_and_is_read_back`.
        token = next(self._tokens)
        self._worker.enqueue(key, value, token)
        QMetaObject.invokeMethod(self._worker, "drain_writes", Qt.ConnectionType.QueuedConnection)
        return True

    def attach_tap(self, job) -> bool:
        """Feed `job` EVERY frame the camera delivers, on the camera thread. False if it cannot be.

        This is how a noise measurement or a face-learning session is done in the window against
        the live camera without opening a second handle -- which is the one thing this class
        exists to make impossible. The job sees the undecimated stream; the preview goes on
        showing its ~15 fps of it (see `camera_worker._tap` for why that distinction is not
        cosmetic).
        """
        if self._state != STREAMING or self._worker.tap is not None:
            return False
        self._worker.set_tap(job)
        return True

    def detach_tap(self) -> None:
        """Stop feeding the attached job. Safe when there is none."""
        self._worker.set_tap(None)

    @property
    def tap(self):
        """The attached job, or None. Read from the GUI thread for its progress figures -- those
        are counters written by one thread and read by another, stale by at most one frame."""
        return self._worker.tap

    def refresh_ranges(self) -> None:
        if self._state == STREAMING:
            QMetaObject.invokeMethod(self._worker, "refresh_ranges",
                                     Qt.ConnectionType.QueuedConnection)

    def read_values(self) -> None:
        if self._state == STREAMING:
            QMetaObject.invokeMethod(self._worker, "read_values",
                                     Qt.ConnectionType.QueuedConnection)

    def shutdown(self) -> None:
        """Release the camera and stop the thread, IN THAT ORDER, before the window goes away.

        Ordering is the whole content of this method: stop the grab loop, close the device on the
        thread that opened it, then quit and join. Tearing the thread down first would leave the
        SDK handle open in a process that is exiting, and the next session would meet a camera held
        by nothing anyone can find.
        """
        if self._thread.isRunning():
            QMetaObject.invokeMethod(self._worker, "close", Qt.ConnectionType.BlockingQueuedConnection)
            self._thread.quit()
            self._thread.wait(SHUTDOWN_WAIT_MS)
        self._set_state(CLOSED, "")

    # -- worker signals, all arriving on the GUI thread ---------------------------------------
    @Slot(dict)
    def _on_opened(self, info: dict) -> None:
        self._serial = info.get("serial")
        size = info.get("size") or (0, 0)
        who = self._serial or "camera"
        self._set_state(STREAMING, "%s is yours - %dx%d" % (who, size[0], size[1]))
        self.values_read.emit(info.get("values") or {})

    @Slot(str, bool)
    def _on_failed(self, message: str, busy: bool) -> None:
        self._set_state(ERROR_BUSY if busy else ERROR_OTHER, message)

    @Slot()
    def _on_closed(self) -> None:
        self._serial = None
        self._set_state(CLOSED, "")

    @Slot(str, bool, object, str)
    def _on_written(self, key: str, ok: bool, confirmed, message: str) -> None:
        self.written.emit(key, ok, confirmed, message)

    @Slot(dict)
    def _on_ranges(self, ranges: dict) -> None:
        self.ranges_changed.emit(ranges)

    @Slot(dict)
    def _on_values(self, values: dict) -> None:
        self.values_read.emit(values)

    @Slot(str)
    def _on_read_error(self, message: str) -> None:
        """A read that threw has ended the stream; say so rather than showing a frozen picture."""
        self._set_state(ERROR_OTHER, "the camera stopped delivering frames: %s" % message)

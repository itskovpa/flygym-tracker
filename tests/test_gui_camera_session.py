"""The camera thread: where frames arrive, what may write to the SDK, and what happens under load.

THERE IS NO CAMERA ON THIS MACHINE. Every camera here is a fake, and what these tests prove is the
CALL PATTERN -- that frames are delivered on the GUI thread, that a write nobody authorised is
refused, that the frame path drops instead of accumulating, and that shutdown releases the device
before the thread goes away. They do NOT prove the sensor accepts any of it; that needs the rig.

THE TEST THAT WOULD HAVE CAUGHT THE LAMBDA BUG is `test_frames_are_delivered_on_the_gui_thread`.
Measured three connection styles for a signal emitted from a worker thread, recording
`threading.get_ident()` inside the slot:

    lambda, no context                    -> ran on the WORKER thread
    lambda, explicit QueuedConnection     -> ran on the WORKER thread
    bound method of a GUI-thread QObject  -> ran on the GUI thread

A lambda has no receiver QObject, so the connection's receiver is the sender -- which lives on the
worker thread -- and even an explicit QueuedConnection posts to the worker's own loop. A lambda
touching a QWidget would be undefined behaviour, silent, in a run that lasts days.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QObject, QThread, Signal, Slot

from flygym_tracker.gui.camera_session import (CLOSED, ERROR_BUSY, ERROR_OTHER, STREAMING,
                                               CameraSession)
from flygym_tracker.gui.camera_worker import LatestFrame

MAIN_IDENT = threading.get_ident()


class FakeFrame:
    def __init__(self, image):
        self.image = image


class FakeSource:
    """A `HikCameraSource` stand-in that records everything it was told, and can be made to fail."""

    def __init__(self, *, fail_open=None, serial="DA4282883", size=(64, 48)):
        self.serial = serial
        self._fail_open = fail_open
        self._size = size
        self.is_acquiring = False
        self.opened = 0
        self.closed = 0
        self.sent = []
        self.frames_read = 0
        self.frame_rate = None
        self.exposure_us = None
        self.gain_db = None
        self.width = None
        self.height = None
        self.ranges_reads = 0

    # -- lifecycle
    def open(self):
        if self._fail_open is not None:
            raise RuntimeError(self._fail_open)
        self.opened += 1
        self.is_acquiring = True

    def close(self):
        self.closed += 1
        self.is_acquiring = False

    def read(self):
        self.frames_read += 1
        image = np.full((self._size[1], self._size[0]), self.frames_read % 256, dtype=np.uint8)
        return FakeFrame(image)

    @property
    def frame_size(self):
        return self._size

    # -- limits + values
    def ranges(self):
        from flygym_tracker.frame_source import fallback_camera_ranges

        self.ranges_reads += 1
        return fallback_camera_ranges()

    def refresh_ranges(self):
        return self.ranges()

    def current_values(self):
        return {"frame_rate": 88.5, "exposure_us": 5000.0}

    # -- setters
    def set_frame_rate(self, value):
        self.sent.append(("frame_rate", value))
        self.frame_rate = value

    def set_exposure_us(self, value):
        self.sent.append(("exposure_us", value))
        self.exposure_us = value

    def set_gain_db(self, value):
        self.sent.append(("gain_db", value))
        self.gain_db = value

    def set_width(self, value):
        if self.is_acquiring:
            raise RuntimeError("Width can only be changed while the stream is stopped")
        self.sent.append(("width", value))
        self.width = value

    def set_height(self, value):
        if self.is_acquiring:
            raise RuntimeError("Height can only be changed while the stream is stopped")
        self.sent.append(("height", value))
        self.height = value


@pytest.fixture
def session(qapp):
    made = []

    def factory():
        source = FakeSource()
        made.append(source)
        return source

    s = CameraSession(factory)
    s.made = made
    yield s
    s.shutdown()


# =============================================================================================
# The state machine
# =============================================================================================
def test_a_new_session_holds_nothing(qapp, session):
    """The app NEVER takes the camera at launch: USB3 Vision is exclusive, so an app that grabs it
    on startup is an app that blocks the rig."""
    assert session.state == CLOSED
    assert session.is_open is False
    assert session.made == [], "constructing the session already built a camera"


def test_opening_reaches_streaming_and_names_the_serial(qapp, session, pump):
    session.open()
    assert pump(lambda: session.state == STREAMING, timeout=5.0), session.state
    assert session.serial == "DA4282883"
    assert session.made[0].opened == 1


def test_a_busy_camera_is_distinguished_from_any_other_failure(qapp, pump):
    """`0x80000203` names no culprit, which is the whole problem -- so it gets its own state, and
    that state is the one that offers to show what is holding the camera."""
    busy = CameraSession(lambda: FakeSource(fail_open="MV_CC_OpenDevice failed (ret=0x80000203)"))
    other = CameraSession(lambda: FakeSource(fail_open="no camera with serial 'X' found"))
    try:
        busy.open()
        other.open()
        assert pump(lambda: busy.state == ERROR_BUSY, timeout=5.0), busy.state
        assert pump(lambda: other.state == ERROR_OTHER, timeout=5.0), other.state
    finally:
        busy.shutdown()
        other.shutdown()


def test_closing_releases_the_device(qapp, session, pump):
    session.open()
    pump(lambda: session.state == STREAMING, timeout=5.0)
    session.close()
    assert pump(lambda: session.state == CLOSED, timeout=5.0)
    assert session.made[0].closed >= 1


def test_shutdown_releases_the_camera_before_the_thread_goes_away(qapp, pump):
    """LEAKING AN EXCLUSIVE USB3 HANDLE IS WHAT CREATES THE NEXT SESSION'S "camera is busy" -- with
    no window on screen to explain it."""
    source = FakeSource()
    s = CameraSession(lambda: source)
    s.open()
    pump(lambda: s.state == STREAMING, timeout=5.0)
    s.shutdown()
    assert source.closed >= 1
    assert source.is_acquiring is False


# =============================================================================================
# Thread affinity -- the test that would have caught the lambda bug
# =============================================================================================
def test_camera_signals_are_delivered_on_the_gui_thread(qapp, session, pump):
    idents = []

    class Receiver(QObject):
        @Slot(str, str)
        def on_state(self, state, detail):
            idents.append(threading.get_ident())

    receiver = Receiver()
    session.state_changed.connect(receiver.on_state)
    session.open()
    pump(lambda: session.state == STREAMING, timeout=5.0)
    assert idents, "no state change was delivered at all"
    assert set(idents) == {MAIN_IDENT}, \
        "a camera signal ran on the worker thread: %r vs main %r" % (set(idents), MAIN_IDENT)


def test_a_lambda_would_have_run_on_the_wrong_thread(qapp):
    """The measurement this rule is built on, kept as a test so the rule is not folklore.

    If this ever starts passing on the GUI thread, Qt's connection semantics changed and the hard
    rule in `camera_session` can be revisited. Until then, bound methods only.
    """
    ran_on = []

    class Emitter(QObject):
        fired = Signal()

        @Slot()
        def go(self):
            self.fired.emit()

    thread = QThread()
    emitter = Emitter()
    emitter.moveToThread(thread)
    emitter.fired.connect(lambda: ran_on.append(threading.get_ident()))
    thread.start()
    from PySide6.QtCore import QMetaObject, Qt

    QMetaObject.invokeMethod(emitter, "go", Qt.ConnectionType.QueuedConnection)
    end = time.monotonic() + 3.0
    while time.monotonic() < end and not ran_on:
        qapp.processEvents()
        time.sleep(0.002)
    thread.quit()
    thread.wait(3000)
    assert ran_on and ran_on[0] != MAIN_IDENT, \
        "a lambda now delivers on the GUI thread; the connection rule can be re-examined"


# =============================================================================================
# Writes are originated by the session or not at all
# =============================================================================================
def test_a_live_setting_reaches_the_camera_and_is_read_back(qapp, session, pump):
    session.open()
    pump(lambda: session.state == STREAMING, timeout=5.0)
    seen = []

    class Receiver(QObject):
        @Slot(str, bool, object, str)
        def on_written(self, key, ok, confirmed, message):
            seen.append((key, ok, confirmed))

    receiver = Receiver()
    session.written.connect(receiver.on_written)
    assert session.write("source.camera.exposure_us", 5000.0) is True
    assert pump(lambda: bool(seen), timeout=5.0)
    assert seen[0][:2] == ("source.camera.exposure_us", True)
    assert seen[0][2] == 5000.0, "the confirmation must be read back AFTER the write"
    assert session.made[0].sent == [("exposure_us", 5000.0)]


def test_a_write_the_session_did_not_originate_is_refused(qapp, session, pump):
    """`setEnabled(False)` IS NOT A GUARD -- measured, a programmatic `setValue` on a disabled
    spinbox succeeds. So a future contributor calling the worker slot directly has to hit a wall
    rather than a comment asking them not to."""
    session.open()
    pump(lambda: session.state == STREAMING, timeout=5.0)
    seen = []

    class Receiver(QObject):
        @Slot(str, bool, object, str)
        def on_written(self, key, ok, confirmed, message):
            seen.append((ok, message))

    receiver = Receiver()
    session.written.connect(receiver.on_written)
    # Straight at the worker, with no token -- the shape of an accidental direct call.
    session._worker.apply_setting("source.camera.exposure_us", 99.0, None)
    assert pump(lambda: bool(seen), timeout=5.0)
    assert seen[0][0] is False
    assert "did not come from the settings surface" in seen[0][1]
    assert session.made[0].sent == [], "an unauthorised write reached the camera"


def test_a_token_cannot_be_replayed(qapp, session, pump):
    """One token, one write. A stale token left lying around must not become a second write."""
    session.open()
    pump(lambda: session.state == STREAMING, timeout=5.0)
    session.write("source.camera.gain_db", 2.0)
    pump(lambda: session.made[0].sent != [], timeout=5.0)
    session._worker.apply_setting("source.camera.gain_db", 9.0, 1)
    qapp.processEvents()
    assert session.made[0].sent == [("gain_db", 2.0)]


def test_a_blocked_key_is_never_sent(qapp, session, pump):
    """INVARIANT 3 at the session boundary: the caller passes the reason it already computed, and
    this refuses without asking the question a second time from a second place."""
    session.open()
    pump(lambda: session.state == STREAMING, timeout=5.0)
    assert session.write("source.camera.width", 640,
                         block_reason="applies at next start - this run is recording") is False
    qapp.processEvents()
    assert session.made[0].sent == []


def test_nothing_is_sent_while_the_camera_is_closed(qapp, session):
    assert session.write("source.camera.gain_db", 2.0) is False


# =============================================================================================
# The frame path -- bounded by construction
# =============================================================================================
def test_the_frame_box_holds_one_frame_and_counts_the_rest(qapp):
    """Qt does not coalesce queued signals: measured, 300 frame-sized payloads emitted at a stalled
    GUI thread were ALL still queued after 0.5 s, holding every one of them. A run is watched for
    days; a one-minute stall would queue thousands. Dropping is the only thing that bounds it."""
    box = LatestFrame()
    for i in range(300):
        box.put(np.full((4, 4), i % 256, dtype=np.uint8))
    frame = box.take()
    assert frame is not None
    assert frame[0][0] == 299 % 256, "the box must keep the NEWEST frame, not the oldest"
    shown, dropped = box.stats
    assert shown == 1
    assert dropped == 299
    assert box.take() is None, "taking must empty the box"


def test_a_stalled_gui_does_not_accumulate_frames(qapp, pump):
    """The property that matters over days: memory stays flat and the picture goes stale, instead
    of memory growing and the app dying."""
    source = FakeSource()
    s = CameraSession(lambda: source)
    try:
        s.open()
        pump(lambda: s.state == STREAMING, timeout=5.0)
        pump(lambda: source.frames_read > 50, timeout=5.0)
        # The GUI never pulls. The box still holds exactly one frame.
        assert s.latest.take() is not None
        assert s.latest.take() is None
    finally:
        s.shutdown()


def test_frames_keep_being_READ_even_when_the_preview_is_not_shown_them(qapp, pump):
    """The SDK buffer fills if reads stop, so every frame is read; the throttle only decides which
    ones are OFFERED. That is why the caption can honestly say how many were not shown."""
    source = FakeSource()
    s = CameraSession(lambda: source)
    try:
        s.open()
        pump(lambda: s.state == STREAMING, timeout=5.0)
        pump(lambda: source.frames_read > 30, timeout=5.0)
        shown, dropped = s.latest.stats
        assert source.frames_read > shown + dropped - 1
        assert dropped > 0, "the throttle counted nothing as skipped"
    finally:
        s.shutdown()


def test_the_delivered_frame_rate_is_counted_not_asked_for(qapp, pump):
    """On this rig's camera the frame-rate limiter is documented to disengage mid-stream while its
    registers still read back CORRECT. So the only trustworthy number is the one counted from
    frames that actually arrived."""
    source = FakeSource()
    s = CameraSession(lambda: source)
    try:
        s.open()
        pump(lambda: s.state == STREAMING, timeout=5.0)
        pump(lambda: source.frames_read > 40, timeout=5.0)
        assert s.measured_fps > 0
    finally:
        s.shutdown()


def test_a_read_that_throws_ends_the_stream_and_says_so_rather_than_freezing(qapp, pump):
    """A frozen picture with a green status bar would be the worst possible outcome: it looks like
    a working camera showing a still scene."""
    class Failing(FakeSource):
        def read(self):
            raise RuntimeError("MV_CC_GetImageBuffer failed (ret=0x80000007)")

    s = CameraSession(lambda: Failing())
    try:
        s.open()
        assert pump(lambda: s.state == ERROR_OTHER, timeout=5.0), s.state
        assert "stopped delivering frames" in s.detail
    finally:
        s.shutdown()


# =============================================================================================
# The shared block rule, through the session's own source
# =============================================================================================
def test_the_session_exposes_its_source_only_so_the_shared_rule_can_ask_is_acquiring(qapp, session,
                                                                                     pump):
    from flygym_tracker.settings_model import start_only_block_reason

    session.open()
    pump(lambda: session.state == STREAMING, timeout=5.0)
    assert start_only_block_reason("source.camera.width", session.source, closable=True)
    assert start_only_block_reason("source.camera.frame_rate", session.source) is None

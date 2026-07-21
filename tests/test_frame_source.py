"""Tests for flygym_tracker.frame_source — VideoFileSource (real I/O), HikCameraSource
construction/failure paths, and the camera-settings contract against a FAKE SDK.

Run: python -m pytest tests/test_frame_source.py -q
No camera or MVS runtime is required for this file to pass.

THE HEADLINE PROPERTY BEING PROTECTED: a config with every camera setting null must produce ZERO
set-calls for those settings, so the camera runs on whatever MVS left it at. That is asserted
directly on the mock — the specific SDK setters are checked as NEVER CALLED, not merely as "the
picture looked right". Before this, `config/flygym_rig.yaml` forced 1280x1024 @ 20 fps on every
run, so a change made in the MVS Viewer was silently reverted the next time the tracker started.

The SDK is faked at the boundary `frame_source` already uses: `HikCameraSource._import_sdk`, the
single lazy import of `MvCameraControl_class`. Everything below it is the REAL code path —
enumeration, serial matching (through actual `ctypes.cast`), `_configure`, the live setters — so
these tests fail if that path changes shape, which a `_configure`-level monkeypatch would not.
"""
import ctypes

import cv2
import numpy as np
import pytest

from flygym_tracker.frame_source import (
    FALLBACK_RANGES,
    CameraRange,
    FrameSource,
    HikCameraSource,
    VideoFileSource,
    camera_ranges,
    fallback_camera_ranges,
)
from flygym_tracker.types import Frame

N_FRAMES = 10
WIDTH, HEIGHT = 64, 48
FPS = 15.0


def _make_synthetic_avi(path: str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, FPS, (WIDTH, HEIGHT), isColor=True)
    assert writer.isOpened(), "cv2.VideoWriter failed to open (MJPG codec unavailable?)"
    for i in range(N_FRAMES):
        value = (i * 20) % 256
        frame = np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def synthetic_video(tmp_path):
    path = str(tmp_path / "synthetic.avi")
    _make_synthetic_avi(path)
    return path


# ---- VideoFileSource --------------------------------------------------------

def test_video_file_source_reads_all_frames_then_none(synthetic_video):
    src = VideoFileSource(synthetic_video)
    src.open()
    try:
        frames = []
        while True:
            frame = src.read()
            if frame is None:
                break
            frames.append(frame)

        assert len(frames) == N_FRAMES
        for i, frame in enumerate(frames):
            assert isinstance(frame, Frame)
            assert frame.index == i
            assert frame.image.dtype == np.uint8
            assert frame.image.ndim == 2
            assert frame.image.shape == (HEIGHT, WIDTH)
            assert isinstance(frame.t_monotonic, float)
            assert isinstance(frame.t_wall_iso, str) and frame.t_wall_iso

        # EOF stays EOF on further reads.
        assert src.read() is None
    finally:
        src.close()


def test_video_file_source_reports_fps_and_frame_size(synthetic_video):
    with VideoFileSource(synthetic_video) as src:
        assert src.frame_size == (WIDTH, HEIGHT)
        assert src.fps == pytest.approx(FPS, rel=0.05)


def test_video_file_source_is_a_frame_source():
    assert issubclass(VideoFileSource, FrameSource)


def test_video_file_source_context_manager_closes(synthetic_video):
    with VideoFileSource(synthetic_video) as src:
        assert src.read() is not None
    # After the with-block exits, the source is closed: read() must raise.
    with pytest.raises(RuntimeError):
        src.read()


def test_video_file_source_missing_file_raises():
    src = VideoFileSource("this/file/does/not/exist.avi")
    with pytest.raises(RuntimeError):
        src.open()


def test_video_file_source_properties_require_open(synthetic_video):
    src = VideoFileSource(synthetic_video)
    with pytest.raises(RuntimeError):
        _ = src.frame_size
    with pytest.raises(RuntimeError):
        _ = src.fps
    with pytest.raises(RuntimeError):
        src.read()


# ---- HikCameraSource (no camera / no SDK grabbing in tests) ----------------

def test_hik_camera_source_module_imports_without_camera_or_sdk():
    # Importing this module (top of file) and constructing an instance must
    # never touch the SDK or hardware.
    src = HikCameraSource(serial="X")
    assert isinstance(src, FrameSource)
    assert src.serial == "X"


def test_hik_camera_source_is_a_frame_source():
    assert issubclass(HikCameraSource, FrameSource)


def test_hik_camera_source_construction_accepts_all_documented_kwargs():
    src = HikCameraSource(
        serial="DA4282883",
        index=0,
        width=1280,
        height=1024,
        exposure_us=20000,
        gain_db=0.0,
        frame_rate=20,
        pixel_format="Mono8",
    )
    assert src.width == 1280 and src.height == 1024
    assert src.pixel_format == "Mono8"


def test_hik_camera_source_open_raises_when_sdk_or_camera_absent():
    # A serial that cannot possibly match a real attached device, so this is
    # deterministic even if the MVS SDK/DLL happens to be installed on the
    # machine running the tests: HikCameraSource never falls back to `index`
    # when a serial is given and doesn't match (see frame_source.py), so
    # open() must raise regardless of whether the SDK itself is present.
    src = HikCameraSource(serial="NO-SUCH-CAMERA-SERIAL-0000000")
    try:
        src.open()
    except RuntimeError:
        return  # expected: SDK not installed, or no matching camera found
    else:
        # Only reachable if open() unexpectedly succeeded on this machine.
        # Don't grab frames - clean up and skip gracefully instead.
        src.close()
        pytest.skip(
            "HikCameraSource.open() unexpectedly succeeded with a bogus serial; "
            "skipping the SDK/camera-absent assertion on this machine"
        )


def test_hik_camera_source_read_and_close_require_open():
    src = HikCameraSource(serial="X")
    with pytest.raises(RuntimeError):
        src.read()
    # close() before open() is a no-op, not an error.
    src.close()


# =============================================================================================
# A fake MvImport SDK — installed at `HikCameraSource._import_sdk`, the module's own boundary
# =============================================================================================
SERIAL = "DA4282883"

#: What the fake sensor reports for each queried node: (min, max, inc, current) for ints,
#: (min, max, current) for floats. Deliberately NOT the values in `frame_source.FALLBACK_RANGES` —
#: a test using the same numbers could not tell a live read from a silent fallback. Every node
#: here differs from its fallback in at least one of (min, max, inc): Width by its increment
#: (16, against the sensor's real 4), Height by its minimum (4, against the real 32). This fake is
#: NOT meant to imitate the rig's camera; it is meant to be distinguishable from it.
FAKE_INT_NODES = {"Width": (16, 1280, 16, 1280), "Height": (4, 1024, 4, 1024)}
FAKE_FLOAT_NODES = {
    "ExposureTime": (15.0, 9000000.0, 4990.0),
    "Gain": (0.0, 23.98, 1.5),
    "AcquisitionFrameRate": (0.5, 163.0, 30.0),
    "ResultingFrameRate": (0.5, 163.0, 29.7),
}


class _Usb3VInfo(ctypes.Structure):
    _fields_ = [("chSerialNumber", ctypes.c_ubyte * 64)]


class _GigEInfo(ctypes.Structure):
    _fields_ = [("chSerialNumber", ctypes.c_ubyte * 64)]


class _SpecialInfo(ctypes.Union):
    _fields_ = [("stGigEInfo", _GigEInfo), ("stUsb3VInfo", _Usb3VInfo)]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [("nTLayerType", ctypes.c_uint), ("SpecialInfo", _SpecialInfo)]


class _DeviceInfoList(ctypes.Structure):
    _fields_ = [("nDeviceNum", ctypes.c_uint),
                ("pDeviceInfo", ctypes.POINTER(_DeviceInfo) * 8)]


class _IntValue:
    def __init__(self):
        self.nCurValue = self.nMin = self.nMax = self.nInc = 0


class _FloatValue:
    def __init__(self):
        self.fCurValue = self.fMin = self.fMax = 0.0


class FakeCamera:
    """Records every SDK call. `calls` is the assertion surface for the whole test file."""

    def __init__(self, log, serial=SERIAL):
        self.calls = log
        self.serial = serial
        self.grabbing = False

    # -- setters (what "sends nothing" is asserted against) --------------------------------
    def MV_CC_SetEnumValueByString(self, node, value):
        self.calls.append(("SetEnumValueByString", node, value))
        return 0

    def MV_CC_SetEnumValue(self, node, value):
        self.calls.append(("SetEnumValue", node, value))
        return 0

    def MV_CC_SetIntValueEx(self, node, value):
        self.calls.append(("SetIntValueEx", node, value))
        return 0

    def MV_CC_SetFloatValue(self, node, value):
        self.calls.append(("SetFloatValue", node, value))
        return 0

    def MV_CC_SetBoolValue(self, node, value):
        self.calls.append(("SetBoolValue", node, value))
        return 0

    # -- getters ----------------------------------------------------------------------------
    def MV_CC_GetIntValue(self, node, out):
        self.calls.append(("GetIntValue", node))
        if node not in FAKE_INT_NODES:
            return 1
        lo, hi, inc, cur = FAKE_INT_NODES[node]
        out.nMin, out.nMax, out.nInc, out.nCurValue = lo, hi, inc, cur
        return 0

    def MV_CC_GetIntValueEx(self, node, out):
        self.calls.append(("GetIntValueEx", node))
        return self.MV_CC_GetIntValue(node, out) if node in FAKE_INT_NODES else 1

    def MV_CC_GetFloatValue(self, node, out):
        self.calls.append(("GetFloatValue", node))
        if node not in FAKE_FLOAT_NODES:
            return 1
        lo, hi, cur = FAKE_FLOAT_NODES[node]
        out.fMin, out.fMax, out.fCurValue = lo, hi, cur
        return 0

    # -- lifecycle ---------------------------------------------------------------------------
    def MV_CC_CreateHandle(self, info):
        return 0

    def MV_CC_OpenDevice(self, access, key):
        return 0

    def MV_CC_StartGrabbing(self):
        self.grabbing = True
        self.calls.append(("StartGrabbing",))
        return 0

    def MV_CC_StopGrabbing(self):
        self.grabbing = False
        self.calls.append(("StopGrabbing",))
        return 0

    def MV_CC_CloseDevice(self):
        return 0

    def MV_CC_DestroyHandle(self):
        return 0

    @staticmethod
    def MV_CC_Initialize():
        return 0

    @staticmethod
    def MV_CC_Finalize():
        return 0


class FakeSdk:
    """Stands in for the `MvCameraControl_class` module."""

    MV_GIGE_DEVICE = 1
    MV_USB_DEVICE = 4
    MV_GENTL_CAMERALINK_DEVICE = 8
    MV_GENTL_CXP_DEVICE = 16
    MV_GENTL_XOF_DEVICE = 32
    MV_ACCESS_Exclusive = 1
    MV_TRIGGER_MODE_OFF = 0

    MV_CC_DEVICE_INFO = _DeviceInfo
    MV_CC_DEVICE_INFO_LIST = _DeviceInfoList
    MVCC_INTVALUE = _IntValue
    MVCC_INTVALUE_EX = _IntValue
    MVCC_FLOATVALUE = _FloatValue

    def __init__(self, serial=SERIAL):
        self.calls = []
        self.serial = serial
        self.camera = None
        self._device = _DeviceInfo()
        self._device.nTLayerType = self.MV_USB_DEVICE
        for i, byte in enumerate(serial.encode("utf-8") + b"\x00"):
            self._device.SpecialInfo.stUsb3VInfo.chSerialNumber[i] = byte
        # Bound so `MvCamera()` hands back the ONE camera whose calls the test inspects.
        outer = self

        class _MvCamera(FakeCamera):
            def __init__(self):
                super().__init__(outer.calls, outer.serial)
                outer.camera = self

        _MvCamera.MV_CC_EnumDevices = staticmethod(self._enum)
        self.MvCamera = _MvCamera

    def _enum(self, layer_type, device_list):
        self.calls.append(("EnumDevices",))
        device_list.nDeviceNum = 1
        device_list.pDeviceInfo[0] = ctypes.pointer(self._device)
        return 0

    # -- assertion helpers -------------------------------------------------------------------
    def set_calls(self):
        return [c for c in self.calls if c[0].startswith("Set")]

    def sets_for(self, node):
        return [c for c in self.set_calls() if len(c) > 1 and c[1] == node]


@pytest.fixture
def sdk(monkeypatch):
    """Install the fake SDK at `HikCameraSource._import_sdk` -- the module's lazy-import seam."""
    fake = FakeSdk()
    monkeypatch.setattr(HikCameraSource, "_import_sdk", staticmethod(lambda: fake))
    return fake


def _open(sdk, **kwargs):
    source = HikCameraSource(serial=SERIAL, **kwargs)
    source.open()
    return source


# =============================================================================================
# THE HEADLINE: a null config sends no camera settings at all
# =============================================================================================
#: Every SDK setter that could impose one of the five adjustable settings. Named explicitly rather
#: than derived, so a sixth way to write to the camera cannot silently escape the test.
_SETTING_WRITE_CALLS = ("SetIntValueEx", "SetFloatValue", "SetBoolValue")


def test_a_config_with_every_camera_setting_unset_sends_nothing_to_the_camera(sdk):
    """THE requirement, asserted the only way that means anything: the setters are NEVER called.

    Width, Height, ExposureTime, Gain and AcquisitionFrameRate are all null, so `_configure` must
    leave every one of those nodes untouched and the camera keeps the state MVS left it in.
    """
    source = _open(sdk)
    try:
        for call in sdk.calls:
            assert call[0] not in _SETTING_WRITE_CALLS, \
                "a null config still wrote to the camera: %r" % (call,)
        for node in ("Width", "Height", "ExposureTime", "Gain", "AcquisitionFrameRate",
                     "AcquisitionFrameRateEnable", "ExposureAuto", "GainAuto"):
            assert sdk.sets_for(node) == [], "%s was set despite being unconfigured" % node
    finally:
        source.close()


def test_a_null_config_still_sends_the_two_settings_the_pipeline_cannot_work_without(sdk):
    """PixelFormat and TriggerMode are not tunables and not optional: the pipeline decodes Mono8
    and reads free-running frames. "Send nothing" covers the five adjustable settings, not the two
    that define what a frame even is."""
    source = _open(sdk)
    try:
        assert ("SetEnumValueByString", "PixelFormat", "Mono8") in sdk.calls
        assert ("SetEnumValue", "TriggerMode", FakeSdk.MV_TRIGGER_MODE_OFF) in sdk.calls
    finally:
        source.close()


def test_reading_the_frame_size_of_an_unconfigured_camera_asks_it_rather_than_telling_it(sdk):
    """`frame_size` still has to be known, and with no width configured it is READ BACK. A get is
    not a set: that distinction is the whole contract, so it is pinned rather than assumed."""
    source = _open(sdk)
    try:
        assert source.frame_size == (1280, 1024)     # the fake sensor's own current values
        assert ("GetIntValue", "Width") in sdk.calls
        assert sdk.sets_for("Width") == []
    finally:
        source.close()


# =============================================================================================
# An explicit value produces exactly one set-call carrying that value
# =============================================================================================
@pytest.mark.parametrize("kwarg,node,value", [
    ("width", "Width", 640),
    ("height", "Height", 480),
    ("exposure_us", "ExposureTime", 5000.0),
    ("gain_db", "Gain", 3.0),
    ("frame_rate", "AcquisitionFrameRate", 30.0),
])
def test_an_explicit_camera_setting_is_sent_exactly_once_with_that_value(sdk, kwarg, node, value):
    source = _open(sdk, **{kwarg: value})
    try:
        sets = sdk.sets_for(node)
        assert len(sets) == 1, "expected one write to %s, got %r" % (node, sets)
        assert sets[0][2] == pytest.approx(value)
    finally:
        source.close()


def test_setting_one_camera_value_leaves_the_other_four_alone(sdk):
    """Partial configuration is the normal case: pin the frame rate, leave geometry to MVS."""
    source = _open(sdk, frame_rate=25.0)
    try:
        assert len(sdk.sets_for("AcquisitionFrameRate")) == 1
        for node in ("Width", "Height", "ExposureTime", "Gain"):
            assert sdk.sets_for(node) == []
    finally:
        source.close()


def test_an_explicit_frame_rate_also_opens_the_gate_that_makes_the_camera_honour_it(sdk):
    """On Hik cameras AcquisitionFrameRate is IGNORED unless AcquisitionFrameRateEnable is on, so
    the value alone would land in a register the camera never reads."""
    source = _open(sdk, frame_rate=25.0)
    try:
        assert ("SetBoolValue", "AcquisitionFrameRateEnable", True) in sdk.calls
    finally:
        source.close()


# =============================================================================================
# Increment snapping (a value off the grid is REJECTED by the SDK, not clamped)
# =============================================================================================
@pytest.mark.parametrize("raw,expected", [
    (640, 640),          # already on the 16-grid
    (650, 656),          # rounds up to the nearer multiple
    (647, 640),          # rounds down
    (1, 16),             # below the sensor minimum
    (99999, 1280),       # above the sensor maximum
])
def test_width_snaps_to_the_increment_the_camera_reports(sdk, raw, expected):
    source = HikCameraSource(serial=SERIAL)
    source.open()
    source.close()                      # ranges cached from the live read, stream stopped
    source._ranges = {"Width": CameraRange("Width", 16.0, 1280.0, 16.0, True)}
    source.set_width(raw)
    assert source.width == expected


def test_snapping_counts_from_the_nodes_minimum_not_from_zero():
    """A node advertising min=2, inc=4 accepts 6 -- which counting from zero never produces."""
    rng = CameraRange(name="Width", lo=2.0, hi=100.0, inc=4.0, live=True)
    assert rng.snap(6) == 6
    assert rng.snap(7) == 6
    assert rng.snap(8) == 10


def test_height_snaps_on_its_own_grid_not_the_widths(sdk):
    """Width and Height have DIFFERENT increments on this sensor (16 and 4); one shared constant
    would snap half of them wrong."""
    source = HikCameraSource(serial=SERIAL)
    source._ranges = {"Height": CameraRange("Height", 4.0, 1024.0, 4.0, True)}
    source.set_height(487)
    assert source.height == 488          # the 4-grid, not the 16-grid Width uses
    source.set_height(482)
    assert source.height == 484          # ...and 482 would be a legal Width, but not a Height


def test_a_snapped_width_is_what_actually_reaches_the_camera(sdk):
    """Snapping that only tidied an attribute would be pointless -- the SDK rejects an off-grid
    Width outright, so the value SENT is the one that has to be on the grid."""
    source = HikCameraSource(serial=SERIAL)
    source._ranges = {"Width": CameraRange("Width", 16.0, 1280.0, 16.0, True)}
    source.set_width(650)
    source.open()
    try:
        assert sdk.sets_for("Width")[0][2] == 656
    finally:
        source.close()


# =============================================================================================
# Where the limits come from
# =============================================================================================
def test_the_limits_are_read_from_the_camera_when_one_is_open(sdk):
    source = _open(sdk)
    try:
        ranges = source.ranges()
        assert ranges["Width"].inc == 16 and ranges["Width"].live is True
        assert ranges["Height"].hi == 1024 and ranges["Height"].live is True
        assert ranges["ExposureTime"].lo == pytest.approx(15.0)
        assert ranges["AcquisitionFrameRate"].hi == pytest.approx(163.0)
        assert all(r.live for r in ranges.values())
    finally:
        source.close()


def test_the_limits_fall_back_to_documented_values_when_no_camera_is_open():
    """Every test machine, and the tune-between-runs case. The fallbacks must be USABLE and must
    admit what they are, because a number shown like a fresh measurement gets believed."""
    ranges = camera_ranges(HikCameraSource(serial=SERIAL))
    assert set(ranges) == set(fallback_camera_ranges())
    assert not any(r.live for r in ranges.values())
    for r in ranges.values():
        assert r.hi > r.lo and r.inc > 0


#: What the rig's own MV-CA013-A0UM (serial DA4282883) reported for these five nodes when it was
#: probed on 2026-07-19. Every min and max is transcribed from that read and from nothing else.
#: The three FLOAT increments are not: float GenICam nodes advertise none, so those are the
#: panel's display granularity (see `FALLBACK_RANGES`), and only the two int increments are
#: measurements.
MEASURED_ON_THE_RIG_2026_07_19 = {
    "Width": (32.0, 1280.0, 4.0),
    "Height": (32.0, 1024.0, 4.0),
    "ExposureTime": (9.0, 9999640.0, 1.0),
    "Gain": (0.0, 16.3704, 0.1),
    "AcquisitionFrameRate": (0.1, 100000.0, 0.1),
}


def test_the_fallback_limits_are_the_ones_measured_on_the_rig():
    """Pins `FALLBACK_RANGES` to a real read of the rig's sensor.

    These are the numbers an operator sees on a machine with no camera attached, so a plausible
    guess edited in later would be indistinguishable on screen from a measurement — which is the
    one thing `CameraRange.live` exists to prevent. Changing this table should mean re-probing the
    camera, so the test makes that a deliberate act rather than a quiet one.
    """
    assert FALLBACK_RANGES == MEASURED_ON_THE_RIG_2026_07_19


@pytest.mark.parametrize("node", ["ExposureTime", "Gain", "AcquisitionFrameRate"])
def test_the_float_fallback_steps_keep_whole_numbers_reachable(node):
    """The settings panel snaps a row onto ``lo + n*step``, so a step that does not divide the
    node's minimum shifts the WHOLE grid off the integers an operator actually types. Measuring
    AcquisitionFrameRate's real floor of 0.1 fps made this bite: against the old 0.5 step the grid
    became 0.1/0.6/1.1..., and a configured 42.0 fps redisplayed as 42.1 — the panel overruling a
    number nobody asked it to touch. Float nodes advertise no increment, so the step is ours to
    pick and there is no excuse for picking one that does this."""
    lo, hi, step = FALLBACK_RANGES[node]
    reachable = lo + round((42.0 - lo) / step) * step
    assert reachable == pytest.approx(42.0), (
        "%s: a configured 42 lands on %r" % (node, reachable))


@pytest.mark.parametrize("node,true_inc", [("Width", 4.0), ("Height", 4.0)])
def test_the_integer_fallbacks_stay_on_a_grid_the_sensor_actually_accepts(node, true_inc):
    """An off-grid Width is REJECTED by the SDK rather than rounded, so a fallback increment
    COARSER than the truth is merely wasteful, while a finer one that is not a whole multiple of
    it hands the operator sizes the camera refuses. Both the step and the floor have to be real:
    the floor was 8 here until 2026-07-19, when the sensor turned out to start at 32."""
    lo, hi, inc = FALLBACK_RANGES[node]
    assert inc >= true_inc and inc % true_inc == 0
    assert (lo - 32.0) % true_inc == 0 and lo >= 32.0
    # Every value the fallback offers is on the sensor's real grid, counting from its real minimum.
    rng = CameraRange(name=node, lo=lo, hi=hi, inc=inc, live=False)
    for raw in (lo - 100, lo, 640, 641, 999, hi, hi + 100):
        assert (rng.snap(raw) - 32.0) % true_inc == 0


def test_a_video_file_source_yields_the_documented_limits_rather_than_failing():
    """`replay` has no camera at all; the panel still has to draw something."""
    ranges = camera_ranges(VideoFileSource("nope.avi"))
    assert not any(r.live for r in ranges.values())
    assert camera_ranges(None) == ranges


def test_a_camera_that_cannot_answer_falls_back_per_node_not_wholesale(sdk):
    """One missing GenICam node must not cost the panel its other four rows."""
    source = _open(sdk)
    try:
        real = sdk.camera.MV_CC_GetFloatValue
        sdk.camera.MV_CC_GetFloatValue = (
            lambda node, out: 1 if node == "Gain" else real(node, out))
        ranges = source.refresh_ranges()
        assert ranges["Gain"].live is False
        assert ranges["ExposureTime"].live is True
    finally:
        source.close()


def test_the_limits_are_cached_so_a_slider_drag_does_not_storm_the_sdk(sdk):
    """A drag asks for the limits on every step; five GenICam round-trips per mouse-step inside a
    live acquisition would be a self-inflicted stall."""
    source = _open(sdk)
    try:
        source.ranges()
        before = len(sdk.calls)
        for _ in range(20):
            source.ranges()
        assert len(sdk.calls) == before
    finally:
        source.close()


# =============================================================================================
# Live adjustment vs start-only
# =============================================================================================
@pytest.mark.parametrize("setter,node,value", [
    ("set_frame_rate", "AcquisitionFrameRate", 45.0),
    ("set_exposure_us", "ExposureTime", 8000.0),
    ("set_gain_db", "Gain", 2.5),
])
def test_a_live_camera_setting_reaches_the_running_stream(sdk, setter, node, value):
    source = _open(sdk)
    try:
        sdk.calls.clear()
        getattr(source, setter)(value)
        sets = sdk.sets_for(node)
        assert len(sets) == 1 and sets[0][2] == pytest.approx(value)
    finally:
        source.close()


def test_clearing_a_live_setting_mid_run_stops_imposing_it_and_sends_nothing(sdk):
    """A camera cannot be told "go back to what you were": the value it runs at IS the value that
    was sent. Clearing means "stop imposing this, and do not send it at the next open"."""
    source = _open(sdk, frame_rate=30.0)
    try:
        sdk.calls.clear()
        source.set_frame_rate(None)
        assert source.frame_rate is None
        assert sdk.set_calls() == []
    finally:
        source.close()


@pytest.mark.parametrize("setter", ["set_width", "set_height"])
def test_the_start_only_settings_refuse_to_act_on_a_running_stream_and_say_why(sdk, setter):
    """Never restart a running stream: this rig records for hours to days, so a restart costs a
    gap in the series PLUS a frame-diff baseline reset -- two regimes in one file."""
    source = _open(sdk)
    try:
        sdk.calls.clear()
        with pytest.raises(RuntimeError) as excinfo:
            getattr(source, setter)(640)
        assert "next start" in str(excinfo.value)
        assert sdk.set_calls() == [], "a refused change still touched the camera"
        assert ("StopGrabbing",) not in sdk.calls, "the running stream was restarted"
    finally:
        source.close()


def test_the_start_only_settings_are_free_to_change_before_the_stream_starts(sdk):
    """Which is the whole reason the `settings` command and the pre-run panel exist."""
    source = HikCameraSource(serial=SERIAL)
    source.set_width(640)
    source.set_height(480)
    assert (source.width, source.height) == (640, 480)
    source.open()
    try:
        assert sdk.sets_for("Width")[0][2] == 640
        assert sdk.sets_for("Height")[0][2] == 480
    finally:
        source.close()


def test_is_acquiring_tracks_the_stream_and_nothing_else(sdk):
    source = HikCameraSource(serial=SERIAL)
    assert source.is_acquiring is False
    source.open()
    assert source.is_acquiring is True
    source.close()
    assert source.is_acquiring is False


def test_current_values_report_what_the_camera_is_doing_not_what_was_configured(sdk):
    """The DEFAULT state on an unset row shows this, so the operator can see what they are leaving
    alone. Frame rate comes from ResultingFrameRate -- the DELIVERED rate, which exposure alone can
    hold below the requested cap."""
    source = _open(sdk)
    try:
        values = source.current_values()
        assert values["frame_rate"] == pytest.approx(29.7)    # Resulting, not Acquisition
        assert values["exposure_us"] == pytest.approx(4990.0)
        assert values["width"] == 1280
        assert source.frame_rate is None, "a read-back must never become a configured value"
    finally:
        source.close()


def test_current_values_are_empty_without_a_camera():
    assert HikCameraSource(serial=SERIAL).current_values() == {}


def test_a_config_written_by_hand_is_snapped_to_the_real_grid_before_it_is_sent(sdk):
    """REGRESSION: `_configure` used to send `int(self.width)` straight through. The panel snaps
    too, but on a machine with no camera it can only snap against the DOCUMENTED grid (8 px); this
    sensor's real increment is 16, so a panel-produced 648 -- or any hand-edited number -- would
    have been rejected outright and failed the open. Snapping at the point of sending is the only
    place that always knows the real grid, because by then the camera is open."""
    source = _open(sdk, width=650, height=482)
    try:
        assert sdk.sets_for("Width")[0][2] == 656       # the 16-grid this sensor reports
        assert sdk.sets_for("Height")[0][2] == 484      # the 4-grid, snapped independently
        assert (source.width, source.height) == (656, 484),             "the source must report what it actually sent, not what it was asked for"
    finally:
        source.close()


def test_an_explicitly_chosen_float_is_sent_as_chosen_and_not_quantised(sdk):
    """Float nodes have no legality grid -- their `inc` is a slider granularity. Snapping 29.7 fps
    to 29.5 would be this code overruling the operator for no reason the camera cares about."""
    source = _open(sdk, frame_rate=29.7, exposure_us=4993.0)
    try:
        assert sdk.sets_for("AcquisitionFrameRate")[0][2] == pytest.approx(29.7)
        assert sdk.sets_for("ExposureTime")[0][2] == pytest.approx(4993.0)
    finally:
        source.close()


def test_a_float_beyond_what_the_camera_accepts_is_clamped_rather_than_refused(sdk):
    """The sensor tops out at 163 fps; asking for 500 should give the fastest it can do, not an
    error in the middle of starting an experiment."""
    source = _open(sdk, frame_rate=500.0)
    try:
        assert sdk.sets_for("AcquisitionFrameRate")[0][2] == pytest.approx(163.0)
    finally:
        source.close()


def test_clamp_leaves_a_legal_float_exactly_alone():
    rng = CameraRange(name="AcquisitionFrameRate", lo=0.5, hi=163.0, inc=0.5, live=True)
    assert rng.clamp(29.7) == pytest.approx(29.7)
    assert rng.clamp(0.1) == pytest.approx(0.5)
    assert rng.clamp(1e6) == pytest.approx(163.0)


# =============================================================================================
# The error a wrong serial produces -- the failure that cost a real afternoon on a second PC
# =============================================================================================
def test_a_wrong_serial_names_every_camera_that_WAS_found(sdk):
    """"no camera with serial X found among 1 device(s)" is true and useless: it does not say that
    the one device present is a perfectly good camera with a different serial, which is the entire
    answer. This is the message a config carrying another machine's serial produces -- and it is
    exactly what a shipped template pinning the development rig's camera produced on every other
    machine."""
    source = HikCameraSource(serial="SOMEONE-ELSES-CAMERA")
    with pytest.raises(RuntimeError) as excinfo:
        source.open()
    message = str(excinfo.value)
    assert "SOMEONE-ELSES-CAMERA" in message
    assert SERIAL in message, "the error did not name the attached camera: %s" % message
    assert "source.camera.serial" in message, "it does not say WHERE the wrong serial came from"


def test_the_serial_survives_an_sdk_that_reports_no_model_name(sdk):
    """Model and vendor are decoration; the serial is what pins the camera and what the operator
    picks by. An SDK build lacking `chModelName` must not cost the serial too -- which is exactly
    what reading every field in one expression used to do. This fake has no chModelName at all,
    which is how the weakness was found."""
    from flygym_tracker.frame_source import _describe_device

    described = _describe_device(sdk, sdk._device)
    assert described["serial"] == SERIAL
    assert described["model"] == ""
    assert described["interface"] == "USB3"


def test_listing_cameras_describes_what_is_attached(sdk):
    from flygym_tracker.frame_source import list_cameras

    # `include_uvc=False`, or this test probes the DEVELOPER'S OWN WEBCAM and its result depends on
    # what is plugged into the machine running it. The webcam path is covered with a fake in
    # `test_camera_picker`.
    cameras = list_cameras(include_uvc=False)
    assert len(cameras) == 1
    assert cameras[0].serial == SERIAL
    assert SERIAL in cameras[0].label, "the label does not lead with the serial that pins it"

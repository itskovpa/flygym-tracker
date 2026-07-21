"""Frame sources for flygym_tracker: offline video replay and live HikRobot capture.

See DESIGN.md section 3 (architecture: "frame source") and section 4
(`frame_source.py` responsibility). Both sources yield `flygym_tracker.types.Frame`
(HxW uint8 grayscale + index + timestamps) via a shared `FrameSource` interface so
`pipeline.py` can be built against either without caring which one it has.

IMPORTANT: importing this module must never touch the camera or the HikRobot SDK.
`HikCameraSource.__init__` only stores configuration; the vendored `MvImport` SDK
is imported lazily inside `open()` (see `HikCameraSource._import_sdk`).

UNSET MEANS UNSET. Every camera setting is `Optional`, and `None` means "send
nothing for this node, so the camera keeps whatever MVS last left it at". That is
not a convenience — it is the contract the rig owner asked for, and it only holds
if `_configure` makes NO SDK call for a `None` field. A config that writes
`width: 1280` because that happens to be the current value would silently take
ownership of the sensor geometry, and the next operator who changed it in MVS
would find their change reverted by a tracker they never told to touch it.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

from flygym_tracker.types import Frame

#: Default install location of the HikRobot MvImport Python SDK on this rig.
#: Override with the MVS_PYTHON_SDK env var on a machine with a different install path.
DEFAULT_MVS_SDK_PATH = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"

#: GenICam node names for the five settings this software will ever impose, keyed by the
#: `HikCameraSource` attribute (and, one level up, the `source.camera.*` config key) that carries
#: them. Spelled out once here so the panel, the router and `_configure` cannot drift apart.
CAMERA_NODES = {
    "width": "Width",
    "height": "Height",
    "exposure_us": "ExposureTime",
    "gain_db": "Gain",
    "frame_rate": "AcquisitionFrameRate",
}

#: Settings that cannot be changed while the stream is running: the sensor's region of interest is
#: fixed at StartGrabbing time, so honouring a mid-run change would mean stopping and restarting
#: acquisition. This rig records continuously for hours to days, and a restart costs both a gap in
#: the series AND a reset of the frame-diff baseline (DESIGN.md §5.3) — i.e. it silently splits one
#: experiment into two incomparable measurement regimes. Never worth it for a geometry tweak.
START_ONLY_ATTRS = ("width", "height")

#: Fallback limits for the MV-CA013-A0UM, used ONLY when no camera is open (every test machine,
#: and the tune-between-runs case). MEASURED on the rig's own sensor (serial DA4282883) on
#: 2026-07-19 by reading these five GenICam nodes over `--probe-camera`, replacing the documented
#: guesses that were here before. They are still flagged `live=False` when shown — see
#: `CameraRange.live` and the "not live" note the settings panel prints under its Camera group —
#: because they describe THE rig's camera rather than whatever camera is actually attached to the
#: machine drawing the panel.
#:
#: The two ints carry the sensor's real increment of 4 (it was a conservative 8 before). The old
#: Width/Height floor of 8 was not merely conservative but WRONG in the unsafe direction: the
#: sensor's true minimum is 32, so the panel was offering an operator sizes the SDK would reject.
#: Nothing shipped a bad Width because `_configure` re-snaps against the live range at send time,
#: but the number on screen was not a real one.
#:
#: The float maxima are the nodes' absolute bounds, not what the rig can achieve: ExposureTime's
#: ~10 s ceiling and AcquisitionFrameRate's 100 kfps are both cut down in practice by each other
#: and by the ROI (this sensor delivers ~88 fps at full frame). Floats advertise no increment, so
#: the third entry for those three is the panel's display granularity, NOT a measurement and not a
#: legality constraint — `_query_float_range` reads the live min/max and keeps the step from here.
#: AcquisitionFrameRate's step is 0.1 rather than 0.5 for a reason worth keeping: the panel snaps a
#: row onto `lo + n*step`, so a 0.5 step over the measured floor of 0.1 would put the grid at
#: 0.1/0.6/1.1... and quietly redisplay an operator's configured 42.0 fps as 42.1. A step that
#: divides the floor keeps whole numbers reachable.
#:
#: The SDK hands these back as float32, so two entries are the readable rounding of what it said:
#: Gain's max read 16.370399475097656 and the frame-rate floor 0.10000000149011612. Rounding the
#: gain ceiling UP by 5e-7 dB is safe in a way rounding a Width would not be — floats are clamped
#: by the camera, never rejected, and `_configure` clamps against the live range whenever one is
#: actually attached.
FALLBACK_RANGES = {
    "Width": (32.0, 1280.0, 4.0),
    "Height": (32.0, 1024.0, 4.0),
    "ExposureTime": (9.0, 9999640.0, 1.0),
    "Gain": (0.0, 16.3704, 0.1),
    "AcquisitionFrameRate": (0.1, 100000.0, 0.1),
}


@dataclass(frozen=True)
class CameraRange:
    """The legal values for one camera node, and whether they came from the camera itself.

    `live=False` means these are `FALLBACK_RANGES` entries: measured on the rig's camera, but not
    read from whatever camera this machine has (if any). The distinction is shown to the operator
    rather than smoothed over — a limit that is merely inherited must not look like one this
    sensor just reported, because the two disagree the moment a different camera is plugged in.
    """

    name: str
    lo: float
    hi: float
    inc: float
    live: bool = False

    def clamp(self, value) -> float:
        """`value` held inside ``[lo, hi]``, with no quantisation.

        This is what the FLOAT nodes get. Their `inc` here is a slider granularity, not a legality
        constraint, so snapping an explicitly-chosen 29.7 fps to 29.5 would be this code quietly
        overruling the operator for no reason the camera cares about.
        """
        return min(self.hi, max(self.lo, float(value)))

    def snap(self, value) -> float:
        """Clamp `value` into ``[lo, hi]`` and snap it onto the node's increment grid.

        This is what the INTEGER nodes get, and it is not cosmetic: an off-grid Width is REJECTED
        by the SDK, not rounded, so an unsnapped value is a failed open rather than a slightly
        wrong picture.

        Snapping is measured FROM `lo`, not from zero, because that is the grid the SDK accepts:
        a Width node advertising ``min=8, inc=8`` rejects 12 but accepts 16, and a node advertising
        ``min=2, inc=4`` accepts 6 — which counting from zero would never produce.
        """
        v = self.clamp(value)
        if v in (self.lo, self.hi):
            return v
        inc = float(self.inc)
        if inc <= 0:
            return v
        snapped = self.lo + round((v - self.lo) / inc) * inc
        return min(self.hi, max(self.lo, snapped))


def fallback_camera_ranges() -> Dict[str, CameraRange]:
    """`FALLBACK_RANGES` as `CameraRange`s, every one flagged ``live=False``."""
    return {
        name: CameraRange(name=name, lo=lo, hi=hi, inc=inc, live=False)
        for name, (lo, hi, inc) in FALLBACK_RANGES.items()
    }


def camera_ranges(source=None) -> Dict[str, CameraRange]:
    """The limits to offer for each camera node, from `source` if it can supply them.

    Deliberately duck-typed and total: anything without a usable `ranges()` (no source at all, a
    `VideoFileSource`, a camera that is not open) yields the documented fallbacks rather than an
    error. The settings panel has to render SOMETHING on a machine with no camera attached, and
    "no numbers at all" would make the whole Camera group unusable exactly where it is most
    needed — tuning between runs.
    """
    probe = getattr(source, "ranges", None)
    if callable(probe):
        try:
            return probe()
        except Exception:
            pass
    return fallback_camera_ranges()


class FrameSource(ABC):
    """Common interface for offline (video file) and live (camera) frame sources."""

    @abstractmethod
    def open(self) -> None:
        """Acquire resources (open the file/device). Must be called before read()."""
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next Frame, or None at end of stream (video EOF)."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources. Safe to call more than once."""
        raise NotImplementedError

    @property
    @abstractmethod
    def fps(self) -> float:
        """Nominal frames per second of the source."""
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """(width, height) in pixels."""
        raise NotImplementedError

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


class VideoFileSource(FrameSource):
    """Offline replay from a recorded video file (dev/replay path per DESIGN.md S3).

    cv2 IS IMPORTED INSIDE THE METHODS, not at module scope, and that is not style. This module is
    what the settings layer reads camera limits from (`camera_ranges`, `START_ONLY_ATTRS`), so a
    module-level `import cv2` would put OpenCV -- the flakiest dependency in this stack, with a
    headless build that silently shadows the GUI one -- underneath a Qt settings window that draws
    nothing with it. `tests/test_frame_source.py` asserts that importing this module with cv2
    blocked still works, so a settings app can open on a machine whose OpenCV install is broken,
    which is exactly the machine whose operator most needs to look at their settings. Only VIDEO
    REPLAY needs cv2, and only when it actually runs.
    """

    def __init__(self, path: str):
        self.path = path
        self._cap = None
        self._next_index = 0
        self._fps = 0.0
        self._frame_size = (0, 0)

    def open(self) -> None:
        import cv2

        if self._cap is not None:
            return
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video file: {self.path!r}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (width, height)
        self._next_index = 0

    def read(self) -> Optional[Frame]:
        import cv2

        if self._cap is None:
            raise RuntimeError("VideoFileSource is not open; call open() first")
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        frame = Frame(
            image=image,
            index=self._next_index,
            t_monotonic=time.monotonic(),
            t_wall_iso=datetime.now().isoformat(),
        )
        self._next_index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        if self._cap is None:
            raise RuntimeError("VideoFileSource is not open; call open() first")
        return self._fps

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._cap is None:
            raise RuntimeError("VideoFileSource is not open; call open() first")
        return self._frame_size


#: The two kinds of camera this software can open, and they are NOT interchangeable.
#:
#: `hikrobot`  the rig's machine-vision camera, reached through the MVS SDK. Mono8, global
#:             shutter, exposure/gain/frame-rate all controllable. What an experiment runs on.
#: `uvc`       an ordinary webcam, reached through OpenCV. Listed and selectable because it is the
#:             only way to exercise the whole program on a machine with no rig attached -- and
#:             because a picker that showed nothing on a laptop with a working built-in camera
#:             looks broken. It is NOT suitable for the assay: see `CameraInfo.suitable`.
KIND_HIKROBOT = "hikrobot"
KIND_UVC = "uvc"

#: How many webcam indices to probe. Probing OPENS each device briefly (there is no handle-free
#: enumeration for UVC the way there is for GenICam), so this is deliberately small: a laptop has
#: one or two, and every additional index costs a real open attempt.
MAX_UVC_PROBE = 4


class CameraInfo:
    """One attached camera, as `list_cameras` reports it. Plain data, never a live handle."""

    def __init__(self, index: int, serial: Optional[str], model: str = "", vendor: str = "",
                 interface: str = "", kind: str = KIND_HIKROBOT) -> None:
        self.index = int(index)
        self.serial = serial
        self.model = model or ""
        self.vendor = vendor or ""
        self.interface = interface or ""
        self.kind = kind

    @property
    def suitable(self) -> bool:
        """Whether an EXPERIMENT can run on this camera.

        A webcam is rolling-shutter, auto-exposing, auto-white-balancing and colour. On a drum
        rotating past an IR backlight, auto-exposure alone re-levels the whole image between frames
        -- which is precisely the signal the activity measurement reads. It will produce numbers.
        They will be a measurement of the camera's gain control, not of the flies.
        """
        return self.kind == KIND_HIKROBOT

    @property
    def label(self) -> str:
        """What to show in a picker. Serial FIRST for a rig camera, because that is what pins it."""
        if self.kind == KIND_UVC:
            # The index is ALWAYS shown, named or not: it is what actually gets opened, and the
            # name -- when Windows offers a trustworthy one -- is only a convenience on top of it.
            head = ("%s (webcam %d)" % (self.model, self.index)) if self.model \
                else "webcam %d" % self.index
            return "%s  -  NOT for experiments" % head
        name = self.model or "camera"
        if self.serial:
            return "%s  -  %s" % (self.serial, name)
        return "%s  -  no serial reported (index %d)" % (name, self.index)

    #: How this camera is written into the config, and read back by `_camera_source_from_config`.
    @property
    def config_id(self) -> str:
        return "uvc:%d" % self.index if self.kind == KIND_UVC else (self.serial or "")

    def __repr__(self) -> str:
        return "CameraInfo(kind=%r, index=%d, serial=%r, model=%r)" % (
            self.kind, self.index, self.serial, self.model)


def list_cameras(include_uvc: bool = True) -> list:
    """Every camera on the machine: the rig's, and -- if asked -- ordinary webcams too.

    TWO SEPARATE ENUMERATIONS, because they are two unrelated device stacks and the MVS SDK is
    blind to webcams. Neither failure hides the other: a machine with no MVS installed still lists
    its built-in camera, and a rig with MVS working still lists nothing extra if OpenCV is broken.
    That matters because the question being asked is usually "is ANY camera visible to this
    program?", and answering it with half the picture is how the wrong thing gets blamed.

    RAISES ONLY IF NOTHING COULD BE ENUMERATED AT ALL. If the MVS SDK is missing but webcams were
    found, that is a successful answer to "what cameras are there" -- and the caller can still see
    from the result that no rig camera is among them.
    """
    hik, hik_error = [], None
    try:
        hik = _list_hikrobot_cameras()
    except Exception as exc:
        hik_error = exc
    uvc = _list_uvc_cameras() if include_uvc else []
    if hik_error is not None and not uvc:
        raise hik_error
    return hik + uvc


def _list_hikrobot_cameras() -> list:
    """Every camera the MVS SDK can see, WITHOUT opening any of them.

    THE POINT IS THAT IT DOES NOT OPEN THEM. USB3 Vision access is exclusive, so a "list the
    cameras" that opened each one in turn would fight the running experiment for the device -- and
    on this rig the thing most likely to be asked "what cameras are there?" is a machine where a
    camera is already streaming. Enumeration is a separate SDK call and takes no handle.

    Raises RuntimeError if the MVS SDK is not installed -- the caller can then say THAT, which is a
    different problem from "no cameras found" and has a different fix.
    """
    mv = HikCameraSource._import_sdk()
    try:
        mv.MvCamera.MV_CC_Initialize()
    except Exception:
        pass                      # not present on all SDK builds; enumeration works either way

    device_list = mv.MV_CC_DEVICE_INFO_LIST()
    layer_type = (mv.MV_GIGE_DEVICE | mv.MV_USB_DEVICE | mv.MV_GENTL_CAMERALINK_DEVICE
                  | mv.MV_GENTL_CXP_DEVICE | mv.MV_GENTL_XOF_DEVICE)
    ret = mv.MvCamera.MV_CC_EnumDevices(layer_type, device_list)
    if ret != 0:
        raise RuntimeError("MV_CC_EnumDevices failed (ret=0x%x)" % ret)

    cameras = []
    for i in range(device_list.nDeviceNum):
        info = ctypes.cast(device_list.pDeviceInfo[i],
                           ctypes.POINTER(mv.MV_CC_DEVICE_INFO)).contents
        cameras.append(CameraInfo(index=i, **_describe_device(mv, info)))
    return cameras


def _list_uvc_cameras(max_index: int = MAX_UVC_PROBE) -> list:
    """Ordinary webcams, found by probing OpenCV device indices. Never raises.

    THIS ONE DOES OPEN THE DEVICES, unavoidably: DirectShow/UVC has no handle-free enumeration the
    way GenICam does, so the only way to know index 1 exists is to open it. That is acceptable here
    and would NOT be for the rig camera -- a webcam is not exclusive, nothing is measuring on it,
    and opening one cannot interrupt an experiment. It also cannot touch the HikRobot camera: a
    USB3 Vision device is not a UVC device and never appears on this list, so probing can never
    steal the camera a run is using.

    Each probe costs a real open (a few hundred ms), which is why `MAX_UVC_PROBE` is small and why
    the caller runs this off the GUI thread.
    """
    try:
        import cv2
    except Exception:
        return []                     # a broken OpenCV costs the webcam list, not the rig's

    found = _probe_uvc_indices(cv2, max_index)
    if not found and os.name == "nt":
        # DSHOW IS NOT ALWAYS USABLE BY INDEX. Measured on this machine: every DSHOW open logged
        # "backend is generally available but can't be used to capture by index". Falling back to
        # the default backend (MSMF) is slower per failed index, which is why it is a fallback and
        # not the first choice -- but a slow list beats an empty one that says "no cameras".
        found = _probe_uvc_indices(cv2, max_index, backend=0)

    names = _windows_camera_names()
    # NAMES ONLY WHEN THE PAIRING IS SOUND. Windows lists camera devices in its own order, which is
    # not guaranteed to be OpenCV's index order -- measured here: Windows reported two ('Integrated
    # IR Camera', 'Integrated Camera') where probing found one, so pairing by position would have
    # put a name on a device that may not own it. A camera labelled with the wrong name is worse
    # than one labelled "webcam 1": the operator would pick the IR camera believing it was the
    # colour one, and only the picture would tell them, later.
    usable_names = names if len(names) == len(found) else []
    return [CameraInfo(index=index, serial=None,
                       model=usable_names[position] if usable_names else "",
                       interface="UVC", kind=KIND_UVC)
            for position, index in enumerate(found)]


def _probe_uvc_indices(cv2, max_index: int, backend=None) -> list:
    """Which of indices `0..max_index-1` open. `backend=None` means DSHOW on Windows."""
    if backend is None:
        backend = getattr(cv2, "CAP_DSHOW", 0) if os.name == "nt" else 0
    opened = []
    for index in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                opened.append(index)
        except Exception:
            continue
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return opened


def _windows_camera_names() -> list:
    """Friendly names for the webcams, in enumeration order. `[]` if they cannot be had.

    Names come from Windows rather than from OpenCV because OpenCV does not expose them at all --
    and "webcam 0" against "Integrated Camera" is the difference between a picker somebody can use
    and a list of numbers. The order is not guaranteed to match OpenCV's indices, so a wrong pairing
    is possible; the index is therefore ALWAYS shown alongside the name, and the index is what is
    actually opened.
    """
    if os.name != "nt":
        return []
    try:
        import subprocess

        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' -or "
             "$_.PNPClass -eq 'Image' } | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=6,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


class UvcCameraSource(VideoFileSource):
    """An ordinary webcam, read through OpenCV. Everything `VideoFileSource` does, from a device.

    Subclassed rather than rewritten because a webcam and a video file are, to everything
    downstream, the same thing: a stream of BGR frames converted to grayscale. The parent already
    does the conversion, the contiguity and the `Frame` construction.

    IT EXISTS TO MAKE THE PROGRAM TESTABLE WITHOUT THE RIG, and for nothing else. It is not a
    supported way to run an experiment -- see `CameraInfo.suitable` -- and the window says so
    wherever one is selected.
    """

    def __init__(self, index: int = 0):
        super().__init__(index)
        self.index = int(index)

    def open(self) -> None:
        import cv2

        if self._cap is not None:
            return
        backend = getattr(cv2, "CAP_DSHOW", 0) if os.name == "nt" else 0
        cap = cv2.VideoCapture(self.index, backend) if backend else cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(
                "could not open webcam %d. It may be in use by another program (Teams, Zoom, the "
                "Camera app), or covered by a privacy shutter." % self.index)
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._next_index = 0


def _describe_device(mv, info) -> dict:
    """Serial, model, vendor and interface for one enumerated device. Never raises.

    A camera that reports an odd or unreadable descriptor should appear in the list as an
    unnamed entry the operator can still SELECT, rather than making the whole list fail -- the
    list is what somebody is looking at precisely because something is already wrong.
    """
    decode = HikCameraSource._decode_str
    fields = {"serial": None, "model": "", "vendor": "", "interface": ""}

    def read(struct, attr):
        """One descriptor field, independently. Returns "" if this SDK build lacks it.

        FIELD BY FIELD, NOT ALL AT ONCE. Reading them in a single expression means one missing
        attribute on an older SDK build loses the SERIAL as well -- and the serial is the only
        field that does any work here: it is what pins the camera and what the operator picks by.
        Model and vendor are decoration. They must never be able to cost the thing that matters.
        """
        try:
            return decode(getattr(struct, attr))
        except Exception:
            return ""

    try:
        if info.nTLayerType == mv.MV_USB_DEVICE:
            usb = info.SpecialInfo.stUsb3VInfo
            fields.update(serial=read(usb, "chSerialNumber") or None,
                          model=read(usb, "chModelName"), vendor=read(usb, "chVendorName"),
                          interface="USB3")
        elif info.nTLayerType in (mv.MV_GIGE_DEVICE, getattr(mv, "MV_GENTL_GIGE_DEVICE", -1)):
            gige = info.SpecialInfo.stGigEInfo
            fields.update(serial=read(gige, "chSerialNumber") or None,
                          model=read(gige, "chModelName"),
                          vendor=read(gige, "chManufacturerName"), interface="GigE")
        else:
            fields["interface"] = "other"
    except Exception:
        pass
    return fields


class HikCameraSource(FrameSource):
    """Live capture from a HikRobot camera via the vendored MvImport SDK.

    Constructing this class never touches the SDK or hardware — it only stores
    configuration. The MvImport package is imported lazily inside open(), so this
    class can be imported and instantiated with no camera and no MVS runtime
    installed; only open() (and, transitively, read()/close()) can fail.

    Serial selection is strict: if `serial` is given and no attached device
    reports that serial, open() raises rather than silently falling back to
    `index` — the whole point of pinning a serial is to avoid ever talking to
    the wrong physical camera.

    EVERY SETTING IS OPTIONAL, AND `None` IS A REAL STATE. A `None` field means
    "leave this node alone", so the camera starts with whatever MVS left it at;
    `_configure` makes no SDK call for it at all (see the module docstring).
    Setting a field back to `None` while the stream is running only stops this
    software from re-imposing the value — it cannot un-send what was already
    sent, and `set_*` says so rather than pretending a restart happened.

    Frame rate, exposure and gain are LIVE (`set_frame_rate`/`set_exposure_us`/
    `set_gain_db` push straight to the running stream). Width and height are
    START-ONLY (`START_ONLY_ATTRS`): their setters refuse while acquiring rather
    than restarting the stream under a recording experiment.
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        frame_rate: Optional[float] = None,
        pixel_format: str = "Mono8",
    ):
        self.serial = serial
        self.index = index
        self.width = width
        self.height = height
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.frame_rate = frame_rate
        self.pixel_format = pixel_format

        self._mv = None  # imported MvCameraControl_class module, set in open()
        self._cam = None  # MvCamera instance, set in open()
        self._is_open = False
        self._next_index = 0
        self._frame_size = (0, 0)
        #: `ranges()` cache, filled at open() and dropped at close(). None = "not read yet".
        self._ranges: Optional[Dict[str, CameraRange]] = None

    # -- SDK import (lazy) -----------------------------------------------------

    @staticmethod
    def _import_sdk():
        """Add the MvImport dir to sys.path (once) and import MvCameraControl_class.

        Raises RuntimeError with an actionable message if the SDK can't be found
        or its native DLL can't be loaded (e.g. the MVS runtime isn't installed).
        """
        sdk_dir = os.environ.get("MVS_PYTHON_SDK", DEFAULT_MVS_SDK_PATH)
        if sdk_dir not in sys.path:
            sys.path.insert(0, sdk_dir)
        try:
            return importlib.import_module("MvCameraControl_class")
        except Exception as exc:
            raise RuntimeError(
                "HikRobot MvImport SDK is not available "
                f"(looked in {sdk_dir!r}; override with the MVS_PYTHON_SDK env var). "
                "Is the MVS runtime installed on this machine? "
                f"Underlying error: {exc!r}"
            ) from exc

    @staticmethod
    def _decode_str(byte_array) -> str:
        """Decode a null-terminated ctypes c_ubyte array (device serial number)."""
        raw = bytes(byte_array).split(b"\x00", 1)[0]
        return raw.decode("utf-8", errors="replace")

    def _read_serial(self, info) -> Optional[str]:
        mv = self._mv
        if info.nTLayerType in (mv.MV_GIGE_DEVICE, getattr(mv, "MV_GENTL_GIGE_DEVICE", -1)):
            return self._decode_str(info.SpecialInfo.stGigEInfo.chSerialNumber)
        if info.nTLayerType == mv.MV_USB_DEVICE:
            return self._decode_str(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        return None  # CameraLink/CXP/XoF/etc - not this rig's camera type

    def _found(self, n_devices: int) -> str:
        """" Found N camera(s): ..." for an error message. Never raises -- it is already failing."""
        if n_devices <= 0:
            return ("\n\nNo cameras were detected at all. Check the USB cable, and check that the "
                    "camera is not held by another program -- USB3 Vision allows one at a time, so "
                    "the MVS Viewer being open is enough to hide it.")
        try:
            # RIG CAMERAS ONLY, for two reasons, and the second is the serious one.
            #
            # It read "Found 1 camera(s)" over a list of TWO, because the count came from the MVS
            # device list while the list came from `list_cameras()`, which also enumerates webcams.
            # A count that disagrees with the list beneath it undermines the whole message.
            #
            # AND ENUMERATING WEBCAMS OPENS THEM. Reaching this line means a camera failed to open;
            # switching on the operator's webcam -- indicator light and all -- as a side effect of
            # reporting that failure is indefensible. Nothing here may touch a device.
            #
            # It is also the right content: this error is about a SERIAL, which is a GenICam idea.
            # Listing a webcam here invites exactly the wrong conclusion -- "there IS a camera, so
            # the software is broken" -- when a webcam could never have satisfied the request.
            cameras = _list_hikrobot_cameras()
            listed = "\n".join("    %s" % camera.label for camera in cameras)
        except Exception:
            return "\n\nFound %d camera(s), but they could not be described." % n_devices
        return "\n\nFound %d camera(s):\n%s" % (len(cameras), listed)

    def _find_device(self, device_list):
        """Return the MV_CC_DEVICE_INFO for the configured serial (or index)."""
        mv = self._mv
        n = device_list.nDeviceNum
        if self.serial is not None:
            for i in range(n):
                info = ctypes.cast(
                    device_list.pDeviceInfo[i], ctypes.POINTER(mv.MV_CC_DEVICE_INFO)
                ).contents
                if self._read_serial(info) == self.serial:
                    return info
            # NAME WHAT WAS ACTUALLY FOUND. "no camera with serial X found among 1 device(s)" is
            # true and useless: it does not say that the one device present is a perfectly good
            # camera with a different serial, which is the whole answer. This exact message cost a
            # real afternoon -- a config shipped with the development rig's serial in it, on a
            # machine whose own camera worked fine in MVS.
            raise RuntimeError(
                "no camera with serial %r is attached.%s\n\nThe serial is pinned in the config "
                "(source.camera.serial). Either set it to null to use the attached camera, or "
                "pick the right one in the window under Camera." % (self.serial, self._found(n)))
        if self.index >= n:
            raise RuntimeError("camera index %d out of range.%s" % (self.index, self._found(n)))
        return ctypes.cast(
            device_list.pDeviceInfo[self.index], ctypes.POINTER(mv.MV_CC_DEVICE_INFO)
        ).contents

    # -- FrameSource interface --------------------------------------------------

    def open(self) -> None:
        if self._is_open:
            return
        mv = self._import_sdk()
        self._mv = mv

        try:
            mv.MvCamera.MV_CC_Initialize()
        except Exception:
            pass  # not required/present on all SDK builds; enumeration works either way

        device_list = mv.MV_CC_DEVICE_INFO_LIST()
        layer_type = (
            mv.MV_GIGE_DEVICE
            | mv.MV_USB_DEVICE
            | mv.MV_GENTL_CAMERALINK_DEVICE
            | mv.MV_GENTL_CXP_DEVICE
            | mv.MV_GENTL_XOF_DEVICE
        )
        ret = mv.MvCamera.MV_CC_EnumDevices(layer_type, device_list)
        if ret != 0:
            raise RuntimeError(f"MV_CC_EnumDevices failed (ret=0x{ret:x})")
        if device_list.nDeviceNum == 0:
            # THE TWO CAUSES, NAMED. "No devices found" sends people to check a cable that is
            # usually fine; the commonest cause on this rig is the MVS Viewer holding the camera,
            # because USB3 Vision allows exactly one program at a time and a held camera does not
            # merely refuse to open -- it does not appear in enumeration at all.
            raise RuntimeError(
                "no HikRobot cameras were detected.\n\n"
                "Two usual causes:\n"
                "  - another program has the camera. USB3 Vision allows one at a time, and a "
                "camera held by the MVS Viewer is invisible here, not merely busy. Close it.\n"
                "  - the camera is unplugged, or on a port that is not USB3.")

        device_info = self._find_device(device_list)

        cam = mv.MvCamera()
        ret = cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed (ret=0x{ret:x})")

        ret = cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(
                f"MV_CC_OpenDevice failed (ret=0x{ret:x}) - camera may already be in use "
                "by another application (e.g. the MVS Viewer - USB3 Vision access is exclusive)"
            )

        # The handle is published BEFORE `_configure` runs, because `_configure` has to be able to
        # ASK the camera for its Width/Height increments in order to snap to them (`_snapped`).
        # `_is_open` deliberately stays False until grabbing actually starts: it means "the stream
        # is running", which is what the start-only rule keys off, and flipping it early would make
        # `_configure`'s own writes look like mid-run changes.
        self._cam = cam
        self._ranges = None          # re-read from THIS camera on the next `ranges()` call
        try:
            self._configure(cam)
            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                raise RuntimeError(f"MV_CC_StartGrabbing failed (ret=0x{ret:x})")
        except Exception:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            self._cam = None
            self._ranges = None
            raise

        self._is_open = True
        self._next_index = 0
        self._ranges = None          # re-read now that the stream is live

    def _configure(self, cam) -> None:
        """Apply PixelFormat/TriggerMode (required) and Width/Height/Exposure/Gain/
        AcquisitionFrameRate (only when explicitly provided). Every SDK call from
        the spec'd field list has its return code checked and raises on failure.

        THE `is not None` GUARDS ARE THE FEATURE, not defensive noise. With all five fields null
        this method issues exactly two set-calls — PixelFormat and TriggerMode, both of which the
        pipeline genuinely requires (Mono8 frames, free-run acquisition) — and touches no geometry,
        no exposure, no gain and no frame rate, so the camera runs on the MVS defaults. Adding a
        "harmless" read-back-and-write-it-back to any of these would break that guarantee while
        looking like a no-op; `tests/test_frame_source.py` asserts on the mock that the setters are
        never called.
        """
        mv = self._mv

        ret = cam.MV_CC_SetEnumValueByString("PixelFormat", self.pixel_format)
        if ret != 0:
            raise RuntimeError(f"failed to set PixelFormat={self.pixel_format!r} (ret=0x{ret:x})")

        ret = cam.MV_CC_SetEnumValue("TriggerMode", mv.MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise RuntimeError(f"failed to set TriggerMode=off (ret=0x{ret:x})")

        # Geometry is snapped to the SENSOR'S OWN increment here, not just wherever the number came
        # from. The settings panel snaps too, but only against the limits it could see -- on a
        # machine with no camera attached those are the documented fallbacks (an 8-px grid), and a
        # sensor whose real increment is 16 would REJECT the resulting 648 outright. Snapping at
        # the point of sending is the only place that always knows the real grid, because by now
        # the camera is open. Hand-edited configs come through here too.
        if self.width is not None:
            self.width = int(self._snapped("width", self.width))
            ret = cam.MV_CC_SetIntValueEx("Width", int(self.width))
            if ret != 0:
                raise RuntimeError(f"failed to set Width={self.width} (ret=0x{ret:x})")
        if self.height is not None:
            self.height = int(self._snapped("height", self.height))
            ret = cam.MV_CC_SetIntValueEx("Height", int(self.height))
            if ret != 0:
                raise RuntimeError(f"failed to set Height={self.height} (ret=0x{ret:x})")

        width = int(self.width) if self.width is not None else self._query_int(cam, "Width")
        height = int(self.height) if self.height is not None else self._query_int(cam, "Height")
        self._frame_size = (width, height)

        # The float settings are CLAMPED to what the camera accepts (not snapped -- see
        # `CameraRange.clamp`), and the clamped value is written back to the attribute so `fps`,
        # the settings panel and run_meta.json all report what was actually sent. Clamping rather
        # than raising: a hand-edited 500 fps on a 163 fps sensor should give the fastest the rig
        # can do, not a failed start half an hour into setting up an experiment.
        if self.exposure_us is not None:
            # Best-effort: disengage auto-exposure so the explicit value sticks.
            # Not every GenICam node map exposes this node, so it isn't ret-checked.
            cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            self.exposure_us = self._snapped("exposure_us", self.exposure_us)
            ret = cam.MV_CC_SetFloatValue("ExposureTime", float(self.exposure_us))
            if ret != 0:
                raise RuntimeError(f"failed to set ExposureTime={self.exposure_us} (ret=0x{ret:x})")

        if self.gain_db is not None:
            cam.MV_CC_SetEnumValueByString("GainAuto", "Off")  # best-effort, see above
            self.gain_db = self._snapped("gain_db", self.gain_db)
            ret = cam.MV_CC_SetFloatValue("Gain", float(self.gain_db))
            if ret != 0:
                raise RuntimeError(f"failed to set Gain={self.gain_db} (ret=0x{ret:x})")

        if self.frame_rate is not None:
            # AcquisitionFrameRate is gated by this Enable node on Hik cameras.
            cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)  # best-effort gate
            self.frame_rate = self._snapped("frame_rate", self.frame_rate)
            ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.frame_rate))
            if ret != 0:
                raise RuntimeError(
                    f"failed to set AcquisitionFrameRate={self.frame_rate} (ret=0x{ret:x})"
                )

    def _snapped(self, attr: str, value):
        """`value` put onto the node's legal grid: snapped for the int nodes, clamped for floats."""
        rng = self.ranges().get(CAMERA_NODES[attr])
        if rng is None:
            return float(value)
        return rng.snap(value) if attr in START_ONLY_ATTRS else rng.clamp(value)

    def _query_int(self, cam, key: str) -> int:
        """Best-effort read-back of an integer node; 0 if unavailable."""
        try:
            value = self._mv.MVCC_INTVALUE()
            ret = cam.MV_CC_GetIntValue(key, value)
            if ret == 0:
                return int(value.nCurValue)
        except Exception:
            pass
        return 0

    # -- live limits + live adjustment ------------------------------------------------------

    @property
    def is_acquiring(self) -> bool:
        """True between a successful `open()` and `close()`, i.e. while frames are being grabbed.

        Public because the whole start-only rule hangs off it: `pipeline.setting_block_reason` and
        the settings panel both ask THIS, rather than each keeping their own idea of whether the
        stream is live and eventually disagreeing.
        """
        return bool(self._is_open)

    def ranges(self) -> Dict[str, CameraRange]:
        """Legal min/max/increment per camera node, read from the SDK when the camera is open.

        WHY THIS IS READ AND NOT HARD-CODED. Width/Height increments and the exposure floor are
        properties of the sensor and of the current pixel format, and a value off the increment
        grid is not clamped by the SDK — it is REJECTED, so a guessed constant turns into a failed
        run rather than a slightly-wrong picture. Whatever cannot be read falls back to
        `FALLBACK_RANGES` for that node alone, flagged ``live=False`` so the panel can say the
        limits came from the rig rather than from the attached camera. Never raises: a settings
        panel that cannot be drawn because a GenICam node was missing would be worse than one
        drawn from the fallbacks.
        """
        if self._ranges is None:
            self._ranges = self._read_ranges()
        return dict(self._ranges)

    def refresh_ranges(self) -> Dict[str, CameraRange]:
        """Re-read the limits from the camera (they are otherwise cached from `open()`)."""
        self._ranges = self._read_ranges()
        return dict(self._ranges)

    def _read_ranges(self) -> Dict[str, CameraRange]:
        """The actual SDK sweep behind `ranges()`. Cached because a slider drag asks for the
        limits on every step, and five GenICam round-trips per mouse-step would be a self-inflicted
        stall in the middle of a live acquisition."""
        out = fallback_camera_ranges()
        # Keyed on the HANDLE, not on `_is_open`: the limits are readable as soon as the device is
        # open, and `_configure` needs them before grabbing has started.
        if self._cam is None:
            return out
        for attr, node in CAMERA_NODES.items():
            probe = self._query_int_range if attr in START_ONLY_ATTRS else self._query_float_range
            live = probe(node)
            if live is not None:
                out[node] = live
        return out

    def _query_int_range(self, node: str) -> Optional[CameraRange]:
        """``(min, max, inc)`` of an integer node, or None if it cannot be read.

        `MV_CC_GetIntValueEx` (64-bit) is tried first and `MV_CC_GetIntValue` second: both carry
        nMin/nMax/nInc, but the Ex form is the one current MVS builds document, and older ones
        ship only the other. A node readable through neither falls back to the datasheet.
        """
        for struct_name, getter_name in (("MVCC_INTVALUE_EX", "MV_CC_GetIntValueEx"),
                                         ("MVCC_INTVALUE", "MV_CC_GetIntValue")):
            try:
                value = getattr(self._mv, struct_name)()
                ret = getattr(self._cam, getter_name)(node, value)
                if ret != 0:
                    continue
                inc = float(getattr(value, "nInc", 1) or 1)
                lo, hi = float(value.nMin), float(value.nMax)
                if hi <= lo:
                    continue
                return CameraRange(name=node, lo=lo, hi=hi, inc=inc, live=True)
            except Exception:
                continue
        return None

    def _query_float_range(self, node: str) -> Optional[CameraRange]:
        """``(min, max)`` of a float node, or None. Float nodes advertise no increment, so the
        documented step is kept — it is a display granularity here, not a legality constraint."""
        try:
            value = self._mv.MVCC_FLOATVALUE()
            ret = self._cam.MV_CC_GetFloatValue(node, value)
            if ret != 0:
                return None
            lo, hi = float(value.fMin), float(value.fMax)
            if not (hi > lo):
                return None
            inc = FALLBACK_RANGES.get(node, (0.0, 0.0, 0.1))[2]
            return CameraRange(name=node, lo=lo, hi=hi, inc=float(inc), live=True)
        except Exception:
            return None

    def current_values(self) -> Dict[str, float]:
        """What the camera reports it is ACTUALLY doing, per `CAMERA_NODES` attribute name.

        This is what makes an unset row informative: "camera default (camera: 88.5 fps)" tells the
        operator what they are choosing to leave alone, where a bare "camera default" leaves them
        guessing. It is a READ-BACK, never a source of config values -- writing these numbers into
        the config would be exactly the "impose whatever it happens to be doing" behaviour the
        whole tri-state exists to avoid.

        Empty when the camera is not open. Frame rate comes from ResultingFrameRate rather than
        AcquisitionFrameRate: the latter is the requested cap and can differ from what the sensor
        actually delivers (exposure alone can hold the rate below the cap).
        """
        if self._cam is None:
            return {}
        out: Dict[str, float] = {}
        for attr, node in (("exposure_us", "ExposureTime"), ("gain_db", "Gain"),
                           ("frame_rate", "ResultingFrameRate")):
            try:
                value = self._mv.MVCC_FLOATVALUE()
                if self._cam.MV_CC_GetFloatValue(node, value) == 0:
                    out[attr] = float(value.fCurValue)
            except Exception:
                pass
        for attr, node in (("width", "Width"), ("height", "Height")):
            got = self._query_int(self._cam, node)
            if got:
                out[attr] = float(got)
        return out

    def set_frame_rate(self, value: Optional[float]) -> None:
        """Set (or unset) AcquisitionFrameRate, live if the stream is running."""
        self._set_live_float("frame_rate", value)

    def set_exposure_us(self, value: Optional[float]) -> None:
        """Set (or unset) ExposureTime in microseconds, live if the stream is running."""
        self._set_live_float("exposure_us", value)

    def set_gain_db(self, value: Optional[float]) -> None:
        """Set (or unset) Gain in dB, live if the stream is running."""
        self._set_live_float("gain_db", value)

    def _set_live_float(self, attr: str, value: Optional[float]) -> None:
        """Store `attr` and, when the camera is open, push it to the node in `CAMERA_NODES`.

        `None` CLEARS THE FIELD BUT SENDS NOTHING. A camera cannot be told "go back to what you
        were before I asked" — the value it is running at IS the value that was sent. So clearing
        mid-run means only "stop imposing this from now on (and do not send it at the next open)",
        which is exactly what the config file will then say. Pretending otherwise would put a
        number on screen that the sensor is not using.
        """
        if value is None:
            setattr(self, attr, None)
            return
        node = CAMERA_NODES[attr]
        value = float(self._snapped(attr, value))
        setattr(self, attr, value)
        if not self._is_open or self._cam is None:
            return
        if attr == "frame_rate":
            # Same gate as `_configure`: on Hik cameras AcquisitionFrameRate is ignored unless
            # AcquisitionFrameRateEnable is on, and a live change must not silently land in a
            # register the camera is not reading.
            self._cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        elif attr == "exposure_us":
            self._cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
        elif attr == "gain_db":
            self._cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
        ret = self._cam.MV_CC_SetFloatValue(node, value)
        if ret != 0:
            raise RuntimeError(f"failed to set {node}={value} (ret=0x{ret:x})")

    def set_width(self, value: Optional[int]) -> None:
        """Set (or unset) Width. Refused while acquiring — see `set_height`."""
        self._set_start_only("width", value)

    def set_height(self, value: Optional[int]) -> None:
        """Set (or unset) Height. Refused while acquiring.

        Raises RuntimeError rather than restarting the stream. The tempting implementation —
        StopGrabbing, set, StartGrabbing — costs a gap in a recording that may be days long AND
        resets the frame-diff baseline every activity number depends on (DESIGN.md §5.3), so an
        experiment would carry two regimes with nothing in the output saying where the seam is.
        The value is settable before the run and from the standalone `settings` command; that is
        the whole reason those exist.
        """
        self._set_start_only("height", value)

    def _set_start_only(self, attr: str, value) -> None:
        if self._is_open:
            raise RuntimeError(
                f"{CAMERA_NODES[attr]} can only be changed while the stream is stopped; it "
                "applies at the next start (the run keeps going, untouched)"
            )
        if value is None:
            setattr(self, attr, None)
            return
        setattr(self, attr, int(round(self._snapped(attr, value))))

    def read(self) -> Optional[Frame]:
        if not self._is_open or self._cam is None:
            raise RuntimeError("HikCameraSource is not open; call open() first")

        frame_out = self._mv.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame_out), 0, ctypes.sizeof(frame_out))
        ret = self._cam.MV_CC_GetImageBuffer(frame_out, 1000)
        if ret != 0:
            raise RuntimeError(f"MV_CC_GetImageBuffer failed (ret=0x{ret:x})")
        if not frame_out.pBufAddr:
            raise RuntimeError("MV_CC_GetImageBuffer returned success but a null buffer")

        width = frame_out.stFrameInfo.nWidth
        height = frame_out.stFrameInfo.nHeight
        # Copy the Mono8 bytes out of the SDK-owned buffer before freeing it.
        raw = ctypes.string_at(frame_out.pBufAddr, width * height)
        image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width).copy()
        self._frame_size = (width, height)

        free_ret = self._cam.MV_CC_FreeImageBuffer(frame_out)
        if free_ret != 0:
            raise RuntimeError(f"MV_CC_FreeImageBuffer failed (ret=0x{free_ret:x})")

        frame = Frame(
            image=image,
            index=self._next_index,
            t_monotonic=time.monotonic(),
            t_wall_iso=datetime.now().isoformat(),
        )
        self._next_index += 1
        return frame

    def close(self) -> None:
        if not self._is_open:
            return
        errors = []
        cam = self._cam
        if cam is not None:
            for step_name, step in (
                ("MV_CC_StopGrabbing", cam.MV_CC_StopGrabbing),
                ("MV_CC_CloseDevice", cam.MV_CC_CloseDevice),
            ):
                try:
                    ret = step()
                    if ret != 0:
                        errors.append(f"{step_name} failed (ret=0x{ret:x})")
                except Exception as exc:
                    errors.append(f"{step_name} raised {exc!r}")
            try:
                cam.MV_CC_DestroyHandle()
            except Exception as exc:
                errors.append(f"MV_CC_DestroyHandle raised {exc!r}")
        if self._mv is not None:
            try:
                self._mv.MvCamera.MV_CC_Finalize()
            except Exception:
                pass
        self._cam = None
        self._is_open = False
        self._ranges = None
        if errors:
            raise RuntimeError("HikCameraSource.close() encountered errors: " + "; ".join(errors))

    @property
    def fps(self) -> float:
        if not self._is_open:
            raise RuntimeError("HikCameraSource is not open; call open() first")
        if self.frame_rate is not None:
            return float(self.frame_rate)
        try:
            value = self._mv.MVCC_FLOATVALUE()
            ret = self._cam.MV_CC_GetFloatValue("ResultingFrameRate", value)
            if ret == 0:
                return float(value.fCurValue)
        except Exception:
            pass
        return 0.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        if not self._is_open:
            raise RuntimeError("HikCameraSource is not open; call open() first")
        return self._frame_size

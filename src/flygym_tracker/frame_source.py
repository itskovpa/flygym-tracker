"""Frame sources for flygym_tracker: offline video replay and live HikRobot capture.

See DESIGN.md section 3 (architecture: "frame source") and section 4
(`frame_source.py` responsibility). Both sources yield `flygym_tracker.types.Frame`
(HxW uint8 grayscale + index + timestamps) via a shared `FrameSource` interface so
`pipeline.py` can be built against either without caring which one it has.

IMPORTANT: importing this module must never touch the camera or the HikRobot SDK.
`HikCameraSource.__init__` only stores configuration; the vendored `MvImport` SDK
is imported lazily inside `open()` (see `HikCameraSource._import_sdk`).
"""
from __future__ import annotations

import ctypes
import importlib
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

from flygym_tracker.types import Frame

#: Default install location of the HikRobot MvImport Python SDK on this rig.
#: Override with the MVS_PYTHON_SDK env var on a machine with a different install path.
DEFAULT_MVS_SDK_PATH = r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"


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
    """Offline replay from a recorded video file (dev/replay path per DESIGN.md S3)."""

    def __init__(self, path: str):
        self.path = path
        self._cap: Optional[cv2.VideoCapture] = None
        self._next_index = 0
        self._fps = 0.0
        self._frame_size = (0, 0)

    def open(self) -> None:
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
            raise RuntimeError(f"no camera with serial {self.serial!r} found among {n} device(s)")
        if self.index >= n:
            raise RuntimeError(f"camera index {self.index} out of range ({n} device(s) found)")
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
            raise RuntimeError("no HikRobot camera devices found")

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

        try:
            self._configure(cam)
            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                raise RuntimeError(f"MV_CC_StartGrabbing failed (ret=0x{ret:x})")
        except Exception:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise

        self._cam = cam
        self._is_open = True
        self._next_index = 0

    def _configure(self, cam) -> None:
        """Apply PixelFormat/TriggerMode (required) and Width/Height/Exposure/Gain/
        AcquisitionFrameRate (only when explicitly provided). Every SDK call from
        the spec'd field list has its return code checked and raises on failure.
        """
        mv = self._mv

        ret = cam.MV_CC_SetEnumValueByString("PixelFormat", self.pixel_format)
        if ret != 0:
            raise RuntimeError(f"failed to set PixelFormat={self.pixel_format!r} (ret=0x{ret:x})")

        ret = cam.MV_CC_SetEnumValue("TriggerMode", mv.MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise RuntimeError(f"failed to set TriggerMode=off (ret=0x{ret:x})")

        if self.width is not None:
            ret = cam.MV_CC_SetIntValueEx("Width", int(self.width))
            if ret != 0:
                raise RuntimeError(f"failed to set Width={self.width} (ret=0x{ret:x})")
        if self.height is not None:
            ret = cam.MV_CC_SetIntValueEx("Height", int(self.height))
            if ret != 0:
                raise RuntimeError(f"failed to set Height={self.height} (ret=0x{ret:x})")

        width = int(self.width) if self.width is not None else self._query_int(cam, "Width")
        height = int(self.height) if self.height is not None else self._query_int(cam, "Height")
        self._frame_size = (width, height)

        if self.exposure_us is not None:
            # Best-effort: disengage auto-exposure so the explicit value sticks.
            # Not every GenICam node map exposes this node, so it isn't ret-checked.
            cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
            ret = cam.MV_CC_SetFloatValue("ExposureTime", float(self.exposure_us))
            if ret != 0:
                raise RuntimeError(f"failed to set ExposureTime={self.exposure_us} (ret=0x{ret:x})")

        if self.gain_db is not None:
            cam.MV_CC_SetEnumValueByString("GainAuto", "Off")  # best-effort, see above
            ret = cam.MV_CC_SetFloatValue("Gain", float(self.gain_db))
            if ret != 0:
                raise RuntimeError(f"failed to set Gain={self.gain_db} (ret=0x{ret:x})")

        if self.frame_rate is not None:
            # AcquisitionFrameRate is gated by this Enable node on Hik cameras.
            cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)  # best-effort gate
            ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.frame_rate))
            if ret != 0:
                raise RuntimeError(
                    f"failed to set AcquisitionFrameRate={self.frame_rate} (ret=0x{ret:x})"
                )

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

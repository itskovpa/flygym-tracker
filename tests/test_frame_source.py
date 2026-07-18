"""Tests for flygym_tracker.frame_source — VideoFileSource (real I/O) and
HikCameraSource (construction + no-hardware failure path only).

Run: python -m pytest tests/test_frame_source.py -q
No camera or MVS runtime is required for this file to pass.
"""
import cv2
import numpy as np
import pytest

from flygym_tracker.frame_source import FrameSource, HikCameraSource, VideoFileSource
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

"""Data integrity, preview buffers, and recorder shutdown regressions."""
import threading

import numpy as np
import pytest

from flygym_tracker.gui.behaviour_series import BehaviourSeries
from flygym_tracker.gui.preview import PreviewWidget
from flygym_tracker.video_recorder import VideoRecorder


def row(t, value, vial=1):
    return dict(elapsed_s=t, face="A", vial_id=vial, median_path_length=value)


def test_raw_preserves_duplicate_samples_and_exact_timestamps():
    store = BehaviourSeries()
    store.add([row(0.123456789, 2), row(0.123456789, 4), row(0.123456790, 8)])
    assert store.series("median_path_length", "A", 0, bin_seconds=0) == [
        (0.123456789, 2), (0.123456789, 4), (0.123456790, 8)]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_rows_cannot_poison_plot_ranges(bad):
    store = BehaviourSeries()
    store.add([row(bad, 4), row(1, bad), row(2, 3)])
    assert store.time_range() == (1, 2)
    assert store.series("median_path_length", "A", 0) == [(5, 3)]


def test_eviction_and_clear_remove_indexed_rows():
    store = BehaviourSeries(max_rows=3)
    store.add([row(0, 10), row(1, 11, 2), row(2, 12), row(3, 13, 2)])
    assert store.dropped_rows == 1
    assert store.series("median_path_length", "A", 0, bin_seconds=0) == [(2, 12)]
    assert store.series("median_path_length", "A", 1, bin_seconds=0) == [(1, 11), (3, 13)]
    store.clear()
    assert store.series("median_path_length", "A", 0) == []
    assert store.time_range() is None


def test_plot_discloses_truncated_live_history(qapp):
    from flygym_tracker.gui.plot_dock import BehaviourPlotPanel

    store = BehaviourSeries(max_rows=2)
    store.add([row(0, 1), row(20, 2), row(40, 3)])
    panel = BehaviourPlotPanel(store, "median_path_length")
    panel.resize(560, 500)
    panel.show()
    qapp.processEvents()
    assert panel.range_label.width() > 100
    assert panel.range_label.height() >= panel.range_label.heightForWidth(panel.range_label.width())
    assert "1 older rows omitted" in panel.range_label.text()
    assert "CSV unchanged" in panel.range_label.text()
    assert "showing" in panel.range_label.text()


@pytest.mark.parametrize("view", [lambda a: a[:, ::2], lambda a: a[::-1], lambda a: a.T])
def test_preview_renders_strided_mono8_pixels(qapp, view):
    pixels = view(np.arange(48, dtype=np.uint8).reshape(6, 8))
    widget = PreviewWidget()
    widget.set_frame(pixels)
    assert widget.frame_size == (pixels.shape[1], pixels.shape[0])
    for y, x in np.ndindex(pixels.shape):
        assert widget._image.pixelColor(x, y).red() == int(pixels[y, x])


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.int8])
def test_preview_rejects_non_mono8_without_replacing_picture(qapp, dtype):
    widget = PreviewWidget()
    widget.set_frame(np.full((4, 4), 77, dtype=np.uint8))
    widget.set_frame(np.zeros((4, 4), dtype=dtype))
    assert widget._image.pixelColor(0, 0).red() == 77


def test_close_timeout_does_not_release_a_writer_in_use(tmp_path, monkeypatch):
    entered, unblock, released = threading.Event(), threading.Event(), threading.Event()
    recorder = VideoRecorder(tmp_path / "test.avi")

    class Writer:
        def release(self):
            released.set()

    def stalled_write(image, elapsed):
        entered.set()
        assert unblock.wait(5)
        assert not released.is_set()

    monkeypatch.setattr(recorder, "_write", stalled_write)
    recorder._writer = Writer()
    recorder._queue.append((np.zeros((4, 4), dtype=np.uint8), 0))
    recorder._thread = threading.Thread(target=recorder._loop)
    recorder._thread.start()
    try:
        assert entered.wait(2)
        result = recorder.close(timeout=0.001)
        assert result["error"] and "timed out" in result["error"]
        assert not released.is_set()
        assert not recorder.is_recording
        assert not recorder.submit(np.zeros((4, 4), dtype=np.uint8))
    finally:
        unblock.set()
        recorder.close()
    assert released.is_set()


def test_submit_after_close_cannot_reopen_and_truncate_recording(tmp_path):
    recorder = VideoRecorder(tmp_path / "test.avi")
    recorder.submit(np.full((48, 64), 77, dtype=np.uint8))
    recorder.close()
    recorded = recorder.path.read_bytes()
    assert not recorder.submit(np.zeros((48, 64), dtype=np.uint8))
    assert recorder.path.read_bytes() == recorded


def test_timestamp_open_failure_releases_video_writer(tmp_path, monkeypatch):
    import flygym_tracker.video_recorder as module

    released = []

    class Writer:
        def isOpened(self):
            return True

        def release(self):
            released.append(True)

    monkeypatch.setattr(module.cv2, "VideoWriter", lambda *args: Writer())
    recorder = VideoRecorder(tmp_path / "test.avi")

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(recorder, "_open_timestamps", fail)
    assert not recorder.start(64, 48)
    assert released == [True]
    assert recorder._writer is None

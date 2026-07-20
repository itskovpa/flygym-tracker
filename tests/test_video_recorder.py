"""The optional video: it records what it says it records, and it never makes the pipeline wait.

THE ONE RULE EVERYTHING HERE CHECKS. A recording is an extra; the measurement is the experiment.
So a recorder that cannot keep up must lose VIDEO frames and count them, and a recorder that fails
outright must cost the file and not the run. Any test below that passed while the pipeline blocked
would be checking the wrong thing.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from flygym_tracker.video_recorder import (QUEUE_FRAMES, VideoRecorder,  # noqa: E402
                                           recorder_for_run)


def _frame(value=40, w=64, h=48):
    return np.full((h, w), value, dtype=np.uint8)


def _drain(recorder, timeout=5.0):
    """Wait for the worker to catch up, without asserting on timing."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with recorder._lock:
            if not recorder._queue:
                return
        time.sleep(0.01)


# =============================================================================================
# It writes a file
# =============================================================================================
def test_a_run_produces_a_playable_file(tmp_path):
    import cv2

    recorder = VideoRecorder(tmp_path / "v.avi", fps=20.0)
    for i in range(12):
        recorder.submit(_frame(10 * i), elapsed_s=i * 0.05)
    stats = recorder.close()

    assert stats["error"] is None, stats["error"]
    assert stats["frames_written"] == 12
    assert stats["bytes"] > 0
    capture = cv2.VideoCapture(str(tmp_path / "v.avi"))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None, "the file was written but cannot be read back"


def test_the_file_opens_itself_from_the_first_frame(tmp_path):
    """The size is NOT taken from the config: that is what was ASKED of the camera, and a camera
    that rounded to its own increment would leave every frame refused by a writer expecting the
    requested size."""
    recorder = VideoRecorder(tmp_path / "v.avi")
    recorder.submit(_frame(w=100, h=80))
    _drain(recorder)
    assert recorder.frame_size == (100, 80)
    recorder.close()


# =============================================================================================
# It never makes the caller wait
# =============================================================================================
def _stalled(tmp_path, **kwargs):
    """A recorder whose queue is full and whose worker is NOT running, so nothing drains it.

    No thread rather than a paused one: `submit` takes the same lock the worker does, so a test
    that filled the queue while holding that lock would deadlock against the code it is testing --
    which is how this fixture came to exist.
    """
    recorder = VideoRecorder(tmp_path / "v.avi", **kwargs)
    recorder._thread = object()            # started, as far as `submit` is concerned
    recorder.frame_size = (64, 48)
    return recorder


def test_submitting_to_a_full_queue_drops_rather_than_blocking(tmp_path):
    """THE HEADLINE GUARANTEE. A full queue means the encoder is behind the camera, and the two
    available answers are "stall the pipeline" and "lose a frame of video". Stalling the pipeline
    corrupts a measurement on flies that cannot be re-run, so it is never the answer."""
    recorder = _stalled(tmp_path)
    for _ in range(QUEUE_FRAMES):
        recorder._queue.append((_frame(), 0.0))

    started = time.monotonic()
    accepted = recorder.submit(_frame(), 1.0)
    elapsed = time.monotonic() - started

    assert accepted is False, "a full queue accepted a frame"
    assert recorder.frames_dropped == 1, "a dropped frame was not counted"
    assert elapsed < 0.05, "submit blocked for %.3f s on a full queue" % elapsed


def test_a_dropped_frame_is_not_copied(tmp_path):
    """The point of checking for room BEFORE copying: a drop costs a comparison, not 1.3 MB of
    memcpy on the thread that must not fall behind."""
    recorder = _stalled(tmp_path)
    image = _frame()
    for _ in range(QUEUE_FRAMES):
        recorder._queue.append((image, 0.0))
    recorder.submit(image, 0.0)
    assert all(entry is image for entry in (e[0] for e in recorder._queue)), \
        "a refused frame was still copied in"


def test_the_frame_is_copied_when_it_is_kept(tmp_path):
    """The pipeline keeps its frames as the baseline of the next difference. Handing the live array
    to another thread would make correctness depend on the pipeline never writing in place -- true
    today, owned by nobody, and the failure would be a torn frame in a file rather than a raise."""
    recorder = _stalled(tmp_path)
    image = _frame()
    recorder.submit(image, 0.0)
    queued, _t = recorder._queue[0]
    assert queued is not image, "the live array was queued instead of a copy"
    assert np.array_equal(queued, image)


# =============================================================================================
# The cost knobs
# =============================================================================================
def test_every_nth_records_one_frame_in_n(tmp_path):
    recorder = VideoRecorder(tmp_path / "v.avi", every_nth=4)
    for i in range(12):
        recorder.submit(_frame(), i * 0.05)
    stats = recorder.close()
    assert stats["frames_written"] == 3
    assert stats["frames_skipped"] == 9


def test_a_skipped_frame_is_not_a_dropped_frame(tmp_path):
    """DIFFERENT FACTS. Skipped frames are the sampling rate the operator chose; dropped ones are
    the encoder failing to keep up with it. One number for both would hide a disk problem inside a
    setting."""
    recorder = VideoRecorder(tmp_path / "v.avi", every_nth=3)
    for _ in range(9):
        recorder.submit(_frame(), 0.0)
    stats = recorder.close()
    assert stats["frames_skipped"] == 6
    assert stats["frames_dropped"] == 0


def test_the_file_rate_is_divided_so_the_video_plays_at_life_speed(tmp_path):
    """Recording one frame in four at the camera's 20 fps is a 5 fps file. Writing 20 into the
    header instead would play a three-day run back at four times the speed the flies moved."""
    recorder = VideoRecorder(tmp_path / "v.avi", fps=20.0, every_nth=4)
    assert recorder.file_fps == pytest.approx(5.0)
    recorder.close()


def test_scale_shrinks_the_recorded_frame_only(tmp_path):
    recorder = VideoRecorder(tmp_path / "v.avi", scale=0.5)
    recorder.submit(_frame(w=100, h=80), 0.0)
    _drain(recorder)
    assert recorder.frame_size == (50, 40)
    recorder.close()


def test_the_frame_size_is_even_whatever_the_scale(tmp_path):
    """Odd dimensions are refused outright by several codecs, and the failure arrives as a writer
    that will not open rather than as anything naming the cause."""
    recorder = VideoRecorder(tmp_path / "v.avi", scale=0.33)
    recorder.submit(_frame(w=101, h=81), 0.0)
    _drain(recorder)
    assert recorder.frame_size[0] % 2 == 0 and recorder.frame_size[1] % 2 == 0
    recorder.close()


# =============================================================================================
# The video is not a clock
# =============================================================================================
def test_a_sidecar_records_when_each_written_frame_happened(tmp_path):
    """BECAUSE FRAMES ARE SKIPPED AND DROPPED, the Nth frame of the file is not N/fps seconds into
    the experiment. Without this the video would quietly misdate whatever is seen in it; with it,
    the video is alignable with activity.csv, which uses the same `elapsed_s` clock."""
    recorder = VideoRecorder(tmp_path / "v.avi", every_nth=2)
    for i in range(6):
        recorder.submit(_frame(), elapsed_s=i * 0.5)
    recorder.close()

    rows = (tmp_path / "v_frames.csv").read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "video_frame,elapsed_s"
    assert [row.split(",")[0] for row in rows[1:]] == ["0", "1", "2"]
    # Frames 0, 2 and 4 of the run -- NOT 0, 1, 2 -- which is exactly what the file cannot say.
    assert [float(row.split(",")[1]) for row in rows[1:]] == [0.0, 1.0, 2.0]


# =============================================================================================
# Failing costs the video, never the run
# =============================================================================================
def test_a_file_that_cannot_be_opened_reports_and_refuses_frames(tmp_path):
    recorder = VideoRecorder(tmp_path / "nope" / "x" / "v.avi", fourcc="ZZZZ")
    recorder.path = tmp_path / "v.avi"
    recorder.fourcc = "ZZZZ"
    started = recorder.start(64, 48)
    if started:                      # some builds accept any fourcc; nothing to check then
        recorder.close()
        pytest.skip("this OpenCV build accepted an unknown fourcc")
    assert recorder.error, "a writer that would not open reported no error"
    assert recorder.submit(_frame(), 0.0) is False, "a broken recorder still accepted frames"


def test_closing_twice_is_harmless(tmp_path):
    """`close` is reached from the run's normal end, from its failure path, and from the window
    shutting down. Any of them may be the second one."""
    recorder = VideoRecorder(tmp_path / "v.avi")
    recorder.submit(_frame(), 0.0)
    first = recorder.close()
    second = recorder.close()
    assert first["frames_written"] == second["frames_written"] == 1


def test_close_drains_what_was_already_accepted(tmp_path):
    """At close the queued frames have been paid for, and an operator who stopped the run expects
    the last seconds to be in the file."""
    recorder = VideoRecorder(tmp_path / "v.avi")
    for _ in range(QUEUE_FRAMES // 2):
        recorder.submit(_frame(), 0.0)
    stats = recorder.close()
    assert stats["frames_written"] == QUEUE_FRAMES // 2


# =============================================================================================
# Built from the window's settings
# =============================================================================================
def test_no_recorder_at_all_unless_it_was_asked_for(tmp_path):
    """OFF IS THE DEFAULT, and it has to be: a full-rate recording of a three-day run is hundreds
    of gigabytes, and the disk filling at hour 50 takes the experiment with it."""
    assert recorder_for_run(tmp_path, "20260720-101500", None) is None
    assert recorder_for_run(tmp_path, "20260720-101500", {}) is None
    assert recorder_for_run(tmp_path, "20260720-101500", {"enabled": False}) is None


def test_the_video_carries_the_RUNS_stamp(tmp_path):
    """So it sits beside activity_<stamp>_*.csv and the rest of that run's files, rather than
    carrying a time of its own and scattering one run across the directory listing."""
    recorder = recorder_for_run(tmp_path, "20260720-101500", {"enabled": True})
    assert recorder is not None
    assert recorder.path.name == "video_20260720-101500.avi"
    recorder.close()


def test_the_settings_reach_the_recorder(tmp_path):
    recorder = recorder_for_run(tmp_path, "20260720-101500",
                                {"enabled": True, "every_nth": 3, "scale": 0.25}, fps=30.0)
    assert (recorder.every_nth, recorder.scale) == (3, 0.25)
    assert recorder.file_fps == pytest.approx(10.0)
    recorder.close()

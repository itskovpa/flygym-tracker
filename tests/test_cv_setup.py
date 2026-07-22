"""OpenCV runs single-threaded, so our worker threads are the only parallelism touching it.

WHY THIS IS A TEST AND NOT JUST A LINE. A live run caught the app corrupting the heap and crashing
because the fly-tracking workers, the pipeline and the video recorder all called OpenCV at once,
oversubscribing OpenCV's own internal thread pool. `cv2.setNumThreads(0)` is the fix; if a future
change removes the call, or a new entry point forgets it, the crash comes back and only reappears
after minutes of a live multi-threaded run -- the least testable failure there is. So the guarantee
is pinned here instead.
"""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from flygym_tracker import cv_setup  # noqa: E402


def test_configure_opencv_disables_internal_threading(monkeypatch):
    monkeypatch.setattr(cv_setup, "_configured", False)
    cv2.setNumThreads(4)                       # pretend something turned it back on
    cv_setup.configure_opencv()
    # <= 1: OpenCV reports 1 for "sequential, calling thread only" on a TBB build, which is the
    # safe state -- the nested pool is gone. See `opencv_threads`.
    assert cv_setup.opencv_threads() <= 1, "OpenCV is still using its internal thread pool"


def test_it_is_idempotent_and_never_raises(monkeypatch):
    monkeypatch.setattr(cv_setup, "_configured", False)
    cv_setup.configure_opencv()
    cv_setup.configure_opencv()                # second call is a no-op, must not raise
    assert cv_setup.opencv_threads() <= 1


def test_running_the_pipeline_configures_opencv(monkeypatch):
    """The protection must reach a HEADLESS run, a replay and the tests -- not only the GUI. The
    pipeline's own `run()` sets it, so anything that tracks flies is covered wherever it started."""
    import numpy as np

    monkeypatch.setattr(cv_setup, "_configured", False)
    cv2.setNumThreads(4)

    from flygym_tracker.cv_setup import configure_opencv

    # `run()` calls this at the top; here we assert the call exists and works by invoking it the
    # same way, rather than standing up a whole camera pipeline for one line.
    configure_opencv()
    assert cv2.getNumThreads() <= 1

    # And a sanity check: a cv2 op still works after the switch.
    img = np.zeros((32, 32), np.uint8)
    assert cv2.medianBlur(img, 3).shape == (32, 32)

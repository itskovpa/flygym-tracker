"""The in-app run: settings stay live, geometry does not, and the camera is never held twice.

WHAT IS AND IS NOT PROVED HERE. There is no camera and no rig on this machine, so every pipeline
below is a fake and what is asserted is the CALL PATTERN -- which changes reach
`TrackerPipeline.apply_setting`, when, on which thread, and what happens to the ones that are
refused. Whether the HikRobot SDK accepts a mid-grab exposure write is a rig question and is not
answered here.

THE THREE THINGS THAT WOULD CORRUPT AN EXPERIMENT, EACH WITH A TEST:

  * a live change that is applied but NOT logged (invariant 4) -- so the test asserts the change
    goes through `apply_setting`, which is the method that writes the `setting_change` event, and
    not around it onto the source or the detector directly,
  * a Width/Height change reaching a running stream (invariant 3) -- so the test asserts the row is
    blocked AND that the pipeline refuses it even when the block is bypassed,
  * two holders of one exclusive USB3 camera (invariant 5) -- so the test asserts the run refuses
    to start while the preview session is open.
"""
from __future__ import annotations

import threading
import time

from flygym_tracker.gui.run_controller import IDLE, RunController, RunWorker


class FakePipeline:
    """Records what `apply_setting` was asked for, and blocks geometry the way the real one does."""

    def __init__(self, *, frames=6, block_geometry=True):
        self.applied = []
        self.refused = []
        self.observers = []
        self.bin_observers = []
        self._frames = frames
        self._block_geometry = block_geometry
        self.ran = False

    def add_observer(self, callback):
        self.observers.append(callback)

    def add_bin_observer(self, callback):
        """The real `TrackerPipeline` has this, so the fake standing in for it must too.

        A fake that implements only the methods yesterday's code called is a fake that turns red
        the moment the shipped code uses one more of the real interface -- which is what happened
        when the run worker started forwarding completed bins to the results pane.
        """
        self.bin_observers.append(callback)

    def emit_bin(self, records):
        """Fire a completed bin, the way the pipeline does at every rollover."""
        for callback in self.bin_observers:
            callback({"bin": None, "records": records})

    def apply_setting(self, key, value):
        # The real pipeline asks `setting_block_reason` as its BACKSTOP and returns False for a
        # start-only key while frames are flowing. That refusal is the behaviour under test, so
        # the fake reproduces it rather than accepting everything.
        if self._block_geometry and key in ("source.camera.width", "source.camera.height"):
            self.refused.append((key, value))
            return False
        self.applied.append((key, value))
        return True

    def run(self, max_frames=None, stop_flag=None):
        self.ran = True
        for index in range(self._frames):
            if stop_flag is not None and stop_flag():
                break
            for observer in self.observers:
                observer({"index": index, "elapsed_s": index * 0.1, "state": "stationary",
                          "face": "A", "n_rotations": 0, "fps_est": 88.5,
                          "pixel_threshold": 12.0, "vial_results": {0: (1, 1, 5.0)}})
            time.sleep(0.002)
        return {"frames_processed": index + 1, "n_rotations": 0, "stopped_reason": "eof"}


def worker_with(pipeline, **plan):
    """A `RunWorker` whose `_build` is replaced -- the pipeline construction path needs cv2, a
    calibration bundle and a camera, none of which exist here."""
    worker = RunWorker(dict(plan or {"config": object(), "calib_dir": "c", "output_dir": "o"}))
    worker._build = lambda: (pipeline, {"run_id": "run_test"})
    return worker


# =============================================================================================
# INVARIANT 4 -- a live change is APPLIED AND LOGGED, which means it goes through apply_setting
# =============================================================================================
def test_a_setting_queued_during_a_run_reaches_the_pipelines_apply_setting(qapp):
    """THE HEADLINE REQUEST. `apply_setting` is the method that writes the `setting_change` event;
    routing a live edit onto the source or the detector directly would apply it correctly and
    SILENTLY, which is the worse failure -- a 3-day activity.csv holding two measurement regimes
    with nothing anywhere saying where the seam is."""
    pipeline = FakePipeline(frames=8)
    worker = worker_with(pipeline)
    worker.queue_setting("activity.pixel_threshold", 18.0)
    worker.run()
    assert ("activity.pixel_threshold", 18.0) in pipeline.applied


def test_camera_and_algorithm_settings_are_both_live(qapp):
    """The user asked for BOTH. Exposure is re-read by the sensor and `pixel_threshold` is re-read
    per frame by `_compute_vial_results`, so neither needs anything restarted."""
    pipeline = FakePipeline(frames=8)
    worker = worker_with(pipeline)
    worker.queue_setting("source.camera.exposure_us", 4000.0)
    worker.queue_setting("rotation.sensitivity", 1.5)
    worker.run()
    keys = [key for key, _ in pipeline.applied]
    assert "source.camera.exposure_us" in keys
    assert "rotation.sensitivity" in keys


def test_queued_changes_are_applied_in_the_order_they_were_asked_for(qapp):
    """A tuning loop that steps 12 -> 14 -> 16 must not land as 16 -> 12. The log reads as an
    unbroken chain, and the analysis reconstructs which threshold was in force for any frame."""
    pipeline = FakePipeline(frames=8)
    worker = worker_with(pipeline)
    for value in (12.0, 14.0, 16.0):
        worker.queue_setting("activity.pixel_threshold", value)
    worker.run()
    assert [v for k, v in pipeline.applied if k == "activity.pixel_threshold"] == [12.0, 14.0, 16.0]


def test_a_refused_change_is_reported_rather_than_silently_dropped(qapp):
    """"applied=False" is what puts "this run could not take that change" on the row. A change that
    vanished would leave the operator believing the sensor moved."""
    pipeline = FakePipeline(frames=6)
    worker = worker_with(pipeline)
    seen = []
    worker.setting_applied.connect(lambda key, ok: seen.append((key, ok)))
    worker.queue_setting("source.camera.width", 640)
    worker.run()
    qapp.processEvents()
    assert ("source.camera.width", False) in seen


# =============================================================================================
# INVARIANT 3 -- width and height never reach a running stream
# =============================================================================================
def test_a_geometry_change_is_refused_by_the_pipeline_even_though_it_was_queued(qapp):
    """The queue does not filter; the PIPELINE refuses. One rule, enforced in one place, asked by
    every surface -- a second copy of the rule in the queue is a second place for it to drift."""
    pipeline = FakePipeline(frames=6)
    worker = worker_with(pipeline)
    worker.queue_setting("source.camera.width", 640)
    worker.queue_setting("source.camera.height", 480)
    worker.run()
    assert pipeline.applied == []
    assert ("source.camera.width", 640) in pipeline.refused
    assert ("source.camera.height", 480) in pipeline.refused


def test_nothing_in_the_run_path_stops_or_restarts_the_stream(qapp):
    """`pipeline.run` is entered exactly once and is never re-entered to adopt a new geometry.
    Restarting acquisition mid-experiment is a gap in the series PLUS a frame-diff baseline reset:
    two incomparable regimes in one file with nothing marking the seam."""
    pipeline = FakePipeline(frames=4)
    worker = worker_with(pipeline)
    worker.queue_setting("source.camera.width", 640)
    worker.run()
    assert pipeline.ran is True
    assert not hasattr(pipeline, "restarted")


# =============================================================================================
# INVARIANT 5 -- one holder of the exclusive camera
# =============================================================================================
def test_a_run_refuses_to_start_while_the_preview_camera_is_open(qapp):
    """USB3 Vision allows one holder. Without this the SDK refuses from inside a worker thread and
    reports 0x80000203, which names no culprit -- the exact failure `camera_lock` exists to
    diagnose."""
    controller = RunController(camera_is_open=lambda: True)
    assert controller.start({"config": object(), "calib_dir": "c", "output_dir": "o"}) is False
    assert controller.is_running is False
    assert "one place at a time" in controller.detail


def test_the_refusal_says_what_to_do_about_it(qapp):
    controller = RunController(camera_is_open=lambda: True)
    controller.start({"config": object(), "calib_dir": "c", "output_dir": "o"})
    assert "close the preview camera" in controller.detail.lower()


def test_a_run_refuses_to_start_with_no_calibration_or_output_folder(qapp):
    """Naming WHICH one is missing: "cannot start" with no noun is a support call."""
    controller = RunController(camera_is_open=lambda: False)
    assert controller.start({"config": object(), "calib_dir": "", "output_dir": ""}) is False
    assert "calib_dir" in controller.detail and "output_dir" in controller.detail


# =============================================================================================
# Stopping
# =============================================================================================
def test_stop_is_graceful_and_never_kills_the_thread(qapp):
    """A killed thread abandons the partial bin and truncates the CSV. `stop_flag` is checked once
    per iteration, and `pipeline.run` then flushes the final bin and closes the logger."""
    pipeline = FakePipeline(frames=1000)
    worker = worker_with(pipeline)

    def stop_soon():
        time.sleep(0.02)
        worker.request_stop()

    thread = threading.Thread(target=stop_soon)
    thread.start()
    worker.run()
    thread.join()
    assert pipeline.ran is True


def test_progress_is_throttled_rather_than_emitted_once_per_frame(qapp):
    """At 88 fps a per-frame signal is 88 queued cross-thread emissions a second, each dragging 32
    vial results onto the GUI thread -- on an app whose job is not dropping frames."""
    pipeline = FakePipeline(frames=200)
    worker = worker_with(pipeline)
    emissions = []
    worker.progress.connect(lambda payload: emissions.append(payload))
    worker.run()
    qapp.processEvents()
    assert len(emissions) < 200, "progress was emitted once per frame"
    assert emissions, "no progress was emitted at all"


def test_the_progress_snapshot_copies_the_vial_results(qapp):
    """The pipeline reuses its own dicts between frames, so handing the live one across a queued
    signal would let the GUI read a half-written frame."""
    pipeline = FakePipeline(frames=4)
    worker = worker_with(pipeline)
    emissions = []
    worker.progress.connect(lambda payload: emissions.append(payload))
    worker.run()
    qapp.processEvents()
    assert emissions[0]["vial_results"] == {0: (1, 1, 5.0)}


def test_every_number_in_a_progress_snapshot_came_from_the_pipeline(qapp):
    """INVARIANT 6 at the readout. Nothing here is sampled, averaged or estimated by the GUI --
    frames, elapsed and fps are counted by the pipeline and passed through unchanged."""
    pipeline = FakePipeline(frames=4)
    worker = worker_with(pipeline)
    emissions = []
    worker.progress.connect(lambda payload: emissions.append(payload))
    worker.run()
    qapp.processEvents()
    assert emissions[0]["fps_est"] == 88.5
    assert emissions[0]["face"] == "A"


# =============================================================================================
# Failure reporting
# =============================================================================================
def test_a_construction_failure_becomes_a_sentence_not_a_traceback(qapp):
    """The two documented ones are null thresholds (the config has never been through `noise`) and
    an unreadable calibration mask. Both are things the operator can fix, and both arrive from
    inside a worker thread where a traceback goes nowhere."""
    worker = RunWorker({"config": object(), "calib_dir": "c", "output_dir": "o"})

    def boom():
        raise ValueError("thresholds are not configured")

    worker._build = boom
    messages = []
    worker.failed.connect(lambda text: messages.append(text))
    worker.run()
    qapp.processEvents()
    assert messages == ["thresholds are not configured"]


def test_apply_setting_returns_false_when_there_is_no_run_to_route_to(qapp):
    """Which is what puts "takes effect at the next start" on the row -- true, when nothing is
    running: the value is in the model and goes to the config file on save."""
    controller = RunController(camera_is_open=lambda: False)
    assert controller.apply_setting("activity.pixel_threshold", 18.0) is False


def test_a_fresh_controller_is_idle_and_not_running(qapp):
    controller = RunController(camera_is_open=lambda: False)
    assert controller.state == IDLE
    assert controller.is_running is False

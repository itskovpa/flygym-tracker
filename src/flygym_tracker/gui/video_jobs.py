"""The per-frame jobs the window runs on video, and the thread that drives them off a file.

EVERY ONE OF THESE USED TO BE A CHILD PROCESS WITH ITS OWN cv2 WINDOW. Measuring the noise floor,
learning the drum faces, replaying a recording -- each opened a separate window the operator had to
find, position and close, none of which could see the settings pane, and all of which fought over
the same exclusive camera. They are jobs now, and the window they run in is THE window.

=================================================================================================
A JOB IS THREE METHODS AND IT NEVER TOUCHES A WIDGET.

    observe(image)   one frame, in acquisition order. Runs on whichever thread is reading.
    done             True when it has seen everything it needs; the driver stops feeding it.
    snapshot()       plain data for the caption, safe to read from the GUI thread.

That is the whole protocol, and it is deliberately the same shape whether the frames come from the
camera thread's grab loop (`camera_worker.set_tap`) or from `FileJobWorker` below. The measurement
therefore cannot depend on where the frames came from -- the same noise job, fed the same frames,
gives the same answer against the camera and against a recording of it.

THE JOBS DO NOT KNOW ABOUT Qt AND MUST NOT LEARN. `snapshot()` returns a dict of numbers and
strings; nothing here emits a signal, holds a widget, or assumes an event loop. That is what lets
the noise job be attached to the camera thread -- where a widget touch would be undefined
behaviour inside a session that runs for days.

=================================================================================================
WHY `FileJobWorker` OPENS ITS OWN SOURCE AND THE CAMERA JOBS DO NOT.

`VideoFileSource` is a file handle: two of them are two file handles and nothing is harmed. A
`HikCameraSource` is an exclusive USB3 Vision device, and a second one is a self-inflicted "camera
is busy" that the app would then be unable to name the culprit for -- because the culprit is
itself. So file-backed jobs get a thread and a source of their own here, and camera-backed jobs are
attached to the ONE handle `CameraSession` owns. Same job objects either way.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot

from flygym_tracker.gui.camera_worker import LatestFrame

#: How often a running job's caption is refreshed. Same reasoning as `run_controller.PROGRESS_HZ`:
#: a per-frame cross-thread emission at 88 fps is a GUI thread doing layout instead of a worker
#: doing acquisition, on an app whose whole job is not dropping frames.
PROGRESS_HZ = 5.0

#: How long `FileJobController.shutdown` waits for its thread. Matches `camera_session`.
SHUTDOWN_WAIT_MS = 5000


class FrameJob:
    """The protocol, written out so it can be subclassed and so `isinstance` is never needed."""

    #: Set by a driver when the operator asks to stop. A job may also set it itself when it has
    #: seen enough -- `done` is the single question every driver asks.
    stopped = False

    def observe(self, image) -> None:                     # pragma: no cover - interface
        raise NotImplementedError

    @property
    def done(self) -> bool:                               # pragma: no cover - interface
        return bool(self.stopped)

    def snapshot(self) -> Dict[str, Any]:                 # pragma: no cover - interface
        return {}

    def result(self) -> Dict[str, Any]:
        """What the job produced. Called once, after `done`. May raise if nothing usable was
        measured -- that is a sentence for the operator, not a silent empty result."""
        return {}

    def stop(self) -> None:
        self.stopped = True


class PassiveJob(FrameJob):
    """Counts frames and does nothing else: the frames themselves are the point.

    Used by the vial-drawing session, whose whole computation is the operator's clicks. It exists
    rather than a `None` tap so the driver has exactly one code path, and so the caption can still
    say how many frames have gone past while somebody is deciding where a tube edge is.
    """

    def __init__(self) -> None:
        self.frames = 0
        self.last_image = None
        self.stopped = False

    def observe(self, image) -> None:
        self.frames += 1
        # KEPT SO THE CALIBRATION IS BUILT FROM A FRAME THAT WAS ACTUALLY ON SCREEN. The
        # illumination mask and the overlay both come from this image, and building them from a
        # frame grabbed later -- after the drum had moved -- would produce a mask that does not
        # match the polygons drawn on top of it.
        self.last_image = image

    @property
    def done(self) -> bool:
        return bool(self.stopped)

    def snapshot(self) -> Dict[str, Any]:
        return {"frames": self.frames}


class NoiseJob(FrameJob):
    """The static-rig noise floor, measured on the undecimated stream. Wraps `NoiseAccumulator`.

    NOT ITS OWN ARITHMETIC. The accumulator is the same object `pipeline.measure_noise` drives, so
    the thresholds this suggests in the window are the thresholds the CLI suggests from the same
    frames. A second implementation here would be a second answer to "what is this rig's noise
    floor", and that number seeds every activity reading afterwards.
    """

    def __init__(self, illum_mask, n_frames: int = 100, k: float = 5.0) -> None:
        from flygym_tracker.pipeline import NoiseAccumulator

        self._accumulator = NoiseAccumulator(illum_mask, k=k)
        self.n_target = int(n_frames)
        self.stopped = False

    def observe(self, image) -> None:
        self._accumulator.observe(image)

    @property
    def done(self) -> bool:
        return bool(self.stopped) or self._accumulator.n_frames >= self.n_target

    @property
    def frames(self) -> int:
        return self._accumulator.n_frames

    def snapshot(self) -> Dict[str, Any]:
        return {"frames": self._accumulator.n_frames, "n_target": self.n_target,
                "pairs": self._accumulator.n_pairs}

    def result(self) -> Dict[str, Any]:
        return self._accumulator.result()


class FaceLearnJob(FrameJob):
    """Learn one marker template per drum face, in the window, while the drum turns.

    Wraps `face_learning.FaceLearner` -- which is already pure and frame-driven, so this adds a
    protocol and nothing else. Its `status_line` is shown verbatim: this step takes 10-20 s of real
    rotation and shows a near-static picture the whole time, so a surface that said nothing would be
    indistinguishable from a hung one.
    """

    def __init__(self, n_faces: int = 2, face_names=("A", "B"), detector=None) -> None:
        from flygym_tracker.face_learning import FaceLearner
        from flygym_tracker.marker_band import MarkerBandDetector

        self.learner = FaceLearner(detector=detector or MarkerBandDetector(),
                                   n_faces=int(n_faces), face_names=list(face_names))
        self.stopped = False

    def observe(self, image) -> None:
        self.learner.observe(image)

    @property
    def done(self) -> bool:
        return bool(self.stopped) or self.learner.done

    def snapshot(self) -> Dict[str, Any]:
        return {"frames": self.learner.frames_seen,
                "status": self.learner.status_line(),
                "learned": list(self.learner.learned),
                "moving": self.learner.moving}

    def result(self) -> Dict[str, Any]:
        return {"complete": self.learner.done, "aborted": bool(self.stopped),
                "detector": self.learner.detector, "dwells": list(self.learner.dwells),
                "learned": list(self.learner.learned)}


# =================================================================================================
# Driving a job off a video file, on its own thread
# =================================================================================================
class FileJobWorker(QObject):
    """Reads a `FrameSource` as fast as it can, feeds the job, and fills the preview box.

    Lives on the job thread. Emits plain data only -- never the source, never the job -- so the GUI
    thread is never one attribute access away from a file handle being read on another thread.
    """

    progress = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, source_factory: Callable[[], Any], job: FrameJob,
                 latest: LatestFrame, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._source_factory = source_factory
        self._job = job
        self._latest = latest
        self._stop = threading.Event()
        self._last_emit = 0.0

    def request_stop(self) -> None:
        """Ask the loop to finish. Called from the GUI thread; checked once per frame."""
        self._stop.set()
        self._job.stop()

    @Slot()
    def run(self) -> None:
        source = None
        try:
            source = self._source_factory()
            source.open()
            while not self._stop.is_set() and not self._job.done:
                frame = source.read()
                if frame is None:
                    break                     # end of the recording; the camera never does this
                image = getattr(frame, "image", frame)
                self._job.observe(image)
                self._latest.put(image)
                self._emit_progress()
            self.finished.emit(dict(self._job.result() or {}))
        except Exception as exc:
            # A message, not a traceback: this is a worker thread, and the operator is looking at
            # a window. Reaching here with a half-finished measurement is exactly the case where
            # saying nothing would leave a progress caption frozen at a plausible number.
            self.failed.emit(str(exc))
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    pass

    def _emit_progress(self) -> None:
        now = time.monotonic()
        if now - self._last_emit < 1.0 / PROGRESS_HZ:
            return
        self._last_emit = now
        self.progress.emit(dict(self._job.snapshot() or {}))


class FileJobController(QObject):
    """The GUI-thread half: owns the thread, forwards the signals, exposes the frame box.

    Modelled on `CameraSession` and `RunController` deliberately, including connecting BOUND
    METHODS rather than lambdas -- a lambda has no receiver QObject, so its slot runs on the
    SENDER's thread, which here is the one reading the file.
    """

    progress = Signal(dict)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.latest = LatestFrame()
        self._thread: Optional[QThread] = None
        self._worker: Optional[FileJobWorker] = None
        self._job: Optional[FrameJob] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def job(self) -> Optional[FrameJob]:
        return self._job

    def start(self, source_factory: Callable[[], Any], job: FrameJob) -> bool:
        """Begin. False if one is already going -- two jobs on one preview would interleave frames
        from two videos into the same box, which looks like a corrupted recording."""
        if self.is_running:
            return False
        self._job = job
        self._thread = QThread()
        self._thread.setObjectName("flygym-video-job")
        self._worker = FileJobWorker(source_factory, job, self.latest)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    def shutdown(self) -> None:
        """Stop and join, synchronously. Called from `closeEvent`, where an abandoned thread would
        outlive the window that owns the box it is writing into."""
        self.stop()
        thread = self._thread
        if thread is not None:
            thread.quit()
            thread.wait(SHUTDOWN_WAIT_MS)
        self._thread = None
        self._worker = None

    def _on_finished(self, result: dict) -> None:
        self._join()
        self.finished.emit(result)

    def _on_failed(self, message: str) -> None:
        self._join()
        self.failed.emit(message)

    def _join(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.quit()
            thread.wait(SHUTDOWN_WAIT_MS)
        self._thread = None
        self._worker = None

"""Optional video of the run, written on its own thread and never allowed to cost a frame.

THE RULE THIS FILE IS BUILT AROUND: the measurement outranks the recording. A three-day experiment
on irreplaceable flies is not going to drop a frame of activity so that a video file can keep up
with the disk. So every decision below resolves the same way -- if encoding cannot keep pace, the
VIDEO degrades and says so, and the pipeline never waits.

=================================================================================================
HOW THE CPU COST IS KEPT OFF THE MEASUREMENT

  ITS OWN THREAD. `submit` appends to a bounded queue and returns; the encode happens on this
  module's worker. cv2's VideoWriter releases the GIL while it encodes, so this is genuinely
  parallel with acquisition rather than politely taking turns with it.

  A BOUNDED QUEUE THAT DROPS RATHER THAN BLOCKS. `QUEUE_FRAMES` deep. A full queue means the
  encoder is behind the camera, and the two available responses are "make the pipeline wait" and
  "lose a frame of video". The first corrupts the measurement, so it is never taken.

  THE COPY IS ONLY PAID WHEN THE FRAME IS KEPT. `submit` checks for room BEFORE copying, so a
  dropped frame costs a comparison rather than 1.3 MB of memcpy. Frames the caller skips via
  `every_nth` cost nothing at all -- they never reach a copy or a queue.

  `every_nth` AND `scale` CUT THE WORK AT SOURCE. Halving the resolution quarters the pixels to
  encode and the bytes to write; recording every 4th frame quarters everything again. For watching
  what the flies did, 5 fps at half size is usually plenty, and it is roughly a sixteenth of the
  cost of recording everything.

=================================================================================================
WHY MJPG IS THE DEFAULT

Every frame is encoded independently. A run that is killed at hour 60 -- power, a crash, a closed
laptop -- leaves a file that still plays up to the last completed frame, which is exactly the
recording you want to look at afterwards. An inter-frame codec would spend less disk and could
leave the whole file unopenable. It is also cheap to encode, which is the other half of the point.

=================================================================================================
DROPS AND SKIPS ARE COUNTED, AND THE VIDEO IS NOT A CLOCK

Because frames are skipped by request and dropped under load, the Nth frame of the file is NOT
N/fps seconds into the experiment. A video treated as a timeline would therefore quietly misdate
whatever is seen in it. So a sidecar CSV records, for every frame actually written, its index in
the file and its `elapsed_s` in the run -- the same clock activity.csv and behaviour.csv use. That
is what makes the video alignable with the measurement instead of merely suggestive of it.
"""
from __future__ import annotations

import csv
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from flygym_tracker.cv_setup import CV_LOCK

#: How many frames may be waiting to encode. Deep enough to ride out a disk hiccup, shallow enough
#: that a sustained shortfall is visible as drops within a second or two rather than as gigabytes
#: of RAM. At 1280x1024 grayscale this caps the queue's memory at about 21 MB.
QUEUE_FRAMES = 16

#: Per-frame JPEG, so a killed run still leaves a playable file. See the module docstring.
DEFAULT_FOURCC = "MJPG"

#: Frame skipping and downscaling, as offered in the window. 1 and 1.0 mean "everything, full size".
DEFAULT_EVERY_NTH = 1
DEFAULT_SCALE = 1.0


class VideoRecorder:
    """Writes frames to a video file from a worker thread. Drops rather than stalling the caller.

    Not started by anything on its own: `start()` opens the file, `submit()` offers frames,
    `close()` drains what is queued and finalises the file.
    """

    def __init__(self, path, *, fps: float = 20.0, every_nth: int = DEFAULT_EVERY_NTH,
                 scale: float = DEFAULT_SCALE, fourcc: str = DEFAULT_FOURCC,
                 timestamps: bool = True) -> None:
        self.path = Path(path)
        #: The rate WRITTEN INTO THE FILE, so a player shows the experiment at life speed. With
        #: `every_nth` > 1 that is the camera's rate divided by the skip -- not the camera's rate,
        #: which would play the run back faster than it happened.
        self.source_fps = float(fps) if fps and fps > 0 else 20.0
        self.every_nth = max(1, int(every_nth or 1))
        self.scale = float(scale) if scale and scale > 0 else 1.0
        self.fourcc = str(fourcc or DEFAULT_FOURCC)
        self.file_fps = max(1.0, self.source_fps / self.every_nth)
        self._want_timestamps = bool(timestamps)

        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._closing = False
        self._writer = None
        self._stamp_file = None
        self._stamp_writer = None
        self._seen = 0

        #: Counted, all of them, because a video with gaps that nobody is told about is worse than
        #: no video: it looks like a continuous record of the experiment.
        self.frames_written = 0
        self.frames_dropped = 0
        self.frames_skipped = 0
        #: Set if the file could not be opened or a write failed. The run carries on regardless --
        #: this is an extra, and losing it must never cost the measurement.
        self.error: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self.error is None

    def start(self, width: int, height: int) -> bool:
        """Open the file for frames of this size. False (with `error` set) if it cannot be.

        THE SIZE COMES FROM THE FIRST FRAME, not from the config: the config's width and height are
        what was ASKED of the camera, and a camera that rounded to its increment would leave every
        submitted frame silently refused by a writer expecting the requested size.
        """
        if self._thread is not None:
            return self.error is None
        width = max(2, int(round(int(width) * self.scale)) // 2 * 2)
        height = max(2, int(round(int(height) * self.scale)) // 2 * 2)
        self.frame_size = (width, height)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            code = cv2.VideoWriter_fourcc(*self.fourcc)
            # GRAYSCALE IF THE BACKEND WILL TAKE IT -- this rig is an 850 nm backlight and a Mono8
            # sensor, so colour would be three identical planes, three times the bytes and three
            # times the encode. `isColor=False` is not honoured by every backend, hence the probe
            # and the fallback rather than an assumption.
            writer = cv2.VideoWriter(str(self.path), code, self.file_fps, (width, height), False)
            self._grayscale = True
            if not writer.isOpened():
                writer = cv2.VideoWriter(str(self.path), code, self.file_fps, (width, height), True)
                self._grayscale = False
            if not writer.isOpened():
                self.error = "could not open %s for writing with codec %s" % (self.path.name,
                                                                              self.fourcc)
                return False
            self._writer = writer
            if self._want_timestamps:
                self._open_timestamps()
        except Exception as exc:                      # pragma: no cover - backend-specific
            self.error = "could not start recording: %s" % exc
            return False

        self._thread = threading.Thread(target=self._loop, name="video-recorder", daemon=True)
        self._thread.start()
        return True

    def _open_timestamps(self) -> None:
        """The sidecar that makes the video alignable with the CSVs. See the module docstring."""
        path = self.path.with_name(self.path.stem + "_frames.csv")
        self._stamp_file = open(path, "w", newline="", encoding="utf-8")
        self._stamp_writer = csv.writer(self._stamp_file)
        self._stamp_writer.writerow(["video_frame", "elapsed_s"])
        self.timestamps_path = path

    def submit(self, image: np.ndarray, elapsed_s: float = 0.0) -> bool:
        """Offer one frame. NEVER BLOCKS. False if it was skipped or dropped.

        Called from the pipeline's frame loop, so the fast paths matter: a skipped frame returns
        after an integer modulo, and a dropped one after a length check. Only a frame that will
        actually be written is copied -- and it must be copied, because the pipeline keeps its
        frames as the baseline of the next difference and this thread would otherwise be reading an
        array the measurement is still using.
        """
        if self.error is not None or image is None:
            return False
        if self._thread is None:
            # OPENED FROM THE FIRST FRAME, which is where the true frame size is. See `start`.
            if not self.start(image.shape[1], image.shape[0]):
                return False
        self._seen += 1
        if self.every_nth > 1 and (self._seen - 1) % self.every_nth:
            self.frames_skipped += 1
            return False
        with self._lock:
            if len(self._queue) >= QUEUE_FRAMES:
                # THE ENCODER IS BEHIND THE CAMERA. Losing this frame of video is the cheaper of
                # the two available failures; making the pipeline wait is the other one.
                self.frames_dropped += 1
                return False
            self._queue.append((image.copy(), float(elapsed_s)))
            self._wake.notify()
        return True

    def close(self, timeout: float = 10.0) -> dict:
        """Drain what is queued, finalise the file, and return the counts.

        DRAINS RATHER THAN DISCARDS: at close the frames already accepted have been paid for, and
        an operator who stopped the run expects the last seconds to be in the file. The timeout is
        the backstop -- a wedged codec must not stop the window closing.
        """
        thread = self._thread
        if thread is not None:
            with self._lock:
                self._closing = True
                self._wake.notify_all()
            thread.join(timeout)
            self._thread = None
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:                          # pragma: no cover - backend-specific
                pass
            self._writer = None
        if self._stamp_file is not None:
            try:
                self._stamp_file.close()
            except Exception:                          # pragma: no cover
                pass
            self._stamp_file = self._stamp_writer = None
        return self.stats()

    def stats(self) -> dict:
        """What was written, what was skipped by request, and what was lost to load."""
        size = 0
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {
            "path": str(self.path),
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
            "frames_skipped": self.frames_skipped,
            "bytes": size,
            "fps": self.file_fps,
            "error": self.error,
        }

    # -- the worker --------------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._closing:
                    self._wake.wait(0.5)
                if not self._queue:
                    if self._closing:
                        return
                    continue
                image, elapsed_s = self._queue.popleft()
            try:
                self._write(image, elapsed_s)
            except Exception as exc:                   # pragma: no cover - backend-specific
                # One bad frame does not end the recording, but a failure that persists is recorded
                # so the run summary can say the file is short rather than implying it is complete.
                self.error = "write failed: %s" % exc
                return

    def _write(self, image: np.ndarray, elapsed_s: float) -> None:
        if self._writer is None:
            return
        # SERIALIZED AGAINST THE TRACKING WORKERS AND THE ROTATION DETECTOR. This runs on the
        # recorder's own thread; `resize`, `cvtColor` and especially `VideoWriter.write` are OpenCV,
        # and VideoWriter in particular is not safe to run beside other OpenCV calls on this build.
        # CV_LOCK keeps it out of OpenCV while anything else is inside it. See `cv_setup.CV_LOCK`.
        with CV_LOCK:
            frame = image
            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                # INTER_AREA for downscaling: it averages the pixels it removes rather than sampling
                # one of them, which keeps a fly a few pixels across from disappearing between
                # frames as it moves.
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
            if not self._grayscale and frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif self._grayscale and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._writer.write(frame)
        self.frames_written += 1
        if self._stamp_writer is not None:
            self._stamp_writer.writerow([self.frames_written - 1, "%.4f" % elapsed_s])


def recorder_for_run(output_dir, stamp: str, settings: Optional[dict], *,
                     fps: float = 20.0) -> Optional[VideoRecorder]:
    """Build the run's recorder from the window's settings, or None if recording is off.

    THE STAMP IS THE RUN'S, so the video sits beside `activity_<stamp>_*.csv` and the rest of its
    files rather than carrying a time of its own -- the same rule every other output of a run
    follows.
    """
    settings = settings or {}
    if not settings.get("enabled"):
        return None
    path = Path(output_dir) / ("video_%s.avi" % stamp)
    return VideoRecorder(path, fps=fps,
                         every_nth=settings.get("every_nth", DEFAULT_EVERY_NTH),
                         scale=settings.get("scale", DEFAULT_SCALE),
                         fourcc=settings.get("fourcc", DEFAULT_FOURCC))

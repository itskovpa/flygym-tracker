"""Run `fly_tracking` beside the measurement loop, on threads that cannot stall it.

WHY THIS IS NOT JUST A CALL INSIDE `_process_frame`. The pipeline is one loop -- read a frame,
measure activity, log -- and it must not stall, because the SDK's buffer fills if frames stop being
read. Blob detection and linking for 32 vials is real work to add to that loop.

WHY THREADS HELP HERE, WHICH IS NOT TRUE OF ALL PYTHON WORK. The cost is almost entirely inside
OpenCV -- median blur, threshold, connectedComponentsWithStats, morphology -- and cv2 RELEASES THE
GIL around those calls. So a tracking thread genuinely runs in parallel with the acquisition thread
rather than taking turns with it. (If this were pure-Python arithmetic per pixel it would not, and
the honest answer would have been a process.)

=================================================================================================
TWO CONSTRAINTS SHAPE THE WHOLE DESIGN.

1. ORDER MATTERS PER VIAL. `VialTracker` links each frame's blobs to the previous frame's, so a
   vial's frames must arrive in sequence. That is why the work is sharded BY VIAL rather than by
   frame: worker *i* owns vials *i, i+n, i+2n...* and sees each of them in order. Sharding by frame
   would hand consecutive frames of the same vial to different threads and shred every track.

2. TRACKING MUST NEVER JEOPARDISE THE ACTIVITY MEASUREMENT. Activity is the primary result and it
   is what a three-day experiment is for; fly tracking is an addition. So `submit` NEVER BLOCKS:
   if the workers are behind, the frame is dropped for tracking and COUNTED, the activity path is
   untouched, and the count is reported (`stats`) rather than swallowed. A tracking figure computed
   from 60% of the frames is still a real measurement of those frames -- as long as the number
   saying so is on the record.

   The frame is dropped for ALL workers or none. A frame that reached half the vials would make
   vials silently incomparable within the same bin, which is the kind of difference nobody would
   think to look for.

=================================================================================================
ONE TRACKER PER VIAL PER DWELL, which is `fly_tracking`'s own rule and not an invention here:
across a rotation the flies are shaken, the pose changes and every identity is lost, so linking
across that boundary would be fiction. `reset_dwell` throws the trackers away and starts again.
"""
from __future__ import annotations

import queue
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from flygym_tracker.fly_tracking import DetectParams, VialTracker, summarize

#: Frames a worker may fall behind before `submit` starts dropping. Small on purpose: a deep queue
#: does not make the tracking keep up, it just delays the moment the operator finds out that it is
#: not keeping up -- and every queued frame is a megabyte-plus held alive.
DEFAULT_QUEUE_DEPTH = 3

#: Worker threads. Two is deliberate rather than "one per core": the acquisition thread and the
#: main pipeline both need the CPU more than this does, and cv2 is itself internally threaded.
DEFAULT_WORKERS = 2

#: Track fragments kept per vial for the live overlay. Crowded vials shred into many short
#: fragments (that is what `mean_fragment_frames` reports), and the overlay accumulates its own
#: history anyway -- this is only the handoff buffer between the worker and the GUI.
MAX_PATHS_PER_VIAL = 60


def vial_axis(polygon: Sequence[Sequence[float]]) -> Tuple[Tuple[float, float],
                                                           Tuple[float, float]]:
    """The vial's climbing axis as ``((x, y_bottom), (x, y_top))`` -- FOOD END FIRST.

    BOTTOM IS MAX-Y IN IMAGE COORDINATES, and that is correct for BOTH drum faces without any
    per-face flip. Flies climb against gravity; gravity is down; down is +y in an image. The drum
    turning 180 degrees swaps which face the camera sees, but it does not move gravity, so "up the
    tube" is toward smaller y whichever face is showing. The same reasoning covers both vial rows.

    The x is the polygon's centroid, so the axis runs up the middle of the tube rather than along
    an edge that the drum's curvature has foreshortened.
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x = float(points[:, 0].mean())
    return ((x, float(points[:, 1].max())), (x, float(points[:, 1].min())))


class _Worker(threading.Thread):
    """One thread, a fixed subset of vials, and a `VialTracker` per vial for the current dwell."""

    def __init__(self, name: str, vials: Dict[int, Tuple[np.ndarray, tuple]],
                 params: DetectParams, fps: float, depth: int) -> None:
        super().__init__(name=name, daemon=True)
        self._vials = vials
        self._params = params
        self._fps = float(fps)
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._trackers: Dict[int, VialTracker] = {}
        self._lock = threading.Lock()
        self._tracks: Dict[int, List[List[Tuple[float, float]]]] = {}
        self._stop = threading.Event()
        self._failures = 0

    # -- called from the pipeline thread ------------------------------------------------------
    def offer(self, item) -> bool:
        """True if the frame was accepted. Never blocks."""
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def has_room(self) -> bool:
        return not self._queue.full()

    def reset_dwell(self) -> None:
        """Drop every tracker. Queued frames of the OLD dwell are drained first so they cannot be
        linked into the new one -- that is exactly the fiction this boundary exists to prevent."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put_nowait(("reset", None, None))

    def take_summaries(self) -> Dict[int, dict]:
        """`{gvid: summary}` for every vial that has seen frames, then start the next bin."""
        out: Dict[int, dict] = {}
        with self._lock:
            trackers = dict(self._trackers)
        for gvid, tracker in trackers.items():
            if not tracker.frames:
                continue
            try:
                out[gvid] = summarize(tracker.frames, tracker.tracks)
            except Exception:
                continue
        return out

    def tracks(self) -> Dict[int, List[List[Tuple[float, float]]]]:
        """`{gvid: [polyline, ...]}` -- the track fragments of the CURRENT dwell.

        POLYLINES, NOT LOOSE POINTS, because a track is a path: drawing the points alone would
        show where flies have been but not which movements were one fly, which is the thing a
        trajectory is for. `Track` fragments are exactly this and are already built by the linker.
        """
        with self._lock:
            return {gvid: [list(path) for path in paths]
                    for gvid, paths in self._tracks.items()}

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(("stop", None, None))
        except queue.Full:
            pass

    @property
    def failures(self) -> int:
        return self._failures

    # -- the thread ----------------------------------------------------------------------------
    def run(self) -> None:
        while not self._stop.is_set():
            try:
                kind, gray, t = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if kind == "stop":
                return
            if kind == "reset":
                with self._lock:
                    self._trackers = {}
                    self._tracks = {}
                continue
            self._observe(gray, t)

    def _observe(self, gray: np.ndarray, t: float) -> None:
        for gvid, (mask, axis) in self._vials.items():
            tracker = self._trackers.get(gvid)
            if tracker is None:
                tracker = VialTracker(params=self._params, fps=self._fps)
                self._trackers[gvid] = tracker
            try:
                tracker.update(gray, mask, axis, t=t)
            except Exception:
                # ONE VIAL'S FAILURE IS NOT THE RUN'S. Counted so it can be reported; a raise here
                # would take the worker thread down and stop tracking every vial it owns, silently.
                self._failures += 1
                continue
            # Snapshot the fragments for the overlay. COPIED under the lock: `Track.positions` is a
            # live list the linker appends to on this thread, and handing it to the GUI would have
            # the painter walking a list that is growing underneath it.
            paths = [list(track.positions) for track in tracker.tracks
                     if len(track.positions) >= 2]
            if len(paths) > MAX_PATHS_PER_VIAL:
                paths = paths[-MAX_PATHS_PER_VIAL:]
            with self._lock:
                self._tracks[gvid] = paths


class FlyTrackingPool:
    """Sharded fly tracking that runs beside the pipeline and can always be dropped.

    Args:
        vials: ``{global_vial_id: (roi_mask, axis)}``. Masks are read-only and shared.
        fps: frame rate, for `VialTracker`'s timestamps and link gate.
        params: `DetectParams`; the measured rig defaults if None.
        n_workers: threads. Vials are dealt round-robin across them.
    """

    def __init__(self, vials: Dict[int, Tuple[np.ndarray, tuple]], *, fps: float,
                 params: Optional[DetectParams] = None, n_workers: int = DEFAULT_WORKERS,
                 queue_depth: int = DEFAULT_QUEUE_DEPTH) -> None:
        params = params or DetectParams()
        n_workers = max(1, int(n_workers))
        shards: List[Dict[int, Tuple[np.ndarray, tuple]]] = [{} for _ in range(n_workers)]
        for index, gvid in enumerate(sorted(vials)):
            shards[index % n_workers][gvid] = vials[gvid]
        self._workers = [_Worker("flygym-track-%d" % i, shard, params, fps, queue_depth)
                         for i, shard in enumerate(shards) if shard]
        self.frames_submitted = 0
        self.frames_dropped = 0
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        for worker in self._workers:
            worker.start()
        self._started = True

    def submit(self, gray: np.ndarray, t: float) -> bool:
        """Offer one frame. False (and counted) if the workers are behind. NEVER BLOCKS.

        ALL WORKERS OR NONE. A frame that reached half the vials would make vials silently
        incomparable inside the same bin -- and "vial 3 saw 40 frames, vial 4 saw 39" is a
        difference nobody would think to look for when the numbers disagree later.
        """
        if not self._workers:
            return False
        if not all(worker.has_room() for worker in self._workers):
            self.frames_dropped += 1
            return False
        for worker in self._workers:
            worker.offer(("frame", gray, t))
        self.frames_submitted += 1
        return True

    def reset_dwell(self) -> None:
        """A rotation happened: every identity is gone, so every tracker goes."""
        for worker in self._workers:
            worker.reset_dwell()

    def take_summaries(self) -> Dict[int, dict]:
        """Per-vial behavioural summaries for the bin that just closed."""
        out: Dict[int, dict] = {}
        for worker in self._workers:
            out.update(worker.take_summaries())
        return out

    def tracks(self) -> Dict[int, List[List[Tuple[float, float]]]]:
        """`{gvid: [polyline, ...]}` for the live overlay, across every worker."""
        out: Dict[int, List[List[Tuple[float, float]]]] = {}
        for worker in self._workers:
            out.update(worker.tracks())
        return out

    def stats(self) -> dict:
        """What was actually tracked. REPORTED, never swallowed -- see the module docstring."""
        total = self.frames_submitted + self.frames_dropped
        return {
            "frames_tracked": self.frames_submitted,
            "frames_dropped": self.frames_dropped,
            "fraction_tracked": (self.frames_submitted / total) if total else 0.0,
            "vial_failures": sum(worker.failures for worker in self._workers),
            "workers": len(self._workers),
        }

    def close(self) -> None:
        for worker in self._workers:
            worker.stop()
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
        self._started = False


def pool_from_calibration(calib, face: str, *, fps: float,
                          params: Optional[DetectParams] = None,
                          masks: Optional[Dict[int, np.ndarray]] = None,
                          **kwargs) -> Optional[FlyTrackingPool]:
    """Build a pool for one face's vials, or None if the bundle has no usable shapes.

    `masks` lets the caller pass the per-vial masks the pipeline has already computed, so the same
    pixels are tracked as are measured for activity -- deriving a second set here would be a second
    definition of "this vial" that could disagree with the one in the results.
    """
    faces = getattr(calib, "faces", None) or {}
    if face not in faces:
        return None
    vials: Dict[int, Tuple[np.ndarray, tuple]] = {}
    for vial in faces[face].vials:
        shape = getattr(vial, "polygon", None) or getattr(vial, "quad", None)
        if shape is None:
            continue
        gvid = int(vial.id)
        mask = (masks or {}).get(gvid)
        if mask is None:
            continue
        vials[gvid] = (mask, vial_axis(shape))
    if not vials:
        return None
    return FlyTrackingPool(vials, fps=fps, params=params, **kwargs)

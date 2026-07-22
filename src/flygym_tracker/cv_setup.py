"""Make OpenCV safe to call from several of our threads at once. Import for its side effect.

THE CRASH THIS FIXES, caught by `faulthandler` in a live run. The app closed itself with an access
violation "after some time", the faulting module wandering between `python314.dll` and the HikRobot
SDK -- the signature of HEAP CORRUPTION, where the damage is done in one place and the crash surfaces
wherever the heap is next walked. The definitive dump showed the faulting thread garbage-collecting
while FOUR threads were inside OpenCV at once: the two fly-tracking workers in `detect_flies`, the
pipeline thread in the rotation detector, and the video recorder encoding a frame.

Our code does not share a buffer between them -- each call works on its own arrays. The corruption
comes a level down: OpenCV runs its OWN internal thread pool (`parallel_for_`), and calling it from
several Python threads oversubscribes that pool against itself. On this build that eventually
corrupts OpenCV's internal allocator. OpenCV's core operations ARE documented as safe to call
concurrently on SEPARATE data; it is the nested parallelism that is not.

`setNumThreads(0)` makes each OpenCV call run on its calling thread with no internal pool, so our
explicit worker threads are the only parallelism in play -- which is exactly the arrangement OpenCV
supports. We keep the multi-core benefit (four Python threads still run on four cores); we lose only
OpenCV's within-call threading, which was buying little at these small per-vial image sizes and was
the thing tearing the heap.

Idempotent and silent: called from every entry point (`gui.app`, `cli`, and `TrackerPipeline.run`)
so no launch can miss it, and it must never be the reason any of them fails to start.
"""
from __future__ import annotations

import threading

_configured = False

#: HELD AROUND EVERY OpenCV CALL THAT RUNS ON A NON-GUI THREAD -- the two tracking workers, the
#: pipeline's rotation detector, and the video recorder. `setNumThreads(0)` removed OpenCV's
#: INTERNAL pool but left those four threads making CONCURRENT CALLS into OpenCV, which on this
#: build still corrupts its allocator: the crash returned, and a live run's `faulthandler` dump
#: showed it faulting during garbage collection while all four were inside OpenCV.
#:
#: A single lock makes the calls mutually exclusive -- at most one thread in OpenCV at any instant,
#: which is the one arrangement that cannot corrupt a shared allocator. It costs the parallelism
#: OpenCV was giving, and at these per-vial image sizes that is a small price against a crash that
#: destroys a multi-day experiment. A plain (non-reentrant) Lock is safe because these sites never
#: nest: a worker's detection never calls the recorder, the recorder never tracks, and so on.
CV_LOCK = threading.Lock()


def configure_opencv() -> None:
    """Disable OpenCV's internal threading, once per process. Never raises."""
    global _configured
    if _configured:
        return
    _configured = True
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        # No cv2, or a build without setNumThreads: nothing to configure, and a failure here must
        # not stop the program -- the worst case is the pre-fix behaviour, not a dead app.
        pass


def opencv_threads() -> int:
    """How many threads OpenCV will use internally.

    After `configure_opencv` this is <= 1 -- OpenCV reports 1 for "sequential, the calling thread
    only, no internal pool", which is the arrangement that stops the nested parallelism from
    corrupting the allocator. It is NOT always 0: `getNumThreads()` returns 1 on a TBB build even
    after `setNumThreads(0)`, and 1 is the safe state, so the tests assert <= 1 rather than == 0.
    """
    try:
        import cv2

        return int(cv2.getNumThreads())
    except Exception:
        return -1

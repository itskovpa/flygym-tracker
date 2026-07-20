"""Watch the drum flip and learn one marker template per face. The step that makes face IDs REAL.

WHY THIS EXISTS (read before changing anything here). The rig's drum flips ~180 degrees and shows
BOTH faces for roughly equal time (DESIGN.md section 2). Telling the two apart is
`marker_band.MarkerBandDetector`'s job, and it does it well -- but only once it has been shown one
template per face, and a template can only be registered from a frame ALREADY KNOWN to show that
face. That bootstrap is the whole problem this module solves.

THE REGRESSION THIS EXISTS FOR. The hand-drawing flow
(`calibration.build_two_face_calibration_from_polygons`) writes a 32-vial two-face bundle whose
``marker`` dict carries no templates at all -- just ``{"source": "live_vial_selector", ...}``. A
run started from such a bundle had nothing to identify faces with, so every stationary onset fell
through to the default face and a real 3-day event log reads, over and over:

    marker_absent,defaulted to face A

Face B's 16 vials were in the calibration, were measured, and were all recorded as face A. Half
the experiment's identities were wrong and nothing in the output said so. Drawing the vials and
learning the faces are therefore ONE session, not two: `live_vial_selector.load_or_select_vials`
runs this immediately after the polygons are saved.

HOW A FACE IS RECOGNISED AS NEW
-------------------------------
Walk the drum's dwells in time order (`adaptive_rotation.AdaptiveRotationDetector` says which
frames are stationary; nothing is ever learned mid-rotation, where the band smears):

  * the first readable dwell defines the first face -- there is nothing to compare it against;
  * every later dwell is scored against the faces already registered with
    `MarkerBandDetector.score_faces`. A dwell whose best score falls below the detector's own
    ``params.min_score`` matches NOTHING known, so it is a NEW face. A dwell that clears
    ``min_score`` with at least ``params.min_margin`` over the runner-up is a face already seen.

Both cuts are the detector's own published thresholds -- this module invents none of its own, so
re-tuning `MarkerBandParams` for a different rig automatically re-tunes the learning step too.

MEASURED ON THE REAL RIG (`Good Markers.avi`, 1280 frames, 1280x1024, 30.27 fps). The dwell walk
above finds 14 dwells and learns exactly 2 faces, from dwell 0 (face A) and dwell 1 (face B).
Dwell 1's best score against the only face then registered is **-0.2415**, against a ``min_score``
of 0.45 -- so "this is a face I have never seen" is not a marginal call, it is a gap of 0.69.
Every later dwell matches a known face at **0.918..0.927** with a margin of **1.14..1.18** over the
runner-up. Feeding the learned detector back over all 943 stationary frames of the clip:
**943/943 identified, 0 unreadable, 14/14 dwells labelled consistently at 100%**.

WHAT IT MUST NEVER DO
---------------------
`calib_faces/` holds polygons the rig owner drew BY HAND, one click per vertex. That is
irreplaceable work. Learning faces ADDS marker templates to an existing bundle and touches
nothing else -- see `calibration.attach_face_templates`, which edits the saved JSON surgically
(it never rebuilds a `VialROI`) precisely so the vial geometry stays byte-identical.

The split is the same as `live_vial_selector` / `roi_editor`, for the same reason: `FaceLearner`
is pure -- frames in, decisions out, no window, no I/O -- so tests never need a display.
`learn_faces` is a thin driver that pumps cv2 events into it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from flygym_tracker.adaptive_rotation import AdaptiveRotationDetector
from flygym_tracker.frame_source import FrameSource
from flygym_tracker.gui_support import require_gui
from flygym_tracker.live_vial_selector import (
    PANEL_WIDTH,
    _text,
    _window_is_gone,
    decode_key,
    place_window_on_screen,
    screen_view_limit,
    view_scale,
)
from flygym_tracker.marker_band import MarkerBandDetector
from flygym_tracker.types import TrackState

#: Face names handed out to distinct faces in the order they are first seen. Matches
#: `calibration.FACE_NAMES` and the ("A", "B") default of the hand-drawing flow, so a bundle's
#: face keys and its learned templates line up without any translation.
FACE_NAMES: Tuple[str, ...] = ("A", "B")

DEFAULT_WINDOW = "Learn drum faces"
#: `cv2.waitKeyEx` timeout per iteration -- the frame rate of the preview (~30 fps).
POLL_MS = 33

COLOR_TEXT = (255, 255, 255)
COLOR_LEARNED = (80, 220, 80)
COLOR_WAITING = (0, 235, 255)
COLOR_PANEL = (26, 22, 20)
COLOR_RULE = (70, 64, 60)
COLOR_LABEL = (150, 150, 150)
COLOR_MOVING = (60, 200, 255)


# ==========================================================================================
# State (pure -- no cv2 window, no I/O)
# ==========================================================================================
#: What `FaceLearner.observe` decided about one frame. `None` = nothing decided yet.
Decision = Optional[str]

#: A dwell was used to register a face that had never been seen before.
LEARNED = "learned"
#: A dwell was recognised as a face already registered.
MATCHED = "matched"
#: The frame scored above `min_score` against two faces at once, too close to call. Only
#: reachable while at least one face slot is still open -- once every face has a template the
#: learner is `done` and stops deciding, and telling faces apart becomes `identify_face`'s job.
AMBIGUOUS = "ambiguous"


@dataclass
class FaceLearner:
    """Learns one marker template per drum face from a live (or replayed) stream. Pure.

    Feed it every frame in acquisition order with `observe`; it runs the rotation detector itself
    so that "is the drum stationary right now" is answered from the same signal the tracker uses,
    not from a second, differently-tuned rule that could disagree with it.

    ONE DECISION PER DWELL, and only from a settled one. A dwell is `settle_frames` consecutive
    STATIONARY frames; the first frame that yields a confident decision closes the dwell and the
    rest of it is ignored. A frame whose band is unreadable decides nothing and does NOT close the
    dwell, so a single smeared or occluded frame costs one frame rather than a whole flip.
    """

    detector: MarkerBandDetector = field(default_factory=MarkerBandDetector)
    rotation: AdaptiveRotationDetector = field(default_factory=AdaptiveRotationDetector)
    n_faces: int = 2
    face_names: Sequence[str] = FACE_NAMES
    #: STATIONARY frames a dwell must hold before it is trusted enough to register FROM. The
    #: templates seed the whole run, so a marginal frame here would poison every later
    #: identification; 5 frames is ~0.17 s at 30 fps, i.e. free. (Measured on `Good Markers.avi`
    #: the caution is not strictly needed -- identification was correct on 943/943 stationary
    #: frames including the earliest of each dwell -- but a template is written once and read
    #: for days.)
    settle_frames: int = 5

    #: Faces registered so far, in the order they were first seen.
    learned: List[str] = field(default_factory=list)
    #: Dwells that produced a decision, as ``(face, score, margin, decision)`` in time order.
    dwells: List[Tuple[str, float, float, str]] = field(default_factory=list)
    #: True once the operator asked to stop early (q / ESC / closed window).
    aborted: bool = False

    state: TrackState = TrackState.UNKNOWN
    frames_seen: int = 0
    #: Frames whose marker band could not be read at all, over the whole session.
    unreadable_frames: int = 0
    _stationary_run: int = 0
    _dwell_closed: bool = False
    _last_decision: Decision = None
    #: Settled frames in a row whose marker band could not be read at all.
    #:
    #: MEASURED ON THE RIG, and it is why this exists. A session ran for 45 s with the drum
    #: turning: 4 dwells settled, face B was offered 520 times, and the band was unreadable on
    #: 661 of 661 stationary frames -- because the exposure was 10 ms and, in a controlled sweep,
    #: this band only becomes readable at 40 ms. The whole time the status line said "waiting for
    #: the drum to show the next one", so the operator kept turning a drum that was never going to
    #: help. "I cannot see the band" and "I have not been shown a new face" are completely
    #: different problems with completely different fixes, and the surface said the wrong one.
    _unreadable_run: int = 0

    def __post_init__(self) -> None:
        if self.n_faces < 1:
            raise ValueError("n_faces must be >= 1")
        if self.n_faces > len(self.face_names):
            raise ValueError(
                "n_faces=%d but only %d face name(s) were supplied (%s)"
                % (self.n_faces, len(self.face_names), ", ".join(self.face_names))
            )
        if self.settle_frames < 1:
            raise ValueError("settle_frames must be >= 1")
        # A detector handed in with templates already on it (e.g. re-learning one missing face)
        # counts those as learned, so `done` and the progress line tell the truth from frame 0.
        for name in self.face_names:
            if name in self.detector.templates and name not in self.learned:
                self.learned.append(name)

    # -- progress -------------------------------------------------------------------------
    @property
    def done(self) -> bool:
        """True once every face has a template. The driver stops here."""
        return len(self.learned) >= self.n_faces

    @property
    def moving(self) -> bool:
        return self.state == TrackState.ROTATING

    #: Consecutive unreadable settled frames after which the status line stops saying "waiting for
    #: the drum" and starts saying "I cannot see the band". ~1 s at 20 fps: long enough that a
    #: single smeared frame says nothing, short enough that the operator is not left turning a drum
    #: for a minute for no reason.
    UNREADABLE_RUN_TO_REPORT = 20

    def _note_unreadable(self) -> None:
        self.unreadable_frames += 1
        self._unreadable_run += 1

    @property
    def band_unreadable(self) -> bool:
        """True when settled frames keep arriving with no readable marker band in them."""
        return self._unreadable_run >= self.UNREADABLE_RUN_TO_REPORT

    def status_line(self) -> str:
        """The one line that tells the operator whether anything is happening.

        This step takes 10-20 s of real rotation and shows a near-static picture the whole time,
        so a window that said nothing would be indistinguishable from a hung one. It always names
        both how far along the learning is and what the drum is doing right now.

        AN UNREADABLE BAND OUTRANKS EVERYTHING ELSE HERE, because it is the one state in which the
        operator's obvious action is the wrong one. Measured on the rig: 45 s with the drum
        turning, 4 dwells settled, face B offered 520 times, band unreadable on 661 of 661
        stationary frames -- and the line read "waiting for the drum to show the next one" the
        whole time. It was not waiting for the drum. It could not see.
        """
        if self.band_unreadable and not self.done:
            return ("the marker band cannot be read - the picture is probably too dark "
                    "(%d frames in a row). Turning the drum will not help; raise the exposure or "
                    "the illumination until the two lit strips are clearly visible."
                    % self._unreadable_run)
        if self.done:
            return "all %d faces learned (%s) - finishing" % (
                len(self.learned), ", ".join(self.learned))
        if not self.learned:
            head = "waiting for the drum to turn... 0 of %d faces learned" % self.n_faces
        else:
            head = "face %d of %d learned (%s) - waiting for the drum to show the next one" % (
                len(self.learned), self.n_faces, ", ".join(self.learned))
        if self.moving:
            return head + "  [drum turning]"
        if self._last_decision == AMBIGUOUS:
            return head + "  [this view matches two faces at once - waiting]"
        return head + "  [drum still]"

    # -- the loop -------------------------------------------------------------------------
    def observe(self, frame_gray: np.ndarray) -> Decision:
        """Feed one frame. Returns what it decided, or None if it decided nothing.

        Nothing is ever learned outside a settled dwell: mid-rotation the marker band is smeared
        across columns and the profile it yields describes no face at all.
        """
        self.frames_seen += 1
        self.state = self.rotation.update(frame_gray)

        if self.state != TrackState.STATIONARY:
            # SETTLING and UNKNOWN are not moving either, but they are not settled; only a real
            # rotation opens a NEW dwell, so a momentary demotion cannot re-decide the same one.
            if self.state == TrackState.ROTATING:
                self._dwell_closed = False
            self._stationary_run = 0
            self._last_decision = None
            return None

        self._stationary_run += 1
        if self._dwell_closed or self._stationary_run < self.settle_frames or self.done:
            return None

        decision = self._decide(frame_gray)
        self._last_decision = decision
        return decision

    def _decide(self, gray: np.ndarray) -> Decision:
        """Classify one settled frame against the faces learned so far, registering if it is new."""
        if not self.learned:
            # Nothing to compare against: the first readable dwell simply DEFINES the first face.
            name = self._next_free_name()
            try:
                self.detector.register_face(gray, name)
            except ValueError:
                self._note_unreadable()
                return None            # no band in this frame; the next one may be cleaner
            self._unreadable_run = 0
            return self._record(name, 1.0, float("nan"), LEARNED)

        scores = self.detector.score_faces(gray)
        if not scores:
            self._note_unreadable()
            return None
        self._unreadable_run = 0
        order = sorted(scores, key=lambda f: -scores[f])
        best = order[0]
        best_score = float(scores[best])
        margin = best_score - float(scores[order[1]]) if len(order) > 1 else float("nan")

        if best_score < self.detector.params.min_score:
            # Matches nothing known. That is exactly what the far side of the drum looks like.
            # There is always a free name here: `observe` stops deciding once every face has a
            # template, so this is only ever reached with `len(learned) < n_faces`.
            #
            # The name is the first UNUSED one, not `face_names[len(learned)]`. Those are the same
            # thing for a session that starts from nothing, but not for one seeded with a detector
            # that already knows a later face -- re-learning only face "A" of a bundle that still
            # has "B" would otherwise keep picking "B" and overwrite the good template forever,
            # never finishing. This module's docstring offers that case, so it has to work.
            name = self._next_free_name()
            try:
                self.detector.register_face(gray, name)
            except ValueError:         # pragma: no cover - score_faces already read the band
                self._note_unreadable()
                return None
            return self._record(name, best_score, margin, LEARNED)

        if len(self.learned) > 1 and margin < self.detector.params.min_margin:
            # Two faces score alike. `identify_face` would refuse this frame, and so does this.
            return AMBIGUOUS

        return self._record(best, best_score, margin, MATCHED)

    def _next_free_name(self) -> str:
        """The first face name that has no template yet. Never called when all of them do."""
        return next(n for n in self.face_names[:self.n_faces] if n not in self.learned)

    def _record(self, face: str, score: float, margin: float, decision: str) -> str:
        if decision == LEARNED and face not in self.learned:
            self.learned.append(face)
        self.dwells.append((face, float(score), float(margin), decision))
        self._dwell_closed = True
        return decision

    def abort(self) -> None:
        """Stop early, keeping whatever has been learned so far."""
        self.aborted = True

    # -- result ---------------------------------------------------------------------------
    def result(self) -> "FaceLearnResult":
        return FaceLearnResult(
            detector=self.detector,
            learned=list(self.learned),
            aborted=self.aborted,
            dwells=list(self.dwells),
            frames_seen=self.frames_seen,
            unreadable_frames=self.unreadable_frames,
        )


@dataclass
class FaceLearnResult:
    """What a learning session produced. `complete` is the only thing worth branching on."""
    detector: MarkerBandDetector
    learned: List[str]
    aborted: bool
    dwells: List[Tuple[str, float, float, str]]
    frames_seen: int = 0
    unreadable_frames: int = 0

    @property
    def complete(self) -> bool:
        """True if enough faces were learned for `identify_face` to ever return a face.

        Two is not a policy choice: `MarkerBandDetector.identify_face` returns None with fewer
        than two templates, because with one there is nothing to discriminate against.
        """
        return len(self.learned) >= 2

    def summary(self) -> str:
        """One line for the terminal, with the numbers that say whether to trust the templates."""
        if not self.learned:
            return "no faces were learned (%d frames seen)" % self.frames_seen
        scored = [(f, s, m, d) for (f, s, m, d) in self.dwells if d == MATCHED]
        bit = ""
        if scored:
            bit = "; %d dwell(s) re-identified, score %.3f..%.3f, margin %.3f..%.3f" % (
                len(scored), min(s for _, s, _, _ in scored), max(s for _, s, _, _ in scored),
                min(m for _, _, m, _ in scored), max(m for _, _, m, _ in scored))
        return "learned %d face(s): %s (%d dwell(s) over %d frames%s)" % (
            len(self.learned), ", ".join(self.learned), len(self.dwells), self.frames_seen, bit)


# ==========================================================================================
# Rendering (no window -- returns the canvas)
# ==========================================================================================
def render_progress(image: np.ndarray, learner: FaceLearner, scale: float = 1.0,
                    panel_width: int = PANEL_WIDTH) -> np.ndarray:
    """The canvas shown while learning: the live frame beside a panel of what is known so far.

    Deliberately the same shape as `live_vial_selector.render_frame` -- the operator has just
    finished drawing in that window and should not have to re-learn where to look.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    colour = COLOR_MOVING if learner.moving else (
        COLOR_LEARNED if learner.done else COLOR_WAITING)
    _text(vis, learner.status_line(), (12, 26), colour, 0.6)

    panel = np.full((vis.shape[0], panel_width, 3), COLOR_PANEL, np.uint8)
    _draw_panel(panel, learner)
    return np.hstack([vis, panel])


def _draw_panel(panel: np.ndarray, learner: FaceLearner) -> None:
    y = 34
    _text(panel, "LEARNING THE DRUM FACES", (16, y), COLOR_TEXT, 0.6)
    y += 14
    cv2.line(panel, (16, y), (panel.shape[1] - 16, y), COLOR_RULE, 1)
    y += 30

    for i in range(learner.n_faces):
        known = i < len(learner.learned)
        name = learner.learned[i] if known else "?"
        mark = "[x]" if known else "[ ]"
        _text(panel, "%s face %d of %d   %s" % (mark, i + 1, learner.n_faces, name), (16, y),
              COLOR_LEARNED if known else COLOR_LABEL, 0.55)
        y += 26

    y += 10
    cv2.line(panel, (16, y), (panel.shape[1] - 16, y), COLOR_RULE, 1)
    y += 28
    for line in (
        "The drum must TURN for this to",
        "finish: each face is learned the",
        "first time it is shown and held",
        "still. Nothing is learned while",
        "it is moving.",
        "",
        "drum: %s" % ("TURNING" if learner.moving else "still"),
        "frames seen: %d" % learner.frames_seen,
        "dwells decided: %d" % len(learner.dwells),
    ):
        if line:
            _text(panel, line, (16, y), COLOR_LABEL, 0.48)
        y += 20

    y = panel.shape[0] - 30
    _text(panel, "q / ESC  skip - keeps your vials", (16, y), COLOR_TEXT, 0.5)


# ==========================================================================================
# Driver (thin -- pumps cv2 events into FaceLearner and nothing else)
# ==========================================================================================
def learn_faces(
    source: FrameSource,
    n_faces: int = 2,
    face_names: Sequence[str] = FACE_NAMES,
    window: str = DEFAULT_WINDOW,
    detector: Optional[MarkerBandDetector] = None,
    learner: Optional[FaceLearner] = None,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    max_frames: Optional[int] = None,
    poll_ms: int = POLL_MS,
    max_view: Optional[Tuple[int, int]] = None,
    show: bool = True,
) -> FaceLearnResult:
    """Watch the drum until every face has been shown once, learning a template from each.

    The window is not decoration. The step needs the drum to complete at least one flip, which
    takes 10-20 s of real rotation, and the picture barely changes in that time -- so the panel
    reports what has been learned and whether the drum is currently moving, and q/ESC (or closing
    the window) aborts. AN ABORT IS SAFE: this function never writes anything. The caller keeps
    whatever it already had, which is the operator's hand-drawn polygons.

    Args:
        source: the live camera (or a video, for a dry run). Opened here; CLOSING IT IS THE
            CALLER'S JOB -- same contract as `live_vial_selector.select_vials_live`, so one
            session can draw and then learn without releasing the camera in between.
        n_faces: how many distinct faces to learn (2 for this rig's drum).
        face_names: names handed out in first-seen order; must match the bundle's face keys.
        detector: an existing detector to add templates to. A fresh one by default.
        learner: a fully built `FaceLearner`, when the caller wants non-default settings.
        on_frame: called with every frame read, for a caller that wants to keep the last one.
        max_frames: give up after this many frames (None = until done or aborted). A run that
            never sees a flip would otherwise wait forever on a rig whose drum is not turning.
        show: False runs the whole loop headlessly (no window, no key handling) -- for replaying
            a recorded clip in a test.

    Returns:
        A `FaceLearnResult`. Check `.complete` before saving anything: fewer than two faces
        cannot identify anything, and writing such a bundle would look like success.
    """
    if show:
        require_gui("Learning the drum faces")

    learner = learner or FaceLearner(
        detector=detector or MarkerBandDetector(), n_faces=n_faces, face_names=face_names)
    source.open()   # idempotent on both FrameSource implementations
    max_view = max_view or (screen_view_limit() if show else None)

    if show:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    print(_startup_banner(learner))

    scale = 1.0
    placed = False
    frames = 0
    try:
        while not learner.done and not learner.aborted:
            if max_frames is not None and frames >= max_frames:
                break
            frame = source.read()
            if frame is None:
                break                       # video ran out; the camera never does
            frames += 1
            image = frame.image
            if on_frame is not None:
                on_frame(image)
            learner.observe(image)

            if not show:
                continue
            scale = view_scale((image.shape[1], image.shape[0]), max_view)
            cv2.imshow(window, render_progress(image, learner, scale))
            if not placed:
                # Only now does the window have its real size, so only now can it be positioned.
                place_window_on_screen(window, max_view)
                placed = True
            if decode_key(cv2.waitKeyEx(poll_ms)) in ("q", "esc"):
                learner.abort()
            elif _window_is_gone(window):
                learner.abort()             # closing with the X counts as "skip"
    finally:
        if show:
            try:
                cv2.destroyWindow(window)
                cv2.waitKey(1)
            except Exception:
                pass

    result = learner.result()
    print("  " + result.summary())
    return result


def _startup_banner(learner: FaceLearner) -> str:
    return (
        "\nLearning the drum faces - the drum needs to turn at least once.\n"
        "  %d face(s) to learn. Each is learned the first time it is shown and held still.\n"
        "  Press q or ESC to skip; your drawn vials are kept either way.\n"
        % learner.n_faces
    )

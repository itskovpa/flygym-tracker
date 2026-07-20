"""Learning one marker template per drum face, and the run path that has to USE them.

THE BUG THESE TESTS EXIST FOR. The rig's drum flips 180 degrees and shows both faces for equal
time, and for a while only face A ever reached the output. Three things stacked up:

  1. `config/flygym_rig.yaml` said ``markers.enabled: false`` under a comment claiming the rig
     "rocks (one face), it does not flip to a back side" -- which is not what the rig does;
  2. `cli._build_marker_detector` built the generic `markers.MarkerDetector` (which reads
     ``marker["signature"]``) instead of the `marker_band.MarkerBandDetector` validated at 43/43
     on real footage (which reads ``marker["band_templates"]``). `marker_detector_from_calibration`
     already did the right thing and was called from NOWHERE in ``src/`` -- only from a test;
  3. the hand-drawing flow wrote no templates at all, so even fixing 1 and 2 could not have helped.

Every face-ID unit test passed in isolation the whole time. That is the actual gap, so the tests
here are deliberately weighted towards the SEAMS: what `cli` builds for a real bundle, what the
pipeline does when identification fails, and one end-to-end run over a flipping sequence that
asserts both faces reach the CSV.

Frames are synthetic and deterministic (`make_rig_frame`), reproducing only what the detectors
actually depend on: two bright strips with one opaque sticker per column alternating up/down, a
bright stage along the bottom, and 16 vial rectangles. Face "B" is the same frame with the two
strips exchanged, which is what a 180 degree flip about the horizontal axis produces.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest

from flygym_tracker.calibration import (
    attach_face_templates,
    build_two_face_calibration_from_polygons,
    calibration_band_faces,
    calibration_signature_faces,
    load_calibration,
    marker_detector_from_calibration,
    save_calibration,
)
from flygym_tracker.cli import _build_marker_detector, face_id_readiness
from flygym_tracker.config import load_config
from flygym_tracker.face_learning import (
    AMBIGUOUS,
    LEARNED,
    MATCHED,
    FaceLearner,
    learn_faces,
)
from flygym_tracker.frame_source import FrameSource, VideoFileSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.marker_band import MarkerBandDetector
from flygym_tracker.markers import MarkerDetector
from flygym_tracker.pipeline import TrackerPipeline
from flygym_tracker.types import Frame

# =============================================================================================
# A synthetic rig frame: marker band + 16 vials + illuminated stage
# =============================================================================================
H, W = 700, 800
UPPER_STRIP = (300, 340)          # inclusive rows of the upper LED strip
LOWER_STRIP = (365, 405)          # inclusive rows of the lower LED strip
BAND = (270, 435)                 # dark hardware surrounding both strips
STAGE_ROW = H - 60                # the illuminated stage: brightest thing in the frame
N_COLS = 8
VIAL_X0, VIAL_PITCH = 80, 80
UPPER_VIALS = (150, 250)          # inclusive rows of the upper vial row
LOWER_VIALS = (450, 550)
BG, DARK, BRIGHT, TUBE = 30, 12, 250, 200

#: 16 vials as 4-point polygons, in the order a hand would draw them: upper row left to right
#: (local ids 1..8), then lower row (9..16). Matches DESIGN.md section 2's canonical numbering.
def vial_polygons() -> list:
    polys = []
    for y0, y1 in (UPPER_VIALS, LOWER_VIALS):
        for c in range(N_COLS):
            x0 = VIAL_X0 + c * VIAL_PITCH + 8
            x1 = VIAL_X0 + (c + 1) * VIAL_PITCH - 9
            polys.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return polys


def make_rig_frame(swap: bool = False, flies: dict | None = None) -> np.ndarray:
    """One synthetic rig frame.

    Args:
        swap: emit the OTHER face -- the same sticker pattern with the upper and lower strip
            contents exchanged, which is what a 180 degree flip about the drum's horizontal
            rotation axis produces (see `marker_band`'s module docstring for the real-data
            measurement that establishes this).
        flies: ``{vial_index: n}`` -- paint `n` dark blobs in that vial (0-based, in the same
            order as `vial_polygons`). This is the per-vial activity signal.
    """
    img = np.full((H, W), BG, np.uint8)
    img[STAGE_ROW:, :] = 255
    img[BAND[0]:BAND[1] + 1, :] = DARK

    for c in range(N_COLS):
        x0, x1 = VIAL_X0 + c * VIAL_PITCH, VIAL_X0 + (c + 1) * VIAL_PITCH - 1
        # One opaque sticker per column, alternating up/down; the LIT run for column c is
        # therefore on the opposite strip. `swap` exchanges the two.
        for strip, lit in ((UPPER_STRIP, (c % 2 == 1) != swap),
                           (LOWER_STRIP, (c % 2 == 0) != swap)):
            if lit:
                img[strip[0]:strip[1] + 1, x0:x1 + 1] = BRIGHT

    for i, poly in enumerate(vial_polygons()):
        (x0, y0), (x1, _), (_, y1), _ = poly
        img[y0:y1 + 1, x0:x1 + 1] = TUBE
        for k in range(int((flies or {}).get(i, 0))):
            cy = y0 + 12 + k * 9
            img[cy:cy + 6, x0 + 6:x0 + 6 + 20] = 40      # a fly silhouette

    return img


def rotating(frame: np.ndarray, k: int) -> np.ndarray:
    """The scene displaced bodily -- what the rotation detector keys on (it measures global
    DISPLACEMENT, not pixel-change magnitude; see `adaptive_rotation`)."""
    return np.roll(frame, (k + 1) * 9, axis=1)


class Seq(FrameSource):
    """A fixed list of frames, replayed once."""

    def __init__(self, frames, fps: float = 20.0):
        self._frames, self._i, self._fps = list(frames), 0, float(fps)

    def open(self):
        self._i = 0

    def read(self):
        if self._i >= len(self._frames):
            return None
        fr = Frame(image=self._frames[self._i], index=self._i, t_monotonic=self._i / self._fps,
                   t_wall_iso="2026-07-19T00:00:%02d" % (int(self._i / self._fps) % 60))
        self._i += 1
        return fr

    def close(self):
        pass

    @property
    def fps(self):
        return self._fps

    @property
    def frame_size(self):
        return (W, H)


def flip_sequence(dwell: int = 40, turn: int = 12, faces: str = "ABA") -> list:
    """Dwell on each face in `faces` in turn, with a bodily displacement between them."""
    frames: list = []
    for i, face in enumerate(faces):
        swap = face == "B"
        if i:
            base = make_rig_frame(swap=swap)
            frames += [rotating(base, k) for k in range(turn)]
        for j in range(dwell):
            # A blob count that changes every frame gives the activity meter something to find.
            frames.append(make_rig_frame(swap=swap, flies={0: 1 + (j % 3), 8: 1 + (j % 2)}))
    return frames


# =============================================================================================
# FaceLearner -- the pure core
# =============================================================================================
def _feed(learner: FaceLearner, frames) -> list:
    return [learner.observe(f) for f in frames]


def test_a_learner_with_no_templates_yet_learns_the_first_settled_dwell_as_face_a():
    learner = FaceLearner()
    decisions = _feed(learner, [make_rig_frame()] * 20)

    assert learner.learned == ["A"]
    assert decisions.count(LEARNED) == 1
    assert "A" in learner.detector.templates


def test_a_dwell_that_matches_nothing_already_registered_is_a_new_face():
    learner = FaceLearner()
    _feed(learner, [make_rig_frame()] * 20)
    _feed(learner, [rotating(make_rig_frame(swap=True), k) for k in range(12)])
    decisions = _feed(learner, [make_rig_frame(swap=True)] * 20)

    assert learner.learned == ["A", "B"]
    assert decisions.count(LEARNED) == 1
    assert learner.done


def test_a_dwell_that_matches_a_registered_face_is_that_face_again_not_a_third_one():
    learner = FaceLearner(n_faces=3, face_names=("A", "B", "C"))
    _feed(learner, [make_rig_frame()] * 20)
    _feed(learner, [rotating(make_rig_frame(swap=True), k) for k in range(12)])
    _feed(learner, [make_rig_frame(swap=True)] * 20)
    _feed(learner, [rotating(make_rig_frame(), k) for k in range(12)])
    decisions = _feed(learner, [make_rig_frame()] * 20)

    assert learner.learned == ["A", "B"], "face A came back; it must not be registered twice"
    assert decisions.count(MATCHED) == 1
    assert decisions.count(LEARNED) == 0


def test_the_new_face_cut_is_the_detectors_own_min_score_not_a_number_of_our_own():
    """Re-tuning `MarkerBandParams` must re-tune the learning step with it.

    The threshold is moved to just ABOVE what a re-shown face actually scores, so the very same
    footage that was recognised as face A a moment ago now reads as something never seen. That
    only changes the outcome if `params.min_score` really is the number being consulted -- a
    hard-coded cut of our own would ignore it entirely.
    """
    learner = FaceLearner(n_faces=3, face_names=("A", "B", "C"))
    _feed(learner, [make_rig_frame()] * 20)
    _feed(learner, [rotating(make_rig_frame(swap=True), k) for k in range(12)])
    _feed(learner, [make_rig_frame(swap=True)] * 20)
    assert learner.learned == ["A", "B"]

    # What face A genuinely scores against the templates just learned. On these synthetic frames
    # it is an exact 1.0 (identical pixels); on real rig footage the same quantity measures
    # 0.918..0.927, comfortably over the shipped 0.45 cut.
    scored = max(learner.detector.score_faces(make_rig_frame()).values())
    assert scored >= learner.detector.params.min_score, "a re-shown face must match by default"

    learner.detector.params.min_score = scored + 0.01
    _feed(learner, [rotating(make_rig_frame(), k) for k in range(12)])
    decisions = _feed(learner, [make_rig_frame()] * 20)

    assert decisions.count(LEARNED) == 1
    assert learner.learned == ["A", "B", "C"], "raising the cut did not change what counts as new"


def test_two_faces_that_score_alike_are_refused_rather_than_guessed_between():
    """A frame both templates match is exactly what `identify_face` abstains on; so does this."""
    learner = FaceLearner(n_faces=3, face_names=("A", "B", "C"))
    learner.detector.params.min_margin = 1.9      # nothing can clear a margin this wide
    _feed(learner, [make_rig_frame()] * 20)
    _feed(learner, [rotating(make_rig_frame(swap=True), k) for k in range(12)])
    _feed(learner, [make_rig_frame(swap=True)] * 20)
    _feed(learner, [rotating(make_rig_frame(), k) for k in range(12)])
    decisions = _feed(learner, [make_rig_frame()] * 20)

    assert AMBIGUOUS in decisions
    assert learner.learned == ["A", "B"], "an unresolvable frame must not invent face C"


def test_nothing_is_ever_learned_while_the_drum_is_turning():
    """Mid-rotation the marker band is smeared across columns and describes no face at all."""
    learner = FaceLearner()
    decisions = _feed(learner, [rotating(make_rig_frame(), k) for k in range(40)])

    assert learner.learned == []
    assert set(decisions) == {None}


def test_one_dwell_yields_at_most_one_decision():
    learner = FaceLearner()
    decisions = _feed(learner, [make_rig_frame()] * 60)

    assert len([d for d in decisions if d is not None]) == 1
    assert len(learner.dwells) == 1


def test_a_dwell_shorter_than_settle_frames_is_not_registered_from():
    """Templates seed the whole run, so they are only taken from a drum that has actually stopped."""
    learner = FaceLearner(settle_frames=25)
    _feed(learner, [make_rig_frame()] * 12)

    assert learner.learned == []


def test_aborting_keeps_what_was_already_learned():
    learner = FaceLearner()
    _feed(learner, [make_rig_frame()] * 20)
    learner.abort()

    result = learner.result()
    assert result.aborted is True
    assert result.learned == ["A"]
    assert result.complete is False, "one face cannot discriminate against anything"


def test_a_detector_handed_in_with_templates_counts_them_as_already_learned():
    det = MarkerBandDetector()
    det.register_face(make_rig_frame(), "A")
    det.register_face(make_rig_frame(swap=True), "B")

    learner = FaceLearner(detector=det)
    assert learner.learned == ["A", "B"]
    assert learner.done


def test_re_learning_only_the_missing_face_fills_the_gap_instead_of_overwriting_the_good_one():
    """Seeding with a detector that already knows a LATER face must still learn the earlier one.

    Naming the next face positionally (``face_names[len(learned)]``) looks right and is right for
    a session that starts from nothing -- but a detector seeded with only "B" would then keep
    choosing "B" again, overwrite the one good template on every dwell, and never finish.
    """
    seeded = MarkerBandDetector()
    seeded.register_face(make_rig_frame(swap=True), "B")     # B known, A missing
    learner = FaceLearner(detector=seeded, n_faces=2)

    assert learner.learned == ["B"]
    assert not learner.done

    before = seeded.templates["B"]
    _feed(learner, [make_rig_frame()] * 20)                  # show it face A

    assert sorted(learner.learned) == ["A", "B"]
    assert learner.done
    assert np.array_equal(seeded.templates["B"][0], before[0]), "the good template was overwritten"
    assert seeded.identify_face(make_rig_frame()) == "A"
    assert seeded.identify_face(make_rig_frame(swap=True)) == "B"


def test_the_status_line_always_says_how_far_along_it_is_and_what_the_drum_is_doing():
    """The window shows a near-static picture for 10-20 s; a silent one reads as hung."""
    learner = FaceLearner()
    assert "0 of 2" in learner.status_line()

    _feed(learner, [make_rig_frame()] * 20)
    assert "face 1 of 2 learned" in learner.status_line()

    _feed(learner, [rotating(make_rig_frame(swap=True), k) for k in range(12)])
    assert "drum turning" in learner.status_line()

    _feed(learner, [make_rig_frame(swap=True)] * 20)
    assert "all 2 faces learned" in learner.status_line()


def test_the_progress_canvas_is_built_without_a_window_and_shows_the_frame_beside_a_panel():
    """The pure/driver split exists so no test needs a display -- this is the half that proves it."""
    from flygym_tracker.face_learning import PANEL_WIDTH, render_progress

    learner = FaceLearner()
    _feed(learner, [make_rig_frame()] * 20)
    canvas = render_progress(make_rig_frame(), learner, scale=1.0)

    assert canvas.shape == (H, W + PANEL_WIDTH, 3)
    assert canvas.dtype == np.uint8
    # the panel really was drawn on, rather than left as flat background
    assert len(np.unique(canvas[:, W:].reshape(-1, 3), axis=0)) > 1


def test_learn_faces_driver_runs_headlessly_and_stops_as_soon_as_both_faces_are_known():
    frames = flip_sequence(dwell=25, turn=12, faces="ABAB")
    result = learn_faces(Seq(frames), n_faces=2, show=False)

    assert result.learned == ["A", "B"]
    assert result.complete
    assert result.frames_seen < len(frames), "it must stop at the second face, not run the clip out"
    assert result.detector.can_identify()


def test_learn_faces_gives_up_at_max_frames_rather_than_waiting_on_a_drum_that_never_turns():
    result = learn_faces(Seq([make_rig_frame()] * 500), n_faces=2, show=False, max_frames=60)

    assert result.learned == ["A"]
    assert result.complete is False
    assert result.frames_seen <= 60


# =============================================================================================
# Merging into a bundle -- and NOT touching the hand-drawn vials
# =============================================================================================
def _bundle(tmp_path, faces=("A", "B")) -> str:
    out = str(tmp_path / "calib_faces")
    calib, masks, overlays = build_two_face_calibration_from_polygons(
        vial_polygons(), make_rig_frame(), (W, H), faces=faces)
    save_calibration(calib, masks, out, overlay=overlays)
    return out


def _learned_detector() -> MarkerBandDetector:
    det = MarkerBandDetector()
    det.register_face(make_rig_frame(), "A")
    det.register_face(make_rig_frame(swap=True), "B")
    return det


def _vials_json(path: str) -> str:
    """The vials of every face, re-serialized identically -- the thing that must not change."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return json.dumps({n: fc["vials"] for n, fc in d["faces"].items()}, sort_keys=True)


def test_learning_faces_leaves_the_hand_drawn_polygons_byte_identical(tmp_path):
    """THE THING THAT MUST NEVER BREAK.

    `calib_faces/` holds 16 polygons the rig owner drew by hand, one click per vertex. They are
    not reproducible: redrawing them is a fresh session of clicking and a fresh set of slightly
    different shapes. Adding marker templates is allowed to add; it is not allowed to renumber,
    reshape, reorder or drop a single vertex of what is already there.
    """
    out = _bundle(tmp_path)
    path = os.path.join(out, "calibration.json")

    before_raw = open(path, "rb").read()
    before_vials = _vials_json(path)
    before_masks = {n: open(os.path.join(out, n), "rb").read()
                    for n in sorted(os.listdir(out)) if n.endswith(".png")}

    attach_face_templates(out, _learned_detector())

    assert _vials_json(path) == before_vials, "the drawn vials changed"
    assert open(path, "rb").read() != before_raw, "nothing was written at all"
    assert {n: open(os.path.join(out, n), "rb").read()
            for n in sorted(os.listdir(out)) if n.endswith(".png")} == before_masks, \
        "the mask/overlay PNGs were rewritten; they are not this step's business"

    # and the shapes still survive a full load, which is what the pipeline actually reads
    reloaded = load_calibration(out)
    for name in ("A", "B"):
        assert [v.polygon for v in reloaded.faces[name].vials] == vial_polygons()
        assert [v.id for v in reloaded.faces[name].vials] == list(range(1, 17))


def test_attaching_templates_preserves_whatever_the_marker_dict_already_said(tmp_path):
    """The hand-drawing flow records its own provenance in `marker`; adding to it must not erase it."""
    out = _bundle(tmp_path)
    attach_face_templates(out, _learned_detector())

    marker = load_calibration(out).faces["A"].marker
    assert marker["source"] == "live_vial_selector"      # written by the drawing flow
    assert marker["n_vials"] == 16
    assert marker["band_templates"]


def test_attached_templates_round_trip_through_the_function_the_run_path_uses(tmp_path):
    out = _bundle(tmp_path)
    attach_face_templates(out, _learned_detector())

    rebuilt = marker_detector_from_calibration(load_calibration(out))
    assert isinstance(rebuilt, MarkerBandDetector)
    assert sorted(rebuilt.templates) == ["A", "B"]
    assert rebuilt.identify_face(make_rig_frame()) == "A"
    assert rebuilt.identify_face(make_rig_frame(swap=True)) == "B"


def test_the_detector_is_rebuilt_with_the_settings_its_templates_were_learned_with(tmp_path):
    """Profiles are only comparable to profiles extracted the same way.

    A detector rebuilt on defaults when the templates were learned on custom settings would score
    every frame against subtly different profiles and quietly lose margin.
    """
    out = _bundle(tmp_path)
    det = _learned_detector()
    det.params.min_margin = 0.42
    det.min_run_px = 17
    attach_face_templates(out, det)

    rebuilt = marker_detector_from_calibration(load_calibration(out))
    assert rebuilt.params.min_margin == 0.42
    assert rebuilt.min_run_px == 17


def test_a_face_the_bundle_does_not_contain_is_skipped_rather_than_invented(tmp_path):
    out = _bundle(tmp_path, faces=("A",))
    det = _learned_detector()          # knows A and B; the bundle only has A

    assert attach_face_templates(out, det) == ["A"]
    assert sorted(load_calibration(out).faces) == ["A"]


def test_attaching_to_a_directory_with_no_bundle_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        attach_face_templates(str(tmp_path / "nothing"), _learned_detector())


def test_attaching_a_detector_that_learned_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no face templates"):
        attach_face_templates(_bundle(tmp_path), MarkerBandDetector())


# =============================================================================================
# The seam: what `cli` actually builds
# =============================================================================================
def _config(**markers):
    return load_config(overrides={"markers": {"enabled": True, **markers}})


def test_cli_builds_the_validated_band_detector_for_a_band_template_bundle(tmp_path):
    """THE ASSERTION THAT WOULD HAVE CAUGHT THE BUG, cheaply and directly.

    Every face-ID unit test passed while `cli` built the OTHER detector, so the validated code
    was never in the run path. This pins the type that comes out of the production factory.
    """
    out = _bundle(tmp_path)
    attach_face_templates(out, _learned_detector())

    detector = _build_marker_detector(_config(), load_calibration(out))

    assert isinstance(detector, MarkerBandDetector)
    assert detector.can_identify()
    assert detector.identify_face(make_rig_frame(swap=True)) == "B"


def test_cli_still_builds_the_generic_detector_for_a_bundle_that_really_has_signatures(tmp_path):
    out = _bundle(tmp_path)
    path = os.path.join(out, "calibration.json")
    with open(path) as f:
        d = json.load(f)
    for i, name in enumerate(("A", "B")):
        d["faces"][name]["marker"] = {"signature": [float(i)] * 7}
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

    detector = _build_marker_detector(_config(), load_calibration(out))
    assert isinstance(detector, MarkerDetector)


def test_a_two_face_bundle_with_no_marker_data_at_all_says_so_loudly(tmp_path):
    """Rather than starting a multi-day run that silently produces half the data."""
    calib = load_calibration(_bundle(tmp_path))
    lines = face_id_readiness(_config(), calib)

    assert lines, "a bundle that cannot identify faces must not start a run in silence"
    text = " ".join(lines)
    assert "CANNOT identify faces" in text
    assert "attributed to face A" in text
    assert "face-learning" in text


def test_the_warning_also_calls_out_markers_being_switched_off(tmp_path):
    """Both halves of the original failure at once -- turning markers on alone would not help."""
    calib = load_calibration(_bundle(tmp_path))
    lines = face_id_readiness(load_config(overrides={"markers": {"enabled": False}}), calib)

    assert any("markers.enabled" in line for line in lines)


def test_a_single_face_bundle_never_warns_and_never_needs_a_marker(tmp_path):
    """A rig that only ever shows one side has nothing to identify; inventing a requirement for
    it would break it for no gain."""
    calib = load_calibration(_bundle(tmp_path, faces=("A",)))

    assert face_id_readiness(_config(), calib) == []
    assert face_id_readiness(load_config(overrides={"markers": {"enabled": False}}), calib) == []


def test_a_bundle_that_can_identify_faces_warns_about_nothing(tmp_path):
    out = _bundle(tmp_path)
    attach_face_templates(out, _learned_detector())

    assert face_id_readiness(_config(), load_calibration(out)) == []


def test_the_helpers_report_which_kind_of_marker_data_a_bundle_carries(tmp_path):
    out = _bundle(tmp_path)
    assert calibration_band_faces(load_calibration(out)) == []
    assert calibration_signature_faces(load_calibration(out)) == []

    attach_face_templates(out, _learned_detector())
    assert calibration_band_faces(load_calibration(out)) == ["A", "B"]
    assert calibration_signature_faces(load_calibration(out)) == []


# =============================================================================================
# Failing safe in the pipeline
# =============================================================================================
def _run(tmp_path, frames, calib, detector, name="run"):
    out = tmp_path / name
    cfg = load_config(overrides={
        "rotation": {"detector": "adaptive", "debounce_frames": 3, "min_stationary_frames": 3},
        "activity": {"pixel_threshold": 30},
        "binning": {"bin_seconds": 1},
        "markers": {"enabled": True},
    })
    logger = ActivityLogger(str(out), run_id=name, fmt="csv")
    pipe = TrackerPipeline(cfg, calib, Seq(frames), logger, marker_detector=detector,
                           clock="index", pixel_threshold=30)
    summary = pipe.run()
    events = pd.read_csv(out / "events.csv", keep_default_na=False)
    # A run that attributed nothing writes no activity rows at all -- which is the POINT of some
    # of these tests, so an absent table is an empty one here, not an error.
    tables = [pd.read_csv(out / n) for n in sorted(os.listdir(out)) if n.startswith("activity")]
    tables = [t for t in tables if not t.empty]
    activity = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["face", "vial_id", "n_stationary_frames", "motion_px_sum"])
    return pipe, summary, events, activity


class _Blind:
    """A detector that is capable in principle but never actually identifies anything."""

    def can_identify(self):
        return True

    def identify_face(self, gray):
        return None


class _OnceThenBlind:
    """Identifies face B exactly once, then goes blind -- an occluded or smeared marker."""

    def __init__(self, face="B"):
        self.face, self.calls = face, 0

    def can_identify(self):
        return True

    def identify_face(self, gray):
        self.calls += 1
        return self.face if self.calls == 1 else None


def test_a_failed_identification_keeps_the_last_known_face_instead_of_resetting_to_a(tmp_path):
    """THE EXACT BEHAVIOUR THAT PRODUCED THE BUG.

    Resetting to the default face on every unreadable onset is indistinguishable from a correct
    answer, so a whole experiment can be mislabelled without a single failure showing anywhere.
    """
    out = _bundle(tmp_path)
    attach_face_templates(out, _learned_detector())
    calib = load_calibration(out)

    detector = _OnceThenBlind("B")
    pipe, summary, events, _ = _run(tmp_path, flip_sequence(faces="ABA"), calib, detector)

    assert pipe._current_face == "B", "it fell back to face A after B had been identified"
    assert summary["faces_seen"] == ["B"]
    absent = events[events["event"] == "marker_absent"]
    assert len(absent) >= 1
    assert absent["detail"].str.contains("kept last known face B").all()


def test_before_the_first_identification_no_face_is_guessed_and_nothing_is_attributed(tmp_path):
    """A short gap at the start of a 3-day run is fine. Mislabelled vial identities are not."""
    out = _bundle(tmp_path)
    attach_face_templates(out, _learned_detector())
    calib = load_calibration(out)

    pipe, summary, events, activity = _run(tmp_path, flip_sequence(faces="AB"), calib, _Blind())

    assert pipe._current_face is None
    assert summary["faces_seen"] == []
    assert activity.empty, "activity was attributed to a face that was never identified"
    assert events[events["event"] == "marker_absent"]["detail"].str.contains(
        "not identified yet").any()


def test_a_single_face_bundle_still_defaults_with_no_detector_exactly_as_before(tmp_path):
    """No marker requirement and no new behaviour for a rig that shows one side."""
    calib = load_calibration(_bundle(tmp_path, faces=("A",)))
    pipe, summary, events, activity = _run(tmp_path, flip_sequence(faces="AA"), calib, None)

    assert pipe._face_id_required is False
    assert pipe._current_face == "A"
    assert summary["faces_seen"] == ["A"]
    assert not activity.empty
    assert events[events["event"] == "marker_absent"]["detail"].str.contains(
        "defaulted to face A").all()


def test_a_two_face_bundle_with_a_detector_that_cannot_discriminate_still_records_something(
        tmp_path):
    """Half the data beats none. The loud startup warning is what covers this case, not silence."""
    calib = load_calibration(_bundle(tmp_path))
    pipe, summary, events, activity = _run(tmp_path, flip_sequence(faces="AB"), calib,
                                           MarkerBandDetector())

    assert pipe._face_id_required is False
    assert pipe._current_face == "A"
    assert not activity.empty


def test_a_face_name_the_calibration_does_not_contain_is_refused_not_adopted(tmp_path):
    class _Wrong:
        def can_identify(self):
            return True

        def identify_face(self, gray):
            return "Z"

    calib = load_calibration(_bundle(tmp_path))
    pipe, _summary, events, activity = _run(tmp_path, flip_sequence(faces="AB"), calib, _Wrong())

    assert pipe._current_face is None
    assert activity.empty
    assert (events["event"] == "mis_registration").any()


# =============================================================================================
# END TO END -- the test whose absence let all of this survive
# =============================================================================================
def test_a_flipping_drum_produces_activity_rows_for_BOTH_faces(tmp_path):
    """The whole chain, in production order, over a sequence that actually flips.

    Draw 16 vials -> learn a template per face -> merge into the bundle -> build the detector the
    way `cli` builds it -> run `TrackerPipeline` -> both faces are in the CSV, with global vial
    ids spanning 1..16 AND 17..32.

    Every unit below this passed throughout the bug. Only a test that crosses all the seams at
    once could have caught it, because the defect was entirely in how they were wired together.
    """
    out = _bundle(tmp_path)

    # -- learn the faces the way a session does, from a clip that flips -----------------------
    learned = learn_faces(Seq(flip_sequence(dwell=25, faces="AB")), n_faces=2, show=False)
    assert learned.complete, "the learning step is the premise of everything below"
    assert attach_face_templates(out, learned.detector) == ["A", "B"]

    # -- build the detector through the PRODUCTION factory, not by hand ----------------------
    calib = load_calibration(out)
    detector = _build_marker_detector(_config(), calib)
    assert isinstance(detector, MarkerBandDetector), "cli must hand the run the validated detector"

    # -- run ---------------------------------------------------------------------------------
    pipe, summary, events, activity = _run(
        tmp_path, flip_sequence(dwell=40, faces="ABAB"), calib, detector, name="e2e")

    assert summary["n_rotations"] >= 3
    assert sorted(summary["faces_seen"]) == ["A", "B"]
    assert (events["event"] == "face_change").sum() >= 2

    # -- THE ASSERTION: both faces are in the output ------------------------------------------
    faces = set(activity["face"])
    assert faces == {"A", "B"}, f"only {faces} reached the activity table"

    ids = set(int(v) for v in activity["vial_id"])
    assert ids & set(range(1, 17)), "no face-A vial ids (1..16) in the output"
    assert ids & set(range(17, 33)), "no face-B vial ids (17..32) in the output"

    # both faces were really MEASURED, not merely listed with zero frames
    measured = activity[activity["n_stationary_frames"] > 0]
    assert set(measured["face"]) == {"A", "B"}
    for lo, hi in ((1, 17), (17, 33)):
        side = measured[measured["vial_id"].between(lo, hi - 1)]
        assert side["motion_px_sum"].sum() > 0, f"vials {lo}..{hi - 1} recorded no motion at all"


def test_the_same_run_without_templates_is_exactly_the_bug_and_only_reaches_one_face(tmp_path):
    """The counter-example, pinned so the regression cannot come back unnoticed.

    Identical footage and identical calibration, minus the learned templates: everything lands on
    face A, face B's vials 17..32 never appear, and the event log fills with `marker_absent`.
    """
    calib = load_calibration(_bundle(tmp_path))
    detector = _build_marker_detector(_config(), calib)

    _pipe, _summary, events, activity = _run(
        tmp_path, flip_sequence(dwell=40, faces="ABAB"), calib, detector, name="nobundle")

    assert set(activity["face"]) == {"A"}
    assert not set(int(v) for v in activity["vial_id"]) & set(range(17, 33))
    assert (events["event"] == "marker_absent").sum() >= 1


# =============================================================================================
# Wiring into the session (live_vial_selector)
# =============================================================================================
def test_a_two_face_bundle_without_templates_is_offered_the_learning_step(tmp_path):
    from flygym_tracker import live_vial_selector as LVS

    assert LVS.faces_need_learning(load_calibration(_bundle(tmp_path))) is True


def test_a_bundle_that_already_has_templates_is_never_re_learned(tmp_path):
    from flygym_tracker import live_vial_selector as LVS

    out = _bundle(tmp_path)
    attach_face_templates(out, _learned_detector())
    assert LVS.faces_need_learning(load_calibration(out)) is False


def test_a_single_face_bundle_is_never_offered_the_learning_step(tmp_path):
    from flygym_tracker import live_vial_selector as LVS

    assert LVS.faces_need_learning(load_calibration(_bundle(tmp_path, faces=("A",)))) is False


def test_declining_the_learning_step_keeps_the_drawn_vials_and_says_what_it_costs(tmp_path, capsys):
    from flygym_tracker import live_vial_selector as LVS

    out = _bundle(tmp_path)
    before = _vials_json(os.path.join(out, "calibration.json"))
    result = LVS.SelectionResult(polygons=vial_polygons(), calibration=load_calibration(out),
                                 reused=True, out_dir=out)

    after = LVS.learn_faces_for_bundle(Seq(flip_sequence()), result, input_fn=lambda _p: "n")

    assert after.n_vials == 16
    assert _vials_json(os.path.join(out, "calibration.json")) == before
    text = capsys.readouterr().out
    assert "NOT learned" in text and "attributed to face A" in text


def test_accepting_the_learning_step_merges_templates_and_keeps_the_vials(tmp_path):
    from flygym_tracker import live_vial_selector as LVS

    out = _bundle(tmp_path)
    before = _vials_json(os.path.join(out, "calibration.json"))
    result = LVS.SelectionResult(polygons=vial_polygons(), calibration=load_calibration(out),
                                 reused=False, out_dir=out)

    after = LVS.learn_faces_for_bundle(
        Seq(flip_sequence(dwell=25, faces="AB")), result, learn=True, show=False)

    assert after.faces_learned == ["A", "B"]
    assert calibration_band_faces(after.calibration) == ["A", "B"]
    assert _vials_json(os.path.join(out, "calibration.json")) == before
    assert after.n_vials == 16


def test_an_unanswerable_prompt_skips_learning_rather_than_hanging_a_scripted_run(tmp_path):
    """Learning needs a window and a turning drum, so it must never block an unattended start."""
    from flygym_tracker import live_vial_selector as LVS

    out = _bundle(tmp_path)
    result = LVS.SelectionResult(polygons=vial_polygons(), calibration=load_calibration(out),
                                 reused=True, out_dir=out)

    after = LVS.learn_faces_for_bundle(Seq(flip_sequence()), result)   # no input_fn, no terminal

    assert after.faces_learned == []
    assert calibration_band_faces(load_calibration(out)) == []


# =============================================================================================
# Real rig data (skipped when the clip is not on this machine)
# =============================================================================================
REAL_VIDEO = os.path.join(os.path.dirname(__file__), "..", "..",
                          "Bonsai related devs", "Good Markers.avi")


@pytest.mark.skipif(not os.path.isfile(REAL_VIDEO), reason="real flip video not available")
def test_the_learning_step_learns_both_faces_from_real_rig_footage():
    """Measured on `Good Markers.avi` (1280 frames, 1280x1024, 30.27 fps, a flipping drum).

    The driver learns both faces within ~91 frames (~3 s), face B scoring -0.24 against the only
    template then registered -- a gap of 0.69 below the 0.45 `min_score` cut, so "never seen this
    before" is not a marginal call.
    """
    source = VideoFileSource(REAL_VIDEO)
    try:
        result = learn_faces(source, n_faces=2, show=False)
    finally:
        source.close()

    assert result.learned == ["A", "B"]
    assert result.complete
    assert result.unreadable_frames == 0
    assert result.detector.can_identify()

    scores = [s for _f, s, _m, how in result.dwells if how == LEARNED]
    assert scores[1] < result.detector.params.min_score


@pytest.mark.skipif(not os.path.isfile(REAL_VIDEO), reason="real flip video not available")
def test_templates_learned_from_real_footage_identify_every_dwell_of_that_footage():
    """Measured: 943/943 stationary frames identified, 0 unreadable, and the dwell labels
    strictly alternate ABABAB... -- which is the only sequence a flipping drum can produce."""
    from flygym_tracker.adaptive_rotation import AdaptiveRotationDetector
    from flygym_tracker.types import TrackState

    source = VideoFileSource(REAL_VIDEO)
    try:
        detector = learn_faces(source, n_faces=2, show=False).detector
    finally:
        source.close()

    rot = AdaptiveRotationDetector()
    cap = cv2.VideoCapture(REAL_VIDEO)
    dwells, prev, n_total, n_none = [], None, 0, 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            state = rot.update(gray)
            if state == TrackState.STATIONARY:
                if prev != TrackState.STATIONARY:
                    dwells.append([])
                face = detector.identify_face(gray)
                dwells[-1].append(face)
                n_total += 1
                n_none += face is None
            prev = state
    finally:
        cap.release()

    assert len(dwells) >= 10
    assert n_total >= 900
    assert n_none == 0, f"{n_none}/{n_total} stationary frames could not be identified"

    labels = []
    for votes in dwells:
        assert len(set(votes)) == 1, "a single dwell was labelled with more than one face"
        labels.append(votes[0])
    assert set(labels) == {"A", "B"}
    assert all(labels[i] != labels[i + 1] for i in range(len(labels) - 1)), \
        "a flipping drum must alternate faces; these labels do not"


# =============================================================================================
# "The drum kept turning, but only one face was detected" -- diagnosed on the real rig
# =============================================================================================
def test_an_unreadable_band_is_reported_instead_of_waiting_for_the_drum():
    """MEASURED ON THE RIG, and it is why this test exists. A 45 s session with the drum turning:
    4 dwells settled, face B was offered 520 times, and the marker band was unreadable on 661 of
    661 stationary frames because the picture was too dark. The status line said "waiting for the
    drum to show the next one" the entire time -- so the operator kept turning a drum that was
    never going to help. It was not waiting for the drum. It could not see.
    """
    import numpy as np

    from flygym_tracker.face_learning import FaceLearner

    learner = FaceLearner(n_faces=2)
    # A STILL frame with texture but no lit marker band -- which is exactly what the rig produced
    # when the illumination was down. A flat black frame will not do: phase correlation on a
    # featureless image never settles, so it would never reach a dwell and would test nothing.
    dark = _dim_textured_frame()
    for _ in range(learner.UNREADABLE_RUN_TO_REPORT + learner.settle_frames + 60):
        learner.observe(dark)

    assert learner.band_unreadable, "a wholly unreadable stream was not reported as one"
    line = learner.status_line()
    assert "cannot be read" in line
    assert "Turning the drum will not help" in line, \
        "the line does not tell the operator to stop doing the thing that cannot work"


def test_a_readable_frame_clears_the_unreadable_report():
    """The warning must not stick once the band comes back -- a stale alarm is one nobody reads."""
    import numpy as np

    from flygym_tracker.face_learning import FaceLearner

    learner = FaceLearner(n_faces=2)
    dark = _dim_textured_frame()
    for _ in range(learner.UNREADABLE_RUN_TO_REPORT + learner.settle_frames + 60):
        learner.observe(dark)
    assert learner.band_unreadable

    learner._unreadable_run = 0                      # what a readable decision does
    assert not learner.band_unreadable
    assert "cannot be read" not in learner.status_line()


def test_the_finished_message_names_the_unreadable_band_as_the_obstacle():
    """"0 of 2 faces learned" and "0 of 2 faces learned, and the band was unreadable in 615 of 871
    frames" send the operator to completely different places. The second one is where the fault
    actually was."""
    from flygym_tracker.gui.video_stage import _job_message

    message = _job_message("faces", {"complete": False, "learned": ["A"],
                                     "unreadable": 615, "frames": 871})
    assert "COULD NOT BE READ" in message
    assert "615 of 871" in message
    assert "too dark" in message

    healthy = _job_message("faces", {"complete": False, "learned": ["A"],
                                     "unreadable": 3, "frames": 871})
    assert "COULD NOT BE READ" not in healthy, "a healthy session was blamed on the light"


def _dim_textured_frame():
    """A still, dim frame with structure but NO lit marker band.

    Deterministic, and deliberately not flat: `AdaptiveRotationDetector` needs something to
    correlate on before it will call a stream stationary, so a black rectangle never reaches a
    dwell at all and a test built on one asserts nothing.
    """
    import numpy as np

    rng = np.random.default_rng(11)
    frame = rng.integers(0, 24, size=(240, 320), dtype=np.uint8)
    frame[60:70, 40:280] = 30          # faint structure, nowhere near a lit LED strip
    return frame

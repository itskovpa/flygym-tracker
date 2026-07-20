"""The operator marks WHERE the marker band is, instead of the detector inferring it every frame.

WHY THIS EXISTS, in one measurement. `MarkerBandDetector` re-derives the band's location from
brightness on every frame, inside a 20%-80% search window. With this rig's backlight unplugged the
two lit strips stopped being two runs: the row profile collapsed to a SINGLE 141-row run, which was
rejected for exceeding `max_strip_h=110`, and the band came back "not found" on 661 of 661
stationary frames. Nothing in that chain was wrong -- there was nothing to see -- but it showed what
the automatic search costs: the band's LOCATION, the one thing about this rig that does not change
between experiments because it is bolted to the rotation axis, was being re-guessed from the very
signal that had degraded.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from flygym_tracker.calibration import (attach_band_rows, calibration_band_rows, load_calibration,
                                        marker_detector_from_calibration)

BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib_faces")


def _bundle(tmp_path) -> str:
    out = str(tmp_path / "calib_faces")
    shutil.copytree(BUNDLE, out)
    return out


pytestmark = pytest.mark.skipif(not os.path.isdir(BUNDLE), reason="no calibration bundle in repo")


# =============================================================================================
# Persistence -- it goes in the vial-positions bundle, which is the rig owner's call
# =============================================================================================
def test_the_band_is_saved_beside_the_vial_positions(tmp_path):
    """The band's location is a fact about this rig's geometry, exactly like the vial polygons,
    so it belongs with them rather than in a config file that travels between rigs."""
    out = _bundle(tmp_path)
    assert calibration_band_rows(load_calibration(out)) is None
    written = attach_band_rows(out, (416, 554))
    assert written == ["A", "B"], "the band was not recorded for every face"
    assert calibration_band_rows(load_calibration(out)) == (416, 554)


def test_saving_the_band_does_not_touch_a_single_hand_drawn_vertex(tmp_path):
    """`calib_faces/` holds polygons drawn by hand, one click per vertex, and they are NOT
    reproducible. The same protection `attach_face_templates` has, for the same reason."""
    out = _bundle(tmp_path)
    before = json.load(open(os.path.join(BUNDLE, "calibration.json"), encoding="utf-8"))
    attach_band_rows(out, (400, 600))
    after = json.load(open(os.path.join(out, "calibration.json"), encoding="utf-8"))
    for face in before["faces"]:
        assert before["faces"][face]["vials"] == after["faces"][face]["vials"], \
            "face %s's hand-drawn vials were modified" % face


def test_a_backwards_or_empty_band_is_refused(tmp_path):
    """A zero-height band finds nothing at all, and would be saved as a silent way to break
    every later identification."""
    out = _bundle(tmp_path)
    for bad in ((500, 500), (600, 400)):
        with pytest.raises(ValueError):
            attach_band_rows(out, bad)


def test_a_bundle_with_no_band_still_reads_as_none(tmp_path):
    """Every bundle written before this feature existed must keep working, on the automatic
    search, rather than raising."""
    assert calibration_band_rows(load_calibration(_bundle(tmp_path))) is None


# =============================================================================================
# The drawn band must actually reach the detector
# =============================================================================================
def test_the_rebuilt_detector_uses_the_drawn_band(tmp_path):
    """Otherwise the operator draws a band and every run goes on guessing -- the failure this
    whole feature exists to remove, made invisible."""
    import cv2

    from flygym_tracker.calibration import attach_face_templates
    from flygym_tracker.marker_band import MarkerBandDetector

    out = _bundle(tmp_path)
    rows = (416, 554)
    detector = MarkerBandDetector(band_rows=rows)
    for face in ("A", "B"):
        image = cv2.imread(os.path.join(BUNDLE, "overlay_%s.png" % face), cv2.IMREAD_GRAYSCALE)
        detector.register_face(image, face)
    attach_face_templates(out, detector)
    attach_band_rows(out, rows)

    rebuilt = marker_detector_from_calibration(load_calibration(out))
    assert rebuilt is not None
    assert rebuilt.band_rows == rows, "the run would have gone back to guessing the band"


def test_hand_picked_rows_win_over_a_stale_stored_window(tmp_path):
    """A bundle whose templates were learned BEFORE the band was drawn must still use the drawn
    band: the operator marked it on the actual rig picture, and the snapshot predates that."""
    import cv2

    from flygym_tracker.calibration import attach_face_templates
    from flygym_tracker.marker_band import MarkerBandDetector

    out = _bundle(tmp_path)
    detector = MarkerBandDetector()                      # learned on the AUTOMATIC window
    for face in ("A", "B"):
        image = cv2.imread(os.path.join(BUNDLE, "overlay_%s.png" % face), cv2.IMREAD_GRAYSCALE)
        detector.register_face(image, face)
    attach_face_templates(out, detector)
    assert marker_detector_from_calibration(load_calibration(out)).band_rows is None

    attach_band_rows(out, (416, 554))                    # drawn afterwards
    assert marker_detector_from_calibration(load_calibration(out)).band_rows == (416, 554)


# =============================================================================================
# The selection session
# =============================================================================================
def test_a_band_that_contains_two_strips_is_reported_as_working(qapp, tmp_path):
    """The preview is the point: a band with fewer than two strips in it identifies nothing, and
    the operator is looking straight at the picture that shows why."""
    import cv2

    from flygym_tracker.gui.band_select import BandSelectSession

    out = _bundle(tmp_path)
    image = cv2.imread(os.path.join(BUNDLE, "overlay_A.png"), cv2.IMREAD_GRAYSCALE)
    session = BandSelectSession(out_dir=out, frame_height=image.shape[0])
    session.on_frame(image)
    session.on_press(0, 410)
    session.on_release(0, 560)
    assert session.rows == (410, 560)
    assert len(session.strips) == 2, "the two lit strips were not found in the drawn band"
    assert "will work" in session.status()


def test_a_band_with_no_strips_in_it_says_so_before_it_is_saved(qapp, tmp_path):
    import cv2

    from flygym_tracker.gui.band_select import BandSelectSession

    out = _bundle(tmp_path)
    image = cv2.imread(os.path.join(BUNDLE, "overlay_A.png"), cv2.IMREAD_GRAYSCALE)
    session = BandSelectSession(out_dir=out, frame_height=image.shape[0])
    session.on_frame(image)
    session.on_press(0, 20)                       # the top of the frame: vials, no LED slots
    session.on_release(0, 120)
    assert len(session.strips) < 2
    assert "needs two" in session.status()


def test_saving_a_band_with_too_few_strips_warns_rather_than_refuses(qapp, tmp_path):
    """It is still saved -- the operator may be marking a rig whose lights are off right now --
    but the message says what it will do, because a silent save reads as success."""
    import cv2

    from flygym_tracker.gui.band_select import BandSelectSession

    out = _bundle(tmp_path)
    image = cv2.imread(os.path.join(BUNDLE, "overlay_A.png"), cv2.IMREAD_GRAYSCALE)
    session = BandSelectSession(out_dir=out, frame_height=image.shape[0])
    session.on_frame(image)
    session.on_press(0, 20)
    session.on_release(0, 120)
    results = []
    session.finished.connect(results.append)
    session.save()
    assert results[0]["saved"] is True
    assert "WARNING" in results[0]["message"]


def test_a_stray_click_does_not_become_a_zero_height_band(qapp, tmp_path):
    """A click is not a drag. A collapsed band would be saved as a region that finds nothing."""
    from flygym_tracker.gui.band_select import BandSelectSession

    session = BandSelectSession(out_dir=str(tmp_path), frame_height=1024)
    session.on_frame(np.zeros((1024, 1280), dtype=np.uint8))
    session.on_press(0, 500)
    session.on_release(0, 502)
    assert not session.has_band
    results = []
    session.finished.connect(results.append)
    session.save()
    assert results[0]["saved"] is False


def test_the_session_shows_what_the_automatic_search_would_have_used(qapp, tmp_path):
    """So the operator can see what they are overriding rather than guessing at it."""
    from flygym_tracker.gui.band_select import BandSelectSession

    session = BandSelectSession(out_dir=str(tmp_path))
    session.on_frame(np.zeros((1000, 1280), dtype=np.uint8))
    assert session.auto_rows == (200, 799), session.auto_rows      # search_frac (0.20, 0.80)

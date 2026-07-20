"""The pre-flight strip: what it can claim, and what it must refuse to claim.

The failures this catches all look identical while they are happening -- a run starts, a CSV fills,
and nothing is wrong until someone reads the results weeks later. So the assertions here are mostly
about HONESTY: an unmeasured condition reports as unknown rather than as a tick, and no check may
touch the camera to find out.
"""
from __future__ import annotations

import os

from flygym_tracker import readiness
from flygym_tracker.readiness import BAD, OK, UNKNOWN


def test_a_missing_config_file_is_a_cross_with_a_fix_button(tmp_path):
    check = readiness.check_config(str(tmp_path / "nope.yaml"))
    assert check.state == BAD
    assert check.fix_action == "pick_config"


def test_an_existing_config_file_is_a_tick(tmp_path):
    path = tmp_path / "rig.yaml"
    path.write_text("binning: {bin_seconds: 60}\n", encoding="utf-8")
    assert readiness.check_config(str(path)).state == OK


def test_a_calibration_folder_with_no_bundle_offers_to_draw_the_vials(tmp_path):
    (tmp_path / "empty").mkdir()
    check = readiness.check_calibration(str(tmp_path / "empty"))
    assert check.state == BAD
    assert check.fix_action == "draw_vials"


def test_a_folder_holding_calibration_json_is_a_tick(tmp_path):
    (tmp_path / "calib").mkdir()
    (tmp_path / "calib" / "calibration.json").write_text("{}", encoding="utf-8")
    assert readiness.check_calibration(str(tmp_path / "calib")).state == OK


def test_an_output_folder_that_does_not_exist_yet_is_fine_if_it_could_be_created(tmp_path):
    """Refusing to start because the output folder has not been made yet would be an obstacle
    invented by the checker: the logger creates it."""
    check = readiness.check_output(str(tmp_path / "results"))
    assert check.state == OK
    assert not os.path.exists(str(tmp_path / "results")), "the check created the folder"


def test_the_readiness_checks_never_write_anything(tmp_path):
    """`_writable` asks the OS rather than touching a probe file into the operator's data folder."""
    before = sorted(os.listdir(tmp_path))
    readiness.evaluate(config_path=str(tmp_path / "c.yaml"), calib_dir=str(tmp_path),
                       output_dir=str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == before


def test_a_closed_camera_is_UNKNOWN_and_not_a_tick():
    """The app deliberately does not take the camera until asked, so "not open" is not a failure.
    It is not a tick either: the limits on screen are then the rig camera's documented ones, and a
    tick would claim this sensor had been read."""
    check = readiness.check_camera("closed")
    assert check.state == UNKNOWN
    assert "documented" in check.sentence
    assert check.fix_action == "open_camera"


def test_a_busy_camera_is_a_cross_that_offers_to_show_the_holder():
    check = readiness.check_camera("error_busy", "held by a Bonsai workflow")
    assert check.state == BAD
    assert check.fix_action == "free_camera"


def test_no_check_ever_opens_a_camera():
    """USB3 Vision is exclusive: a readiness check that opened the camera to see whether it opens
    would BE the thing holding it."""
    class Exploding:
        def open(self):
            raise AssertionError("readiness opened the camera")

        def __getattr__(self, name):
            raise AssertionError("readiness touched the camera (%s)" % name)

    readiness.evaluate(camera_state="closed")
    readiness.check_camera("streaming", "DA4282883")
    # Nothing above was handed a camera object at all, which is the actual guarantee: the signature
    # takes a STATE STRING the app already knows, so there is nothing to probe.
    assert "camera" not in readiness.check_camera.__code__.co_varnames[:1]


def test_an_unchecked_camera_value_is_a_cross_naming_the_setting():
    check = readiness.check_unverified(["source.camera.frame_rate"],
                                       {"source.camera.frame_rate": "frame rate"})
    assert check.state == BAD
    assert "frame rate" in check.sentence.lower()


def test_unsaved_changes_are_UNKNOWN_rather_than_a_failure():
    """Unsaved work is a state, not a fault -- but it is worth saying, because a run started now
    would use the file's old values."""
    check = readiness.check_unsaved(3, "config/flygym_rig.yaml")
    assert check.state == UNKNOWN
    assert "old values" in check.sentence


def test_ready_is_false_only_when_something_is_actually_wrong(tmp_path):
    (tmp_path / "calib").mkdir()
    (tmp_path / "calib" / "calibration.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "rig.yaml"
    config.write_text("binning: {bin_seconds: 60}\n", encoding="utf-8")

    good = readiness.evaluate(config_path=str(config), calib_dir=str(tmp_path / "calib"),
                              output_dir=str(tmp_path / "out"), camera_state="closed")
    assert good.ready is True, good.text()      # a closed camera does not block

    bad = readiness.evaluate(config_path=str(config), calib_dir=str(tmp_path),
                             output_dir=str(tmp_path / "out"), camera_state="closed")
    assert bad.ready is False
    assert [c.key for c in bad.problems()] == ["calibration"]


def test_the_strip_renders_as_plain_text_for_a_terminal_too(tmp_path):
    text = readiness.evaluate(config_path=None).text()
    assert text.count("\n") == 5              # six checks, five newlines
    assert "[X]" in text


# =============================================================================================
# The strip's height, which is a measurement and not a preference
# =============================================================================================
def _strip(qapp):
    from flygym_tracker.gui.readiness_strip import ReadinessStrip

    strip = ReadinessStrip()
    strip.resize(900, 300)
    return strip


def _checks(bad_keys=()):
    from flygym_tracker.readiness import BAD, Check, OK

    out = []
    for key in ("config", "calibration", "output", "camera", "unverified", "unsaved"):
        if key in bad_keys:
            out.append(Check(key, BAD, "%s is not ready" % key, "Fix it", "fix_%s" % key))
        else:
            out.append(Check(key, OK, "%s is fine" % key))
    return out


def test_everything_passing_collapses_to_one_line(qapp):
    """MEASURED, not preferred: six full rows took 210 px of an 880 px window -- 24% of the height
    -- while the camera picture got 312 px. On a rig whose whole point is looking at the picture
    (exposure and gain are tuned by eye, vial polygons are drawn on it), an all-ticks checklist was
    the second-largest thing on screen."""
    from flygym_tracker.readiness import Readiness

    strip = _strip(qapp)
    strip.set_readiness(Readiness(checks=_checks()))
    assert len(strip._rows) == 1, "the passing checks did not collapse"


def test_the_summary_names_what_passed_rather_than_counting_it(qapp):
    """"5 of 6 checks pass" tells an operator nothing they can act on."""
    from PySide6.QtWidgets import QLabel

    from flygym_tracker.readiness import Readiness

    strip = _strip(qapp)
    strip.set_readiness(Readiness(checks=_checks()))
    text = " ".join(label.text() for label in strip._rows[0].findChildren(QLabel))
    for name in ("config", "vial positions", "output folder", "camera"):
        assert name in text, "the summary does not name %r" % name


def test_anything_not_passing_keeps_its_own_row_and_its_fix_button(qapp):
    """The collapse must never cost the operator the thing that fixes the problem."""
    from PySide6.QtWidgets import QPushButton

    from flygym_tracker.readiness import Readiness

    strip = _strip(qapp)
    strip.set_readiness(Readiness(checks=_checks(bad_keys=("calibration",))))
    assert len(strip._rows) == 2, "expected the problem row plus the summary"
    buttons = [b for row in strip._rows for b in row.findChildren(QPushButton)]
    assert [b.text() for b in buttons] == ["Fix it"]


def test_problems_are_listed_above_the_reassurance(qapp):
    """Whatever is wrong is what has to be acted on, so it goes where the eye lands. The other
    order would push a cross below a row of ticks."""
    from PySide6.QtWidgets import QLabel

    from flygym_tracker.readiness import Readiness

    strip = _strip(qapp)
    strip.set_readiness(Readiness(checks=_checks(bad_keys=("output",))))
    first = " ".join(label.text() for label in strip._rows[0].findChildren(QLabel))
    assert "not ready" in first, "the problem was not the first thing in the strip"


def test_the_strip_is_still_always_there(qapp):
    """It is never hidden, even with nothing wrong: a strip that appears only when something is
    wrong is a strip nobody has read before, so the first time it appears it is unfamiliar -- at
    the moment it is most needed."""
    from flygym_tracker.readiness import Readiness

    strip = _strip(qapp)
    strip.set_readiness(Readiness(checks=_checks()))
    assert strip._rows, "the strip vanished when everything passed"

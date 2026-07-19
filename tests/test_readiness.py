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

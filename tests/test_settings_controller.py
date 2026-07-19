"""The rules a settings surface obeys, tested with no toolkit anywhere near them.

WHY THESE ARE NOT WIDGET TESTS. Each one protects an experiment that runs for days on flies that
cannot be replaced, and the protection has to hold whichever surface is driving -- the Qt app, the
cv2 panel, or whatever Stage 3 grows. Testing them through offscreen widgets would tie the proof
of a measurement rule to the layout of a window. So the rules live in `settings_controller`, these
tests import no Qt at all, and the widget tests that follow only have to show the buttons are
wired to them.

Nothing here needs a camera, and there is none on the machine this was written on. Where a camera
is implied it is a fake, and what is being asserted is the CALL PATTERN -- what would be sent, and
when nothing would be.
"""
from __future__ import annotations

import pytest

from flygym_tracker.config import load_config
from flygym_tracker.settings_controller import (
    NEVER_CHECKED_BADGE,
    SettingsController,
    arming_plan,
    camera_block_reason,
    to_default_notice,
)
from flygym_tracker.settings_model import (
    CAMERA_START_ONLY_KEYS,
    DEFAULT_TEXT,
    build_app_settings,
    build_camera_settings,
    start_only_block_reason,
)

CAMERA_KEYS = ["source.camera.frame_rate", "source.camera.exposure_us", "source.camera.gain_db",
               "source.camera.width", "source.camera.height"]


# =============================================================================================
# Fixtures: a config with every camera key null (which is what the rig actually ships), and a
# fake camera that records what it was told rather than owning hardware.
# =============================================================================================
@pytest.fixture
def config():
    return load_config(path="config/flygym_rig.yaml")


@pytest.fixture
def model(config):
    return build_app_settings(config)


@pytest.fixture
def controller(model, tmp_path, config):
    path = tmp_path / "rig.yaml"
    path.write_text("binning:\n  bin_seconds: 60\nsource:\n  camera:\n"
                    "    frame_rate: null   # fps\n    exposure_us: null\n"
                    "    gain_db: null\n    width: null\n    height: null\n"
                    "activity:\n  pixel_threshold: 12.0  # measured noise floor\n",
                    encoding="utf-8")
    return SettingsController(model, config_path=str(path))


class FakeCamera:
    """Just enough `HikCameraSource` for the block rule and the arming hints."""

    def __init__(self, *, acquiring=False, values=None):
        self.is_acquiring = acquiring
        self._values = dict(values or {})
        self.sent = []

    def current_values(self):
        return dict(self._values)

    def ranges(self):
        from flygym_tracker.frame_source import fallback_camera_ranges

        return fallback_camera_ranges()

    def set_frame_rate(self, value):
        self.sent.append(("frame_rate", value))


# =============================================================================================
# INVARIANT 1 -- a row left at "camera default" produces no write, through the whole funnel
# =============================================================================================
def test_a_model_built_from_the_rig_config_has_every_camera_row_at_the_camera_default(model):
    """The shipped config sets all five to null, and null must survive into the model as None --
    not be folded into a default number on the way."""
    for key in CAMERA_KEYS:
        assert model.value(key) is None, "%s did not arrive as 'camera default'" % key


def test_reading_and_reporting_a_default_row_never_calls_the_change_router(controller):
    """Every read-only thing a surface does to a default row -- formatting it, counting it, asking
    whether it is blocked, planning what arming WOULD do -- must not reach `on_change`.

    This is the funnel-level half of invariant 1. The SDK-level half is asserted through the real
    fake SDK in `test_gui_settings_view.py`.
    """
    calls = []
    controller.on_change = lambda key, value: calls.append((key, value))
    for key in CAMERA_KEYS:
        controller.block_reason(key)
        controller.group_title("Camera")
        arming_plan(controller.model.get(key))
        to_default_notice(controller.model.get(key), camera_open=True)
    assert calls == []


def test_committing_the_value_a_row_already_holds_reports_nothing(controller):
    """A no-op edit is not a change. A spinbox emits on programmatic writes and on every focus
    change, and a router that logged a `setting_change` for each would bury the real transitions
    in its own noise."""
    calls = []
    controller.on_change = lambda key, value: calls.append((key, value))
    before = controller.model.value("activity.pixel_threshold")
    result = controller.commit("activity.pixel_threshold", before)
    assert result.moved is False
    assert calls == []


# =============================================================================================
# INVARIANT 4 -- provenance: one call per distinct value, with the value that was STORED
# =============================================================================================
def test_the_router_is_told_the_coerced_value_not_the_one_that_was_typed(controller):
    """The model clamps and snaps; the router must hear what was stored, because that is what the
    run will actually be using and what the events log will name."""
    seen = []
    controller.on_change = lambda key, value: seen.append((key, value))
    controller.commit("activity.pixel_threshold", 999.0)      # hi is 60.0
    assert seen == [("activity.pixel_threshold", 60.0)]


def test_three_distinct_values_report_three_times_and_a_repeat_reports_once(controller):
    """A drag that passes through several values really did measure frames at each one, so each is
    reported. Repeating a value it already holds is not a transition."""
    seen = []
    controller.on_change = lambda key, value: seen.append(value)
    for value in (10.0, 20.0, 30.0, 30.0, 30.0):
        controller.commit("activity.pixel_threshold", value)
    assert seen == [10.0, 20.0, 30.0]


def test_a_router_that_refuses_keeps_the_value_and_says_so(controller):
    """`on_change` returning False means "this run cannot apply it" -- which is shown, never
    reverted. Snapping the widget back would look broken; silently keeping it would misrepresent
    the run. The value still belongs in the config file."""
    controller.on_change = lambda key, value: False
    result = controller.commit("activity.pixel_threshold", 30.0)
    assert result.moved is True
    assert result.applied is False
    assert controller.model.value("activity.pixel_threshold") == 30.0
    assert "not applied to this run" in result.message


def test_a_router_that_raises_does_not_take_the_surface_down_with_it(controller):
    def boom(key, value):
        raise RuntimeError("the camera fell over")

    controller.on_change = boom
    result = controller.commit("activity.pixel_threshold", 30.0)
    assert result.applied is False
    assert "the camera fell over" in result.message


# =============================================================================================
# INVARIANT 3 -- start-only, and ONE rule shared with the pipeline
# =============================================================================================
def test_width_and_height_are_blocked_while_the_camera_is_acquiring(model):
    camera = FakeCamera(acquiring=True)
    controller = SettingsController(model, block_reason=camera_block_reason(lambda: camera))
    for key in CAMERA_START_ONLY_KEYS:
        assert controller.is_blocked(key), "%s was editable during acquisition" % key


def test_nothing_is_blocked_while_the_camera_is_closed(model):
    camera = FakeCamera(acquiring=False)
    controller = SettingsController(model, block_reason=camera_block_reason(lambda: camera))
    assert [k for k in model.keys() if controller.is_blocked(k)] == []


def test_the_live_camera_rows_stay_editable_during_acquisition(model):
    """Blocking exposure mid-run would defeat the point of the preview -- these are the knobs the
    operator is watching the picture to set."""
    camera = FakeCamera(acquiring=True)
    controller = SettingsController(model, block_reason=camera_block_reason(lambda: camera))
    for key in ("source.camera.frame_rate", "source.camera.exposure_us", "source.camera.gain_db"):
        assert not controller.is_blocked(key)


def test_a_blocked_row_stores_nothing_and_tells_no_one(model):
    """The refusal is at the funnel, not at the control: `setEnabled(False)` was MEASURED not to
    stop a programmatic `setValue`, so a forced commit has to hit a wall here."""
    camera = FakeCamera(acquiring=True)
    calls = []
    controller = SettingsController(model, block_reason=camera_block_reason(lambda: camera),
                                    on_change=lambda k, v: calls.append((k, v)))
    before = controller.model.value("source.camera.width")
    result = controller.commit("source.camera.width", 640)
    assert result.blocked is True
    assert controller.model.value("source.camera.width") == before
    assert calls == []


def test_a_blocked_row_cannot_be_moved_by_reset_either(model):
    """Reset would otherwise be the one way round the guard, and would leave the surface showing a
    value the run is not using."""
    camera = FakeCamera(acquiring=False)
    controller = SettingsController(model, block_reason=camera_block_reason(lambda: camera))
    controller.commit("source.camera.width", 640)
    camera.is_acquiring = True                     # the run starts
    controller.reset()
    assert controller.model.value("source.camera.width") == 640


def test_the_pipeline_and_the_app_answer_the_start_only_question_with_the_same_function():
    """The rule has ONE implementation. `TrackerPipeline.setting_block_reason` calls it and so does
    the app's shim, so the two cannot drift into disagreeing about whether Width is safe to change
    under a recording experiment."""
    import inspect

    from flygym_tracker.pipeline import TrackerPipeline

    source = inspect.getsource(TrackerPipeline.setting_block_reason)
    assert "start_only_block_reason(" in source
    assert "CAMERA_START_ONLY_KEYS" not in source.split('"""')[-1], \
        "the pipeline is deciding for itself again instead of asking the shared rule"


def test_the_shared_rule_gives_the_recording_wording_only_once_frames_have_been_measured():
    """Two situations, two messages: "the stream is running" would be a baffling thing to read in a
    panel opened BEFORE the run, and `run` hands the pipeline an already-open camera."""
    camera = FakeCamera(acquiring=True)
    key = CAMERA_START_ONLY_KEYS[0]
    assert "this run is recording" in start_only_block_reason(key, camera, recording=True)
    assert "`settings` command" in start_only_block_reason(key, camera, recording=False)
    assert "close it" in start_only_block_reason(key, camera, recording=False, closable=True)


def test_the_app_gets_the_wording_that_names_a_button_it_actually_has(model):
    """Telling an operator who is looking at a "Close camera" button to go and use a command-line
    tool would be absurd, and a greyed control whose reason does not fit is worse than none."""
    camera = FakeCamera(acquiring=True)
    controller = SettingsController(model, block_reason=camera_block_reason(lambda: camera))
    reason = controller.block_reason("source.camera.width")
    assert "close it" in reason
    assert "`settings` command" not in reason


# =============================================================================================
# THE ARMING HAZARD -- the one thing in this surface that can quietly ruin an experiment
# =============================================================================================
def test_arming_a_row_with_a_camera_open_lands_on_what_the_camera_reports(config):
    """The good case: the first value written equals what the sensor is already doing, so nothing
    physical changes -- only provenance does."""
    camera = FakeCamera(acquiring=True, values={"frame_rate": 88.5, "exposure_us": 5000.0})
    settings, _notes = build_camera_settings(config, camera)
    frame_rate = [s for s in settings if s.key == "source.camera.frame_rate"][0]
    plan = arming_plan(frame_rate)
    assert plan.blind is False
    assert plan.landing == pytest.approx(88.5)
    assert "88.5" in plan.button_text
    assert plan.warning == "", "there is nothing to warn about: this came from the sensor"


def test_arming_a_row_with_NO_camera_lands_on_the_lowest_legal_value_and_says_so(model):
    """THE MEASURED HAZARD. With no camera there is no hint, and the landing value is `lo`: 0.1 fps,
    9 us, 32 px. Saving that into the rig config is a ruined experiment. It is not blocked -- away-
    from-the-rig editing is a workflow the CLI supports -- so the number is put in the button's own
    label, where it is read before the press rather than discovered after it."""
    landings = {}
    for key in CAMERA_KEYS:
        plan = arming_plan(model.get(key))
        landings[key] = plan.landing
        assert plan.blind is True
        # The VALUE is on the button (read before the press); the reason it is a guess is the
        # warning shown beside it. Both are on the row -- the split is a layout decision, not a
        # decision to say less. See `arming_plan` on why the button label had to get shorter.
        assert str(plan.landing).rstrip("0").rstrip(".") in plan.button_text.replace(",", "")
        assert "no camera" in plan.warning
        assert "not a reading" in plan.warning
    assert landings["source.camera.frame_rate"] == pytest.approx(0.1)
    assert landings["source.camera.exposure_us"] == pytest.approx(9.0)
    assert landings["source.camera.width"] == pytest.approx(32.0)


def test_a_row_armed_with_no_camera_is_badged_never_checked(controller):
    controller.arm("source.camera.frame_rate")
    assert controller.never_checked() == ["source.camera.frame_rate"]


def test_saving_an_unchecked_camera_value_asks_first_and_names_the_value_and_the_file(controller):
    """The guard is on the IRREVERSIBLE step. Arming is reversible and unconfirmed; writing a
    number the next experiment will start from is not."""
    controller.arm("source.camera.frame_rate")
    asked = []
    result = controller.save(confirm=lambda text: asked.append(text) or False)
    assert result.saved is False
    assert result.needed_confirm is True
    assert len(asked) == 1
    assert "0.1 fps" in asked[0]
    assert controller.config_path in asked[0]
    # The FULL wording, not the row's short badge: this dialog has room for it, and it is where an
    # operator meets the phrase in full before an irreversible write.
    assert NEVER_CHECKED_BADGE.split(" against")[0] in asked[0]


def test_saying_no_to_that_question_writes_nothing_at_all(controller, tmp_path):
    before = open(controller.config_path, encoding="utf-8").read()
    controller.arm("source.camera.frame_rate")
    controller.save(confirm=lambda text: False)
    assert open(controller.config_path, encoding="utf-8").read() == before


def test_saying_yes_writes_it_and_keeps_the_files_comments(controller):
    controller.arm("source.camera.frame_rate")
    result = controller.save(confirm=lambda text: True)
    assert result.saved is True
    text = open(controller.config_path, encoding="utf-8").read()
    assert "frame_rate: 0.1" in text
    assert "# fps" in text, "the comment justifying the line was lost"
    assert "# measured noise floor" in text


def test_an_ordinary_save_is_never_confirmed(controller):
    """Over-confirming trains click-through, which protects nothing. This save rewrites only what
    changed and preserves every comment -- it is neither destructive nor surprising."""
    controller.commit("activity.pixel_threshold", 15.0)
    asked = []
    result = controller.save(confirm=lambda text: asked.append(text) or True)
    assert result.saved is True
    assert asked == []


def test_a_row_put_back_to_the_camera_default_is_no_longer_unchecked(controller):
    controller.arm("source.camera.frame_rate")
    controller.to_default("source.camera.frame_rate")
    assert controller.never_checked() == []
    assert controller.model.value("source.camera.frame_rate") is None


def test_opening_a_camera_clears_the_unchecked_badge_without_changing_the_value(controller):
    """The operator's number is still theirs; it is simply no longer unverifiable, because a sensor
    is now attached to check it against."""
    controller.arm("source.camera.exposure_us")
    controller.commit("source.camera.exposure_us", 5000.0)
    controller.confirm_against_camera({"exposure_us": 4998.0})
    assert controller.never_checked() == []
    assert controller.model.value("source.camera.exposure_us") == pytest.approx(5000.0)


# =============================================================================================
# RETURNING TO "camera default" -- the words, and when they are needed at all
# =============================================================================================
def test_going_back_to_default_with_the_camera_open_says_the_camera_keeps_the_value(config):
    camera = FakeCamera(acquiring=True, values={"frame_rate": 30.0})
    settings, _ = build_camera_settings(config, camera)
    row = [s for s in settings if s.key == "source.camera.frame_rate"][0]
    row.value = 30.0
    notice = to_default_notice(row, camera_open=True)
    assert "KEEP running at 30.0 fps" in notice
    assert DEFAULT_TEXT in notice
    assert "recording" not in notice


def test_the_same_notice_mid_run_adds_that_the_recording_continues_at_that_value(config):
    camera = FakeCamera(acquiring=True, values={"frame_rate": 30.0})
    settings, _ = build_camera_settings(config, camera)
    row = [s for s in settings if s.key == "source.camera.frame_rate"][0]
    row.value = 30.0
    assert "continues to record at 30.0 fps" in to_default_notice(row, camera_open=True,
                                                                 recording=True)


def test_with_the_camera_closed_there_is_nothing_to_say_because_nothing_was_ever_sent(model):
    """This exemption is what keeps the notice rare enough to still be read: most tuning happens
    with the camera closed, and there it is plain config editing."""
    row = model.get("source.camera.frame_rate")
    row.value = 30.0
    assert to_default_notice(row, camera_open=False) == ""


def test_clearing_a_row_goes_out_through_the_router_like_any_other_change(controller):
    """Clearing a value mid-run is a measurement-regime change too, and `apply_setting` logs it as
    one -- so it must not take a side door that skips the router."""
    controller.commit("source.camera.frame_rate", 30.0)
    seen = []
    controller.on_change = lambda key, value: seen.append((key, value))
    controller.to_default("source.camera.frame_rate")
    assert seen == [("source.camera.frame_rate", None)]


# =============================================================================================
# The group-level count -- the tri-state's fourth channel
# =============================================================================================
def test_the_camera_group_title_counts_how_many_rows_are_left_alone(controller):
    assert controller.group_title("Camera") == "Camera - 5 of 5 left at %s" % DEFAULT_TEXT
    controller.commit("source.camera.gain_db", 2.0)
    assert controller.group_title("Camera") == "Camera - 4 of 5 left at %s" % DEFAULT_TEXT


def test_a_group_with_no_nullable_rows_has_a_plain_title(controller):
    assert controller.group_title("Activity") == "Activity"


# =============================================================================================
# Saving, generally
# =============================================================================================
def test_saving_with_nothing_changed_writes_nothing_and_says_so(controller):
    result = controller.save(confirm=lambda text: True)
    assert result.saved is False
    assert "nothing changed" in result.message


def test_saving_with_no_config_file_refuses_rather_than_guessing_a_path(model):
    controller = SettingsController(model, config_path=None)
    controller.commit("activity.pixel_threshold", 15.0)
    result = controller.save(confirm=lambda text: True)
    assert result.saved is False
    assert "no config file" in result.message


def test_a_save_advances_the_baseline_so_the_rows_stop_showing_as_unsaved(controller):
    controller.commit("activity.pixel_threshold", 15.0)
    assert len(controller.changed()) == 1
    controller.save(confirm=lambda text: True)
    assert controller.changed() == []

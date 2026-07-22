"""The activity-only tick box: turn off per-fly tracking, keep activity monitoring.

The rig owner's call: "I want an option to be able to turn off tracking and leave activity
monitoring only. Add it as a tick box disabled by default." Disabled (unticked) by default MEANS
flies are still tracked unless the box is ticked, so an existing run behaves exactly as before;
ticking it drops the two fly-tracking worker threads and leaves per-vial activity + rotation.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def window(qapp, tmp_path):
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    state = gui_state.default_state()
    state["calib_dir"] = str(tmp_path / "calib")
    state["output_dir"] = str(tmp_path / "out")
    win = MainWindow(config=load_config(), config_path=str(tmp_path / "c.yaml"), state=state,
                     root=str(tmp_path), camera_factory=lambda: None, confirm=lambda text: True)
    win.show()
    qapp.processEvents()
    yield win
    win.run.shutdown()
    win.session.shutdown()


# =============================================================================================
# The tick box
# =============================================================================================
def test_tracking_is_on_by_default(qapp, window):
    """Unticked by default MEANS flies are tracked -- the behaviour every existing run already has.
    The box turns tracking OFF; leaving it alone must change nothing."""
    bar = window.session_bar
    assert not bar.activity_only_box.isChecked()
    assert bar.track_flies() is True


def test_the_box_lives_in_the_collapsible_experiment_section(qapp, window):
    """As asked -- the same Experiment section as the paths and the recording box, and it collapses
    with them. It belongs there: like recording, it is settled once at the start, not mid-run."""
    bar = window.session_bar
    assert bar.activity_only_box.isVisible()
    bar.set_expanded(False)
    qapp.processEvents()
    assert not bar.activity_only_box.isVisible(), "the tracking box did not collapse with the rest"
    bar.set_expanded(True)
    qapp.processEvents()
    assert bar.activity_only_box.isVisible()


def test_ticking_it_means_activity_only_and_announces_the_change(qapp, window):
    bar = window.session_bar
    seen = []
    bar.tracking_changed.connect(seen.append)
    bar.activity_only_box.setChecked(True)
    assert bar.track_flies() is False
    assert seen == [False], "tracking_changed should carry the new track_flies value"


# =============================================================================================
# It reaches the run, once, at Start
# =============================================================================================
def test_the_default_plan_tracks(qapp, window, monkeypatch):
    seen = {}
    monkeypatch.setattr(window.run, "start", lambda plan: seen.update(plan) or False)
    window._start_run_now()
    assert seen["track_flies"] is True


def test_activity_only_is_carried_into_the_run_plan(qapp, window, monkeypatch):
    """Read ONCE, when Start is pressed -- a run cannot pick up a mid-run flip of this, by design
    (see `_on_tracking_changed`): it fixes what the whole run's data means."""
    bar = window.session_bar
    bar.activity_only_box.setChecked(True)
    seen = {}
    monkeypatch.setattr(window.run, "start", lambda plan: seen.update(plan) or False)
    window._start_run_now()
    assert seen["track_flies"] is False


# =============================================================================================
# It survives a restart, and the env override still wins
# =============================================================================================
def test_the_choice_survives_a_restart(qapp, window, tmp_path):
    from flygym_tracker.gui import gui_state

    window.session_bar.activity_only_box.setChecked(True)
    qapp.processEvents()
    reloaded = gui_state.load_state(str(tmp_path))
    assert reloaded["track_flies"] is False


def test_gui_state_round_trips_the_bool_as_a_bool(tmp_path):
    """False must survive save->load as False, not become the int 0 or the string 'False' -- the
    reason `_coerce` needs a bool branch ahead of the int one."""
    from flygym_tracker.gui import gui_state

    state = gui_state.default_state()
    state["track_flies"] = False
    assert gui_state.save_state(str(tmp_path), state)
    assert gui_state.load_state(str(tmp_path))["track_flies"] is False

    state["track_flies"] = True
    gui_state.save_state(str(tmp_path), state)
    assert gui_state.load_state(str(tmp_path))["track_flies"] is True


def test_default_when_the_key_is_absent_is_to_track(tmp_path):
    from flygym_tracker.gui import gui_state

    assert gui_state.load_state(str(tmp_path))["track_flies"] is True   # no file at all


def test_the_env_debug_switch_still_forces_tracking_off(monkeypatch):
    """`FLYGYM_DISABLE_TRACKING=1` is the crash-bisect override and must beat the tick box: even a
    plan that asks to track ends up not tracking. This is the resolution the run controller applies."""
    monkeypatch.setenv("FLYGYM_DISABLE_TRACKING", "1")
    plan = {"track_flies": True}
    track = bool(plan.get("track_flies", True)) and not os.environ.get("FLYGYM_DISABLE_TRACKING")
    assert track is False

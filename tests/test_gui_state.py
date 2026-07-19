"""The app's remembered paths, and the one property that matters: LOADING NEVER RAISES.

This file is written by whichever version of the app last ran and read by whichever version the rig
happens to have. A startup `KeyError` on a lab machine at the beginning of an experiment is a worse
outcome than a forgotten folder, every time -- so every test below feeds it something broken and
asserts that a usable state comes back.

Qt-free: `gui_state` is plain JSON, and it stays that way so this runs on any machine.
"""
from __future__ import annotations

import json
import os

import pytest

from flygym_tracker.gui import gui_state


def test_a_missing_file_loads_the_defaults(tmp_path):
    state = gui_state.load_state(str(tmp_path))
    assert state["config_path"] == gui_state.DEFAULTS["config_path"]
    assert state["recent_configs"] == []


@pytest.mark.parametrize("junk", [
    "",                                   # empty file
    "{not json at all",                   # a truncated write, e.g. power loss mid-save
    "[]",                                 # valid JSON, wrong shape
    "null",
    '{"config_path": 42}',                # a hand-edit that changed a type
    '{"recent_configs": "not a list"}',
    '{"recent_configs": [1, 2, {"a": 1}]}',
])
def test_a_broken_state_file_still_loads_a_usable_state(tmp_path, junk):
    (tmp_path / gui_state.STATE_FILENAME).write_text(junk, encoding="utf-8")
    state = gui_state.load_state(str(tmp_path))
    assert isinstance(state["config_path"], str)
    assert isinstance(state["recent_configs"], list)
    assert all(isinstance(p, str) for p in state["recent_configs"])


def test_a_key_from_a_future_version_is_dropped_rather_than_carried(tmp_path):
    """Carrying an unknown key would mean writing it back out, guessing at what it meant."""
    (tmp_path / gui_state.STATE_FILENAME).write_text(
        json.dumps({"config_path": "a.yaml", "some_future_thing": {"x": 1}}), encoding="utf-8")
    state = gui_state.load_state(str(tmp_path))
    assert "some_future_thing" not in state
    assert state["config_path"] == "a.yaml"


def test_a_partial_file_keeps_what_it_has_and_defaults_the_rest(tmp_path):
    (tmp_path / gui_state.STATE_FILENAME).write_text(
        json.dumps({"output_dir": "D:/results"}), encoding="utf-8")
    state = gui_state.load_state(str(tmp_path))
    assert state["output_dir"] == "D:/results"
    assert state["calib_dir"] == gui_state.DEFAULTS["calib_dir"]


def test_saving_then_loading_round_trips(tmp_path):
    state = gui_state.default_state()
    state["output_dir"] = "D:/somewhere else"
    assert gui_state.save_state(str(tmp_path), state) is True
    assert gui_state.load_state(str(tmp_path))["output_dir"] == "D:/somewhere else"


def test_saving_into_a_folder_that_cannot_be_written_reports_rather_than_raises(tmp_path):
    """A read-only install directory is a real deployment -- a shared rig, a locked-down lab
    machine -- and being unable to remember a folder is not a reason to refuse to run."""
    assert gui_state.save_state(str(tmp_path / "does" / "not" / "exist"),
                                gui_state.default_state()) is False


def test_the_recent_list_is_most_recent_first_with_no_duplicates():
    state = gui_state.default_state()
    for path in ("a.yaml", "b.yaml", "a.yaml"):
        gui_state.remember_config(state, path)
    assert state["recent_configs"] == ["a.yaml", "b.yaml"]


def test_the_recent_list_is_capped():
    state = gui_state.default_state()
    for i in range(gui_state.MAX_RECENT + 5):
        gui_state.remember_config(state, "config_%d.yaml" % i)
    assert len(state["recent_configs"]) == gui_state.MAX_RECENT


def test_the_state_file_is_visible_next_to_the_program_not_in_the_registry(tmp_path):
    """A scientist has to be able to see it, copy it to the second rig, back it up, and delete it
    when it goes wrong. "Delete this file and start again" is a support instruction that works;
    "open regedit" is not."""
    gui_state.save_state(str(tmp_path), gui_state.default_state())
    assert os.path.isfile(tmp_path / "gui_state.json")


def test_bin_seconds_is_deliberately_not_an_app_setting():
    """It decides what one row of the results MEANS, which makes it a measurement parameter
    belonging in the experiment's YAML (and, through run_meta.json, in the record of the run) --
    not an app path. `build_app_settings` puts it on screen as a settings row instead."""
    assert "bin_seconds" not in gui_state.DEFAULTS

    from flygym_tracker.settings_model import BIN_SECONDS_KEY, build_app_settings
    from flygym_tracker.config import load_config

    model = build_app_settings(load_config(path="config/flygym_rig.yaml"))
    assert BIN_SECONDS_KEY in model
    assert model.get(BIN_SECONDS_KEY).live is False      # it says "next run" on its own face

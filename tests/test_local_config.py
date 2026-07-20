"""The app writes to THIS RIG's config, never to the shipped template.

THE PROBLEM THIS CLOSES. `config/flygym_rig.yaml` is in version control: it is what a fresh clone
gets, and a whole family of tests asserts that it leaves every camera field null, so an untouched
install imposes nothing on the sensor (invariant 1). But it was also the file the app opened and
saved into. So an ordinary afternoon of tuning rewrote the shipped default to whatever the last
operator happened to be trying -- silently, visible only in a `git diff` nobody reads before an
experiment, and it broke 18 tests that were doing exactly their job.

The fix is a layer, not a rule about being careful: `config/flygym_rig.local.yaml` holds this
machine's values and is merged ON TOP of the template. Nothing is lost, the template stays a
template, and a save aimed at the template is redirected rather than refused.
"""
from __future__ import annotations

import io
import os

import pytest

from flygym_tracker.config import (DEFAULT_CONFIG_PATH, LOCAL_SUFFIX, RIG_CONFIG_PATH,
                                   config_layers, ensure_local_config, is_tracked_template,
                                   load_config, local_config_path, template_for_local)


# =============================================================================================
# The layering
# =============================================================================================
def test_a_local_file_is_merged_on_top_of_its_template_not_instead_of_it(tmp_path):
    """THE LOAD-BEARING CLAIM. If the local file REPLACED the template, a rig whose local file
    holds one tuned threshold would silently lose every other value the template carries -- and
    the run would still start, and the CSV would still fill."""
    template = tmp_path / "rig.yaml"
    template.write_text("binning:\n  bin_seconds: 60\nrotation:\n  sensitivity: 1.0\n")
    local = tmp_path / ("rig" + LOCAL_SUFFIX)
    local.write_text("rotation:\n  sensitivity: 1.2\n")

    config = load_config(path=str(local))
    assert config.rotation.sensitivity == 1.2, "the local value did not win"
    assert config.binning.bin_seconds == 60, "the template's value was lost"


def test_a_template_improvement_still_reaches_a_rig_that_has_a_local_file(tmp_path):
    """This is why the local file is an override layer and not a copy: a value the operator never
    touched keeps tracking the shipped template, including after an update."""
    template = tmp_path / "rig.yaml"
    local = tmp_path / ("rig" + LOCAL_SUFFIX)
    local.write_text("rotation:\n  sensitivity: 1.2\n")

    template.write_text("binning:\n  bin_seconds: 60\n")
    assert load_config(path=str(local)).binning.bin_seconds == 60
    template.write_text("binning:\n  bin_seconds: 30\n")      # the template is updated
    assert load_config(path=str(local)).binning.bin_seconds == 30, \
        "the rig is pinned to an old template value it never chose"


def test_an_ordinary_config_path_is_unaffected(tmp_path):
    """Only `.local.yaml` gets the extra layer. Every existing caller passing an ordinary path --
    every CLI subcommand, every test -- must behave exactly as before."""
    path = tmp_path / "whatever.yaml"
    path.write_text("binning:\n  bin_seconds: 15\n")
    assert config_layers(str(path)) == [DEFAULT_CONFIG_PATH, path]
    assert load_config(path=str(path)).binning.bin_seconds == 15


def test_a_local_file_with_no_template_beside_it_is_just_a_config(tmp_path):
    local = tmp_path / ("orphan" + LOCAL_SUFFIX)
    local.write_text("binning:\n  bin_seconds: 12\n")
    assert template_for_local(str(local)) is None
    assert load_config(path=str(local)).binning.bin_seconds == 12


def test_the_layers_are_reportable_because_they_are_the_provenance_of_a_measurement():
    names = [p.name for p in config_layers("config/flygym_rig.local.yaml")]
    assert names == ["default_config.yaml", "flygym_rig.yaml", "flygym_rig.local.yaml"]


# =============================================================================================
# Which files are shipped
# =============================================================================================
def test_the_shipped_configs_are_recognised_as_templates():
    assert is_tracked_template(str(RIG_CONFIG_PATH))
    assert is_tracked_template(str(DEFAULT_CONFIG_PATH))


def test_a_local_file_is_never_a_template():
    assert not is_tracked_template("config/flygym_rig.local.yaml")
    assert not is_tracked_template(str(local_config_path(RIG_CONFIG_PATH)))


def test_a_path_that_does_not_exist_is_not_a_template():
    """`is_tracked_template` guards a WRITE. It must answer, not raise, for anything at all."""
    assert not is_tracked_template("")
    assert not is_tracked_template(None)
    assert not is_tracked_template("no/such/file.yaml")


def test_naming_the_local_sibling_is_idempotent():
    """It is applied both when choosing where to save and when deciding whether a chosen file
    needs redirecting, so applying it twice must not produce `x.local.local.yaml`."""
    once = local_config_path("config/flygym_rig.yaml")
    assert once.name == "flygym_rig.local.yaml"
    assert local_config_path(once) == once


# =============================================================================================
# Creating it
# =============================================================================================
def test_a_fresh_rig_gets_an_empty_local_file_and_behaves_exactly_like_the_template(tmp_path):
    """Created EMPTY, not as a copy: until somebody changes something the rig must behave exactly
    as a fresh clone does -- in particular, still imposing nothing on the camera."""
    template = tmp_path / "rig.yaml"
    template.write_text("binning:\n  bin_seconds: 60\n")
    path = ensure_local_config(str(template))
    assert path.endswith(LOCAL_SUFFIX)
    assert os.path.exists(path)

    text = io.open(path, encoding="utf-8").read()
    assert text.strip().startswith("#"), "the new file is not just a comment header"
    import yaml

    assert not (yaml.safe_load(text) or {}), "a fresh local config already overrides something"
    assert load_config(path=path).binning.bin_seconds == 60


def test_making_it_twice_does_not_wipe_what_is_in_it(tmp_path):
    """`ensure_local_config` runs at EVERY launch. Overwriting would delete the rig's settings on
    the next start, which is as bad as this module gets."""
    template = tmp_path / "rig.yaml"
    template.write_text("binning:\n  bin_seconds: 60\n")
    path = ensure_local_config(str(template))
    io.open(path, "a", encoding="utf-8").write("binning:\n  bin_seconds: 5\n")
    again = ensure_local_config(str(template))
    assert again == path
    assert load_config(path=again).binning.bin_seconds == 5, "the rig's own settings were wiped"


def test_a_read_only_install_still_starts(tmp_path, monkeypatch):
    """A shared rig or a locked-down lab machine is a real deployment. Being unable to create a
    settings file is not a reason to refuse to run -- it falls back to the template, which is
    read-only but perfectly usable."""
    template = tmp_path / "rig.yaml"
    template.write_text("binning:\n  bin_seconds: 60\n")

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(io, "open", refuse, raising=False)
    monkeypatch.setattr("builtins.open", refuse)
    path = ensure_local_config(str(template))
    assert path == str(template), "it did not fall back to something usable"


# =============================================================================================
# The shipped template, which is the thing all of this protects
# =============================================================================================
def test_the_shipped_rig_template_still_imposes_nothing_on_the_camera():
    """The invariant the whole mechanism exists to keep true. This is the assertion that 18 other
    tests were making when the app started saving into this file."""
    camera = load_config(path=str(RIG_CONFIG_PATH)).source.camera
    for key in ("width", "height", "frame_rate", "exposure_us", "gain_db"):
        assert camera.get(key) is None, "%s is forced by the SHIPPED template" % key


def test_local_configs_are_not_version_controlled():
    """One machine's tuning must not arrive in somebody else's clone as though it were the
    project's chosen default."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = io.open(os.path.join(root, ".gitignore"), encoding="utf-8").read()
    assert "config/*.local.yaml" in text


# =============================================================================================
# The redirect, in the window
# =============================================================================================
@pytest.mark.parametrize("start_at", ["template", "local"])
def test_saving_never_writes_to_the_shipped_template(tmp_path, start_at, monkeypatch):
    """END TO END through the window's own save. A save aimed at the template is REDIRECTED, not
    refused: the values are the operator's and they asked for them to be kept -- what changes is
    only which file keeps them, and the status line says so by name."""
    pytest.importorskip("PySide6")
    import shutil

    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    # A private copy of the repo's config folder, so the real shipped template cannot be touched
    # even if this test is wrong.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    template = config_dir / "flygym_rig.yaml"
    shutil.copy(str(RIG_CONFIG_PATH), str(template))
    local = config_dir / "flygym_rig.local.yaml"
    monkeypatch.setattr("flygym_tracker.config.TRACKED_TEMPLATES",
                        (DEFAULT_CONFIG_PATH, template))
    before = template.read_text()

    path = str(template if start_at == "template" else local)
    if start_at == "local":
        ensure_local_config(str(template))
    window = MainWindow(config=load_config(path=str(template)), config_path=path,
                        state=gui_state.default_state(), root=str(tmp_path),
                        camera_factory=lambda: None, confirm=lambda text: True)
    try:
        window.controller.commit("activity.pixel_threshold", 17.0)
        window.save_settings()

        assert template.read_text() == before, "THE SHIPPED TEMPLATE WAS MODIFIED"
        assert local.exists(), "the value went nowhere"
        assert load_config(path=str(local)).activity.pixel_threshold == 17.0
        if start_at == "template":
            assert "not the shipped template" in window.settings_view.change_label.text()
    finally:
        window.session.shutdown()


def test_a_save_leaves_a_message_on_screen_at_all(tmp_path):
    """SEPARATE BUG, found while testing the redirect and fixed with it.

    `save_settings` called `set_status(result.message)` and then `refresh_titles()`, which rewrites
    the same label from the change COUNT alone -- so the save result was wiped off the screen every
    single time. The operator saw "no changes" whether the write had succeeded, been refused or
    silently skipped, which is precisely what `set_status`'s docstring says it exists to prevent.
    """
    pytest.importorskip("PySide6")

    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    local = tmp_path / "rig.local.yaml"
    local.write_text("")
    window = MainWindow(config=load_config(), config_path=str(local),
                        state=gui_state.default_state(), root=str(tmp_path),
                        camera_factory=lambda: None, confirm=lambda text: True)
    try:
        window.controller.commit("activity.pixel_threshold", 21.0)
        window.save_settings()
        text = window.settings_view.change_label.text()
        assert "wrote" in text, "the save result was wiped off the screen: %r" % text
        assert str(local) in text or local.name in text, \
            "the message does not name the file that was written: %r" % text
    finally:
        window.session.shutdown()

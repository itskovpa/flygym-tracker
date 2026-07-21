"""Installed, the program must not try to write inside itself.

WHY THIS FILE EXISTS AT ALL. Every path rule here only takes effect when `sys.frozen` is set --
that is, only in the built app, which is the one configuration no test run ever exercises. Left
untested, the first time this code runs is on a customer's machine, and its failure mode is the
worst kind: on Windows, writing into `C:\\Program Files` from a normal user account does not raise.
It silently redirects into a per-user `VirtualStore` shadow copy, so the app looks like it saved
and the files are not where anyone will look. That would land on a three-day experiment.

So `frozen` is simulated here, and every rule is checked against it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from flygym_tracker import config as config_module  # noqa: F401  -- see below
from flygym_tracker import paths

# IMPORTED AT MODULE SCOPE ON PURPOSE. `config.REPO_ROOT` is computed when that module is first
# imported, so importing it for the first time INSIDE the `frozen` fixture would bake this file's
# fake install directory into it for the rest of the pytest process -- and every later GUI test
# would then fail to find `default_config.yaml`. (In the real program that cannot happen: the
# PyInstaller bootloader sets `sys.frozen` before any of this code runs, so the value is right the
# first time and never changes. It is a test-isolation hazard, not a shipping one.)


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    """Pretend to be a PyInstaller build installed at `tmp_path/install`."""
    install = tmp_path / "install"
    (install / "config").mkdir(parents=True)
    monkeypatch.setattr(paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(paths.sys, "executable", str(install / "FlyGymTracker.exe"))
    monkeypatch.delenv("FLYGYM_DATA_DIR", raising=False)
    if hasattr(paths.sys, "_MEIPASS"):
        monkeypatch.delattr(paths.sys, "_MEIPASS")
    return install


# =============================================================================================
# From a clone, nothing changes
# =============================================================================================
def test_a_source_checkout_behaves_exactly_as_it_always_did():
    """The two roots are the SAME folder from a clone, which is what keeps a developer's config,
    state file and output folder where they have always been."""
    assert not paths.is_frozen()
    assert paths.bundle_root() == paths.user_data_root()
    assert (paths.bundle_root() / "src" / "flygym_tracker").is_dir()


def test_the_defaults_stay_relative_from_a_clone():
    assert paths.default_output_dir() == "output"
    assert paths.default_calib_dir() == "calib_faces"
    assert paths.default_config_path() == "config/flygym_rig.local.yaml"


# =============================================================================================
# Installed, they must separate
# =============================================================================================
def test_the_writable_root_is_never_inside_the_install(frozen):
    """THE WHOLE POINT. Everything the program saves -- settings, vial positions, and the results
    of an experiment -- must land somewhere the user can actually write."""
    data = paths.user_data_root()
    assert paths.bundle_root() == frozen
    with pytest.raises(ValueError):
        data.resolve().relative_to(frozen.resolve())


def test_the_shipped_templates_are_read_from_the_install(frozen):
    assert paths.bundle_root() == frozen


def test_the_defaults_become_absolute_once_installed(frozen):
    """A relative default resolves against the working directory, and a desktop shortcut starts the
    app in whatever folder Windows feels like -- so the same button would write a run's results to
    a different place depending on how the app was launched."""
    for value in (paths.default_output_dir(), paths.default_calib_dir(),
                  paths.default_config_path()):
        assert os.path.isabs(value), "%r is relative on an installed copy" % value


def test_the_installed_defaults_all_land_in_the_writable_root(frozen):
    """Not merely absolute -- absolute AND outside the install. An absolute path pointing back into
    `C:\\Program Files` would pass the test above and still fail on the customer's machine."""
    data = str(paths.user_data_root())
    for value in (paths.default_output_dir(), paths.default_calib_dir(),
                  paths.default_config_path()):
        assert value.startswith(data), "%r is not under the writable root" % value


def test_a_shared_rig_can_pin_the_data_folder(frozen, monkeypatch, tmp_path):
    """`FLYGYM_DATA_DIR` is what a locked-down lab machine or a shared rig needs: one folder on a
    data drive that several accounts reach, rather than a copy per user profile."""
    shared = tmp_path / "shared"
    monkeypatch.setenv("FLYGYM_DATA_DIR", str(shared))
    assert paths.user_data_root() == shared


def test_the_override_works_from_a_clone_too(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYGYM_DATA_DIR", str(tmp_path / "elsewhere"))
    assert paths.user_data_root() == tmp_path / "elsewhere"


def test_a_data_folder_that_cannot_be_made_still_returns_its_path(monkeypatch, tmp_path):
    """Never raises: a folder that cannot be created is not a reason to refuse to start, but the
    path must still come back so whatever fails next names the folder somebody has to fix."""
    monkeypatch.setenv("FLYGYM_DATA_DIR", str(tmp_path / "no"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
    assert paths.ensure_user_data_root() == tmp_path / "no"


# =============================================================================================
# Documents, not `~/Documents`
# =============================================================================================
@pytest.mark.skipif(os.name != "nt", reason="Windows folder redirection")
def test_documents_is_resolved_through_windows_rather_than_guessed():
    """IT IS OFTEN NOT `~/Documents`. OneDrive Known Folder Move -- on by default on consumer
    Windows, and on THIS rig -- redirects Documents into the OneDrive folder, leaving the literal
    `~/Documents` missing or a stale empty one. Writing a three-day experiment into the wrong one
    is how results go missing."""
    documents = paths._documents_dir()
    assert documents.is_dir(), "resolved a Documents folder that does not exist: %s" % documents


# =============================================================================================
# The config layering still works across the split
# =============================================================================================
def test_an_installed_copy_keeps_its_settings_out_of_program_files(frozen):
    """`flygym_rig.local.yaml` is what the app WRITES. Left beside its template it would be inside
    the install directory."""
    from flygym_tracker import config

    local = config.local_config_path()
    assert local.name.endswith(".local.yaml")
    with pytest.raises(ValueError):
        local.resolve().relative_to(frozen.resolve())


def test_the_local_file_still_finds_the_template_it_layers_on(frozen):
    """THE FAILURE THIS PREVENTS IS SILENT AND EXPENSIVE. The operator's overrides live under their
    profile while the template lives in Program Files; without the second lookup the local file
    would be read as the WHOLE config rather than as a layer on top, dropping every value the
    template supplies -- including the ones that decide what gets measured."""
    from flygym_tracker import config

    template = frozen / "config" / "flygym_rig.yaml"
    template.write_text("binning:\n  bin_seconds: 10\n", encoding="utf-8")
    # `REPO_ROOT` is bound at import; point it at this fake install for the lookup, and put it back
    # afterwards -- every other test in the process reads the real one.
    original = config_module.REPO_ROOT
    config_module.REPO_ROOT = frozen
    try:
        found = config.template_for_local(paths.user_data_root() / "config"
                                          / "flygym_rig.local.yaml")
        assert found == template, "the local override lost the template underneath it"
    finally:
        config_module.REPO_ROOT = original


# =============================================================================================
# Knowing WHICH BUILD you are looking at
# =============================================================================================
def test_the_two_version_declarations_agree():
    """`__init__.py` drives the installer filename and the title bar; `pyproject.toml` drives the
    wheel. If they drift, the file somebody downloads and the version the app reports stop being
    the same thing -- which is exactly the confusion the version exists to prevent."""
    import re

    import flygym_tracker

    text = (paths.bundle_root() / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'(?m)^version = "([^"]+)"', text).group(1)
    assert declared == flygym_tracker.__version__


def test_the_window_title_names_the_version(qapp, tmp_path):
    """FIVE BUILDS WERE HANDED OVER IN ONE AFTERNOON, all named
    `FlyGymTracker-0.1.0.dev0-Setup.exe`, none of which said which build it was once installed. So
    testing a fix on a second machine meant trusting that the right file had been double-clicked."""
    import flygym_tracker
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    win = MainWindow(config=load_config(), config_path=str(tmp_path / "c.yaml"),
                     state=gui_state.default_state(), root=str(tmp_path),
                     camera_factory=lambda: None, confirm=lambda t: True)
    try:
        assert flygym_tracker.__version__ in win.windowTitle()
    finally:
        win.run.shutdown()
        win.session.shutdown()

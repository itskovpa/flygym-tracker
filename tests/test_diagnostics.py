"""Always-on crash logging: every build leaves a report a customer can send back.

WHY THIS MATTERS ENOUGH TO TEST. The app has been closing itself with a native access violation,
which a windowed build shows nothing for -- the only way to debug it on a machine we cannot see is a
log the app wrote itself. If diagnostics silently fail to install, that capability is gone exactly
when it is needed, and nobody notices until the next unexplained crash on a customer's bench.
"""
from __future__ import annotations

import zipfile

import pytest

from flygym_tracker import diagnostics


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    """Point the user-data root at a temp folder and reset the module's install state."""
    monkeypatch.setenv("FLYGYM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_installed", False)
    monkeypatch.setattr(diagnostics, "_log", None)
    monkeypatch.setattr(diagnostics, "_log_path", None)
    return tmp_path


# =============================================================================================
# Installing
# =============================================================================================
def test_install_writes_a_session_log_with_the_environment(data_dir):
    """The header answers "what is different about that machine" before a single crash -- version,
    OS, CPU count, and whether the MVS camera SDK was even found."""
    path = diagnostics.install(app_version="9.9.9-test")
    assert path is not None
    text = open(path, encoding="utf-8").read()
    assert "9.9.9-test" in text
    assert "cpu count" in text
    assert "MVS SDK" in text, "the camera environment -- the usual culprit -- is not in the header"


def test_install_is_idempotent(data_dir):
    first = diagnostics.install()
    second = diagnostics.install()
    assert first == second


def test_install_never_raises_even_if_the_folder_cannot_be_made(monkeypatch, tmp_path):
    """Diagnostics failing to start is not a reason to stop the app starting."""
    monkeypatch.setenv("FLYGYM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_installed", False)
    monkeypatch.setattr(diagnostics, "_log", None)

    from flygym_tracker import paths

    monkeypatch.setattr(paths, "ensure_user_data_root",
                        lambda: (_ for _ in ()).throw(OSError("denied")))
    assert diagnostics.install() is None          # must not raise


def test_only_the_last_few_session_logs_are_kept(data_dir, monkeypatch):
    """A machine that runs for months must not accumulate a heap of logs."""
    logs = data_dir / "logs"
    logs.mkdir()
    for i in range(diagnostics.KEEP_SESSIONS + 6):
        (logs / ("session_2026010%02d-000000.log" % i)).write_text("old", encoding="utf-8")
    diagnostics.install()
    remaining = list(logs.glob("session_*.log"))
    assert len(remaining) <= diagnostics.KEEP_SESSIONS + 1     # +1 for the one just opened


# =============================================================================================
# Did the last run crash?
# =============================================================================================
def test_a_log_without_the_clean_marker_reads_as_a_crash(data_dir):
    """A native fault leaves the faulthandler dump but never reaches the clean-shutdown line, so
    the ABSENCE of that line is exactly the crash signal -- which is what lets the next launch
    offer to send the report."""
    logs = data_dir / "logs"
    logs.mkdir()
    (logs / "session_20260101-000000.log").write_text(
        "started\nWindows fatal exception: access violation\n", encoding="utf-8")
    diagnostics.install()          # opens THIS session's log (which has no marker yet either)
    crashed = diagnostics.previous_session_crashed()
    assert crashed is not None
    assert "session_20260101-000000.log" in crashed


def test_a_cleanly_closed_previous_session_is_not_a_crash(data_dir):
    logs = data_dir / "logs"
    logs.mkdir()
    (logs / "session_20260101-000000.log").write_text(
        "started\n...\nclean shutdown\n", encoding="utf-8")
    diagnostics.install()
    assert diagnostics.previous_session_crashed() is None


def test_this_sessions_own_log_is_never_mistaken_for_a_past_crash(data_dir):
    """The current log has no clean-shutdown marker yet -- it must not report ITSELF as the crash."""
    diagnostics.install()
    assert diagnostics.previous_session_crashed() is None


# =============================================================================================
# One file to send
# =============================================================================================
def test_collect_report_bundles_the_logs_and_a_system_report(data_dir):
    diagnostics.install(app_version="9.9.9-test")
    diagnostics.write("something happened")
    out = diagnostics.collect_report(destination=str(data_dir / "report.zip"))
    assert out is not None
    with zipfile.ZipFile(out) as bundle:
        names = bundle.namelist()
        assert "system_report.txt" in names
        assert any(n.startswith("logs/session_") for n in names), "the session logs are not in it"
        report = bundle.read("system_report.txt").decode("utf-8")
        assert "cpu count" in report
        # The report carries the PACKAGE version (`__version__`), while the session-log header
        # carries whatever `install()` was handed -- checked separately above.
        assert "version" in report
        session = next(n for n in names if n.startswith("logs/session_"))
        assert "9.9.9-test" in bundle.read(session).decode("utf-8")


def test_the_report_contains_no_experiment_data(data_dir):
    """The dialog promises this. The report is logs + machine details, nothing from a run -- so an
    operator can send it without a second thought about what is in it."""
    diagnostics.install()
    out = diagnostics.collect_report(destination=str(data_dir / "report.zip"))
    with zipfile.ZipFile(out) as bundle:
        for name in bundle.namelist():
            assert "activity" not in name and "behaviour" not in name and "output" not in name

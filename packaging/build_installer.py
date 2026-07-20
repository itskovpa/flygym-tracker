"""Build the whole installer in one command: PyInstaller bundle, then Inno Setup wrapper.

    python packaging/build_installer.py

WHY A SCRIPT AND NOT A README OF STEPS. A release built by hand is built differently every time,
and the differences are invisible until a customer's machine finds one. This also puts the version
in exactly one place -- `flygym_tracker.__version__` -- so the exe, the installer filename and the
About text cannot drift apart.

WHAT IT DOES NOT DO: sign the installer. Without a code-signing certificate Windows SmartScreen
warns every customer that the publisher is unrecognised. That is a commercial purchase, not a
build step, and `INSTALL.md` tells the operator what the warning means and how to verify the file
instead.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGING = Path(__file__).resolve().parent
PROJECT = PACKAGING.parent
DIST = PACKAGING / "dist"
BUILD = PACKAGING / "build"
BUNDLE = DIST / "FlyGymTracker"

#: Where winget puts Inno Setup, plus the classic per-machine location. Checked in order.
ISCC_CANDIDATES = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
    Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
    Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
)


def version() -> str:
    """The one source of truth, read from the package rather than repeated here."""
    sys.path.insert(0, str(PROJECT / "src"))
    import flygym_tracker

    return flygym_tracker.__version__


def find_iscc() -> Path | None:
    for candidate in ISCC_CANDIDATES:
        if candidate.is_file():
            return candidate
    found = shutil.which("ISCC.exe")
    return Path(found) if found else None


def run(command: list, what: str) -> None:
    print("\n=== %s ===\n%s\n" % (what, " ".join(str(c) for c in command)), flush=True)
    result = subprocess.run(command, cwd=str(PROJECT))
    if result.returncode != 0:
        raise SystemExit("%s failed (exit %d)" % (what, result.returncode))


def build_bundle() -> None:
    # CLEAN EVERY TIME. PyInstaller reuses its work directory, so a stale build can carry a file
    # that the current spec no longer includes -- and it would ship.
    for folder in (DIST, BUILD):
        shutil.rmtree(folder, ignore_errors=True)
    run([sys.executable, "-m", "PyInstaller", str(PACKAGING / "flygym_tracker.spec"),
         "--distpath", str(DIST), "--workpath", str(BUILD), "--noconfirm", "--log-level", "WARN"],
        "PyInstaller bundle")
    exe = BUNDLE / "FlyGymTracker.exe"
    if not exe.is_file():
        raise SystemExit("the bundle has no FlyGymTracker.exe -- the build did not produce an app")
    print("bundle: %s (%s)" % (BUNDLE, human(folder_size(BUNDLE))))


def build_installer(app_version: str) -> Path:
    iscc = find_iscc()
    if iscc is None:
        raise SystemExit(
            "Inno Setup was not found. Install it with:\n"
            "    winget install --id JRSoftware.InnoSetup\n"
            "or build only the folder bundle with --bundle-only.")
    run([str(iscc),
         "/DAppVersion=%s" % app_version,
         "/DSourceDir=%s" % BUNDLE,
         "/DOutputDir=%s" % DIST,
         str(PACKAGING / "flygym_tracker.iss")], "Inno Setup installer")
    setup = DIST / ("FlyGymTracker-%s-Setup.exe" % app_version)
    if not setup.is_file():
        raise SystemExit("Inno Setup reported success but %s is not there" % setup.name)
    return setup


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.0f %s" % (size, unit)
        size /= 1024.0
    return "%.0f GB" % size


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-only", action="store_true",
                        help="Build the folder bundle and stop, without the installer.")
    args = parser.parse_args()

    app_version = version()
    print("FlyGym Tracker %s" % app_version)
    build_bundle()
    if args.bundle_only:
        return 0

    setup = build_installer(app_version)
    checksum = sha256(setup)
    # THE CHECKSUM IS PUBLISHED WITH THE RELEASE because the installer is unsigned: it is the only
    # way a customer who is nervous about the SmartScreen warning can verify what they downloaded.
    (setup.parent / (setup.name + ".sha256")).write_text(
        "%s  %s\n" % (checksum, setup.name), encoding="utf-8")

    print("\n" + "=" * 70)
    print("installer : %s" % setup)
    print("size      : %s" % human(setup.stat().st_size))
    print("sha256    : %s" % checksum)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

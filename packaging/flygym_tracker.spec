# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of the FlyGym Tracker window, for a machine with no Python on it.

ONE FOLDER, NOT ONE FILE, and that is a deliberate trade. A one-file build unpacks the whole
bundle -- several hundred megabytes of Qt, OpenCV and NumPy -- into a temp folder on EVERY launch,
which costs seconds of startup and leaves the app dead in the water if the temp drive is full or
scanned by antivirus. A folder starts instantly and is what the installer copies anyway.

WHAT CANNOT GO IN HERE. The HikRobot MVS SDK. `frame_source.HikCameraSource` imports `MvImport`
from `C:\\Program Files (x86)\\MVS\\Development\\Samples\\Python\\MvImport`, and that is a
proprietary vendor package that ships with USB3 Vision DRIVERS -- a driver cannot be delivered by
copying files, and the licence is not ours to redistribute. So MVS stays a prerequisite the
installer checks for and names, and everything else is bundled.
"""
import os
import sys
from pathlib import Path

# `SPECPATH` is where this file lives; the project is its parent.
PROJECT = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

block_cipher = None

datas = [
    # THE SHIPPED CONFIG TEMPLATES. `config.DEFAULT_CONFIG_PATH` reads these from the install
    # directory (see `paths.bundle_root`), and the app cannot start without default_config.yaml.
    (str(PROJECT / "config" / "default_config.yaml"), "config"),
    (str(PROJECT / "config" / "flygym_rig.yaml"), "config"),
]

hiddenimports = [
    # openpyxl's writer is reached through pandas by string, so the analyser never sees it -- and
    # its absence would surface only when a run tried to write its workbook, hours in.
    "openpyxl",
    "openpyxl.cell._writer",
    "pandas._libs.tslibs.base",

    # ---- what the HikRobot MVS SDK needs, and WHY IT MUST BE LISTED BY HAND -------------------
    # THIS IS THE BUG THAT MADE THE RIG CAMERA IMPOSSIBLE TO USE IN ANY INSTALLED BUILD.
    #
    # The MVS SDK is NOT bundled -- it is loaded at runtime from the operator's MVS installation
    # (see `frame_source._import_sdk`). PyInstaller therefore never analyses it, never sees its
    # imports, and does not bundle them. `MvCameraControl_class.py` opens with `import platform`,
    # nothing in this application imports `platform` itself, so the module was absent from the
    # build and the SDK import died with `ModuleNotFoundError: No module named 'platform'`.
    #
    # The symptom was perfect camouflage: the app listed the built-in webcam (OpenCV is bundled)
    # and no rig camera, on every machine, which reads as "the camera is not connected properly".
    # It was reported from a second PC and blamed first on a hard-coded SDK path -- a genuine bug,
    # but not this one.
    #
    # Every module the MVS SDK's Python files import, taken from the SDK source rather than
    # guessed. Keep this in step if HikRobot ship a new SDK; the test suite checks it.
    "platform",
    "copy",
    "ctypes",
    "ctypes.util",
    "os",
    "sys",
]

excludes = [
    # ---- machine-learning stack that this app does not use at all -----------------------------
    # MEASURED, NOT ASSUMED. The first build came to 892 MB, of which torch was 323 MB, llvmlite
    # 102 MB, scipy 53 MB and onnxruntime 32 MB. None of them is imported by anything here: a
    # runtime check that imported numpy, cv2, pandas, yaml, openpyxl, PySide6 and this package's
    # own modules and then read `sys.modules` found not one of them loaded. They are dragged in
    # statically by a hook doing `collect_all` on a package that merely *could* use them, and they
    # happen to be installed in this Python because the machine is also used for other work.
    #
    # This is why the build must never be trusted to a machine's ambient site-packages: what a
    # customer downloads would otherwise depend on what the developer happened to `pip install`.
    "torch", "torchvision", "torchaudio", "scipy", "numba", "llvmlite", "onnxruntime",
    "sympy", "h5py", "sklearn", "scikit-learn", "tensorboard", "transformers", "networkx",
    # PIL arrives with the same crowd; openpyxl only needs it to EMBED images, which nothing here
    # does -- the workbook this app writes is numbers.
    "PIL", "Pillow",
    # ---- developer and plotting tooling -------------------------------------------------------
    # `tkinter` in particular gets pulled in by anything that touches matplotlib.
    "matplotlib", "tkinter", "PyQt5", "PyQt6", "PySide2", "IPython", "jupyter", "notebook",
    "pytest", "sphinx", "setuptools", "pip",
    # Qt modules this app never opens. WebEngine alone is ~130 MB.
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick", "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtMultimedia",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtBluetooth",
    "PySide6.QtPositioning", "PySide6.QtNetworkAuth", "PySide6.QtSql", "PySide6.QtTest",
]

a = Analysis(
    [str(PROJECT / "packaging" / "launcher.py")],
    pathex=[str(PROJECT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FlyGymTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-packed Qt DLLs are a reliable way to get flagged by antivirus
    console=False,      # a windowed app; startup failures are reported by `launcher.py`
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FlyGymTracker",
)

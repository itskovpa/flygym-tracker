"""Which experiment: the config file, the vial positions, and where results go.

THIS BAND EXISTS TO DELETE A BATCH FILE HEADER. Today the only way to change any of these is to
open `run.bat` in Notepad and edit:

    set "CONFIG=config\\flygym_rig.yaml"
    set "CALIB=calib_faces"
    set "OUTDIR=output"

That is squarely what "I want all the settings to be in one usable GUI, not as command line
prompts" was about, and a settings app that still needed Notepad for the path to the file it edits
would be a strange object.

THE PATH FIELDS ARE READ-ONLY, WITH A PICKER. A free-text path field invites a typo that is only
discovered when the run cannot find the calibration bundle -- half an hour into setting up an
experiment. The config field is the exception: it is an editable `QComboBox` because its recent
list is how an operator alternates between the rig's config and the one they are trying, and
because typing a path there is occasionally the fastest way to reach a network share.

THE ORDER OF THE WINDOW IS DELIBERATE, and this is band two of it: whose camera is it -> WHICH
EXPERIMENT -> what do I see -> what am I imposing -> am I ready.
"""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QSizePolicy, QToolButton, QVBoxLayout, QWidget)


class PathField(QWidget):
    """A read-only path plus a "..." button. One row of the grid."""

    picked = Signal(str)

    def __init__(self, *, directory: bool, caption: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._directory = directory
        self._caption = caption
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        layout.addWidget(self.edit, 0, 0)
        self.button = QToolButton()
        self.button.setText("...")
        self.button.clicked.connect(self._pick)
        layout.addWidget(self.button, 0, 1)

    def value(self) -> str:
        return self.edit.text()

    def set_value(self, path: str) -> None:
        self.edit.setText(path or "")
        self.edit.setToolTip(os.path.abspath(path) if path else "")

    def _pick(self) -> None:
        start = self.edit.text() or os.getcwd()
        if self._directory:
            chosen = QFileDialog.getExistingDirectory(self, self._caption, start)
        else:
            chosen, _filter = QFileDialog.getOpenFileName(self, self._caption, start,
                                                          "Config files (*.yaml *.yml);;All files (*)")
        if chosen:
            self.set_value(chosen)
            self.picked.emit(chosen)


class SessionBar(QWidget):
    """Config file, vial-position folder, output folder -- and what camera is being asked for."""

    config_changed = Signal(str)
    calib_changed = Signal(str)
    output_changed = Signal(str)

    def __init__(self, state: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # THE HEADER, AND WHY COLLAPSING DOES NOT JUST HIDE THIS BAND. These four paths decide
        # which experiment runs, where its results land and which camera is opened -- they are set
        # once at the start of a session and then never touched, which is exactly what makes four
        # full-width rows of them a poor use of the screen for the other three days.
        #
        # Collapsed, the header keeps a ONE-LINE SUMMARY of what is chosen. A collapsed section
        # that showed only its title would hide the answer to "am I about to overwrite yesterday's
        # output folder", which is a question worth answering without a click.
        header = QHBoxLayout()
        header.setContentsMargins(10, 4, 10, 4)
        header.setSpacing(8)
        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setAutoRaise(True)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle.setToolTip("Show or hide the experiment's paths")
        self.toggle.toggled.connect(self.set_expanded)
        header.addWidget(self.toggle)

        title = QLabel("EXPERIMENT")
        title.setProperty("role", "grouptitle")
        header.addWidget(title)

        self.summary = QLabel("")
        self.summary.setProperty("role", "note")
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        header.addWidget(self.summary, 1)
        outer.addLayout(header)

        self.body = QWidget()
        outer.addWidget(self.body)
        layout = QGridLayout(self.body)
        layout.setContentsMargins(10, 2, 10, 6)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)

        layout.addWidget(_label("config file"), 0, 0)
        self.config_combo = QComboBox()
        self.config_combo.setEditable(True)
        self.config_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.config_combo.setMinimumWidth(320)
        for path in state.get("recent_configs") or []:
            self.config_combo.addItem(path)
        current = state.get("config_path") or ""
        if current and self.config_combo.findText(current) < 0:
            self.config_combo.insertItem(0, current)
        self.config_combo.setCurrentText(current)
        self.config_combo.activated.connect(self._on_config_activated)
        self.config_combo.lineEdit().editingFinished.connect(self._on_config_typed)
        layout.addWidget(self.config_combo, 0, 1)

        self.config_browse = QToolButton()
        self.config_browse.setText("Browse...")
        self.config_browse.clicked.connect(self._browse_config)
        layout.addWidget(self.config_browse, 0, 2)

        layout.addWidget(_label("vial positions"), 1, 0)
        self.calib_field = PathField(directory=True, caption="Folder holding the vial positions")
        self.calib_field.set_value(state.get("calib_dir") or "")
        self.calib_field.picked.connect(self.calib_changed)
        self.calib_field.picked.connect(self._on_path_picked)
        layout.addWidget(self.calib_field, 1, 1, 1, 2)

        layout.addWidget(_label("results go to"), 2, 0)
        self.output_field = PathField(directory=True, caption="Folder for the results")
        self.output_field.set_value(state.get("output_dir") or "")
        self.output_field.picked.connect(self.output_changed)
        self.output_field.picked.connect(self._on_path_picked)
        layout.addWidget(self.output_field, 2, 1, 1, 2)

        # Which physical camera the config asks for. It is READ-ONLY here because it is a property
        # of the config file, edited there -- but it is SHOWN, because `cli._camera_source_from_
        # config` reads a `source.camera.index` key that no shipped YAML defines, and an
        # undocumented knob the app inherits silently is exactly the kind of thing that makes a
        # two-camera bench behave differently from a one-camera one for no visible reason.
        layout.addWidget(_label("camera"), 3, 0)
        self.camera_label = QLabel("")
        self.camera_label.setProperty("role", "note")
        layout.addWidget(self.camera_label, 3, 1, 1, 2)

        layout.setColumnStretch(1, 1)
        self._refresh_summary()

    # -- collapsing --------------------------------------------------------------------------
    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the path rows. The header and its summary always stay."""
        expanded = bool(expanded)
        if self.toggle.isChecked() != expanded:
            self.toggle.setChecked(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(expanded)
        self._refresh_summary()

    def is_expanded(self) -> bool:
        return self.body.isVisible()

    def _refresh_summary(self) -> None:
        """What is chosen, in one line, named by folder rather than by full path.

        Basenames because the full paths are long enough to push each other off the end of the
        line, and the question a collapsed header has to answer is "is this the right bundle and
        the right output folder", which the last component answers. The full path is the tooltip.
        """
        parts = [os.path.basename(self.config_path()) or "no config",
                 os.path.basename(self.calib_field.value().rstrip("\\/")) or "no vial positions",
                 os.path.basename(self.output_field.value().rstrip("\\/")) or "no output folder"]
        self.summary.setText("   ".join(parts))
        self.summary.setToolTip("config: %s\nvial positions: %s\nresults: %s"
                                % (self.config_path(), self.calib_field.value(),
                                   self.output_field.value()))

    def set_camera_identity(self, serial, index) -> None:
        """Say which camera will be opened, in the terms the config actually selects by.

        Serial wins in `HikCameraSource._find_device` and index is only consulted when no serial is
        pinned, so the line names whichever one is load-bearing rather than printing both.
        """
        if serial:
            self.camera_label.setText("serial %s (pinned - index is ignored)" % serial)
        else:
            self.camera_label.setText(
                "no serial pinned, so device index %s is used - pin a serial in the config if this "
                "bench has more than one camera" % (index if index is not None else 0))

    def config_path(self) -> str:
        return self.config_combo.currentText().strip()

    def set_recent(self, recent) -> None:
        current = self.config_combo.currentText()
        self.config_combo.blockSignals(True)
        self.config_combo.clear()
        for path in recent:
            self.config_combo.addItem(path)
        self.config_combo.setCurrentText(current)
        self.config_combo.blockSignals(False)

    def _on_path_picked(self, _path: str) -> None:
        self._refresh_summary()

    def _on_config_activated(self, _index: int) -> None:
        self._refresh_summary()
        self.config_changed.emit(self.config_path())

    def _on_config_typed(self) -> None:
        self._refresh_summary()
        self.config_changed.emit(self.config_path())

    def _browse_config(self) -> None:
        start = self.config_path() or os.getcwd()
        chosen, _f = QFileDialog.getOpenFileName(self, "Config file to edit", start,
                                                 "Config files (*.yaml *.yml);;All files (*)")
        if chosen:
            self.config_combo.setCurrentText(chosen)
            self._refresh_summary()
            self.config_changed.emit(chosen)


def _label(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "note")
    return label

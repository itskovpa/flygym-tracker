"""Camera-independent monitoring and control for the FlyGym rotation device."""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QSpinBox, QTabWidget, QTableWidget,
                               QHeaderView, QScrollArea,
                               QVBoxLayout, QWidget)

from flygym_tracker.rig_device import RigDeviceService


class SensorPlot(QWidget):
    """Small dependency-free rolling plot of ADC average and digital sensor states."""

    def __init__(self, parent=None, *, limit: int = 300):
        super().__init__(parent)
        self._points = deque(maxlen=max(2, int(limit)))
        self.setMinimumHeight(150)

    def set_readings(self, readings) -> None:
        self._points.clear()
        self._points.extend(list(readings)[-self._points.maxlen:])
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor("#111820"))
            if len(self._points) < 2:
                painter.setPen(QColor("#8c99a5"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                                 "Waiting for sensor samples")
                return
            values = [p.adc_avg for p in self._points]
            low, high = min(values), max(values)
            span = max(1, high - low)
            left, top, width, height = 48.0, 20.0, max(1.0, self.width() - 58.0), \
                max(1.0, self.height() - 44.0)
            times = [p.received_monotonic for p in self._points]
            time_span = max(0.001, times[-1] - times[0])
            points = QPolygonF([
                QPointF(left + (reading.received_monotonic - times[0]) * width / time_span,
                        top + height - (value - low) * height / span)
                for reading, value in zip(self._points, values)
            ])
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("#65c7f2"), 2))
            painter.drawPolyline(points)
            for reading in self._points:
                x = left + (reading.received_monotonic - times[0]) * width / time_span
                if reading.raw:
                    painter.setPen(QPen(QColor("#ffb454"), 1))
                    painter.drawLine(int(x), int(top), int(x), int(top + 5))
                if reading.debounced:
                    painter.setPen(QPen(QColor("#83d17a"), 1))
                    painter.drawLine(int(x), int(top + height - 5), int(x), int(top + height))
            painter.setPen(QColor("#b7c2cc"))
            painter.drawText(2, 14, str(high))
            painter.drawText(2, int(top + height), str(low))
            painter.drawText(int(left), self.height() - 5, "%.1f s" % time_span)
            painter.drawText(int(left + width - 150), self.height() - 5,
                             "raw orange | debounced green")
        finally:
            painter.end()


class DeviceControlWindow(QMainWindow):
    """Monitor a rig device and expose only protocol-defined mutations."""

    def __init__(self, service=None, *, service_factory: Callable[..., object] = RigDeviceService,
                 parent=None, poll_ms: int = 100):
        super().__init__(parent)
        self.setWindowTitle("FlyGym device control")
        self._service = service
        self._service_port = str(service.port) if service is not None else None
        self._service_factory = service_factory
        self._started = service is not None
        self._pending_id: Optional[str] = None
        self._pending_command: Optional[str] = None
        self._command_queue = deque()
        self._response_requests = {}
        self._response_payloads = {}
        self._settings_dirty = False
        self._program_dirty = False
        self._settings_loaded = False
        self._loaded_program_id = None
        self._program_count = None
        self._max_programs = 10
        self._last_boot_id = None
        self._delete_requested_id = None
        self._rig_port_detected = False
        self._build()
        self._connect_widgets()
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    @property
    def service(self):
        return self._service

    def _build(self) -> None:
        body = QWidget()
        outer = QVBoxLayout(body)

        connection = QHBoxLayout()
        connection.addWidget(QLabel("USB port"))
        self.port_box = QComboBox()
        self.port_box.setEditable(True)
        self.port_box.lineEdit().setPlaceholderText("No rig USB port found")
        connection.addWidget(self.port_box, 1)
        self.scan_button = QPushButton("Scan")
        connection.addWidget(self.scan_button)
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        connection.addWidget(self.connect_button)
        connection.addWidget(self.disconnect_button)
        outer.addLayout(connection)

        self.connection_label = QLabel("Disconnected — monitoring starts when you connect")
        self.connection_label.setWordWrap(True)
        outer.addWidget(self.connection_label)

        status = QGroupBox("Device status")
        form = QFormLayout(status)
        self.status_labels = {}
        for key, title in (("boot_id", "Boot"), ("build", "Build"), ("owner", "Control"),
                           ("phase", "Phase"), ("position", "Position"),
                           ("motion", "Motion"), ("fault", "Fault"),
                           ("sensor", "Sensor")):
            label = QLabel("—")
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.status_labels[key] = label
            form.addRow(title, label)
        outer.addWidget(status)

        ownership = QHBoxLayout()
        self.take_control_button = QPushButton("Take USB control")
        self.release_control_button = QPushButton("Return local control")
        ownership.addWidget(self.take_control_button)
        ownership.addWidget(self.release_control_button)
        ownership.addStretch(1)
        outer.addLayout(ownership)

        tabs = QTabWidget()
        tabs.addTab(self._motion_tab(), "Rotation")
        tabs.addTab(self._scroll_tab(self._program_tab()), "Programs")
        tabs.addTab(self._scroll_tab(self._settings_tab()), "Settings")
        tabs.setMaximumHeight(240)
        outer.addWidget(tabs)

        sensor_box = QGroupBox("Live sensor — ADC average")
        sensor_layout = QVBoxLayout(sensor_box)
        self.sensor_plot = SensorPlot()
        sensor_layout.addWidget(self.sensor_plot)
        self.sensor_detail = QLabel("ADC —   — mV   raw —   debounced —")
        sensor_layout.addWidget(self.sensor_detail)
        outer.addWidget(sensor_box, 1)

        self.command_label = QLabel("No command sent")
        self.command_label.setWordWrap(True)
        outer.addWidget(self.command_label)
        self.setCentralWidget(body)
        self.resize(720, 760)
        self.scan_ports()

    @staticmethod
    def _scroll_tab(widget):
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        return scroll

    @property
    def port_edit(self):
        """Editable line component retained as a small convenience for callers and tests."""
        return self.port_box.lineEdit()

    def scan_ports(self) -> None:
        current = self.port_box.currentText().strip()
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
        except Exception:
            ports = []
        self.port_box.clear()
        preferred = None
        for port in ports:
            device = str(port.device)
            self.port_box.addItem(device)
            if ((port.vid, port.pid) == (0x239A, 0x80F7)
                    or str(getattr(port, "serial_number", "")) == "DF63C856E7615633"):
                preferred = device
        self._rig_port_detected = preferred is not None
        if preferred:
            self.port_box.setCurrentText(preferred)
        elif current and any(str(port.device) == current for port in ports):
            self.port_box.setCurrentText(current)
        else:
            self.port_box.setCurrentIndex(-1)

    def _motion_tab(self):
        tab = QWidget()
        row = QHBoxLayout(tab)
        self.home_button = QPushButton("Home")
        self.half_turn_button = QPushButton("Half-turn")
        self.abort_motion_button = QPushButton("Abort motion")
        for button in (self.home_button, self.half_turn_button, self.abort_motion_button):
            row.addWidget(button)
        row.addStretch(1)
        return tab

    def _program_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        row.addWidget(QLabel("Program"))
        self.program_box = QComboBox()
        self.program_box.addItem("Program 1", 0)
        row.addWidget(self.program_box)
        self.start_program_button = QPushButton("Start")
        self.pause_program_button = QPushButton("Pause")
        self.resume_program_button = QPushButton("Resume")
        self.cancel_program_button = QPushButton("Cancel program")
        self.cancel_program_button.setToolTip(
            "Stop scheduling future stages. The current half-turn finishes; use Abort motion "
            "to interrupt it.")
        for button in (self.start_program_button, self.pause_program_button,
                       self.resume_program_button, self.cancel_program_button):
            row.addWidget(button)
        layout.addLayout(row)
        edit_row = QHBoxLayout()
        self.read_program_button = QPushButton("Read program")
        self.create_program_button = QPushButton("Create")
        self.delete_program_button = QPushButton("Delete")
        self.stage_count = QSpinBox()
        self.stage_count.setRange(1, 10)
        self.stage_count.setPrefix("Stages: ")
        self.save_program_button = QPushButton("Save program")
        edit_row.addWidget(self.read_program_button)
        edit_row.addWidget(self.create_program_button)
        edit_row.addWidget(self.delete_program_button)
        edit_row.addWidget(self.stage_count)
        edit_row.addWidget(self.save_program_button)
        edit_row.addStretch(1)
        layout.addLayout(edit_row)
        self.program_table = QTableWidget(10, 4)
        self.program_table.setMaximumHeight(180)
        self.program_table.setEnabled(False)
        self.program_table.setHorizontalHeaderLabels(
            ["Turning (min)", "Resting (min)", "Half-turns/min", "Cycles"])
        self.program_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.program_cells = []
        for row in range(10):
            cells = []
            for column in range(4):
                spin = QSpinBox()
                spin.setRange(0 if column == 1 else 1, 65535)
                spin.valueChanged.connect(self._program_edited)
                self.program_table.setCellWidget(row, column, spin)
                cells.append(spin)
            self.program_cells.append(cells)
        layout.addWidget(self.program_table)
        self.stage_count.setEnabled(False)
        self.save_program_button.setEnabled(False)
        return tab

    def _settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.refresh_settings_button = QPushButton("Read settings from device")
        layout.addWidget(self.refresh_settings_button)
        form = QFormLayout()
        specs = (("ACCEL", 5000, 400000, " steps/s²"),
                 ("SPEED", 100, 100000, " steps/s"),
                 ("SPEED_HOMING_SEEK", 100, 9000, " steps/s"),
                 ("ACCEL_HOMING", 10, 9000, " steps/s²"),
                 ("SPEED_HOMING", 10, 9000, " steps/s"),
                 ("HOME_ADJUST", -500, 500, " pulses"),
                 ("SENSOR_POLL_INTERVAL", 1, 1000, " ms"),
                 ("PULL_OFF_DIST", 1, 12800, " pulses"))
        self.setting_spins = {}
        for name, low, high, suffix in specs:
            spin = QSpinBox()
            spin.setRange(low, high)
            spin.setSuffix(suffix)
            spin.setEnabled(False)
            spin.valueChanged.connect(self._setting_edited)
            self.setting_spins[name] = spin
            form.addRow(name.replace("_", " ").title(), spin)
        layout.addLayout(form)
        self.save_settings_button = QPushButton("Save settings to device")
        self.save_settings_button.setEnabled(False)
        layout.addWidget(self.save_settings_button)
        return tab

    def _connect_widgets(self) -> None:
        self.connect_button.clicked.connect(self.connect_device)
        self.disconnect_button.clicked.connect(self.disconnect_device)
        self.scan_button.clicked.connect(self.scan_ports)
        commands = ((self.take_control_button, "TAKE_CONTROL"),
                    (self.release_control_button, "RELEASE_CONTROL"),
                    (self.home_button, "HOME"), (self.half_turn_button, "HALF_TURN"),
                    (self.abort_motion_button, "ABORT_MOTION"),
                    (self.pause_program_button, "PAUSE_PROGRAM"),
                    (self.resume_program_button, "RESUME_PROGRAM"),
                    (self.cancel_program_button, "CANCEL_PROGRAM"),
                    (self.refresh_settings_button, "GET_SETTINGS"))
        for button, command in commands:
            button.clicked.connect(lambda _checked=False, name=command: self._send(name))
        self.start_program_button.clicked.connect(
            lambda: self._send("START_PROGRAM", {"program_id": self.program_box.currentData()}))
        self.read_program_button.clicked.connect(self.read_program)
        self.create_program_button.clicked.connect(lambda: self._send("CREATE_PROGRAM"))
        self.delete_program_button.clicked.connect(self.delete_program)
        self.save_program_button.clicked.connect(self.save_program)
        self.save_settings_button.clicked.connect(self.save_settings)
        self.stage_count.valueChanged.connect(self._program_edited)
        self.stage_count.valueChanged.connect(self._update_stage_rows)
        self.program_box.currentIndexChanged.connect(self._program_selection_changed)

    def _program_selection_changed(self, _index=None):
        self._program_dirty = False
        self._loaded_program_id = None
        self.program_table.setEnabled(False)
        self.stage_count.setEnabled(False)
        self.save_program_button.setEnabled(False)

    def _update_stage_rows(self, count):
        for row, cells in enumerate(self.program_cells):
            for spin in cells:
                spin.setEnabled(row < int(count))

    def _setting_edited(self, _value=None):
        if any(spin.isEnabled() for spin in self.setting_spins.values()):
            self._settings_dirty = True
            self.save_settings_button.setEnabled(True)

    def _program_edited(self, _value=None):
        if self.program_table.isEnabled():
            self._program_dirty = True
            self.save_program_button.setEnabled(True)

    def read_program(self):
        self._send("GET_PROGRAMS", {"program_id": self.program_box.currentData()})

    def delete_program(self):
        self._delete_requested_id = self.program_box.currentData()
        self._send("DELETE_PROGRAM", {"program_id": self._delete_requested_id})

    def save_settings(self):
        if not self._settings_dirty or not self._settings_loaded:
            return
        commands = [("SET_SETTING", {"name": name, "value": spin.value()})
                    for name, spin in self.setting_spins.items()]
        commands.append(("SAVE_SETTINGS", {}))
        self._queue_commands(commands)

    def save_program(self):
        if not self._program_dirty or self._loaded_program_id != self.program_box.currentData():
            return
        program_id = self.program_box.currentData()
        count = self.stage_count.value()
        fields = ("time_turning", "time_resting", "rot_per_min", "num_cycles")
        commands = []
        for stage_id in range(count):
            values = {name: spin.value() for name, spin in
                      zip(fields, self.program_cells[stage_id])}
            commands.append(("SET_PROGRAM_STAGE",
                             dict(program_id=program_id, stage_id=stage_id, **values)))
        commands.extend((("SET_PROGRAM_STAGE_COUNT", {"program_id": program_id,
                                                        "count": count}),
                         ("SAVE_PROGRAMS", {})))
        self._queue_commands(commands)

    def _queue_commands(self, commands):
        if self._pending_id is not None or self._command_queue:
            return
        snapshot = self._service.snapshot() if self._service is not None else None
        if (snapshot is None or not bool(getattr(self._service, "owns_control", False))
                or snapshot.message["moving"]):
            return
        self._command_queue.extend(commands)
        self._send_next()

    def _send_next(self):
        if self._pending_id is None and self._command_queue:
            command, args = self._command_queue.popleft()
            self._send(command, args)

    def connect_device(self) -> None:
        port = self.port_edit.text().strip()
        if not port:
            self.connection_label.setText("No rig USB port is available")
            return
        if self._service is None or port != self._service_port:
            if self._service is not None:
                self._service.close()
            self._service = self._service_factory(port)
            self._service_port = port
        self._service.start()
        self._started = True
        self.connection_label.setText("Opening port; waiting for valid status…")
        self.refresh()

    def disconnect_device(self) -> None:
        if self._service is not None:
            self._service.close()
        self._started = False
        self._abandon_pending("Connection closed; pending command outcome is uncertain")
        self._invalidate_editors()
        self.refresh()

    def _abandon_pending(self, message=None):
        if message and (self._pending_id is not None or self._command_queue):
            self.command_label.setText(message)
        if self._pending_id is not None:
            self._response_requests.pop(self._pending_id, None)
            self._response_payloads.pop(self._pending_id, None)
        self._pending_id = None
        self._pending_command = None
        self._command_queue.clear()
        self._delete_requested_id = None

    def _invalidate_editors(self):
        self._settings_loaded = False
        self._settings_dirty = False
        for spin in self.setting_spins.values():
            spin.setEnabled(False)
        self._loaded_program_id = None
        self._program_dirty = False
        self.program_table.setEnabled(False)
        self.stage_count.setEnabled(False)

    def _send(self, command: str, args=None) -> None:
        # Enforce the same guard as the disabled buttons. Programmatic clicks on disabled Qt
        # widgets can still emit, so presentation alone is not a safety boundary.
        snapshot = self._service.snapshot() if self._service is not None else None
        if snapshot is None or (self._pending_id is not None and command != "ABORT_MOTION"):
            return
        message = snapshot.message
        read_only = command in ("GET_STATUS", "GET_SETTINGS", "GET_PROGRAMS", "PING", "HELLO")
        if command == "ABORT_MOTION" and self._pending_id is not None:
            self._abandon_pending("Earlier command superseded by emergency abort")
        if read_only:
            allowed = True
        elif command == "TAKE_CONTROL":
            allowed = message["owner"] == "LOCAL" and not message["moving"]
        elif command == "ABORT_MOTION":
            allowed = bool(getattr(self._service, "owns_control", False)) and message["moving"]
        elif command in ("PAUSE_PROGRAM", "CANCEL_PROGRAM"):
            allowed = bool(getattr(self._service, "owns_control", False))
        else:
            allowed = bool(getattr(self._service, "owns_control", False)) and not message["moving"]
        if not allowed:
            self._abandon_pending("Command blocked because device state or USB ownership changed")
            return
        try:
            self._pending_command = command
            self._pending_id = self._service.send_command(command, args)
            if command in ("GET_SETTINGS", "GET_PROGRAMS", "CREATE_PROGRAM"):
                self._response_requests[self._pending_id] = command
            self.command_label.setText("%s sent; waiting for acceptance" % command)
        except Exception as exc:
            self._abandon_pending()
            self.command_label.setText(
                "%s outcome is uncertain after a write error; it will not be retried: %s"
                % (command, exc))
        self.refresh()

    def refresh(self) -> None:
        service = self._service
        snapshot = service.snapshot() if service is not None else None
        connected = snapshot is not None
        self.connect_button.setEnabled(not self._started)
        self.disconnect_button.setEnabled(self._started)
        self.port_box.setEnabled(not self._started)
        self.scan_button.setEnabled(not self._started)

        if snapshot is None and self._pending_id is not None:
            self._abandon_pending("Status became stale; pending command outcome is uncertain")
            self._invalidate_editors()

        if self._pending_id is not None:
            request_id = self._pending_id
            expected_response = self._response_requests.get(request_id)
            response = (service.response(request_id)
                        if service is not None and hasattr(service, "response") else None)
            if response is not None:
                self._response_payloads[request_id] = response
            state = (service.command_state(request_id)
                     if service is not None and self._pending_id == request_id else None)
            if state is not None:
                detail = ": %s" % state.detail if state.detail else ""
                self.command_label.setText("%s %s%s" % (state.command, state.state, detail))
                if state.state in ("rejected", "aborted", "uncertain"):
                    self._abandon_pending()
                elif (state.state == "completed"
                      and (expected_response is None or request_id in self._response_payloads)):
                    try:
                        payload = self._response_payloads.get(request_id)
                        if expected_response == "GET_SETTINGS":
                            self._load_settings(payload)
                        elif expected_response in ("GET_PROGRAMS", "CREATE_PROGRAM"):
                            self._load_programs(payload,
                                                select_program=expected_response == "CREATE_PROGRAM")
                    except (KeyError, TypeError, ValueError, OverflowError) as exc:
                        self._abandon_pending()
                        self.command_label.setText("Invalid device response: %s" % exc)
                        state = None
                    if state is None:
                        return
                    self._response_requests.pop(request_id, None)
                    self._response_payloads.pop(request_id, None)
                    self._pending_id = None
                    self._pending_command = None
                    if state.command == "SAVE_SETTINGS":
                        self._settings_dirty = False
                    if state.command == "SAVE_PROGRAMS":
                        self._program_dirty = False
                    if state.command == "DELETE_PROGRAM":
                        deleted = self._delete_requested_id
                        self._delete_requested_id = None
                        old_count = self._program_count
                        self._invalidate_editors()
                        if deleted is not None and old_count is not None and old_count > 1:
                            target = min(deleted, old_count - 2)
                            self.program_box.blockSignals(True)
                            self.program_box.setCurrentIndex(target)
                            self.program_box.blockSignals(False)
                            self._command_queue.appendleft(
                                ("GET_PROGRAMS", {"program_id": target}))
            if self._pending_id is None:
                self._send_next()

        if snapshot is None:
            if self._settings_loaded or self._loaded_program_id is not None:
                self._invalidate_editors()
            error = getattr(service, "last_error", None) if service is not None else None
            idle_text = ("Disconnected — no FlyGym USB device detected"
                         if not self._rig_port_detected else
                         "Disconnected — monitoring starts when you connect")
            self.connection_label.setText(error or ("Waiting for fresh valid status…" if service
                                                     else idle_text))
            for label in self.status_labels.values():
                label.setText("—")
            self.sensor_detail.setText("ADC —   — mV   raw —   debounced —")
            self.sensor_plot.set_readings(service.sensor_history() if service is not None else [])
            self._set_action_states(None)
            return

        message = snapshot.message
        if self._last_boot_id is not None and message["boot_id"] != self._last_boot_id:
            self._invalidate_editors()
        self._last_boot_id = message["boot_id"]
        self.connection_label.setText("Connected to %s; status is fresh" % service.port)
        self.status_labels["boot_id"].setText(str(message["boot_id"]))
        self.status_labels["build"].setText(str(message.get("build_id") or "not reported"))
        if bool(getattr(service, "owns_control", False)):
            owner_text = "USB — controlled by this app"
        elif message["owner"] == "USB":
            owner_text = "USB — controlled elsewhere"
        else:
            owner_text = "Local controls"
        self.status_labels["owner"].setText(owner_text)
        self.status_labels["phase"].setText("—" if message["phase"] is None else str(message["phase"]))
        self.status_labels["position"].setText(
            "unknown" if not message["position_valid"] else
            ("homed" if message["homed"] else "valid"))
        self.status_labels["motion"].setText("moving (id %s)" % message["move_id"]
                                              if message["moving"] else "idle")
        self.status_labels["fault"].setText(str(message["fault"] or "none"))
        self.status_labels["sensor"].setText("raw %s; debounced %s" %
                                               (message["sensor_raw"], message["sensor_debounced"]))
        self.sensor_detail.setText("ADC %d   %d mV   raw %s   debounced %s" %
                                   (message["sensor_adc_avg"], message["sensor_mv"],
                                    message["sensor_raw"], message["sensor_debounced"]))
        self.sensor_plot.set_readings(service.sensor_history())
        self._set_action_states(message)

    def _set_action_states(self, message) -> None:
        pending = self._pending_id is not None
        fresh = message is not None
        usb = fresh and bool(getattr(self._service, "owns_control", False))
        moving = fresh and message["moving"]
        self.take_control_button.setEnabled(fresh and message["owner"] == "LOCAL"
                                            and not moving and not pending)
        self.release_control_button.setEnabled(usb and not moving and not pending)
        for button in (self.home_button, self.half_turn_button, self.start_program_button,
                       self.resume_program_button):
            button.setEnabled(usb and not moving and not pending)
        # These stop future program scheduling and remain available during a physical move.
        self.pause_program_button.setEnabled(usb and not pending)
        self.cancel_program_button.setEnabled(usb and not pending)
        # Abort is the escape from an in-progress move, including one whose HOME/HALF_TURN
        # command has only been accepted. Do not let the pending-command guard remove it.
        self.abort_motion_button.setEnabled(usb and moving)
        self.refresh_settings_button.setEnabled(fresh and not pending)
        self.read_program_button.setEnabled(fresh and not pending)
        self.create_program_button.setEnabled(
            usb and not pending and (self._program_count is None
                                     or self._program_count < self._max_programs))
        self.delete_program_button.setEnabled(
            usb and not pending and self._program_count is not None and self._program_count > 1)
        self.save_settings_button.setEnabled(usb and self._settings_dirty and not pending)
        self.save_program_button.setEnabled(usb and self._program_dirty and not pending)

    def _load_settings(self, response):
        if response.get("type") != "settings":
            raise ValueError("expected settings payload")
        values = response.get("values")
        bounds = response.get("bounds", {})
        if not hasattr(values, "get"):
            raise ValueError("settings values are missing")
        missing = set(self.setting_spins) - set(values)
        if missing:
            raise ValueError("settings missing %s" % ", ".join(sorted(missing)))
        checked = []
        for name, spin in self.setting_spins.items():
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("%s is not an integer" % name)
            device_bounds = bounds.get(name) if hasattr(bounds, "get") else None
            if device_bounds is not None:
                if len(device_bounds) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                                                  for v in device_bounds):
                    raise ValueError("%s bounds are invalid" % name)
                low, high = int(device_bounds[0]), int(device_bounds[1])
                if low > high:
                    raise ValueError("%s bounds are reversed" % name)
            else:
                low, high = spin.minimum(), spin.maximum()
            if value < low or value > high:
                raise ValueError("%s=%s is outside %s..%s" %
                                 (name, value, low, high))
            checked.append((spin, low, high, value))
        for spin, low, high, value in checked:
            spin.blockSignals(True)
            spin.setRange(low, high)
            spin.setValue(value)
            spin.setEnabled(True)
            spin.blockSignals(False)
        self._settings_dirty = False
        self._settings_loaded = True

    def _load_programs(self, response, *, select_program=False):
        if response.get("type") != "programs":
            raise ValueError("expected programs payload")
        count = response.get("program_count")
        limits = response.get("limits")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10:
            raise ValueError("invalid program count")
        if not hasattr(limits, "get"):
            raise ValueError("program limits are missing")
        required_limits = ("max_programs", "max_stages", "max_cycles", "max_rot_per_min")
        if any(isinstance(limits.get(key), bool) or not isinstance(limits.get(key), int)
               or limits.get(key) < 1 for key in required_limits):
            raise ValueError("program limits are invalid")
        if count > limits["max_programs"] or limits["max_programs"] > 10:
            raise ValueError("program count exceeds device limit")
        wanted = self.program_box.currentData()
        program = response.get("program")
        if not hasattr(program, "get"):
            raise ValueError("program is missing")
        program_id = program.get("program_id")
        if isinstance(program_id, bool) or not isinstance(program_id, int) or not 0 <= program_id < count:
            raise ValueError("program id is invalid")
        if not select_program and program_id != wanted:
            raise ValueError("program response does not match the selection")
        stages = list(program.get("stages", ()))
        if not 1 <= len(stages) <= min(10, limits["max_stages"]):
            raise ValueError("program stage count is invalid")
        fields = ("time_turning", "time_resting", "rot_per_min", "num_cycles")
        for stage in stages:
            if not hasattr(stage, "get"):
                raise ValueError("program stage is invalid")
            maxima = (255, 255, limits["max_rot_per_min"], limits["max_cycles"])
            minima = (1, 0, 1, 1)
            for field, low, high in zip(fields, minima, maxima):
                value = stage.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                    raise ValueError("invalid %s in program stage" % field)
        self.program_box.blockSignals(True)
        self.program_box.clear()
        for number in range(count):
            self.program_box.addItem("Program %d" % (number + 1), number)
        self.program_box.setCurrentIndex(program_id)
        self.program_box.blockSignals(False)
        self._program_count = count
        self._max_programs = limits["max_programs"]
        self.stage_count.setMaximum(min(10, limits["max_stages"]))
        for cells in self.program_cells:
            cells[0].setRange(1, 255)
            cells[1].setRange(0, 255)
            cells[2].setRange(1, limits["max_rot_per_min"])
            cells[3].setRange(1, limits["max_cycles"])
        self.program_table.setEnabled(False)
        self.stage_count.blockSignals(True)
        if not stages:
            raise ValueError("program must contain at least one stage")
        self.stage_count.setValue(len(stages))
        self.stage_count.blockSignals(False)
        for row, cells in enumerate(self.program_cells):
            stage = stages[row] if row < len(stages) else {}
            for field, spin in zip(fields, cells):
                spin.blockSignals(True)
                spin.setValue(int(stage.get(field, 0)))
                spin.blockSignals(False)
                spin.setEnabled(row < len(stages))
        self.program_table.setEnabled(True)
        self.stage_count.setEnabled(True)
        self._update_stage_rows(len(stages))
        self._program_dirty = False
        self._loaded_program_id = program_id

    def closeEvent(self, event) -> None:
        self._timer.stop()
        # An injected service may be application-owned and shared with a future pipeline. Closing
        # this view must not silently tear down that shared connection.
        event.accept()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._timer.start()
        self.refresh()

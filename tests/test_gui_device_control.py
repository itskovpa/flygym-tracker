from dataclasses import dataclass

from flygym_tracker.gui.device_control import DeviceControlWindow
from flygym_tracker.rig_device import CommandState, RigSnapshot, SensorReading


def status(**changes):
    value = dict(boot_id="boot-a", seq=1, device_ms=100, position_valid=True,
                 homed=True, moving=False, phase=0, sensor_raw=False,
                 sensor_debounced=False, sensor_adc_avg=123, sensor_mv=456,
                 owner="LOCAL", fault=None, move_id=0)
    value.update(changes)
    return value


class FakeService:
    def __init__(self, port="COM22"):
        self.port = port
        self.current = None
        self.history = []
        self.commands = []
        self.states = {}
        self.started = 0
        self.closed = 0
        self.last_error = None
        self.local_owns = False
        self.responses = {}

    @property
    def owns_control(self):
        return self.local_owns and self.current is not None \
            and self.current.message["owner"] == "USB"

    def start(self):
        self.started += 1

    def close(self):
        self.closed += 1
        self.current = None

    def snapshot(self):
        return self.current

    def sensor_history(self):
        return list(self.history)

    def send_command(self, command, args=None):
        request_id = "request-%d" % (len(self.commands) + 1)
        self.commands.append((command, args or {}))
        self.states[request_id] = CommandState(request_id, command, "pending")
        return request_id

    def command_state(self, request_id):
        return self.states.get(request_id)

    def response(self, request_id):
        return self.responses.get(request_id)


def snap(**changes):
    return RigSnapshot(status(**changes), 10.0)


def programs_response(program_id=0, count=1, stages=None):
    return {"type": "programs", "program_count": count,
            "limits": {"max_programs": 10, "max_stages": 10,
                       "max_cycles": 100, "max_rot_per_min": 100},
            "program": {"program_id": program_id, "stages": stages or [
                {"time_turning": 1, "time_resting": 0,
                 "rot_per_min": 1, "num_cycles": 1}]}}


def test_connect_does_not_need_or_touch_a_camera(qapp):
    made = []
    window = DeviceControlWindow(service_factory=lambda port: made.append(FakeService(port)) or made[-1])
    window.port_edit.setText("COM22")
    window.connect_button.click()
    assert made[0].port == "COM22"
    assert made[0].started == 1
    assert "waiting for fresh valid status" in window.connection_label.text().lower()


def test_fresh_status_reports_identity_validity_phase_and_all_sensor_forms(qapp):
    service = FakeService()
    service.current = snap(build_id="2026.09", homed=True, position_valid=False, phase=1,
                           sensor_raw=True, sensor_debounced=False, sensor_adc_avg=712,
                           sensor_mv=2870)
    service.history = [SensorReading(1.0, 100, True, False, 712, 2870)] * 400
    window = DeviceControlWindow(service)
    window.refresh()
    assert window.status_labels["boot_id"].text() == "boot-a"
    assert window.status_labels["build"].text() == "2026.09"
    assert window.status_labels["position"].text() == "unknown"
    assert window.status_labels["phase"].text() == "1"
    assert "ADC 712" in window.sensor_detail.text()
    assert "2870 mV" in window.sensor_detail.text()
    assert "raw True" in window.sensor_detail.text()
    assert len(window.sensor_plot._points) == 300


def test_monitoring_local_device_only_offers_explicit_take_control(qapp):
    service = FakeService()
    service.current = snap(owner="LOCAL")
    window = DeviceControlWindow(service)
    window.refresh()
    assert window.take_control_button.isEnabled()
    assert not window.home_button.isEnabled()
    assert not window.start_program_button.isEnabled()
    window.home_button.click()
    assert service.commands == []
    window.read_program_button.click()
    assert service.commands == [("GET_PROGRAMS", {"program_id": 0})]
    request_id = window._pending_id
    service.states[request_id] = CommandState(request_id, "GET_PROGRAMS", "completed")
    service.responses[request_id] = programs_response()
    window.refresh()
    window.take_control_button.click()
    assert service.commands[-1] == ("TAKE_CONTROL", {})


def test_usb_owner_can_home_half_turn_and_start_selected_existing_program(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    service.local_owns = True
    window = DeviceControlWindow(service)
    window.refresh()
    window.half_turn_button.click()
    assert service.commands == [("HALF_TURN", {})]
    # A pending mutation blocks every incompatible mutation.
    window.start_program_button.click()
    assert len(service.commands) == 1
    request_id = window._pending_id
    service.states[request_id] = CommandState(request_id, "HALF_TURN", "completed")
    window.refresh()
    window._load_programs(programs_response(0, 10))
    window.program_box.setCurrentIndex(6)
    window.start_program_button.click()
    assert service.commands[-1] == ("START_PROGRAM", {"program_id": 6})


def test_acceptance_is_shown_as_acceptance_and_keeps_mutations_blocked(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    service.local_owns = True
    window = DeviceControlWindow(service)
    window.home_button.click()
    request_id = window._pending_id
    service.states[request_id] = CommandState(request_id, "HOME", "accepted")
    window.refresh()
    assert window.command_label.text() == "HOME accepted"
    assert not window.half_turn_button.isEnabled()
    assert window._pending_id == request_id


def test_stale_status_disables_mutation_but_keeps_bounded_history_visible(qapp):
    service = FakeService()
    service.history = [SensorReading(float(i), i, False, False, i, i * 2) for i in range(350)]
    window = DeviceControlWindow(service)
    window.refresh()
    assert not window.take_control_button.isEnabled()
    assert not window.abort_motion_button.isEnabled()
    assert len(window.sensor_plot._points) == 300


def test_abort_motion_is_distinct_from_cancel_program(qapp):
    service = FakeService()
    service.current = snap(owner="USB", moving=True)
    service.local_owns = True
    window = DeviceControlWindow(service)
    window.refresh()
    assert window.abort_motion_button.isEnabled()
    assert window.cancel_program_button.isEnabled()
    window.abort_motion_button.click()
    assert service.commands == [("ABORT_MOTION", {})]


def test_pause_and_cancel_reach_service_during_motion(qapp):
    for button_name, expected in (("pause_program_button", "PAUSE_PROGRAM"),
                                  ("cancel_program_button", "CANCEL_PROGRAM")):
        service = FakeService()
        service.current = snap(owner="USB", moving=True)
        service.local_owns = True
        window = DeviceControlWindow(service)
        getattr(window, button_name).click()
        assert service.commands == [(expected, {})]
        window.close()


def test_abort_remains_available_when_a_move_command_is_pending(qapp):
    service = FakeService()
    service.current = snap(owner="USB", moving=False)
    service.local_owns = True
    window = DeviceControlWindow(service)
    window.home_button.click()
    service.current = snap(owner="USB", moving=True)
    window.refresh()
    assert window.abort_motion_button.isEnabled()
    window.abort_motion_button.click()
    assert service.commands[-1] == ("ABORT_MOTION", {})


def test_disconnect_allows_a_different_port(qapp):
    made = []
    window = DeviceControlWindow(service_factory=lambda port: made.append(FakeService(port)) or made[-1])
    window.port_edit.setText("COM22")
    window.connect_device()
    window.disconnect_device()
    window.port_edit.setText("COM24")
    window.connect_device()
    assert [service.port for service in made] == ["COM22", "COM24"]


def test_settings_are_loaded_from_device_and_only_saved_explicitly(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    service.local_owns = True
    window = DeviceControlWindow(service)
    window.refresh_settings_button.click()
    request_id = window._pending_id
    service.responses[request_id] = {"type": "settings", "values": {
        "ACCEL": 6000, "SPEED": 2000, "SPEED_HOMING_SEEK": 300,
        "ACCEL_HOMING": 400, "SPEED_HOMING": 500, "HOME_ADJUST": -2,
        "SENSOR_POLL_INTERVAL": 10, "PULL_OFF_DIST": 100}}
    service.states[request_id] = CommandState(request_id, "GET_SETTINGS", "completed")
    window.refresh()
    assert window.setting_spins["ACCEL"].value() == 6000
    assert not window.save_settings_button.isEnabled()
    window.setting_spins["ACCEL"].setValue(7000)
    assert window.save_settings_button.isEnabled()


def test_external_usb_owner_does_not_grant_this_app_a_control_lease(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    window = DeviceControlWindow(service)
    window.refresh()
    assert not window.home_button.isEnabled()
    assert not window.release_control_button.isEnabled()
    assert not window.take_control_button.isEnabled()
    window.take_control_button.click()
    assert service.commands == []


def test_program_read_is_paginated_and_edits_wait_for_explicit_save(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    service.local_owns = True
    window = DeviceControlWindow(service)
    window._load_programs(programs_response(0, 3))
    window.program_box.setCurrentIndex(2)
    assert not window.program_table.isEnabled()
    window.read_program_button.click()
    assert service.commands[-1] == ("GET_PROGRAMS", {"program_id": 2})
    request_id = window._pending_id
    service.responses[request_id] = programs_response(2, 3, [
        {"time_turning": 2, "time_resting": 3, "rot_per_min": 4, "num_cycles": 5}])
    service.states[request_id] = CommandState(request_id, "GET_PROGRAMS", "completed")
    window.refresh()
    assert window.program_table.isEnabled()
    assert window.program_cells[0][0].value() == 2
    assert not window.save_program_button.isEnabled()
    window.program_cells[0][0].setValue(7)
    assert window.save_program_button.isEnabled()
    assert len(service.commands) == 1


def test_created_program_response_expands_list_and_selects_new_program(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    service.local_owns = True
    window = DeviceControlWindow(service)
    window.create_program_button.click()
    request_id = window._pending_id
    service.responses[request_id] = programs_response(2, 3)
    service.states[request_id] = CommandState(request_id, "CREATE_PROGRAM", "completed")
    window.refresh()
    assert window.program_box.count() == 3
    assert window.program_box.currentData() == 2
    assert window._loaded_program_id == 2


def test_deleting_program_invalidates_editor_and_reads_safe_remaining_id(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    service.local_owns = True
    window = DeviceControlWindow(service)
    window._load_programs(programs_response(2, 3), select_program=True)
    window.refresh()
    window.delete_program_button.click()
    request_id = window._pending_id
    assert service.commands[-1] == ("DELETE_PROGRAM", {"program_id": 2})
    service.states[request_id] = CommandState(request_id, "DELETE_PROGRAM", "completed")
    window.refresh()
    assert window._loaded_program_id is None
    assert service.commands[-1] == ("GET_PROGRAMS", {"program_id": 1})


def test_stage_units_minima_and_cancel_help_match_firmware(qapp):
    window = DeviceControlWindow(FakeService())
    assert window.program_table.horizontalHeaderItem(2).text() == "Half-turns/min"
    assert [cell.minimum() for cell in window.program_cells[0]] == [1, 0, 1, 1]
    assert "current half-turn finishes" in window.cancel_program_button.toolTip()
    assert "Abort motion" in window.cancel_program_button.toolTip()


def test_response_and_completion_are_both_required_in_either_order(qapp):
    service = FakeService()
    service.current = snap(owner="LOCAL")
    window = DeviceControlWindow(service)
    window.refresh_settings_button.click()
    request_id = window._pending_id
    service.states[request_id] = CommandState(request_id, "GET_SETTINGS", "completed")
    window.refresh()
    assert window._pending_id == request_id
    assert not window._settings_loaded
    values = {name: spin.minimum() for name, spin in window.setting_spins.items()}
    service.responses[request_id] = {"type": "settings", "values": values}
    window.refresh()
    assert window._pending_id is None


def test_rejected_batch_stops_before_save(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    service.local_owns = True
    window = DeviceControlWindow(service)
    window._settings_loaded = True
    window._settings_dirty = True
    window.save_settings()
    assert service.commands[0][0] == "SET_SETTING"
    request_id = window._pending_id
    service.states[request_id] = CommandState(request_id, "SET_SETTING", "rejected", "bad")
    window.refresh()
    assert len(service.commands) == 1
    assert not window._command_queue


def test_selecting_another_program_invalidates_old_table(qapp):
    service = FakeService()
    service.current = snap(owner="USB")
    service.local_owns = True
    window = DeviceControlWindow(service)
    window._loaded_program_id = 1
    window._program_dirty = True
    window.program_table.setEnabled(True)
    window.program_box.setCurrentIndex(1)
    assert window._loaded_program_id is None
    window.save_program()
    assert service.commands == []


def test_malformed_settings_do_not_partially_populate_editor(qapp):
    service = FakeService()
    service.current = snap(owner="LOCAL")
    window = DeviceControlWindow(service)
    window.refresh_settings_button.click()
    request_id = window._pending_id
    values = {name: spin.minimum() for name, spin in window.setting_spins.items()}
    values["PULL_OFF_DIST"] = 999999
    service.responses[request_id] = {"type": "settings", "values": values}
    service.states[request_id] = CommandState(request_id, "GET_SETTINGS", "completed")
    window.refresh()
    assert not window._settings_loaded
    assert all(not spin.isEnabled() for spin in window.setting_spins.values())
    assert "Invalid device response" in window.command_label.text()


def test_closing_view_does_not_close_application_owned_service(qapp):
    service = FakeService()
    window = DeviceControlWindow(service)
    window.close()
    assert service.closed == 0

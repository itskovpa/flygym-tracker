import json
import threading

import pytest

from flygym_tracker.rig_device import MAX_INBOUND_LINE_BYTES, PREFIX, ProtocolError, RigDeviceService, parse_line


def wire(message):
    return PREFIX + json.dumps(message).encode() + b"\n"


def status(**changes):
    value = dict(v=1, type="status", boot_id="boot-a", seq=1, device_ms=100,
                 position_valid=True, homed=True, moving=False, phase=0,
                 sensor_raw=False, sensor_debounced=False, sensor_adc_avg=123,
                 sensor_mv=456, owner="LOCAL", fault=None, move_id=0)
    value.update(changes)
    return value


def acquire(service):
    request_id = service.send_command("TAKE_CONTROL")
    service.feed_line(wire({"v": 1, "type": "ack", "id": request_id, "accepted": True}))
    service.feed_line(wire({"v": 1, "type": "event", "id": request_id,
                            "outcome": "completed"}))


class FakeTransport:
    def __init__(self):
        self.lines = []
        self.writes = []
        self.closed = threading.Event()

    def read(self, _limit):
        if self.lines:
            return self.lines.pop(0)
        self.closed.wait(0.01)
        return b""

    def write(self, value):
        self.writes.append(value)
        return len(value)

    def close(self):
        self.closed.set()


def test_parser_requires_prefix_version_type_and_bounds():
    assert parse_line(wire(status()).rstrip())["boot_id"] == "boot-a"
    for bad in (b"console text", PREFIX + b"[]", PREFIX + b'{"v":2,"type":"status"}',
                PREFIX + b"x" * MAX_INBOUND_LINE_BYTES):
        with pytest.raises(ProtocolError):
            parse_line(bad)


def test_status_validates_stales_and_keeps_sensor_history():
    now = [10.0]
    service = RigDeviceService("COM22", stale_after=2, clock=lambda: now[0])
    service.feed_line(wire(status()), now[0])
    assert service.connected
    assert service.snapshot().message["phase"] == 0
    assert service.sensor_history()[0].millivolts == 456
    now[0] = 12.01
    assert service.snapshot() is None
    assert not service.connected
    with pytest.raises(ProtocolError, match="invalid status"):
        service.feed_line(wire(status(owner="BAD")), now[0])


def test_duplicate_sequence_ignored_and_boot_or_gap_accepts_new_snapshot():
    now = [1.0]
    service = RigDeviceService("COM22", clock=lambda: now[0])
    service.feed_line(wire(status(seq=4, sensor_mv=400)), 1)
    service.feed_line(wire(status(seq=4, sensor_mv=999)), 2)
    assert service.snapshot().message["sensor_mv"] == 400
    service.feed_line(wire(status(seq=8, sensor_mv=800)), 3)
    assert service.snapshot().message["seq"] == 8
    service.feed_line(wire(status(boot_id="boot-b", seq=0, sensor_mv=200)), 4)
    assert service.snapshot().message["boot_id"] == "boot-b"


def test_command_acceptance_is_distinct_from_completion():
    transport = FakeTransport()
    service = RigDeviceService("COM22", transport_factory=lambda: transport)
    service._transport = transport
    service.feed_line(wire(status(owner="USB")), service._clock())
    acquire(service)
    request_id = service.send_command("home")
    assert service.command_state(request_id).state == "pending"
    service.feed_line(wire({"v": 1, "type": "ack", "id": request_id, "accepted": True}), 1)
    assert service.command_state(request_id).state == "accepted"
    service.feed_line(wire({"v": 1, "type": "event", "id": request_id,
                            "outcome": "completed"}), 2)
    assert service.command_state(request_id).state == "completed"
    service.feed_line(wire({"v": 1, "type": "ack", "id": request_id, "accepted": False}), 3)
    assert service.command_state(request_id).state == "completed"


def test_service_thread_closes_cleanly_after_valid_status():
    transport = FakeTransport()
    transport.lines.append(wire(status()))
    service = RigDeviceService("COM22", transport_factory=lambda: transport)
    service.start()
    for _ in range(100):
        if service.connected:
            break
        threading.Event().wait(0.005)
    assert service.connected
    service.close()
    assert transport.closed.is_set()
    assert service._thread is None


def test_partial_and_oversized_lines_do_not_break_stream():
    transport = FakeTransport()
    good = wire(status())
    second = wire(status(seq=2))
    transport.lines.extend([good[:7], good[7:], b"x" * (MAX_INBOUND_LINE_BYTES + 2) + b"\n", second])
    service = RigDeviceService("COM22", transport_factory=lambda: transport)
    service.start()
    for _ in range(100):
        if len(service.sensor_history()) == 2:
            break
        threading.Event().wait(0.005)
    assert len(service.sensor_history()) == 2
    service.close()


def test_phase_bool_rejected_and_boot_invalidates_pending_command():
    service = RigDeviceService("COM22")
    with pytest.raises(ProtocolError):
        service.feed_line(wire(status(phase=True)), service._clock())
    service._transport = FakeTransport()
    service.feed_line(wire(status(owner="USB")), service._clock())
    acquire(service)
    request_id = service.send_command("HOME")
    service.feed_line(wire(status(boot_id="new", seq=0, owner="LOCAL")), service._clock())
    assert service.command_state(request_id).state == "uncertain"


def test_monitor_cannot_mutate_leftover_usb_owner_and_stale_revokes_local_lease():
    now = [1.0]
    service = RigDeviceService("COM22", clock=lambda: now[0])
    service._transport = FakeTransport()
    service.feed_line(wire(status(owner="USB")), now[0])
    with pytest.raises(PermissionError):
        service.send_command("HOME")
    acquire(service)
    assert service.owns_control
    now[0] += 3
    assert not service.owns_control


def test_payload_response_is_deeply_immutable_and_command_size_is_bounded():
    service = RigDeviceService("COM22")
    service._transport = FakeTransport()
    request_id = service.send_command("GET_SETTINGS")
    service.feed_line(wire({"v": 1, "type": "settings", "id": request_id,
                            "staged": False, "values": {"speed": 10}}))
    response = service.response(request_id)
    assert response["values"]["speed"] == 10
    with pytest.raises(TypeError):
        response["values"]["speed"] = 20
    with pytest.raises(ValueError, match="line limit"):
        service.send_command("GET_STATUS", {"padding": "x" * 600})


def test_status_ranges_and_local_owner_revoke_control():
    service = RigDeviceService("COM22")
    service._transport = FakeTransport()
    for bad in (status(phase=0.0), status(sensor_adc_avg=4096), status(sensor_mv=6001)):
        with pytest.raises(ProtocolError):
            service.feed_line(wire(bad), service._clock())
    service.feed_line(wire(status(owner="USB")), service._clock())
    acquire(service)
    assert service.owns_control
    service.feed_line(wire(status(seq=2, owner="LOCAL")), service._clock())
    assert not service.owns_control


def test_pong_completes_known_ping_and_unsolicited_response_is_ignored():
    service = RigDeviceService("COM22")
    service._transport = FakeTransport()
    request_id = service.send_command("PING")
    service.feed_line(wire({"v": 1, "type": "pong", "id": request_id, "device_ms": 5}))
    assert service.command_state(request_id).state == "completed"
    service.feed_line(wire({"v": 1, "type": "settings", "id": "unknown",
                            "staged": False, "values": {}}))
    assert service.response("unknown") is None


def test_late_terminal_can_resolve_response_timeout():
    now = [1.0]
    service = RigDeviceService("COM22", clock=lambda: now[0])
    service._transport = FakeTransport()
    request_id = service.send_command("GET_STATUS")
    now[0] = 5.0
    service._expire_commands()
    assert service.command_state(request_id).state == "uncertain"
    service.feed_line(wire({"v": 1, "type": "event", "id": request_id,
                            "outcome": "completed"}), now[0])
    assert service.command_state(request_id).state == "completed"

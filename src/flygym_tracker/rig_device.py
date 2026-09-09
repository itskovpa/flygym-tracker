"""PC-side transport for the versioned FlyGym USB protocol.

This module has no Qt, camera, or tracker dependencies.  It opens no port until
``start`` is called and accepts a transport factory so tests need no hardware.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from queue import Empty, Queue
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Deque, Mapping, Optional


PREFIX = b"@FG "
MAX_LINE_BYTES = 511
MAX_INBOUND_LINE_BYTES = 4096
PROTOCOL_VERSION = 1
_STATUS_FIELDS = {
    "boot_id", "seq", "device_ms", "position_valid", "homed", "moving", "phase",
    "sensor_raw", "sensor_debounced", "sensor_adc_avg", "sensor_mv", "owner", "fault",
    "move_id",
}


class ProtocolError(ValueError):
    pass


def parse_line(line: bytes) -> dict:
    """Parse one bounded protocol line and reject non-protocol console output."""
    if len(line) > MAX_INBOUND_LINE_BYTES:
        raise ProtocolError("line exceeds 4096 bytes")
    if not line.startswith(PREFIX):
        raise ProtocolError("missing @FG prefix")
    try:
        message = json.loads(line[len(PREFIX):].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON") from exc
    if not isinstance(message, dict):
        raise ProtocolError("message must be an object")
    if isinstance(message.get("v"), bool) or message.get("v") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported protocol version")
    if message.get("type") not in {"status", "ack", "event", "pong", "settings", "programs"}:
        raise ProtocolError("unsupported message type")
    return message


def _valid_status(message: dict) -> bool:
    if not _STATUS_FIELDS.issubset(message):
        return False
    if not isinstance(message["boot_id"], str) or not message["boot_id"]:
        return False
    ints = ("seq", "device_ms", "sensor_adc_avg", "sensor_mv", "move_id")
    if any(isinstance(message[k], bool) or not isinstance(message[k], int) for k in ints):
        return False
    if any(not isinstance(message[k], bool) for k in
           ("position_valid", "homed", "moving", "sensor_raw", "sensor_debounced")):
        return False
    if message["phase"] is not None and (type(message["phase"]) is not int
                                          or message["phase"] not in (0, 1)):
        return False
    if any(message[k] < 0 for k in ints):
        return False
    if message["sensor_adc_avg"] > 4095 or message["sensor_mv"] > 6000:
        return False
    return message["owner"] in ("LOCAL", "USB") \
        and (message["fault"] is None or isinstance(message["fault"], str))


@dataclass(frozen=True)
class RigSnapshot:
    message: Mapping
    received_monotonic: float


@dataclass(frozen=True)
class SensorReading:
    received_monotonic: float
    device_ms: int
    raw: bool
    debounced: bool
    adc_avg: int
    millivolts: int


@dataclass(frozen=True)
class CommandState:
    request_id: str
    command: str
    state: str                 # pending | accepted | completed | rejected | aborted
    detail: Optional[str] = None


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


class RigDeviceService:
    """Own one serial connection and expose validated status without blocking callers."""

    def __init__(self, port: str, *, baudrate: int = 115200, stale_after: float = 2.0,
                 transport_factory: Optional[Callable[[], object]] = None,
                 clock: Callable[[], float] = time.monotonic, history_size: int = 300) -> None:
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.stale_after = float(stale_after)
        self._clock = clock
        self._transport_factory = transport_factory or self._open_serial
        self._history: Deque[SensorReading] = deque(maxlen=max(1, int(history_size)))
        self._snapshot: Optional[RigSnapshot] = None
        self._commands: dict[str, CommandState] = {}
        self._responses: dict[str, Mapping] = {}
        self._command_order: Deque[str] = deque(maxlen=256)
        self._writes: Queue[tuple[str, bytes]] = Queue(maxsize=64)
        self._command_deadlines: dict[str, float] = {}
        self._owns_control = False
        self._transport = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    def _open_serial(self):
        import serial
        return serial.Serial(self.port, self.baudrate, timeout=0.1, write_timeout=0.5)

    @property
    def owns_control(self) -> bool:
        snap = self.snapshot()
        return bool(self._owns_control and snap is not None and snap.message["owner"] == "USB")

    @property
    def connected(self) -> bool:
        """True only after a fresh, fully validated status message."""
        return self.snapshot() is not None

    def snapshot(self) -> Optional[RigSnapshot]:
        with self._lock:
            value = self._snapshot
        if value is None or self._clock() - value.received_monotonic > self.stale_after:
            self._owns_control = False
            return None
        return value

    def sensor_history(self) -> list[SensorReading]:
        with self._lock:
            return list(self._history)

    def command_state(self, request_id: str) -> Optional[CommandState]:
        with self._lock:
            return self._commands.get(request_id)

    def response(self, request_id: str) -> Optional[Mapping]:
        with self._lock:
            return self._responses.get(request_id)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._writes = Queue(maxsize=64)
        self._thread = threading.Thread(target=self._run, name="rig-device", daemon=True)
        self._thread.start()

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        transport = self._transport
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("rig device thread did not stop")
        self._thread = None
        with self._lock:
            self._snapshot = None
            self._owns_control = False
            self._invalidate_pending("connection closed")
            self._commands.clear()
            self._command_order.clear()
            self._command_deadlines.clear()
            self._responses.clear()

    def send_command(self, command: str, args: Optional[dict] = None) -> str:
        """Send once. An I/O failure is reported; mutation is never blindly replayed."""
        if self._transport is None:
            raise RuntimeError("rig device is not open")
        request_id = uuid.uuid4().hex
        name = str(command).upper()
        read_only = name in {"GET_STATUS", "GET_SETTINGS", "GET_PROGRAMS", "PING", "HELLO"}
        snap = self.snapshot()
        if not read_only and name != "TAKE_CONTROL":
            if not self.owns_control:
                raise PermissionError("USB control is not held")
        if name == "TAKE_CONTROL" and (snap is None or snap.message["moving"]):
            raise PermissionError("control requires a fresh idle status")
        message = {"v": 1, "type": "command", "id": request_id,
                   "cmd": name, "args": dict(args or {})}
        wire = PREFIX + json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(wire) - 1 > MAX_LINE_BYTES:
            raise ValueError("command exceeds protocol line limit")
        with self._lock:
            if len(self._command_order) == self._command_order.maxlen:
                evicted = self._command_order[0]
                self._commands.pop(evicted, None)
                self._command_deadlines.pop(evicted, None)
                self._responses.pop(evicted, None)
            self._command_order.append(request_id)
            self._commands[request_id] = CommandState(request_id, name, "pending")
            self._command_deadlines[request_id] = self._clock() + 3.0
        try:
            self._writes.put_nowait((request_id, wire))
        except Exception:
            with self._lock:
                self._commands[request_id] = CommandState(request_id, name, "uncertain", "write not queued")
            raise
        return request_id

    def _run(self) -> None:
        buffer = bytearray()
        discarding = False
        try:
            self._transport = self._transport_factory()
            self._queue_internal("HELLO")
            next_ping = self._clock() + 1.0
            while not self._stop.is_set():
                self._expire_commands()
                if self.owns_control and self._clock() >= next_ping:
                    self._queue_internal("PING")
                    next_ping = self._clock() + 1.0
                try:
                    request_id, wire = self._writes.get_nowait()
                    command = self.command_state(request_id)
                    if command is None or command.state != "pending":
                        continue
                    if command.command not in {"GET_STATUS", "GET_SETTINGS", "GET_PROGRAMS",
                                                "PING", "HELLO", "TAKE_CONTROL"} and not self.owns_control:
                        with self._lock:
                            self._commands[request_id] = CommandState(
                                request_id, command.command, "uncertain", "USB ownership lost before write")
                        continue
                    written = self._transport.write(wire)
                    if written is not None and written != len(wire):
                        raise OSError("short serial write")
                except Empty:
                    pass
                except Exception as exc:
                    with self._lock:
                        current = self._commands.get(request_id)
                        if current:
                            self._commands[request_id] = CommandState(
                                request_id, current.command, "uncertain", "write outcome unknown")
                    raise exc
                chunk = self._transport.read(256)
                for byte in chunk or b"":
                    if byte == 10:
                        if not discarding and buffer:
                            try:
                                self.feed_line(bytes(buffer).rstrip(b"\r"), self._clock())
                            except ProtocolError as exc:
                                self.last_error = str(exc)
                        buffer.clear()
                        discarding = False
                    elif not discarding:
                        buffer.append(byte)
                        if len(buffer) > MAX_INBOUND_LINE_BYTES:
                            buffer.clear()
                            discarding = True
                            self.last_error = "line exceeds 4096 bytes"
        except Exception as exc:
            if not self._stop.is_set():
                self.last_error = str(exc)
        finally:
            transport, self._transport = self._transport, None
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass
            with self._lock:
                self._snapshot = None
                self._owns_control = False
                self._invalidate_pending("connection lost")

    def feed_line(self, line: bytes, received_monotonic: Optional[float] = None) -> None:
        """Process one line; public for deterministic transport-level tests."""
        message = parse_line(line.rstrip(b"\r\n"))
        received = self._clock() if received_monotonic is None else float(received_monotonic)
        kind = message["type"]
        if kind == "status":
            if not _valid_status(message):
                raise ProtocolError("invalid status fields")
            with self._lock:
                previous = self._snapshot
                if previous is not None:
                    old = previous.message
                    if old["boot_id"] == message["boot_id"] and message["seq"] <= old["seq"]:
                        return
                    if old["boot_id"] != message["boot_id"]:
                        self._snapshot = None
                        self._owns_control = False
                        for key, command in list(self._commands.items()):
                            if command.state in ("pending", "accepted"):
                                self._commands[key] = CommandState(
                                    key, command.command, "uncertain", "device rebooted")
                self._snapshot = RigSnapshot(_freeze(message), received)
                if message["owner"] == "LOCAL":
                    self._owns_control = False
                self._history.append(SensorReading(
                    received, message["device_ms"], message["sensor_raw"],
                    message["sensor_debounced"], message["sensor_adc_avg"], message["sensor_mv"]))
            return
        if kind in ("ack", "event"):
            self._update_command(message)
        elif kind in ("settings", "programs", "pong"):
            request_id = message.get("id")
            if isinstance(request_id, str) and self.command_state(request_id) is not None:
                with self._lock:
                    self._responses[request_id] = _freeze(message)
                if kind == "pong":
                    self._finish_ping(request_id)

    def _update_command(self, message: dict) -> None:
        request_id = message.get("id")
        if not isinstance(request_id, str):
            return
        with self._lock:
            current = self._commands.get(request_id)
            if current is None:
                return
            if current.state in ("completed", "rejected", "aborted"):
                return
            if current.state == "uncertain" and current.detail != "response timeout":
                return
            if message["type"] == "ack":
                state = "accepted" if message.get("accepted") is True else "rejected"
            else:
                outcome = message.get("outcome")
                if outcome not in ("completed", "rejected", "aborted"):
                    return
                state = outcome
            self._commands[request_id] = CommandState(
                request_id, current.command, state,
                message.get("detail") if isinstance(message.get("detail"), str) else None)
            if current.command == "TAKE_CONTROL" and state == "completed":
                self._owns_control = True
            elif current.command == "RELEASE_CONTROL" and state in ("completed", "accepted"):
                self._owns_control = False
            if state == "accepted":
                self._command_deadlines[request_id] = self._clock() + 120.0
            else:
                self._command_deadlines.pop(request_id, None)

    def _finish_ping(self, request_id: str) -> None:
        with self._lock:
            current = self._commands.get(request_id)
            if current is not None and current.command == "PING" and current.state in ("pending", "accepted"):
                self._commands[request_id] = CommandState(request_id, "PING", "completed")
                self._command_deadlines.pop(request_id, None)

    def _queue_internal(self, command: str) -> None:
        try:
            self.send_command(command)
        except Exception:
            pass

    def _expire_commands(self) -> None:
        now = self._clock()
        with self._lock:
            for key, deadline in list(self._command_deadlines.items()):
                command = self._commands.get(key)
                if deadline <= now and command is not None and command.state in ("pending", "accepted"):
                    self._commands[key] = CommandState(key, command.command, "uncertain", "response timeout")
                    self._command_deadlines.pop(key, None)

    def _invalidate_pending(self, detail: str) -> None:
        for key, command in list(self._commands.items()):
            if command.state in ("pending", "accepted"):
                self._commands[key] = CommandState(key, command.command, "uncertain", detail)
            self._command_deadlines.pop(key, None)

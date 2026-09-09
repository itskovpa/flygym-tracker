# FlyGym USB protocol v1

Every record is one newline-terminated UTF-8 line: `@FG ` followed by a JSON object. Host-to-device lines are at most 511 bytes before the newline, matching the firmware receive buffer. Device-to-host lines are at most 4096 bytes so full status/config payloads fit. Console prose without the prefix is not protocol data.

Host command:

```json
{"v":1,"type":"command","id":"unique-string","cmd":"UPPERCASE_COMMAND","args":{}}
```

The device answers with `ack` carrying the same `id` and `accepted: true|false`. Acceptance is not completion. A later `event` with the same `id` and `outcome: completed|rejected|aborted` reports the result. The host does not retry mutating commands automatically after write failure or reconnect.

Command vocabulary:

- `HELLO {}`, `PING {}`, `GET_STATUS {}`, `GET_SETTINGS {}`, and `GET_PROGRAMS {}` are read-only.
- `TAKE_CONTROL {}` and `RELEASE_CONTROL {}` change the owner; success is confirmed by a later status whose `owner` reflects the change.
- `HOME {}`, `HALF_TURN {}`, and `ABORT_MOTION {}` control an individual move. Completion requires an outcome event, not only an ACK.
- `START_PROGRAM {"program_id": integer}`, `PAUSE_PROGRAM {}`, `RESUME_PROGRAM {}`, and `CANCEL_PROGRAM {}` control a program. Cancel is distinct from aborting the current move.
- `SET_SETTING {"name": string, "value": integer}`, `SAVE_SETTINGS {}`, and `DISCARD_SETTINGS {}` edit device configuration. Applied values must be read back from the device.
- `CREATE_PROGRAM {}`, `DELETE_PROGRAM {"program_id": integer}`, `SET_PROGRAM_STAGE {"program_id": integer, "stage_id": integer, "time_turning": integer, "time_resting": integer, "rot_per_min": integer, "num_cycles": integer}`, `SET_PROGRAM_STAGE_COUNT {"program_id": integer, "count": integer}`, and `SAVE_PROGRAMS {}` edit programs.

`settings` responses are `{v:1,type:"settings",id,staged:boolean,values:{...},bounds:{SETTING:[min,max]}}`. Programs are paginated by program: request `GET_PROGRAMS {"program_id": integer}` and receive `{v:1,type:"programs",id,program_count,limits:{max_programs,max_stages,max_cycles,max_rot_per_min},program:{program_id,stages:[{time_turning,time_resting,rot_per_min,num_cycles}]}}`. `CREATE_PROGRAM {}` returns the same `programs` payload for the new program between its accepted ACK and completed terminal event. `DELETE_PROGRAM` returns a payload for the program now occupying the deleted index, or the new last program when the old last was deleted; deleting the sole program resets and returns program 0, so `program_count` is always at least one. The service stores these bounded responses by request ID through `response(id)`; callers must not infer success without the terminal command event.

Program stage ranges are: `time_turning` 1..255 minutes, `time_resting` 0..255 minutes, `rot_per_min` 1..`max_rot_per_min`, and `num_cycles` 1..`max_cycles`. `CANCEL_PROGRAM` cancels future scheduling and allows an already active half-turn to finish; use `ABORT_MOTION` separately to stop that move and invalidate position.

Status is emitted every 250 ms. A USB control lease lasts 3000 ms and is refreshed only by `PING` and a successful `TAKE_CONTROL`. On expiry, homing aborts and invalidates position; an accepted half-turn may finish, after which firmware releases USB ownership, prevents new moves, and pauses program scheduling.

A `status` requires: `v`, `type`, `boot_id`, monotonic `seq`, `device_ms`, `position_valid`, `homed`, `moving`, `phase` (`0`, `1`, or `null`), `sensor_raw`, `sensor_debounced`, `sensor_adc_avg`, `sensor_mv`, `owner` (`LOCAL` or `USB`), `fault` (string or null), and `move_id`. Status must be periodic. Opening a port is not connected; the host reports connected only after a fresh, valid status.

Device/host clock alignment is intentionally outside this milestone. Every parsed message receives a host monotonic receipt timestamp; no receipt-time offset formula is treated as frame synchronization.

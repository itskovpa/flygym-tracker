"""Manual ROI calibration wizard (DESIGN.md §5.5, PRIMARY calibration path).

THIN interactive driver only: it collects 16 vial boxes + present/absent marks from the
user (optionally pre-seeded from `calibration.detect_calibration`) and hands them to the
pure, unit-tested `calibration.build_calibration_from_boxes`, which does all the real work
(illuminated sub-masks, central-band exclusion, bundle assembly). Keeping the logic out of
here is deliberate so the CV behaviour stays testable headlessly.

Typical use::

    from flygym_tracker import calibration as C
    from flygym_tracker.calibrate_wizard import run_wizard

    seed, _ = C.detect_seed_boxes(frame, "A")          # optional accelerator
    calib, mask, overlay = run_wizard(frame, "A", seed_boxes=seed)
    C.save_calibration(calib, mask, "calib", overlay=overlay)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from flygym_tracker import calibration as C
from flygym_tracker.calibration import Box, CalibParams

N_SLOTS = 16
_HELP = (
    "FlyGym calibration wizard\n"
    "  For each of the 16 slots (row-major: 1-8 upper, 9-16 lower):\n"
    "    - drag a rectangle over the lit tube body, then ENTER/SPACE to accept\n"
    "    - press 'a' to mark the slot ABSENT (empty / missing tube)\n"
    "    - press 's' to keep the auto-seeded box as-is (if seeded)\n"
    "    - press 'u' to undo the previous slot\n"
    "    - press 'q'/ESC to finish early (remaining seeded slots are kept)\n"
)


def run_wizard(
    frame_gray: np.ndarray,
    face: str = "A",
    seed_boxes: Optional[Sequence[Box]] = None,
    seed_present: Optional[Sequence[bool]] = None,
    params: Optional[CalibParams] = None,
    window: str = "FlyGym calibration",
):
    """Interactively collect vial boxes for one face, then build the bundle.

    Args:
        frame_gray: HxW grayscale face frame.
        face: face name ("A"/"B").
        seed_boxes: optional pre-filled boxes (e.g. from `detect_calibration`) so the user
            only nudges instead of drawing from scratch.
        seed_present: optional present flags matching `seed_boxes`.
        params: `CalibParams` forwarded to the builder.
        window: OpenCV window title.

    Returns:
        (calibration, illum_mask_uint8, overlay_bgr) -- identical shape to
        `detect_calibration`.
    """
    import cv2  # local import: interactive-only dependency, keeps headless import light

    gray = C._as_gray(frame_gray)
    seed_boxes = list(seed_boxes) if seed_boxes is not None else None
    seed_present = list(seed_present) if seed_present is not None else None

    boxes: List[Box] = []
    flags: List[Optional[bool]] = []
    disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    print(_HELP)

    i = 0
    while i < N_SLOTS:
        seed = seed_boxes[i] if (seed_boxes and i < len(seed_boxes)) else None
        preview = disp.copy()
        for b, f in zip(boxes, flags):
            c = (0, 0, 255) if f is False else (0, 200, 0)
            cv2.rectangle(preview, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), c, 2)
        if seed is not None:
            cv2.rectangle(preview, (seed[0], seed[1]),
                          (seed[0] + seed[2], seed[1] + seed[3]), (0, 200, 200), 1)
        cv2.putText(preview, "slot %d/%d  (a=absent s=seed u=undo q=quit)" % (i + 1, N_SLOTS),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        cv2.imshow(window, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("a"), ord("A")):  # absent
            boxes.append(seed if seed is not None else (0, 0, 1, 1))
            flags.append(False)
            i += 1
            continue
        if key in (ord("s"), ord("S")) and seed is not None:  # accept seed
            boxes.append(seed)
            flags.append(seed_present[i] if (seed_present and i < len(seed_present)) else None)
            i += 1
            continue
        if key in (ord("u"), ord("U")) and boxes:  # undo
            boxes.pop()
            flags.pop()
            i -= 1
            continue
        if key in (ord("q"), 27):  # quit early: keep remaining seeds
            while i < N_SLOTS and seed_boxes and i < len(seed_boxes):
                boxes.append(seed_boxes[i])
                flags.append(seed_present[i] if (seed_present and i < len(seed_present)) else None)
                i += 1
            break
        if key in (ord("d"), ord("D"), 13, 32):  # draw / accept via selectROI
            r = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
            if r and r[2] > 0 and r[3] > 0:
                boxes.append((int(r[0]), int(r[1]), int(r[2]), int(r[3])))
                flags.append(None)
                i += 1
            continue

    cv2.destroyWindow(window)
    return C.build_calibration_from_boxes(
        gray, face, boxes, present_flags=flags, params=params,
        notes="manual ROI wizard",
    )

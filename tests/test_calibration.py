"""Tests for flygym_tracker.calibration (DESIGN.md §5.4 / §5.5, §8).

Covers the auto-detect path and the pure `build_calibration_from_boxes` core used by the
manual wizard, on a synthetic lattice with a known missing slot, plus a round-trip and (when
present) an assertion on the real reference frame `docs/frame_full.png`.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from flygym_tracker.calibration import (
    CalibParams,
    build_calibration_from_boxes,
    detect_calibration,
    load_calibration,
    save_calibration,
    boxes_from_calibration,
)
from flygym_tracker.types import Calibration, FaceCalibration, VialROI

# --------------------------------------------------------------------------------------
# Synthetic lattice (mirrors the real rig so one code path serves both):
#   dark background, 2 rows x 8 bright vertical bars (4+4 split by a central gap), a
#   saturated horizontal central LED band, and bright mouth rims on present tubes.
#   A "missing tube" keeps its back-lit body (the diffuser glows through an empty slot, as
#   on the real rig) but loses its mouth rim -- so the empty is flagged by the rim cue, not
#   by a dark body, and the lit lattice geometry stays intact regardless of which slot is
#   missing.
# --------------------------------------------------------------------------------------
SYN_H, SYN_W = 520, 680
SYN_BG, SYN_BODY, SYN_RIM, SYN_LED = 12, 175, 240, 255
SYN_UPPER = (110, 240)
SYN_LOWER = (350, 480)
SYN_CENTRAL = (270, 320)          # LED band (lies inside the excluded central region)
SYN_GROUPS = ((70, 330), (350, 610))
SYN_PITCH = 65


def _syn_centers():
    centers = []
    for gx0, _ in SYN_GROUPS:
        for i in range(4):
            centers.append(int(gx0 + SYN_PITCH * (i + 0.5)))
    return centers


def make_synthetic(blank_id: int = 6):
    """Return (image, expected_empty_ids). `blank_id` in 1..16 is the missing tube.

    The missing tube keeps a lit body but has no mouth rim (see module note)."""
    img = np.full((SYN_H, SYN_W), SYN_BG, np.uint8)
    centers = _syn_centers()
    for band in (SYN_UPPER, SYN_LOWER):                 # bright tube bodies (always lit)
        for gx0, gx1 in SYN_GROUPS:
            img[band[0]:band[1], gx0:gx1] = SYN_BODY
    img[SYN_CENTRAL[0]:SYN_CENTRAL[1], :] = SYN_LED     # saturated central band
    for ids, band, mrow in ((range(1, 9), SYN_UPPER, (70, 84)),
                            (range(9, 17), SYN_LOWER, (500, 514))):
        for j, vid in enumerate(ids):
            cx = centers[j]
            if vid != blank_id:                         # present tube: bright mouth rim
                img[mrow[0]:mrow[1], cx - 16:cx + 16] = SYN_RIM
            # missing tube: body stays lit, no rim drawn
    return img, [blank_id]


# --------------------------------------------------------------------------------------
# Auto-detect on the synthetic lattice
# --------------------------------------------------------------------------------------
def test_detect_returns_16_rois_row_major():
    img, _ = make_synthetic(blank_id=6)
    calib, mask, overlay = detect_calibration(img, face="A")

    assert isinstance(calib, Calibration)
    assert set(calib.faces) == {"A"}
    vials = calib.faces["A"].vials
    assert len(vials) == 16

    # ids 1..16 row-major; rows 0 (upper) then 1 (lower); cols 0..7 each row.
    assert [v.id for v in vials] == list(range(1, 17))
    assert [v.row for v in vials] == [0] * 8 + [1] * 8
    assert [v.col for v in vials] == list(range(8)) * 2

    # boxes are ordered left->right within each row and don't straddle the central gap.
    upper_x = [v.x for v in vials[:8]]
    assert upper_x == sorted(upper_x)
    # left group (cols 0-3) entirely left of right group (cols 4-7)
    assert max(v.x + v.w for v in vials[:4]) <= min(v.x for v in vials[4:8])

    assert mask.dtype == np.uint8 and mask.shape == img.shape
    assert overlay.shape == (SYN_H, SYN_W, 3)


def test_detect_flags_missing_tube_only():
    for blank in (6, 3, 12):
        img, expected_empty = make_synthetic(blank_id=blank)
        calib, _, _ = detect_calibration(img, face="A")
        vials = calib.faces["A"].vials
        empty = [v.id for v in vials if not v.present]
        present = [v.id for v in vials if v.present]
        assert empty == expected_empty, "blank_id=%d -> empty=%s" % (blank, empty)
        assert len(present) == 15


def test_illum_mask_excludes_central_band():
    img, _ = make_synthetic()
    _, mask, _ = detect_calibration(img, face="A")
    # every pixel in the LED/central band is excluded ...
    assert int(mask[SYN_CENTRAL[0]:SYN_CENTRAL[1]].max()) == 0
    # ... while the tube bodies remain trackable.
    assert int(mask[SYN_UPPER[0]:SYN_UPPER[1]].max()) == 255
    assert int(mask[SYN_LOWER[0]:SYN_LOWER[1]].max()) == 255


# --------------------------------------------------------------------------------------
# Pure builder used by the manual wizard
# --------------------------------------------------------------------------------------
def _grid_boxes():
    """16 body boxes matching the synthetic lattice (as a wizard would draw them)."""
    centers = _syn_centers()
    boxes = []
    for band in (SYN_UPPER, SYN_LOWER):
        for c in centers:
            boxes.append((c - 24, band[0] + 6, 48, (band[1] - band[0]) - 12))
    return boxes


def test_build_from_boxes_explicit_flags():
    img, _ = make_synthetic(blank_id=6)
    boxes = _grid_boxes()
    flags = [True] * 16
    flags[6 - 1] = False        # mark slot 6 absent, as the user would
    flags[13 - 1] = False       # and an extra user-marked absent slot

    calib, mask, overlay = build_calibration_from_boxes(img, "A", boxes, present_flags=flags)
    vials = calib.faces["A"].vials

    assert len(vials) == 16
    assert [v.id for v in vials] == list(range(1, 17))
    assert [v.present for v in vials] == flags
    # boxes are copied through faithfully
    for v, b in zip(vials, boxes):
        assert (v.x, v.y, v.w, v.h) == b
    # illuminated sub-mask: a present, lit box has trackable pixels; central band excluded.
    v1 = vials[0]
    assert mask[v1.y:v1.y + v1.h, v1.x:v1.x + v1.w].max() == 255
    assert int(mask[SYN_CENTRAL[0]:SYN_CENTRAL[1]].max()) == 0
    assert overlay.shape == (SYN_H, SYN_W, 3)


def test_build_from_boxes_derives_presence_both_empty_types():
    """present_flags=None -> presence derived. Covers both empty signatures:
    a lit body with no mouth rim (the real-frame case) AND a dark/blanked body."""
    img, _ = make_synthetic(blank_id=6)     # slot 6: lit body, no rim -> rim cue
    # make slot 8 a dark/blanked body -> dark-body cue
    centers = _syn_centers()
    c8 = centers[7]
    img[SYN_UPPER[0]:SYN_UPPER[1], c8 - 26:c8 + 26] = SYN_BG
    img[70:84, c8 - 16:c8 + 16] = SYN_BG    # (no rim either)

    calib, _, _ = build_calibration_from_boxes(img, "A", _grid_boxes(), present_flags=None)
    empty = [v.id for v in calib.faces["A"].vials if not v.present]
    assert empty == [6, 8]


def test_build_matches_detect_boxes():
    """Feeding detect's boxes back through the pure builder reproduces the same bundle."""
    img, _ = make_synthetic(blank_id=6)
    calib_d, _, _ = detect_calibration(img, face="A")
    boxes, flags = boxes_from_calibration(calib_d, "A")
    calib_b, _, _ = build_calibration_from_boxes(img, "A", boxes, present_flags=flags)
    a = [(v.id, v.row, v.col, v.x, v.y, v.w, v.h, v.present) for v in calib_d.faces["A"].vials]
    b = [(v.id, v.row, v.col, v.x, v.y, v.w, v.h, v.present) for v in calib_b.faces["A"].vials]
    assert a == b


# --------------------------------------------------------------------------------------
# Persistence round-trip
# --------------------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    img, _ = make_synthetic(blank_id=6)
    calib, mask, overlay = detect_calibration(img, face="A")
    out = str(tmp_path / "calib")
    save_calibration(calib, mask, out, overlay=overlay)

    assert os.path.isfile(os.path.join(out, "calibration.json"))
    assert os.path.isfile(os.path.join(out, "illum_mask_A.png"))
    assert os.path.isfile(os.path.join(out, "overlay_A.png"))

    loaded = load_calibration(out)
    assert loaded.image_width == calib.image_width
    assert loaded.image_height == calib.image_height
    orig = calib.faces["A"].vials
    got = loaded.faces["A"].vials
    assert len(orig) == len(got) == 16
    for a, b in zip(orig, got):
        assert (a.id, a.row, a.col, a.x, a.y, a.w, a.h, a.present) == \
               (b.id, b.row, b.col, b.x, b.y, b.w, b.h, b.present)

    # mask path resolves to a readable PNG whose central band is excluded.
    m = cv2.imread(loaded.faces["A"].illum_mask_path, cv2.IMREAD_GRAYSCALE)
    assert m is not None
    assert int(m[SYN_CENTRAL[0]:SYN_CENTRAL[1]].max()) == 0


def test_manual_flags_survive_roundtrip(tmp_path):
    img, _ = make_synthetic(blank_id=6)
    flags = [True] * 16
    flags[3] = False            # user marks slot 4 absent (overrides auto)
    calib, mask, _ = build_calibration_from_boxes(img, "A", _grid_boxes(), present_flags=flags)
    out = str(tmp_path / "calib")
    save_calibration(calib, mask, out)
    loaded = load_calibration(out)
    assert [v.present for v in loaded.faces["A"].vials] == flags


# --------------------------------------------------------------------------------------
# Real reference frame (ground truth from the rig owner: empties are ids 7 and 10)
# --------------------------------------------------------------------------------------
REAL = os.path.join(os.path.dirname(__file__), "..", "docs", "frame_full.png")


@pytest.mark.skipif(not os.path.isfile(REAL), reason="real reference frame not available")
def test_real_frame_detects_two_empties():
    img = cv2.imread(REAL, cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (1024, 1280)
    calib, mask, overlay = detect_calibration(img, face="A")
    vials = calib.faces["A"].vials

    assert len(vials) == 16
    assert [v.id for v in vials] == list(range(1, 17))
    # ground truth: exactly ids 7 (upper) and 10 (lower) are empty
    assert sorted(v.id for v in vials if not v.present) == [7, 10]

    # central band (the two ultra-bright LED slots ~ rows 470-665) is excluded
    assert int(mask[520:560].max()) == 0
    assert mask.max() == 255 and (mask > 0).sum() > 0

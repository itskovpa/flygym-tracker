"""Rotation measured from the marker band, so flies can't be misread as the drum turning.

THE PROBLEM THIS SOLVES. At 30 flies per vial the flies' collective motion pollutes a whole-frame
rotation estimate and gets read as a drum flip, which wipes the fly tracks. The marker strip is
rigid drum structure with no flies in it, so a rotation signal taken from there cannot be faked by
anything happening inside the tubes.
"""
from __future__ import annotations

import numpy as np

from flygym_tracker.pipeline import TrackerPipeline


class _Calib:
    image_height = 480
    image_width = 640


class _BandDetector:
    band_rows = (100, 160)      # inclusive rows of the marker strip


def _pipe():
    p = TrackerPipeline.__new__(TrackerPipeline)   # exercise the helper without a camera
    p.calibration = _Calib()
    return p


def test_the_mask_covers_exactly_the_band_rows_full_width():
    mask = _pipe()._marker_band_roi_mask(_BandDetector())
    assert mask is not None
    assert mask.shape == (480, 640)
    assert mask[100:161, :].all(), "the band rows are not all inside the ROI"
    assert not mask[:100, :].any(), "rows above the band leaked into the ROI"
    assert not mask[161:, :].any(), "rows below the band leaked into the ROI"


def test_no_band_rows_falls_back_to_none_not_a_wrong_mask():
    """A detector with no band -- uncalibrated, or the non-marker kind -- returns None so the
    caller uses the whole frame, rather than a mask of the wrong region that would measure rotation
    from nothing."""
    class NoBand:
        band_rows = None

    assert _pipe()._marker_band_roi_mask(NoBand()) is None
    assert _pipe()._marker_band_roi_mask(object()) is None


def test_band_rows_out_of_the_frame_are_clamped():
    class Overflow:
        band_rows = (-20, 100000)

    mask = _pipe()._marker_band_roi_mask(Overflow())
    assert mask.shape == (480, 640)
    assert mask.all(), "a band spanning the frame should mask the whole frame, not crash"


def test_the_config_default_is_the_validated_whole_frame():
    """A generic install must keep the validated behaviour unless it opts in -- the marker-band ROI
    changes the signal and needs a calibrated band, so it is not the silent default everywhere."""
    import yaml

    from flygym_tracker.config import DEFAULT_CONFIG_PATH

    loaded = yaml.safe_load(open(DEFAULT_CONFIG_PATH, encoding="utf-8"))
    assert loaded["rotation"]["roi"] == "full"


# =============================================================================================
# Auto-location: build the band ROI LIVE from the marker detector, for a calibration that stores no
# fixed band_rows (this rig re-finds the band every frame). This is what makes `roi: marker_band`
# actually engage instead of silently falling back to whole-frame.
# =============================================================================================
class _FakeRotation:
    roi_mask = None


class _FindStrips:
    """Marker detector that returns two strips (upper 100-120, lower 150-165) -> band rows 100..165."""

    def find_strips(self, gray):
        return [(100, 120), (150, 165)]


def _pipe_for_autolocate(marker_detector, rotation=None):
    p = _pipe()
    p.marker_detector = marker_detector
    p.rotation = rotation if rotation is not None else _FakeRotation()
    p._rotation_roi_mode = True
    p._band_roi_located = False
    p._rotation_roi = "marker_band (locating)"
    return p


def test_auto_location_points_the_roi_at_the_located_band():
    p = _pipe_for_autolocate(_FindStrips())
    p._refresh_rotation_band_roi(np.zeros((480, 640), dtype=np.uint8))

    assert p._band_roi_located is True
    m = p.rotation.roi_mask
    assert m is not None and m.shape == (480, 640)
    assert m[100:166, :].all(), "the band rows (100..165 inclusive) are not all inside the ROI"
    assert not m[:100, :].any() and not m[166:, :].any(), "rows outside the band leaked into the ROI"
    assert "marker_band" in p._rotation_roi


def test_auto_location_builds_the_mask_to_the_frame_shape_not_the_calibration():
    """The mask must match the frame phaseCorrelate is actually given, even if a frame arrives at a
    different size than the calibration -- otherwise the correlation would raise on a shape clash."""
    p = _pipe_for_autolocate(_FindStrips())
    p._refresh_rotation_band_roi(np.zeros((300, 500), dtype=np.uint8))   # != _Calib's 480x640
    assert p.rotation.roi_mask.shape == (300, 500)


def test_auto_location_no_band_this_frame_leaves_the_previous_roi():
    """A frame with no findable band is a no-op: the previous ROI (whole-frame or a prior band)
    stands, rather than being blanked to a wrong region."""
    class NoStrips:
        def find_strips(self, gray):
            return []

    rot = _FakeRotation()
    rot.roi_mask = "PREVIOUS"          # sentinel: whatever was in force before
    p = _pipe_for_autolocate(NoStrips(), rotation=rot)
    p._refresh_rotation_band_roi(np.zeros((480, 640), dtype=np.uint8))
    assert p._band_roi_located is False
    assert p.rotation.roi_mask == "PREVIOUS"


def test_auto_location_tolerates_a_detector_without_find_strips():
    """The generic (non-band) marker detector has no find_strips; refresh must be a safe no-op."""
    p = _pipe_for_autolocate(object())
    p._refresh_rotation_band_roi(np.zeros((480, 640), dtype=np.uint8))
    assert p._band_roi_located is False

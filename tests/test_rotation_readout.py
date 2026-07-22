"""The live rotation signal on screen, so a fake rotation is visible as it fires.

WHY THIS EXISTS. At 30 flies per vial the flies' own motion can be misread as the drum rotating,
and every false rotation wipes the fly tracks. A rotation COUNT tells the operator one already
happened; it does not let them SEE the detector being fooled. This readout shows the live
displacement against the threshold it must cross, so "disp sat just under the line and then jumped
while the drum was still" is something the operator can watch.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from flygym_tracker.gui.run_panel import _rotation_readout  # noqa: E402


def test_the_signal_is_shown_against_its_threshold():
    text = _rotation_readout({"rotation": {"disp": 0.14, "enter": 0.20}})
    assert "0.140" in text and "0.200" in text
    assert "stationary" in text, "below the threshold should read stationary"


def test_crossing_the_threshold_reads_as_rotating():
    text = _rotation_readout({"rotation": {"disp": 0.33, "enter": 0.20}})
    assert "ROTATING" in text


def test_a_detector_without_the_signal_shows_nothing_rather_than_a_fake_zero():
    """The fixed-threshold detector has no `last_disp`. A "0.000 / 0.000" would be a measurement
    the detector never made -- same rule as blank vs zero everywhere else in this program."""
    assert _rotation_readout({"rotation": {"disp": None, "enter": None}}) == ""
    assert _rotation_readout({}) == ""


def test_the_pipeline_payload_carries_the_signal():
    """`_rotation_signal` reads the adaptive detector's live diagnostics into plain floats. A fake
    detector with the two attributes stands in for the real one, which needs a camera."""
    from flygym_tracker.pipeline import TrackerPipeline

    class FakeDetector:
        last_disp = 0.12
        enter_threshold = 0.20
        last_consistency = 0.4

    pipe = TrackerPipeline.__new__(TrackerPipeline)   # no __init__: we only exercise the helper
    pipe.rotation = FakeDetector()
    signal = pipe._rotation_signal()
    assert signal == {"disp": 0.12, "enter": 0.20, "consistency": 0.4}


def test_the_readout_shows_which_roi_the_signal_came_from():
    """So the operator can SEE the marker band is engaged (or still locating), not just trust it."""
    band = _rotation_readout({"rotation": {"disp": 0.02, "enter": 0.11},
                              "rotation_roi": "marker_band rows=485-594"})
    assert "band 485-594" in band

    locating = _rotation_readout({"rotation": {"disp": 0.02, "enter": 0.11},
                                  "rotation_roi": "marker_band (locating)"})
    assert "band?" in locating

    full = _rotation_readout({"rotation": {"disp": 0.02, "enter": 0.11}, "rotation_roi": "full"})
    assert "full" in full

    # No label (an older pipeline) leaves the readout exactly as it was -- signal only, no ROI tag.
    plain = _rotation_readout({"rotation": {"disp": 0.02, "enter": 0.11}})
    assert plain.endswith("stationary")

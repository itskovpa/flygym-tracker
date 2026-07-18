"""Tests for flygym_tracker.markers.MarkerDetector (DESIGN.md §5.2, §8).

Fully synthetic (no physical marker exists yet -- see markers.py's module docstring). Covers the
parts of the framework that ARE real: dark-silhouette detection scoped to a search region,
area/aspect filtering, the Hu+skew signature, and nearest-signature matching -- in particular that
the signature distinguishes a marker from its own 180 degree flip, which is the whole point for
this rig (Face A vs Face B is exactly the drum's ~180 degree flip, DESIGN.md §2).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from flygym_tracker.markers import MarkerDetector, MarkerParams

# --------------------------------------------------------------------------------------
# Synthetic frame builders
# --------------------------------------------------------------------------------------
SHAPE = (200, 220)                     # H, W -- used for every synthetic frame below
BRIGHT = 220                           # bright back-lit ground level
DARK = 30                              # opaque marker level
SEARCH_REGION = (20, 10, 160, 170)     # (x, y, w, h): generously contains every marker bbox below

# A single asymmetric right-triangle bbox, reused by several tests: right angle at its top-left
# corner. Its 180-degree point reflection (about its own bbox center) is the complementary
# right-angle triangle occupying the SAME bbox -- i.e. cutting the bbox rectangle along one
# diagonal gives marker "A" and marker "B" as the two halves. This is the hard case DESIGN.md
# calls out: same footprint/bbox, same Hu moments (rotation invariant), different signature only
# via the orientation-sensitive (skew) half of `_signature`.
TRI_X0, TRI_Y0, TRI_W, TRI_H = 60, 50, 60, 90


def _bright_frame(shape=SHAPE, bright=BRIGHT, noise_std=0.0, seed=0):
    """A uniform back-lit-bright frame, optionally with sensor-noise-like jitter."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, float(bright))
    if noise_std > 0:
        img = img + rng.normal(0.0, noise_std, size=shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_polygon(img, points, dark=DARK):
    out = img.copy()
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(out, [pts], dark)
    return out


def _draw_rect(img, x, y, w, h, dark=DARK):
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), dark, -1)
    return out


def _triangle_a(x0=TRI_X0, y0=TRI_Y0, w=TRI_W, h=TRI_H):
    """Right angle at the top-left of its bbox."""
    return [(x0, y0), (x0 + w, y0), (x0, y0 + h)]


def _reflect_180(points, cx, cy):
    """Point-reflect `points` through (cx, cy) -- a 180 degree rotation about that center."""
    return [(2 * cx - x, 2 * cy - y) for x, y in points]


def _triangle_b_flipped(x0=TRI_X0, y0=TRI_Y0, w=TRI_W, h=TRI_H):
    """`_triangle_a`'s own bbox, rotated 180 degrees about the bbox center (the "other half")."""
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    return _reflect_180(_triangle_a(x0, y0, w, h), cx, cy)


def _triangle_c_different(x0=TRI_X0, y0=TRI_Y0, w=TRI_W, h=TRI_H):
    """A different asymmetric triangle in the same bbox (not a flip of `_triangle_a`)."""
    return [(x0, y0), (x0 + w, y0 + h // 3), (x0 + w // 4, y0 + h)]


# --------------------------------------------------------------------------------------
# Core spec scenarios
# --------------------------------------------------------------------------------------
def test_register_and_identify_marker():
    frame = _draw_polygon(_bright_frame(), _triangle_a())
    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)

    det.register_marker(frame, "A")
    assert det.identify_face(frame) == "A"


def test_distinguishes_marker_from_180_degree_flip():
    """The whole point for this rig: Face A vs Face B is exactly the drum's ~180 degree flip."""
    frame_a = _draw_polygon(_bright_frame(), _triangle_a())
    frame_b = _draw_polygon(_bright_frame(), _triangle_b_flipped())

    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)
    det.register_marker(frame_a, "A")
    det.register_marker(frame_b, "B")

    assert det.identify_face(frame_a) == "A"
    assert det.identify_face(frame_b) == "B"


def test_distinguishes_two_clearly_different_markers():
    """The gentler sibling case: genuinely different shapes, not just a flip of one another."""
    frame_a = _draw_polygon(_bright_frame(), _triangle_a())
    frame_c = _draw_polygon(_bright_frame(), _triangle_c_different())

    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)
    det.register_marker(frame_a, "A")
    det.register_marker(frame_c, "B")

    assert det.identify_face(frame_a) == "A"
    assert det.identify_face(frame_c) == "B"


def test_disabled_returns_none_even_with_a_matching_registry():
    frame = _draw_polygon(_bright_frame(), _triangle_a())

    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)
    det.register_marker(frame, "A")
    det.enabled = False
    assert det.identify_face(frame) is None

    # also via the constructor flag directly, with a pre-populated registry that WOULD match.
    det2 = MarkerDetector(enabled=False, search_region=SEARCH_REGION, registry=det.to_dict())
    assert det2.identify_face(frame) is None


def test_no_dark_shape_returns_none():
    frame = _bright_frame(noise_std=2.0, seed=1)  # bright, only sensor-noise-scale jitter
    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)
    assert det.identify_face(frame) is None


def test_to_dict_from_dict_roundtrips_registry():
    frame_a = _draw_polygon(_bright_frame(), _triangle_a())
    frame_b = _draw_polygon(_bright_frame(), _triangle_b_flipped())

    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)
    det.register_marker(frame_a, "A")
    det.register_marker(frame_b, "B")

    data = det.to_dict()
    assert set(data.keys()) == {"A", "B"}
    assert all(isinstance(v, list) and all(isinstance(x, float) for x in v) for v in data.values())

    restored = MarkerDetector.from_dict(data, enabled=True, search_region=SEARCH_REGION)
    assert set(restored.registry.keys()) == {"A", "B"}
    for face in ("A", "B"):
        assert restored.registry[face] == pytest.approx(det.registry[face])

    # round-tripped detector behaves identically to the original.
    assert restored.identify_face(frame_a) == "A"
    assert restored.identify_face(frame_b) == "B"


# --------------------------------------------------------------------------------------
# Detection mechanics: search_region / area / aspect filtering
# --------------------------------------------------------------------------------------
def test_search_region_scopes_detection():
    region = (100, 100, 80, 80)  # x:100-180, y:100-180
    outside = _draw_rect(_bright_frame(), 10, 10, 30, 30)  # well outside `region`

    det = MarkerDetector(enabled=True, search_region=region)
    with pytest.raises(ValueError):
        det.register_marker(outside, "A")  # nothing dark inside the region

    inside = _draw_rect(outside, 120, 120, 30, 30)  # add a second blob inside the region
    det.register_marker(inside, "A")  # now finds the one inside the region
    assert "A" in det.registry


def test_area_filter_rejects_undersized_and_oversized_blobs():
    det = MarkerDetector(enabled=True, min_area=200, max_area=1000)

    too_small = _draw_rect(_bright_frame(), 60, 60, 8, 8)      # area ~64 < min_area
    too_big = _draw_rect(_bright_frame(), 20, 20, 150, 150)    # area ~22500 > max_area
    just_right = _draw_rect(_bright_frame(), 60, 60, 25, 25)   # area ~625, in range

    with pytest.raises(ValueError):
        det.register_marker(too_small, "A")
    with pytest.raises(ValueError):
        det.register_marker(too_big, "A")
    det.register_marker(just_right, "A")  # must not raise
    assert "A" in det.registry


def test_aspect_filter_rejects_thin_sliver():
    det = MarkerDetector(enabled=True, params=MarkerParams(max_aspect_ratio=6.0))

    sliver = _draw_rect(_bright_frame(), 60, 30, 4, 120)     # area 480, long/short = 30
    squarish = _draw_rect(_bright_frame(), 60, 60, 24, 20)   # area 480, long/short = 1.2

    with pytest.raises(ValueError):
        det.register_marker(sliver, "A")
    det.register_marker(squarish, "A")  # must not raise
    assert "A" in det.registry


def test_register_marker_raises_without_candidate():
    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)
    with pytest.raises(ValueError):
        det.register_marker(_bright_frame(), "A")


# --------------------------------------------------------------------------------------
# Signature matching mechanics
# --------------------------------------------------------------------------------------
def test_max_match_distance_rejects_a_far_match():
    """`MarkerParams.max_match_distance` gates acceptance -- a mechanism test, not a claim about
    what the real threshold should be once physical markers exist (see markers.py's docstring)."""
    frame_a = _draw_polygon(_bright_frame(), _triangle_a())
    frame_c = _draw_polygon(_bright_frame(), _triangle_c_different())  # unregistered, dissimilar

    det = MarkerDetector(
        enabled=True, search_region=SEARCH_REGION, params=MarkerParams(max_match_distance=5.0)
    )
    det.register_marker(frame_a, "A")

    assert det.identify_face(frame_a) == "A"    # self-match: distance ~0, well under 5.0
    assert det.identify_face(frame_c) is None   # dissimilar unregistered shape: rejected, not "A"


def test_max_match_distance_none_never_rejects():
    """Default behaviour: with no distance cap, the nearest registered signature always wins."""
    frame_a = _draw_polygon(_bright_frame(), _triangle_a())
    frame_c = _draw_polygon(_bright_frame(), _triangle_c_different())

    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)  # max_match_distance=None
    assert det.params.max_match_distance is None
    det.register_marker(frame_a, "A")

    assert det.identify_face(frame_c) == "A"  # only registered face, no rejection threshold


def test_empty_registry_returns_none():
    frame = _draw_polygon(_bright_frame(), _triangle_a())
    det = MarkerDetector(enabled=True, search_region=SEARCH_REGION)
    assert det.registry == {}
    assert det.identify_face(frame) is None  # candidate found, but nothing to match against

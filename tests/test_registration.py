"""Tests for flygym_tracker.registration (DESIGN.md §5.2)."""
import numpy as np
import pytest

from flygym_tracker.registration import apply_shift, estimate_shift


def _textured_frame(seed, shape=(240, 320)):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


# ---- estimate_shift -----------------------------------------------------------


def test_estimate_shift_recovers_known_integer_shift():
    ref = _textured_frame(123)
    dx_true, dy_true = 6, -4
    # np.roll(shift=(dy,dx), axis=(0,1)): content at (x,y) in ref lands at (x+dx,y+dy) in cur.
    cur = np.roll(ref, shift=(dy_true, dx_true), axis=(0, 1))

    dx, dy, residual = estimate_shift(cur, ref)

    assert dx == pytest.approx(dx_true, abs=0.5)
    assert dy == pytest.approx(dy_true, abs=0.5)
    assert 0.0 <= residual <= 1.0
    # exact circular shift of a textured frame -> near-perfect phase correlation peak
    assert residual < 0.05


def test_estimate_shift_recovers_another_known_shift_different_sign():
    ref = _textured_frame(77)
    dx_true, dy_true = -11, 9
    cur = np.roll(ref, shift=(dy_true, dx_true), axis=(0, 1))

    dx, dy, residual = estimate_shift(cur, ref)

    assert dx == pytest.approx(dx_true, abs=0.5)
    assert dy == pytest.approx(dy_true, abs=0.5)
    assert residual < 0.05


def test_estimate_shift_zero_shift_gives_zero_and_low_residual():
    ref = _textured_frame(9)
    dx, dy, residual = estimate_shift(ref, ref)
    assert dx == pytest.approx(0.0, abs=0.5)
    assert dy == pytest.approx(0.0, abs=0.5)
    assert residual < 0.05


def test_estimate_shift_masked_recovers_shift():
    ref = _textured_frame(55)
    dx_true, dy_true = 4, -2
    cur = np.roll(ref, shift=(dy_true, dx_true), axis=(0, 1))

    mask = np.ones(ref.shape, dtype=bool)
    mask[:, :30] = False  # exclude a strip, proving the mask path is exercised

    dx, dy, residual = estimate_shift(cur, ref, mask=mask)
    assert dx == pytest.approx(dx_true, abs=0.5)
    assert dy == pytest.approx(dy_true, abs=0.5)


def test_estimate_shift_unrelated_frames_have_high_residual():
    a = _textured_frame(3, shape=(150, 150))
    b = _textured_frame(4, shape=(150, 150))
    _, _, residual = estimate_shift(a, b)
    # two independent random frames share no coherent phase relationship
    assert residual > 0.8


# ---- apply_shift -----------------------------------------------------------


def test_apply_shift_moves_bbox_and_rounds():
    bbox = (10, 20, 30, 40)  # x, y, w, h
    new_bbox = apply_shift(bbox, dx=2.6, dy=-1.4)
    assert new_bbox == (13, 19, 30, 40)  # round(12.6)=13, round(18.6)=19


def test_apply_shift_negative_shift():
    bbox = (0, 0, 5, 5)
    new_bbox = apply_shift(bbox, dx=-3.0, dy=4.0)
    assert new_bbox == (-3, 4, 5, 5)


def test_apply_shift_preserves_width_height():
    bbox = (100, 200, 42, 17)
    new_bbox = apply_shift(bbox, dx=0.0, dy=0.0)
    assert new_bbox == (100, 200, 42, 17)


def test_apply_shift_result_is_integer_tuple():
    bbox = (5, 5, 10, 10)
    x, y, w, h = apply_shift(bbox, dx=1.2, dy=2.8)
    assert all(isinstance(v, int) for v in (x, y, w, h))


def test_apply_shift_roundtrips_with_estimate_shift():
    """End-to-end sanity: a bbox tracked through a known content shift lands within 1px."""
    ref = _textured_frame(21)
    dx_true, dy_true = 8, 5
    cur = np.roll(ref, shift=(dy_true, dx_true), axis=(0, 1))

    dx, dy, _residual = estimate_shift(cur, ref)
    bbox_ref = (50, 60, 20, 20)
    bbox_cur = apply_shift(bbox_ref, dx, dy)

    assert bbox_cur[0] == pytest.approx(50 + dx_true, abs=1)
    assert bbox_cur[1] == pytest.approx(60 + dy_true, abs=1)
    assert bbox_cur[2:] == (20, 20)

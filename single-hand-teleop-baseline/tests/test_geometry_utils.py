import math

import pytest

from features.geometry_utils import clamp01, euclidean, joint_angle, normalize_between, polyline_length


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.5, 0.0),
        (0.0, 0.0),
        (0.25, 0.25),
        (1.0, 1.0),
        (1.5, 1.0),
    ],
)
def test_clamp01_bounds_values(value, expected):
    assert clamp01(value) == expected


def test_joint_angle_returns_default_for_degenerate_segments():
    assert joint_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == math.pi


def test_joint_angle_detects_right_angle():
    angle = joint_angle((1.0, 0.0), (0.0, 0.0), (0.0, 1.0))
    assert math.isclose(angle, math.pi / 2.0, rel_tol=1e-6)


def test_polyline_length_and_euclidean_support_3d_points():
    assert math.isclose(euclidean((0.0, 0.0, 0.0), (0.0, 3.0, 4.0)), 5.0)
    assert math.isclose(polyline_length([(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]), 10.0)
    assert polyline_length([(1.0, 2.0)]) == 0.0


@pytest.mark.parametrize(
    ("value", "open_ref", "closed_ref", "expected"),
    [
        (0.02, 0.02, 0.55, 0.0),
        (0.55, 0.02, 0.55, 1.0),
        (0.30, 0.02, 0.55, pytest.approx((0.30 - 0.02) / (0.55 - 0.02))),
        (0.95, 0.95, 0.25, 0.0),
        (0.25, 0.95, 0.25, 1.0),
        (0.0, 0.02, 0.55, 0.0),
        (2.0, 0.02, 0.55, 1.0),
    ],
)
def test_normalize_between_handles_increasing_and_decreasing_ranges(value, open_ref, closed_ref, expected):
    assert normalize_between(value, open_ref, closed_ref) == expected

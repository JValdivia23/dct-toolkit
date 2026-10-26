"""Coordinate precision and lazy nominal-count support regressions."""

from typing import Callable, Tuple

import numpy as np
import pytest

from dct_toolkit import dct_count, dct_mean, dct_prefill, dct_smooth, dct_std, dct_variance
from dct_toolkit._geometry import _resolve_polar_geometry
from dct_toolkit.polar import smooth_polar


_COORDINATE_ENTRY_POINTS = (
    smooth_polar, dct_smooth, dct_mean, dct_variance, dct_std, dct_count, dct_prefill,
)


@pytest.mark.parametrize("function", _COORDINATE_ENTRY_POINTS)
@pytest.mark.parametrize("role", ["time", "linear"])
@pytest.mark.parametrize(
    "coordinate, width",
    [
        (np.array([-40000.0, 40000.0], dtype=np.float16), 80000.0),
        (np.array([-3e38, 3e38], dtype=np.float32), 6e38),
    ],
    ids=["float16-step-overflow", "float32-step-overflow"],
)
def test_low_precision_coordinates_use_promoted_spacing(
    function: Callable[..., np.ndarray], coordinate: np.ndarray, width: float, role: str
) -> None:
    """Valid coordinate steps need not be representable in their input float dtype."""
    original = coordinate.copy()
    data = np.array([0.0, 1.0])
    values = np.ones(2, dtype=bool) if function is dct_count else data
    kwargs = {"axis_types": (role,)}
    if function is not smooth_polar:
        kwargs["coordinates"] = "polar"
    with np.errstate(over="raise", invalid="raise"):
        actual = function(values, width, axis_coords=(coordinate,), **kwargs)
        expected = function(
            values, width, axis_coords=(coordinate.astype(np.float64),), **kwargs
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(coordinate, original)


@pytest.mark.parametrize("function", _COORDINATE_ENTRY_POINTS)
@pytest.mark.parametrize("role", ["time", "linear"])
def test_low_precision_subtraction_does_not_hide_irregularity(
    function: Callable[..., np.ndarray], role: str
) -> None:
    """The exact steps 59999 and 60001 must not both round to float16 60000."""
    coordinate = np.array([-60000.0, -1.0, 60000.0], dtype=np.float16)
    values = np.ones(3, dtype=bool) if function is dct_count else np.array([0.0, 1.0, 0.0])
    kwargs = {"axis_types": (role,), "axis_coords": (coordinate,)}
    if function is not smooth_polar:
        kwargs["coordinates"] = "polar"
    with pytest.raises(ValueError, match="uniformly spaced"):
        function(values, 60000.0, **kwargs)


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps,
    reason="The platform does not provide a wider floating coordinate type",
)
def test_promoting_small_floats_does_not_downcast_wider_coordinate_types() -> None:
    """Preserve native wider precision when validating a large floating epoch."""
    coordinate = np.longdouble("1e18") + np.arange(4, dtype=np.longdouble) * 1000
    data = np.arange(4, dtype=float)
    actual = smooth_polar(
        data, 2250.0, axis_types=("time",), axis_coords=(coordinate,)
    )
    expected = smooth_polar(data, 2.25, axis_types=("time",), width_units="grid")
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("role", ["time", "linear", "range"])
@pytest.mark.parametrize(
    "coordinate",
    [
        10**18 + 1000 * np.arange(4, dtype=np.int64),
        10**18 + np.arange(4, dtype=np.int64),
        np.array([-(2**63) + i for i in range(4)], dtype=np.int64),
        np.array([2**63 - 4 + i for i in range(4)], dtype=np.int64),
        np.array([2**64 - 4 + i for i in range(4)], dtype=np.uint64),
        np.array([-(2**63), 2**63 - 1], dtype=np.int64),
        np.array([0, 2**64 - 1], dtype=np.uint64),
    ],
    ids=["epoch-milliseconds", "epoch-unit-steps", "int64-min", "int64-max",
         "uint64-max", "signed-full-span", "unsigned-full-span"],
)
def test_integer_spacing_matches_relative_and_grid_coordinates(
    coordinate: np.ndarray, role: str
) -> None:
    """Large absolute integer epochs do not affect exact original spacings."""
    step = int(coordinate[1]) - int(coordinate[0])
    relative = np.arange(coordinate.size, dtype=float) * step
    data = np.random.default_rng(1301).standard_normal(coordinate.size)
    original = coordinate.copy()
    kwargs = {"axis_types": (role,), "boundary": "periodic"}
    width = 2.25 * step
    with np.errstate(over="raise", invalid="raise"):
        actual = smooth_polar(data, width, axis_coords=(coordinate,), **kwargs)
        expected = smooth_polar(data, width, axis_coords=(relative,), **kwargs)
        grid = smooth_polar(data, 2.25, width_units="grid", **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual, grid, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(coordinate, original)
    np.testing.assert_allclose(
        smooth_polar(np.ones(coordinate.size), step, axis_coords=(coordinate,), **kwargs),
        1.0, rtol=0.0, atol=1e-14,
    )


@pytest.mark.parametrize(
    "coordinate",
    [
        10**18 + np.array([0, 1000, 2001, 3001], dtype=np.int64),
        10**18 + np.array([0, 1, 2, 4], dtype=np.int64),
        np.array([2**64 - 5, 2**64 - 4, 2**64 - 2, 2**64 - 1], dtype=np.uint64),
        np.array([-(2**63), 0, 1], dtype=np.int64),
    ],
    ids=["epoch-milliseconds", "epoch-unit-steps", "uint64-max", "signed-wide-step"],
)
def test_true_integer_irregularity_is_not_erased_by_float_rounding(
    coordinate: np.ndarray,
) -> None:
    """Exact original differences still expose genuinely irregular integer grids."""
    with pytest.raises(ValueError, match="uniformly spaced"):
        smooth_polar(
            np.ones(coordinate.size), 1.0, axis_types=("time",), axis_coords=(coordinate,)
        )


@pytest.mark.parametrize(
    "coordinate",
    [
        np.array([2**63 - 1, -(2**63)], dtype=np.int64),
        np.array([2**64 - 1, 0], dtype=np.uint64),
        10**18 + np.array([3, 2, 1, 0], dtype=np.int64),
        np.array([2**64 - 2, 2**64 - 2], dtype=np.uint64),
    ],
    ids=["signed-wrap", "unsigned-wrap", "epoch-descending", "uint64-duplicate"],
)
def test_integer_descending_and_duplicate_values_cannot_wrap_into_positive_steps(
    coordinate: np.ndarray,
) -> None:
    """Original-dtype comparisons reject descent before unsigned differencing."""
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="strictly increasing"):
            smooth_polar(
                np.ones(coordinate.size), 1.0, axis_types=("time",), axis_coords=(coordinate,)
            )


@pytest.mark.parametrize("origin", [10**18, -(10**18), 2**64 - 360])
def test_integer_azimuth_spacing_is_derived_before_radian_conversion(origin: int) -> None:
    """A large azimuth origin cannot change the step or full-circle validation."""
    dtype = np.uint64 if origin > np.iinfo(np.int64).max else np.int64
    angles = np.array([origin + 90 * i for i in range(4)], dtype=dtype)
    data = np.array([0.0, 4.0, -1.0, 2.0])[:, None]
    width = 2.25 * 5.0 * np.pi / 2
    kwargs = {
        "axis_types": ("azimuth", "range"), "filter_axes": (0,), "boundary": "periodic",
    }
    actual = smooth_polar(data, (width, 1.0), axis_coords=(angles, np.array([5.0])), **kwargs)
    expected = smooth_polar(
        data, (width, 1.0), axis_coords=(np.arange(4) * 90.0, np.array([5.0])), **kwargs
    )
    grid = smooth_polar(data, (2.25, 1.0), width_units="grid", **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual, grid, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("angle_units", ["degrees", "radians"])
def test_float_azimuth_spacing_precedes_unit_conversion(angle_units: str) -> None:
    """Even representable floating-point steps can be lost by converting offsets first."""
    angles = 1e16 + 2.0 * np.arange(4)
    step = np.deg2rad(2.0) if angle_units == "degrees" else 2.0
    data = np.array([0.0, 4.0, -1.0, 2.0])[:, None]
    kwargs = {"axis_types": ("azimuth", "range"), "filter_axes": (0,)}
    actual = smooth_polar(
        data, (2.25 * 5.0 * step, 1.0), axis_coords=(angles, np.array([5.0])),
        angle_units=angle_units, **kwargs,
    )
    expected = smooth_polar(data, (2.25, 1.0), width_units="grid", **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("width", [1e100, 1e-100])
@pytest.mark.parametrize("kernel", ["gaussian", "boxcar", "boxcar_discrete"])
@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
def test_smoothing_and_moments_do_not_require_representable_nominal_support(
    width: float, kernel: str, boundary: str
) -> None:
    """Finite per-axis widths remain valid even when their product over/underflows."""
    data = np.ones((2, 2, 2, 2))
    kwargs = {
        "axis_types": ("time",) * 4, "width_units": "grid",
        "kernel_type": kernel, "boundary": boundary,
    }
    geometry = _resolve_polar_geometry(data.shape, width, **kwargs)
    assert geometry.widths_in_grid == (width,) * 4
    np.testing.assert_allclose(smooth_polar(data, width, **kwargs), 1.0, atol=2e-14)
    for function, expected in (
        (dct_smooth, 1.0), (dct_mean, 1.0), (dct_variance, 0.0), (dct_std, 0.0)
    ):
        actual = function(data, width, coordinates="polar", **kwargs)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)
    with pytest.raises(ValueError, match="dct_count.*represented"):
        _ = geometry.nominal_support
    with pytest.raises(ValueError, match="dct_count.*represented"):
        dct_count(data.astype(bool), width, coordinates="polar", **kwargs)


@pytest.mark.parametrize(
    "widths",
    [
        (1e200, 1e200, 1e-200, 1e-200),
        (1e-200, 1e-200, 1e200, 1e200),
        (1e-200, 1e-120, 1e200, 1e120),
    ],
    ids=["intermediate-overflow", "intermediate-underflow", "subnormal-intermediate"],
)
def test_representable_count_factor_survives_extreme_intermediate_products(
    widths: Tuple[float, ...],
) -> None:
    """Count support rescales exponents rather than rejecting a valid final factor."""
    mask = np.ones((2, 2, 2, 2), dtype=bool)
    kwargs = {"axis_types": ("time",) * 4, "width_units": "grid"}
    geometry = _resolve_polar_geometry(mask.shape, widths, **kwargs)
    np.testing.assert_allclose(geometry.nominal_support, 1.0, rtol=2e-15, atol=0.0)
    actual = dct_count(mask, widths, coordinates="polar", **kwargs)
    np.testing.assert_allclose(actual, 1.0, rtol=2e-15, atol=0.0)


def test_physical_count_and_compact_support_are_unchanged() -> None:
    """Lazy support retains physical geometry, role order, and pre-rounding widths."""
    elevations = np.array([-30.0, 0.0, 30.0])
    ranges = 3.0 + np.arange(4) * 2.0
    mask = np.ones((3, 8, 4, 2), dtype=bool)
    widths = (2.0, 4.0, 6.0, 5.0)
    kwargs = {
        "axis_types": ("elevation", "azimuth", "range", "time"),
        "axis_coords": (elevations, np.arange(8) * 45.0, ranges, np.array([0.0, 2.0])),
        "kernel_type": "boxcar_discrete", "az_boundary": "periodic",
    }
    radii = ranges.reshape(1, 1, 4, 1)
    cosines = np.cos(np.deg2rad(elevations)).reshape(3, 1, 1, 1)
    expected = (2.0 / (radii * np.pi / 6)) * (4.0 / (radii * cosines * np.pi / 4))
    expected = expected * 3.0 * 2.5
    geometry = _resolve_polar_geometry(mask.shape, widths, **kwargs)
    support = geometry.nominal_support
    assert support.shape == (3, 1, 4, 1)
    assert not support.flags.writeable
    np.testing.assert_allclose(support, expected, rtol=2e-15, atol=0.0)
    actual = dct_count(mask, widths, coordinates="polar", **kwargs)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, mask.shape), rtol=2e-15)


def test_legacy_nominal_support_retains_original_numerics() -> None:
    """Legacy count factors keep the same multiplication and index-radius origin."""
    geometry = _resolve_polar_geometry((8, 4), (3.5, 2.3), az_res_deg=45.0)
    expected = 2.3 * (3.5 / (np.arange(1, 5) * np.deg2rad(45.0)))
    np.testing.assert_array_equal(geometry.nominal_support, expected.reshape(1, 4))
    for width, value in ((1e200, np.inf), (1e-200, 0.0)):
        geometry = _resolve_polar_geometry((8, 4), width, az_res_deg=45.0)
        with np.errstate(over="ignore", under="ignore"):
            np.testing.assert_array_equal(geometry.nominal_support, np.full((1, 4), value))


@pytest.mark.parametrize("width, spacing", [(1e100, 1e-300), (1e-100, 1e300)])
def test_each_effective_width_is_validated_without_requesting_count_support(
    width: float, spacing: float
) -> None:
    """Lazy products do not weaken validation of an unrepresentable individual width."""
    data = np.ones(4)
    kwargs = {"axis_types": ("time",), "axis_coords": (np.arange(4) * spacing,)}
    for function in (smooth_polar, dct_smooth, dct_mean, dct_std, dct_count):
        coordinates = {} if function is smooth_polar else {"coordinates": "polar"}
        with pytest.raises(ValueError, match="Effective grid widths.*finite and positive"):
            function(data, width, **coordinates, **kwargs)

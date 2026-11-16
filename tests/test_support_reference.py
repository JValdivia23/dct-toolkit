"""Independent spatial acceptance tests for local statistical support.

Expected weights are assembled by visiting reflected/wrapped spatial samples,
or by the independent spatial polar reference. No runtime geometry, transfer,
support, or order-statistic helper is used to construct an expected answer.
Dense matrices are deliberately restricted to tiny acceptance-test grids.
"""

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pytest

from dct_toolkit.support import SupportDiagnostics, local_support
from test_polar_nd import _polar_reference


WEIGHT_TOLERANCE = 64 * np.finfo(float).eps
SUPPORT_FIELDS = (
    "original_valid_weight", "valid_weight", "available_weight", "valid_fraction",
    "self_weight", "excluded_weight", "supported",
)


def _spatial_line_matrix(
    length: int, width: float, kernel: str = "gaussian", boundary: str = "reflective"
) -> np.ndarray:
    """Sum spatial sample visits, including every reflected or periodic duplicate."""
    if kernel == "gaussian":
        sigma = width / np.sqrt(12.0)
        # Twelve sigma leaves less than 4e-33 omitted probability; this is an
        # oracle precision choice, not the runtime's allowed kernel definition.
        radius = max(1, int(np.ceil(12 * sigma)))
        offsets = np.arange(-radius, radius + 1)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= weights.sum()
    else:
        assert kernel == "boxcar_discrete"
        size = max(1, int(np.round(width)))
        size += int(size % 2 == 0)
        offsets = np.arange(-(size // 2), size // 2 + 1)
        weights = np.full(size, 1.0 / size)
    sources = np.arange(length)[:, None] + offsets
    if boundary == "periodic":
        sources %= length
    else:
        assert boundary == "reflective"
        sources %= 2 * length
        sources = np.where(sources < length, sources, 2 * length - 1 - sources)
    matrix = np.zeros((length, length))
    np.add.at(matrix, (np.arange(length)[:, None], sources), weights[None, :])
    return matrix


def _cartesian_matrix(
    shape: Tuple[int, ...], widths: Tuple[float, ...], kernels: Tuple[str, ...]
) -> np.ndarray:
    """Form a tensor-product operator from independently assembled spatial rows."""
    matrix = np.ones((1, 1))
    for length, width, kernel in zip(shape, widths, kernels):
        matrix = np.kron(matrix, _spatial_line_matrix(length, width, kernel))
    return matrix


def _polar_matrix(
    shape: Tuple[int, ...], widths: Any, geometry: Dict[str, Any]
) -> np.ndarray:
    """Apply only the independent spatial polar reference to small basis arrays."""
    basis = np.eye(int(np.prod(shape)))
    return np.column_stack([
        _polar_reference(source.reshape(shape), widths, **geometry).ravel()
        for source in basis
    ])


def _physical_polar_case() -> Tuple[Tuple[int, ...], Tuple[float, ...], Dict[str, Any]]:
    """Describe a small volume whose physical angular and range passes do not commute."""
    return (3, 4, 3), (4.0, 8.0, 2.2), {
        "axis_types": ("elevation", "azimuth", "range"),
        "axis_coords": (
            np.array([-45.0, 0.0, 45.0]), np.arange(4) * 90.0, np.array([2.0, 3.0, 4.0]),
        ),
        "width_units": "physical",
        "kernel_type": ("boxcar_discrete", "gaussian", "boxcar_discrete"),
        "boundary": ("reflective", "periodic", "reflective"),
    }


def _support_reference(
    matrix: np.ndarray,
    mask: np.ndarray,
    exclude_center: bool = False,
    min_weight: float = 1e-3,
    min_valid_fraction: float = 0.0,
    weight_tolerance: float = WEIGHT_TOLERANCE,
) -> Dict[str, np.ndarray]:
    """Evaluate the support equations directly from the full spatial matrix."""
    valid_input = mask.ravel().astype(float)
    original = matrix @ valid_input
    diagonal = np.diag(matrix).copy()
    excluded = diagonal * valid_input if exclude_center else np.zeros_like(diagonal)
    available = matrix @ np.ones(mask.size) - (diagonal if exclude_center else 0.0)
    valid = original - excluded
    for values in (original, available, valid):
        values[np.abs(values) <= WEIGHT_TOLERANCE] = 0.0
    fraction = np.divide(valid, available, out=np.full_like(valid, np.nan),
                         where=available > weight_tolerance)
    supported = (
        (available > weight_tolerance) & (valid > max(min_weight, weight_tolerance))
        & (fraction + WEIGHT_TOLERANCE >= min_valid_fraction)
    )
    return {
        key: values.reshape(mask.shape) for key, values in {
            "original_valid_weight": original, "valid_weight": valid,
            "available_weight": available, "valid_fraction": fraction,
            "self_weight": diagonal, "excluded_weight": excluded, "supported": supported,
        }.items()
    }


def _assert_support_matches(
    actual: SupportDiagnostics, expected: Dict[str, np.ndarray]
) -> None:
    """Check every public support field, including its array shape and boolean gate."""
    assert isinstance(actual, SupportDiagnostics)
    assert isinstance(actual.geometry, Mapping)
    assert actual.geometry
    for field in SUPPORT_FIELDS:
        observed = np.asarray(getattr(actual, field))
        assert observed.shape == expected[field].shape, field
        if field == "supported":
            assert observed.dtype == np.dtype(bool)
            np.testing.assert_array_equal(observed, expected[field], err_msg=field)
        else:
            if field != "valid_fraction":
                assert np.all(np.isfinite(observed)), field
            np.testing.assert_allclose(observed, expected[field], atol=3e-13, rtol=3e-12,
                                       err_msg=field)


@pytest.mark.parametrize("exclude_center", [False, True])
@pytest.mark.parametrize("shape,widths,kernels", [
    ((7,), (3.7,), ("gaussian",)),
    ((2, 3), (17.0, 11.0), ("gaussian", "boxcar_discrete")),
    ((3, 4), (3.0, 5.0), ("boxcar_discrete", "boxcar_discrete")),
    ((2, 3, 2), (2.2, 7.0, 4.5), ("gaussian", "boxcar_discrete", "gaussian")),
])
def test_cartesian_support_matches_independent_spatial_operator(
    shape: Tuple[int, ...], widths: Tuple[float, ...], kernels: Tuple[str, ...],
    exclude_center: bool,
) -> None:
    """Mixed kernels, broad filters, and multidimensional diagonals share one operator."""
    mask = (np.arange(int(np.prod(shape))).reshape(shape) % 3) != 1
    before = mask.copy()
    expected = _support_reference(_cartesian_matrix(shape, widths, kernels), mask,
                                  exclude_center)
    actual = local_support(mask, widths, kernel_type=kernels, exclude_center=exclude_center)
    _assert_support_matches(actual, expected)
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
@pytest.mark.parametrize("kernel,width", [("gaussian", 18.0), ("boxcar_discrete", 13.0)])
def test_center_exclusion_subtracts_all_repeated_boundary_visits(
    boundary: str, kernel: str, width: float,
) -> None:
    """Removing a lone seed removes its full diagonal mass even for many wraps."""
    mask = np.array([True, False, False])
    matrix = _spatial_line_matrix(3, width, kernel, boundary)
    nominal_center = (1.0 / 13.0 if kernel == "boxcar_discrete" else
                      1.0 / np.exp(-0.5 * (np.arange(-100, 101)
                                          / (width / np.sqrt(12))) ** 2).sum())
    assert np.all(np.diag(matrix) > nominal_center + 0.1)
    actual = local_support(
        mask, width, coordinates="polar", axis_types=("linear",), width_units="grid",
        boundary=boundary, kernel_type=kernel, exclude_center=True,
    )
    _assert_support_matches(actual, _support_reference(matrix, mask, True))
    assert actual.valid_weight[0] == pytest.approx(0.0, abs=3e-13)
    assert not actual.supported[0]
    assert np.all(actual.supported[1:])
    assert np.all(actual.excluded_weight[1:] == 0)
    assert np.all(actual.self_weight[1:] > 0)


@pytest.mark.parametrize("exclude_center", [False, True])
def test_physical_polar_support_uses_complete_noncommuting_operator(
    exclude_center: bool,
) -> None:
    """Range/elevation-dependent angular widths are composed in canonical role order."""
    shape, widths, geometry = _physical_polar_case()
    matrix = _polar_matrix(shape, widths, geometry)
    reversed_matrix = _polar_matrix(shape, widths, dict(geometry, order=(2, 0, 1)))
    assert np.max(np.abs(matrix - reversed_matrix)) > 0.01
    mask = np.arange(int(np.prod(shape))).reshape(shape) % 4 != 0
    expected = _support_reference(matrix, mask, exclude_center)
    actual = local_support(mask, widths, coordinates="polar", exclude_center=exclude_center,
                           **geometry)
    _assert_support_matches(actual, expected)


def test_polar_support_and_diagonal_follow_permuted_array_axes() -> None:
    """Geometry metadata follows storage order while physical filter order is fixed."""
    shape, widths, geometry = _physical_polar_case()
    mask = np.arange(int(np.prod(shape))).reshape(shape) % 5 != 1
    expected = _support_reference(_polar_matrix(shape, widths, geometry), mask, True)
    permutation = (2, 0, 1)
    permuted = {
        key: value if isinstance(value, str) else tuple(value[i] for i in permutation)
        for key, value in geometry.items()
    }
    actual = local_support(
        mask.transpose(permutation), tuple(widths[i] for i in permutation),
        coordinates="polar", exclude_center=True, **permuted,
    )
    _assert_support_matches(actual, {
        key: values.transpose(permutation) for key, values in expected.items()
    })


def test_legacy_polar_support_retains_index_based_range_and_azimuth_units() -> None:
    """Legacy two-dimensional calls use r=j+1 and the supplied azimuth resolution."""
    shape, widths = (5, 3), (7.3, 3.0)
    reference_geometry = {
        "axis_types": ("azimuth", "range"),
        "axis_coords": (np.arange(5) * 72.0, np.arange(1.0, 4.0)),
        "width_units": ("physical", "grid"),
        "kernel_type": ("gaussian", "boxcar_discrete"),
        "boundary": ("periodic", "reflective"),
    }
    mask = np.arange(15).reshape(shape) % 3 != 0
    expected = _support_reference(_polar_matrix(shape, widths, reference_geometry), mask, True)
    actual = local_support(
        mask, widths, coordinates="polar", az_res_deg=72.0, az_boundary="periodic",
        kernel_type=reference_geometry["kernel_type"], exclude_center=True,
    )
    _assert_support_matches(actual, expected)


@pytest.mark.parametrize("shape", [(1,), (1, 1), (3,)])
def test_identity_and_singleton_exclusion_leave_no_external_support(
    shape: Tuple[int, ...],
) -> None:
    """A positive input cannot become its own fallback after diagonal exclusion."""
    mask = np.ones(shape, dtype=bool)
    width = 17.0 if mask.size == 1 else 0.1
    actual = local_support(mask, width, kernel_type="boxcar_discrete", exclude_center=True,
                           min_weight=0.0)
    _assert_support_matches(actual, _support_reference(np.eye(mask.size), mask, True, 0.0))


def test_empty_filter_selection_is_an_identity_support_operator() -> None:
    """No selected axes means no borrowing, regardless of requested widths."""
    mask = np.array([[True, False], [False, True]])
    actual = local_support(
        mask, 999.0, coordinates="polar", axis_types=("linear", "linear"),
        width_units="grid", filter_axes=(), exclude_center=True,
    )
    _assert_support_matches(actual, _support_reference(np.eye(4), mask, True))


def test_batch_and_unfiltered_axes_are_independent_support_domains() -> None:
    """Entirely empty lines do not acquire support from other batches or times."""
    mask = np.ones((2, 3, 4), dtype=bool)
    mask[1] = False
    mask[0, 1] = False
    mask[0, 2, 1] = False
    matrix = np.kron(np.eye(6), _spatial_line_matrix(4, 5.0))
    actual = local_support(
        mask, (999.0, 999.0, 5.0), coordinates="polar",
        axis_types=("batch", "time", "linear"), filter_axes=(-1,),
        width_units="grid", exclude_center=True,
    )
    _assert_support_matches(actual, _support_reference(matrix, mask, True))
    assert not np.any(actual.supported[1])
    assert not np.any(actual.supported[0, 1])


@pytest.mark.parametrize("exclude_center", [False, True])
@pytest.mark.parametrize("min_weight,min_fraction", [(0.0, 0.0), (0.4, 0.0), (0.0, 0.75)])
def test_support_thresholds_apply_to_actual_mass_and_exclusion_adjusted_fraction(
    exclude_center: bool, min_weight: float, min_fraction: float,
) -> None:
    """The valid fraction is divided by K1-Kii after exclusion, rather than by one."""
    mask = np.array([True, True, False])
    matrix = _spatial_line_matrix(3, 3.0, "boxcar_discrete", "periodic")
    expected = _support_reference(matrix, mask, exclude_center, min_weight, min_fraction)
    actual = local_support(
        mask, 3.0, coordinates="polar", axis_types=("linear",), width_units="grid",
        boundary="periodic", kernel_type="boxcar_discrete", exclude_center=exclude_center,
        min_weight=min_weight, min_valid_fraction=min_fraction,
    )
    _assert_support_matches(actual, expected)


def test_support_weight_floor_is_strict_and_not_additive_regularization() -> None:
    """Mass exactly equal to the floor is unsupported, including an identity row."""
    actual = local_support(np.ones(3, dtype=bool), 0.1, kernel_type="boxcar_discrete",
                           min_weight=1.0)
    assert not np.any(actual.supported)
    np.testing.assert_allclose(actual.valid_weight, 1.0, atol=3e-13)


@pytest.mark.parametrize("all_valid", [False, True])
def test_empty_and_complete_masks_have_roundoff_safe_support(all_valid: bool) -> None:
    """Roundoff cannot fabricate support in an empty mask or reduce a full fraction."""
    mask = np.full(9, all_valid)
    expected = _support_reference(_spatial_line_matrix(9, 3.0), mask, True, 0.0, 1.0)
    actual = local_support(mask, 3.0, exclude_center=True, min_weight=0.0,
                           min_valid_fraction=1.0)
    _assert_support_matches(actual, expected)


def test_gaussian_support_retains_resolvable_tails_beyond_four_sigma() -> None:
    """A remote valid observation remains support without an arbitrary finite window."""
    mask = np.zeros(17, dtype=bool)
    mask[13] = True
    matrix = _spatial_line_matrix(17, np.sqrt(12.0))
    actual = local_support(mask, np.sqrt(12.0), min_weight=0.0)
    _assert_support_matches(actual, _support_reference(matrix, mask, min_weight=0.0))
    assert actual.valid_weight[8] > 1e-6
    assert actual.supported[8]


def test_numerical_tolerance_gates_support_without_relabeling_actual_weight() -> None:
    """A numerical floor is recorded separately from valid mass and scientific thresholds."""
    mask = np.zeros(17, dtype=bool)
    mask[13] = True
    tolerance = 1e-5
    matrix = _spatial_line_matrix(17, np.sqrt(12.0))
    actual = local_support(mask, np.sqrt(12.0), min_weight=0.0, weight_tolerance=tolerance)
    _assert_support_matches(actual, _support_reference(
        matrix, mask, min_weight=0.0, weight_tolerance=tolerance,
    ))
    assert 0 < actual.valid_weight[8] < tolerance
    assert not actual.supported[8]
    assert actual.weight_tolerance == tolerance
    assert actual.min_weight == 0.0
    assert actual.min_valid_fraction == 0.0


def test_numeric_support_masks_exclude_nonfinite_and_zero_entries() -> None:
    """Numeric validity masks select finite nonzero values, without modifying their input."""
    mask = np.array([0.0, 1.0, np.nan, -2.0, np.inf, 0.0])
    before = mask.copy()
    valid = np.array([False, True, False, True, False, False])
    expected = _support_reference(_spatial_line_matrix(6, 5.0), valid, True)
    _assert_support_matches(local_support(mask, 5.0, exclude_center=True), expected)
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("kernel", ["boxcar", "gaussian_spectral"])
def test_support_rejects_signed_active_kernels_even_for_empty_masks(kernel: str) -> None:
    """Positive statistical support is a kernel contract, not an input-dependent test."""
    with pytest.raises(ValueError):
        local_support(np.zeros(5, dtype=bool), 2.3, kernel_type=kernel)


def test_support_allows_signed_kernel_names_on_unfiltered_axes() -> None:
    """Inactive axes are isolated domains and contribute no statistical weights."""
    mask = np.ones((2, 4, 2), dtype=bool)
    mask[1, 1, 1] = False
    matrix = np.kron(np.eye(2), np.kron(_spatial_line_matrix(4, 3.0), np.eye(2)))
    actual = local_support(
        mask, 3.0, coordinates="polar", axis_types=("batch", "linear", "time"),
        filter_axes=(1,), width_units="grid",
        kernel_type=("boxcar", "gaussian", "gaussian_spectral"),
    )
    _assert_support_matches(actual, _support_reference(matrix, mask))


@pytest.mark.parametrize("kwargs", [
    {"min_weight": -1.0}, {"min_weight": np.nan}, {"min_weight": np.inf},
    {"min_valid_fraction": -0.1}, {"min_valid_fraction": 1.1},
    {"min_valid_fraction": np.nan}, {"weight_tolerance": -1.0},
    {"weight_tolerance": np.nan}, {"weight_tolerance": np.inf},
    {"coordinates": "spherical"}, {"width": 0.0}, {"width": -1.0},
    {"width": np.nan}, {"unrecognized_geometry_option": True},
])
def test_support_rejects_invalid_public_controls(kwargs: Dict[str, Any]) -> None:
    """Invalid controls must be rejected before an all-empty-mask early return."""
    arguments = {"width": 3.0, **kwargs}
    with pytest.raises((ValueError, TypeError)):
        local_support(np.zeros(4, dtype=bool), **arguments)


@pytest.mark.parametrize("coords", [np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 4.0])])
def test_support_rejects_descending_or_irregular_physical_coordinates(coords: np.ndarray) -> None:
    """Adapters must regularize/reverse coordinates before calling the toolkit."""
    with pytest.raises(ValueError):
        local_support(np.ones(3, dtype=bool), 3.0, coordinates="polar",
                      axis_types=("linear",), axis_coords=(coords,))


def test_support_rejects_incomplete_physical_periodic_azimuth() -> None:
    """A periodic physical angular grid must span a full circle without duplicates."""
    with pytest.raises(ValueError):
        local_support(
            np.zeros((3, 2), dtype=bool), 2.0, coordinates="polar",
            axis_types=("azimuth", "range"),
            axis_coords=(np.array([0.0, 20.0, 40.0]), np.array([2.0, 3.0])),
            boundary=("periodic", "reflective"),
        )

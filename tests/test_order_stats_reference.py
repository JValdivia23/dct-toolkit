"""Independent weighted-distribution acceptance tests for local order statistics.

Every expected answer comes from spatial operator rows and a sorted weighted
CDF. In particular, no runtime smoother, transfer function, histogram, rank
filter, or private implementation helper supplies an expected distribution.
Tiny matrices keep the independent oracle cheap; larger histogram examples
remain one-dimensional. No wall-clock performance threshold is imposed.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pytest

from dct_toolkit.order_stats import OrderStatisticResult, local_mad, local_median, local_quantile
from test_support_reference import (
    SUPPORT_FIELDS, WEIGHT_TOLERANCE, _assert_support_matches, _cartesian_matrix,
    _physical_polar_case, _polar_matrix, _spatial_line_matrix, _support_reference,
)


def _weighted_inverse_cdf(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Return lower weighted inverses, with explicit allowance for transform roundoff."""
    positive = weights > WEIGHT_TOLERANCE
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    tolerance = WEIGHT_TOLERANCE * max(1.0, float(cumulative[-1]))
    indices = np.searchsorted(cumulative + tolerance, quantiles * cumulative[-1], side="left")
    indices[quantiles == 0.0] = 0
    indices[quantiles == 1.0] = len(values) - 1
    return values[np.minimum(indices, len(values) - 1)]


def _quantile_reference(
    matrix: np.ndarray,
    data: np.ndarray,
    q: Any,
    mask: Optional[np.ndarray] = None,
    *,
    exclude_center: bool = False,
    restore_input_nan: bool = False,
    min_weight: float = 1e-3,
    min_valid_fraction: float = 0.0,
) -> np.ndarray:
    """Sort original valid observations independently for each complete operator row."""
    valid = np.isfinite(data) if mask is None else np.isfinite(data) & mask
    support = _support_reference(matrix, valid, exclude_center, min_weight, min_valid_fraction)
    quantiles = np.atleast_1d(q).astype(float)
    output = np.full((len(quantiles), data.size), np.nan)
    for target in np.flatnonzero(support["supported"].ravel()):
        if restore_input_nan and not valid.ravel()[target]:
            continue
        weights = np.where(valid.ravel(), matrix[target], 0.0)
        if exclude_center:
            weights[target] = 0.0
        output[:, target] = _weighted_inverse_cdf(data.ravel(), weights, quantiles)
    output = output.reshape((len(quantiles),) + data.shape)
    return output[0] if np.ndim(q) == 0 else output


def _mad_reference(
    matrix: np.ndarray,
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    exclude_center: bool = False,
    restore_input_nan: bool = False,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Center every source deviation on the target's own weighted neighborhood median."""
    medians = _quantile_reference(matrix, data, 0.5, mask, exclude_center=exclude_center)
    valid = np.isfinite(data) if mask is None else np.isfinite(data) & mask
    output = np.full(data.size, np.nan)
    for target in np.flatnonzero(np.isfinite(medians.ravel())):
        weights = np.where(valid.ravel(), matrix[target], 0.0)
        if exclude_center:
            weights[target] = 0.0
        deviations = np.abs(data.ravel() - medians.ravel()[target])
        output[target] = scale * _weighted_inverse_cdf(deviations, weights, np.array([0.5]))[0]
    output = output.reshape(data.shape)
    if restore_input_nan:
        output = np.where(valid, output, np.nan)
        medians = np.where(valid, medians, np.nan)
    return output, medians


def _assert_result_contract(
    result: OrderStatisticResult, expected: np.ndarray, method: str,
) -> None:
    """Check the public container and exact-valued outputs without using runtime internals."""
    assert isinstance(result, OrderStatisticResult)
    assert result.method == method
    assert np.asarray(result.resolved).dtype == np.dtype(bool)
    np.testing.assert_array_equal(result.resolved, np.isfinite(result.values))
    assert isinstance(result.bin_count, (int, np.integer))
    assert np.asarray(result.values).shape == expected.shape
    np.testing.assert_allclose(result.values, expected, atol=2e-12, rtol=2e-13)
    bounds = np.asarray(result.error_bound)
    np.broadcast_to(bounds, expected.shape)
    assert np.all(bounds[np.isfinite(bounds)] >= 0)
    if method in ("exact", "spatial"):
        assert result.bin_count == 0
        assert np.all(bounds[np.isfinite(bounds)] == 0)


@pytest.mark.parametrize("method", ["exact", "histogram"])
@pytest.mark.parametrize("exclude_center", [False, True])
def test_quantiles_match_independent_weighted_cdfs_for_mixed_cartesian_kernels(
    method: str, exclude_center: bool,
) -> None:
    """Low-cardinality histograms and exact rows implement the same lower inverse CDF."""
    data = np.array([[4.0, -2.0, 4.0, np.nan], [1.0, 8.0, np.inf, -2.0],
                     [8.0, 1.0, 4.0, -2.0]])
    mask = np.ones(data.shape, dtype=bool)
    mask[1, 1] = False
    original, mask_before = data.copy(), mask.copy()
    q = np.array([1.0, 0.0, 0.5, 0.2, 0.8, 0.5])
    widths, kernels = (4.3, 5.0), ("gaussian", "boxcar_discrete")
    matrix = _cartesian_matrix(data.shape, widths, kernels)
    expected = _quantile_reference(matrix, data, q, mask, exclude_center=exclude_center)
    result = local_quantile(
        data, q, widths, mask=mask, restore_input_nan=False, method=method, bins=8,
        kernel_type=kernels, exclude_center=exclude_center, return_info=True,
    )
    _assert_result_contract(result, expected, method)
    _assert_support_matches(result.support, _support_reference(
        matrix, np.isfinite(data) & mask, exclude_center,
    ))
    assert np.nanmax(result.error_bound) <= 2e-12
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(mask, mask_before)
    np.testing.assert_array_equal(q, [1.0, 0.0, 0.5, 0.2, 0.8, 0.5])


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
@pytest.mark.parametrize("q", [0.0, 0.5, 1.0, np.array([0.5]), np.array([0.9, 0.1, 0.5])])
def test_scalar_and_vector_quantiles_preserve_documented_shape(method: str, q: Any) -> None:
    """Only a one-dimensional quantile vector introduces a leading result axis."""
    data = np.arange(12).reshape(3, 4)
    matrix = _cartesian_matrix(data.shape, (3.0, 5.0), ("boxcar_discrete",) * 2)
    expected = _quantile_reference(matrix, data, q)
    result = local_quantile(data, q, (3.0, 5.0), method=method, bins=16,
                            kernel_type="boxcar_discrete")
    assert result.shape == expected.shape
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, expected, atol=2e-12)


@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_lower_cdf_ties_endpoints_and_center_exclusion_are_roundoff_stable(method: str) -> None:
    """An equal-weight even neighborhood returns its lower middle value, without averaging."""
    data = np.array([2.0, -5.0, 10.0])
    q = np.array([0.0, 0.5 - 1e-8, 0.5, 0.5 + 1e-8, 1.0])
    matrix = _spatial_line_matrix(3, 3.0, "boxcar_discrete", "periodic")
    expected = _quantile_reference(matrix, data, q, exclude_center=True)
    assert expected[2, 0] == -5.0
    assert expected[3, 0] == 10.0
    result = local_quantile(
        data, q, 3.0, coordinates="polar", axis_types=("linear",), width_units="grid",
        boundary="periodic", kernel_type="boxcar_discrete", exclude_center=True,
        method=method, bins=4,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
def test_repeated_values_use_total_tied_mass_at_lower_cdf_boundary(method: str) -> None:
    """Several equal observations represent one CDF jump, including a floating-point tie."""
    data = np.array([4.0, 4.0, 9.0])
    q = np.array([0.0, 2.0 / 3.0, 2.0 / 3.0 + 1e-8, 1.0])
    matrix = _spatial_line_matrix(3, 3.0, "boxcar_discrete")
    expected = _quantile_reference(matrix, data, q)
    result = local_quantile(data, q, 3.0, kernel_type="boxcar_discrete", method=method, bins=3)
    np.testing.assert_array_equal(result, expected)
    assert result[1, 1] == 4.0 and result[2, 1] == 9.0


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
def test_identity_quantile_endpoints_ignore_spurious_transform_mass(method: str) -> None:
    """q=0 and q=1 select only positive spatial support, not remote roundoff artifacts."""
    data = np.array([-1000.0, 7.0, 2000.0, -1.0, 3.0])
    result = local_quantile(data, [0.0, 1.0], 0.1, kernel_type="boxcar_discrete",
                            method=method, bins=8, min_weight=0.0)
    np.testing.assert_array_equal(result, np.stack([data, data]))


@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_restore_input_nan_only_changes_output_validity_not_distribution(method: str) -> None:
    """NaNs, infinities, and masked finite outliers never contribute observation mass."""
    data = np.array([1.0, np.nan, 1e8, 4.0, np.inf, -np.inf, 5.0])
    mask = np.ones(7, dtype=bool)
    mask[2] = False
    valid = np.isfinite(data) & mask
    matrix = _spatial_line_matrix(7, 6.0)
    unmasked = local_quantile(data, 0.5, 6.0, mask=mask, method=method, bins=8,
                              restore_input_nan=False, return_info=True)
    restored = local_quantile(data, 0.5, 6.0, mask=mask, method=method, bins=8,
                              return_info=True)
    expected = _quantile_reference(matrix, data, 0.5, mask)
    np.testing.assert_array_equal(unmasked.values, expected)
    np.testing.assert_array_equal(restored.values, np.where(valid, expected, np.nan))
    for field in SUPPORT_FIELDS:
        np.testing.assert_allclose(getattr(unmasked.support, field),
                                   getattr(restored.support, field), equal_nan=True)
    assert np.all(unmasked.values < 10)


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
def test_unsupported_local_distributions_remain_nan_without_prefilling(method: str) -> None:
    """Distant valid seeds do not repair empty or insufficient local mass."""
    data = np.array([7.0, np.nan, np.nan, np.nan, np.nan, 1e6])
    matrix = _spatial_line_matrix(6, 3.0, "boxcar_discrete")
    expected = _quantile_reference(matrix, data, 0.5, min_weight=0.5)
    result = local_quantile(
        data, 0.5, 3.0, kernel_type="boxcar_discrete", restore_input_nan=False,
        min_weight=0.5, method=method, bins=8, return_info=True,
    )
    _assert_result_contract(result, expected, method)
    _assert_support_matches(result.support, _support_reference(
        matrix, np.isfinite(data), min_weight=0.5,
    ))
    assert np.all(np.isnan(result.values[1:-1]))


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
@pytest.mark.parametrize("shape", [(1,), (3,), (1, 1)])
def test_identity_exclusion_never_falls_back_to_tested_observation(
    method: str, shape: Tuple[int, ...],
) -> None:
    """Excluding the sole source leaves an unsupported quantile, median, and MAD."""
    data = np.full(shape, 7.0)
    for function in (local_quantile, local_median, local_mad):
        args = (data, 0.5, 0.1) if function is local_quantile else (data, 0.1)
        result = function(*args, method=method, kernel_type="boxcar_discrete",
                          exclude_center=True, min_weight=0.0, return_info=True)
        assert np.all(np.isnan(result.values))
        assert not np.any(result.support.supported)
        assert np.all(np.isnan(result.support.valid_fraction))


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
def test_all_missing_quantile_and_mad_return_unsupported_diagnostics(method: str) -> None:
    """An empty distribution is unresolved even when the scientific weight floor is zero."""
    data = np.full((2, 3), np.nan)
    matrix = _cartesian_matrix(data.shape, (3.0, 5.0), ("boxcar_discrete",) * 2)
    for function in (local_quantile, local_mad):
        q = np.array([0.0, 0.5, 1.0])
        args = (data, q, (3.0, 5.0)) if function is local_quantile else (data, (3.0, 5.0))
        result = function(
            *args, method=method, kernel_type="boxcar_discrete", restore_input_nan=False,
            min_weight=0.0, return_info=True,
        )
        expected_shape = (3,) + data.shape if function is local_quantile else data.shape
        _assert_result_contract(result, np.full(expected_shape, np.nan), method)
        _assert_support_matches(result.support, _support_reference(
            matrix, np.zeros(data.shape, dtype=bool), min_weight=0.0,
        ))
        if function is local_mad:
            np.testing.assert_array_equal(result.median, np.full(data.shape, np.nan))


@pytest.mark.parametrize("function", [local_quantile, local_median, local_mad])
@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
def test_order_statistics_apply_fraction_gate_after_center_exclusion(
    function: Callable[..., Any], method: str,
) -> None:
    """Scientific support uses valid/available mass and never triggers a filling fallback."""
    data = np.array([1.0, 2.0, np.nan, 4.0, np.nan, np.nan, 7.0])
    matrix = _spatial_line_matrix(7, 5.0, "boxcar_discrete")
    support = _support_reference(matrix, np.isfinite(data), True, 0.1, 0.7)
    assert np.any(support["supported"]) and np.any(~support["supported"])
    assert support["supported"][1]
    assert support["valid_weight"][1] < 0.7 < support["valid_fraction"][1]
    if function is local_mad:
        expected, median = _mad_reference(matrix, data, exclude_center=True)
    else:
        expected = _quantile_reference(matrix, data, 0.5, exclude_center=True)
    expected = np.where(support["supported"], expected, np.nan)
    args = (data, 0.5, 5.0) if function is local_quantile else (data, 5.0)
    result = function(
        *args, method=method, bins=8, kernel_type="boxcar_discrete", exclude_center=True,
        min_weight=0.1, min_valid_fraction=0.7, restore_input_nan=False, return_info=True,
    )
    _assert_result_contract(result, expected, method)
    _assert_support_matches(result.support, support)
    if function is local_mad:
        np.testing.assert_array_equal(result.median,
                                      np.where(support["supported"], median, np.nan))


@pytest.mark.parametrize("method", ["exact", "histogram"])
@pytest.mark.parametrize("permutation", [(0, 1, 2), (2, 0, 1)])
def test_polar_quantiles_use_full_noncommuting_operator_in_any_array_order(
    method: str, permutation: Tuple[int, ...],
) -> None:
    """Geometry ordering precedes one weighted CDF, rather than sequential axis quantiles."""
    shape, widths, geometry = _physical_polar_case()
    data = np.random.default_rng(1801).permutation(36).astype(float).reshape(shape)
    data[0, 0, 0] = np.nan
    mask = np.ones(shape, dtype=bool)
    mask[1, 2, 2] = False
    q = np.array([0.0, 0.2, 0.5, 0.85, 1.0])
    matrix = _polar_matrix(shape, widths, geometry)
    expected = _quantile_reference(matrix, data, q, mask, exclude_center=True)
    reverse = _polar_matrix(shape, widths, dict(geometry, order=(2, 0, 1)))
    wrong = _quantile_reference(reverse, data, q, mask, exclude_center=True)
    assert np.nanmax(np.abs(expected - wrong)) >= 2.0
    permuted = {
        key: value if isinstance(value, str) else tuple(value[i] for i in permutation)
        for key, value in geometry.items()
    }
    result = local_quantile(
        data.transpose(permutation), q, tuple(widths[i] for i in permutation),
        coordinates="polar", mask=mask.transpose(permutation), restore_input_nan=False,
        method=method, bins=64, exclude_center=True, return_info=True, **permuted,
    )
    quantile_permutation = (0,) + tuple(axis + 1 for axis in permutation)
    _assert_result_contract(result, expected.transpose(quantile_permutation), method)
    _assert_support_matches(result.support, {
        key: value.transpose(permutation) for key, value in _support_reference(
            matrix, np.isfinite(data) & mask, True,
        ).items()
    })


@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_legacy_polar_quantiles_match_independent_index_geometry(method: str) -> None:
    """The legacy azimuth/range interface retains its original width interpretation."""
    data = np.random.default_rng(1802).permutation(15).astype(float).reshape(5, 3)
    data[1, 1] = np.nan
    widths = (7.3, 3.0)
    geometry = {
        "axis_types": ("azimuth", "range"),
        "axis_coords": (np.arange(5) * 72.0, np.arange(1.0, 4.0)),
        "width_units": ("physical", "grid"), "kernel_type": "boxcar_discrete",
        "boundary": ("periodic", "reflective"),
    }
    matrix = _polar_matrix(data.shape, widths, geometry)
    expected = _quantile_reference(matrix, data, [0.1, 0.5, 0.9], exclude_center=True)
    actual = local_quantile(
        data, [0.1, 0.5, 0.9], widths, coordinates="polar", az_res_deg=72.0,
        az_boundary="periodic", kernel_type="boxcar_discrete", exclude_center=True,
        restore_input_nan=False, method=method, bins=16,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_histogram_distinct_values_are_resolved_within_each_isolated_domain(method: str) -> None:
    """Distant batch offsets cannot widen bins or contaminate an entirely missing batch."""
    data = np.array([[1.0, 2.0, 1.0, 4.0, 2.0],
                     [1e9, 1e9 + 2, 1e9 + 4, 1e9, 1e9 + 2], [np.nan] * 5])
    matrix = np.kron(np.eye(3), _spatial_line_matrix(5, 4.0))
    q = np.array([0.0, 0.5, 1.0])
    result = local_quantile(
        data, q, (999.0, 4.0), coordinates="polar", axis_types=("batch", "linear"),
        width_units="grid", restore_input_nan=False, method=method, bins=3, return_info=True,
    )
    _assert_result_contract(result, _quantile_reference(matrix, data, q), method)
    assert np.all(np.isnan(result.values[:, 2]))
    assert not np.any(result.support.supported[2])
    assert np.nanmax(result.error_bound) <= 2e-12


@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_unfiltered_axes_keep_independent_value_domains_and_allow_inactive_signed_kernels(
    method: str,
) -> None:
    """Only active axes define weights; even histogram value bins stay within each domain."""
    data = np.arange(16, dtype=float).reshape(2, 4, 2)
    data[1, :, 0] = np.nan
    matrix = np.kron(np.eye(2), np.kron(_spatial_line_matrix(4, 3.0), np.eye(2)))
    result = local_median(
        data, 3.0, coordinates="polar", axis_types=("batch", "linear", "time"),
        filter_axes=(1,), width_units="grid", method=method, bins=4,
        kernel_type=("boxcar", "gaussian", "gaussian_spectral"), restore_input_nan=False,
    )
    np.testing.assert_array_equal(result, _quantile_reference(matrix, data, 0.5))


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
@pytest.mark.parametrize("exclude_center", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1.4826])
def test_mad_on_a_ramp_is_centered_on_each_targets_own_median(
    method: str, exclude_center: bool, scale: float,
) -> None:
    """Target-centered MAD stays positive where a two-median residual shortcut gives zero."""
    data = np.arange(9, dtype=float)
    matrix = _spatial_line_matrix(9, 5.0, "boxcar_discrete")
    expected, median = _mad_reference(matrix, data, exclude_center=exclude_center, scale=scale)
    if not exclude_center:
        residual = np.abs(data - median)
        shortcut = scale * _quantile_reference(matrix, residual, 0.5)
        assert np.max(np.abs(expected - shortcut)) >= scale
    result = local_mad(
        data, 5.0, method=method, bins=16, kernel_type="boxcar_discrete", scale=scale,
        exclude_center=exclude_center, return_info=True,
    )
    _assert_result_contract(result, expected, method)
    np.testing.assert_array_equal(result.median, median)
    np.testing.assert_array_equal(local_median(
        data, 5.0, method=method, bins=16, kernel_type="boxcar_discrete",
        exclude_center=exclude_center,
    ), median)


@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_masked_physical_polar_mad_matches_target_distribution(method: str) -> None:
    """MAD reuses the complete polar target distribution, including exclusion and masks."""
    shape, widths, geometry = _physical_polar_case()
    data = np.random.default_rng(1803).permutation(36).astype(float).reshape(shape)
    data[0, 1, 0] = np.nan
    mask = np.arange(36).reshape(shape) % 5 != 0
    matrix = _polar_matrix(shape, widths, geometry)
    expected, median = _mad_reference(matrix, data, mask, exclude_center=True, scale=1.4826)
    result = local_mad(
        data, widths, coordinates="polar", mask=mask, restore_input_nan=False,
        exclude_center=True, scale=1.4826, method=method, bins=64, return_info=True, **geometry,
    )
    _assert_result_contract(result, expected, method)
    np.testing.assert_allclose(result.median, median, atol=2e-12)


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
def test_constant_masked_data_have_zero_unscaled_mad_and_no_hidden_floor(method: str) -> None:
    """Missing observations and explicit scaling do not manufacture nonzero spread."""
    data = np.full((3, 4), 17.5)
    data[1, 1] = np.nan
    mask = np.ones(data.shape, dtype=bool)
    mask[0, 3] = False
    matrix = _cartesian_matrix(data.shape, (3.0, 5.0), ("boxcar_discrete",) * 2)
    result = local_mad(
        data, (3.0, 5.0), mask=mask, restore_input_nan=False, method=method, bins=3,
        kernel_type="boxcar_discrete", exclude_center=True, return_info=True,
    )
    expected, median = _mad_reference(matrix, data, mask, exclude_center=True)
    _assert_result_contract(result, expected, method)
    np.testing.assert_array_equal(result.median, median)
    np.testing.assert_array_equal(result.values, np.zeros(data.shape))


def test_histogram_quantization_converges_with_reported_half_bin_error() -> None:
    """Uniform-bin estimates obey the value error guarantee against the exact estimator."""
    data = np.linspace(0.0, 1.0, 65) ** 1.3
    q = np.array([0.0, 0.23, 0.5, 0.83, 1.0])
    matrix = _spatial_line_matrix(65, 11.0)
    expected = _quantile_reference(matrix, data, q)
    errors, bounds = [], []
    for bins in (4, 8, 16, 32):
        result = local_quantile(data, q, 11.0, method="histogram", bins=bins,
                                value_range=(0.0, 1.0), return_info=True)
        assert result.method == "histogram"
        error = np.abs(result.values - expected)
        bound = np.broadcast_to(result.error_bound, error.shape)
        assert np.all(error <= bound + 3e-12)
        assert np.nanmax(bound) <= 0.5 / bins + 3e-12
        assert 0 < np.max(np.asarray(result.bin_count)) <= bins
        errors.append(float(np.max(error)))
        bounds.append(float(np.nanmax(bound)))
    assert np.all(np.diff(bounds) < 0)
    assert errors[-1] < errors[0]


def test_histogram_mad_obeys_whole_bin_error_including_explicit_scale() -> None:
    """Both median centering and deviation quantization enter the MAD error budget."""
    data = np.linspace(-2.0, 3.0, 49) ** 3
    matrix = _spatial_line_matrix(49, 9.0)
    expected, median = _mad_reference(matrix, data, scale=1.4826)
    value_span = float(np.ptp(data))
    bounds = []
    for bins in (4, 8, 16, 32):
        result = local_mad(data, 9.0, method="histogram", bins=bins, scale=1.4826,
                           return_info=True)
        error = np.abs(result.values - expected)
        bound = np.broadcast_to(result.error_bound, error.shape)
        assert np.all(error <= bound + 3e-11)
        assert np.nanmax(bound) <= 1.4826 * value_span / bins + 3e-11
        assert np.max(np.abs(result.median - median)) <= value_span / (2 * bins) + 3e-11
        bounds.append(float(np.nanmax(bound)))
    assert np.all(np.diff(bounds) < 0)


@pytest.mark.parametrize("batch_size,max_memory_mb", [(1, 0.01), (3, 0.05), (8, 64.0)])
def test_histogram_transform_batching_does_not_change_estimates_or_error_bounds(
    batch_size: int, max_memory_mb: float,
) -> None:
    """Different bounded transform batches represent the same weighted histogram."""
    data = np.random.default_rng(1804).normal(size=(4, 5))
    data[1, 2] = np.nan
    kwargs = dict(method="histogram", bins=7, restore_input_nan=False, return_info=True)
    reference = local_quantile(data, [0.25, 0.5, 0.75], (3.0, 5.0), **kwargs)
    result = local_quantile(
        data, [0.25, 0.5, 0.75], (3.0, 5.0), batch_size=batch_size,
        max_memory_mb=max_memory_mb, **kwargs,
    )
    np.testing.assert_allclose(result.values, reference.values, atol=2e-12)
    np.testing.assert_array_equal(result.error_bound, reference.error_bound)


def test_histogram_mad_is_invariant_under_small_working_memory_budget() -> None:
    """Bounded histogram storage preserves the target-centered estimator and its bound."""
    data = np.linspace(-1.0, 2.0, 17) ** 3
    kwargs = dict(method="histogram", bins=11, exclude_center=True, return_info=True)
    expected = local_mad(data, 8.0, **kwargs)
    result = local_mad(data, 8.0, batch_size=1, max_memory_mb=0.002, **kwargs)
    np.testing.assert_array_equal(result.values, expected.values)
    np.testing.assert_array_equal(result.median, expected.median)
    np.testing.assert_array_equal(result.error_bound, expected.error_bound)


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
@pytest.mark.parametrize("masked", [False, True])
def test_spatial_boxcar_quantiles_and_mad_match_broad_reflected_operator(
    method: str, masked: bool,
) -> None:
    """Compiled rank and masked/excluded spatial paths preserve aggregated reflection weights."""
    data = np.array([[9.0, -2.0, 4.0], [100.0, 7.0, 1.0]])
    mask = np.ones(data.shape, dtype=bool)
    if masked:
        data[0, 1] = np.nan
        mask[1, 0] = False
    matrix = _cartesian_matrix(data.shape, (7.0, 9.0), ("boxcar_discrete",) * 2)
    q = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    kwargs = dict(kernel_type="boxcar_discrete", method=method, bins=8, mask=mask,
                  restore_input_nan=False, exclude_center=masked, batch_size=2)
    expected = _quantile_reference(matrix, data, q, mask, exclude_center=masked)
    np.testing.assert_array_equal(local_quantile(data, q, (7.0, 9.0), **kwargs), expected)
    mad, median = _mad_reference(matrix, data, mask, exclude_center=masked)
    result = local_mad(data, (7.0, 9.0), return_info=True, **kwargs)
    np.testing.assert_array_equal(result.values, mad)
    np.testing.assert_array_equal(result.median, median)


def test_histogram_value_range_rejects_outliers_instead_of_clipping_mass() -> None:
    """Finite valid observations outside a declared histogram range are an explicit error."""
    for data in (np.array([-1.0, 1.0, 2.0]), np.array([0.0, 1.0, 3.0])):
        with pytest.raises(ValueError):
            local_quantile(data, 0.5, 3.0, value_range=(0.0, 2.0))
        with pytest.raises(ValueError):
            local_mad(data, 3.0, value_range=(0.0, 2.0))


def test_histogram_value_range_ignores_masked_or_nonfinite_observations() -> None:
    """An excluded outlier does not affect domain values, range validation, or estimates."""
    data = np.array([1.0, 1e8, np.nan, np.inf])
    mask = np.array([True, False, True, True])
    result = local_quantile(data, [0.0, 0.5, 1.0], 9.0, mask=mask, restore_input_nan=False,
                            bins=2, value_range=(0.0, 2.0))
    np.testing.assert_array_equal(result, np.ones((3, 4)))


@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_full_gaussian_tails_define_remote_quantiles_without_prefilling(method: str) -> None:
    """A sole observation five sigma away is still the actual weighted distribution."""
    data = np.full(17, np.nan)
    data[13] = 23.0
    matrix = _spatial_line_matrix(17, np.sqrt(12.0))
    expected = _quantile_reference(matrix, data, [0.0, 0.5, 1.0], min_weight=0.0)
    result = local_quantile(
        data, [0.0, 0.5, 1.0], np.sqrt(12.0), method=method, restore_input_nan=False,
        min_weight=0.0, return_info=True,
    )
    _assert_result_contract(result, expected, method)
    assert result.support.valid_weight[8] > 1e-6
    np.testing.assert_array_equal(result.values[:, 8], [23.0, 23.0, 23.0])


@pytest.mark.parametrize("method", ["exact", "histogram"])
def test_quantile_numerical_weight_tolerance_does_not_fabricate_support(method: str) -> None:
    """A caller's numerical floor can suppress a tiny real tail without filling its target."""
    data = np.full(17, np.nan)
    data[13] = 23.0
    matrix = _spatial_line_matrix(17, np.sqrt(12.0))
    support = _support_reference(matrix, np.isfinite(data), min_weight=0.0,
                                 weight_tolerance=1e-5)
    expected = np.where(support["supported"], 23.0, np.nan)
    result = local_quantile(
        data, 0.5, np.sqrt(12.0), method=method, min_weight=0.0, weight_tolerance=1e-5,
        restore_input_nan=False, return_info=True,
    )
    _assert_result_contract(result, expected, method)
    _assert_support_matches(result.support, support)
    assert np.isnan(result.values[8])


def test_spatial_method_rejects_unsupported_kernel_or_coordinate_system() -> None:
    """The exact spatial backend advertises only Cartesian discrete-boxcar geometry."""
    with pytest.raises(ValueError):
        local_quantile(np.arange(5.0), 0.5, 3.0, method="spatial", kernel_type="gaussian")
    with pytest.raises(ValueError):
        local_quantile(
            np.arange(5.0), 0.5, 3.0, method="spatial", kernel_type="boxcar_discrete",
            coordinates="polar", axis_types=("linear",), width_units="grid",
        )


@pytest.mark.parametrize("function", [local_quantile, local_median, local_mad])
@pytest.mark.parametrize("kernel", ["boxcar", "gaussian_spectral"])
def test_order_statistics_reject_signed_active_kernels_before_empty_data_shortcut(
    function: Callable[..., Any], kernel: str,
) -> None:
    """An all-missing input does not bypass the nonnegative-weight kernel contract."""
    data = np.full(5, np.nan)
    args = (data, 0.5, 2.3) if function is local_quantile else (data, 2.3)
    with pytest.raises(ValueError):
        function(*args, kernel_type=kernel)


@pytest.mark.parametrize("q", [-0.01, 1.01, np.nan, np.inf, [0.5, np.nan], [[0.5]], 0.5j])
def test_quantile_rejects_invalid_probability_or_quantile_dimensions(q: Any) -> None:
    """Probabilities are finite real scalars or one-dimensional vectors within [0, 1]."""
    with pytest.raises((ValueError, TypeError)):
        local_quantile(np.full(4, np.nan), q, 3.0)


@pytest.mark.parametrize("kwargs", [
    {"method": "unrecognized"}, {"bins": 0}, {"bins": -1}, {"bins": 2.5},
    {"batch_size": 0}, {"batch_size": -1}, {"batch_size": 1.5},
    {"max_memory_mb": 0.0}, {"max_memory_mb": -1.0}, {"max_memory_mb": np.nan},
    {"min_weight": -1.0}, {"min_weight": np.nan},
    {"min_valid_fraction": -0.1}, {"min_valid_fraction": 1.1},
    {"weight_tolerance": -1.0}, {"weight_tolerance": np.nan},
    {"value_range": (1.0, 0.0)}, {"value_range": (0.0, np.inf)},
    {"value_range": (0.0, 1.0, 2.0)}, {"unrecognized_geometry_option": True},
])
def test_quantile_validates_controls_even_without_any_valid_data(kwargs: Dict[str, Any]) -> None:
    """Input-independent validation must happen before returning all-missing estimates."""
    with pytest.raises((ValueError, TypeError)):
        local_quantile(np.full(4, np.nan), 0.5, 3.0, **kwargs)


@pytest.mark.parametrize("scale", [-1.0, np.nan, np.inf])
def test_mad_rejects_invalid_explicit_scale(scale: float) -> None:
    """A scaling option must not silently create negative or nonfinite spread."""
    with pytest.raises((ValueError, TypeError)):
        local_mad(np.arange(5.0), 3.0, scale=scale)


def test_order_statistics_reject_mask_shape_mismatch() -> None:
    """A mask must describe observations on the same grid, without silent broadcasting."""
    with pytest.raises((ValueError, TypeError)):
        local_quantile(np.ones((3, 4)), 0.5, 3.0, mask=np.ones(4, dtype=bool))


@pytest.mark.parametrize("coords", [np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 4.0])])
def test_quantile_rejects_descending_or_irregular_coordinates(coords: np.ndarray) -> None:
    """Coordinate adapters must handle axis reversal; the kernel requires a regular grid."""
    with pytest.raises(ValueError):
        local_quantile(np.ones(3), 0.5, 3.0, coordinates="polar", axis_types=("linear",),
                       axis_coords=(coords,))


def test_quantile_rejects_empty_explicit_geometry_and_zero_range_singularity() -> None:
    """Invalid geometric domains must be diagnosed even without usable observations."""
    with pytest.raises(ValueError):
        local_quantile(np.empty((0, 2)), 0.5, 3.0, coordinates="polar",
                       axis_types=("azimuth", "range"), width_units="grid")
    with pytest.raises(ValueError):
        local_quantile(
            np.full((4, 2), np.nan), 0.5, 3.0, coordinates="polar",
            axis_types=("azimuth", "range"),
            axis_coords=(np.arange(4) * 90.0, np.array([0.0, 1.0])), boundary="periodic",
        )

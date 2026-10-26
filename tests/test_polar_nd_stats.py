"""N-D polar normalized statistics, support, restoration, and isolated fallback.

Truth is built from the independent ordered spatial/NumPy operator in
``test_polar_nd``. No package transfer, geometry, or private statistics helper
is used to calculate an expected value.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pytest

from dct_toolkit import dct_count, dct_mean, dct_prefill, dct_smooth, dct_std, dct_variance
from test_polar_nd import _nominal_volume, _polar_reference, _volume_geometry


MOMENTS = (dct_mean, dct_variance, dct_std)
STATISTICS = (*MOMENTS, dct_count)
ENTRY_POINTS = (*STATISTICS, dct_prefill, dct_smooth)


def _statistics_case() -> Tuple[np.ndarray, Tuple[float, ...], Dict[str, Any]]:
    """Construct a nonseparable gapped time/elevation/azimuth/range field."""
    data = np.random.default_rng(811).standard_normal((3, 4, 9, 6))
    data += np.arange(6)[None, None, None, :] / 3.0
    data[0, 0, 0, 0] = np.nan
    data[1, 1:3, 3:5, 2:4] = np.nan
    data[2, 3, 8, 5] = np.nan
    volume = _volume_geometry()
    geometry = {
        "axis_types": ("time", *volume["axis_types"]),
        "axis_coords": (np.array([3.0, 3.5, 4.0]), *volume["axis_coords"]),
        "width_units": ("physical", "grid", "physical", "physical"),
        "kernel_type": ("gaussian", "boxcar", "boxcar_discrete", "gaussian"),
        "boundary": ("periodic", "reflective", "periodic", "reflective"),
    }
    return data, (1.2, 2.6, 9.4, 1.6), geometry


def _moments_reference(
    data: np.ndarray,
    width: Any,
    geometry: Dict[str, Any],
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the complete operator to each raw moment and normalize only once."""
    support = np.isfinite(data)
    if mask is not None:
        support &= mask
    zero_filled = np.where(support, data, 0.0)
    density = _polar_reference(support.astype(float), width, **geometry)
    numerator = _polar_reference(zero_filled, width, **geometry)
    second_numerator = _polar_reference(zero_filled**2, width, **geometry)
    mean = np.full(data.shape, np.nan)
    second = np.full(data.shape, np.nan)
    np.divide(numerator, density, out=mean, where=density > 1e-3)
    np.divide(second_numerator, density, out=second, where=density > 1e-3)
    variance = np.maximum(second - mean**2, 0.0)
    count = np.clip(density, 0.0, 1.0) * _nominal_volume(data.shape, width, **geometry)
    return mean, variance, density, count, support


def _circular_nearest(values: np.ndarray) -> np.ndarray:
    """Fill a single periodic line by distance, preferring the preceding tied seed."""
    seeds = np.flatnonzero(np.isfinite(values))
    result = values.copy()
    n = len(values)
    for target in np.flatnonzero(~np.isfinite(values)):
        seed = min(
            seeds,
            key=lambda j: (min(abs(target - j), n - abs(target - j)), (target - j) % n),
        )
        result[target] = values[seed]
    return result


def _periodic_line_prefill_reference(
    data: np.ndarray, width: float, geometry: Dict[str, Any]
) -> np.ndarray:
    """Execute three public density-gated updates, then spatial circular nearest fill."""
    result = data.copy()
    for _ in range(3):
        mean, _, density, _, support = _moments_reference(result, width, geometry)
        accepted = ~support & np.isfinite(mean) & (np.clip(density, 0.0, 1.0) > 0.35)
        if not np.any(accepted):
            break
        result[accepted] = mean[accepted]
    return _circular_nearest(result.ravel()).reshape(data.shape)


@pytest.mark.parametrize("func", STATISTICS)
@pytest.mark.parametrize("restore", [None, False, True])
@pytest.mark.parametrize("explicit_mask", [False, True])
@pytest.mark.parametrize("mixed_case", [0, 1])
def test_4d_statistics_match_full_ordered_raw_moment_reference(
    func: Callable, restore: Optional[bool], explicit_mask: bool, mixed_case: int
) -> None:
    """Moments, counts, masks, and both restoration modes share the same 4-D operator."""
    data, widths, geometry = _statistics_case()
    if mixed_case:
        geometry.update(
            kernel_type=("boxcar_discrete", "gaussian", "boxcar", "boxcar_discrete"),
            boundary=("periodic", "periodic", "reflective", "periodic"),
        )
    mask = None
    if explicit_mask:
        mask = np.ones(data.shape, dtype=bool)
        mask[0, 1, 4:6, 4] = False
        mask[2, 2, 1, 2] = False
    mean, variance, density, count, support = _moments_reference(data, widths, geometry, mask)
    assert np.min(density) > 0.35  # Stay on the documented normal-ratio path.
    expected = {
        dct_mean: mean, dct_variance: variance, dct_std: np.sqrt(variance), dct_count: count,
    }[func]
    if restore is None or restore:
        expected = np.where(support, expected, np.nan)
    extra: Dict[str, Any] = {}
    if restore is not None:
        extra["restore_input_nan"] = restore
    if mask is not None and func is not dct_count:
        extra["mask"] = mask
    original = data.copy()
    mask_before = None if mask is None else mask.copy()
    values = support if func is dct_count else data
    result = func(values, widths, coordinates="polar", **geometry, **extra)
    np.testing.assert_allclose(result, expected, rtol=3e-11, atol=3e-11)
    np.testing.assert_array_equal(data, original)
    if mask is not None:
        np.testing.assert_array_equal(mask, mask_before)


@pytest.mark.parametrize("func", STATISTICS)
def test_statistics_follow_permuted_array_order(func: Callable) -> None:
    """Moment geometry and nominal count volume both transpose with the input."""
    data, widths, geometry = _statistics_case()
    mean, variance, _, count, support = _moments_reference(data, widths, geometry)
    expected = {
        dct_mean: mean, dct_variance: variance, dct_std: np.sqrt(variance), dct_count: count,
    }[func]
    permutation = (3, 1, 0, 2)
    permuted = {key: tuple(values[i] for i in permutation) for key, values in geometry.items()}
    values = support if func is dct_count else data
    result = func(
        values.transpose(permutation), tuple(widths[i] for i in permutation),
        coordinates="polar", restore_input_nan=False, **permuted,
    )
    np.testing.assert_allclose(result, expected.transpose(permutation), rtol=3e-11, atol=3e-11)


def test_mean_is_not_a_chain_of_normalized_axis_means() -> None:
    """A deliberately uneven mask exposes normalization between angular/range passes."""
    data = np.arange(75, dtype=float).reshape(3, 5, 5) / 7.0
    data[:, :2, :2] = np.nan
    data[1, 3, 3] = np.nan
    geometry = {
        "axis_types": ("elevation", "azimuth", "range"), "width_units": "grid",
        "kernel_type": "boxcar_discrete", "boundary": ("reflective", "periodic", "reflective"),
    }
    expected, _, density, _, _ = _moments_reference(data, 3.0, geometry)
    assert np.min(density) > 1e-3
    wrong = data.copy()
    for axis in (1, 0, 2):
        wrong, _, _, _, _ = _moments_reference(wrong, 3.0, dict(geometry, filter_axes=(axis,)))
    assert np.max(np.abs(expected - wrong)) > 0.05
    result = dct_mean(data, 3.0, coordinates="polar", restore_input_nan=False, **geometry)
    np.testing.assert_allclose(result, expected, atol=2e-12)


@pytest.mark.parametrize("func", MOMENTS)
def test_density_gate_uses_full_original_mask_response(func: Callable) -> None:
    """Density gating masks the reference moments, not a product of per-axis ratios."""
    data, widths, geometry = _statistics_case()
    mean, variance, density, _, _ = _moments_reference(data, widths, geometry)
    threshold = 0.98
    assert np.any(density < threshold) and np.any(density > threshold)
    base = {dct_mean: mean, dct_variance: variance, dct_std: np.sqrt(variance)}[func]
    expected = np.where(density >= threshold, base, np.nan)
    result = func(
        data, widths, coordinates="polar", restore_input_nan=False,
        min_effective_density=threshold, **geometry,
    )
    np.testing.assert_allclose(result, expected, rtol=3e-11, atol=3e-11)


def test_count_uses_requested_effective_widths_not_rounded_taps_or_ess() -> None:
    """Sub-grid requested widths remain in nominal support even when taps round to one."""
    mask = np.ones((4, 7, 5, 2), dtype=bool)
    mask[1, 2, 2, 0] = False
    geometry = {
        "axis_types": ("time", "azimuth", "range", "batch"),
        "width_units": "grid", "filter_axes": (0, -2),
        "kernel_type": "boxcar_discrete", "boundary": "periodic",
    }
    widths = (0.25, 19.3, 2.2, 500.0)
    density = _polar_reference(mask.astype(float), widths, **geometry)
    expected = np.clip(density, 0.0, 1.0) * (0.25 * 2.2)
    result = dct_count(mask, widths, coordinates="polar", restore_input_nan=False, **geometry)
    np.testing.assert_allclose(result, expected, atol=2e-12)
    assert np.max(np.abs(result - np.clip(density, 0.0, 1.0) * 3.0)) > 1.0


def test_analytical_kernel_count_clips_negative_density_and_overshoot() -> None:
    """Signed analytical responses cannot produce negative or over-nominal counts."""
    mask = np.zeros((2, 1, 9, 1), dtype=bool)
    mask[0, 0, 4, 0] = True
    mask[1] = ~mask[0]
    geometry = {
        "axis_types": ("batch", "elevation", "azimuth", "range"),
        "filter_axes": (2,), "width_units": "grid", "boundary": "periodic",
        "kernel_type": "boxcar",
    }
    density = _polar_reference(mask.astype(float), 1.7, **geometry)
    assert np.min(density) < -1e-3 and np.max(density) > 1.0 + 1e-3
    result = dct_count(mask, 1.7, coordinates="polar", restore_input_nan=False, **geometry)
    np.testing.assert_allclose(result, np.clip(density, 0.0, 1.0) * 1.7, atol=2e-12)
    assert np.min(result) >= 0.0 and np.max(result) <= 1.7 + 2e-12


def test_physical_count_uses_local_cosine_spacing_and_singleton_effective_width_one() -> None:
    """Singleton dependencies retain their values but contribute one to count volume."""
    mask = np.ones((1, 8, 1, 3, 2), dtype=bool)
    geometry = {
        "axis_types": ("elevation", "azimuth", "range", "time", "batch"),
        "axis_coords": (
            np.array([60.0]), np.arange(8) * 45.0, np.array([4.0]),
            np.array([0.0, 2.0, 4.0]), None,
        ),
        "boundary": ("reflective", "periodic", "reflective", "periodic", "periodic"),
    }
    widths = (111.0, 3.2 * 4.0 * 0.5 * np.pi / 4.0, 999.0, 3.4, 10000.0)
    expected_volume = 3.2 * (3.4 / 2.0)
    result = dct_count(mask, widths, coordinates="polar", restore_input_nan=False, **geometry)
    np.testing.assert_allclose(result, np.full(mask.shape, expected_volume), atol=2e-12)


def test_targeted_finite_prefill_excludes_targets_and_preserves_every_other_value() -> None:
    """Finite requested replacements must not contribute to their own estimates."""
    data, widths, geometry = _statistics_case()
    target = np.zeros(data.shape, dtype=bool)
    target[0, 1, 4, 3] = True
    target[2, 2, 6, 1] = True
    data[target] = 1000.0
    mean, _, density, _, _ = _moments_reference(data, widths, geometry, ~target)
    assert np.min(density) > 0.35
    expected = np.where(target, mean, data)
    original = data.copy()
    result = dct_prefill(
        data, widths, coordinates="polar", fill_mask=target, max_iter=1, **geometry
    )
    np.testing.assert_allclose(result, expected, rtol=3e-11, atol=3e-11)
    np.testing.assert_array_equal(result[~target], data[~target])
    np.testing.assert_array_equal(data, original)
    assert np.max(result[target]) < 10.0


def test_prefill_and_wrapper_use_full_ordered_reference_then_restore_original_nans() -> None:
    """One supported fill iteration has a direct numerical truth, including the wrapper."""
    data, widths, geometry = _statistics_case()
    mean, _, density, _, support = _moments_reference(data, widths, geometry)
    assert np.min(density) > 0.35
    filled = np.where(support, data, mean)
    expected = np.where(support, _polar_reference(filled, widths, **geometry), np.nan)
    actual_fill = dct_prefill(data, widths, coordinates="polar", max_iter=1, **geometry)
    np.testing.assert_allclose(actual_fill, filled, rtol=3e-11, atol=3e-11)
    np.testing.assert_array_equal(actual_fill[support], data[support])
    for iteration_kwarg in ({"prefill_max_iter": 1}, {"max_iter": 1}):
        result = dct_smooth(data, widths, coordinates="polar", **geometry, **iteration_kwarg)
        np.testing.assert_allclose(result, expected, rtol=3e-11, atol=3e-11)
        np.testing.assert_array_equal(np.isnan(result), ~support)


@pytest.mark.parametrize("selection", [(2,), ()])
def test_unestimable_finite_targets_keep_original_without_borrowing_other_batches(
    selection: Tuple[int, ...],
) -> None:
    """Replacing an entire isolated domain cannot consume its own targeted observations."""
    data = np.array([[[1.0, 2.0, 3.0]], [[100.0, 100.0, 100.0]]])
    target = np.zeros(data.shape, dtype=bool)
    target[0] = True
    result = dct_prefill(
        data, 3.0, coordinates="polar", axis_types=("batch", "azimuth", "range"),
        width_units="grid", filter_axes=selection, fill_mask=target, max_iter=1,
    )
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize("func", [*MOMENTS, dct_smooth])
def test_integer_statistics_and_wrapper_match_float_reference_without_mutation(
    func: Callable,
) -> None:
    """Integer arrays are promoted before moments, geometry, or smoothing are evaluated."""
    data = np.random.default_rng(823).integers(-20, 21, size=(2, 3, 7, 4))
    geometry = {
        "axis_types": ("batch", "time", "azimuth", "range"), "width_units": "grid",
        "boundary": ("periodic", "periodic", "periodic", "reflective"),
        "kernel_type": ("boxcar", "gaussian", "boxcar_discrete", "boxcar"),
    }
    widths = (99.0, 2.3, 3.2, 1.7)
    mean, variance, _, _, _ = _moments_reference(data.astype(float), widths, geometry)
    expected = {
        dct_mean: mean, dct_variance: variance, dct_std: np.sqrt(variance),
        dct_smooth: _polar_reference(data, widths, **geometry),
    }[func]
    original = data.copy()
    result = func(data, widths, coordinates="polar", **geometry)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_array_equal(data, original)


def test_density_gated_iteration_defers_to_circular_nearest_with_numeric_truth() -> None:
    """A high gate deliberately forces nearest, including preceding-beam circular ties."""
    data = np.full((1, 9, 1), np.nan)
    data[0, 1, 0], data[0, 6, 0] = 2.0, 10.0
    geometry = {
        "axis_types": ("time", "azimuth", "range"), "width_units": "grid",
        "boundary": ("reflective", "periodic", "reflective"),
        "filter_axes": (1,), "kernel_type": "boxcar_discrete",
    }
    density = _polar_reference(np.isfinite(data).astype(float), 3.0, **geometry)
    assert np.max(density[~np.isfinite(data)]) < 0.99
    expected = _circular_nearest(data.ravel()).reshape(data.shape)
    result = dct_prefill(
        data, 3.0, coordinates="polar", max_iter=1, min_effective_density=0.99, **geometry
    )
    np.testing.assert_allclose(result, expected, atol=2e-12)
    assert result[0, 8, 0] == 10.0  # Equidistant: preceding circular seed at beam 6.


def test_low_support_variance_prefills_first_and_second_moments_independently() -> None:
    """Fallback must smooth separately prefilled x and x**2, not square a filled mean."""
    data = np.full((1, 32, 1), np.nan)
    data[0, 8, 0], data[0, 10, 0], data[0, 24, 0] = 1.0, 5.0, 9.0
    geometry = {
        "axis_types": ("time", "azimuth", "range"), "width_units": "grid",
        "filter_axes": (1,), "boundary": "periodic", "kernel_type": "boxcar_discrete",
    }
    mean, _, density, _, _ = _moments_reference(data, 5.0, geometry)
    first_filled = _periodic_line_prefill_reference(data, 5.0, geometry)
    second_filled = _periodic_line_prefill_reference(data**2, 5.0, geometry)
    fallback_first = _polar_reference(first_filled, 5.0, **geometry)
    fallback_second = _polar_reference(second_filled, 5.0, **geometry)
    unsupported = density <= 1e-3
    expected_mean = np.where(unsupported, fallback_first, mean)
    raw_second = np.full(data.shape, np.nan)
    np.divide(
        _polar_reference(np.nan_to_num(data)**2, 5.0, **geometry), density,
        out=raw_second, where=~unsupported,
    )
    expected_second = np.where(unsupported, fallback_second, raw_second)
    expected_variance = np.maximum(expected_second - expected_mean**2, 0.0)
    wrong_second = _polar_reference(first_filled**2, 5.0, **geometry)
    assert np.any(unsupported)
    assert np.max(np.abs(second_filled - first_filled**2)) > 0.5
    assert np.max(np.abs(fallback_second[unsupported] - wrong_second[unsupported])) > 0.05
    np.testing.assert_allclose(
        dct_prefill(data, 5.0, coordinates="polar", **geometry), first_filled, atol=2e-11
    )
    for func, expected in (
        (dct_mean, expected_mean), (dct_variance, expected_variance),
        (dct_std, np.sqrt(expected_variance)),
    ):
        result = func(data, 5.0, coordinates="polar", restore_input_nan=False, **geometry)
        if func is dct_std:
            # Squaring avoids sqrt amplifying sub-ULP cancellation at zero variance.
            np.testing.assert_allclose(result**2, expected_variance, rtol=3e-11, atol=3e-11)
        else:
            np.testing.assert_allclose(result, expected, rtol=3e-11, atol=3e-11)


@pytest.mark.parametrize("periodic_role", ["azimuth", "time", "both"])
@pytest.mark.parametrize("periodic", [False, True])
def test_targeted_global_nearest_honors_angular_and_extra_time_periodic_seams(
    periodic_role: str, periodic: bool
) -> None:
    """No same-axis donors exist, so global nearest must use every selected boundary."""
    data = np.full((8, 1, 8, 3), np.nan)
    if periodic_role == "azimuth":
        seed_a, seed_b = (1, 0, 7, 0), (1, 0, 3, 0)
    elif periodic_role == "time":
        seed_a, seed_b = (7, 0, 1, 0), (3, 0, 1, 0)
    else:
        seed_a, seed_b = (7, 0, 7, 0), (3, 0, 3, 0)
    data[seed_a], data[seed_b] = 10.0, 30.0
    target = np.zeros(data.shape, dtype=bool)
    target[0, 0, 0, 2] = True
    boundary = ["reflective"] * 4
    periodic_axes = (
        (0, 2) if periodic_role == "both" else ((0,) if periodic_role == "time" else (2,))
    )
    if periodic:
        for axis in periodic_axes:
            boundary[axis] = "periodic"
    geometry = {
        "axis_types": ("time", "elevation", "azimuth", "range"),
        "width_units": "grid", "kernel_type": "boxcar_discrete", "boundary": tuple(boundary),
    }
    expected = data.copy()
    expected[target] = 10.0 if periodic else 30.0
    result = dct_prefill(
        data, 1.0, coordinates="polar", fill_mask=target, max_iter=1, **geometry
    )
    np.testing.assert_allclose(result, expected, atol=2e-12)
    np.testing.assert_array_equal(result[~target], data[~target])
    if periodic:
        for axis in periodic_axes:
            rotated = dct_prefill(
                np.roll(data, 2, axis=axis), 1.0, coordinates="polar",
                fill_mask=np.roll(target, 2, axis=axis), max_iter=1, **geometry,
            )
            np.testing.assert_allclose(np.roll(rotated, -2, axis=axis), expected, atol=2e-12)


@pytest.mark.parametrize("func", ENTRY_POINTS)
def test_5d_allmissing_batch_is_isolated_from_supported_batches(func: Callable) -> None:
    """A wholly missing batch remains missing through ratios, prefill, and all fallbacks."""
    data = np.empty((3, 2, 3, 7, 4))
    data[0], data[1], data[2] = 2.0, np.nan, 17.0
    data[0, 0, 1, 2, 1] = np.nan
    data[2, 1, 1, 4, 2] = np.nan
    geometry = {
        "axis_types": ("batch", "time", "elevation", "azimuth", "range"),
        "width_units": "grid", "boundary": ("periodic",) * 5,
        "kernel_type": ("boxcar", "gaussian", "boxcar_discrete", "gaussian", "boxcar_discrete"),
    }
    widths = (500.0, 2.1, 3.1, 3.7, 2.2)
    mean, variance, _, count, support = _moments_reference(data, widths, geometry)
    expected = {
        dct_mean: mean, dct_variance: variance, dct_std: np.sqrt(variance), dct_count: count,
    }
    if func is dct_prefill:
        truth = np.empty_like(data)
        truth[0], truth[1], truth[2] = 2.0, np.nan, 17.0
    elif func is dct_smooth:
        truth = data.copy()
    else:
        truth = expected[func]
    extra = {"restore_input_nan": False} if func in STATISTICS else {}
    result = func(
        support if func is dct_count else data, widths, coordinates="polar", **geometry, **extra
    )
    if func is dct_std:
        np.testing.assert_allclose(result**2, variance, rtol=3e-11, atol=3e-11)
    else:
        np.testing.assert_allclose(result, truth, rtol=3e-11, atol=3e-11)
    if func is dct_count:
        np.testing.assert_allclose(result[1], 0.0, atol=2e-12)
    else:
        assert np.all(np.isnan(result[1]))


@pytest.mark.parametrize("func", [*MOMENTS, dct_prefill, dct_smooth])
def test_inactive_nonbatch_axes_are_isolated_nearest_and_fallback_domains(func: Callable) -> None:
    """Selecting range alone cannot borrow observations across time or azimuth."""
    data = np.full((2, 3, 5), np.nan)
    data[0, 0, 1] = 3.0
    data[1, 2, 3] = 11.0
    geometry = {
        "axis_types": ("time", "azimuth", "range"), "width_units": "grid",
        "filter_axes": (-1,), "kernel_type": "boxcar_discrete", "boundary": "periodic",
    }
    expected = np.full(data.shape, np.nan)
    expected[0, 0, :], expected[1, 2, :] = 3.0, 11.0
    if func in (dct_variance, dct_std):
        expected[np.isfinite(expected)] = 0.0
    if func is dct_smooth:
        expected[~np.isfinite(data)] = np.nan
    extra = {"restore_input_nan": False} if func in MOMENTS else {}
    result = func(data, (99.0, 99.0, 1.0), coordinates="polar", **geometry, **extra)
    if func is dct_std:
        np.testing.assert_allclose(result**2, expected, rtol=2e-11, atol=2e-11)
    else:
        np.testing.assert_allclose(result, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("func", ENTRY_POINTS)
def test_no_active_axes_are_identity_and_cannot_fill_from_other_cells(func: Callable) -> None:
    """An empty filter selection makes each individual cell its own support domain."""
    data = np.arange(24, dtype=float).reshape(2, 3, 4)
    data[0, 1, 2] = np.nan
    data[1, 2, 0] = np.nan
    support = np.isfinite(data)
    geometry = {
        "axis_types": ("batch", "azimuth", "range"), "filter_axes": (),
        "width_units": "physical", "boundary": "periodic",
    }
    if func is dct_count:
        expected = support.astype(float)
    elif func in (dct_variance, dct_std):
        expected = np.where(support, 0.0, np.nan)
    else:
        expected = data
    extra = {"restore_input_nan": False} if func in STATISTICS else {}
    result = func(
        support if func is dct_count else data, 99.0, coordinates="polar", **geometry, **extra
    )
    np.testing.assert_allclose(result, expected, atol=2e-12)


@pytest.mark.parametrize("func", ENTRY_POINTS)
def test_all_nan_nd_valid_geometry_keeps_nan_and_zero_count_semantics(func: Callable) -> None:
    """With no support anywhere, valid explicit geometry has a deterministic result."""
    data = np.full((2, 3, 7, 4), np.nan)
    geometry = {
        "axis_types": ("batch", "time", "azimuth", "range"), "width_units": "grid",
        "boundary": "periodic",
        "kernel_type": ("gaussian", "boxcar", "boxcar_discrete", "gaussian"),
    }
    extra = {"restore_input_nan": False} if func is dct_count else {}
    values = np.isfinite(data) if func is dct_count else data
    result = func(values, (99.0, 3.0, 2.2, 1.7), coordinates="polar", **geometry, **extra)
    expected = np.zeros(data.shape) if func is dct_count else data
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("func", ENTRY_POINTS)
@pytest.mark.parametrize("all_missing", [False, True])
@pytest.mark.parametrize(
    "override, message", [
        ({"axis_types": ("azimuth", "range")}, "axis|axes|length|dimension"),
        ({"axis_coords": (None, np.array([0.0, 1.0, 2.2, 3.0]), None)}, "coord|uniform|spacing"),
        ({"filter_axes": (0, -3)}, "filter|axis|axes|duplicate"),
        ({"boundary": ("reflective", "open", "periodic")}, "bound|axis"),
        ({"width_units": "metres"}, "unit"),
        ({"width": 0.0}, "width|positive"),
    ],
)
def test_geometry_validation_precedes_all_nan_and_no_target_early_returns(
    func: Callable, all_missing: bool, override: Dict[str, Any], message: str
) -> None:
    """Every public path validates even when there is nothing to estimate or prefill."""
    data = np.full((2, 4, 3), np.nan if all_missing else 7.0)
    kwargs = dict(
        width=2.0, coordinates="polar", axis_types=("time", "azimuth", "range"),
        width_units="grid",
    )
    kwargs.update(override)
    values = np.isfinite(data) if func is dct_count else data
    with pytest.raises(ValueError, match="(?i)" + message):
        func(values, **kwargs)


@pytest.mark.parametrize("func", [dct_mean, dct_variance, dct_std, dct_prefill])
def test_nd_mask_shape_validation_is_not_bypassed(func: Callable) -> None:
    """Both normalized support masks and targeted replacement masks retain shape checks."""
    data, widths, geometry = _statistics_case()
    keyword = "fill_mask" if func is dct_prefill else "mask"
    with pytest.raises(ValueError, match="(?i)mask|shape"):
        func(
            data, widths, coordinates="polar", **geometry,
            **{keyword: np.ones(data.shape[:-1], dtype=bool)},
        )

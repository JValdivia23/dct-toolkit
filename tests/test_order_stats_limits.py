"""Storage, extreme-value, and broad-window regressions for local distributions."""

import numpy as np
import pytest

from dct_toolkit import local_mad, local_median, local_quantile, local_support


@pytest.mark.parametrize("exclude", [False, True])
def test_mad_disk_spill_and_memory_histograms_agree(exclude: bool) -> None:
    """Changing working-memory/storage policy does not change the estimator."""
    data = np.random.default_rng(2401).normal(size=(32, 32))
    data[3:5, 9:12] = np.nan
    options = dict(bins=16, exclude_center=exclude, return_info=True)
    memory = local_mad(data, 5, max_memory_mb=64, **options)
    disk = local_mad(data, 5, max_memory_mb=.2, **options)
    np.testing.assert_array_equal(memory.values, disk.values)
    np.testing.assert_array_equal(memory.median, disk.median)
    np.testing.assert_array_equal(memory.error_bound, disk.error_bound)


@pytest.mark.parametrize("coordinates", ["cartesian", "polar"])
def test_broad_discrete_operator_does_not_enumerate_requested_window(coordinates: str) -> None:
    """Many repeated visits reduce to a uniform finite-domain limit to roundoff."""
    data = np.arange(1., 7.).reshape(3, 2)
    options = dict(coordinates=coordinates, kernel_type="boxcar_discrete")
    actual = local_median(data, 1e100, **options)
    np.testing.assert_allclose(actual, 3.)
    np.testing.assert_allclose(local_mad(data, 1e100, **options), 1.)
    support = local_support(np.ones_like(data, bool), 1e100, **options)
    np.testing.assert_allclose(support.self_weight, 1 / data.size, atol=1e-14)


@pytest.mark.parametrize("method", ["exact", "histogram", "spatial"])
def test_numeric_mask_nonfinite_entries_never_contribute(method: str) -> None:
    """Numeric masks follow the same validity convention as local_support."""
    data = np.array([1., 1000., 3., 4., 5.])
    mask = np.array([1., np.nan, 1., 1., 1.])
    options = dict(method=method, kernel_type="boxcar_discrete", restore_input_nan=False)
    actual = local_quantile(data, .5, 3, mask=mask, **options)
    expected = local_quantile(data, .5, 3, mask=np.isfinite(mask), **options)
    np.testing.assert_array_equal(actual, expected)


def test_integer_histogram_range_is_promoted_before_subtraction() -> None:
    """A valid int16 full-span range must not wrap to a negative bin width."""
    data = np.linspace(-30000., 30000., 17)
    limits = np.array([-32768, 32767], dtype=np.int16)
    options = dict(bins=4, kernel_type="boxcar_discrete", restore_input_nan=False,
                   return_info=True)
    actual = local_quantile(data, [.25, .5, .75], 7, value_range=limits, **options)
    expected = local_quantile(data, [.25, .5, .75], 7,
                              value_range=limits.astype(float), **options)
    assert np.all(actual.resolved)
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.error_bound, expected.error_bound)


def test_tiny_tied_observations_are_aggregated_before_weight_tolerance() -> None:
    """Two five-sigma tails have resolvable combined mass despite tiny individual weights."""
    data = np.zeros(21)
    data[[5, 15]] = 100.
    options = dict(weight_tolerance=2e-6, bins=2, return_info=True)
    exact = local_quantile(data, 1., np.sqrt(12), method="exact", **options)
    histogram = local_quantile(data, 1., np.sqrt(12), method="histogram", **options)
    assert exact.values[10] == 100.
    assert histogram.values[10] == 100.
    np.testing.assert_array_equal(exact.values, histogram.values)
    np.testing.assert_array_equal(histogram.error_bound, 0.)


def test_total_mass_without_an_eligible_value_remains_unresolved() -> None:
    """Argmax on an empty CDF-hit set cannot fabricate a quantile."""
    data = np.full(21, np.nan)
    data[5], data[15] = 1., 2.
    options = dict(bins=2, weight_tolerance=2e-6, min_weight=0.,
                   restore_input_nan=False, return_info=True)
    exact = local_quantile(data, [0., .5, 1.], np.sqrt(12), method="exact", **options)
    histogram = local_quantile(data, [0., .5, 1.], np.sqrt(12),
                               method="histogram", **options)
    assert exact.support.supported[10]
    assert not exact.resolved[:, 10].any()
    np.testing.assert_array_equal(exact.values, histogram.values)


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(float).max,
                    reason="No wider floating range on this platform")
def test_unrepresentable_finite_observations_are_not_silently_discarded() -> None:
    """Explicit masks may exclude extended values, but unmasked finite mass is preserved."""
    data = np.array([np.longdouble('1e400'), 1., 2., 3.], dtype=np.longdouble)
    with pytest.raises(ValueError, match="float64"):
        local_median(data, 3)
    result = local_median(data, 3, mask=np.array([False, True, True, True]))
    assert np.isnan(result[0]) and np.isfinite(result[1:]).all()
    with pytest.raises(ValueError, match="float64"):
        local_median(np.arange(5.), 3, bins=2,
                      value_range=(-np.longdouble('1e400'), np.longdouble('1e400')))

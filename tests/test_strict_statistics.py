"""Independent no-filling and center-exclusion contracts for existing APIs."""

from typing import Callable
from unittest.mock import patch

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from dct_toolkit import dct_mean, dct_smooth, dct_std, dct_variance, local_support
from test_polar_nd import _polar_reference


@pytest.mark.parametrize("polar", [False, True])
@pytest.mark.parametrize("exclude", [False, True])
def test_strict_statistics_match_original_weighted_moments(polar: bool, exclude: bool) -> None:
    """Every original moment passes through all filters before normalization once."""
    shape = (4, 3, 3) if polar else (7, 5)
    rng = np.random.default_rng(2101)
    data = rng.uniform(-2, 5, shape)
    mask = rng.random(shape) > 0.25
    data.ravel()[::11] = np.nan
    mask &= np.isfinite(data)
    if polar:
        widths = (4., 2., 3.)
        geometry = {
            "axis_types": ("azimuth", "elevation", "range"),
            "axis_coords": (np.arange(4) * 90., np.array([-15., 0., 15.]),
                            np.array([3., 4., 5.])),
            "boundary": ("periodic", "reflective", "reflective"),
            "kernel_type": ("gaussian", "boxcar_discrete", "boxcar_discrete"),
        }

        def smooth(values: np.ndarray) -> np.ndarray:
            """Apply an independent spatial polar operator."""
            return _polar_reference(values, widths, **geometry)
    else:
        widths = (3., 5.)
        geometry = {"kernel_type": ("gaussian", "boxcar_discrete")}

        def smooth(values: np.ndarray) -> np.ndarray:
            """Apply independent sampled Gaussian and discrete spatial passes."""
            first = gaussian_filter1d(values, 3 / np.sqrt(12), axis=0, truncate=12)
            return uniform_filter1d(first, 5, axis=1, mode="reflect")

    n = data.size
    matrix = np.stack([smooth(e.reshape(shape)).ravel() for e in np.eye(n)], axis=1)
    if exclude:
        np.fill_diagonal(matrix, 0.)
    weights = matrix * mask.ravel()
    denominator = weights.sum(axis=1)
    good = denominator > 1e-3
    x = np.where(mask, data, 0).ravel()
    mean = np.full(n, np.nan)
    second = np.full(n, np.nan)
    np.divide(weights @ x, denominator, out=mean, where=good)
    np.divide(weights @ (x**2), denominator, out=second, where=good)
    variance = np.maximum(second - mean**2, 0)
    options = {"coordinates": "polar" if polar else "cartesian", "mask": mask,
               "restore_input_nan": False, "fallback": "none", "exclude_center": exclude,
               **geometry}
    with patch("dct_toolkit.stats.dct_prefill", side_effect=AssertionError("Unexpected fill")):
        np.testing.assert_allclose(dct_mean(data, widths, **options), mean.reshape(shape),
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dct_variance(data, widths, **options),
                                   variance.reshape(shape), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dct_std(data, widths, **options)**2,
                                   variance.reshape(shape), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("function", [dct_mean, dct_variance, dct_std, dct_smooth])
def test_strict_statistics_never_complete_unsupported_regions(function: Callable) -> None:
    """An isolated observation does not acquire support across a distant empty region."""
    data = np.full(19, np.nan)
    data[3] = 8.
    with patch("dct_toolkit.stats.dct_prefill", side_effect=AssertionError("Unexpected fill")):
        actual = function(data, 3, kernel_type="boxcar_discrete", fallback="none")
    assert np.isfinite(actual[3])
    assert np.isnan(actual[10])
    actual = function(data, 3, kernel_type="boxcar_discrete", fallback="none",
                      restore_input_nan=False)
    assert np.isnan(actual[10])
    assert np.isfinite(actual[2])


@pytest.mark.parametrize("function", [dct_mean, dct_variance, dct_std, dct_smooth])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (4,)])
def test_strict_identity_exclusion_has_no_self_fallback(function: Callable, shape: tuple) -> None:
    """Singleton and identity operators have no external support after exclusion."""
    actual = function(np.full(shape, 4.), .01, kernel_type="boxcar_discrete",
                      fallback="none", exclude_center=True)
    assert actual.shape == shape
    assert np.all(np.isnan(actual))


@pytest.mark.parametrize("function", [dct_mean, dct_variance, dct_std, dct_smooth])
def test_strict_controls_reject_ambiguous_policies(function: Callable) -> None:
    """Exclusion cannot silently reintroduce tested observations through filling."""
    with pytest.raises(ValueError, match="fallback"):
        function(np.ones(5), 3, fallback="other")
    with pytest.raises(ValueError, match="fallback='none'"):
        function(np.ones(5), 3, exclude_center=True)
    with pytest.raises(ValueError, match="Nonnegative"):
        function(np.ones(5), 2.5, kernel_type="boxcar", fallback="none", exclude_center=True)


def test_strict_smooth_is_normalized_not_zero_filled_smoothing() -> None:
    """Turning off prefill retains normalized weighting rather than introducing zero signal."""
    data = np.array([10., np.nan, 10., 10., 10.])
    actual = dct_smooth(data, 3, kernel_type="boxcar_discrete", fallback="none")
    np.testing.assert_allclose(actual[np.isfinite(data)], 10.)
    assert np.isnan(actual[1])


def test_strict_density_gate_uses_available_external_mass() -> None:
    """Center exclusion changes the available denominator used by the fraction gate."""
    data = np.array([1., 2., 3., 4., 5.])
    for function in (dct_mean, dct_variance, dct_std):
        actual = function(data, 3, kernel_type="boxcar_discrete", fallback="none",
                          exclude_center=True, min_effective_density=.95)
        assert np.all(np.isfinite(actual))


def test_excluded_full_fraction_does_not_reject_transform_roundoff() -> None:
    """A 100% external support threshold is compared with numerical mass tolerance."""
    data = np.ones(15)
    data[7] = np.nan
    support = local_support(np.isfinite(data), 2.3, exclude_center=True,
                            min_valid_fraction=1)
    assert support.supported[7]
    for function in (dct_mean, dct_variance, dct_std):
        actual = function(data, 2.3, fallback="none", exclude_center=True,
                          min_effective_density=1, restore_input_nan=False)
        assert np.isfinite(actual[7])

    complete = local_support(np.ones(15, bool), 1., exclude_center=True,
                             min_valid_fraction=1)
    assert complete.supported.all()


def test_support_retains_coordinate_snapshot_for_audit() -> None:
    """Caller coordinate mutations cannot change the recorded operator geometry."""
    coordinates = np.arange(5.)
    support = local_support(np.ones(5, bool), 3., coordinates="polar",
                            axis_types=("linear",), axis_coords=(coordinates,))
    coordinates[:] = 100
    np.testing.assert_array_equal(support.geometry["axis_coords"][0], np.arange(5.))
    assert support.geometry["shape"] == (5,)

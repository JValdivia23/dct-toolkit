"""Independent spatial references for sampled Gaussian filtering and moments."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from dct_toolkit import dct_mean, dct_prefill, dct_std, dct_variance
from dct_toolkit.cartesian import smooth_cartesian
from dct_toolkit.core import dct_convolve_1d, get_dct_transfer_function
from dct_toolkit.polar import compute_polar_transfer_functions, smooth_polar


@pytest.mark.parametrize("n", [1, 2, 7, 16, 65])
@pytest.mark.parametrize("width", [1e-100, 0.25, 1.0, 2.0, 3.0, 5.0, 8.0, 40.0, 250.0])
@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
def test_sampled_gaussian_matches_spatial_convolution(
    n: int, width: float, boundary: str
) -> None:
    """Odd/even axes, subgrid widths and repeated boundary visits match SciPy."""
    data = np.random.default_rng(1401).standard_normal(n)
    mode = "wrap" if boundary == "periodic" else "reflect"
    expected = gaussian_filter1d(data, width / np.sqrt(12), mode=mode, truncate=12)
    actual = smooth_polar(
        data, width, axis_types=("linear",), width_units="grid", boundary=boundary
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
    if boundary == "reflective":
        response = get_dct_transfer_function(n, "gaussian", width)
        np.testing.assert_allclose(
            dct_convolve_1d(data, response), expected, rtol=2e-13, atol=2e-14
        )
        np.testing.assert_allclose(smooth_cartesian(data, width), expected, atol=2e-14)
    else:
        # A one-gate legacy polar grid has effective beam width = width.
        response, _ = compute_polar_transfer_functions(
            (n, 1), 360 / n, (width * (2 * np.pi / n), 1.0), az_boundary=boundary
        )
        legacy = np.fft.irfft(np.fft.rfft(data) * response[:, 0], n=n)
        np.testing.assert_allclose(legacy, expected, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("sigma", [0.999, 1.0, 1.001, 10.0, 100.0, 1000.0])
@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
def test_broad_gaussian_matches_direct_sample_sums(
    n: int, sigma: float, boundary: str
) -> None:
    """Full Gaussian summation agrees across narrow/broad evaluation crossover."""
    data = np.random.default_rng(1403).standard_normal(n)
    expected = gaussian_filter1d(
        data, sigma, mode="wrap" if boundary == "periodic" else "reflect", truncate=12
    )
    actual = smooth_polar(
        data, sigma * np.sqrt(12), axis_types=("linear",),
        width_units="grid", boundary=boundary,
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize("width", [np.nextafter(0.0, 1.0), 1e300])
@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
def test_extreme_gaussian_widths_remain_finite(width: float, boundary: str) -> None:
    """Tiny widths are identity; enormous widths reach the uniform finite-grid limit."""
    data = np.arange(7, dtype=float)
    expected = data if width < 1 else np.full_like(data, data.mean())
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = smooth_polar(
            data, width, axis_types=("linear",), width_units="grid", boundary=boundary
        )
        cartesian = smooth_cartesian(data, width)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_allclose(cartesian, expected, atol=2e-14)


@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
def test_spectral_gaussian_retains_original_response(boundary: str) -> None:
    """The compatibility name preserves the exponential spectral definition."""
    n, width = 16, 2.0
    if boundary == "reflective":
        omega = np.pi * np.arange(n) / n
        actual = get_dct_transfer_function(n, "gaussian_spectral", width)
    else:
        omega = 2 * np.pi * np.fft.rfftfreq(n)
        actual, _ = compute_polar_transfer_functions(
            (n, 1), 360 / n, (width * 2 * np.pi / n, 1.0),
            kernel_type="gaussian_spectral", az_boundary=boundary,
        )
        actual = actual[:, 0]
    expected = np.exp(-0.5 * (omega * width / np.sqrt(12)) ** 2)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_default_gaussian_removes_substantive_negative_impulse_variance() -> None:
    """Spatial positivity fixes the raw estimator before its final variance clamp."""
    data = np.zeros(65)
    data[32] = 1
    sampled = smooth_cartesian(data, 2.0)
    spectral = smooth_cartesian(data, 2.0, kernel_type="gaussian_spectral")
    assert np.min(spectral - spectral**2) < -0.01
    assert np.min(sampled) >= -2e-15
    assert np.min(sampled - sampled**2) >= -2e-15
    expected = gaussian_filter1d(data, 2 / np.sqrt(12), truncate=12)
    expected = np.maximum(expected - expected**2, 0)
    np.testing.assert_allclose(dct_variance(data, 2), expected, atol=2e-14)
    np.testing.assert_allclose(dct_std(data, 2)**2, expected, atol=2e-14)


def test_full_gaussian_preserves_tails_beyond_four_sigma() -> None:
    """The standard kernel includes resolvable tails beyond a four-sigma window."""
    data = np.zeros(65)
    data[32] = 1
    actual = smooth_cartesian(data, np.sqrt(12))
    expected = gaussian_filter1d(data, 1.0, truncate=12)
    assert actual[37] > 1e-6
    assert actual[38] > 5e-9
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


def test_gaussian_discrete_boxcar_hybrid_statistics_match_spatial_reference() -> None:
    """Fractional requested widths retain positive kernels on every hybrid axis."""
    data = np.random.default_rng(1407).standard_normal((7, 9, 11))
    mask = np.random.default_rng(1408).random(data.shape) > 0.2
    finite_data = np.where(mask, data, 0)
    data[~mask] = np.nan

    def spatial_smooth(values: np.ndarray) -> np.ndarray:
        """Apply the independent sampled/discrete kernels before normalization."""
        result = gaussian_filter1d(values, 2.3 / np.sqrt(12), axis=0, truncate=12)
        result = uniform_filter1d(result, size=5, axis=1, mode="reflect")
        return uniform_filter1d(result, size=5, axis=2, mode="reflect")

    density = spatial_smooth(mask.astype(float))
    assert density.min() > 1e-3
    expected_mean = spatial_smooth(finite_data) / density
    expected_variance = spatial_smooth(finite_data**2) / density - expected_mean**2
    assert expected_variance.min() > 0
    kwargs = {
        "width": (2.3, 3.7, 5.1),
        "kernel_type": ("gaussian", "boxcar_discrete", "boxcar_discrete"),
        "restore_input_nan": False,
    }
    np.testing.assert_allclose(dct_mean(data, **kwargs), expected_mean, atol=2e-14)
    np.testing.assert_allclose(dct_variance(data, **kwargs), expected_variance, atol=2e-14)
    np.testing.assert_allclose(dct_std(data, **kwargs)**2, expected_variance, atol=2e-14)


@pytest.mark.parametrize("width", [1.0, 2.0, 3.0, 8.0])
@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
def test_gaussian_moments_match_positive_weighted_squared_deviations(
    width: float, boundary: str
) -> None:
    """Masked variance agrees with an independent nonnegative central-moment sum."""
    n = 33
    data = np.random.default_rng(1409).uniform(-2, 3, n)
    data[::5] = np.nan
    mask = np.isfinite(data)
    matrix = gaussian_filter1d(
        np.eye(n), width / np.sqrt(12), axis=0,
        mode="wrap" if boundary == "periodic" else "reflect", truncate=12,
    )
    weights = matrix * mask
    density = weights.sum(axis=1)
    assert np.min(density) > 1e-3
    weights /= density[:, None]
    finite_data = np.where(mask, data, 0)
    mean = weights @ finite_data
    variance = np.sum(weights * (finite_data[None, :] - mean[:, None])**2, axis=1)
    kwargs = {
        "coordinates": "polar", "axis_types": ("linear",), "width_units": "grid",
        "boundary": boundary, "restore_input_nan": False,
    }
    np.testing.assert_allclose(dct_mean(data, width, **kwargs), mean, atol=2e-14)
    # Transform roundoff is amplified by normalization at the sparsest support.
    np.testing.assert_allclose(dct_variance(data, width, **kwargs), variance, atol=1e-12)
    np.testing.assert_allclose(dct_std(data, width, **kwargs)**2, variance, atol=1e-12)


@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
def test_sparse_gaussian_prefill_and_fallback_preserve_data_bounds(boundary: str) -> None:
    """Low-support fallback does not introduce substantive signed-weight overshoots."""
    data = np.full((21, 2), np.nan)
    data[2] = [2.0, 30.0]
    data[18] = [5.0, 40.0]
    kwargs = {
        "coordinates": "polar", "axis_types": ("linear", "batch"),
        "width_units": "grid", "boundary": boundary,
    }
    for actual in (
        dct_mean(data, 2, restore_input_nan=False, **kwargs),
        dct_prefill(data, 2, **kwargs),
    ):
        assert np.all(np.isfinite(actual))
        assert np.all(actual >= np.array([2.0, 30.0]) - 1e-11)
        assert np.all(actual <= np.array([5.0, 40.0]) + 1e-11)

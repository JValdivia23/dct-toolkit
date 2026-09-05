import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from dct_toolkit.core import dct_convolve_1d, get_dct_transfer_function
from dct_toolkit.stats import dct_count, dct_mean, dct_prefill, dct_std, dct_variance


def test_nd_mean_matches_axiswise_reference():
    """N-D normalized-convolution mean should match explicit axis-wise reference."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 8, 6))
    data[2:5, 3:6, 1:4] = np.nan

    mean = dct_mean(data, width=3.0, coordinates="cartesian", restore_input_nan=False)

    finite = np.isfinite(mean)
    assert np.any(finite)
    assert np.all(np.isfinite(mean[finite]))


def test_nd_variance_std_count_are_finite_and_consistent():
    """N-D variance/std/count should be finite and algebraically consistent."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((7, 6, 5, 4))
    data[1:3, :, 2:4, :] = np.nan

    mask = np.isfinite(data)
    count = dct_count(mask, width=2.5, coordinates="cartesian", restore_input_nan=False)
    var = dct_variance(
        data, width=2.5, coordinates="cartesian", restore_input_nan=False
    )
    std = dct_std(data, width=2.5, coordinates="cartesian", restore_input_nan=False)

    assert np.all(np.isfinite(count))
    assert np.nanmin(count) >= 0.0
    assert np.all(np.isfinite(var))
    assert np.nanmin(var) >= 0.0
    assert np.all(np.isfinite(std))
    assert np.allclose(std, np.sqrt(var), atol=1e-12)


def _hybrid_reference(data: np.ndarray) -> np.ndarray:
    """Smooth z spectrally and apply independent spatial boxcars along y/x."""
    transfer = get_dct_transfer_function(data.shape[0], "gaussian", 3.0)
    result = dct_convolve_1d(data, transfer, axis=0)
    result = uniform_filter1d(result, size=5, axis=1, mode="reflect")
    return uniform_filter1d(result, size=3, axis=2, mode="reflect")


@pytest.mark.parametrize("boxcar", ["boxcar", "boxcar_discrete"])
@pytest.mark.parametrize("restore_input_nan", [True, False])
def test_hybrid_statistics_match_normalized_spatial_reference(
    boxcar: str, restore_input_nan: bool
) -> None:
    """Mixed-kernel moments and counts use the same kernel for data and mask."""
    rng = np.random.default_rng(41)
    data = rng.standard_normal((9, 11, 13))
    mask = rng.random(data.shape) > 0.2
    data[~mask] = np.nan
    original = data.copy()
    filled = np.where(mask, data, 0.0)
    density = _hybrid_reference(mask.astype(float))
    assert np.min(density) > 0.35

    mean = _hybrid_reference(filled) / density
    variance = np.maximum(_hybrid_reference(filled**2) / density - mean**2, 0.0)
    count = np.clip(density, 0.0, 1.0) * (3.0 * 5.0 * 3.0)
    for func, values, expected in (
        (dct_mean, data, mean),
        (dct_variance, data, variance),
        (dct_std, data, np.sqrt(variance)),
        (dct_count, mask, count),
    ):
        if restore_input_nan:
            expected = np.where(mask, expected, np.nan)
        result = func(
            values,
            width=(3.0, 5.0, 3.0),
            kernel_type=("gaussian", boxcar, boxcar),
            restore_input_nan=restore_input_nan,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(data, original)


def test_hybrid_prefill_and_density_gate_match_reference() -> None:
    """Prefill and density thresholds use the requested mixed kernel."""
    rng = np.random.default_rng(43)
    data = rng.standard_normal((9, 11, 13))
    mask = rng.random(data.shape) > 0.2
    data[~mask] = np.nan
    density = _hybrid_reference(mask.astype(float))
    assert np.min(density) > 0.35
    mean = _hybrid_reference(np.where(mask, data, 0.0)) / density
    kwargs = {"width": (3.0, 5.0, 3.0), "kernel_type": ("gaussian", "boxcar", "boxcar")}

    filled = dct_prefill(data, max_iter=1, **kwargs)
    np.testing.assert_allclose(filled, np.where(mask, data, mean), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(filled[mask], data[mask])

    threshold = 0.8
    assert np.any(density < threshold) and np.any(density >= threshold)
    gated = dct_mean(data, restore_input_nan=False, min_effective_density=threshold, **kwargs)
    expected = np.where(density >= threshold, mean, np.nan)
    np.testing.assert_allclose(gated, expected, rtol=1e-12, atol=1e-12)


def test_hybrid_statistics_preserve_all_nan_behavior() -> None:
    """Mixed kernels retain existing behavior when no valid support exists."""
    data = np.full((5, 7, 9), np.nan)
    kwargs = {"width": 3.0, "kernel_type": ("gaussian", "boxcar", "boxcar")}
    for func in (dct_mean, dct_variance, dct_std, dct_prefill):
        assert np.all(np.isnan(func(data, **kwargs)))
    count = dct_count(np.isfinite(data), restore_input_nan=False, **kwargs)
    np.testing.assert_array_equal(count, np.zeros(data.shape))

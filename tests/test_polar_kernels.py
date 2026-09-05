"""Independent references for mixed polar kernels and normalized statistics."""

from itertools import product
from typing import Callable, Tuple

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from dct_toolkit import dct_smooth
from dct_toolkit.core import dct_convolve_1d, get_dct_transfer_function
from dct_toolkit.polar import (
    compute_polar_transfer_functions,
    compute_polar_transfer_functions_v2,
    smooth_polar,
)
from dct_toolkit.stats import dct_count, dct_mean, dct_prefill, dct_std, dct_variance


KERNELS = ("gaussian", "boxcar", "boxcar_discrete")
MIXED_PAIRS = [(az, rg) for az, rg in product(KERNELS, repeat=2) if az != rg]


def _odd_window(width: float) -> int:
    """Return the odd integer window specified by the public width convention."""
    size = max(1, int(np.round(width)))
    return size + 1 if size % 2 == 0 else size


def _polar_reference(
    data: np.ndarray,
    widths: Tuple[float, float],
    kernels: Tuple[str, str],
    az_boundary: str,
) -> np.ndarray:
    """Filter columns independently, using spatial averaging for discrete boxcars."""
    n_az, n_range = data.shape
    dtheta = 2 * np.pi / n_az
    step1 = np.empty_like(data, dtype=float)
    for j in range(n_range):
        width = widths[0] / ((j + 1) * dtheta)
        column = data[:, j]
        if kernels[0] == "boxcar_discrete":
            mode = "wrap" if az_boundary == "periodic" else "reflect"
            step1[:, j] = uniform_filter1d(column, size=_odd_window(width), mode=mode)
        elif az_boundary == "reflective":
            transfer = get_dct_transfer_function(n_az, kernels[0], width)
            step1[:, j] = dct_convolve_1d(column, transfer)
        else:
            omega = 2 * np.pi * np.fft.rfftfreq(n_az)
            if kernels[0] == "gaussian":
                transfer = np.exp(-0.5 * (omega * width / np.sqrt(12)) ** 2)
            else:
                transfer = np.ones(omega.size)
                transfer[1:] = (
                    np.sin(width * omega[1:] / 2) / (width * np.sin(omega[1:] / 2))
                )
            step1[:, j] = np.fft.irfft(np.fft.rfft(column) * transfer, n=n_az)

    if kernels[1] == "boxcar_discrete":
        return uniform_filter1d(step1, size=_odd_window(widths[1]), axis=1, mode="reflect")
    transfer = get_dct_transfer_function(n_range, kernels[1], widths[1])
    return dct_convolve_1d(step1, transfer, axis=1)


@pytest.mark.parametrize("kernels", MIXED_PAIRS)
@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
@pytest.mark.parametrize("n_az", [31, 32])
@pytest.mark.parametrize("scalar_width", [False, True])
def test_mixed_polar_matches_columnwise_reference(
    kernels: Tuple[str, str], az_boundary: str, n_az: int, scalar_width: bool
) -> None:
    """Mixed filtering preserves axis order and range adaptation in both modes."""
    data = np.random.default_rng(53).standard_normal((n_az, 13))
    original = data.copy()
    widths = (4.5, 4.5) if scalar_width else (4.5, 3.0)
    result = smooth_polar(
        data,
        width_pixels=widths[0] if scalar_width else widths,
        az_res_deg=360.0 / n_az,
        az_boundary=az_boundary,
        kernel_type=kernels,
    )
    expected = _polar_reference(data, widths, kernels, az_boundary)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
@pytest.mark.parametrize("container", [tuple, list, np.array])
def test_polar_string_matches_repeated_pair(
    kernel: str, az_boundary: str, container: Callable
) -> None:
    """Strings broadcast to both axes and all documented sequence types work."""
    data = np.random.default_rng(59).standard_normal((32, 11))
    kwargs = {"width_pixels": (5.0, 3.0), "az_res_deg": 11.25, "az_boundary": az_boundary}
    result = smooth_polar(data, kernel_type=container([kernel, kernel]), **kwargs)
    expected = _polar_reference(data, (5.0, 3.0), (kernel, kernel), az_boundary)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result, smooth_polar(data, kernel_type=kernel, **kwargs))


@pytest.mark.parametrize("n_az", [15, 16])
@pytest.mark.parametrize("beam_width", [0.25, 1.0, 2.0, 2.5, 4.6, 45.0])
def test_periodic_discrete_boxcar_matches_explicit_circular_average(
    n_az: int, beam_width: float
) -> None:
    """Discrete azimuth boxcars wrap correctly, including multi-revolution windows."""
    data = np.zeros((n_az, 1))
    data[0, 0] = 1.0
    width_azimuth = beam_width * (2 * np.pi / n_az)
    result = smooth_polar(
        data,
        width_pixels=(width_azimuth, 1.0),
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
        kernel_type=("boxcar_discrete", "boxcar"),
    )
    size = _odd_window(beam_width)
    half = size // 2
    expected = sum(np.roll(data, shift, axis=0) for shift in range(-half, half + 1)) / size
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.sum(result), 1.0, atol=1e-12)


@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (31, 13)])
def test_mixed_polar_preserves_constants_and_singleton_axes(
    az_boundary: str, shape: Tuple[int, int]
) -> None:
    """DC preservation holds for all mixed pairs, including singleton axes."""
    data = np.full(shape, 7, dtype=np.int64)
    for kernels in MIXED_PAIRS:
        result = smooth_polar(
            data, width_pixels=(5.0, 3.0), kernel_type=kernels, az_boundary=az_boundary
        )
        np.testing.assert_allclose(result, data, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kernels, message",
    [
        ([], "length-2"),
        (["gaussian"], "length-2"),
        (["boxcar"] * 3, "length-2"),
        ([["boxcar", "gaussian"]], "1-D sequence"),
        (None, "string or 1-D sequence"),
        (["gaussian", 2], "entries must be strings.*axis 1"),
        (["unknown", "gaussian"], "Unknown kernel type.*axis 0"),
        (["gaussian", "unknown"], "Unknown kernel type.*axis 1"),
    ],
)
@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
def test_invalid_polar_kernel_specifications(
    kernels: object, message: str, az_boundary: str
) -> None:
    """Polar validation rejects malformed pairs and never substitutes another kernel."""
    with pytest.raises(ValueError, match=message):
        smooth_polar(np.ones((16, 9)), 3.0, kernel_type=kernels, az_boundary=az_boundary)


@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
def test_polar_transfer_alias_supports_mixed_kernels(az_boundary: str) -> None:
    """The compatibility transfer-function entry point forwards kernel pairs."""
    kwargs = {
        "shape": (31, 13), "az_res_deg": 360.0 / 31, "width_pixels": (4.5, 3.0),
        "kernel_type": ("boxcar_discrete", "gaussian"), "az_boundary": az_boundary,
    }
    result = compute_polar_transfer_functions_v2(**kwargs)
    expected = compute_polar_transfer_functions(**kwargs)
    for actual, reference in zip(result, expected):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
@pytest.mark.parametrize("kernels", [("boxcar_discrete", "gaussian"), ("gaussian", "boxcar")])
def test_mixed_polar_statistics_prefill_and_wrapper_match_reference(
    az_boundary: str, kernels: Tuple[str, str]
) -> None:
    """Statistics, gates, and prefill share the full azimuth-then-range operator."""
    data = np.random.default_rng(61).standard_normal((32, 13))
    widths = (4.0, 3.0)
    kwargs = {
        "coordinates": "polar", "width": widths, "kernel_type": kernels,
        "az_res_deg": 11.25, "az_boundary": az_boundary,
    }
    finite_result = dct_smooth(data, **kwargs)
    np.testing.assert_allclose(
        finite_result, _polar_reference(data, widths, kernels, az_boundary),
        rtol=1e-12, atol=1e-12,
    )

    data[::7, 4] = np.nan
    data[2::11, 8] = np.nan
    mask = np.isfinite(data)
    zero_filled = np.where(mask, data, 0.0)
    density = _polar_reference(mask.astype(float), widths, kernels, az_boundary)
    assert np.min(density) > 0.35
    mean = _polar_reference(zero_filled, widths, kernels, az_boundary) / density
    second_moment = _polar_reference(zero_filled**2, widths, kernels, az_boundary) / density
    variance = np.maximum(second_moment - mean**2, 0.0)
    area = widths[1] * widths[0] / (np.arange(1, data.shape[1] + 1) * (2 * np.pi / 32))
    count = np.clip(density, 0.0, 1.0) * area

    for func, values, expected in (
        (dct_mean, data, mean), (dct_variance, data, variance),
        (dct_std, data, np.sqrt(variance)), (dct_count, mask, count),
    ):
        for restore in (False, True):
            target = np.where(mask, expected, np.nan) if restore else expected
            result = func(values, restore_input_nan=restore, **kwargs)
            np.testing.assert_allclose(result, target, rtol=1e-12, atol=1e-12)

    filled = dct_prefill(data, max_iter=1, **kwargs)
    expected_filled = np.where(mask, data, mean)
    np.testing.assert_allclose(filled, expected_filled, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(filled[mask], data[mask])
    expected_smoothed = _polar_reference(expected_filled, widths, kernels, az_boundary)
    result = dct_smooth(data, prefill_max_iter=1, **kwargs)
    np.testing.assert_allclose(
        result, np.where(mask, expected_smoothed, np.nan), rtol=1e-12, atol=1e-12
    )

    threshold = 0.98
    assert np.any(density < threshold) and np.any(density >= threshold)
    gated = dct_mean(data, restore_input_nan=False, min_effective_density=threshold, **kwargs)
    np.testing.assert_allclose(
        gated, np.where(density >= threshold, mean, np.nan), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
def test_mixed_polar_all_nan_support(az_boundary: str) -> None:
    """An absent validity mask retains existing NaN and zero-count behavior."""
    data = np.full((16, 9), np.nan)
    kwargs = {
        "coordinates": "polar", "width": (5.0, 3.0),
        "kernel_type": ("boxcar_discrete", "gaussian"), "az_boundary": az_boundary,
    }
    for func in (dct_smooth, dct_mean, dct_variance, dct_std, dct_prefill):
        assert np.all(np.isnan(func(data, **kwargs)))
    count = dct_count(np.isfinite(data), restore_input_nan=False, **kwargs)
    np.testing.assert_array_equal(count, np.zeros(data.shape))

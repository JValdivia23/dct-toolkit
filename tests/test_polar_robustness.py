"""Regression tests for polar fallback boundaries and azimuth spacing validation."""

from typing import Callable, Tuple

import numpy as np
import pytest

from dct_toolkit import dct_count, dct_mean, dct_prefill, dct_smooth, dct_std, dct_variance
from dct_toolkit.polar import (
    compute_polar_transfer_functions,
    compute_polar_transfer_functions_v2,
    smooth_polar,
)


@pytest.mark.parametrize("n_az", [15, 16])
@pytest.mark.parametrize(
    "kernels", [("boxcar_discrete", "gaussian"), ("gaussian", "boxcar_discrete")]
)
def test_sparse_periodic_prefill_and_smoothing_rotate_with_data(
    n_az: int, kernels: Tuple[str, str]
) -> None:
    """Sparse sweeps use circular nearest neighbors, independent of the seam location."""
    data = np.full((n_az, 1), np.nan)
    data[4, 0], data[-1, 0] = 0.0, 10.0
    kwargs = {
        "width": (2.05 * 2 * np.pi / n_az, 1.0), "coordinates": "polar",
        "az_res_deg": 360.0 / n_az, "az_boundary": "periodic", "kernel_type": kernels,
    }
    seeds = (4, n_az - 1)
    expected = np.empty_like(data)
    for i in range(n_az):
        # At equal circular distance, prefer the preceding beam.
        nearest = min(seeds, key=lambda j: (min(abs(i - j), n_az - abs(i - j)), (i - j) % n_az))
        expected[i, 0] = data[nearest, 0]
    filled = dct_prefill(data, **kwargs)
    np.testing.assert_array_equal(filled, expected)
    np.testing.assert_array_equal(filled[np.isfinite(data)], data[np.isfinite(data)])

    for func in (dct_prefill, dct_smooth, dct_mean):
        extra = {"restore_input_nan": False} if func is dct_mean else {}
        original = func(data, **kwargs, **extra)
        for shift in range(1, n_az):
            rotated = func(np.roll(data, shift, axis=0), **kwargs, **extra)
            np.testing.assert_allclose(
                np.roll(rotated, -shift, axis=0), original, rtol=1e-11, atol=1e-11
            )
        if func is dct_smooth:
            np.testing.assert_array_equal(np.isnan(original), np.isnan(data))


@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
def test_polar_nearest_fallback_keeps_range_nonperiodic(az_boundary: str) -> None:
    """The first and last range gates do not become neighbors in a periodic sweep."""
    data = np.tile([np.nan, 1.0, np.nan, np.nan, 9.0], (8, 1))
    filled = dct_prefill(
        data, width=(0.01, 1.0), coordinates="polar", az_res_deg=45.0,
        az_boundary=az_boundary, kernel_type="boxcar_discrete", max_iter=1,
    )
    np.testing.assert_array_equal(filled, np.tile([1.0, 1.0, 1.0, 9.0, 9.0], (8, 1)))


@pytest.mark.parametrize("az_boundary, expected", [("reflective", 30.0), ("periodic", 10.0)])
def test_targeted_polar_prefill_global_fallback_respects_boundary(
    az_boundary: str, expected: float
) -> None:
    """A target lacking any same-row/column support uses the correct global distance."""
    data = np.full((8, 4), np.nan)
    data[7, 0], data[3, 0] = 10.0, 30.0
    target = np.zeros(data.shape, dtype=bool)
    target[0, 3] = True
    kwargs = {
        "width": (0.01, 1.0), "coordinates": "polar", "az_res_deg": 45.0,
        "az_boundary": az_boundary, "kernel_type": "boxcar_discrete", "max_iter": 1,
    }
    result = dct_prefill(data, fill_mask=target, **kwargs)
    assert result[0, 3] == expected
    np.testing.assert_array_equal(result[~target], data[~target])
    if az_boundary == "periodic":
        for shift in range(1, data.shape[0]):
            rotated = dct_prefill(
                np.roll(data, shift, axis=0), fill_mask=np.roll(target, shift, axis=0), **kwargs
            )
            np.testing.assert_array_equal(np.roll(rotated, -shift, axis=0), result)


@pytest.mark.parametrize("shape", [(1, 4), (8, 1)])
def test_periodic_fallback_with_one_seed_and_singleton_axis(shape: Tuple[int, int]) -> None:
    """One finite observation can fill a sweep even when an axis has length one."""
    data = np.full(shape, np.nan)
    data[-1, -1] = 7.0
    result = dct_prefill(
        data, width=(0.01, 1.0), coordinates="polar", az_res_deg=1.0,
        az_boundary="periodic", kernel_type="boxcar_discrete",
    )
    np.testing.assert_allclose(result, 7.0, rtol=1e-12, atol=1e-12)


INVALID_SPACINGS = [
    0.0, -1.0, np.nan, np.inf, -np.inf, [], [1.0], [1.0, 2.0],
    np.array([[1.0]]), None, "1.0", 1.0j, True,
]


@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
@pytest.mark.parametrize(
    "func", [
        compute_polar_transfer_functions, compute_polar_transfer_functions_v2, smooth_polar,
        dct_smooth, dct_count, dct_mean, dct_prefill, dct_variance, dct_std,
    ],
)
def test_polar_entry_points_reject_invalid_azimuth_spacing(
    func: Callable, az_boundary: str
) -> None:
    """Invalid spacing raises a clear ValueError before spectral or area arithmetic."""
    data = np.ones((8, 4))
    data[2, 1] = np.nan
    kwargs = {"az_boundary": az_boundary, "kernel_type": ("boxcar_discrete", "gaussian")}
    if func in (compute_polar_transfer_functions, compute_polar_transfer_functions_v2):
        args = (data.shape,)
        kwargs["width_pixels"] = 2.0
    elif func is smooth_polar:
        args = (data,)
        kwargs["width_pixels"] = 2.0
    else:
        args = (np.isfinite(data) if func is dct_count else data,)
        kwargs.update(width=2.0, coordinates="polar")
    for spacing in INVALID_SPACINGS:
        with pytest.raises(ValueError, match="az_res_deg.*finite.*positive.*scalar"):
            func(*args, az_res_deg=spacing, **kwargs)


@pytest.mark.parametrize("func", [dct_smooth, dct_prefill])
@pytest.mark.parametrize("value", [1.0, np.nan])
def test_polar_spacing_validation_precedes_early_returns(func: Callable, value: float) -> None:
    """No-op prefill and all-NaN wrapper inputs still reject invalid geometry."""
    data = np.full((8, 4), value)
    for spacing in INVALID_SPACINGS:
        with pytest.raises(ValueError, match="az_res_deg.*finite.*positive.*scalar"):
            func(data, width=2.0, coordinates="polar", az_res_deg=spacing)


@pytest.mark.parametrize("spacing", [0.5, np.float32(0.5), np.array(0.5), np.int64(2)])
@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
def test_valid_scalar_spacing_preserves_smoothing_and_count_area(
    spacing: float, az_boundary: str
) -> None:
    """Fractional and NumPy scalar spacings retain the same geometry in both operations."""
    data = np.ones((8, 4))
    kwargs = {
        "coordinates": "polar", "width": (2.0, 3.0), "az_res_deg": spacing,
        "az_boundary": az_boundary, "kernel_type": ("boxcar_discrete", "gaussian"),
    }
    np.testing.assert_allclose(dct_smooth(data, **kwargs), data, rtol=1e-12, atol=1e-12)
    area = 6.0 / (np.arange(1, 5) * np.deg2rad(float(spacing)))
    expected = np.broadcast_to(area, data.shape)
    np.testing.assert_allclose(dct_count(data.astype(bool), **kwargs), expected, rtol=1e-12)

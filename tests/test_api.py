import numpy as np

import dct_toolkit as dct


def test_dct_smooth_nan_safe_cartesian_restores_mask():
    """dct_smooth should prefill NaNs, smooth, and restore original NaN mask."""
    data = np.arange(25, dtype=float).reshape(5, 5)
    data[1, 2] = np.nan
    data[3, 4] = np.nan

    out = dct.dct_smooth(data, width=2.0, coordinates="cartesian")

    assert out.shape == data.shape
    assert np.all(np.isnan(out[np.isnan(data)]))
    assert np.all(np.isfinite(out[~np.isnan(data)]))


def test_dct_smooth_nan_safe_polar_restores_mask():
    """dct_smooth should remain finite on valid polar cells with NaN inputs."""
    n_az, n_range = 120, 80
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    rng = np.arange(1, n_range + 1)
    AZ, R = np.meshgrid(az, rng, indexing="ij")
    data = np.sin(AZ) * np.exp(-R / 40.0)

    data[20:50, 10:20] = np.nan
    data[10, :] = np.nan

    out = dct.dct_smooth(
        data,
        width=5.0,
        coordinates="polar",
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
    )

    assert out.shape == data.shape
    assert np.all(np.isnan(out[np.isnan(data)]))
    assert np.all(np.isfinite(out[~np.isnan(data)]))


def test_dct_smooth_all_nan_returns_all_nan():
    """All-NaN input has no support and remains all-NaN."""
    data = np.full((10, 12), np.nan)
    out = dct.dct_smooth(data, width=3.0, coordinates="cartesian")
    assert np.all(np.isnan(out))


def test_dct_smooth_prefill_max_iter_alias_max_iter_kwarg():
    """max_iter kwarg should map to prefill_max_iter in dct_smooth."""
    data = np.array([1.0, np.nan, np.nan, 4.0], dtype=float)
    out = dct.dct_smooth(data, width=2.0, max_iter=1)

    assert np.isnan(out[1]) and np.isnan(out[2])
    assert np.isfinite(out[0]) and np.isfinite(out[3])

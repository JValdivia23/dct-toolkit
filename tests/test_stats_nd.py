import numpy as np

from dct_toolkit.stats import dct_count, dct_mean, dct_std, dct_variance


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

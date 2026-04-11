import numpy as np
import pytest
from dct_toolkit.stats import dct_count, dct_mean, dct_prefill, dct_variance, dct_std


@pytest.fixture
def uniform_with_gaps():
    """Uniform data with 50% random gaps."""
    data = np.ones(1000)
    mask = np.random.rand(1000) > 0.5
    data[~mask] = np.nan
    return data, mask


@pytest.fixture
def normal_with_gaps():
    """Normal distribution with known variance and gaps."""
    np.random.seed(42)
    data = np.random.randn(2000)  # True variance = 1.0
    data[::3] = np.nan  # 33% gaps
    return data


def test_count_uniform_gaps():
    """Count should reflect gap density."""
    np.random.seed(42)
    mask = np.random.rand(1000) > 0.5  # 50% gaps
    width = 20.0
    count = dct_count(mask, width=width)

    # In interior, should be approximately density * width
    # 0.5 * 20 = 10
    interior_mean = np.mean(count[20:-20])
    assert np.abs(interior_mean - 10.0) < 1.0


def test_mean_uniform_with_gaps(uniform_with_gaps):
    """Mean of uniform data should be 1.0 even with 50% gaps."""
    data, mask = uniform_with_gaps
    mean = dct_mean(data, width=20.0)
    # The mean should be very close to 1.0 everywhere (ignoring edges)
    assert np.allclose(mean[20:-20], 1.0, atol=0.05)


def test_variance_normal_distribution(normal_with_gaps):
    """Variance of normal data should be ~1.0 even with gaps."""
    var = dct_variance(normal_with_gaps, width=30.0)

    # Check interior mean variance
    mean_var = np.mean(var[30:-30])
    # Expect ~1.0. Tolerance 0.2 allows for sample variance fluctuation
    assert np.abs(mean_var - 1.0) < 0.2


def test_variance_uniform():
    """Variance of uniform [0,1] should be 1/12 ≈ 0.083."""
    np.random.seed(42)
    data = np.random.rand(5000)  # Uniform [0,1]
    data[::4] = np.nan  # 25% gaps

    var = dct_variance(data, width=50.0)
    mean_var = np.mean(var[50:-50])

    expected = 1.0 / 12.0
    assert np.abs(mean_var - expected) < 0.01


def test_std_consistency(normal_with_gaps):
    """std = sqrt(variance)."""
    var = dct_variance(normal_with_gaps, width=30.0)
    std = dct_std(normal_with_gaps, width=30.0)
    assert np.allclose(std, np.sqrt(var))


def test_all_nan_handling():
    """Should handle all-NaN regions gracefully."""
    data = np.full(100, np.nan)
    mean = dct_mean(data, width=10.0)
    assert np.all(np.isnan(mean))


def test_single_point_influence():
    """Single point should influence neighbors."""
    data = np.full(100, np.nan)
    data[50] = 10.0
    mean = dct_mean(data, width=10.0)

    # At index 50, mean should be exactly the value (normalized conv property)
    assert np.isclose(mean[50], 10.0)
    # Nearby points should also be close to 10.0 (constant model assumption)
    assert np.isclose(mean[51], 10.0)


def test_integer_input_matches_float_mean():
    """Integer inputs should produce the same mean as float inputs."""
    data_int = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    data_float = data_int.astype(float)

    mean_int = dct_mean(data_int, width=2.0)
    mean_float = dct_mean(data_float, width=2.0)

    assert np.issubdtype(mean_int.dtype, np.floating)
    assert np.allclose(mean_int, mean_float, atol=1e-12)


def test_integer_input_matches_float_variance():
    """Integer inputs should produce the same variance as float inputs."""
    data_int = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    data_float = data_int.astype(float)

    var_int = dct_variance(data_int, width=2.0)
    var_float = dct_variance(data_float, width=2.0)

    assert np.issubdtype(var_int.dtype, np.floating)
    assert np.allclose(var_int, var_float, atol=1e-12)


def test_mask_shape_validation():
    """Mask shape must match data shape for normalized convolution."""
    data = np.arange(10, dtype=float)
    bad_mask = np.ones((10, 1), dtype=bool)

    with pytest.raises(ValueError, match="Mask shape"):
        dct_mean(data, width=2.0, mask=bad_mask)


def test_prefill_fills_nan_gap_1d():
    """Iterative prefill should recover finite values in a contiguous 1D gap."""
    x = np.linspace(0, 2 * np.pi, 200)
    truth = np.sin(x)
    data = truth.copy()
    data[80:120] = np.nan

    filled = dct_prefill(data, width=10.0, max_iter=5)
    assert np.all(np.isfinite(filled[80:120]))
    mae = np.mean(np.abs(filled[80:120] - truth[80:120]))
    assert mae < 0.35


def test_prefill_preserves_non_target_values():
    """Values outside fill targets should remain unchanged."""
    data = np.linspace(0, 1, 100)
    fill_mask = np.zeros(100, dtype=bool)
    fill_mask[30:40] = True

    filled = dct_prefill(data, width=6.0, fill_mask=fill_mask, max_iter=2)
    assert np.allclose(filled[~fill_mask], data[~fill_mask])


def test_prefill_all_nan_returns_all_nan():
    """All-NaN input remains all-NaN because no valid support exists."""
    data = np.full((20, 30), np.nan)
    filled = dct_prefill(data, width=5.0, max_iter=3)
    assert np.all(np.isnan(filled))


def test_prefill_shape_validation_fill_mask():
    """fill_mask shape must match data shape."""
    data = np.arange(12, dtype=float).reshape(3, 4)
    bad_fill = np.zeros((3, 5), dtype=bool)

    with pytest.raises(ValueError, match="fill_mask shape"):
        dct_prefill(data, width=2.0, fill_mask=bad_fill)


def test_prefill_invalid_parameters():
    """Width and iteration count must be valid."""
    data = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="width must be > 0"):
        dct_prefill(data, width=0.0)
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        dct_prefill(data, width=2.0, max_iter=0)


def test_prefill_polar_periodic_smoke():
    """Prefill should operate in polar mode with periodic azimuth."""
    n_az, n_range = 120, 60
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    rng = np.arange(1, n_range + 1, dtype=float)
    AZ, R = np.meshgrid(az, rng, indexing="ij")
    truth = np.sin(AZ) * np.exp(-R / 40.0)

    data = truth.copy()
    data[30:50, 20:30] = np.nan

    filled = dct_prefill(
        data,
        width=6.0,
        coordinates="polar",
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
        max_iter=4,
    )
    assert np.all(np.isfinite(filled[30:50, 20:30]))


def test_prefill_then_smooth_reduces_boundary_jump():
    """Prefill before smoothing should reduce edge artifacts near a gap boundary."""
    x = np.linspace(0, 4 * np.pi, 300)
    truth = np.sin(x)
    data = truth + 0.2 * np.random.default_rng(0).standard_normal(x.size)
    data[110:170] = np.nan

    direct = dct_mean(data, width=10.0)
    prefilled = dct_prefill(data, width=10.0, max_iter=5)
    smooth_prefilled = dct_mean(prefilled, width=10.0)

    # Compare discontinuity at gap edge (left boundary around index 109/110)
    jump_direct = abs(direct[109] - direct[110])
    jump_prefilled = abs(smooth_prefilled[109] - smooth_prefilled[110])
    assert jump_prefilled <= jump_direct

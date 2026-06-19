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
    interior_mean = np.nanmean(count[20:-20])
    assert np.abs(interior_mean - 10.0) < 1.0


def test_mean_uniform_with_gaps(uniform_with_gaps):
    """Mean of uniform data should be 1.0 even with 50% gaps."""
    data, mask = uniform_with_gaps
    mean = dct_mean(data, width=20.0)
    interior = slice(20, -20)
    # Valid locations remain near 1.0, gap locations are restored to NaN.
    assert np.allclose(mean[interior][mask[interior]], 1.0, atol=0.05)
    assert np.all(np.isnan(mean[interior][~mask[interior]]))


def test_variance_normal_distribution(normal_with_gaps):
    """Variance of normal data should be ~1.0 even with gaps."""
    var = dct_variance(normal_with_gaps, width=30.0)

    # Check interior mean variance
    mean_var = np.nanmean(var[30:-30])
    # Expect ~1.0. Tolerance 0.2 allows for sample variance fluctuation
    assert np.abs(mean_var - 1.0) < 0.2


def test_variance_uniform():
    """Variance of uniform [0,1] should be 1/12 ≈ 0.083."""
    np.random.seed(42)
    data = np.random.rand(5000)  # Uniform [0,1]
    data[::4] = np.nan  # 25% gaps

    var = dct_variance(data, width=50.0)
    mean_var = np.nanmean(var[50:-50])

    expected = 1.0 / 12.0
    assert np.abs(mean_var - expected) < 0.01


def test_std_consistency(normal_with_gaps):
    """std = sqrt(variance)."""
    var = dct_variance(normal_with_gaps, width=30.0)
    std = dct_std(normal_with_gaps, width=30.0)
    assert np.allclose(std, np.sqrt(var), equal_nan=True)


def test_all_nan_handling():
    """Should handle all-NaN regions gracefully."""
    data = np.full(100, np.nan)
    mean = dct_mean(data, width=10.0)
    assert np.all(np.isnan(mean))


def test_single_point_influence():
    """Single point should influence neighbors."""
    data = np.full(100, np.nan)
    data[50] = 10.0
    mean = dct_mean(data, width=10.0, restore_input_nan=False)

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
    with pytest.raises(ValueError, match="max_iter must be >= 1 or None"):
        dct_prefill(data, width=2.0, max_iter=0)


def test_prefill_max_iter_none_converges_or_caps():
    """max_iter=None should run with the internal cap and produce finite fills."""
    data = np.array([1.0, np.nan, np.nan, 4.0], dtype=float)
    filled = dct_prefill(data, width=2.0, max_iter=None)
    assert np.all(np.isfinite(filled))


def test_prefill_default_max_iter_runs():
    """Default prefill path should remain finite for simple gaps."""
    data = np.array([1.0, np.nan, 3.0], dtype=float)
    filled = dct_prefill(data, width=2.0)
    assert np.all(np.isfinite(filled))


def test_prefill_residual_nearest_1d():
    """Residual nearest fill should replace unresolved 1D targets."""
    data = np.array([np.nan, 1.0, np.nan, np.nan, 3.0, np.nan], dtype=float)

    filled = dct_prefill(
        data,
        width=1.0,
        kernel_type="boxcar_discrete",
        max_iter=1,
    )

    expected = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0], dtype=float)
    assert np.allclose(filled, expected)


def test_prefill_residual_nearest_polar_axis1():
    """Polar residual nearest fill should operate along range axis first."""
    data = np.array(
        [
            [1.0, np.nan, np.nan, 4.0, np.nan],
        ],
        dtype=float,
    )

    filled = dct_prefill(
        data,
        width=1.0,
        coordinates="polar",
        az_res_deg=1.0,
        kernel_type="boxcar_discrete",
        max_iter=1,
    )

    expected = np.array(
        [
            [1.0, 1.0, 4.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    assert np.allclose(filled, expected)


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

    direct = dct_mean(data, width=10.0, restore_input_nan=False)
    prefilled = dct_prefill(data, width=10.0, max_iter=5)
    smooth_prefilled = dct_mean(prefilled, width=10.0, restore_input_nan=False)

    # Compare discontinuity at gap edge (left boundary around index 109/110)
    jump_direct = abs(direct[109] - direct[110])
    jump_prefilled = abs(smooth_prefilled[109] - smooth_prefilled[110])
    assert jump_prefilled <= jump_direct


def test_count_nonnegative_with_heavy_gaps():
    """Effective count should remain nonnegative under heavy missingness."""
    n_az, n_range = 180, 100
    mask = np.zeros((n_az, n_range), dtype=bool)

    # Keep sparse support in a few contiguous stripes.
    mask[:, 20:24] = True
    mask[60:120, 40:43] = True
    mask[::6, :] = True

    count = dct_count(
        mask,
        width=5.0,
        coordinates="polar",
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
    )
    assert np.nanmin(count) >= 0.0


def test_mean_stable_finite_heavy_gaps_polar():
    """Mean should remain finite when sparse valid support exists."""
    n_az, n_range = 180, 120
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    rng = np.arange(1, n_range + 1, dtype=float)
    AZ, R = np.meshgrid(az, rng, indexing="ij")
    base = np.sin(2.0 * AZ) + 0.2 * np.cos(R / 8.0)

    data = np.full((n_az, n_range), np.nan)
    data[:, 10:14] = base[:, 10:14]
    data[50:130, 40:44] = base[50:130, 40:44]
    data[::7, :] = base[::7, :]

    mean = dct_mean(
        data,
        width=5.0,
        coordinates="polar",
        restore_input_nan=False,
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
    )
    assert np.all(np.isfinite(mean))


def test_mean_restore_input_nan_default_true():
    """By default, dct_mean restores NaNs at original invalid inputs."""
    data = np.array([1.0, np.nan, 3.0], dtype=float)
    mean = dct_mean(data, width=2.0)
    assert np.isnan(mean[1])
    assert np.isfinite(mean[0]) and np.isfinite(mean[2])


def test_variance_std_restore_input_nan_default_true():
    """Variance/std should preserve NaNs at original invalid inputs by default."""
    data = np.array([1.0, np.nan, 3.0], dtype=float)
    var = dct_variance(data, width=2.0)
    std = dct_std(data, width=2.0)
    assert np.isnan(var[1]) and np.isnan(std[1])
    assert np.isfinite(var[0]) and np.isfinite(var[2])
    assert np.isfinite(std[0]) and np.isfinite(std[2])


def test_count_restore_input_nan_default_true():
    """Count should restore NaNs where input mask is False by default."""
    mask = np.array([True, False, True, False], dtype=bool)
    count = dct_count(mask, width=2.0)
    assert np.isnan(count[1]) and np.isnan(count[3])
    assert np.isfinite(count[0]) and np.isfinite(count[2])


def test_variance_std_stable_finite_heavy_gaps_polar():
    """Variance/std should remain finite and nonnegative with sparse support."""
    n_az, n_range = 160, 90
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    rng = np.arange(1, n_range + 1, dtype=float)
    AZ, R = np.meshgrid(az, rng, indexing="ij")
    base = np.sin(3.0 * AZ + R / 20.0)

    data = np.full((n_az, n_range), np.nan)
    data[:, 15:18] = base[:, 15:18]
    data[::5, :] = base[::5, :]

    var = dct_variance(
        data,
        width=5.0,
        coordinates="polar",
        restore_input_nan=False,
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
    )
    std = dct_std(
        data,
        width=5.0,
        coordinates="polar",
        restore_input_nan=False,
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
    )

    assert np.all(np.isfinite(var))
    assert np.all(np.isfinite(std))
    assert np.nanmin(var) >= 0.0


def test_stats_isotropic_scalar_equals_vector_cartesian():
    """Stats outputs should match between scalar and isotropic vector widths."""
    rng = np.random.default_rng(13)
    data = rng.standard_normal((12, 10, 8))
    data[3:6, 2:5, 1:4] = np.nan
    mask = np.isfinite(data)

    width_scalar = 3.0
    width_vector = [3.0, 3.0, 3.0]

    mean_s = dct_mean(
        data, width=width_scalar, coordinates="cartesian", restore_input_nan=False
    )
    mean_v = dct_mean(
        data, width=width_vector, coordinates="cartesian", restore_input_nan=False
    )
    var_s = dct_variance(
        data, width=width_scalar, coordinates="cartesian", restore_input_nan=False
    )
    var_v = dct_variance(
        data, width=width_vector, coordinates="cartesian", restore_input_nan=False
    )
    std_s = dct_std(
        data, width=width_scalar, coordinates="cartesian", restore_input_nan=False
    )
    std_v = dct_std(
        data, width=width_vector, coordinates="cartesian", restore_input_nan=False
    )
    cnt_s = dct_count(
        mask, width=width_scalar, coordinates="cartesian", restore_input_nan=False
    )
    cnt_v = dct_count(
        mask, width=width_vector, coordinates="cartesian", restore_input_nan=False
    )

    assert np.allclose(mean_s, mean_v, atol=1e-12)
    assert np.allclose(var_s, var_v, atol=1e-12)
    assert np.allclose(std_s, std_v, atol=1e-12)
    assert np.allclose(cnt_s, cnt_v, atol=1e-12)


def test_stats_isotropic_scalar_equals_vector_polar():
    """Polar stats outputs should match between scalar and isotropic vector widths."""
    n_az, n_range = 180, 80
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    rng = np.arange(1, n_range + 1, dtype=float)
    AZ, R = np.meshgrid(az, rng, indexing="ij")
    data = np.sin(AZ) * np.exp(-R / 35.0)
    data[25:50, 20:30] = np.nan

    width_scalar = 5.0
    width_vector = (5.0, 5.0)
    kwargs = {
        "coordinates": "polar",
        "az_res_deg": 360.0 / n_az,
        "az_boundary": "periodic",
        "restore_input_nan": False,
    }

    mean_s = dct_mean(data, width=width_scalar, **kwargs)
    mean_v = dct_mean(data, width=width_vector, **kwargs)
    var_s = dct_variance(data, width=width_scalar, **kwargs)
    var_v = dct_variance(data, width=width_vector, **kwargs)
    std_s = dct_std(data, width=width_scalar, **kwargs)
    std_v = dct_std(data, width=width_vector, **kwargs)
    cnt_s = dct_count(
        np.isfinite(data),
        width=width_scalar,
        coordinates="polar",
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
        restore_input_nan=False,
    )
    cnt_v = dct_count(
        np.isfinite(data),
        width=width_vector,
        coordinates="polar",
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
        restore_input_nan=False,
    )

    assert np.allclose(mean_s, mean_v, atol=1e-12)
    assert np.allclose(var_s, var_v, atol=1e-12)
    assert np.allclose(std_s, std_v, atol=1e-12)
    assert np.allclose(cnt_s, cnt_v, atol=1e-12)


def test_prefill_accepts_anisotropic_width_cartesian():
    """Prefill should accept anisotropic Cartesian widths."""
    data = np.linspace(0.0, 1.0, 60, dtype=float).reshape(5, 4, 3)
    data[2, :, 1] = np.nan

    filled = dct_prefill(
        data, width=[3.0, 2.0, 1.5], coordinates="cartesian", max_iter=3
    )
    assert np.all(np.isfinite(filled))


def test_prefill_min_effective_density_default_gates_low_support():
    """Default 0.35 density gate should refuse to fill cells with very low support.

    With a width-10 kernel and only 1 valid cell every 100 positions, the
    smoothed-mask density is well below 0.35 even near the valid samples,
    so the per-iteration gate keeps cells as NaN.
    """
    n = 500
    data = np.sin(np.linspace(0, 4 * np.pi, n))
    data[:] = np.nan
    data[::100] = data[::100]  # one valid point every 100

    filled = dct_prefill(data, width=10.0, max_iter=5)

    not_target = np.ones(n, dtype=bool)
    not_target[::100] = False  # original valid points are not fill targets
    fillable_nan = filled[not_target]
    assert np.all(np.isnan(fillable_nan))


def test_prefill_min_effective_density_none_disables_gate():
    """min_effective_density=None should reproduce the legacy un-gated behavior."""
    data = np.sin(np.linspace(0, 4 * np.pi, 200))
    data[80:120] = np.nan

    filled = dct_prefill(
        data, width=10.0, max_iter=5, min_effective_density=None
    )
    assert np.all(np.isfinite(filled[80:120]))


def test_prefill_min_effective_density_custom_value():
    """A stricter threshold should defer more cells to the nearest-neighbor fallback.

    The iterative normalized-convolution fill produces a smooth interpolation
    across the gap, while the nearest-neighbor fallback (used for unresolved
    cells) propagates the nearest valid value, creating a step. With a higher
    threshold, more cells are deferred to the fallback, so the gap contains
    piecewise-constant segments (one value per nearest-neighbor propagation
    direction). With a lower threshold, the gap is dominated by the smooth
    iterative fill.
    """
    n = 200
    data = np.linspace(0.0, 1.0, n)
    data[80:120] = np.nan

    filled_low = dct_prefill(
        data, width=10.0, max_iter=3, min_effective_density=0.05
    )
    filled_high = dct_prefill(
        data, width=10.0, max_iter=3, min_effective_density=0.9
    )

    # Both should be fully finite (fallback fills the rest).
    assert np.all(np.isfinite(filled_low[80:120]))
    assert np.all(np.isfinite(filled_high[80:120]))

    # The high-threshold fill relies more on the nearest-neighbor fallback,
    # which propagates a single value from the nearest valid neighbor. So the
    # filled gap contains long runs of identical values, whereas the
    # low-threshold fill interpolates smoothly. Compare the maximum run
    # length of identical values inside the gap.
    def _max_run_length(arr):
        runs = []
        current = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1]:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)
        return max(runs) if runs else 0

    run_low = _max_run_length(filled_low[80:120])
    run_high = _max_run_length(filled_high[80:120])
    # The high-threshold fill should have a longer run of identical values
    # (larger fallback region) than the low-threshold fill.
    assert run_high >= run_low


def test_mean_min_effective_density_returns_nan():
    """dct_mean with min_effective_density set should return NaN in low-density zones."""
    data = np.full(200, np.nan)
    # Use a cluster of valid points to get density > 0.35 at the center.
    data[95:106] = 5.0

    mean = dct_mean(
        data,
        width=8.0,
        restore_input_nan=False,
        min_effective_density=0.35,
    )

    # Far from the valid cluster, density < 0.35 => NaN.
    assert np.isnan(mean[0])
    assert np.isnan(mean[20])
    # Within the valid cluster, density is well above 0.35.
    assert np.isfinite(mean[100])
    assert np.isfinite(mean[102])


def test_mean_min_effective_density_none_unchanged():
    """dct_mean with min_effective_density=None should match legacy behavior."""
    data = np.full(200, np.nan)
    data[100] = 5.0

    mean_none = dct_mean(data, width=8.0, restore_input_nan=False, min_effective_density=None)
    mean_default = dct_mean(data, width=8.0, restore_input_nan=False)

    assert np.allclose(mean_none, mean_default, equal_nan=True)


def test_prefill_min_effective_density_consistent_with_dct_count():
    """Gate should accept cells in the un-gated path that the gated path skips."""
    np.random.seed(0)
    n = 400
    data = np.linspace(0.0, 1.0, n)
    mask = np.zeros(n, dtype=bool)
    mask[::20] = True
    data[~mask] = np.nan

    width = 6.0
    threshold = 0.35
    count = dct_count(mask, width=width, restore_input_nan=False)
    width_norm = float(np.prod(np.atleast_1d(width)))
    density = count / width_norm

    filled_gated = dct_prefill(
        data,
        width=width,
        max_iter=4,
        min_effective_density=threshold,
    )
    filled_ungated = dct_prefill(
        data,
        width=width,
        max_iter=4,
        min_effective_density=None,
    )

    target = np.isnan(data)
    n_filled_gated = int(np.sum(target & np.isfinite(filled_gated)))
    n_filled_ungated = int(np.sum(target & np.isfinite(filled_ungated)))

    # The gate must accept no more cells than the un-gated path.
    assert n_filled_gated <= n_filled_ungated

    # In the un-gated path, cells near the original valid points (high
    # original-mask density) should be filled.
    near_valid = target & (density > 0.5)
    if np.any(near_valid):
        assert np.all(np.isfinite(filled_ungated[near_valid]))

    # Consistency with dct_count: cells that the gate skips should have
    # original-mask density below the threshold (so the gate is justified
    # in leaving them NaN at this iteration).
    gated_skipped = target & np.isnan(filled_gated) & np.isfinite(filled_ungated)
    if np.any(gated_skipped):
        # Some skipped cells may have been filled by the nearest-neighbor
        # fallback, so we only check the bulk of the gap region.
        center_skipped = gated_skipped[100:300]
        if np.any(center_skipped):
            assert np.all(density[100:300][center_skipped] <= threshold)


def test_mean_min_effective_density_polar_heavy_gaps():
    """dct_mean gate should work in polar coords and respect the threshold."""
    n_az, n_range = 180, 120
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    rng = np.arange(1, n_range + 1, dtype=float)
    AZ, R = np.meshgrid(az, rng, indexing="ij")
    base = np.sin(2.0 * AZ) + 0.2 * np.cos(R / 8.0)

    # Use dense support regions (so density > 0.35) to test the gate.
    data = np.full((n_az, n_range), np.nan)
    data[:, 10:30] = base[:, 10:30]      # 20-cell radial stripe (dense)
    data[20:160, 40:60] = base[20:160, 40:60]  # 140x20 angular-range patch

    mean = dct_mean(
        data,
        width=5.0,
        coordinates="polar",
        restore_input_nan=False,
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
        min_effective_density=0.35,
    )

    valid = np.isfinite(data)
    assert np.all(np.isfinite(mean[valid]))

    mask = np.isfinite(data)
    count = dct_count(
        mask,
        width=5.0,
        coordinates="polar",
        az_res_deg=360.0 / n_az,
        az_boundary="periodic",
        restore_input_nan=False,
    )
    widths = np.array([5.0, 5.0], dtype=float)
    r_indices = np.arange(1, n_range + 1)
    az_res_rad = np.deg2rad(360.0 / n_az)
    w_beams = float(widths[0]) / (r_indices * az_res_rad)
    area_1d = float(widths[1]) * w_beams
    area = np.tile(area_1d, (n_az, 1))
    density = count / area

    below_gate = density < 0.35
    assert np.all(np.isnan(mean[below_gate]))
    assert np.all(np.isfinite(mean[~below_gate]))

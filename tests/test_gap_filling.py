"""
Test suite for dct_inpaint (DCT spectral inpainting).

Tests cover:
- Basic functionality (1D, 2D, no NaN output)
- Data fidelity (valid pixels preserved exactly)
- Curvature preservation (order=2 advantage)
- Convergence behaviour
- Edge cases (all NaN, no NaN, single point)
- Polar coordinate support (reflective + periodic)
- Order parameter (p=1 vs p=2)
- smooth_output option
"""

import numpy as np
import pytest

from dct_toolkit.gap_filling import (
    dct_inpaint,
    _width_to_lambda,
    _eigenvalues_dct,
    _eigenvalues_dft,
    _compute_eigenvalues_2d,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def sine_1d():
    """1-D sine wave with a contiguous gap."""
    x = np.linspace(0, 2 * np.pi, 200)
    truth = np.sin(x)
    data = truth.copy()
    data[80:120] = np.nan
    return data, truth


@pytest.fixture
def smooth_2d():
    """2-D smooth field with a circular hole."""
    Y, X = np.meshgrid(
        np.linspace(0, 4 * np.pi, 100),
        np.linspace(0, 4 * np.pi, 100),
    )
    truth = np.sin(X) * np.cos(Y)
    data = truth.copy()
    cx, cy, r = 50, 50, 15
    yy, xx = np.ogrid[:100, :100]
    hole = ((xx - cx) ** 2 + (yy - cy) ** 2) < r**2
    data[hole] = np.nan
    return data, truth, hole


@pytest.fixture
def polar_field():
    """Polar grid (720 x 200) with a centered hole."""
    n_az, n_range = 360, 100
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    rng = np.arange(1, n_range + 1, dtype=float)
    AZ, RNG = np.meshgrid(az, rng, indexing="ij")
    truth = 5.0 * np.sin(AZ) * np.exp(-RNG / 50.0)

    # Circular hole (index space)
    cx_az, cx_r, radius = n_az // 2, n_range // 2, 12
    yy, xx = np.ogrid[:n_az, :n_range]
    hole = ((yy - cx_az) ** 2 + (xx - cx_r) ** 2) < radius**2

    data = truth.copy()
    data[hole] = np.nan
    return data, truth, hole


# ===================================================================
# Helper function tests
# ===================================================================


class TestHelpers:
    """Tests for private helper functions."""

    def test_width_to_lambda_positive(self):
        """Lambda should be positive for any positive width."""
        for w in [1.0, 5.0, 20.0, 100.0]:
            for p in [1, 2, 3]:
                lam = _width_to_lambda(w, p)
                assert lam > 0, f"lambda should be > 0 for width={w}, order={p}"

    def test_width_to_lambda_monotonic(self):
        """Larger width → larger lambda (more smoothing)."""
        widths = [1.0, 5.0, 10.0, 50.0]
        for p in [1, 2]:
            lambdas = [_width_to_lambda(w, p) for w in widths]
            for i in range(len(lambdas) - 1):
                assert lambdas[i] < lambdas[i + 1], (
                    f"Lambda should increase with width (order={p})"
                )

    def test_eigenvalues_dct_dc_is_zero(self):
        """DC component (k=0) eigenvalue should be zero."""
        for n in [32, 64, 128]:
            for p in [1, 2]:
                E = _eigenvalues_dct(n, p)
                assert E[0] == 0.0, "DC eigenvalue must be 0"

    def test_eigenvalues_dct_nonnegative(self):
        """All eigenvalues should be non-negative."""
        E = _eigenvalues_dct(100, 2)
        assert np.all(E >= 0), "Eigenvalues must be >= 0"

    def test_eigenvalues_dft_dc_is_zero(self):
        """DC component eigenvalue should be zero for DFT too."""
        E = _eigenvalues_dft(64, 2)
        assert E[0] == 0.0, "DC eigenvalue must be 0"

    def test_eigenvalues_dft_length(self):
        """DFT eigenvalue array length should be n//2+1 (RFFT half-spectrum)."""
        for n in [32, 64, 100]:
            E = _eigenvalues_dft(n, 1)
            assert len(E) == n // 2 + 1

    def test_eigenvalues_2d_shape_reflective(self):
        """2D eigenvalue tensor shape for reflective BC."""
        E = _compute_eigenvalues_2d((60, 80), 2, "reflective")
        assert E.shape == (60, 80)

    def test_eigenvalues_2d_shape_periodic(self):
        """2D eigenvalue tensor shape for periodic axis-0."""
        E = _compute_eigenvalues_2d((60, 80), 2, "periodic")
        assert E.shape == (60 // 2 + 1, 80)


# ===================================================================
# Core functionality tests
# ===================================================================


class TestDCTInpaint1D:
    """Tests for 1-D inpainting."""

    def test_no_nan_output(self, sine_1d):
        """Output should contain no NaN values."""
        data, _ = sine_1d
        filled = dct_inpaint(data, width=10.0)
        assert not np.any(np.isnan(filled))

    def test_valid_data_preserved(self, sine_1d):
        """Valid pixels should be preserved exactly."""
        data, truth = sine_1d
        filled = dct_inpaint(data, width=10.0)
        valid = ~np.isnan(data)
        np.testing.assert_array_equal(filled[valid], truth[valid])

    def test_accuracy(self, sine_1d):
        """Gap fill should be accurate for a smooth signal."""
        data, truth = sine_1d
        filled = dct_inpaint(data, width=10.0)
        gap_mae = np.mean(np.abs(filled[80:120] - truth[80:120]))
        assert gap_mae < 0.01, f"1D gap MAE={gap_mae:.4f}, expected < 0.01"

    def test_no_gaps_passthrough(self):
        """Data with no NaN should be returned unchanged."""
        data = np.arange(50, dtype=float)
        result = dct_inpaint(data, width=5.0)
        np.testing.assert_array_equal(result, data)


class TestDCTInpaint2D:
    """Tests for 2-D Cartesian inpainting."""

    def test_no_nan_output(self, smooth_2d):
        """Output should contain no NaN values."""
        data, _, _ = smooth_2d
        filled = dct_inpaint(data, width=10.0)
        assert not np.any(np.isnan(filled))

    def test_valid_data_preserved(self, smooth_2d):
        """Valid pixels should be preserved exactly."""
        data, truth, hole = smooth_2d
        filled = dct_inpaint(data, width=10.0)
        valid = ~np.isnan(data)
        np.testing.assert_array_equal(filled[valid], truth[valid])

    def test_accuracy(self, smooth_2d):
        """Gap fill should be reasonably accurate for a smooth field."""
        data, truth, hole = smooth_2d
        filled = dct_inpaint(data, width=10.0)
        gap_mae = np.mean(np.abs(filled[hole] - truth[hole]))
        assert gap_mae < 0.2, f"2D gap MAE={gap_mae:.4f}, expected < 0.2"

    def test_curvature_preservation(self, smooth_2d):
        """Order=2 should outperform order=1 on curved fields."""
        data, truth, hole = smooth_2d
        filled_p1 = dct_inpaint(data, width=10.0, order=1)
        filled_p2 = dct_inpaint(data, width=10.0, order=2)
        mae_p1 = np.mean(np.abs(filled_p1[hole] - truth[hole]))
        mae_p2 = np.mean(np.abs(filled_p2[hole] - truth[hole]))
        assert mae_p2 < mae_p1, (
            f"Order 2 (MAE={mae_p2:.4f}) should beat order 1 (MAE={mae_p1:.4f})"
        )

    def test_constant_field(self):
        """Constant field with gap should fill with the constant."""
        data = np.ones((50, 50)) * 7.0
        data[20:30, 20:30] = np.nan
        filled = dct_inpaint(data, width=5.0)
        np.testing.assert_allclose(filled, 7.0, atol=1e-8)


class TestDCTInpaintPolar:
    """Tests for polar coordinate inpainting."""

    def test_no_nan_output(self, polar_field):
        """Output should contain no NaN values (polar, periodic)."""
        data, _, _ = polar_field
        filled = dct_inpaint(
            data,
            width=10.0,
            coordinates="polar",
            az_res_deg=1.0,
            az_boundary="periodic",
        )
        assert not np.any(np.isnan(filled))

    def test_accuracy_periodic(self, polar_field):
        """Periodic BC should give accurate fill for smooth polar field."""
        data, truth, hole = polar_field
        filled = dct_inpaint(
            data,
            width=10.0,
            coordinates="polar",
            az_res_deg=1.0,
            az_boundary="periodic",
        )
        gap_mae = np.mean(np.abs(filled[hole] - truth[hole]))
        assert gap_mae < 0.5, f"Polar gap MAE={gap_mae:.4f}, expected < 0.5"

    def test_accuracy_reflective(self, polar_field):
        """Reflective BC should also work for polar data."""
        data, truth, hole = polar_field
        filled = dct_inpaint(
            data,
            width=10.0,
            coordinates="polar",
            az_res_deg=1.0,
            az_boundary="reflective",
        )
        gap_mae = np.mean(np.abs(filled[hole] - truth[hole]))
        assert gap_mae < 0.5, f"Polar reflective gap MAE={gap_mae:.4f}, expected < 0.5"

    def test_wrapping_hole(self):
        """Periodic BC should handle holes crossing the 0/360 boundary."""
        n_az, n_range = 360, 80
        az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
        rng = np.arange(1, n_range + 1, dtype=float)
        AZ, RNG = np.meshgrid(az, rng, indexing="ij")
        truth = 3.0 * np.cos(AZ) * (RNG / n_range)

        # Wrapping hole near azimuth=0
        hole = np.zeros((n_az, n_range), dtype=bool)
        for i in range(n_az):
            for j in range(n_range):
                az_dist = min(i, n_az - i)
                if az_dist**2 + (j - 40) ** 2 < 10**2:
                    hole[i, j] = True

        data = truth.copy()
        data[hole] = np.nan

        filled = dct_inpaint(
            data,
            width=8.0,
            coordinates="polar",
            az_res_deg=1.0,
            az_boundary="periodic",
        )
        gap_mae = np.mean(np.abs(filled[hole] - truth[hole]))
        assert gap_mae < 0.2, f"Wrapping hole MAE={gap_mae:.4f}, expected < 0.2"


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_all_nan(self):
        """All-NaN input should return zeros with a warning."""
        data = np.full(50, np.nan)
        with pytest.warns(UserWarning, match="All values are NaN"):
            filled = dct_inpaint(data, width=5.0)
        np.testing.assert_array_equal(filled, 0.0)

    def test_single_valid_point(self):
        """Single valid point should still produce a result."""
        data = np.full(50, np.nan)
        data[25] = 3.0
        filled = dct_inpaint(data, width=5.0)
        assert not np.any(np.isnan(filled))
        assert filled[25] == 3.0  # valid point preserved

    def test_3d_raises(self):
        """3-D input should raise ValueError."""
        data = np.ones((10, 10, 10))
        data[5, 5, 5] = np.nan
        with pytest.raises(ValueError, match="1-D and 2-D"):
            dct_inpaint(data, width=3.0)

    def test_smooth_output(self, sine_1d):
        """smooth_output=True should smooth valid data too."""
        data, truth = sine_1d
        filled_exact = dct_inpaint(data, width=10.0, smooth_output=False)
        filled_smooth = dct_inpaint(data, width=10.0, smooth_output=True)

        valid = ~np.isnan(data)
        # Exact mode: valid data preserved
        np.testing.assert_array_equal(filled_exact[valid], truth[valid])
        # Smooth mode: valid data may differ (smoothed)
        assert not np.array_equal(filled_smooth[valid], truth[valid])

    def test_zeros_init(self, smooth_2d):
        """init='zeros' should still produce a valid result."""
        data, truth, hole = smooth_2d
        filled = dct_inpaint(data, width=10.0, init="zeros")
        assert not np.any(np.isnan(filled))

    def test_invalid_init_raises(self):
        """Unknown init strategy should raise ValueError."""
        data = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="Unknown init"):
            dct_inpaint(data, width=2.0, init="magic")


# ===================================================================
# Comparison with iterative_gap_fill (v3)
# ===================================================================


class TestComparisonWithV3:
    """Ensure dct_inpaint improves upon iterative_gap_fill."""

    def test_inpaint_beats_diffusion_2d(self, smooth_2d):
        """dct_inpaint (order=2) should be more accurate than v3 diffusion."""
        from dct_toolkit.gap_filling import iterative_gap_fill

        data, truth, hole = smooth_2d
        filled_v3 = iterative_gap_fill(data, width=10.0)
        filled_v4 = dct_inpaint(data, width=10.0)

        mae_v3 = np.mean(np.abs(filled_v3[hole] - truth[hole]))
        mae_v4 = np.mean(np.abs(filled_v4[hole] - truth[hole]))
        assert mae_v4 < mae_v3, f"v4 MAE={mae_v4:.4f} should be < v3 MAE={mae_v3:.4f}"

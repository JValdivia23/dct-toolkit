import numpy as np
import pytest
from dct_toolkit.core import get_dct_transfer_function, dct_convolve_1d

def test_transfer_function_dc_preservation():
    """All kernels should preserve DC (H[0] ≈ 1)."""
    for kernel in ['boxcar', 'boxcar_discrete', 'gaussian']:
        H = get_dct_transfer_function(64, kernel, width=8.0)
        assert np.isclose(H[0], 1.0), f"{kernel} failed DC preservation"

def test_boxcar_analytical_vs_discrete():
    """For large n, analytical and discrete should be similar."""
    n = 256
    w = 8.0
    H_analytical = get_dct_transfer_function(n, 'boxcar', w)
    H_discrete = get_dct_transfer_function(n, 'boxcar_discrete', w)
    # Should be close for low frequencies (k < n/8)
    # Relaxed tolerance as discrete/continuous difference is expected at higher freq
    assert np.allclose(H_analytical[:32], H_discrete[:32], atol=0.1)

def test_gaussian_monotonicity():
    """Gaussian transfer function should decrease monotonically."""
    H = get_dct_transfer_function(64, 'gaussian', width=8.0)
    assert np.all(np.diff(H) <= 0), "Gaussian not monotonically decreasing"

def test_convolve_1d_identity():
    """Smoothing with vanishing width should return original signal."""
    data = np.random.rand(100)
    # Extremely small width -> Sigma ~ 0 -> H ~ 1
    H = get_dct_transfer_function(100, 'gaussian', width=1e-6)
    smoothed = dct_convolve_1d(data, H)
    assert np.allclose(data, smoothed, atol=1e-10)

def test_shape_mismatch():
    """Should raise error if data and H shape mismatch."""
    data = np.zeros(50)
    H = np.zeros(60)
    with pytest.raises(ValueError):
        dct_convolve_1d(data, H)

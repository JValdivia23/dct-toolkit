import numpy as np
import pytest
from dct_toolkit.cartesian import smooth_cartesian

@pytest.fixture
def constant_field_2d():
    """Returns array of ones."""
    return np.ones((64, 64))

@pytest.fixture
def linear_ramp_2d():
    """Returns linear ramp along axis 0."""
    x = np.linspace(0, 1, 64)
    # Shape (64, 1) broadcasted to (64, 64) implicitly or explicitly
    # Let's make it (64, 64) varying only along axis 0
    return x[:, np.newaxis] * np.ones((1, 64))

def test_smooth_constant_cartesian(constant_field_2d):
    """Smoothing a constant field should return constant."""
    result = smooth_cartesian(constant_field_2d, width=5.0)
    assert np.allclose(result, 1.0)

def test_smooth_linear_cartesian(linear_ramp_2d):
    """Boxcar smoothing of linear function should be exact."""
    # Boxcar kernel preserves linear trends (first moment)
    result = smooth_cartesian(linear_ramp_2d, width=5.0, kernel_type='boxcar')
    # Edges might have boundary effects, check interior
    interior = result[5:-5, 5:-5]
    expected = linear_ramp_2d[5:-5, 5:-5]
    assert np.allclose(interior, expected, atol=1e-3)

def test_separability():
    """Sequential 1D smoothing equals 2D smoothing."""
    np.random.seed(42)
    data = np.random.randn(32, 32)
    
    # 2D Smoothing
    result_2d = smooth_cartesian(data, width=4.0, kernel_type='gaussian')
    
    # Explicit separable smoothing
    from dct_toolkit.core import get_dct_transfer_function, dct_convolve_1d
    H0 = get_dct_transfer_function(32, 'gaussian', 4.0)
    H1 = get_dct_transfer_function(32, 'gaussian', 4.0)
    
    step1 = dct_convolve_1d(data, H0, axis=0)
    result_sep = dct_convolve_1d(step1, H1, axis=1)
    
    assert np.allclose(result_2d, result_sep)

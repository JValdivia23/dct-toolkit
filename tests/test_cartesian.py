import numpy as np
import pytest

from dct_toolkit.cartesian import smooth_cartesian
from dct_toolkit.core import dct_convolve_1d, get_dct_transfer_function


def _smooth_cartesian_reference(
    data: np.ndarray,
    width: float,
    kernel_type: str = "gaussian",
) -> np.ndarray:
    """Reference separable smoothing using explicit axis-wise 1D operations."""
    result = data.copy()
    for axis, n in enumerate(data.shape):
        transfer = get_dct_transfer_function(n, kernel_type, width)
        result = dct_convolve_1d(result, transfer, axis=axis)
    return result


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
    result = smooth_cartesian(linear_ramp_2d, width=5.0, kernel_type="boxcar")
    # Edges might have boundary effects, check interior
    interior = result[5:-5, 5:-5]
    expected = linear_ramp_2d[5:-5, 5:-5]
    assert np.allclose(interior, expected, atol=1e-3)


def test_separability():
    """N-D DCT smoothing matches explicit 2D separable reference."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((32, 32))

    result_nd = smooth_cartesian(data, width=4.0, kernel_type="gaussian")
    result_ref = _smooth_cartesian_reference(data, width=4.0, kernel_type="gaussian")

    assert np.allclose(result_nd, result_ref, atol=1e-12)


@pytest.mark.parametrize("shape", [(12, 10, 8), (6, 5, 4, 3)])
@pytest.mark.parametrize("kernel_type", ["gaussian", "boxcar", "boxcar_discrete"])
def test_nd_matches_axiswise_reference(shape, kernel_type):
    """N-D Cartesian smoothing remains artifact-free for 3D/4D arrays."""
    rng = np.random.default_rng(123)
    data = rng.standard_normal(shape)

    result_nd = smooth_cartesian(data, width=3.5, kernel_type=kernel_type)
    result_ref = _smooth_cartesian_reference(data, width=3.5, kernel_type=kernel_type)

    assert np.allclose(result_nd, result_ref, atol=1e-12)

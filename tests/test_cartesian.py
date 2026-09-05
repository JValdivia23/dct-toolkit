from typing import Callable, Sequence, Tuple, Union

import numpy as np
import pytest

from dct_toolkit._widths import WidthLike
from dct_toolkit.cartesian import smooth_cartesian
from dct_toolkit.core import dct_convolve_1d, get_dct_transfer_function


def _smooth_cartesian_reference(
    data: np.ndarray,
    width: WidthLike,
    kernel_type: Union[str, Sequence[str], np.ndarray] = "gaussian",
) -> np.ndarray:
    """Reference separable smoothing using explicit axis-wise 1D operations."""
    width_values = np.asarray(width, dtype=float)
    if width_values.ndim == 0:
        widths = np.full(data.ndim, float(width_values), dtype=float)
    else:
        widths = width_values

    kernels = [kernel_type] * data.ndim if isinstance(kernel_type, str) else kernel_type
    result = data.copy()
    for axis, (n, width_axis, kernel_axis) in enumerate(zip(data.shape, widths, kernels)):
        transfer = get_dct_transfer_function(n, kernel_axis, float(width_axis))
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


def test_isotropic_scalar_equals_isotropic_vector():
    """Scalar width should match equivalent isotropic vector widths."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((11, 9, 7))

    result_scalar = smooth_cartesian(data, width=3.0, kernel_type="gaussian")
    result_vector = smooth_cartesian(
        data, width=[3.0, 3.0, 3.0], kernel_type="gaussian"
    )

    assert np.allclose(result_scalar, result_vector, atol=1e-12)


def test_anisotropic_width_matches_axiswise_reference():
    """Per-axis widths should match explicit axis-wise separable reference."""
    rng = np.random.default_rng(11)
    data = rng.standard_normal((13, 10, 8, 6))
    widths = [4.0, 2.5, 1.5, 3.0]

    result_nd = smooth_cartesian(data, width=widths, kernel_type="gaussian")
    result_ref = _smooth_cartesian_reference(data, width=widths, kernel_type="gaussian")

    assert np.allclose(result_nd, result_ref, atol=1e-12)


@pytest.mark.parametrize(
    "shape, kernels",
    [
        ((17,), ("boxcar",)),
        ((11, 9), ("gaussian", "boxcar")),
        ((13, 10, 8), ("gaussian", "boxcar", "boxcar")),
        ((5, 7, 6, 4), ("boxcar_discrete", "gaussian", "boxcar", "gaussian")),
    ],
)
@pytest.mark.parametrize("scalar_width", [True, False])
def test_per_axis_kernels_match_reference(
    shape: Tuple[int, ...], kernels: Tuple[str, ...], scalar_width: bool
) -> None:
    """Per-axis kernels match successive 1-D convolutions for 1D through 4D."""
    data = np.random.default_rng(23).standard_normal(shape)
    widths = 3.5 if scalar_width else np.linspace(2.5, 5.5, data.ndim)
    original = data.copy()

    result = smooth_cartesian(data, width=widths, kernel_type=kernels)
    expected = _smooth_cartesian_reference(data, width=widths, kernel_type=kernels)

    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("kernel", ["gaussian", "boxcar", "boxcar_discrete"])
@pytest.mark.parametrize("container", [tuple, list, np.array])
def test_repeated_kernels_preserve_single_string_behavior(
    kernel: str, container: Callable
) -> None:
    """Lists, tuples, and arrays of one repeated kernel retain legacy results."""
    data = np.random.default_rng(29).standard_normal((7, 9, 11))
    widths = (2.5, 3.0, 5.0)

    expected = smooth_cartesian(data, width=widths, kernel_type=kernel)
    result = smooth_cartesian(data, width=widths, kernel_type=container([kernel] * 3))

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(7, 9, 11), (1, 9, 7), (1, 1, 1)])
def test_mixed_kernels_preserve_constant_fields(shape: Tuple[int, ...]) -> None:
    """Mixed kernels preserve DC, including singleton axes and integer data."""
    data = np.full(shape, 7, dtype=np.int64)
    result = smooth_cartesian(
        data, width=(3.0, 5.0, 3.0), kernel_type=("gaussian", "boxcar", "boxcar_discrete")
    )
    np.testing.assert_allclose(result, data, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kernels, message",
    [
        ([], "length-3"),
        (["gaussian"], "length-3"),
        (["boxcar"] * 4, "length-3"),
        ([["gaussian", "boxcar", "boxcar"]], "1-D sequence"),
        (None, "string or 1-D sequence"),
        (3, "string or 1-D sequence"),
        (["gaussian", None, "boxcar"], "entries must be strings.*axis 1"),
        (["gaussian", 3, "boxcar"], "entries must be strings.*axis 1"),
        ("unknown", "Unknown kernel type"),
        (["gaussian", "boxcar", "unknown"], "Unknown kernel type.*axis 2"),
    ],
)
def test_invalid_kernel_specifications_raise(kernels: object, message: str) -> None:
    """Malformed or unsupported per-axis specifications raise clear errors."""
    with pytest.raises(ValueError, match=message):
        smooth_cartesian(np.ones((5, 7, 9)), width=3.0, kernel_type=kernels)


def test_mixed_kernels_reject_empty_axes() -> None:
    """Empty axes remain unsupported by the DCT backend."""
    with pytest.raises(ValueError):
        smooth_cartesian(
            np.empty((0, 5, 3)), width=3.0, kernel_type=("gaussian", "boxcar", "boxcar")
        )


def test_scalar_input_remains_a_copy() -> None:
    """A zero-dimensional input remains a no-op with no per-axis kernels."""
    data = np.array(4.0)
    result = smooth_cartesian(data, width=3.0, kernel_type=())
    np.testing.assert_array_equal(result, data)
    assert result is not data

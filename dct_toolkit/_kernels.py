"""Internal helpers for per-axis kernel selection."""

from typing import Sequence, Tuple, Union

import numpy as np


KernelLike = Union[str, Sequence[str], np.ndarray]


def normalize_kernel_types(kernel_type: KernelLike, ndim: int) -> Tuple[str, ...]:
    """Validate kernel names and expand a single name to every axis."""
    if isinstance(kernel_type, str):
        kernels = (kernel_type,) * ndim
    else:
        values = np.asarray(kernel_type, dtype=object)
        if values.ndim != 1:
            raise ValueError("kernel_type must be a string or 1-D sequence of strings")
        if values.size != ndim:
            raise ValueError(
                f"kernel_type must be a string or length-{ndim} sequence, "
                f"got length {values.size}"
            )
        kernels = tuple(values.tolist())

    for axis, kernel in enumerate(kernels):
        if not isinstance(kernel, str):
            raise ValueError(f"kernel_type entries must be strings, got {kernel!r} at axis {axis}")
        if kernel not in ("boxcar", "boxcar_discrete", "gaussian"):
            raise ValueError(f"Unknown kernel type: {kernel!r} at axis {axis}")
    return kernels

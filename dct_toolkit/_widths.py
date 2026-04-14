"""Internal helpers for isotropic/anisotropic width handling."""

from typing import Sequence, Union

import numpy as np


WidthLike = Union[float, Sequence[float], np.ndarray]


def normalize_widths(width: WidthLike, ndim: int, name: str = "width") -> np.ndarray:
    """Normalize scalar/sequence width input to a positive 1-D array."""
    if ndim < 0:
        raise ValueError(f"ndim must be >= 0, got {ndim}")

    values = np.asarray(width, dtype=float)

    if values.ndim == 0:
        widths = np.full(ndim, float(values), dtype=float)
    elif values.ndim == 1:
        if values.size != ndim:
            raise ValueError(
                f"{name} must be a scalar or length-{ndim} sequence, got length {values.size}"
            )
        widths = values.astype(float, copy=False)
    else:
        raise ValueError(f"{name} must be a scalar or 1-D sequence")

    if not np.all(np.isfinite(widths)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(widths <= 0):
        raise ValueError(f"{name} must be > 0")

    return widths

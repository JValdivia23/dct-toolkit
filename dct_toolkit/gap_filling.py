"""
Iterative Gap Filling using DCT Statistics.

This module implements a constructive gap filling algorithm built from the
``dct_mean`` primitive.

Algorithm (Linear Initialization — default):
1. Initialize missing values via axis-wise linear interpolation
   (row-wise ``np.interp``, then column-wise for remaining NaNs,
   then global-mean fallback).  This preserves spatial gradients across
   holes from the very first iterate.
2. Iteratively smooth the entire field and update ONLY the missing values
   with the smooth trend (diffusion process at the user's target width).
3. Converges to a solution where missing values are consistent with the
   smooth trends of the surrounding valid data.

Alternative initializations (``init`` parameter):
- ``'multiscale'``: Coarse-to-fine DCT cascade (legacy v0.2.0 approach).
- ``'dct'``: Single-pass Normalized Convolution at the target width.
"""

import warnings
from typing import Optional

import numpy as np

from .stats import dct_mean


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _linear_init_1d(data: np.ndarray) -> np.ndarray:
    """
    Fill NaN gaps in a 1-D array using linear interpolation.

    Parameters
    ----------
    data : np.ndarray
        1-D array, possibly containing NaNs.

    Returns
    -------
    filled : np.ndarray
        Copy of *data* with NaN gaps linearly interpolated from valid
        neighbours.  If the entire array is NaN the values are set to 0.
    """
    filled = data.copy()
    mask = np.isnan(filled)
    if not np.any(mask):
        return filled
    valid = ~mask
    if not np.any(valid):
        filled[:] = 0.0
        return filled
    coords = np.arange(len(filled))
    filled[mask] = np.interp(coords[mask], coords[valid], data[valid])
    return filled


def _linear_init_2d(data: np.ndarray) -> np.ndarray:
    """
    Fill NaN gaps in a 2-D array using axis-wise linear interpolation.

    Strategy:
    1. Azimuth-wise (axis=0): interpolate along columns (azimuth).
       Avoids extrapolation at edges (left/right=NaN) to handle periodic
       boundaries correctly (leaves gaps for range fill).
    2. Range-wise (axis=1): interpolate along rows (range).
       Avoids extrapolation at edges.
    3. Fallback: Repeated passes allowing extrapolation to fill corners/edges.
    4. Global-mean fallback: Final safety net.

    Parameters
    ----------
    data : np.ndarray
        2-D array, possibly containing NaNs. Shape (n_az, n_range).

    Returns
    -------
    filled : np.ndarray
        Copy of *data* with NaN gaps filled.
    """
    filled = data.copy()
    rows, cols = filled.shape

    # --- Pass 1: Azimuth-wise (axis=0), NO extrapolation ---
    for c in range(cols):
        col = filled[:, c]
        m = np.isnan(col)
        if np.any(m) and not np.all(m):
            v = ~m
            coords = np.arange(len(col))
            # left=NaN, right=NaN: Don't extrapolate wrapping gaps
            filled[m, c] = np.interp(coords[m], coords[v], col[v], left=np.nan, right=np.nan)

    # --- Pass 2: Range-wise (axis=1), NO extrapolation ---
    for r in range(rows):
        row = filled[r]
        m = np.isnan(row)
        if np.any(m) and not np.all(m):
            v = ~m
            coords = np.arange(len(row))
            filled[r, m] = np.interp(coords[m], coords[v], row[v], left=np.nan, right=np.nan)

    # --- Pass 3: Fill remaining with extrapolation (Azimuth then Range) ---
    if np.any(np.isnan(filled)):
        # Azimuth (axis=0) with extrapolation
        for c in range(cols):
            col = filled[:, c]
            m = np.isnan(col)
            if np.any(m) and not np.all(m):
                v = ~m
                coords = np.arange(len(col))
                filled[m, c] = np.interp(coords[m], coords[v], col[v])
        
        # Range (axis=1) with extrapolation
        for r in range(rows):
            row = filled[r]
            m = np.isnan(row)
            if np.any(m) and not np.all(m):
                v = ~m
                coords = np.arange(len(row))
                filled[r, m] = np.interp(coords[m], coords[v], row[v])

    # --- Global-mean fallback ---
    if np.any(np.isnan(filled)):
        global_mean = np.nanmean(data)
        if np.isnan(global_mean):
            global_mean = 0.0
        filled[np.isnan(filled)] = global_mean

    return filled


def _build_width_cascade(data_shape: tuple, target_width: float) -> list:
    """
    Build an adaptive coarse-to-fine width cascade.

    Starts from ``max(data_shape) / 4`` and halves until reaching the
    target width.  Always ends with *target_width*.

    Parameters
    ----------
    data_shape : tuple
        Shape of the data array.
    target_width : float
        The user's requested smoothing width (final scale).

    Returns
    -------
    cascade : list of float
        Widths from coarsest to finest, ending with *target_width*.
    """
    max_dim = max(data_shape)
    w_start = max_dim / 4.0

    cascade = []
    w = w_start
    while w > target_width:
        cascade.append(w)
        w /= 2.0

    cascade.append(target_width)
    return cascade


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iterative_gap_fill(
    data: np.ndarray,
    width: float,
    coordinates: str = 'cartesian',
    max_iter: int = 50,
    tol: float = 1e-4,
    init: str = 'linear',
    multiscale: Optional[bool] = None,
    smooth_output: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Fill gaps using iterative robust smoothing (Constructive Gap Filling).

    Missing values (NaN) are first initialized, then iteratively refined
    by replacing gap values with the DCT-smoothed trend of the current
    field.  Valid data is preserved exactly (unless ``smooth_output=True``).

    Parameters
    ----------
    data : np.ndarray
        Input data with NaNs marking gaps.  1-D or 2-D.
    width : float
        Smoothing width.  Controls the "stiffness" of the interpolated
        surface — larger values produce smoother gap fills.
    coordinates : str, default='cartesian'
        Coordinate system: ``'cartesian'`` or ``'polar'``.
    max_iter : int, default=50
        Maximum number of diffusion iterations.
    tol : float, default=1e-4
        Convergence tolerance (relative change in L2 norm).
    init : str, default='linear'
        Initialization strategy for gap values:

        * ``'linear'`` — axis-wise linear interpolation (default).
          Preserves spatial gradients across holes; recommended for
          contiguous holes larger than a few times *width*.
        * ``'multiscale'`` — coarse-to-fine DCT cascade.  Starts at
          ``max(data_shape)/4`` and halves down to *width*.
        * ``'dct'`` — single-pass Normalized Convolution at *width*.

    multiscale : bool or None, default=None
        **Deprecated.**  Retained for backward compatibility only.
        If explicitly set to ``False``, maps to ``init='dct'``.
        If explicitly set to ``True``, maps to ``init='multiscale'``.
        Ignored when *init* is set to a non-default value by the caller.
    smooth_output : bool, default=False
        If True, the final iteration smooths **all** values (valid + gap),
        producing a combined gap-fill + noise-reduction result.  If False
        (default), valid data is preserved exactly.
    **kwargs
        Additional arguments passed to the smoothing function
        (e.g., ``kernel_type='gaussian'``, ``az_res_deg=0.5``).

    Returns
    -------
    filled : np.ndarray
        Data with gaps filled.  If ``smooth_output=False``, original valid
        values are preserved exactly.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    >>> filled = iterative_gap_fill(data, width=2.0)
    >>> np.all(~np.isnan(filled))
    True
    """
    # ------------------------------------------------------------------
    # Handle deprecated `multiscale` parameter
    # ------------------------------------------------------------------
    if multiscale is not None:
        warnings.warn(
            "The `multiscale` parameter is deprecated and will be removed "
            "in a future version.  Use `init='multiscale'` or "
            "`init='dct'` instead.",
            FutureWarning,
            stacklevel=2,
        )
        # Only override init if caller left it at the default
        if init == 'linear':
            init = 'multiscale' if multiscale else 'dct'

    # ------------------------------------------------------------------
    # Identify gaps
    # ------------------------------------------------------------------
    valid_mask = ~np.isnan(data)
    if np.all(valid_mask):
        return data.copy()

    gap_mask = ~valid_mask          # True where NaN

    # ------------------------------------------------------------------
    # Phase 1: Initialization
    # ------------------------------------------------------------------
    if init == 'linear':
        if data.ndim == 1:
            filled = _linear_init_1d(data)
        elif data.ndim == 2:
            filled = _linear_init_2d(data)
        else:
            raise ValueError(
                f"Linear init supports 1-D and 2-D data, got {data.ndim}-D"
            )

    elif init == 'multiscale':
        filled = data.copy()
        cascade = _build_width_cascade(data.shape, width)

        for w in cascade:
            current_mask = ~np.isnan(filled)
            estimate = dct_mean(filled, w, coordinates,
                                mask=current_mask, **kwargs)

            still_nan = np.isnan(filled)
            if not np.any(still_nan):
                break
            valid_estimate = ~np.isnan(estimate)
            fill_these = still_nan & valid_estimate
            filled[fill_these] = estimate[fill_these]

        # Fallback for remaining NaNs
        still_nan = np.isnan(filled)
        if np.any(still_nan):
            global_mean = np.nanmean(data)
            if np.isnan(global_mean):
                global_mean = 0.0
            filled[still_nan] = global_mean

    elif init == 'dct':
        filled = data.copy()
        estimate = dct_mean(data, width, coordinates,
                            mask=valid_mask, **kwargs)
        filled[gap_mask] = estimate[gap_mask]

        # Fallback
        still_nan = np.isnan(filled)
        if np.any(still_nan):
            global_mean = np.nanmean(data)
            if np.isnan(global_mean):
                global_mean = 0.0
            filled[still_nan] = global_mean

    else:
        raise ValueError(
            f"Unknown init strategy '{init}'. "
            f"Choose from 'linear', 'multiscale', 'dct'."
        )

    # ------------------------------------------------------------------
    # Phase 2: Iterative diffusion at target width
    # ------------------------------------------------------------------
    all_valid = np.ones_like(filled, dtype=bool)

    for i in range(max_iter):
        prev_filled = filled.copy()

        # Smooth the current filled field (all pixels treated as valid)
        trend = dct_mean(filled, width, coordinates,
                         mask=all_valid, **kwargs)

        # Update ONLY the gaps — preserve valid data exactly
        filled[gap_mask] = trend[gap_mask]

        # Check convergence
        diff = np.linalg.norm(filled - prev_filled)
        norm = np.linalg.norm(filled)
        rel_change = diff / (norm + 1e-10)

        if rel_change < tol:
            break

    # ------------------------------------------------------------------
    # Optional: smooth the entire output
    # ------------------------------------------------------------------
    if smooth_output:
        filled = dct_mean(filled, width, coordinates,
                          mask=all_valid, **kwargs)

    return filled

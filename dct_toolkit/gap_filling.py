"""
Gap Filling using DCT Statistics.

This module provides two gap-filling algorithms:

1. **Iterative Diffusion** (``iterative_gap_fill``):
   Simple iterative smoothing with Dirichlet constraints.  Equivalent to
   solving the heat equation ΔU = 0 in gaps (v0.2.x).

2. **DCT Spectral Inpainting** (``dct_inpaint``):
   Penalized Least Squares in the DCT/FFT domain (v0.3.0).
   Minimises  ‖W·(y − û)‖² + λ·‖Δᵖ û‖²  where p=2 (bi-harmonic)
   preserves curvature across gaps.  Based on Garcia (2010) with a
   user-friendly ``width`` parameterisation instead of raw λ.

Both algorithms:
- Initialize gaps with fast linear interpolation (baseline).
- Iterate: forward-transform → spectral filter → inverse-transform →
  reset valid data.
- Converge to the spectrally-optimal smooth surface through gaps.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.fft

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
            filled[m, c] = np.interp(
                coords[m], coords[v], col[v], left=np.nan, right=np.nan
            )

    # --- Pass 2: Range-wise (axis=1), NO extrapolation ---
    for r in range(rows):
        row = filled[r]
        m = np.isnan(row)
        if np.any(m) and not np.all(m):
            v = ~m
            coords = np.arange(len(row))
            filled[r, m] = np.interp(
                coords[m], coords[v], row[v], left=np.nan, right=np.nan
            )

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
    coordinates: str = "cartesian",
    max_iter: int = 50,
    tol: float = 1e-4,
    init: str = "linear",
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
        if init == "linear":
            init = "multiscale" if multiscale else "dct"

    # ------------------------------------------------------------------
    # Identify gaps
    # ------------------------------------------------------------------
    valid_mask = ~np.isnan(data)
    if np.all(valid_mask):
        return data.copy()

    gap_mask = ~valid_mask  # True where NaN

    # ------------------------------------------------------------------
    # Phase 1: Initialization
    # ------------------------------------------------------------------
    if init == "linear":
        if data.ndim == 1:
            filled = _linear_init_1d(data)
        elif data.ndim == 2:
            filled = _linear_init_2d(data)
        else:
            raise ValueError(
                f"Linear init supports 1-D and 2-D data, got {data.ndim}-D"
            )

    elif init == "multiscale":
        filled = data.copy()
        cascade = _build_width_cascade(data.shape, width)

        for w in cascade:
            current_mask = ~np.isnan(filled)
            estimate = dct_mean(filled, w, coordinates, mask=current_mask, **kwargs)

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

    elif init == "dct":
        filled = data.copy()
        estimate = dct_mean(data, width, coordinates, mask=valid_mask, **kwargs)
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
        trend = dct_mean(filled, width, coordinates, mask=all_valid, **kwargs)

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
        filled = dct_mean(filled, width, coordinates, mask=all_valid, **kwargs)

    return filled


# ===================================================================
# DCT Spectral Inpainting  (v0.3.0)
# ===================================================================


def _width_to_lambda(width: float, order: int) -> float:
    """
    Derive the Tikhonov regularisation parameter λ from a smoothing width.

    Maps the user's intuitive ``width`` (correlation length in grid points)
    to the spectral penalty λ used in DCT-PLS.  The mapping is derived by
    matching the half-power frequency of a Gaussian kernel with
    σ = width / √12.

    Parameters
    ----------
    width : float
        Smoothing / correlation width in grid points.
    order : int
        Order of the difference penalty (1 = gradient, 2 = curvature).

    Returns
    -------
    lam : float
        Regularisation parameter λ ≥ 0.
    """
    sigma = width / np.sqrt(12.0)
    lam = (sigma**2 / 2.0) ** order
    return lam


def _eigenvalues_dct(n: int, order: int) -> np.ndarray:
    """
    Eigenvalues of the p-th order difference operator in the DCT-II basis.

    For reflective (even-symmetric) boundary conditions the discrete
    Laplacian diagonalises under DCT-II with eigenvalues
    ``2 − 2 cos(π k / N)``, raised to the power ``order``.

    Parameters
    ----------
    n : int
        Length of the axis.
    order : int
        Derivative order (1 = gradient, 2 = curvature, …).

    Returns
    -------
    E : np.ndarray, shape (n,)
        Eigenvalue array.  ``E[0] = 0`` (DC component unpenalised).
    """
    k = np.arange(n, dtype=np.float64)
    E = (2.0 - 2.0 * np.cos(np.pi * k / n)) ** order
    return E


def _eigenvalues_dft(n: int, order: int) -> np.ndarray:
    """
    Eigenvalues of the p-th order difference operator in the DFT basis.

    For periodic boundary conditions the discrete Laplacian diagonalises
    under the DFT with eigenvalues ``2 − 2 cos(2π k / N)``, raised to
    the power ``order``.  Returns the real-FFT half-spectrum
    (k = 0 … N//2).

    Parameters
    ----------
    n : int
        Full length of the periodic axis.
    order : int
        Derivative order.

    Returns
    -------
    E : np.ndarray, shape (n // 2 + 1,)
        Eigenvalue array for the RFFT frequencies.
    """
    n_freq = n // 2 + 1
    k = np.arange(n_freq, dtype=np.float64)
    E = (2.0 - 2.0 * np.cos(2.0 * np.pi * k / n)) ** order
    return E


def _compute_eigenvalues_2d(
    shape: Tuple[int, int],
    order: int,
    az_boundary: str = "reflective",
) -> np.ndarray:
    """
    Build the 2-D eigenvalue tensor E_p[k0, k1].

    Parameters
    ----------
    shape : (int, int)
        Data shape ``(n_axis0, n_axis1)``.
    order : int
        Derivative order.
    az_boundary : str
        Boundary condition for axis-0 (azimuth in polar coordinates).
        ``'reflective'`` → DCT eigenvalues; ``'periodic'`` → DFT eigenvalues.
        Axis-1 (range) is always reflective.

    Returns
    -------
    E : np.ndarray
        2-D eigenvalue tensor.  Shape matches the spectral-domain
        representation: ``(n0, n1)`` for reflective or
        ``(n0 // 2 + 1, n1)`` for periodic axis-0.
    """
    n0, n1 = shape

    # Axis 0 eigenvalues
    if az_boundary == "periodic":
        E0 = _eigenvalues_dft(n0, order)  # (n0//2+1,)
    else:
        E0 = _eigenvalues_dct(n0, order)  # (n0,)

    # Axis 1 eigenvalues (always reflective)
    E1 = _eigenvalues_dct(n1, order)  # (n1,)

    # Outer sum → 2-D tensor
    E = E0[:, np.newaxis] + E1[np.newaxis, :]
    return E


def _forward_transform(
    data: np.ndarray,
    az_boundary: str = "reflective",
) -> np.ndarray:
    """
    Full forward spectral transform (2-D or 1-D).

    For 2-D data:
      - Axis 0: DCT-II (reflective) or RFFT (periodic).
      - Axis 1: DCT-II (reflective).

    For 1-D data:
      - DCT-II (reflective).

    Parameters
    ----------
    data : np.ndarray
        Spatial-domain data (1-D or 2-D, real-valued, no NaN).
    az_boundary : str
        Boundary condition for axis 0.

    Returns
    -------
    spectrum : np.ndarray
        Spectral coefficients (may be complex for periodic axis 0).
    """
    if data.ndim == 1:
        return scipy.fft.dct(data, type=2, norm="ortho")

    # 2-D
    if az_boundary == "periodic":
        # Axis 0: RFFT (periodic)
        step = scipy.fft.rfft(data, axis=0, norm="ortho")
        # Axis 1: DCT (reflective) — works on complex arrays
        return scipy.fft.dct(step, axis=1, type=2, norm="ortho")
    else:
        # Both axes: DCT (reflective)
        step = scipy.fft.dct(data, axis=0, type=2, norm="ortho")
        return scipy.fft.dct(step, axis=1, type=2, norm="ortho")


def _inverse_transform(
    spectrum: np.ndarray,
    n_axis0: int,
    az_boundary: str = "reflective",
) -> np.ndarray:
    """
    Full inverse spectral transform (2-D or 1-D).

    Parameters
    ----------
    spectrum : np.ndarray
        Spectral coefficients.
    n_axis0 : int
        Original length of axis 0 (needed for ``irfft``).
    az_boundary : str
        Boundary condition for axis 0.

    Returns
    -------
    data : np.ndarray
        Spatial-domain data (real-valued).
    """
    if spectrum.ndim == 1:
        return scipy.fft.idct(spectrum, type=2, norm="ortho")

    # 2-D: invert in reverse order (axis 1 first, then axis 0)
    if az_boundary == "periodic":
        step = scipy.fft.idct(spectrum, axis=1, type=2, norm="ortho")
        return scipy.fft.irfft(step, n=n_axis0, axis=0, norm="ortho")
    else:
        step = scipy.fft.idct(spectrum, axis=1, type=2, norm="ortho")
        return scipy.fft.idct(step, axis=0, type=2, norm="ortho")


# ---------------------------------------------------------------------------
# Public API — DCT Spectral Inpainting
# ---------------------------------------------------------------------------


def dct_inpaint(
    data: np.ndarray,
    width: float,
    coordinates: str = "cartesian",
    order: int = 2,
    max_iter: int = 100,
    tol: float = 1e-5,
    init: str = "linear",
    smooth_output: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Fill gaps via DCT-domain Penalised Least Squares (spectral inpainting).

    Minimises the functional

        J(û) = ‖W·(y − û)‖² + λ·‖Δᵖ û‖²

    where W = diag(mask), Δᵖ is the p-th order difference operator, and
    λ is derived automatically from ``width``.  With the default
    ``order=2`` (bi-harmonic penalty), this is equivalent to thin-plate
    spline interpolation: it preserves curvature across gaps rather than
    flattening them (as the Laplace / heat-equation approach of
    ``iterative_gap_fill`` does).

    Algorithm
    ---------
    1. **Initialise** gap values with fast linear interpolation (baseline).
    2. **Iterate** (Garcia, 2010):

       a. Compose field: z = M·y + (1−M)·û   (observed where valid,
          current estimate where gap).
       b. Forward spectral transform (DCT or RFFT depending on BC).
       c. Apply filter: Û = Z / (1 + λ·Eₚ).
       d. Inverse transform → new estimate û.
       e. Convergence check on gap pixels.

    The linear-interpolation initialisation ensures the DCT/FFT operates
    on a fully-populated (no NaN) continuous field from the very first
    iteration, accelerating convergence significantly.

    Parameters
    ----------
    data : np.ndarray
        Input data with NaNs marking gaps.  1-D or 2-D.
    width : float
        Correlation length scale in grid points.  Controls smoothness
        of the reconstruction: larger width → smoother fill across gaps.
        Internally mapped to the Tikhonov parameter λ.
    coordinates : str, default ``'cartesian'``
        Coordinate system: ``'cartesian'`` or ``'polar'``.
    order : int, default 2
        Order of the smoothness penalty:

        * 1 — gradient penalty (Laplace equation, membrane).
        * 2 — curvature penalty (bi-harmonic, thin-plate spline).
        * 3 — third-order (very smooth).
    max_iter : int, default 100
        Maximum number of iterations.
    tol : float, default 1e-5
        Convergence tolerance (relative L2 change of gap pixels).
    init : str, default ``'linear'``
        Initialisation strategy for gap values:

        * ``'linear'`` — axis-wise linear interpolation (default).
        * ``'zeros'`` — fill gaps with zero (useful for zero-mean fields).
    smooth_output : bool, default False
        If True, the final result is the spectrally-smoothed field
        (valid + gap).  If False, valid data is preserved exactly.
    **kwargs
        Keyword arguments forwarded to polar smoothing when
        ``coordinates='polar'``:

        * ``az_res_deg`` (float) — azimuth resolution in degrees.
        * ``az_boundary`` (str) — ``'reflective'`` or ``'periodic'``.

    Returns
    -------
    filled : np.ndarray
        Data with gaps filled.  Valid pixels are preserved exactly
        unless ``smooth_output=True``.

    See Also
    --------
    iterative_gap_fill : Simpler iterative diffusion approach (v0.2.x).

    References
    ----------
    Garcia, D. (2010). Robust smoothing of gridded data in one and higher
    dimensions with missing values. *Computational Statistics & Data
    Analysis*, 54(4), 1167–1178.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> data = np.sin(x)
    >>> data[30:50] = np.nan
    >>> filled = dct_inpaint(data, width=10.0)
    >>> np.all(~np.isnan(filled))
    True
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if data.ndim not in (1, 2):
        raise ValueError(f"dct_inpaint supports 1-D and 2-D data, got {data.ndim}-D")

    valid_mask = ~np.isnan(data)
    if np.all(valid_mask):
        return data.copy()
    if not np.any(valid_mask):
        warnings.warn(
            "All values are NaN; returning zeros.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros_like(data)

    gap_mask = ~valid_mask
    y_obs = np.where(valid_mask, data, 0.0)  # observed data, 0 in gaps

    # ------------------------------------------------------------------
    # Determine boundary conditions
    # ------------------------------------------------------------------
    az_boundary = kwargs.get("az_boundary", "reflective")
    if coordinates == "cartesian":
        az_boundary = "reflective"  # Cartesian always reflective

    # ------------------------------------------------------------------
    # Compute regularisation parameter and eigenvalue tensor
    # ------------------------------------------------------------------
    lam = _width_to_lambda(width, order)

    if data.ndim == 1:
        E = _eigenvalues_dct(data.shape[0], order)
    else:
        E = _compute_eigenvalues_2d(data.shape, order, az_boundary)

    # Spectral filter denominator: Γ⁻¹ = 1 + λ·Eₚ
    gamma_inv = 1.0 + lam * E

    # ------------------------------------------------------------------
    # Phase 1: Initialisation (linear interpolation baseline)
    # ------------------------------------------------------------------
    if init == "linear":
        if data.ndim == 1:
            filled = _linear_init_1d(data)
        else:
            filled = _linear_init_2d(data)
    elif init == "zeros":
        filled = y_obs.copy()
    else:
        raise ValueError(f"Unknown init strategy '{init}'. Choose 'linear' or 'zeros'.")

    # ------------------------------------------------------------------
    # Phase 2: Iterative DCT-PLS
    # ------------------------------------------------------------------
    n_axis0 = data.shape[0]

    for iteration in range(max_iter):
        prev_gap_vals = filled[gap_mask].copy()

        # Step (a): compose field — observed where valid, estimate where gap
        z = np.where(valid_mask, y_obs, filled)

        # Step (b): forward transform
        Z = _forward_transform(z, az_boundary)

        # Step (c): spectral filter  Û = Z / (1 + λ·Eₚ)
        U = Z / gamma_inv

        # Step (d): inverse transform
        filled = _inverse_transform(U, n_axis0, az_boundary)

        # Step (e): convergence check on gap pixels only
        new_gap_vals = filled[gap_mask]
        diff = np.linalg.norm(new_gap_vals - prev_gap_vals)
        norm = np.linalg.norm(new_gap_vals) + 1e-10
        rel_change = diff / norm

        if rel_change < tol:
            break

    # ------------------------------------------------------------------
    # Enforce data fidelity: valid pixels preserved exactly
    # ------------------------------------------------------------------
    if not smooth_output:
        filled[valid_mask] = data[valid_mask]

    return filled

"""
2D Polarimetric Smoothing.

This module provides smoothing for polar coordinate data (Azimuth x Range).
It handles the non-uniform physical width of azimuth beams by adapting
the smoothing kernel in the spectral domain.
"""

import warnings
from typing import Tuple

import numpy as np
import scipy.fft

from ._kernels import KernelLike, normalize_kernel_types
from ._widths import WidthLike, normalize_widths
from .core import get_dct_transfer_function


def _azimuth_spacing_radians(az_res_deg: float) -> float:
    """Validate a positive real scalar azimuth spacing and convert it to radians."""
    message = "az_res_deg must be a finite, positive real scalar"
    try:
        value = np.asarray(az_res_deg)
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise ValueError(message)
        degrees = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(degrees) or degrees <= 0:
        raise ValueError(message)
    radians = float(np.deg2rad(degrees))
    if radians == 0:
        raise ValueError(f"{message}; spacing is too small to represent in radians")
    return radians


def compute_polar_transfer_functions(
    shape: Tuple[int, int],
    az_res_deg: float,
    width_pixels: WidthLike,
    kernel_type: KernelLike = "gaussian",
    az_boundary: str = "reflective",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer functions for polar smoothing.

    The azimuth kernel width adapts with range to maintain constant physical
    width.

    Parameters
    ----------
    shape : tuple
        (n_azimuth, n_range).
    az_res_deg : float
        Azimuth resolution in degrees, as a finite, positive real scalar.
    width_pixels : float or sequence of float
        Width specification in pixels as ``(width_azimuth, width_range)``.
        A scalar applies equal nominal widths. At range index ``r`` (starting
        at 1), the azimuth width in beams is ``width_azimuth / (r * dtheta)``,
        where ``dtheta`` is in radians. Range width is in range gates.
    kernel_type : str or sequence of str, default='gaussian'
        'boxcar', 'boxcar_discrete', or 'gaussian'. A string applies to both
        dimensions. A pair selects ``(kernel_azimuth, kernel_range)``.
        Gaussian ``sigma = width / sqrt(12)`` uses the effective width on
        each axis. Discrete boxcars round that width to an integer, clamp
        it to at least 1, and increment even values to obtain an odd window.
    az_boundary : str, default='reflective'
        Azimuth boundary handling: ``'reflective'`` (DCT) or
        ``'periodic'`` (RFFT).

    Returns
    -------
    H_az : np.ndarray
        Azimuth transfer function.
        Shape is ``(n_azimuth, n_range)`` for reflective boundaries and
        ``(n_azimuth // 2 + 1, n_range)`` for periodic boundaries.
    H_range : np.ndarray
        Range transfer function of shape (n_range,).

    Raises
    ------
    ValueError
        If the kernel specification is not a supported string or a 1-D
        pair of supported strings, the azimuth boundary is unknown, or
        ``az_res_deg`` is not a finite, positive real scalar.
    """
    n_az, n_range = shape
    az_res_rad = _azimuth_spacing_radians(az_res_deg)
    width_azimuth, width_range = normalize_widths(width_pixels, 2, name="width_pixels")
    kernel_azimuth, kernel_range = normalize_kernel_types(kernel_type, 2)

    # Range transfer function (reflective/DCT).
    H_range = get_dct_transfer_function(n_range, kernel_range, float(width_range))

    # Azimuth widths adapt with range: w_az(r) = width_azimuth / (r * dtheta).
    r_indices = np.arange(1, n_range + 1)
    w_beams = float(width_azimuth) / (r_indices * az_res_rad)

    if az_boundary == "reflective":
        H_az_T = np.zeros((n_range, n_az))
        for i in range(n_range):
            H_az_T[i, :] = get_dct_transfer_function(n_az, kernel_azimuth, w_beams[i])
        H_az = H_az_T.T

    elif az_boundary == "periodic":
        n_freq = n_az // 2 + 1
        H_az_T = np.zeros((n_range, n_freq))
        k = np.arange(n_freq)

        for i in range(n_range):
            width = w_beams[i]

            if kernel_azimuth in ("boxcar", "boxcar_discrete"):
                if kernel_azimuth == "boxcar_discrete":
                    w_int = max(1, int(np.round(width)))
                    if w_int % 2 == 0:
                        w_int += 1
                    width = float(w_int)

                # For odd integer widths this is the normalized cosine sum
                # of a centered discrete boxcar, including circular wraps.
                theta_half = (np.pi * k) / n_az
                H = np.zeros(n_freq)
                H[0] = 1.0
                mask = k > 0

                num = np.sin(width * theta_half[mask])
                den = np.sin(theta_half[mask])
                valid = np.abs(den) > 1e-15

                H[mask] = np.where(valid, num / (den * width), 0.0)
                H_az_T[i, :] = H

            elif kernel_azimuth == "gaussian":
                omega = (2 * np.pi * k) / n_az
                sigma = width / np.sqrt(12)
                H_az_T[i, :] = np.exp(-0.5 * (omega * sigma) ** 2)

        H_az = H_az_T.T

    else:
        raise ValueError(f"Unknown azimuth boundary: {az_boundary}")

    return H_az, H_range


def compute_polar_transfer_functions_v2(
    shape: Tuple[int, int],
    az_res_deg: float,
    width_pixels: WidthLike,
    kernel_type: KernelLike = "gaussian",
    az_boundary: str = "reflective",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper for ``compute_polar_transfer_functions``.

    Parameters
    ----------
    shape : tuple of int
        (n_azimuth, n_range).
    az_res_deg : float
        Azimuth resolution in degrees, as a finite, positive real scalar.
    width_pixels : float or sequence of float
        Scalar width or ``(width_azimuth, width_range)``; see
        ``compute_polar_transfer_functions`` for the adaptive width convention.
    kernel_type : str or sequence of str, default='gaussian'
        A supported kernel name or ``(kernel_azimuth, kernel_range)`` pair.
    az_boundary : str, default='reflective'
        'reflective' or 'periodic'.

    Returns
    -------
    H_az : np.ndarray
        Azimuth transfer function, varying with range.
    H_range : np.ndarray
        Range transfer function.
    """
    return compute_polar_transfer_functions(
        shape=shape,
        az_res_deg=az_res_deg,
        width_pixels=width_pixels,
        kernel_type=kernel_type,
        az_boundary=az_boundary,
    )


def smooth_polar(
    data: np.ndarray,
    width_pixels: WidthLike,
    az_res_deg: float = 1.0,
    az_boundary: str = "reflective",
    range_boundary: str = "reflective",
    kernel_type: KernelLike = "gaussian",
) -> np.ndarray:
    """
    Apply smoothing to 2D polar data (Azimuth x Range).

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (n_azimuth, n_range).
    width_pixels : float or sequence of float
        Width specification in pixels as ``(width_azimuth, width_range)``.
        A scalar applies equal nominal widths. Azimuth width adapts with
        range; see ``compute_polar_transfer_functions`` for units and scaling.
    az_res_deg : float
        Azimuth resolution in degrees, as a finite, positive real scalar.
    az_boundary : str
        'reflective' (default) or 'periodic'.
    range_boundary : str
        'reflective' (default).
    kernel_type : str or sequence of str, default='gaussian'
        'boxcar', 'boxcar_discrete', or 'gaussian'. A string applies to both
        dimensions. A pair selects ``(kernel_azimuth, kernel_range)``,
        independently of whether ``width_pixels`` is scalar or a pair.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data with the same shape as input.

    Raises
    ------
    ValueError
        If data is not 2D, the kernel specification is invalid, or the
        azimuth boundary is unknown, or ``az_res_deg`` is not a finite,
        positive real scalar.
    NotImplementedError
        If the range boundary is not 'reflective'.

    Notes
    -----
    Smoothing applies the range-adaptive azimuth filter first, then the range
    filter. Reversing this order generally changes the result. All three
    kernels support both azimuth boundary modes. Input should be finite;
    use ``dct_mean`` or ``dct_smooth`` for data containing NaNs.

    Examples
    --------
    Apply a circular boxcar in azimuth and a Gaussian in range:

    >>> data = np.ones((36, 20))
    >>> smoothed = smooth_polar(
    ...     data, width_pixels=(5.0, 3.0), az_res_deg=10.0,
    ...     kernel_type=("boxcar", "gaussian"), az_boundary="periodic",
    ... )
    >>> np.allclose(smoothed, data)
    True
    """
    # Validation
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D (az, range), got {data.ndim}D")

    n_az, n_range = data.shape
    if n_az > 720 and n_range < n_az:
        warnings.warn(
            f"Data shape ({n_az}, {n_range}) has large first dimension. "
            "Ensure format is (n_azimuth, n_range).",
            UserWarning,
        )

    # Get transfer functions
    H_az, H_range = compute_polar_transfer_functions(
        data.shape, az_res_deg, width_pixels, kernel_type, az_boundary
    )

    # 1. Azimuth Smoothing (Axis 0)
    if az_boundary == "reflective":
        # DCT
        X = scipy.fft.dct(data, axis=0, type=2, norm="ortho")
        Y = X * H_az
        step1 = scipy.fft.idct(Y, axis=0, type=2, norm="ortho")

    elif az_boundary == "periodic":
        # Real FFT
        X = scipy.fft.rfft(data, axis=0, norm="ortho")
        # H_az shape matches rfft output (n_az//2 + 1, n_range)
        Y = X * H_az
        step1 = scipy.fft.irfft(Y, n=n_az, axis=0, norm="ortho")

    else:
        raise ValueError(f"Unknown azimuth boundary: {az_boundary}")

    # 2. Range Smoothing (Axis 1)
    # Always reflective (DCT) for now
    if range_boundary != "reflective":
        raise NotImplementedError("Only 'reflective' range boundary supported")

    # H_range is (n_range,), broadcast to (n_az, n_range)
    X = scipy.fft.dct(step1, axis=1, type=2, norm="ortho")
    Y = X * H_range.reshape(1, -1)
    result = scipy.fft.idct(Y, axis=1, type=2, norm="ortho")

    return result

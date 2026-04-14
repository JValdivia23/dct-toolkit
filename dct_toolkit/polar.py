"""
2D Polarimetric Smoothing.

This module provides smoothing for polar coordinate data (Azimuth x Range).
It handles the non-uniform physical width of azimuth beams by adapting
the smoothing kernel in the spectral domain.
"""

import numpy as np
import scipy.fft
import warnings
from typing import Tuple

from ._widths import WidthLike, normalize_widths
from .core import get_dct_transfer_function


def compute_polar_transfer_functions(
    shape: Tuple[int, int],
    az_res_deg: float,
    width_pixels: WidthLike,
    kernel_type: str = "gaussian",
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
        Azimuth resolution in degrees.
    width_pixels : float or sequence of float
        Width specification in pixels as ``(width_azimuth, width_range)``.
        Scalar input applies isotropic width in both dimensions.
    kernel_type : str
        'boxcar', 'boxcar_discrete', 'gaussian'.
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
    """
    n_az, n_range = shape
    az_res_rad = np.deg2rad(az_res_deg)
    width_azimuth, width_range = normalize_widths(width_pixels, 2, name="width_pixels")

    # Range transfer function (reflective/DCT).
    H_range = get_dct_transfer_function(n_range, kernel_type, float(width_range))

    # Azimuth widths adapt with range: w_az(r) = width_azimuth / (r * dtheta).
    r_indices = np.arange(1, n_range + 1)
    w_beams = float(width_azimuth) / (r_indices * az_res_rad)

    if az_boundary == "reflective":
        H_az_T = np.zeros((n_range, n_az))
        for i in range(n_range):
            H_az_T[i, :] = get_dct_transfer_function(n_az, kernel_type, w_beams[i])
        H_az = H_az_T.T

    elif az_boundary == "periodic":
        n_freq = n_az // 2 + 1
        H_az_T = np.zeros((n_range, n_freq))
        k = np.arange(n_freq)

        for i in range(n_range):
            width = w_beams[i]

            if kernel_type == "boxcar":
                theta_half = (np.pi * k) / n_az
                H = np.zeros(n_freq)
                H[0] = 1.0
                mask = k > 0

                num = np.sin(width * theta_half[mask])
                den = np.sin(theta_half[mask])
                valid = np.abs(den) > 1e-15

                H[mask] = np.where(valid, num / (den * width), 0.0)
                H_az_T[i, :] = H

            elif kernel_type == "gaussian":
                omega = (2 * np.pi * k) / n_az
                sigma = width / np.sqrt(12)
                H_az_T[i, :] = np.exp(-0.5 * (omega * sigma) ** 2)

            else:
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
    kernel_type: str = "gaussian",
    az_boundary: str = "reflective",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper for ``compute_polar_transfer_functions``.
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
    kernel_type: str = "gaussian",
) -> np.ndarray:
    """
    Apply smoothing to 2D polar data (Azimuth x Range).

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (n_azimuth, n_range).
    width_pixels : float or sequence of float
        Width specification in pixels as ``(width_azimuth, width_range)``.
        Scalar input applies isotropic width in both dimensions.
    az_res_deg : float
        Azimuth resolution in degrees.
    az_boundary : str
        'reflective' (default) or 'periodic'.
    range_boundary : str
        'reflective' (default).
    kernel_type : str
        Kernel type.

    Returns
    -------
    smoothed : np.ndarray
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

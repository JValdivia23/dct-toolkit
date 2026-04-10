"""
DCT Statistical Operations.

This module implements robust local statistics (Mean, Variance, Std, Count)
using Normalized Convolution. This approach naturally handles gaps (NaNs)
without requiring explicit pre-filling.
"""

import numpy as np
from .core import get_dct_transfer_function, dct_convolve_1d
from .cartesian import smooth_cartesian
from .polar import smooth_polar


def _get_smooth_func(coordinates: str):
    """Select smoothing function based on coordinate system."""
    if coordinates == "cartesian":
        return smooth_cartesian
    elif coordinates == "polar":
        return smooth_polar
    else:
        raise ValueError(f"Unknown coordinates: {coordinates}")


def dct_count(
    mask: np.ndarray, width: float, coordinates: str = "cartesian", **kwargs
) -> np.ndarray:
    """
    Compute effective sample count (local density * window area).

    Parameters
    ----------
    mask : np.ndarray
        Boolean or binary mask (1=valid, 0=invalid).
    width : float
        Smoothing width.
    coordinates : str
        'cartesian' or 'polar'.
    **kwargs
        Additional arguments passed to smoothing function (e.g. az_res_deg).

    Returns
    -------
    count : np.ndarray
        Effective count of valid samples within the smoothing window.
    """
    smooth_func = _get_smooth_func(coordinates)

    # Density = Smooth(Indicator)
    density = smooth_func(mask.astype(float), width, **kwargs)

    # Area Calculation
    if coordinates == "cartesian":
        # Area = width^ndim (assuming isotropic width)
        area = width**mask.ndim
    elif coordinates == "polar":
        # Area varies with range: w_az(r) * w_range
        # w_range = width
        # w_az(r) = width / (r * d_theta) [in beam units]
        # But wait, w_beams was used in smoothing.
        # Area in index-space (which DCT operates on) is what matters for "count".
        # DCT smoothing effectively averages over a window defined in index space
        # but weighted by the kernel.
        # For polar, the azimuth kernel width in indices is w_beams[r].
        # So Area[r] = width * w_beams[r]

        n_az, n_range = mask.shape
        az_res_deg = kwargs.get("az_res_deg", 1.0)
        az_res_rad = np.deg2rad(az_res_deg)
        r_indices = np.arange(1, n_range + 1)

        # w_beams[r] is width in azimuth indices
        w_beams = width / (r_indices * az_res_rad)

        # Effective area in (az, range) index space
        area_1d = width * w_beams
        area = np.tile(area_1d, (n_az, 1))
    else:
        area = 1.0

    return density * area


def dct_mean(
    data: np.ndarray,
    width: float,
    coordinates: str = "cartesian",
    mask: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """
    Compute robust local mean using Normalized Convolution.

    Mean = Smooth(Data * Mask) / Smooth(Mask)

    Parameters
    ----------
    data : np.ndarray
        Input data (can contain NaNs).
    width : float
        Smoothing width.
    coordinates : str
        'cartesian' or 'polar'.
    mask : np.ndarray, optional
        Valid data mask. If None, inferred from ~isnan(data).

    Returns
    -------
    mean : np.ndarray
        Local mean (floating-point array).
    """
    smooth_func = _get_smooth_func(coordinates)
    data_array = np.asarray(data)
    out_dtype = np.result_type(data_array.dtype, np.float64)

    if mask is None:
        mask = ~np.isnan(data_array)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data_array.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data shape {data_array.shape}"
        )

    # 1. Numerator: Smooth(Data * Mask)
    # Fill NaNs with 0 for the convolution (they are masked out anyway)
    data_filled = data_array.astype(out_dtype, copy=True)
    data_filled[~mask] = 0.0
    numerator = smooth_func(data_filled, width, **kwargs)

    # 2. Denominator: Smooth(Mask)
    denominator = smooth_func(mask.astype(out_dtype), width, **kwargs)

    # 3. Normalized Ratio
    # Handle division by zero where denominator is very small (no valid data nearby)
    valid_den = denominator > 1e-10
    mean = np.full(data_array.shape, np.nan, dtype=out_dtype)
    np.divide(numerator, denominator, out=mean, where=valid_den)

    return mean


def dct_prefill(
    data: np.ndarray,
    width: float,
    coordinates: str = "cartesian",
    fill_mask: np.ndarray = None,
    max_iter: int = 3,
    **kwargs,
) -> np.ndarray:
    """
    Fill gaps using iterative normalized convolution.

    The function repeatedly applies ``dct_mean`` to grow valid support into
    missing regions. It is useful as a pre-processing step before full-field
    smoothing so spectral filters operate on a continuous field.

    Parameters
    ----------
    data : np.ndarray
        Input data array. NaN values indicate gaps when ``fill_mask`` is None.
    width : float
        Smoothing width used by normalized convolution. Must be > 0.
    coordinates : str, default='cartesian'
        Coordinate mode for smoothing: ``'cartesian'`` or ``'polar'``.
    fill_mask : np.ndarray, optional
        Boolean mask where True marks positions to fill/replace.
        If None, all NaN positions are filled.
    max_iter : int, default=3
        Maximum number of fill iterations.
    **kwargs
        Additional keyword arguments passed to ``dct_mean``
        (e.g. ``az_res_deg``, ``az_boundary``, ``kernel_type``).
        If ``kernel_type`` is not provided, ``'gaussian'`` is used to keep
        prefill behavior stable in low-support regions.

    Returns
    -------
    np.ndarray
        Filled array (floating-point). Non-target values are preserved exactly.

    Notes
    -----
    - ``fill_mask`` uses the convention True = "fill this position".
    - If ``fill_mask`` is provided and a marked value cannot be estimated,
      original finite values are kept.
    """
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    data_array = np.asarray(data)
    out_dtype = np.result_type(data_array.dtype, np.float64)
    original = data_array.astype(out_dtype, copy=True)

    if fill_mask is None:
        target = np.isnan(original)
        use_target_fallback = False
    else:
        target = np.asarray(fill_mask, dtype=bool)
        if target.shape != original.shape:
            raise ValueError(
                f"fill_mask shape {target.shape} does not match data shape {original.shape}"
            )
        use_target_fallback = True

    if not np.any(target):
        return original.copy()

    out = original.copy()
    out[target] = np.nan

    mean_kwargs = dict(kwargs)
    mean_kwargs.setdefault("kernel_type", "gaussian")

    for _ in range(int(max_iter)):
        candidates = target & np.isnan(out)
        if not np.any(candidates):
            break

        valid_current = np.isfinite(out)
        if not np.any(valid_current):
            break

        estimate = dct_mean(
            out,
            width=width,
            coordinates=coordinates,
            mask=valid_current,
            **mean_kwargs,
        )
        newly_filled = candidates & np.isfinite(estimate)
        if not np.any(newly_filled):
            break

        out[newly_filled] = estimate[newly_filled]

    if use_target_fallback:
        fallback = target & np.isnan(out) & np.isfinite(original)
        out[fallback] = original[fallback]

    return out


def dct_variance(
    data: np.ndarray,
    width: float,
    coordinates: str = "cartesian",
    mask: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """
    Compute robust local variance.

    Var = E[X^2] - (E[X])^2
    Both expectations are computed using Normalized Convolution.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    width : float
        Smoothing width.

    Returns
    -------
    variance : np.ndarray
    """
    data_array = np.asarray(data)
    out_dtype = np.result_type(data_array.dtype, np.float64)
    data_float = data_array.astype(out_dtype, copy=False)

    if mask is None:
        mask = ~np.isnan(data_float)

    # E[X]
    mean = dct_mean(data_float, width, coordinates, mask, **kwargs)

    # E[X^2]
    data_sq = data_float**2
    mean_sq = dct_mean(data_sq, width, coordinates, mask, **kwargs)

    # Var = E[X^2] - E[X]^2
    # Use maximum(0) to avoid negative variance due to numerical precision
    variance = np.maximum(mean_sq - mean**2, 0.0)

    return variance


def dct_std(
    data: np.ndarray,
    width: float,
    coordinates: str = "cartesian",
    mask: np.ndarray = None,
    **kwargs,
) -> np.ndarray:
    """Compute robust local standard deviation."""
    var = dct_variance(data, width, coordinates, mask, **kwargs)
    return np.sqrt(var)

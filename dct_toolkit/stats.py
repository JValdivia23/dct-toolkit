"""
DCT Statistical Operations.

This module implements robust local statistics (Mean, Variance, Std, Count)
using Normalized Convolution. This approach naturally handles gaps (NaNs)
without requiring explicit pre-filling.
"""

from typing import Optional

import numpy as np
import scipy.ndimage
from .core import get_dct_transfer_function, dct_convolve_1d
from .cartesian import smooth_cartesian
from .polar import smooth_polar


_PREFILL_MAX_ITER_CAP = 20


def _get_smooth_func(coordinates: str):
    """Select smoothing function based on coordinate system."""
    if coordinates == "cartesian":
        return smooth_cartesian
    elif coordinates == "polar":
        return smooth_polar
    else:
        raise ValueError(f"Unknown coordinates: {coordinates}")


def _fill_nan_nearest_along_axis(
    data: np.ndarray,
    fill_target: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Fill target NaNs by nearest finite neighbor along one axis."""
    out = np.asarray(data).copy()
    target = np.asarray(fill_target, dtype=bool)

    if out.shape != target.shape:
        raise ValueError(
            f"fill_target shape {target.shape} does not match data shape {out.shape}"
        )

    axis = int(axis)
    if axis < 0:
        axis += out.ndim
    if axis < 0 or axis >= out.ndim:
        raise ValueError(f"Invalid axis {axis} for shape {out.shape}")

    moved = np.moveaxis(out, axis, -1)
    moved_target = np.moveaxis(target, axis, -1)
    n_axis = moved.shape[-1]
    flat_data = moved.reshape(-1, n_axis)
    flat_target = moved_target.reshape(-1, n_axis)
    idx = np.arange(n_axis)

    for row, row_target in zip(flat_data, flat_target):
        unresolved = row_target & np.isnan(row)
        if not np.any(unresolved):
            continue

        finite_idx = np.flatnonzero(np.isfinite(row))
        if finite_idx.size == 0:
            continue

        miss_idx = idx[unresolved]
        insert_pos = np.searchsorted(finite_idx, miss_idx, side="left")

        left_pos = np.clip(insert_pos - 1, 0, finite_idx.size - 1)
        right_pos = np.clip(insert_pos, 0, finite_idx.size - 1)
        left_idx = finite_idx[left_pos]
        right_idx = finite_idx[right_pos]

        use_left = (miss_idx - left_idx) <= (right_idx - miss_idx)
        nearest_idx = np.where(use_left, left_idx, right_idx)
        row[miss_idx] = row[nearest_idx]

    return np.moveaxis(flat_data.reshape(moved.shape), -1, axis)


def _fill_nan_nearest(
    data: np.ndarray,
    fill_target: np.ndarray,
    primary_axis: int,
) -> np.ndarray:
    """Fill target NaNs using nearest neighbors with axis-first/global fallback."""
    out = np.asarray(data).copy()
    target = np.asarray(fill_target, dtype=bool)

    if out.shape != target.shape:
        raise ValueError(
            f"fill_target shape {target.shape} does not match data shape {out.shape}"
        )

    if out.ndim == 0:
        return out

    axis = int(primary_axis)
    if axis < 0:
        axis += out.ndim
    if axis < 0 or axis >= out.ndim:
        raise ValueError(f"Invalid primary_axis {primary_axis} for shape {out.shape}")

    unresolved = target & np.isnan(out)
    if not np.any(unresolved):
        return out

    axis_order = [axis] + [ax for ax in range(out.ndim) if ax != axis]

    # First pass: fast axis-wise nearest fills.
    for ax in axis_order:
        if not np.any(unresolved):
            break
        out = _fill_nan_nearest_along_axis(out, unresolved, axis=ax)
        unresolved = target & np.isnan(out)

    if not np.any(unresolved):
        return out

    # Final fallback: global nearest finite neighbor in N-D index space.
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    nearest_indices = scipy.ndimage.distance_transform_edt(
        ~finite,
        return_distances=False,
        return_indices=True,
    )
    nearest_coords = tuple(idx[unresolved] for idx in nearest_indices)
    out[unresolved] = out[nearest_coords]
    return out


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
    max_iter: Optional[int] = None,
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
    max_iter : int or None, default=None
        Maximum number of normalized-convolution iterations.
        If None, iterate until convergence or until a safety cap of 20
        iterations is reached.
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
    - After iterative normalized-convolution filling, any remaining target NaNs
      are filled by nearest-neighbor propagation to guarantee finite output when
      at least one finite value exists.
    """
    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")
    if max_iter is not None and max_iter < 1:
        raise ValueError(f"max_iter must be >= 1 or None, got {max_iter}")

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

    max_iter_eff = _PREFILL_MAX_ITER_CAP if max_iter is None else int(max_iter)

    for _ in range(max_iter_eff):
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

    unresolved = target & np.isnan(out)
    if np.any(unresolved):
        fill_axis = 1 if (coordinates == "polar" and out.ndim >= 2) else -1
        out = _fill_nan_nearest(out, unresolved, primary_axis=fill_axis)

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

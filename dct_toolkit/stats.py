"""
DCT Statistical Operations.

This module implements robust local statistics (Mean, Variance, Std, Count)
using Normalized Convolution. This approach naturally handles gaps (NaNs)
without requiring explicit pre-filling.
"""

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage
import scipy.spatial

from ._geometry import _resolve_polar_geometry
from ._kernels import normalize_kernel_types
from ._widths import WidthLike, normalize_widths
from .cartesian import smooth_cartesian
from .polar import smooth_polar


_PREFILL_MAX_ITER_CAP = 20
_MEAN_DENOMINATOR_FLOOR = 1e-3
DEFAULT_MIN_EFFECTIVE_DENSITY = 0.35
_MEAN_PRIVATE_OPTIONS = ("_stable_fallback", "_prefill_max_iter", "_denominator_floor")
_POLAR_GEOMETRY_OPTIONS = (
    "kernel_type", "az_res_deg", "az_boundary", "range_boundary", "axis_types",
    "axis_coords", "boundary", "width_units", "angle_units", "filter_axes",
)


def _normalize_width_for_coordinates(
    width: WidthLike,
    coordinates: str,
    data_ndim: int,
    axis_types: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Normalize width input for Cartesian or polar calls."""
    if coordinates == "cartesian":
        expected_ndim = data_ndim
    elif coordinates == "polar":
        expected_ndim = 2 if axis_types is None else data_ndim
    else:
        raise ValueError(f"Unknown coordinates: {coordinates}")
    return normalize_widths(width, expected_ndim, name="width")


def _resolve_smoothing_geometry(
    shape: Tuple[int, ...],
    width: WidthLike,
    coordinates: str,
    **kwargs: Any,
) -> Any:
    """Validate smoothing options without filtering, returning polar geometry if used."""
    _normalize_width_for_coordinates(width, coordinates, len(shape), kwargs.get("axis_types"))
    if coordinates == "cartesian":
        normalize_kernel_types(kwargs.get("kernel_type", "gaussian"), len(shape))
        return None
    geometry_kwargs = {key: kwargs[key] for key in _POLAR_GEOMETRY_OPTIONS if key in kwargs}
    return _resolve_polar_geometry(shape, width, **geometry_kwargs)


def _smoothing_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Separate private normalized-convolution controls from backend options."""
    return {key: value for key, value in kwargs.items() if key not in _MEAN_PRIVATE_OPTIONS}


def _get_smooth_func(coordinates: str) -> Callable[..., np.ndarray]:
    """Select smoothing function based on coordinate system."""
    if coordinates == "cartesian":
        return smooth_cartesian
    elif coordinates == "polar":
        return smooth_polar
    else:
        raise ValueError(f"Unknown coordinates: {coordinates}")


def _smooth_with_coordinates(
    data: np.ndarray,
    width: WidthLike,
    coordinates: str,
    **kwargs: Any,
) -> np.ndarray:
    """Apply smoothing while mapping width to Cartesian/polar signatures."""
    smooth_func = _get_smooth_func(coordinates)
    if coordinates == "polar":
        return smooth_func(data, width_pixels=width, **kwargs)
    return smooth_func(data, width, **kwargs)


def _smooth_completed_field(
    data: np.ndarray,
    width: WidthLike,
    coordinates: str,
    geometry: Any = None,
    **kwargs: Any,
) -> np.ndarray:
    """Smooth a completed field without sending unsupported N-D domains through FFTs."""
    if geometry is None or geometry.legacy:
        return _smooth_with_coordinates(data, width, coordinates, **kwargs)

    finite = np.isfinite(data)
    if geometry.active_axes:
        supported = np.any(finite, axis=tuple(geometry.active_axes), keepdims=True)
    else:
        # With no filtering axes every individual cell is its own domain.
        supported = finite
    safe_data = np.where(supported, data, 0.0)
    result = _smooth_with_coordinates(safe_data, width, coordinates, **kwargs)
    return np.where(supported, result, np.nan)


def _fill_nan_nearest_along_axis(
    data: np.ndarray,
    fill_target: np.ndarray,
    axis: int,
    periodic: bool = False,
) -> np.ndarray:
    """Fill target NaNs along an axis, optionally using circular distances."""
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

        if periodic:
            left_idx = finite_idx[(insert_pos - 1) % finite_idx.size]
            right_idx = finite_idx[insert_pos % finite_idx.size]
            # Resolve ties toward the preceding beam, independently of the seam.
            use_left = (miss_idx - left_idx) % n_axis <= (right_idx - miss_idx) % n_axis
        else:
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
    periodic_axis: Optional[int] = None,
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

    if periodic_axis is not None:
        periodic_axis = int(periodic_axis)
        if periodic_axis < 0:
            periodic_axis += out.ndim
        if periodic_axis < 0 or periodic_axis >= out.ndim:
            raise ValueError(f"Invalid periodic_axis {periodic_axis} for shape {out.shape}")

    unresolved = target & np.isnan(out)
    if not np.any(unresolved):
        return out

    axis_order = [axis] + [ax for ax in range(out.ndim) if ax != axis]

    # First pass: fast axis-wise nearest fills.
    for ax in axis_order:
        if not np.any(unresolved):
            break
        out = _fill_nan_nearest_along_axis(
            out, unresolved, axis=ax, periodic=(ax == periodic_axis)
        )
        unresolved = target & np.isnan(out)

    if not np.any(unresolved):
        return out

    # Final fallback: global nearest finite neighbor in N-D index space.
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    search_finite = finite
    center = [slice(None)] * out.ndim
    if periodic_axis is not None:
        # Adjacent copies provide neighbors across either end of the periodic axis.
        search_finite = np.concatenate((finite, finite, finite), axis=periodic_axis)
        n_periodic = out.shape[periodic_axis]
        center[periodic_axis] = slice(n_periodic, 2 * n_periodic)

    nearest_indices = scipy.ndimage.distance_transform_edt(
        ~search_finite,
        return_distances=False,
        return_indices=True,
    )
    nearest_indices = nearest_indices[(slice(None),) + tuple(center)]
    if periodic_axis is not None:
        nearest_indices[periodic_axis] %= n_periodic
    nearest_coords = tuple(idx[unresolved] for idx in nearest_indices)
    out[unresolved] = out[nearest_coords]
    return out


def _fill_nan_nearest_in_domains(
    data: np.ndarray,
    fill_target: np.ndarray,
    active_axes: Tuple[int, ...],
    boundaries: Tuple[str, ...],
    range_axis: Optional[int],
    axis_types: Tuple[str, ...],
) -> np.ndarray:
    """Complete explicit-geometry targets without layout-dependent role priorities.

    Parameters
    ----------
    data : np.ndarray
        Partially completed field.
    fill_target : np.ndarray
        Boolean targets with the same shape as ``data``.
    active_axes : tuple of int
        Selected axes in canonical role order.
    boundaries : tuple of str
        Validated boundaries in array order.
    range_axis : int or None
        Range axis, if present; it has first priority when active.
    axis_types : tuple of str
        Canonical roles in array order, with ``cartesian`` mapped to ``linear``.

    Returns
    -------
    np.ndarray
        Completed copy, leaving unsupported inactive-axis slices missing.

    Notes
    -----
    Range-first and unique-role axis passes retain their axis-wise tie rules.
    Repeated active ``time`` or ``linear`` roles skip individual axis passes;
    residual targets are completed simultaneously by Euclidean nearest search
    over all active axes, wrapping each periodic axis. Exact global ties average
    all nearest finite seed values after the retained passes. Newly completed
    global targets do not become seeds in the same pass. Legacy 2-D completion
    uses a separate helper and retains its original precedence and tie rules.
    """
    out = np.asarray(data).copy()
    target = np.asarray(fill_target, dtype=bool)
    if out.shape != target.shape:
        raise ValueError(
            f"fill_target shape {target.shape} does not match data shape {out.shape}"
        )
    if not active_axes:
        return out

    # Only unique roles have an intrinsic axis priority. Choosing array order
    # between repeated independent roles would change the result on transpose.
    repeated_roles = {
        role for role in ("time", "linear")
        if sum(axis_types[axis] == role for axis in active_axes) > 1
    }
    axis_order = [axis for axis in active_axes if axis_types[axis] not in repeated_roles]
    if range_axis in axis_order:
        axis_order.remove(range_axis)
        axis_order.insert(0, range_axis)
    unresolved = target & np.isnan(out)
    for axis in axis_order:
        if not np.any(unresolved):
            return out
        out = _fill_nan_nearest_along_axis(
            out, unresolved, axis=axis, periodic=(boundaries[axis] == "periodic")
        )
        unresolved = target & np.isnan(out)

    if not np.any(unresolved):
        return out

    batch_axes = tuple(axis for axis in range(out.ndim) if axis not in active_axes)
    permutation = batch_axes + tuple(active_axes)
    domains = np.transpose(out, permutation)
    domain_targets = np.transpose(unresolved, permutation)
    batch_shape = tuple(out.shape[axis] for axis in batch_axes)
    # Zero box lengths make the corresponding tree dimensions nonperiodic.
    # Mixed toroidal distances avoid 3**n_periodic copies of the full domain.
    boxsize = np.array(
        [out.shape[axis] if boundaries[axis] == "periodic" else 0 for axis in active_axes],
        dtype=float,
    )
    periodic = boxsize > 0
    periods = boxsize[periodic].astype(np.intp)
    for batch_index in np.ndindex(batch_shape):
        domain = domains[batch_index]
        domain_target = domain_targets[batch_index]
        if not np.any(domain_target):
            continue
        finite_coords = np.column_stack(np.nonzero(np.isfinite(domain)))
        if finite_coords.shape[0] == 0:
            continue
        target_coords = np.column_stack(np.nonzero(domain_target))
        tree = scipy.spatial.cKDTree(finite_coords, boxsize=boxsize)
        distances, nearest = tree.query(target_coords, k=2)
        completed = domain[tuple(finite_coords[nearest[:, 0]].T)]
        # A single seed yields an infinite second distance, so it needs no
        # special branch. Radius neighborhoods are only built for actual ties.
        tied_targets = np.flatnonzero(distances[:, 0] == distances[:, 1])
        for target_index in tied_targets:
            point = target_coords[target_index]
            radius = np.nextafter(distances[target_index, 0], np.inf)
            neighbors = np.asarray(tree.query_ball_point(point, radius), dtype=np.intp)
            deltas = np.abs(finite_coords[neighbors] - point)
            if np.any(periodic):
                deltas[:, periodic] = np.minimum(deltas[:, periodic], periods - deltas[:, periodic])
            # Integer squared index distances distinguish exact ties from any
            # slightly farther points included by the expanded search radius.
            squared_distances = np.sum(deltas**2, axis=1)
            tied_seeds = neighbors[squared_distances == squared_distances.min()]
            tied_values = domain[tuple(finite_coords[tied_seeds].T)]
            # Fix floating-point summation order independently of array layout.
            completed[target_index] = np.mean(np.sort(tied_values))
        domain[domain_target] = completed
    return out


def dct_count(
    mask: np.ndarray,
    width: WidthLike,
    coordinates: str = "cartesian",
    restore_input_nan: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """
    Compute nominal weighted support (local density * nominal window support).

    Parameters
    ----------
    mask : np.ndarray
        Boolean or binary mask (1=valid, 0=invalid).
    width : float or sequence of float
        Smoothing width. A scalar applies the same width to every axis.
        A sequence follows array axis order and must match dimensionality
        for Cartesian or explicit N-D polar geometry. Legacy polar calls
        use ``(width_azimuth, width_range)``.
    coordinates : str
        'cartesian' or 'polar'.
    restore_input_nan : bool, default=True
        If True, output is set to NaN where ``mask`` is False.
    **kwargs
        Additional arguments passed to smoothing function (e.g. az_res_deg).
        ``kernel_type`` accepts a string or one kernel name per array axis.
        Explicit polar geometry accepts ``axis_types``, ``axis_coords``,
        ``boundary``, ``width_units``, ``angle_units``, and ``filter_axes``;
        all per-axis sequences follow array axis order. Legacy polar calls
        accept an (azimuth, range) kernel pair. See ``smooth_polar`` for units.

    Returns
    -------
    count : np.ndarray
        Nominal weighted support of valid samples. This is not statistical
        effective sample size or a literal count of distinct neighbors.
        Output shape matches the mask, including a zero-dimensional array
        for a scalar mask.

    Notes
    -----
    Support is ``clip(smooth(mask), 0, 1) * nominal_support``. In explicit
    polar mode, the shared geometry resolver determines the nominal support
    from requested grid-equivalent widths on active axes only. Inactive axes
    are independent domains; explicit singleton axes contribute one.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    geometry = _resolve_smoothing_geometry(mask_bool.shape, width, coordinates, **kwargs)

    # Density = Smooth(Indicator)
    density = _smooth_with_coordinates(
        mask_bool.astype(float), width, coordinates, **kwargs
    )

    # Keep density physically valid for indicator masks.
    density = np.nan_to_num(density, nan=0.0, posinf=1.0, neginf=0.0)
    density = np.clip(density, 0.0, 1.0)

    # Nominal support is a width convention, not statistical effective sample size.
    if coordinates == "cartesian":
        nominal_support = float(np.prod(normalize_widths(width, mask_bool.ndim)))
    else:
        nominal_support = geometry.nominal_support

    count = np.asarray(density * nominal_support)
    if restore_input_nan:
        count = count.astype(np.result_type(count.dtype, np.float64), copy=False)
        count[~mask_bool] = np.nan
    return count


def dct_mean(
    data: np.ndarray,
    width: WidthLike,
    coordinates: str = "cartesian",
    mask: Optional[np.ndarray] = None,
    restore_input_nan: bool = True,
    min_effective_density: Optional[float] = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Compute robust local mean using Normalized Convolution.

    Mean = Smooth(Data * Mask) / Smooth(Mask)

    Parameters
    ----------
    data : np.ndarray
        Input data (can contain NaNs).
    width : float or sequence of float
        Smoothing width. A scalar applies the same width to every axis.
        A sequence follows array axis order and must match dimensionality
        for Cartesian or explicit N-D polar geometry. Legacy polar calls
        use ``(width_azimuth, width_range)``.
    coordinates : str
        'cartesian' or 'polar'.
    mask : np.ndarray, optional
        Valid data mask. If None, inferred from finite input values.
    restore_input_nan : bool, default=True
        If True, output is set to NaN where input support is invalid.
    min_effective_density : float or None, optional
        When positive, set final output to NaN wherever the original raw mask
        response ``smooth(valid_mask)`` is below this threshold. This is a
        dimensionless support diagnostic, not statistical effective sample size.
        Signed-weight kernels can produce responses outside [0, 1]; unlike
        ``dct_count / nominal_support``, this gate does not clip the response.
        ``None`` (default) or a non-positive value disables this additional gate,
        retaining the internal ``_MEAN_DENOMINATOR_FLOOR`` stability rule.
    **kwargs
        Additional arguments passed to smoothing, including ``kernel_type``
        and the polar ``axis_types``, ``axis_coords``, ``boundary``,
        ``width_units``, ``angle_units``, and ``filter_axes`` options. Per-axis
        sequences follow array axis order; legacy polar calls use (azimuth,
        range) pairs. The same complete linear operator filters data and mask,
        with normalization only after all filters. See ``smooth_polar`` for units.

    Returns
    -------
    mean : np.ndarray
        Local mean (floating-point array).

    Notes
    -----
    The default ratio path requires a finite original mask response strictly
    greater than ``1e-3``. At other positions this function uses a selective
    prefill-and-smooth fallback when support exists, not additive-epsilon
    regularization. A positive ``min_effective_density`` gate is applied after
    fallback, but tests the original raw mask response, not the prefilled values
    or support grown during prefill.
    In explicit N-D polar mode, all inactive axes define isolated domains.
    A domain with no finite valid support remains NaN, including during fallback.
    """
    stable_fallback = bool(kwargs.pop("_stable_fallback", True))
    prefill_max_iter = kwargs.pop("_prefill_max_iter", 3)
    den_floor = kwargs.pop("_denominator_floor", None)
    if den_floor is None:
        den_floor = _MEAN_DENOMINATOR_FLOOR if stable_fallback else 1e-10

    data_array = np.asarray(data)
    geometry = _resolve_smoothing_geometry(data_array.shape, width, coordinates, **kwargs)
    out_dtype = np.result_type(data_array.dtype, np.float64)

    if mask is None:
        mask = np.isfinite(data_array)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data_array.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data shape {data_array.shape}"
        )

    valid_data = mask & np.isfinite(data_array)

    # 1. Numerator: Smooth(Data * Mask)
    # Fill NaNs with 0 for the convolution (they are masked out anyway)
    data_filled = data_array.astype(out_dtype, copy=True)
    data_filled[~valid_data] = 0.0
    numerator = _smooth_with_coordinates(data_filled, width, coordinates, **kwargs)

    # 2. Denominator: Smooth(Mask)
    denominator = _smooth_with_coordinates(
        valid_data.astype(out_dtype), width, coordinates, **kwargs
    )

    # 3. Normalized Ratio
    # Handle division by zero/instability where denominator is very small.
    valid_den = np.isfinite(denominator) & (denominator > float(den_floor))
    mean = np.full(data_array.shape, np.nan, dtype=out_dtype)
    np.divide(numerator, denominator, out=mean, where=valid_den)

    # 4. Stability fallback in under-supported regions.
    invalid_den = ~valid_den
    if stable_fallback and np.any(invalid_den) and np.any(valid_data):
        fill_target = ~valid_data
        fallback_input = data_array.astype(out_dtype, copy=True)
        fallback_input[fill_target] = np.nan

        prefilled = dct_prefill(
            fallback_input,
            width=width,
            coordinates=coordinates,
            fill_mask=fill_target,
            max_iter=prefill_max_iter,
            _stable_fallback=False,
            **kwargs,
        )
        fallback = _smooth_completed_field(
            prefilled, width, coordinates, geometry=geometry, **kwargs
        )
        mean[invalid_den] = fallback[invalid_den]

    # 5. Optional density-based gate (min_effective_density).
    #    Applied after the stable-fallback so that any value that survives
    #    the fallback is still discarded if the local support is too sparse.
    if min_effective_density is not None and float(min_effective_density) > 0:
        below_threshold = denominator < float(min_effective_density)
        if np.any(below_threshold):
            mean[below_threshold] = np.nan

    if restore_input_nan:
        mean[~valid_data] = np.nan

    return mean


def dct_prefill(
    data: np.ndarray,
    width: WidthLike,
    coordinates: str = "cartesian",
    fill_mask: Optional[np.ndarray] = None,
    max_iter: Optional[int] = 3,
    min_effective_density: Optional[float] = DEFAULT_MIN_EFFECTIVE_DENSITY,
    **kwargs: Any,
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
    width : float or sequence of float
        Smoothing width used by normalized convolution. A scalar applies
        the same width to every axis. A sequence follows array axis order
        and must match dimensionality for Cartesian or explicit N-D polar
        geometry. Legacy polar calls use ``(width_azimuth, width_range)``.
        All values must be > 0.
    coordinates : str, default='cartesian'
        Coordinate mode for smoothing: ``'cartesian'`` or ``'polar'``.
    fill_mask : np.ndarray, optional
        Boolean mask where True marks positions to fill/replace.
        If None, all NaN positions are filled.
    max_iter : int or None, default=3
        Maximum number of normalized-convolution iterations.
        If None, iterate until convergence or until a safety cap of 20
        iterations is reached.
    min_effective_density : float or None, default=0.35
        Minimum local valid-sample density (in [0, 1]) required to accept a
        per-iteration mean estimate. The density is the same
        ``smooth(mask)`` signal used internally by ``dct_count`` (i.e.
        ``dct_count / nominal_support``), so the threshold is dimensionless and works
        uniformly across Cartesian and polar coordinates. Cells whose local
        density is below the threshold are left as NaN for that iteration
        and may be filled in a later iteration as the valid mask grows.
        Set to ``None`` or a non-positive value to disable the gate and
        reproduce the legacy un-gated behavior.
    **kwargs
        Additional keyword arguments passed to ``dct_mean``. ``kernel_type``
        accepts a string or one kernel per array axis. Polar options include
        ``axis_types``, ``axis_coords``, ``boundary``, ``width_units``,
        ``angle_units``, and ``filter_axes``. All per-axis sequences follow
        array order; legacy polar calls use an (azimuth, range) kernel pair.
        See ``smooth_cartesian`` and ``smooth_polar`` for kernels and units.
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
      at least one finite value exists in the same filtering domain.
      Explicit polar propagation searches only active axes, trying range
      first when active, and respects every active periodic boundary. Each
      inactive-axis slice is isolated; an unsupported slice remains NaN.
      With no active axes, no values propagate between cells. Legacy polar
      propagation tries range first, then azimuth, wrapping only periodic azimuth.
      During a periodic axis pass, equal-distance neighbors prefer the preceding cell.
      Repeated active ``time`` or ``linear`` roles skip individual axis passes
      and use simultaneous multidimensional nearest completion, after range-first
      and unique-role passes. Exact global ties average every equidistant nearest
      seed; globally completed targets do not become seeds in that same pass.
    - The ``min_effective_density`` gate is also passed to the internal
      ``dct_mean`` call so the per-iteration mean statistic is consistently
      gated by the same criterion.
    """
    data_array = np.asarray(data)
    geometry = _resolve_smoothing_geometry(data_array.shape, width, coordinates, **kwargs)

    if max_iter is not None and max_iter < 1:
        raise ValueError(f"max_iter must be >= 1 or None, got {max_iter}")

    gate_enabled = min_effective_density is not None and float(min_effective_density) > 0
    gate_threshold = float(min_effective_density) if gate_enabled else None

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
    mean_kwargs["_stable_fallback"] = False
    if gate_enabled:
        mean_kwargs["min_effective_density"] = gate_threshold

    # Raw density uses exactly the same backend options as the normalized mean.
    smooth_kwargs = _smoothing_kwargs(mean_kwargs)
    smooth_kwargs.pop("min_effective_density", None)

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
            restore_input_nan=False,
            **mean_kwargs,
        )
        if gate_enabled:
            density = _smooth_with_coordinates(
                valid_current.astype(out_dtype), width, coordinates, **smooth_kwargs
            )
            density = np.clip(density, 0.0, 1.0)
            density_ok = density > gate_threshold
        else:
            density_ok = np.ones(out.shape, dtype=bool)
        newly_filled = candidates & np.isfinite(estimate) & density_ok
        if not np.any(newly_filled):
            break

        out[newly_filled] = estimate[newly_filled]

    if use_target_fallback:
        fallback = target & np.isnan(out) & np.isfinite(original)
        out[fallback] = original[fallback]

    unresolved = target & np.isnan(out)
    if np.any(unresolved):
        if geometry is not None and not geometry.legacy:
            out = _fill_nan_nearest_in_domains(
                out,
                unresolved,
                active_axes=tuple(geometry.active_axes),
                boundaries=tuple(geometry.boundaries),
                range_axis=geometry.range_axis,
                axis_types=tuple(geometry.axis_types),
            )
        else:
            fill_axis = 1 if coordinates == "polar" and out.ndim >= 2 else -1
            periodic_axis = (
                0 if coordinates == "polar" and kwargs.get("az_boundary") == "periodic" else None
            )
            out = _fill_nan_nearest(
                out, unresolved, primary_axis=fill_axis, periodic_axis=periodic_axis
            )

    return out


def dct_variance(
    data: np.ndarray,
    width: WidthLike,
    coordinates: str = "cartesian",
    mask: Optional[np.ndarray] = None,
    restore_input_nan: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """
    Compute robust local variance.

    Var = E[X^2] - (E[X])^2
    Both expectations are computed using Normalized Convolution.

    Parameters
    ----------
    data : np.ndarray
        Input data.
    width : float or sequence of float
        Smoothing width. A scalar applies the same width to every axis.
        A sequence follows array axis order and must match dimensionality
        for Cartesian or explicit N-D polar geometry. Legacy polar calls
        use ``(width_azimuth, width_range)``.
    coordinates : str, default='cartesian'
        'cartesian' or 'polar'.
    mask : np.ndarray, optional
        Boolean validity mask with the same shape as ``data``. If None,
        inferred from finite input values.
    restore_input_nan : bool, default=True
        If True, output is set to NaN where input support is invalid.
    **kwargs
        Additional arguments passed to ``dct_mean`` for both moments.
        ``kernel_type`` accepts a string or one kernel per array axis.
        Polar geometry options (``axis_types``, ``axis_coords``, ``boundary``,
        ``width_units``, ``angle_units``, ``filter_axes``) reach both moments
        and their independent completion fallbacks unchanged. Legacy polar
        calls use an (azimuth, range) kernel pair. See ``smooth_polar`` for units.

    Returns
    -------
    variance : np.ndarray
        Local variance (floating-point array, including a zero-dimensional
        array for scalar input).
    """
    data_array = np.asarray(data)
    _normalize_width_for_coordinates(
        width, coordinates, data_array.ndim, kwargs.get("axis_types")
    )
    out_dtype = np.result_type(data_array.dtype, np.float64)
    data_float = data_array.astype(out_dtype, copy=False)

    if mask is None:
        mask = np.isfinite(data_float)

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data_float.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data shape {data_float.shape}"
        )
    valid_data = mask & np.isfinite(data_float)

    # E[X]
    mean = dct_mean(
        data_float,
        width,
        coordinates,
        mask,
        restore_input_nan=False,
        **kwargs,
    )

    # E[X^2]
    data_sq = data_float**2
    mean_sq = dct_mean(
        data_sq,
        width,
        coordinates,
        mask,
        restore_input_nan=False,
        **kwargs,
    )

    # Var = E[X^2] - E[X]^2
    # Use maximum(0) to avoid negative variance due to numerical precision
    variance = np.asarray(np.maximum(mean_sq - mean**2, 0.0))
    if restore_input_nan:
        variance[~valid_data] = np.nan

    return variance


def dct_std(
    data: np.ndarray,
    width: WidthLike,
    coordinates: str = "cartesian",
    mask: Optional[np.ndarray] = None,
    restore_input_nan: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """
    Compute robust local standard deviation.

    Parameters
    ----------
    data : np.ndarray
        Input data (can contain NaNs).
    width : float or sequence of float
        Smoothing width. A scalar applies the same width to every axis.
        A sequence follows array axis order and must match dimensionality
        for Cartesian or explicit N-D polar geometry. Legacy polar calls
        use ``(width_azimuth, width_range)``.
    coordinates : str, default='cartesian'
        'cartesian' or 'polar'.
    mask : np.ndarray, optional
        Boolean validity mask with the same shape as ``data``. If None,
        inferred from finite input values.
    restore_input_nan : bool, default=True
        If True, output is set to NaN where input support is invalid.
    **kwargs
        Additional arguments passed to ``dct_variance``, including all polar
        geometry, units, boundary, and filtering-axis options. ``kernel_type``
        accepts a string or one kernel per array axis; legacy polar calls use
        an (azimuth, range) pair. See ``smooth_polar`` for kernels and units.

    Returns
    -------
    std : np.ndarray
        Local standard deviation (square root of the local variance), retaining
        a zero-dimensional array for scalar input.
    """
    data_array = np.asarray(data)
    _normalize_width_for_coordinates(
        width, coordinates, data_array.ndim, kwargs.get("axis_types")
    )

    var = dct_variance(
        data,
        width,
        coordinates,
        mask,
        restore_input_nan=restore_input_nan,
        **kwargs,
    )
    return np.asarray(np.sqrt(var))

"""
Geometry-aware polar smoothing with a backward-compatible 2-D interface.

Explicit N-D geometry adapts angular smoothing to actual range and elevation
coordinates. Without explicit axis roles, the original azimuth-by-range,
range-index-based filtering convention is retained.
"""

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.fft

from ._geometry import (
    AxisCoordinates,
    AxisStrings,
    GridWidth,
    _azimuth_spacing_radians,
    _resolve_polar_geometry,
)
from ._kernels import KernelLike, normalize_kernel_types
from ._widths import WidthLike, normalize_widths
from .core import get_dct_transfer_function


def compute_polar_transfer_functions(
    shape: Tuple[int, int],
    az_res_deg: float,
    width_pixels: WidthLike,
    kernel_type: KernelLike = "gaussian",
    az_boundary: str = "reflective",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer functions for polar smoothing.

    The azimuth width adapts with the index radius ``r = j + 1``. Its nominal
    physical-width interpretation is limited to that legacy convention; this
    helper does not accept actual range offsets or elevation coordinates.

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


def _polar_axis_transfer_function(
    shape: Tuple[int, ...],
    axis: int,
    width: GridWidth,
    kernel_type: str,
    boundary: str,
) -> np.ndarray:
    """Build a compact response for one explicit-geometry filtering pass.

    Parameters
    ----------
    shape : tuple of int
        Data dimensions.
    axis : int
        Filtering axis; widths must not vary along this dimension.
    width : float or ndarray
        Resolved grid widths, broadcastable to the data.
    kernel_type : str
        Validated kernel name.
    boundary : str
        ``reflective`` for DCT-II or ``periodic`` for RFFT.

    Returns
    -------
    ndarray
        Response varying only along frequency and geometry-dependent axes.
        Its DC component is exactly one at every geometry location.
    """
    n_axis = shape[axis]
    periodic = boundary == "periodic"
    n_freq = n_axis // 2 + 1 if periodic else n_axis
    frequency_shape = [1] * len(shape)
    frequency_shape[axis] = n_freq
    k = np.arange(n_freq).reshape(frequency_shape)
    theta = np.pi * k / (n_axis if periodic else 2 * n_axis)
    widths = np.asarray(width, dtype=float)
    if kernel_type == "gaussian":
        with np.errstate(over="ignore", under="ignore"):
            response = np.exp(-0.5 * (2 * theta * (widths / np.sqrt(12))) ** 2)
    else:
        if kernel_type == "boxcar_discrete":
            widths = np.maximum(1.0, np.rint(widths))
            widths = widths + (np.remainder(widths, 2) == 0)
        # All sampled phases repeat after 2*N (periodic) or 4*N (DCT)
        # width units. Reducing the numerator avoids overflow and preserves
        # multi-wrap windows without a loop over discrete boxcar neighbors.
        phase_width = np.remainder(widths, (2 if periodic else 4) * n_axis)
        response = np.sinc(phase_width * (theta / np.pi)) / np.sinc(theta / np.pi)
        response = response * (phase_width / widths)
    dc = [slice(None)] * len(shape)
    dc[axis] = 0
    response[tuple(dc)] = 1.0
    return response


def smooth_polar(
    data: np.ndarray,
    width_pixels: WidthLike,
    az_res_deg: float = 1.0,
    az_boundary: str = "reflective",
    range_boundary: str = "reflective",
    kernel_type: KernelLike = "gaussian",
    *,
    axis_types: Optional[Sequence[str]] = None,
    axis_coords: Optional[AxisCoordinates] = None,
    boundary: Optional[AxisStrings] = None,
    width_units: AxisStrings = "physical",
    angle_units: str = "degrees",
    filter_axes: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Apply legacy 2-D polar smoothing or explicit geometry-aware N-D filtering.

    Parameters
    ----------
    data : np.ndarray
        Input array. Without ``axis_types``, shape must be (n_azimuth, n_range).
    width_pixels : float or sequence of float
        Positive scalar or widths in array axis order. Legacy pairs are
        ``(width_azimuth, width_range)`` and retain index-based scaling; see
        ``compute_polar_transfer_functions``. Explicit geometry interprets
        widths according to ``width_units``.
    az_res_deg : float
        Legacy azimuth resolution in degrees, as a finite, positive real
        scalar. Explicit geometry uses ``axis_coords`` instead.
    az_boundary : str
        'reflective' (default) or 'periodic'.
    range_boundary : str
        Legacy mode supports only 'reflective'. Explicit geometry also
        supports 'periodic', unless overridden by ``boundary``.
    kernel_type : str or sequence of str, default='gaussian'
        'boxcar', 'boxcar_discrete', or 'gaussian'. A string applies to every
        axis. A sequence selects one kernel per array axis.
    axis_types : sequence of str or None, optional
        Flat sequence of one role per axis: ``azimuth``, ``elevation``,
        ``range``, ``time``, ``linear`` (alias ``cartesian``), or ``batch``.
        Azimuth, elevation, and range may each occur at most once. None
        preserves the legacy 2-D convention exactly.
    axis_coords : sequence or None, optional
        One finite real coordinate vector or None per axis. Supplied vectors
        must match axis length, be strictly increasing, and have uniform
        spacing (relative tolerance 1e-6). Physical active nonsingleton axes
        require coordinates, as do range/elevation dependencies of physical
        angular axes. Actual coordinate values, not gate indices, determine
        angular conversion. Selected nonsingleton physical angular filters
        require strictly positive range coordinates; range-only filtering
        may start at zero.
    boundary : str or sequence of str or None, optional
        ``reflective`` (DCT-II) or ``periodic`` (RFFT), in array order. A
        string broadcasts. None uses ``az_boundary`` for azimuth,
        ``range_boundary`` for range, and reflective for other axes.
    width_units : str or sequence of str, default='physical'
        ``grid`` uses literal index widths. ``physical`` divides linear,
        time, and range widths by coordinate spacing. Angular widths become
        ``L / (R*cos(elevation)*daz)`` for azimuth and ``L / (R*del)`` for
        elevation, with angular steps in radians. Without elevation, cos=1.
        Angular physical widths L are distances in the range coordinate's
        units, not degrees or radians.
        Singleton axes are identity filters with effective width one.
    angle_units : str, default='degrees'
        Global angular coordinate units: ``degrees`` or ``radians``.
        Elevation must lie within [-90, 90] degrees or [-pi/2, pi/2] radians.
    filter_axes : sequence of int or None, optional
        Exact axes to filter, with negative indices accepted. None filters all
        nonbatch axes; an empty sequence is identity. Duplicate or batch
        entries are invalid. Unfiltered geometry axes retain their roles.

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data with the same shape as input.

    Raises
    ------
    ValueError
        If geometry or filter metadata are invalid, physical angular range
        coordinates are nonpositive, azimuth geometry is singular near a
        pole, or physical periodic azimuth does not cover one full circle
        without a duplicate endpoint. New geometry options require
        ``axis_types``; nondefault options are never silently ignored.
    NotImplementedError
        If the legacy range boundary is not 'reflective'.

    Notes
    -----
    Filtering order is azimuth, elevation, range, time, then generic linear
    axes, regardless of array placement. Axes with the same independent role
    retain array order. Reversing dependent filters generally changes the
    result. Batch axes are never filtered. The operator does not normalize
    masked moments; use ``dct_mean`` for normalized convolution with gaps or
    ``dct_smooth`` for automatic prefill. All three kernels support either
    boundary on each explicit axis. Widths are not capped at the axis length.
    Origin/pole checks concern selected nonsingleton physical angular filters:
    both angular conversions require positive ranges; only physical azimuth
    conversion rejects near-pole elevation where cos(elevation) <= 1e-12.
    Grid, unfiltered, and singleton angular axes skip their own conversion,
    but their coordinates are still required if another selected filter
    depends on them.

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
    geometry = _resolve_polar_geometry(
        data.shape,
        width_pixels,
        kernel_type=kernel_type,
        az_res_deg=az_res_deg,
        az_boundary=az_boundary,
        range_boundary=range_boundary,
        axis_types=axis_types,
        axis_coords=axis_coords,
        boundary=boundary,
        width_units=width_units,
        angle_units=angle_units,
        filter_axes=filter_axes,
    )
    if not geometry.legacy:
        result = np.array(data, dtype=np.result_type(data.dtype, np.float64), copy=True)
        for axis in geometry.active_axes:
            if data.shape[axis] == 1:
                continue
            response = _polar_axis_transfer_function(
                data.shape, axis, geometry.widths_in_grid[axis],
                geometry.kernels[axis], geometry.boundaries[axis],
            )
            if geometry.boundaries[axis] == "periodic":
                spectrum = scipy.fft.rfft(result, axis=axis, norm="ortho")
                result = scipy.fft.irfft(
                    spectrum * response, n=data.shape[axis], axis=axis, norm="ortho"
                )
            else:
                spectrum = scipy.fft.dct(result, axis=axis, type=2, norm="ortho")
                result = scipy.fft.idct(
                    spectrum * response, axis=axis, type=2, norm="ortho"
                )
        return result

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

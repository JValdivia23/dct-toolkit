"""Internal, compact geometry resolution for legacy and N-D polar filters."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._kernels import KernelLike, normalize_kernel_types
from ._widths import WidthLike, normalize_widths


GridWidth = Union[float, np.ndarray]
AxisStrings = Union[str, Sequence[str], np.ndarray]
AxisCoordinates = Sequence[Optional[Union[Sequence[float], np.ndarray]]]
_AXIS_ROLES = ("azimuth", "elevation", "range", "time", "linear", "cartesian", "batch")
_FILTER_ORDER = ("azimuth", "elevation", "range", "time", "linear")
_GRID_RTOL = 1e-6


@dataclass(frozen=True)
class _PolarGeometry:
    """Resolved polar geometry shared by smoothing and statistics.

    Attributes
    ----------
    legacy : bool
        Whether the original two-dimensional, range-index convention applies.
    axis_types : tuple of str
        Roles in array order, with ``cartesian`` canonicalized to ``linear``.
    kernels, boundaries : tuple of str
        Kernel names and boundary conditions in array order.
    active_axes : tuple of int
        Selected axes in azimuth, elevation, range, time, then linear order.
    batch_axes : tuple of int
        All unfiltered axes in array order, including explicitly excluded axes.
    range_axis : int or None
        Array position of the range role, even when that axis is unfiltered.
    widths_in_grid : tuple of float or ndarray
        Effective index widths, constant along their own axis. Arrays have
        singleton dimensions outside their geometry dependencies. In explicit
        mode, singleton and unfiltered axes have the identity width of one.
    nominal_support : float or ndarray
        Lazily computed product of the requested, converted widths over active
        axes. Discrete boxcar rounding is deliberately not applied. Explicit
        support is checked for representability only when requested for counts.
    """

    legacy: bool
    axis_types: Tuple[str, ...]
    kernels: Tuple[str, ...]
    boundaries: Tuple[str, ...]
    active_axes: Tuple[int, ...]
    batch_axes: Tuple[int, ...]
    range_axis: Optional[int]
    widths_in_grid: Tuple[GridWidth, ...]

    @property
    def nominal_support(self) -> GridWidth:
        """Compute the nominal count factor without constraining smoothing.

        Returns
        -------
        float or ndarray
            Product of effective widths over active axes, with compact arrays
            read-only. Singleton and inactive explicit axes contribute one.

        Raises
        ------
        ValueError
            If the explicit ``dct_count`` factor cannot be represented as a
            finite, positive float64 value. Legacy products retain their
            original numerical behavior, including overflow and underflow.
        """
        if self.legacy:
            support = self.widths_in_grid[1] * self.widths_in_grid[0]
        else:
            # Accumulate binary exponents separately so a representable final
            # product is not lost to intermediate overflow or underflow.
            mantissa = 1.0
            exponent = 0
            for axis in self.active_axes:
                fraction, shift = np.frexp(self.widths_in_grid[axis])
                mantissa, carry = np.frexp(mantissa * fraction)
                exponent = exponent + shift + carry
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                support = np.ldexp(mantissa, exponent)
            if not np.all(np.isfinite(support)) or np.any(np.asarray(support) <= 0):
                raise ValueError(
                    "Nominal support for dct_count cannot be represented as a finite, "
                    "positive float64 value"
                )
        if isinstance(support, np.ndarray):
            support.setflags(write=False)
            return support
        return float(support)


def _azimuth_spacing_radians(az_res_deg: float) -> float:
    """Validate the legacy positive scalar spacing and convert it to radians.

    Parameters
    ----------
    az_res_deg : float
        Finite, positive, real scalar spacing in degrees.

    Returns
    -------
    float
        Positive spacing in radians.
    """
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


def _normalize_axis_strings(
    values: AxisStrings,
    ndim: int,
    name: str,
    allowed: Tuple[str, ...],
    broadcast: bool = True,
) -> Tuple[str, ...]:
    """Validate flat per-axis strings, optionally broadcasting one string.

    Parameters
    ----------
    values : str or sequence of str
        User specification in array order.
    ndim : int
        Required number of entries.
    name : str
        Parameter name used in error messages.
    allowed : tuple of str
        Accepted values.
    broadcast : bool, default=True
        Whether a single string can specify all axes.

    Returns
    -------
    tuple of str
        Validated entries in array order.
    """
    if isinstance(values, str):
        if not broadcast:
            raise ValueError(f"{name} must be a flat length-{ndim} sequence of strings")
        if values not in allowed:
            raise ValueError(f"Unknown {name}: {values!r}; expected one of {allowed}")
        entries = (values,) * ndim
    else:
        try:
            array = np.asarray(values, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a flat length-{ndim} sequence") from exc
        if array.ndim != 1 or array.size != ndim:
            raise ValueError(f"{name} must be a flat length-{ndim} sequence")
        entries = tuple(array.tolist())
    for axis, entry in enumerate(entries):
        if not isinstance(entry, str) or entry not in allowed:
            raise ValueError(f"Invalid {name} {entry!r} at axis {axis}; expected one of {allowed}")
    return entries


def _normalize_filter_axes(
    filter_axes: Optional[Sequence[int]], axis_types: Tuple[str, ...]
) -> Tuple[int, ...]:
    """Select nonbatch axes and arrange them in geometry filtering order.

    Parameters
    ----------
    filter_axes : sequence of int or None
        Exact selected axes, allowing negative indices.
    axis_types : tuple of str
        Canonical axis roles.

    Returns
    -------
    tuple of int
        Validated axes in role order, not caller or array order.
    """
    ndim = len(axis_types)
    if filter_axes is None:
        selected = {axis for axis, role in enumerate(axis_types) if role != "batch"}
    else:
        try:
            entries = np.asarray(filter_axes, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError("filter_axes must be a flat sequence of integer axes") from exc
        if entries.ndim != 1:
            raise ValueError("filter_axes must be a flat sequence of integer axes")
        selected = set()
        for entry in entries:
            if isinstance(entry, (bool, np.bool_)) or not isinstance(entry, (int, np.integer)):
                raise ValueError("filter_axes entries must be integers")
            axis = int(entry)
            if axis < 0:
                axis += ndim
            if axis < 0 or axis >= ndim:
                raise ValueError(f"filter_axes axis {entry} is out of bounds for {ndim}D data")
            if axis in selected:
                raise ValueError(f"filter_axes contains duplicate axis {axis}")
            if axis_types[axis] == "batch":
                raise ValueError(f"filter_axes cannot include batch axis {axis}")
            selected.add(axis)
    return tuple(
        axis
        for role in _FILTER_ORDER
        for axis, axis_role in enumerate(axis_types)
        if axis in selected and axis_role == role
    )


def _uniform_coordinate_spacing(values: np.ndarray, axis: int) -> Optional[float]:
    """Find a uniform step before coordinate conversion can erase precision.

    Parameters
    ----------
    values : ndarray
        Validated finite, real one-dimensional coordinates in their input dtype.
    axis : int
        Array axis used in validation messages.

    Returns
    -------
    float or None
        Positive, representable coordinate spacing, or None for a singleton.
        Integer differences are formed exactly before conversion to float64.
        Floating differences use at least float64, retaining wider input types.
    """
    if values.size < 2:
        return None
    if np.any(values[1:] <= values[:-1]):
        raise ValueError(f"axis_coords at axis {axis} must be strictly increasing")
    if values.dtype.kind in "iu":
        # After checking order in the original dtype, unsigned subtraction
        # gives exact positive differences up to 2**64 - 1. Signed values map
        # modulo 2**64, so even a step across both int64 extremes is safe.
        steps = np.diff(values.astype(np.uint64, copy=False))
    else:
        # Half/single-precision subtraction can overflow or round distinct
        # steps together. Promote before differencing, without downcasting
        # higher-precision floating coordinate types.
        working_values = values.astype(np.result_type(values.dtype, np.float64), copy=False)
        with np.errstate(over="ignore", invalid="ignore"):
            steps = np.diff(working_values)
    spacing = float(steps[0])
    if not np.all(np.isfinite(steps)) or not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"axis_coords at axis {axis} must have finite, positive spacing")
    if not np.allclose(steps, spacing, rtol=_GRID_RTOL, atol=0.0):
        raise ValueError(f"axis_coords at axis {axis} must be uniformly spaced")
    return spacing


def _normalize_axis_coords(
    axis_coords: Optional[AxisCoordinates],
    shape: Tuple[int, ...],
    axis_types: Tuple[str, ...],
    angle_units: str,
) -> Tuple[Tuple[Optional[np.ndarray], ...], Tuple[Optional[float], ...]]:
    """Validate supplied real uniform coordinates and derive their spacings.

    Parameters
    ----------
    axis_coords : sequence or None
        A coordinate vector or None for every array axis.
    shape : tuple of int
        Array dimensions.
    axis_types : tuple of str
        Canonical roles in array order.
    angle_units : str
        ``degrees`` or ``radians`` for angular coordinates.

    Returns
    -------
    coords : tuple
        Validated coordinates, with angular coordinates converted to radians.
    spacings : tuple
        Positive coordinate steps, or None for absent or singleton vectors.
        Integer steps are derived exactly before float/unit conversion;
        floating differences use at least float64 precision. Large integer
        epochs need not be distinguishable as absolute float64 values.
    """
    if axis_coords is None:
        return (None,) * len(shape), (None,) * len(shape)
    try:
        if isinstance(axis_coords, (str, bytes)):
            raise TypeError
        entries = tuple(axis_coords)
    except TypeError as exc:
        raise ValueError(f"axis_coords must be a length-{len(shape)} sequence") from exc
    if len(entries) != len(shape):
        raise ValueError(f"axis_coords must be a length-{len(shape)} sequence")

    coords = []
    spacings = []
    for axis, (entry, size, role) in enumerate(zip(entries, shape, axis_types)):
        if entry is None:
            coords.append(None)
            spacings.append(None)
            continue
        message = f"axis_coords at axis {axis} must be a finite real 1-D vector of length {size}"
        try:
            values = np.asarray(entry)
            if values.ndim != 1 or values.size != size or values.dtype.kind not in "iuf":
                raise ValueError(message)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(message)
        spacing = _uniform_coordinate_spacing(values, axis)
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(float, copy=False)
        if not np.all(np.isfinite(values)):
            raise ValueError(message)
        if role in ("azimuth", "elevation") and angle_units == "degrees":
            values = np.deg2rad(values)
            if spacing is not None:
                spacing = float(np.deg2rad(spacing))
                if spacing == 0:
                    raise ValueError(
                        f"axis_coords at axis {axis} spacing is too small to represent in radians"
                    )
        if role == "elevation" and np.any(np.abs(values) > np.pi / 2):
            raise ValueError(
                "elevation axis_coords must be within [-90, 90] degrees or [-pi/2, pi/2]"
            )
        coords.append(values)
        spacings.append(spacing)
    return tuple(coords), tuple(spacings)


def _coordinate_view(values: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    """Reshape one coordinate vector into a compact N-D broadcast view.

    Parameters
    ----------
    values : ndarray
        One-dimensional coordinate vector.
    axis : int
        Array axis represented by the vector.
    ndim : int
        Number of dimensions in the data.

    Returns
    -------
    ndarray
        View with singleton dimensions on every other axis.
    """
    shape = [1] * ndim
    shape[axis] = values.size
    return values.reshape(shape)


def _freeze_grid_width(width: GridWidth) -> GridWidth:
    """Validate a resolved width and make any compact array read-only.

    Parameters
    ----------
    width : float or ndarray
        Computed index width for an individual axis.

    Returns
    -------
    float or ndarray
        Finite, positive value that cannot mutate the resolved geometry.
    """
    if not np.all(np.isfinite(width)) or np.any(np.asarray(width) <= 0):
        raise ValueError("Effective grid widths must be finite and positive")
    if isinstance(width, np.ndarray):
        width.setflags(write=False)
        return width
    return float(width)


def _resolve_polar_geometry(
    shape: Tuple[int, ...],
    width_pixels: WidthLike,
    *,
    kernel_type: KernelLike = "gaussian",
    az_res_deg: float = 1.0,
    az_boundary: str = "reflective",
    range_boundary: str = "reflective",
    axis_types: Optional[Sequence[str]] = None,
    axis_coords: Optional[AxisCoordinates] = None,
    boundary: Optional[AxisStrings] = None,
    width_units: AxisStrings = "physical",
    angle_units: str = "degrees",
    filter_axes: Optional[Sequence[int]] = None,
) -> _PolarGeometry:
    """Resolve widths, coordinate roles, filter order, and nominal support.

    Parameters
    ----------
    shape : tuple of int
        Data shape. Explicit geometry rejects zero-length axes.
    width_pixels : float or sequence of float
        Positive widths in array order; a scalar broadcasts to all axes.
    kernel_type : str or sequence of str, default='gaussian'
        Supported kernel names in array order.
    az_res_deg : float, default=1.0
        Legacy azimuth step; explicit geometry uses coordinates instead.
    az_boundary, range_boundary : str, default='reflective'
        Legacy boundaries and defaults for corresponding explicit roles.
    axis_types : sequence of str or None, optional
        Per-axis roles. None retains the original 2-D index-based convention.
    axis_coords : sequence or None, optional
        Uniform real coordinate vectors or None in array order.
    boundary : str or sequence of str or None, optional
        Reflective or periodic boundaries, overriding role-based defaults.
    width_units : str or sequence of str, default='physical'
        ``physical`` converts through geometry; ``grid`` uses literal widths.
    angle_units : str, default='degrees'
        Global angular coordinate units: ``degrees`` or ``radians``.
    filter_axes : sequence of int or None, optional
        Selected nonbatch axes; None selects all nonbatch axes.

    Returns
    -------
    _PolarGeometry
        Frozen geometry with compact broadcastable widths and nominal support.

    Raises
    ------
    ValueError
        If metadata are invalid, required coordinates are missing, physical
        angular geometry is singular, or coordinates are not uniform.
    NotImplementedError
        If a legacy call requests a nonreflective range boundary.
    """
    ndim = len(shape)
    if axis_types is None:
        if (
            axis_coords is not None
            or boundary is not None
            or filter_axes is not None
            or not isinstance(width_units, str)
            or width_units != "physical"
            or not isinstance(angle_units, str)
            or angle_units != "degrees"
        ):
            raise ValueError("axis_types is required when using explicit polar geometry options")
        if ndim != 2:
            raise ValueError(f"Data must be 2D (az, range), got {ndim}D")
        spacing = _azimuth_spacing_radians(az_res_deg)
        widths = normalize_widths(width_pixels, 2, name="width_pixels")
        kernels = normalize_kernel_types(kernel_type, 2)
        if az_boundary not in ("reflective", "periodic"):
            raise ValueError(f"Unknown azimuth boundary: {az_boundary}")
        if range_boundary != "reflective":
            raise NotImplementedError("Only 'reflective' range boundary supported")
        azimuth_width = float(widths[0]) / (np.arange(1, shape[1] + 1) * spacing)
        azimuth_width = azimuth_width.reshape(1, -1)
        # Preserve the old singleton convention and range-index origin.
        azimuth_width.setflags(write=False)
        return _PolarGeometry(
            legacy=True,
            axis_types=("azimuth", "range"),
            kernels=kernels,
            boundaries=(az_boundary, range_boundary),
            active_axes=(0, 1),
            batch_axes=(),
            range_axis=1,
            widths_in_grid=(azimuth_width, float(widths[1])),
        )

    if any(size <= 0 for size in shape):
        raise ValueError("Explicit polar geometry does not support zero-length axes")
    roles = _normalize_axis_strings(axis_types, ndim, "axis_types", _AXIS_ROLES, broadcast=False)
    roles = tuple("linear" if role == "cartesian" else role for role in roles)
    for role in ("azimuth", "elevation", "range"):
        if roles.count(role) > 1:
            raise ValueError(f"axis_types may contain at most one {role!r} axis")
    if not isinstance(angle_units, str) or angle_units not in ("degrees", "radians"):
        raise ValueError("angle_units must be 'degrees' or 'radians'")
    units = _normalize_axis_strings(width_units, ndim, "width_units", ("physical", "grid"))
    widths = normalize_widths(width_pixels, ndim, name="width_pixels")
    kernels = normalize_kernel_types(kernel_type, ndim)
    # Scalar metadata must still be validated for an explicit zero-dimensional array.
    if ndim == 0:
        if np.asarray(width_pixels).ndim == 0:
            normalize_widths(width_pixels, 1, name="width_pixels")
        if isinstance(kernel_type, str):
            normalize_kernel_types(kernel_type, 1)
    if boundary is None:
        boundary = tuple(
            az_boundary if role == "azimuth" else range_boundary if role == "range"
            else "reflective" for role in roles
        )
    boundaries = _normalize_axis_strings(boundary, ndim, "boundary", ("reflective", "periodic"))
    active_axes = _normalize_filter_axes(filter_axes, roles)
    batch_axes = tuple(axis for axis in range(ndim) if axis not in active_axes)
    coords, spacings = _normalize_axis_coords(axis_coords, shape, roles, angle_units)
    range_axis = roles.index("range") if "range" in roles else None
    elevation_axis = roles.index("elevation") if "elevation" in roles else None

    grid_widths = [1.0] * ndim
    for axis in active_axes:
        if shape[axis] == 1:
            continue
        width = float(widths[axis])
        if units[axis] == "grid":
            grid_widths[axis] = width
            continue
        role = roles[axis]
        spacing = spacings[axis]
        if spacing is None:
            raise ValueError(f"Physical smoothing requires axis_coords for {role} axis {axis}")
        if role in ("time", "linear", "range"):
            grid_widths[axis] = width / spacing
            continue

        if range_axis is None or coords[range_axis] is None:
            raise ValueError("Physical angular smoothing requires a range axis with axis_coords")
        ranges = coords[range_axis]
        if np.any(ranges <= 0):
            raise ValueError(
                "Physical angular smoothing requires strictly positive range coordinates"
            )
        if elevation_axis is not None and coords[elevation_axis] is None:
            raise ValueError(
                "Physical angular smoothing requires elevation axis_coords when present"
            )
        if role == "azimuth" and boundaries[axis] == "periodic":
            if not np.isclose(shape[axis] * spacing, 2 * np.pi, rtol=_GRID_RTOL, atol=0.0):
                raise ValueError(
                    "Physical periodic azimuth coordinates must cover one full circle "
                    "without a duplicate endpoint"
                )
        range_view = _coordinate_view(ranges, range_axis, ndim)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            effective_width = width / range_view / spacing
        if role == "azimuth" and elevation_axis is not None:
            cosine = np.cos(coords[elevation_axis])
            if np.any(cosine <= 1e-12):
                raise ValueError("Physical azimuth smoothing is singular near elevation poles")
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                effective_width = effective_width / _coordinate_view(cosine, elevation_axis, ndim)
        grid_widths[axis] = effective_width

    frozen_widths = tuple(_freeze_grid_width(width) for width in grid_widths)
    return _PolarGeometry(
        legacy=False,
        axis_types=roles,
        kernels=kernels,
        boundaries=boundaries,
        active_axes=active_axes,
        batch_axes=batch_axes,
        range_axis=range_axis,
        widths_in_grid=frozen_widths,
    )

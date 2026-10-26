"""Independent numerical contracts for geometry-aware N-D polar smoothing.

The reference deliberately uses spatial indexing for discrete boxcars and a
NumPy cosine matrix/FFT for the other kernels, never package transfer helpers.
The statistics tests reuse this complete ordered operator, not normalized 1-D
passes or implementation geometry objects.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

from dct_toolkit.polar import smooth_polar


KERNELS = ("gaussian", "boxcar", "boxcar_discrete")
_ORDER = {"azimuth": 0, "elevation": 1, "range": 2, "time": 3, "linear": 4}


def _axis_values(value: Any, ndim: int) -> Tuple[Any, ...]:
    """Broadcast scalar specifications in array order for valid reference cases."""
    if isinstance(value, str) or np.ndim(value) == 0:
        return (value,) * ndim
    return tuple(value)


def _active_axes(
    axis_types: Sequence[str], filter_axes: Optional[Sequence[int]]
) -> Tuple[int, ...]:
    """Return the selected array axes without implementing validation."""
    if filter_axes is None:
        return tuple(i for i, kind in enumerate(axis_types) if kind != "batch")
    return tuple(int(i) % len(axis_types) for i in filter_axes)


def _effective_widths(
    shape: Tuple[int, ...],
    width_pixels: Any,
    *,
    axis_types: Sequence[str],
    axis_coords: Optional[Sequence[Optional[np.ndarray]]] = None,
    width_units: Any = "physical",
    angle_units: str = "degrees",
    filter_axes: Optional[Sequence[int]] = None,
    **_unused: Any,
) -> Tuple[np.ndarray, ...]:
    """Evaluate the public physical metric directly at every grid point."""
    ndim = len(shape)
    widths = _axis_values(width_pixels, ndim)
    units = _axis_values(width_units, ndim)
    coords = (None,) * ndim if axis_coords is None else tuple(axis_coords)
    active = _active_axes(axis_types, filter_axes)
    radians = np.pi / 180.0 if angle_units == "degrees" else 1.0
    fields = []
    for axis, kind in enumerate(axis_types):
        if axis not in active or shape[axis] == 1:
            fields.append(np.ones(shape))
            continue
        denominator: Any = 1.0
        if units[axis] == "physical":
            spacing = 1.0 if coords[axis] is None else coords[axis][1] - coords[axis][0]
            denominator = spacing
            if kind in ("azimuth", "elevation"):
                range_axis = tuple(axis_types).index("range")
                range_shape = [1] * ndim
                range_shape[range_axis] = shape[range_axis]
                ranges = np.asarray(coords[range_axis]).reshape(range_shape)
                denominator = ranges * spacing * radians
                if kind == "azimuth" and "elevation" in axis_types:
                    elevation_axis = tuple(axis_types).index("elevation")
                    elevation_shape = [1] * ndim
                    elevation_shape[elevation_axis] = shape[elevation_axis]
                    elevations = np.asarray(coords[elevation_axis]).reshape(elevation_shape)
                    denominator = denominator * np.cos(elevations * radians)
        fields.append(np.full(shape, float(widths[axis])) / denominator)
    return tuple(fields)


def _filter_line(values: np.ndarray, width: float, kernel: str, boundary: str) -> np.ndarray:
    """Apply a stand-alone spatial boxcar or explicit Fourier/cosine expansion."""
    n = len(values)
    if n == 1:
        return values.astype(float, copy=True)
    if kernel == "boxcar_discrete":
        size = max(1, int(np.round(width)))
        size += int(size % 2 == 0)
        offsets = np.arange(-(size // 2), size // 2 + 1)
        indices = np.arange(n)[:, None] + offsets
        if boundary == "periodic":
            indices %= n
        else:
            indices %= 2 * n
            indices = np.where(indices < n, indices, 2 * n - 1 - indices)
        return np.mean(values[indices], axis=1)

    if boundary == "periodic":
        omega = 2 * np.pi * np.fft.rfftfreq(n)
    else:
        omega = np.pi * np.arange(n) / n
    if kernel == "gaussian":
        response = np.exp(-(omega * width) ** 2 / 24.0)
    else:
        response = np.sinc(width * omega / (2 * np.pi)) / np.sinc(omega / (2 * np.pi))
    if boundary == "periodic":
        return np.fft.irfft(np.fft.rfft(values) * response, n=n)
    basis = np.sqrt(2.0 / n) * np.cos(omega[:, None] * (np.arange(n) + 0.5))
    basis[0] /= np.sqrt(2.0)
    return basis.T @ (response * (basis @ values))


def _polar_reference(
    data: np.ndarray,
    width_pixels: Any,
    *,
    axis_types: Sequence[str],
    axis_coords: Optional[Sequence[Optional[np.ndarray]]] = None,
    boundary: Any = None,
    width_units: Any = "physical",
    angle_units: str = "degrees",
    filter_axes: Optional[Sequence[int]] = None,
    kernel_type: Any = "gaussian",
    az_boundary: str = "reflective",
    range_boundary: str = "reflective",
    order: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Filter independent columns in canonical geometry order, then return array order."""
    kinds = tuple("linear" if kind == "cartesian" else kind for kind in axis_types)
    active = _active_axes(kinds, filter_axes)
    if order is None:
        order = sorted(active, key=lambda axis: (_ORDER[kinds[axis]], axis))
    if boundary is None:
        boundaries = tuple(
            az_boundary if kind == "azimuth" else
            range_boundary if kind == "range" else "reflective"
            for kind in kinds
        )
    else:
        boundaries = _axis_values(boundary, data.ndim)
    kernels = _axis_values(kernel_type, data.ndim)
    fields = _effective_widths(
        data.shape, width_pixels, axis_types=kinds, axis_coords=axis_coords,
        width_units=width_units, angle_units=angle_units, filter_axes=filter_axes,
    )
    result = data.astype(float, copy=True)
    for axis in order:
        moved = np.moveaxis(result, axis, 0)
        widths = np.moveaxis(fields[axis], axis, 0)[0]
        output = np.empty_like(moved)
        for index in np.ndindex(moved.shape[1:]):
            column = (slice(None),) + index
            output[column] = _filter_line(
                moved[column], float(widths[index]), kernels[axis], boundaries[axis]
            )
        result = np.moveaxis(output, 0, axis)
    return result


def _nominal_volume(shape: Tuple[int, ...], width_pixels: Any, **geometry: Any) -> np.ndarray:
    """Multiply requested effective widths, not rounded taps or statistical ESS."""
    return np.prod(_effective_widths(shape, width_pixels, **geometry), axis=0)


def _volume_geometry() -> Dict[str, Any]:
    """Return a small physical (elevation, azimuth, range) test geometry."""
    return {
        "axis_types": ("elevation", "azimuth", "range"),
        "axis_coords": (
            np.array([-30.0, -10.0, 10.0, 30.0]),
            -160.0 + np.arange(9) * 40.0,
            2.25 + np.arange(6) * 0.75,
        ),
    }


@pytest.mark.parametrize(
    "kernels", [
        ("boxcar_discrete",) * 3,
        ("gaussian", "boxcar", "boxcar_discrete"),
        ("boxcar", "gaussian", "boxcar"),
    ],
)
@pytest.mark.parametrize(
    "boundaries", [
        ("reflective", "periodic", "reflective"),
        ("periodic", "reflective", "periodic"),
    ],
)
def test_physical_3d_volume_matches_independent_columns(
    kernels: Tuple[str, ...], boundaries: Tuple[str, ...]
) -> None:
    """Both angles use local physical geometry, with mixed kernels and boundaries."""
    data = np.random.default_rng(701).standard_normal((4, 9, 6))
    original = data.copy()
    geometry = dict(_volume_geometry(), kernel_type=kernels, boundary=boundaries)
    coords_before = tuple(coord.copy() for coord in geometry["axis_coords"])
    widths = (2.2, 7.7, 1.6)
    expected = _polar_reference(data, widths, **geometry)
    result = smooth_polar(data, widths, **geometry)
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(data, original)
    for actual, before in zip(geometry["axis_coords"], coords_before):
        np.testing.assert_array_equal(actual, before)


def test_real_first_gate_offset_is_not_replaced_by_range_indices() -> None:
    """Changing the actual range origin changes angular filtering, not range spacing."""
    data = np.random.default_rng(703).standard_normal((4, 9, 6))
    geometry = dict(_volume_geometry(), kernel_type="boxcar_discrete", boundary="reflective")
    widths = (3.7, 7.7, 0.75)
    expected = _polar_reference(data, widths, **geometry)
    indexed = dict(geometry)
    indexed["axis_coords"] = (*geometry["axis_coords"][:2], np.arange(1.0, 7.0))
    wrong = _polar_reference(data, widths, **indexed)
    assert np.max(np.abs(expected - wrong)) > 0.1
    np.testing.assert_allclose(smooth_polar(data, widths, **geometry), expected, atol=2e-12)


@pytest.mark.parametrize("permutation", [(0, 1, 2, 3), (3, 2, 0, 1), (1, 0, 3, 2)])
def test_4d_mixed_specifications_follow_array_order_under_permutation(
    permutation: Tuple[int, ...],
) -> None:
    """Storage layout cannot change the ordered nonstationary physical operator."""
    data = np.random.default_rng(709).standard_normal((3, 4, 9, 6))
    volume = _volume_geometry()
    geometry = {
        "axis_types": ("time", *volume["axis_types"]),
        "axis_coords": (np.array([4.0, 4.5, 5.0]), *volume["axis_coords"]),
        "kernel_type": ("boxcar_discrete", "gaussian", "boxcar_discrete", "boxcar"),
        "boundary": ("periodic", "reflective", "periodic", "reflective"),
        "width_units": ("physical", "grid", "physical", "physical"),
    }
    widths = (1.2, 2.6, 7.7, 1.6)
    expected = _polar_reference(data, widths, **geometry)
    permuted = {key: tuple(values[i] for i in permutation) for key, values in geometry.items()}
    result = smooth_polar(
        data.transpose(permutation), tuple(widths[i] for i in permutation), **permuted
    )
    np.testing.assert_allclose(result, expected.transpose(permutation), rtol=2e-12, atol=2e-12)


def test_canonical_azimuth_elevation_range_order_is_not_array_order() -> None:
    """Azimuth/elevation/range do not commute when their physical metrics vary."""
    data = np.random.default_rng(719).standard_normal((4, 9, 6))
    geometry = dict(_volume_geometry(), kernel_type="boxcar_discrete")
    widths = (3.1, 7.7, 2.1)
    expected = _polar_reference(data, widths, **geometry)
    storage_order = _polar_reference(data, widths, order=(0, 1, 2), **geometry)
    reverse_order = _polar_reference(data, widths, order=(2, 0, 1), **geometry)
    assert np.max(np.abs(expected - storage_order)) > 0.01
    assert np.max(np.abs(expected - reverse_order)) > 0.05
    np.testing.assert_allclose(smooth_polar(data, widths, **geometry), expected, atol=2e-12)


def test_actual_5d_batch_time_volume_matches_independent_domains() -> None:
    """An extra batch dimension is neither smoothed nor included in geometry width."""
    data = np.random.default_rng(727).integers(-5, 6, size=(2, 3, 4, 9, 6))
    data[1] += 100
    original = data.copy()
    volume = _volume_geometry()
    geometry = {
        "axis_types": ("batch", "time", *volume["axis_types"]),
        "axis_coords": (None, np.array([0.0, 2.0, 4.0]), *volume["axis_coords"]),
        "kernel_type": ("boxcar", "gaussian", "boxcar_discrete", "boxcar", "gaussian"),
        "boundary": ("periodic", "periodic", "reflective", "periodic", "reflective"),
    }
    widths = (999.0, 4.6, 2.2, 7.7, 1.6)
    result = smooth_polar(data, widths, **geometry)
    expected = _polar_reference(data, widths, **geometry)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, expected, rtol=3e-12, atol=3e-12)
    assert np.max(result[0]) < 10 and np.min(result[1]) > 90
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("kernel", KERNELS)
def test_nd_scalar_and_string_broadcasting_preserves_constants(kernel: str) -> None:
    """Scalar widths/units/boundaries/kernels broadcast across arbitrary array order."""
    data = np.full((2, 4, 9, 6), 7, dtype=np.int64)
    volume = _volume_geometry()
    geometry = {
        "axis_types": ("batch", *volume["axis_types"]),
        "axis_coords": (None, *volume["axis_coords"]),
        "boundary": "periodic", "kernel_type": kernel,
    }
    result = smooth_polar(data, 3.2, **geometry)
    repeated = smooth_polar(
        data, (3.2,) * 4, **dict(
            geometry, boundary=("periodic",) * 4, kernel_type=(kernel,) * 4,
            width_units=("physical",) * 4,
        )
    )
    np.testing.assert_allclose(result, 7.0, atol=2e-12)
    np.testing.assert_allclose(result, repeated, rtol=2e-12, atol=2e-12)


def test_degrees_and_radians_give_the_same_physical_volume() -> None:
    """Both angular coordinate spacings and elevation cosines use angle_units."""
    data = np.random.default_rng(733).standard_normal((4, 9, 6))
    degrees = dict(_volume_geometry(), boundary=("reflective", "periodic", "reflective"))
    radians = dict(degrees, angle_units="radians")
    radians["axis_coords"] = (
        np.deg2rad(degrees["axis_coords"][0]),
        np.deg2rad(degrees["axis_coords"][1]),
        degrees["axis_coords"][2],
    )
    expected = _polar_reference(data, (2.2, 7.7, 1.6), **degrees)
    for geometry in (degrees, radians):
        np.testing.assert_allclose(
            smooth_polar(data, (2.2, 7.7, 1.6), **geometry), expected, atol=2e-12
        )


def test_physical_linear_time_range_widths_use_coordinate_spacing() -> None:
    """Physical nonangular widths convert to grid widths independently in array order."""
    data = np.random.default_rng(739).standard_normal((4, 8, 5, 6))
    geometry = {
        "axis_types": ("time", "azimuth", "range", "linear"),
        "axis_coords": (
            10.0 + 2.0 * np.arange(4), np.arange(8) * 45.0,
            100.0 + 3.0 * np.arange(5), 0.5 * np.arange(6),
        ),
        "boundary": ("periodic", "periodic", "reflective", "periodic"),
        "kernel_type": ("gaussian", "boxcar_discrete", "boxcar", "gaussian"),
    }
    grid_widths = (1.5, 3.2, 2.3, 4.6)
    physical_widths = (3.0, 3.2, 6.9, 2.3)
    mixed = dict(geometry, width_units=("physical", "grid", "physical", "physical"))
    expected = _polar_reference(data, grid_widths, **dict(geometry, width_units="grid"))
    np.testing.assert_allclose(smooth_polar(data, physical_widths, **mixed), expected, atol=2e-12)
    np.testing.assert_allclose(
        smooth_polar(data, grid_widths, **dict(geometry, width_units="grid")), expected,
        atol=2e-12,
    )


def test_singleton_coordinates_still_control_other_axes_physical_metrics() -> None:
    """A singleton range/elevation is identity but its actual radius/cosine still matter."""
    data = np.arange(8, dtype=float).reshape(1, 8, 1)
    geometry = {
        "axis_types": ("elevation", "azimuth", "range"),
        "axis_coords": (np.array([60.0]), np.arange(8) * 45.0, np.array([4.0])),
        "boundary": ("reflective", "periodic", "reflective"),
    }
    widths = (100.0, 3.2 * 4.0 * 0.5 * np.pi / 4.0, 900.0)
    expected = _polar_reference(data, widths, **geometry)
    np.testing.assert_allclose(smooth_polar(data, widths, **geometry), expected, atol=2e-12)
    grid_expected = _polar_reference(
        data, (1.0, 3.2, 1.0), **dict(geometry, width_units="grid")
    )
    np.testing.assert_allclose(expected, grid_expected, atol=2e-12)


def test_grid_angular_axes_need_no_physical_dependencies() -> None:
    """Grid-unit angular filtering does not invent a radius or require a full sweep."""
    data = np.random.default_rng(743).standard_normal((3, 7, 4))
    geometry = {
        "axis_types": ("elevation", "azimuth", "time"), "width_units": "grid",
        "boundary": ("periodic", "periodic", "reflective"),
        "kernel_type": ("boxcar_discrete", "gaussian", "boxcar"),
    }
    expected = _polar_reference(data, (2.3, 3.4, 1.7), **geometry)
    np.testing.assert_allclose(
        smooth_polar(data, (2.3, 3.4, 1.7), **geometry), expected, atol=2e-12
    )


@pytest.mark.parametrize("selection", [(0, 2), (-2, 0), ()])
def test_filter_axes_selects_only_requested_nonbatch_axes(selection: Tuple[int, ...]) -> None:
    """Unselected angular/batch slices remain independent, including negative indices."""
    data = np.random.default_rng(751).standard_normal((3, 7, 5, 2))
    data[:, :, :, 1] += 100.0
    geometry = {
        "axis_types": ("time", "azimuth", "range", "batch"),
        "axis_coords": (np.arange(3) * 2.0, None, 10.0 + np.arange(5) * 0.5, None),
        "filter_axes": selection,
        "boundary": ("periodic", "periodic", "reflective", "periodic"),
        "kernel_type": ("boxcar_discrete", "gaussian", "boxcar_discrete", "boxcar"),
    }
    widths = (5.1, 99.0, 1.4, 500.0)
    expected = _polar_reference(data, widths, **geometry)
    np.testing.assert_allclose(smooth_polar(data, widths, **geometry), expected, atol=2e-12)
    if not selection:
        np.testing.assert_array_equal(expected, data)


def test_cartesian_alias_and_repeated_linear_axes_have_array_order_widths() -> None:
    """Independent linear roles may repeat, and cartesian is an exact linear alias."""
    data = np.random.default_rng(757).standard_normal((4, 5, 3))
    geometry = {
        "axis_types": ("cartesian", "linear", "time"),
        "axis_coords": (np.arange(4) * 0.5, np.arange(5) * 2.0, np.arange(3) * 3.0),
        "kernel_type": ("boxcar", "gaussian", "boxcar_discrete"),
        "boundary": ("periodic", "reflective", "periodic"),
    }
    expected = _polar_reference(data, (1.3, 5.1, 6.9), **geometry)
    for roles in (geometry["axis_types"], ("linear", "linear", "time")):
        result = smooth_polar(data, (1.3, 5.1, 6.9), **dict(geometry, axis_types=roles))
        np.testing.assert_allclose(result, expected, atol=2e-12)


@pytest.mark.parametrize("boundary", ["reflective", "periodic"])
@pytest.mark.parametrize("width", [0.25, 1.0, 2.0, 2.5, 4.6, 25.0])
@pytest.mark.parametrize("kernel", KERNELS)
def test_nd_subgrid_and_multiwrap_kernels_match_independent_reference(
    boundary: str, width: float, kernel: str
) -> None:
    """Small and oversized widths use the requested kernel in either boundary mode."""
    data = np.array([0.0, 1.0, 4.0, -2.0, 7.0])
    geometry = {
        "axis_types": ("time",), "width_units": "grid",
        "boundary": boundary, "kernel_type": kernel,
    }
    expected = _polar_reference(data, width, **geometry)
    np.testing.assert_allclose(smooth_polar(data, width, **geometry), expected, atol=2e-12)


@pytest.mark.parametrize("az_boundary", ["reflective", "periodic"])
@pytest.mark.parametrize("kernel", KERNELS)
def test_explicit_2d_matches_reference_and_physically_consistent_legacy_call(
    az_boundary: str, kernel: str
) -> None:
    """Legacy index-radius geometry is one special case of explicit physical axes."""
    data = np.random.default_rng(761).standard_normal((12, 7))
    geometry = {
        "axis_types": ("azimuth", "range"),
        "axis_coords": (np.arange(12) * 30.0, np.arange(1.0, 8.0)),
        "az_boundary": az_boundary, "kernel_type": (kernel, "gaussian"),
    }
    widths = (4.3, 2.7)
    expected = _polar_reference(data, widths, **geometry)
    explicit = smooth_polar(data, widths, **geometry)
    legacy = smooth_polar(
        data, widths, 30.0, az_boundary, "reflective", (kernel, "gaussian")
    )
    np.testing.assert_allclose(explicit, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(legacy, expected, rtol=2e-12, atol=2e-12)


def test_boundary_none_uses_role_defaults_not_array_positions() -> None:
    """Old boundary names only affect their matching explicit roles."""
    data = np.random.default_rng(769).standard_normal((4, 9, 6))
    geometry = dict(
        _volume_geometry(), az_boundary="periodic", range_boundary="periodic",
        kernel_type=("boxcar_discrete", "gaussian", "boxcar_discrete"),
    )
    expected = _polar_reference(data, (2.2, 7.7, 1.6), **geometry)
    default = smooth_polar(data, (2.2, 7.7, 1.6), boundary=None, **geometry)
    explicit = smooth_polar(
        data, (2.2, 7.7, 1.6),
        **dict(geometry, boundary=("reflective", "periodic", "periodic")),
    )
    np.testing.assert_allclose(default, expected, atol=2e-12)
    np.testing.assert_allclose(explicit, expected, atol=2e-12)


@pytest.mark.parametrize(
    "roles", [
        ("azimuth", "range"), ("azimuth", "azimuth", "range"),
        ("elevation", "elevation", "range"), ("range", "time", "range"),
        ("latitude", "azimuth", "range"), ("elevation", 2, "range"),
    ],
)
def test_invalid_axis_roles_raise_meaningful_value_error(roles: Tuple[Any, ...]) -> None:
    """Role length, known names, and unique spherical axes are part of validation."""
    with pytest.raises(ValueError, match="(?i)axis|axes|type|role|dimension|length"):
        smooth_polar(np.ones((3, 4, 5)), 2.0, axis_types=roles, width_units="grid")


@pytest.mark.parametrize(
    "override, message", [
        ({"width_pixels": (2.0, 3.0)}, "width|length|dimension"),
        ({"width_pixels": (2.0, 0.0, 3.0)}, "width|positive"),
        ({"width_pixels": (2.0, np.nan, 3.0)}, "width|finite"),
        ({"kernel_type": ("gaussian", "boxcar")}, "kernel|length"),
        ({"kernel_type": ("gaussian", "unknown", "boxcar")}, "kernel|axis"),
        ({"boundary": ("periodic", "reflective")}, "bound|length"),
        ({"boundary": ("reflective", "open", "periodic")}, "bound|axis"),
        ({"width_units": ("grid", "physical")}, "unit|length"),
        ({"width_units": ("grid", "metres", "grid")}, "unit|axis"),
        ({"angle_units": "turns"}, "angle|unit"),
        ({"filter_axes": (0, -3)}, "filter|axis|axes|duplicate"),
        ({"filter_axes": (3,)}, "filter|axis|axes|bound"),
        ({"filter_axes": (1.5,)}, "filter|axis|axes|integer"),
    ],
)
def test_invalid_nd_axis_specifications_are_rejected(
    override: Dict[str, Any], message: str
) -> None:
    """Specifications are validated in array order before any transform is attempted."""
    kwargs = dict(
        axis_types=("elevation", "azimuth", "range"), width_units="grid", width_pixels=2.0
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match="(?i)" + message):
        smooth_polar(np.ones((3, 4, 5)), **kwargs)


@pytest.mark.parametrize("selection", [(0,), (-3,), (0, 2)])
def test_filter_axes_cannot_select_batch(selection: Tuple[int, ...]) -> None:
    """Batch roles cannot be enabled through either positive or negative axis indices."""
    with pytest.raises(ValueError, match="(?i)batch|filter"):
        smooth_polar(
            np.ones((2, 4, 5)), 2.0, axis_types=("batch", "time", "range"),
            width_units="grid", filter_axes=selection,
        )


@pytest.mark.parametrize(
    "coordinate", [
        np.arange(3.0), np.arange(4.0).reshape(2, 2),
        np.array([0.0, 1.0, np.nan, 3.0]), np.array([0.0, 1.0, np.inf, 3.0]),
        np.array([0.0, 1.0, 1.0, 3.0]), np.array([3.0, 2.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 2.1, 3.0]), np.arange(4.0).astype(complex) + 1j,
        np.array(["0", "1", "2", "3"]),
    ],
)
@pytest.mark.parametrize("inactive", [False, True])
def test_every_supplied_coordinate_is_a_finite_uniform_real_vector(
    coordinate: np.ndarray, inactive: bool
) -> None:
    """Inactive coordinates are still validated rather than silently ignored."""
    with pytest.raises(ValueError, match="(?i)coord|axis|vector|finite|real|spacing|uniform"):
        smooth_polar(
            np.ones((4, 5)), 2.0, axis_types=("time", "range"),
            axis_coords=(coordinate, np.arange(5.0)), width_units="grid",
            filter_axes=(1,) if inactive else None,
        )


def test_axis_coords_length_must_match_ndim() -> None:
    """A missing array-order coordinate slot is not interpreted as a different role."""
    with pytest.raises(ValueError, match="(?i)coord|length|dimension"):
        smooth_polar(
            np.ones((3, 4, 5)), 2.0, axis_types=("time", "azimuth", "range"),
            axis_coords=(np.arange(3.0), np.arange(4.0)), width_units="grid",
        )


def test_batch_coordinates_are_validated_even_though_batch_is_never_filtered() -> None:
    """A batch coordinate vector follows the same supplied-coordinate contract."""
    with pytest.raises(ValueError, match="(?i)coord|uniform|spacing|axis"):
        smooth_polar(
            np.ones((4, 5)), 2.0, axis_types=("batch", "time"),
            axis_coords=(np.array([0.0, 1.0, 2.2, 3.0]), np.arange(5.0)),
            width_units="grid",
        )


@pytest.mark.parametrize("missing_axis", [0, 1, 2])
def test_physical_angles_require_real_range_and_angular_coordinates(missing_axis: int) -> None:
    """Physical angular filtering never substitutes gate indices or az_res_deg."""
    geometry = _volume_geometry()
    coords = list(geometry["axis_coords"])
    coords[missing_axis] = None
    with pytest.raises(ValueError, match="(?i)coord|range|azimuth|elevation|physical"):
        smooth_polar(np.ones((4, 9, 6)), 2.0, **dict(geometry, axis_coords=tuple(coords)))


@pytest.mark.parametrize("first_range", [0.0, -1.0])
def test_physical_angular_width_rejects_origin_and_nonpositive_range(first_range: float) -> None:
    """Positive range is required where angular physical widths use the local metric."""
    geometry = _volume_geometry()
    coords = (*geometry["axis_coords"][:2], first_range + np.arange(6))
    with pytest.raises(ValueError, match="(?i)range|positive|origin"):
        smooth_polar(np.ones((4, 9, 6)), 2.0, **dict(geometry, axis_coords=coords))


@pytest.mark.parametrize("elevation", [90.0, -90.0, 89.99999999999])
def test_azimuth_physical_metric_rejects_poles(elevation: float) -> None:
    """Near-zero elevation cosine is rejected even on a singleton dependency axis."""
    with pytest.raises(ValueError, match="(?i)pole|cos|elevation|singular"):
        smooth_polar(
            np.ones((1, 8, 2)), 2.0, axis_types=("elevation", "azimuth", "range"),
            axis_coords=(np.array([elevation]), np.arange(8) * 45.0, np.array([2.0, 3.0])),
            filter_axes=(1,), boundary="periodic",
        )


@pytest.mark.parametrize("angle_units, elevation", [("degrees", 91.0), ("radians", 1.6)])
def test_supplied_elevation_coordinates_stay_within_legal_limits(
    angle_units: str, elevation: float
) -> None:
    """Elevation bounds are geometry validation, including grid-unit filtering."""
    with pytest.raises(ValueError, match="(?i)elevation|limit|angle|range"):
        smooth_polar(
            np.ones((1, 4)), 2.0, axis_types=("elevation", "time"),
            axis_coords=(np.array([elevation]), None), width_units="grid", angle_units=angle_units,
        )


@pytest.mark.parametrize("angles", [np.linspace(0.0, 360.0, 8), np.arange(8) * 20.0])
def test_physical_periodic_azimuth_requires_complete_sweep_without_endpoint(
    angles: np.ndarray,
) -> None:
    """A periodic physical metric requires n*daz=2*pi rather than an endpoint duplicate."""
    with pytest.raises(ValueError, match="(?i)azimuth|period|sweep|360|2.*pi|full"):
        smooth_polar(
            np.ones((8, 3)), 2.0, axis_types=("azimuth", "range"),
            axis_coords=(angles, np.arange(1.0, 4.0)), boundary="periodic",
        )


@pytest.mark.parametrize("shape", [(0, 4, 5), (3, 0, 5), (3, 4, 0)])
def test_explicit_zero_length_axes_are_invalid(shape: Tuple[int, ...]) -> None:
    """Empty explicit domains fail clearly, including an unfiltered batch dimension."""
    with pytest.raises(ValueError, match="(?i)empty|zero|length|non.?empty|size"):
        smooth_polar(
            np.ones(shape), 2.0, axis_types=("batch", "time", "range"), width_units="grid"
        )

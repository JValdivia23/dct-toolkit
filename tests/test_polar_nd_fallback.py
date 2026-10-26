"""Regression contracts for layout-independent fallback and scalar array outputs."""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest
import scipy.spatial
from scipy.ndimage import uniform_filter1d

from dct_toolkit import dct_count, dct_mean, dct_prefill, dct_smooth, dct_std, dct_variance


_REPEATED_ROLES = [("linear", "linear"), ("time", "time"), ("cartesian", "linear")]
_MOMENTS = (dct_mean, dct_variance, dct_std)


def _nearest_reference(data: np.ndarray, boundaries: Tuple[str, ...]) -> np.ndarray:
    """Complete simultaneously from all exactly nearest original seeds in index space."""
    result = data.copy()
    seeds = np.column_stack(np.nonzero(np.isfinite(data)))
    if seeds.shape[0] == 0:
        return result
    for point in np.column_stack(np.nonzero(np.isnan(data))):
        deltas = np.abs(seeds - point)
        for axis, boundary in enumerate(boundaries):
            if boundary == "periodic":
                deltas[:, axis] = np.minimum(deltas[:, axis], data.shape[axis] - deltas[:, axis])
        distances = np.sum(deltas**2, axis=1)
        tied_seeds = seeds[distances == distances.min()]
        result[tuple(point)] = np.mean(data[tuple(tied_seeds.T)])
    return result


def _spatial_smooth(data: np.ndarray, boundaries: Tuple[str, ...]) -> np.ndarray:
    """Apply independent spatial size-three windows, not package transfer functions."""
    result = data.copy()
    for axis, boundary in enumerate(boundaries):
        mode = "wrap" if boundary == "periodic" else "reflect"
        result = uniform_filter1d(result, size=3, axis=axis, mode=mode)
    return result


def _reproducer() -> np.ndarray:
    """Return seeds whose nearest choice previously depended on the first array axis."""
    data = np.full((3, 4), np.nan)
    data[0, 3], data[1, 0] = 1.0, 2.0
    return data


@pytest.mark.parametrize("roles", _REPEATED_ROLES)
def test_repeated_roles_prefill_uses_multidimensional_nearest_not_axis_order(
    roles: Tuple[str, ...],
) -> None:
    """The review's non-tie target has an unambiguously closer seed after transpose."""
    data = _reproducer()
    original = data.copy()
    kwargs = dict(
        coordinates="polar", axis_types=roles, width_units="grid", kernel_type="boxcar_discrete"
    )
    expected = _nearest_reference(data, ("reflective", "reflective"))
    assert expected[0, 0] == 2.0  # Squared distances are 1 and 9, not a tie.
    result = dct_prefill(data, 1.0, **kwargs)
    transposed = dct_prefill(data.T, 1.0, **dict(kwargs, axis_types=roles[::-1])).T
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(transposed, expected)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("roles", _REPEATED_ROLES[:2])
@pytest.mark.parametrize("func", _MOMENTS)
def test_repeated_roles_complete_both_moments_independently_and_equivariantly(
    roles: Tuple[str, ...], func: Callable,
) -> None:
    """An exact nearest tie averages x and x**2 independently, giving positive variance."""
    data = _reproducer()
    boundaries = ("reflective", "reflective")
    first = _nearest_reference(data, boundaries)
    second = _nearest_reference(data**2, boundaries)
    variance = np.maximum(second - first**2, 0.0)
    assert first[2, 2] == 1.5 and variance[2, 2] == 0.25
    expected = first if func is dct_mean else variance
    kwargs = dict(
        coordinates="polar", axis_types=roles, width_units="grid",
        kernel_type="boxcar_discrete", restore_input_nan=False,
    )
    result = func(data, 1.0, **kwargs)
    transposed = func(data.T, 1.0, **kwargs).T
    if func is dct_std:
        result, transposed = result**2, transposed**2
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(transposed, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("roles", _REPEATED_ROLES[:2])
@pytest.mark.parametrize("boundaries", [("reflective", "reflective"), ("reflective", "periodic")])
def test_repeated_roles_wrapper_original_observations_are_layout_independent(
    roles: Tuple[str, ...], boundaries: Tuple[str, ...],
) -> None:
    """A gate of one forces residual nearest fill before nontrivial smoothing."""
    data = _reproducer()
    support = np.isfinite(data)
    expected = _spatial_smooth(_nearest_reference(data, boundaries), boundaries)
    expected[~support] = np.nan
    kwargs = dict(
        coordinates="polar", axis_types=roles, width_units="grid", boundary=boundaries,
        kernel_type="boxcar_discrete", min_effective_density=1.0,
    )
    result = dct_smooth(data, 3.0, **kwargs)
    transposed = dct_smooth(data.T, 3.0, **dict(kwargs, boundary=boundaries[::-1])).T
    np.testing.assert_allclose(result[support], expected[support], atol=2e-12)
    np.testing.assert_allclose(transposed[support], expected[support], atol=2e-12)
    np.testing.assert_array_equal(np.isnan(result), ~support)
    np.testing.assert_array_equal(np.isnan(transposed), ~support)


@pytest.mark.parametrize("roles", _REPEATED_ROLES[:2])
def test_global_nearest_averages_every_exactly_equidistant_seed(roles: Tuple[str, ...]) -> None:
    """The tied neighborhood can contain more than the two nearest-query results."""
    data = np.full((5, 5), np.nan)
    data[0, 2], data[2, 0], data[4, 2], data[2, 4] = 1.0, 3.0, 9.0, 15.0
    kwargs = dict(
        coordinates="polar", axis_types=roles, width_units="grid", kernel_type="boxcar_discrete"
    )
    expected = _nearest_reference(data, ("reflective", "reflective"))
    assert expected[2, 2] == 7.0
    np.testing.assert_array_equal(dct_prefill(data, 1.0, **kwargs), expected)
    np.testing.assert_array_equal(dct_prefill(data.T, 1.0, **kwargs).T, expected)


def test_unique_role_global_ties_average_seeds_without_changing_non_targets() -> None:
    """The explicit global tie rule also applies after unique-axis passes cannot reach a target."""
    data = np.full((5, 5), np.nan)
    data[0, 2], data[2, 0], data[4, 2], data[2, 4] = 1.0, 3.0, 9.0, 15.0
    target = np.zeros(data.shape, dtype=bool)
    target[1, 1] = True  # No donor shares a time or range coordinate.
    result = dct_prefill(
        data, 1.0, coordinates="polar", axis_types=("time", "range"),
        width_units="grid", kernel_type="boxcar_discrete", fill_mask=target,
    )
    assert result[1, 1] == 2.0
    np.testing.assert_array_equal(result[~target], data[~target])


def test_global_nearest_queries_radius_neighborhoods_only_for_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A two-neighbor tree query identifies the sole tie without expanding every target."""
    tree_type = scipy.spatial.cKDTree
    query_sizes, radius_points = [], []

    class ObservedTree:
        """Record neighborhood-query work while delegating actual geometry to SciPy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.tree = tree_type(*args, **kwargs)

        def query(self, *args: Any, k: int = 1, **kwargs: Any) -> Any:
            query_sizes.append(k)
            return self.tree.query(*args, k=k, **kwargs)

        def query_ball_point(self, point: np.ndarray, radius: float) -> Any:
            radius_points.append(tuple(point))
            return self.tree.query_ball_point(point, radius)

    monkeypatch.setattr(scipy.spatial, "cKDTree", ObservedTree)
    data = _reproducer()
    result = dct_prefill(
        data, 1.0, coordinates="polar", axis_types=("linear", "linear"),
        width_units="grid", kernel_type="boxcar_discrete",
    )
    np.testing.assert_array_equal(result, _nearest_reference(data, ("reflective",) * 2))
    assert query_sizes == [2]
    assert radius_points == [(2, 2)]


@pytest.mark.parametrize("roles", _REPEATED_ROLES[:2])
@pytest.mark.parametrize(
    "boundaries", [("periodic", "periodic"), ("reflective", "periodic"), ("periodic", "reflective")]
)
def test_repeated_roles_nearest_respects_mixed_toroidal_distances_and_seams(
    roles: Tuple[str, ...], boundaries: Tuple[str, ...],
) -> None:
    """All periodic active axes participate in both ties and unique nearest choices."""
    data = np.full((6, 8), np.nan)
    data[5, 7], data[1, 1], data[3, 4] = -2.0, 6.0, 99.0
    kwargs = dict(
        coordinates="polar", axis_types=roles, width_units="grid", boundary=boundaries,
        kernel_type="boxcar_discrete",
    )
    expected = _nearest_reference(data, boundaries)
    if boundaries == ("periodic", "periodic"):
        assert expected[0, 0] == 2.0
    np.testing.assert_array_equal(dct_prefill(data, 1.0, **kwargs), expected)
    transposed = dct_prefill(data.T, 1.0, **dict(kwargs, boundary=boundaries[::-1])).T
    np.testing.assert_array_equal(transposed, expected)
    for axis, boundary in enumerate(boundaries):
        if boundary == "periodic":
            shifted = dct_prefill(np.roll(data, 2, axis=axis), 1.0, **kwargs)
            np.testing.assert_array_equal(np.roll(shifted, -2, axis=axis), expected)


def test_repeated_roles_retain_range_first_priority() -> None:
    """Range completion still precedes simultaneous independent-role completion."""
    data = np.full((3, 4, 5), np.nan)
    data[0, 3, 0], data[1, 0, 4] = 1.0, 2.0
    range_completed = np.full(data.shape, np.nan)
    range_completed[0, 3, :], range_completed[1, 0, :] = 1.0, 2.0
    expected = np.stack(
        [_nearest_reference(range_completed[..., i], ("reflective", "reflective"))
         for i in range(data.shape[-1])], axis=-1,
    )
    assert expected[0, 0, 0] == 2.0
    assert _nearest_reference(data, ("reflective",) * 3)[0, 0, 0] == 1.0
    kwargs = dict(
        coordinates="polar", axis_types=("linear", "linear", "range"), width_units="grid",
        kernel_type="boxcar_discrete",
    )
    result = dct_prefill(data, 1.0, **kwargs)
    np.testing.assert_array_equal(result, expected)
    permutation = (1, 2, 0)
    transposed = dct_prefill(
        data.transpose(permutation), 1.0,
        **dict(kwargs, axis_types=tuple(kwargs["axis_types"][axis] for axis in permutation)),
    )
    np.testing.assert_array_equal(transposed, expected.transpose(permutation))


@pytest.mark.parametrize("func", [dct_prefill, *_MOMENTS, dct_smooth])
def test_repeated_roles_never_borrow_across_batch_or_inactive_role_axes(func: Callable) -> None:
    """Repeated-role completion cannot use an inactive time/batch axis as a bridge."""
    data = np.full((3, 3, 2, 4), np.nan)
    data[:, 0, 0, :] = _reproducer()
    data[2, 0, 1, 2] = 50.0
    data[:, 2, 0, :] = _reproducer() * 10.0
    original = data.copy()
    boundaries = ("reflective", "periodic")
    first, second = np.full(data.shape, np.nan), np.full(data.shape, np.nan)
    for batch in range(3):
        for time in range(2):
            domain = data[:, batch, time, :]
            first[:, batch, time, :] = _nearest_reference(domain, boundaries)
            second[:, batch, time, :] = _nearest_reference(domain**2, boundaries)
    expected = first if func in (dct_prefill, dct_mean) else np.maximum(second - first**2, 0.0)
    kwargs: Dict[str, Any] = dict(
        coordinates="polar", axis_types=("linear", "batch", "time", "linear"),
        width_units="grid", filter_axes=(0, 3), kernel_type="boxcar_discrete",
        boundary=("reflective", "periodic", "periodic", "periodic"),
    )
    extra = {"restore_input_nan": False} if func in _MOMENTS else {}
    width = 1.0
    if func is dct_smooth:
        width = 3.0
        extra = {"min_effective_density": 1.0}
        expected = first.copy()
        for batch in range(3):
            for time in range(2):
                expected[:, batch, time, :] = _spatial_smooth(first[:, batch, time, :], boundaries)
        expected[np.isnan(data)] = np.nan
    result = func(data, width, **kwargs, **extra)
    permutation = (3, 2, 0, 1)
    permuted = dict(kwargs)
    for key in ("axis_types", "boundary"):
        permuted[key] = tuple(kwargs[key][axis] for axis in permutation)
    permuted["filter_axes"] = tuple(permutation.index(axis) for axis in kwargs["filter_axes"])
    transposed = func(data.transpose(permutation), width, **permuted, **extra)
    if func is dct_std:
        result, transposed = result**2, transposed**2
    np.testing.assert_allclose(result, expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(transposed, expected.transpose(permutation), atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("coordinates", ["cartesian", "polar"])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("func", [dct_count, *_MOMENTS])
def test_scalar_statistics_return_zero_dimensional_arrays(
    coordinates: str, restore: bool, valid: bool, func: Callable,
) -> None:
    """Scalar arithmetic must not turn a writable zero-dimensional array into a scalar."""
    kwargs: Dict[str, Any] = {"coordinates": coordinates, "restore_input_nan": restore}
    if coordinates == "polar":
        kwargs["axis_types"] = ()
    if func is dct_count:
        values = np.array(valid)
        expected = 1.0 if valid else (np.nan if restore else 0.0)
    else:
        values = np.array(3.0 if valid else np.nan)
        expected = (3.0 if func is dct_mean else 0.0) if valid else np.nan
    result = func(values, 2.0, **kwargs)
    assert isinstance(result, np.ndarray) and result.shape == ()
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, np.array(expected), equal_nan=True)


@pytest.mark.parametrize("coordinates", ["cartesian", "polar"])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("func", _MOMENTS)
def test_scalar_explicit_validity_masks_preserve_zero_dimensional_outputs(
    coordinates: str, restore: bool, valid: bool, func: Callable,
) -> None:
    """An explicitly invalid scalar cannot borrow support, even when its value is finite."""
    kwargs: Dict[str, Any] = dict(
        coordinates=coordinates, restore_input_nan=restore, mask=np.array(valid),
    )
    if coordinates == "polar":
        kwargs["axis_types"] = ()
    result = func(np.array(3.0), 2.0, **kwargs)
    assert isinstance(result, np.ndarray) and result.shape == ()
    expected = (3.0 if func is dct_mean else 0.0) if valid else np.nan
    np.testing.assert_allclose(result, np.array(expected), equal_nan=True)


@pytest.mark.parametrize("coordinates", ["cartesian", "polar"])
@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("func", [dct_prefill, dct_smooth])
def test_scalar_prefill_and_wrapper_keep_shape_and_missingness(
    coordinates: str, valid: bool, func: Callable,
) -> None:
    """No-op scalar completion/smoothing retains the array and its original NaN state."""
    kwargs: Dict[str, Any] = {"coordinates": coordinates}
    if coordinates == "polar":
        kwargs["axis_types"] = ()
    data = np.array(3.0 if valid else np.nan)
    original = data.copy()
    result = func(data, 2.0, **kwargs)
    assert isinstance(result, np.ndarray) and result.shape == ()
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("coordinates", ["cartesian", "polar"])
def test_scalar_targeted_finite_prefill_keeps_unestimable_original(coordinates: str) -> None:
    """A finite scalar replacement target retains its original value without other support."""
    kwargs: Dict[str, Any] = {"coordinates": coordinates}
    if coordinates == "polar":
        kwargs["axis_types"] = ()
    result = dct_prefill(np.array(3.0), 2.0, fill_mask=np.array(True), **kwargs)
    assert isinstance(result, np.ndarray) and result.shape == ()
    np.testing.assert_array_equal(result, np.array(3.0))

"""Cached complete smoothing operators for batched statistical calculations."""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np
import scipy.fft

from ._geometry import _resolve_polar_geometry
from ._kernels import normalize_kernel_types
from ._widths import WidthLike, normalize_widths
from .core import get_dct_transfer_function
from .polar import _polar_axis_transfer_function


@dataclass
class _OperatorPlan:
    """Resolve geometry once and reuse responses across data/value-bin batches."""

    shape: Tuple[int, ...]
    active_axes: Tuple[int, ...]
    kernels: Tuple[str, ...]
    boundaries: Tuple[str, ...]
    responses: Tuple[np.ndarray, ...]
    metadata: Mapping[str, Any]
    cartesian: bool = False

    def require_positive(self) -> None:
        """Reject kernels without validated nonnegative statistical weights."""
        if any(self.kernels[a] not in ("gaussian", "boxcar_discrete")
               for a in self.active_axes):
            raise ValueError(
                "Nonnegative statistical weights require gaussian or boxcar_discrete "
                "on every filtered axis"
            )

    def apply(self, values: np.ndarray, adjoint: bool = False) -> np.ndarray:
        """Apply K or its adjoint, retaining arbitrary leading batch dimensions."""
        result = np.asarray(values, dtype=float)
        extra = result.ndim - len(self.shape)
        if extra < 0 or result.shape[extra:] != self.shape:
            raise ValueError("Operator input does not match the resolved data shape")
        if not self.active_axes:
            return result.copy()
        if self.cartesian:
            axes = tuple(a + extra for a in self.active_axes)
            spectrum = scipy.fft.dctn(result, axes=axes, type=2, norm="ortho")
            for response in self.responses:
                spectrum *= response.reshape((1,) * extra + response.shape)
            return scipy.fft.idctn(spectrum, axes=axes, type=2, norm="ortho")
        passes = list(zip(self.active_axes, self.responses))
        if adjoint:
            passes.reverse()
        for axis, response in passes:
            transformed_axis = axis + extra
            response = response.reshape((1,) * extra + response.shape)
            if self.boundaries[axis] == "periodic":
                spectrum = scipy.fft.rfft(result, axis=transformed_axis, norm="ortho")
                result = scipy.fft.irfft(
                    spectrum * response, n=self.shape[axis], axis=transformed_axis,
                    norm="ortho",
                )
            else:
                spectrum = scipy.fft.dct(result, axis=transformed_axis, type=2, norm="ortho")
                result = scipy.fft.idct(
                    spectrum * response, axis=transformed_axis, type=2, norm="ortho"
                )
        return result

    def diagonal(self) -> np.ndarray:
        """Compute the complete Kii, including all reflection/wrap self-visits."""
        diagonal = np.ones(self.shape)
        for axis, response in zip(self.active_axes, self.responses):
            n = self.shape[axis]
            if self.boundaries[axis] == "periodic":
                kernel = scipy.fft.irfft(response, n=n, axis=axis)
                term = np.take(kernel, [0], axis=axis)
            else:
                pad_shape = list(response.shape)
                pad_shape[axis] = 1
                spectrum = np.concatenate((response, np.zeros(pad_shape)), axis=axis)
                kernel = scipy.fft.irfft(spectrum, n=2 * n, axis=axis)
                term = np.take(kernel, [0], axis=axis) + np.take(
                    kernel, np.arange(1, 2 * n, 2), axis=axis
                )
            # Each axis is visited once. Returning to the same input therefore
            # requires the diagonal contribution at every ordered axis pass.
            diagonal *= term
        return diagonal


def _make_plan(
    shape: Tuple[int, ...], width: WidthLike, coordinates: str, **kwargs: Any
) -> _OperatorPlan:
    """Build reusable axis responses without adding statistical batch axes to geometry."""
    if any(n == 0 for n in shape):
        raise ValueError("Local statistics do not support zero-length axes")
    if coordinates == "cartesian":
        unknown = set(kwargs) - {"kernel_type"}
        if unknown:
            raise TypeError(f"Unexpected Cartesian options: {sorted(unknown)}")
        widths = normalize_widths(width, len(shape))
        kernels = normalize_kernel_types(kwargs.get("kernel_type", "gaussian"), len(shape))
        if not shape:
            normalize_widths(width, 1)
            if isinstance(kwargs.get("kernel_type", "gaussian"), str):
                normalize_kernel_types(kwargs.get("kernel_type", "gaussian"), 1)
        responses = []
        for axis, n in enumerate(shape):
            response_shape = [1] * len(shape)
            response_shape[axis] = n
            if kernels[axis] == "boxcar_discrete":
                # Use the same cosine-sum identity as explicit polar filtering:
                # work scales with axis length, not with an oversized window.
                response = _polar_axis_transfer_function(
                    shape, axis, float(widths[axis]), kernels[axis], "reflective"
                )
            else:
                response = get_dct_transfer_function(n, kernels[axis], widths[axis]).reshape(
                    response_shape
                )
            responses.append(response)
        active = tuple(range(len(shape)))
        boundaries = ("reflective",) * len(shape)
        metadata = {"coordinates": coordinates, "shape": shape, "legacy": False,
                    "axis_types": ("linear",) * len(shape), "axis_coords": (None,) * len(shape),
                    "filter_axes": active, "kernel_type": kernels, "boundary": boundaries,
                    "width_units": "grid", "widths_requested": tuple(widths),
                    "widths_in_grid": tuple(widths)}
        return _OperatorPlan(shape, active, kernels, boundaries, tuple(responses), metadata, True)
    if coordinates != "polar":
        raise ValueError(f"Unknown coordinates: {coordinates}")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        geometry = _resolve_polar_geometry(shape, width, **kwargs)
    if any(not np.all(np.isfinite(w)) or np.any(np.asarray(w) <= 0)
           for w in geometry.widths_in_grid):
        raise ValueError("Effective grid widths must be finite and positive")
    responses = tuple(_polar_axis_transfer_function(
        shape, axis, geometry.widths_in_grid[axis],
        geometry.kernels[axis], geometry.boundaries[axis],
    ) for axis in geometry.active_axes)
    coordinate_snapshots = []
    original_coords = kwargs.get("axis_coords")
    for values in (None,) * len(shape) if original_coords is None else original_coords:
        if values is None:
            coordinate_snapshots.append(None)
        else:
            values = np.array(values, copy=True)
            values.setflags(write=False)
            coordinate_snapshots.append(values)
    units = kwargs.get("width_units", "legacy" if geometry.legacy else "physical")
    metadata = {"coordinates": coordinates, "shape": shape, "legacy": geometry.legacy,
                "axis_types": geometry.axis_types, "axis_coords": tuple(coordinate_snapshots),
                "filter_axes": geometry.active_axes, "kernel_type": geometry.kernels,
                "boundary": geometry.boundaries,
                "width_units": units if isinstance(units, str) else tuple(units),
                "widths_requested": tuple(normalize_widths(width, len(shape))),
                "widths_in_grid": geometry.widths_in_grid,
                "angle_units": kwargs.get("angle_units", "degrees"),
                "az_res_deg": kwargs.get("az_res_deg", 1.0) if geometry.legacy else None}
    return _OperatorPlan(shape, geometry.active_axes, geometry.kernels, geometry.boundaries,
                         responses, metadata)

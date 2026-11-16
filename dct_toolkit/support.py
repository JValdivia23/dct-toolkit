"""Actual operator-weight support diagnostics, separate from nominal counts."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ._operator import _OperatorPlan, _make_plan
from ._widths import WidthLike


DEFAULT_WEIGHT_TOLERANCE = 64 * np.finfo(float).eps


@dataclass(frozen=True)
class SupportDiagnostics:
    """Original and center-excluded weight masses under the complete operator.

    Attributes
    ----------
    original_valid_weight, valid_weight : ndarray
        K(M), and K(M) minus Kii*Mi when exclusion is requested.
    available_weight : ndarray
        K(1), minus Kii when excluding the center. Not nominal window size.
    valid_fraction : ndarray
        Valid / available mass; NaN where no available mass exists.
    self_weight, excluded_weight : ndarray
        Full Kii and the actual subtracted valid-center contribution.
    supported : ndarray of bool
        Valid mass strictly exceeds max(min_weight, weight_tolerance), with
        available mass > weight_tolerance and the requested valid fraction,
        compared in mass units with the declared numerical tolerance.
    geometry : mapping
        Resolved roles, selected axes, kernels, boundaries, units and grid widths.
    min_weight, min_valid_fraction, weight_tolerance : float
        Recorded scientific support rules and separate numerical tolerance.
    """

    original_valid_weight: np.ndarray
    valid_weight: np.ndarray
    available_weight: np.ndarray
    valid_fraction: np.ndarray
    self_weight: np.ndarray
    excluded_weight: np.ndarray
    supported: np.ndarray
    geometry: Mapping[str, Any]
    min_weight: float
    min_valid_fraction: float
    weight_tolerance: float


def _support_from_plan(
    mask: np.ndarray, plan: _OperatorPlan, exclude_center: bool = False,
    min_weight: float = 1e-3, min_valid_fraction: float = 0.0,
    weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
) -> SupportDiagnostics:
    """Reuse a resolved plan for original and center-excluded mass diagnostics."""
    for name, value in (("min_weight", min_weight), ("weight_tolerance", weight_tolerance)):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative scalar")
    if not np.isscalar(min_valid_fraction) or not np.isfinite(min_valid_fraction) or not (
        0 <= min_valid_fraction <= 1
    ):
        raise ValueError("min_valid_fraction must lie in [0, 1]")
    plan.require_positive()
    valid = np.asarray(mask, dtype=bool)
    if np.all(valid):
        original = np.ones(plan.shape)  # K(1)=1 exactly; avoid an unnecessary transform.
    elif not np.any(valid):
        original = np.zeros(plan.shape)
    else:
        original = plan.apply(valid.astype(float))
    diagonal = plan.diagonal()
    if np.any(original < -weight_tolerance) or np.any(diagonal < -weight_tolerance):
        raise ArithmeticError("Operator produced negative weight beyond roundoff tolerance")
    original = np.clip(original, 0.0, 1.0)
    diagonal = np.clip(diagonal, 0.0, 1.0)
    excluded = diagonal * valid if exclude_center else np.zeros(plan.shape)
    available = np.maximum(1.0 - diagonal, 0.0) if exclude_center else np.ones(plan.shape)
    remaining = np.maximum(original - excluded, 0.0)
    fraction = np.full(plan.shape, np.nan)
    np.divide(remaining, available, out=fraction, where=available > weight_tolerance)
    fraction = np.clip(fraction, 0.0, 1.0)
    # Compare fraction thresholds in weight units so cancellation in a small
    # external available mass does not amplify roundoff into false missingness.
    fraction_pass = remaining >= min_valid_fraction * available - weight_tolerance
    supported = ((remaining > max(min_weight, weight_tolerance)) &
                 (available > weight_tolerance) & fraction_pass)
    return SupportDiagnostics(original, remaining, available, fraction, diagonal, excluded,
                              supported, MappingProxyType(dict(plan.metadata)),
                              float(min_weight), float(min_valid_fraction), float(weight_tolerance))


def local_support(
    mask: np.ndarray, width: WidthLike, coordinates: str = "cartesian", *,
    exclude_center: bool = False, min_weight: float = 1e-3,
    min_valid_fraction: float = 0.0, weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
    **kwargs: Any,
) -> SupportDiagnostics:
    """Return actual valid/available mass and complete center-exclusion diagnostics.

    Parameters
    ----------
    mask : ndarray
        Validity mask. Numeric nonzero finite entries are valid; nonfinite
        entries are invalid. Combine an explicit mask with finite data first.
    width : float or sequence of float
        Existing grid/physical width convention, in array axis order.
    coordinates : {'cartesian', 'polar'}, default='cartesian'
        Existing coordinate and geometry vocabulary is retained.
    exclude_center : bool, default=False
        Remove the tested physical input, including every repeated boundary visit.
    min_weight : float, default=1e-3
        Required valid mass, strictly exceeded. This is a support rule, not epsilon
        regularization. Use zero to require only numerically resolvable mass.
    min_valid_fraction : float, default=0
        Minimum valid / available mass fraction, in [0, 1].
    weight_tolerance : float, default=64*float64 epsilon
        Absolute numerical tolerance, separate from scientific support thresholds.
    **kwargs
        Existing kernel/geometry options. Only sampled Gaussian and discrete
        boxcar kernels are accepted on selected axes; inactive axes remain isolated.

    Returns
    -------
    SupportDiagnostics
        Unmasked arrays of actual weight masses and the declared support rule.
        Unlike dct_count, these are not nominal counts or statistical ESS.

    Examples
    --------
    >>> support = local_support(np.ones(5, dtype=bool), 3, exclude_center=True)
    >>> np.allclose(support.valid_fraction, 1)
    True
    """
    values = np.asarray(mask)
    valid = np.isfinite(values) & values.astype(bool)
    plan = _make_plan(valid.shape, width, coordinates, **kwargs)
    return _support_from_plan(valid, plan, exclude_center, min_weight, min_valid_fraction,
                              weight_tolerance)

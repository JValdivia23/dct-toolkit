"""Weighted local order statistics over the complete geometry-aware operator."""

from contextlib import ExitStack
from dataclasses import dataclass
from tempfile import TemporaryFile
from typing import Any, Optional, Tuple, Union

import numpy as np
import scipy.ndimage

from ._operator import _OperatorPlan, _make_plan
from ._widths import WidthLike
from .support import DEFAULT_WEIGHT_TOLERANCE, SupportDiagnostics, _support_from_plan


@dataclass(frozen=True)
class OrderStatisticResult:
    """An estimate, its support, and recorded numerical approximation information.

    Attributes
    ----------
    values : ndarray
        Statistic; vector quantiles have a leading quantile axis.
    support : SupportDiagnostics
        Original/excluded masses and the declared support rule, in input shape.
    error_bound : ndarray
        Value-quantization bound (not statistical uncertainty). Zero for exact
        and spatial evaluation or unquantized categorical histograms. CDF
        roundoff and input floating-point representation are separate tolerances.
    resolved : ndarray of bool
        Whether the returned estimate is finite after support/mask restoration.
    method : str
        Evaluation method: histogram, exact, or spatial.
    bin_count : int
        Maximum bins used per independent domain; zero outside histogram mode.
    median : ndarray or None
        Target-centered median reused inside MAD, without another histogram pass.
    """

    values: np.ndarray
    support: SupportDiagnostics
    error_bound: np.ndarray
    resolved: np.ndarray
    method: str
    bin_count: int
    median: Optional[np.ndarray] = None


StatisticOutput = Union[np.ndarray, OrderStatisticResult]


def _positive_integer(value: Any, name: str) -> int:
    """Validate a positive integer without silently rounding user controls."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _quantiles(q: Any) -> Tuple[np.ndarray, bool]:
    """Validate scalar/vector probabilities and retain the requested axis semantics."""
    values = np.asarray(q)
    if values.dtype.kind not in "iuf" or values.ndim > 1 or values.size == 0:
        raise ValueError("q must be a real scalar or nonempty 1-D probability vector")
    values = values.astype(float)
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("q must lie in [0, 1]")
    return values.reshape(-1), values.ndim == 0


def _weighted_rows(
    values: np.ndarray, weights: np.ndarray, q: np.ndarray, tolerance: float
) -> np.ndarray:
    """Evaluate lower weighted inverse CDFs for rows with ascending sample values."""
    # Tolerance applies to a value's total mass, not each observation separately.
    # Otherwise a histogram category can resolve tied tiny weights that the
    # exact evaluator discarded individually. Sum signed roundoff before clipping.
    if values.ndim == 1 and np.any(values[1:] == values[:-1]):
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        values = values[starts]
        weights = np.add.reduceat(weights, starts, axis=1)
    elif values.ndim == 2 and np.any(values[:, 1:] == values[:, :-1]):
        columns = values.shape[1]
        flat = values.ravel()
        boundaries = ((flat[1:] != flat[:-1]) |
                      (np.arange(1, flat.size) % columns == 0))
        starts = np.r_[0, np.flatnonzero(boundaries) + 1]
        stops = np.r_[starts[1:] - 1, flat.size - 1]
        grouped = np.zeros_like(weights)
        grouped.ravel()[stops] = np.add.reduceat(weights.ravel(), starts)
        weights = grouped
    if np.any(weights < -tolerance):
        raise ArithmeticError("Negative aggregate value mass exceeds numerical tolerance")
    weights = np.maximum(weights, 0.0)
    cumulative = np.cumsum(weights, axis=1)
    total = cumulative[:, -1]
    positive = weights > tolerance
    result = np.full((len(q), len(weights)), np.nan)
    for index, probability in enumerate(q):
        if probability == 0:
            selected = np.argmax(positive, axis=1)
            found = np.any(positive, axis=1)
        elif probability == 1:
            selected = weights.shape[1] - 1 - np.argmax(positive[:, ::-1], axis=1)
            found = np.any(positive, axis=1)
        else:
            hit = positive & (cumulative >= probability * total[:, None] - tolerance)
            selected = np.argmax(hit, axis=1)
            found = np.any(hit, axis=1)
        if values.ndim == 1:
            result[index] = values[selected]
        else:
            result[index] = values[np.arange(len(weights)), selected]
        result[index, (total <= tolerance) | ~found] = np.nan
    return result


def _exact(
    data: np.ndarray, valid: np.ndarray, plan: _OperatorPlan, support: SupportDiagnostics,
    q: np.ndarray, mad: bool, batch_size: int, budget: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reference evaluation using bounded batches of rows of K, via its adjoint."""
    n = data.size
    result = np.full((len(q), n), np.nan)
    medians = np.full(n, np.nan) if mad else None
    targets = np.flatnonzero(support.supported.ravel())
    order = np.flatnonzero(valid.ravel())
    order = order[np.argsort(data.ravel()[order], kind="stable")]
    if len(order) == 0 or len(targets) == 0:
        return result, medians
    count = min(batch_size, budget // max(64 * n, 1))
    if count < 1:
        raise ValueError("max_memory_mb is too small for one exact operator-row batch")
    samples = data.ravel()[order]
    for start in range(0, len(targets), count):
        indices = targets[start:start + count]
        basis = np.zeros((len(indices), n))
        basis[np.arange(len(indices)), indices] = 1
        rows = plan.apply(basis.reshape((len(indices),) + data.shape), adjoint=True)
        rows = rows.reshape(len(indices), n)
        if np.any(support.excluded_weight):
            rows[np.arange(len(indices)), indices] = 0
        weights = rows[:, order]
        if np.min(weights) < -support.weight_tolerance:
            raise ArithmeticError("Negative operator weights exceed numerical tolerance")
        if not mad:
            result[:, indices] = _weighted_rows(samples, weights, q, support.weight_tolerance)
        else:
            center = _weighted_rows(samples, weights, np.array([0.5]),
                                    support.weight_tolerance)[0]
            medians[indices] = center
            with np.errstate(over="ignore", invalid="ignore"):
                deviations = np.abs(samples[None, :] - center[:, None])
            permutation = np.argsort(deviations, axis=1, kind="stable")
            result[:, indices] = _weighted_rows(
                np.take_along_axis(deviations, permutation, axis=1),
                np.take_along_axis(weights, permutation, axis=1),
                np.array([0.5]), support.weight_tolerance,
            )
    return result, medians


def _bin_codes(
    data: np.ndarray, valid: np.ndarray, plan: _OperatorPlan, bins: int,
    value_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode values independently in each inactive-axis domain, without clipping tails."""
    active = tuple(sorted(plan.active_axes))
    inactive = tuple(a for a in range(data.ndim) if a not in active)
    permutation = inactive + active
    inverse = np.argsort(permutation)
    inactive_shape = tuple(data.shape[a] for a in inactive)
    domains = int(np.prod(inactive_shape)) if inactive_shape else 1
    values = np.transpose(data, permutation).reshape(domains, -1)
    masks = np.transpose(valid, permutation).reshape(domains, -1)
    codes = np.full(values.shape, -1, dtype=np.intp)
    representatives = []
    errors = np.zeros(domains)
    if value_range is not None:
        limits = np.asarray(value_range)
        if (limits.shape != (2,) or limits.dtype.kind not in "iuf" or
                not np.all(np.isfinite(limits)) or limits[0] >= limits[1]):
            raise ValueError("value_range must be a finite increasing (lower, upper) pair")
        with np.errstate(over="ignore", invalid="ignore"):
            limits = limits.astype(float)
        if not np.all(np.isfinite(limits)):
            raise ValueError("value_range must be representable in float64")
        if np.any(valid & ((data < limits[0]) | (data > limits[1]))):
            raise ValueError("value_range would discard valid observations; enlarge the range")
    for domain in range(domains):
        selected = values[domain, masks[domain]]
        unique, indices = np.unique(selected, return_inverse=True)
        if len(unique) <= bins:
            representatives.append(unique)
            codes[domain, masks[domain]] = indices
            continue
        lower, upper = (selected.min(), selected.max()) if value_range is None else limits
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            span = upper - lower
            if np.isfinite(span):
                fractions = (selected - lower) / span
            else:
                scale = max(abs(lower), abs(upper))
                fractions = (selected / scale - lower / scale) / (upper / scale - lower / scale)
            centers = (np.arange(bins) + 0.5) / bins
            points = (1 - centers) * lower + centers * upper
            errors[domain] = upper / (2 * bins) - lower / (2 * bins)
        # The maximum is assigned to the final bin, not discarded as overflow.
        codes[domain, masks[domain]] = np.minimum((fractions * bins).astype(np.intp), bins - 1)
        representatives.append(points)
    count = max(1, max((len(r) for r in representatives), default=0))
    table = np.full((count, domains), np.nan)
    for domain, points in enumerate(representatives):
        table[:len(points), domain] = points
    canonical_shape = tuple(data.shape[a] for a in permutation)
    domain_ids = np.broadcast_to(
        np.arange(domains).reshape(inactive_shape + (1,) * len(active)), canonical_shape
    )
    ids = np.transpose(domain_ids, inverse).ravel()
    codes = np.transpose(codes.reshape(canonical_shape), inverse)
    return codes, table, ids, errors[ids]


def _histogram(
    data: np.ndarray, valid: np.ndarray, plan: _OperatorPlan, support: SupportDiagnostics,
    q: np.ndarray, mad: bool, bins: int, batch_size: int, budget: int,
    value_range: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, int]:
    """Stream spectral bin masses; retain bounded-memory histogram storage only for MAD."""
    codes, table, domains, errors = _bin_codes(data, valid, plan, bins, value_range)
    n, count = data.size, len(table)
    stored_bytes = 8 * count * n
    in_memory = mad and stored_bytes <= budget // 3
    work_budget = budget - stored_bytes if in_memory else budget
    batch = min(batch_size, work_budget // max(64 * n, 1))
    if batch < 1:
        raise ValueError("max_memory_mb is too small for one histogram transform batch")
    probabilities = np.array([0.5]) if mad else q
    result = np.full((len(probabilities), n), np.nan)
    cumulative = np.zeros(n)
    mass = support.valid_weight.ravel()
    tolerance = support.weight_tolerance
    with ExitStack() as stack:
        storage = np.empty((count, n)) if in_memory else None
        spool = stack.enter_context(TemporaryFile()) if mad and not in_memory else None
        for start in range(0, count, batch):
            indices = np.arange(start, min(count, start + batch))
            batch_shape = (len(indices),) + (1,) * data.ndim
            fields = codes[None, ...] == indices.reshape(batch_shape)
            hist = plan.apply(fields).reshape(len(indices), n)
            if np.any(support.excluded_weight):
                hist -= (codes.ravel()[None, :] == indices[:, None]) * (
                    support.excluded_weight.ravel()[None, :]
                )
            if np.min(hist) < -tolerance:
                raise ArithmeticError("Negative histogram mass exceeds numerical tolerance")
            np.maximum(hist, 0.0, out=hist)
            for offset, index in enumerate(indices):
                weights = hist[offset]
                if storage is not None:
                    storage[index] = weights
                elif spool is not None:
                    weights.tofile(spool)
                cumulative += weights
                positive = weights > tolerance
                points = table[index, domains]
                for qi, probability in enumerate(probabilities):
                    if probability == 0:
                        hit = positive & np.isnan(result[qi])
                    elif probability == 1:
                        hit = positive
                    else:
                        hit = (np.isnan(result[qi]) & positive &
                               (cumulative >= probability * mass - tolerance))
                    result[qi, hit] = points[hit]
        if not mad:
            return result, None, errors, count
        medians = result[0].copy()
        result[:] = np.nan
        chunk = max(1, work_budget // max(64 * count, 1))
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            if storage is not None:
                weights = storage[:, start:stop].T
            else:
                weights = np.empty((stop - start, count))
                for index in range(count):
                    spool.seek((index * n + start) * 8)
                    weights[:, index] = np.fromfile(spool, dtype=float, count=stop - start)
            with np.errstate(over="ignore", invalid="ignore"):
                distances = np.abs(table[:, domains[start:stop]].T - medians[start:stop, None])
            distances[~np.isfinite(distances)] = np.inf
            order = np.argsort(distances, axis=1, kind="stable")
            result[:, start:stop] = _weighted_rows(
                np.take_along_axis(distances, order, axis=1),
                np.take_along_axis(weights, order, axis=1),
                np.array([0.5]), tolerance,
            )
    return result, medians, 2 * errors, count


def _spatial(
    data: np.ndarray, valid: np.ndarray, plan: _OperatorPlan, support: SupportDiagnostics,
    q: np.ndarray, mad: bool, budget: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Evaluate constant Cartesian discrete-boxcar distributions with vectorized tiles."""
    if not plan.cartesian or any(k != "boxcar_discrete" for k in plan.kernels):
        raise ValueError("method='spatial' requires Cartesian boxcar_discrete kernels")
    sizes = [max(1, int(np.rint(w))) for w in plan.metadata["widths_in_grid"]]
    sizes = [w + (w % 2 == 0) for w in sizes]
    volume = 1
    for size in sizes:
        volume *= size
    if volume * 64 > budget:
        raise ValueError("Spatial footprint exceeds max_memory_mb; use histogram evaluation")
    if not mad and np.all(valid) and not np.any(support.excluded_weight) and sizes:
        # Compiled rank filtering avoids Python target loops and N*window storage.
        if volume <= 4096:
            result = np.stack([scipy.ndimage.rank_filter(
                data, rank=min(volume - 1, max(0, int(np.ceil(p * volume)) - 1)),
                size=sizes, mode="reflect",
            ).ravel() for p in q])
            return result, None
    if not data.shape:
        return _exact(data, valid, plan, support, q, mad, 1, budget)
    offsets = np.array(np.meshgrid(*[
        np.arange(-(size // 2), size // 2 + 1) for size in sizes
    ], indexing="ij")).reshape(data.ndim, -1)
    result = np.full((len(q), data.size), np.nan)
    medians = np.full(data.size, np.nan) if mad else None
    chunk = max(1, budget // (64 * volume))
    for start in range(0, data.size, chunk):
        targets = np.arange(start, min(data.size, start + chunk))
        locations = np.array(np.unravel_index(targets, data.shape))
        neighbors = []
        for axis, size in enumerate(data.shape):
            indices = (locations[axis, :, None] + offsets[axis]) % (2 * size)
            neighbors.append(np.where(indices < size, indices, 2 * size - 1 - indices))
        indices = np.ravel_multi_index(tuple(neighbors), data.shape)
        weights = valid.ravel()[indices].astype(float)
        if np.any(support.excluded_weight):
            weights[indices == targets[:, None]] = 0
        values = data.ravel()[indices]
        values = np.where(weights > 0, values, np.inf)
        order = np.argsort(values, axis=1, kind="stable")
        ordered_values = np.take_along_axis(values, order, axis=1)
        ordered_weights = np.take_along_axis(weights, order, axis=1)
        probabilities = np.array([0.5]) if mad else q
        estimate = _weighted_rows(ordered_values, ordered_weights, probabilities, 0.0)
        if mad:
            medians[targets] = estimate[0]
            with np.errstate(over="ignore", invalid="ignore"):
                distances = np.abs(values - estimate[0, :, None])
            order = np.argsort(distances, axis=1, kind="stable")
            estimate = _weighted_rows(
                np.take_along_axis(distances, order, axis=1),
                np.take_along_axis(weights, order, axis=1), np.array([0.5]), 0.0,
            )
        result[:, targets] = estimate
    return result, medians


def _calculate(
    data: np.ndarray, q: Any, width: WidthLike, coordinates: str,
    mask: Optional[np.ndarray], restore_input_nan: bool, method: str, bins: int,
    batch_size: int, max_memory_mb: float, exclude_center: bool, min_weight: float,
    min_valid_fraction: float, weight_tolerance: float, return_info: bool,
    value_range: Optional[Tuple[float, float]], mad: bool, scale: float, **kwargs: Any,
) -> StatisticOutput:
    """Share input, support and method contracts across quantile, median and MAD."""
    values = np.asarray(data)
    if values.dtype.kind not in "iuf":
        raise ValueError("data must be a real numeric array")
    original_values = values
    with np.errstate(over="ignore", invalid="ignore"):
        values = values.astype(float, copy=False)
    probabilities, scalar = _quantiles(q)
    bins = _positive_integer(bins, "bins")
    batch_size = _positive_integer(batch_size, "batch_size")
    if not np.isscalar(max_memory_mb) or not np.isfinite(max_memory_mb) or max_memory_mb <= 0:
        raise ValueError("max_memory_mb must be finite and positive")
    if method not in ("exact", "histogram", "spatial"):
        raise ValueError("method must be 'exact', 'histogram', or 'spatial'")
    if not np.isscalar(scale) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    valid = np.isfinite(values)
    if mask is not None:
        mask_values = np.asarray(mask)
        mask = np.isfinite(mask_values) & mask_values.astype(bool)
        if mask.shape != values.shape:
            raise ValueError("mask must have the same shape as data")
        valid &= mask
    if original_values.dtype.kind == "f" and original_values.dtype.itemsize > 8:
        selected = True if mask is None else mask
        if np.any(selected & np.isfinite(original_values) & ~np.isfinite(values)):
            raise ValueError("Finite observations outside float64 range must be rescaled")
    plan = _make_plan(values.shape, width, coordinates, **kwargs)
    support = _support_from_plan(valid, plan, exclude_center, min_weight,
                                 min_valid_fraction, weight_tolerance)
    budget = int(max_memory_mb * 1024**2)
    count = 0
    errors = np.zeros(values.size)
    if method == "histogram":
        result, medians, errors, count = _histogram(
            values, valid, plan, support, probabilities, mad, bins, batch_size, budget, value_range
        )
    elif method == "exact":
        result, medians = _exact(values, valid, plan, support, probabilities,
                                 mad, batch_size, budget)
    else:
        result, medians = _spatial(values, valid, plan, support, probabilities, mad, budget)
    with np.errstate(over="ignore", invalid="ignore"):
        result *= scale
        errors *= scale
    permitted = support.supported.ravel().copy()
    if restore_input_nan:
        permitted &= valid.ravel()
    result[:, ~permitted] = np.nan
    result[~np.isfinite(result)] = np.nan
    output_shape = values.shape if scalar else (len(probabilities),) + values.shape
    result = result.reshape(output_shape)
    error = np.broadcast_to(errors.reshape(values.shape), output_shape).copy()
    error[~np.isfinite(result)] = np.nan
    if medians is not None:
        medians[~permitted] = np.nan
        medians = medians.reshape(values.shape)
    if return_info:
        return OrderStatisticResult(result, support, error, np.isfinite(result),
                                    method, count, medians)
    return result


def local_quantile(
    data: np.ndarray, q: Any, width: WidthLike, coordinates: str = "cartesian",
    mask: Optional[np.ndarray] = None, restore_input_nan: bool = True, *,
    method: str = "histogram", bins: int = 128, batch_size: int = 8,
    max_memory_mb: float = 64.0, exclude_center: bool = False, min_weight: float = 1e-3,
    min_valid_fraction: float = 0.0, weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
    return_info: bool = False, value_range: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> StatisticOutput:
    """Compute masked local weighted quantiles without filling missing observations.

    Parameters
    ----------
    data : ndarray
        Real input. Nonfinite observations have zero weight, not zero signal.
    q : float or 1-D array_like
        Probabilities in [0, 1]. Use the lower weighted inverse CDF; the median
        need not average two middle values. q=0/1 selects the first/last positive
        mass, above the declared numerical weight tolerance.
    width, coordinates, mask, restore_input_nan
        Existing statistics/geometry conventions. The mask must match data.
        Mask restoration affects returned values, not neighborhood membership.
    method : {'histogram', 'exact', 'spatial'}, default='histogram'
        Histogram batches DCT/FFT bin masses under the complete operator, then
        normalizes once. Exact is a small-problem reference using adjoint rows.
        Spatial is exact vectorized/compiled Cartesian discrete-boxcar evaluation.
        These evaluate the same weighted distribution, not sequential 1-D medians.
    bins : int, default=128
        Maximum value bins per isolated domain. At most bins distinct values
        use exact categories. Otherwise uniform bins return bin centers, with
        half-bin value-quantization error (available in return_info).
    batch_size : int, default=8
        Maximum simultaneous histogram transforms or reference operator rows.
    max_memory_mb : float, default=64
        Approximate working-array budget for transform batches and histogram
        evaluation, excluding inputs, outputs, geometry, and FFT library scratch.
        MAD spills its histogram to a temporary file when necessary.
    exclude_center : bool, default=False
        Remove all contributions of the target input, including boundary repeats.
    min_weight, min_valid_fraction, weight_tolerance
        Scientific mass/fraction rules and separate numerical tolerance, as in
        local_support. Unsupported estimates remain NaN; no fallback is performed.
    return_info : bool, default=False
        Return OrderStatisticResult with support, quantization bound and method.
    value_range : (float, float), optional
        Histogram range. Valid values outside it raise ValueError; no tail mass
        is silently clipped. Default ranges are independent per inactive domain.
    **kwargs
        Existing kernel/geometry options. Selected kernels must be gaussian or
        boxcar_discrete. Gaussian retains its full sampled response, without a
        fixed spatial cutoff. Batch axes never become filtering or quantile axes.

    Returns
    -------
    ndarray or OrderStatisticResult
        Scalar q preserves input shape; vector q prepends a quantile axis.
        Error bounds describe value quantization, not statistical uncertainty or
        CDF ambiguity within weight_tolerance of a probability jump.

    Examples
    --------
    >>> local_quantile(np.array([1., 9., 3.]), 0.5, 3, kernel_type='boxcar_discrete')
    array([1., 3., 3.])
    """
    return _calculate(data, q, width, coordinates, mask, restore_input_nan, method,
                      bins, batch_size, max_memory_mb, exclude_center, min_weight,
                      min_valid_fraction, weight_tolerance, return_info, value_range,
                      False, 1.0, **kwargs)


def local_median(
    data: np.ndarray, width: WidthLike, coordinates: str = "cartesian",
    mask: Optional[np.ndarray] = None, restore_input_nan: bool = True, **kwargs: Any,
) -> StatisticOutput:
    """Compute the lower weighted local median, exactly local_quantile(q=0.5).

    Parameters
    ----------
    data, width, coordinates, mask, restore_input_nan
        As in local_quantile; missing observations never create support.
    **kwargs
        All numerical, histogram, support, exclusion and geometry options of
        local_quantile, including return_info. Defaults are identical.

    Returns
    -------
    ndarray or OrderStatisticResult
        Local medians in input shape, with optional diagnostics.
    """
    return local_quantile(data, 0.5, width, coordinates, mask, restore_input_nan, **kwargs)


def local_mad(
    data: np.ndarray, width: WidthLike, coordinates: str = "cartesian",
    mask: Optional[np.ndarray] = None, restore_input_nan: bool = True, *,
    method: str = "histogram", bins: int = 128, batch_size: int = 8,
    max_memory_mb: float = 64.0, exclude_center: bool = False, min_weight: float = 1e-3,
    min_valid_fraction: float = 0.0, weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
    return_info: bool = False, value_range: Optional[Tuple[float, float]] = None,
    scale: float = 1.0, **kwargs: Any,
) -> StatisticOutput:
    """Compute weighted median absolute deviation about each target's own median.

    Parameters
    ----------
    data, width, coordinates, mask, restore_input_nan
        As in local_quantile.
    method, bins, batch_size, max_memory_mb
        The same exact/spatial/histogram evaluation and bounded-memory controls
        as local_quantile. Histogram median and MAD share bin masses and support.
    exclude_center, min_weight, min_valid_fraction, weight_tolerance
        The same support/exclusion contract as local_quantile and local_support.
    return_info : bool, default=False
        Return diagnostics, including the target median reused in the estimate.
    value_range : (float, float), optional
        Range as in local_quantile; out-of-range valid values are rejected.
    scale : float, default=1
        Explicit positive multiplier. Use 1.4826 for approximate normal-reference
        scaling. No uncertainty floor is imposed; zero MAD remains zero.
    **kwargs
        Existing positive-kernel geometry options, as in local_quantile.

    Returns
    -------
    ndarray or OrderStatisticResult
        Unscaled MAD by default, in input shape. Histogram value-quantization
        error is at most a whole bin width (times scale), apart from declared
        CDF roundoff. This is not median_filter(abs(x - median_filter(x))).
    """
    return _calculate(data, 0.5, width, coordinates, mask, restore_input_nan, method,
                      bins, batch_size, max_memory_mb, exclude_center, min_weight,
                      min_valid_fraction, weight_tolerance, return_info, value_range,
                      True, scale, **kwargs)

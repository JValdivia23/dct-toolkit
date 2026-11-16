# Local weighted order statistics (v2.1.0)

Introduced in v2.1.0, the `order_stats` and `support` modules extend the spectral operators with
geometry-aware weighted distributions. Public names describe the statistic:
`local_quantile`, `local_median`, `local_mad`, and `local_support`. They are also
exported from `dct_toolkit`. Existing `dct_*` names and their defaults remain.
Radar interpretation, clutter thresholds, terrain rules, and reconstruction
eligibility belong to the application. No new gap-filling API is exported.

## A complete neighborhood distribution

Let K be the existing complete finite-domain operator: Cartesian tensor product,
or ordered azimuth/elevation/range/time/linear polar passes. M combines the
explicit mask and finite original data. An observation's weight at target i is
K[i,j]*M[j]. These are weight masses, not physical-volume integrals or independent
sample counts. Only `gaussian` and `boxcar_discrete` are accepted on selected
axes. Signed kernels on inactive axes are harmless; active `gaussian_spectral`
and analytical `boxcar` are rejected even on empty masks.

Filtering bin indicators uses the same complete K. Normalize after every axis
has been applied, once. Sequential one-dimensional medians and medians of DCT
coefficients define different estimators. The sampled Gaussian retains its full
response to floating-point precision, with no fixed spatial cutoff.

Scalar q preserves the input shape. A nonempty vector q creates a **leading**
quantile axis in requested order, including repeated probabilities. This axis is
never filtered. Coordinates, widths, units, boundaries, and selected axes use
the existing vocabulary. Unfiltered/batch domains never exchange observations;
even automatic histogram ranges/categories are chosen independently per domain.
Descending or irregular coordinate vectors remain unsupported; adapt them before
calling the toolkit.

## Quantiles, ties, and MAD

The convention is the lower weighted inverse CDF: choose the first supported
value whose cumulative mass reaches q times the total mass. Tied values have
their combined mass, aggregated **before** numerical mass/tolerance comparisons
in every method. This differs from averaging two middle unweighted values.
The median is exactly q=0.5 under that convention.

For every target i, first find its own median m[i], then take the weighted median
of `abs(x[j] - m[i])` with the **same target's weights**. The toolkit never uses
`median_filter(abs(x - median_filter(x)))`. MAD is unscaled by default; request
`scale=1.4826` explicitly for approximate normal-reference scaling. Zero MAD is
preserved. Measurement/statistical uncertainty floors belong to the application.

q=0/1 selects the first/last numerically positive mass under the declared weight
tolerance. In a quantized histogram, returned bin centers approximate these
extrema too; use `method="exact"` or the supported `"spatial"` method when exact
data-valued endpoints are needed. The tolerance prevents roundoff outside an
otherwise finite boxcar neighborhood from becoming a false extreme.

```python
import numpy as np
from dct_toolkit import local_quantile, local_mad

x = np.array([1., 9., np.nan, 3., 4., 2.])
options = dict(kernel_type="boxcar_discrete", exclude_center=True,
               restore_input_nan=False)
quartiles = local_quantile(x, [0.25, 0.5, 0.75], 3, method="spatial", **options)
result = local_mad(x, 3, method="histogram", return_info=True, **options)
assert quartiles.shape == (3,) + x.shape
assert result.values.shape == x.shape
# result.median is the target median already reused in this MAD calculation.
```

## Evaluation methods and approximation controls

```python
local_quantile(data, q, width, coordinates="cartesian", mask=None,
               restore_input_nan=True, *, method="histogram", bins=128,
               batch_size=8, max_memory_mb=64.0, exclude_center=False,
               min_weight=1e-3, min_valid_fraction=0.0,
               weight_tolerance=64*np.finfo(float).eps,
               return_info=False, value_range=None, **geometry)

local_median(data, width, coordinates="cartesian", mask=None,
             restore_input_nan=True, **options)

local_mad(data, width, coordinates="cartesian", mask=None,
          restore_input_nan=True, *, scale=1.0, **options)

local_support(mask, width, coordinates="cartesian", *, exclude_center=False,
              min_weight=1e-3, min_valid_fraction=0.0,
              weight_tolerance=64*np.finfo(float).eps, **geometry)
```

Median accepts the same evaluation controls as quantile. MAD accepts those same
controls except q, plus scale. Geometry options are the existing per-axis kernel,
role, coordinate, unit, boundary and filtering-axis parameters.

| Method | Mathematical contract | Intended use |
|---|---|---|
| `histogram` (default) | Same distribution, with explicit value quantization | General Cartesian/polar workloads |
| `exact` | Full operator-row weighted inverse CDF | Independent small-problem reference |
| `spatial` | Same exact weighted distribution with repeated boundary samples | Fixed Cartesian discrete boxcars |

`bins=128` is the maximum number of value categories per independent domain.
When the number of distinct valid values is no greater than bins, categories
preserve those values exactly. Otherwise, uniform bins span that domain's actual
finite extrema and estimates use bin centers. `value_range=(low, high)` can fix
the range explicitly; values outside it raise an error instead of losing tail
mass. Invalid/masked observations do not constrain the range.
Evaluation uses float64. Finite unmasked observations or explicit ranges outside
float64's representable range raise an error and must be rescaled explicitly;
they are never silently converted into missing observations.

`return_info=True` returns `OrderStatisticResult`:

- `values`: the estimate, retaining scalar/vector-q shape conventions;
- `support`: `SupportDiagnostics` in original input shape;
- `error_bound`: half-bin quantile or whole-bin MAD value-quantization bound,
  with MAD scaling included; zero for categorical/exact/spatial evaluation;
- `resolved`: finite returned estimates after support and mask restoration;
- `method`, `bin_count`: the evaluation and maximum categories actually used;
- `median`: the reused target median for MAD, otherwise None.

The bound concerns value quantization, not measurement uncertainty or CDF
ambiguity within the numerical tolerance of a probability jump. Increasing bins
tightens the worst-case value bound; the realized error need not decrease
monotonically for every target. Overflowing MAD/scaled estimates are unresolved
NaNs rather than silently clipped extreme values.
An estimate also remains unresolved if no eligible value mass/CDF hit exceeds
the numerical tolerance, even when total observation mass passes the scientific
support rule. `support.supported` records that mass rule; `resolved` additionally
records whether a finite statistic could actually be returned.

Histogram evaluation resolves geometry and transfer functions once, batches bin
indicators (`batch_size=8` maximum), and reuses the original mask response.
Quantiles stream bin masses without storing B complete arrays. MAD reuses the
same histogram to obtain its center and deviations; it does not run a second
set of bin convolutions. When necessary, temporary disk storage replaces the
full in-memory histogram and is removed on return or error.

`max_memory_mb=64` controls approximate transform/histogram working-array sizes.
It is **not a process RSS limit**: inputs, outputs, quantization encodings,
geometry, allocator retention, and FFT library scratch also consume memory.
A single transform must fit the estimated batch budget or the call raises a
clear error. MAD temporary storage can require `8 * B * data.size` bytes on disk.

The exact reference takes O(N^2) operator-row work and should not be selected for
large arrays. It evaluates row batches through the adjoint, reversing ordered
polar passes; K is not assumed globally symmetric. Spatial evaluation uses a
compiled rank filter on fully valid, nonexcluded boxes where practical, and
bounded vectorized neighbor tiles for masks, exclusion and MAD. Broad footprints
can make this method expensive; histogram work is largely independent of spatial
width. See [performance measurements](ORDER_STATISTICS_PERFORMANCE.md).

## Actual support and full center exclusion

```python
from dct_toolkit import local_support

support = local_support(np.isfinite(x), 3, kernel_type="boxcar_discrete",
                        exclude_center=True)
```

`SupportDiagnostics` provides:

| Field | Definition |
|---|---|
| `original_valid_weight` | K(M) |
| `self_weight` | Full operator diagonal kappa[i]=K[i,i] |
| `excluded_weight` | kappa[i]*M[i] when excluding, otherwise zero |
| `valid_weight` | Original valid weight minus excluded weight |
| `available_weight` | K(1)-kappa[i] when excluding, otherwise K(1)=1 |
| `valid_fraction` | valid_weight / available_weight, NaN if unavailable |
| `supported` | Recorded mass, availability, and fraction rule |
| `geometry` | Resolved roles, axes, widths/units, kernels and boundaries |

Repeated reflections/wraps of the same physical input are aggregated in Kii;
subtracting one nominal kernel-center coefficient would be wrong. Available
external mass need not be one. Identity/singleton-only support vanishes after
exclusion and cannot fall back to the tested observation.

By default, valid mass must **strictly exceed** `min_weight=1e-3`, available mass
must exceed `weight_tolerance=64*float64_epsilon`, and valid fraction must be at
least `min_valid_fraction=0`. Set min_weight=0 to require only numerically
resolvable mass. The numerical tolerance is separate from scientific support
thresholds; tiny negative transform weights are treated as roundoff, while
larger violations raise an error. Unsupported estimates remain NaN.
Fraction thresholds are compared in mass units, allowing weight_tolerance of
roundoff: `valid_weight >= min_valid_fraction*available_weight - weight_tolerance`.
This prevents a complete external neighborhood from failing a 100% fraction
requirement solely because subtraction of Kii magnified transform roundoff.

Diagnostics themselves are not masked at the target. `restore_input_nan=True`
on order statistics restores invalid original targets without changing their
neighborhood distributions. Setting it False permits supported estimates there;
it does not manufacture new observations. `dct_count` still means nominal
weighted support and is not replaced by these diagnostics. Neither quantity
is statistical ESS for correlated observations.

## Strict no-filling mode on existing statistics

```python
from dct_toolkit import dct_mean, dct_variance, dct_smooth

strict_mean = dct_mean(x, 3, fallback="none", **options)
strict_variance = dct_variance(x, 3, fallback="none", **options)
strict_smoothed = dct_smooth(x, 3, fallback="none", **options)
```

`fallback="prefill"` preserves all existing defaults. `fallback="none"` on
`dct_mean`, `dct_variance`, `dct_std`, and `dct_smooth` uses only original valid
observations: `K(M*x)/K(M)`, with original-mask restoration as requested.
Unsupported ratios return NaN without prefill, nearest-neighbor fallback, or
additive-epsilon regularization. Strict smoothing is normalized convolution,
not merely applying a transform to zero-filled data.

`exclude_center=True` on these existing functions requires `fallback="none"`
and validated nonnegative kernels. Both first and second moments subtract the
full tested-observation contribution. Strict variance fuses original first
moment, second moment and mask into one cached batch. Its usual final numerical
variance clamp remains. `min_effective_density` keeps its original raw-density
meaning without exclusion; with exclusion it gates valid/available external
mass. The raw denominator must still exceed the existing default floor 1e-3.

New order statistics always use this no-fabricated-support policy; there is no
prefill switch on them. Applications can reconstruct eligible isolated gaps in
a separate stage, then explicitly request final strict normalized smoothing.

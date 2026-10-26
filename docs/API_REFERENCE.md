# API Reference

This reference targets **v1.0.0**, including the public smoothing/statistics API
and the backward-compatible v0.6.0 legacy 2D polar interface.

## Core Primitives

### `dct_toolkit.core`

#### `get_dct_transfer_function(n, kernel_type, width)`
Generate a 1D spectral transfer function `H[k]`.

- `n` (int): Number of points.
- `kernel_type` (str): `'boxcar'`, `'boxcar_discrete'`, `'gaussian'`.
- `width` (float): Kernel width in pixel units.

#### `dct_convolve_1d(data, H, axis=-1)`
Apply 1D convolution using DCT-II (`norm='ortho'`).

- `data` (`np.ndarray`): Input data.
- `H` (`np.ndarray`): Transfer function matching `data.shape[axis]`.
- `axis` (int): Axis to smooth.

---

## High-Level Smoothing

### `dct_toolkit.dct_smooth(data, width, coordinates='cartesian', prefill_max_iter=3, **kwargs)`
Top-level convenience wrapper for Cartesian and polar smoothing.

- `coordinates='cartesian'`: calls `smooth_cartesian(data, width, **kwargs)`.
- `coordinates='polar'`: calls `smooth_polar(data, width_pixels=width, **kwargs)`.
- Default kernel is `'gaussian'` unless `kernel_type` is explicitly provided.
- Cartesian kernels follow `smooth_cartesian`. Legacy 2D polar calls accept a
  string or `(kernel_azimuth, kernel_range)` pair. Explicit N-D polar calls accept
  a string or one kernel per array axis, as documented below. The same kernels
  and geometry are used in prefill and final smoothing.
- If input contains NaNs, smoothing automatically runs a NaN-safe prefill step,
  applies spectral smoothing on the finite field, then restores the original NaN mask.
- All new polar arguments (`axis_types`, `axis_coords`, `boundary`, `width_units`,
  `angle_units`, `filter_axes`) pass through `**kwargs` with `coordinates='polar'`.
  `width` remains positional or keyword; the wrapper maps it to `width_pixels`.
- Polar geometry is validated even on all-NaN input. Unfiltered/batch domains
  never exchange support during prefill, smoothing, or fallback; wholly missing
  independent domains remain missing without contaminating supported domains.
- `prefill_max_iter=3` by default for predictable runtime.
- `prefill_max_iter=None` means iterate prefill until convergence (or safety cap of 20).
- Legacy alias: passing `max_iter=...` in kwargs is mapped to `prefill_max_iter`.

### `dct_toolkit.cartesian`

#### `smooth_cartesian(data, width, kernel_type='gaussian')`
Apply separable DCT smoothing to N-D Cartesian data.

- `data` (`np.ndarray`): Any-dimensional array.
- `width` (`float` or sequence): A scalar applies the same width to every axis.
  A sequence must have length `data.ndim` and follows array axis order.
  Values must be finite and positive, in grid cells along the corresponding axis.
  Gaussian widths use `sigma = width / sqrt(12)`; boxcar widths are full widths.
  `'boxcar_discrete'` rounds width to an integer, clamps it to at least 1, and
  increments even values to obtain an odd window size.
- `kernel_type` (`str` or sequence of `str`): `'boxcar'`, `'boxcar_discrete'`,
  or `'gaussian'`. A string selects the same kernel for every axis. A 1-D list,
  tuple, or NumPy array must have length `data.ndim` and selects one kernel per
  axis, independently of whether `width` is scalar or a sequence.
- Invalid kernel sequence length, dimensionality, non-string entries, or unknown
  kernel names raise `ValueError`. Errors for invalid entries identify the axis.
- Boundaries are reflective on every axis. Mixed kernels retain a single N-D
  forward/inverse DCT pair. Equal widths with different kernel types can still
  produce anisotropic smoothing.
- Input should be finite. Use `dct_mean` for normalized convolution with gaps,
  or `dct_smooth` for automatic prefill followed by smoothing.

For data in **(z, y, x)** order:

```python
import numpy as np
from dct_toolkit import smooth_cartesian

volume = np.random.default_rng(42).standard_normal((16, 32, 32))
smoothed = smooth_cartesian(
    volume,
    width=(3.0, 5.0, 5.0),
    kernel_type=("gaussian", "boxcar", "boxcar"),
)
```

If the volume is stored in `(x, y, z)` order, the corresponding sequences are
`width=(5.0, 5.0, 3.0)` and `kernel_type=("boxcar", "boxcar", "gaussian")`.
Physical widths must first be divided by the grid spacing along each axis.

### `dct_toolkit.polar`

#### `smooth_polar`

```python
def smooth_polar(
    data, width_pixels, az_res_deg=1.0, az_boundary="reflective",
    range_boundary="reflective", kernel_type="gaussian", *,
    axis_types=None, axis_coords=None, boundary=None,
    width_units="physical", angle_units="degrees", filter_axes=None,
):
    ...
```

The six existing positional arguments and `width_pixels` name are unchanged.
The new arguments are keyword-only. Output has the same shape as input. The
low-level smoother expects finite input; use `dct_mean` for normalized statistics
on gaps, `dct_prefill` to fill them, or `dct_smooth` to prefill automatically and
restore the original NaN mask.

##### Compatibility mode: `axis_types=None`

This is the existing 2D **(azimuth, range)** path, not inferred physical geometry.

- `width_pixels` is a finite positive scalar or `(width_azimuth, width_range)`
  pair. At range index `r=1,...,n_range`, the azimuth width in beams is
  `width_azimuth / (r * dtheta)`; range width is in range gates. Actual first-gate
  distance and elevation are not modeled.
- `kernel_type` is a supported string or a length-2 sequence in `(azimuth, range)`
  order. Supported names are `'boxcar'`, `'boxcar_discrete'`, and `'gaussian'`.
- `az_res_deg` is a finite, positive real scalar in degrees and must stay positive
  after conversion to radians. Python/NumPy numeric scalars and zero-dimensional
  numeric arrays are accepted; boolean, complex, string, and nonscalar inputs
  are rejected.
- `az_boundary` is `'reflective'` (DCT) or `'periodic'` (RFFT).
  `range_boundary` retains its existing reflective-only behavior.
- Filtering applies adaptive azimuth first, then range. It preserves the v0.6.0
  index-based units, even though the new `width_units` default is `'physical'`.
  This default has no reinterpretation effect in compatibility mode.
- Nondefault new geometry options require explicit `axis_types`: do not pass
  coordinates, boundary overrides, filtering selection, or nondefault unit options
  and expect the legacy path to infer their meaning.

```python
import numpy as np
from dct_toolkit import dct_smooth

radar_data = np.random.default_rng(42).standard_normal((360, 100))
smoothed = dct_smooth(
    radar_data, (5.0, 3.0), coordinates="polar",
    kernel_type=("boxcar", "gaussian"), az_res_deg=1.0, az_boundary="periodic",
)
```

The module-level helpers `compute_polar_transfer_functions` and
`compute_polar_transfer_functions_v2` remain legacy 2D interfaces: they accept the
same kernel pair and return the adaptive azimuth transfer array and range
transfer vector. They are not an N-D geometry construction API.

##### Explicit N-D geometry: `axis_types` supplied

**Axis metadata and selection**

- `axis_types`: a length-`data.ndim` sequence in **array axis order**. Roles are
  `'azimuth'`, `'elevation'`, `'range'`, `'time'`, `'linear'`, `'cartesian'` (alias
  of `'linear'`), and `'batch'`. There is at most one axis of each spherical role
  (azimuth/elevation/range), with arbitrary extra time/linear/batch dimensions.
- `axis_coords`: `None`, or a length-`ndim` sequence whose entries are `None` or
  numeric 1D arrays of length `data.shape[axis]`. Every supplied array must be
  finite, real, strictly increasing, and uniformly spaced. Singleton coordinate
  arrays are allowed; explicit mode rejects zero-length array axes. Elevation
  coordinates must lie in `[-90, 90]` degrees or `[-pi/2, pi/2]` radians.
  Supply actual coordinates wherever a physical conversion or its geometry
  dependencies require them; `None` is not a substitute for actual range/elevation
  in a physical angular filter. All supplied coordinates are validated, including
  those on unfiltered axes.
  Integer coordinate spacings are checked before floating-point conversion, so
  a large integer time origin does not destroy otherwise regular sampling.
  Floating coordinates use at least `float64` precision for spacing checks,
  preventing low-precision subtraction from overflowing or hiding irregularity.
- `filter_axes=None`: filter every non-batch axis. An explicit sequence of axis
  indices selects only those axes; negative indices are supported and duplicates
  are invalid. Indices are array axes, not role-order positions or a requested
  filter order. An empty sequence selects no axes (identity smoothing, no
  inter-cell filling). Selecting a `'batch'` axis is invalid.
  Unselected axes remain independent domains, including in **all** iterative
  filling, nearest-neighbor propagation, and low-support fallbacks. A wholly
  missing independent domain cannot borrow support from another one.

**Widths, kernels, units, and boundaries**

- `width_pixels`: despite its retained compatibility name, it follows
  `width_units` in explicit mode. A finite positive scalar broadcasts; a 1D
  sequence must contain one finite positive width per array axis, including
  placeholders for batch/unfiltered axes.
- `kernel_type`: one supported kernel string or a length-`ndim` sequence.
  Each selected axis uses its own kernel, independently of width/boundary choice.
  Gaussian `sigma = effective_grid_width / sqrt(12)`. Discrete boxcars round
  effective grid width to an integer, clamp to at least 1, and increment even
  sizes to obtain an odd window, locally where geometry changes the width.
- `width_units='physical'`: a string broadcasts, or a length-`ndim` sequence
  selects `'physical'` or `'grid'` independently per axis. `'grid'` means literal
  grid cells on that axis; it disables angular distance conversion for that axis.
  Physical time/linear/range widths use that axis's coordinate units and are
  divided by its coordinate spacing. Physical angular widths are **distances in
  the range coordinate's units**, not angular coordinate units.
- `angle_units='degrees'`: a single global string, `'degrees'` or `'radians'`,
  describes all angular coordinate arrays. Angular spacings and elevations are
  converted to radians internally. This is not a per-axis sequence.
  `az_res_deg` is a legacy option, not a substitute for explicit coordinates.
- `boundary=None`: inherit `az_boundary` for the azimuth role and `range_boundary`
  for range; other roles default to reflective. An explicit string broadcasts, or
  a length-`ndim` sequence selects `'reflective'` (DCT-II) or `'periodic'` (RFFT)
  per axis, overriding inherited choices. Unlike legacy mode, explicit mode can
  select a periodic range/time/linear/elevation axis as well as periodic azimuth.
  The user is responsible for whether that periodic model is appropriate.
- All per-axis sequences follow **array order**, including widths, kernels,
  boundaries, and width units. Reorder all of them, plus metadata and selected
  axis indices, when transposing data.
- Singleton axes are identity operations and have nominal support width 1;
  batch/unfiltered axes contribute no support-area factor.

For physical azimuth/elevation filtering, the effective grid widths are

```text
w_az(R, el) = L_az / (R * cos(el) * delta_az)
w_el(R)     = L_el / (R * delta_el)
```

Here `R` is the actual range coordinate, and angular quantities are in radians.
If no elevation axis is present, `cos(el)=1`. A non-singleton physical angular
filter requires a range axis with actual positive coordinates and the relevant
angular coordinates. Physical azimuth also needs elevation coordinates if that
role exists, even if elevation is unfiltered. The actual first range gate is
honored. Origins and poles that make the conversion singular are rejected rather than silently
replaced or capped. Non-singleton physical periodic azimuth filtering requires
`n_azimuth * delta_az = 2*pi`: a complete circle with no duplicate endpoint.
Reflective azimuth can represent a regular sector instead.

**Operator order and limitations**

Selected filters run in canonical **azimuth, elevation, range, time, linear**
order, independent of array storage order. Geometry-dependent angular filters
are applied on their local range/elevation slices. This generally nonseparable,
noncommuting composition is not an exact isotropic Cartesian/geodesic convolution
or a cell-volume-weighted average. Coordinates must describe a regular tensor
grid, not an irregular concatenation of radar sweeps. The API does not resample
irregular coordinates or perform geodesic/beam-volume integration.

See [`examples/nd_polar.py`](../examples/nd_polar.py) for a runnable synthetic
`(time, elevation, azimuth, range)` example with physical widths, mixed kernels,
independent boundaries, missing data, axis permutation, and unfiltered time
domains. Only NumPy/SciPy and this package are needed.

---

## Statistical Operations

### `dct_toolkit.stats`

All functions below accept positional or keyword `width`. Cartesian widths and
kernels follow `smooth_cartesian`; legacy polar scalar/pair conventions follow
the compatibility path above. With `coordinates='polar'`, **all** statistics,
including `dct_count` and `dct_prefill`, forward the new geometry arguments through
`**kwargs`. Explicit N-D widths/kernels follow array order. Validation also applies
to all-NaN inputs and prefill calls with no targets.

Let `S` be the complete selected smoothing operator and `m` the validity mask.
Invalid samples are zero-filled before filtering. Both numerator and mask pass
through **all** filters before a single division: `S(m*x)/S(m)`. Normalizing
separately after each axis is not the same operation. Both variance moments,
prefill, and density diagnostics use the same geometry, kernels, boundaries, and
canonical filter order. Unfiltered/batch dimensions never contribute support or
exchange values, even during fallback.

The mask response `S(m)` is a support-density diagnostic, not statistical ESS.
Nonnegative spatial weights imply a response in `[0, 1]`; arbitrary fractional
spectral kernels need not have nonnegative spatial weights and may overshoot.
`dct_count` explicitly clips its density. Do not assume positivity merely from
the kernel name or from a nonnegative spectral response.

#### `dct_count(mask, width, coordinates='cartesian', restore_input_nan=True, **kwargs)`

Report **nominal weighted support**, retaining the existing API name:

```text
count(i) = clip(S(mask)(i), 0, 1) * A(i)
```

- `mask` is a boolean/binary validity array. `restore_input_nan=True` sets output
  to NaN where it is False; set it to False to inspect support at missing cells.
- Cartesian `A` is `prod(widths)`, preserving the existing convention. Legacy
  polar `A(r)` is `width_range * width_azimuth / (r * dtheta)` with `r` starting at 1.
- Explicit N-D polar `A(i)` is the product of the local effective grid widths on
  **filtered axes only**. Singleton axes contribute 1; batch/unfiltered axes
  contribute no factor. Angular widths can depend on range and elevation.
- The nominal factor is not replaced with the discrete-boxcar rounded window
  length or a Gaussian statistical-ESS factor. It is not a literal count of
  distinct neighbors, a physical area/volume, or a cell-volume quadrature weight.
  Reflective boundaries and oversized circular windows can revisit samples.
- The explicit count factor is computed only when requested. If its final product
  cannot be represented as a finite positive `float64`, `dct_count` raises
  `ValueError`. Smoothing and normalized moments do not require that product;
  their individual effective widths must still be finite and positive.
- For independent samples with combined spatial weights `q_j`, statistical ESS
  would instead use `(sum(q_j))**2 / sum(q_j**2)`. Repeated visits to the same
  sample must be combined before squaring. `dct_count` does not compute this.

#### `dct_mean`

```python
dct_mean(data, width, coordinates="cartesian", mask=None,
         restore_input_nan=True, min_effective_density=None, **kwargs)
```

- Returns floating-point output. `mask`, when supplied, must match `data.shape`;
  only entries both masked valid and finite contribute. Otherwise finite input
  defines validity.
- The normal public ratio path requires a **finite denominator strictly greater
  than `1e-3`** (`_MEAN_DENOMINATOR_FLOOR`). No additive epsilon is used.
- At unsupported/poorly conditioned positions, the existing stability fallback
  prefills the field and smooths it. Support must exist in the same independent
  filtered domain; wholly missing domains remain NaN.
- `restore_input_nan=True` masks output at invalid input support. Set it to False
  to retain estimates in gaps, without permitting cross-domain borrowing.
- `min_effective_density=None` retains the default stability behavior. A positive
  threshold, typically in `[0, 1]`, discards results where the original `S(m)` is
  below it, **after** fallback. This is a dimensionless support-density gate,
  not an ESS threshold. It does not turn the fallback into a ratio estimator.

#### `dct_prefill`

```python
dct_prefill(data, width, coordinates="cartesian", fill_mask=None,
            max_iter=3, min_effective_density=0.35, **kwargs)
```

- Iteratively fills gaps using normalized-convolution estimates; non-target
  values are preserved exactly. `fill_mask` uses `True = fill/replace here`;
  without it, NaN positions are targets. If a finite targeted value cannot be
  estimated, its original value is retained.
- `max_iter=3` is the default; `None` runs until convergence or a safety cap of
  20 iterations. Iteration can stop early when no targets remain or no progress
  can be made.
- `min_effective_density=0.35` gates iterative estimates using local mask
  response; accepted iterative candidates must exceed the clipped-density
  threshold. Set it to `None` or a non-positive value to disable this gate.
  Residual nearest-neighbor fill is a separate fallback, not a guarantee that
  every returned value passed the iterative density gate.
- Residual targets use nearest-neighbor propagation within their selected
  filtered domain. A finite output is possible only where that domain has
  support; an all-missing independent domain stays missing.
- Legacy polar propagation tries range before azimuth. Periodic azimuth wraps
  index distances across the seam in both axis-wise and global fallback;
  equidistant axis-wise azimuth neighbors prefer the preceding beam.
- Explicit geometry honors selected axes and their reflective/periodic
  boundaries in fallback as well. Nearest-neighbor fallback is an index-space
  filling rule, not physical-distance/geodesic interpolation. It cannot use an
  unfiltered time/elevation/range/batch axis as a bridge to another domain.
- Explicit fallback retains range-first and other unique-role axis passes.
  Repeated active time/linear roles skip individual passes; residual targets use
  simultaneous mixed-boundary nearest completion. Exact global ties average all
  nearest seeds after the retained passes, without using newly completed global
  targets as donors. This avoids layout-dependent results for repeated roles;
  legacy 2D nearest-neighbor rules are unchanged.

#### `dct_variance`

```python
dct_variance(data, width, coordinates="cartesian", mask=None,
             restore_input_nan=True, **kwargs)
```

Compute `max(0, E[x**2] - E[x]**2)` from independent normalized first and second
moments using the same complete `S`. The second moment is `S(m*x**2)/S(m)`, not
the square of a smoothed/prefilled field. Where fallback is needed, `x` and
`x**2` are **prefilled separately** before smoothing. `mask` follows `dct_mean`;
`restore_input_nan=True` masks invalid input support. Additional mean options,
including the density gate, pass to both moments.

This is a weighted population-moment estimate, not a Bessel-corrected unbiased
sample variance. Nonnegative-weight variance guarantees apply only when the
actual spatial weights are nonnegative. Clamping does not make an arbitrary
signed-weight spectral estimator a probability-weighted variance.

#### `dct_std`

```python
dct_std(data, width, coordinates="cartesian", mask=None,
        restore_input_nan=True, **kwargs)
```

Return `sqrt(dct_variance(...))`, with the same mask, geometry, fallback, and
restoration conventions.

---

## Workflow Diagram

All high-level functions below support `coordinates='cartesian'` or
`coordinates='polar'` where applicable (`dct_smooth`, `dct_count`, `dct_mean`,
`dct_prefill`, `dct_variance`, `dct_std`).

```text
                             +-------------------------------+
                             | get_dct_transfer_function     |
                             +---------------+---------------+
                                             |
                                             v
                             +-------------------------------+
                             | dct_convolve_1d               |
                             +---------------+---------------+
                                             |
                        +--------------------+--------------------+
                        |                                         |
                        v                                         v
            +---------------------------+             +---------------------------+
            | smooth_cartesian          |             | smooth_polar              |
            | (N-D Cartesian smoother)  |             | (N-D / legacy 2D polar)   |
            +-------------+-------------+             +-------------+-------------+
                          |                                         |
                          +--------------------+--------------------+
                                               |
                                               v
                                      +------------------+
                                      | dct_smooth       |
                                      | wrapper          |
                                      +--------+---------+
                                               |
                                      if NaNs: v
                                      +------------------+
                                      | dct_prefill      |
                                      | iterative fill   |
                                      +--------+---------+
                                               |
                                               v
                                   (smooth again, restore NaN mask)


    +--------------------+    clips smooth(mask), scales by nominal support
    | dct_count          |-----------------------------------------------> output support
    +--------------------+

    +--------------------+    normalized convolution: smooth(x*mask)/smooth(mask)
    | dct_mean           |-----------------------------------------------> output mean
    +---------+----------+
              |
              +--> internal low-support fallback may call dct_prefill

    +--------------------+    independent E[x], E[x^2], then clamp
    | dct_variance       |-----------------------------------------------> output variance
    +---------+----------+
              |
              v
    +--------------------+    sqrt(variance)
    | dct_std            |-----------------------------------------------> output std
    +--------------------+
```

## Quick Cheat Sheet

- `get_dct_transfer_function`: Build 1D spectral kernel `H[k]` from width and kernel type.
- `dct_convolve_1d`: Apply one 1D DCT convolution along one axis using a precomputed `H`.
- `smooth_cartesian`: Low-level separable smoothing for N-D Cartesian arrays.
- `smooth_polar`: Low-level explicit N-D geometry or compatible legacy 2D polar smoothing.
- `dct_smooth`: Top-level "just smooth this" wrapper (auto-pre-fills NaNs, then restores NaN mask).
- `dct_mean`: NaN-robust local mean via normalized convolution.
- `dct_count`: Nominal weighted support from a validity mask, not statistical ESS.
- `dct_prefill`: Iterative normalized-convolution gap fill (often used before full-field smoothing).
- `dct_variance`: NaN-robust local variance from normalized-convolution moments.
- `dct_std`: NaN-robust local standard deviation (`sqrt(dct_variance)`).

---

## Scope Note

`dct_prefill` is public. The separate spectral-inpainting/diffusion algorithms in
[`dct_toolkit/gap_filling.py`](../dct_toolkit/gap_filling.py) remain
research-internal, are not exported by the top-level API, and do not inherit this
N-D geometry contract. Their module/function docstrings are the local reference;
the old gap-filling basis document is not present in this repository.

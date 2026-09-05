# API Reference

This reference documents the publication-focused public API surface
(convolution + statistics).

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
- For Cartesian data, `kernel_type` accepts a string or a sequence with one
  supported kernel name per array axis, as documented for `smooth_cartesian`.
  Polar data accepts a string or `(kernel_azimuth, kernel_range)` pair, as
  documented for `smooth_polar`. The same kernels are used in prefill and
  final smoothing.
- If input contains NaNs, smoothing automatically runs a NaN-safe prefill step,
  applies spectral smoothing on the finite field, then restores the original NaN mask.
- For polar input, `az_res_deg` must be a finite, positive real scalar. This
  validation also applies when the input is entirely NaN.
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

#### `smooth_polar(data, width_pixels, az_res_deg=1.0, az_boundary='reflective', range_boundary='reflective', kernel_type='gaussian')`
Apply smoothing to 2D polar data (`n_azimuth`, `n_range`) with adaptive azimuth kernels.

- `width_pixels` (`float` or `(width_azimuth, width_range)`): A scalar applies
  equal nominal widths; a pair specifies independent widths. Values must be
  finite and positive. At range index `r` (starting at 1), azimuth width is
  `width_azimuth / (r * dtheta)` beams, with `dtheta` in radians; range width is
  in range gates. Gaussian `sigma = effective_width / sqrt(12)` on either axis.
- `az_res_deg`: A finite, positive real scalar in degrees. Python and NumPy
  numeric scalars, including zero-dimensional numeric arrays, are accepted.
  Zero, negative, nonfinite, complex, boolean, string, and nonscalar inputs
  raise `ValueError`. The spacing must remain positive when converted to radians.
- `az_boundary='reflective'`: DCT-based reflective boundary.
- `az_boundary='periodic'`: real FFT periodic boundary (0/360 wrap).
- `range_boundary`: currently `'reflective'`.
- `kernel_type` (`str` or sequence of `str`): `'boxcar'`, `'boxcar_discrete'`,
  or `'gaussian'`. A string applies to both dimensions; a 1-D list, tuple, or
  NumPy array of length 2 selects `(kernel_azimuth, kernel_range)`. Kernel and
  width pairs use the same order and can be specified independently.
- All three kernels support reflective and periodic azimuth. Discrete boxcars
  round the effective width to an integer, clamp it to at least 1, and increment
  even sizes. In azimuth this conversion occurs separately at each range gate.
  Periodic windows use circular averaging, including windows larger than a sweep.
- Invalid kernel pair length, dimensionality, non-string entries, or unknown
  names raise `ValueError`, using the same validation as Cartesian smoothing.
- Filtering applies the adaptive azimuth operator first, then the range
  operator. Reversing these operations generally changes the result because
  the azimuth width varies with range. The polar backend accepts 2D data.
- Input should be finite; `dct_mean` and `dct_smooth` handle NaNs.

For data in **(azimuth, range)** order:

```python
import numpy as np
from dct_toolkit import dct_smooth

radar_data = np.random.default_rng(42).standard_normal((360, 100))
smoothed = dct_smooth(
    radar_data,
    coordinates="polar",
    width=(5.0, 3.0),
    kernel_type=("boxcar", "gaussian"),
    az_res_deg=1.0,
    az_boundary="periodic",
)
```

`compute_polar_transfer_functions` and its compatibility alias
`compute_polar_transfer_functions_v2` accept the same kernel pair. Their return
values remain the adaptive azimuth transfer array and the range transfer vector.

**Behavior correction:** periodic azimuth with `'boxcar_discrete'` now uses the
requested discrete boxcar. Earlier versions substituted a Gaussian in this
branch. Existing single-string Gaussian and analytical boxcar calls retain
their behavior.

---

## Statistical Operations

### `dct_toolkit.stats`

All statistical functions use normalized convolution and support NaN-containing inputs.

For Cartesian coordinates, all functions below accept `kernel_type` through
`**kwargs` as a single string or a length-`data.ndim` sequence (`mask.ndim` for
`dct_count`), using the same axis order and width conventions as
`smooth_cartesian`. This includes `dct_prefill`. The default is `'gaussian'`.
For polar coordinates, `kernel_type` accepts a string or a pair in
`(azimuth, range)` order, with the width conventions of `smooth_polar`.
All polar statistics require the same finite, positive scalar `az_res_deg` as
`smooth_polar`; `dct_count` uses the same validated spacing for its window area.
Validation also applies to all-NaN input and prefill calls with no targets.

Mixed-kernel means apply the complete smoothing operator to both the zero-filled
data and validity mask before dividing. Variance uses that same kernel for both
moments; prefill and density thresholds also use the requested kernels.
For polar data this includes the same azimuth-then-range order in every pass.
`dct_count` retains the convention `clip(smooth(mask), 0, 1) * prod(widths)` for
Cartesian data; this is an effective window count, including for Gaussian or
mixed kernels, rather than a literal number of neighbors in a finite kernel.
Polar counts retain the corresponding range-dependent factor
`width_range * width_azimuth / (r * dtheta)`.

For heavily gapped fields, statistical functions include internal stability
fallbacks to avoid unstable normalized-convolution ratios in poorly supported
regions.

#### `dct_count(mask, width, coordinates='cartesian', **kwargs)`
Compute effective local sample count.

- `width` (`float` or sequence): Scalar = equal widths, sequence = per-axis widths.
- Density is clipped to `[0, 1]` to keep counts physically valid.
- `restore_input_nan=True` by default masks output where input `mask` is False.

#### `dct_mean(data, width, coordinates='cartesian', mask=None, min_effective_density=None, **kwargs)`
Compute robust local mean:

`mu = smooth(data * mask) / smooth(mask)`

- Returns floating-point output.
- `width` (`float` or sequence): Scalar = equal widths, sequence = per-axis widths.
- If `mask` is provided, it must match `data.shape`.
- In low-support regions, an internal prefill-and-smooth fallback is used to
  keep results finite and stable when support exists. The denominator threshold
  is `1e-3` (internal constant `_MEAN_DENOMINATOR_FLOOR`).
- `restore_input_nan=True` by default masks output where input support is invalid.
- `min_effective_density=None` (default) keeps the legacy behavior. When set
  (e.g. `0.35`), the final output is set to NaN wherever the local
  valid-sample density `smooth(mask)` is below this threshold. The threshold
  is dimensionless (in `[0, 1]`) and works uniformly for Cartesian and polar
  coordinates. This is the same density signal that `dct_count / area`
  represents internally.

#### `dct_prefill(data, width, coordinates='cartesian', fill_mask=None, max_iter=3, min_effective_density=0.35, **kwargs)`
Fill gaps using iterative normalized convolution based on `dct_mean`.

- `width` (`float` or sequence): Scalar = equal widths, sequence = per-axis widths.
- `fill_mask` uses `True = fill this position`.
- If `fill_mask` is None, NaN positions are filled.
- Preserves non-target values exactly.
- Intended as a pre-processing step before full-field smoothing.
- `max_iter=3` by default for predictable runtime.
- `max_iter=None` runs until convergence with safety cap `20`; fixed integer values
  stop early once no NaNs remain.
- Any unresolved targets after iterations are filled with nearest-neighbor fallback
  to guarantee finite outputs when at least one finite value exists.
- Polar nearest-neighbor propagation tries range first, then azimuth. With
  `az_boundary='periodic'`, azimuth neighbors wrap across the first/last beam,
  including in the global fallback. Range distances remain nonperiodic.
  During the azimuth pass, equidistant neighbors prefer the preceding beam
  around the circle. Reflective azimuth uses ordinary index distances.
- `min_effective_density=0.35` (default) gates per-iteration mean estimates by
  the local valid-sample density `smooth(mask)`. Cells whose local density is
  below the threshold are left as NaN for that iteration and may be filled in
  a later iteration as the valid mask grows. This stabilizes the iterative
  prefill in low-support regions and uses the same density signal as
  `dct_count / area`. Set to `None` or a non-positive value to disable the
  gate and reproduce the legacy un-gated behavior.

#### `dct_variance(data, width, coordinates='cartesian', mask=None, **kwargs)`
Compute robust local variance:

`Var = E[X^2] - (E[X])^2`

- `restore_input_nan=True` by default masks output where input support is invalid.
- `width` (`float` or sequence): Scalar = equal widths, sequence = per-axis widths.

#### `dct_std(data, width, coordinates='cartesian', mask=None, **kwargs)`
Compute robust local standard deviation:

`Std = sqrt(Var)`

- `restore_input_nan=True` by default masks output where input support is invalid.
- `width` (`float` or sequence): Scalar = equal widths, sequence = per-axis widths.

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
            | (N-D Cartesian smoother)  |             | (2D polar smoother)       |
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


    +--------------------+    uses smooth_* on mask and scales by area
    | dct_count          |-----------------------------------------------> output count
    +--------------------+

    +--------------------+    normalized convolution: smooth(x*mask)/smooth(mask)
    | dct_mean           |-----------------------------------------------> output mean
    +---------+----------+
              |
              +--> internal low-support fallback may call dct_prefill

    +--------------------+    calls dct_mean twice: E[x] and E[x^2]
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
- `smooth_polar`: Low-level smoothing for 2D polar arrays with adaptive azimuth width.
- `dct_smooth`: Top-level "just smooth this" wrapper (auto-pre-fills NaNs, then restores NaN mask).
- `dct_mean`: NaN-robust local mean via normalized convolution.
- `dct_count`: Effective local sample count from a validity mask.
- `dct_prefill`: Iterative normalized-convolution gap fill (often used before full-field smoothing).
- `dct_variance`: NaN-robust local variance from normalized-convolution moments.
- `dct_std`: NaN-robust local standard deviation (`sqrt(dct_variance)`).

---

## Scope Note

Gap-filling methods are intentionally excluded from this public API reference for the
initial stats-first release track.

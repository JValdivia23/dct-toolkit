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

### `dct_toolkit.dct_smooth(data, width, coordinates='cartesian', **kwargs)`
Top-level convenience wrapper for Cartesian and polar smoothing.

- `coordinates='cartesian'`: calls `smooth_cartesian(data, width, **kwargs)`.
- `coordinates='polar'`: calls `smooth_polar(data, width_pixels=width, **kwargs)`.
- Default kernel is `'gaussian'` unless `kernel_type` is explicitly provided.

### `dct_toolkit.cartesian`

#### `smooth_cartesian(data, width, kernel_type='gaussian')`
Apply separable DCT smoothing to N-D Cartesian data.

- `data` (`np.ndarray`): Any-dimensional array.
- `width` (float): Isotropic smoothing width.
- `kernel_type` (str): `'boxcar'`, `'boxcar_discrete'`, `'gaussian'`.

### `dct_toolkit.polar`

#### `smooth_polar(data, width_pixels, az_res_deg=1.0, az_boundary='reflective', range_boundary='reflective', kernel_type='gaussian')`
Apply smoothing to 2D polar data (`n_azimuth`, `n_range`) with adaptive azimuth kernels.

- `az_boundary='reflective'`: DCT-based reflective boundary.
- `az_boundary='periodic'`: real FFT periodic boundary (0/360 wrap).
- `range_boundary`: currently `'reflective'`.

---

## Statistical Operations

### `dct_toolkit.stats`

All statistical functions use normalized convolution and support NaN-containing inputs.

#### `dct_count(mask, width, coordinates='cartesian', **kwargs)`
Compute effective local sample count.

#### `dct_mean(data, width, coordinates='cartesian', mask=None, **kwargs)`
Compute robust local mean:

`mu = smooth(data * mask) / smooth(mask)`

- Returns floating-point output.
- If `mask` is provided, it must match `data.shape`.

#### `dct_prefill(data, width, coordinates='cartesian', fill_mask=None, max_iter=3, **kwargs)`
Fill gaps using iterative normalized convolution based on `dct_mean`.

- `fill_mask` uses `True = fill this position`.
- If `fill_mask` is None, NaN positions are filled.
- Preserves non-target values exactly.
- Intended as a pre-processing step before full-field smoothing.

#### `dct_variance(data, width, coordinates='cartesian', mask=None, **kwargs)`
Compute robust local variance:

`Var = E[X^2] - (E[X])^2`

#### `dct_std(data, width, coordinates='cartesian', mask=None, **kwargs)`
Compute robust local standard deviation:

`Std = sqrt(Var)`

---

## Scope Note

Gap-filling methods are intentionally excluded from this public API reference for the
initial stats-first release track.

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
- If input contains NaNs, smoothing automatically runs a NaN-safe prefill step,
  applies spectral smoothing on the finite field, then restores the original NaN mask.
- `prefill_max_iter=3` by default for predictable runtime.
- `prefill_max_iter=None` means iterate prefill until convergence (or safety cap of 20).
- Legacy alias: passing `max_iter=...` in kwargs is mapped to `prefill_max_iter`.

### `dct_toolkit.cartesian`

#### `smooth_cartesian(data, width, kernel_type='gaussian')`
Apply separable DCT smoothing to N-D Cartesian data.

- `data` (`np.ndarray`): Any-dimensional array.
- `width` (`float` or sequence): Scalar = isotropic. Sequence length must be `data.ndim`.
- `kernel_type` (str): `'boxcar'`, `'boxcar_discrete'`, `'gaussian'`.

### `dct_toolkit.polar`

#### `smooth_polar(data, width_pixels, az_res_deg=1.0, az_boundary='reflective', range_boundary='reflective', kernel_type='gaussian')`
Apply smoothing to 2D polar data (`n_azimuth`, `n_range`) with adaptive azimuth kernels.

- `width_pixels` (`float` or `(width_azimuth, width_range)`): Scalar keeps isotropic
  behavior; tuple enables anisotropic polar smoothing.
- `az_boundary='reflective'`: DCT-based reflective boundary.
- `az_boundary='periodic'`: real FFT periodic boundary (0/360 wrap).
- `range_boundary`: currently `'reflective'`.

---

## Statistical Operations

### `dct_toolkit.stats`

All statistical functions use normalized convolution and support NaN-containing inputs.

For heavily gapped fields, statistical functions include internal stability
fallbacks to avoid unstable normalized-convolution ratios in poorly supported
regions.

#### `dct_count(mask, width, coordinates='cartesian', **kwargs)`
Compute effective local sample count.

- `width` (`float` or sequence): Scalar = isotropic, sequence = anisotropic.
- Density is clipped to `[0, 1]` to keep counts physically valid.
- `restore_input_nan=True` by default masks output where input `mask` is False.

#### `dct_mean(data, width, coordinates='cartesian', mask=None, **kwargs)`
Compute robust local mean:

`mu = smooth(data * mask) / smooth(mask)`

- Returns floating-point output.
- `width` (`float` or sequence): Scalar = isotropic, sequence = anisotropic.
- If `mask` is provided, it must match `data.shape`.
- In low-support regions, an internal prefill-and-smooth fallback is used to
  keep results finite and stable when support exists.
- `restore_input_nan=True` by default masks output where input support is invalid.

#### `dct_prefill(data, width, coordinates='cartesian', fill_mask=None, max_iter=3, **kwargs)`
Fill gaps using iterative normalized convolution based on `dct_mean`.

- `width` (`float` or sequence): Scalar = isotropic, sequence = anisotropic.
- `fill_mask` uses `True = fill this position`.
- If `fill_mask` is None, NaN positions are filled.
- Preserves non-target values exactly.
- Intended as a pre-processing step before full-field smoothing.
- `max_iter=3` by default for predictable runtime.
- `max_iter=None` runs until convergence with safety cap `20`; fixed integer values
  stop early once no NaNs remain.
- Any unresolved targets after iterations are filled with nearest-neighbor fallback
  to guarantee finite outputs when at least one finite value exists.

#### `dct_variance(data, width, coordinates='cartesian', mask=None, **kwargs)`
Compute robust local variance:

`Var = E[X^2] - (E[X])^2`

- `restore_input_nan=True` by default masks output where input support is invalid.
- `width` (`float` or sequence): Scalar = isotropic, sequence = anisotropic.

#### `dct_std(data, width, coordinates='cartesian', mask=None, **kwargs)`
Compute robust local standard deviation:

`Std = sqrt(Var)`

- `restore_input_nan=True` by default masks output where input support is invalid.
- `width` (`float` or sequence): Scalar = isotropic, sequence = anisotropic.

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

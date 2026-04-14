# DCT-Toolkit: Convolutional Statistics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`dct_toolkit` provides DCT-based smoothing and normalized-convolution statistics for
data with gaps. The current publication-prep track is intentionally focused on the
convolution/statistics surface.

## Current Scope (Stats-First)

- DCT-based smoothing for 1D/N-D Cartesian data.
- Polar smoothing with adaptive azimuth kernels.
- Reflective boundaries via DCT and periodic azimuth boundaries via real FFT.
- Robust local statistics (`dct_mean`, `dct_variance`, `dct_std`, `dct_count`) with NaN support.

## Installation

### pip

```bash
pip install dct-toolkit
```

### conda (pending conda-forge approval)

Conda-forge packaging has been submitted (PR under review). This option would be available soon:

```bash
conda install dct-toolkit
```

## Quick Start

### 1) Robust Mean with Missing Data

```python
import numpy as np
import dct_toolkit as dct

data = np.sin(np.linspace(0, 10, 200))
data[70:110] = np.nan

mu = dct.dct_mean(data, width=10.0)
```

### 1b) Prefill Before Full-Field Smoothing

```python
smoothed = dct.dct_smooth(prefilled, width=10.0)
```
This function will call `dct.dct_prefill()` internally. We need to fill NaNs as requirement for DCT or FFT operations by default.

### 2) Polar Smoothing with Periodic Azimuth

```python
smoothed = dct.dct_smooth(
    radar_data,
    width=5.0,
    coordinates="polar",
    az_boundary="periodic",
    az_res_deg=1.0,
)
```

### 3) Local Variability Estimates

```python
var = dct.dct_variance(data, width=10.0)
std = dct.dct_std(data, width=10.0)
```

## Quick Cheat Sheet

| Function | Description |
|----------|-------------|
| `get_dct_transfer_function` | Build 1D spectral kernel `H[k]` from width and kernel type. |
| `dct_convolve_1d` | Apply one 1D DCT convolution along one axis using a precomputed `H`. |
| `smooth_cartesian` | Low-level separable smoothing for N-D Cartesian arrays. |
| `smooth_polar` | Low-level smoothing for 2D polar arrays with adaptive azimuth width. |
| `dct_smooth` | Top-level "just smooth this" wrapper (auto-pre-fills NaNs, then restores NaN mask). |
| `dct_mean` | NaN-robust local mean via normalized convolution. |
| `dct_count` | Effective local sample count from a validity mask. |
| `dct_prefill` | Iterative normalized-convolution gap fill (often used before full-field smoothing). |
| `dct_variance` | NaN-robust local variance from normalized-convolution moments. |
| `dct_std` | NaN-robust local standard deviation (`sqrt(dct_variance)`). |

## Pre-filling (`dct_prefill`)

`dct_prefill` provides an **iterative normalized-convolution gap fill** based on `dct_mean`:

- Fills gaps (NaNs or a specified mask) using iterative local averaging.
- Preserves non-target values exactly.
- Intended as a **pre-processing step** before full-field smoothing.
- `max_iter=3` by default for predictable runtime; use `max_iter=None` to iterate until convergence (all NaNs filled). By default, any remaining NaNs are filled with the nearest valid data.

See the Quick Start section above for usage examples.

## Documentation

- `docs/MATHEMATICAL_BASIS.md` — Mathematical foundations and theory
- `docs/API_REFERENCE.md` — Complete API documentation

## License

MIT. See `LICENSE`.

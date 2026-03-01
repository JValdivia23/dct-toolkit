# API Reference

## Core Primitives

### `dct_toolkit.core`

#### `get_dct_transfer_function(n, kernel_type, width)`
Generate the 1D DCT Transfer Function H[k].

- **n** (int): Number of points.
- **kernel_type** (str):
  - `'boxcar'`: Continuous Sinc function (analytical).
  - `'boxcar_discrete'`: Sum of cosines (discrete approximation).
  - `'gaussian'`: $\exp(-\omega^2 \sigma^2/2)$ where $\sigma = \text{width}/\sqrt{12}$.
- **width** (float): Kernel width in pixels.

#### `dct_convolve_1d(data, H, axis=-1)`
Apply 1D convolution using DCT-II (Ortho).

- **data** (np.ndarray): Input data.
- **H** (np.ndarray): Transfer function matching `data.shape[axis]`.
- **axis** (int): Axis to smooth along.

---

## High-Level Smoothing

### `dct_toolkit.cartesian`

#### `smooth_cartesian(data, width, kernel_type='boxcar')`
Apply separable smoothing to N-D Cartesian data.

- **data**: Input array (any dimension).
- **width**: Isotropic smoothing width.
- **kernel_type**: Kernel type.

### `dct_toolkit.polar`

#### `smooth_polar(data, width_pixels, az_res_deg=1.0, az_boundary='reflective', ...)`
Apply smoothing to 2D polar data (Azimuth x Range).

- **data**: Shape must be `(n_azimuth, n_range)`.
- **width_pixels**: Target physical width in pixel units (at reference range).
- **az_res_deg**: Azimuth resolution in degrees.
- **az_boundary**:
  - `'reflective'`: Standard DCT (mirror symmetry).
  - `'periodic'`: Real FFT (circular convolution 0-360).

---

## Statistical Operations

### `dct_toolkit.stats`

All functions support NaN handling via Normalized Convolution.

#### `dct_count(mask, width, coordinates='cartesian', ...)`
Compute effective sample size (density * window area).

#### `dct_mean(data, width, coordinates='cartesian', mask=None, ...)`
Compute robust local mean.
$$ \mu = \frac{\text{smooth}(data \cdot mask)}{\text{smooth}(mask)} $$

#### `dct_variance(data, width, ...)`
Compute robust local variance.
$$ \text{Var} = E[X^2] - (E[X])^2 $$

#### `dct_std(data, width, ...)`
Compute robust local standard deviation ($\sqrt{\text{Var}}$).

---

## Gap Filling

### `dct_toolkit.gap_filling`

#### `dct_inpaint(data, width, coordinates='cartesian', order=2, max_iter=100, tol=1e-5, init='linear', smooth_output=False, **kwargs)`
Fill gaps via DCT-domain Penalised Least Squares (spectral inpainting).

Minimises the functional **J(u) = ||W(y - u)||^2 + lambda ||D^p u||^2** where
W = diag(mask), D^p is the p-th order difference operator, and lambda is derived
automatically from `width`.  With the default `order=2` (bi-harmonic penalty),
this is equivalent to thin-plate spline interpolation: it preserves curvature
across gaps rather than flattening them.

- **data** (np.ndarray): Input data with NaNs marking gaps. 1-D or 2-D.
- **width** (float): Correlation length scale in grid points (must be > 0). Controls smoothness of the reconstruction: larger width = smoother fill across gaps. Internally mapped to Tikhonov parameter lambda via `lambda = (width^2 / 24)^order`.
- **coordinates** (str): `'cartesian'` or `'polar'`. Default `'cartesian'`.
- **order** (int): Order of the smoothness penalty (must be >= 1). Default `2`.
  - `1` — gradient penalty (Laplace equation, membrane).
  - `2` — curvature penalty (bi-harmonic, thin-plate spline). **Recommended.**
  - `3` — third-order (very smooth).
- **max_iter** (int): Maximum number of iterations. Default `100`.
- **tol** (float): Convergence tolerance (relative L2 change of gap pixels). Default `1e-5`.
- **init** (str): Initialisation strategy for gap values:
  - `'linear'` (default): Axis-wise linear interpolation.
  - `'zeros'`: Fill gaps with zero (useful for zero-mean fields).
- **smooth_output** (bool): If True, the final result is the spectrally-smoothed field (valid + gap). If False, valid data is preserved exactly. Default `False`.
- **\*\*kwargs**: Keyword arguments for polar mode:
  - `az_res_deg` (float) — azimuth resolution in degrees. Enables range-adaptive azimuth penalty: at range index *j*, the effective azimuth penalty scales as `1 / (j * dtheta)^(2p)`, mirroring the adaptive kernel widths in `smooth_polar`.
  - `az_boundary` (str) — `'reflective'` or `'periodic'`. Default `'reflective'`.

**Returns**: `np.ndarray` — Data with gaps filled. Original valid values preserved exactly when `smooth_output=False`.

**Raises**: `ValueError` — If `width <= 0`, `order < 1`, `coordinates` or `az_boundary` is invalid, or data is not 1-D/2-D.

**Example** (1-D):
```python
import numpy as np
from dct_toolkit import dct_inpaint

x = np.linspace(0, 2 * np.pi, 200)
data = np.sin(x)
data[80:120] = np.nan
filled = dct_inpaint(data, width=10.0)
```

**Example** (2-D polar with wrapping):
```python
filled = dct_inpaint(
    radar_data,                  # (n_az, n_range) with NaN gaps
    width=50.0,
    coordinates='polar',
    az_res_deg=1.0,              # adaptive azimuth penalty
    az_boundary='periodic',      # correct 0/360 wrapping
)
```

See [Gap Filling Basis](GAP_FILLING_BASIS.md) for the mathematical foundation.

#### `iterative_gap_fill(data, width, coordinates='cartesian', max_iter=50, tol=1e-4, init='linear', multiscale=None, smooth_output=False, **kwargs)`
Fill gaps (NaN values) using iterative DCT-based diffusion.

Missing values are first initialized, then iteratively refined by replacing
gap values with the DCT-smoothed trend. Valid data is preserved exactly
(unless `smooth_output=True`).

- **data** (np.ndarray): Input data with NaNs marking gaps. 1-D or 2-D.
- **width** (float): Smoothing width. Controls "stiffness" of the interpolated surface — larger values produce smoother fills. For best results on contiguous holes, use a width that is a significant fraction (~25%) of the hole diameter.
- **coordinates** (str): `'cartesian'` or `'polar'`. Default `'cartesian'`.
- **max_iter** (int): Maximum number of diffusion iterations. Default `50`.
- **tol** (float): Convergence tolerance (relative change in L2 norm). Default `1e-4`.
- **init** (str): Initialization strategy for gap values:
  - `'linear'` (default): Axis-wise linear interpolation. Preserves spatial gradients across holes. Recommended for contiguous holes larger than a few times *width*.
  - `'multiscale'`: Coarse-to-fine DCT cascade. Starts at `max(data_shape)/4` and halves to *width*.
  - `'dct'`: Single-pass Normalized Convolution at *width*.
- **multiscale** (bool or None): **Deprecated.** Use `init='multiscale'` or `init='dct'` instead. Emits `FutureWarning`.
- **smooth_output** (bool): If True, the final iteration smooths all values (valid + gap), producing a combined gap-fill + noise-reduction result. Default `False`.
- **\*\*kwargs**: Additional arguments passed to the smoothing function (e.g., `kernel_type='gaussian'`, `az_res_deg=0.5`).

**Returns**: `np.ndarray` — Data with gaps filled. Original valid values preserved exactly when `smooth_output=False`.

**Example**:
```python
import numpy as np
from dct_toolkit import iterative_gap_fill

data = np.random.randn(100, 100)
data[40:60, 40:60] = np.nan  # 20x20 hole
filled = iterative_gap_fill(data, width=10.0, max_iter=10)
```

See [Gap Filling Basis](GAP_FILLING_BASIS.md) for the mathematical foundation.

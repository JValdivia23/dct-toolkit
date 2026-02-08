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

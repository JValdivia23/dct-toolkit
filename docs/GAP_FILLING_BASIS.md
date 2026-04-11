# Mathematical Basis: Gap Filling

## 1. Problem Statement

Given a signal $y$ observed only on a subset of points $\Omega$ (where mask $M=1$), we want to estimate the values on the complement $\Omega^c$ (where $M=0$).

$$ y_{obs} = M \cdot y_{true} + \epsilon $$

We assume the true signal $y_{true}$ is smooth in some sense (band-limited or low-pass).

---

## 2. Method 1: Iterative Normalised Convolution (`iterative_gap_fill`)

Unlike Tikhonov regularisation (which requires inverting large matrices) or DCT-PLS (which requires solving a penalised least squares system), this approach is **constructive**.

### Step 1: Initialisation
We initialise the estimate $\hat{y}^{(0)}$ using one of three strategies (controlled by the `init` parameter):

- **`'linear'` (default)**: Axis-wise linear interpolation — first row-wise (`np.interp`), then column-wise for remaining NaNs, with a global-mean fallback.  This preserves spatial gradients across holes from the very first iterate and is equivalent to `scipy.interpolate.griddata` (linear) for most hole geometries.
- **`'multiscale'`**: Coarse-to-fine DCT cascade — starts at `max(data_shape)/4` and halves down to the target width.
- **`'dct'`**: Single-pass Normalised Convolution at the target width.

$$ \hat{y}^{(0)} = \text{LinearInterp}(y_{obs}) \quad (\text{default}) $$

### Step 2: Iteration
At each step $k$, we compute a "smooth trend" of the current estimate using Normalised Convolution. Importantly, we treat **all** points in the current estimate as valid sources of information, but we "reset" the known valid points to their ground truth after smoothing.

$$ \text{Trend}^{(k)} = \text{Smooth}(\hat{y}^{(k)}) $$

$$ \hat{y}^{(k+1)} = M \cdot y_{obs} + (1-M) \cdot \text{Trend}^{(k)} $$

### Step 3: Convergence
The iteration repeats until $||\hat{y}^{(k+1)} - \hat{y}^{(k)}|| < \delta$.

### Why it works
This process is equivalent to solving the heat equation (diffusion) with Dirichlet boundary conditions (fixed values at valid points):
- The smoothing operation acts as a diffusion operator.
- Resetting the valid points acts as the boundary condition.
- The gaps "fill up" via diffusion from the valid boundaries.

### Limitation
Because the smoother is a low-pass filter (Gaussian or boxcar), the solution corresponds to solving the **Laplace equation** $\Delta u = 0$ in gaps.  This produces the smoothest possible surface — a *membrane* — but tends to **flatten curvature** across large gaps.

---

## 3. Method 2: DCT Spectral Inpainting (`dct_inpaint`)

### 3.1 The Optimisation Problem

`dct_inpaint` solves the problem properly as a **Penalised Least Squares** (PLS) optimisation in the spectral domain (Garcia, 2010):

$$
\hat{u} = \arg\min_u \left\{ \| W \cdot (y - u) \|^2 + \lambda \| \Delta^p u \|^2 \right\}
$$

where:
- $W = \text{diag}(M)$ — data fidelity only at observed points
- $\Delta^p$ — p-th order finite difference operator
- $\lambda$ — regularisation parameter (derived from user's `width`)
- $p$ — order of the penalty (default $p=2$, bi-harmonic)

The first term enforces fidelity to observed data.  The second term penalises roughness in the reconstruction.

### 3.2 DCT-Domain Diagonalisation

The key insight is that the discrete Laplacian $\Delta$ **diagonalises** under the DCT-II (for reflective BC) and the DFT (for periodic BC).  The eigenvalues are:

**DCT-II (reflective BC):**
$$E_p[k] = \left(2 - 2\cos\frac{\pi k}{N}\right)^p, \quad k = 0, \ldots, N-1$$

**DFT (periodic BC, RFFT half-spectrum):**
$$E_p[k] = \left(2 - 2\cos\frac{2\pi k}{N}\right)^p, \quad k = 0, \ldots, N/2$$

**2-D (separable):**
$$E_p[k_1, k_2] = E_p^{(1)}[k_1] + E_p^{(2)}[k_2]$$

This means the entire optimisation can be solved iteratively in $O(N \log N)$ per iteration.

### 3.3 Iterative Algorithm

The DCT-PLS iteration is:

1. **Compose** the field: $z = M \cdot y + (1-M) \cdot \hat{u}^{(k)}$
2. **Forward transform**: $Z = \mathcal{T}\{z\}$ (DCT or FFT)
3. **Spectral filter**: $\hat{U}^{(k+1)} = \frac{Z}{1 + \lambda \cdot E_p}$
4. **Inverse transform**: $\hat{u}^{(k+1)} = \mathcal{T}^{-1}\{\hat{U}^{(k+1)}\}$
5. **Convergence check** on gap pixels.

The spectral filter $\Gamma[k] = 1 / (1 + \lambda \cdot E_p[k])$ is the key:
- At DC ($k=0$): $E_p = 0$, so $\Gamma = 1$ — the mean is unpenalised.
- At high frequencies: $E_p \gg 1$, so $\Gamma \approx 0$ — high-frequency noise is suppressed.
- At intermediate frequencies: smooth roll-off determined by $\lambda$.

### 3.4 Width-to-Lambda Bridge

Rather than exposing the raw $\lambda$ parameter (hard to tune), we derive it from the user's intuitive `width` (correlation length in grid points).  The mapping matches the half-power frequency of a Gaussian kernel with $\sigma = \text{width}/\sqrt{12}$:

$$
\lambda = \left(\frac{\sigma^2}{2}\right)^p = \left(\frac{w^2}{24}\right)^p
$$

This gives users the same intuitive control as the rest of the toolkit:
- **Small width** → small $\lambda$ → sharp reconstruction
- **Large width** → large $\lambda$ → smooth reconstruction

### 3.5 Why Order p=2 Preserves Curvature

The choice of penalty order $p$ determines what physical quantity is minimised in gaps:

| Order | Penalty | PDE equivalent | Behaviour |
|-------|---------|---------------|-----------|
| $p=1$ | $\|\nabla u\|^2$ (gradient) | $\Delta u = 0$ (Laplace) | Membrane: flat in gaps |
| $p=2$ | $\|\nabla^2 u\|^2$ (curvature) | $\Delta^2 u = 0$ (Bi-harmonic) | Thin-plate spline: curvature preserved |
| $p=3$ | $\|\nabla^3 u\|^2$ | $\Delta^3 u = 0$ | Very smooth, may overshoot |

With $p=2$, the algorithm minimises the total bending energy of the surface, naturally preserving curved features (ridges, valleys, sinusoidal patterns) across gaps.

### 3.6 Initialisation: Linear Interpolation as Baseline

The algorithm initialises gap values with **fast axis-wise linear interpolation** before the DCT-PLS iterations begin.  This is important because:

1. The DCT/FFT forward transform requires a fully-populated array (no NaN).
2. Linear interpolation provides a reasonable initial guess that preserves gradients.
3. Starting close to the solution reduces the number of iterations needed.

The iterative DCT-PLS then refines this baseline, replacing the piecewise-linear fill with the spectrally-optimal curvature-preserving solution.

### 3.7 Polar Coordinate Support

For polar data (azimuth × range):
- **Azimuth** (axis 0): periodic BC → RFFT with DFT eigenvalues.
- **Range** (axis 1): reflective BC → DCT with DCT eigenvalues.
- The 2D eigenvalue tensor is the outer sum of the 1D eigenvalue vectors.
- This handles the 0°/360° azimuth wrap-around naturally.

---

## 4. Comparison

| Feature | `iterative_gap_fill` (v3) | `dct_inpaint` (v4) |
|---------|--------------------------|---------------------|
| **Formulation** | Heuristic (smooth + reset) | Optimal (penalised LS) |
| **PDE** | $\Delta u = 0$ (membrane) | $\Delta^2 u = 0$ (thin-plate) |
| **Curvature** | Flattened in gaps | Preserved |
| **Convergence** | Slow for large gaps | Fast (spectral acceleration) |
| **Parameters** | width, init, kernel_type | width, order |
| **Speed** | $k \times O(N \log N)$ | $k \times O(N \log N)$ (fewer $k$) |

### Benchmark Results (720×1000 polar grid, circular hole r=100)

| Method | Wrapping hole MAE | Non-wrapping hole MAE | Speed |
|--------|-------------------|-----------------------|-------|
| griddata (baseline) | 1.236 | 1.134 | 7.2s |
| v3 best | 2.338 | 1.117 | 0.7s |
| **v4 (w=50, p=2)** | **0.810** | **0.406** | 0.8s |

v4 is **2.9× more accurate** than v3 and **34% more accurate than griddata** on the wrapping hole, at 9× the speed of griddata.

---

## 5. References

- Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values. *Computational Statistics & Data Analysis*, 54(4), 1167–1178.
- Knutsson, H. & Westin, C.-F. (1993). Normalised Convolution.
- Bertalmio, M. et al. (2000). Image inpainting. *SIGGRAPH*.

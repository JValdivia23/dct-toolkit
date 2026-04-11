# Exp v4 Plan: DCT Spectral Inpainting (`dct_inpaint`)

## 1. Goal

Implement a mathematically rigorous gap-filling method based on **DCT-domain
Penalized Least Squares** (DCT-PLS).  This is classical spectral inpainting —
analogous to diffusion-model denoising but without neural networks: pure,
explainable, solid mathematics.

The method iteratively reconstructs missing data by solving a penalized
optimization problem entirely in the spectral (DCT/FFT) domain.

**Key improvements over v3 (`iterative_gap_fill`):**

| Aspect | v3 (iterative diffusion) | v4 (`dct_inpaint`) |
|--------|--------------------------|---------------------|
| PDE equivalent | Laplace Δu = 0 (membrane) | Bi-harmonic Δ²u = 0 (thin-plate) |
| Curvature in gaps | Flattened | Preserved |
| Convergence | Slow for large gaps | Fast (spectral acceleration) |
| Formulation | Heuristic (smooth + reset) | Optimal (penalized LS) |
| Parameters | width + init + kernel_type | width + order (cleaner) |

---

## 2. Mathematical Basis

### 2.1 The Optimization Problem

Given data $y$ observed on mask $M$ (valid=1, gap=0), find $\hat{u}$ that
minimizes:

$$
J(\hat{u}) = \| W \cdot (y - \hat{u}) \|^2 + \lambda \| \Delta^p \hat{u} \|^2
$$

where:
- $W = \text{diag}(M)$ — data fidelity only at observed points
- $\Delta^p$ — p-th order finite difference operator
- $\lambda$ — smoothness penalty (derived from user's `width` parameter)
- $p = 2$ (default) — bi-harmonic / curvature penalty

### 2.2 DCT-Domain Solution (Garcia, 2010)

In the DCT domain, the finite difference operator $\Delta^p$ diagonalizes.
Its eigenvalues are:

**1-D:**
$$E_p[k] = \left(2 - 2\cos\frac{\pi k}{N}\right)^p, \quad k = 0, \ldots, N-1$$

**2-D (Cartesian, DCT-II reflective BC):**
$$E_p[k_1, k_2] = \left(2 - 2\cos\frac{\pi k_1}{N_1}\right)^p
                  + \left(2 - 2\cos\frac{\pi k_2}{N_2}\right)^p$$

**2-D (Polar, periodic azimuth + reflective range):**
$$E_p[k_{az}, k_r] = \left(2 - 2\cos\frac{2\pi k_{az}}{N_{az}}\right)^p
                    + \left(2 - 2\cos\frac{\pi k_r}{N_r}\right)^p$$

The iterative DCT-PLS algorithm is:

$$
\hat{U}^{(k+1)} = \frac{\text{DCT}\left\{ M \cdot y + (1-M) \cdot \hat{u}^{(k)} \right\}}{1 + \lambda \cdot E_p}
$$
$$
\hat{u}^{(k+1)} = \text{IDCT}\left\{ \hat{U}^{(k+1)} \right\}
$$

Each iteration:
1. Compose the field: observed data where valid, current estimate where gap
2. Forward transform (DCT or FFT depending on BC)
3. Apply spectral filter $\Gamma[k] = 1 / (1 + \lambda \cdot E_p[k])$
4. Inverse transform → new estimate

### 2.3 Width-to-Lambda Bridge

The user specifies `width` (physical correlation length).  We derive the
optimal $\lambda$ by matching the spectral half-power point of a Gaussian
kernel with $\sigma = \text{width} / \sqrt{12}$:

$$
\lambda = \left(\frac{\sigma^2}{2}\right)^p = \left(\frac{w^2}{24}\right)^p
$$

where $w$ is the width in grid points.  This ensures:
- Small `width` → small `λ` → sharp reconstruction (less smoothing)
- Large `width` → large `λ` → smooth reconstruction (more regularization)

The user never sees $\lambda$.

### 2.4 Why This Preserves Curvature

With $p=1$ (v3 equivalent): the penalty minimizes $\|\nabla u\|^2$ — the
solution is a membrane (flat in gaps, flattens curvature).

With $p=2$ (v4 default): the penalty minimizes $\|\nabla^2 u\|^2$ — the
solution minimizes bending energy, preserving curvature across gaps.  This
is equivalent to thin-plate spline interpolation.

### 2.5 Initialization Strategy

We initialize with **linear interpolation** (`_linear_init_2d` from v3).
This is critical because:
1. The DCT/FFT requires a fully-populated array (no NaN) for the forward
   transform.
2. A good initial guess reduces the number of iterations needed.
3. Linear interpolation preserves spatial gradients from the start.

The iterative DCT-PLS then refines this initial guess, replacing the
linear fill with the spectrally-optimal curvature-preserving solution.

---

## 3. Implementation

### 3.1 New Function: `dct_inpaint`

```python
def dct_inpaint(
    data: np.ndarray,
    width: float,
    coordinates: str = 'cartesian',
    order: int = 2,
    max_iter: int = 100,
    tol: float = 1e-5,
    init: str = 'linear',
    smooth_output: bool = False,
    **kwargs,
) -> np.ndarray:
```

### 3.2 Helpers

- `_compute_eigenvalues(shape, order, bc)` — compute eigenvalue tensor $E_p$
- `_width_to_lambda(width, order)` — derive $\lambda$ from `width`

### 3.3 Polar Support

For polar data (periodic azimuth, reflective range):
- Azimuth: use `scipy.fft.rfft` / `irfft` with DFT eigenvalues
- Range: use `scipy.fft.dct` / `idct` with DCT eigenvalues
- Separable application: azimuth filter → range filter (matches `smooth_polar` pattern)

---

## 4. Experiments

### 4.1 Reproduce v3 Benchmarks
- Same 720×1000 polar grid
- Same circular hole (radius=100)
- Compare: griddata baseline vs v3 vs v4
- Metrics: MAE, convergence iterations, wall time

### 4.2 Expected Outcome
v4 should approach griddata accuracy (MAE ~1.2) while maintaining DCT speed
advantage (~10x faster than griddata).

---

## 5. Files Modified/Created

| File | Action |
|------|--------|
| `dct_toolkit/gap_filling.py` | Add `dct_inpaint`, helpers |
| `dct_toolkit/__init__.py` | Export `dct_inpaint` |
| `tests/test_gap_filling.py` | New test suite |
| `exp_v4/test_inpaint_vs_v3.py` | Benchmark comparison |
| `docs/GAP_FILLING_BASIS.md` | Update with DCT-PLS theory |
| `CHANGELOG.md` | Document v0.3.0 |

---

## 6. References

- Garcia, D. (2010). Robust smoothing of gridded data in one and higher
  dimensions with missing values. *Computational Statistics & Data Analysis*.
- Bertalmio, M. et al. (2000). Image inpainting. *SIGGRAPH*.
- Wang, G. et al. (2006). DCT-based regularization for inverse problems.

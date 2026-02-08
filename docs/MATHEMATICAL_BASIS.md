# Mathematical Basis

## 1. DCT-Based Smoothing

The core operation in `dct-toolkit` is spectral smoothing using the Discrete Cosine Transform (DCT).

### 1.1 Convolution Theorem
Smoothing a signal $x[n]$ with a symmetric kernel $g[n]$ is equivalent to multiplication in the spectral domain:
$$ \text{DCT}(x * g) \approx \text{DCT}(x) \cdot H[k] $$
where $H[k]$ is the transfer function (spectral representation) of the kernel.

### 1.2 Transfer Functions

We define analytical transfer functions for common kernels to avoid discretization errors and ensure exact spectral behavior.

#### Boxcar (Rectangular) Kernel
For a boxcar of width $W$, the continuous Fourier Transform is a Sinc function. In the DCT domain (Type-II, Ortho), the transfer function is:
$$ H[k] = \frac{\sin(W \cdot \theta_k)}{W \cdot \sin(\theta_k)} $$
where $\theta_k = \frac{\pi k}{2N}$ for $k=0, \dots, N-1$.

#### Gaussian Kernel
For a Gaussian with standard deviation $\sigma$, the transfer function is:
$$ H[k] = \exp\left(-\frac{1}{2} (\omega_k \sigma)^2\right) $$
where $\omega_k = \frac{\pi k}{N}$.
We map the user-provided `width` to $\sigma$ via $\sigma = \text{width} / \sqrt{12}$ to match the variance of a boxcar of the same width.

---

## 2. Normalized Convolution (Robust Statistics)

Standard convolution fails in the presence of missing data (NaNs). We use **Normalized Convolution** to compute robust local statistics.

### 2.1 The Concept
Normalized convolution treats the signal as a weighted field $f(x)$ with a certainty mask $m(x)$ (where $m=1$ for valid data, $m=0$ for gaps).
$$ \hat{f}(x) = \frac{(f \cdot m) * g}{m * g} $$
where $*$ denotes convolution with the smoothing kernel $g$.

### 2.2 Robust Mean
This directly gives the robust local mean:
$$ \mu(x) = \frac{\text{smooth}(data \cdot mask)}{\text{smooth}(mask) + \epsilon} $$
- The **numerator** represents the weighted sum of valid data in the neighborhood.
- The **denominator** represents the effective sample size (density) in the neighborhood.
- The ratio is the correct local average, ignoring gaps.

### 2.3 Robust Variance
Variance is defined as $E[X^2] - (E[X])^2$. We compute both terms using normalized convolution:
$$ E[X^2](x) = \frac{\text{smooth}(data^2 \cdot mask)}{\text{smooth}(mask) + \epsilon} $$
$$ \text{Var}(x) = \max\left(0, E[X^2](x) - (\mu(x))^2\right) $$

---

## 3. Polar Coordinate Smoothing

Radar and Lidar data often come in Polar coordinates (Azimuth $\phi$, Range $r$). Smoothing on this grid requires special handling to maintain physical consistency.

### 3.1 Adaptive Azimuth Kernel
In a polar grid, the physical arc length $\Delta s$ corresponding to an azimuthal angle $\Delta \phi$ increases with range: $\Delta s \approx r \cdot \Delta \phi$.
To smooth with a constant physical width $W_{phys}$, the effective kernel width in the azimuthal grid domain (radians or indices) must decrease with range:
$$ W_{grid}(r) = \frac{W_{phys}}{r \cdot \Delta \phi_{res}} $$
`dct-toolkit` computes a unique transfer function $H_{az}[k, r]$ for each range gate $r$ to apply this adaptive smoothing efficiently.

### 3.2 Boundary Conditions
- **Range**: Typically non-periodic. We use DCT-II (Reflective BC), effectively assuming $df/dr = 0$ at the boundaries.
- **Azimuth**: The angular domain wraps around ($0^\circ \equiv 360^\circ$).
  - **Periodic BC**: We use the Real FFT (RFFT) instead of DCT. This enforces circular convolution, ensuring values at $359^\circ$ smooth correctly into values at $0^\circ$.

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

### 1.3 Separable N-D and Mixed Kernels

The Cartesian smoother builds one transfer function $H_a$ for each array axis
$a$. The combined spectral response is their product:

$$
H(k_0, \ldots, k_{d-1}) = \prod_{a=0}^{d-1} H_a(k_a; W_a, t_a),
$$

where $W_a$ is the width and $t_a$ is the kernel type along axis $a$.
Each axis can use a different supported kernel. For boxcars in x and y and
a Gaussian in z:

$$
H(k_x, k_y, k_z) = H_{\mathrm{boxcar}}(k_x; W_x)
                   H_{\mathrm{boxcar}}(k_y; W_y)
                   H_{\mathrm{gaussian}}(k_z; W_z).
$$

This is the tensor product of the corresponding one-dimensional smoothing
operators. It is equivalent, up to floating-point rounding, to applying those
operators successively along the axes. The implementation multiplies the N-D
spectrum by each broadcast 1-D response, using one forward and one inverse DCT.
All Cartesian axes retain reflective boundary conditions. Since each supported
response preserves its DC component ($H_a(0)=1$), the product preserves constant
fields as well.

`width` and `kernel_type` sequences follow NumPy array axis order. A scalar width
or single kernel string is repeated along all axes. Equal widths with different
kernel types can still produce anisotropic smoothing. Widths are measured in
grid cells along each axis: for grid spacing $\Delta_a$ and requested physical
width $L_a$, use $W_a=L_a/\Delta_a$. A Gaussian axis has
$\sigma_a=W_a/\sqrt{12}$ in grid cells. Mixed kernels do not change the individual
kernel width conventions or their discretization behavior.

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

For mixed Cartesian kernels, let $S$ be the complete separable smoothing operator
from Section 1.3. The normalized mean is $S(fm)/S(m)$: apply all per-axis filters
to the zero-filled data and mask separately, then divide once. Successively
computing a normalized mean along each axis generally differs when gaps are
present, because normalization changes the weights between passes. The same
combined operator is used for both variance moments and for the density used by
prefill and support thresholds.

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

### 3.3 Mixed Polar Kernels and Filter Order

Polar `kernel_type` pairs follow `(azimuth, range)` order. The selected azimuth
kernel still uses the effective width
$w_{\mathrm{az}}(r)=W_{\mathrm{az}}/(r\,\Delta\theta)$ at each range gate, while
the range kernel uses $W_{\mathrm{range}}$ gates. Here $r=1,\ldots,n_{\mathrm{range}}$
is the existing range-index convention and $\Delta\theta$ is in radians.
Gaussian standard deviations are the corresponding effective widths divided
by $\sqrt{12}$.

Let $A$ apply the chosen adaptive azimuth filter independently at each range
gate, and let $R$ apply reflective smoothing in range. The polar smoother is
$S=R\circ A$: azimuth first, range second. Because $A$ varies with range,
$R\circ A$ and $A\circ R$ generally differ. This is not the stationary tensor
product used for Cartesian data. Both filters preserve constants, so their
composition does too. Normalized statistics use $S(fm)/S(m)$ with the same order
and kernel pair for data and mask; variance, prefill, and density thresholds
use that same operator.

### 3.4 Periodic Discrete Boxcar

For `'boxcar_discrete'`, round the effective azimuth width to an integer, clamp
it to at least 1, and increment even sizes to obtain an odd window $M=2m+1$.
On a periodic axis with $N$ beams, its response at $\omega_k=2\pi k/N$ is

$$
H(k)=\frac{1+2\sum_{j=1}^{m}\cos(j\omega_k)}{M}
    =\frac{\sin(M\omega_k/2)}{M\sin(\omega_k/2)},\qquad H(0)=1.
$$

The implementation uses the sine ratio, with DC set explicitly to 1. It is
equivalent to averaging $M$ centered samples with circular wrapping; a window
larger than $N$ counts repeated visits to a beam. The window is chosen separately
at each range gate after applying the adaptive width conversion. Reflective
azimuth uses the same odd-window convention with its DCT response. Periodic
discrete boxcars now use this response; the former Gaussian substitution is
removed.

---

## 4. Spectral Inpainting (Gap Filling)

DCT-based smoothing extends naturally to gap filling via Penalised Least Squares. The full mathematical formulation — including the Tikhonov functional, eigenvalue diagonalisation, iterative algorithm, and adaptive polar eigenvalues — is documented in [Gap Filling Basis](GAP_FILLING_BASIS.md).

Key ideas:
- **Penalised Least Squares**: Minimise $J(\hat{u}) = \|W(y - \hat{u})\|^2 + \lambda \|\Delta^p \hat{u}\|^2$ where $W = \text{diag}(\text{mask})$.
- **Spectral filter**: In the DCT/FFT domain, the solution is $\hat{U}[k] = Z[k] / (1 + \lambda \cdot E_p[k])$ where $E_p$ are the eigenvalues of the $p$-th order difference operator.
- **Width-to-lambda bridge**: The user-facing `width` parameter maps to the Tikhonov parameter via $\lambda = (w^2/24)^p$.
- **Adaptive polar eigenvalues**: For polar coordinates, the azimuth eigenvalues are scaled per range gate: $E[k_0, j] = \frac{E_0[k_0]}{(j \cdot \Delta\theta)^{2p}} + E_1[j]$, ensuring physically consistent smoothness across the polar grid.

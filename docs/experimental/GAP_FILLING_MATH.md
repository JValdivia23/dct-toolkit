# Mathematical Basis: Iterative Gap Filling

## 1. Problem Statement

Given a signal $y$ observed only on a subset of points $\Omega$ (where mask $M=1$), we want to estimate the values on the complement $\Omega^c$ (where $M=0$).

$$ y_{obs} = M \cdot y_{true} + \epsilon $$

We assume the true signal $y_{true}$ is smooth in some sense (band-limited or low-pass).

## 2. Iterative Normalized Convolution Algorithm

Unlike Tikhonov regularization (which requires inverting large matrices) or DCT-PLS (which requires solving a penalized least squares system), our approach is **constructive**.

### Step 1: Initialization
We initialize the estimate $\hat{y}^{(0)}$ by filling gaps with a neutral value (e.g., 0 or global mean), or better, the result of a single pass of Normalized Convolution.

$$ \hat{y}^{(0)} = y_{obs} \quad (\text{with 0 at gaps}) $$

### Step 2: Iteration
At each step $k$, we compute a "smooth trend" of the current estimate using Normalized Convolution. Importantly, we treat **all** points in the current estimate as valid sources of information, but we "reset" the known valid points to their ground truth after smoothing.

$$ \text{Trend}^{(k)} = \text{Smooth}(\hat{y}^{(k)}) $$

$$ \hat{y}^{(k+1)} = M \cdot y_{obs} + (1-M) \cdot \text{Trend}^{(k)} $$

### Step 3: Convergence
The iteration repeats until $||\hat{y}^{(k+1)} - \hat{y}^{(k)}|| < \delta$.

### Why it works
This process is equivalent to solving the heat equation (diffusion) with Dirichlet boundary conditions (fixed values at valid points).
- The smoothing operation acts as a diffusion operator.
- Resetting the valid points acts as the boundary condition.
- The gaps "fill up" via diffusion from the valid boundaries.

## 3. Comparison with Garcia (DCT-PLS)

| Feature | DCT-PLS (Garcia 2010) | Iterative Normalized Conv (Ours) |
|---------|-----------------------|----------------------------------|
| **Objective** | Minimize $||y-\hat{y}||^2 + \lambda||\nabla^2 \hat{y}||^2$ | Diffusion limit |
| **Method** | Solve linear system in spectral domain | Iterative smoothing |
| **Parameters** | Smoothness $\lambda$ (hard to tune) | Kernel width $\sigma$ (physically intuitive) |
| **Speed** | Fast ($O(N \log N)$) | Fast ($k \times O(N \log N)$) |
| **Flexibility** | Rigid Laplacian prior | Any smoothing kernel (Gaussian, Boxcar) |

Our method is more intuitive ("smooth local mean") and easier to control via the kernel width, which corresponds directly to the correlation length scale of the data.

# Mathematical Basis

This guide targets **v1.0.0**, based on v0.6.0. It
distinguishes the retained legacy 2D polar convention from explicit N-D geometry.
See the [API reference](API_REFERENCE.md) for signatures and validation rules.

## 1. Spectral Smoothing Operators

### 1.1 Reflective and periodic axes

For an orthonormal DCT-II matrix $C$ and a selected spectral response $H$, the
reflective-axis operator is

$$
K=C^\mathsf{T}\operatorname{diag}(H)C.
$$

This defines the discrete operator exactly, up to numerical rounding. It is a
spectral filter with half-sample even boundary extension, not ordinary circular
convolution. A periodic axis instead uses the real FFT and its inverse, with
the corresponding circular spectral response. Both use `norm='ortho'`.
Setting $H(0)=1$ preserves constant fields. It does **not** by itself guarantee
nonnegative spatial weights.

### 1.2 Widths and transfer functions

Let $W$ be effective width in grid cells and $N$ the axis length. Frequencies are
$\omega_k=\pi k/N$ for a reflective axis and $\omega_k=2\pi k/N$ for a periodic
axis, using the frequencies represented by the respective transform.

For `'boxcar'`, the response is

$$
H(k)=\frac{\sin(W\omega_k/2)}{W\sin(\omega_k/2)},\qquad H(0)=1.
$$

For odd integer widths this equals the centered discrete boxcar response.
Arbitrary fractional widths define a spectral extension of that formula; they
must not automatically be interpreted as a nonnegative sampled rectangular
kernel or an exact continuous-space convolution.

For `'boxcar_discrete'`, round $W$ to an integer, clamp to at least 1, and
increment even sizes to obtain an odd window $M=2h+1$. Its response is

$$
H(k)=\frac{1+2\sum_{j=1}^{h}\cos(j\omega_k)}{M}
    =\frac{\sin(M\omega_k/2)}{M\sin(\omega_k/2)},\qquad H(0)=1.
$$

This averages centered samples under the chosen boundary extension. A periodic
window larger than $N$ visits some input samples more than once; reflection can
also repeat samples. Discrete angular window rounding occurs after the local
geometry conversion, independently at each relevant range/elevation slice.
The periodic discrete response is an actual boxcar, not a Gaussian substitution.

For `'gaussian'`,

$$
H(k)=\exp\!\left[-\tfrac12(\omega_k\sigma)^2\right],
\qquad \sigma=W/\sqrt{12}.
$$

The width-to-sigma convention matches the variance of a continuous boxcar of
width $W$. It does not equate the kernels' shapes, support, or statistical ESS.
More generally, a nonnegative spectrum does not prove that the discrete inverse
transform has nonnegative spatial weights. Statistical positivity arguments
below are conditional on the actual spatial weights.

### 1.3 Cartesian tensor-product smoothing

The Cartesian backend uses fixed widths/kernels on all axes. Its N-D response is

$$
H(k_0,\ldots,k_{d-1})=\prod_{a=0}^{d-1}H_a(k_a;W_a,t_a).
$$

It is a tensor product of independent 1D reflective operators. The axes commute;
one forward/inverse N-D DCT pair implements the product, up to rounding.
`width` and `kernel_type` sequences follow array order. Cartesian widths are
grid cells; physical widths must be divided by axis spacing before calling
`smooth_cartesian`. Equal widths with different kernels can still be anisotropic.

## 2. Explicit N-D Polar/Spherical Geometry

### 2.1 Coordinates, units, and selection

With `axis_types` supplied, roles identify azimuth, elevation, range, time,
linear (`cartesian` alias), and batch axes in arbitrary array order. At most one
axis of each spherical role is allowed. Every per-axis option follows **array
order**: coordinates, widths, kernel types, boundaries, and width-unit entries.

Coordinates describe a regular tensor grid: each supplied array is numeric,
real, finite, 1D, strictly increasing, and uniformly spaced. The API does not
regrid irregular/concatenated multi-sweep radar data. Elevation coordinates lie
in $[-\pi/2,\pi/2]$ (or $[-90,90]$ degrees); explicit mode rejects zero-length
array axes. `filter_axes=None` selects all non-batch axes; explicit indices select
only those axes. Batch axes cannot be selected. An empty selection is identity
smoothing, without inter-cell filling. Singleton axes are identities with
nominal support width 1.

Explicit geometry defaults to `width_units='physical'`. For a time, linear, or
range axis with coordinate spacing $\Delta_a$, a physical width $L_a$ becomes

$$
W_a=L_a/\Delta_a.
$$

`width_units='grid'` instead uses literal cells on that axis, including angular
axes. Physical/grid choices may be mixed with a length-`ndim` sequence. Actual
coordinates must be provided wherever physical conversion needs them; a missing
coordinate array is not an implicit physical spacing or range origin.

### 2.2 Local angular metric

Write $R$ for range, $\phi$ for azimuth, and $\theta$ for elevation from the
horizontal plane. The local spherical coordinate scale factors are

$$
ds_{\mathrm{az}}=R\cos\theta\,d\phi,
\qquad ds_{\mathrm{el}}=R\,d\theta.
$$

Consequently, physical angular filter widths, expressed as distances in the
**range coordinate's units**, become

$$
W_{\mathrm{az}}(R,\theta)
  =\frac{L_{\mathrm{az}}}{R\cos\theta\,\Delta\phi},
\qquad
W_{\mathrm{el}}(R)
  =\frac{L_{\mathrm{el}}}{R\,\Delta\theta}.
$$

Angles/spacings in these formulas are radians. The single global `angle_units`
option (`'degrees'` or `'radians'`) describes the coordinate arrays, which are
converted internally. Without an elevation role the azimuth cosine factor is 1.

A non-singleton physical angular filter requires actual positive range
coordinates and the relevant angular coordinates. Physical azimuth also uses
actual elevation coordinates when an elevation role exists, even if that axis
is not filtered. The true first range gate is honored; it is not replaced with
the index 1. Origins/poles that make conversion singular are rejected rather
than assigned an arbitrary cap. Singleton filters remain identities.

These scale factors prescribe **local coordinate-line widths**. They do not
turn the algorithm into exact isotropic Cartesian/geodesic convolution, an
angular-area integral, or cell-volume quadrature. In particular, no spherical
Jacobian such as $R^2\cos\theta$ is introduced as an averaging weight.

### 2.3 Independent boundaries

In explicit mode each axis may be reflective (DCT-II) or periodic (RFFT).
`boundary=None` inherits `az_boundary` for azimuth, `range_boundary` for range,
and reflective boundaries for the other roles. A string overrides all axes;
a length-`ndim` sequence supplies independent choices in array order.

A selected non-singleton **physical periodic azimuth** grid must satisfy

$$
N_{\mathrm{az}}\Delta\phi=2\pi,
$$

with no duplicate endpoint. For example, 0 through 359 degrees at 1-degree
spacing is a complete periodic grid; including both 0 and 360 is not. A regular
sector can use reflective boundaries. Periodicity on other coordinate roles is
a user-selected model, not a claim that those physical coordinates wrap.

### 2.4 Ordered, generally nonseparable composition

Let $K_{\mathrm{az}}$ filter azimuth lines with widths depending on range and
elevation, and $K_{\mathrm{el}}$ filter elevation lines with widths depending
on range. With all these roles selected, the canonical operator is

$$
S=K_{\mathrm{linear}}\circ K_{\mathrm{time}}\circ K_{\mathrm{range}}
  \circ K_{\mathrm{el}}\circ K_{\mathrm{az}}.
$$

Read right to left: **azimuth, elevation, range, time, linear**. Missing or
unselected roles are omitted; independent time/linear dimensions use their own
fixed axis operators. This order is semantic, not array storage order. If $P$
permutes the data axes and every per-axis option is reordered consistently, the
permuted operator is $P S P^{-1}$, up to floating-point rounding.

For fixed geometry this is a linear, constant-preserving operator. It is
generally **nonseparable and noncommuting**: range smoothing changes which
range-dependent angular widths would be applied, and elevation smoothing can
mix different azimuth widths. Thus, for example,
$K_{\mathrm{range}}K_{\mathrm{az}}\ne
K_{\mathrm{az}}K_{\mathrm{range}}$ in general. It is not represented by one
global stationary tensor-product spectral kernel. Grid-width special cases
can be separable; that does not extend to general physical angular widths.

### 2.5 Legacy 2D compatibility

`axis_types=None` retains the v0.6.0 **(azimuth, range)** interpretation. It uses
$r_j=j+1$ and the legacy spacing `az_res_deg` to define

$$
W_{\mathrm{az}}(j)=\frac{L_{\mathrm{az}}}{(j+1)\Delta\phi},
\qquad W_{\mathrm{range}}=L_{\mathrm{range}}\ \text{gates}.
$$

Here $L_{\mathrm{az}}$ is the old nominal index-based width, **not** a newly
inferred physical distance. Scalar widths/kernels or pairs in `(azimuth, range)`
order retain their meaning. The operator is azimuth then reflective range;
azimuth may be reflective or periodic. The existing positive finite scalar
`az_res_deg` validation remains. Legacy periodic grids are not retroactively
required to describe physical full-circle coordinates. Nondefault new geometry
options require explicit `axis_types`; old calls are never silently reinterpreted.

## 3. Normalized Moments with Missing Data

### 3.1 One normalization after the complete operator

Let $m$ be 1 only for finite, valid input samples and 0 otherwise. Form $x_m$ by
copying valid $x$ and putting **zero** at invalid entries; do not evaluate
`NaN * 0` and expect a finite result. For either the Cartesian or polar operator
$S$, the normal mean path is

$$
d_i=S(m)_i,\qquad
\mu_i=\frac{S(x_m)_i}{d_i},
\qquad d_i\ \text{finite and}\ d_i>10^{-3}.
$$

The strict `> 1e-3` condition is the default public stability threshold. There is
**no additive epsilon** in this ratio. The denominator is a smoothed validity
response (support density), not statistical effective sample size.

Every filter acts on data and mask separately before division **once**. Doing a
normalized mean along azimuth and then another along elevation/range generally
gives different weights when gaps are present. The same complete operator,
geometry, kernels, and boundaries are used for both moments and density checks.

Where this ratio is unsupported or poorly conditioned, the existing mean
fallback separately prefills the field and applies $S$ to that filled field at
affected positions. It is a different estimator, not epsilon regularization.
No fallback can borrow across batch/unfiltered dimensions. By default invalid
input support is restored to NaN; `restore_input_nan=False` retains estimates in
gaps where support exists in the same domain.

### 3.2 Second moment and variance

The normal second-moment path is independent of the first:

$$
M_{2,i}=\frac{S((x^2)_m)_i}{d_i},\qquad
V_i=\max(0,M_{2,i}-\mu_i^2),\qquad
\sigma_i=\sqrt{V_i}.
$$

At low-support positions, the first-moment fallback prefills $x$ while the
second-moment fallback **independently prefills $x^2$**. Squaring a prefilled
field would not implement the same estimator. The final variance clamp is
always applied; the result is a population-moment estimate, not a
Bessel-corrected unbiased sample variance.

If a row of $S$ has nonnegative spatial weights $q_{ij}$, a positive denominator
defines normalized valid-data weights $q_{ij}m_j/d_i$. For the ratio path,
the associated variance is nonnegative in exact arithmetic. This guarantee is
conditional on **actual nonnegative spatial weights**, not a kernel label.
Arbitrary fractional spectral kernels need not be positive; a clamp cannot
make a signed-weight moment estimator a probability-weighted average. Separate
fallback estimates also should not be conflated with the ratio-path weights.

## 4. Nominal Weighted Support, Not Statistical ESS

The public name `dct_count` is retained. It computes

$$
C_i=\operatorname{clip}(S(m)_i,0,1)\,A_i.
$$

In Cartesian mode the legacy factor is $A=\prod_a W_a$. Legacy 2D polar mode uses
$A_j=L_{\mathrm{range}}L_{\mathrm{az}}/[(j+1)\Delta\phi]$. For explicit N-D
geometry it is the product of the local effective grid widths on **selected
filtered axes only**. Singleton axes contribute 1; batch/unfiltered dimensions
do not contribute support area. The factor is nominal: discrete kernel rounding
does not replace it with the rounded window size, nor does Gaussian filtering
replace it with a squared-weight ESS factor.

The explicit nominal factor is evaluated only for the count diagnostic. Its
final value must be representable as a finite positive `float64`; otherwise
`dct_count` raises `ValueError`. This bookkeeping limit does not constrain the
smoothing operator or normalized moments, although each individual effective
grid width must be finite and positive.

This quantity is **nominal weighted support**, not the number of distinct
neighbors, a physical cell area/volume, or the number of independent
observations. Full validity gives the nominal factor even when a boundary
extension or an oversized periodic window revisits the same sample. Clipping
the mask response for this diagnostic does not make spatial weights positive.

For independent samples with nonnegative combined weights $v_j$, statistical
effective sample size would instead be

$$
N_{\mathrm{eff}}=\frac{(\sum_j v_j)^2}{\sum_j v_j^2}.
$$

For a supported ratio row one can take $v_j=q_{ij}m_j$. Repeated boundary visits
to the **same input observation** must be aggregated before squaring its weight;
they are not additional independent data. `dct_count` does not compute these
squared weights, and neither it nor a weight-only ESS accounts for correlations
between different observations.

## 5. Prefill, Density Gates, and Independent Domains

`dct_prefill` grows valid support by iterative normalized-convolution estimates,
preserving non-target values. Its default `min_effective_density=0.35` gates
iterative candidates using the smoothed validity mask; `None` disables the gate.
`dct_mean` has an optional gate (default `None`) on the original mask response,
applied after its fallback. These are dimensionless support-density criteria,
not ESS thresholds. The nominal `dct_count` diagnostic uses a clipped response;
spectral mask overshoots must not be confused with a probability density.

Prefill uses three iterations by default; `max_iter=None` has a safety cap of
20, not an unconditional guarantee that every gap is filled. Unresolved targets
use nearest-neighbor fallback when their own domain has finite support. This
residual fallback is distinct from the iterative density gate.

All unfiltered/batch indices define separate domains. The smoothing operator
is block diagonal with respect to those domains, and **every** filling/fallback
stage respects the same partition. A wholly missing domain remains missing;
values in another time slice or batch are not available unless that axis is
selected for filtering. An unfiltered range/elevation axis may still supply
geometry for a physical angular filter without exchanging data along that axis.

Nearest-neighbor fallback uses index-space distances, not physical/geodesic
distances. For a periodic selected axis of length $N$, the circular distance is

$$
d(i,j)=\min(|i-j|,N-|i-j|).
$$

Legacy polar propagation tries range then azimuth; periodic azimuth respects
the sweep seam in axis-wise and global fallback. Equal-distance neighbors in
the axis-wise azimuth pass prefer the preceding beam around the circle.
Explicit geometry likewise respects the selected periodic/reflective axes
without crossing independent-domain boundaries.

Explicit fallback retains range-first and other unique-role axis passes, but
does not process repeated active time/linear roles individually in array order.
Residual targets use simultaneous multidimensional nearest completion within
each independent domain. Exact global ties average all nearest seeds after the
retained passes; newly completed global targets are not reused as donors. This
keeps repeated-role completion equivariant under axis permutations rather than
letting storage layout select an interpolation direction. Legacy 2D rules remain
unchanged.

## 6. Research-Internal Gap Filling

Public `dct_prefill` is distinct from the diffusion/inpainting algorithms in
[`dct_toolkit/gap_filling.py`](../dct_toolkit/gap_filling.py). Their module and
function docstrings describe the research-internal implementation. Those
algorithms are not exported by the top-level API, and this N-D geometry contract
does not extend them. The formerly referenced gap-filling basis document is not
present in this repository.

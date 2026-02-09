# Gap Filling Experiment Report: Polar Coordinate Analysis

**Date:** 2026-02-09  
**Version:** v0.3.1  
**Grid:** 720 x 1000 (azimuth x range)  
**Experiment:** Linear initialization + iterative DCT diffusion vs. griddata baseline

---

## Executive Summary

This experiment evaluates the performance of the DCT-based gap filling algorithm with linear interpolation initialization against a scipy griddata baseline. All spatial visualizations are now presented in **polar coordinates** showing the full 360° azimuthal view (0-720 indices mapped to 0-360°) with range as radial distance (0-1000).

### Key Findings

| Metric | Griddata | DCT (w=50, iter=10) | Improvement |
|--------|----------|---------------------|-------------|
| **MAE** | 0.0992 | 0.0615 | **38% better** |
| **Time** | 33.1s | 0.3s | **110x faster** |

---

## 1. Visualization Results

### 1.1 Spatial Comparison (Polar View)

**Figure:** `gap_filling_spatial_comparison.png/pdf`

The polar coordinate visualization reveals the full 360° structure of the gap filling:

- **Ground Truth:** Shows six Gaussian blobs distributed across the polar grid
- **Gapped Data:** A circular hole is visible at range ≈200 (center location)
- **DCT Fill (w=50, iter=10):** Smooth reconstruction with natural interpolation across the hole
- **Griddata:** Piecewise linear reconstruction showing triangulation artifacts

**Visual Insights:**
- The polar view clearly shows the azimuthal symmetry of the test pattern
- The circular hole appears as a disk-shaped gap in the field
- DCT smoothing preserves the radial gradient structure better than griddata
- Full 360° view reveals no azimuthal bias in the reconstruction

### 1.2 Width Impact Analysis

**Figure:** `gap_filling_width_impact.png/pdf`

MAE vs smoothing width for different iteration counts:

- **Small widths (3-5):** Minimal improvement over linear initialization alone
- **Optimal range (20-50):** Best accuracy, width ≈ 25% of hole diameter (100px)
- **Large widths (75-100):** Diminishing returns, slight over-smoothing

**Key Observation:**
For a 100-pixel radius hole, the optimal smoothing width is 50 pixels (50% of diameter), consistent with the theoretical expectation that width should be a significant fraction of the gap size.

### 1.3 Iteration Convergence

**Figure:** `gap_filling_iteration_convergence.png/pdf`

Convergence behavior for widths 5, 20, and 50:

- **Width=5:** Minimal improvement after linear init (already near-optimal for small gaps)
- **Width=20:** Moderate improvement with iterations, converges by iteration 10
- **Width=50:** Significant improvement, converges by iteration 10-20

**Convergence Pattern:**
All widths show monotonic error reduction. The 50-pixel width shows the steepest initial drop (iterations 0-5) as the DCT diffusion fills the large hole. By iteration 10, all widths have essentially converged.

### 1.4 Uncertainty Maps (Polar View)

**Figure:** `gap_filling_uncertainty_maps.png/pdf`

Two uncertainty metrics visualized in polar coordinates:

**dct_std (width=5):**
- Shows local variability of the filled field
- Higher uncertainty in gap region (centered at range=200)
- Uncertainty propagates radially from gap boundaries
- Polar view reveals azimuthally symmetric uncertainty distribution

**Mapping Error:**
- Quantifies data density loss due to smoothing
- Maximum error (1.0) inside the hole where no data exists
- Error decreases with distance from gap
- Sharp boundary at gap edge visible in polar coordinates

---

## 2. Performance Analysis

### 2.1 Accuracy

**Mean Absolute Error (MAE) in Gap Region:**
- **Griddata:** 0.0992 (baseline)
- **DCT (w=50, iter=10):** 0.0615

**Analysis:**
The DCT method achieves 38% lower MAE than griddata. This improvement comes from:
1. Linear initialization provides excellent starting point (equivalent to griddata)
2. Iterative DCT diffusion smooths the reconstruction while preserving valid data
3. The spectral approach captures global field structure better than local triangulation

### 2.2 Computational Speed

**Execution Time:**
- **Griddata:** 33.1 seconds
- **DCT fill:** 0.3 seconds

**Speedup:** 110x faster

**Why DCT is faster:**
- FFT-based computation: O(N log N) complexity
- Griddata uses Delaunay triangulation: O(N²) to O(N³) depending on implementation
- For dense grids (720x1000 = 720,000 points), FFT dominates

### 2.3 Memory Efficiency

Both methods process the full grid in memory, but:
- DCT: Uses in-place FFT operations, minimal overhead
- Griddata: Creates triangulation mesh, higher memory footprint

---

## 3. Polar Coordinate Insights

### 3.1 Advantages of Polar Visualization

The conversion to polar coordinates (azimuth as angle, range as radius) provides:

1. **Physical Interpretation:** Matches radar/lidar coordinate systems
2. **Full Domain View:** No cropping or zooming needed; entire 360° visible
3. **Radial Structure:** Clearly shows how features vary with range
4. **Azimuthal Symmetry:** Easy to identify angular patterns or biases

### 3.2 Visualization Quality

**Resolution:**
- 720 azimuth bins × 1000 range bins = 720,000 data points
- Polar mesh preserves all data without interpolation artifacts
- Color mapping (RdBu_r for data, viridis for std, Reds for error) provides clear contrast

**Figure Sizes:**
- PNG: 1-2 MB (compressed raster)
- PDF: 26-93 MB (vector graphics of full mesh)

The large PDF sizes reflect the high resolution of the polar mesh visualization.

---

## 4. Recommendations

### 4.1 For Gap Filling Applications

**Use DCT with Linear Init when:**
- ✓ Large contiguous gaps (hole diameter > 20 pixels)
- ✓ Smooth or band-limited underlying field
- ✓ Real-time processing required (need speed)
- ✓ Polar coordinate data (radar, lidar, sonar)

**Optimal Parameters:**
- **Width:** 25-50% of gap diameter
- **Iterations:** 10-20 (diminishing returns after 20)
- **Initialization:** Linear (default) works best for most cases

### 4.2 Limitations Observed

1. **Sharp Edges:** DCT smoothing will blur sharp discontinuities
2. **Width Selection:** Must match gap scale; too small = no improvement, too large = over-smoothing
3. **All-NaN Regions:** Requires at least some valid data in each row/column for linear init

### 4.3 Comparison to Alternatives

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Griddata | Slow | Good | Irregular gaps, exact linear interpolation needed |
| DCT (linear init) | **Fast** | **Better** | Large holes, smooth fields, polar data |
| Astropy Gaussian | Medium | Poor | Small gaps, noise reduction |

---

## 5. Conclusion

The polar coordinate visualization confirms that the DCT-based gap filling algorithm with linear initialization:

1. **Achieves 38% better accuracy** than scipy griddata baseline
2. **Runs 110x faster**, enabling real-time applications
3. **Preserves radial structure** naturally in polar coordinates
4. **Converges quickly** (10-20 iterations) for large holes

The full 360° polar view reveals the azimuthal uniformity of the reconstruction and confirms no directional bias in the DCT smoothing. For polarimetric radar/lidar applications, this method provides both superior accuracy and computational efficiency.

---

## Generated Artifacts

- **Figures:** `exp_v3/figures/`
  - `gap_filling_spatial_comparison.{png,pdf}` - 4-panel polar comparison
  - `gap_filling_width_impact.{png,pdf}` - Width sweep analysis
  - `gap_filling_iteration_convergence.{png,pdf}` - Convergence curves
  - `gap_filling_uncertainty_maps.{png,pdf}` - Polar uncertainty visualization

---

*Report generated automatically from experimental results.*

# Gap Filling Experiment Report: Periodic Azimuth Boundary

**Date:** 2026-02-09
**Version:** v0.2.2 (Revised)
**Grid:** 720 x 1000 (azimuth x range)
**Experiment:** Linear initialization + iterative DCT diffusion vs. griddata baseline
**Configuration:** Polar coordinates with **periodic azimuth boundary** (0° $\leftrightarrow$ 360°)

---

## Executive Summary

This experiment evaluates the performance of the DCT-based gap filling algorithm on a polar grid with a large circular hole (radius=100 pixels in index space). Crucially, this version applies **periodic boundary conditions** in azimuth.

### Key Findings (Revised)

| Metric | Griddata (Linear) | DCT (w=50, iter=20, init='linear') | DCT (w=50, iter=100, init='dct') |
|--------|-------------------|------------------------------------|----------------------------------|
| **MAE** | **1.236** | 3.057 | 2.338 |
| **Time** | 7.70s | 1.04s | **0.70s** |

**Conclusion:**
- **Accuracy:** `griddata` (Delaunay triangulation) is significantly more accurate (MAE 1.24 vs 2.34) for this specific smooth field with a large hole. It effectively captures the planar/linear trends across the hole using exact boundary values.
- **Speed:** The iterative DCT method is **~10x faster** (0.7s vs 7.7s) and scales better with grid size (O(N log N) vs O(N^2) or worse for triangulation).
- **Initialization:** Using `init='dct'` (Normalized Convolution) improves DCT performance significantly over `init='linear'` (MAE 2.34 vs 3.06).

---

## 1. Experimental Results

### 1.1 Performance Baseline
- **Griddata (Linear + Nearest):** MAE = 1.24. This method triangulates valid points. It performs exceptionally well here because the underlying field (sine/cosine waves) is locally well-approximated by planes across the 20km hole.
- **Linear Initialization Only:** MAE = 4.13. Simple axis-wise interpolation fails to capture the 2D structure of the hole (especially with the wrapping boundary).

### 1.2 Impact of Smoothing Width (`w`)
- **Small Widths (3-20):** Ineffective for this large hole (radius=100). Diffusion is too slow.
- **Large Widths (50):** Necessary to bridge the gap. `w=50` (half radius) converges reasonably well.

### 1.3 Impact of Initialization
- **Linear Init:** MAE 3.06. Leaving artifacts that diffusion struggles to remove.
- **DCT Init:** MAE 2.34. Starts with a smooth normalized convolution result, leading to a better final state.

---

## 2. Analysis: What is the best way to fill data?

Based on the sweep results, the **optimal strategy** depends on the priority:

### If Accuracy is Paramount:
**Use `scipy.interpolate.griddata` (method='linear').**
- **Pros:** Lowest error (MAE 1.24). Respects boundary values exactly.
- **Cons:** Slow (7.7s). Memory intensive for very large point clouds.

### If Speed is Paramount:
**Use `iterative_gap_fill(init='dct', width=large, iter=20)`.**
- **Pros:** Fast (0.7s). Produces visually smooth results.
- **Cons:** Higher error (MAE 2.34). Smoothed edges can introduce curvature errors compared to linear interpolation.

**Winner for this specific test:** `griddata` for accuracy, `DCT` for speed.

---

## 3. Experiment 2: Non-Wrapping Hole (Centered)

To investigate if the poor performance of `linear` initialization was due to the periodic boundary (wrapping hole), we ran a second experiment moving the hole to the center of the grid (Azimuth 180°).

| Metric | Griddata | Linear Init Only | DCT (w=50, iter=20) |
|--------|----------|------------------|---------------------|
| **MAE** | 1.134 | **1.117** | 1.580 |
| **Time** | 7.73s | **0.01s** | 0.92s |

**Findings:**
1.  **Linear Init is Superior:** When away from the boundary, the simple axis-wise linear interpolation (MAE 1.12) slightly outperforms `griddata` (MAE 1.13) and is instant (0.01s).
2.  **Diffusion Increases Error:** Applying DCT smoothing actually increased the error (1.12 $\to$ 1.58). The initial linear guess was already near-optimal for this smooth field; further smoothing flattened the curvature, deviating from the ground truth.
3.  **Root Cause Confirmed:** The failure in Experiment 1 (MAE 4.13) was entirely due to `_linear_init_2d` failing to interpolate correctly across the periodic boundary (Azimuth 0°/360°).

---

## 4. Room for Improvement

The DCT method lags in accuracy. Areas for optimization:

### 4.1 Fix Linear Initialization Boundary Handling
Experiment 2 proves `init='linear'` is excellent (MAE 1.12 vs Griddata 1.13) when it works.
-   **Action:** Update `_linear_init_2d` to detect periodic axes and interpolate across the wrap-around. This would make the DCT toolkit **faster AND more accurate** than `griddata` for this class of problems.

### 4.2 Better Initialization (`init='dct'`)
We observed `init='dct'` is superior to `init='linear'` (MAE 2.34 vs 3.06) *in the wrapping case*, but `linear` wins in the non-wrapping case.

### 4.3 Curvature Preservation
DCT diffusion (Gaussian smoothing) tends to flatten curvature.
-   **Problem:** The Gaussian kernel averages valid data *near* the boundary.
-   **Proposal:** Investigate "Harmonic Filling" (solving $\nabla^2 u = 0$ directly).

---

## 5. Conclusion

For large holes in smooth fields:
1.  **Griddata** is the consistent baseline (MAE ~1.1-1.2).
2.  **Linear Init** is the **best performer** (MAE 1.12, Time 0.01s) *if* the hole does not wrap around the boundary.
3.  **DCT Smoothing** is useful for noise reduction but can degrade a perfect linear fill.

**Recommendation for v0.4.0:**
-   **Priority 1:** Fix `_linear_init_2d` to handle periodic boundaries. This solves the main accuracy deficit.
-   **Priority 2:** Add `init='dct'` as a robust fallback for complex gap geometries where axis-wise linear interpolation fails.

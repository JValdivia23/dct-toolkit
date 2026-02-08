# AGENTS.md - DCT Toolkit Agent Guidelines

**Project**: dct-toolkit  
**Version**: 0.1.0-alpha  
**Status**: Standalone Private Repository  
**Last Updated**: 2026-02-08

---

## Critical Rule: Standalone Project

⚠️ **THIS IS A COMPLETELY STANDALONE PROJECT**

- ✅ ONLY import from within `dct_toolkit/dct_toolkit/` directory
- ✅ External dependencies: `numpy`, `scipy` ONLY
- ❌ NEVER import from parent directories (`../src/`, `../dct_stats/`, legacy code)
- ❌ NEVER import from any code outside this repository
- ❌ NEVER assume other projects are available

**Verification Command**:
```bash
grep -r "^from \.\./" dct_toolkit/
grep -r "^import .*\.\." dct_toolkit/
```
Should return NO matches.

---

## Environment Setup

### Conda Environment
```bash
conda activate myenv
python --version  # Should be 3.8+
```

### PYTHONPATH
When running tests or examples from project root:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
```

---

## Code Style - Scientific Repository Standards

### Docstrings (REQUIRED)
Use NumPy style:
```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Short description.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description
        
    Returns
    -------
    return_type
        Description
        
    Examples
    --------
    >>> result = function_name(1.0, 2.0)
    """
```

### Type Hints (REQUIRED)
All function parameters and returns must be typed:
```python
def smooth_data(data: np.ndarray, width: float) -> np.ndarray:
    ...
```

### Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `get_transfer_function`)
- **Classes**: `CamelCase` (e.g., `DCTSmoother`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_KERNEL`)
- **Private**: `_leading_underscore` (e.g., `_helper_func`)

### Formatting
- **Indentation**: 4 spaces (NO tabs)
- **Line Length**: 100 characters (soft limit)
- **Blank Lines**: 
  - 2 lines between top-level functions/classes
  - 1 line between methods
- **Operators**: Spaces around operators (`x = y + 1`)

### Import Organization
```python
# 1. Standard library
import os
import sys
from typing import Tuple, Optional

# 2. Third-party (numpy, scipy)
import numpy as np
import scipy.fft

# 3. Local modules (relative imports)
from .core import get_dct_transfer_function
from .stats import dct_mean
```

---

## Project Structure

```
dct_toolkit/                    # PROJECT ROOT
│
├── dct_toolkit/               # INSTALLABLE PACKAGE ⚠️ CODE GOES HERE
│   ├── __init__.py           # Public API exports
│   ├── core.py               # Transfer functions & 1D primitives
│   ├── cartesian.py          # 2D Cartesian separable smoothing
│   ├── polar.py              # 2D Polar smoothing (adaptive kernels)
│   ├── stats.py              # Statistical ops (Normalized Convolution)
│   └── utils.py              # Validation & helpers
│
├── tests/                     # TEST SUITE ⚠️ TESTS GO HERE
│   ├── test_core.py
│   ├── test_cartesian.py
│   ├── test_polar.py
│   ├── test_stats.py
│   └── conftest.py           # Fixtures
│
├── examples/                  # USAGE EXAMPLES
│   ├── basic_polar.py
│   └── comprehensive_demo.py
│
├── experimental/              # BETA CODE (not installed)
│   └── gap_filling/
│       ├── iterative_fill.py
│       └── benchmark.py
│
├── docs/                      # DOCUMENTATION
│   ├── MATHEMATICAL_BASIS.md
│   ├── API_REFERENCE.md
│   ├── TEST_REPORT.md
│   ├── PLAN.md
│   └── experimental/
│       ├── GAP_FILLING_MATH.md
│       └── GAP_FILLING_TESTS.md
│
├── README.md                  # Main documentation
├── CHANGELOG.md               # Version history
└── AGENTS.md                  # This file
```

---

## Development Workflow

### Before Making Changes

1. **Activate Environment**
   ```bash
   conda activate myenv
   ```

2. **Verify Baseline**
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
   python -m pytest dct_toolkit/tests/ -v
   ```
   All 19 tests must pass before you start.

3. **Review Documentation**
   - Check `docs/PLAN.md` for design decisions
   - Check `docs/MATHEMATICAL_BASIS.md` for theory
   - Check `docs/API_REFERENCE.md` for existing API

### Making Changes

1. **Edit Code**
   - Modify appropriate module in `dct_toolkit/dct_toolkit/`
   - Follow code style guidelines
   - Add docstrings and type hints

2. **Add Tests**
   - Add/update tests in `dct_toolkit/tests/`
   - Tests should validate correctness with synthetic data
   - Include edge cases (empty arrays, all NaN, etc.)

3. **Update Documentation**
   - Update `API_REFERENCE.md` if API changes
   - Update `CHANGELOG.md` with changes
   - Update this file if workflow changes

### Testing (MANDATORY)

**Run Tests**:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
python -m pytest dct_toolkit/tests/ -v
```

**Expected Output**: 19 tests passing

**If Tests Fail**:
- Fix the code, not the tests
- Tests represent ground truth requirements
- Never commit with failing tests

### Pre-Completion Checklist

Before declaring task complete:

- [ ] All 19 tests pass (`pytest dct_toolkit/tests/ -v`)
- [ ] No imports from outside `dct_toolkit/`
- [ ] All public functions have NumPy-style docstrings
- [ ] All functions have type hints
- [ ] Code follows naming conventions
- [ ] No tabs, 4-space indentation
- [ ] Line length ≤ 100 characters
- [ ] CHANGELOG.md updated
- [ ] Documentation updated if API changed

---

## Current Progress & Roadmap

### ✅ Completed (v0.1.0)
- [x] Core primitives (`core.py`)
  - Transfer functions (boxcar, boxcar_discrete, gaussian)
  - 1D DCT convolution
- [x] Cartesian smoothing (`cartesian.py`)
  - Separable N-D smoothing
- [x] Polar smoothing (`polar.py`)
  - Adaptive azimuth kernels
  - Boundary conditions (reflective/periodic)
- [x] Statistical operations (`stats.py`)
  - Normalized Convolution
  - dct_count, dct_mean, dct_variance, dct_std
- [x] Test suite (19 tests, 100% pass)
- [x] Documentation suite
- [x] Experimental gap filling (100x accuracy improvement)

### 🚧 In Progress / Next Phase (v0.2.0)
- [ ] 3D support (volumetric data)
- [ ] Additional kernels (Savitzky-Golay, Hanning)
- [ ] Performance optimizations (numba?)
- [ ] Jupyter notebook examples
- [ ] pip package setup (`setup.py`, `pyproject.toml`)

### 📋 Future (v0.3.0+)
- [ ] conda-forge distribution
- [ ] Public GitHub repository
- [ ] CI/CD with GitHub Actions
- [ ] Advanced gap filling (constrained optimization)
- [ ] GPU acceleration (CuPy)

---

## Key Technical Details

### Transfer Functions
- **Boxcar (Analytical)**: `H[k] = sin(W·θ)/(W·sin(θ))`, θ = πk/(2n)
- **Boxcar (Discrete)**: Sum of cosines for discrete kernels
- **Gaussian**: `H[k] = exp(-0.5·(ω·σ)²)`, σ = width/√12

### Normalized Convolution
Formula for robust mean with gaps:
```
μ = smooth(data * mask) / (smooth(mask) + ε)
```

### Polar Coordinates
- **Data Layout**: (n_azimuth, n_range)
- **Adaptive Width**: w_az(r) = width / (r · dθ)
- **Boundary Conditions**:
  - Reflective: DCT-II (even symmetry)
  - Periodic: Real FFT (circular convolution)

### Testing Philosophy
- Synthetic data with known ground truth
- Verify theoretical properties (DC preservation, linearity)
- Edge case coverage (all NaN, single point, etc.)

---

## Common Pitfalls & Solutions

### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'dct_toolkit'`
**Solution**: 
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit
```

### 2. Path Confusion
**Problem**: Tests run from wrong directory
**Solution**: Always run from project root (`dct_toolkit/`)

### 3. Polar Boundary Issues
**Problem**: Discontinuity at 360°/0°
**Solution**: Use `az_boundary='periodic'` for azimuth

### 4. Test Failures
**Problem**: Tests fail after changes
**Solution**: Fix the code, don't modify tests to pass

### 5. Legacy Code Contamination
**Problem**: Accidentally importing from `../src/`
**Solution**: Check with grep command in Critical Rule section

---

## Repository Management

### Private Repository (Current)
- Hosted on: GitHub (private)
- Access: Core team only
- Branching: Feature branches → main
- No CI/CD yet (manual testing required)

### Future Public Release
- Will be made public after v0.3.0
- MIT License
- Conda-forge distribution
- Community contributions welcome

---

## Emergency Contacts & References

### Documentation
- **Design Decisions**: `docs/PLAN.md`
- **Math Theory**: `docs/MATHEMATICAL_BASIS.md`
- **API Details**: `docs/API_REFERENCE.md`
- **Test Results**: `docs/TEST_REPORT.md`

### Code Examples
- **Polar Smoothing**: `examples/basic_polar.py`
- **Full Demo**: `examples/comprehensive_demo.py`
- **Gap Filling**: `experimental/gap_filling/benchmark.py`

### Key Papers & References
- Garcia (2010): DCT-PLS (contrast/background)
- Knutsson et al.: Normalized Convolution
- Ahmed et al.: DCT fundamentals
- Lecture 17 Notes: Objective Mapping / Kriging (for context)

---

## Quick Reference Card

```bash
# Setup
conda activate myenv
export PYTHONPATH=$PYTHONPATH:$(pwd)/dct_toolkit

# Test
python -m pytest dct_toolkit/tests/ -v

# Run example
python dct_toolkit/examples/basic_polar.py

# Benchmark
python dct_toolkit/experimental/gap_filling/benchmark.py

# Check imports (should be empty)
grep -r "^from \.\./" dct_toolkit/dct_toolkit/
```

---

**Remember**: This is a standalone scientific package. Quality, correctness, and clarity are more important than speed. When in doubt, refer to the test suite - it defines the expected behavior.

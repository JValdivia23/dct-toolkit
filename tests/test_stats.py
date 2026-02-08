import numpy as np
import pytest
from dct_toolkit.stats import dct_count, dct_mean, dct_variance, dct_std

@pytest.fixture
def uniform_with_gaps():
    """Uniform data with 50% random gaps."""
    data = np.ones(1000)
    mask = np.random.rand(1000) > 0.5
    data[~mask] = np.nan
    return data, mask

@pytest.fixture
def normal_with_gaps():
    """Normal distribution with known variance and gaps."""
    np.random.seed(42)
    data = np.random.randn(2000)  # True variance = 1.0
    data[::3] = np.nan  # 33% gaps
    return data

def test_count_uniform_gaps():
    """Count should reflect gap density."""
    np.random.seed(42)
    mask = np.random.rand(1000) > 0.5  # 50% gaps
    width = 20.0
    count = dct_count(mask, width=width)
    
    # In interior, should be approximately density * width
    # 0.5 * 20 = 10
    interior_mean = np.mean(count[20:-20])
    assert np.abs(interior_mean - 10.0) < 1.0

def test_mean_uniform_with_gaps(uniform_with_gaps):
    """Mean of uniform data should be 1.0 even with 50% gaps."""
    data, mask = uniform_with_gaps
    mean = dct_mean(data, width=20.0)
    # The mean should be very close to 1.0 everywhere (ignoring edges)
    assert np.allclose(mean[20:-20], 1.0, atol=0.05)

def test_variance_normal_distribution(normal_with_gaps):
    """Variance of normal data should be ~1.0 even with gaps."""
    var = dct_variance(normal_with_gaps, width=30.0)
    
    # Check interior mean variance
    mean_var = np.mean(var[30:-30])
    # Expect ~1.0. Tolerance 0.2 allows for sample variance fluctuation
    assert np.abs(mean_var - 1.0) < 0.2

def test_variance_uniform():
    """Variance of uniform [0,1] should be 1/12 ≈ 0.083."""
    np.random.seed(42)
    data = np.random.rand(5000)  # Uniform [0,1]
    data[::4] = np.nan  # 25% gaps
    
    var = dct_variance(data, width=50.0)
    mean_var = np.mean(var[50:-50])
    
    expected = 1.0 / 12.0
    assert np.abs(mean_var - expected) < 0.01

def test_std_consistency(normal_with_gaps):
    """std = sqrt(variance)."""
    var = dct_variance(normal_with_gaps, width=30.0)
    std = dct_std(normal_with_gaps, width=30.0)
    assert np.allclose(std, np.sqrt(var))

def test_all_nan_handling():
    """Should handle all-NaN regions gracefully."""
    data = np.full(100, np.nan)
    mean = dct_mean(data, width=10.0)
    assert np.all(np.isnan(mean))

def test_single_point_influence():
    """Single point should influence neighbors."""
    data = np.full(100, np.nan)
    data[50] = 10.0
    mean = dct_mean(data, width=10.0)
    
    # At index 50, mean should be exactly the value (normalized conv property)
    assert np.isclose(mean[50], 10.0)
    # Nearby points should also be close to 10.0 (constant model assumption)
    assert np.isclose(mean[51], 10.0)

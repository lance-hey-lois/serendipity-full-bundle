"""
FAISS Vector Operations Tests
============================

Comprehensive unit tests for FAISS-based similarity search and vector operations.
Tests both FAISS-accelerated and numpy fallback implementations.

Key Test Areas:
- FAISS availability detection
- Top-k cosine similarity search
- Performance comparison: FAISS vs numpy
- Accuracy validation between implementations
- Edge case handling and robustness
- Memory efficiency and scaling
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import patch, MagicMock

# Mock FAISS if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = MagicMock()

class TestFAISSDetection:
    """Test FAISS availability detection."""
    
    def test_have_faiss_detection(self):
        """Test FAISS availability detection."""
        from serendipity_engine_ui.engine.faiss_helper import have_faiss
        
        # Test actual availability
        actual_availability = have_faiss()
        assert isinstance(actual_availability, bool), "have_faiss should return boolean"
        
        # Test with mocked unavailability
        with patch('builtins.__import__', side_effect=ImportError("No FAISS")):
            assert not have_faiss(), "Should return False when FAISS unavailable"
    
    def test_have_faiss_consistency(self):
        """Test that FAISS detection is consistent across calls."""
        from serendipity_engine_ui.engine.faiss_helper import have_faiss
        
        # Should return same result on multiple calls
        result1 = have_faiss()
        result2 = have_faiss()
        result3 = have_faiss()
        
        assert result1 == result2 == result3, "FAISS detection should be consistent"

class TestTopMCosine:
    """Test top-M cosine similarity search."""
    
    def test_top_m_cosine_basic_functionality(self, rng):
        """Test basic top-M cosine similarity functionality."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        # Create test data with known similarities
        n_candidates = 100
        n_dims = 32
        m = 10
        
        # Create seed vector
        seed_vec = np.array([1.0] + [0.0] * (n_dims - 1))  # Unit vector along first dimension
        
        # Create candidate vectors with varying similarity to seed
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)  # Normalize
        
        # Make some candidates similar to seed by boosting first dimension
        for i in range(0, 20):  # First 20 candidates
            cand_vecs[i, 0] = 0.9  # High similarity
            cand_vecs[i] = cand_vecs[i] / np.linalg.norm(cand_vecs[i])  # Re-normalize
        
        # Find top-M
        top_indices = top_m_cosine(seed_vec, cand_vecs, m)
        
        # Validate output
        assert top_indices.shape == (m,), f"Expected shape ({m},), got {top_indices.shape}"
        assert top_indices.dtype == np.int64, f"Expected int64 indices, got {top_indices.dtype}"
        assert np.all(top_indices >= 0), "All indices should be non-negative"
        assert np.all(top_indices < n_candidates), f"All indices should be < {n_candidates}"
        
        # Check uniqueness
        assert len(np.unique(top_indices)) == m, "All returned indices should be unique"
        
        # Verify ordering (top similarities first)
        similarities = cand_vecs @ seed_vec
        top_similarities = similarities[top_indices]
        
        # Should be in descending order
        for i in range(m - 1):
            assert top_similarities[i] >= top_similarities[i + 1], \
                f"Similarities should be descending: {top_similarities[i]} < {top_similarities[i+1]} at position {i}"
    
    def test_top_m_cosine_exact_results(self, rng):
        """Test that top-M returns exactly the top M similarities."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_candidates = 50
        n_dims = 16
        m = 15
        
        # Create test vectors
        seed_vec = rng.normal(0, 1, n_dims)
        seed_vec = seed_vec / np.linalg.norm(seed_vec)
        
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        # Get top-M via FAISS/function
        top_indices = top_m_cosine(seed_vec, cand_vecs, m)
        
        # Get top-M via direct computation
        similarities = cand_vecs @ seed_vec
        expected_indices = np.argsort(-similarities)[:m]  # Top M in descending order
        
        # Sort both for comparison (order might vary for equal similarities)
        top_similarities = similarities[top_indices]
        expected_similarities = similarities[expected_indices]
        
        # Should have same similarities (within numerical precision)
        top_similarities_sorted = np.sort(top_similarities)[::-1]  # Descending
        expected_similarities_sorted = np.sort(expected_similarities)[::-1]
        
        assert np.allclose(top_similarities_sorted, expected_similarities_sorted, atol=1e-6), \
            "Top-M should return exactly the highest similarities"
    
    def test_top_m_cosine_edge_cases(self, rng):
        """Test edge cases for top-M cosine similarity."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_dims = 8
        seed_vec = rng.normal(0, 1, n_dims)
        seed_vec = seed_vec / np.linalg.norm(seed_vec)
        
        # Test m = 0
        cand_vecs = rng.normal(0, 1, (10, n_dims))
        cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        top_0 = top_m_cosine(seed_vec, cand_vecs, 0)
        assert top_0.shape == (0,), "m=0 should return empty array"
        
        # Test m >= n_candidates
        top_large = top_m_cosine(seed_vec, cand_vecs, 20)  # More than 10 candidates
        assert top_large.shape[0] <= 10, "Should not return more indices than candidates"
        
        # Test single candidate
        single_cand = cand_vecs[:1]
        top_single = top_m_cosine(seed_vec, single_cand, 5)
        assert top_single.shape == (1,), "Single candidate should return single index"
        assert top_single[0] == 0, "Single candidate index should be 0"
        
        # Test empty candidate set
        empty_cands = np.array([]).reshape(0, n_dims)
        top_empty = top_m_cosine(seed_vec, empty_cands, 5)
        assert top_empty.shape == (0,), "Empty candidates should return empty result"
    
    def test_top_m_cosine_identical_vectors(self, rng):
        """Test top-M with identical vectors."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_dims = 6
        seed_vec = rng.normal(0, 1, n_dims)
        seed_vec = seed_vec / np.linalg.norm(seed_vec)
        
        # Create candidates with some identical to seed
        n_candidates = 20
        m = 8
        
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        # Make first 5 candidates identical to seed
        for i in range(5):
            cand_vecs[i] = seed_vec.copy()
        
        top_indices = top_m_cosine(seed_vec, cand_vecs, m)
        
        # First 5 should be in the top results (they have similarity = 1.0)
        similarities = cand_vecs @ seed_vec
        top_similarities = similarities[top_indices]
        
        # At least 5 should have similarity ≈ 1.0
        perfect_matches = np.sum(np.abs(top_similarities - 1.0) < 1e-10)
        assert perfect_matches >= 5, f"Should find at least 5 perfect matches, found {perfect_matches}"
    
    def test_top_m_cosine_numerical_precision(self, rng):
        """Test numerical precision of top-M cosine similarity."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_dims = 10
        n_candidates = 30
        m = 10
        
        # Create vectors with very small differences
        base_vec = rng.normal(0, 1, n_dims)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        seed_vec = base_vec.copy()
        
        # Create candidates with tiny variations
        cand_vecs = []
        for i in range(n_candidates):
            noise = rng.normal(0, 1e-8, n_dims)  # Very small noise
            cand_vec = base_vec + noise
            cand_vec = cand_vec / np.linalg.norm(cand_vec)
            cand_vecs.append(cand_vec)
        
        cand_vecs = np.array(cand_vecs)
        
        # Should still work with high precision
        top_indices = top_m_cosine(seed_vec, cand_vecs, m)
        
        assert top_indices.shape == (m,), "Should handle high-precision vectors"
        assert np.all(np.isfinite(top_indices)), "Indices should be finite"
        
        # All similarities should be very close to 1
        similarities = cand_vecs @ seed_vec
        top_similarities = similarities[top_indices]
        assert np.all(top_similarities > 0.999), f"All similarities should be very high, min: {np.min(top_similarities)}"

class TestFAISSvsNumpyComparison:
    """Test FAISS vs numpy implementation comparison."""
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_faiss_vs_numpy_accuracy(self, rng):
        """Test accuracy comparison between FAISS and numpy implementations."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_candidates = 100
        n_dims = 32
        m = 20
        
        # Create test data
        seed_vec = rng.normal(0, 1, n_dims)
        seed_vec = seed_vec / np.linalg.norm(seed_vec)
        
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        # Get results with FAISS (if available)
        faiss_indices = top_m_cosine(seed_vec, cand_vecs, m)
        
        # Get results with numpy fallback (force by mocking FAISS as unavailable)
        with patch('serendipity_engine_ui.engine.faiss_helper.have_faiss', return_value=False):
            numpy_indices = top_m_cosine(seed_vec, cand_vecs, m)
        
        # Compare similarities (not necessarily same indices due to ties)
        similarities = cand_vecs @ seed_vec
        faiss_similarities = similarities[faiss_indices]
        numpy_similarities = similarities[numpy_indices]
        
        # Sort for comparison
        faiss_similarities_sorted = np.sort(faiss_similarities)[::-1]
        numpy_similarities_sorted = np.sort(numpy_similarities)[::-1]
        
        # Should be nearly identical
        assert np.allclose(faiss_similarities_sorted, numpy_similarities_sorted, atol=1e-10), \
            "FAISS and numpy should give same top similarities"
    
    def test_numpy_fallback_functionality(self, rng):
        """Test numpy fallback when FAISS unavailable."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_candidates = 50
        n_dims = 16
        m = 12
        
        seed_vec = rng.normal(0, 1, n_dims)
        seed_vec = seed_vec / np.linalg.norm(seed_vec)
        
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        # Force numpy fallback
        with patch('serendipity_engine_ui.engine.faiss_helper.have_faiss', return_value=False):
            numpy_indices = top_m_cosine(seed_vec, cand_vecs, m)
        
        # Validate numpy fallback results
        assert numpy_indices.shape == (m,), f"Numpy fallback shape should be ({m},)"
        assert numpy_indices.dtype == np.int64, "Numpy fallback should return int64"
        assert np.all(numpy_indices >= 0) and np.all(numpy_indices < n_candidates), \
            "Numpy fallback indices should be valid"
        
        # Verify correctness
        similarities = cand_vecs @ seed_vec
        numpy_similarities = similarities[numpy_indices]
        
        # Should be in descending order
        for i in range(m - 1):
            assert numpy_similarities[i] >= numpy_similarities[i + 1], \
                f"Numpy fallback should be ordered: pos {i}"
    
    def test_implementation_consistency(self, rng):
        """Test consistency between implementations across multiple runs."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_candidates = 40
        n_dims = 12
        m = 10
        
        # Fixed test data for reproducibility
        np.random.seed(12345)
        seed_vec = np.random.normal(0, 1, n_dims)
        seed_vec = seed_vec / np.linalg.norm(seed_vec)
        
        cand_vecs = np.random.normal(0, 1, (n_candidates, n_dims))
        cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        # Test multiple runs with numpy fallback (should be deterministic)
        with patch('serendipity_engine_ui.engine.faiss_helper.have_faiss', return_value=False):
            result1 = top_m_cosine(seed_vec, cand_vecs, m)
            result2 = top_m_cosine(seed_vec, cand_vecs, m)
            result3 = top_m_cosine(seed_vec, cand_vecs, m)
        
        # Results should be identical (deterministic)
        assert np.array_equal(result1, result2), "Numpy implementation should be deterministic"
        assert np.array_equal(result2, result3), "Numpy implementation should be deterministic"

class TestFAISSPerformance:
    """Test FAISS performance characteristics."""
    
    def test_top_m_cosine_performance_scaling(self, rng, performance_monitor, test_config):
        """Test performance scaling of top-M cosine similarity."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        # Test different problem sizes
        test_cases = [
            (100, 32, 10),
            (500, 32, 25),
            (1000, 32, 50),
            (2000, 64, 100),
        ]
        
        performance_results = {}
        
        for n_candidates, n_dims, m in test_cases:
            # Create test data
            seed_vec = rng.normal(0, 1, n_dims)
            seed_vec = seed_vec / np.linalg.norm(seed_vec)
            
            cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
            cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
            
            # Time the operation
            performance_monitor.reset()
            performance_monitor.start_monitoring()
            
            top_indices = top_m_cosine(seed_vec, cand_vecs, m)
            
            performance_monitor.stop_monitoring()
            
            # Validate result
            assert top_indices.shape == (m,), f"Invalid result shape for {n_candidates}×{n_dims}"
            
            key = f"{n_candidates}x{n_dims}_top{m}"
            performance_results[key] = {
                "time_ms": performance_monitor.elapsed_time_ms,
                "time_per_candidate_ms": performance_monitor.elapsed_time_ms / n_candidates,
                "memory_mb": performance_monitor.memory_delta_mb,
                "throughput_candidates_per_sec": n_candidates / (performance_monitor.elapsed_time_ms / 1000)
            }
        
        # Analyze and report performance
        for key, result in performance_results.items():
            print(f"FAISS {key}: {result['time_per_candidate_ms']:.4f} ms/candidate, "
                  f"throughput: {result['throughput_candidates_per_sec']:.0f} candidates/sec")
            
            # Check against performance thresholds
            threshold = test_config["performance_thresholds"]["faiss_search_ms_per_query"]
            if result['time_ms'] > threshold:
                pytest.warn(f"FAISS search time {result['time_ms']:.2f} ms exceeds threshold {threshold} ms for {key}")
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_faiss_vs_numpy_performance_comparison(self, rng, performance_monitor):
        """Compare FAISS vs numpy performance."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        # Large dataset for meaningful comparison
        n_candidates = 5000
        n_dims = 64
        m = 100
        
        seed_vec = rng.normal(0, 1, n_dims)
        seed_vec = seed_vec / np.linalg.norm(seed_vec)
        
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        # Time FAISS version
        performance_monitor.reset()
        performance_monitor.start_monitoring()
        faiss_result = top_m_cosine(seed_vec, cand_vecs, m)
        performance_monitor.stop_monitoring()
        faiss_time = performance_monitor.elapsed_time_ms
        
        # Time numpy version
        with patch('serendipity_engine_ui.engine.faiss_helper.have_faiss', return_value=False):
            performance_monitor.reset()
            performance_monitor.start_monitoring()
            numpy_result = top_m_cosine(seed_vec, cand_vecs, m)
            performance_monitor.stop_monitoring()
            numpy_time = performance_monitor.elapsed_time_ms
        
        # Performance comparison
        speedup = numpy_time / faiss_time if faiss_time > 0 else float('inf')
        
        print(f"Performance comparison ({n_candidates} candidates, top-{m}):")
        print(f"  FAISS: {faiss_time:.2f} ms")
        print(f"  NumPy: {numpy_time:.2f} ms")
        print(f"  Speedup: {speedup:.1f}x")
        
        # FAISS should generally be faster for large datasets
        if speedup < 0.5:  # Allow FAISS to be slower for small datasets due to overhead
            print(f"Warning: FAISS slower than numpy (speedup = {speedup:.1f}x)")
        
        # Results should be equivalent
        similarities = cand_vecs @ seed_vec
        faiss_sims = np.sort(similarities[faiss_result])[::-1]
        numpy_sims = np.sort(similarities[numpy_result])[::-1]
        assert np.allclose(faiss_sims, numpy_sims, atol=1e-10), "FAISS and numpy should give equivalent results"
    
    def test_faiss_memory_efficiency(self, rng, performance_monitor):
        """Test FAISS memory efficiency."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        # Test memory usage with increasing dataset sizes
        sizes = [(1000, 32), (2000, 64), (5000, 128)]
        baseline_memory = performance_monitor.peak_memory_mb
        
        for n_candidates, n_dims in sizes:
            seed_vec = rng.normal(0, 1, n_dims)
            seed_vec = seed_vec / np.linalg.norm(seed_vec)
            
            cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
            cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
            
            performance_monitor.reset()
            performance_monitor.start_monitoring()
            _ = top_m_cosine(seed_vec, cand_vecs, 50)
            performance_monitor.stop_monitoring()
            
            memory_growth = performance_monitor.peak_memory_mb - baseline_memory
            data_size_mb = (n_candidates * n_dims * 4) / (1024 * 1024)  # 4 bytes per float32
            
            print(f"FAISS memory for {n_candidates}×{n_dims}: "
                  f"{memory_growth:.1f} MB growth (data: {data_size_mb:.1f} MB)")
            
            # Memory growth should be reasonable (not excessive overhead)
            overhead_ratio = memory_growth / data_size_mb if data_size_mb > 0 else 0
            assert overhead_ratio < 10.0, f"FAISS memory overhead too high: {overhead_ratio:.1f}x data size"

class TestFAISSRobustness:
    """Test FAISS robustness and error handling."""
    
    def test_faiss_invalid_inputs(self, rng):
        """Test FAISS handling of invalid inputs."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        # Mismatched dimensions
        seed_2d = rng.normal(0, 1, 2)
        seed_2d = seed_2d / np.linalg.norm(seed_2d)
        
        cand_3d = rng.normal(0, 1, (10, 3))  # Different dimension
        cand_3d = cand_3d / np.linalg.norm(cand_3d, axis=1, keepdims=True)
        
        try:
            result = top_m_cosine(seed_2d, cand_3d, 5)
            # If it doesn't crash, result should be sensible
            assert isinstance(result, np.ndarray), "Should return array even with dimension mismatch"
        except (ValueError, AssertionError):
            # Acceptable to fail on dimension mismatch
            pass
    
    def test_faiss_extreme_values(self, rng):
        """Test FAISS with extreme values."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_dims = 16
        n_candidates = 20
        m = 5
        
        # Very large values
        seed_large = rng.normal(0, 1000, n_dims)
        seed_large = seed_large / np.linalg.norm(seed_large)  # Still unit vector
        
        cand_large = rng.normal(0, 1000, (n_candidates, n_dims))
        cand_large = cand_large / np.linalg.norm(cand_large, axis=1, keepdims=True)
        
        result_large = top_m_cosine(seed_large, cand_large, m)
        assert result_large.shape == (m,), "Should handle large values"
        assert np.all(np.isfinite(result_large)), "Should return finite indices with large values"
        
        # Very small values
        seed_small = rng.normal(0, 0.001, n_dims)
        seed_small = seed_small / np.linalg.norm(seed_small)
        
        cand_small = rng.normal(0, 0.001, (n_candidates, n_dims))
        cand_small = cand_small / np.linalg.norm(cand_small, axis=1, keepdims=True)
        
        result_small = top_m_cosine(seed_small, cand_small, m)
        assert result_small.shape == (m,), "Should handle small values"
        assert np.all(np.isfinite(result_small)), "Should return finite indices with small values"
    
    def test_faiss_non_unit_vectors(self, rng):
        """Test FAISS behavior with non-unit vectors."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        # Note: Function expects unit vectors, but test graceful handling
        n_dims = 8
        n_candidates = 15
        m = 5
        
        # Non-unit seed
        seed_non_unit = rng.normal(0, 1, n_dims) * 10  # Not normalized
        
        # Non-unit candidates
        cand_non_unit = rng.normal(0, 1, (n_candidates, n_dims)) * rng.uniform(0.1, 10, (n_candidates, 1))
        
        try:
            # Function may normalize internally or handle gracefully
            result = top_m_cosine(seed_non_unit, cand_non_unit, m)
            
            # If it works, should still return valid indices
            assert result.shape == (m,), "Should handle non-unit vectors"
            assert np.all(result >= 0) and np.all(result < n_candidates), "Indices should be valid"
        except Exception:
            # Acceptable to fail with non-unit vectors if documented behavior
            pass
    
    def test_faiss_data_type_handling(self, rng):
        """Test FAISS with different data types."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        n_dims = 6
        n_candidates = 12
        m = 4
        
        # Test with float64
        seed_f64 = rng.normal(0, 1, n_dims).astype(np.float64)
        seed_f64 = seed_f64 / np.linalg.norm(seed_f64)
        
        cand_f64 = rng.normal(0, 1, (n_candidates, n_dims)).astype(np.float64)
        cand_f64 = cand_f64 / np.linalg.norm(cand_f64, axis=1, keepdims=True)
        
        result_f64 = top_m_cosine(seed_f64, cand_f64, m)
        assert result_f64.shape == (m,), "Should handle float64"
        assert result_f64.dtype == np.int64, "Should return int64 indices"
        
        # Test with float32
        seed_f32 = seed_f64.astype(np.float32)
        cand_f32 = cand_f64.astype(np.float32)
        
        result_f32 = top_m_cosine(seed_f32, cand_f32, m)
        assert result_f32.shape == (m,), "Should handle float32"
        
        # Results should be similar (allowing for precision differences)
        similarities_f64 = cand_f64 @ seed_f64
        similarities_f32 = cand_f32.astype(np.float64) @ seed_f32.astype(np.float64)
        
        # Top similarities should be close
        top_sims_f64 = similarities_f64[result_f64]
        top_sims_f32 = similarities_f32[result_f32]
        
        # Allow some difference due to precision
        assert np.allclose(np.sort(top_sims_f64)[::-1], np.sort(top_sims_f32)[::-1], atol=1e-6), \
            "Float32 and float64 should give similar top similarities"

if __name__ == "__main__":
    pytest.main([__file__])
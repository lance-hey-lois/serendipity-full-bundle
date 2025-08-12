"""
Photonic Gaussian Boson Sampling (GBS) Tests
============================================

Comprehensive unit tests for the photonic GBS module, testing quantum photonics
simulations, PCA projections, mode activity analysis, and community density scoring.

Key Test Areas:
- GBS simulation accuracy and correctness
- PCA projection for photonic modes
- Mode activity computation and analysis
- Community density scoring validation
- Performance and scalability testing
- Fallback mechanisms when Strawberry Fields unavailable
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import warnings
from typing import List, Dict, Any, Tuple

# Mock Strawberry Fields if not available
try:
    import strawberryfields as sf
    STRAWBERRY_FIELDS_AVAILABLE = True
except ImportError:
    STRAWBERRY_FIELDS_AVAILABLE = False
    sf = MagicMock()

class TestGBSCore:
    """Test core GBS functionality and algorithms."""
    
    def test_have_photonics_detection(self):
        """Test detection of Strawberry Fields availability."""
        from serendipity_engine_ui.engine.photonic_gbs import have_photonics
        
        # Test actual availability
        actual_availability = have_photonics()
        assert isinstance(actual_availability, bool), "have_photonics should return boolean"
        
        # Test mocked unavailability
        with patch('builtins.__import__', side_effect=ImportError("No SF")):
            assert not have_photonics(), "Should return False when SF unavailable"
    
    def test_zscore_normalization_internal(self, rng):
        """Test internal z-score normalization function."""
        from serendipity_engine_ui.engine.photonic_gbs import _zscore
        
        # Test normal case
        x = rng.normal(10, 3, 1000)
        z = _zscore(x)
        
        assert abs(np.mean(z)) < 1e-10, f"Mean should be ~0, got {np.mean(z)}"
        assert abs(np.std(z) - 1) < 1e-10, f"Std should be ~1, got {np.std(z)}"
        
        # Test constant array (edge case)
        const_arr = np.full(50, 5.0)
        z_const = _zscore(const_arr)
        assert np.all(z_const == 0), "Z-score of constants should be zero"
        assert np.all(np.isfinite(z_const)), "Z-score should always be finite"
        
        # Test single element
        single = np.array([42.0])
        z_single = _zscore(single)
        assert z_single[0] == 0, "Z-score of single element should be zero"
    
    def test_pca_project_basic_functionality(self, rng):
        """Test PCA projection for photonic modes."""
        from serendipity_engine_ui.engine.photonic_gbs import pca_project
        
        # Create structured test data
        n_candidates = 100
        n_dims = 20
        modes = 4
        
        # Generate data with clear structure
        centers = rng.normal(0, 2, (3, n_dims))
        cand_vecs = []
        for _ in range(n_candidates):
            center = centers[rng.integers(0, 3)]
            noise = rng.normal(0, 0.5, n_dims)
            cand_vecs.append(center + noise)
        cand_vecs = np.array(cand_vecs)
        
        # Apply PCA projection
        Z, var = pca_project(cand_vecs, modes)
        
        # Validate outputs
        assert Z.shape == (n_candidates, modes), f"Z shape should be {(n_candidates, modes)}, got {Z.shape}"
        assert var.shape == (modes,), f"Variance shape should be {(modes,)}, got {var.shape}"
        assert np.all(var > 0), "All variances should be positive"
        assert np.all(np.isfinite(Z)), "All Z values should be finite"
        assert np.all(np.isfinite(var)), "All variance values should be finite"
    
    def test_pca_project_mathematical_properties(self, rng):
        """Test mathematical properties of PCA projection."""
        from serendipity_engine_ui.engine.photonic_gbs import pca_project
        
        # Create data with known principal components
        n_samples = 200
        n_dims = 10
        modes = 3
        
        # First component: strong signal
        comp1 = np.outer(np.linspace(-2, 2, n_samples), 
                        np.concatenate([np.ones(3), np.zeros(n_dims-3)]))
        # Second component: weaker signal  
        comp2 = np.outer(np.sin(np.linspace(0, 4*np.pi, n_samples)),
                        np.concatenate([np.zeros(3), np.ones(2), np.zeros(n_dims-5)]))
        # Add noise
        noise = rng.normal(0, 0.1, (n_samples, n_dims))
        
        X = comp1 + 0.5 * comp2 + noise
        
        Z, var = pca_project(X, modes)
        
        # First mode should capture most variance
        assert var[0] >= var[1], "First mode should have highest variance"
        assert var[1] >= var[2], "Second mode should have second highest variance"
        
        # Verify variance ordering is preserved
        for i in range(modes - 1):
            assert var[i] >= var[i+1], f"Variance should decrease: var[{i}]={var[i]} < var[{i+1}]={var[i+1]}"
    
    def test_pca_project_edge_cases(self, rng):
        """Test PCA projection edge cases."""
        from serendipity_engine_ui.engine.photonic_gbs import pca_project
        
        # Test single sample
        single_sample = rng.normal(0, 1, (1, 10))
        Z_single, var_single = pca_project(single_sample, 3)
        assert Z_single.shape == (1, 3)
        assert var_single.shape == (3,)
        assert np.all(np.isfinite(Z_single))
        assert np.all(var_single > 0)  # Should add epsilon for numerical stability
        
        # Test modes > dimensions
        small_data = rng.normal(0, 1, (20, 5))
        Z_over, var_over = pca_project(small_data, 10)
        # Should handle gracefully (likely clamp to available dimensions)
        assert Z_over.shape[1] <= 10
        assert var_over.shape[0] <= 10
        
        # Test constant data
        const_data = np.ones((50, 8))
        Z_const, var_const = pca_project(const_data, 4)
        assert np.all(np.isfinite(Z_const))
        assert np.all(var_const > 0)  # Numerical stability epsilon

class TestGBSSimulation:
    """Test GBS simulation and measurement."""
    
    @pytest.mark.skipif(not STRAWBERRY_FIELDS_AVAILABLE, reason="Strawberry Fields not available")
    def test_gbs_boost_with_strawberry_fields(self, rng):
        """Test GBS boost with actual Strawberry Fields."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        # Create test data
        n_candidates = 50
        n_dims = 16
        seed_vec = rng.normal(0, 1, n_dims)
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        
        # Run GBS boost
        community_scores = gbs_boost(
            seed_vec, cand_vecs, 
            modes=4, shots=50, cutoff=4, r_max=0.3
        )
        
        # Validate output
        assert community_scores.shape == (n_candidates,), f"Expected shape {(n_candidates,)}, got {community_scores.shape}"
        assert np.all(np.isfinite(community_scores)), "All community scores should be finite"
        
        # After z-score normalization, mean should be ~0, std ~1
        if n_candidates > 1:  # Need multiple samples for meaningful stats
            assert abs(np.mean(community_scores)) < 0.1, f"Mean should be ~0, got {np.mean(community_scores)}"
            assert abs(np.std(community_scores) - 1) < 0.2, f"Std should be ~1, got {np.std(community_scores)}"
    
    def test_gbs_boost_fallback_behavior(self, rng):
        """Test GBS fallback when Strawberry Fields unavailable."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        n_candidates = 30
        n_dims = 12
        seed_vec = rng.normal(0, 1, n_dims)
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        
        # Mock Strawberry Fields as unavailable
        with patch('serendipity_engine_ui.engine.photonic_gbs.have_photonics', return_value=False):
            community_scores = gbs_boost(seed_vec, cand_vecs)
            
            assert community_scores.shape == (n_candidates,)
            assert np.all(community_scores == 0), "Fallback should return zeros"
    
    def test_gbs_boost_parameter_sensitivity(self, rng):
        """Test sensitivity to GBS parameters."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        n_candidates = 25
        n_dims = 8
        seed_vec = rng.normal(0, 1, n_dims)
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        
        # Test different parameter settings
        param_sets = [
            {"modes": 2, "shots": 30, "cutoff": 3, "r_max": 0.2},
            {"modes": 4, "shots": 60, "cutoff": 5, "r_max": 0.4},
            {"modes": 6, "shots": 100, "cutoff": 6, "r_max": 0.6},
        ]
        
        results = []
        for params in param_sets:
            scores = gbs_boost(seed_vec, cand_vecs, **params)
            results.append({
                "params": params,
                "scores": scores,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "range": np.max(scores) - np.min(scores)
            })
        
        # All results should be valid
        for result in results:
            assert result["scores"].shape == (n_candidates,)
            assert np.all(np.isfinite(result["scores"]))
            
        # Different parameters should potentially give different results
        # (though due to randomness, this isn't guaranteed)
        print(f"GBS parameter sensitivity test:")
        for i, result in enumerate(results):
            print(f"  Params {i}: mean={result['mean']:.3f}, std={result['std']:.3f}")
    
    def test_gbs_boost_empty_input_handling(self):
        """Test GBS boost handles empty inputs gracefully."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        # Empty candidate set
        seed_vec = np.array([0.1, 0.2, 0.3])
        empty_cands = np.array([]).reshape(0, 3)
        
        scores_empty = gbs_boost(seed_vec, empty_cands)
        assert scores_empty.shape == (0,), "Empty input should return empty output"
        
        # Single candidate
        single_cand = np.array([[0.4, 0.5, 0.6]])
        scores_single = gbs_boost(seed_vec, single_cand)
        assert scores_single.shape == (1,)
        assert np.all(np.isfinite(scores_single))

class TestGBSPerformance:
    """Test GBS performance characteristics."""
    
    def test_gbs_boost_scaling_performance(self, rng, performance_monitor, test_config):
        """Test how GBS performance scales with problem size."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        candidate_counts = [20, 50, 100]
        dimension_sizes = [8, 16, 32]
        
        performance_results = {}
        
        for n_cands in candidate_counts:
            for n_dims in dimension_sizes:
                seed_vec = rng.normal(0, 1, n_dims)
                cand_vecs = rng.normal(0, 1, (n_cands, n_dims))
                
                performance_monitor.reset()
                performance_monitor.start_monitoring()
                
                # Use smaller parameters for faster testing
                scores = gbs_boost(
                    seed_vec, cand_vecs,
                    modes=min(4, min(n_dims, n_cands)), 
                    shots=30, cutoff=4, r_max=0.3
                )
                
                performance_monitor.stop_monitoring()
                
                key = f"cands_{n_cands}_dims_{n_dims}"
                performance_results[key] = {
                    "time_ms": performance_monitor.elapsed_time_ms,
                    "memory_mb": performance_monitor.memory_delta_mb,
                    "time_per_candidate": performance_monitor.elapsed_time_ms / n_cands,
                    "valid_output": scores.shape == (n_cands,) and np.all(np.isfinite(scores))
                }
        
        # Analyze performance trends
        for key, result in performance_results.items():
            print(f"GBS {key}: {result['time_per_candidate']:.2f} ms/candidate, "
                  f"memory: {result['memory_mb']:.1f} MB")
            
            # Check against performance thresholds
            threshold = test_config["performance_thresholds"]["gbs_boost_ms_per_candidate"]
            if result['time_per_candidate'] > threshold:
                pytest.warn(f"GBS performance {result['time_per_candidate']:.2f} ms/candidate "
                           f"exceeds threshold {threshold} ms/candidate for {key}")
    
    def test_gbs_boost_memory_efficiency(self, rng, performance_monitor):
        """Test memory efficiency of GBS computations."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        # Test with various problem sizes
        sizes = [(50, 16), (100, 32), (200, 64)]
        
        baseline_memory = performance_monitor.peak_memory_mb
        
        for n_cands, n_dims in sizes:
            seed_vec = rng.normal(0, 1, n_dims)
            cand_vecs = rng.normal(0, 1, (n_cands, n_dims))
            
            performance_monitor.reset()
            performance_monitor.start_monitoring()
            
            _ = gbs_boost(seed_vec, cand_vecs, modes=4, shots=50, cutoff=4)
            
            performance_monitor.stop_monitoring()
            
            memory_growth = performance_monitor.peak_memory_mb - baseline_memory
            
            print(f"GBS memory for {n_cands}×{n_dims}: {memory_growth:.2f} MB growth")
            
            # Memory growth should be reasonable
            assert memory_growth < 200, f"Excessive memory growth: {memory_growth} MB for {n_cands}×{n_dims}"
    
    def test_gbs_numerical_stability(self, rng):
        """Test numerical stability of GBS computations."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        # Test with various numerical challenges
        test_cases = [
            {
                "name": "large_values",
                "seed": rng.normal(0, 100, 8),
                "cands": rng.normal(0, 100, (30, 8))
            },
            {
                "name": "small_values", 
                "seed": rng.normal(0, 0.01, 8),
                "cands": rng.normal(0, 0.01, (30, 8))
            },
            {
                "name": "mixed_scales",
                "seed": np.concatenate([rng.normal(0, 100, 4), rng.normal(0, 0.01, 4)]),
                "cands": np.concatenate([
                    rng.normal(0, 100, (30, 4)), 
                    rng.normal(0, 0.01, (30, 4))
                ], axis=1)
            }
        ]
        
        for case in test_cases:
            scores = gbs_boost(case["seed"], case["cands"], modes=4, shots=40, cutoff=4)
            
            assert np.all(np.isfinite(scores)), f"GBS scores should be finite for {case['name']}"
            assert scores.shape == (case["cands"].shape[0],), f"Wrong shape for {case['name']}"
            
            print(f"GBS numerical stability test '{case['name']}': "
                  f"range=[{np.min(scores):.3f}, {np.max(scores):.3f}], "
                  f"std={np.std(scores):.3f}")

class TestGBSIntegration:
    """Test GBS integration with other system components."""
    
    def test_gbs_with_faiss_preprocessing(self, rng):
        """Test GBS integration with FAISS preprocessing."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine, have_faiss
        
        # Create test data
        n_candidates = 200
        n_dims = 32
        seed_vec = rng.normal(0, 1, n_dims)
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        
        # Normalize for cosine similarity
        seed_unit = seed_vec / np.linalg.norm(seed_vec)
        cand_unit = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        # Get top candidates via FAISS/cosine similarity
        top_k = 50
        if have_faiss():
            top_indices = top_m_cosine(seed_unit, cand_unit, top_k)
        else:
            # Fallback to numpy
            similarities = cand_unit @ seed_unit
            top_indices = np.argsort(-similarities)[:top_k]
        
        # Apply GBS to top candidates
        top_cand_vecs = cand_vecs[top_indices]
        gbs_scores = gbs_boost(seed_vec, top_cand_vecs, modes=4, shots=60)
        
        # Validate integration results
        assert gbs_scores.shape == (top_k,), f"GBS scores shape should be {(top_k,)}"
        assert np.all(np.isfinite(gbs_scores)), "All GBS scores should be finite"
        
        # Combined scores should be reasonable
        if np.std(gbs_scores) > 0:  # If GBS provides differentiation
            # Higher GBS scores should correspond to meaningful community density
            top_gbs_idx = np.argmax(gbs_scores)
            assert 0 <= top_gbs_idx < top_k, "Top GBS index should be valid"
    
    def test_gbs_consistency_across_runs(self, rng):
        """Test GBS consistency across multiple runs."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        # Fixed test data
        seed_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        cand_vecs = rng.normal(0, 1, (20, 6))
        
        # Run GBS multiple times with same parameters
        n_runs = 5
        results = []
        
        for run in range(n_runs):
            scores = gbs_boost(
                seed_vec, cand_vecs,
                modes=3, shots=100, cutoff=4, r_max=0.4
            )
            results.append(scores)
        
        # Analyze consistency
        results = np.array(results)  # Shape: (n_runs, n_candidates)
        
        # Calculate run-to-run variations
        mean_scores = np.mean(results, axis=0)
        std_scores = np.std(results, axis=0)
        cv_scores = std_scores / (np.abs(mean_scores) + 1e-9)  # Coefficient of variation
        
        print(f"GBS consistency: mean CV = {np.mean(cv_scores):.3f}, max CV = {np.max(cv_scores):.3f}")
        
        # Due to quantum sampling, some variation is expected
        # But extreme variation might indicate implementation issues
        assert np.mean(cv_scores) < 2.0, f"Mean coefficient of variation too high: {np.mean(cv_scores)}"
        
        # All runs should produce valid outputs
        for run_idx, scores in enumerate(results):
            assert np.all(np.isfinite(scores)), f"Run {run_idx} produced non-finite scores"

class TestGBSQuantumProperties:
    """Test quantum-specific properties of GBS implementation."""
    
    @pytest.mark.skipif(not STRAWBERRY_FIELDS_AVAILABLE, reason="Strawberry Fields not available")
    def test_gbs_quantum_state_properties(self):
        """Test that GBS produces valid quantum states."""
        # This test requires deeper integration with SF internals
        # For now, we test the interface and statistical properties
        
        import strawberryfields as sf
        from strawberryfields import ops as O
        
        # Test basic GBS circuit construction
        modes = 4
        prog = sf.Program(modes)
        
        with prog.context as q:
            # Test squeezing operations
            for i in range(modes):
                O.Sgate(0.2) | q[i]  # Small squeezing for stability
            O.MeasureFock() | q
        
        # Test program compilation
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        
        # Should not raise exceptions
        try:
            result = eng.run(prog, shots=10)
            samples = result.samples
            
            # Validate sample properties
            assert samples.shape == (10, modes), f"Expected shape (10, {modes}), got {samples.shape}"
            assert np.all(samples >= 0), "Photon counts should be non-negative integers"
            assert samples.dtype in [np.int64, np.int32, int], f"Samples should be integers, got {samples.dtype}"
            
        except Exception as e:
            pytest.fail(f"GBS quantum state preparation failed: {e}")
    
    def test_gbs_squeezing_parameter_effects(self, rng):
        """Test effects of different squeezing parameters."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        seed_vec = rng.normal(0, 1, 6)
        cand_vecs = rng.normal(0, 1, (25, 6))
        
        # Test different r_max values (squeezing strength)
        r_values = [0.1, 0.3, 0.5, 0.7]
        results = {}
        
        for r_max in r_values:
            scores = gbs_boost(
                seed_vec, cand_vecs,
                modes=3, shots=80, cutoff=5, r_max=r_max
            )
            
            results[r_max] = {
                "scores": scores,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "range": np.max(scores) - np.min(scores)
            }
        
        # Validate all results
        for r_max, result in results.items():
            assert np.all(np.isfinite(result["scores"])), f"Invalid scores for r_max={r_max}"
            print(f"GBS r_max={r_max}: mean={result['mean']:.3f}, "
                  f"std={result['std']:.3f}, range={result['range']:.3f}")
        
        # Different squeezing should potentially give different statistics
        # (though this is probabilistic and not guaranteed)
        stds = [results[r]["std"] for r in r_values]
        print(f"GBS squeezing effect on std deviation: {stds}")

if __name__ == "__main__":
    pytest.main([__file__])
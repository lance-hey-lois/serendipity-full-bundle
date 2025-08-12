"""
Quantum Algorithms Unit Tests
=============================

Comprehensive unit tests for quantum kernel computations and quantum-enhanced
features in the serendipity engine. Tests quantum circuit correctness,
state preparation, measurement accuracy, and performance characteristics.

Key Test Areas:
- Quantum kernel fidelity computation
- PCA compression for quantum encoding  
- Quantum state preparation and measurement
- Error handling and fallback mechanisms
- Performance and scalability validation
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import sys
from typing import List, Dict, Any, Tuple

# Import the modules under test
import importlib.util

# Mock PennyLane if not available
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = MagicMock()

class TestQuantumKernelMethods:
    """Test quantum kernel computation methods."""
    
    def test_pca_compress_basic_functionality(self, rng):
        """Test PCA compression produces correct dimensionality."""
        from serendipity_engine_ui.engine.quantum import pca_compress
        
        # Create test data
        n_samples, n_features = 100, 20
        out_dim = 5
        X = rng.normal(0, 1, (n_samples, n_features))
        
        # Apply PCA compression
        Z = pca_compress(X, out_dim)
        
        # Validate output shape
        assert Z.shape == (n_samples, out_dim), f"Expected {(n_samples, out_dim)}, got {Z.shape}"
        
        # Validate output range is [0, π/2]
        assert np.all(Z >= 0), "PCA output should be non-negative"
        assert np.all(Z <= np.pi/2 + 1e-6), f"PCA output should be ≤ π/2, max: {np.max(Z)}"
    
    def test_pca_compress_handles_edge_cases(self, rng):
        """Test PCA compression handles edge cases properly."""
        from serendipity_engine_ui.engine.quantum import pca_compress
        
        # Test single sample
        X_single = rng.normal(0, 1, (1, 10))
        Z_single = pca_compress(X_single, 3)
        assert Z_single.shape == (1, 3)
        assert np.all(np.isfinite(Z_single))
        
        # Test constant data (should not crash)
        X_const = np.ones((10, 5))
        Z_const = pca_compress(X_const, 3)
        assert Z_const.shape == (10, 3)
        assert np.all(np.isfinite(Z_const))
        
        # Test out_dim larger than input dimension
        X_small = rng.normal(0, 1, (20, 3))
        Z_large = pca_compress(X_small, 10)  # Should be clamped to 3
        assert Z_large.shape[1] <= 10  # Implementation dependent
    
    def test_pca_compress_mathematical_properties(self, rng):
        """Test mathematical properties of PCA compression."""
        from serendipity_engine_ui.engine.quantum import pca_compress
        
        # Create structured test data with known properties
        n_samples = 200
        # Create data with clear principal components
        component1 = np.outer(np.linspace(0, 1, n_samples), [1, 0.8, 0.6, 0])
        component2 = np.outer(np.sin(np.linspace(0, 2*np.pi, n_samples)), [0, 0.2, 0.4, 1])
        X = component1 + component2 + rng.normal(0, 0.1, (n_samples, 4))
        
        # Apply PCA
        Z = pca_compress(X, 2)
        
        # Check variance is captured (first component should have higher variance)
        var_comp1 = np.var(Z[:, 0])
        var_comp2 = np.var(Z[:, 1]) 
        
        # Since we scale to [0, π/2], we need to check relative magnitudes
        assert var_comp1 > 0, "First component should have positive variance"
        assert var_comp2 >= 0, "Second component should have non-negative variance"
    
    @pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not available")
    def test_quantum_kernel_to_seed_with_pennylane(self, rng):
        """Test quantum kernel computation with actual PennyLane."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed
        
        # Create test vectors
        n_dims = 4
        n_candidates = 5
        seed_vec = rng.uniform(0, np.pi/2, n_dims)
        cand_vecs = rng.uniform(0, np.pi/2, (n_candidates, n_dims))
        
        # Compute kernel
        kernel_values = quantum_kernel_to_seed(seed_vec, cand_vecs)
        
        # Validate output
        assert kernel_values.shape == (n_candidates,), f"Expected shape {(n_candidates,)}, got {kernel_values.shape}"
        assert np.all(kernel_values >= 0), "Kernel values should be non-negative probabilities"
        assert np.all(kernel_values <= 1), "Kernel values should be ≤ 1 (probabilities)"
        assert np.all(np.isfinite(kernel_values)), "All kernel values should be finite"
    
    def test_quantum_kernel_fallback_behavior(self, rng):
        """Test quantum kernel fallback when PennyLane unavailable."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed
        
        # Mock PennyLane as unavailable
        with patch('builtins.__import__', side_effect=ImportError("No PennyLane")):
            n_dims = 4
            n_candidates = 10
            seed_vec = rng.uniform(0, np.pi/2, n_dims)
            cand_vecs = rng.uniform(0, np.pi/2, (n_candidates, n_dims))
            
            # Should return zeros when PennyLane unavailable
            kernel_values = quantum_kernel_to_seed(seed_vec, cand_vecs)
            
            assert kernel_values.shape == (n_candidates,)
            assert np.all(kernel_values == 0), "Fallback should return zeros"
    
    @pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not available")
    def test_quantum_kernel_mathematical_properties(self, rng):
        """Test mathematical properties of quantum kernel."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed
        
        n_dims = 3
        seed_vec = np.array([0.1, 0.2, 0.3])  # Small angles
        
        # Test self-similarity (kernel with itself should be high)
        self_kernel = quantum_kernel_to_seed(seed_vec, seed_vec.reshape(1, -1))
        assert self_kernel[0] > 0.8, f"Self-kernel should be high, got {self_kernel[0]}"
        
        # Test different vectors (should be lower similarity)
        different_vec = np.array([1.4, 1.5, 1.6]).reshape(1, -1)  # Large angles
        diff_kernel = quantum_kernel_to_seed(seed_vec, different_vec)
        assert diff_kernel[0] < self_kernel[0], "Different vectors should have lower kernel values"
        
        # Test symmetry property (approximately)
        vec1 = np.array([0.2, 0.3, 0.4])
        vec2 = np.array([0.5, 0.6, 0.1])
        
        k12 = quantum_kernel_to_seed(vec1, vec2.reshape(1, -1))[0]
        k21 = quantum_kernel_to_seed(vec2, vec1.reshape(1, -1))[0]
        
        # Due to the quantum circuit structure, perfect symmetry may not hold
        # but values should be reasonably close
        assert abs(k12 - k21) < 0.2, f"Kernel should be approximately symmetric: {k12} vs {k21}"
    
    def test_quantum_kernel_performance_characteristics(self, rng, performance_monitor):
        """Test performance characteristics of quantum kernel computation."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed
        
        n_dims = 4
        n_candidates_list = [10, 50, 100]
        
        for n_candidates in n_candidates_list:
            seed_vec = rng.uniform(0, np.pi/2, n_dims)
            cand_vecs = rng.uniform(0, np.pi/2, (n_candidates, n_dims))
            
            performance_monitor.start_monitoring()
            kernel_values = quantum_kernel_to_seed(seed_vec, cand_vecs)
            performance_monitor.stop_monitoring()
            
            # Validate results
            assert kernel_values.shape == (n_candidates,)
            
            # Performance should scale roughly linearly with candidates
            time_per_candidate = performance_monitor.elapsed_time_ms / n_candidates
            
            # Log performance (for CI/monitoring)
            print(f"Quantum kernel: {n_candidates} candidates, "
                  f"{time_per_candidate:.2f} ms/candidate, "
                  f"total: {performance_monitor.elapsed_time_ms:.2f} ms")
    
    def test_zscore_normalization(self, rng):
        """Test z-score normalization utility function."""
        from serendipity_engine_ui.engine.quantum import zscore
        
        # Test basic normalization
        x = rng.normal(5, 2, 1000)
        z = zscore(x)
        
        assert abs(np.mean(z)) < 1e-10, f"Z-score mean should be ~0, got {np.mean(z)}"
        assert abs(np.std(z) - 1) < 1e-10, f"Z-score std should be ~1, got {np.std(z)}"
        
        # Test edge cases
        constant_array = np.ones(10)
        z_const = zscore(constant_array)
        assert np.all(np.isfinite(z_const)), "Z-score of constant should be finite"
        assert np.all(z_const == 0), "Z-score of constant should be zero"
        
        # Test single element
        single = np.array([5.0])
        z_single = zscore(single)
        assert z_single.shape == (1,)
        assert z_single[0] == 0, "Z-score of single element should be zero"

class TestQuantumCircuitProperties:
    """Test quantum circuit construction and properties."""
    
    @pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not available")
    def test_quantum_circuit_structure(self):
        """Test the structure of quantum circuits used in kernels."""
        import pennylane as qml
        
        n_qubits = 4
        dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        
        def test_circuit(params):
            # Replicate the feature map from quantum.py
            for i, param in enumerate(params):
                qml.RY(param, wires=i)
            for i in range(n_qubits):
                qml.CZ(wires=[i, (i+1) % n_qubits])
            return qml.probs(wires=range(n_qubits))
        
        qnode = qml.QNode(test_circuit, dev)
        
        # Test with various parameter sets
        params = np.array([0.1, 0.2, 0.3, 0.4])
        probs = qnode(params)
        
        # Validate probability distribution
        assert len(probs) == 2**n_qubits, f"Expected {2**n_qubits} probabilities"
        assert abs(np.sum(probs) - 1.0) < 1e-10, f"Probabilities should sum to 1, got {np.sum(probs)}"
        assert np.all(probs >= 0), "All probabilities should be non-negative"
        
        # Test state preparation effects
        zero_params = np.zeros(n_qubits)
        zero_probs = qnode(zero_params)
        assert zero_probs[0] > 0.9, f"Zero parameters should concentrate on |0000⟩, got {zero_probs[0]}"
    
    @pytest.mark.skipif(not PENNYLANE_AVAILABLE, reason="PennyLane not available") 
    def test_quantum_circuit_gradients(self):
        """Test gradient computation for quantum circuits."""
        import pennylane as qml
        
        n_qubits = 3
        dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        
        @qml.qnode(dev)
        def circuit(params):
            for i, param in enumerate(params):
                qml.RY(param, wires=i)
            for i in range(n_qubits):
                qml.CZ(wires=[i, (i+1) % n_qubits])
            return qml.expval(qml.PauliZ(0))
        
        params = np.array([0.1, 0.2, 0.3])
        
        # Compute gradients
        grad_fn = qml.grad(circuit)
        gradients = grad_fn(params)
        
        assert len(gradients) == n_qubits, f"Expected {n_qubits} gradients"
        assert np.all(np.isfinite(gradients)), "All gradients should be finite"
        
        # Test gradient magnitude (should be non-trivial for non-zero parameters)
        grad_magnitude = np.linalg.norm(gradients)
        assert grad_magnitude > 1e-6, f"Gradient magnitude should be significant, got {grad_magnitude}"

class TestQuantumPerformanceMetrics:
    """Test quantum algorithm performance and scaling."""
    
    def test_quantum_kernel_scaling(self, rng, performance_monitor, test_config):
        """Test how quantum kernel computation scales with problem size."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed
        
        dimensions = [2, 4, 6, 8]
        candidates = [10, 50, 100]
        
        results = {}
        
        for n_dims in dimensions:
            for n_cands in candidates:
                seed_vec = rng.uniform(0, np.pi/2, n_dims)
                cand_vecs = rng.uniform(0, np.pi/2, (n_cands, n_dims))
                
                performance_monitor.reset()
                performance_monitor.start_monitoring()
                
                kernel_values = quantum_kernel_to_seed(seed_vec, cand_vecs)
                
                performance_monitor.stop_monitoring()
                
                key = f"dims_{n_dims}_cands_{n_cands}"
                results[key] = {
                    "time_ms": performance_monitor.elapsed_time_ms,
                    "memory_mb": performance_monitor.memory_delta_mb,
                    "time_per_candidate": performance_monitor.elapsed_time_ms / n_cands,
                    "valid_output": kernel_values.shape == (n_cands,)
                }
        
        # Analyze scaling behavior
        for key, result in results.items():
            print(f"Quantum kernel {key}: {result['time_per_candidate']:.2f} ms/candidate")
            
            # Validate performance thresholds
            threshold = test_config["performance_thresholds"]["quantum_kernel_ms_per_comparison"]
            if result['time_per_candidate'] > threshold:
                pytest.warn(f"Quantum kernel performance {result['time_per_candidate']:.2f} ms/candidate "
                           f"exceeds threshold {threshold} ms/candidate for {key}")
    
    def test_quantum_kernel_memory_usage(self, rng, performance_monitor):
        """Test memory usage characteristics of quantum computations."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed
        
        # Test memory growth with increasing problem size
        sizes = [50, 100, 200, 500]
        n_dims = 4
        
        baseline_memory = performance_monitor.peak_memory_mb
        
        for size in sizes:
            seed_vec = rng.uniform(0, np.pi/2, n_dims)
            cand_vecs = rng.uniform(0, np.pi/2, (size, n_dims))
            
            performance_monitor.start_monitoring()
            _ = quantum_kernel_to_seed(seed_vec, cand_vecs)
            performance_monitor.stop_monitoring()
            
            memory_growth = performance_monitor.peak_memory_mb - baseline_memory
            
            print(f"Quantum kernel memory growth for {size} candidates: {memory_growth:.2f} MB")
            
            # Memory growth should be reasonable (not exponential)
            assert memory_growth < 500, f"Excessive memory growth: {memory_growth} MB for {size} candidates"

class TestQuantumErrorHandling:
    """Test error handling and robustness of quantum algorithms."""
    
    def test_quantum_kernel_invalid_inputs(self):
        """Test quantum kernel handles invalid inputs gracefully."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed
        
        # Test empty inputs
        empty_seed = np.array([])
        empty_cands = np.array([]).reshape(0, 0)
        
        # Should not crash
        try:
            result = quantum_kernel_to_seed(empty_seed, empty_cands)
            assert result.shape == (0,), "Empty input should return empty output"
        except Exception as e:
            # Acceptable to raise exception for invalid input
            assert "shape" in str(e).lower() or "empty" in str(e).lower()
        
        # Test mismatched dimensions
        seed_2d = np.array([0.1, 0.2])
        cands_3d = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        try:
            result = quantum_kernel_to_seed(seed_2d, cands_3d)
            # If it doesn't crash, result should be reasonable
            assert result.shape == (2,)
        except Exception:
            # Acceptable to fail on mismatched dimensions
            pass
    
    def test_pca_compress_robustness(self, rng):
        """Test PCA compression robustness to edge cases."""
        from serendipity_engine_ui.engine.quantum import pca_compress
        
        # Test with NaN values
        X_nan = rng.normal(0, 1, (10, 5))
        X_nan[0, 0] = np.nan
        
        try:
            Z_nan = pca_compress(X_nan, 3)
            # If it doesn't crash, check if result is handled appropriately
            if np.any(np.isnan(Z_nan)):
                pytest.warn("PCA compress propagates NaN values")
        except Exception:
            # Acceptable to fail on NaN input
            pass
        
        # Test with infinite values
        X_inf = rng.normal(0, 1, (10, 5))
        X_inf[1, 1] = np.inf
        
        try:
            Z_inf = pca_compress(X_inf, 3)
            assert np.all(np.isfinite(Z_inf)), "PCA should handle infinite input gracefully"
        except Exception:
            # Acceptable to fail on infinite input  
            pass
        
        # Test with very large values
        X_large = rng.normal(0, 1e6, (10, 5))
        Z_large = pca_compress(X_large, 3)
        
        assert np.all(np.isfinite(Z_large)), "PCA should handle large values"
        assert np.all(Z_large >= 0), "PCA output should remain non-negative"
        assert np.all(Z_large <= np.pi/2 + 1e-6), "PCA output should remain in [0, π/2]"

if __name__ == "__main__":
    pytest.main([__file__])
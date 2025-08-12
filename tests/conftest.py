"""
Comprehensive Test Configuration for Quantum-Enhanced Serendipity Engine
========================================================================

This module provides shared fixtures, utilities, and configuration for testing
the quantum-enhanced serendipity discovery system. It supports quantum-specific
testing methodologies and performance benchmarking.

Key Features:
- Quantum algorithm testing with controlled environments
- Performance benchmarking with statistical validation
- Mock data generation for reproducible tests
- Cross-platform compatibility testing
- Memory usage monitoring and validation
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import time
import psutil
import os
from typing import Dict, List, Any, Tuple, Generator
from unittest.mock import MagicMock, patch
import warnings

# Suppress known warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration and constants."""
    return {
        "random_seed": 42,
        "small_dataset_size": 100,
        "medium_dataset_size": 1000,
        "large_dataset_size": 10000,
        "vector_dimensions": [16, 32, 64, 128],
        "performance_thresholds": {
            "quantum_kernel_ms_per_comparison": 50,
            "gbs_boost_ms_per_candidate": 10,
            "faiss_search_ms_per_query": 5,
            "memory_growth_mb_limit": 100,
        },
        "tolerance": {
            "float_comparison": 1e-6,
            "quantum_fidelity": 1e-4,
            "statistical_significance": 0.05,
        }
    }

@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(seed=42)

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

class MockQuantumDevice:
    """Mock quantum device for testing without actual quantum hardware."""
    
    def __init__(self, wires: int, shots: int = None):
        self.wires = wires
        self.shots = shots
        self._state = np.zeros(2**wires, dtype=complex)
        self._state[0] = 1.0  # |0...0⟩ state
    
    def reset(self):
        """Reset to computational basis state |0...0⟩."""
        self._state = np.zeros_like(self._state)
        self._state[0] = 1.0
    
    def apply_rotation(self, angle: float, wire: int):
        """Apply rotation gate for testing."""
        # Simplified mock rotation for testing
        self._state = self._state * np.exp(1j * angle * 0.1)
    
    def measure(self) -> np.ndarray:
        """Mock measurement returning probabilities."""
        probs = np.abs(self._state)**2
        return probs / np.sum(probs)

@pytest.fixture
def mock_quantum_device():
    """Provide mock quantum device for testing."""
    return MockQuantumDevice

@pytest.fixture
def sample_vectors(rng):
    """Generate sample embedding vectors for testing."""
    def _generate(n_samples: int = 100, n_dims: int = 64, n_clusters: int = 5):
        """Generate clustered sample vectors."""
        # Create cluster centers
        centers = rng.normal(0, 2, (n_clusters, n_dims))
        
        # Assign samples to clusters
        cluster_assignments = rng.integers(0, n_clusters, n_samples)
        
        # Generate samples around centers with noise
        vectors = []
        for i in range(n_samples):
            cluster_id = cluster_assignments[i]
            noise = rng.normal(0, 0.5, n_dims)
            vector = centers[cluster_id] + noise
            vectors.append(vector)
        
        return np.array(vectors), cluster_assignments, centers
    
    return _generate

@pytest.fixture
def sample_people_data(sample_vectors, rng):
    """Generate realistic people data for testing."""
    def _generate(n_people: int = 100, n_dims: int = 64):
        vectors, clusters, centers = sample_vectors(n_people, n_dims, 5)
        
        people = []
        for i, (vec, cluster) in enumerate(zip(vectors, clusters)):
            person = {
                "id": f"person_{i:04d}",
                "vec": vec.tolist(),
                "novelty": rng.random(),
                "availability": rng.random(),
                "pathTrust": rng.random(),
                "cluster": int(cluster),
                "metadata": {
                    "skills": rng.choice(["python", "ml", "quantum", "data"], 
                                       size=rng.integers(1, 4), replace=False).tolist(),
                    "experience": rng.integers(0, 20),
                }
            }
            people.append(person)
        
        return people, vectors, clusters, centers
    
    return _generate

@pytest.fixture
def performance_monitor():
    """Monitor system performance during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.reset()
        
        def reset(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.process = psutil.Process()
        
        def start_monitoring(self):
            self.start_time = time.perf_counter()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def stop_monitoring(self):
            self.end_time = time.perf_counter()
            self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        @property
        def elapsed_time_ms(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return (self.end_time - self.start_time) * 1000
        
        @property
        def memory_delta_mb(self) -> float:
            if self.start_memory is None or self.end_memory is None:
                return 0.0
            return self.end_memory - self.start_memory
        
        @property
        def peak_memory_mb(self) -> float:
            return self.process.memory_info().rss / 1024 / 1024
    
    return PerformanceMonitor()

@pytest.fixture
def quantum_test_environment():
    """Set up quantum testing environment with fallbacks."""
    class QuantumTestEnv:
        def __init__(self):
            self.has_pennylane = self._check_pennylane()
            self.has_strawberryfields = self._check_strawberryfields()
            self.mock_mode = not (self.has_pennylane and self.has_strawberryfields)
        
        def _check_pennylane(self) -> bool:
            try:
                import pennylane as qml
                return True
            except ImportError:
                return False
        
        def _check_strawberryfields(self) -> bool:
            try:
                import strawberryfields as sf
                return True
            except ImportError:
                return False
        
        def require_pennylane(self):
            if not self.has_pennylane:
                pytest.skip("PennyLane not available")
        
        def require_strawberryfields(self):
            if not self.has_strawberryfields:
                pytest.skip("Strawberry Fields not available")
        
        def get_mock_quantum_result(self, shape: Tuple[int, ...]) -> np.ndarray:
            """Generate mock quantum results for testing."""
            return np.random.random(shape)
    
    return QuantumTestEnv()

@pytest.fixture
def statistical_validator():
    """Validate statistical properties of results."""
    class StatisticalValidator:
        def __init__(self, significance_level: float = 0.05):
            self.significance_level = significance_level
        
        def validate_distribution(self, samples: np.ndarray, 
                                expected_mean: float = None,
                                expected_std: float = None) -> Dict[str, Any]:
            """Validate statistical properties of sample distribution."""
            from scipy import stats
            
            results = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "min": float(np.min(samples)),
                "max": float(np.max(samples)),
                "n_samples": len(samples),
            }
            
            # Test normality
            if len(samples) > 8:  # Minimum for Shapiro-Wilk
                stat, p_value = stats.shapiro(samples)
                results["normality_test"] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "is_normal": p_value > self.significance_level
                }
            
            # Test expected mean if provided
            if expected_mean is not None:
                t_stat, p_value = stats.ttest_1samp(samples, expected_mean)
                results["mean_test"] = {
                    "expected": expected_mean,
                    "statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significantly_different": p_value < self.significance_level
                }
            
            return results
        
        def validate_quantum_fidelity(self, state1: np.ndarray, state2: np.ndarray,
                                    min_fidelity: float = 0.9) -> Dict[str, Any]:
            """Validate quantum state fidelity."""
            # Normalize states
            state1 = state1 / np.linalg.norm(state1)
            state2 = state2 / np.linalg.norm(state2)
            
            # Calculate fidelity |⟨ψ₁|ψ₂⟩|²
            fidelity = abs(np.vdot(state1, state2))**2
            
            return {
                "fidelity": float(fidelity),
                "meets_threshold": fidelity >= min_fidelity,
                "threshold": min_fidelity
            }
    
    return StatisticalValidator()

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically set up test environment for each test."""
    # Set reproducible random seeds
    np.random.seed(42)
    
    # Mock external dependencies that might not be available
    mock_modules = []
    
    try:
        import pennylane
    except ImportError:
        mock_modules.append("pennylane")
    
    try:
        import strawberryfields
    except ImportError:
        mock_modules.append("strawberryfields")
    
    try:
        import faiss
    except ImportError:
        mock_modules.append("faiss")
    
    # Apply mocks if needed
    for module in mock_modules:
        monkeypatch.setattr(f"sys.modules['{module}']", MagicMock())

# Utility functions for test data generation and validation

def generate_quantum_test_cases(n_cases: int = 10) -> List[Dict[str, Any]]:
    """Generate test cases for quantum algorithm validation."""
    rng = np.random.default_rng(42)
    test_cases = []
    
    for i in range(n_cases):
        n_qubits = rng.integers(2, 6)
        angles = rng.uniform(0, np.pi, n_qubits)
        
        test_cases.append({
            "case_id": i,
            "n_qubits": n_qubits,
            "rotation_angles": angles,
            "expected_properties": {
                "trace": 1.0,  # Density matrix trace
                "hermitian": True,
                "positive_semidefinite": True,
            }
        })
    
    return test_cases

def assert_quantum_properties(density_matrix: np.ndarray, tolerance: float = 1e-6):
    """Assert essential quantum state properties."""
    # Trace should be 1
    trace = np.trace(density_matrix)
    assert abs(trace - 1.0) < tolerance, f"Trace {trace} != 1"
    
    # Should be Hermitian
    assert np.allclose(density_matrix, density_matrix.conj().T, atol=tolerance), \
        "Matrix is not Hermitian"
    
    # Should be positive semidefinite
    eigenvals = np.linalg.eigvals(density_matrix)
    assert np.all(eigenvals >= -tolerance), \
        f"Matrix not positive semidefinite, min eigenvalue: {np.min(eigenvals)}"

def create_benchmark_suite() -> Dict[str, callable]:
    """Create comprehensive benchmark suite for performance testing."""
    
    def benchmark_quantum_kernel(n_vectors: int, n_dims: int) -> Dict[str, float]:
        """Benchmark quantum kernel computation."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed
        
        rng = np.random.default_rng(42)
        seed_vec = rng.random(n_dims)
        cand_vecs = rng.random((n_vectors, n_dims))
        
        start_time = time.perf_counter()
        result = quantum_kernel_to_seed(seed_vec, cand_vecs)
        end_time = time.perf_counter()
        
        return {
            "total_time_ms": (end_time - start_time) * 1000,
            "time_per_vector_ms": ((end_time - start_time) * 1000) / n_vectors,
            "result_shape": result.shape,
            "result_range": [float(np.min(result)), float(np.max(result))]
        }
    
    def benchmark_gbs_boost(n_candidates: int, n_dims: int) -> Dict[str, float]:
        """Benchmark GBS boost computation."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        rng = np.random.default_rng(42)
        seed_vec = rng.random(n_dims)
        cand_vecs = rng.random((n_candidates, n_dims))
        
        start_time = time.perf_counter()
        result = gbs_boost(seed_vec, cand_vecs, modes=4, shots=60)
        end_time = time.perf_counter()
        
        return {
            "total_time_ms": (end_time - start_time) * 1000,
            "time_per_candidate_ms": ((end_time - start_time) * 1000) / n_candidates,
            "result_shape": result.shape,
            "result_range": [float(np.min(result)), float(np.max(result))]
        }
    
    def benchmark_faiss_search(n_candidates: int, n_dims: int, k: int) -> Dict[str, float]:
        """Benchmark FAISS-based similarity search."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        rng = np.random.default_rng(42)
        seed_vec = rng.random(n_dims)
        seed_unit = seed_vec / np.linalg.norm(seed_vec)
        
        cand_vecs = rng.random((n_candidates, n_dims))
        cand_unit = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
        
        start_time = time.perf_counter()
        indices = top_m_cosine(seed_unit, cand_unit, k)
        end_time = time.perf_counter()
        
        return {
            "total_time_ms": (end_time - start_time) * 1000,
            "time_per_candidate_ms": ((end_time - start_time) * 1000) / n_candidates,
            "result_shape": indices.shape,
            "indices_range": [int(np.min(indices)), int(np.max(indices))]
        }
    
    return {
        "quantum_kernel": benchmark_quantum_kernel,
        "gbs_boost": benchmark_gbs_boost,
        "faiss_search": benchmark_faiss_search,
    }

# Export key testing utilities
__all__ = [
    "generate_quantum_test_cases",
    "assert_quantum_properties", 
    "create_benchmark_suite",
    "MockQuantumDevice"
]
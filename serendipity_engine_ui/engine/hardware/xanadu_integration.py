"""
Enhanced Xanadu Cloud integration with quantum kernel computation
================================================================

This module provides high-level quantum kernel computation using Xanadu Cloud
hardware with graceful fallback to simulation.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Import Xanadu integration components
try:
    from .xanadu_factory import xanadu_factory
    FACTORY_AVAILABLE = True
except ImportError:
    logger.warning("Xanadu factory not available")
    xanadu_factory = None
    FACTORY_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    import pennylane as qml
    import strawberryfields as sf
    QUANTUM_LIBRARIES_AVAILABLE = True
except ImportError:
    QUANTUM_LIBRARIES_AVAILABLE = False
    logger.warning("Quantum libraries not available - using classical fallback")


class QuantumKernel:
    """
    Quantum kernel computation class with hardware acceleration
    """
    
    def __init__(self):
        self.use_hardware = True
        self.max_shots = 1000
        self.cache = {}
        self.device_cache = {}
        
    def compute_kernel_matrix(self, vectors: np.ndarray, max_shots: int = 1000) -> np.ndarray:
        """
        Compute quantum kernel matrix between all pairs of vectors
        
        Args:
            vectors: Array of shape (n_samples, n_features)
            max_shots: Maximum shots for hardware execution
            
        Returns:
            Kernel matrix of shape (n_samples, n_samples)
        """
        n_samples = vectors.shape[0]
        kernel_matrix = np.zeros((n_samples, n_samples))
        
        # Compute all pairwise kernel values
        for i in range(n_samples):
            for j in range(n_samples):
                if i == j:
                    kernel_matrix[i, j] = 1.0
                elif i < j:
                    # Compute kernel value
                    kernel_value = self._compute_single_kernel(
                        vectors[i], vectors[j], max_shots
                    )
                    kernel_matrix[i, j] = kernel_value
                    kernel_matrix[j, i] = kernel_value  # Symmetric
                    
        return kernel_matrix
    
    def _compute_single_kernel(self, x: np.ndarray, y: np.ndarray, max_shots: int) -> float:
        """
        Compute quantum kernel between two vectors
        """
        # Create cache key
        x_str = np.array2string(x, precision=4)
        y_str = np.array2string(y, precision=4)
        cache_key = f"{x_str}_{y_str}_{self.use_hardware}_{max_shots}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if self.use_hardware and FACTORY_AVAILABLE and QUANTUM_LIBRARIES_AVAILABLE:
                # Try hardware computation
                result = self._hardware_kernel_computation(x, y, max_shots)
            else:
                # Fallback to simulation
                result = self._simulation_kernel_computation(x, y)
                
            # Cache result
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.warning(f"Kernel computation failed: {e}")
            # Ultimate fallback to classical similarity
            return self._classical_similarity(x, y)
    
    def _hardware_kernel_computation(self, x: np.ndarray, y: np.ndarray, max_shots: int) -> float:
        """
        Compute kernel using Xanadu hardware
        """
        try:
            # Get device
            wires = min(len(x), 8)  # Hardware limit
            device = xanadu_factory.create_pennylane_device(
                wires=wires,
                shots=max_shots,
                force_local=False
            )
            
            if device is None:
                raise Exception("Device creation failed")
            
            # Create quantum circuit for kernel computation
            @qml.qnode(device)
            def kernel_circuit():
                # Encode first vector
                for i in range(wires):
                    if i < len(x):
                        qml.RY(x[i] * np.pi / 2, wires=i)
                
                # Entangling layer
                for i in range(wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Ring connection
                if wires > 2:
                    qml.CNOT(wires=[wires - 1, 0])
                
                # Inverse entangling
                if wires > 2:
                    qml.CNOT(wires=[wires - 1, 0])
                for i in range(wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Inverse encoding of second vector
                for i in range(wires):
                    if i < len(y):
                        qml.RY(-y[i] * np.pi / 2, wires=i)
                
                # Measurement
                return qml.probs(wires=range(wires))
            
            # Execute circuit
            probs = kernel_circuit()
            
            # Return fidelity with ground state
            return float(probs[0])
            
        except Exception as e:
            logger.warning(f"Hardware kernel computation failed: {e}")
            raise
    
    def _simulation_kernel_computation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute kernel using local simulation
        """
        if not QUANTUM_LIBRARIES_AVAILABLE:
            return self._classical_similarity(x, y)
        
        try:
            wires = min(len(x), 16)  # Simulation limit
            device = qml.device("default.qubit", wires=wires, shots=None)
            
            @qml.qnode(device)
            def kernel_circuit():
                # Same circuit as hardware version
                for i in range(wires):
                    if i < len(x):
                        qml.RY(x[i] * np.pi / 2, wires=i)
                
                for i in range(wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                if wires > 2:
                    qml.CNOT(wires=[wires - 1, 0])
                    qml.CNOT(wires=[wires - 1, 0])
                
                for i in range(wires - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                for i in range(wires):
                    if i < len(y):
                        qml.RY(-y[i] * np.pi / 2, wires=i)
                
                return qml.probs(wires=range(wires))
            
            probs = kernel_circuit()
            return float(probs[0])
            
        except Exception as e:
            logger.warning(f"Simulation kernel computation failed: {e}")
            return self._classical_similarity(x, y)
    
    def _classical_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Classical cosine similarity fallback
        """
        try:
            # Normalize vectors
            x_norm = x / (np.linalg.norm(x) + 1e-10)
            y_norm = y / (np.linalg.norm(y) + 1e-10)
            
            # Cosine similarity
            similarity = np.dot(x_norm, y_norm)
            
            # Convert to [0, 1] range
            return (similarity + 1) / 2
            
        except Exception:
            return 0.5  # Neutral similarity
    
    def clear_cache(self):
        """Clear computation cache"""
        self.cache.clear()
        self.device_cache.clear()


# Global quantum kernel instance
quantum_kernel = QuantumKernel()


def compute_quantum_tiebreaker(seed_vector: List[float], 
                             candidate_vectors: List[List[float]],
                             use_hardware: bool = True,
                             max_shots: int = 1000) -> int:
    """
    Compute quantum tiebreaker to select winning candidate
    
    Args:
        seed_vector: Reference vector
        candidate_vectors: List of candidate vectors
        use_hardware: Whether to use hardware if available
        max_shots: Maximum shots for hardware execution
        
    Returns:
        Index of selected candidate
    """
    if not candidate_vectors:
        return 0
    
    # Set hardware preference
    quantum_kernel.use_hardware = use_hardware
    quantum_kernel.max_shots = max_shots
    
    # Convert to numpy arrays
    seed = np.array(seed_vector, dtype=float)
    candidates = np.array(candidate_vectors, dtype=float)
    
    # Compute kernel values
    kernel_values = []
    for candidate in candidates:
        kernel_value = quantum_kernel._compute_single_kernel(seed, candidate, max_shots)
        kernel_values.append(kernel_value)
    
    # Select candidate with highest kernel value
    return int(np.argmax(kernel_values))


def get_hardware_status() -> Dict[str, Any]:
    """Get current hardware status"""
    status = {
        "quantum_libraries_available": QUANTUM_LIBRARIES_AVAILABLE,
        "factory_available": FACTORY_AVAILABLE,
        "cache_size": len(quantum_kernel.cache)
    }
    
    if FACTORY_AVAILABLE and xanadu_factory:
        try:
            factory_status = xanadu_factory.get_device_info()
            status.update(factory_status)
        except Exception as e:
            status["factory_error"] = str(e)
    
    return status
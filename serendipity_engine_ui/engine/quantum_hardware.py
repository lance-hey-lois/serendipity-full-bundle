"""
Xanadu Cloud Quantum Hardware Integration
==========================================
Real photonic quantum computing on Xanadu hardware with graceful fallbacks.

Environment Variables:
- SF_API_KEY: Your Xanadu Cloud API key
- XANADU_DEVICE: Device name (default: X8)
- XANADU_SHOTS: Number of shots (default: 100)
"""

import os
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Check for quantum dependencies
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    logger.warning("PennyLane not installed - quantum features will use fallback")

try:
    import strawberryfields as sf
    HAS_STRAWBERRYFIELDS = True
except ImportError:
    HAS_STRAWBERRYFIELDS = False
    logger.warning("Strawberry Fields not installed - photonic features will use fallback")

try:
    import pennylane_sf
    HAS_PENNYLANE_SF = True
except ImportError:
    HAS_PENNYLANE_SF = False
    logger.info("PennyLane-SF not installed - hardware acceleration unavailable")


def make_pl_device(wires: int, cutoff: int = 5, shots: int = 100):
    """
    Create a PennyLane device. If SF_API_KEY is set, use Xanadu Cloud hardware;
    otherwise fall back to a local simulator.
    
    Args:
        wires: Number of quantum wires/modes
        cutoff: Cutoff dimension for Fock basis
        shots: Number of measurement shots
    
    Returns:
        PennyLane device (hardware or simulator)
    """
    if not HAS_PENNYLANE:
        logger.error("PennyLane not available")
        return None
    
    # Get configuration from environment
    sf_key = os.getenv("SF_API_KEY", "").strip()
    device_name = os.getenv("XANADU_DEVICE", "X8")
    shots = int(os.getenv("XANADU_SHOTS", str(shots)))
    
    # Apply hardware constraints
    wires = min(wires, 6)  # Hardware limit
    cutoff = min(cutoff, 8)  # Memory limit
    shots = min(shots, 200)  # Cost optimization
    
    if sf_key and HAS_PENNYLANE_SF:
        # Use real quantum hardware
        try:
            logger.info(f"Creating Xanadu Cloud device: {device_name} with {wires} wires")
            
            # Remote photonic hardware via PennyLane-SF
            device = qml.device(
                "strawberryfields.remote",
                backend=device_name,
                wires=wires,
                cutoff_dim=int(cutoff),
                shots=shots,
            )
            
            logger.info("✅ Connected to Xanadu quantum hardware")
            return device
            
        except Exception as e:
            logger.warning(f"Failed to connect to hardware: {e}")
            logger.info("Falling back to local simulation")
    
    # Local fallback: fock simulator
    if HAS_STRAWBERRYFIELDS:
        try:
            device = qml.device(
                "strawberryfields.fock",
                wires=wires,
                cutoff_dim=int(cutoff),
                shots=shots if shots else None,
            )
            logger.info(f"Using local Strawberry Fields simulator with {wires} wires")
            return device
        except Exception as e:
            logger.warning(f"Strawberry Fields device failed: {e}")
    
    # Ultimate fallback: default qubit
    device = qml.device("default.qubit", wires=wires, shots=shots if shots else None)
    logger.info(f"Using default qubit simulator with {wires} wires")
    return device


def make_sf_engine(cutoff: int = 5, shots: int = 100):
    """
    Return a Strawberry Fields Engine. Uses RemoteEngine on Xanadu Cloud
    if SF_API_KEY is set, else local fock simulator Engine.
    
    Args:
        cutoff: Cutoff dimension for Fock basis
        shots: Number of measurement shots
    
    Returns:
        Strawberry Fields Engine (hardware or simulator)
    """
    if not HAS_STRAWBERRYFIELDS:
        logger.error("Strawberry Fields not available")
        return None
    
    # Get configuration from environment
    sf_key = os.getenv("SF_API_KEY", "").strip()
    device_name = os.getenv("XANADU_DEVICE", "X8")
    shots = int(os.getenv("XANADU_SHOTS", str(shots)))
    
    # Apply hardware constraints
    cutoff = min(cutoff, 8)
    shots = min(shots, 200)
    
    if sf_key:
        # Try to use real quantum hardware
        try:
            logger.info(f"Connecting to Xanadu Cloud device: {device_name}")
            
            # Authenticate if not already done
            if not sf.Account.is_logged_in():
                sf.Account.login(sf_key, save=True)
                logger.info("Authenticated with Xanadu Cloud")
            
            # Create remote engine
            engine = sf.RemoteEngine(device_name)
            logger.info(f"✅ Connected to Xanadu hardware: {device_name}")
            return engine
            
        except Exception as e:
            logger.warning(f"Failed to connect to hardware: {e}")
            logger.info("Falling back to local simulation")
    
    # Local fallback: fock simulator
    engine = sf.Engine("fock", backend_options={"cutoff_dim": int(cutoff)})
    logger.info(f"Using local Fock simulator with cutoff={cutoff}")
    return engine


def check_hardware_availability() -> dict:
    """
    Check if quantum hardware is available and configured.
    
    Returns:
        Dictionary with hardware status information
    """
    status = {
        "pennylane": HAS_PENNYLANE,
        "strawberryfields": HAS_STRAWBERRYFIELDS,
        "pennylane_sf": HAS_PENNYLANE_SF,
        "api_key_set": bool(os.getenv("SF_API_KEY")),
        "device": os.getenv("XANADU_DEVICE", "X8"),
        "shots": int(os.getenv("XANADU_SHOTS", "100")),
        "hardware_available": False,
        "authenticated": False
    }
    
    # Check authentication status
    if HAS_STRAWBERRYFIELDS and status["api_key_set"]:
        try:
            status["authenticated"] = sf.Account.is_logged_in()
            status["hardware_available"] = status["authenticated"]
        except:
            pass
    
    return status


def validate_circuit_constraints(wires: int, cutoff: int, shots: int) -> Tuple[int, int, int]:
    """
    Validate and adjust circuit parameters for hardware constraints.
    
    Args:
        wires: Requested number of wires
        cutoff: Requested cutoff dimension
        shots: Requested number of shots
    
    Returns:
        Tuple of (adjusted_wires, adjusted_cutoff, adjusted_shots)
    """
    # Hardware limits
    MAX_WIRES = 6
    MAX_CUTOFF = 8
    MAX_SHOTS = 200
    MIN_SHOTS = 10
    
    # Apply constraints
    wires = max(1, min(wires, MAX_WIRES))
    cutoff = max(2, min(cutoff, MAX_CUTOFF))
    shots = max(MIN_SHOTS, min(shots, MAX_SHOTS))
    
    if wires > MAX_WIRES or cutoff > MAX_CUTOFF:
        logger.warning(f"Circuit parameters adjusted for hardware: wires={wires}, cutoff={cutoff}")
    
    return wires, cutoff, shots


# Example quantum circuit using hardware
def hardware_quantum_kernel(x: np.ndarray, y: np.ndarray, use_hardware: bool = True) -> float:
    """
    Compute quantum kernel between two vectors using hardware if available.
    
    Args:
        x: First vector
        y: Second vector
        use_hardware: Whether to attempt hardware execution
    
    Returns:
        Quantum kernel value
    """
    if not HAS_PENNYLANE:
        return 0.0
    
    # Prepare data
    dim = min(len(x), len(y), 6)  # Hardware limit
    x_scaled = x[:dim] * np.pi / 2
    y_scaled = y[:dim] * np.pi / 2
    
    # Create device
    device = make_pl_device(wires=dim, cutoff=5, shots=100) if use_hardware else None
    if device is None:
        return 0.0
    
    @qml.qnode(device)
    def kernel_circuit():
        # Encode x
        for i in range(dim):
            qml.Rotation(x_scaled[i], wires=i)
        
        # Entangle
        for i in range(dim - 1):
            qml.CZ(wires=[i, i + 1])
        
        # Inverse encode y
        for i in range(dim - 1):
            qml.CZ(wires=[i, i + 1])
        
        for i in range(dim):
            qml.Rotation(-y_scaled[i], wires=i)
        
        # Measure
        return qml.probs(wires=range(dim))
    
    try:
        probs = kernel_circuit()
        return float(probs[0])  # Return |<0|psi>|^2
    except Exception as e:
        logger.error(f"Quantum kernel execution failed: {e}")
        return 0.0


if __name__ == "__main__":
    # Test hardware availability
    print("🔬 Xanadu Hardware Status:")
    print("=" * 50)
    
    status = check_hardware_availability()
    for key, value in status.items():
        icon = "✅" if value else "❌"
        if key in ["device", "shots"]:
            print(f"  {key}: {value}")
        else:
            print(f"{icon} {key}: {value}")
    
    # Test device creation
    print("\n🧪 Testing Device Creation:")
    print("-" * 50)
    
    device = make_pl_device(wires=4, cutoff=5, shots=50)
    if device:
        print(f"✅ Created device: {device}")
    
    engine = make_sf_engine(cutoff=5, shots=50)
    if engine:
        print(f"✅ Created engine: {engine}")
    
    # Test quantum kernel
    print("\n🚀 Testing Quantum Kernel:")
    print("-" * 50)
    
    x = np.array([0.5, 1.0, 0.3, 0.8])
    y = np.array([0.6, 0.9, 0.4, 0.7])
    
    kernel_value = hardware_quantum_kernel(x, y, use_hardware=True)
    print(f"Quantum kernel value: {kernel_value:.4f}")
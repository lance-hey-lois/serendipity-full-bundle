"""
Xanadu Cloud Quantum Device Factory for Serendipity Engine

This module provides a unified interface for creating and managing quantum devices
that can run on Xanadu's photonic quantum hardware through PennyLane and Strawberry Fields.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XanaduDeviceFactory:
    """Factory for creating quantum devices with Xanadu Cloud support."""
    
    def __init__(self):
        self.api_token = os.getenv("XANADU_CLOUD_TOKEN")
        self.default_device = os.getenv("XANADU_DEFAULT_DEVICE", "X8_01")
        self.use_hardware = os.getenv("XANADU_USE_HARDWARE", "false").lower() == "true"
        self.max_shots = int(os.getenv("XANADU_MAX_SHOTS", "1000"))
        self.connection_timeout = int(os.getenv("XANADU_TIMEOUT", "300"))
        
    def check_hardware_availability(self) -> bool:
        """Check if Xanadu Cloud hardware is available and accessible."""
        if not self.api_token:
            logger.warning("XANADU_CLOUD_TOKEN not set. Hardware unavailable.")
            return False
            
        if not self.use_hardware:
            logger.info("Hardware disabled via XANADU_USE_HARDWARE=false")
            return False
            
        try:
            import pennylane as qml
            # Test connection to Xanadu Cloud
            test_dev = qml.device("strawberryfields.remote", 
                                wires=2, 
                                backend="X8_01",
                                shots=10)
            logger.info("Xanadu Cloud hardware connection successful")
            return True
        except Exception as e:
            logger.error(f"Hardware connection failed: {e}")
            return False
    
    def create_pennylane_device(self, 
                              wires: int, 
                              shots: Optional[int] = None,
                              backend: Optional[str] = None,
                              force_local: bool = False) -> Any:
        """
        Create a PennyLane device with Xanadu Cloud support.
        
        Args:
            wires: Number of qubits/modes
            shots: Number of shots (None for exact simulation)
            backend: Specific backend to use
            force_local: Force local simulation even if hardware available
            
        Returns:
            PennyLane device instance
        """
        try:
            import pennylane as qml
        except ImportError:
            raise ImportError("PennyLane not installed. Run: pip install pennylane")
        
        # Determine if we should use hardware
        use_hw = (not force_local and 
                 self.use_hardware and 
                 self.check_hardware_availability())
        
        if use_hw:
            try:
                device_backend = backend or self.default_device
                device_shots = min(shots or self.max_shots, self.max_shots)
                
                logger.info(f"Creating Xanadu hardware device: {device_backend}, shots={device_shots}")
                
                return qml.device("strawberryfields.remote",
                                wires=wires,
                                backend=device_backend,
                                shots=device_shots)
                
            except Exception as e:
                logger.error(f"Hardware device creation failed: {e}")
                logger.info("Falling back to local simulation")
        
        # Fallback to local simulation
        if shots is None:
            logger.info("Creating local exact simulation device")
            return qml.device("default.qubit", wires=wires)
        else:
            logger.info(f"Creating local shot-based simulation device: {shots} shots")
            return qml.device("default.qubit", wires=wires, shots=shots)
    
    def create_strawberryfields_engine(self, 
                                     backend: Optional[str] = None,
                                     force_local: bool = False) -> Any:
        """
        Create a Strawberry Fields engine with Xanadu Cloud support.
        
        Args:
            backend: Specific backend to use
            force_local: Force local simulation
            
        Returns:
            Strawberry Fields engine instance
        """
        try:
            import strawberryfields as sf
        except ImportError:
            raise ImportError("Strawberry Fields not installed. Run: pip install strawberryfields")
        
        use_hw = (not force_local and 
                 self.use_hardware and 
                 self.check_hardware_availability())
        
        if use_hw:
            try:
                device_backend = backend or self.default_device
                logger.info(f"Creating Xanadu SF remote engine: {device_backend}")
                
                return sf.RemoteEngine(device_backend)
                
            except Exception as e:
                logger.error(f"Remote SF engine creation failed: {e}")
                logger.info("Falling back to local Fock simulation")
        
        # Fallback to local simulation
        logger.info("Creating local Fock simulation engine")
        return sf.Engine("fock", backend_options={"cutoff_dim": 5})
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices and current configuration."""
        return {
            "api_token_set": bool(self.api_token),
            "hardware_enabled": self.use_hardware,
            "default_device": self.default_device,
            "max_shots": self.max_shots,
            "connection_timeout": self.connection_timeout,
            "hardware_available": self.check_hardware_availability()
        }
    
    def validate_circuit_constraints(self, 
                                   wires: int, 
                                   shots: Optional[int] = None,
                                   backend: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate circuit constraints for hardware execution.
        
        Returns:
            Dictionary with validation results and recommendations
        """
        constraints = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check wire count limits
        if wires > 8:
            constraints["errors"].append(f"Wire count {wires} exceeds hardware limit of 8")
            constraints["valid"] = False
        elif wires > 4:
            constraints["warnings"].append(f"High wire count {wires} may have limited hardware availability")
        
        # Check shot limits
        if shots and shots > self.max_shots:
            constraints["warnings"].append(f"Shot count {shots} exceeds recommended limit {self.max_shots}")
            constraints["recommendations"].append(f"Consider reducing shots to {self.max_shots}")
        
        # Backend-specific constraints
        device_backend = backend or self.default_device
        if device_backend.startswith("X8"):
            if wires > 8:
                constraints["errors"].append(f"X8 devices support maximum 8 modes, requested {wires}")
                constraints["valid"] = False
        
        return constraints

# Global factory instance
xanadu_factory = XanaduDeviceFactory()

def get_quantum_device(wires: int, 
                      shots: Optional[int] = None,
                      backend: Optional[str] = None,
                      force_local: bool = False) -> Any:
    """Convenience function to get a quantum device."""
    return xanadu_factory.create_pennylane_device(wires, shots, backend, force_local)

def get_sf_engine(backend: Optional[str] = None,
                 force_local: bool = False) -> Any:
    """Convenience function to get a Strawberry Fields engine."""
    return xanadu_factory.create_strawberryfields_engine(backend, force_local)
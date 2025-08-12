"""
Tests for Xanadu Cloud hardware integration.

This module tests both hardware and fallback functionality to ensure
robust operation in all environments.
"""

import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock

# Import modules to test
from engine.hardware.xanadu_factory import XanaduDeviceFactory, get_quantum_device, get_sf_engine
from engine.hardware.xanadu_integration import XanaduQuantumKernel, XanaduPhotonicGBS
from config.xanadu_config import XanaduConfig

class TestXanaduDeviceFactory:
    """Test the Xanadu device factory."""
    
    def test_factory_initialization(self):
        """Test factory initializes with environment variables."""
        factory = XanaduDeviceFactory()
        assert hasattr(factory, 'api_token')
        assert hasattr(factory, 'use_hardware')
        assert hasattr(factory, 'default_device')
    
    def test_hardware_availability_no_token(self):
        """Test hardware availability check without token."""
        with patch.dict(os.environ, {}, clear=True):
            factory = XanaduDeviceFactory()
            assert not factory.check_hardware_availability()
    
    def test_hardware_availability_disabled(self):
        """Test hardware availability when disabled."""
        with patch.dict(os.environ, {'XANADU_USE_HARDWARE': 'false'}):
            factory = XanaduDeviceFactory()
            assert not factory.check_hardware_availability()
    
    @patch('pennylane.device')
    def test_create_pennylane_device_local(self, mock_device):
        """Test creating local PennyLane device."""
        mock_device.return_value = MagicMock()
        factory = XanaduDeviceFactory()
        
        device = factory.create_pennylane_device(wires=4, force_local=True)
        
        mock_device.assert_called_once()
        assert device is not None
    
    @patch('strawberryfields.Engine')
    def test_create_sf_engine_local(self, mock_engine):
        """Test creating local Strawberry Fields engine."""
        mock_engine.return_value = MagicMock()
        factory = XanaduDeviceFactory()
        
        engine = factory.create_strawberryfields_engine(force_local=True)
        
        mock_engine.assert_called_once()
        assert engine is not None
    
    def test_validate_circuit_constraints(self):
        """Test circuit constraint validation."""
        factory = XanaduDeviceFactory()
        
        # Valid constraints
        result = factory.validate_circuit_constraints(wires=4, shots=100)
        assert result['valid']
        assert len(result['errors']) == 0
        
        # Invalid wire count
        result = factory.validate_circuit_constraints(wires=10)
        assert not result['valid']
        assert any('exceeds hardware limit' in error for error in result['errors'])
        
        # High shot count
        result = factory.validate_circuit_constraints(wires=4, shots=10000)
        assert any('exceeds recommended limit' in warning for warning in result['warnings'])
    
    def test_get_device_info(self):
        """Test device information retrieval."""
        factory = XanaduDeviceFactory()
        info = factory.get_device_info()
        
        assert 'api_token_set' in info
        assert 'hardware_enabled' in info
        assert 'default_device' in info
        assert 'max_shots' in info
        assert 'hardware_available' in info

class TestXanaduQuantumKernel:
    """Test the quantum kernel implementation."""
    
    def test_kernel_initialization(self):
        """Test kernel initialization."""
        kernel = XanaduQuantumKernel(use_hardware=False)
        assert not kernel.use_hardware
        assert kernel.max_dimension == 4
    
    @patch('pennylane.device')
    def test_compute_kernel_matrix_local(self, mock_device):
        """Test kernel matrix computation with local simulation."""
        mock_qnode = MagicMock()
        mock_qnode.return_value = np.array([0.8, 0.1, 0.05, 0.05])
        
        with patch('pennylane.qnode', return_value=mock_qnode):
            kernel = XanaduQuantumKernel(use_hardware=False)
            X = np.random.rand(3, 2) * np.pi/2
            
            K = kernel.compute_kernel_matrix(X)
            
            assert K.shape == (3, 3)
            assert np.allclose(np.diag(K), 1.0, atol=0.1)  # Diagonal should be close to 1
    
    def test_pca_compression(self):
        """Test PCA compression for high-dimensional inputs."""
        kernel = XanaduQuantumKernel(use_hardware=False, max_dimension=2)
        X = np.random.rand(5, 10)  # 10 features > max_dimension
        
        # This should trigger compression internally
        with patch.object(kernel, '_get_device') as mock_get_device:
            mock_get_device.return_value = MagicMock()
            kernel.compute_kernel_matrix(X)
            
            # Check that device was called with compressed dimensions
            mock_get_device.assert_called()

class TestXanaduPhotonicGBS:
    """Test the photonic GBS implementation."""
    
    def test_gbs_initialization(self):
        """Test GBS initialization."""
        gbs = XanaduPhotonicGBS(use_hardware=False)
        assert not gbs.use_hardware
    
    @patch('strawberryfields.Engine')
    @patch('strawberryfields.Program')
    def test_gbs_sampling_local(self, mock_program, mock_engine):
        """Test GBS sampling with local simulation."""
        # Mock SF components
        mock_result = MagicMock()
        mock_result.samples = np.array([[1, 0, 2], [0, 1, 1], [2, 0, 0]])
        
        mock_eng_instance = MagicMock()
        mock_eng_instance.run.return_value = mock_result
        mock_engine.return_value = mock_eng_instance
        
        gbs = XanaduPhotonicGBS(use_hardware=False)
        cand_vecs = np.random.rand(5, 4)
        
        scores = gbs.gbs_sampling(cand_vecs, modes=3, shots=50)
        
        assert scores.shape == (5,)
        assert np.isfinite(scores).all()
    
    def test_pca_projection(self):
        """Test PCA projection functionality."""
        gbs = XanaduPhotonicGBS(use_hardware=False)
        cand_vecs = np.random.rand(10, 6)
        
        Z, var = gbs._pca_project(cand_vecs, modes=3)
        
        assert Z.shape == (10, 3)
        assert var.shape == (3,)
        assert np.all(var > 0)
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        gbs = XanaduPhotonicGBS(use_hardware=False)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        normalized = gbs._zscore(x)
        
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)

class TestXanaduConfig:
    """Test the configuration management."""
    
    def test_config_from_env(self):
        """Test configuration creation from environment."""
        with patch.dict(os.environ, {
            'XANADU_CLOUD_TOKEN': 'test_token',
            'XANADU_USE_HARDWARE': 'true',
            'XANADU_DEFAULT_DEVICE': 'X12_01',
            'XANADU_MAX_SHOTS': '2000'
        }):
            config = XanaduConfig.from_env()
            
            assert config.api_token == 'test_token'
            assert config.use_hardware is True
            assert config.default_device == 'X12_01'
            assert config.max_shots == 2000
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = XanaduConfig(
            use_hardware=True,
            api_token=None,  # Missing token
            max_shots=-1,    # Invalid shots
            max_wires=0      # Invalid wires
        )
        
        issues = config.validate()
        
        assert len(issues) > 0
        assert any('XANADU_CLOUD_TOKEN' in issue for issue in issues)
        assert any('max_shots' in issue for issue in issues)
        assert any('max_wires' in issue for issue in issues)
    
    def test_device_list(self):
        """Test device list functionality."""
        config = XanaduConfig()
        devices = config.get_device_list()
        
        assert 'X8_01' in devices
        assert 'X12_01' in devices
        assert 'gaussian' in devices
        assert 'fock' in devices
    
    def test_hardware_device_detection(self):
        """Test hardware device detection."""
        config = XanaduConfig()
        
        assert config.is_hardware_device('X8_01')
        assert config.is_hardware_device('X12_01')
        assert not config.is_hardware_device('gaussian')
        assert not config.is_hardware_device('fock')

class TestIntegrationFallbacks:
    """Test integration and fallback mechanisms."""
    
    def test_quantum_kernel_fallback(self):
        """Test quantum kernel fallback to local simulation."""
        from engine.quantum import quantum_kernel_to_seed
        
        seed_vec = np.random.rand(3) * np.pi/2
        cand_vecs = np.random.rand(5, 3) * np.pi/2
        
        # Should work regardless of hardware availability
        result = quantum_kernel_to_seed(seed_vec, cand_vecs, use_hardware=False)
        
        assert result.shape == (5,)
        assert np.isfinite(result).all()
    
    def test_gbs_fallback(self):
        """Test GBS fallback to local simulation."""
        from engine.photonic_gbs import gbs_boost
        
        seed_vec = np.random.rand(4)
        cand_vecs = np.random.rand(6, 4)
        
        # Should work regardless of hardware availability
        result = gbs_boost(seed_vec, cand_vecs, use_hardware=False)
        
        assert result.shape == (6,)
        assert np.isfinite(result).all()

if __name__ == '__main__':
    pytest.main([__file__])
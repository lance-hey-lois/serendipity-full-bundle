# Xanadu Cloud Hardware Setup Guide

This guide explains how to configure and use Xanadu's photonic quantum hardware with the Serendipity Engine.

## Overview

The Serendipity Engine now supports real quantum computing through Xanadu Cloud's photonic quantum processors. This integration provides:

- **Hardware-accelerated quantum kernel computations**
- **Real Gaussian Boson Sampling (GBS) on photonic hardware**
- **Automatic fallback to local simulation**
- **Seamless integration with existing workflows**

## Prerequisites

### 1. Software Requirements

```bash
# Install required packages
pip install pennylane strawberryfields pennylane-sf thewalrus
```

### 2. Xanadu Cloud Account

1. Sign up at [Xanadu Cloud](https://cloud.xanadu.ai/)
2. Generate an API token from your dashboard
3. Note your available devices (X8, X12, etc.)

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required: API Authentication
export XANADU_CLOUD_TOKEN="your_api_token_here"

# Hardware Settings
export XANADU_USE_HARDWARE="true"
export XANADU_DEFAULT_DEVICE="X8_01"
export XANADU_MAX_SHOTS="1000"
export XANADU_TIMEOUT="300"

# Circuit Constraints
export XANADU_MAX_WIRES="8"
export XANADU_MAX_CUTOFF="5"
export XANADU_MAX_SQUEEZING="1.0"

# Fallback Settings
export XANADU_AUTO_FALLBACK="true"
export XANADU_FALLBACK_TIMEOUT="30"

# Performance Settings
export XANADU_CACHE_DEVICES="true"
export XANADU_BATCH_SIZE="16"
```

### Configuration File

Alternatively, create a JSON configuration file:

```json
{
  "api_token": "your_token_here",
  "use_hardware": true,
  "default_device": "X8_01",
  "max_shots": 1000,
  "connection_timeout": 300,
  "max_wires": 8,
  "max_cutoff": 5,
  "max_squeezing": 1.0,
  "auto_fallback": true,
  "fallback_timeout": 30,
  "cache_devices": true,
  "batch_size": 16
}
```

## Hardware Devices

### Available Devices

| Device | Description | Max Modes | Notes |
|--------|-------------|-----------|-------|
| X8_01  | 8-mode X-Series | 8 | General purpose |
| X12_01 | 12-mode X-Series | 12 | Higher capacity |
| gaussian | Local Gaussian | Unlimited | Simulation only |
| fock | Local Fock | Unlimited | Simulation only |

### Device Selection

The system automatically selects the best available device based on:
- Circuit requirements (number of modes/qubits)
- Hardware availability
- Configuration preferences

## Usage Examples

### Basic Quantum Kernel Computation

```python
import numpy as np
from engine.quantum import quantum_kernel_to_seed

# Prepare data
seed_vec = np.random.rand(4) * np.pi/2
cand_vecs = np.random.rand(10, 4) * np.pi/2

# Compute kernel with hardware acceleration
kernel_scores = quantum_kernel_to_seed(
    seed_vec, 
    cand_vecs, 
    use_hardware=True,
    max_shots=1000
)

print(f"Kernel scores: {kernel_scores}")
```

### Photonic GBS Analysis

```python
from engine.photonic_gbs import gbs_boost

# Analyze candidate vectors
cand_vecs = np.random.rand(20, 6)
seed_vec = np.random.rand(6)

# Run GBS with hardware
gbs_scores = gbs_boost(
    seed_vec,
    cand_vecs,
    modes=4,
    shots=200,
    use_hardware=True,
    backend="X8_01"
)

print(f"GBS community scores: {gbs_scores}")
```

### Configuration Management

```python
from config.xanadu_config import get_config, update_config

# Check current configuration
config = get_config()
print(f"Hardware enabled: {config.use_hardware}")
print(f"Default device: {config.default_device}")

# Update configuration
update_config(max_shots=2000, default_device="X12_01")

# Validate configuration
issues = config.validate()
if issues:
    print(f"Configuration issues: {issues}")
```

## Hardware Optimization

### Circuit Design Guidelines

1. **Keep circuits small**: Limit to 8 modes/qubits for best hardware compatibility
2. **Minimize shots**: Use 100-1000 shots for balance of accuracy and speed
3. **Batch operations**: Process multiple small circuits rather than one large circuit
4. **Use compression**: Let the system automatically compress high-dimensional data

### Performance Tips

1. **Enable caching**: Set `XANADU_CACHE_DEVICES=true` to reuse device connections
2. **Batch processing**: Use smaller batch sizes (16-32) for hardware execution
3. **Parallel fallback**: System automatically falls back to local simulation if hardware fails
4. **Monitor quotas**: Track your Xanadu Cloud usage to avoid quota exhaustion

## Troubleshooting

### Common Issues

#### 1. Hardware Connection Failed

```
ERROR: Hardware connection failed: [error details]
INFO: Falling back to local simulation
```

**Solutions:**
- Check your API token: `echo $XANADU_CLOUD_TOKEN`
- Verify internet connectivity
- Check Xanadu Cloud service status
- Ensure your account has available credits

#### 2. Circuit Too Large

```
ERROR: Wire count 12 exceeds hardware limit of 8
```

**Solutions:**
- Reduce problem size
- Enable automatic compression
- Use local simulation for large problems

#### 3. Authentication Error

```
WARNING: XANADU_CLOUD_TOKEN not set. Hardware unavailable.
```

**Solutions:**
- Set the environment variable: `export XANADU_CLOUD_TOKEN="your_token"`
- Check token validity in Xanadu Cloud dashboard
- Ensure token has necessary permissions

### Debugging Tools

#### Check Hardware Status

```python
from engine.hardware.xanadu_factory import xanadu_factory

# Get device information
info = xanadu_factory.get_device_info()
print(f"Hardware available: {info['hardware_available']}")
print(f"API token set: {info['api_token_set']}")

# Check circuit constraints
constraints = xanadu_factory.validate_circuit_constraints(
    wires=6, 
    shots=500
)
print(f"Valid circuit: {constraints['valid']}")
if constraints['warnings']:
    print(f"Warnings: {constraints['warnings']}")
```

#### Test Hardware Connection

```python
from engine.hardware.xanadu_factory import get_quantum_device

try:
    # Try to create hardware device
    device = get_quantum_device(wires=2, shots=10, force_local=False)
    print("Hardware connection successful!")
except Exception as e:
    print(f"Hardware connection failed: {e}")
```

## Performance Monitoring

### Execution Metrics

The system automatically logs:
- Hardware vs. local execution
- Circuit execution times
- Fallback events
- Device utilization

### Monitoring Integration

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('engine.hardware')

# Monitor execution
logger.info("Starting quantum computation")
result = quantum_kernel_to_seed(seed_vec, cand_vecs)
logger.info(f"Computation completed: {len(result)} results")
```

## Advanced Configuration

### Custom Device Factory

```python
from engine.hardware.xanadu_factory import XanaduDeviceFactory

# Create custom factory
factory = XanaduDeviceFactory()
factory.default_device = "X12_01"
factory.max_shots = 2000

# Create specialized device
device = factory.create_pennylane_device(
    wires=6,
    shots=500,
    backend="X8_01"
)
```

### Hardware-Specific Optimization

```python
from engine.hardware.xanadu_integration import XanaduQuantumKernel

# Configure kernel for hardware optimization
kernel = XanaduQuantumKernel(
    use_hardware=True,
    max_dimension=6
)

# Optimize for hardware constraints
X = np.random.rand(50, 10)  # Will be compressed to 6 dimensions
K = kernel.compute_kernel_matrix(X, max_shots=800)
```

## Security Considerations

1. **API Token Security**: Never commit tokens to version control
2. **Environment Isolation**: Use separate tokens for development/production
3. **Access Control**: Limit token permissions to necessary operations
4. **Audit Logging**: Monitor API usage through Xanadu Cloud dashboard

## Support and Resources

- **Xanadu Documentation**: [docs.xanadu.ai](https://docs.xanadu.ai)
- **PennyLane Documentation**: [pennylane.ai/doc](https://pennylane.ai/doc)
- **Strawberry Fields**: [strawberryfields.readthedocs.io](https://strawberryfields.readthedocs.io)
- **Issue Reporting**: Submit issues to the Serendipity Engine repository

## Migration from Local-Only

### Gradual Migration

1. **Start with testing**: Enable hardware for test environments first
2. **Validate results**: Compare hardware vs. local simulation results
3. **Monitor performance**: Track execution times and success rates
4. **Scale gradually**: Increase hardware usage as confidence grows

### Backward Compatibility

The integration maintains full backward compatibility:
- Existing code works without modification
- Local simulation remains the default fallback
- Hardware usage is opt-in via configuration

This ensures smooth migration with minimal risk to existing workflows.
"""
Xanadu Cloud Configuration for Serendipity Engine

This module handles all configuration settings for Xanadu Cloud integration,
including environment variables, device selection, and fallback settings.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

@dataclass
class XanaduConfig:
    """Configuration class for Xanadu Cloud integration."""
    
    # Authentication
    api_token: Optional[str] = None
    
    # Hardware settings
    use_hardware: bool = False
    default_device: str = "X8_01"
    max_shots: int = 1000
    connection_timeout: int = 300
    
    # Circuit constraints
    max_wires: int = 8
    max_cutoff: int = 5
    max_squeezing: float = 1.0
    
    # Fallback settings
    auto_fallback: bool = True
    fallback_timeout: int = 30
    
    # Performance settings
    cache_devices: bool = True
    batch_size: int = 16
    
    @classmethod
    def from_env(cls) -> 'XanaduConfig':
        """Create configuration from environment variables."""
        return cls(
            api_token=os.getenv("XANADU_CLOUD_TOKEN"),
            use_hardware=os.getenv("XANADU_USE_HARDWARE", "false").lower() == "true",
            default_device=os.getenv("XANADU_DEFAULT_DEVICE", "X8_01"),
            max_shots=int(os.getenv("XANADU_MAX_SHOTS", "1000")),
            connection_timeout=int(os.getenv("XANADU_TIMEOUT", "300")),
            max_wires=int(os.getenv("XANADU_MAX_WIRES", "8")),
            max_cutoff=int(os.getenv("XANADU_MAX_CUTOFF", "5")),
            max_squeezing=float(os.getenv("XANADU_MAX_SQUEEZING", "1.0")),
            auto_fallback=os.getenv("XANADU_AUTO_FALLBACK", "true").lower() == "true",
            fallback_timeout=int(os.getenv("XANADU_FALLBACK_TIMEOUT", "30")),
            cache_devices=os.getenv("XANADU_CACHE_DEVICES", "true").lower() == "true",
            batch_size=int(os.getenv("XANADU_BATCH_SIZE", "16"))
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'XanaduConfig':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'api_token': self.api_token,
            'use_hardware': self.use_hardware,
            'default_device': self.default_device,
            'max_shots': self.max_shots,
            'connection_timeout': self.connection_timeout,
            'max_wires': self.max_wires,
            'max_cutoff': self.max_cutoff,
            'max_squeezing': self.max_squeezing,
            'auto_fallback': self.auto_fallback,
            'fallback_timeout': self.fallback_timeout,
            'cache_devices': self.cache_devices,
            'batch_size': self.batch_size
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if self.use_hardware and not self.api_token:
            issues.append("Hardware enabled but XANADU_CLOUD_TOKEN not set")
        
        if self.max_shots <= 0:
            issues.append("max_shots must be positive")
        
        if self.max_wires <= 0 or self.max_wires > 8:
            issues.append("max_wires must be between 1 and 8")
        
        if self.max_cutoff <= 0:
            issues.append("max_cutoff must be positive")
        
        if self.max_squeezing < 0:
            issues.append("max_squeezing must be non-negative")
        
        if self.connection_timeout <= 0:
            issues.append("connection_timeout must be positive")
        
        if self.fallback_timeout <= 0:
            issues.append("fallback_timeout must be positive")
        
        if self.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        return issues
    
    def get_device_list(self) -> List[str]:
        """Get list of available Xanadu devices."""
        return [
            "X8_01",  # 8-mode X-Series
            "X12_01", # 12-mode X-Series  
            "gaussian", # Local Gaussian simulation
            "fock"    # Local Fock simulation
        ]
    
    def is_hardware_device(self, device_name: str) -> bool:
        """Check if device name corresponds to hardware."""
        return device_name.startswith("X")

# Global configuration instance
config = XanaduConfig.from_env()

def get_config() -> XanaduConfig:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs) -> None:
    """Update global configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

def reset_config() -> None:
    """Reset configuration to environment defaults."""
    global config
    config = XanaduConfig.from_env()

# Environment setup helper
def setup_environment() -> Dict[str, str]:
    """
    Setup environment variables for Xanadu Cloud.
    Returns dict of required environment variables.
    """
    required_vars = {
        "XANADU_CLOUD_TOKEN": "Your Xanadu Cloud API token",
        "XANADU_USE_HARDWARE": "Enable hardware execution (true/false)",
        "XANADU_DEFAULT_DEVICE": "Default device (X8_01, X12_01, etc.)",
        "XANADU_MAX_SHOTS": "Maximum shots per circuit",
        "XANADU_TIMEOUT": "Connection timeout in seconds"
    }
    
    missing_vars = {}
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars[var] = description
    
    return missing_vars

def create_env_template(output_path: str = ".env.xanadu") -> None:
    """Create a template environment file for Xanadu configuration."""
    template = """# Xanadu Cloud Configuration for Serendipity Engine

# API Authentication
XANADU_CLOUD_TOKEN=your_token_here

# Hardware Settings
XANADU_USE_HARDWARE=false
XANADU_DEFAULT_DEVICE=X8_01
XANADU_MAX_SHOTS=1000
XANADU_TIMEOUT=300

# Circuit Constraints
XANADU_MAX_WIRES=8
XANADU_MAX_CUTOFF=5
XANADU_MAX_SQUEEZING=1.0

# Fallback Settings
XANADU_AUTO_FALLBACK=true
XANADU_FALLBACK_TIMEOUT=30

# Performance Settings
XANADU_CACHE_DEVICES=true
XANADU_BATCH_SIZE=16
"""
    
    with open(output_path, 'w') as f:
        f.write(template)
import { HardwareStatus, DeviceInfo } from '../../types/quantum';

/**
 * Quantum Hardware Service
 * 
 * Manages quantum hardware connections and status for Xanadu devices.
 */
class QuantumHardwareService {
  private readonly backendUrl: string;
  private readonly cacheTimeout: number = 30000; // 30 seconds
  private lastStatusCheck: number = 0;
  private cachedStatus: HardwareStatus | null = null;

  constructor() {
    this.backendUrl = process.env.QUANTUM_BACKEND_URL || 'http://localhost:8000';
  }

  /**
   * Get current hardware status with caching
   */
  async getHardwareStatus(): Promise<HardwareStatus> {
    const now = Date.now();
    
    // Return cached status if still valid
    if (this.cachedStatus && (now - this.lastStatusCheck) < this.cacheTimeout) {
      return this.cachedStatus;
    }

    try {
      // Try to get status from backend
      const response = await fetch(`${this.backendUrl}/api/quantum/hardware/status`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${process.env.QUANTUM_API_TOKEN || ''}`,
          'Content-Type': 'application/json'
        },
        timeout: 10000 // 10 second timeout
      });

      if (response.ok) {
        const backendStatus = await response.json();
        this.cachedStatus = {
          hardwareAvailable: backendStatus.hardware_available,
          device: backendStatus.device,
          authenticated: backendStatus.authenticated,
          apiKeySet: backendStatus.api_key_set,
          maxShots: backendStatus.max_shots,
          maxWires: backendStatus.max_wires,
          devices: backendStatus.devices || [],
          lastCheck: new Date().toISOString(),
          status: backendStatus.hardware_available ? 'online' : 'offline'
        };
      } else {
        throw new Error(`Backend status check failed: ${response.status}`);
      }
    } catch (error) {
      console.warn('Failed to get hardware status from backend:', error);
      
      // Fallback to local hardware check
      this.cachedStatus = await this.performLocalHardwareCheck();
    }

    this.lastStatusCheck = now;
    return this.cachedStatus;
  }

  /**
   * Perform local hardware availability check
   */
  private async performLocalHardwareCheck(): Promise<HardwareStatus> {
    const apiKeySet = Boolean(
      process.env.SF_API_KEY || 
      process.env.XANADU_API_KEY || 
      process.env.XANADU_CLOUD_TOKEN
    );

    // Basic environment check
    const status: HardwareStatus = {
      hardwareAvailable: false,
      device: 'local_simulator',
      authenticated: false,
      apiKeySet,
      maxShots: 1000,
      maxWires: 8,
      devices: [],
      lastCheck: new Date().toISOString(),
      status: 'offline'
    };

    // If API key is available, assume hardware might be accessible
    if (apiKeySet) {
      status.hardwareAvailable = true;
      status.authenticated = true;
      status.device = process.env.XANADU_DEVICE || 'X8';
      status.status = 'online';
      status.devices = [
        {
          name: status.device,
          type: 'photonic',
          wires: 8,
          available: true,
          queue_size: 0
        }
      ];
    }

    return status;
  }

  /**
   * Get available quantum devices
   */
  async getAvailableDevices(): Promise<DeviceInfo[]> {
    try {
      const response = await fetch(`${this.backendUrl}/api/quantum/hardware/devices`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${process.env.QUANTUM_API_TOKEN || ''}`,
          'Content-Type': 'application/json'
        },
        timeout: 10000
      });

      if (response.ok) {
        const data = await response.json();
        return data.devices || [];
      }
    } catch (error) {
      console.warn('Failed to get device list from backend:', error);
    }

    // Fallback device list
    return [
      {
        name: 'X8',
        type: 'photonic',
        wires: 8,
        available: true,
        queue_size: 0
      },
      {
        name: 'local_simulator',
        type: 'simulator',
        wires: 16,
        available: true,
        queue_size: 0
      }
    ];
  }

  /**
   * Test hardware connection
   */
  async testHardwareConnection(deviceName?: string): Promise<{
    success: boolean;
    device: string;
    latency?: number;
    error?: string;
  }> {
    const startTime = Date.now();
    
    try {
      const response = await fetch(`${this.backendUrl}/api/quantum/hardware/test`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.QUANTUM_API_TOKEN || ''}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          device: deviceName || 'X8'
        }),
        timeout: 30000 // 30 second timeout for hardware test
      });

      const latency = Date.now() - startTime;

      if (response.ok) {
        const result = await response.json();
        return {
          success: true,
          device: result.device,
          latency
        };
      } else {
        const error = await response.json().catch(() => ({}));
        return {
          success: false,
          device: deviceName || 'unknown',
          latency,
          error: error.detail || `HTTP ${response.status}`
        };
      }
    } catch (error) {
      return {
        success: false,
        device: deviceName || 'unknown',
        latency: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Connection failed'
      };
    }
  }

  /**
   * Clear cached status (force refresh on next call)
   */
  clearCache(): void {
    this.cachedStatus = null;
    this.lastStatusCheck = 0;
  }

  /**
   * Get recommended device for given parameters
   */
  async getRecommendedDevice(wires: number, shots: number): Promise<string> {
    const devices = await this.getAvailableDevices();
    const status = await this.getHardwareStatus();

    // If hardware is available and parameters fit
    if (status.hardwareAvailable && wires <= 8 && shots <= 1000) {
      const hardwareDevices = devices.filter(d => d.type === 'photonic' && d.available);
      if (hardwareDevices.length > 0) {
        return hardwareDevices[0].name;
      }
    }

    // Fallback to simulator
    return 'local_simulator';
  }
}

export const quantumHardwareService = new QuantumHardwareService();
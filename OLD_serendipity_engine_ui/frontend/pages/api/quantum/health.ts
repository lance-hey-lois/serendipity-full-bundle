import { NextApiRequest, NextApiResponse } from 'next';
import { quantumCollapseService } from '../../../lib/quantum/collapse-service';
import { quantumHardwareService } from '../../../lib/quantum/hardware-service';

interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    collapse: boolean;
    hardware: boolean;
    backend: boolean;
  };
  timestamp: string;
  version: string;
}

/**
 * Quantum System Health Check API Endpoint
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<HealthResponse>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({
      status: 'unhealthy',
      services: { collapse: false, hardware: false, backend: false },
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0'
    });
  }

  try {
    // Check individual services
    const [collapseHealthy, hardwareStatus, backendHealthy] = await Promise.allSettled([
      quantumCollapseService.healthCheck(),
      quantumHardwareService.getHardwareStatus(),
      checkBackendHealth()
    ]);

    const services = {
      collapse: collapseHealthy.status === 'fulfilled' && collapseHealthy.value,
      hardware: hardwareStatus.status === 'fulfilled',
      backend: backendHealthy.status === 'fulfilled' && backendHealthy.value
    };

    // Determine overall status
    let status: 'healthy' | 'degraded' | 'unhealthy';
    const healthyCount = Object.values(services).filter(Boolean).length;
    
    if (healthyCount === 3) {
      status = 'healthy';
    } else if (healthyCount >= 1) {
      status = 'degraded';
    } else {
      status = 'unhealthy';
    }

    const response: HealthResponse = {
      status,
      services,
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0'
    };

    const statusCode = status === 'healthy' ? 200 : status === 'degraded' ? 206 : 503;
    return res.status(statusCode).json(response);

  } catch (error) {
    console.error('Health check error:', error);
    
    return res.status(503).json({
      status: 'unhealthy',
      services: { collapse: false, hardware: false, backend: false },
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0'
    });
  }
}

async function checkBackendHealth(): Promise<boolean> {
  try {
    const backendUrl = process.env.QUANTUM_BACKEND_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/health`, {
      method: 'GET',
      timeout: 5000
    });
    return response.ok;
  } catch {
    return false;
  }
}
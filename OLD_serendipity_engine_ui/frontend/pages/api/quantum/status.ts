import { NextApiRequest, NextApiResponse } from 'next';
import { quantumHardwareService } from '../../../lib/quantum/hardware-service';
import { HardwareStatus, QuantumError } from '../../../types/quantum';

/**
 * Quantum Hardware Status API Endpoint
 * 
 * Returns current status of quantum hardware and simulation capabilities.
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<HardwareStatus | QuantumError>
) {
  if (req.method !== 'GET') {
    return res.status(405).json({
      error: 'Method not allowed',
      message: 'Only GET requests are supported',
      timestamp: new Date().toISOString()
    });
  }

  try {
    const status = await quantumHardwareService.getHardwareStatus();
    
    return res.status(200).json({
      ...status,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Hardware status error:', error);
    
    return res.status(500).json({
      error: 'Hardware status check failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
}
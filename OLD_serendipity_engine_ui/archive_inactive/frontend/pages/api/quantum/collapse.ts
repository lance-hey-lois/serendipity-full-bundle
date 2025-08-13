import { NextApiRequest, NextApiResponse } from 'next';
import { quantumCollapseService } from '../../../lib/quantum/collapse-service';
import { validateQuantumRequest, handleApiError } from '../../../lib/quantum/utils';
import { CollapseRequest, CollapseResponse, QuantumError } from '../../../types/quantum';

/**
 * Quantum Collapse API Endpoint
 * 
 * Handles quantum tiebreaker requests using the Serendipity Engine's
 * quantum collapse mechanism with Xanadu hardware acceleration.
 */
export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<CollapseResponse | QuantumError>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({
      error: 'Method not allowed',
      message: 'Only POST requests are supported',
      timestamp: new Date().toISOString()
    });
  }

  try {
    // Validate request payload
    const validation = validateQuantumRequest(req.body);
    if (!validation.valid) {
      return res.status(400).json({
        error: 'Invalid request',
        message: validation.message || 'Request validation failed',
        timestamp: new Date().toISOString()
      });
    }

    const collapseRequest: CollapseRequest = req.body;
    
    // Process quantum collapse
    const startTime = Date.now();
    const result = await quantumCollapseService.performCollapse(collapseRequest);
    const processingTime = Date.now() - startTime;

    // Return successful response
    const response: CollapseResponse = {
      success: true,
      result: {
        selectedIndex: result.selectedIndex,
        confidence: result.confidence,
        quantumKernel: result.quantumKernel,
        hardwareUsed: result.hardwareUsed,
        processingTime,
        metadata: {
          timestamp: new Date().toISOString(),
          shots: result.shots,
          device: result.device,
          method: result.method
        }
      }
    };

    // Log successful quantum operation
    console.log(`Quantum collapse completed in ${processingTime}ms`, {
      selectedIndex: result.selectedIndex,
      confidence: result.confidence,
      hardwareUsed: result.hardwareUsed
    });

    return res.status(200).json(response);

  } catch (error) {
    console.error('Quantum collapse error:', error);
    return handleApiError(res, error as Error);
  }
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '10mb', // Handle large embedding vectors
    },
  },
};
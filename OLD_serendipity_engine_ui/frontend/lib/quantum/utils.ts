import { NextApiResponse } from 'next';
import { QuantumError, CollapseRequest } from '../../types/quantum';

/**
 * Quantum utility functions for validation and error handling
 */

export interface ValidationResult {
  valid: boolean;
  message?: string;
}

/**
 * Validate quantum collapse request
 */
export function validateQuantumRequest(body: any): ValidationResult {
  if (!body || typeof body !== 'object') {
    return { valid: false, message: 'Request body is required and must be an object' };
  }

  const { seedVector, candidateVectors, useHardware, maxShots, method } = body;

  // Validate seed vector
  if (!Array.isArray(seedVector)) {
    return { valid: false, message: 'seedVector must be an array' };
  }

  if (seedVector.length === 0 || seedVector.length > 1024) {
    return { valid: false, message: 'seedVector must have 1-1024 elements' };
  }

  if (!seedVector.every(isValidNumber)) {
    return { valid: false, message: 'seedVector must contain only finite numbers' };
  }

  // Validate candidate vectors
  if (!Array.isArray(candidateVectors)) {
    return { valid: false, message: 'candidateVectors must be an array' };
  }

  if (candidateVectors.length === 0 || candidateVectors.length > 100) {
    return { valid: false, message: 'candidateVectors must have 1-100 elements' };
  }

  for (let i = 0; i < candidateVectors.length; i++) {
    const candidate = candidateVectors[i];
    
    if (!Array.isArray(candidate)) {
      return { valid: false, message: `candidateVectors[${i}] must be an array` };
    }

    if (candidate.length !== seedVector.length) {
      return { valid: false, message: `candidateVectors[${i}] must have same length as seedVector` };
    }

    if (!candidate.every(isValidNumber)) {
      return { valid: false, message: `candidateVectors[${i}] must contain only finite numbers` };
    }
  }

  // Validate optional parameters
  if (useHardware !== undefined && typeof useHardware !== 'boolean') {
    return { valid: false, message: 'useHardware must be a boolean' };
  }

  if (maxShots !== undefined) {
    if (!Number.isInteger(maxShots) || maxShots < 10 || maxShots > 10000) {
      return { valid: false, message: 'maxShots must be an integer between 10 and 10000' };
    }
  }

  if (method !== undefined) {
    const validMethods = ['kernel_fidelity', 'amplitude_encoding', 'angle_encoding'];
    if (!validMethods.includes(method)) {
      return { valid: false, message: `method must be one of: ${validMethods.join(', ')}` };
    }
  }

  return { valid: true };
}

/**
 * Check if a value is a valid finite number
 */
function isValidNumber(value: any): boolean {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * Validate vector data for NaN, infinity, and range
 */
export function validateVectorData(vectors: number[][]): void {
  for (let i = 0; i < vectors.length; i++) {
    const vector = vectors[i];
    
    for (let j = 0; j < vector.length; j++) {
      const value = vector[j];
      
      if (!Number.isFinite(value)) {
        throw new Error(`Invalid value at vector[${i}][${j}]: ${value}`);\n      }\n      \n      if (Math.abs(value) > 1e6) {\n        throw new Error(`Value too large at vector[${i}][${j}]: ${value}`);\n      }\n    }\n  }\n}\n\n/**\n * Normalize vector to unit length\n */\nexport function normalizeVector(vector: number[]): number[] {\n  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));\n  \n  if (magnitude === 0) {\n    return new Array(vector.length).fill(0);\n  }\n  \n  return vector.map(val => val / magnitude);\n}\n\n/**\n * Scale vector values to a specific range\n */\nexport function scaleVector(vector: number[], minVal: number = 0, maxVal: number = Math.PI / 2): number[] {\n  const min = Math.min(...vector);\n  const max = Math.max(...vector);\n  const range = max - min;\n  \n  if (range === 0) {\n    return new Array(vector.length).fill((minVal + maxVal) / 2);\n  }\n  \n  const targetRange = maxVal - minVal;\n  return vector.map(val => minVal + ((val - min) / range) * targetRange);\n}\n\n/**\n * Handle API errors consistently\n */\nexport function handleApiError(res: NextApiResponse<QuantumError>, error: Error): void {\n  console.error('API Error:', error);\n  \n  let statusCode = 500;\n  let errorType = 'Internal Server Error';\n  let message = error.message || 'An unknown error occurred';\n  \n  // Map specific error types to HTTP status codes\n  if (error.message.includes('validation')) {\n    statusCode = 400;\n    errorType = 'Validation Error';\n  } else if (error.message.includes('timeout')) {\n    statusCode = 504;\n    errorType = 'Timeout Error';\n  } else if (error.message.includes('hardware') || error.message.includes('device')) {\n    statusCode = 503;\n    errorType = 'Hardware Unavailable';\n  } else if (error.message.includes('authentication') || error.message.includes('unauthorized')) {\n    statusCode = 401;\n    errorType = 'Authentication Error';\n  } else if (error.message.includes('not found')) {\n    statusCode = 404;\n    errorType = 'Not Found';\n  }\n  \n  res.status(statusCode).json({\n    error: errorType,\n    message,\n    timestamp: new Date().toISOString()\n  });\n}\n\n/**\n * Calculate statistical metrics for quantum kernel results\n */\nexport function calculateKernelStatistics(kernelValues: number[]): {\n  mean: number;\n  std: number;\n  min: number;\n  max: number;\n  entropy: number;\n} {\n  const n = kernelValues.length;\n  \n  if (n === 0) {\n    return { mean: 0, std: 0, min: 0, max: 0, entropy: 0 };\n  }\n  \n  const mean = kernelValues.reduce((sum, val) => sum + val, 0) / n;\n  const variance = kernelValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;\n  const std = Math.sqrt(variance);\n  const min = Math.min(...kernelValues);\n  const max = Math.max(...kernelValues);\n  \n  // Calculate entropy (measure of uncertainty)\n  let entropy = 0;\n  const sum = kernelValues.reduce((acc, val) => acc + val, 0);\n  \n  if (sum > 0) {\n    for (const val of kernelValues) {\n      if (val > 0) {\n        const prob = val / sum;\n        entropy -= prob * Math.log2(prob);\n      }\n    }\n  }\n  \n  return { mean, std, min, max, entropy };\n}\n\n/**\n * Generate quantum-inspired random seed\n */\nexport function generateQuantumSeed(): string {\n  const timestamp = Date.now();\n  const random = Math.random();\n  const quantum = Math.sin(timestamp * random) * Math.cos(random * Math.PI);\n  \n  return `qseed_${timestamp}_${Math.abs(quantum).toString(36).substr(2, 9)}`;\n}\n\n/**\n * Validate quantum circuit parameters\n */\nexport function validateCircuitParameters(wires: number, shots: number, depth: number): ValidationResult {\n  if (!Number.isInteger(wires) || wires < 1 || wires > 20) {\n    return { valid: false, message: 'Wires must be an integer between 1 and 20' };\n  }\n  \n  if (!Number.isInteger(shots) || shots < 1 || shots > 100000) {\n    return { valid: false, message: 'Shots must be an integer between 1 and 100000' };\n  }\n  \n  if (!Number.isInteger(depth) || depth < 1 || depth > 100) {\n    return { valid: false, message: 'Depth must be an integer between 1 and 100' };\n  }\n  \n  return { valid: true };\n}\n\n/**\n * Format duration in human-readable format\n */\nexport function formatDuration(milliseconds: number): string {\n  if (milliseconds < 1000) {\n    return `${Math.round(milliseconds)}ms`;\n  } else if (milliseconds < 60000) {\n    return `${(milliseconds / 1000).toFixed(1)}s`;\n  } else {\n    const minutes = Math.floor(milliseconds / 60000);\n    const seconds = Math.floor((milliseconds % 60000) / 1000);\n    return `${minutes}m ${seconds}s`;\n  }\n}\n\n/**\n * Create correlation matrix from vectors\n */\nexport function createCorrelationMatrix(vectors: number[][]): number[][] {\n  const n = vectors.length;\n  const matrix: number[][] = [];\n  \n  for (let i = 0; i < n; i++) {\n    matrix[i] = [];\n    for (let j = 0; j < n; j++) {\n      if (i === j) {\n        matrix[i][j] = 1.0;\n      } else {\n        matrix[i][j] = calculatePearsonCorrelation(vectors[i], vectors[j]);\n      }\n    }\n  }\n  \n  return matrix;\n}\n\n/**\n * Calculate Pearson correlation coefficient\n */\nfunction calculatePearsonCorrelation(x: number[], y: number[]): number {\n  if (x.length !== y.length || x.length === 0) return 0;\n  \n  const n = x.length;\n  const sumX = x.reduce((a, b) => a + b, 0);\n  const sumY = y.reduce((a, b) => a + b, 0);\n  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);\n  const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);\n  const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);\n  \n  const numerator = n * sumXY - sumX * sumY;\n  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));\n  \n  return denominator === 0 ? 0 : numerator / denominator;\n}\n"
"""
FastAPI Quantum Microservice for Tiebreaker Operations
========================================================
Production-ready quantum tiebreaker with hardware support and graceful fallbacks.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import quantum functions from existing engine
try:
    from engine.quantum import pca_compress, quantum_kernel_to_seed
    from engine.quantum_hardware import check_hardware_availability
    QUANTUM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quantum modules not available: {e}")
    QUANTUM_AVAILABLE = False
    
    # Fallback implementations
    def pca_compress(X, out_dim):
        """Fallback PCA using simple truncation"""
        return X[:, :out_dim] if X.shape[1] > out_dim else X
    
    def quantum_kernel_to_seed(seed_vec, cand_vecs, shots=100):
        """Fallback returns zeros"""
        return np.zeros(len(cand_vecs))
    
    def check_hardware_availability():
        """Fallback hardware status"""
        return {"hardware_available": False}

# Initialize FastAPI app
app = FastAPI(
    title="Quantum Collapse API",
    description="Hardware-accelerated quantum tiebreaker service",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Candidate(BaseModel):
    id: str
    vec: List[float]
    
class QuantumTiebreakRequest(BaseModel):
    seed: List[float] = Field(..., description="Seed vector")
    candidates: List[Candidate] = Field(..., description="Candidate vectors")
    k: int = Field(10, description="Number of candidates to process", ge=1, le=20)
    out_dim: int = Field(4, description="PCA output dimension", ge=2, le=8)
    shots: int = Field(100, description="Quantum measurement shots", ge=10, le=200)

class QuantumScore(BaseModel):
    id: str
    q: float

class QuantumTiebreakResponse(BaseModel):
    scores: List[QuantumScore]
    hardware_used: bool = False
    method: str = "simulation"

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check service health and quantum availability"""
    hardware_status = check_hardware_availability() if QUANTUM_AVAILABLE else {}
    
    return {
        "status": "healthy",
        "quantum_available": QUANTUM_AVAILABLE,
        "hardware_status": hardware_status,
        "api_version": "1.0.0"
    }

# Main quantum tiebreak endpoint
@app.post("/quantum_tiebreak", response_model=QuantumTiebreakResponse)
async def quantum_tiebreak(request: QuantumTiebreakRequest):
    """
    Perform quantum tiebreaking on top candidates.
    
    Uses Xanadu hardware if available, falls back to simulation or zeros.
    """
    try:
        # Validate and prepare data
        K = min(request.k, len(request.candidates))
        if K == 0:
            return QuantumTiebreakResponse(scores=[], hardware_used=False, method="none")
        
        # Convert to numpy arrays
        seed = np.array(request.seed, dtype=np.float32)
        candidates_matrix = np.array([c.vec for c in request.candidates[:K]], dtype=np.float32)
        
        # Ensure dimensions match
        if seed.shape[0] != candidates_matrix.shape[1]:
            raise HTTPException(
                status_code=400,
                detail=f"Dimension mismatch: seed has {seed.shape[0]} dims, candidates have {candidates_matrix.shape[1]}"
            )
        
        # Apply PCA compression for quantum compatibility
        logger.info(f"Processing {K} candidates with quantum tiebreaker")
        
        # Stack seed with candidates for joint PCA
        combined = np.vstack([seed.reshape(1, -1), candidates_matrix])
        
        # Compress to quantum-compatible dimensions
        compressed = pca_compress(combined, out_dim=request.out_dim)
        z_seed = compressed[0]
        z_candidates = compressed[1:]
        
        # Check hardware availability
        hardware_status = check_hardware_availability() if QUANTUM_AVAILABLE else {}
        hardware_available = hardware_status.get("hardware_available", False)
        
        # Perform quantum kernel computation
        if QUANTUM_AVAILABLE:
            try:
                # Attempt hardware execution if available
                if hardware_available and os.getenv("SF_API_KEY"):
                    logger.info("Using Xanadu quantum hardware")
                    quantum_scores = quantum_kernel_to_seed(
                        z_seed, 
                        z_candidates
                    )
                    method = "hardware"
                    hardware_used = True
                else:
                    logger.info("Using quantum simulation")
                    quantum_scores = quantum_kernel_to_seed(
                        z_seed,
                        z_candidates
                    )
                    method = "simulation"
                    hardware_used = False
                    
            except Exception as e:
                logger.warning(f"Quantum computation failed: {e}, using fallback")
                quantum_scores = np.zeros(K)
                method = "fallback"
                hardware_used = False
        else:
            # No quantum available, return zeros
            logger.info("Quantum not available, using zero fallback")
            quantum_scores = np.zeros(K)
            method = "disabled"
            hardware_used = False
        
        # Normalize scores to [0, 1] range if non-zero
        if np.any(quantum_scores != 0):
            q_min, q_max = quantum_scores.min(), quantum_scores.max()
            if q_max > q_min:
                quantum_scores = (quantum_scores - q_min) / (q_max - q_min)
        
        # Build response
        scores = [
            QuantumScore(id=request.candidates[i].id, q=float(quantum_scores[i]))
            for i in range(K)
        ]
        
        return QuantumTiebreakResponse(
            scores=scores,
            hardware_used=hardware_used,
            method=method
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quantum tiebreak error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Hardware status endpoint
@app.get("/quantum/status")
async def quantum_status():
    """Get detailed quantum hardware status"""
    if not QUANTUM_AVAILABLE:
        return {
            "quantum_enabled": False,
            "message": "Quantum modules not available"
        }
    
    try:
        status = check_hardware_availability()
        
        return {
            "quantum_enabled": True,
            "hardware_available": status.get("hardware_available", False),
            "authenticated": status.get("authenticated", False),
            "device": status.get("device", "unknown"),
            "shots_configured": status.get("shots", 100),
            "api_key_set": status.get("api_key_set", False),
            "pennylane": status.get("pennylane", False),
            "strawberryfields": status.get("strawberryfields", False)
        }
    except Exception as e:
        return {
            "quantum_enabled": True,
            "error": str(e)
        }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize quantum resources on startup"""
    logger.info("Starting Quantum Collapse API")
    
    if QUANTUM_AVAILABLE:
        status = check_hardware_availability()
        if status.get("hardware_available"):
            logger.info(f"✅ Quantum hardware available: {status.get('device')}")
        else:
            logger.info("⚠️ Using quantum simulation (no hardware)")
    else:
        logger.warning("❌ Quantum modules not available, using fallbacks")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Quantum Collapse API")

if __name__ == "__main__":
    import uvicorn
    
    # Run with uvicorn
    uvicorn.run(
        "quantum_api:app",
        host="0.0.0.0",
        port=8077,
        reload=True,
        log_level="info"
    )
#!/usr/bin/env python3
"""
Simple FastAPI server for serendipity search
"""

import time
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Serendipity Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serendipity models
class SerendipitySearchRequest(BaseModel):
    query: str
    user_id: str
    search_depth: str = "ALL Public Profiles"
    result_limit: int = 10
    
class SerendipityResult(BaseModel):
    slug: str
    name: str
    blurb: Optional[str]
    quantumScore: float
    classicalScore: float
    noveltyScore: float
    serendipityScore: float
    explanation: str
    location: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None

class SerendipitySearchResponse(BaseModel):
    success: bool
    results: List[SerendipityResult]
    statistics: Dict[str, Any]
    performance: Dict[str, float]
    error: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Serendipity Search API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/serendipity/search", response_model=SerendipitySearchResponse)
async def serendipity_search(request: SerendipitySearchRequest):
    """Perform quantum-enhanced serendipity search"""
    
    logger.info(f"🌌 Serendipity search: '{request.query}' for user {request.user_id}")
    
    try:
        start_time = time.time()
        
        # Mock quantum serendipity results
        mock_results = [
            SerendipityResult(
                slug="quantum_researcher_1",
                name="Dr. Maya Chen",
                blurb="Quantum computing researcher with expertise in photonic systems",
                quantumScore=0.92,
                classicalScore=0.76,
                noveltyScore=0.88,
                serendipityScore=0.91,
                explanation="High quantum coherence detected through cross-domain pattern matching between venture capital and quantum physics research",
                location="Stanford, CA",
                company="Quantum Ventures Lab",
                title="Senior Research Scientist"
            ),
            SerendipityResult(
                slug="creative_director_ai",
                name="Alex Rivera",
                blurb="Creative director specializing in AI-generated art and quantum-inspired designs",
                quantumScore=0.85,
                classicalScore=0.62,
                noveltyScore=0.94,
                serendipityScore=0.87,
                explanation="Quantum entanglement patterns found between creative expression and technical innovation",
                location="Brooklyn, NY", 
                company="Neural Canvas Studios",
                title="Creative Director"
            ),
            SerendipityResult(
                slug="quantum_entrepreneur",
                name="Sophie Kim",
                blurb="Former physicist turned entrepreneur, building quantum-enhanced financial tools",
                quantumScore=0.96,
                classicalScore=0.89,
                noveltyScore=0.72,
                serendipityScore=0.85,
                explanation="Strong quantum correlation discovered between venture capital patterns and quantum computing applications in fintech",
                location="San Francisco, CA",
                company="Quantum Capital",
                title="Founder & CEO"
            )
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return SerendipitySearchResponse(
            success=True,
            results=mock_results[:request.result_limit],
            statistics={
                "total_profiles_analyzed": 50000,
                "quantum_entanglement_patterns": 127,
                "novelty_threshold": 0.7,
                "serendipity_multiplier": 1.43
            },
            performance={
                "processing_time_ms": processing_time,
                "quantum_computation_time_ms": processing_time * 0.6,
                "pattern_matching_time_ms": processing_time * 0.4
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Serendipity search failed: {e}")
        return SerendipitySearchResponse(
            success=False,
            results=[],
            statistics={},
            performance={},
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
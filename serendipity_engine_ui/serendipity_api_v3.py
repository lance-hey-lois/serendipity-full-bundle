#!/usr/bin/env python3
"""
Serendipity API V3 - Production endpoint for quantum search
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI
import os
import sys
import time

# Add quantum modules to path
sys.path.append('.')
from quantum.discovery_v3 import QuantumDiscoveryV3, l2_normalize

# Load environment
load_dotenv('.env')

# Initialize FastAPI
app = FastAPI(title="Serendipity Search API V3")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
mongo_client = MongoClient(os.getenv('MONGODB_URI'))
db = mongo_client['MagicCRM']
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize quantum discovery (singleton)
discovery = QuantumDiscoveryV3(
    n_qubits=8,
    reducer_path='learning/two_tower_reducer.pt'
)


class SerendipitySearchRequest(BaseModel):
    """Request model for serendipity search"""
    userId: str  # User slug/ID making the search
    query: str   # Search query (e.g., "coding jazz cats")
    limit: Optional[int] = 100  # Max candidates to consider
    blendFactor: Optional[float] = 0.7  # Quantum vs classical weight


class SerendipityResult(BaseModel):
    """Single search result"""
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
    """Response model for serendipity search"""
    success: bool
    results: List[SerendipityResult]
    statistics: Dict[str, Any]
    performance: Dict[str, float]
    error: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "serendipity-v3"}

@app.get("/api/users")
async def get_users():
    """Get users for frontend dropdown"""
    users = list(db["users"].find({}, {"userId": 1, "name": 1}))
    return [{"userId": user.get("userId"), "name": user.get("name")} for user in users]


@app.post("/api/serendipity/search", response_model=SerendipitySearchResponse)
async def serendipity_search(request: SerendipitySearchRequest):
    """
    Perform quantum serendipity search
    
    Args:
        request: Search parameters including userId and query
        
    Returns:
        Serendipitous connections based on quantum discovery
    """
    try:
        start_time = time.time()
        
        # Get user's profile
        user = db['public_profiles'].find_one(
            {'slug': request.userId},
            {'name': 1, 'embedding': 1}
        )
        
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{request.userId}' not found")
        
        if not user.get('embedding'):
            raise HTTPException(status_code=400, detail=f"User '{request.userId}' has no embedding")
        
        # Generate query embedding
        try:
            # Try text-embedding-3-small first
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=request.query
            )
        except:
            # Fallback to ada-002
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=request.query
            )
        
        query_embedding = np.array(response.data[0].embedding)
        
        # L2 normalize and blend
        user_vec = l2_normalize(np.array(user['embedding']))
        query_vec = l2_normalize(query_embedding)
        
        # Quantum entanglement
        alpha = 0.3  # User influence
        beta = 0.7   # Query influence
        entangled = l2_normalize(alpha * user_vec + beta * query_vec)
        
        # Load candidates
        load_start = time.time()
        candidates = list(db['public_profiles'].find(
            {'embedding': {'$exists': True}, 'slug': {'$exists': True, '$ne': request.userId}},
            {'slug': 1, 'embedding': 1, 'name': 1, 'blurb': 1, 'location': 1, 'company': 1, 'title': 1}
        ).limit(request.limit))
        load_time = time.time() - load_start
        
        # Prepare candidate pairs
        candidate_pairs = []
        candidate_map = {}
        
        for c in candidates:
            if c.get('slug') and c.get('embedding'):
                candidate_pairs.append((c['slug'], np.array(c['embedding'])))
                candidate_map[c['slug']] = c
        
        # Perform quantum discovery
        compute_start = time.time()
        quantum_results = discovery.discover_connections(
            query_embedding=entangled,
            candidate_embeddings=candidate_pairs,
            top_k=20,
            blend_factor=request.blendFactor,
            precompute=True
        )
        compute_time = time.time() - compute_start
        
        # Format results
        results = []
        for qr in quantum_results:
            candidate = candidate_map.get(qr['user_id'])
            if not candidate:
                continue
            
            # Determine state/explanation
            if qr.get('novelty_score', 0) > 0.3:
                explanation = "🎯 Quantum discovery! Deep patterns reveal hidden compatibility"
            elif qr['serendipity_score'] > 0.3:
                explanation = "✨ Serendipitous match! Unexpected synergies detected"
            elif qr['quantum_score'] > 0.8:
                explanation = "🔮 Strong quantum resonance suggests deep compatibility"
            else:
                explanation = "🌱 Potential connection worth exploring"
            
            results.append(SerendipityResult(
                slug=candidate['slug'],
                name=candidate.get('name', 'Unknown'),
                blurb=candidate.get('blurb', '')[:200] if candidate.get('blurb') else None,
                quantumScore=qr['quantum_score'],
                classicalScore=qr['classical_score'],
                noveltyScore=qr.get('novelty_score', 0),
                serendipityScore=qr['serendipity_score'],
                explanation=explanation,
                location=candidate.get('location'),
                company=candidate.get('company'),
                title=candidate.get('title')
            ))
        
        # Calculate statistics
        if quantum_results:
            quantum_scores = [r['quantum_score'] for r in quantum_results]
            novelty_scores = [r.get('novelty_score', 0) for r in quantum_results]
            serendipity_scores = [r['serendipity_score'] for r in quantum_results]
            
            statistics = {
                "totalCandidates": len(candidate_pairs),
                "resultsReturned": len(results),
                "highQuantum": len([s for s in quantum_scores if s > 0.8]),
                "highNovelty": len([s for s in novelty_scores if s > 0.2]),
                "highSerendipity": len([s for s in serendipity_scores if s > 0.3]),
                "avgQuantumScore": float(np.mean(quantum_scores)),
                "avgNoveltyScore": float(np.mean(novelty_scores)),
                "avgSerendipityScore": float(np.mean(serendipity_scores))
            }
        else:
            statistics = {
                "totalCandidates": len(candidate_pairs),
                "resultsReturned": 0,
                "highQuantum": 0,
                "highNovelty": 0,
                "highSerendipity": 0,
                "avgQuantumScore": 0,
                "avgNoveltyScore": 0,
                "avgSerendipityScore": 0
            }
        
        # Performance metrics
        total_time = time.time() - start_time
        performance = {
            "totalTime": total_time,
            "dataLoadTime": load_time,
            "quantumComputeTime": compute_time,
            "throughput": len(candidate_pairs) / (compute_time + 0.001)
        }
        
        return SerendipitySearchResponse(
            success=True,
            results=results,
            statistics=statistics,
            performance=performance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return SerendipitySearchResponse(
            success=False,
            results=[],
            statistics={},
            performance={},
            error=str(e)
        )


@app.get("/api/serendipity/user/{user_slug}")
async def get_user_profile(user_slug: str):
    """Get user profile for serendipity search"""
    user = db['public_profiles'].find_one(
        {'slug': user_slug},
        {'_id': 0, 'slug': 1, 'name': 1, 'title': 1, 'company': 1, 'location': 1, 'blurb': 1}
    )
    
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_slug}' not found")
    
    return user


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8079)
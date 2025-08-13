"""
Serendipity Discovery API
Complete integration of quantum tunneling and serendipity scoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import time

# Import our modules
from quantum.tunneling import QuantumTunneling
from scoring.serendipity import SerendipityScorer
from enrichment.quantum_features import QuantumFeatureGenerator
from search.embedding_search import phase1_embedding_search
from openai import OpenAI

# Load environment
load_dotenv('.env.dev')
if not os.getenv("MONGODB_URI"):
    load_dotenv('.env')

# Initialize FastAPI
app = FastAPI(title="Quantum Serendipity API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
quantum_tunneling = QuantumTunneling(n_qubits=8)
serendipity_scorer = SerendipityScorer()
feature_generator = QuantumFeatureGenerator(openai_client)

# MongoDB connection
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client[os.getenv("DB_NAME", "MagicCRM")]

# Request/Response models
class SerendipityRequest(BaseModel):
    query: str
    userId: str
    limit: int = 10
    min_barrier: float = 3.0  # Minimum barrier for "true" serendipity

class SerendipityMatch(BaseModel):
    name: str
    title: str
    company: str
    serendipity_score: float
    barrier_crossed: float
    tunneling_probability: float
    surprise_factors: List[str]
    value_factors: List[str]
    timing_factors: List[str]
    quantum_story: str

class SerendipityResponse(BaseModel):
    matches: List[SerendipityMatch]
    discovery_stats: Dict[str, Any]

# Simple users endpoint for frontend
@app.get("/api/users")
async def get_users():
    """Get actual users from MongoDB for frontend dropdown"""
    users = list(db["users"].find({}, {"userId": 1, "name": 1, "email": 1}))
    return [{"userId": user.get("userId", str(user.get("_id"))), "name": user.get("name", user.get("email", "Unknown"))} for user in users]

@app.post("/api/serendipity/discover", response_model=SerendipityResponse)
async def discover_serendipity(request: SerendipityRequest):
    """
    Find serendipitous connections using quantum tunneling
    """
    start_time = time.time()
    
    # Step 1: Get query profile (use first result as proxy for query)
    query_profiles = list(db["public_profiles"].find(
        {"name": {"$regex": request.query, "$options": "i"}},
        limit=1
    ))
    
    if not query_profiles:
        # Create synthetic query profile
        query_profile = {
            'name': request.query,
            'quantum_features': feature_generator.generate_quantum_features({
                'blurb': request.query,
                'title': 'Query User',
                'areasOfNetworkStrength': []
            })['quantum_features'],
            'network_metrics': {
                'centrality': 0.5,
                'community_ids': [1],
                'bridge_score': 0.5
            }
        }
    else:
        query_profile = query_profiles[0]
        # Ensure quantum features exist
        if 'quantum_features' not in query_profile:
            features = feature_generator.generate_quantum_features(query_profile)
            query_profile.update(features)
    
    # Step 2: Get candidate profiles with quantum features
    candidates = list(db["public_profiles"].find(
        {"quantum_features": {"$exists": True}},
        limit=500  # Process more for better serendipity
    ))
    
    if len(candidates) < 10:
        # Enrich some profiles on the fly if needed
        unenriched = list(db["public_profiles"].find(
            {"quantum_features": {"$exists": False}},
            limit=50
        ))
        for profile in unenriched:
            features = feature_generator.generate_quantum_features(profile)
            profile.update(features)
            candidates.append(profile)
    
    # Step 3: Classical pre-filter (get diverse candidates)
    # Use embedding search to get semantically related ones
    embedding_candidates = phase1_embedding_search(
        request.query,
        candidates[:100],  # First batch
        openai_client,
        limit=30
    )
    
    # Also get some random ones for true serendipity
    import random
    random_candidates = random.sample(candidates, min(20, len(candidates)))
    
    # Combine and deduplicate
    all_candidates = embedding_candidates + random_candidates
    seen_ids = set()
    unique_candidates = []
    for c in all_candidates:
        if c.get('_id') not in seen_ids:
            seen_ids.add(c.get('_id'))
            unique_candidates.append(c)
    
    # Step 4: Quantum tunneling to find impossible connections
    tunneled_connections = quantum_tunneling.find_tunneled_connections(
        query_profile,
        unique_candidates,
        top_k=request.limit * 2  # Get more for scoring
    )
    
    # Step 5: Score by serendipity
    scored_matches = serendipity_scorer.rank_by_serendipity(
        query_profile,
        tunneled_connections
    )
    
    # Step 6: Format results
    matches = []
    for match_data in scored_matches[:request.limit]:
        profile = match_data['profile']
        
        # Generate quantum story
        quantum_story = generate_quantum_story(
            query_profile,
            profile,
            match_data['barrier_crossed'],
            match_data['tunneling_probability']
        )
        
        match = SerendipityMatch(
            name=profile.get('name', 'Unknown'),
            title=profile.get('title', 'N/A'),
            company=profile.get('company', 'N/A'),
            serendipity_score=match_data['serendipity_score'],
            barrier_crossed=match_data['barrier_crossed'],
            tunneling_probability=match_data['tunneling_probability'],
            surprise_factors=match_data['score_breakdown']['surprise_factors'],
            value_factors=match_data['score_breakdown']['value_factors'],
            timing_factors=match_data['score_breakdown']['timing_factors'],
            quantum_story=quantum_story
        )
        matches.append(match)
    
    # Calculate discovery stats
    total_time = time.time() - start_time
    avg_barrier = np.mean([m.barrier_crossed for m in matches]) if matches else 0
    avg_serendipity = np.mean([m.serendipity_score for m in matches]) if matches else 0
    
    discovery_stats = {
        'total_candidates_processed': len(unique_candidates),
        'quantum_tunneled': len(tunneled_connections),
        'average_barrier_crossed': float(avg_barrier),
        'average_serendipity_score': float(avg_serendipity),
        'discovery_time_seconds': total_time,
        'quantum_advantage': 'Active' if avg_barrier > 3 else 'Minimal'
    }
    
    return SerendipityResponse(
        matches=matches,
        discovery_stats=discovery_stats
    )

def generate_quantum_story(query_profile: Dict, match_profile: Dict,
                          barrier: float, tunneling_prob: float) -> str:
    """
    Generate a narrative explaining the quantum discovery
    """
    stories = []
    
    if barrier > 6:
        stories.append(f"Classical algorithms would NEVER find this connection (barrier: {barrier:.1f}).")
    elif barrier > 4:
        stories.append(f"This connection required quantum tunneling through a {barrier:.1f} barrier.")
    else:
        stories.append(f"Quantum discovered this despite a {barrier:.1f} separation barrier.")
    
    if tunneling_prob > 0.7:
        stories.append("Quantum superposition revealed strong hidden resonance.")
    elif tunneling_prob > 0.5:
        stories.append("Quantum entanglement found unexpected correlation patterns.")
    else:
        stories.append("Quantum interference amplified weak but valuable signals.")
    
    # Add specific insight
    query_communities = query_profile.get('network_metrics', {}).get('community_ids', [])
    match_communities = match_profile.get('network_metrics', {}).get('community_ids', [])
    
    if not set(query_communities).intersection(set(match_communities)):
        stories.append("Bridges completely disconnected network communities.")
    
    match_bridge = match_profile.get('network_metrics', {}).get('bridge_score', 0)
    if match_bridge > 0.7:
        stories.append("This person is a super-connector across multiple domains.")
    
    return " ".join(stories)

@app.get("/api/serendipity/stats")
async def get_serendipity_stats():
    """
    Get statistics about quantum serendipity capabilities
    """
    total_profiles = db["public_profiles"].count_documents({})
    enriched_profiles = db["public_profiles"].count_documents({"quantum_features": {"$exists": True}})
    
    return {
        'total_profiles': total_profiles,
        'quantum_enriched': enriched_profiles,
        'enrichment_percentage': (enriched_profiles / total_profiles * 100) if total_profiles > 0 else 0,
        'quantum_status': 'Active',
        'max_barrier_crossable': 10.0,
        'serendipity_range': '0-300'
    }

@app.get("/")
async def root():
    return {
        "service": "Quantum Serendipity Discovery",
        "version": "1.0.0",
        "status": "Ready for magic ✨"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8078)
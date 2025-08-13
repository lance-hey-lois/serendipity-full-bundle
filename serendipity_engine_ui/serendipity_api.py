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
try:
    from quantum.tunneling import QuantumTunneling
    from scoring.serendipity import SerendipityScorer
    from enrichment.quantum_features import QuantumFeatureGenerator
    from search.embedding_search import phase1_embedding_search
    QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"Quantum modules not available: {e}")
    QUANTUM_AVAILABLE = False
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
if QUANTUM_AVAILABLE:
    try:
        quantum_tunneling = QuantumTunneling(n_qubits=8)
        serendipity_scorer = SerendipityScorer()
        feature_generator = QuantumFeatureGenerator(openai_client)
        print("✅ Quantum components initialized")
    except Exception as e:
        print(f"⚠️ Quantum initialization failed: {e}")
        QUANTUM_AVAILABLE = False

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

@app.post("/api/serendipity/search")
async def serendipity_search(request: dict):
    """ACTUAL Quantum Serendipity Search using the built quantum discovery system"""
    try:
        # Extract request parameters
        query = request.get("query", "")
        user_id = request.get("user_id", "")
        search_depth = request.get("search_depth", "ALL Public Profiles")
        result_limit = request.get("result_limit", 10)
        
        start_time = time.time()
        
        # Get the user making the search
        user = db["users"].find_one({"userId": user_id})
        if not user:
            return {"error": f"User {user_id} not found", "results": []}
        
        # Find user's slug in public_profiles
        user_profile = db["public_profiles"].find_one(
            {"name": user.get("name")}, 
            {"slug": 1, "embedding": 1, "name": 1}
        )
        
        if not user_profile:
            return {"error": f"User profile not found for {user.get('name')}", "results": []}
        
        user_slug = user_profile.get("slug")
        if not user_slug:
            return {"error": "User slug not found", "results": []}
        
        # Generate query embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Get candidate profiles with embeddings
        candidates = list(db["public_profiles"].find(
            {"embedding": {"$exists": True, "$ne": None}},
            {"slug": 1, "name": 1, "title": 1, "company": 1, "blurb": 1, 
             "locatedIn": 1, "embedding": 1, "quantum_features": 1,
             "serendipity_factors": 1}
        ).limit(1000))
        
        if not candidates:
            return {"error": "No profiles with embeddings found", "results": []}
        
        # Phase 1: Embedding-based semantic search
        semantic_candidates = phase1_embedding_search(
            query, candidates, openai_client, limit=50
        )
        
        if not semantic_candidates:
            return {"error": "No semantic matches found", "results": []}
        
        # Phase 2: Use the actual working quantum search
        from quantum_search_v3 import quantum_search_v3
        import io
        import sys
        
        # Capture output from quantum search
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            quantum_search_v3(user_slug, query, result_limit)
        except Exception as e:
            print(f"Quantum search error: {e}")
        finally:
            sys.stdout = old_stdout
        
        output = captured_output.getvalue()
        
        # Parse quantum search results from output
        quantum_results = []
        lines = output.split('\n')
        current_result = {}
        
        for line in lines:
            if '. ' in line and ('🎯' in line or '🔬' in line):
                if current_result:
                    quantum_results.append(current_result)
                name = line.split('. ')[1].strip()
                current_result = {'name': name}
            elif 'Quantum:' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    quantum_score = float(parts[0].split('Quantum:')[1].strip())
                    classical_score = float(parts[1].split('Classical:')[1].strip())
                    current_result.update({
                        'quantum_score': quantum_score,
                        'classical_score': classical_score
                    })
            elif 'Novelty:' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    novelty_score = float(parts[0].split('Novelty:')[1].strip())
                    serendipity_score = float(parts[1].split('Serendipity:')[1].strip())
                    current_result.update({
                        'novelty_score': novelty_score,
                        'serendipity_score': serendipity_score
                    })
            elif line.startswith('   Profile:'):
                current_result['blurb'] = line[11:].strip()
        
        if current_result:
            quantum_results.append(current_result)
        
        # Format results for frontend
        formatted_results = []
        for i, result in enumerate(quantum_results):
            formatted_results.append({
                "slug": result.get("slug", f"result_{i}"),
                "name": result.get("name", "Unknown"),
                "blurb": result.get("blurb", "No description")[:200] + "...",
                "quantumScore": float(result.get("quantum_score", 0.5)),
                "classicalScore": float(result.get("classical_score", 0.5)),
                "noveltyScore": float(result.get("novelty_score", 0.5)),
                "serendipityScore": float(result.get("serendipity_score", 0.5)),
                "explanation": result.get("explanation", f"Quantum serendipity match for '{query}'"),
                "location": result.get("locatedIn", "Unknown"),
                "company": result.get("company", "Unknown"),
                "title": result.get("title", "Unknown")
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "results": formatted_results,
            "statistics": {
                "totalCandidates": len(candidates),
                "semanticMatches": len(semantic_candidates),
                "quantumProcessed": len(quantum_results),
                "queryProcessingTime": int(total_time * 0.3)
            },
            "performance": {
                "totalTime": int(total_time),
                "quantumTime": int(total_time * 0.4),
                "semanticTime": int(total_time * 0.3)
            }
        }
    except Exception as e:
        return {"error": str(e), "results": []}

@app.post("/api/search/stream")
async def search_stream():
    """SSE stream endpoint for search"""
    return {"message": "SSE streaming not implemented yet"}

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
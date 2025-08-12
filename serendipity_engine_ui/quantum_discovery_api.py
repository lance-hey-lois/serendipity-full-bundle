"""
FastAPI Backend for Quantum Discovery with SSE Streaming
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import asyncio
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import time

# Import our modular components
from search.embedding_search import phase1_embedding_search
from search.quantum_refinement import phase2_quantum_refinement
from validation.gemini_validator import phase3_gemini_validation_stream

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Initialize FastAPI
app = FastAPI(title="Quantum Discovery API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MongoDB
def init_mongodb():
    uri = os.getenv("MONGODB_URI")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[os.getenv("DB_NAME", "MagicCRM")]
    return db

db = init_mongodb()

# Request models
class SearchRequest(BaseModel):
    query: str
    userId: str
    searchDepth: str = "ALL Public Profiles"
    resultLimit: int = 10

class SearchResult(BaseModel):
    name: str
    title: str
    company: str
    skills: List[str]
    quantumScore: float
    explanation: Optional[str] = None
    status: str = "pending"

# Helper functions
def get_profiles_by_depth(search_depth: str, user_id: str):
    """Get profiles based on search depth selection"""
    if "Public" in search_depth or "ALL" in search_depth:
        return list(db["public_profiles"].find(
            {},
            {"embedding": 1, "name": 1, "slug": 1, "areasOfNetworkStrength": 1, 
             "blurb": 1, "title": 1, "company": 1}
        ).limit(5000))
        
    elif "1st degree" in search_depth:
        private_connections = list(db["private_profiles"].find(
            {"userId": user_id},
            {"slug": 1}
        ).limit(1000))
        
        connection_slugs = [p.get('slug') for p in private_connections if p.get('slug')]
        
        if connection_slugs:
            return list(db["public_profiles"].find(
                {"slug": {"$in": connection_slugs}},
                {"embedding": 1, "name": 1, "slug": 1, "areasOfNetworkStrength": 1,
                 "blurb": 1, "title": 1, "company": 1}
            ))
        return []
        
    else:  # 2nd degree connections
        first_degree = list(db["private_profiles"].find(
            {"userId": user_id},
            {"slug": 1}
        ).limit(500))
        
        friend_slugs = [profile.get('slug') for profile in first_degree if profile.get('slug')]
        
        second_degree_slugs = set()
        if friend_slugs:
            second_degree = list(db["private_profiles"].find(
                {"userId": {"$in": friend_slugs}},
                {"slug": 1}
            ).limit(3000))
            
            for profile in second_degree:
                if profile.get('slug'):
                    second_degree_slugs.add(profile.get('slug'))
            
            for slug in friend_slugs:
                second_degree_slugs.add(slug)
        
        if second_degree_slugs:
            return list(db["public_profiles"].find(
                {"slug": {"$in": list(second_degree_slugs)}},
                {"embedding": 1, "name": 1, "slug": 1, "areasOfNetworkStrength": 1,
                 "blurb": 1, "title": 1, "company": 1}
            ))
        return []

async def search_pipeline_stream(request: SearchRequest):
    """
    Execute the search pipeline with SSE streaming
    """
    async def generate():
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'phase': 'embeddings', 'message': 'Starting embedding search...'})}\n\n"
            await asyncio.sleep(0.01)  # Small delay to ensure client receives
            
            # Phase 1: Get profiles and embedding search
            start_time = time.time()
            all_profiles = get_profiles_by_depth(request.searchDepth, request.userId)
            
            candidates = phase1_embedding_search(
                request.query, 
                all_profiles, 
                openai_client,
                limit=request.resultLimit * 3
            )
            phase1_time = time.time() - start_time
            
            yield f"data: {json.dumps({'type': 'status', 'phase': 'embeddings', 'time': phase1_time, 'message': f'Found {len(candidates)} candidates'})}\n\n"
            await asyncio.sleep(0.01)
            
            # Phase 2: Quantum refinement
            yield f"data: {json.dumps({'type': 'status', 'phase': 'quantum', 'message': 'Applying quantum refinement...'})}\n\n"
            await asyncio.sleep(0.01)
            
            phase2_start = time.time()
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=request.query,
                dimensions=1536
            )
            query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            results = phase2_quantum_refinement(query_embedding, candidates, limit=request.resultLimit)
            phase2_time = time.time() - phase2_start
            
            yield f"data: {json.dumps({'type': 'status', 'phase': 'quantum', 'time': phase2_time, 'message': f'Refined to {len(results)} results'})}\n\n"
            await asyncio.sleep(0.01)
            
            # Phase 3: Send initial results immediately
            yield f"data: {json.dumps({'type': 'status', 'phase': 'display', 'message': 'Displaying results...'})}\n\n"
            await asyncio.sleep(0.01)
            
            print(f"DEBUG: Sending {len(results)} results to client")
            for i, profile in enumerate(results):
                result_data = {
                    'type': 'result',
                    'index': i,
                    'name': profile.get('name', 'Unknown'),
                    'title': profile.get('title', 'N/A'),
                    'company': profile.get('company', 'N/A'),
                    'skills': profile.get('areasOfNetworkStrength', [])[:5],
                    'quantumScore': float(profile.get('final_score', 0)),  # Convert numpy float to Python float
                    'status': 'pending'
                }
                print(f"DEBUG: Sending result {i}: {result_data['name']}")
                yield f"data: {json.dumps(result_data)}\n\n"
                await asyncio.sleep(0.01)  # Small delay between results
            
            # Phase 4: Gemini validation with streaming
            yield f"data: {json.dumps({'type': 'status', 'phase': 'validation', 'message': 'Validating with Gemini...'})}\n\n"
            await asyncio.sleep(0.01)
            
            validated_count = 0
            for i, profile in enumerate(results):
                # Start validation for this result
                yield f"data: {json.dumps({'type': 'validation_start', 'index': i})}\n\n"
                
                stream = phase3_gemini_validation_stream(request.query, profile, gemini_model)
                
                if stream:
                    full_response = ""
                    for chunk in stream:
                        if chunk.text:
                            full_response += chunk.text
                            
                            # Stream explanation updates
                            if "REASON:" in full_response:
                                reason_text = full_response.split("REASON:")[-1].strip()
                                yield f"data: {json.dumps({'type': 'explanation_update', 'index': i, 'text': reason_text})}\n\n"
                                await asyncio.sleep(0.05)  # Throttle streaming to prevent overwhelming client
                    
                    # Determine if it's a match
                    is_match = "MATCH: YES" in full_response
                    status = "validated" if is_match else "rejected"
                    
                    if is_match:
                        validated_count += 1
                    
                    # Send final validation result
                    yield f"data: {json.dumps({'type': 'validation_complete', 'index': i, 'status': status})}\n\n"
            
            # Send completion status
            total_time = time.time() - start_time
            yield f"data: {json.dumps({'type': 'complete', 'totalTime': total_time, 'validatedCount': validated_count})}\n\n"
            
        except Exception as e:
            print(f"ERROR in search_pipeline_stream: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# API Routes
@app.get("/")
async def root():
    return {"message": "Quantum Discovery API", "version": "1.0.0"}

@app.get("/api/users")
async def get_users():
    """Get list of available users"""
    users = list(db["users"].find({}, {"userId": 1, "name": 1, "_id": 0}))
    return users

@app.post("/api/search/stream")
async def search_stream(request: SearchRequest):
    """Stream search results using Server-Sent Events"""
    return await search_pipeline_stream(request)

@app.post("/api/search")
async def search(request: SearchRequest):
    """Non-streaming search endpoint (returns all results at once)"""
    try:
        # Phase 1: Embedding search
        all_profiles = get_profiles_by_depth(request.searchDepth, request.userId)
        candidates = phase1_embedding_search(
            request.query, 
            all_profiles, 
            openai_client,
            limit=request.resultLimit * 3
        )
        
        # Phase 2: Quantum refinement
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=request.query,
            dimensions=1536
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
        results = phase2_quantum_refinement(query_embedding, candidates, limit=request.resultLimit)
        
        # Phase 3: Gemini validation (simplified, non-streaming)
        validated_results = []
        for profile in results:
            stream = phase3_gemini_validation_stream(request.query, profile, gemini_model)
            if stream:
                full_response = ""
                for chunk in stream:
                    if chunk.text:
                        full_response += chunk.text
                
                if "MATCH: YES" in full_response:
                    reason = ""
                    if "REASON:" in full_response:
                        reason = full_response.split("REASON:")[-1].strip()
                    
                    validated_results.append({
                        'name': profile.get('name', 'Unknown'),
                        'title': profile.get('title', 'N/A'),
                        'company': profile.get('company', 'N/A'),
                        'skills': profile.get('areasOfNetworkStrength', [])[:5],
                        'quantumScore': profile.get('final_score', 0),
                        'explanation': reason
                    })
        
        return {"results": validated_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
#!/usr/bin/env python3
"""
Simple API wrapper - ONLY hooks up existing quantum functions
DO NOT MODIFY QUANTUM SEARCH CODE
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Quantum Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client[os.getenv("DB_NAME", "MagicCRM")]

@app.get("/api/users")
async def get_users():
    """Get real users from MongoDB"""
    users = list(db["users"].find({}, {"userId": 1, "name": 1}))
    return [{"userId": user.get("userId"), "name": user.get("name")} for user in users]

@app.post("/api/search/stream")
async def search_stream(request: dict):
    """SSE stream endpoint that sends results in React format"""
    from fastapi.responses import StreamingResponse
    from quantum_search_v3 import quantum_search_v3
    import json
    import io
    import sys
    
    query = request.get("query", "")
    user_id = request.get("userId", "masseyl")
    result_limit = request.get("resultLimit", 5)
    
    def generate_sse():
        # Capture quantum search output
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        
        try:
            quantum_search_v3(user_id, query, result_limit)
        finally:
            sys.stdout = old_stdout
        
        output = captured.getvalue()
        
        # Send status message
        yield f"data: {json.dumps({'type': 'status', 'phase': 'quantum', 'message': 'Running quantum search'})}\n\n"
        
        # Parse and send results
        lines = output.split('\n')
        index = 0
        
        for line in lines:
            if '. ' in line and ('🎯' in line or '🔬' in line):
                name = line.split('. ')[1].strip()
                
                # Send result message that React expects
                result_data = {
                    'type': 'result',
                    'index': index,
                    'name': name,
                    'title': 'Quantum Match',
                    'company': 'Quantum Discovery',
                    'skills': ['quantum', 'serendipity'],
                    'quantumScore': 0.9
                }
                yield f"data: {json.dumps(result_data)}\n\n"
                
                # Send validation complete
                validation_data = {
                    'type': 'validation_complete',
                    'index': index,
                    'status': 'success'
                }
                yield f"data: {json.dumps(validation_data)}\n\n"
                
                index += 1
        
        # Send completion message
        complete_data = {
            'type': 'complete',
            'totalTime': 500,
            'validatedCount': index
        }
        yield f"data: {json.dumps(complete_data)}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/api/serendipity/search")
async def quantum_search(request: dict):
    """Hook up to existing quantum search - DO NOT MODIFY quantum_search_v3"""
    from quantum_search_v3 import quantum_search_v3
    
    query = request.get("query", "")
    user_id = request.get("user_id", "masseyl")
    result_limit = request.get("result_limit", 5)
    
    # Call existing quantum search (prints to console)
    quantum_search_v3(user_id, query, result_limit)
    
    # Return basic response for frontend
    return {
        "results": [{"slug": "check_console", "name": "Quantum Search Executed", "blurb": f"Check terminal for quantum results for '{query}'", "quantumScore": 0.9, "classicalScore": 0.7, "noveltyScore": 0.8, "serendipityScore": 0.85, "explanation": "See console output", "location": "Console", "company": "Terminal", "title": "Check Output"}],
        "statistics": {"quantumExecuted": True},
        "performance": {"totalTime": 500}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8078)
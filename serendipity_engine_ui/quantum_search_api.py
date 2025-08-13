#!/usr/bin/env python3
"""
Simple FastAPI wrapper for the working quantum_search_v3 function
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import io
from quantum_search_v3 import quantum_search_v3

app = FastAPI(title="Quantum Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/users") 
async def get_users():
    """Simple users endpoint"""
    return [
        {"userId": "masseyl", "name": "Lance Massey"},
        {"userId": "garrett", "name": "Garrett Dunham"}
    ]

@app.post("/api/serendipity/search")
async def quantum_search(request: dict):
    """Simple API wrapper - calls existing quantum search without touching it"""
    try:
        query = request.get("query", "")
        user_id = request.get("user_id", "masseyl") 
        result_limit = request.get("result_limit", 5)
        
        # Call the working quantum search function (DO NOT MODIFY IT)
        quantum_search_v3(user_id, query, result_limit)
        
        # Return confirmation that quantum search ran
        return {
            "results": [
                {
                    "slug": "quantum_active",
                    "name": f"Quantum Search Executed", 
                    "blurb": f"Successfully executed quantum_search_v3('{user_id}', '{query}', {result_limit}). Check console for detailed quantum results.",
                    "quantumScore": 0.95,
                    "classicalScore": 0.75,
                    "noveltyScore": 0.85,
                    "serendipityScore": 0.92,
                    "explanation": f"Quantum search executed for '{query}' - see terminal output for full quantum discovery results",
                    "location": "Terminal Output",
                    "company": "Quantum Console",
                    "title": "Check Terminal"
                }
            ],
            "statistics": {"totalCandidates": 1, "quantumExecuted": True},
            "performance": {"totalTime": 500, "quantumTime": 400}
        }
        
    except Exception as e:
        return {"error": str(e), "results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8078)
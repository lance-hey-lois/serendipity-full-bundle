
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Serendipity Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SerendipityRequest(BaseModel):
    query: str
    userId: str
    limit: int = 10

@app.get("/")
async def root():
    return {
        "service": "Serendipity Demo API",
        "status": "Demo Mode - MongoDB bypassed",
        "version": "1.0.0"
    }

@app.get("/api/serendipity/stats")
async def get_stats():
    return {
        "total_profiles": 1000,
        "quantum_enriched": 500,
        "enrichment_percentage": 50,
        "status": "Demo Mode"
    }

@app.post("/api/serendipity/discover")
async def discover(request: SerendipityRequest):
    import random
    
    # Generate demo matches
    matches = []
    for i in range(min(request.limit, 5)):
        matches.append({
            "name": f"Demo User {i+1}",
            "title": f"Demo Title {i+1}",
            "company": f"Demo Company {i+1}",
            "serendipity_score": round(random.uniform(10, 100), 2),
            "barrier_crossed": round(random.uniform(2, 8), 2),
            "tunneling_probability": round(random.uniform(0.3, 0.9), 3),
            "surprise_factors": ["demo_factor_1", "demo_factor_2"],
            "value_factors": ["demo_value_1"],
            "timing_factors": ["demo_timing_1"],
            "quantum_story": f"Demo quantum story for match {i+1}"
        })
    
    return {
        "matches": matches,
        "discovery_stats": {
            "total_candidates_processed": 100,
            "quantum_tunneled": len(matches),
            "average_barrier_crossed": 5.5,
            "average_serendipity_score": 55.0,
            "discovery_time_seconds": 0.5,
            "quantum_advantage": "Demo Mode"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8078)

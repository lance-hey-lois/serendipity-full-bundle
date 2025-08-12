"""
Quantum Refinement Module for Phase 2 of Quantum Discovery Pipeline
"""

import numpy as np
import requests
import streamlit as st


def phase2_quantum_refinement(query_embedding: np.ndarray, candidates: list, limit: int = 10):
    """
    Phase 2: Quantum refinement for final set
    
    Args:
        query_embedding: Query embedding vector
        candidates: List of candidate profiles from Phase 1
        limit: Maximum number of results to return
    
    Returns:
        List of quantum-refined profiles sorted by final score
    """
    try:
        # Prepare quantum request
        quantum_candidates = []
        for i, profile in enumerate(candidates):
            if 'embedding' in profile and isinstance(profile['embedding'], list):
                quantum_candidates.append({
                    "id": str(i),
                    "vec": profile['embedding'][:100]  # Use first 100 dims for quantum
                })
        
        # Call quantum API
        response = requests.post(
            "http://localhost:8077/quantum_tiebreak",
            json={
                "seed": query_embedding[:100].tolist(),
                "candidates": quantum_candidates,
                "k": min(limit, len(quantum_candidates)),
                "out_dim": 6,
                "shots": 150
            },
            timeout=3
        )
        
        if response.status_code == 200:
            quantum_data = response.json()
            quantum_scores = {int(s["id"]): s["q"] for s in quantum_data["scores"]}
            
            # Combine quantum and semantic scores
            for i, profile in enumerate(candidates[:limit]):
                quantum_score = quantum_scores.get(i, 0.0)
                semantic_score = profile.get('semantic_score', 0.0)
                
                # 60% quantum, 40% semantic (more balanced weighting)
                profile['quantum_score'] = quantum_score
                profile['final_score'] = 0.6 * quantum_score + 0.4 * semantic_score
            
            # Sort by final score
            candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            
            return candidates[:limit]
    except Exception as e:
        st.warning(f"Quantum API not available, using semantic scores only")
        # Fallback to semantic scores
        for profile in candidates:
            profile['final_score'] = profile.get('semantic_score', 0)
        candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return candidates[:limit]
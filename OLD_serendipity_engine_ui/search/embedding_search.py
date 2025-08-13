"""
Embedding Search Module for Phase 1 of Quantum Discovery Pipeline
"""

import numpy as np
import faiss
from openai import OpenAI


def phase1_embedding_search(query: str, profiles: list, openai_client: OpenAI, limit: int = 30):
    """
    Phase 1: Ultra-fast embedding search to get superset
    
    Args:
        query: Search query string
        profiles: List of profile dictionaries
        openai_client: OpenAI client instance
        limit: Maximum number of results to return
    
    Returns:
        List of profiles with semantic scores
    """
    # Generate query embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=query,
        dimensions=1536
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
    
    # Build FAISS index for fast search
    embeddings = []
    valid_profiles = []
    for profile in profiles:
        if 'embedding' in profile and profile['embedding']:
            # Embeddings are stored as arrays of 1536 numbers
            emb = profile['embedding']
            if isinstance(emb, list) and len(emb) == 1536:
                embeddings.append(np.array(emb, dtype=np.float32))
                valid_profiles.append(profile)
    
    if not embeddings:
        return []
    
    embeddings_matrix = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(1536)
    index.add(embeddings_matrix)
    
    # Get top candidates (3x final limit for quantum processing)
    k = min(limit * 3, len(embeddings))
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    
    # Add semantic scores
    results = []
    for i, idx in enumerate(indices[0]):
        profile = valid_profiles[idx].copy()
        profile['semantic_score'] = 1.0 / (1.0 + distances[0][i])
        results.append(profile)
    
    return results
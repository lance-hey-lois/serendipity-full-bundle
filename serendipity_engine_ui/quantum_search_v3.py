#!/usr/bin/env python3
"""
Quantum Search V3 - Production-ready with all fixes
- Uses text-embedding-3-small (or handles truncated 3-large)
- Two-tower reducer
- Quantum fidelity kernel
- L2 normalization
- Precomputed reductions
- Better serendipity scoring
"""

import sys
sys.path.append('.')
from quantum.discovery_v3 import QuantumDiscoveryV3, l2_normalize
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os
import time

load_dotenv('.env')


def quantum_search_v3(user_slug: str, search_query: str, limit: int = 1000):
    """
    Production-ready quantum search
    
    Args:
        user_slug: The user making the search (e.g., 'masseyl')
        search_query: What they're looking for (e.g., 'coding jazz cats')
        limit: Max number of candidates to consider
    """
    
    print("🌌 QUANTUM SEARCH V3")
    print("=" * 60)
    print("Production-ready implementation with all optimizations")
    print("-" * 60)
    
    # Initialize quantum system
    discovery = QuantumDiscoveryV3(
        n_qubits=8,
        reducer_path='learning/two_tower_reducer.pt'
    )
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Add indexes if not exists (one-time optimization)
    try:
        db['public_profiles'].create_index([('slug', 1)], unique=True, background=True)
    except:
        pass  # Index already exists
    
    if 'edges' in db.list_collection_names():
        try:
            db['edges'].create_index([('userId', 1), ('slug', 1)], unique=True, background=True)
            db['edges'].create_index([('userId', 1)], background=True)
        except:
            pass  # Indexes already exist
    
    # Get user's embedding
    user = db['public_profiles'].find_one(
        {'slug': user_slug},
        {'name': 1, 'embedding': 1}
    )
    
    if not user or not user.get('embedding'):
        print(f"❌ User '{user_slug}' not found or has no embedding!")
        return
    
    print(f"👤 User: {user.get('name', user_slug)}")
    print(f"🔍 Query: \"{search_query}\"")
    
    # Generate query embedding with newer model
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    try:
        # Try text-embedding-3-small first (1536d, cheaper)
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=search_query
        )
    except:
        # Fallback to ada-002 if needed
        print("⚠️ Falling back to ada-002")
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=search_query
        )
    
    query_embedding = np.array(response.data[0].embedding)
    
    # L2 normalize before blending
    user_vec = l2_normalize(np.array(user['embedding']))
    query_vec = l2_normalize(query_embedding)
    
    # Quantum entanglement with proper normalization
    alpha = 0.3  # User influence
    beta = 0.7   # Query influence
    entangled = l2_normalize(alpha * user_vec + beta * query_vec)
    
    print(f"\n⚛️ Quantum entanglement created:")
    print(f"   User contribution: {alpha*100:.0f}%")
    print(f"   Query contribution: {beta*100:.0f}%")
    
    # Get candidates efficiently
    print("\n📡 Loading quantum field of candidates...")
    start = time.time()
    
    # Use projection to minimize data transfer
    candidates = list(db['public_profiles'].find(
        {'embedding': {'$exists': True}, 'slug': {'$exists': True, '$ne': user_slug}},
        {'slug': 1, 'embedding': 1, 'name': 1, 'blurb': 1}
    ).limit(limit))
    
    load_time = time.time() - start
    print(f"   {len(candidates)} profiles loaded in {load_time:.2f}s")
    
    # Prepare for quantum collapse with safer join
    candidate_pairs = []
    candidate_map = {}
    
    for c in candidates:
        if c.get('slug') and c.get('embedding'):
            candidate_pairs.append((c['slug'], np.array(c['embedding'])))
            candidate_map[c['slug']] = c
    
    # Collapse the wave function with precomputation
    print("\n🔮 Collapsing quantum wave function with precomputation...")
    start = time.time()
    
    results = discovery.discover_connections(
        query_embedding=entangled,
        candidate_embeddings=candidate_pairs,
        top_k=20,
        blend_factor=0.7,  # 70% quantum, 30% classical
        precompute=True    # Precompute all reductions for speed
    )
    
    compute_time = time.time() - start
    print(f"   Computed in {compute_time:.2f}s")
    
    print("\n" + "=" * 60)
    print(f"🎆 QUANTUM SEARCH RESULTS: \"{search_query}\"")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        # Safer lookup
        candidate = candidate_map.get(result['user_id'])
        if not candidate:
            continue
        
        # Enhanced interpretation
        quantum_score = result['quantum_score']
        novelty = result.get('novelty_score', 0)
        serendipity = result['serendipity_score']
        
        # Determine quantum state based on scores
        if novelty > 0.3:
            state = "🎯 QUANTUM DISCOVERY"
        elif quantum_score > 0.9 and serendipity > 0.5:
            state = "⚛️ QUANTUM ENTANGLED"
        elif quantum_score > 0.8:
            state = "🌟 SUPERPOSITION"
        elif serendipity > 0.3:
            state = "✨ QUANTUM TUNNELING"
        else:
            state = "🔬 OBSERVED"
        
        print(f"\n{i}. {candidate.get('name', 'Unknown')}")
        print(f"   {state}")
        print(f"   Quantum: {quantum_score:.3f} | Classical: {result['classical_score']:.3f}")
        print(f"   Novelty: {novelty:.3f} | Serendipity: {serendipity:.3f}")
        
        if candidate.get('blurb'):
            print(f"   Profile: {candidate['blurb'][:150]}...")
        
        print("   " + "-" * 50)
    
    # Enhanced statistics
    print("\n⚛️ QUANTUM STATISTICS")
    print("=" * 60)
    
    high_quantum = [r for r in results if r['quantum_score'] > 0.8]
    high_novelty = [r for r in results if r.get('novelty_score', 0) > 0.2]
    high_serendipity = [r for r in results if r['serendipity_score'] > 0.3]
    
    print(f"Quantum resonance (>0.8): {len(high_quantum)} profiles")
    print(f"Novel discoveries (>0.2): {len(high_novelty)} profiles")
    print(f"Serendipitous matches (>0.3): {len(high_serendipity)} profiles")
    
    if high_novelty:
        print(f"\n🎯 {len(high_novelty)} quantum discoveries with novelty > classical!")
        print("These connections emerge from quantum patterns invisible to classical matching.")
    
    # Performance metrics
    print(f"\n⚡ Performance:")
    print(f"   Data load: {load_time:.2f}s")
    print(f"   Quantum computation: {compute_time:.2f}s")
    print(f"   Total time: {load_time + compute_time:.2f}s")
    print(f"   Throughput: {len(candidate_pairs)/(compute_time+0.001):.0f} profiles/sec")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantum Search V3")
    parser.add_argument("--user", default="masseyl", help="User slug")
    parser.add_argument("--query", default="coding jazz cats", help="Search query")
    parser.add_argument("--limit", type=int, default=1000, help="Max candidates")
    
    args = parser.parse_args()
    
    quantum_search_v3(args.user, args.query, args.limit)
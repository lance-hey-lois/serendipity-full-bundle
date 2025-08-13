#!/usr/bin/env python3
"""
Pure Quantum Search - NO KEYWORDS!
Just embeddings, quantum mechanics, and spooky action at a distance
"""

import sys
sys.path.append('.')
from quantum.discovery_with_learned_v2 import LearnedQuantumDiscoveryV2
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os

load_dotenv('.env')

def quantum_search(user_slug: str, search_query: str):
    """
    Pure quantum search - no keywords, just embeddings and quantum mechanics
    
    Args:
        user_slug: The user making the search (e.g., 'masseyl')
        search_query: What they're looking for (e.g., 'coding jazz cats')
    """
    
    print("🌌 QUANTUM SEARCH ENGINE")
    print("=" * 60)
    print("No keywords. No classical matching. Just quantum mechanics.")
    print("-" * 60)
    
    # Initialize quantum system
    discovery = LearnedQuantumDiscoveryV2(
        n_qubits=8,
        reducer_path='learning/learned_reducer.pt'
    )
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
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
    
    # Generate query embedding
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=search_query
    )
    query_embedding = np.array(response.data[0].embedding)
    
    # Quantum entanglement: blend user and query embeddings
    # This creates a superposition of user interests and search intent
    alpha = 0.3  # User influence
    beta = 0.7   # Query influence
    
    entangled_embedding = alpha * np.array(user['embedding']) + beta * query_embedding
    entangled_embedding = entangled_embedding / np.linalg.norm(entangled_embedding)
    
    print(f"\n⚛️ Quantum entanglement created:")
    print(f"   User contribution: {alpha*100:.0f}%")
    print(f"   Query contribution: {beta*100:.0f}%")
    
    # Get ALL candidates with embeddings (no keyword filtering!)
    print("\n📡 Loading quantum field of candidates...")
    candidates = list(db['public_profiles'].find(
        {'embedding': {'$exists': True}, 'slug': {'$exists': True, '$ne': user_slug}},
        {'name': 1, 'slug': 1, 'embedding': 1, 'blurb': 1}
    ).limit(1000))  # Cast a wide quantum net
    
    print(f"   {len(candidates)} profiles in quantum superposition")
    
    # Prepare for quantum collapse
    candidate_pairs = [(c['slug'], np.array(c['embedding'])) for c in candidates if c.get('embedding')]
    
    # Collapse the wave function through measurement
    print("\n🔮 Collapsing quantum wave function...")
    results = discovery.discover_connections(
        query_embedding=entangled_embedding,
        candidate_embeddings=candidate_pairs,
        top_k=20,
        blend_factor=0.8  # Heavy quantum weight - we trust the spooky stuff
    )
    
    print("\n" + "=" * 60)
    print(f"🎆 QUANTUM SEARCH RESULTS: \"{search_query}\"")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        candidate = next(c for c in candidates if c['slug'] == result['user_id'])
        
        # Quantum interpretation
        quantum_score = result['quantum_score']
        serendipity = result['serendipity_score']
        
        # Determine quantum state
        if quantum_score > 0.9 and serendipity > 0.5:
            state = "⚛️ QUANTUM ENTANGLED"
        elif quantum_score > 0.8:
            state = "🌟 SUPERPOSITION"
        elif serendipity > 0.3:
            state = "✨ QUANTUM TUNNELING"
        else:
            state = "🔬 OBSERVED"
        
        print(f"\n{i}. {candidate.get('name', 'Unknown')}")
        print(f"   {state}")
        print(f"   Quantum Score: {quantum_score:.3f}")
        print(f"   Serendipity: {serendipity:.3f}")
        
        if candidate.get('blurb'):
            # Show blurb but NO keyword highlighting - we don't care about keywords!
            print(f"   Profile: {candidate['blurb'][:150]}...")
        
        print("   " + "-" * 50)
    
    # Quantum statistics
    print("\n⚛️ QUANTUM STATISTICS")
    print("=" * 60)
    
    high_quantum = [r for r in results if r['quantum_score'] > 0.8]
    high_serendipity = [r for r in results if r['serendipity_score'] > 0.3]
    
    print(f"Quantum entangled profiles (>0.8): {len(high_quantum)}")
    print(f"Serendipitous discoveries (>0.3): {len(high_serendipity)}")
    
    if high_serendipity:
        print(f"\n✨ The quantum field suggests {len(high_serendipity)} unexpected connections!")
        print("These emerge from quantum patterns, not classical similarity.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pure quantum search - no keywords!")
    parser.add_argument("--user", default="masseyl", help="User slug")
    parser.add_argument("--query", default="coding jazz cats", help="Search query")
    
    args = parser.parse_args()
    
    quantum_search(args.user, args.query)
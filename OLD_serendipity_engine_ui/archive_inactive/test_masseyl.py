#!/usr/bin/env python3
"""
Test quantum discovery for specific user: masseyl
"""

import sys
sys.path.append('.')
from quantum.discovery_with_learned_v2 import LearnedQuantumDiscoveryV2
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import os

# Load environment
load_dotenv('.env')

def find_connections_for_masseyl():
    """Find serendipitous connections for masseyl"""
    
    # Initialize quantum discovery
    print("🚀 Initializing Quantum Discovery System...")
    discovery = LearnedQuantumDiscoveryV2(
        n_qubits=8,
        reducer_path='learning/learned_reducer.pt'
    )
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Find masseyl's profile
    print("\n🔍 Looking for user 'masseyl'...")
    masseyl = db['public_profiles'].find_one(
        {'slug': 'masseyl'},
        {'name': 1, 'slug': 1, 'embedding': 1, 'bio': 1, 'skills': 1, 'location': 1}
    )
    
    if not masseyl:
        print("❌ User 'masseyl' not found!")
        return
    
    if not masseyl.get('embedding'):
        print("❌ User 'masseyl' has no embedding!")
        return
    
    print(f"✅ Found: {masseyl.get('name', 'masseyl')}")
    if masseyl.get('bio'):
        print(f"   Bio: {masseyl['bio'][:200]}...")
    if masseyl.get('skills'):
        print(f"   Skills: {', '.join(masseyl['skills'][:10])}")
    if masseyl.get('location'):
        print(f"   Location: {masseyl['location']}")
    
    # Get all other users with embeddings
    print("\n📊 Loading candidate profiles...")
    candidates = list(db['public_profiles'].find(
        {'embedding': {'$exists': True}, 'slug': {'$exists': True, '$ne': 'masseyl'}},
        {'name': 1, 'slug': 1, 'embedding': 1, 'bio': 1, 'skills': 1, 'location': 1}
    ).limit(500))  # Analyze top 500 candidates
    
    print(f"   Found {len(candidates)} candidates with embeddings")
    
    # Prepare candidate list - skip if no slug
    candidate_pairs = []
    for c in candidates:
        if c.get('slug') and c.get('embedding'):
            candidate_pairs.append((c['slug'], np.array(c['embedding'])))
    
    # Discover connections
    print("\n🔮 Running quantum discovery algorithm...")
    results = discovery.discover_connections(
        query_embedding=np.array(masseyl['embedding']),
        candidate_embeddings=candidate_pairs,
        top_k=20,
        blend_factor=0.7  # 70% quantum, 30% classical
    )
    
    print("\n" + "=" * 80)
    print("🌟 TOP SERENDIPITOUS CONNECTIONS FOR MASSEYL")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        # Find the full profile
        candidate = next(c for c in candidates if c['slug'] == result['user_id'])
        
        print(f"\n{i}. {candidate.get('name', 'Unknown')} (@{candidate['slug']})")
        print(f"   📍 {candidate.get('location', 'Location unknown')}")
        
        # Scores
        print(f"\n   📊 Scores:")
        print(f"      Quantum:      {result['quantum_score']:.3f} {'⭐' * int(result['quantum_score'] * 5)}")
        print(f"      Classical:    {result['classical_score']:.3f} {'⭐' * int(result['classical_score'] * 5)}")
        print(f"      Blended:      {result['blended_score']:.3f} {'⭐' * int(result['blended_score'] * 5)}")
        print(f"      Serendipity:  {result['serendipity_score']:.3f} {'✨' * int(result['serendipity_score'] * 5)}")
        
        # Explanation
        print(f"\n   💡 {discovery.explain_connection(result)}")
        
        # Profile details
        if candidate.get('bio'):
            print(f"\n   📝 Bio: {candidate['bio'][:150]}...")
        
        if candidate.get('skills'):
            skills_preview = ', '.join(candidate['skills'][:7])
            if len(candidate['skills']) > 7:
                skills_preview += f" (+{len(candidate['skills'])-7} more)"
            print(f"   🛠️  Skills: {skills_preview}")
        
        print("\n   " + "-" * 70)
    
    # Show statistics
    print("\n📈 DISCOVERY STATISTICS")
    print("=" * 80)
    
    quantum_scores = [r['quantum_score'] for r in results]
    classical_scores = [r['classical_score'] for r in results]
    serendipity_scores = [r['serendipity_score'] for r in results]
    
    print(f"Quantum scores:     Min={min(quantum_scores):.3f}, Max={max(quantum_scores):.3f}, Avg={np.mean(quantum_scores):.3f}")
    print(f"Classical scores:   Min={min(classical_scores):.3f}, Max={max(classical_scores):.3f}, Avg={np.mean(classical_scores):.3f}")
    print(f"Serendipity scores: Min={min(serendipity_scores):.3f}, Max={max(serendipity_scores):.3f}, Avg={np.mean(serendipity_scores):.3f}")
    
    # Identify truly serendipitous connections
    serendipitous = [r for r in results if r['serendipity_score'] > 0.3]
    if serendipitous:
        print(f"\n✨ Found {len(serendipitous)} highly serendipitous connections!")
        print("These are people with hidden compatibility despite surface differences.")

if __name__ == "__main__":
    find_connections_for_masseyl()
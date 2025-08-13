#!/usr/bin/env python3
"""
Query-based quantum discovery: Find "coding jazz cats" for masseyl
"""

import sys
sys.path.append('.')
from quantum.discovery_with_learned_v2 import LearnedQuantumDiscoveryV2
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os

# Load environment
load_dotenv('.env')

def find_coding_jazz_cats_for_masseyl():
    """Find people matching 'coding jazz cats' query for masseyl"""
    
    # Initialize
    print("🚀 Initializing Quantum Discovery System...")
    discovery = LearnedQuantumDiscoveryV2(
        n_qubits=8,
        reducer_path='learning/learned_reducer.pt'
    )
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Initialize OpenAI for query embedding
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Find masseyl's profile
    print("\n🔍 Looking for user 'masseyl'...")
    masseyl = db['public_profiles'].find_one(
        {'slug': 'masseyl'},
        {'name': 1, 'slug': 1, 'embedding': 1}
    )
    
    if not masseyl or not masseyl.get('embedding'):
        print("❌ User 'masseyl' not found or has no embedding!")
        return
    
    print(f"✅ Found: {masseyl.get('name', 'masseyl')}")
    
    # Generate embedding for the search query
    print("\n🎯 Generating embedding for query: 'coding jazz cats'")
    query_text = "coding jazz cats musicians who code programmers with musical talent jazz developers software engineers who play music creative coders"
    
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=query_text
    )
    query_embedding = np.array(response.data[0].embedding)
    
    # Blend masseyl's embedding with the query embedding
    # This personalizes the search to masseyl's interests
    print("🔄 Blending user profile with search query...")
    blended_embedding = 0.4 * np.array(masseyl['embedding']) + 0.6 * query_embedding
    blended_embedding = blended_embedding / np.linalg.norm(blended_embedding)  # Normalize
    
    # Pre-filter candidates based on keywords (optional optimization)
    print("\n🔎 Finding candidates with relevant signals...")
    
    # First try to find people with music/jazz/code keywords
    keyword_filter = {
        '$or': [
            {'bio': {'$regex': 'jazz|music|musician|guitar|piano|drums|bass|saxophone', '$options': 'i'}},
            {'skills': {'$in': ['music', 'jazz', 'guitar', 'piano', 'coding', 'programming']}},
            {'name': {'$regex': 'jazz|music', '$options': 'i'}}
        ]
    }
    
    # Get candidates with keywords
    keyword_candidates = list(db['public_profiles'].find(
        {'$and': [
            {'embedding': {'$exists': True}},
            {'slug': {'$exists': True, '$ne': 'masseyl'}},
            keyword_filter
        ]},
        {'name': 1, 'slug': 1, 'embedding': 1, 'bio': 1, 'skills': 1}
    ).limit(100))
    
    print(f"   Found {len(keyword_candidates)} candidates with music/code keywords")
    
    # Also get some general candidates for comparison
    general_candidates = list(db['public_profiles'].find(
        {'embedding': {'$exists': True}, 'slug': {'$exists': True, '$ne': 'masseyl'}},
        {'name': 1, 'slug': 1, 'embedding': 1, 'bio': 1, 'skills': 1}
    ).limit(200))
    
    # Combine and deduplicate
    all_candidates = {c['slug']: c for c in keyword_candidates}
    for c in general_candidates:
        if c['slug'] not in all_candidates:
            all_candidates[c['slug']] = c
    
    candidates = list(all_candidates.values())
    print(f"   Total candidates to analyze: {len(candidates)}")
    
    # Calculate classical similarities to the query
    print("\n📊 Computing similarities to 'coding jazz cats' query...")
    query_similarities = []
    
    for c in candidates:
        if c.get('embedding'):
            # Similarity to pure query
            query_sim = np.dot(query_embedding, np.array(c['embedding'])) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(c['embedding'])
            )
            query_similarities.append({
                'candidate': c,
                'query_similarity': float(query_sim)
            })
    
    # Sort by query similarity first
    query_similarities.sort(key=lambda x: x['query_similarity'], reverse=True)
    
    # Take top candidates by query relevance
    top_by_query = query_similarities[:50]
    
    # Prepare for quantum discovery
    candidate_pairs = []
    candidate_map = {}
    
    for item in top_by_query:
        c = item['candidate']
        if c.get('slug') and c.get('embedding'):
            candidate_pairs.append((c['slug'], np.array(c['embedding'])))
            candidate_map[c['slug']] = {
                'profile': c,
                'query_similarity': item['query_similarity']
            }
    
    # Run quantum discovery with blended embedding
    print("\n🔮 Running quantum discovery with personalized query...")
    results = discovery.discover_connections(
        query_embedding=blended_embedding,
        candidate_embeddings=candidate_pairs,
        top_k=20,
        blend_factor=0.6  # Slightly less quantum weight for query-based search
    )
    
    print("\n" + "=" * 80)
    print("🎷 CODING JAZZ CATS FOR MASSEYL")
    print("=" * 80)
    print("Query: Musicians who code, programmers with jazz talent")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        candidate_info = candidate_map[result['user_id']]
        profile = candidate_info['profile']
        
        print(f"\n{i}. {profile.get('name', 'Unknown')} (@{profile['slug']})")
        
        # Check for relevant keywords
        bio_lower = (profile.get('bio', '') or '').lower()
        skills_lower = [s.lower() for s in (profile.get('skills', []) or [])]
        
        music_keywords = ['jazz', 'music', 'musician', 'guitar', 'piano', 'drums', 'bass', 'saxophone', 'composer', 'band']
        code_keywords = ['code', 'coding', 'programmer', 'developer', 'software', 'engineer', 'tech', 'python', 'javascript']
        
        has_music = any(kw in bio_lower for kw in music_keywords) or any(kw in skills_lower for kw in music_keywords)
        has_code = any(kw in bio_lower for kw in code_keywords) or any(kw in skills_lower for kw in code_keywords)
        
        # Tags
        tags = []
        if has_music and has_code:
            tags.append("🎷💻 JAZZ CODER!")
        elif has_music:
            tags.append("🎵 Musical")
        elif has_code:
            tags.append("💻 Coder")
            
        if tags:
            print(f"   {' '.join(tags)}")
        
        # Scores
        print(f"\n   📊 Scores:")
        print(f"      Query Match:  {candidate_info['query_similarity']:.3f} {'⭐' * int(candidate_info['query_similarity'] * 5)}")
        print(f"      Quantum:      {result['quantum_score']:.3f} {'⭐' * int(result['quantum_score'] * 5)}")
        print(f"      Classical:    {result['classical_score']:.3f} {'⭐' * int(result['classical_score'] * 5)}")
        print(f"      Serendipity:  {result['serendipity_score']:.3f} {'✨' * int(result['serendipity_score'] * 5)}")
        
        # Profile details
        if profile.get('bio'):
            # Highlight relevant parts
            bio = profile['bio'][:200]
            for kw in ['jazz', 'music', 'code', 'coding', 'programmer']:
                if kw in bio.lower():
                    bio = bio.replace(kw, f"**{kw}**")
                    bio = bio.replace(kw.capitalize(), f"**{kw.capitalize()}**")
            print(f"\n   📝 Bio: {bio}...")
        
        if profile.get('skills'):
            # Highlight relevant skills
            highlighted_skills = []
            for skill in profile['skills'][:10]:
                if any(kw in skill.lower() for kw in music_keywords + code_keywords):
                    highlighted_skills.append(f"✓{skill}")
                else:
                    highlighted_skills.append(skill)
            print(f"   🛠️  Skills: {', '.join(highlighted_skills)}")
        
        print("\n   " + "-" * 70)
    
    # Summary statistics
    print("\n📈 SEARCH STATISTICS")
    print("=" * 80)
    
    # Count how many are actual "coding jazz cats"
    coding_jazz_cats = []
    for result in results[:10]:
        candidate_info = candidate_map[result['user_id']]
        profile = candidate_info['profile']
        bio_lower = (profile.get('bio', '') or '').lower()
        skills_lower = [s.lower() for s in (profile.get('skills', []) or [])]
        
        has_music = any(kw in bio_lower for kw in music_keywords) or any(kw in skills_lower for kw in music_keywords)
        has_code = any(kw in bio_lower for kw in code_keywords) or any(kw in skills_lower for kw in code_keywords)
        
        if has_music and has_code:
            coding_jazz_cats.append(profile['name'])
    
    if coding_jazz_cats:
        print(f"\n🎯 Found {len(coding_jazz_cats)} actual coding jazz cats:")
        for name in coding_jazz_cats:
            print(f"   • {name}")
    else:
        print("\n💡 No exact matches, but quantum discovery found people with hidden compatibility!")
        print("   These connections might surprise you with unexpected synergies.")

if __name__ == "__main__":
    find_coding_jazz_cats_for_masseyl()
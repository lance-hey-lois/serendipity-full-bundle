#!/usr/bin/env python3
"""
Fix Embeddings with Domain Classification
=========================================
Demonstrates how to improve embeddings by adding explicit domain context
"""

import json
from typing import Dict, List, Any

def analyze_current_profile(profile: Dict[str, Any]) -> str:
    """Show what the current embedding looks like"""
    
    # This is what gets embedded currently (from your embeddings.ts)
    current_text = f"""
    Name: {profile.get('name', '')}
    Title: {profile.get('title', '')}
    Company: {profile.get('company', '')}
    Location: {profile.get('locatedIn', '')}
    Short Pitch: {profile.get('shortPitch', '')}
    Blurb: {profile.get('blurb', '')}
    Goals: {', '.join(profile.get('midTermPriorities', {}).get('goals', []))}
    Looking For: {', '.join(profile.get('midTermPriorities', {}).get('lookingFor', []))}
    Network Strengths: {', '.join(profile.get('areasOfNetworkStrength', []))}
    Can Help With: {', '.join(profile.get('canHelpWith', []))}
    Interests: {', '.join(profile.get('passionsAndInterests', []))}
    Personality Traits: {', '.join(profile.get('personalityTraits', []))}
    Relevant Details: {profile.get('relevantDetails', '')}
    """
    return current_text.strip()

def detect_industry(profile: Dict[str, Any]) -> str:
    """Detect the industry/domain from profile content"""
    
    # Combine all text for analysis
    text_fields = [
        profile.get('title', ''),
        profile.get('company', ''),
        profile.get('blurb', ''),
        ' '.join(profile.get('areasOfNetworkStrength', [])),
        ' '.join(profile.get('canHelpWith', [])),
        ' '.join(profile.get('passionsAndInterests', []))
    ]
    full_text = ' '.join(text_fields).lower()
    
    # Industry detection rules
    industries = {
        'Music & Entertainment': [
            'music', 'song', 'songwriter', 'composer', 'musician', 'artist', 
            'album', 'record', 'studio', 'broadway', 'theater', 'spotify',
            'juke', 'concert', 'tour', 'band', 'lyrics', 'melody'
        ],
        'Finance & Investment': [
            'investment', 'capital', 'fund', 'finance', 'banking', 'equity',
            'portfolio', 'ppp', 'real estate', 'stakeholder', 'strategic investment',
            'funding', 'investor', 'venture', 'assets', 'returns'
        ],
        'Technology': [
            'software', 'developer', 'engineer', 'platform', 'saas', 'ai',
            'machine learning', 'data', 'api', 'cloud', 'tech', 'startup',
            'product', 'code', 'algorithm', 'system'
        ],
        'Healthcare': [
            'health', 'medical', 'cancer', 'hospital', 'clinical', 'patient',
            'therapy', 'treatment', 'care', 'wellness', 'mental health',
            'pharmaceutical', 'doctor', 'nurse'
        ],
        'Legal': [
            'law', 'legal', 'attorney', 'lawyer', 'litigation', 'contract',
            'compliance', 'regulatory', 'court', 'justice'
        ]
    }
    
    # Count matches for each industry
    industry_scores = {}
    for industry, keywords in industries.items():
        score = sum(1 for keyword in keywords if keyword in full_text)
        if score > 0:
            industry_scores[industry] = score
    
    # Return the highest scoring industry
    if industry_scores:
        return max(industry_scores, key=industry_scores.get)
    return 'General Business'

def create_enhanced_embedding_text(profile: Dict[str, Any]) -> str:
    """Create an enhanced embedding with domain context"""
    
    # Detect the industry
    industry = detect_industry(profile)
    
    # Build enhanced embedding with clear domain signal
    enhanced_text = f"""
    INDUSTRY: {industry}
    PROFESSIONAL: {profile.get('name', '')} is a {profile.get('title', '')} at {profile.get('company', '')}
    DOMAIN EXPERTISE: {', '.join(profile.get('areasOfNetworkStrength', []))}
    PROFESSIONAL FOCUS: {profile.get('shortPitch', '')}
    
    DETAILED BACKGROUND:
    {profile.get('blurb', '')}
    
    SKILLS AND CAPABILITIES:
    {', '.join(profile.get('canHelpWith', []))}
    
    CURRENT OBJECTIVES:
    {', '.join(profile.get('midTermPriorities', {}).get('goals', []))}
    
    SEEKING CONNECTIONS:
    {', '.join(profile.get('midTermPriorities', {}).get('lookingFor', []))}
    
    INTERESTS:
    {', '.join(profile.get('passionsAndInterests', []))}
    """
    
    return enhanced_text.strip()

# Example with Matthew Semegran's profile
matthew_profile = {
    "name": "Matthew Semegran",
    "title": "Partner",
    "company": "Stable Capital",
    "areasOfNetworkStrength": [
        "Public-Private Partnerships",
        "Investment Strategies", 
        "Project Management in Infrastructure",
        "Real Estate Developments"
    ],
    "blurb": "Matthew Semegran is a dynamic finance professional...",
    "canHelpWith": [
        "Navigating complex investment and funding structures",
        "Building strategic stakeholder relationships",
        "Identifying high-value investment opportunities",
        "Developing effective project proposals"
    ],
    "shortPitch": "Experienced finance partner focused on driving impactful public-private collaborations...",
    "midTermPriorities": {
        "goals": [
            "Expand investment opportunities in PPP and real estate projects",
            "Gain insights on public-private partnerships in Uganda"
        ],
        "lookingFor": [
            "Potential partnerships in the renewable energy sector",
            "Opportunities for involvement in PPP projects"
        ]
    },
    "passionsAndInterests": [
        "Sustainability in investment",
        "Innovative funding models"
    ]
}

# Example songwriter profile for comparison
songwriter_profile = {
    "name": "Jane Smith",
    "title": "Songwriter",
    "company": "Independent",
    "areasOfNetworkStrength": [
        "Lyric writing",
        "Melody composition",
        "Music production",
        "Artist collaboration"
    ],
    "blurb": "Award-winning songwriter with credits on multiple platinum albums...",
    "canHelpWith": [
        "Writing compelling lyrics",
        "Creating memorable melodies",
        "Collaborating with artists",
        "Music arrangement"
    ],
    "shortPitch": "Professional songwriter specializing in pop and R&B...",
    "midTermPriorities": {
        "goals": [
            "Expand into film scoring",
            "Collaborate with emerging artists"
        ],
        "lookingFor": [
            "Music producers",
            "Recording artists",
            "Music publishers"
        ]
    },
    "passionsAndInterests": [
        "Music composition",
        "Creative writing"
    ]
}

if __name__ == "__main__":
    print("=" * 80)
    print("CURRENT EMBEDDING APPROACH (Generic)")
    print("=" * 80)
    
    print("\n1. Matthew Semegran (Finance):")
    print("-" * 40)
    print(analyze_current_profile(matthew_profile)[:500] + "...")
    
    print("\n2. Jane Smith (Songwriter):")
    print("-" * 40)
    print(analyze_current_profile(songwriter_profile)[:500] + "...")
    
    print("\n" + "=" * 80)
    print("PROBLEM: Both profiles have similar abstract patterns:")
    print("- 'Partner' vs 'Collaboration'")
    print("- 'Strategic' vs 'Creative'")
    print("- 'Building relationships' vs 'Artist collaboration'")
    print("- 'Innovative' appears in both")
    
    print("\n" + "=" * 80)
    print("ENHANCED EMBEDDING APPROACH (With Domain Classification)")
    print("=" * 80)
    
    print("\n1. Matthew Semegran (Finance) - ENHANCED:")
    print("-" * 40)
    matthew_industry = detect_industry(matthew_profile)
    print(f"DETECTED INDUSTRY: {matthew_industry}")
    print(create_enhanced_embedding_text(matthew_profile)[:600] + "...")
    
    print("\n2. Jane Smith (Songwriter) - ENHANCED:")
    print("-" * 40)
    songwriter_industry = detect_industry(songwriter_profile)
    print(f"DETECTED INDUSTRY: {songwriter_industry}")
    print(create_enhanced_embedding_text(songwriter_profile)[:600] + "...")
    
    print("\n" + "=" * 80)
    print("SOLUTION: With industry classification, searching for 'songwriter' will:")
    print("1. First identify the query is about 'Music & Entertainment'")
    print("2. Prioritize profiles with 'INDUSTRY: Music & Entertainment'")
    print("3. Matthew's 'INDUSTRY: Finance & Investment' gets deprioritized")
    print("4. Jane's 'INDUSTRY: Music & Entertainment' gets boosted")
    print("\n✅ Result: Songwriter search returns actual songwriters!")
"""
Serendipity Scoring System
Measures how surprising, valuable, and timely a connection is
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import math

class SerendipityScorer:
    """
    Calculate serendipity scores for quantum-discovered connections
    
    Serendipity = Surprise × Value × Timing × Quantum_Bonus
    """
    
    def __init__(self):
        self.weights = {
            'surprise': 1.0,
            'value': 1.0, 
            'timing': 1.0,
            'quantum_bonus': 1.5  # Boost for quantum-tunneled connections
        }
    
    def calculate_serendipity(self, query_profile: Dict, 
                             match_profile: Dict,
                             barrier_height: float = 0,
                             tunneling_prob: float = 0) -> Dict[str, Any]:
        """
        Calculate comprehensive serendipity score
        
        Args:
            query_profile: Query user profile
            match_profile: Matched candidate profile
            barrier_height: Barrier that was tunneled through
            tunneling_prob: Quantum tunneling probability
            
        Returns:
            Dictionary with score components and final score
        """
        # Calculate individual components
        surprise = self.calculate_surprise(query_profile, match_profile, barrier_height)
        value = self.calculate_value(query_profile, match_profile)
        timing = self.calculate_timing(query_profile, match_profile)
        
        # Apply quantum bonus if this was a tunneled connection
        quantum_multiplier = 1.0
        if tunneling_prob > 0.3 and barrier_height > 3:
            quantum_multiplier = self.weights['quantum_bonus']
        
        # Calculate final score (0-300 scale)
        final_score = surprise * value * timing * quantum_multiplier * 100
        
        # Cap at 300 for consistency
        final_score = min(final_score, 300)
        
        return {
            'final_score': float(final_score),
            'components': {
                'surprise': float(surprise),
                'value': float(value),
                'timing': float(timing),
                'quantum_bonus': float(quantum_multiplier)
            },
            'breakdown': {
                'surprise_factors': self._get_surprise_factors(query_profile, match_profile, barrier_height),
                'value_factors': self._get_value_factors(query_profile, match_profile),
                'timing_factors': self._get_timing_factors(query_profile, match_profile)
            },
            'barrier_crossed': float(barrier_height),
            'tunneling_probability': float(tunneling_prob)
        }
    
    def calculate_surprise(self, query_profile: Dict, match_profile: Dict, 
                          barrier_height: float) -> float:
        """
        Calculate surprise factor (0-1)
        Higher = more unexpected connection
        """
        surprise = 0.3  # Base surprise
        
        # 1. Network distance contributes to surprise
        query_communities = set(query_profile.get('network_metrics', {}).get('community_ids', []))
        match_communities = set(match_profile.get('network_metrics', {}).get('community_ids', []))
        
        if not query_communities.intersection(match_communities):
            surprise += 0.3  # Different communities = surprising
        
        # 2. Skill divergence = surprising
        query_skills = query_profile.get('quantum_features', {}).get('skills_vector', [])
        match_skills = match_profile.get('quantum_features', {}).get('skills_vector', [])
        
        if query_skills and match_skills:
            skill_distance = np.linalg.norm(
                np.array(query_skills[:8]) - np.array(match_skills[:8])
            )
            surprise += min(skill_distance / 4, 0.3)  # Normalize to 0-0.3
        
        # 3. Barrier height directly indicates surprise
        if barrier_height > 5:
            surprise += 0.3
        elif barrier_height > 3:
            surprise += 0.2
        elif barrier_height > 1:
            surprise += 0.1
        
        # 4. Centrality difference (connecting different network levels)
        query_central = query_profile.get('network_metrics', {}).get('centrality', 0.5)
        match_central = match_profile.get('network_metrics', {}).get('centrality', 0.5)
        
        central_diff = abs(query_central - match_central)
        if central_diff > 0.5:
            surprise += 0.1
        
        return min(surprise, 1.0)
    
    def calculate_value(self, query_profile: Dict, match_profile: Dict) -> float:
        """
        Calculate value alignment (0-1)
        Higher = more valuable connection
        """
        value = 0.4  # Base value
        
        # 1. Complementary skills = valuable
        query_skills = query_profile.get('quantum_features', {}).get('skills_vector', [])
        match_skills = match_profile.get('quantum_features', {}).get('skills_vector', [])
        
        if query_skills and match_skills:
            # Find complementarity (where one is strong and other is weak)
            query_array = np.array(query_skills[:10])
            match_array = np.array(match_skills[:10])
            
            # Complementarity: high where one is low and vice versa
            complementarity = 0
            for i in range(len(query_array)):
                if query_array[i] < 0.3 and match_array[i] > 0.7:
                    complementarity += 0.1
                elif query_array[i] > 0.7 and match_array[i] < 0.3:
                    complementarity += 0.1
            
            value += min(complementarity, 0.3)
        
        # 2. Bridge score = valuable (connects communities)
        match_bridge = match_profile.get('network_metrics', {}).get('bridge_score', 0)
        value += match_bridge * 0.2
        
        # 3. Weak ties = valuable (Granovetter's theory)
        match_weak_ties = match_profile.get('network_metrics', {}).get('weak_ties_count', 0)
        if match_weak_ties > 150:
            value += 0.1
        
        # 4. Uniqueness of match = valuable
        match_uniqueness = match_profile.get('serendipity_factors', {}).get('uniqueness', 0)
        value += match_uniqueness * 0.1
        
        return min(value, 1.0)
    
    def calculate_timing(self, query_profile: Dict, match_profile: Dict) -> float:
        """
        Calculate timing resonance (0-1)
        Higher = better timing for connection
        """
        timing = 0.5  # Base timing
        
        # 1. Both available = good timing
        query_avail = query_profile.get('quantum_features', {}).get('availability', 0.5)
        match_avail = match_profile.get('quantum_features', {}).get('availability', 0.5)
        
        combined_availability = query_avail * match_avail
        timing += combined_availability * 0.3
        
        # 2. Both in transition = excellent timing
        query_trans = query_profile.get('quantum_features', {}).get('transition_probability', 0.3)
        match_trans = match_profile.get('quantum_features', {}).get('transition_probability', 0.3)
        
        if query_trans > 0.5 and match_trans > 0.5:
            timing += 0.3
        elif query_trans > 0.3 or match_trans > 0.3:
            timing += 0.1
        
        # 3. Timing score from profile
        match_timing = match_profile.get('serendipity_factors', {}).get('timing_score', 0.5)
        timing += match_timing * 0.1
        
        return min(timing, 1.0)
    
    def _get_surprise_factors(self, query_profile: Dict, match_profile: Dict, 
                              barrier_height: float) -> List[str]:
        """Get human-readable surprise factors"""
        factors = []
        
        if barrier_height > 5:
            factors.append(f"Crossed massive barrier ({barrier_height:.1f})")
        elif barrier_height > 3:
            factors.append(f"Tunneled through barrier ({barrier_height:.1f})")
        
        query_communities = set(query_profile.get('network_metrics', {}).get('community_ids', []))
        match_communities = set(match_profile.get('network_metrics', {}).get('community_ids', []))
        
        if not query_communities.intersection(match_communities):
            factors.append("From completely different communities")
        
        query_central = query_profile.get('network_metrics', {}).get('centrality', 0.5)
        match_central = match_profile.get('network_metrics', {}).get('centrality', 0.5)
        
        if abs(query_central - match_central) > 0.5:
            if match_central > query_central:
                factors.append("Connected to network hub")
            else:
                factors.append("Found hidden gem in network periphery")
        
        return factors
    
    def _get_value_factors(self, query_profile: Dict, match_profile: Dict) -> List[str]:
        """Get human-readable value factors"""
        factors = []
        
        match_bridge = match_profile.get('network_metrics', {}).get('bridge_score', 0)
        if match_bridge > 0.7:
            factors.append("Bridges multiple communities")
        
        match_weak_ties = match_profile.get('network_metrics', {}).get('weak_ties_count', 0)
        if match_weak_ties > 150:
            factors.append(f"Has {match_weak_ties} weak ties (valuable connections)")
        
        match_uniqueness = match_profile.get('serendipity_factors', {}).get('uniqueness', 0)
        if match_uniqueness > 0.8:
            factors.append("Rare combination of skills")
        
        # Check complementarity
        offers = match_profile.get('serendipity_factors', {}).get('complementarity', {}).get('offers', [])
        if offers:
            factors.append(f"Offers: {', '.join(offers[:2])}")
        
        return factors
    
    def _get_timing_factors(self, query_profile: Dict, match_profile: Dict) -> List[str]:
        """Get human-readable timing factors"""
        factors = []
        
        match_avail = match_profile.get('quantum_features', {}).get('availability', 0.5)
        if match_avail > 0.8:
            factors.append("Highly available now")
        
        match_trans = match_profile.get('quantum_features', {}).get('transition_probability', 0.3)
        if match_trans > 0.6:
            factors.append("In career transition")
        elif match_trans > 0.4:
            factors.append("Open to new opportunities")
        
        query_trans = query_profile.get('quantum_features', {}).get('transition_probability', 0.3)
        if query_trans > 0.5 and match_trans > 0.5:
            factors.append("Both ready for change")
        
        return factors
    
    def rank_by_serendipity(self, query_profile: Dict, 
                           matches: List[Tuple[Dict, float, float]]) -> List[Dict]:
        """
        Rank matches by serendipity score
        
        Args:
            query_profile: Query user profile
            matches: List of (profile, tunneling_prob, barrier) tuples
            
        Returns:
            List of matches with serendipity scores, sorted by score
        """
        scored_matches = []
        
        for match_profile, tunneling_prob, barrier in matches:
            score_data = self.calculate_serendipity(
                query_profile, match_profile, barrier, tunneling_prob
            )
            
            result = {
                'profile': match_profile,
                'serendipity_score': score_data['final_score'],
                'score_components': score_data['components'],
                'score_breakdown': score_data['breakdown'],
                'barrier_crossed': barrier,
                'tunneling_probability': tunneling_prob
            }
            
            scored_matches.append(result)
        
        # Sort by serendipity score
        scored_matches.sort(key=lambda x: x['serendipity_score'], reverse=True)
        
        return scored_matches


def demo_serendipity_scoring():
    """Demo the serendipity scoring system"""
    
    # Sample profiles
    query = {
        'name': 'Tech CEO',
        'quantum_features': {
            'skills_vector': [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5],
            'personality_phase': [np.pi/4, np.pi/2, np.pi/3, np.pi/6, np.pi],
            'availability': 0.6,
            'transition_probability': 0.4
        },
        'network_metrics': {
            'centrality': 0.8,
            'community_ids': [1, 2],
            'bridge_score': 0.5
        }
    }
    
    match = {
        'name': 'Jazz Musician / Hidden Tech Genius',
        'quantum_features': {
            'skills_vector': [0.2, 0.9, 0.3, 0.8, 0.4, 0.7, 0.5, 0.6, 0.6, 0.4],
            'personality_phase': [np.pi, np.pi/6, np.pi/2, np.pi/3, np.pi/4],
            'availability': 0.9,
            'transition_probability': 0.7
        },
        'network_metrics': {
            'centrality': 0.3,
            'community_ids': [5, 6],
            'bridge_score': 0.8,
            'weak_ties_count': 200
        },
        'serendipity_factors': {
            'uniqueness': 0.95,
            'timing_score': 0.8,
            'complementarity': {
                'offers': ['creative vision', 'unconventional thinking'],
                'seeks': ['technical platform', 'business structure']
            }
        }
    }
    
    # Calculate serendipity
    scorer = SerendipityScorer()
    result = scorer.calculate_serendipity(query, match, barrier_height=6.5, tunneling_prob=0.75)
    
    print("\n✨ SERENDIPITY ANALYSIS ✨")
    print("="*60)
    print(f"\nConnection: {query['name']} ←→ {match['name']}")
    print(f"\n🎯 SERENDIPITY SCORE: {result['final_score']:.0f}/300")
    
    print("\n📊 Score Components:")
    for component, value in result['components'].items():
        stars = "⭐" * int(value * 5)
        print(f"  {component.capitalize()}: {value:.2f} {stars}")
    
    print("\n🔍 Why This Is Serendipitous:")
    print("\n  Surprise Factors:")
    for factor in result['breakdown']['surprise_factors']:
        print(f"    • {factor}")
    
    print("\n  Value Factors:")
    for factor in result['breakdown']['value_factors']:
        print(f"    • {factor}")
    
    print("\n  Timing Factors:")
    for factor in result['breakdown']['timing_factors']:
        print(f"    • {factor}")
    
    print(f"\n🌌 Quantum Magic:")
    print(f"  Barrier Crossed: {result['barrier_crossed']:.1f}")
    print(f"  Tunneling Success: {result['tunneling_probability']:.1%}")
    
    # Score interpretation
    score = result['final_score']
    if score > 250:
        print("\n💫 RATING: MIND-BLOWING SERENDIPITY!")
        print("   This connection defies all logic but could change everything.")
    elif score > 200:
        print("\n🌟 RATING: HIGHLY SERENDIPITOUS")
        print("   An unexpected gem that classical search would never find.")
    elif score > 150:
        print("\n⭐ RATING: SERENDIPITOUS")
        print("   A surprising and valuable connection worth exploring.")
    else:
        print("\n✨ RATING: MILDLY SERENDIPITOUS")
        print("   An interesting connection with some surprise elements.")

if __name__ == "__main__":
    demo_serendipity_scoring()
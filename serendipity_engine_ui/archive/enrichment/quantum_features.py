"""
Quantum Feature Enrichment Pipeline
Generates quantum-compatible features from profile data for serendipitous discovery
"""

import numpy as np
from typing import Dict, List, Any, Optional
import hashlib
import re
from openai import OpenAI
import os
import json

class QuantumFeatureGenerator:
    """Generate quantum features from profile data"""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.openai_client = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Skill categories for vector encoding
        self.skill_categories = [
            "technical", "creative", "leadership", "communication", "analytical",
            "strategic", "operational", "financial", "sales", "research"
        ]
        
        # Personality dimensions (simplified Big Five)
        self.personality_dims = [
            "openness", "conscientiousness", "extraversion", 
            "agreeableness", "neuroticism"
        ]
    
    def generate_quantum_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate all quantum features for a profile
        
        Args:
            profile: MongoDB profile document
            
        Returns:
            Dictionary of quantum features to add to profile
        """
        # Extract text content
        bio = profile.get('blurb', '') or ''
        title = profile.get('title', '') or ''
        skills = profile.get('areasOfNetworkStrength', []) or []
        company = profile.get('company', '') or ''
        
        combined_text = f"{title} at {company}. {bio}. Skills: {', '.join(skills[:10])}"
        
        # Generate feature vectors
        skills_vector = self._extract_skills_vector(combined_text, skills)
        personality_phase = self._generate_personality_phase(bio)
        availability = self._calculate_availability(profile)
        transition_prob = self._calculate_transition_probability(profile)
        
        # Calculate network metrics from limited data
        network_metrics = self._calculate_network_metrics(profile)
        
        # Generate serendipity factors
        serendipity_factors = self._calculate_serendipity_factors(
            profile, skills_vector, personality_phase
        )
        
        return {
            "quantum_features": {
                "skills_vector": skills_vector.tolist(),
                "personality_phase": personality_phase.tolist(),
                "availability": float(availability),
                "transition_probability": float(transition_prob)
            },
            "network_metrics": network_metrics,
            "serendipity_factors": serendipity_factors
        }
    
    def _extract_skills_vector(self, text: str, skills_list: List[str]) -> np.ndarray:
        """
        Extract 10-dimensional skills vector from text
        """
        vector = np.zeros(10)
        
        # Map skills to categories
        skill_keywords = {
            0: ["python", "java", "code", "software", "api", "backend", "frontend"],  # technical
            1: ["design", "creative", "art", "music", "writing", "content"],  # creative
            2: ["lead", "manage", "director", "executive", "ceo", "founder"],  # leadership
            3: ["communication", "marketing", "pr", "social", "community"],  # communication
            4: ["data", "analysis", "research", "metrics", "analytics"],  # analytical
            5: ["strategy", "planning", "vision", "roadmap", "business"],  # strategic
            6: ["operations", "process", "efficiency", "logistics"],  # operational
            7: ["finance", "accounting", "investment", "revenue", "budget"],  # financial
            8: ["sales", "business development", "growth", "customer"],  # sales
            9: ["research", "science", "academic", "phd", "innovation"]  # research
        }
        
        text_lower = text.lower()
        
        # Score each dimension based on keyword presence
        for dim, keywords in skill_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            vector[dim] = min(score / 3.0, 1.0)  # Normalize to 0-1
        
        # Add some noise for quantum superposition
        vector += np.random.normal(0, 0.05, 10)
        vector = np.clip(vector, 0, 1)
        
        # Normalize to unit vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _generate_personality_phase(self, bio: str) -> np.ndarray:
        """
        Generate 5-dimensional personality phase vector (0 to 2π)
        Using text analysis to estimate Big Five traits
        """
        phases = np.zeros(5)
        
        if not bio:
            # Random phases if no bio
            return np.random.uniform(0, 2*np.pi, 5)
        
        bio_lower = bio.lower()
        
        # Openness indicators
        openness_words = ["innovative", "creative", "curious", "explore", "novel", "unique"]
        openness_score = sum(1 for w in openness_words if w in bio_lower)
        phases[0] = (openness_score / 3.0) * np.pi
        
        # Conscientiousness indicators
        conscient_words = ["organized", "reliable", "detail", "thorough", "responsible"]
        conscient_score = sum(1 for w in conscient_words if w in bio_lower)
        phases[1] = (conscient_score / 3.0) * np.pi
        
        # Extraversion indicators
        extra_words = ["team", "collaborate", "social", "energetic", "enthusiastic"]
        extra_score = sum(1 for w in extra_words if w in bio_lower)
        phases[2] = (extra_score / 3.0) * np.pi
        
        # Agreeableness indicators
        agree_words = ["help", "support", "kind", "cooperat", "empath"]
        agree_score = sum(1 for w in agree_words if w in bio_lower)
        phases[3] = (agree_score / 3.0) * np.pi
        
        # Neuroticism (inverse - looking for stability)
        stable_words = ["calm", "resilient", "confident", "steady", "balanced"]
        stable_score = sum(1 for w in stable_words if w in bio_lower)
        phases[4] = (1 - stable_score / 3.0) * np.pi
        
        # Add quantum uncertainty
        phases += np.random.normal(0, 0.1, 5)
        phases = np.mod(phases, 2*np.pi)  # Wrap to [0, 2π]
        
        return phases
    
    def _calculate_availability(self, profile: Dict) -> float:
        """
        Calculate availability score (0-1) based on profile signals
        """
        availability = 0.5  # Default neutral
        
        bio = (profile.get('blurb', '') or '').lower()
        title = (profile.get('title', '') or '').lower()
        
        # Positive availability signals
        if any(word in bio for word in ["looking", "open to", "seeking", "exploring"]):
            availability += 0.3
        if any(word in title for word in ["consultant", "freelance", "advisor", "interim"]):
            availability += 0.2
        
        # Negative availability signals
        if any(word in title for word in ["ceo", "founder", "president", "partner"]):
            availability -= 0.2
        if "happy" in bio or "love my" in bio:
            availability -= 0.1
        
        return np.clip(availability, 0, 1)
    
    def _calculate_transition_probability(self, profile: Dict) -> float:
        """
        Calculate probability of career transition (0-1)
        """
        transition_prob = 0.3  # Base probability
        
        bio = (profile.get('blurb', '') or '').lower()
        
        # Transition indicators
        transition_words = [
            "transition", "pivot", "exploring", "new chapter", "next step",
            "recently", "formerly", "ex-", "was", "used to"
        ]
        
        for word in transition_words:
            if word in bio:
                transition_prob += 0.1
        
        # Check for multiple past roles mentioned
        if bio.count("previously") + bio.count("former") > 1:
            transition_prob += 0.2
        
        return np.clip(transition_prob, 0, 1)
    
    def _calculate_network_metrics(self, profile: Dict) -> Dict[str, Any]:
        """
        Calculate network metrics from available data
        """
        # Since we don't have real network data, we'll estimate based on profile
        
        # Use profile completeness as proxy for centrality
        centrality = 0.5
        if profile.get('blurb'):
            centrality += 0.1
        if profile.get('areasOfNetworkStrength'):
            centrality += 0.1
        if profile.get('title'):
            centrality += 0.1
        if profile.get('company'):
            centrality += 0.1
        
        # Bridge score - people who span multiple domains
        skills = profile.get('areasOfNetworkStrength', [])
        skill_diversity = len(set(skills[:10])) / 10.0 if skills else 0.3
        bridge_score = skill_diversity
        
        # Community detection (simplified - based on keywords)
        communities = []
        bio = (profile.get('blurb', '') or '').lower()
        
        community_keywords = {
            1: ["tech", "software", "startup"],
            2: ["finance", "investment", "banking"],
            3: ["health", "medical", "bio"],
            4: ["education", "academic", "research"],
            5: ["creative", "design", "art"],
            6: ["social", "nonprofit", "impact"]
        }
        
        for comm_id, keywords in community_keywords.items():
            if any(kw in bio for kw in keywords):
                communities.append(comm_id)
        
        if not communities:
            communities = [np.random.randint(1, 7)]  # Random assignment
        
        # Weak ties estimate (random but consistent per profile)
        profile_hash = hashlib.md5(str(profile.get('_id', '')).encode()).hexdigest()
        weak_ties = int(profile_hash[:4], 16) % 200 + 50  # 50-250 range
        
        return {
            "centrality": float(centrality),
            "bridge_score": float(bridge_score),
            "community_ids": communities[:3],  # Max 3 communities
            "weak_ties_count": weak_ties
        }
    
    def _calculate_serendipity_factors(self, profile: Dict, 
                                      skills_vector: np.ndarray,
                                      personality_phase: np.ndarray) -> Dict[str, Any]:
        """
        Calculate factors that contribute to serendipitous discovery
        """
        # Uniqueness - how rare is this combination?
        uniqueness = float(np.std(skills_vector) * np.std(personality_phase))
        uniqueness = np.clip(uniqueness * 2, 0, 1)
        
        # Timing score - are they ready for something new?
        timing_score = self._calculate_availability(profile) * 0.5 + \
                      self._calculate_transition_probability(profile) * 0.5
        
        # What they offer vs what they need (simplified)
        skills = profile.get('areasOfNetworkStrength', [])
        
        complementarity = {
            "offers": skills[:5] if skills else ["general expertise"],
            "seeks": self._infer_needs(profile)
        }
        
        return {
            "uniqueness": float(uniqueness),
            "timing_score": float(timing_score),
            "complementarity": complementarity
        }
    
    def _infer_needs(self, profile: Dict) -> List[str]:
        """
        Infer what someone might need based on their profile
        """
        title = (profile.get('title', '') or '').lower()
        needs = []
        
        # Role-based needs inference
        if "founder" in title or "ceo" in title:
            needs.extend(["technical expertise", "operations", "funding"])
        elif "engineer" in title or "developer" in title:
            needs.extend(["product vision", "business strategy", "design"])
        elif "designer" in title:
            needs.extend(["technical implementation", "user research", "growth"])
        elif "sales" in title or "marketing" in title:
            needs.extend(["product development", "analytics", "content"])
        else:
            needs.extend(["collaboration", "growth opportunities", "mentorship"])
        
        return needs[:3]  # Return top 3 needs

def enrich_profile(profile: Dict[str, Any], generator: Optional[QuantumFeatureGenerator] = None) -> Dict[str, Any]:
    """
    Convenience function to enrich a single profile with quantum features
    """
    if generator is None:
        generator = QuantumFeatureGenerator()
    
    features = generator.generate_quantum_features(profile)
    
    # Merge features into profile
    enriched_profile = profile.copy()
    enriched_profile.update(features)
    
    return enriched_profile
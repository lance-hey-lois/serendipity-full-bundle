"""
Quantum Tunneling Algorithm for Serendipitous Discovery
Finds connections that classical algorithms would never explore by tunneling through barriers
"""

import numpy as np
import pennylane as qml
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class QuantumTunneling:
    """
    Quantum tunneling to find impossible connections
    """
    
    def __init__(self, n_qubits: int = 8):
        """
        Initialize quantum tunneling system
        
        Args:
            n_qubits: Number of qubits (max 8 for simulation)
        """
        self.n_qubits = min(n_qubits, 8)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
    
    def calculate_barriers(self, query_profile: Dict, candidates: List[Dict]) -> np.ndarray:
        """
        Calculate barrier heights between query and candidates
        Higher barriers = less likely to connect classically
        
        Args:
            query_profile: Query user profile with quantum features
            candidates: List of candidate profiles
            
        Returns:
            Array of barrier heights (0-10 scale)
        """
        barriers = []
        
        for candidate in candidates:
            barrier = 0.0
            
            # 1. Network distance barrier (no mutual connections)
            query_communities = set(query_profile.get('network_metrics', {}).get('community_ids', []))
            cand_communities = set(candidate.get('network_metrics', {}).get('community_ids', []))
            
            if not query_communities.intersection(cand_communities):
                barrier += 3.0  # Different communities = high barrier
            
            # 2. Industry/domain barrier
            query_skills = query_profile.get('quantum_features', {}).get('skills_vector', [])
            cand_skills = candidate.get('quantum_features', {}).get('skills_vector', [])
            
            if query_skills and cand_skills:
                # Cosine distance as barrier
                query_skills = np.array(query_skills[:self.n_qubits])
                cand_skills = np.array(cand_skills[:self.n_qubits])
                
                cos_sim = np.dot(query_skills, cand_skills) / (
                    np.linalg.norm(query_skills) * np.linalg.norm(cand_skills) + 1e-10
                )
                skill_barrier = (1 - cos_sim) * 2  # 0-2 scale
                barrier += skill_barrier
            
            # 3. Personality phase difference barrier
            query_phase = query_profile.get('quantum_features', {}).get('personality_phase', [])
            cand_phase = candidate.get('quantum_features', {}).get('personality_phase', [])
            
            if query_phase and cand_phase:
                phase_diff = np.mean(np.abs(np.array(query_phase[:5]) - np.array(cand_phase[:5])))
                barrier += phase_diff / np.pi  # Normalize to 0-2
            
            # 4. Availability mismatch barrier
            query_avail = query_profile.get('quantum_features', {}).get('availability', 0.5)
            cand_avail = candidate.get('quantum_features', {}).get('availability', 0.5)
            
            avail_diff = abs(query_avail - cand_avail)
            if avail_diff > 0.5:
                barrier += 1.5
            
            # 5. Centrality difference (power imbalance)
            query_central = query_profile.get('network_metrics', {}).get('centrality', 0.5)
            cand_central = candidate.get('network_metrics', {}).get('centrality', 0.5)
            
            central_diff = abs(query_central - cand_central)
            if central_diff > 0.6:
                barrier += 1.0
            
            barriers.append(min(barrier, 10.0))  # Cap at 10
        
        return np.array(barriers)
    
    def create_tunneling_circuit(self, n_wires):
        """Create a quantum circuit for tunneling"""
        dev = qml.device("default.qubit", wires=n_wires)
        
        @qml.qnode(dev)
        def circuit(query_features, cand_features, barrier_height):
            n = len(query_features)
            
            # Layer 1: Encode query state with barrier influence
            for i in range(n):
                qml.RY(query_features[i] * np.exp(-barrier_height/10), wires=i)
                qml.RZ(query_features[i] * 0.5, wires=i)
            
            # Layer 2: Create superposition for tunneling
            for i in range(n):
                qml.Hadamard(wires=i)
            
            # Layer 3: Entanglement creates tunneling paths
            # The more entanglement, the more tunneling possibilities
            for _ in range(min(int(barrier_height), 3)):  # Cap iterations
                # Ring entanglement
                for i in range(n-1):
                    qml.CNOT(wires=[i, i+1])
                if n > 2:
                    qml.CNOT(wires=[n-1, 0])
                
                # Cross entanglement for long-range correlations
                for i in range(0, n-2, 2):
                    qml.CNOT(wires=[i, i+2])
            
            # Layer 4: Interference with candidate features
            for i in range(n):
                # Tunneling operator: exp(-barrier * distance)
                tunneling_strength = np.exp(-barrier_height / 5)
                qml.RY(cand_features[i] * tunneling_strength, wires=i)
                
                # Phase kickback for quantum advantage
                qml.RZ((query_features[i] - cand_features[i]) * np.pi, wires=i)
            
            # Layer 5: Reverse entanglement to amplify tunneled states
            for i in range(n-1, 0, -1):
                qml.CNOT(wires=[i, i-1])
            
            # Layer 6: Final encoding
            for i in range(n):
                qml.RY(-cand_features[i] * np.exp(-barrier_height/10), wires=i)
            
            # Measure tunneling probability
            return qml.probs(wires=range(n))
        
        return circuit
    
    def quantum_tunnel(self, query_features: np.ndarray,
                      candidate_features: List[np.ndarray],
                      barrier_heights: np.ndarray) -> np.ndarray:
        """
        Apply quantum tunneling to find connections through barriers
        
        Args:
            query_features: Query profile quantum features
            candidate_features: List of candidate quantum features
            barrier_heights: Barrier height for each candidate
            
        Returns:
            Tunneling probabilities for each candidate
        """
        tunneling_probs = []
        
        # Ensure features are the right size
        query_feat = np.array(query_features[:self.n_qubits])
        query_feat = np.pad(query_feat, (0, self.n_qubits - len(query_feat)), 'constant')
        
        for i, (cand_feat, barrier) in enumerate(zip(candidate_features, barrier_heights)):
            try:
                # Prepare candidate features
                cand_feat = np.array(cand_feat[:self.n_qubits])
                cand_feat = np.pad(cand_feat, (0, self.n_qubits - len(cand_feat)), 'constant')
                
                # Create and run quantum circuit
                circuit = self.create_tunneling_circuit(self.n_qubits)
                probs = circuit(query_feat, cand_feat, barrier)
                
                # Calculate tunneling probability
                # High probability of |0...0⟩ state means successful tunneling
                tunneling_prob = float(probs[0])
                
                # Boost probability for high barriers (reward quantum tunneling)
                if barrier > 5:
                    tunneling_prob *= (1 + barrier/10)
                
                tunneling_probs.append(tunneling_prob)
                
            except Exception as e:
                logger.warning(f"Tunneling circuit failed for candidate {i}: {e}")
                tunneling_probs.append(0.0)
        
        # Normalize probabilities
        probs_array = np.array(tunneling_probs)
        if probs_array.max() > 0:
            probs_array = probs_array / probs_array.max()
        
        return probs_array
    
    def find_tunneled_connections(self, query_profile: Dict, 
                                 candidates: List[Dict],
                                 top_k: int = 10) -> List[Tuple[Dict, float, float]]:
        """
        Find connections that require quantum tunneling to discover
        
        Args:
            query_profile: Query user profile
            candidates: List of candidate profiles
            top_k: Number of top tunneled connections to return
            
        Returns:
            List of (profile, tunneling_probability, barrier_height) tuples
        """
        if not candidates:
            return []
        
        # Calculate barriers
        barriers = self.calculate_barriers(query_profile, candidates)
        
        # Extract quantum features
        query_features = query_profile.get('quantum_features', {}).get('skills_vector', [])
        if not query_features:
            # Fallback to random features
            query_features = np.random.rand(self.n_qubits)
        
        candidate_features = []
        for cand in candidates:
            feat = cand.get('quantum_features', {}).get('skills_vector', [])
            if not feat:
                feat = np.random.rand(self.n_qubits)
            candidate_features.append(feat)
        
        # Process in batches for quantum circuit
        batch_size = 10
        all_tunneling_probs = []
        
        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidate_features[i:i+batch_size]
            batch_barriers = barriers[i:i+batch_size]
            
            batch_probs = self.quantum_tunnel(
                query_features,
                batch_candidates,
                batch_barriers
            )
            
            all_tunneling_probs.extend(batch_probs)
        
        # Combine results
        results = []
        for i, (cand, prob, barrier) in enumerate(zip(candidates, all_tunneling_probs, barriers)):
            # Only include high-barrier connections with good tunneling probability
            if barrier > 3.0 and prob > 0.3:  # High barrier but quantum found a way
                results.append((cand, float(prob), float(barrier)))
        
        # Sort by tunneling probability
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


def demo_tunneling():
    """
    Demo the quantum tunneling algorithm
    """
    # Create sample profiles
    query = {
        'name': 'Tech Founder',
        'quantum_features': {
            'skills_vector': [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            'personality_phase': [np.pi/4, np.pi/2, np.pi/3, np.pi/6, np.pi],
            'availability': 0.8
        },
        'network_metrics': {
            'centrality': 0.7,
            'community_ids': [1, 2]
        }
    }
    
    # Create candidates with various barrier levels
    candidates = [
        {
            'name': 'Similar Tech Person',
            'quantum_features': {
                'skills_vector': [0.85, 0.15, 0.75, 0.25, 0.65, 0.35, 0.55, 0.45],
                'personality_phase': [np.pi/4, np.pi/2, np.pi/3, np.pi/6, np.pi],
                'availability': 0.7
            },
            'network_metrics': {
                'centrality': 0.6,
                'community_ids': [1, 3]  # Some overlap
            }
        },
        {
            'name': 'Jazz Musician with Hidden Tech Skills',
            'quantum_features': {
                'skills_vector': [0.3, 0.9, 0.2, 0.8, 0.1, 0.7, 0.4, 0.6],
                'personality_phase': [np.pi, np.pi/6, np.pi/2, np.pi/3, np.pi/4],
                'availability': 0.9
            },
            'network_metrics': {
                'centrality': 0.3,
                'community_ids': [5, 6]  # No overlap - high barrier!
            }
        },
        {
            'name': 'Retired NASA Engineer',
            'quantum_features': {
                'skills_vector': [0.7, 0.2, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
                'personality_phase': [np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6],
                'availability': 0.95
            },
            'network_metrics': {
                'centrality': 0.2,
                'community_ids': [8, 9]  # Completely different world
            }
        }
    ]
    
    # Run tunneling
    tunneler = QuantumTunneling(n_qubits=8)
    results = tunneler.find_tunneled_connections(query, candidates, top_k=3)
    
    print("\n🌌 Quantum Tunneling Results:")
    print("="*60)
    for profile, prob, barrier in results:
        print(f"\n✨ {profile['name']}")
        print(f"   Barrier Height: {barrier:.1f} (classical wouldn't cross)")
        print(f"   Tunneling Probability: {prob:.3f}")
        print(f"   Magic Factor: {'🔮' * int(prob * 5)}")

if __name__ == "__main__":
    demo_tunneling()
#!/usr/bin/env python3
"""
Quantum Discovery with Learned Reducer V2
Fixed quantum kernel implementation
"""

import numpy as np
import pennylane as qml
from typing import List, Dict, Any, Tuple
import torch
import sys
import os
sys.path.append('..')
from learning.learned_reducer import load_reducer

class LearnedQuantumDiscoveryV2:
    """
    Quantum discovery using learned embedding reducer with fixed kernel
    """
    
    def __init__(self, n_qubits: int = 8, reducer_path: str = '../learning/learned_reducer.pt'):
        """
        Initialize with learned reducer
        """
        self.n_qubits = n_qubits
        
        # Load the learned reducer
        if os.path.exists(reducer_path):
            self.reducer = load_reducer(reducer_path)
            print(f"✅ Loaded learned reducer from {reducer_path}")
        else:
            self.reducer = None
            print("⚠️ No learned reducer found, will use raw features")
        
        # Initialize quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
    def reduce_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Reduce embedding using learned reducer with tanh normalization
        """
        if self.reducer is None:
            # Fallback
            reduced = embedding[:self.n_qubits]
            return (np.tanh(reduced) + 1) * np.pi / 2
        
        # Use learned reducer
        with torch.no_grad():
            emb_tensor = torch.FloatTensor(embedding)
            reduced = self.reducer.reduce_single(emb_tensor).detach().numpy()
            
        # Use tanh normalization for stable range [0, π]
        return (np.tanh(reduced) + 1) * np.pi / 2
    
    def quantum_kernel(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Simplified quantum kernel that actually works
        """
        @qml.qnode(self.dev)
        def kernel_circuit(f1, f2):
            # Encode difference directly
            diff = f1 - f2
            
            # Apply rotation based on difference
            for i in range(self.n_qubits):
                qml.RY(diff[i], wires=i)
            
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # More rotations for complexity
            for i in range(self.n_qubits):
                qml.RZ(diff[i] * 0.5, wires=i)
            
            # Measure expectation of all-zeros
            return qml.probs(wires=list(range(self.n_qubits)))
        
        probs = kernel_circuit(features1, features2)
        # High prob[0] means similar (small difference)
        return float(probs[0])
    
    def classical_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def discover_connections(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[Tuple[str, np.ndarray]],
        top_k: int = 10,
        blend_factor: float = 0.7  # How much to weight quantum vs classical
    ) -> List[Dict[str, Any]]:
        """
        Discover connections using blended quantum + classical approach
        """
        # Reduce query embedding
        query_reduced = self.reduce_embedding(query_embedding)
        
        results = []
        
        for user_id, candidate_embedding in candidate_embeddings:
            # Reduce candidate
            candidate_reduced = self.reduce_embedding(candidate_embedding)
            
            # Quantum similarity (in reduced space)
            quantum_score = self.quantum_kernel(query_reduced, candidate_reduced)
            
            # Classical similarity (in original space)
            classical_score = self.classical_similarity(query_embedding, candidate_embedding)
            
            # Blended score
            blended_score = blend_factor * quantum_score + (1 - blend_factor) * classical_score
            
            # Serendipity: High quantum but moderate classical
            # This finds hidden connections
            serendipity_factor = quantum_score * np.exp(-abs(classical_score - 0.4))
            
            results.append({
                'user_id': user_id,
                'quantum_score': float(quantum_score),
                'classical_score': float(classical_score),
                'blended_score': float(blended_score),
                'serendipity_score': float(serendipity_factor)
            })
        
        # Sort by blended score but boost serendipitous ones
        results.sort(key=lambda x: x['blended_score'] + 0.3 * x['serendipity_score'], reverse=True)
        
        return results[:top_k]
    
    def explain_connection(self, result: Dict[str, Any]) -> str:
        """
        Explain why a connection is recommended
        """
        quantum = result['quantum_score']
        classical = result['classical_score']
        serendipity = result['serendipity_score']
        
        if serendipity > 0.3:
            return "🎯 Serendipitous discovery! Hidden compatibility detected through quantum patterns"
        elif quantum > 0.8 and classical > 0.8:
            return "⭐ Perfect match! Strong alignment in both quantum and classical spaces"
        elif quantum > 0.7:
            return "🔮 Quantum connection: Deep pattern alignment suggests strong potential"
        elif classical > 0.8:
            return "🤝 Natural fit: Clear similarities in skills and interests"
        elif quantum > 0.5 and classical < 0.4:
            return "💡 Unexpected synergy: Quantum analysis reveals hidden complementarity"
        else:
            return "🌱 Potential connection: Some interesting overlaps to explore"


def test_discovery_v2():
    """Test the improved quantum discovery"""
    from pymongo import MongoClient
    from dotenv import load_dotenv
    
    load_dotenv('../.env.dev')
    if not os.getenv("MONGODB_URI"):
        load_dotenv('../.env')
    
    # Initialize
    discovery = LearnedQuantumDiscoveryV2()
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Get test users - relax the bio requirement
    users = list(db['public_profiles'].find(
        {'embedding': {'$exists': True}},
        {'name': 1, 'slug': 1, 'embedding': 1, 'bio': 1, 'skills': 1}
    ).limit(50))
    
    if len(users) < 2:
        print("Not enough users with embeddings")
        return
    
    # Find a good query user (someone with bio containing specific interests)
    query_user = None
    for user in users:
        bio = user.get('bio', '').lower()
        if any(word in bio for word in ['music', 'jazz', 'code', 'tech', 'creative']):
            query_user = user
            break
    
    if not query_user:
        query_user = users[0]
    
    # Get candidates (everyone except query user)
    candidates = [(u['slug'], np.array(u['embedding'])) 
                  for u in users if u['slug'] != query_user['slug']]
    
    print(f"\n🔍 Finding connections for: {query_user.get('name', 'Unknown')}")
    if query_user.get('bio'):
        print(f"   Bio: {query_user['bio'][:150]}...")
    if query_user.get('skills'):
        print(f"   Skills: {', '.join(query_user['skills'][:5])}")
    
    # Discover connections
    results = discovery.discover_connections(
        np.array(query_user['embedding']),
        candidates,
        top_k=10
    )
    
    print("\n🌟 Top Connections (Quantum + Classical Blended):")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        user = next(u for u in users if u['slug'] == result['user_id'])
        print(f"\n{i}. {user.get('name', 'Unknown')}")
        print(f"   Quantum: {result['quantum_score']:.3f} | Classical: {result['classical_score']:.3f}")
        print(f"   Blended: {result['blended_score']:.3f} | Serendipity: {result['serendipity_score']:.3f}")
        print(f"   {discovery.explain_connection(result)}")
        if user.get('bio'):
            print(f"   Bio: {user['bio'][:100]}...")
        if user.get('skills'):
            print(f"   Skills: {', '.join(user['skills'][:3])}")
    
    # Show distribution of scores
    print("\n📊 Score Distribution:")
    quantum_scores = [r['quantum_score'] for r in results]
    classical_scores = [r['classical_score'] for r in results]
    print(f"   Quantum range: {min(quantum_scores):.3f} - {max(quantum_scores):.3f}")
    print(f"   Classical range: {min(classical_scores):.3f} - {max(classical_scores):.3f}")


if __name__ == "__main__":
    test_discovery_v2()
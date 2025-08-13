#!/usr/bin/env python3
"""
Quantum Discovery with Learned Reducer
Uses neural network-learned embedding reduction instead of PCA
"""

import numpy as np
import pennylane as qml
from typing import List, Dict, Any, Tuple
import torch
import sys
import os
sys.path.append('..')
from learning.learned_reducer import load_reducer

class LearnedQuantumDiscovery:
    """
    Quantum discovery using learned embedding reducer
    """
    
    def __init__(self, n_qubits: int = 8, reducer_path: str = '../learning/learned_reducer.pt'):
        """
        Initialize with learned reducer
        
        Args:
            n_qubits: Number of qubits (should match reducer output dim)
            reducer_path: Path to trained reducer model
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
        Reduce embedding using learned reducer
        
        Args:
            embedding: 1536-dim embedding
            
        Returns:
            8-dim reduced representation
        """
        if self.reducer is None:
            # Fallback: take first n_qubits dimensions and normalize
            reduced = embedding[:self.n_qubits]
            reduced = (reduced - reduced.min()) / (reduced.max() - reduced.min() + 1e-10)
            return reduced * np.pi
        
        # Use learned reducer
        with torch.no_grad():
            emb_tensor = torch.FloatTensor(embedding)
            reduced = self.reducer.reduce_single(emb_tensor).detach().numpy()
            
        # Scale to quantum angle range [0, π] - with better normalization
        # Add small noise to avoid all-zero issues
        reduced = reduced + np.random.normal(0, 0.01, size=reduced.shape)
        
        # Normalize to [0, 1] then scale to [0.1, π-0.1] to avoid extremes
        reduced = (reduced - reduced.min()) / (reduced.max() - reduced.min() + 1e-10)
        return reduced * (np.pi - 0.2) + 0.1
    
    def quantum_kernel(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute quantum kernel between two feature vectors
        Uses learned reduction for better similarity
        """
        @qml.qnode(self.dev)
        def kernel_circuit(f1, f2):
            # Encode first feature vector
            for i in range(self.n_qubits):
                qml.RY(f1[i], wires=i)
                qml.RZ(f1[i], wires=i)
            
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Encode second feature vector (inverse)
            for i in range(self.n_qubits):
                qml.RZ(-f2[i], wires=i)
                qml.RY(-f2[i], wires=i)
            
            # Measure overlap
            return qml.probs(wires=list(range(self.n_qubits)))
        
        probs = kernel_circuit(features1, features2)
        # Return probability of all zeros (maximum overlap)
        return float(probs[0])
    
    def discover_connections(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[Tuple[str, np.ndarray]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Discover serendipitous connections using learned reducer
        
        Args:
            query_embedding: Query user's embedding
            candidate_embeddings: List of (user_id, embedding) tuples
            top_k: Number of top connections to return
            
        Returns:
            List of connection recommendations with scores
        """
        # Reduce query embedding
        query_reduced = self.reduce_embedding(query_embedding)
        
        results = []
        
        for user_id, candidate_embedding in candidate_embeddings:
            # Reduce candidate embedding
            candidate_reduced = self.reduce_embedding(candidate_embedding)
            
            # Compute quantum kernel similarity
            quantum_score = self.quantum_kernel(query_reduced, candidate_reduced)
            
            # Compute classical cosine similarity for comparison
            cos_sim = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            
            # Learned reducer should capture non-obvious patterns
            # High quantum score with moderate cosine = serendipity!
            serendipity_score = quantum_score * (1.0 - abs(cos_sim - 0.5))
            
            results.append({
                'user_id': user_id,
                'quantum_score': float(quantum_score),
                'cosine_similarity': float(cos_sim),
                'serendipity_score': float(serendipity_score),
                'reduced_query': query_reduced.tolist(),
                'reduced_candidate': candidate_reduced.tolist()
            })
        
        # Sort by serendipity score
        results.sort(key=lambda x: x['serendipity_score'], reverse=True)
        
        return results[:top_k]
    
    def explain_connection(self, result: Dict[str, Any]) -> str:
        """
        Explain why a connection is serendipitous
        """
        quantum_score = result['quantum_score']
        cos_sim = result['cosine_similarity']
        serendipity = result['serendipity_score']
        
        if quantum_score > 0.7 and cos_sim < 0.3:
            return "🎯 Hidden gem: Quantum patterns reveal deep compatibility despite surface differences"
        elif quantum_score > 0.6 and cos_sim > 0.7:
            return "✨ Natural fit: Strong alignment in both quantum and classical spaces"
        elif serendipity > 0.5:
            return "🌟 Serendipitous match: Unexpected synergies discovered through quantum analysis"
        elif quantum_score > 0.5:
            return "🔮 Quantum connection: Subtle patterns suggest potential collaboration"
        else:
            return "🤝 Worth exploring: Some interesting overlaps detected"


def test_learned_discovery():
    """Test the learned quantum discovery"""
    import sys
    sys.path.append('..')
    from pymongo import MongoClient
    from dotenv import load_dotenv
    
    load_dotenv('../.env.dev')
    if not os.getenv("MONGODB_URI"):
        load_dotenv('../.env')
    
    # Initialize
    discovery = LearnedQuantumDiscovery()
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Get test users
    users = list(db['public_profiles'].find(
        {'embedding': {'$exists': True}},
        {'name': 1, 'slug': 1, 'embedding': 1, 'bio': 1}
    ).limit(20))
    
    if len(users) < 2:
        print("Not enough users with embeddings")
        return
    
    # Use first user as query
    query_user = users[0]
    candidates = [(u['slug'], np.array(u['embedding'])) for u in users[1:]]
    
    print(f"\n🔍 Finding connections for: {query_user.get('name', 'Unknown')}")
    if query_user.get('bio'):
        print(f"   Bio: {query_user['bio'][:100]}...")
    
    # Discover connections
    results = discovery.discover_connections(
        np.array(query_user['embedding']),
        candidates,
        top_k=5
    )
    
    print("\n🌟 Top Serendipitous Connections:")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        user = next(u for u in users if u['slug'] == result['user_id'])
        print(f"\n{i}. {user.get('name', 'Unknown')}")
        print(f"   Quantum Score: {result['quantum_score']:.3f}")
        print(f"   Cosine Similarity: {result['cosine_similarity']:.3f}")
        print(f"   Serendipity Score: {result['serendipity_score']:.3f}")
        print(f"   {discovery.explain_connection(result)}")
        if user.get('bio'):
            print(f"   Bio: {user['bio'][:100]}...")


if __name__ == "__main__":
    test_learned_discovery()
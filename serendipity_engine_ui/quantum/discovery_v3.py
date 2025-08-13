#!/usr/bin/env python3
"""
Quantum Discovery V3 - Production-ready implementation
Fixes:
- Proper two-tower reducer
- Quantum fidelity kernel
- L2 normalization
- Precomputed reductions
- Better serendipity scoring
"""

import numpy as np
import pennylane as qml
from typing import List, Dict, Any, Tuple, Optional
import torch
import sys
import os
sys.path.append('..')
from learning.two_tower_reducer import load_two_tower_reducer, Tower


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalize a vector"""
    return x / (np.linalg.norm(x) + 1e-12)


class QuantumDiscoveryV3:
    """
    Production-ready quantum discovery with all fixes applied
    """
    
    def __init__(self, n_qubits: int = 8, reducer_path: str = '../learning/two_tower_reducer.pt'):
        """
        Initialize with two-tower reducer and quantum fidelity kernel
        
        Args:
            n_qubits: Number of qubits (should match reducer output dim)
            reducer_path: Path to trained two-tower reducer model
        """
        self.n_qubits = n_qubits
        
        # Load the two-tower reducer (just the tower for inference)
        if os.path.exists(reducer_path):
            self.tower = load_two_tower_reducer(reducer_path)
            print(f"✅ Loaded two-tower reducer from {reducer_path}")
        else:
            # Fallback to old reducer if available
            old_path = reducer_path.replace('two_tower_', '')
            if os.path.exists(old_path):
                print(f"⚠️ Using old reducer from {old_path}")
                from learning.learned_reducer import load_reducer
                self.tower = None
                self.old_reducer = load_reducer(old_path)
            else:
                self.tower = None
                self.old_reducer = None
                print("⚠️ No reducer found, will use raw features")
        
        # Initialize quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Create cached quantum kernel function
        self._create_kernel()
    
    def _create_kernel(self):
        """Create cached quantum fidelity kernel"""
        
        def _feature_map(x):
            """Quantum feature map"""
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        @qml.qnode(self.dev)
        def _fidelity(x, y):
            """Quantum fidelity |<ψ(x)|ψ(y)>|²"""
            _feature_map(x)
            qml.adjoint(_feature_map)(y)
            return qml.probs(wires=range(self.n_qubits))
        
        self._fidelity_kernel = _fidelity
    
    def reduce_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Reduce embedding using two-tower reducer with proper normalization
        
        Args:
            embedding: 1536 or 3072-dim embedding
            
        Returns:
            8-dim reduced representation scaled to quantum angles
        """
        # Truncate if text-embedding-3-large (3072d)
        if len(embedding) > 1536:
            embedding = embedding[:1536]
        
        # L2 normalize input
        embedding = l2_normalize(embedding)
        
        if self.tower is not None:
            # Use two-tower reducer
            with torch.no_grad():
                emb_tensor = torch.FloatTensor(embedding)
                reduced = self.tower(emb_tensor).numpy()
        elif self.old_reducer is not None:
            # Fallback to old reducer
            with torch.no_grad():
                emb_tensor = torch.FloatTensor(embedding)
                reduced = self.old_reducer.reduce_single(emb_tensor).detach().numpy()
        else:
            # Fallback: take first n_qubits dimensions
            reduced = embedding[:self.n_qubits]
        
        # Z-score normalization then clamp to stable range
        mean = np.mean(reduced)
        std = np.std(reduced) + 1e-8
        reduced = (reduced - mean) / std
        reduced = np.clip(reduced, -2, 2)  # Clamp to ±2 std
        
        # Map to quantum angles [0.1, π-0.1] (avoid extremes)
        reduced = (reduced + 2) / 4  # Now in [0, 1]
        reduced = reduced * (np.pi - 0.2) + 0.1
        
        return reduced
    
    def quantum_kernel(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute quantum fidelity kernel between two feature vectors
        Well-behaved, PSD kernel
        """
        probs = self._fidelity_kernel(features1, features2)
        return float(probs[0])  # |<ψ(x)|ψ(y)>|²
    
    def classical_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity with proper normalization"""
        emb1 = l2_normalize(emb1)
        emb2 = l2_normalize(emb2)
        return float(np.dot(emb1, emb2))
    
    def discover_connections(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[Tuple[str, np.ndarray]],
        top_k: int = 10,
        blend_factor: float = 0.7,
        precompute: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Discover connections with all optimizations
        
        Args:
            query_embedding: Query user's embedding (will be L2 normalized)
            candidate_embeddings: List of (user_id, embedding) tuples
            top_k: Number of top connections to return
            blend_factor: Weight for quantum vs classical (0.7 = 70% quantum)
            precompute: Whether to precompute all reductions (faster)
            
        Returns:
            List of connection recommendations with scores
        """
        # L2 normalize query
        query_embedding = l2_normalize(query_embedding)
        
        # Reduce query once
        query_reduced = self.reduce_embedding(query_embedding)
        
        # Precompute all candidate reductions if requested
        if precompute:
            print("Precomputing candidate reductions...")
            candidate_reductions = {}
            for user_id, emb in candidate_embeddings:
                emb_normalized = l2_normalize(emb)
                candidate_reductions[user_id] = self.reduce_embedding(emb_normalized)
        
        results = []
        
        for user_id, candidate_embedding in candidate_embeddings:
            # L2 normalize candidate
            candidate_embedding = l2_normalize(candidate_embedding)
            
            # Get or compute reduction
            if precompute:
                candidate_reduced = candidate_reductions[user_id]
            else:
                candidate_reduced = self.reduce_embedding(candidate_embedding)
            
            # Quantum similarity (in reduced space)
            quantum_score = self.quantum_kernel(query_reduced, candidate_reduced)
            
            # Classical similarity (in original space)
            classical_score = self.classical_similarity(query_embedding, candidate_embedding)
            
            # Better serendipity: reward "quantum >> classical"
            novelty = max(0.0, quantum_score - max(0.0, classical_score))
            
            # Blended score with novelty bonus
            blended_score = (
                blend_factor * quantum_score + 
                (1 - blend_factor) * classical_score + 
                0.2 * novelty
            )
            
            # Serendipity: high quantum with moderate classical
            serendipity_score = quantum_score * np.exp(-abs(classical_score - 0.4))
            
            results.append({
                'user_id': user_id,
                'quantum_score': float(quantum_score),
                'classical_score': float(classical_score),
                'blended_score': float(blended_score),
                'serendipity_score': float(serendipity_score),
                'novelty_score': float(novelty)
            })
        
        # Sort by blended score
        results.sort(key=lambda x: x['blended_score'], reverse=True)
        
        return results[:top_k]
    
    def explain_connection(self, result: Dict[str, Any]) -> str:
        """
        Explain why a connection is recommended
        """
        quantum = result['quantum_score']
        classical = result['classical_score']
        novelty = result.get('novelty_score', 0)
        serendipity = result['serendipity_score']
        
        if novelty > 0.3:
            return "🎯 Quantum discovery! Deep patterns reveal hidden compatibility"
        elif serendipity > 0.3:
            return "✨ Serendipitous match! Unexpected synergies detected"
        elif quantum > 0.8 and classical > 0.8:
            return "⭐ Perfect alignment in both quantum and classical spaces"
        elif quantum > 0.7:
            return "🔮 Strong quantum resonance suggests deep compatibility"
        elif classical > 0.8:
            return "🤝 Clear surface-level similarities and shared interests"
        else:
            return "🌱 Potential connection worth exploring"


def test_v3():
    """Test the V3 implementation"""
    from pymongo import MongoClient
    from dotenv import load_dotenv
    
    load_dotenv('../.env')
    
    # Initialize
    discovery = QuantumDiscoveryV3()
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Get test users
    users = list(db['public_profiles'].find(
        {'embedding': {'$exists': True}},
        {'name': 1, 'slug': 1, 'embedding': 1}
    ).limit(50))
    
    if len(users) < 2:
        print("Not enough users with embeddings")
        return
    
    # Use first user as query
    query_user = users[0]
    candidates = [(u['slug'], np.array(u['embedding'])) for u in users[1:]]
    
    print(f"\n🔍 Finding connections for: {query_user.get('name', 'Unknown')}")
    
    # Discover connections with precomputation
    results = discovery.discover_connections(
        np.array(query_user['embedding']),
        candidates,
        top_k=10,
        precompute=True  # Precompute all reductions
    )
    
    print("\n🌟 Top Connections (V3 Implementation):")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        user = next(u for u in users if u['slug'] == result['user_id'])
        print(f"\n{i}. {user.get('name', 'Unknown')}")
        print(f"   Quantum: {result['quantum_score']:.3f} | Classical: {result['classical_score']:.3f}")
        print(f"   Blended: {result['blended_score']:.3f} | Novelty: {result['novelty_score']:.3f}")
        print(f"   {discovery.explain_connection(result)}")


if __name__ == "__main__":
    test_v3()
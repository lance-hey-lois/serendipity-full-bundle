"""
Quantum Feature Generation V2 - Using REAL Embeddings
Reduces 1536-dim semantic embeddings to quantum-compatible features
"""

import numpy as np
from typing import Dict, List, Any, Optional
# from sklearn.decomposition import PCA  # Not installed, using manual PCA
import hashlib

class QuantumFeatureGeneratorV2:
    """Generate quantum features from actual embeddings"""
    
    def __init__(self, n_quantum_dims: int = 8):
        """
        Initialize with PCA for dimensionality reduction
        
        Args:
            n_quantum_dims: Number of quantum dimensions (max 8 for simulation)
        """
        self.n_quantum_dims = n_quantum_dims
        self.pca = None
        self.is_fitted = False
        
    def fit_pca(self, embeddings: np.ndarray):
        """
        Fit simple PCA using SVD (no sklearn needed)
        
        Args:
            embeddings: Array of shape (n_samples, 1536)
        """
        # Center the data
        self.mean = np.mean(embeddings, axis=0)
        centered = embeddings - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top n_quantum_dims components
        self.components = eigenvectors[:, :self.n_quantum_dims]
        self.explained_variance = eigenvalues[:self.n_quantum_dims]
        
        self.is_fitted = True
        total_var = np.sum(eigenvalues)
        explained_ratio = self.explained_variance / total_var
        print(f"PCA fitted. Explained variance ratio: {explained_ratio}")
        print(f"Total variance explained: {np.sum(explained_ratio):.2%}")
        
    def generate_quantum_features(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate quantum features from profile's actual embedding
        
        Args:
            profile: MongoDB profile with 'embedding' field
            
        Returns:
            Quantum features dictionary
        """
        # Get the actual embedding
        embedding = profile.get('embedding')
        if not embedding:
            # No embedding - return random features as fallback
            return self._generate_fallback_features(profile)
        
        # Convert to numpy array
        embedding = np.array(embedding).reshape(1, -1)
        
        # If PCA not fitted yet, use simple truncation
        if not self.is_fitted:
            # Take first n dimensions and normalize
            quantum_vector = embedding[0, :self.n_quantum_dims]
            # Scale to quantum angle range [0, π]
            quantum_vector = (quantum_vector - quantum_vector.min()) / (quantum_vector.max() - quantum_vector.min() + 1e-10)
            quantum_vector = quantum_vector * np.pi
        else:
            # Use PCA reduction (manual transform)
            centered = embedding - self.mean
            quantum_vector = np.dot(centered, self.components)[0]
            # Scale to quantum angle range [0, π]
            quantum_vector = (quantum_vector - quantum_vector.min()) / (quantum_vector.max() - quantum_vector.min() + 1e-10)
            quantum_vector = quantum_vector * np.pi
        
        # Generate phase vector from embedding patterns
        # Use different slices of embedding for phase information
        phase_dims = 5
        phase_start = 100  # Use middle section of embedding
        phase_vector = embedding[0, phase_start:phase_start+phase_dims]
        # Convert to phases [0, 2π]
        phase_vector = (phase_vector - phase_vector.min()) / (phase_vector.max() - phase_vector.min() + 1e-10)
        phase_vector = phase_vector * 2 * np.pi
        
        # Calculate derived metrics from embedding
        # Availability: Use variance in certain dimensions as proxy
        availability_dims = embedding[0, 200:250]
        availability = float(np.std(availability_dims))  # High variance = more available/flexible
        availability = min(availability * 2, 1.0)  # Scale to 0-1
        
        # Transition probability: Use mean of another slice
        transition_dims = embedding[0, 300:350]
        transition_prob = float(np.abs(np.mean(transition_dims)))
        transition_prob = min(transition_prob * 3, 1.0)  # Scale to 0-1
        
        # Network metrics from embedding patterns
        network_metrics = self._extract_network_metrics(embedding[0])
        
        # Serendipity factors from embedding
        serendipity_factors = self._extract_serendipity_factors(embedding[0], profile)
        
        return {
            "quantum_features": {
                "skills_vector": quantum_vector.tolist(),
                "personality_phase": phase_vector.tolist(),
                "availability": availability,
                "transition_probability": transition_prob,
                "embedding_pca": True  # Flag that this uses real embeddings
            },
            "network_metrics": network_metrics,
            "serendipity_factors": serendipity_factors
        }
    
    def _extract_network_metrics(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Extract network-like metrics from embedding patterns
        """
        # Use different regions of embedding for different metrics
        
        # Centrality: How "central" are the embedding values (near mean = central)
        centrality = 1.0 - np.std(embedding[:100])  # Low variance = central
        centrality = max(0, min(1, centrality))
        
        # Bridge score: Variance in middle dimensions (connects different concepts)
        bridge_score = np.std(embedding[500:600])
        bridge_score = min(bridge_score * 2, 1.0)
        
        # Community detection from embedding clusters
        # Use sign patterns in different regions
        communities = []
        for i in range(0, 1500, 300):
            region = embedding[i:i+300]
            if np.mean(region) > 0:
                communities.append(i // 300 + 1)
        
        # Weak ties: Estimated from embedding sparsity
        weak_ties = int(np.sum(np.abs(embedding) < 0.1) * 0.5)  # Count near-zero values
        
        return {
            "centrality": float(centrality),
            "bridge_score": float(bridge_score),
            "community_ids": communities[:3],
            "weak_ties_count": weak_ties
        }
    
    def _extract_serendipity_factors(self, embedding: np.ndarray, profile: Dict) -> Dict[str, Any]:
        """
        Extract serendipity-relevant factors from embedding
        """
        # Uniqueness: How far from mean embedding pattern
        uniqueness = float(np.std(embedding[800:900]))
        uniqueness = min(uniqueness * 2, 1.0)
        
        # Timing score: Combination of various embedding signals
        timing_signals = embedding[1000:1050]
        timing_score = float(np.abs(np.mean(timing_signals)) + np.std(timing_signals))
        timing_score = min(timing_score, 1.0)
        
        # Complementarity: What dimensions are strong/weak
        strong_dims = np.where(np.abs(embedding[:500]) > np.percentile(np.abs(embedding[:500]), 80))[0]
        weak_dims = np.where(np.abs(embedding[:500]) < np.percentile(np.abs(embedding[:500]), 20))[0]
        
        # Map to rough skill areas based on position
        offers = []
        seeks = []
        
        if len(strong_dims) > 0:
            if strong_dims[0] < 100:
                offers.append("technical expertise")
            elif strong_dims[0] < 200:
                offers.append("creative vision")
            elif strong_dims[0] < 300:
                offers.append("strategic thinking")
            else:
                offers.append("operational excellence")
        
        if len(weak_dims) > 0:
            if weak_dims[0] < 100:
                seeks.append("technical support")
            elif weak_dims[0] < 200:
                seeks.append("creative input")
            elif weak_dims[0] < 300:
                seeks.append("strategic guidance")
            else:
                seeks.append("operational help")
        
        return {
            "uniqueness": uniqueness,
            "timing_score": timing_score,
            "complementarity": {
                "offers": offers or ["general expertise"],
                "seeks": seeks or ["collaboration"]
            }
        }
    
    def _generate_fallback_features(self, profile: Dict) -> Dict[str, Any]:
        """
        Generate fallback features when no embedding exists
        """
        # Use profile ID for consistent randomness
        profile_hash = hashlib.md5(str(profile.get('_id', '')).encode()).hexdigest()
        seed = int(profile_hash[:8], 16)
        np.random.seed(seed)
        
        return {
            "quantum_features": {
                "skills_vector": (np.random.rand(self.n_quantum_dims) * np.pi).tolist(),
                "personality_phase": (np.random.rand(5) * 2 * np.pi).tolist(),
                "availability": float(np.random.rand()),
                "transition_probability": float(np.random.rand()),
                "embedding_pca": False
            },
            "network_metrics": {
                "centrality": float(np.random.rand()),
                "bridge_score": float(np.random.rand()),
                "community_ids": [np.random.randint(1, 10) for _ in range(3)],
                "weak_ties_count": np.random.randint(50, 250)
            },
            "serendipity_factors": {
                "uniqueness": float(np.random.rand()),
                "timing_score": float(np.random.rand()),
                "complementarity": {
                    "offers": ["expertise"],
                    "seeks": ["opportunities"]
                }
            }
        }


def batch_fit_pca(db, collection_name="public_profiles", sample_size=1000):
    """
    Fit PCA on a sample of embeddings from the database
    
    Args:
        db: MongoDB database object
        collection_name: Name of collection
        sample_size: Number of profiles to sample for PCA fitting
        
    Returns:
        Fitted QuantumFeatureGeneratorV2
    """
    print(f"Sampling {sample_size} profiles for PCA fitting...")
    
    # Get profiles with embeddings
    profiles = list(db[collection_name].find(
        {"embedding": {"$exists": True}},
        {"embedding": 1}
    ).limit(sample_size))
    
    if len(profiles) == 0:
        raise ValueError("No profiles with embeddings found!")
    
    # Extract embeddings
    embeddings = []
    for p in profiles:
        if p.get('embedding') and len(p['embedding']) == 1536:
            embeddings.append(p['embedding'])
    
    embeddings = np.array(embeddings)
    print(f"Collected {len(embeddings)} embeddings")
    
    # Fit PCA
    generator = QuantumFeatureGeneratorV2(n_quantum_dims=8)
    generator.fit_pca(embeddings)
    
    return generator
"""
Optimized Fast Scoring Engine
High-performance vectorized scoring with advanced indexing and caching.
Targets 3x speedup over original implementation.
"""

import numpy as np
import numba
from typing import Dict, List, Tuple, Optional
import faiss
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for optimized scoring."""
    batch_size: int = 1024
    use_gpu: bool = True
    n_threads: int = 4
    cache_size: int = 10000
    precompute_norms: bool = True
    use_numba: bool = True


class OptimizedVectorCache:
    """High-performance vector cache with LRU eviction."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get vector from cache."""
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            self.hit_count += 1
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def put(self, key: str, value: np.ndarray):
        """Store vector in cache."""
        if key in self.cache:
            # Update existing
            self.access_order.remove(key)
            self.access_order.append(key)
            self.cache[key] = value
            return
        
        # Add new entry
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        return {
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }


# Numba-accelerated functions for high-performance computation
@numba.jit(nopython=True, parallel=True)
def batch_cosine_similarity_numba(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Numba-accelerated batch cosine similarity computation."""
    n_vectors = vectors.shape[0]
    similarities = np.empty(n_vectors, dtype=np.float32)
    
    # Precompute query norm
    query_norm = np.sqrt(np.sum(query * query))
    
    for i in numba.prange(n_vectors):
        # Compute dot product
        dot_product = 0.0
        for j in range(len(query)):
            dot_product += query[j] * vectors[i, j]
        
        # Compute vector norm
        vector_norm = 0.0
        for j in range(vectors.shape[1]):
            vector_norm += vectors[i, j] * vectors[i, j]
        vector_norm = np.sqrt(vector_norm)
        
        # Compute cosine similarity
        if query_norm > 0 and vector_norm > 0:
            similarities[i] = dot_product / (query_norm * vector_norm)
        else:
            similarities[i] = 0.0
    
    return similarities


@numba.jit(nopython=True, parallel=True)
def vectorized_scoring_numba(
    cos_similarities: np.ndarray,
    trust_scores: np.ndarray,
    availability_scores: np.ndarray,
    novelty_scores: np.ndarray,
    random_values: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """Numba-accelerated vectorized scoring computation."""
    n_candidates = len(cos_similarities)
    scores = np.empty(n_candidates, dtype=np.float32)
    
    # Unpack weights for better performance
    w_fit = weights[0]
    w_trust = weights[1]
    w_avail = weights[2]
    w_div = weights[3]
    w_ser = weights[4]
    
    for i in numba.prange(n_candidates):
        # Base score
        base_score = (
            w_fit * max(0.0, cos_similarities[i]) +
            w_trust * trust_scores[i] +
            w_avail * availability_scores[i] +
            w_div * novelty_scores[i]
        )
        
        # Serendipity component
        serendipity = w_ser * (0.5 * novelty_scores[i] + 0.5 * random_values[i])
        
        scores[i] = base_score + serendipity
    
    return scores


class OptimizedFAISSEngine:
    """
    Ultra-high-performance FAISS engine with GPU acceleration and smart indexing.
    """
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.index = None
        self.dimension = None
        self.is_trained = False
        self.gpu_resources = None
        
        # Initialize GPU resources if available
        if config.use_gpu and faiss.get_num_gpus() > 0:
            self.gpu_resources = faiss.StandardGpuResources()
            logger.info(f"Initialized GPU resources: {faiss.get_num_gpus()} GPUs available")
    
    def build_hierarchical_index(self, vectors: np.ndarray, nlist: int = None):
        """Build hierarchical index with optimal parameters."""
        n_vectors, dimension = vectors.shape
        self.dimension = dimension
        
        # Auto-tune parameters based on dataset size
        if nlist is None:
            nlist = min(4096, max(64, int(np.sqrt(n_vectors))))
        
        logger.info(f"Building hierarchical index for {n_vectors} vectors, dim={dimension}")
        logger.info(f"Using nlist={nlist}")
        
        # Choose index type based on dataset characteristics
        if n_vectors < 1000:
            # Small dataset: exact search
            index = faiss.IndexFlatIP(dimension)
            logger.info("Using exact search (IndexFlatIP)")
        
        elif n_vectors < 100000:
            # Medium dataset: IVF with flat quantizer
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            logger.info("Using IVF with flat quantizer")
        
        else:
            # Large dataset: IVF with PQ compression
            quantizer = faiss.IndexFlatIP(dimension)
            m = 8  # Number of subquantizers
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            logger.info("Using IVF with PQ compression")
        
        # Move to GPU if available
        if self.config.use_gpu and self.gpu_resources:
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
            logger.info("Moved index to GPU")
        
        # Train the index if needed
        if hasattr(index, 'train'):
            logger.info("Training index...")
            index.train(vectors.astype(np.float32))
            self.is_trained = True
        
        # Add vectors
        logger.info("Adding vectors to index...")
        index.add(vectors.astype(np.float32))
        
        self.index = index
        logger.info(f"Index built successfully: {self.index.ntotal} vectors")
    
    def batch_search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform batch search for multiple queries."""
        if self.index is None:
            raise ValueError("Index not built")
        
        # Ensure queries are 2D
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        # Search in batches for memory efficiency
        batch_size = self.config.batch_size
        n_queries = queries.shape[0]
        
        all_distances = []
        all_indices = []
        
        for i in range(0, n_queries, batch_size):
            batch_end = min(i + batch_size, n_queries)
            batch_queries = queries[i:batch_end]
            
            distances, indices = self.index.search(batch_queries.astype(np.float32), k)
            all_distances.append(distances)
            all_indices.append(indices)
        
        return np.vstack(all_distances), np.vstack(all_indices)
    
    def get_index_stats(self) -> Dict:
        """Get detailed index statistics."""
        if self.index is None:
            return {}
        
        stats = {
            'ntotal': self.index.ntotal,
            'dimension': self.dimension,
            'is_trained': self.is_trained,
            'on_gpu': hasattr(self.index, 'getDevice')
        }
        
        # Add index-specific stats
        if hasattr(self.index, 'nlist'):
            stats['nlist'] = self.index.nlist
        if hasattr(self.index, 'nprobe'):
            stats['nprobe'] = self.index.nprobe
        
        return stats


class OptimizedScorer:
    """
    Ultra-fast scoring engine with advanced optimizations.
    """
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.cache = OptimizedVectorCache(config.cache_size)
        self.precomputed_norms = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.n_threads)
        
        logger.info(f"Initialized OptimizedScorer with config: {config}")
    
    def precompute_norms(self, vectors: np.ndarray, key_prefix: str = ""):
        """Precompute vector norms for faster similarity computation."""
        if not self.config.precompute_norms:
            return
        
        logger.info("Precomputing vector norms...")
        norms = np.linalg.norm(vectors, axis=1)
        self.precomputed_norms[key_prefix] = norms
        logger.info(f"Precomputed {len(norms)} norms")
    
    def compute_similarities_optimized(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute similarities using the fastest available method."""
        
        # Try cache first
        query_key = f"query_{hash(query.tobytes())}"
        cached_result = self.cache.get(query_key)
        if cached_result is not None:
            return cached_result
        
        # Choose computation method
        if self.config.use_numba:
            # Numba-accelerated computation
            similarities = batch_cosine_similarity_numba(query, vectors)
        else:
            # NumPy computation
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                similarities = np.zeros(len(vectors))
            else:
                vector_norms = np.linalg.norm(vectors, axis=1)
                dots = vectors @ query
                similarities = dots / (query_norm * vector_norms + 1e-8)
        
        # Cache result
        self.cache.put(query_key, similarities)
        
        return similarities
    
    def score_batch_optimized(self, 
                             query: np.ndarray,
                             candidate_vectors: np.ndarray,
                             trust_scores: np.ndarray,
                             availability_scores: np.ndarray,
                             novelty_scores: np.ndarray,
                             weights: Dict[str, float],
                             rng: np.random.Generator = None) -> np.ndarray:
        """Optimized batch scoring with all enhancements."""
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Compute similarities
        cos_similarities = self.compute_similarities_optimized(query, candidate_vectors)
        
        # Generate random values for serendipity
        random_values = rng.random(len(candidate_vectors))
        
        # Prepare weights array for Numba
        weights_array = np.array([
            weights.get('fit', 0.35),
            weights.get('trust', 0.20),
            weights.get('avail', 0.20),
            weights.get('div', 0.10),
            weights.get('ser', 0.15)
        ], dtype=np.float32)
        
        # Compute scores
        if self.config.use_numba:
            scores = vectorized_scoring_numba(
                cos_similarities.astype(np.float32),
                trust_scores.astype(np.float32),
                availability_scores.astype(np.float32),
                novelty_scores.astype(np.float32),
                random_values.astype(np.float32),
                weights_array
            )
        else:
            # Fallback NumPy computation
            base_scores = (
                weights_array[0] * np.maximum(0.0, cos_similarities) +
                weights_array[1] * trust_scores +
                weights_array[2] * availability_scores +
                weights_array[3] * novelty_scores
            )
            serendipity = weights_array[4] * (0.5 * novelty_scores + 0.5 * random_values)
            scores = base_scores + serendipity
        
        return scores
    
    def parallel_score_batches(self, 
                              queries: List[np.ndarray],
                              candidate_batches: List[np.ndarray],
                              metadata_batches: List[Dict],
                              weights: Dict[str, float]) -> List[np.ndarray]:
        """Score multiple query-candidate batches in parallel."""
        
        def score_single_batch(args):
            query, candidates, metadata = args
            return self.score_batch_optimized(
                query, candidates,
                metadata['trust'], metadata['availability'], metadata['novelty'],
                weights
            )
        
        # Submit all tasks
        futures = []
        for query, candidates, metadata in zip(queries, candidate_batches, metadata_batches):
            future = self.executor.submit(score_single_batch, (query, candidates, metadata))
            futures.append(future)
        
        # Collect results
        results = [future.result() for future in futures]
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics."""
        cache_stats = self.cache.get_stats()
        
        stats = {
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_size': cache_stats['size'],
            'cache_max_size': cache_stats['max_size'],
            'precomputed_norms': len(self.precomputed_norms),
            'use_numba': self.config.use_numba,
            'use_gpu': self.config.use_gpu,
            'n_threads': self.config.n_threads
        }
        
        return stats


class OptimizedSerendipityEngine:
    """
    Complete optimized serendipity engine integrating all performance enhancements.
    """
    
    def __init__(self, config: ScoringConfig = None):
        self.config = config or ScoringConfig()
        self.faiss_engine = OptimizedFAISSEngine(self.config)
        self.scorer = OptimizedScorer(self.config)
        
        self.item_vectors = None
        self.item_metadata = None
        
        logger.info("Initialized OptimizedSerendipityEngine")
    
    def fit(self, item_vectors: np.ndarray, item_metadata: Dict[str, np.ndarray]):
        """Fit the engine with item data."""
        logger.info(f"Fitting engine with {len(item_vectors)} items")
        
        # Store data
        self.item_vectors = item_vectors
        self.item_metadata = item_metadata
        
        # Build optimized index
        self.faiss_engine.build_hierarchical_index(item_vectors)
        
        # Precompute norms
        self.scorer.precompute_norms(item_vectors, "items")
        
        logger.info("Engine fitted successfully")
    
    def recommend(self, user_vector: np.ndarray, 
                  intent: str = "default",
                  k: int = 10,
                  prefilter_k: int = None,
                  serendipity_scale: float = 1.0) -> List[Tuple[int, float]]:
        """Generate optimized recommendations."""
        
        if self.item_vectors is None:
            raise ValueError("Engine not fitted")
        
        # Auto-tune prefilter size
        if prefilter_k is None:
            prefilter_k = min(k * 20, len(self.item_vectors))
        
        logger.debug(f"Generating recommendations: k={k}, prefilter_k={prefilter_k}")
        
        # Stage 1: Fast FAISS prefiltering
        distances, indices = self.faiss_engine.index.search(
            user_vector.astype(np.float32).reshape(1, -1), 
            prefilter_k
        )
        
        candidate_indices = indices[0]
        candidate_vectors = self.item_vectors[candidate_indices]
        
        # Stage 2: Detailed scoring
        weights = self._get_intent_weights(intent, serendipity_scale)
        
        candidate_metadata = {
            'trust': self.item_metadata['trust'][candidate_indices],
            'availability': self.item_metadata['availability'][candidate_indices],
            'novelty': self.item_metadata['novelty'][candidate_indices]
        }
        
        scores = self.scorer.score_batch_optimized(
            user_vector, candidate_vectors,
            candidate_metadata['trust'],
            candidate_metadata['availability'],
            candidate_metadata['novelty'],
            weights
        )
        
        # Stage 3: Final ranking
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        recommendations = [
            (candidate_indices[idx], scores[idx])
            for idx in top_k_indices
        ]
        
        return recommendations
    
    def batch_recommend(self, user_vectors: np.ndarray, 
                       intent: str = "default",
                       k: int = 10) -> List[List[Tuple[int, float]]]:
        """Generate recommendations for multiple users efficiently."""
        
        logger.info(f"Batch recommend for {len(user_vectors)} users")
        
        # Batch FAISS search
        prefilter_k = min(k * 20, len(self.item_vectors))
        all_distances, all_indices = self.faiss_engine.batch_search(user_vectors, prefilter_k)
        
        # Prepare batches for parallel scoring
        queries = []
        candidate_batches = []
        metadata_batches = []
        
        for i, (distances, indices) in enumerate(zip(all_distances, all_indices)):
            candidate_vectors = self.item_vectors[indices]
            candidate_metadata = {
                'trust': self.item_metadata['trust'][indices],
                'availability': self.item_metadata['availability'][indices],
                'novelty': self.item_metadata['novelty'][indices]
            }
            
            queries.append(user_vectors[i])
            candidate_batches.append(candidate_vectors)
            metadata_batches.append(candidate_metadata)
        
        # Parallel scoring
        weights = self._get_intent_weights(intent)
        all_scores = self.scorer.parallel_score_batches(
            queries, candidate_batches, metadata_batches, weights
        )
        
        # Generate final recommendations
        recommendations = []
        for i, (scores, indices) in enumerate(zip(all_scores, all_indices)):
            top_k_indices = np.argsort(scores)[::-1][:k]
            user_recommendations = [
                (indices[idx], scores[idx])
                for idx in top_k_indices
            ]
            recommendations.append(user_recommendations)
        
        return recommendations
    
    def _get_intent_weights(self, intent: str, serendipity_scale: float = 1.0) -> Dict[str, float]:
        """Get scoring weights for different intents."""
        base_weights = {
            "deal": {"fit": 0.35, "trust": 0.35, "avail": 0.20, "div": 0.05, "ser": 0.05},
            "ship": {"fit": 0.25, "trust": 0.10, "avail": 0.25, "div": 0.10, "ser": 0.30},
            "friend": {"fit": 0.40, "trust": 0.15, "avail": 0.20, "div": 0.10, "ser": 0.15},
            "mentor": {"fit": 0.30, "trust": 0.30, "avail": 0.15, "div": 0.10, "ser": 0.15},
            "default": {"fit": 0.35, "trust": 0.20, "avail": 0.20, "div": 0.10, "ser": 0.15}
        }
        
        weights = base_weights.get(intent, base_weights["default"]).copy()
        weights["ser"] = max(0.0, min(1.5, weights["ser"] * serendipity_scale))
        
        return weights
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        stats = {
            'engine': 'OptimizedSerendipityEngine',
            'config': self.config.__dict__,
            'faiss_stats': self.faiss_engine.get_index_stats(),
            'scorer_stats': self.scorer.get_performance_stats()
        }
        
        if self.item_vectors is not None:
            stats['data_stats'] = {
                'n_items': len(self.item_vectors),
                'vector_dimension': self.item_vectors.shape[1],
                'memory_usage_mb': self.item_vectors.nbytes / (1024**2)
            }
        
        return stats


# Example usage and benchmarking
if __name__ == "__main__":
    import time
    
    # Create test data
    logger.info("Creating test data...")
    n_items = 50000
    n_users = 1000
    dimension = 128
    
    np.random.seed(42)
    item_vectors = np.random.randn(n_items, dimension).astype(np.float32)
    item_vectors = item_vectors / np.linalg.norm(item_vectors, axis=1, keepdims=True)
    
    item_metadata = {
        'trust': np.random.uniform(0, 1, n_items),
        'availability': np.random.uniform(0, 1, n_items),
        'novelty': np.random.uniform(0, 1, n_items)
    }
    
    user_vectors = np.random.randn(n_users, dimension).astype(np.float32)
    user_vectors = user_vectors / np.linalg.norm(user_vectors, axis=1, keepdims=True)
    
    # Initialize and benchmark
    config = ScoringConfig(use_gpu=False, use_numba=True)  # CPU for testing
    engine = OptimizedSerendipityEngine(config)
    
    # Fit
    start_time = time.time()
    engine.fit(item_vectors, item_metadata)
    fit_time = time.time() - start_time
    logger.info(f"Fit time: {fit_time:.2f} seconds")
    
    # Single recommendation benchmark
    test_user = user_vectors[0]
    start_time = time.time()
    recommendations = engine.recommend(test_user, k=10)
    single_rec_time = time.time() - start_time
    logger.info(f"Single recommendation time: {single_rec_time*1000:.2f} ms")
    
    # Batch recommendation benchmark
    test_users = user_vectors[:100]
    start_time = time.time()
    batch_recommendations = engine.batch_recommend(test_users, k=10)
    batch_time = time.time() - start_time
    per_user_time = batch_time / len(test_users)
    logger.info(f"Batch recommendation time: {batch_time:.2f} seconds ({per_user_time*1000:.2f} ms per user)")
    
    # Performance stats
    stats = engine.get_comprehensive_stats()
    logger.info(f"Performance stats: {stats}")
    
    # Show sample recommendations
    logger.info("Sample recommendations:")
    for i, (item_id, score) in enumerate(recommendations):
        logger.info(f"  {i+1}. Item {item_id}: {score:.4f}")
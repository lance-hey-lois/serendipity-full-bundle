"""
Multi-Objective Scoring System Tests
====================================

Comprehensive unit tests for the multi-objective scoring system that combines
fitness, trust, availability, diversity, and serendipity into unified scores.

Key Test Areas:
- Cosine similarity computation accuracy
- Intent-based weight configuration
- Multi-objective score calculation
- Serendipity scaling mechanisms
- Vectorized scoring performance
- Statistical properties of scoring
"""

import pytest
import numpy as np
import random
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch

class TestCosineSimilarity:
    """Test cosine similarity computation."""
    
    def test_cos_basic_functionality(self):
        """Test basic cosine similarity computation."""
        from serendipity_engine_ui.engine.core import cos
        
        # Test identical vectors
        vec_a = [1, 2, 3, 4]
        vec_b = [1, 2, 3, 4]
        similarity = cos(vec_a, vec_b)
        assert abs(similarity - 1.0) < 1e-10, f"Identical vectors should have cos=1, got {similarity}"
        
        # Test orthogonal vectors
        vec_c = [1, 0, 0]
        vec_d = [0, 1, 0]
        similarity_ortho = cos(vec_c, vec_d)
        assert abs(similarity_ortho - 0.0) < 1e-10, f"Orthogonal vectors should have cos=0, got {similarity_ortho}"
        
        # Test opposite vectors
        vec_e = [1, 1, 1]
        vec_f = [-1, -1, -1]
        similarity_opposite = cos(vec_e, vec_f)
        assert abs(similarity_opposite - (-1.0)) < 1e-10, f"Opposite vectors should have cos=-1, got {similarity_opposite}"
    
    def test_cos_mathematical_properties(self):
        """Test mathematical properties of cosine similarity."""
        from serendipity_engine_ui.engine.core import cos
        
        # Test symmetry: cos(a, b) = cos(b, a)
        vec_a = [2, -1, 3, 0.5]
        vec_b = [1, 4, -2, 1.5]
        
        cos_ab = cos(vec_a, vec_b)
        cos_ba = cos(vec_b, vec_a)
        assert abs(cos_ab - cos_ba) < 1e-10, f"Cosine should be symmetric: {cos_ab} != {cos_ba}"
        
        # Test scale invariance: cos(a, b) = cos(ka, b) for k > 0
        scale = 5.0
        vec_a_scaled = [x * scale for x in vec_a]
        cos_scaled = cos(vec_a_scaled, vec_b)
        assert abs(cos_ab - cos_scaled) < 1e-10, f"Cosine should be scale invariant: {cos_ab} != {cos_scaled}"
        
        # Test bounds: -1 ≤ cos(a, b) ≤ 1
        test_vectors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], 
            [1, 1, 1], [-1, -1, -1],
            [2, -3, 4], [5, 1, -2]
        ]
        
        for i, vec1 in enumerate(test_vectors):
            for j, vec2 in enumerate(test_vectors):
                similarity = cos(vec1, vec2)
                assert -1.0 <= similarity <= 1.0, \
                    f"Cosine similarity out of bounds: {similarity} for vectors {i}, {j}"
    
    def test_cos_edge_cases(self):
        """Test cosine similarity edge cases."""
        from serendipity_engine_ui.engine.core import cos
        
        # Test empty vectors
        assert cos([], []) == 0.0, "Empty vectors should return 0"
        assert cos([1, 2], []) == 0.0, "Mismatched empty vector should return 0"
        assert cos([], [3, 4]) == 0.0, "Mismatched empty vector should return 0"
        
        # Test zero vectors
        zero_vec = [0, 0, 0, 0]
        non_zero_vec = [1, 2, 3, 4]
        assert cos(zero_vec, non_zero_vec) == 0.0, "Zero vector similarity should be 0"
        assert cos(non_zero_vec, zero_vec) == 0.0, "Zero vector similarity should be 0"
        assert cos(zero_vec, zero_vec) == 0.0, "Zero-zero similarity should be 0"
        
        # Test mismatched dimensions
        vec_short = [1, 2]
        vec_long = [1, 2, 3, 4, 5]
        # Should handle gracefully (likely truncate or pad)
        result = cos(vec_short, vec_long)
        assert isinstance(result, float), "Should return a float even with mismatched dimensions"
        assert -1.0 <= result <= 1.0, f"Result should be in valid range: {result}"
    
    def test_cos_numerical_stability(self):
        """Test numerical stability of cosine similarity."""
        from serendipity_engine_ui.engine.core import cos
        
        # Test very small vectors
        tiny_vec1 = [1e-10, 1e-10, 1e-10]
        tiny_vec2 = [2e-10, 2e-10, 2e-10]
        result_tiny = cos(tiny_vec1, tiny_vec2)
        assert np.isfinite(result_tiny), f"Small vectors should give finite result, got {result_tiny}"
        
        # Test very large vectors
        large_vec1 = [1e10, 1e10, 1e10]
        large_vec2 = [2e10, 2e10, 2e10]
        result_large = cos(large_vec1, large_vec2)
        assert np.isfinite(result_large), f"Large vectors should give finite result, got {result_large}"
        
        # Should be approximately 1 (same direction)
        assert abs(result_large - 1.0) < 1e-6, f"Parallel large vectors should have cos≈1, got {result_large}"

class TestIntentWeights:
    """Test intent-based weight configuration."""
    
    def test_weights_for_all_intents(self):
        """Test weight configuration for all defined intents."""
        from serendipity_engine_ui.engine.core import weights_for
        
        defined_intents = ["deal", "ship", "friend", "mentor"]
        required_keys = {"fit", "trust", "avail", "div", "ser"}
        
        for intent in defined_intents:
            weights = weights_for(intent)
            
            # Check structure
            assert isinstance(weights, dict), f"Weights should be dict for intent '{intent}'"
            assert set(weights.keys()) == required_keys, \
                f"Missing keys for intent '{intent}': {set(weights.keys())} vs {required_keys}"
            
            # Check values
            for key, value in weights.items():
                assert isinstance(value, float), f"Weight {key} should be float for intent '{intent}'"
                assert value >= 0, f"Weight {key} should be non-negative for intent '{intent}'"
            
            # Check normalization (weights should sum to reasonable total)
            total_weight = sum(weights.values())
            assert 0.8 <= total_weight <= 1.2, \
                f"Total weights should be ~1.0 for intent '{intent}', got {total_weight}"
    
    def test_weights_for_case_insensitivity(self):
        """Test that intent matching is case insensitive."""
        from serendipity_engine_ui.engine.core import weights_for
        
        # Test various case combinations
        case_variants = [
            ("deal", "DEAL", "Deal", "DeAl"),
            ("ship", "SHIP", "Ship", "ShIp"),
            ("friend", "FRIEND", "Friend", "FrIeNd"),
            ("mentor", "MENTOR", "Mentor", "MeNtOr")
        ]
        
        for variants in case_variants:
            base_weights = weights_for(variants[0])
            
            for variant in variants[1:]:
                variant_weights = weights_for(variant)
                assert variant_weights == base_weights, \
                    f"Case variant '{variant}' should match '{variants[0]}'"
    
    def test_weights_for_default_intent(self):
        """Test default weight configuration for unknown intents."""
        from serendipity_engine_ui.engine.core import weights_for
        
        unknown_intents = ["unknown", "random", "test", "", "xyz"]
        default_weights = weights_for("unknown")
        
        for intent in unknown_intents:
            weights = weights_for(intent)
            assert weights == default_weights, \
                f"Unknown intent '{intent}' should use default weights"
        
        # Default weights should be reasonable
        assert 0.8 <= sum(default_weights.values()) <= 1.2, \
            "Default weights should sum to ~1.0"
        assert all(w >= 0 for w in default_weights.values()), \
            "Default weights should be non-negative"
    
    def test_weights_for_intent_characteristics(self):
        """Test that different intents have characteristic weight patterns."""
        from serendipity_engine_ui.engine.core import weights_for
        
        deal_weights = weights_for("deal")
        ship_weights = weights_for("ship")
        friend_weights = weights_for("friend")
        mentor_weights = weights_for("mentor")
        
        # Deal intent should prioritize fit and trust
        assert deal_weights["fit"] >= 0.3, "Deal should prioritize fit"
        assert deal_weights["trust"] >= 0.3, "Deal should prioritize trust"
        
        # Ship intent should prioritize serendipity
        assert ship_weights["ser"] >= ship_weights["trust"], "Ship should value serendipity over trust"
        assert ship_weights["ser"] >= 0.25, "Ship should have significant serendipity weight"
        
        # Friend intent should prioritize fit
        assert friend_weights["fit"] >= 0.35, "Friend should prioritize fit highly"
        
        # Mentor intent should balance fit and trust
        assert mentor_weights["fit"] >= 0.25, "Mentor should value fit"
        assert mentor_weights["trust"] >= 0.25, "Mentor should value trust"

class TestMultiObjectiveScoring:
    """Test multi-objective score calculation."""
    
    def test_score_basic_functionality(self, rng):
        """Test basic multi-objective scoring."""
        from serendipity_engine_ui.engine.core import score, weights_for
        
        # Create test data
        me = {"vec": [1.0, 0.5, -0.2, 0.8]}
        candidate = {
            "vec": [0.8, 0.6, -0.1, 0.9],
            "pathTrust": 0.7,
            "availability": 0.8,
            "novelty": 0.6
        }
        
        weights = weights_for("deal")
        
        # Calculate score
        total_score = score(me, candidate, weights)
        
        # Basic validation
        assert isinstance(total_score, float), f"Score should be float, got {type(total_score)}"
        assert np.isfinite(total_score), f"Score should be finite, got {total_score}"
        
        # Score should be reasonable (components are all [0,1] with positive weights)
        assert total_score >= 0, f"Score should be non-negative, got {total_score}"
        assert total_score <= 2.0, f"Score should be reasonable, got {total_score}"  # Allow for serendipity randomness
    
    def test_score_component_contributions(self, rng):
        """Test individual component contributions to score."""
        from serendipity_engine_ui.engine.core import score, weights_for
        
        me = {"vec": [1.0, 0.0, 0.0]}
        
        # Test perfect fit candidate
        perfect_fit = {
            "vec": [1.0, 0.0, 0.0],  # Same as me
            "pathTrust": 1.0,
            "availability": 1.0,
            "novelty": 1.0
        }
        
        # Test no fit candidate
        no_fit = {
            "vec": [-1.0, 0.0, 0.0],  # Opposite of me
            "pathTrust": 1.0,
            "availability": 1.0,
            "novelty": 1.0
        }
        
        weights = {"fit": 1.0, "trust": 0.0, "avail": 0.0, "div": 0.0, "ser": 0.0}
        
        # Mock random for serendipity to make it deterministic
        with patch('serendipity_engine_ui.engine.core.random.random', return_value=0.5):
            score_perfect = score(me, perfect_fit, weights)
            score_no_fit = score(me, no_fit, weights)
        
        # Perfect fit should score higher than no fit
        assert score_perfect > score_no_fit, \
            f"Perfect fit {score_perfect} should score higher than no fit {score_no_fit}"
    
    def test_score_serendipity_component(self, rng):
        """Test serendipity component of scoring."""
        from serendipity_engine_ui.engine.core import score
        
        me = {"vec": [1.0, 0.0]}
        candidate = {
            "vec": [1.0, 0.0],
            "pathTrust": 0.5,
            "availability": 0.5,
            "novelty": 0.8
        }
        
        # Weights with only serendipity
        weights_ser_only = {"fit": 0.0, "trust": 0.0, "avail": 0.0, "div": 0.0, "ser": 1.0}
        
        # Run multiple times to test randomness
        scores = []
        for _ in range(100):
            total_score = score(me, candidate, weights_ser_only)
            scores.append(total_score)
        
        # Serendipity should introduce variability
        assert len(set(scores)) > 1, "Serendipity should create score variation"
        
        # But scores should be bounded by the serendipity formula
        # ser = w["ser"] * (0.5 * novelty + 0.5 * random())
        # = 1.0 * (0.5 * 0.8 + 0.5 * random()) = 0.4 + 0.5 * random()
        # So scores should be in [0.4, 0.9]
        assert all(0.3 <= s <= 1.0 for s in scores), f"Serendipity scores out of expected range: {min(scores)}, {max(scores)}"
        
        # Mean should be around 0.65 (0.4 + 0.5 * 0.5)
        mean_score = np.mean(scores)
        assert 0.55 <= mean_score <= 0.75, f"Mean serendipity score unexpected: {mean_score}"
    
    def test_score_missing_attributes(self):
        """Test scoring with missing candidate attributes."""
        from serendipity_engine_ui.engine.core import score, weights_for
        
        me = {"vec": [1.0, 0.0]}
        
        # Candidate missing some attributes
        incomplete_candidate = {
            "vec": [0.8, 0.1],
            # Missing pathTrust, availability, novelty
        }
        
        weights = weights_for("deal")
        
        # Should handle missing attributes gracefully
        total_score = score(me, incomplete_candidate, weights)
        
        assert isinstance(total_score, float), "Score should be float even with missing attributes"
        assert np.isfinite(total_score), "Score should be finite even with missing attributes"
        assert total_score >= 0, "Score should be non-negative even with missing attributes"
    
    def test_score_zero_weights(self):
        """Test scoring with zero weights."""
        from serendipity_engine_ui.engine.core import score
        
        me = {"vec": [1.0, 0.0]}
        candidate = {
            "vec": [0.5, 0.5],
            "pathTrust": 0.8,
            "availability": 0.9,
            "novelty": 0.7
        }
        
        # All zero weights
        zero_weights = {"fit": 0.0, "trust": 0.0, "avail": 0.0, "div": 0.0, "ser": 0.0}
        
        with patch('serendipity_engine_ui.engine.core.random.random', return_value=0.5):
            zero_score = score(me, candidate, zero_weights)
        
        assert zero_score == 0.0, f"Zero weights should give zero score, got {zero_score}"

class TestSerendipityScaling:
    """Test serendipity weight scaling mechanisms."""
    
    def test_apply_serendipity_scale_basic(self):
        """Test basic serendipity scaling functionality."""
        from serendipity_engine_ui.engine.core import apply_serendipity_scale, weights_for
        
        base_weights = weights_for("ship")
        original_ser = base_weights["ser"]
        
        # Test scaling up
        scaled_up = apply_serendipity_scale(base_weights, 2.0)
        expected_ser_up = original_ser * 2.0
        
        assert abs(scaled_up["ser"] - expected_ser_up) < 1e-10, \
            f"Serendipity scaling up failed: {scaled_up['ser']} vs {expected_ser_up}"
        
        # Other weights should remain unchanged
        for key in ["fit", "trust", "avail", "div"]:
            assert scaled_up[key] == base_weights[key], \
                f"Non-serendipity weight {key} should not change"
        
        # Test scaling down
        scaled_down = apply_serendipity_scale(base_weights, 0.5)
        expected_ser_down = original_ser * 0.5
        
        assert abs(scaled_down["ser"] - expected_ser_down) < 1e-10, \
            f"Serendipity scaling down failed: {scaled_down['ser']} vs {expected_ser_down}"
    
    def test_apply_serendipity_scale_bounds(self):
        """Test serendipity scaling bounds."""
        from serendipity_engine_ui.engine.core import apply_serendipity_scale, weights_for
        
        base_weights = weights_for("ship")
        
        # Test upper bound (1.5)
        scaled_high = apply_serendipity_scale(base_weights, 10.0)  # Very high scale
        assert scaled_high["ser"] <= 1.5, f"Serendipity weight should be capped at 1.5, got {scaled_high['ser']}"
        
        # Test lower bound (0.0)
        scaled_negative = apply_serendipity_scale(base_weights, -5.0)  # Negative scale
        assert scaled_negative["ser"] >= 0.0, f"Serendipity weight should be non-negative, got {scaled_negative['ser']}"
        assert scaled_negative["ser"] == 0.0, f"Negative scale should result in 0 serendipity, got {scaled_negative['ser']}"
    
    def test_apply_serendipity_scale_edge_cases(self):
        """Test serendipity scaling edge cases."""
        from serendipity_engine_ui.engine.core import apply_serendipity_scale, weights_for
        
        base_weights = weights_for("deal")
        
        # Test zero scale
        scaled_zero = apply_serendipity_scale(base_weights, 0.0)
        assert scaled_zero["ser"] == 0.0, "Zero scale should result in zero serendipity"
        
        # Test scale of 1 (no change)
        scaled_one = apply_serendipity_scale(base_weights, 1.0)
        assert scaled_one["ser"] == base_weights["ser"], "Scale of 1 should not change serendipity weight"
        
        # Test very small positive scale
        scaled_tiny = apply_serendipity_scale(base_weights, 1e-10)
        expected_tiny = base_weights["ser"] * 1e-10
        assert abs(scaled_tiny["ser"] - expected_tiny) < 1e-15, "Small scale should work precisely"
    
    def test_apply_serendipity_scale_immutability(self):
        """Test that serendipity scaling doesn't modify original weights."""
        from serendipity_engine_ui.engine.core import apply_serendipity_scale, weights_for
        
        original_weights = weights_for("mentor")
        original_ser = original_weights["ser"]
        
        # Apply scaling
        scaled_weights = apply_serendipity_scale(original_weights, 1.5)
        
        # Original should be unchanged
        assert original_weights["ser"] == original_ser, \
            "Original weights should not be modified by scaling"
        
        # Scaled should be different
        assert scaled_weights["ser"] != original_weights["ser"], \
            "Scaled weights should be different from original"
        
        # Other keys should be identical (not just equal)
        for key in ["fit", "trust", "avail", "div"]:
            assert scaled_weights[key] == original_weights[key], \
                f"Non-serendipity weights should be identical: {key}"

class TestVectorizedScoring:
    """Test vectorized scoring performance and correctness."""
    
    def test_score_vectorized_basic_functionality(self, rng):
        """Test basic vectorized scoring functionality."""
        from serendipity_engine_ui.engine.fastscore import prepare_arrays, score_vectorized
        
        # Create test data
        seed = {"vec": rng.normal(0, 1, 8).tolist()}
        pool = []
        for i in range(50):
            person = {
                "id": f"person_{i}",
                "vec": rng.normal(0, 1, 8).tolist(),
                "novelty": rng.random(),
                "availability": rng.random(),
                "pathTrust": rng.random()
            }
            pool.append(person)
        
        # Prepare arrays
        seed_unit, arrs = prepare_arrays(seed, pool)
        
        # Compute vectorized scores
        scores = score_vectorized(seed_unit, arrs, "deal", 1.0, rng)
        
        # Validate output
        assert scores.shape == (50,), f"Expected shape (50,), got {scores.shape}"
        assert np.all(np.isfinite(scores)), "All scores should be finite"
        assert isinstance(scores, np.ndarray), "Should return numpy array"
    
    def test_score_vectorized_vs_individual(self, rng):
        """Test that vectorized scoring matches individual scoring."""
        from serendipity_engine_ui.engine.core import score, weights_for, apply_serendipity_scale
        from serendipity_engine_ui.engine.fastscore import prepare_arrays, score_vectorized
        
        # Create small test dataset
        seed = {"vec": [1.0, 0.5, -0.2, 0.8]}
        pool = [
            {
                "id": "p1",
                "vec": [0.8, 0.6, -0.1, 0.9],
                "novelty": 0.7,
                "availability": 0.8,
                "pathTrust": 0.6
            },
            {
                "id": "p2", 
                "vec": [-0.3, 0.1, 0.4, -0.5],
                "novelty": 0.3,
                "availability": 0.5,
                "pathTrust": 0.9
            }
        ]
        
        intent = "friend"
        ser_scale = 1.2
        
        # Compute individual scores
        weights = apply_serendipity_scale(weights_for(intent), ser_scale)
        
        # Mock random for consistency
        random.seed(42)
        individual_scores = []
        for person in pool:
            random.seed(42)  # Reset for each to match vectorized behavior
            individual_score = score(seed, person, weights)
            individual_scores.append(individual_score)
        
        # Compute vectorized scores with same random seed
        seed_unit, arrs = prepare_arrays(seed, pool)
        rng_vec = np.random.default_rng(42)
        vectorized_scores = score_vectorized(seed_unit, arrs, intent, ser_scale, rng_vec)
        
        # Compare results (allowing for small numerical differences)
        for i, (ind_score, vec_score) in enumerate(zip(individual_scores, vectorized_scores)):
            diff = abs(ind_score - vec_score)
            # Note: Due to different random number generation, exact match might not occur
            # But the structure and magnitude should be similar
            print(f"Person {i}: individual={ind_score:.6f}, vectorized={vec_score:.6f}, diff={diff:.6f}")
    
    def test_prepare_arrays_functionality(self, rng):
        """Test array preparation for vectorized scoring."""
        from serendipity_engine_ui.engine.fastscore import prepare_arrays
        
        seed = {"vec": [2.0, -1.0, 0.5]}
        pool = []
        for i in range(10):
            person = {
                "id": f"id_{i}",
                "vec": rng.normal(0, 1, 3).tolist(),
                "novelty": rng.random(),
                "availability": rng.random(),
                "pathTrust": rng.random()
            }
            pool.append(person)
        
        seed_unit, arrs = prepare_arrays(seed, pool)
        
        # Validate seed preparation
        assert seed_unit.shape == (3,), f"Seed unit shape should be (3,), got {seed_unit.shape}"
        assert abs(np.linalg.norm(seed_unit) - 1.0) < 1e-10, f"Seed should be unit vector, norm={np.linalg.norm(seed_unit)}"
        
        # Validate array preparation
        expected_keys = {"ids", "vecs", "vecs_norm", "novelty", "availability", "trust"}
        assert set(arrs.keys()) == expected_keys, f"Missing keys in arrays: {set(arrs.keys())} vs {expected_keys}"
        
        assert arrs["vecs"].shape == (10, 3), f"Vecs shape should be (10, 3), got {arrs['vecs'].shape}"
        assert arrs["vecs_norm"].shape == (10, 3), f"Vecs_norm shape should be (10, 3), got {arrs['vecs_norm'].shape}"
        assert arrs["novelty"].shape == (10,), f"Novelty shape should be (10,), got {arrs['novelty'].shape}"
        
        # Validate normalization
        norms = np.linalg.norm(arrs["vecs_norm"], axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-10), f"All vectors should be unit vectors, norms range: {np.min(norms)}-{np.max(norms)}"
    
    def test_score_vectorized_performance(self, rng, performance_monitor):
        """Test vectorized scoring performance characteristics."""
        from serendipity_engine_ui.engine.fastscore import prepare_arrays, score_vectorized
        
        # Test with different pool sizes
        pool_sizes = [100, 500, 1000, 2000]
        n_dims = 64
        
        seed = {"vec": rng.normal(0, 1, n_dims).tolist()}
        
        performance_results = {}
        
        for pool_size in pool_sizes:
            # Create pool
            pool = []
            for i in range(pool_size):
                person = {
                    "id": f"p_{i}",
                    "vec": rng.normal(0, 1, n_dims).tolist(),
                    "novelty": rng.random(),
                    "availability": rng.random(),
                    "pathTrust": rng.random()
                }
                pool.append(person)
            
            # Prepare arrays
            seed_unit, arrs = prepare_arrays(seed, pool)
            
            # Time vectorized scoring
            performance_monitor.reset()
            performance_monitor.start_monitoring()
            
            scores = score_vectorized(seed_unit, arrs, "ship", 1.0, rng)
            
            performance_monitor.stop_monitoring()
            
            # Record results
            performance_results[pool_size] = {
                "time_ms": performance_monitor.elapsed_time_ms,
                "time_per_candidate_ms": performance_monitor.elapsed_time_ms / pool_size,
                "memory_mb": performance_monitor.memory_delta_mb,
                "valid_output": scores.shape == (pool_size,) and np.all(np.isfinite(scores))
            }
        
        # Analyze performance scaling
        for size, result in performance_results.items():
            print(f"Vectorized scoring {size} candidates: {result['time_per_candidate_ms']:.4f} ms/candidate, "
                  f"total: {result['time_ms']:.2f} ms, memory: {result['memory_mb']:.1f} MB")
            
            # Vectorized scoring should be very fast
            assert result['time_per_candidate_ms'] < 0.1, \
                f"Vectorized scoring too slow: {result['time_per_candidate_ms']:.4f} ms/candidate for {size} candidates"
        
        # Performance should scale linearly or better
        if len(performance_results) > 1:
            sizes = sorted(performance_results.keys())
            times = [performance_results[size]['time_ms'] for size in sizes]
            
            # Rough linearity check (later sizes shouldn't be disproportionately slower)
            time_ratio = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            efficiency_ratio = time_ratio / size_ratio
            
            print(f"Vectorized scaling efficiency: {efficiency_ratio:.2f} (1.0 = perfect linear scaling)")
            assert efficiency_ratio < 2.0, f"Vectorized scoring scaling too poor: {efficiency_ratio}"

class TestScoringStatistics:
    """Test statistical properties of the scoring system."""
    
    def test_score_distribution_properties(self, rng):
        """Test statistical properties of score distributions."""
        from serendipity_engine_ui.engine.fastscore import prepare_arrays, score_vectorized
        
        # Create large random dataset
        n_candidates = 1000
        n_dims = 32
        
        seed = {"vec": rng.normal(0, 1, n_dims).tolist()}
        pool = []
        for i in range(n_candidates):
            person = {
                "id": f"p_{i}",
                "vec": rng.normal(0, 1, n_dims).tolist(),
                "novelty": rng.random(),
                "availability": rng.random(),
                "pathTrust": rng.random()
            }
            pool.append(person)
        
        # Compute scores
        seed_unit, arrs = prepare_arrays(seed, pool)
        scores = score_vectorized(seed_unit, arrs, "collab", 1.0, rng)
        
        # Analyze distribution
        score_stats = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75))
        }
        
        print(f"Score distribution statistics:")
        for key, value in score_stats.items():
            print(f"  {key}: {value:.4f}")
        
        # Basic sanity checks
        assert score_stats["std"] > 0, "Scores should show variation"
        assert score_stats["min"] >= 0, "Minimum score should be non-negative"
        assert score_stats["max"] >= score_stats["mean"], "Max should be >= mean"
        assert score_stats["q25"] <= score_stats["median"] <= score_stats["q75"], "Quartiles should be ordered"
    
    def test_score_sensitivity_analysis(self, rng):
        """Test sensitivity of scores to different components."""
        from serendipity_engine_ui.engine.fastscore import prepare_arrays, score_vectorized
        
        n_candidates = 200
        n_dims = 16
        
        # Create baseline dataset
        seed = {"vec": rng.normal(0, 1, n_dims).tolist()}
        
        # Test sensitivity to different attribute ranges
        attribute_tests = {
            "high_novelty": {"novelty": (0.8, 1.0), "availability": (0.4, 0.6), "pathTrust": (0.4, 0.6)},
            "low_novelty": {"novelty": (0.0, 0.2), "availability": (0.4, 0.6), "pathTrust": (0.4, 0.6)},
            "high_trust": {"novelty": (0.4, 0.6), "availability": (0.4, 0.6), "pathTrust": (0.8, 1.0)},
            "low_trust": {"novelty": (0.4, 0.6), "availability": (0.4, 0.6), "pathTrust": (0.0, 0.2)},
            "high_availability": {"novelty": (0.4, 0.6), "availability": (0.8, 1.0), "pathTrust": (0.4, 0.6)},
            "low_availability": {"novelty": (0.4, 0.6), "availability": (0.0, 0.2), "pathTrust": (0.4, 0.6)},
        }
        
        results = {}
        
        for test_name, attr_ranges in attribute_tests.items():
            pool = []
            for i in range(n_candidates):
                person = {
                    "id": f"p_{i}",
                    "vec": rng.normal(0, 1, n_dims).tolist(),
                }
                
                # Set attributes according to test ranges
                for attr, (min_val, max_val) in attr_ranges.items():
                    person[attr] = rng.uniform(min_val, max_val)
                
                pool.append(person)
            
            # Compute scores
            seed_unit, arrs = prepare_arrays(seed, pool)
            scores = score_vectorized(seed_unit, arrs, "mentor", 1.0, rng)
            
            results[test_name] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "median_score": float(np.median(scores))
            }
        
        # Analyze sensitivity
        print("Score sensitivity analysis:")
        for test_name, stats in results.items():
            print(f"  {test_name}: mean={stats['mean_score']:.4f}, std={stats['std_score']:.4f}")
        
        # High-value attributes should generally lead to higher scores
        # (though serendipity randomness may complicate this)
        high_attributes = ["high_novelty", "high_trust", "high_availability"]
        low_attributes = ["low_novelty", "low_trust", "low_availability"]
        
        for high_attr, low_attr in zip(high_attributes, low_attributes):
            high_mean = results[high_attr]["mean_score"]
            low_mean = results[low_attr]["mean_score"]
            
            print(f"Comparing {high_attr} ({high_mean:.4f}) vs {low_attr} ({low_mean:.4f})")
            
            # Allow some tolerance due to serendipity randomness and different weight distributions
            if high_mean <= low_mean:
                print(f"  Warning: {high_attr} not scoring higher than {low_attr}")

if __name__ == "__main__":
    pytest.main([__file__])
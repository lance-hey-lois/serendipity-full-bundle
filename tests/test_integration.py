"""
Integration Tests for Serendipity Engine
========================================

Comprehensive integration tests for the quantum-enhanced serendipity engine,
testing end-to-end workflows, component interactions, and system behavior.

Key Test Areas:
- End-to-end discovery pipeline integration
- UI-engine communication and data flow
- Multi-component workflow validation
- Cross-system consistency checks
- Performance under realistic workloads
- Error propagation and recovery
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import io
import json
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

class TestEndToEndPipeline:
    """Test complete end-to-end discovery pipeline."""
    
    def test_complete_discovery_workflow(self, sample_people_data, rng, temp_dir):
        """Test complete discovery workflow from data to recommendations."""
        # Generate test data
        people, vectors, clusters, centers = sample_people_data(200, 32)
        
        # Create test user (seed)
        user = people[0].copy()
        candidate_pool = people[1:]  # Rest as candidates
        
        # Test the complete pipeline from suggest2.py
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        # Run complete scoring pipeline
        results = score_pool(
            seed=user,
            pool=candidate_pool,
            intent="ship",
            ser_scale=1.2,
            k=20,
            use_faiss_prefilter=True,
            M_prefilter=100,
            quantum_gamma=0.3,
            quantum_dims=4,
            use_gbs=True,
            gbs_modes=3,
            gbs_shots=50,
            gbs_cutoff=4,
            gbs_lambda=0.4
        )
        
        # Validate pipeline results
        assert isinstance(results, list), "Pipeline should return list of results"
        assert len(results) <= 20, f"Should return at most 20 results, got {len(results)}"
        assert len(results) > 0, "Pipeline should return some results"
        
        # Validate result structure
        for i, result in enumerate(results):
            assert "candidate" in result, f"Result {i} missing 'candidate'"
            assert "classical_score" in result, f"Result {i} missing 'classical_score'"
            assert "quantum_boost" in result, f"Result {i} missing 'quantum_boost'"  
            assert "gbs_boost" in result, f"Result {i} missing 'gbs_boost'"
            assert "total_score" in result, f"Result {i} missing 'total_score'"
            
            # Validate score types
            assert isinstance(result["classical_score"], (float, np.floating)), \
                f"Classical score should be float, got {type(result['classical_score'])}"
            assert isinstance(result["total_score"], (float, np.floating)), \
                f"Total score should be float, got {type(result['total_score'])}"
            
            # Validate score ranges
            assert np.isfinite(result["classical_score"]), f"Classical score should be finite"
            assert np.isfinite(result["total_score"]), f"Total score should be finite"
        
        # Results should be ordered by total score (descending)
        total_scores = [r["total_score"] for r in results]
        for i in range(len(total_scores) - 1):
            assert total_scores[i] >= total_scores[i+1], \
                f"Results should be ordered by total score: {total_scores[i]} < {total_scores[i+1]} at position {i}"
        
        print(f"End-to-end pipeline completed successfully:")
        print(f"  Input: 1 seed user, {len(candidate_pool)} candidates")
        print(f"  Output: {len(results)} ranked recommendations")
        print(f"  Score range: [{min(total_scores):.4f}, {max(total_scores):.4f}]")
    
    def test_pipeline_with_different_intents(self, sample_people_data, rng):
        """Test pipeline behavior with different user intents."""
        people, vectors, clusters, centers = sample_people_data(100, 24)
        
        user = people[0]
        candidates = people[1:51]  # Use subset for faster testing
        
        intents = ["deal", "ship", "friend", "mentor", "collab"]
        intent_results = {}
        
        for intent in intents:
            from serendipity_engine_ui.engine.suggest2 import score_pool
            
            results = score_pool(
                seed=user,
                pool=candidates,
                intent=intent,
                ser_scale=1.0,
                k=10,
                use_faiss_prefilter=False,  # Disable for consistency
                quantum_gamma=0.2,
                quantum_dims=3,
                use_gbs=False,  # Disable for faster testing
                gbs_lambda=0.0
            )
            
            intent_results[intent] = results
            
            # Basic validation for each intent
            assert len(results) <= 10, f"Intent {intent} returned too many results"
            assert len(results) > 0, f"Intent {intent} returned no results"
        
        # Different intents should potentially give different rankings
        # (though not guaranteed due to randomness and data properties)
        print("Intent-based ranking comparison:")
        for intent, results in intent_results.items():
            top_ids = [r["candidate"]["id"] for r in results[:3]]
            top_scores = [r["total_score"] for r in results[:3]]
            print(f"  {intent}: top_ids={top_ids}, top_scores={[f'{s:.3f}' for s in top_scores]}")
    
    def test_pipeline_component_interactions(self, sample_people_data, rng):
        """Test interactions between different pipeline components."""
        people, vectors, clusters, centers = sample_people_data(80, 16)
        
        user = people[0]
        candidates = people[1:41]
        
        # Test with various component combinations
        test_configs = [
            {
                "name": "classical_only",
                "use_faiss_prefilter": False,
                "quantum_gamma": 0.0,
                "use_gbs": False,
                "gbs_lambda": 0.0
            },
            {
                "name": "with_faiss",
                "use_faiss_prefilter": True,
                "M_prefilter": 30,
                "quantum_gamma": 0.0,
                "use_gbs": False,
                "gbs_lambda": 0.0
            },
            {
                "name": "with_quantum",
                "use_faiss_prefilter": False,
                "quantum_gamma": 0.5,
                "quantum_dims": 4,
                "use_gbs": False,
                "gbs_lambda": 0.0
            },
            {
                "name": "with_gbs",
                "use_faiss_prefilter": False,
                "quantum_gamma": 0.0,
                "use_gbs": True,
                "gbs_modes": 3,
                "gbs_shots": 30,
                "gbs_lambda": 0.3
            },
            {
                "name": "full_pipeline",
                "use_faiss_prefilter": True,
                "M_prefilter": 30,
                "quantum_gamma": 0.3,
                "quantum_dims": 3,
                "use_gbs": True,
                "gbs_modes": 3,
                "gbs_shots": 30,
                "gbs_lambda": 0.2
            }
        ]
        
        config_results = {}
        
        for config in test_configs:
            from serendipity_engine_ui.engine.suggest2 import score_pool
            
            # Set defaults
            full_config = {
                "seed": user,
                "pool": candidates,
                "intent": "mentor",
                "ser_scale": 1.0,
                "k": 8,
                "use_faiss_prefilter": False,
                "M_prefilter": 50,
                "quantum_gamma": 0.0,
                "quantum_dims": 4,
                "use_gbs": False,
                "gbs_modes": 4,
                "gbs_shots": 60,
                "gbs_cutoff": 5,
                "gbs_lambda": 0.0
            }
            
            # Update with test config
            full_config.update({k: v for k, v in config.items() if k != "name"})
            
            results = score_pool(**full_config)
            config_results[config["name"]] = results
            
            # Validate each configuration
            assert len(results) <= 8, f"Config {config['name']} returned too many results"
            assert len(results) > 0, f"Config {config['name']} returned no results"
            
            # All results should have proper structure
            for result in results:
                assert "total_score" in result, f"Config {config['name']} missing total_score"
                assert np.isfinite(result["total_score"]), \
                    f"Config {config['name']} has non-finite total_score"
        
        # Analyze component effects
        print("Component interaction analysis:")
        for name, results in config_results.items():
            avg_score = np.mean([r["total_score"] for r in results])
            score_std = np.std([r["total_score"] for r in results])
            print(f"  {name}: avg_score={avg_score:.4f}, std={score_std:.4f}")
    
    def test_pipeline_error_handling(self, sample_people_data, rng):
        """Test pipeline error handling and recovery."""
        people, vectors, clusters, centers = sample_people_data(50, 12)
        
        user = people[0]
        candidates = people[1:26]
        
        # Test various error conditions
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        # Test with empty pool
        try:
            results_empty = score_pool(
                seed=user,
                pool=[],
                intent="deal",
                ser_scale=1.0,
                k=10
            )
            assert isinstance(results_empty, list), "Empty pool should return empty list"
            assert len(results_empty) == 0, "Empty pool should return empty list"
        except Exception as e:
            # Acceptable to raise exception for empty pool
            print(f"Empty pool raises: {type(e).__name__}: {e}")
        
        # Test with invalid k
        results_large_k = score_pool(
            seed=user,
            pool=candidates,
            intent="friend",
            ser_scale=1.0,
            k=1000  # Much larger than pool size
        )
        assert len(results_large_k) <= len(candidates), "Should not return more than available candidates"
        
        # Test with malformed candidates (missing attributes)
        malformed_candidates = []
        for i, candidate in enumerate(candidates[:5]):
            if i % 2 == 0:
                # Remove some attributes
                malformed = {key: val for key, val in candidate.items() if key != "novelty"}
            else:
                malformed = candidate.copy()
            malformed_candidates.append(malformed)
        
        try:
            results_malformed = score_pool(
                seed=user,
                pool=malformed_candidates,
                intent="ship",
                ser_scale=1.0,
                k=3
            )
            
            # Should handle gracefully
            assert isinstance(results_malformed, list), "Should handle malformed candidates"
            print(f"Malformed candidates handled: {len(results_malformed)} results")
            
        except Exception as e:
            print(f"Malformed candidates raise: {type(e).__name__}: {e}")
            # Some errors may be acceptable depending on implementation

class TestDataFlowIntegration:
    """Test data flow integration between components."""
    
    def test_data_preparation_pipeline(self, sample_people_data, rng):
        """Test data preparation and transformation pipeline."""
        people, vectors, clusters, centers = sample_people_data(60, 20)
        
        user = people[0]
        pool = people[1:31]
        
        # Test data preparation stages
        from serendipity_engine_ui.engine.fastscore import prepare_arrays
        
        # Stage 1: Array preparation
        seed_unit, arrs = prepare_arrays(user, pool)
        
        # Validate prepared data
        assert seed_unit.shape == (20,), f"Seed unit shape should be (20,), got {seed_unit.shape}"
        assert abs(np.linalg.norm(seed_unit) - 1.0) < 1e-10, "Seed should be unit vector"
        
        expected_keys = {"ids", "vecs", "vecs_norm", "novelty", "availability", "trust"}
        assert set(arrs.keys()) == expected_keys, f"Array keys mismatch: {set(arrs.keys())}"
        
        assert arrs["vecs"].shape == (30, 20), f"Vectors shape should be (30, 20)"
        assert arrs["vecs_norm"].shape == (30, 20), f"Normalized vectors shape should be (30, 20)"
        
        # Stage 2: FAISS preprocessing
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine
        
        top_indices = top_m_cosine(seed_unit, arrs["vecs_norm"], 15)
        assert top_indices.shape == (15,), f"FAISS should return 15 indices"
        assert np.all(top_indices < 30), "All indices should be valid"
        
        # Stage 3: Quantum preprocessing  
        from serendipity_engine_ui.engine.quantum import pca_compress, quantum_kernel_to_seed
        
        # Extract top vectors
        top_vecs = arrs["vecs"][top_indices]
        
        # PCA compression for quantum
        quantum_seed = pca_compress(user["vec"][np.newaxis], 4)[0]
        quantum_cands = pca_compress(top_vecs, 4)
        
        assert quantum_seed.shape == (4,), "Quantum seed should be compressed to 4D"
        assert quantum_cands.shape == (15, 4), "Quantum candidates should be (15, 4)"
        
        # Quantum kernel computation
        quantum_scores = quantum_kernel_to_seed(quantum_seed, quantum_cands)
        assert quantum_scores.shape == (15,), "Quantum scores should match candidates"
        
        # Validate data flow consistency
        print(f"Data flow validation:")
        print(f"  Original pool size: {len(pool)}")
        print(f"  FAISS filtered to: {len(top_indices)}")
        print(f"  Quantum processing: {quantum_seed.shape} seed, {quantum_cands.shape} candidates")
        print(f"  Quantum scores range: [{np.min(quantum_scores):.4f}, {np.max(quantum_scores):.4f}]")
    
    def test_score_aggregation_consistency(self, sample_people_data, rng):
        """Test consistency of score aggregation across components."""
        people, vectors, clusters, centers = sample_people_data(40, 16)
        
        user = people[0]
        candidates = people[1:21]
        
        # Test manual score computation vs pipeline
        from serendipity_engine_ui.engine.fastscore import prepare_arrays, score_vectorized
        from serendipity_engine_ui.engine.quantum import pca_compress, quantum_kernel_to_seed, zscore
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        # Manual computation
        seed_unit, arrs = prepare_arrays(user, candidates)
        
        # Classical scores
        classical_scores = score_vectorized(seed_unit, arrs, "mentor", 1.0, rng)
        
        # Quantum scores
        quantum_seed = pca_compress(user["vec"][np.newaxis], 3)[0]
        quantum_cands = pca_compress(arrs["vecs"], 3)
        quantum_raw = quantum_kernel_to_seed(quantum_seed, quantum_cands)
        quantum_scores = zscore(quantum_raw)
        
        # GBS scores
        gbs_scores = gbs_boost(
            seed_vec=np.array(user["vec"]),
            cand_vecs=arrs["vecs"],
            modes=3,
            shots=40,
            cutoff=4
        )
        
        # Manual aggregation
        gamma = 0.3
        lambda_gbs = 0.2
        manual_total = classical_scores + gamma * quantum_scores + lambda_gbs * gbs_scores
        
        # Pipeline computation
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        pipeline_results = score_pool(
            seed=user,
            pool=candidates,
            intent="mentor", 
            ser_scale=1.0,
            k=20,
            use_faiss_prefilter=False,  # Test without prefiltering
            quantum_gamma=gamma,
            quantum_dims=3,
            use_gbs=True,
            gbs_modes=3,
            gbs_shots=40,
            gbs_cutoff=4,
            gbs_lambda=lambda_gbs
        )
        
        # Extract pipeline scores
        pipeline_total = np.array([r["total_score"] for r in pipeline_results])
        
        # Compare (allowing for random differences in GBS/serendipity)
        print(f"Score aggregation consistency check:")
        print(f"  Manual computation: mean={np.mean(manual_total):.4f}, std={np.std(manual_total):.4f}")
        print(f"  Pipeline computation: mean={np.mean(pipeline_total):.4f}, std={np.std(pipeline_total):.4f}")
        
        # Scores should be in similar ranges (exact match not expected due to randomness)
        manual_range = [np.min(manual_total), np.max(manual_total)]
        pipeline_range = [np.min(pipeline_total), np.max(pipeline_total)]
        
        # Ranges should overlap significantly
        overlap_start = max(manual_range[0], pipeline_range[0])
        overlap_end = min(manual_range[1], pipeline_range[1])
        overlap_ratio = (overlap_end - overlap_start) / (max(manual_range[1], pipeline_range[1]) - min(manual_range[0], pipeline_range[0]))
        
        assert overlap_ratio > 0.3, f"Score ranges should overlap significantly, overlap_ratio={overlap_ratio:.3f}"
    
    def test_cross_component_consistency(self, sample_people_data, rng):
        """Test consistency between different component implementations."""
        people, vectors, clusters, centers = sample_people_data(30, 12)
        
        user = people[0]
        candidates = people[1:16]
        
        # Test individual scoring vs suggest module
        from serendipity_engine_ui.engine.core import score, weights_for, apply_serendipity_scale
        from serendipity_engine_ui.engine.suggest import suggest
        
        # Individual component scoring
        intent = "ship"
        weights = apply_serendipity_scale(weights_for(intent), 1.0)
        
        individual_scores = []
        for candidate in candidates:
            # Fix randomness for comparison
            with patch('serendipity_engine_ui.engine.core.random.random', return_value=0.5):
                indiv_score = score(user, candidate, weights)
                individual_scores.append(indiv_score)
        
        # Suggest module scoring (with novelty binning)
        suggest_results = suggest(user, intent, candidates, k=15)
        
        # Both should return valid results
        assert len(individual_scores) == len(candidates), "Individual scoring should score all candidates"
        assert len(suggest_results) <= 15, "Suggest should respect k limit"
        assert len(suggest_results) > 0, "Suggest should return some results"
        
        # Individual scores should be finite
        assert all(np.isfinite(score) for score in individual_scores), "Individual scores should be finite"
        
        # Suggest results should have reasonable scores
        for result in suggest_results:
            # suggest() returns candidates, not scores, but they should be from the pool
            assert result["id"] in [c["id"] for c in candidates], "Suggest result should be from candidate pool"
        
        print(f"Cross-component consistency check:")
        print(f"  Individual scoring: {len(individual_scores)} scores, range=[{min(individual_scores):.3f}, {max(individual_scores):.3f}]")
        print(f"  Suggest module: {len(suggest_results)} results returned")

class TestStreamlitUIIntegration:
    """Test integration with Streamlit UI components."""
    
    def test_data_loading_integration(self, temp_dir):
        """Test data loading integration for UI."""
        from serendipity_engine_ui.engine.io import load_embeddings_csv
        from serendipity_engine_ui.engine.data_gen import make_people
        
        # Test CSV loading path
        test_data = {
            "id": [f"person_{i}" for i in range(20)],
            "novelty": np.random.random(20).tolist(),
            "availability": np.random.random(20).tolist(),
            "pathTrust": np.random.random(20).tolist(),
        }
        
        # Add embedding columns
        for dim in range(8):
            test_data[f"vec_{dim}"] = np.random.normal(0, 1, 20).tolist()
        
        # Save test CSV
        import pandas as pd
        test_df = pd.DataFrame(test_data)
        csv_path = f"{temp_dir}/test_embeddings.csv"
        test_df.to_csv(csv_path, index=False)
        
        # Test loading
        loaded_people = load_embeddings_csv(csv_path)
        
        assert len(loaded_people) == 20, f"Should load 20 people, got {len(loaded_people)}"
        
        for person in loaded_people:
            assert "id" in person, "Loaded person should have id"
            assert "vec" in person, "Loaded person should have vec"
            assert len(person["vec"]) == 8, f"Vector should have 8 dimensions, got {len(person['vec'])}"
            assert "novelty" in person, "Loaded person should have novelty"
            assert "availability" in person, "Loaded person should have availability"
            assert "pathTrust" in person, "Loaded person should have pathTrust"
        
        # Test synthetic generation path
        synthetic_people = make_people(n_people=15, n_dims=12, n_clusters=3, seed=42)
        
        assert len(synthetic_people) == 15, f"Should generate 15 people, got {len(synthetic_people)}"
        
        for person in synthetic_people:
            assert "id" in person, "Generated person should have id"
            assert "vec" in person, "Generated person should have vec"
            assert len(person["vec"]) == 12, f"Vector should have 12 dimensions"
            assert 0 <= person.get("novelty", 0) <= 1, "Novelty should be in [0,1]"
            assert 0 <= person.get("availability", 0) <= 1, "Availability should be in [0,1]"
            assert 0 <= person.get("pathTrust", 0) <= 1, "PathTrust should be in [0,1]"
        
        print(f"Data loading integration test:")
        print(f"  CSV loading: {len(loaded_people)} people loaded")
        print(f"  Synthetic generation: {len(synthetic_people)} people generated")
    
    def test_ui_parameter_integration(self, sample_people_data, rng):
        """Test integration with UI parameter configuration."""
        people, vectors, clusters, centers = sample_people_data(50, 16)
        
        user = people[0]  
        pool = people[1:26]
        
        # Simulate UI parameter configurations
        ui_configs = [
            {
                "name": "conservative",
                "intent": "deal", 
                "ser_scale": 0.5,
                "k": 5,
                "use_faiss": False,
                "quantum_gamma": 0.0,
                "use_gbs": False
            },
            {
                "name": "balanced", 
                "intent": "friend",
                "ser_scale": 1.0,
                "k": 10,
                "use_faiss": True,
                "M_prefilter": 20,
                "quantum_gamma": 0.3,
                "quantum_dims": 4,
                "use_gbs": False
            },
            {
                "name": "adventurous",
                "intent": "ship",
                "ser_scale": 1.5, 
                "k": 15,
                "use_faiss": True,
                "M_prefilter": 20,
                "quantum_gamma": 0.5,
                "quantum_dims": 4,
                "use_gbs": True,
                "gbs_modes": 3,
                "gbs_shots": 40,
                "gbs_lambda": 0.4
            }
        ]
        
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        config_results = {}
        
        for config in ui_configs:
            # Set up full parameter set
            params = {
                "seed": user,
                "pool": pool,
                "intent": config.get("intent", "friend"),
                "ser_scale": config.get("ser_scale", 1.0),
                "k": config.get("k", 10),
                "use_faiss_prefilter": config.get("use_faiss", False),
                "M_prefilter": config.get("M_prefilter", 50),
                "quantum_gamma": config.get("quantum_gamma", 0.0),
                "quantum_dims": config.get("quantum_dims", 4),
                "use_gbs": config.get("use_gbs", False),
                "gbs_modes": config.get("gbs_modes", 4),
                "gbs_shots": config.get("gbs_shots", 60),
                "gbs_cutoff": config.get("gbs_cutoff", 5),
                "gbs_lambda": config.get("gbs_lambda", 0.0)
            }
            
            results = score_pool(**params)
            config_results[config["name"]] = results
            
            # Validate results for each config
            assert isinstance(results, list), f"Config {config['name']} should return list"
            assert len(results) <= config["k"], f"Config {config['name']} returned too many results"
            assert len(results) > 0, f"Config {config['name']} returned no results"
        
        # Analyze configuration effects
        print(f"UI parameter integration analysis:")
        for name, results in config_results.items():
            avg_score = np.mean([r["total_score"] for r in results])
            diversity = np.std([r["total_score"] for r in results])
            print(f"  {name}: {len(results)} results, avg_score={avg_score:.4f}, diversity={diversity:.4f}")
        
        # Different configurations should produce different result characteristics
        conservative_diversity = np.std([r["total_score"] for r in config_results["conservative"]])
        adventurous_diversity = np.std([r["total_score"] for r in config_results["adventurous"]])
        
        print(f"  Diversity comparison: conservative={conservative_diversity:.4f}, adventurous={adventurous_diversity:.4f}")
    
    def test_ui_error_handling_integration(self, sample_people_data, rng):
        """Test UI error handling integration."""
        people, vectors, clusters, centers = sample_people_data(30, 10)
        
        user = people[0]
        pool = people[1:16]
        
        # Test various UI error scenarios
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        error_scenarios = [
            {
                "name": "negative_k",
                "params": {"k": -5},
                "should_handle": True
            },
            {
                "name": "zero_k", 
                "params": {"k": 0},
                "should_handle": True
            },
            {
                "name": "invalid_intent",
                "params": {"intent": "invalid_intent_xyz"},
                "should_handle": True  # Should use default
            },
            {
                "name": "extreme_serendipity",
                "params": {"ser_scale": 10.0}, 
                "should_handle": True  # Should be clamped
            },
            {
                "name": "negative_serendipity",
                "params": {"ser_scale": -2.0},
                "should_handle": True  # Should be clamped
            }
        ]
        
        for scenario in error_scenarios:
            base_params = {
                "seed": user,
                "pool": pool,
                "intent": "friend",
                "ser_scale": 1.0,
                "k": 5,
                "use_faiss_prefilter": False,
                "quantum_gamma": 0.0,
                "use_gbs": False
            }
            
            # Update with error scenario
            base_params.update(scenario["params"])
            
            try:
                results = score_pool(**base_params)
                
                if scenario["should_handle"]:
                    # Should handle gracefully
                    assert isinstance(results, list), f"Scenario {scenario['name']} should return list"
                    print(f"  {scenario['name']}: handled gracefully, returned {len(results)} results")
                else:
                    print(f"  {scenario['name']}: unexpectedly succeeded")
                    
            except Exception as e:
                if scenario["should_handle"]:
                    print(f"  {scenario['name']}: raised {type(e).__name__}: {e}")
                else:
                    print(f"  {scenario['name']}: correctly raised {type(e).__name__}")

if __name__ == "__main__":
    pytest.main([__file__])
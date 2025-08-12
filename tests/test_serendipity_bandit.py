"""
Serendipity Bandit Algorithm Tests
==================================

Comprehensive unit tests for the serendipity bandit system using Thompson sampling
for exploration-exploitation balance in novelty-based discovery.

Key Test Areas:
- Beta-like sampling approximation accuracy
- Bin selection and Thompson sampling
- Reward update mechanisms and convergence
- Multi-armed bandit properties validation
- Performance under different scenarios
- Statistical convergence properties
"""

import pytest
import numpy as np
import random
from typing import List, Dict, Any, Tuple
from unittest.mock import patch

class TestSerendipityBins:
    """Test serendipity bin creation and management."""
    
    def test_make_bins_basic_functionality(self):
        """Test basic bin creation functionality."""
        from serendipity_engine_ui.engine.serendipity import make_bins
        
        # Test default bin creation
        bins_5 = make_bins(5)
        
        assert len(bins_5) == 5, f"Expected 5 bins, got {len(bins_5)}"
        
        # Validate bin structure
        for i, bin_obj in enumerate(bins_5):
            assert "lo" in bin_obj, f"Bin {i} missing 'lo' key"
            assert "hi" in bin_obj, f"Bin {i} missing 'hi' key"
            assert "alpha" in bin_obj, f"Bin {i} missing 'alpha' key"
            assert "beta" in bin_obj, f"Bin {i} missing 'beta' key"
            
            # Check initial values
            assert bin_obj["alpha"] == 1.0, f"Bin {i} alpha should be 1.0"
            assert bin_obj["beta"] == 1.0, f"Bin {i} beta should be 1.0"
            
            # Check bin boundaries
            expected_lo = i / 5
            expected_hi = (i + 1) / 5
            assert abs(bin_obj["lo"] - expected_lo) < 1e-10, f"Bin {i} lo boundary incorrect"
            assert abs(bin_obj["hi"] - expected_hi) < 1e-10, f"Bin {i} hi boundary incorrect"
    
    def test_make_bins_different_sizes(self):
        """Test bin creation with different numbers of bins."""
        from serendipity_engine_ui.engine.serendipity import make_bins
        
        for n_bins in [2, 3, 5, 10, 20]:
            bins = make_bins(n_bins)
            
            assert len(bins) == n_bins, f"Expected {n_bins} bins, got {len(bins)}"
            
            # Check coverage of [0, 1) interval
            assert bins[0]["lo"] == 0.0, f"First bin should start at 0, got {bins[0]['lo']}"
            assert abs(bins[-1]["hi"] - 1.0) < 1e-10, f"Last bin should end at 1, got {bins[-1]['hi']}"
            
            # Check no gaps between bins
            for i in range(n_bins - 1):
                assert abs(bins[i]["hi"] - bins[i+1]["lo"]) < 1e-10, \
                    f"Gap between bins {i} and {i+1}"
            
            # Check no overlaps
            for i in range(n_bins):
                assert bins[i]["lo"] < bins[i]["hi"], \
                    f"Bin {i} has lo >= hi: {bins[i]['lo']} >= {bins[i]['hi']}"
    
    def test_make_bins_edge_cases(self):
        """Test bin creation edge cases."""
        from serendipity_engine_ui.engine.serendipity import make_bins
        
        # Test single bin
        bins_1 = make_bins(1)
        assert len(bins_1) == 1
        assert bins_1[0]["lo"] == 0.0
        assert bins_1[0]["hi"] == 1.0
        
        # Test large number of bins
        bins_100 = make_bins(100)
        assert len(bins_100) == 100
        assert bins_100[0]["lo"] == 0.0
        assert abs(bins_100[-1]["hi"] - 1.0) < 1e-10
        
        # Verify uniform spacing
        for i in range(100):
            expected_width = 1.0 / 100
            actual_width = bins_100[i]["hi"] - bins_100[i]["lo"]
            assert abs(actual_width - expected_width) < 1e-10, \
                f"Bin {i} width incorrect: {actual_width} vs {expected_width}"

class TestBetaLikeSampling:
    """Test the beta-like sampling approximation."""
    
    def test_beta_like_sample_basic_properties(self):
        """Test basic properties of beta-like sampling."""
        from serendipity_engine_ui.engine.serendipity import _beta_like_sample
        
        # Test with various alpha/beta combinations
        test_cases = [
            (1.0, 1.0),  # Uniform
            (2.0, 1.0),  # Skewed toward 1
            (1.0, 2.0),  # Skewed toward 0
            (5.0, 5.0),  # Symmetric, concentrated around 0.5
            (10.0, 2.0), # Heavily skewed toward 1
            (1.0, 10.0), # Heavily skewed toward 0
        ]
        
        for alpha, beta in test_cases:
            samples = [_beta_like_sample(alpha, beta) for _ in range(1000)]
            
            # Check bounds
            assert all(0 <= s <= 1 for s in samples), \
                f"Samples out of bounds for alpha={alpha}, beta={beta}"
            
            # Check finite values
            assert all(np.isfinite(s) for s in samples), \
                f"Non-finite samples for alpha={alpha}, beta={beta}"
            
            # Check statistical properties approach expected values
            sample_mean = np.mean(samples)
            expected_mean = alpha / (alpha + beta)
            
            # Allow some tolerance due to approximation and sampling variance
            tolerance = 0.1
            assert abs(sample_mean - expected_mean) < tolerance, \
                f"Mean deviation too large for alpha={alpha}, beta={beta}: " \
                f"got {sample_mean}, expected {expected_mean}"
    
    def test_beta_like_sample_edge_cases(self):
        """Test beta-like sampling edge cases."""
        from serendipity_engine_ui.engine.serendipity import _beta_like_sample
        
        # Test extreme values
        edge_cases = [
            (0.1, 0.1),   # Very small parameters
            (100.0, 1.0), # Very large alpha
            (1.0, 100.0), # Very large beta
            (100.0, 100.0), # Both very large
        ]
        
        for alpha, beta in edge_cases:
            samples = [_beta_like_sample(alpha, beta) for _ in range(100)]
            
            # Should still produce valid samples
            assert all(0 <= s <= 1 for s in samples), \
                f"Edge case samples out of bounds for alpha={alpha}, beta={beta}"
            assert all(np.isfinite(s) for s in samples), \
                f"Edge case non-finite samples for alpha={alpha}, beta={beta}"
    
    def test_beta_like_sample_consistency(self):
        """Test consistency of beta-like sampling."""
        from serendipity_engine_ui.engine.serendipity import _beta_like_sample
        
        # Set random seed for reproducibility test
        random.seed(42)
        samples1 = [_beta_like_sample(3.0, 2.0) for _ in range(100)]
        
        random.seed(42)
        samples2 = [_beta_like_sample(3.0, 2.0) for _ in range(100)]
        
        # Should get same sequence with same seed
        assert samples1 == samples2, "Beta-like sampling should be reproducible with same seed"

class TestBinSelection:
    """Test bin selection using Thompson sampling."""
    
    def test_pick_bin_basic_functionality(self):
        """Test basic bin selection functionality."""
        from serendipity_engine_ui.engine.serendipity import make_bins, pick_bin
        
        bins = make_bins(5)
        
        # Should always return a valid bin index
        for _ in range(100):
            selected_idx = pick_bin(bins)
            assert 0 <= selected_idx < 5, f"Invalid bin index: {selected_idx}"
            assert isinstance(selected_idx, int), f"Bin index should be int, got {type(selected_idx)}"
    
    def test_pick_bin_selection_bias(self):
        """Test that bin selection shows appropriate bias based on alpha/beta values."""
        from serendipity_engine_ui.engine.serendipity import make_bins, pick_bin
        
        bins = make_bins(3)
        
        # Modify bins to create clear preferences
        bins[0]["alpha"] = 10.0  # High reward history
        bins[0]["beta"] = 1.0
        
        bins[1]["alpha"] = 1.0   # Medium
        bins[1]["beta"] = 1.0
        
        bins[2]["alpha"] = 1.0   # Low reward history
        bins[2]["beta"] = 10.0
        
        # Sample many times to see selection bias
        selections = [pick_bin(bins) for _ in range(1000)]
        selection_counts = [selections.count(i) for i in range(3)]
        
        # Bin 0 should be selected most often (highest expected reward)
        assert selection_counts[0] > selection_counts[1], \
            f"High-reward bin not preferred: {selection_counts}"
        assert selection_counts[1] > selection_counts[2], \
            f"Medium-reward bin not preferred over low-reward: {selection_counts}"
        
        # But all bins should be selected sometimes (exploration)
        assert all(count > 0 for count in selection_counts), \
            f"Some bins never selected: {selection_counts}"
    
    def test_pick_bin_uniform_selection(self):
        """Test bin selection when all bins have equal parameters."""
        from serendipity_engine_ui.engine.serendipity import make_bins, pick_bin
        
        bins = make_bins(5)  # All initialized with alpha=1, beta=1
        
        # Sample many times
        selections = [pick_bin(bins) for _ in range(5000)]
        selection_counts = [selections.count(i) for i in range(5)]
        
        # With equal parameters, selection should be roughly uniform
        expected_count = 1000  # 5000 / 5
        tolerance = 200  # Allow some random variation
        
        for i, count in enumerate(selection_counts):
            assert abs(count - expected_count) < tolerance, \
                f"Bin {i} selected {count} times, expected ~{expected_count}"
    
    def test_pick_bin_deterministic_case(self):
        """Test bin selection in deterministic scenarios."""
        from serendipity_engine_ui.engine.serendipity import make_bins, pick_bin
        
        bins = make_bins(3)
        
        # Make one bin clearly superior
        bins[1]["alpha"] = 1000.0
        bins[1]["beta"] = 1.0
        
        # Others remain at default (1.0, 1.0)
        
        # Should almost always select the superior bin
        selections = [pick_bin(bins) for _ in range(100)]
        bin_1_count = selections.count(1)
        
        # Should select bin 1 most of the time (allow some exploration)
        assert bin_1_count > 80, f"Superior bin selected only {bin_1_count}/100 times"

class TestRewardUpdate:
    """Test reward update mechanisms."""
    
    def test_update_bin_basic_functionality(self):
        """Test basic bin update functionality."""
        from serendipity_engine_ui.engine.serendipity import make_bins, update_bin
        
        bins = make_bins(3)
        bin_to_update = bins[1]
        
        # Store initial values
        initial_alpha = bin_to_update["alpha"]
        initial_beta = bin_to_update["beta"]
        
        # Test reward update
        reward = 0.7
        update_bin(bin_to_update, reward)
        
        # Check updates
        expected_alpha = initial_alpha + reward
        expected_beta = initial_beta + (1.0 - reward)
        
        assert abs(bin_to_update["alpha"] - expected_alpha) < 1e-10, \
            f"Alpha update incorrect: {bin_to_update['alpha']} vs {expected_alpha}"
        assert abs(bin_to_update["beta"] - expected_beta) < 1e-10, \
            f"Beta update incorrect: {bin_to_update['beta']} vs {expected_beta}"
    
    def test_update_bin_edge_rewards(self):
        """Test bin updates with edge case rewards."""
        from serendipity_engine_ui.engine.serendipity import make_bins, update_bin
        
        bins = make_bins(4)
        
        # Test reward = 0.0 (complete failure)
        bin_0 = bins[0].copy()
        update_bin(bins[0], 0.0)
        assert bins[0]["alpha"] == bin_0["alpha"], "Alpha shouldn't change for 0 reward"
        assert bins[0]["beta"] == bin_0["beta"] + 1.0, "Beta should increase by 1 for 0 reward"
        
        # Test reward = 1.0 (perfect success)
        bin_1 = bins[1].copy()
        update_bin(bins[1], 1.0)
        assert bins[1]["alpha"] == bin_1["alpha"] + 1.0, "Alpha should increase by 1 for perfect reward"
        assert bins[1]["beta"] == bin_1["beta"], "Beta shouldn't change for 1 reward"
        
        # Test intermediate reward
        bin_2 = bins[2].copy()
        reward = 0.3
        update_bin(bins[2], reward)
        assert bins[2]["alpha"] == bin_2["alpha"] + reward
        assert bins[2]["beta"] == bin_2["beta"] + (1.0 - reward)
    
    def test_update_bin_accumulation(self):
        """Test that bin updates accumulate properly over time."""
        from serendipity_engine_ui.engine.serendipity import make_bins, update_bin
        
        bins = make_bins(2)
        bin_obj = bins[0]
        
        initial_alpha = bin_obj["alpha"]
        initial_beta = bin_obj["beta"]
        
        # Apply multiple updates
        rewards = [0.8, 0.6, 0.9, 0.2, 0.7]
        
        for reward in rewards:
            update_bin(bin_obj, reward)
        
        # Check final values
        expected_alpha = initial_alpha + sum(rewards)
        expected_beta = initial_beta + sum(1.0 - r for r in rewards)
        
        assert abs(bin_obj["alpha"] - expected_alpha) < 1e-10, \
            f"Accumulated alpha incorrect: {bin_obj['alpha']} vs {expected_alpha}"
        assert abs(bin_obj["beta"] - expected_beta) < 1e-10, \
            f"Accumulated beta incorrect: {bin_obj['beta']} vs {expected_beta}"
    
    def test_update_bin_out_of_range_rewards(self):
        """Test bin updates with out-of-range rewards."""
        from serendipity_engine_ui.engine.serendipity import make_bins, update_bin
        
        bins = make_bins(2)
        
        # Test rewards outside [0, 1] range
        test_cases = [-0.5, 1.5, -1.0, 2.0]
        
        for reward in test_cases:
            bin_before = bins[0].copy()
            
            # Update should still work (implementation dependent behavior)
            update_bin(bins[0], reward)
            
            # Alpha and beta should change
            assert bins[0]["alpha"] != bin_before["alpha"] or bins[0]["beta"] != bin_before["beta"], \
                f"Bin should update even with out-of-range reward {reward}"
            
            # Values should remain finite
            assert np.isfinite(bins[0]["alpha"]), f"Alpha became non-finite with reward {reward}"
            assert np.isfinite(bins[0]["beta"]), f"Beta became non-finite with reward {reward}"
            
            # Reset for next test
            bins[0] = {"lo": 0.0, "hi": 0.5, "alpha": 1.0, "beta": 1.0}

class TestSerendipityBanditIntegration:
    """Test integration of serendipity bandit components."""
    
    def test_exploration_exploitation_balance(self):
        """Test that the bandit balances exploration and exploitation."""
        from serendipity_engine_ui.engine.serendipity import make_bins, pick_bin, update_bin
        
        bins = make_bins(3)
        
        # Simulate bandit operation
        n_rounds = 1000
        bin_selections = []
        rewards_per_bin = [0.8, 0.3, 0.5]  # Different expected rewards
        
        for round_num in range(n_rounds):
            # Select bin
            selected_bin = pick_bin(bins)
            bin_selections.append(selected_bin)
            
            # Simulate reward (noisy version of expected reward)
            expected_reward = rewards_per_bin[selected_bin]
            noise = (random.random() - 0.5) * 0.2  # ±0.1 noise
            actual_reward = max(0.0, min(1.0, expected_reward + noise))
            
            # Update bin
            update_bin(bins[selected_bin], actual_reward)
        
        # Analyze results
        selection_counts = [bin_selections.count(i) for i in range(3)]
        final_alphas = [bin_obj["alpha"] for bin_obj in bins]
        final_betas = [bin_obj["beta"] for bin_obj in bins]
        
        print(f"Bandit results after {n_rounds} rounds:")
        for i in range(3):
            expected_reward = final_alphas[i] / (final_alphas[i] + final_betas[i])
            print(f"  Bin {i}: selected {selection_counts[i]} times, "
                  f"α={final_alphas[i]:.1f}, β={final_betas[i]:.1f}, "
                  f"expected_reward={expected_reward:.3f}")
        
        # Best bin (0) should be selected most often
        assert selection_counts[0] > selection_counts[1], "Best bin not preferred"
        assert selection_counts[0] > selection_counts[2], "Best bin not preferred over worst"
        
        # All bins should be explored at least a few times
        assert all(count >= 10 for count in selection_counts), \
            f"Insufficient exploration: {selection_counts}"
        
        # Final expected rewards should reflect true rewards
        estimated_rewards = [final_alphas[i] / (final_alphas[i] + final_betas[i]) for i in range(3)]
        assert estimated_rewards[0] > estimated_rewards[1], "Reward estimation incorrect"
        assert estimated_rewards[1] < estimated_rewards[0], "Reward estimation incorrect"
    
    def test_convergence_properties(self):
        """Test convergence properties of the bandit algorithm."""
        from serendipity_engine_ui.engine.serendipity import make_bins, pick_bin, update_bin
        
        bins = make_bins(2)
        
        # Simulate with known reward probabilities
        true_rewards = [0.7, 0.3]
        n_rounds = 2000
        
        # Track selection history
        history = {"selections": [], "rewards": [], "alphas": [[], []], "betas": [[], []]}
        
        for round_num in range(n_rounds):
            selected_bin = pick_bin(bins)
            history["selections"].append(selected_bin)
            
            # Generate reward based on true probability
            reward = 1.0 if random.random() < true_rewards[selected_bin] else 0.0
            history["rewards"].append(reward)
            
            # Update bin
            update_bin(bins[selected_bin], reward)
            
            # Track parameters every 100 rounds
            if round_num % 100 == 0:
                for i in range(2):
                    history["alphas"][i].append(bins[i]["alpha"])
                    history["betas"][i].append(bins[i]["beta"])
        
        # Check convergence
        final_expected_rewards = [
            bins[i]["alpha"] / (bins[i]["alpha"] + bins[i]["beta"]) 
            for i in range(2)
        ]
        
        tolerance = 0.1
        for i, (estimated, true) in enumerate(zip(final_expected_rewards, true_rewards)):
            assert abs(estimated - true) < tolerance, \
                f"Bin {i} didn't converge: estimated {estimated:.3f} vs true {true:.3f}"
        
        # Better bin should be selected more often in later rounds
        late_selections = history["selections"][-500:]  # Last 500 selections
        late_counts = [late_selections.count(i) for i in range(2)]
        
        assert late_counts[0] > late_counts[1], \
            f"Better bin not preferred in late rounds: {late_counts}"

class TestSerendipityBanditStatistics:
    """Test statistical properties of the serendipity bandit."""
    
    def test_regret_bounds(self):
        """Test that cumulative regret is bounded."""
        from serendipity_engine_ui.engine.serendipity import make_bins, pick_bin, update_bin
        
        bins = make_bins(3)
        true_rewards = [0.8, 0.5, 0.2]  # Decreasing rewards
        optimal_reward = max(true_rewards)
        
        cumulative_regret = 0.0
        n_rounds = 1000
        
        for round_num in range(n_rounds):
            selected_bin = pick_bin(bins)
            
            # Simulate noisy reward
            expected = true_rewards[selected_bin]
            noise = (random.random() - 0.5) * 0.1
            actual_reward = max(0.0, min(1.0, expected + noise))
            
            # Calculate regret
            regret = optimal_reward - expected  # Expected regret
            cumulative_regret += regret
            
            # Update bin
            update_bin(bins[selected_bin], actual_reward)
        
        # Regret should be bounded (not grow linearly)
        avg_regret = cumulative_regret / n_rounds
        
        print(f"Average regret per round: {avg_regret:.4f}")
        
        # Should achieve reasonable performance (regret decreases over time)
        assert avg_regret < 0.3, f"Average regret too high: {avg_regret}"
    
    def test_confidence_intervals(self):
        """Test that the bandit maintains appropriate confidence in its estimates."""
        from serendipity_engine_ui.engine.serendipity import make_bins, pick_bin, update_bin
        
        bins = make_bins(2)
        true_reward = 0.6
        
        # Run for different numbers of rounds to see confidence evolution
        round_counts = [50, 200, 500, 1000]
        confidences = []
        
        for n_rounds in round_counts:
            # Reset bins
            test_bins = make_bins(2)
            
            # Only use bin 0 for this test
            for _ in range(n_rounds):
                reward = 1.0 if random.random() < true_reward else 0.0
                update_bin(test_bins[0], reward)
            
            # Calculate confidence (inverse of variance)
            alpha = test_bins[0]["alpha"]
            beta = test_bins[0]["beta"]
            
            # Beta distribution variance: αβ/((α+β)²(α+β+1))
            variance = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            confidence = 1.0 / (variance + 1e-9)
            confidences.append(confidence)
            
            estimated_reward = alpha / (alpha + beta)
            print(f"After {n_rounds} rounds: estimated={estimated_reward:.3f}, "
                  f"confidence={confidence:.1f}")
        
        # Confidence should increase with more samples
        for i in range(len(confidences) - 1):
            assert confidences[i+1] > confidences[i], \
                f"Confidence should increase with more data: {confidences}"

if __name__ == "__main__":
    pytest.main([__file__])
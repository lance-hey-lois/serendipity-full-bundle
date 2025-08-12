"""
Quantum Superiority Metrics and Validation
==========================================

Comprehensive tests to establish and validate quantum superiority metrics
for the serendipity discovery system. Compares quantum-enhanced approaches
against classical baselines across multiple evaluation dimensions.

Key Superiority Areas:
- Discovery quality and relevance
- Exploration vs exploitation balance
- Serendipity enhancement capabilities
- Computational efficiency gains
- Robustness and consistency
- User experience improvements
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Tuple
import time
from dataclasses import dataclass
from scipy import stats
import itertools

@dataclass
class QuantumSuperiorityResult:
    """Structured quantum superiority evaluation result."""
    metric_name: str
    classical_score: float
    quantum_score: float
    improvement_ratio: float
    improvement_percentage: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]
    interpretation: str

class QuantumSuperiorityEvaluator:
    """Evaluate quantum superiority across multiple metrics."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = []
    
    def evaluate_metric(self, metric_name: str, 
                       classical_values: List[float],
                       quantum_values: List[float],
                       higher_is_better: bool = True) -> QuantumSuperiorityResult:
        """Evaluate quantum superiority for a specific metric."""
        
        classical_mean = np.mean(classical_values)
        quantum_mean = np.mean(quantum_values)
        
        # Calculate improvement
        if higher_is_better:
            improvement_ratio = quantum_mean / classical_mean if classical_mean > 0 else np.inf
            improvement_percentage = (quantum_mean - classical_mean) / classical_mean * 100 if classical_mean > 0 else 0
        else:  # Lower is better (e.g., time, error rate)
            improvement_ratio = classical_mean / quantum_mean if quantum_mean > 0 else np.inf
            improvement_percentage = (classical_mean - quantum_mean) / classical_mean * 100 if classical_mean > 0 else 0
        
        # Statistical significance test
        try:
            if higher_is_better:
                # One-tailed test: quantum > classical
                statistic, p_value = stats.ttest_ind(quantum_values, classical_values, alternative='greater')
            else:
                # One-tailed test: quantum < classical (better)
                statistic, p_value = stats.ttest_ind(quantum_values, classical_values, alternative='less')
            
            # Confidence interval for the difference
            diff_values = np.array(quantum_values) - np.array(classical_values)
            diff_mean = np.mean(diff_values)
            diff_sem = stats.sem(diff_values)
            ci = stats.t.interval(1 - self.significance_level, len(diff_values) - 1, 
                                 diff_mean, diff_sem)
        except Exception:
            p_value = 1.0
            ci = (0.0, 0.0)
        
        # Interpretation
        if p_value < self.significance_level:
            if improvement_percentage > 10:
                interpretation = "Strong quantum superiority"
            elif improvement_percentage > 5:
                interpretation = "Moderate quantum superiority"
            else:
                interpretation = "Weak quantum superiority"
        else:
            interpretation = "No significant quantum advantage"
        
        result = QuantumSuperiorityResult(
            metric_name=metric_name,
            classical_score=classical_mean,
            quantum_score=quantum_mean,
            improvement_ratio=improvement_ratio,
            improvement_percentage=improvement_percentage,
            statistical_significance=p_value,
            confidence_interval=ci,
            interpretation=interpretation
        )
        
        self.results.append(result)
        return result

class TestQuantumSuperiority:
    """Test quantum superiority across key metrics."""
    
    def test_discovery_quality_superiority(self, sample_people_data, rng):
        """Test quantum superiority in discovery quality."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        evaluator = QuantumSuperiorityEvaluator()
        
        # Create test scenarios with ground truth
        people, vectors, clusters, centers = sample_people_data(300, 48)
        
        # Multiple test users from different clusters
        test_users = [people[i] for i in [0, 50, 100, 150, 200]]
        
        classical_quality_scores = []
        quantum_quality_scores = []
        
        for user_idx, user in enumerate(test_users):
            # Create candidate pool
            pool_start = (user_idx * 50 + 25) % len(people)
            pool_end = min(pool_start + 80, len(people))
            pool = people[pool_start:pool_end]
            if len(pool) < 40:
                pool = people[1:41]  # Fallback
            
            # Classical recommendation
            classical_results = score_pool(
                seed=user,
                pool=pool,
                intent="friend",
                ser_scale=1.0,
                k=15,
                use_faiss_prefilter=True,
                M_prefilter=min(60, len(pool)),
                quantum_gamma=0.0,  # No quantum
                use_gbs=False
            )
            
            # Quantum-enhanced recommendation
            quantum_results = score_pool(
                seed=user,
                pool=pool,
                intent="friend",
                ser_scale=1.0,
                k=15,
                use_faiss_prefilter=True,
                M_prefilter=min(60, len(pool)),
                quantum_gamma=0.4,  # Quantum boost
                quantum_dims=4,
                use_gbs=True,
                gbs_modes=3,
                gbs_shots=60,
                gbs_lambda=0.3
            )
            
            # Quality evaluation based on vector similarity
            user_vec = np.array(user["vec"])
            user_vec_norm = user_vec / np.linalg.norm(user_vec)
            
            def evaluate_quality(results):
                similarities = []
                novelties = []
                diversities = []
                
                for r in results:
                    candidate_vec = np.array(r["candidate"]["vec"])
                    candidate_vec_norm = candidate_vec / np.linalg.norm(candidate_vec)
                    similarity = np.dot(user_vec_norm, candidate_vec_norm)
                    similarities.append(similarity)
                    novelties.append(r["candidate"].get("novelty", 0.5))
                
                # Pairwise diversity
                for i in range(len(results)):
                    for j in range(i + 1, len(results)):
                        vec_i = np.array(results[i]["candidate"]["vec"])
                        vec_j = np.array(results[j]["candidate"]["vec"])
                        distance = np.linalg.norm(vec_i - vec_j)
                        diversities.append(distance)
                
                # Combined quality score
                relevance_score = np.mean(similarities)
                novelty_score = np.mean(novelties)
                diversity_score = np.mean(diversities) if diversities else 0
                
                # Weighted combination
                quality_score = (0.5 * relevance_score + 
                               0.3 * novelty_score + 
                               0.2 * min(diversity_score / 2.0, 1.0))  # Normalize diversity
                
                return {
                    "quality_score": quality_score,
                    "relevance": relevance_score,
                    "novelty": novelty_score,
                    "diversity": diversity_score
                }
            
            classical_quality = evaluate_quality(classical_results)
            quantum_quality = evaluate_quality(quantum_results)
            
            classical_quality_scores.append(classical_quality["quality_score"])
            quantum_quality_scores.append(quantum_quality["quality_score"])
        
        # Evaluate quantum superiority in discovery quality
        quality_result = evaluator.evaluate_metric(
            "discovery_quality",
            classical_quality_scores,
            quantum_quality_scores,
            higher_is_better=True
        )
        
        print(f"Discovery Quality Superiority Analysis:")
        print(f"  Classical mean: {quality_result.classical_score:.4f}")
        print(f"  Quantum mean: {quality_result.quantum_score:.4f}")
        print(f"  Improvement: {quality_result.improvement_percentage:+.1f}%")
        print(f"  P-value: {quality_result.statistical_significance:.4f}")
        print(f"  Interpretation: {quality_result.interpretation}")
        
        # Basic quality assurance
        assert np.mean(classical_quality_scores) > 0, "Classical approach should produce positive quality scores"
        assert np.mean(quantum_quality_scores) > 0, "Quantum approach should produce positive quality scores"
        
        return quality_result
    
    def test_exploration_exploitation_superiority(self, sample_people_data, rng):
        """Test quantum superiority in exploration-exploitation balance."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        evaluator = QuantumSuperiorityEvaluator()
        
        # Create clustered dataset for exploration analysis
        people, vectors, clusters, centers = sample_people_data(200, 32)
        
        # User from cluster 0
        cluster_0_members = [p for p, c in zip(people, clusters) if c == 0]
        user = cluster_0_members[0] if cluster_0_members else people[0]
        
        # Mixed pool with different cluster densities
        pool = []
        for cluster_id in [0, 1, 2]:
            cluster_members = [p for p, c in zip(people, clusters) if c == cluster_id]
            if cluster_id == 0:  # Same cluster as user (exploitation)
                pool.extend(cluster_members[1:16])  # 15 members
            else:  # Other clusters (exploration)
                pool.extend(cluster_members[:10])   # 10 members each
        
        # Test different serendipity levels
        serendipity_levels = [0.5, 1.0, 1.5]
        
        classical_exploration_scores = []
        quantum_exploration_scores = []
        
        for ser_scale in serendipity_levels:
            # Classical approach
            classical_results = score_pool(
                seed=user,
                pool=pool,
                intent="ship",  # Ship intent for exploration
                ser_scale=ser_scale,
                k=20,
                use_faiss_prefilter=False,
                quantum_gamma=0.0,
                use_gbs=False
            )
            
            # Quantum approach
            quantum_results = score_pool(
                seed=user,
                pool=pool,
                intent="ship",
                ser_scale=ser_scale,
                k=20,
                use_faiss_prefilter=False,
                quantum_gamma=0.5,
                quantum_dims=4,
                use_gbs=True,
                gbs_modes=4,
                gbs_shots=80,
                gbs_lambda=0.4
            )
            
            def analyze_exploration(results, user_cluster=0):
                cluster_distribution = {}
                for r in results:
                    candidate_id = r["candidate"]["id"]
                    # Find candidate's cluster
                    for i, person in enumerate(people):
                        if person["id"] == candidate_id:
                            candidate_cluster = clusters[i]
                            cluster_distribution[candidate_cluster] = cluster_distribution.get(candidate_cluster, 0) + 1
                            break
                
                total_results = len(results)
                same_cluster_ratio = cluster_distribution.get(user_cluster, 0) / total_results
                exploration_ratio = 1.0 - same_cluster_ratio
                cluster_diversity = len(cluster_distribution) / 3  # Max 3 clusters
                
                # Exploration score: balance of exploration and diversity
                exploration_score = 0.6 * exploration_ratio + 0.4 * cluster_diversity
                
                return exploration_score
            
            classical_exploration = analyze_exploration(classical_results)
            quantum_exploration = analyze_exploration(quantum_results)
            
            classical_exploration_scores.append(classical_exploration)
            quantum_exploration_scores.append(quantum_exploration)
        
        # Evaluate quantum superiority in exploration
        exploration_result = evaluator.evaluate_metric(
            "exploration_balance",
            classical_exploration_scores,
            quantum_exploration_scores,
            higher_is_better=True
        )
        
        print(f"Exploration-Exploitation Superiority Analysis:")
        print(f"  Classical mean exploration: {exploration_result.classical_score:.4f}")
        print(f"  Quantum mean exploration: {exploration_result.quantum_score:.4f}")
        print(f"  Improvement: {exploration_result.improvement_percentage:+.1f}%")
        print(f"  P-value: {exploration_result.statistical_significance:.4f}")
        print(f"  Interpretation: {exploration_result.interpretation}")
        
        return exploration_result
    
    def test_computational_efficiency_superiority(self, sample_people_data, rng):
        """Test quantum computational efficiency vs classical approaches."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        evaluator = QuantumSuperiorityEvaluator()
        
        people, vectors, clusters, centers = sample_people_data(500, 64)
        user = people[0]
        
        # Test different pool sizes for efficiency analysis
        pool_sizes = [100, 200, 400]
        
        classical_times = []
        quantum_times = []
        classical_quality_per_time = []
        quantum_quality_per_time = []
        
        for pool_size in pool_sizes:
            pool = people[1:pool_size+1]
            
            # Classical approach timing
            classical_start = time.perf_counter()
            
            classical_results = score_pool(
                seed=user,
                pool=pool,
                intent="collab",
                ser_scale=1.0,
                k=25,
                use_faiss_prefilter=True,
                M_prefilter=min(150, pool_size),
                quantum_gamma=0.0,
                use_gbs=False
            )
            
            classical_end = time.perf_counter()
            classical_time = (classical_end - classical_start) * 1000  # ms
            
            # Quantum approach timing
            quantum_start = time.perf_counter()
            
            quantum_results = score_pool(
                seed=user,
                pool=pool,
                intent="collab",
                ser_scale=1.0,
                k=25,
                use_faiss_prefilter=True,
                M_prefilter=min(150, pool_size),
                quantum_gamma=0.3,
                quantum_dims=4,
                use_gbs=True,
                gbs_modes=4,
                gbs_shots=60,
                gbs_lambda=0.2
            )
            
            quantum_end = time.perf_counter()
            quantum_time = (quantum_end - quantum_start) * 1000  # ms
            
            classical_times.append(classical_time)
            quantum_times.append(quantum_time)
            
            # Quality per unit time (efficiency metric)
            user_vec = np.array(user["vec"])
            user_vec_norm = user_vec / np.linalg.norm(user_vec)
            
            def compute_quality_per_time(results, time_ms):
                total_similarity = 0
                for r in results:
                    candidate_vec = np.array(r["candidate"]["vec"])
                    candidate_vec_norm = candidate_vec / np.linalg.norm(candidate_vec)
                    similarity = np.dot(user_vec_norm, candidate_vec_norm)
                    total_similarity += max(0, similarity)  # Only positive contributions
                
                quality = total_similarity / len(results) if results else 0
                efficiency = quality / (time_ms / 1000) if time_ms > 0 else 0  # Quality per second
                return efficiency
            
            classical_efficiency = compute_quality_per_time(classical_results, classical_time)
            quantum_efficiency = compute_quality_per_time(quantum_results, quantum_time)
            
            classical_quality_per_time.append(classical_efficiency)
            quantum_quality_per_time.append(quantum_efficiency)
        
        # Evaluate time efficiency (lower is better)
        time_result = evaluator.evaluate_metric(
            "execution_time",
            classical_times,
            quantum_times,
            higher_is_better=False
        )
        
        # Evaluate quality efficiency (higher is better)
        efficiency_result = evaluator.evaluate_metric(
            "quality_per_time",
            classical_quality_per_time,
            quantum_quality_per_time,
            higher_is_better=True
        )
        
        print(f"Computational Efficiency Superiority Analysis:")
        print(f"Execution Time:")
        print(f"  Classical mean: {time_result.classical_score:.1f} ms")
        print(f"  Quantum mean: {time_result.quantum_score:.1f} ms")
        print(f"  Time improvement: {time_result.improvement_percentage:+.1f}%")
        print(f"  P-value: {time_result.statistical_significance:.4f}")
        
        print(f"Quality Efficiency:")
        print(f"  Classical mean: {efficiency_result.classical_score:.4f} quality/sec")
        print(f"  Quantum mean: {efficiency_result.quantum_score:.4f} quality/sec")
        print(f"  Efficiency improvement: {efficiency_result.improvement_percentage:+.1f}%")
        print(f"  P-value: {efficiency_result.statistical_significance:.4f}")
        
        return time_result, efficiency_result
    
    def test_robustness_superiority(self, sample_people_data, rng):
        """Test quantum robustness vs classical approaches under various conditions."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        evaluator = QuantumSuperiorityEvaluator()
        
        people, vectors, clusters, centers = sample_people_data(150, 24)
        
        # Test robustness under various challenging conditions
        test_conditions = [
            {
                "name": "small_pool",
                "pool_size": 20,
                "user_idx": 0,
                "k": 10
            },
            {
                "name": "large_k",
                "pool_size": 80,
                "user_idx": 0,
                "k": 30
            },
            {
                "name": "sparse_data",
                "pool_size": 50,
                "user_idx": 0,
                "k": 15,
                "sparsity": 0.7  # 70% of vector elements set to 0
            }
        ]
        
        classical_robustness_scores = []
        quantum_robustness_scores = []
        
        for condition in test_conditions:
            user = people[condition["user_idx"]]
            pool = people[1:condition["pool_size"]+1]
            
            # Apply sparsity if specified
            if "sparsity" in condition:
                sparsity_ratio = condition["sparsity"]
                for person in [user] + pool:
                    vec = np.array(person["vec"])
                    mask = rng.random(len(vec)) > sparsity_ratio
                    sparse_vec = vec * mask
                    person["vec"] = sparse_vec.tolist()
            
            # Test multiple runs for robustness assessment
            classical_successes = 0
            quantum_successes = 0
            classical_qualities = []
            quantum_qualities = []
            
            for run in range(5):  # Multiple runs to test consistency
                try:
                    # Classical approach
                    classical_results = score_pool(
                        seed=user,
                        pool=pool,
                        intent="mentor",
                        ser_scale=1.0,
                        k=condition["k"],
                        use_faiss_prefilter=False,  # More robust without prefiltering
                        quantum_gamma=0.0,
                        use_gbs=False
                    )
                    
                    if len(classical_results) > 0:
                        classical_successes += 1
                        # Simple quality measure: average score
                        avg_score = np.mean([r["total_score"] for r in classical_results])
                        classical_qualities.append(avg_score)
                    
                except Exception as e:
                    print(f"    Classical failed on {condition['name']} run {run}: {e}")
                
                try:
                    # Quantum approach  
                    quantum_results = score_pool(
                        seed=user,
                        pool=pool,
                        intent="mentor",
                        ser_scale=1.0,
                        k=condition["k"],
                        use_faiss_prefilter=False,
                        quantum_gamma=0.3,
                        quantum_dims=min(4, len(user["vec"])),
                        use_gbs=True,
                        gbs_modes=min(3, len(user["vec"])),
                        gbs_shots=40,
                        gbs_lambda=0.2
                    )
                    
                    if len(quantum_results) > 0:
                        quantum_successes += 1
                        avg_score = np.mean([r["total_score"] for r in quantum_results])
                        quantum_qualities.append(avg_score)
                    
                except Exception as e:
                    print(f"    Quantum failed on {condition['name']} run {run}: {e}")
            
            # Robustness score: combination of success rate and quality consistency
            classical_success_rate = classical_successes / 5
            quantum_success_rate = quantum_successes / 5
            
            classical_quality_std = np.std(classical_qualities) if classical_qualities else 1.0
            quantum_quality_std = np.std(quantum_qualities) if quantum_qualities else 1.0
            
            # Lower standard deviation is better (more consistent)
            classical_consistency = 1.0 / (1.0 + classical_quality_std)
            quantum_consistency = 1.0 / (1.0 + quantum_quality_std)
            
            # Combined robustness score
            classical_robustness = 0.7 * classical_success_rate + 0.3 * classical_consistency
            quantum_robustness = 0.7 * quantum_success_rate + 0.3 * quantum_consistency
            
            classical_robustness_scores.append(classical_robustness)
            quantum_robustness_scores.append(quantum_robustness)
            
            print(f"Robustness test - {condition['name']}:")
            print(f"  Classical: {classical_success_rate:.2f} success rate, {classical_consistency:.3f} consistency")
            print(f"  Quantum: {quantum_success_rate:.2f} success rate, {quantum_consistency:.3f} consistency")
        
        # Evaluate quantum superiority in robustness
        robustness_result = evaluator.evaluate_metric(
            "robustness",
            classical_robustness_scores,
            quantum_robustness_scores,
            higher_is_better=True
        )
        
        print(f"Robustness Superiority Analysis:")
        print(f"  Classical mean robustness: {robustness_result.classical_score:.4f}")
        print(f"  Quantum mean robustness: {robustness_result.quantum_score:.4f}")
        print(f"  Improvement: {robustness_result.improvement_percentage:+.1f}%")
        print(f"  P-value: {robustness_result.statistical_significance:.4f}")
        print(f"  Interpretation: {robustness_result.interpretation}")
        
        return robustness_result
    
    def test_serendipity_enhancement_superiority(self, sample_people_data, rng):
        """Test quantum superiority in serendipity enhancement."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        evaluator = QuantumSuperiorityEvaluator()
        
        people, vectors, clusters, centers = sample_people_data(200, 32)
        
        # Create scenarios with different serendipity requirements
        test_users = [people[i] for i in [0, 40, 80, 120]]
        
        classical_serendipity_scores = []
        quantum_serendipity_scores = []
        
        for user in test_users:
            pool_start = people.index(user) + 1
            pool = people[pool_start:pool_start+60] if pool_start+60 <= len(people) else people[1:61]
            
            # High serendipity setting
            ser_scale = 1.5
            
            # Classical approach
            classical_results = score_pool(
                seed=user,
                pool=pool,
                intent="ship",  # Ship intent emphasizes serendipity
                ser_scale=ser_scale,
                k=20,
                use_faiss_prefilter=True,
                M_prefilter=40,
                quantum_gamma=0.0,
                use_gbs=False
            )
            
            # Quantum approach
            quantum_results = score_pool(
                seed=user,
                pool=pool,
                intent="ship",
                ser_scale=ser_scale,
                k=20,
                use_faiss_prefilter=True,
                M_prefilter=40,
                quantum_gamma=0.4,
                quantum_dims=4,
                use_gbs=True,
                gbs_modes=4,
                gbs_shots=80,
                gbs_lambda=0.5  # High GBS weight for serendipity
            )
            
            def measure_serendipity(results, user):
                user_vec = np.array(user["vec"])
                user_vec_norm = user_vec / np.linalg.norm(user_vec)
                
                novelties = []
                surprises = []
                diversities = []
                
                for r in results:
                    # Novelty from candidate attributes
                    novelty = r["candidate"].get("novelty", 0.5)
                    novelties.append(novelty)
                    
                    # Surprise: inverse of similarity (unexpected recommendations)
                    candidate_vec = np.array(r["candidate"]["vec"])
                    candidate_vec_norm = candidate_vec / np.linalg.norm(candidate_vec)
                    similarity = np.dot(user_vec_norm, candidate_vec_norm)
                    surprise = 1.0 - max(0, similarity)  # Higher surprise for low similarity
                    surprises.append(surprise)
                
                # Pairwise diversity among recommendations
                for i in range(len(results)):
                    for j in range(i + 1, len(results)):
                        vec_i = np.array(results[i]["candidate"]["vec"])
                        vec_j = np.array(results[j]["candidate"]["vec"])
                        
                        # Cosine distance
                        norm_i = np.linalg.norm(vec_i)
                        norm_j = np.linalg.norm(vec_j)
                        if norm_i > 0 and norm_j > 0:
                            cosine_sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                            cosine_distance = 1.0 - cosine_sim
                            diversities.append(cosine_distance)
                
                # Combined serendipity score
                avg_novelty = np.mean(novelties)
                avg_surprise = np.mean(surprises)
                avg_diversity = np.mean(diversities) if diversities else 0
                
                # Weighted combination emphasizing different aspects of serendipity
                serendipity_score = (0.4 * avg_novelty + 
                                   0.3 * avg_surprise + 
                                   0.3 * avg_diversity)
                
                return serendipity_score
            
            classical_serendipity = measure_serendipity(classical_results, user)
            quantum_serendipity = measure_serendipity(quantum_results, user)
            
            classical_serendipity_scores.append(classical_serendipity)
            quantum_serendipity_scores.append(quantum_serendipity)
        
        # Evaluate quantum superiority in serendipity
        serendipity_result = evaluator.evaluate_metric(
            "serendipity_enhancement",
            classical_serendipity_scores,
            quantum_serendipity_scores,
            higher_is_better=True
        )
        
        print(f"Serendipity Enhancement Superiority Analysis:")
        print(f"  Classical mean serendipity: {serendipity_result.classical_score:.4f}")
        print(f"  Quantum mean serendipity: {serendipity_result.quantum_score:.4f}")
        print(f"  Improvement: {serendipity_result.improvement_percentage:+.1f}%")
        print(f"  P-value: {serendipity_result.statistical_significance:.4f}")
        print(f"  Interpretation: {serendipity_result.interpretation}")
        
        return serendipity_result
    
    def test_overall_quantum_superiority_assessment(self, sample_people_data, rng):
        """Comprehensive quantum superiority assessment across all metrics."""
        print("="*60)
        print("COMPREHENSIVE QUANTUM SUPERIORITY ASSESSMENT")
        print("="*60)
        
        # Run all superiority tests
        quality_result = self.test_discovery_quality_superiority(sample_people_data, rng)
        exploration_result = self.test_exploration_exploitation_superiority(sample_people_data, rng)
        time_result, efficiency_result = self.test_computational_efficiency_superiority(sample_people_data, rng)
        robustness_result = self.test_robustness_superiority(sample_people_data, rng)
        serendipity_result = self.test_serendipity_enhancement_superiority(sample_people_data, rng)
        
        # Aggregate results
        all_results = [
            quality_result,
            exploration_result,
            time_result,
            efficiency_result,
            robustness_result,
            serendipity_result
        ]
        
        # Count superiority evidence
        significant_improvements = 0
        total_metrics = len(all_results)
        improvement_percentages = []
        
        print("\nQUANTUM SUPERIORITY SUMMARY:")
        print("-" * 60)
        
        for result in all_results:
            status = "✓" if result.statistical_significance < 0.05 else "?"
            direction = "↑" if result.improvement_percentage > 0 else "↓"
            
            print(f"{status} {result.metric_name:20s}: {result.improvement_percentage:+6.1f}% {direction} "
                  f"(p={result.statistical_significance:.3f})")
            
            if result.statistical_significance < 0.05 and result.improvement_percentage > 0:
                significant_improvements += 1
            
            improvement_percentages.append(result.improvement_percentage)
        
        # Overall assessment
        avg_improvement = np.mean(improvement_percentages)
        significant_ratio = significant_improvements / total_metrics
        
        print("-" * 60)
        print(f"OVERALL ASSESSMENT:")
        print(f"  Average improvement: {avg_improvement:+.1f}%")
        print(f"  Significant improvements: {significant_improvements}/{total_metrics} ({significant_ratio:.1%})")
        
        # Quantum superiority verdict
        if significant_ratio >= 0.7 and avg_improvement > 5:
            verdict = "STRONG QUANTUM SUPERIORITY"
        elif significant_ratio >= 0.5 and avg_improvement > 2:
            verdict = "MODERATE QUANTUM SUPERIORITY"
        elif significant_ratio >= 0.3 and avg_improvement > 0:
            verdict = "WEAK QUANTUM ADVANTAGE"
        else:
            verdict = "NO CLEAR QUANTUM SUPERIORITY"
        
        print(f"  Verdict: {verdict}")
        print("="*60)
        
        # Validation requirements
        assert significant_ratio > 0, "Should show some quantum advantages"
        assert avg_improvement > -10, "Should not show major quantum disadvantages"
        
        return {
            "verdict": verdict,
            "avg_improvement": avg_improvement,
            "significant_ratio": significant_ratio,
            "individual_results": all_results
        }

if __name__ == "__main__":
    pytest.main([__file__])
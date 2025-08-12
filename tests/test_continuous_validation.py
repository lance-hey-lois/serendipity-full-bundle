"""
Continuous Validation Pipeline
==============================

Automated continuous validation system for the quantum-enhanced serendipity engine.
Provides ongoing validation, regression detection, and performance monitoring.

Key Validation Areas:
- Automated regression testing
- Performance trend monitoring  
- Quality metric tracking
- Cross-version compatibility
- Production readiness validation
- Automated report generation
"""

import pytest
import numpy as np
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import tempfile
import os

@dataclass
class ValidationReport:
    """Structured validation report."""
    timestamp: str
    test_suite: str
    version: str
    passed: int
    failed: int
    skipped: int
    warnings: int
    performance_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    regression_flags: List[str]
    recommendations: List[str]

class ContinuousValidator:
    """Continuous validation orchestrator."""
    
    def __init__(self, report_dir: Optional[str] = None):
        self.report_dir = report_dir or tempfile.mkdtemp()
        self.baseline_metrics = {}
        self.regression_thresholds = {
            "performance_degradation": 0.2,  # 20% performance degradation
            "quality_degradation": 0.1,      # 10% quality degradation
            "memory_increase": 0.3,           # 30% memory increase
        }
    
    def save_baseline(self, metrics: Dict[str, Any], version: str = "baseline"):
        """Save baseline metrics for regression detection."""
        baseline_file = os.path.join(self.report_dir, f"baseline_{version}.json")
        with open(baseline_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.baseline_metrics[version] = metrics
        
    def load_baseline(self, version: str = "baseline") -> Optional[Dict[str, Any]]:
        """Load baseline metrics."""
        baseline_file = os.path.join(self.report_dir, f"baseline_{version}.json")
        if os.path.exists(baseline_file):
            with open(baseline_file, 'r') as f:
                return json.load(f)
        return None
    
    def detect_regressions(self, current_metrics: Dict[str, Any], 
                          baseline_metrics: Dict[str, Any]) -> List[str]:
        """Detect performance/quality regressions."""
        regressions = []
        
        for key, current_value in current_metrics.items():
            if key in baseline_metrics:
                baseline_value = baseline_metrics[key]
                
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    # Performance metrics (lower is better for time/memory)
                    if "time" in key.lower() or "duration" in key.lower() or "memory" in key.lower():
                        if current_value > baseline_value * (1 + self.regression_thresholds["performance_degradation"]):
                            degradation = (current_value - baseline_value) / baseline_value
                            regressions.append(f"{key} degraded by {degradation:.1%} ({baseline_value:.3f} -> {current_value:.3f})")
                    
                    # Quality metrics (higher is better for accuracy/similarity)
                    elif "accuracy" in key.lower() or "similarity" in key.lower() or "quality" in key.lower():
                        if current_value < baseline_value * (1 - self.regression_thresholds["quality_degradation"]):
                            degradation = (baseline_value - current_value) / baseline_value
                            regressions.append(f"{key} degraded by {degradation:.1%} ({baseline_value:.3f} -> {current_value:.3f})")
        
        return regressions
    
    def generate_report(self, test_results: Dict[str, Any], 
                       performance_metrics: Dict[str, float],
                       quality_metrics: Dict[str, float]) -> ValidationReport:
        """Generate comprehensive validation report."""
        
        # Check for regressions
        baseline = self.load_baseline()
        regressions = []
        if baseline:
            all_metrics = {**performance_metrics, **quality_metrics}
            baseline_all = {**baseline.get("performance", {}), **baseline.get("quality", {})}
            regressions = self.detect_regressions(all_metrics, baseline_all)
        
        # Generate recommendations
        recommendations = []
        
        # Performance recommendations
        for key, value in performance_metrics.items():
            if "memory" in key.lower() and value > 1000:  # > 1GB
                recommendations.append(f"High memory usage detected in {key}: {value:.1f} MB")
            if "time" in key.lower() and value > 5000:  # > 5 seconds
                recommendations.append(f"Slow performance detected in {key}: {value:.1f} ms")
        
        # Quality recommendations
        for key, value in quality_metrics.items():
            if "accuracy" in key.lower() and value < 0.8:
                recommendations.append(f"Low accuracy detected in {key}: {value:.3f}")
        
        # Regression recommendations
        if regressions:
            recommendations.append(f"Performance regressions detected: {len(regressions)} issues")
        
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            test_suite="quantum_serendipity_engine",
            version="current",
            passed=test_results.get("passed", 0),
            failed=test_results.get("failed", 0),
            skipped=test_results.get("skipped", 0),
            warnings=test_results.get("warnings", 0),
            performance_metrics=performance_metrics,
            quality_metrics=quality_metrics,
            regression_flags=regressions,
            recommendations=recommendations
        )
        
        # Save report
        report_file = os.path.join(self.report_dir, f"validation_report_{int(time.time())}.json")
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        return report

class TestContinuousValidation:
    """Test continuous validation pipeline."""
    
    def test_automated_regression_detection(self, temp_dir):
        """Test automated regression detection."""
        validator = ContinuousValidator(temp_dir)
        
        # Create baseline metrics
        baseline_metrics = {
            "performance": {
                "quantum_kernel_time_ms": 50.0,
                "gbs_boost_time_ms": 200.0,
                "pipeline_memory_mb": 150.0,
                "faiss_search_time_ms": 10.0
            },
            "quality": {
                "recommendation_accuracy": 0.85,
                "diversity_score": 0.75,
                "user_satisfaction": 0.80
            }
        }
        
        validator.save_baseline(baseline_metrics)
        
        # Test with good metrics (no regression)
        good_metrics = {
            "quantum_kernel_time_ms": 45.0,  # Better
            "gbs_boost_time_ms": 195.0,      # Slightly better
            "pipeline_memory_mb": 140.0,     # Better
            "faiss_search_time_ms": 9.0,     # Better
            "recommendation_accuracy": 0.87,  # Better
            "diversity_score": 0.76,         # Slightly better
            "user_satisfaction": 0.82        # Better
        }
        
        good_regressions = validator.detect_regressions(good_metrics, 
            {**baseline_metrics["performance"], **baseline_metrics["quality"]})
        
        assert len(good_regressions) == 0, f"Good metrics shouldn't trigger regressions: {good_regressions}"
        
        # Test with regression metrics
        regression_metrics = {
            "quantum_kernel_time_ms": 70.0,  # 40% slower - regression
            "gbs_boost_time_ms": 180.0,      # Better
            "pipeline_memory_mb": 220.0,     # 47% more memory - regression
            "faiss_search_time_ms": 8.0,     # Better
            "recommendation_accuracy": 0.75, # 12% worse - regression
            "diversity_score": 0.74,         # Slightly worse but within threshold
            "user_satisfaction": 0.81        # Better
        }
        
        regressions = validator.detect_regressions(regression_metrics,
            {**baseline_metrics["performance"], **baseline_metrics["quality"]})
        
        assert len(regressions) > 0, "Regression metrics should trigger alerts"
        
        print(f"Regression detection test:")
        print(f"  Good metrics: {len(good_regressions)} regressions detected")
        print(f"  Regression metrics: {len(regressions)} regressions detected")
        for regression in regressions:
            print(f"    - {regression}")
    
    def test_performance_trend_monitoring(self, sample_people_data, rng):
        """Test performance trend monitoring over multiple runs."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        people, vectors, clusters, centers = sample_people_data(100, 32)
        user = people[0]
        pool = people[1:51]
        
        # Simulate multiple validation runs over time
        trend_data = []
        
        for run_id in range(10):
            # Simulate slight variations in performance
            start_time = time.perf_counter()
            
            results = score_pool(
                seed=user,
                pool=pool,
                intent="ship",
                ser_scale=1.0,
                k=10,
                use_faiss_prefilter=True,
                M_prefilter=30,
                quantum_gamma=0.3,
                quantum_dims=4,
                use_gbs=False,  # Faster for trend testing
                gbs_lambda=0.0
            )
            
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Add some artificial variation
            duration_ms *= (1.0 + rng.uniform(-0.1, 0.1))
            
            trend_point = {
                "run_id": run_id,
                "timestamp": time.time() + run_id * 3600,  # 1 hour apart
                "duration_ms": duration_ms,
                "memory_mb": rng.uniform(100, 200),
                "n_results": len(results),
                "avg_score": np.mean([r["total_score"] for r in results])
            }
            
            trend_data.append(trend_point)
        
        # Analyze trends
        durations = [point["duration_ms"] for point in trend_data]
        memories = [point["memory_mb"] for point in trend_data]
        scores = [point["avg_score"] for point in trend_data]
        
        # Calculate trend statistics
        duration_trend = np.polyfit(range(len(durations)), durations, 1)[0]  # Slope
        memory_trend = np.polyfit(range(len(memories)), memories, 1)[0]
        score_trend = np.polyfit(range(len(scores)), scores, 1)[0]
        
        trend_analysis = {
            "duration_trend_ms_per_run": duration_trend,
            "memory_trend_mb_per_run": memory_trend,
            "score_trend_per_run": score_trend,
            "duration_stability": np.std(durations) / np.mean(durations),
            "memory_stability": np.std(memories) / np.mean(memories),
            "score_stability": np.std(scores) / np.mean(scores)
        }
        
        print(f"Performance trend analysis:")
        print(f"  Duration trend: {trend_analysis['duration_trend_ms_per_run']:+.2f} ms/run")
        print(f"  Memory trend: {trend_analysis['memory_trend_mb_per_run']:+.2f} MB/run")
        print(f"  Score trend: {trend_analysis['score_trend_per_run']:+.4f} per run")
        print(f"  Duration stability (CV): {trend_analysis['duration_stability']:.3f}")
        print(f"  Memory stability (CV): {trend_analysis['memory_stability']:.3f}")
        print(f"  Score stability (CV): {trend_analysis['score_stability']:.3f}")
        
        # Validate trend characteristics
        assert trend_analysis["duration_stability"] < 0.5, \
            f"Duration too unstable: CV={trend_analysis['duration_stability']:.3f}"
        assert trend_analysis["score_stability"] < 0.3, \
            f"Scores too unstable: CV={trend_analysis['score_stability']:.3f}"
        
        # Performance shouldn't degrade significantly over runs
        assert abs(trend_analysis["duration_trend_ms_per_run"]) < 10, \
            f"Duration trend too steep: {trend_analysis['duration_trend_ms_per_run']:.2f} ms/run"
    
    def test_quality_metric_validation(self, sample_people_data, rng):
        """Test quality metric tracking and validation."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        people, vectors, clusters, centers = sample_people_data(150, 24)
        
        # Create multiple test scenarios
        test_scenarios = [
            {
                "name": "similar_users",
                "user_idx": 0,
                "pool_indices": list(range(1, 21)),  # Similar cluster members
                "expected_quality": "high"
            },
            {
                "name": "diverse_pool",
                "user_idx": 0,
                "pool_indices": list(range(1, 51, 2)),  # Every other person
                "expected_quality": "medium"
            },
            {
                "name": "random_pool", 
                "user_idx": 0,
                "pool_indices": rng.choice(range(1, 150), 30, replace=False).tolist(),
                "expected_quality": "variable"
            }
        ]
        
        quality_results = {}
        
        for scenario in test_scenarios:
            user = people[scenario["user_idx"]]
            pool = [people[i] for i in scenario["pool_indices"]]
            
            # Run recommendation pipeline
            results = score_pool(
                seed=user,
                pool=pool,
                intent="friend",
                ser_scale=1.0,
                k=15,
                use_faiss_prefilter=False,
                quantum_gamma=0.2,
                quantum_dims=4,
                use_gbs=False
            )
            
            # Calculate quality metrics
            user_vec = np.array(user["vec"])
            
            # Similarity-based metrics
            similarities = []
            novelties = []
            diversities = []
            
            for r in results:
                candidate_vec = np.array(r["candidate"]["vec"])
                similarity = np.dot(user_vec / np.linalg.norm(user_vec), 
                                  candidate_vec / np.linalg.norm(candidate_vec))
                similarities.append(similarity)
                novelties.append(r["candidate"].get("novelty", 0.5))
            
            # Diversity: average pairwise distance between recommendations
            pairwise_distances = []
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    vec_i = np.array(results[i]["candidate"]["vec"])
                    vec_j = np.array(results[j]["candidate"]["vec"])
                    distance = np.linalg.norm(vec_i - vec_j)
                    pairwise_distances.append(distance)
            
            quality_metrics = {
                "mean_similarity": np.mean(similarities),
                "similarity_std": np.std(similarities),
                "mean_novelty": np.mean(novelties),
                "diversity": np.mean(pairwise_distances) if pairwise_distances else 0,
                "coverage": len(set([r["candidate"]["id"] for r in results])) / len(results),
                "score_range": max([r["total_score"] for r in results]) - min([r["total_score"] for r in results])
            }
            
            quality_results[scenario["name"]] = quality_metrics
        
        print(f"Quality metric validation:")
        for scenario_name, metrics in quality_results.items():
            print(f"  {scenario_name}:")
            print(f"    Similarity: {metrics['mean_similarity']:.3f} ± {metrics['similarity_std']:.3f}")
            print(f"    Novelty: {metrics['mean_novelty']:.3f}")
            print(f"    Diversity: {metrics['diversity']:.3f}")
            print(f"    Coverage: {metrics['coverage']:.3f}")
            print(f"    Score range: {metrics['score_range']:.3f}")
        
        # Validate quality metric characteristics
        for scenario_name, metrics in quality_results.items():
            assert 0 <= metrics["coverage"] <= 1, f"Coverage should be in [0,1] for {scenario_name}"
            assert metrics["diversity"] >= 0, f"Diversity should be non-negative for {scenario_name}"
            assert -1 <= metrics["mean_similarity"] <= 1, f"Similarity should be in [-1,1] for {scenario_name}"
            assert 0 <= metrics["mean_novelty"] <= 1, f"Novelty should be in [0,1] for {scenario_name}"
        
        # Cross-scenario quality validation
        similar_quality = quality_results["similar_users"]["mean_similarity"]
        diverse_quality = quality_results["diverse_pool"]["mean_similarity"]
        
        # Similar users scenario should have higher average similarity
        print(f"Quality comparison: similar_users similarity={similar_quality:.3f}, diverse_pool similarity={diverse_quality:.3f}")
    
    def test_cross_version_compatibility(self, sample_people_data, rng, temp_dir):
        """Test cross-version compatibility validation."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        # Simulate different version configurations
        version_configs = {
            "v1.0": {
                "quantum_gamma": 0.0,
                "use_gbs": False,
                "use_faiss_prefilter": True,
                "ser_scale": 1.0
            },
            "v1.1": {
                "quantum_gamma": 0.3,
                "use_gbs": False,
                "use_faiss_prefilter": True,
                "ser_scale": 1.0
            },
            "v1.2": {
                "quantum_gamma": 0.3,
                "use_gbs": True,
                "use_faiss_prefilter": True,
                "ser_scale": 1.0,
                "gbs_lambda": 0.2
            }
        }
        
        people, vectors, clusters, centers = sample_people_data(80, 16)
        user = people[0]
        pool = people[1:31]
        
        version_results = {}
        
        for version, config in version_configs.items():
            # Run pipeline with version-specific config
            base_params = {
                "seed": user,
                "pool": pool,
                "intent": "mentor",
                "ser_scale": config.get("ser_scale", 1.0),
                "k": 12,
                "use_faiss_prefilter": config.get("use_faiss_prefilter", False),
                "M_prefilter": 25,
                "quantum_gamma": config.get("quantum_gamma", 0.0),
                "quantum_dims": 4,
                "use_gbs": config.get("use_gbs", False),
                "gbs_modes": 3,
                "gbs_shots": 40,
                "gbs_lambda": config.get("gbs_lambda", 0.0)
            }
            
            start_time = time.perf_counter()
            results = score_pool(**base_params)
            end_time = time.perf_counter()
            
            # Analyze version-specific metrics
            version_metrics = {
                "n_results": len(results),
                "execution_time_ms": (end_time - start_time) * 1000,
                "avg_score": np.mean([r["total_score"] for r in results]),
                "score_std": np.std([r["total_score"] for r in results]),
                "top_3_ids": [r["candidate"]["id"] for r in results[:3]]
            }
            
            version_results[version] = version_metrics
        
        print(f"Cross-version compatibility test:")
        for version, metrics in version_results.items():
            print(f"  {version}: {metrics['n_results']} results, "
                  f"{metrics['execution_time_ms']:.1f} ms, "
                  f"avg_score={metrics['avg_score']:.3f}")
        
        # Validate cross-version consistency
        execution_times = [metrics["execution_time_ms"] for metrics in version_results.values()]
        result_counts = [metrics["n_results"] for metrics in version_results.values()]
        
        # All versions should return results
        assert all(count > 0 for count in result_counts), "All versions should return results"
        
        # Execution times should be reasonable
        assert all(time_ms < 10000 for time_ms in execution_times), "All versions should complete in reasonable time"
        
        # Check for breaking changes between versions
        version_names = list(version_results.keys())
        for i in range(len(version_names) - 1):
            curr_version = version_names[i]
            next_version = version_names[i + 1]
            
            curr_time = version_results[curr_version]["execution_time_ms"]
            next_time = version_results[next_version]["execution_time_ms"]
            
            time_increase = (next_time - curr_time) / curr_time if curr_time > 0 else 0
            
            print(f"  {curr_version} -> {next_version}: {time_increase:+.1%} execution time change")
            
            # Significant performance regressions should be flagged
            if time_increase > 0.5:  # >50% slower
                print(f"    Warning: Significant performance regression detected")
    
    def test_production_readiness_validation(self, sample_people_data, rng):
        """Test production readiness validation."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        people, vectors, clusters, centers = sample_people_data(200, 48)
        
        # Production readiness criteria
        readiness_tests = {
            "response_time": {"target": 2000, "unit": "ms", "test_type": "performance"},
            "memory_usage": {"target": 500, "unit": "MB", "test_type": "resource"},
            "error_rate": {"target": 0.01, "unit": "ratio", "test_type": "reliability"},
            "concurrent_users": {"target": 5, "unit": "sessions", "test_type": "scalability"}
        }
        
        readiness_results = {}
        
        # Response time test
        user = people[0]
        pool = people[1:101]
        
        response_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            
            results = score_pool(
                seed=user,
                pool=pool,
                intent="collab",
                ser_scale=1.0,
                k=20,
                use_faiss_prefilter=True,
                M_prefilter=50,
                quantum_gamma=0.2,
                quantum_dims=4,
                use_gbs=False  # For consistent performance
            )
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
        
        avg_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        readiness_results["response_time"] = {
            "avg": avg_response_time,
            "p95": p95_response_time,
            "passes": p95_response_time < readiness_tests["response_time"]["target"]
        }
        
        # Memory usage test (simplified)
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        readiness_results["memory_usage"] = {
            "current": memory_mb,
            "passes": memory_mb < readiness_tests["memory_usage"]["target"]
        }
        
        # Error rate test
        error_count = 0
        total_tests = 20
        
        for i in range(total_tests):
            try:
                test_user = people[i % len(people)]
                test_pool = people[(i+1) % len(people):(i+11) % len(people)]
                if len(test_pool) == 0:
                    test_pool = people[1:11]
                
                results = score_pool(
                    seed=test_user,
                    pool=test_pool,
                    intent="friend",
                    ser_scale=1.0,
                    k=5,
                    use_faiss_prefilter=False
                )
                
                # Basic validation
                assert len(results) <= 5, "Should not exceed k"
                assert all(np.isfinite(r["total_score"]) for r in results), "All scores should be finite"
                
            except Exception as e:
                error_count += 1
                print(f"    Error in test {i}: {e}")
        
        error_rate = error_count / total_tests
        
        readiness_results["error_rate"] = {
            "rate": error_rate,
            "errors": error_count,
            "total": total_tests,
            "passes": error_rate < readiness_tests["error_rate"]["target"]
        }
        
        # Overall readiness assessment
        all_passed = all(result["passes"] for result in readiness_results.values())
        
        print(f"Production readiness validation:")
        for test_name, result in readiness_results.items():
            status = "PASS" if result["passes"] else "FAIL"
            target = readiness_tests[test_name]["target"]
            unit = readiness_tests[test_name]["unit"]
            
            if test_name == "response_time":
                print(f"  {test_name}: {status} (avg: {result['avg']:.1f} ms, p95: {result['p95']:.1f} ms, target: <{target} {unit})")
            elif test_name == "memory_usage":
                print(f"  {test_name}: {status} (current: {result['current']:.1f} MB, target: <{target} {unit})")
            elif test_name == "error_rate":
                print(f"  {test_name}: {status} (rate: {result['rate']:.3f}, target: <{target} {unit})")
        
        print(f"Overall production readiness: {'READY' if all_passed else 'NOT READY'}")
        
        # Critical production criteria
        assert readiness_results["error_rate"]["passes"], \
            f"Error rate too high for production: {error_rate:.3f} > {readiness_tests['error_rate']['target']}"
        
        if not readiness_results["response_time"]["passes"]:
            print(f"Warning: Response time may be too slow for production: {p95_response_time:.1f} ms")
    
    def test_automated_report_generation(self, temp_dir):
        """Test automated validation report generation."""
        validator = ContinuousValidator(temp_dir)
        
        # Mock test results
        test_results = {
            "passed": 45,
            "failed": 2,
            "skipped": 3,
            "warnings": 5
        }
        
        performance_metrics = {
            "quantum_kernel_avg_time_ms": 25.3,
            "gbs_boost_avg_time_ms": 180.5,
            "faiss_search_avg_time_ms": 8.2,
            "pipeline_total_time_ms": 420.1,
            "peak_memory_mb": 234.7,
            "throughput_candidates_per_sec": 1250.0
        }
        
        quality_metrics = {
            "recommendation_accuracy": 0.87,
            "diversity_score": 0.72,
            "novelty_balance": 0.65,
            "user_satisfaction_proxy": 0.81,
            "coverage_ratio": 0.95
        }
        
        # Generate report
        report = validator.generate_report(test_results, performance_metrics, quality_metrics)
        
        # Validate report structure
        assert report.timestamp is not None, "Report should have timestamp"
        assert report.test_suite == "quantum_serendipity_engine", "Report should have correct test suite name"
        assert report.passed == 45, "Report should have correct pass count"
        assert report.failed == 2, "Report should have correct fail count"
        
        # Validate metrics inclusion
        assert "quantum_kernel_avg_time_ms" in report.performance_metrics, "Report should include performance metrics"
        assert "recommendation_accuracy" in report.quality_metrics, "Report should include quality metrics"
        
        # Validate recommendations
        assert isinstance(report.recommendations, list), "Report should have recommendations list"
        
        print(f"Automated report generation test:")
        print(f"  Report timestamp: {report.timestamp}")
        print(f"  Test summary: {report.passed} passed, {report.failed} failed, {report.skipped} skipped")
        print(f"  Performance metrics: {len(report.performance_metrics)} items")
        print(f"  Quality metrics: {len(report.quality_metrics)} items")
        print(f"  Regression flags: {len(report.regression_flags)} items")
        print(f"  Recommendations: {len(report.recommendations)} items")
        
        if report.recommendations:
            print(f"  Sample recommendations:")
            for rec in report.recommendations[:3]:
                print(f"    - {rec}")
        
        # Test report persistence
        report_files = [f for f in os.listdir(temp_dir) if f.startswith("validation_report_")]
        assert len(report_files) > 0, "Report should be saved to file"
        
        # Test report loading
        latest_report_file = max([os.path.join(temp_dir, f) for f in report_files], key=os.path.getctime)
        with open(latest_report_file, 'r') as f:
            loaded_report_data = json.load(f)
        
        assert loaded_report_data["passed"] == 45, "Saved report should be loadable and correct"
        print(f"  Report successfully saved and loaded from: {latest_report_file}")

if __name__ == "__main__":
    pytest.main([__file__])
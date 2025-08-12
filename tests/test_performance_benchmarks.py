"""
Performance Benchmarking Suite
==============================

Comprehensive performance benchmarks for the quantum-enhanced serendipity engine.
Establishes performance baselines, scalability characteristics, and quantum superiority metrics.

Key Benchmark Areas:
- Scalability analysis across problem dimensions
- Component-wise performance profiling  
- Memory usage and efficiency validation
- Quantum vs classical performance comparison
- Real-time performance under load
- Cross-platform compatibility metrics
"""

import pytest
import numpy as np
import time
import psutil
import gc
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

@dataclass
class BenchmarkResult:
    """Structured benchmark result."""
    name: str
    duration_ms: float
    memory_mb: float
    throughput_ops_per_sec: float
    scalability_factor: float
    metadata: Dict[str, Any]

class PerformanceBenchmarker:
    """Advanced performance benchmarking utilities."""
    
    def __init__(self):
        self.results = []
        self.baseline_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_function(self, func: Callable, args: tuple, name: str, 
                         n_iterations: int = 1, warmup: int = 1) -> BenchmarkResult:
        """Benchmark a function with multiple iterations."""
        
        # Warmup runs
        for _ in range(warmup):
            func(*args)
        
        # Force garbage collection
        gc.collect()
        start_memory = self._get_memory_usage()
        
        # Actual benchmark
        start_time = time.perf_counter()
        
        for _ in range(n_iterations):
            result = func(*args)
        
        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        
        duration_ms = (end_time - start_time) * 1000 / n_iterations
        memory_mb = max(0, end_memory - start_memory)
        throughput = n_iterations / (end_time - start_time) if end_time > start_time else 0
        
        benchmark_result = BenchmarkResult(
            name=name,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            throughput_ops_per_sec=throughput,
            scalability_factor=1.0,  # Will be calculated later
            metadata={
                "n_iterations": n_iterations,
                "warmup": warmup,
                "result_type": type(result).__name__ if result is not None else "None"
            }
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def analyze_scalability(self, func: Callable, param_name: str, 
                          param_values: List[Any], fixed_args: Dict[str, Any]) -> List[BenchmarkResult]:
        """Analyze scalability across different parameter values."""
        scalability_results = []
        baseline_time = None
        
        for param_value in param_values:
            # Prepare arguments
            args_dict = fixed_args.copy()
            args_dict[param_name] = param_value
            
            # Convert to function arguments (assuming function takes kwargs)
            def test_func():
                return func(**args_dict)
            
            # Benchmark
            result = self.benchmark_function(
                test_func, (), 
                name=f"{func.__name__}_{param_name}_{param_value}",
                n_iterations=3,
                warmup=1
            )
            
            # Calculate scalability factor
            if baseline_time is None:
                baseline_time = result.duration_ms
                result.scalability_factor = 1.0
            else:
                result.scalability_factor = result.duration_ms / baseline_time
            
            result.metadata[param_name] = param_value
            scalability_results.append(result)
            
            print(f"Scalability test {param_name}={param_value}: "
                  f"{result.duration_ms:.2f} ms, factor={result.scalability_factor:.2f}x")
        
        return scalability_results

class TestQuantumPerformanceBenchmarks:
    """Benchmark quantum algorithm performance."""
    
    def test_quantum_kernel_scalability(self, rng, performance_monitor):
        """Benchmark quantum kernel scalability."""
        from serendipity_engine_ui.engine.quantum import quantum_kernel_to_seed, pca_compress
        
        benchmarker = PerformanceBenchmarker()
        
        # Test scalability across dimensions
        def quantum_kernel_test(n_dims, n_candidates):
            seed_vec = rng.uniform(0, np.pi/2, n_dims)
            cand_vecs = rng.uniform(0, np.pi/2, (n_candidates, n_dims))
            return quantum_kernel_to_seed(seed_vec, cand_vecs)
        
        # Dimension scalability
        dim_results = benchmarker.analyze_scalability(
            quantum_kernel_test,
            "n_dims",
            [2, 4, 6, 8],
            {"n_candidates": 50}
        )
        
        # Candidate scalability  
        cand_results = benchmarker.analyze_scalability(
            quantum_kernel_test,
            "n_candidates", 
            [10, 25, 50, 100, 200],
            {"n_dims": 4}
        )
        
        # Analyze results
        print("\nQuantum Kernel Scalability Analysis:")
        print("Dimension scalability:")
        for result in dim_results:
            print(f"  {result.metadata['n_dims']}D: {result.duration_ms:.2f} ms/call, "
                  f"factor={result.scalability_factor:.2f}x")
        
        print("Candidate scalability:")
        for result in cand_results:
            n_cands = result.metadata['n_candidates']
            time_per_cand = result.duration_ms / n_cands
            print(f"  {n_cands} candidates: {result.duration_ms:.2f} ms/call, "
                  f"{time_per_cand:.4f} ms/candidate")
        
        # Performance assertions
        final_dim_result = dim_results[-1]
        final_cand_result = cand_results[-1]
        
        # Dimension scaling should be reasonable (not exponential)
        assert final_dim_result.scalability_factor < 10.0, \
            f"Dimension scaling too poor: {final_dim_result.scalability_factor:.2f}x"
        
        # Candidate scaling should be roughly linear
        assert final_cand_result.scalability_factor < 25.0, \
            f"Candidate scaling too poor: {final_cand_result.scalability_factor:.2f}x"
    
    def test_pca_compression_performance(self, rng):
        """Benchmark PCA compression performance."""
        from serendipity_engine_ui.engine.quantum import pca_compress
        
        benchmarker = PerformanceBenchmarker()
        
        def pca_test(n_samples, n_features, out_dim):
            X = rng.normal(0, 1, (n_samples, n_features))
            return pca_compress(X, out_dim)
        
        # Test different problem sizes
        test_cases = [
            (100, 20, 4),
            (200, 40, 6),
            (500, 80, 8),
            (1000, 160, 10)
        ]
        
        pca_results = []
        for n_samples, n_features, out_dim in test_cases:
            result = benchmarker.benchmark_function(
                pca_test, (n_samples, n_features, out_dim),
                name=f"pca_{n_samples}x{n_features}_to_{out_dim}",
                n_iterations=5
            )
            
            result.metadata.update({
                "n_samples": n_samples,
                "n_features": n_features,
                "out_dim": out_dim,
                "compression_ratio": n_features / out_dim
            })
            
            pca_results.append(result)
        
        print("\nPCA Compression Performance:")
        for result in pca_results:
            meta = result.metadata
            throughput_samples_per_sec = meta["n_samples"] / (result.duration_ms / 1000)
            print(f"  {meta['n_samples']}×{meta['n_features']}→{meta['out_dim']}: "
                  f"{result.duration_ms:.2f} ms, "
                  f"{throughput_samples_per_sec:.0f} samples/sec")
        
        # Performance should be reasonable for real-time use
        for result in pca_results:
            meta = result.metadata
            if meta["n_samples"] <= 200:
                assert result.duration_ms < 100, \
                    f"PCA too slow for small datasets: {result.duration_ms:.2f} ms for {meta['n_samples']} samples"

class TestPhotonicsPerformanceBenchmarks:
    """Benchmark photonic GBS performance."""
    
    def test_gbs_performance_scaling(self, rng):
        """Benchmark GBS performance scaling."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        benchmarker = PerformanceBenchmarker()
        
        def gbs_test(n_candidates, n_dims, modes, shots):
            seed_vec = rng.normal(0, 1, n_dims)
            cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
            return gbs_boost(seed_vec, cand_vecs, modes=modes, shots=shots, cutoff=4)
        
        # Test candidate scaling
        candidate_results = benchmarker.analyze_scalability(
            gbs_test,
            "n_candidates",
            [20, 50, 100, 200],
            {"n_dims": 16, "modes": 4, "shots": 60}
        )
        
        # Test dimension scaling
        dimension_results = benchmarker.analyze_scalability(
            gbs_test,
            "n_dims", 
            [8, 16, 32, 64],
            {"n_candidates": 50, "modes": 4, "shots": 60}
        )
        
        # Test shots scaling
        shots_results = benchmarker.analyze_scalability(
            gbs_test,
            "shots",
            [20, 60, 120, 240],
            {"n_candidates": 50, "n_dims": 16, "modes": 4}
        )
        
        print("\nGBS Performance Scaling Analysis:")
        
        print("Candidate scaling:")
        for result in candidate_results:
            n_cands = result.metadata['n_candidates']
            time_per_cand = result.duration_ms / n_cands
            print(f"  {n_cands} candidates: {result.duration_ms:.1f} ms, "
                  f"{time_per_cand:.3f} ms/candidate")
        
        print("Dimension scaling:")
        for result in dimension_results:
            n_dims = result.metadata['n_dims']
            print(f"  {n_dims}D: {result.duration_ms:.1f} ms, factor={result.scalability_factor:.2f}x")
        
        print("Shots scaling:")
        for result in shots_results:
            shots = result.metadata['shots']
            time_per_shot = result.duration_ms / shots
            print(f"  {shots} shots: {result.duration_ms:.1f} ms, "
                  f"{time_per_shot:.3f} ms/shot")
        
        # Performance assertions
        # GBS should scale reasonably with problem size
        final_cand_result = candidate_results[-1]
        assert final_cand_result.scalability_factor < 15.0, \
            f"GBS candidate scaling too poor: {final_cand_result.scalability_factor:.2f}x"
        
        # Shots should scale roughly linearly
        final_shots_result = shots_results[-1]
        expected_shots_factor = final_shots_result.metadata['shots'] / shots_results[0].metadata['shots']
        assert final_shots_result.scalability_factor < expected_shots_factor * 1.5, \
            f"GBS shots scaling inefficient: {final_shots_result.scalability_factor:.2f}x vs expected ~{expected_shots_factor:.2f}x"
    
    def test_gbs_parameter_sensitivity_performance(self, rng):
        """Test performance sensitivity to GBS parameters."""
        from serendipity_engine_ui.engine.photonic_gbs import gbs_boost
        
        benchmarker = PerformanceBenchmarker()
        
        n_candidates = 80
        n_dims = 24
        seed_vec = rng.normal(0, 1, n_dims)
        cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
        
        # Test different parameter combinations
        param_combinations = [
            {"modes": 2, "shots": 40, "cutoff": 3},
            {"modes": 4, "shots": 80, "cutoff": 4},
            {"modes": 6, "shots": 120, "cutoff": 5},
            {"modes": 8, "shots": 160, "cutoff": 6},
        ]
        
        param_results = []
        for params in param_combinations:
            def gbs_param_test():
                return gbs_boost(seed_vec, cand_vecs, **params, r_max=0.4)
            
            result = benchmarker.benchmark_function(
                gbs_param_test, (),
                name=f"gbs_m{params['modes']}_s{params['shots']}_c{params['cutoff']}",
                n_iterations=3
            )
            
            result.metadata.update(params)
            result.metadata["complexity"] = params["modes"] * params["shots"] * params["cutoff"]
            param_results.append(result)
        
        print("\nGBS Parameter Sensitivity:")
        for result in param_results:
            meta = result.metadata
            complexity = meta["complexity"]
            efficiency = complexity / result.duration_ms if result.duration_ms > 0 else 0
            
            print(f"  modes={meta['modes']}, shots={meta['shots']}, cutoff={meta['cutoff']}: "
                  f"{result.duration_ms:.1f} ms, complexity={complexity}, "
                  f"efficiency={efficiency:.1f} complex_units/ms")
        
        # Higher complexity should generally take longer, but efficiency shouldn't degrade too much
        complexities = [r.metadata["complexity"] for r in param_results]
        durations = [r.duration_ms for r in param_results]
        
        # Basic monotonicity check (allowing some variation)
        increasing_pairs = 0
        total_pairs = 0
        for i in range(len(param_results)):
            for j in range(i + 1, len(param_results)):
                if complexities[i] < complexities[j]:
                    if durations[i] <= durations[j]:
                        increasing_pairs += 1
                    total_pairs += 1
        
        monotonicity_ratio = increasing_pairs / total_pairs if total_pairs > 0 else 0
        print(f"  Performance monotonicity: {monotonicity_ratio:.2f} ({increasing_pairs}/{total_pairs} pairs)")
        
        # Should show reasonable correlation between complexity and time
        assert monotonicity_ratio > 0.5, f"GBS performance should correlate with complexity: {monotonicity_ratio:.2f}"

class TestClassicalPerformanceBenchmarks:
    """Benchmark classical algorithm performance."""
    
    def test_faiss_vs_numpy_performance_comparison(self, rng):
        """Detailed FAISS vs numpy performance comparison."""
        from serendipity_engine_ui.engine.faiss_helper import top_m_cosine, have_faiss
        
        benchmarker = PerformanceBenchmarker()
        
        # Test different problem sizes
        test_configs = [
            (500, 32, 25),
            (1000, 64, 50),
            (2000, 128, 100),
            (5000, 256, 200)
        ]
        
        comparison_results = []
        
        for n_candidates, n_dims, m in test_configs:
            # Prepare test data
            seed_vec = rng.normal(0, 1, n_dims)
            seed_vec = seed_vec / np.linalg.norm(seed_vec)
            
            cand_vecs = rng.normal(0, 1, (n_candidates, n_dims))
            cand_vecs = cand_vecs / np.linalg.norm(cand_vecs, axis=1, keepdims=True)
            
            config_name = f"{n_candidates}x{n_dims}_top{m}"
            
            # Benchmark FAISS version (if available)
            if have_faiss():
                faiss_result = benchmarker.benchmark_function(
                    top_m_cosine, (seed_vec, cand_vecs, m),
                    name=f"faiss_{config_name}",
                    n_iterations=10
                )
                faiss_result.metadata.update({
                    "method": "faiss",
                    "n_candidates": n_candidates,
                    "n_dims": n_dims,
                    "m": m
                })
            else:
                faiss_result = None
            
            # Benchmark numpy version
            with patch('serendipity_engine_ui.engine.faiss_helper.have_faiss', return_value=False):
                numpy_result = benchmarker.benchmark_function(
                    top_m_cosine, (seed_vec, cand_vecs, m),
                    name=f"numpy_{config_name}",
                    n_iterations=10
                )
                numpy_result.metadata.update({
                    "method": "numpy",
                    "n_candidates": n_candidates,
                    "n_dims": n_dims,
                    "m": m
                })
            
            comparison_results.append({
                "config": config_name,
                "faiss": faiss_result,
                "numpy": numpy_result,
                "n_candidates": n_candidates,
                "n_dims": n_dims,
                "m": m
            })
        
        print("\nFAISS vs NumPy Performance Comparison:")
        for result in comparison_results:
            config = result["config"]
            faiss_result = result["faiss"]
            numpy_result = result["numpy"]
            
            if faiss_result:
                speedup = numpy_result.duration_ms / faiss_result.duration_ms
                print(f"  {config}:")
                print(f"    FAISS: {faiss_result.duration_ms:.2f} ms")
                print(f"    NumPy: {numpy_result.duration_ms:.2f} ms")
                print(f"    Speedup: {speedup:.2f}x")
            else:
                print(f"  {config}: NumPy only: {numpy_result.duration_ms:.2f} ms")
            
            # Calculate throughput
            n_cands = result["n_candidates"] 
            numpy_throughput = n_cands / (numpy_result.duration_ms / 1000)
            print(f"    NumPy throughput: {numpy_throughput:.0f} candidates/sec")
            
            if faiss_result:
                faiss_throughput = n_cands / (faiss_result.duration_ms / 1000)
                print(f"    FAISS throughput: {faiss_throughput:.0f} candidates/sec")
        
        # Performance assertions for numpy fallback
        for result in comparison_results:
            numpy_result = result["numpy"]
            n_cands = result["n_candidates"]
            
            # NumPy should handle reasonable sizes efficiently
            if n_cands <= 1000:
                assert numpy_result.duration_ms < 100, \
                    f"NumPy too slow for {n_cands} candidates: {numpy_result.duration_ms:.2f} ms"
    
    def test_multi_objective_scoring_performance(self, rng):
        """Benchmark multi-objective scoring performance."""
        from serendipity_engine_ui.engine.fastscore import prepare_arrays, score_vectorized
        
        benchmarker = PerformanceBenchmarker()
        
        def scoring_test(n_candidates, n_dims):
            # Create test data
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
            
            # Prepare arrays
            seed_unit, arrs = prepare_arrays(seed, pool)
            
            # Score
            scores = score_vectorized(seed_unit, arrs, "mentor", 1.0, rng)
            return scores
        
        # Test scalability
        scoring_results = benchmarker.analyze_scalability(
            scoring_test,
            "n_candidates",
            [100, 500, 1000, 2000, 5000],
            {"n_dims": 64}
        )
        
        print("\nMulti-Objective Scoring Performance:")
        for result in scoring_results:
            n_cands = result.metadata['n_candidates']
            throughput = n_cands / (result.duration_ms / 1000)
            time_per_candidate = result.duration_ms / n_cands
            
            print(f"  {n_cands} candidates: {result.duration_ms:.1f} ms, "
                  f"{throughput:.0f} candidates/sec, "
                  f"{time_per_candidate:.4f} ms/candidate")
        
        # Vectorized scoring should be very fast
        for result in scoring_results:
            n_cands = result.metadata['n_candidates']
            time_per_candidate = result.duration_ms / n_cands
            
            assert time_per_candidate < 0.1, \
                f"Vectorized scoring too slow: {time_per_candidate:.4f} ms/candidate for {n_cands} candidates"

class TestEndToEndPerformanceBenchmarks:
    """Benchmark complete end-to-end pipeline performance."""
    
    def test_complete_pipeline_performance(self, rng):
        """Benchmark complete discovery pipeline performance."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        benchmarker = PerformanceBenchmarker()
        
        def pipeline_test(n_candidates, n_dims, use_quantum, use_gbs):
            # Create test data
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
            
            # Run pipeline
            results = score_pool(
                seed=seed,
                pool=pool,
                intent="ship",
                ser_scale=1.0,
                k=20,
                use_faiss_prefilter=True,
                M_prefilter=min(100, len(pool)),
                quantum_gamma=0.3 if use_quantum else 0.0,
                quantum_dims=4,
                use_gbs=use_gbs,
                gbs_modes=3 if use_gbs else 4,
                gbs_shots=40 if use_gbs else 60,
                gbs_lambda=0.2 if use_gbs else 0.0
            )
            
            return results
        
        # Test different pipeline configurations
        pipeline_configs = [
            {"name": "classical", "use_quantum": False, "use_gbs": False},
            {"name": "quantum", "use_quantum": True, "use_gbs": False},
            {"name": "gbs", "use_quantum": False, "use_gbs": True},
            {"name": "full", "use_quantum": True, "use_gbs": True}
        ]
        
        # Test different problem sizes
        problem_sizes = [
            (200, 32),
            (500, 64),
            (1000, 128)
        ]
        
        pipeline_results = {}
        
        for n_candidates, n_dims in problem_sizes:
            size_name = f"{n_candidates}x{n_dims}"
            pipeline_results[size_name] = {}
            
            for config in pipeline_configs:
                def test_func():
                    return pipeline_test(n_candidates, n_dims, config["use_quantum"], config["use_gbs"])
                
                result = benchmarker.benchmark_function(
                    test_func, (),
                    name=f"pipeline_{config['name']}_{size_name}",
                    n_iterations=3
                )
                
                result.metadata.update({
                    "config": config["name"],
                    "n_candidates": n_candidates,
                    "n_dims": n_dims,
                    "use_quantum": config["use_quantum"],
                    "use_gbs": config["use_gbs"]
                })
                
                pipeline_results[size_name][config["name"]] = result
        
        print("\nComplete Pipeline Performance:")
        for size_name, configs in pipeline_results.items():
            print(f"  Problem size {size_name}:")
            
            for config_name, result in configs.items():
                n_cands = result.metadata["n_candidates"]
                throughput = n_cands / (result.duration_ms / 1000)
                
                print(f"    {config_name:10s}: {result.duration_ms:6.1f} ms, "
                      f"{throughput:5.0f} candidates/sec")
                
                # Real-time performance requirements
                if n_cands <= 500:
                    assert result.duration_ms < 5000, \
                        f"Pipeline too slow for real-time use: {result.duration_ms:.1f} ms for {n_cands} candidates"
    
    def test_concurrent_pipeline_performance(self, rng):
        """Test pipeline performance under concurrent load."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        # Create shared test data
        n_candidates = 300
        n_dims = 48
        
        seeds = []
        pools = []
        
        for session in range(10):
            seed = {"vec": rng.normal(0, 1, n_dims).tolist()}
            pool = []
            for i in range(n_candidates):
                person = {
                    "id": f"s{session}_p{i}",
                    "vec": rng.normal(0, 1, n_dims).tolist(),
                    "novelty": rng.random(),
                    "availability": rng.random(),
                    "pathTrust": rng.random()
                }
                pool.append(person)
            
            seeds.append(seed)
            pools.append(pool)
        
        def single_pipeline_task(session_id):
            """Single pipeline execution task."""
            start_time = time.perf_counter()
            
            results = score_pool(
                seed=seeds[session_id],
                pool=pools[session_id],
                intent="friend",
                ser_scale=1.0,
                k=15,
                use_faiss_prefilter=True,
                M_prefilter=100,
                quantum_gamma=0.2,
                quantum_dims=4,
                use_gbs=False,  # Disable GBS for faster concurrent testing
                gbs_lambda=0.0
            )
            
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000
            
            return {
                "session_id": session_id,
                "duration_ms": duration,
                "n_results": len(results),
                "thread_id": threading.current_thread().ident
            }
        
        # Test sequential execution
        print("\nConcurrent Pipeline Performance Test:")
        
        sequential_start = time.perf_counter()
        sequential_results = []
        for session_id in range(5):  # Test subset for speed
            result = single_pipeline_task(session_id)
            sequential_results.append(result)
        sequential_end = time.perf_counter()
        sequential_total = (sequential_end - sequential_start) * 1000
        
        # Test concurrent execution
        concurrent_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(single_pipeline_task, i) for i in range(5)]
            concurrent_results = [f.result() for f in as_completed(futures)]
        concurrent_end = time.perf_counter()
        concurrent_total = (concurrent_end - concurrent_start) * 1000
        
        print(f"Sequential execution:")
        avg_sequential = np.mean([r["duration_ms"] for r in sequential_results])
        print(f"  Total time: {sequential_total:.1f} ms")
        print(f"  Average per session: {avg_sequential:.1f} ms")
        
        print(f"Concurrent execution (3 workers):")
        avg_concurrent = np.mean([r["duration_ms"] for r in concurrent_results])
        print(f"  Total time: {concurrent_total:.1f} ms")
        print(f"  Average per session: {avg_concurrent:.1f} ms")
        
        # Calculate performance metrics
        speedup = sequential_total / concurrent_total
        efficiency = speedup / 3  # 3 workers
        
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {efficiency:.2f}")
        
        # Concurrent execution should provide some speedup
        assert speedup > 1.2, f"Concurrent execution should provide speedup, got {speedup:.2f}x"
        
        # Individual session times shouldn't degrade significantly
        max_degradation = max(concurrent_results, key=lambda x: x["duration_ms"])["duration_ms"]
        max_sequential = max(sequential_results, key=lambda x: x["duration_ms"])["duration_ms"]
        degradation_factor = max_degradation / max_sequential
        
        assert degradation_factor < 2.0, f"Concurrent execution causes too much degradation: {degradation_factor:.2f}x"
        
        print(f"  Max degradation: {degradation_factor:.2f}x")

class TestQuantumSuperiorityMetrics:
    """Establish quantum superiority metrics and benchmarks."""
    
    def test_quantum_vs_classical_recommendation_quality(self, rng):
        """Compare recommendation quality between quantum and classical approaches."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        # Create structured test dataset with known ground truth
        n_candidates = 200
        n_dims = 32
        
        # Create user with specific preferences
        user_profile = rng.normal(0, 1, n_dims)
        user_profile = user_profile / np.linalg.norm(user_profile)
        seed = {"vec": user_profile.tolist()}
        
        # Create candidates with varying similarity to user
        pool = []
        ground_truth_similarities = []
        
        for i in range(n_candidates):
            if i < 50:  # High similarity candidates
                noise = rng.normal(0, 0.3, n_dims)
                candidate_vec = user_profile + noise
            elif i < 100:  # Medium similarity
                noise = rng.normal(0, 0.8, n_dims)
                candidate_vec = user_profile + noise
            else:  # Random candidates
                candidate_vec = rng.normal(0, 1, n_dims)
            
            candidate_vec = candidate_vec / np.linalg.norm(candidate_vec)
            similarity = np.dot(user_profile, candidate_vec)
            ground_truth_similarities.append(similarity)
            
            person = {
                "id": f"candidate_{i}",
                "vec": candidate_vec.tolist(),
                "novelty": rng.random(),
                "availability": rng.random(),
                "pathTrust": rng.random(),
                "ground_truth_similarity": similarity
            }
            pool.append(person)
        
        # Test different approaches
        approaches = [
            {"name": "classical", "quantum_gamma": 0.0, "use_gbs": False},
            {"name": "quantum", "quantum_gamma": 0.5, "use_gbs": False},
            {"name": "gbs", "quantum_gamma": 0.0, "use_gbs": True, "gbs_lambda": 0.3},
            {"name": "hybrid", "quantum_gamma": 0.3, "use_gbs": True, "gbs_lambda": 0.2}
        ]
        
        quality_results = {}
        
        for approach in approaches:
            # Get recommendations
            results = score_pool(
                seed=seed,
                pool=pool,
                intent="friend",
                ser_scale=1.0,
                k=20,
                use_faiss_prefilter=True,
                M_prefilter=100,
                quantum_gamma=approach.get("quantum_gamma", 0.0),
                quantum_dims=4,
                use_gbs=approach.get("use_gbs", False),
                gbs_modes=3,
                gbs_shots=50,
                gbs_lambda=approach.get("gbs_lambda", 0.0)
            )
            
            # Analyze recommendation quality
            recommended_similarities = [r["candidate"]["ground_truth_similarity"] for r in results]
            
            quality_metrics = {
                "mean_similarity": np.mean(recommended_similarities),
                "top_5_similarity": np.mean(recommended_similarities[:5]),
                "diversity": np.std(recommended_similarities),
                "coverage": len(set(recommended_similarities)) / len(recommended_similarities)
            }
            
            quality_results[approach["name"]] = quality_metrics
        
        print("\nQuantum vs Classical Recommendation Quality:")
        for approach_name, metrics in quality_results.items():
            print(f"  {approach_name:10s}: mean_sim={metrics['mean_similarity']:.4f}, "
                  f"top5_sim={metrics['top_5_similarity']:.4f}, "
                  f"diversity={metrics['diversity']:.4f}")
        
        # Compare approaches
        classical_quality = quality_results["classical"]["mean_similarity"]
        quantum_quality = quality_results["quantum"]["mean_similarity"]
        hybrid_quality = quality_results["hybrid"]["mean_similarity"]
        
        quantum_improvement = (quantum_quality - classical_quality) / classical_quality if classical_quality > 0 else 0
        hybrid_improvement = (hybrid_quality - classical_quality) / classical_quality if classical_quality > 0 else 0
        
        print(f"\nQuantum Superiority Metrics:")
        print(f"  Quantum improvement: {quantum_improvement:+.2%}")
        print(f"  Hybrid improvement: {hybrid_improvement:+.2%}")
        
        # All approaches should produce reasonable results
        for approach_name, metrics in quality_results.items():
            assert metrics["mean_similarity"] > 0, \
                f"Approach {approach_name} should produce positive similarity scores"
            assert metrics["diversity"] > 0, \
                f"Approach {approach_name} should show some diversity"
    
    def test_quantum_exploration_vs_exploitation_balance(self, rng):
        """Test quantum approaches' exploration vs exploitation balance."""
        from serendipity_engine_ui.engine.suggest2 import score_pool
        
        # Create test scenario with clear clusters
        n_dims = 24
        n_per_cluster = 30
        
        # User in cluster 0
        cluster_centers = [
            rng.normal(0, 1, n_dims),
            rng.normal(5, 1, n_dims),  
            rng.normal(-5, 1, n_dims)
        ]
        
        user_vec = cluster_centers[0] + rng.normal(0, 0.2, n_dims)
        user_vec = user_vec / np.linalg.norm(user_vec)
        seed = {"vec": user_vec.tolist()}
        
        # Create candidates from each cluster
        pool = []
        for cluster_id, center in enumerate(cluster_centers):
            for i in range(n_per_cluster):
                candidate_vec = center + rng.normal(0, 0.5, n_dims)
                candidate_vec = candidate_vec / np.linalg.norm(candidate_vec)
                
                person = {
                    "id": f"cluster_{cluster_id}_person_{i}",
                    "vec": candidate_vec.tolist(),
                    "novelty": rng.random(),
                    "availability": rng.random(),
                    "pathTrust": rng.random(),
                    "true_cluster": cluster_id
                }
                pool.append(person)
        
        # Test different serendipity levels
        serendipity_levels = [0.5, 1.0, 1.5]
        exploration_results = {}
        
        for ser_scale in serendipity_levels:
            # Classical approach
            classical_results = score_pool(
                seed=seed,
                pool=pool,
                intent="ship",
                ser_scale=ser_scale,
                k=15,
                use_faiss_prefilter=False,
                quantum_gamma=0.0,
                use_gbs=False
            )
            
            # Quantum approach
            quantum_results = score_pool(
                seed=seed,
                pool=pool,
                intent="ship", 
                ser_scale=ser_scale,
                k=15,
                use_faiss_prefilter=False,
                quantum_gamma=0.4,
                quantum_dims=4,
                use_gbs=False
            )
            
            # Analyze cluster distribution in recommendations
            def analyze_cluster_distribution(results):
                cluster_counts = {}
                for r in results:
                    cluster = r["candidate"]["true_cluster"]
                    cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                
                total = len(results)
                cluster_0_ratio = cluster_counts.get(0, 0) / total  # Same cluster as user
                cluster_diversity = len(cluster_counts) / 3  # How many clusters represented
                
                return {
                    "cluster_0_ratio": cluster_0_ratio,
                    "cluster_diversity": cluster_diversity,
                    "cluster_counts": cluster_counts
                }
            
            classical_analysis = analyze_cluster_distribution(classical_results)
            quantum_analysis = analyze_cluster_distribution(quantum_results)
            
            exploration_results[ser_scale] = {
                "classical": classical_analysis,
                "quantum": quantum_analysis
            }
        
        print("\nExploration vs Exploitation Analysis:")
        for ser_scale, results in exploration_results.items():
            print(f"  Serendipity scale {ser_scale}:")
            
            classical = results["classical"]
            quantum = results["quantum"]
            
            print(f"    Classical - Same cluster: {classical['cluster_0_ratio']:.2f}, "
                  f"diversity: {classical['cluster_diversity']:.2f}")
            print(f"    Quantum   - Same cluster: {quantum['cluster_0_ratio']:.2f}, "
                  f"diversity: {quantum['cluster_diversity']:.2f}")
            
            # Quantum should potentially show different exploration patterns
            exploration_difference = quantum["cluster_diversity"] - classical["cluster_diversity"]
            print(f"    Quantum exploration difference: {exploration_difference:+.2f}")
        
        # Higher serendipity should increase exploration for both approaches
        for approach in ["classical", "quantum"]:
            low_ser_diversity = exploration_results[0.5][approach]["cluster_diversity"]
            high_ser_diversity = exploration_results[1.5][approach]["cluster_diversity"]
            
            diversity_increase = high_ser_diversity - low_ser_diversity
            print(f"  {approach.title()} diversity increase with serendipity: {diversity_increase:+.2f}")
            
            assert diversity_increase >= 0, \
                f"{approach} should explore more with higher serendipity: {diversity_increase}"

if __name__ == "__main__":
    # Run comprehensive performance benchmarks
    pytest.main([__file__, "-v", "-s"])